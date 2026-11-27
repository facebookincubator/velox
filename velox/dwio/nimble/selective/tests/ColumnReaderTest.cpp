/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/dwio/nimble/selective/ColumnReader.h"

#include <numeric>

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Tests rebindColumnReader() and the rebindable(), rebind() and unbind() of
// the readers that buildColumnReader() builds for one column of a file, and
// that move between its stripes through StripeStreams.
class ColumnReaderTest : public ::testing::Test,
                         public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Writes 'batches' to a file, one stripe per batch, and prepares to read it
  // from its first stripe.
  void writeFile(const std::vector<RowVectorPtr>& batches) {
    file_ = test::createNimbleFile(
        *rootPool_, std::vector<VectorPtr>(batches.begin(), batches.end()));
    readerBase_ = ReaderBase::create(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file_),
            FileIoContext{.pool = pool()}),
        dwio::common::ReaderOptions(pool()));
    ASSERT_EQ(readerBase_->tablet().stripeCount(), batches.size());
    streams_ = std::make_unique<StripeStreams>(readerBase_);
    streams_->setStripe(0);
    scanSpec_ = std::make_shared<common::ScanSpec>("root");
    scanSpec_->addAllChildFields(*readerBase_->fileSchema());
  }

  // Returns the params of column 'column' of the file, which read its streams
  // through 'streams'.
  NimbleParams makeParams(column_index_t column, StripeStreams& streams) {
    return NimbleParams(
        *pool(),
        splitStats_,
        readerBase_->nimbleSchema()->asRow().childAt(column),
        streams,
        /*rowSizeTracker=*/nullptr,
        encodingFactory_);
  }

  // Builds the reader of column 'column' of the file for the stripe that
  // 'params' reads.
  std::unique_ptr<dwio::common::SelectiveColumnReader> buildReader(
      column_index_t column,
      NimbleParams& params) {
    const auto& fileSchema = readerBase_->fileSchema();
    return buildColumnReader(
        fileSchema->childAt(column),
        readerBase_->fileSchemaWithId()->childAt(column),
        params,
        *scanSpec_->childByName(fileSchema->nameOf(column)),
        /*isRoot=*/false);
  }

  // Reads the first 'numRows' rows of the current stripe through 'reader'.
  static VectorPtr read(
      dwio::common::SelectiveColumnReader& reader,
      vector_size_t numRows) {
    std::vector<vector_size_t> rowNumbers(numRows);
    std::iota(rowNumbers.begin(), rowNumbers.end(), 0);
    const RowSet rows(rowNumbers.data(), rowNumbers.size());
    reader.read(/*offset=*/0, rows, /*incomingNulls=*/nullptr);
    VectorPtr result;
    reader.getValues(reader.outputRows(), &result);
    return result;
  }

  const EncodingFactory encodingFactory_;
  std::string file_;
  std::shared_ptr<ReaderBase> readerBase_;
  std::unique_ptr<StripeStreams> streams_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  dwio::common::SplitStats splitStats_{dwio::common::FileFormat::NIMBLE};
};

// Rebinds integer, floating-point and list readers to each later stripe, and
// reads every stripe through them.
TEST_F(ColumnReaderTest, rebindColumnReader) {
  constexpr int32_t kNumStripes{3};
  constexpr vector_size_t kRowsPerStripe{4};
  std::vector<RowVectorPtr> batches;
  for (int32_t stripe = 0; stripe < kNumStripes; ++stripe) {
    // Only stripe 1 has a null row.
    const auto isNullAt = [&](vector_size_t row) {
      return stripe == 1 && row == 2;
    };
    batches.emplace_back(makeRowVector(
        {"integers", "doubles", "arrays"},
        {makeFlatVector<int64_t>(
             kRowsPerStripe,
             [&](auto row) { return stripe * 100 + row; },
             isNullAt),
         makeFlatVector<double>(
             kRowsPerStripe,
             [&](auto row) { return (stripe * 100 + row) / 10.0; },
             isNullAt),
         makeArrayVector<int32_t>(
             kRowsPerStripe,
             [](auto /*row*/) { return 2; },
             [&](auto index) { return stripe * 100 + index; },
             isNullAt)}));
  }
  writeFile(batches);

  struct TestParam {
    column_index_t column;
    // True if the reader is unbound before each move to the next stripe, as
    // the readers kept for a later stripe are.
    bool unbind;

    std::string debugString() const {
      return fmt::format("column {}, unbind {}", column, unbind);
    }
  };
  std::vector<TestParam> testSettings;
  for (column_index_t column = 0; column < batches[0]->childrenSize();
       ++column) {
    for (const bool unbind : {false, true}) {
      testSettings.push_back({column, unbind});
    }
  }
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    streams_->setStripe(0);
    auto params = makeParams(testData.column, *streams_);
    auto reader = buildReader(testData.column, params);
    ASSERT_TRUE(reader->rebindable());
    for (int32_t stripe = 0; stripe < kNumStripes; ++stripe) {
      SCOPED_TRACE(fmt::format("stripe {}", stripe));
      if (stripe > 0) {
        if (testData.unbind) {
          reader->unbind();
        }
        streams_->setStripe(stripe);
        ASSERT_TRUE(rebindColumnReader(*reader, params));
      }
      streams_->load();
      const auto result = read(*reader, kRowsPerStripe);
      const auto& expected = batches[stripe]->childAt(testData.column);
      ASSERT_EQ(result->size(), kRowsPerStripe);
      for (vector_size_t row = 0; row < kRowsPerStripe; ++row) {
        EXPECT_TRUE(result->equalValueAt(expected.get(), row, row))
            << "row " << row << ": " << result->toString(row) << " vs "
            << expected->toString(row);
      }
    }
  }
}

// A stripe whose arrays are all empty has no elements stream, so the reader of
// the elements, and with it the list reader, cannot rebind to it.
TEST_F(ColumnReaderTest, rebindColumnReaderWithoutStream) {
  constexpr vector_size_t kRowsPerStripe{4};
  std::vector<RowVectorPtr> batches;
  for (const vector_size_t arraySize : {2, 0}) {
    batches.emplace_back(makeRowVector(
        {"arrays"},
        {makeArrayVector<int32_t>(
            kRowsPerStripe,
            [&](auto /*row*/) { return arraySize; },
            [](auto index) { return index; })}));
  }
  writeFile(batches);
  auto params = makeParams(/*column=*/0, *streams_);
  auto reader = buildReader(/*column=*/0, params);
  streams_->load();
  ASSERT_EQ(read(*reader, kRowsPerStripe)->size(), kRowsPerStripe);
  streams_->setStripe(1);
  EXPECT_FALSE(rebindColumnReader(*reader, params));
}

// A string reader keeps the defaults of SelectiveColumnReader, so it cannot
// rebind, and unbinding it is a bug.
TEST_F(ColumnReaderTest, notRebindableColumnReader) {
  writeFile(
      {makeRowVector({"strings"}, {makeFlatVector<std::string>({"a", "b"})})});
  auto params = makeParams(/*column=*/0, *streams_);
  auto reader = buildReader(/*column=*/0, params);
  EXPECT_FALSE(reader->rebindable());
  EXPECT_FALSE(rebindColumnReader(*reader, params));
  VELOX_ASSERT_THROW(
      reader->unbind(), "Unbinding a reader that cannot rebind.");
}

// A reader rebinds only with params that read through the StripeStreams it
// was built with, since that is the StripeStreams that the caller moves to the
// new stripe.
TEST_F(ColumnReaderTest, rebindColumnReaderWithOtherStripeStreams) {
  writeFile({
      makeRowVector({"integers"}, {makeFlatVector<int64_t>({1, 2})}),
      makeRowVector({"integers"}, {makeFlatVector<int64_t>({3, 4})}),
  });
  auto params = makeParams(/*column=*/0, *streams_);
  auto reader = buildReader(/*column=*/0, params);
  StripeStreams otherStreams(readerBase_);
  otherStreams.setStripe(1);
  auto otherParams = makeParams(/*column=*/0, otherStreams);
  NIMBLE_ASSERT_THROW(
      rebindColumnReader(*reader, otherParams),
      "A reader must rebind with the StripeStreams it was built with.");
}

} // namespace
} // namespace facebook::nimble
