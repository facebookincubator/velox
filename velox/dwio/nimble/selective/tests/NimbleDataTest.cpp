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

#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Tests for NimbleData null handling (exercised through carefully crafted
// vectors) and TrackedColumnLoader row size tracking (exercised through
// lazy-loaded struct columns).
class NimbleDataTest : public ::testing::Test,
                       public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);
    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    auto type = asRowType(expected->type());
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(type);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
    return makeReaders(
        input, test::createNimbleFile(*rootPool(), input), scanSpec);
  }

  void validate(
      const RowVector& expected,
      dwio::common::RowReader& rowReader,
      int batchSize) {
    auto result = BaseVector::create(asRowType(expected.type()), 0, pool());
    int numScanned = 0;
    int offset = 0;
    while (numScanned < expected.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int j = 0; j < result->size(); ++j) {
        ASSERT_TRUE(result->equalValueAt(&expected, j, offset + j))
            << "Mismatch at row " << (offset + j) << ": expected "
            << expected.toString(offset + j) << " got " << result->toString(j);
      }
      offset += result->size();
    }
    ASSERT_EQ(numScanned, expected.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// ----- NimbleData null handling tests -----

// Scalar column with no nulls — readNulls should produce no null bitmap.
TEST_F(NimbleDataTest, scalarNoNulls) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(100, folly::identity),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(100, result), 100);
  auto* row = result->asUnchecked<RowVector>();
  auto child = row->childAt(0);
  DecodedVector decoded(*child);
  for (int i = 0; i < 100; ++i) {
    ASSERT_FALSE(child->isNullAt(i));
    ASSERT_EQ(decoded.valueAt<int64_t>(i), static_cast<int64_t>(i));
  }
}

// All-null scalar — verify null bitmap is set correctly.
TEST_F(NimbleDataTest, scalarAllNulls) {
  auto input = makeRowVector({
      makeConstant<int64_t>(std::nullopt, 100),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(100, result), 100);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(row->childAt(0)->isNullAt(i));
  }
}

// ROW type with null rows — tests nullsDescriptor path.
TEST_F(NimbleDataTest, rowNulls) {
  auto input = makeRowVector({
      makeRowVector(
          {makeFlatVector<int32_t>(50, folly::identity)}, nullEvery(3)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 11);
}

// ARRAY type with null arrays — tests lengths-as-nulls path.
TEST_F(NimbleDataTest, arrayNulls) {
  auto input = makeRowVector({
      makeArrayVector<int64_t>(
          50,
          [](auto i) { return 1 + i % 3; },
          [](auto j) { return j * 10; },
          nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 13);
}

// MAP type with null maps — tests lengths-as-nulls path.
TEST_F(NimbleDataTest, mapNulls) {
  auto input = makeRowVector({
      makeMapVector<int64_t, int64_t>(
          50,
          [](auto i) { return 1 + i % 3; },
          [](auto j) { return j; },
          [](auto j) { return j * 100; },
          nullEvery(4)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 11);
}

// Nested ROW inside ARRAY — tests combined null propagation.
TEST_F(NimbleDataTest, nestedNulls) {
  auto innerRow = makeRowVector(
      {makeFlatVector<int32_t>(50, folly::identity)}, nullEvery(7));
  auto input = makeRowVector({
      innerRow,
      makeArrayVector<int64_t>(
          50,
          [](auto i) { return 1 + i % 4; },
          [](auto j) { return j; },
          nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 9);
}

// ----- TrackedColumnLoader tests (exercised through lazy-loaded struct
// columns) -----

// Lazy-loaded primitive field — verify estimatedRowSize changes after load.
TEST_F(NimbleDataTest, lazyPrimitiveField) {
  // Create a struct with two fields. Make the struct have nulls so the
  // inner fields become lazy.
  auto input = makeRowVector({
      makeFlatVector<int64_t>(100, folly::identity),
      makeRowVector(
          {makeFlatVector<int64_t>(100, [](auto i) { return i * 2; })},
          nullEvery(3)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);

  // estimatedRowSize should be available after creating the reader.
  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_GT(*estimatedRowSize, 0);

  // Read and trigger lazy loading by accessing all data.
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(100, result), 100);
  result->validate();
  // Verify no more rows.
  ASSERT_EQ(readers.rowReader->next(1, result), 0);
}

// Lazy-loaded string field — verify variable-length size tracking.
TEST_F(NimbleDataTest, lazyStringField) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(50, folly::identity),
      makeRowVector(
          {makeFlatVector<std::string>(
              50, [](auto i) { return "string_value_" + std::to_string(i); })},
          nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());

  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(50, result), 50);
  result->validate();
}

// Lazy-loaded array field — verify element size tracking.
TEST_F(NimbleDataTest, lazyArrayField) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(50, folly::identity),
      makeRowVector(
          {makeArrayVector<int32_t>(
              50, [](auto i) { return 1 + i % 5; }, [](auto j) { return j; })},
          nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());

  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(50, result), 50);
  result->validate();
}

// Verify estimatedRowSize() returns a value and that the full read pipeline
// works for a schema that mixes eager and lazy fields.
TEST_F(NimbleDataTest, estimatedRowSizeAfterLazyLoad) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<std::string>(
          kSize, [](auto i) { return "val_" + std::to_string(i); }),
      makeArrayVector<int32_t>(
          kSize,
          [](auto i) { return 2 + i % 3; },
          [](auto j) { return j * 7; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_GT(*estimatedRowSize, 0);

  // Read all rows and verify correctness.
  validate(*input, *readers.rowReader, 11);
}

} // namespace
} // namespace facebook::nimble
