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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/writer/BufferPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble {
namespace {

// Rows of the writer test: an id of 8 bytes and 64 floats of 4 bytes, 264 raw
// bytes per row.
constexpr int32_t kNumRows{100};
constexpr int32_t kDimension{64};
constexpr uint64_t kRowRawSize{8 + kDimension * 4};

class BufferPolicyTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(leafPool_.get());
  }

  // Returns 'numRows' rows of one BIGINT column, 8 raw bytes per row.
  velox::RowVectorPtr makeBigintRows(int64_t numRows) {
    std::vector<int64_t> values(numRows);
    std::iota(values.begin(), values.end(), 0);
    return vectorMaker_->rowVector({vectorMaker_->flatVector<int64_t>(values)});
  }

  // Returns the row count of each stripe flushInput() has ready.
  static std::vector<uint32_t> drainRowCounts(BufferPolicy& policy) {
    std::vector<uint32_t> rowCounts;
    for (auto range = policy.flushInput(); !range.empty();
         range = policy.flushInput()) {
      uint32_t rows{0};
      for (const auto& rowRange : range.rowRanges) {
        rows += rowRange.numRows();
      }
      rowCounts.emplace_back(rows);
    }
    return rowCounts;
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(BufferPolicyTest, cutsBatchLargerThanTarget) {
  // 10 rows of 8 bytes reach the 80-byte target.
  StripeRawSizeBufferPolicy policy{80};
  policy.bufferInput(makeBigintRows(25));
  EXPECT_THAT(drainRowCounts(policy), ::testing::ElementsAre(10, 10));

  policy.finalize();
  EXPECT_THAT(drainRowCounts(policy), ::testing::ElementsAre(5));
}

TEST_F(BufferPolicyTest, accumulatesBatchesSmallerThanTarget) {
  StripeRawSizeBufferPolicy policy{80};
  const auto first = makeBigintRows(4);
  const auto second = makeBigintRows(4);
  const auto third = makeBigintRows(4);
  policy.bufferInput(first);
  policy.bufferInput(second);
  EXPECT_TRUE(drainRowCounts(policy).empty());

  // The third batch completes the stripe with 2 of its 4 rows.
  policy.bufferInput(third);
  const auto range = policy.flushInput();
  ASSERT_EQ(range.inputs.size(), 3);
  EXPECT_EQ(range.inputs[0], first);
  EXPECT_EQ(range.inputs[1], second);
  EXPECT_EQ(range.inputs[2], third);
  EXPECT_EQ(range.rowRanges[0], (RowRange{0, 4}));
  EXPECT_EQ(range.rowRanges[1], (RowRange{0, 4}));
  EXPECT_EQ(range.rowRanges[2], (RowRange{0, 2}));
  EXPECT_TRUE(policy.flushInput().empty());

  policy.finalize();
  EXPECT_THAT(drainRowCounts(policy), ::testing::ElementsAre(2));
}

TEST_F(BufferPolicyTest, rowLargerThanTargetIsItsOwnStripe) {
  StripeRawSizeBufferPolicy policy{4};
  policy.bufferInput(makeBigintRows(3));
  EXPECT_THAT(drainRowCounts(policy), ::testing::ElementsAre(1, 1, 1));

  policy.finalize();
  EXPECT_TRUE(drainRowCounts(policy).empty());
}

TEST_F(BufferPolicyTest, rejectsZeroTarget) {
  NIMBLE_ASSERT_USER_THROW(
      StripeRawSizeBufferPolicy{0}, "Stripe raw size must be positive");
}

TEST_F(BufferPolicyTest, rejectsNullOrEmptyInput) {
  StripeRawSizeBufferPolicy policy{80};
  NIMBLE_ASSERT_USER_THROW(
      policy.bufferInput(nullptr), "Buffered input must not be null");
  NIMBLE_ASSERT_USER_THROW(
      policy.bufferInput(makeBigintRows(0)),
      "Buffered input must not be empty");
}

TEST_F(BufferPolicyTest, writerCutsStripesInsideBatch) {
  const auto batch = vectorMaker_->rowVector(
      {"id", "vector"},
      {vectorMaker_->flatVector<int64_t>(
           kNumRows, [](auto row) { return row; }),
       vectorMaker_->arrayVector<float>(
           kNumRows,
           [](auto /*row*/) { return kDimension; },
           [](auto index) { return static_cast<float>(index) / 7; })});

  WriterOptions options;
  options.bufferPolicyFactory = [] {
    return std::make_unique<StripeRawSizeBufferPolicy>(10 * kRowRawSize);
  };
  std::string file;
  Writer writer{
      batch->type(),
      std::make_unique<velox::InMemoryWriteFile>(&file),
      *rootPool_,
      std::move(options)};
  writer.write(batch);
  writer.close();

  velox::InMemoryReadFile readFile{file};
  BatchReader reader{&readFile, *leafPool_};
  const auto& tablet = reader.tabletReader();
  std::vector<uint32_t> stripeRowCounts;
  stripeRowCounts.reserve(tablet.stripeCount());
  for (uint32_t stripe = 0; stripe < tablet.stripeCount(); ++stripe) {
    stripeRowCounts.emplace_back(tablet.stripeRowCount(stripe));
  }
  EXPECT_EQ(stripeRowCounts, std::vector<uint32_t>(10, 10));

  // Each next() returns at most the rest of the current stripe.
  velox::VectorPtr result;
  int32_t offset{0};
  while (reader.next(kNumRows, result)) {
    for (velox::vector_size_t row = 0; row < result->size(); ++row) {
      EXPECT_TRUE(result->equalValueAt(batch.get(), row, offset + row))
          << "row " << offset + row;
    }
    offset += result->size();
  }
  EXPECT_EQ(offset, kNumRows);
}

} // namespace
} // namespace facebook::nimble
