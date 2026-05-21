/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/exec/window/VectorWindowPartition.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/WindowFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gmock/gmock.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {
namespace {

class VectorWindowPartitionTest : public testing::Test,
                                  public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  window::VectorWindowPartition makePartition(
      std::vector<column_index_t> inputChannels,
      std::vector<std::pair<column_index_t, core::SortOrder>> sortKeyInfo =
          {}) {
    std::vector<column_index_t> inputMapping(inputChannels.size());
    for (auto i = 0; i < inputChannels.size(); ++i) {
      inputMapping[inputChannels[i]] = i;
    }
    return window::VectorWindowPartition{
        inputChannels, std::move(inputMapping), std::move(sortKeyInfo), pool()};
  }
};

TEST_F(VectorWindowPartitionTest, extractsColumnsAcrossBlocksAndRanges) {
  auto firstBlock = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto secondBlock = makeRowVector({makeFlatVector<int64_t>({5, 6, 7})});

  auto partition = makePartition({0});
  partition.addBlock(firstBlock, 1, 4);
  partition.addBlock(secondBlock, 0, 2);

  auto result = makeFlatVector<int64_t>(5);
  partition.extractColumn(0, 0, 5, 0, result);
  velox::test::assertEqualVectors(
      makeFlatVector<int64_t>({2, 3, 4, 5, 6}), result);
}

TEST_F(VectorWindowPartitionTest, rejectsInvalidBlocks) {
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto partition = makePartition({0});

  VELOX_ASSERT_THROW(partition.addBlock(nullptr, 0, 1), "Input vector");
  VELOX_ASSERT_THROW(partition.addBlock(data, 2, 1), "startRow");
  VELOX_ASSERT_THROW(partition.addBlock(data, 0, 4), "endRow");
}

TEST_F(VectorWindowPartitionTest, extractsRandomRowsAndNullRows) {
  auto firstBlock = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto secondBlock = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});

  auto partition = makePartition({0});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());

  std::vector<vector_size_t> rowNumbers{5, WindowFunction::kNullRow, 0, 3};
  auto result = makeFlatVector<int64_t>(rowNumbers.size());
  partition.extractColumn(
      0, folly::Range(rowNumbers.data(), rowNumbers.size()), 0, result);

  EXPECT_EQ(result->valueAt(0), 6);
  EXPECT_TRUE(result->isNullAt(1));
  EXPECT_EQ(result->valueAt(2), 1);
  EXPECT_EQ(result->valueAt(3), 4);
}

TEST_F(VectorWindowPartitionTest, extractsRandomRowsAfterRemoval) {
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6})});
  std::vector<vector_size_t> rowNumbers{2, 3, WindowFunction::kNullRow};
  auto expected = makeNullableFlatVector<int64_t>(
      std::vector<std::optional<int64_t>>{3, 4, std::nullopt});

  auto vectorPartition = makePartition({0});
  vectorPartition.addBlock(data, 0, data->size());
  vectorPartition.removeProcessedRows(2);
  auto vectorResult = makeFlatVector<int64_t>(rowNumbers.size());
  vectorPartition.extractColumn(
      0, folly::Range(rowNumbers.data(), rowNumbers.size()), 0, vectorResult);
  velox::test::assertEqualVectors(expected, vectorResult);
}

TEST_F(VectorWindowPartitionTest, extractsNullsAcrossBlocksAfterRemoval) {
  auto firstBlock = makeRowVector(
      {"s", "v"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              std::vector<std::optional<int64_t>>{10, std::nullopt, 30}),
      });
  auto secondBlock = makeRowVector(
      {"s", "v"},
      {
          makeFlatVector<int32_t>({4, 5, 6}),
          makeNullableFlatVector<int64_t>(std::vector<std::optional<int64_t>>{
              std::nullopt, 50, std::nullopt}),
      });

  auto partition = makePartition({0, 1}, {{0, core::SortOrder{true, true}}});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());
  partition.removeProcessedRows(2);

  auto nulls = allocateNulls(4, pool_.get());
  partition.extractNulls(1, 2, 4, nulls);
  const auto* rawNulls = nulls->as<uint64_t>();
  EXPECT_FALSE(bits::isBitSet(rawNulls, 0));
  EXPECT_TRUE(bits::isBitSet(rawNulls, 1));
  EXPECT_FALSE(bits::isBitSet(rawNulls, 2));
  EXPECT_TRUE(bits::isBitSet(rawNulls, 3));

  auto frameStarts = AlignedBuffer::allocate<vector_size_t>(2, pool_.get());
  auto frameEnds = AlignedBuffer::allocate<vector_size_t>(2, pool_.get());
  auto* rawFrameStarts = frameStarts->asMutable<vector_size_t>();
  auto* rawFrameEnds = frameEnds->asMutable<vector_size_t>();
  rawFrameStarts[0] = 2;
  rawFrameEnds[0] = 3;
  rawFrameStarts[1] = 4;
  rawFrameEnds[1] = 5;

  auto frameNulls = allocateNulls(0, pool_.get());
  const auto frameNullRange = partition.extractNulls(
      1, SelectivityVector(2), frameStarts, frameEnds, &frameNulls);
  ASSERT_TRUE(frameNullRange.has_value());
  EXPECT_EQ(frameNullRange->first, 2);
  EXPECT_EQ(frameNullRange->second, 4);

  rawNulls = frameNulls->as<uint64_t>();
  EXPECT_FALSE(bits::isBitSet(rawNulls, 0));
  EXPECT_TRUE(bits::isBitSet(rawNulls, 1));
  EXPECT_FALSE(bits::isBitSet(rawNulls, 2));
  EXPECT_TRUE(bits::isBitSet(rawNulls, 3));
}

TEST_F(VectorWindowPartitionTest, computesPeerBuffersAfterRemoval) {
  auto firstBlock = makeRowVector({makeFlatVector<int32_t>({10, 10, 10})});
  auto secondBlock = makeRowVector({makeFlatVector<int32_t>({10, 20})});

  auto partition = makePartition({0}, {{0, core::SortOrder{true, true}}});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());
  partition.removeProcessedRows(2);

  std::vector<vector_size_t> peerStarts(3);
  std::vector<vector_size_t> peerEnds(3);
  const auto peerBounds = partition.computePeerBuffers(
      2, 5, 0, 2, peerStarts.data(), peerEnds.data());

  EXPECT_THAT(peerStarts, ::testing::ElementsAre(0, 0, 4));
  EXPECT_THAT(peerEnds, ::testing::ElementsAre(3, 3, 4));
  EXPECT_EQ(peerBounds.first, 4);
  EXPECT_EQ(peerBounds.second, 5);
}

TEST_F(VectorWindowPartitionTest, previousRowDoesNotRetainProcessedInput) {
  auto partition = makePartition({0}, {{0, core::SortOrder{true, true}}});

  std::weak_ptr<RowVector> processedInput;
  {
    auto firstBlock = makeRowVector({makeFlatVector<int32_t>({10})});
    processedInput = firstBlock;
    partition.addBlock(firstBlock, 0, firstBlock->size());
  }

  partition.removeProcessedRows(1);
  EXPECT_TRUE(processedInput.expired());

  auto secondBlock = makeRowVector({makeFlatVector<int32_t>({10})});
  partition.addBlock(secondBlock, 0, secondBlock->size());

  std::vector<vector_size_t> peerStarts(1);
  std::vector<vector_size_t> peerEnds(1);
  const auto peerBounds = partition.computePeerBuffers(
      1, 2, 0, 1, peerStarts.data(), peerEnds.data());

  EXPECT_THAT(peerStarts, ::testing::ElementsAre(0));
  EXPECT_THAT(peerEnds, ::testing::ElementsAre(1));
  EXPECT_EQ(peerBounds.first, 0);
  EXPECT_EQ(peerBounds.second, 2);
}

TEST_F(
    VectorWindowPartitionTest,
    skipsEmptyBlocksAndExtractsAfterRepeatedRemoval) {
  auto firstBlock = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto secondBlock = makeRowVector({makeFlatVector<int64_t>({4, 5})});

  auto partition = makePartition({0});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, 0);
  partition.addBlock(secondBlock, 2, 2);
  partition.removeProcessedRows(2);
  partition.addBlock(secondBlock, 0, secondBlock->size());
  partition.removeProcessedRows(2);

  EXPECT_EQ(partition.numRows(), 1);
  auto result = makeFlatVector<int64_t>(1);
  partition.extractColumn(0, 4, 1, 0, result);
  velox::test::assertEqualVectors(makeFlatVector<int64_t>({5}), result);
}

TEST_F(
    VectorWindowPartitionTest,
    computesKRangeFrameBoundsAfterRemovalFromSecondBlock) {
  auto firstBlock = makeRowVector(
      {"s", "bound"},
      {
          makeFlatVector<int64_t>({10, 20}),
          makeFlatVector<int64_t>({5, 15}),
      });
  auto secondBlock = makeRowVector(
      {"s", "bound"},
      {
          makeFlatVector<int64_t>({30, 40, 50}),
          makeFlatVector<int64_t>({25, 35, 45}),
      });

  auto partition = makePartition({0, 1}, {{0, core::SortOrder{true, true}}});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());
  partition.removeProcessedRows(2);

  std::vector<vector_size_t> peerStarts(3);
  std::vector<vector_size_t> peerEnds(3);
  partition.computePeerBuffers(2, 5, 0, 0, peerStarts.data(), peerEnds.data());

  std::vector<vector_size_t> frameBounds(3);
  SelectivityVector validFrames(3, true);
  partition.computeKRangeFrameBounds(
      true, true, 1, 2, 3, peerStarts.data(), frameBounds.data(), validFrames);

  EXPECT_THAT(peerStarts, ::testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(peerEnds, ::testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(frameBounds, ::testing::ElementsAre(2, 3, 4));
  EXPECT_TRUE(validFrames.isAllSelected());
}

TEST_F(VectorWindowPartitionTest, computesKRangeFrameBoundsForNullOrderValues) {
  auto firstBlock = makeRowVector(
      {"s", "bound"},
      {
          makeNullableFlatVector<int64_t>(
              std::vector<std::optional<int64_t>>{std::nullopt, std::nullopt}),
          makeNullableFlatVector<int64_t>(
              std::vector<std::optional<int64_t>>{std::nullopt, std::nullopt}),
      });
  auto secondBlock = makeRowVector(
      {"s", "bound"},
      {
          makeNullableFlatVector<int64_t>(
              std::vector<std::optional<int64_t>>{10, 20}),
          makeNullableFlatVector<int64_t>(
              std::vector<std::optional<int64_t>>{10, 15}),
      });

  auto partition = makePartition({0, 1}, {{0, core::SortOrder{true, true}}});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());

  std::vector<vector_size_t> peerStarts(4);
  std::vector<vector_size_t> peerEnds(4);
  partition.computePeerBuffers(0, 4, 0, 0, peerStarts.data(), peerEnds.data());

  std::vector<vector_size_t> frameBounds(4);
  SelectivityVector validFrames(4, true);
  partition.computeKRangeFrameBounds(
      true, true, 1, 0, 4, peerStarts.data(), frameBounds.data(), validFrames);

  EXPECT_THAT(peerStarts, ::testing::ElementsAre(0, 0, 2, 3));
  EXPECT_THAT(peerEnds, ::testing::ElementsAre(1, 1, 2, 3));
  EXPECT_THAT(frameBounds, ::testing::ElementsAre(0, 0, 2, 3));
  EXPECT_TRUE(validFrames.isAllSelected());
}

TEST_F(VectorWindowPartitionTest, computesKRangeFrameBoundsAcrossBlocks) {
  auto firstBlock = makeRowVector(
      {"s", "bound"},
      {
          makeFlatVector<int32_t>({1, 1}),
          makeFlatVector<int32_t>({1, 1}),
      });
  auto secondBlock = makeRowVector(
      {"s", "bound"},
      {
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int32_t>({2, 3}),
      });

  auto partition = makePartition({0, 1}, {{0, core::SortOrder{true, true}}});
  partition.addBlock(firstBlock, 0, firstBlock->size());
  partition.addBlock(secondBlock, 0, secondBlock->size());

  std::vector<vector_size_t> peerStarts(4);
  std::vector<vector_size_t> peerEnds(4);
  partition.computePeerBuffers(0, 4, 0, 0, peerStarts.data(), peerEnds.data());

  std::vector<vector_size_t> frameBounds(4);
  SelectivityVector validFrames(4, true);
  partition.computeKRangeFrameBounds(
      true, false, 1, 0, 4, peerStarts.data(), frameBounds.data(), validFrames);

  EXPECT_THAT(frameBounds, ::testing::ElementsAre(0, 0, 2, 3));
  EXPECT_TRUE(validFrames.isAllSelected());
}

} // namespace
} // namespace facebook::velox::exec::test
