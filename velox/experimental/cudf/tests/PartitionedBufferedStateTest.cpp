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

#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <algorithm>
#include <functional>

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::cudf_velox {
namespace {

struct TableLeafState final : public BufferedState {
  explicit TableLeafState(InputChunk chunk) : chunk(std::move(chunk)) {}

  InputChunk chunk;
};

class IdentityBufferedStateOps final : public BufferedStateOps {
 public:
  IdentityBufferedStateOps(memory::MemoryPool* pool, RowTypePtr rowType)
      : pool_(pool), rowType_(std::move(rowType)), keyIndices_{0} {}

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    return makeOwnedChunk(std::move(rawInput));
  }

  size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeaf(leaf).chunk.size() + input.size();
  }

  std::unique_ptr<BufferedState> createLeaf(InputChunk input) override {
    return std::make_unique<TableLeafState>(std::move(input));
  }

  void addInputToLeaf(BufferedState& leaf, InputChunk input) override {
    auto& tableLeaf = asLeaf(leaf);
    tableLeaf.chunk = mergeChunks(std::move(tableLeaf.chunk), std::move(input));
  }

  size_t leafRowCount(const BufferedState& leaf) const override {
    return asLeaf(leaf).chunk.size();
  }

  uint64_t leafFlatSize(const BufferedState& leaf) const override {
    const auto& chunk = asLeaf(leaf).chunk;
    return chunk.owner ? chunk.owner->estimateFlatSize() : 0;
  }

  std::vector<InputChunk> partitionInput(
      InputChunk input,
      const PartitionSpec& spec) override {
    std::vector<std::vector<int64_t>> buckets(spec.numPartitions);
    for (auto key : extractKeys(input)) {
      auto bucket = partitionFn_(key, spec.seed, spec.numPartitions);
      VELOX_CHECK_GE(bucket, 0);
      VELOX_CHECK_LT(bucket, spec.numPartitions);
      buckets[bucket].push_back(key);
    }

    std::vector<InputChunk> partitions(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (!buckets[i].empty()) {
        partitions[i] = makeChunk_(buckets[i]);
      }
    }
    return partitions;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const PartitionSpec& spec) override {
    auto tableLeaf = std::unique_ptr<TableLeafState>(
        static_cast<TableLeafState*>(leaf.release()));
    auto partitions = partitionInput(std::move(tableLeaf->chunk), spec);

    std::vector<std::unique_ptr<BufferedState>> leaves(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (!partitions[i].empty()) {
        leaves[i] = std::make_unique<TableLeafState>(std::move(partitions[i]));
      }
    }
    return leaves;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto tableLeaf = std::unique_ptr<TableLeafState>(
        static_cast<TableLeafState*>(leaf.release()));
    return tableLeaf->chunk.owner;
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

  void setPartitioning(
      std::function<std::vector<int64_t>(const InputChunk&)> extractKeys,
      std::function<InputChunk(const std::vector<int64_t>&)> makeChunk,
      std::function<int32_t(int64_t, uint32_t, int32_t)> partitionFn) {
    extractKeys_ = std::move(extractKeys);
    makeChunk_ = std::move(makeChunk);
    partitionFn_ = std::move(partitionFn);
  }

 private:
  memory::MemoryPool* pool_;
  RowTypePtr rowType_;
  std::vector<cudf::size_type> keyIndices_;

  std::function<std::vector<int64_t>(const InputChunk&)> extractKeys_;
  std::function<InputChunk(const std::vector<int64_t>&)> makeChunk_;
  std::function<int32_t(int64_t, uint32_t, int32_t)> partitionFn_;

  TableLeafState& asLeaf(BufferedState& leaf) const {
    return static_cast<TableLeafState&>(leaf);
  }

  const TableLeafState& asLeaf(const BufferedState& leaf) const {
    return static_cast<const TableLeafState&>(leaf);
  }

  InputChunk makeOwnedChunk(CudfVectorPtr owner) const {
    return InputChunk{
        owner->pool(),
        rowType_,
        owner->getTableView(),
        owner->stream(),
        std::move(owner)};
  }

  InputChunk mergeChunks(InputChunk left, InputChunk right) const {
    if (left.empty()) {
      return right;
    }
    if (right.empty()) {
      return left;
    }

    auto stream = left.stream;
    std::vector<cudf::table_view> views{left.view, right.view};
    std::vector<rmm::cuda_stream_view> inputStreams{left.stream, right.stream};
    auto mergedTable =
        concatenateViews(views, inputStreams, stream, get_output_mr());
    auto merged = std::make_shared<CudfVector>(
        pool_,
        rowType_,
        mergedTable->num_rows(),
        std::move(mergedTable),
        stream);
    return makeOwnedChunk(std::move(merged));
  }

  std::vector<int64_t> extractKeys(const InputChunk& input) const {
    return extractKeys_(input);
  }
};

class PartitionedBufferedStateTest : public ::testing::Test,
                                     public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
  }

  CudfVectorPtr makeCudfVector(const std::vector<int64_t>& keys) {
    auto row = makeRowVector({"c0"}, {makeFlatVector<int64_t>(keys)});
    auto stream = cudfGlobalStreamPool().get_stream();
    auto table =
        with_arrow::toCudfTable(row, pool_.get(), stream, get_output_mr());
    stream.synchronize();
    return std::make_shared<CudfVector>(
        pool_.get(), rowType_, row->size(), std::move(table), stream);
  }

  InputChunk makeChunk(const std::vector<int64_t>& keys) {
    auto vector = makeCudfVector(keys);
    return InputChunk{
        vector->pool(),
        rowType_,
        vector->getTableView(),
        vector->stream(),
        std::move(vector)};
  }

  std::vector<int64_t> toKeys(const InputChunk& input) {
    auto stream = input.stream;
    auto row = with_arrow::toVeloxColumn(
        input.view, pool_.get(), rowType_, "", stream, get_output_mr());
    stream.synchronize();

    std::vector<int64_t> keys;
    auto* flatKeys = row->childAt(0)->as<FlatVector<int64_t>>();
    keys.reserve(row->size());
    for (vector_size_t i = 0; i < row->size(); ++i) {
      keys.push_back(flatKeys->valueAt(i));
    }
    return keys;
  }

  std::vector<std::vector<int64_t>> drainAll(PartitionedBufferedState& state) {
    std::vector<std::vector<int64_t>> outputs;
    while (auto output = state.drainNextOutput()) {
      auto keys = toKeys(
          InputChunk{
              output->pool(),
              rowType_,
              output->getTableView(),
              output->stream(),
              output});
      std::sort(keys.begin(), keys.end());
      outputs.push_back(std::move(keys));
    }

    std::sort(outputs.begin(), outputs.end());
    return outputs;
  }

  RowTypePtr rowType_{ROW({"c0"}, {BIGINT()})};
};

TEST_F(PartitionedBufferedStateTest, mergesLeafDirectlyBelowCap) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 10, 0);

  state.addInput(makeCudfVector({1, 2}));
  state.addInput(makeCudfVector({3, 4}));

  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3, 4}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, topLevelSplitKeepsRoutingStable) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 3, 0);

  state.addInput(makeCudfVector({0, 2}));
  state.addInput(makeCudfVector({1, 3}));
  state.addInput(makeCudfVector({4, 5}));

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 2, 4}, {1, 3, 5}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, overflowingChildSplitsAgain) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t seed, int32_t numPartitions) {
        auto value = seed == 0 ? key : key / 10;
        return static_cast<int32_t>(value % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 2, 0);

  state.addInput(makeCudfVector({0, 10}));
  state.addInput(makeCudfVector({20, 1}));
  state.addInput(makeCudfVector({30}));

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 20}, {1}, {10, 30}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, noProgressSplitFailsFast) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t /* key */, uint32_t /* seed */, int32_t /* numPartitions */) {
        return 0;
      });
  PartitionedBufferedState state(std::move(ops), 2, 0);

  VELOX_ASSERT_THROW(
      state.addInput(makeCudfVector({1, 2, 3})), "made no progress");
}

} // namespace
} // namespace facebook::velox::cudf_velox
