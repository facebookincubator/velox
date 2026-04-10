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

class IdentityBufferedStateOps final
    : public PartitionedBufferedState::BufferedStateOps {
 public:
  IdentityBufferedStateOps(memory::MemoryPool* pool, RowTypePtr rowType)
      : pool_(pool), rowType_(std::move(rowType)), keyIndices_{0} {}

  CudfVectorPtr compactInputBatch(CudfVectorPtr rawInput) override {
    return rawInput;
  }

  CudfVectorPtr mergeBuffered(CudfVectorPtr left, CudfVectorPtr right) override {
    if (!left) {
      return right;
    }
    if (!right) {
      return left;
    }

    auto stream = left->stream();
    std::vector<CudfVectorPtr> tables;
    tables.push_back(std::move(left));
    tables.push_back(std::move(right));
    auto mergedTable =
        getConcatenatedTable(std::move(tables), rowType_, stream, get_output_mr());
    return std::make_shared<CudfVector>(
        pool_, rowType_, mergedTable->num_rows(), std::move(mergedTable), stream);
  }

  CudfVectorPtr finalizeLeaf(CudfVectorPtr bufferedLeaf) override {
    return bufferedLeaf;
  }

  TypePtr bufferedType() const override {
    return rowType_;
  }

  TypePtr outputType() const override {
    return rowType_;
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

 private:
  memory::MemoryPool* pool_;
  RowTypePtr rowType_;
  std::vector<cudf::size_type> keyIndices_;
};

class LambdaHashPartitioner final
    : public PartitionedBufferedState::HashPartitioner {
 public:
  using ExtractKeys = std::function<std::vector<int64_t>(const CudfVectorPtr&)>;
  using MakeTable = std::function<CudfVectorPtr(const std::vector<int64_t>&)>;
  using PartitionFn = std::function<int32_t(int64_t, uint32_t, int32_t)>;

  LambdaHashPartitioner(
      ExtractKeys extractKeys,
      MakeTable makeTable,
      PartitionFn partitionFn)
      : extractKeys_(std::move(extractKeys)),
        makeTable_(std::move(makeTable)),
        partitionFn_(std::move(partitionFn)) {}

  std::vector<CudfVectorPtr> partition(
      const CudfVectorPtr& input,
      const TypePtr& /* tableType */,
      const PartitionedBufferedState::PartitionSpec& spec) const override {
    std::vector<std::vector<int64_t>> buckets(spec.numPartitions);
    for (auto key : extractKeys_(input)) {
      auto bucket = partitionFn_(key, spec.seed, spec.numPartitions);
      VELOX_CHECK_GE(bucket, 0);
      VELOX_CHECK_LT(bucket, spec.numPartitions);
      buckets[bucket].push_back(key);
    }

    std::vector<CudfVectorPtr> partitions(spec.numPartitions);
    for (int32_t i = 0; i < spec.numPartitions; ++i) {
      if (!buckets[i].empty()) {
        partitions[i] = makeTable_(buckets[i]);
      }
    }
    return partitions;
  }

 private:
  ExtractKeys extractKeys_;
  MakeTable makeTable_;
  PartitionFn partitionFn_;
};

class PartitionedBufferedStateTest : public ::testing::Test, public VectorTestBase {
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
    auto table = with_arrow::toCudfTable(row, pool_.get(), stream, get_output_mr());
    stream.synchronize();
    return std::make_shared<CudfVector>(
        pool_.get(), rowType_, row->size(), std::move(table), stream);
  }

  std::vector<int64_t> toKeys(const CudfVectorPtr& vector) {
    auto stream = vector->stream();
    auto row = with_arrow::toVeloxColumn(
        vector->getTableView(), pool_.get(), rowType_, "", stream, get_output_mr());
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
      auto keys = toKeys(output);
      std::sort(keys.begin(), keys.end());
      outputs.push_back(std::move(keys));
    }

    std::sort(outputs.begin(), outputs.end());
    return outputs;
  }

  RowTypePtr rowType_{ROW({"c0"}, {BIGINT()})};
};

TEST_F(PartitionedBufferedStateTest, mergesLeafDirectlyBelowCap) {
  PartitionedBufferedState state(
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_),
      10,
      std::make_unique<LambdaHashPartitioner>(
          [&](const CudfVectorPtr& input) { return toKeys(input); },
          [&](const std::vector<int64_t>& keys) { return makeCudfVector(keys); },
          [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
            return static_cast<int32_t>(key % numPartitions);
          }),
      0);

  state.addInput(makeCudfVector({1, 2}));
  state.addInput(makeCudfVector({3, 4}));

  EXPECT_EQ(
      drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3, 4}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, topLevelSplitKeepsRoutingStable) {
  PartitionedBufferedState state(
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_),
      3,
      std::make_unique<LambdaHashPartitioner>(
          [&](const CudfVectorPtr& input) { return toKeys(input); },
          [&](const std::vector<int64_t>& keys) { return makeCudfVector(keys); },
          [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
            return static_cast<int32_t>(key % numPartitions);
          }),
      0);

  state.addInput(makeCudfVector({0, 2}));
  state.addInput(makeCudfVector({1, 3}));
  state.addInput(makeCudfVector({4, 5}));

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 2, 4}, {1, 3, 5}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, overflowingChildSplitsAgain) {
  PartitionedBufferedState state(
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_),
      2,
      std::make_unique<LambdaHashPartitioner>(
          [&](const CudfVectorPtr& input) { return toKeys(input); },
          [&](const std::vector<int64_t>& keys) { return makeCudfVector(keys); },
          [](int64_t key, uint32_t seed, int32_t numPartitions) {
            auto value = seed == 0 ? key : key / 10;
            return static_cast<int32_t>(value % numPartitions);
          }),
      0);

  state.addInput(makeCudfVector({0, 10}));
  state.addInput(makeCudfVector({20, 1}));
  state.addInput(makeCudfVector({30}));

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 20}, {1}, {10, 30}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, noProgressSplitFailsFast) {
  PartitionedBufferedState state(
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_),
      2,
      std::make_unique<LambdaHashPartitioner>(
          [&](const CudfVectorPtr& input) { return toKeys(input); },
          [&](const std::vector<int64_t>& keys) { return makeCudfVector(keys); },
          [](int64_t /* key */, uint32_t /* seed */, int32_t /* numPartitions */) {
            return 0;
          }),
      0);

  VELOX_ASSERT_THROW(state.addInput(makeCudfVector({1, 2, 3})), "made no progress");
}

} // namespace
} // namespace facebook::velox::cudf_velox
