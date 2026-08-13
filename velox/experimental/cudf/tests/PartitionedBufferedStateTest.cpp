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

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/copying.hpp>

#include <algorithm>
#include <functional>
#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::cudf_velox {
namespace {

struct IdentityBufferedStateOpsStats {
  const BufferedState* firstPartitionedLeaf{nullptr};
  size_t originalLeafPartitionCalls{0};
  bool oldLeafAliveOnRetries{true};
  bool oldLeafDestroyed{false};
  bool oldLeafDestroyedBeforeChildCreation{true};
  size_t createLeafCallsAfterPartition{0};
  size_t createLeafFromInputsCalls{0};
  size_t addInputCalls{0};
  size_t addInputCallsAfterPartition{0};
  std::vector<size_t> spilledLeafRows;
  std::vector<size_t> restoredLeafRows;
  std::vector<const BufferedState*> spilledLeaves;
  const BufferedState* activeLeafDuringCallback{nullptr};
  uint64_t pinnedHostBytes{0};
};

struct TableLeafState final : public BufferedState {
  TableLeafState(
      InputChunk chunk,
      std::shared_ptr<IdentityBufferedStateOpsStats> stats)
      : rowCount(chunk.size()),
        chunk(std::move(chunk)),
        stats(std::move(stats)) {}

  ~TableLeafState() override {
    if (stats && stats->firstPartitionedLeaf == this) {
      stats->oldLeafDestroyed = true;
    }
  }

  bool isResident() const {
    return chunk.owner != nullptr;
  }

  size_t rowCount;
  InputChunk chunk;
  std::optional<SpilledCudfVector> spilled;
  std::shared_ptr<IdentityBufferedStateOpsStats> stats;
};

class IdentityBufferedStateOps final : public BufferedStateOps {
 public:
  IdentityBufferedStateOps(
      memory::MemoryPool* pool,
      RowTypePtr rowType,
      std::shared_ptr<IdentityBufferedStateOpsStats> stats = nullptr)
      : pool_(pool),
        rowType_(std::move(rowType)),
        keyIndices_{0},
        stats_(std::move(stats)) {}

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    return makeOwnedChunk(std::move(rawInput));
  }

  size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeaf(leaf).rowCount + input.size();
  }

  std::unique_ptr<BufferedState> createLeaf(InputChunk input) override {
    recordChildCreation();
    return std::make_unique<TableLeafState>(
        std::move(input).materialize(get_output_mr()), stats_);
  }

  std::unique_ptr<BufferedState> createLeafFromInputs(
      InputChunk first,
      InputChunk second) override {
    if (stats_ && stats_->firstPartitionedLeaf != nullptr) {
      ++stats_->createLeafFromInputsCalls;
      stats_->oldLeafDestroyedBeforeChildCreation &= stats_->oldLeafDestroyed;
    }
    return std::make_unique<TableLeafState>(
        mergeChunks(std::move(first), std::move(second)), stats_);
  }

  void addInputToLeaf(BufferedState& leaf, InputChunk input) override {
    if (leafOperationCallback_) {
      leafOperationCallback_(leaf);
    }
    if (stats_) {
      ++stats_->addInputCalls;
      if (stats_->oldLeafDestroyed) {
        ++stats_->addInputCallsAfterPartition;
      }
    }
    auto& tableLeaf = asLeaf(leaf);
    tableLeaf.chunk = mergeChunks(std::move(tableLeaf.chunk), std::move(input));
    tableLeaf.rowCount = tableLeaf.chunk.size();
  }

  size_t leafRowCount(const BufferedState& leaf) const override {
    return asLeaf(leaf).rowCount;
  }

  uint64_t leafFlatSize(const BufferedState& leaf) const override {
    const auto& chunk = asLeaf(leaf).chunk;
    return chunk.owner ? chunk.owner->estimateFlatSize() : 0;
  }

  uint64_t leafReclaimableBytes(const BufferedState& leaf) const override {
    const auto& tableLeaf = asLeaf(leaf);
    // Row count is a deterministic stand-in for bytes in PBS policy tests.
    return spillingEnabled_ && tableLeaf.isResident() ? tableLeaf.rowCount : 0;
  }

  void spillLeaf(BufferedState& leaf) override {
    auto& tableLeaf = asLeaf(leaf);
    VELOX_CHECK(spillingEnabled_);
    VELOX_CHECK(tableLeaf.isResident());
    VELOX_CHECK(!tableLeaf.spilled.has_value());
    try {
      auto spilled = SpilledCudfVector::spill(tableLeaf.chunk.owner, pool_);
      tableLeaf.chunk = InputChunk{};
      tableLeaf.spilled.emplace(std::move(spilled));
    } catch (...) {
      if (tableLeaf.chunk.owner) {
        tableLeaf.chunk.view = tableLeaf.chunk.owner->getTableView();
        tableLeaf.chunk.stream = tableLeaf.chunk.owner->stream();
      }
      throw;
    }
    if (stats_) {
      stats_->spilledLeafRows.push_back(tableLeaf.rowCount);
      stats_->spilledLeaves.push_back(&leaf);
      stats_->pinnedHostBytes += tableLeaf.spilled->hostBytes();
    }
  }

  void restoreLeaf(BufferedState& leaf) override {
    auto& tableLeaf = asLeaf(leaf);
    if (tableLeaf.isResident()) {
      return;
    }
    VELOX_CHECK(tableLeaf.spilled.has_value());
    auto restored = tableLeaf.spilled->restore(get_output_mr());
    tableLeaf.chunk = makeOwnedChunk(std::move(restored));
    tableLeaf.spilled.reset();
    if (stats_) {
      stats_->restoredLeafRows.push_back(tableLeaf.rowCount);
    }
  }

  std::vector<InputChunk> partitionInput(
      const InputChunk& input,
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

  std::vector<InputChunk> partitionLeaf(
      const BufferedState& leaf,
      const PartitionSpec& spec) override {
    if (stats_ && !stats_->oldLeafDestroyed) {
      if (stats_->firstPartitionedLeaf == nullptr) {
        stats_->firstPartitionedLeaf = &leaf;
      } else if (stats_->firstPartitionedLeaf != &leaf) {
        stats_->oldLeafAliveOnRetries = false;
      }
      stats_->oldLeafAliveOnRetries &= !stats_->oldLeafDestroyed;
      ++stats_->originalLeafPartitionCalls;
    }
    return partitionInput(asLeaf(leaf).chunk, spec);
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto tableLeaf = std::unique_ptr<TableLeafState>(
        static_cast<TableLeafState*>(leaf.release()));
    return std::move(tableLeaf->chunk.owner);
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

  void enableSpilling() {
    spillingEnabled_ = true;
  }

  void enableDeduplicatingMerges() {
    deduplicateMerges_ = true;
  }

  void setLeafOperationCallback(
      std::function<void(const BufferedState&)> callback) {
    leafOperationCallback_ = std::move(callback);
  }

 private:
  memory::MemoryPool* pool_;
  RowTypePtr rowType_;
  std::vector<cudf::size_type> keyIndices_;
  std::shared_ptr<IdentityBufferedStateOpsStats> stats_;

  std::function<std::vector<int64_t>(const InputChunk&)> extractKeys_;
  std::function<InputChunk(const std::vector<int64_t>&)> makeChunk_;
  std::function<int32_t(int64_t, uint32_t, int32_t)> partitionFn_;
  std::function<void(const BufferedState&)> leafOperationCallback_;

  bool spillingEnabled_{false};
  bool deduplicateMerges_{false};

  TableLeafState& asLeaf(BufferedState& leaf) const {
    return static_cast<TableLeafState&>(leaf);
  }

  const TableLeafState& asLeaf(const BufferedState& leaf) const {
    return static_cast<const TableLeafState&>(leaf);
  }

  void recordChildCreation() {
    if (stats_ && stats_->firstPartitionedLeaf != nullptr) {
      ++stats_->createLeafCallsAfterPartition;
      stats_->oldLeafDestroyedBeforeChildCreation &= stats_->oldLeafDestroyed;
    }
  }

  InputChunk makeOwnedChunk(CudfVectorPtr owner) const {
    return InputChunk{
        owner->pool(),
        rowType_,
        owner->getTableView(),
        owner->stream(),
        std::move(owner),
        nullptr,
        InputChunkStorage::kOwned};
  }

  InputChunk mergeChunks(InputChunk left, InputChunk right) const {
    if (left.empty()) {
      return right;
    }
    if (right.empty()) {
      return left;
    }

    if (deduplicateMerges_) {
      auto keys = extractKeys(left);
      auto rightKeys = extractKeys(right);
      keys.insert(keys.end(), rightKeys.begin(), rightKeys.end());
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      return makeChunk_(keys);
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

  CudfVectorPtr makeCudfVectorFromRow(RowVectorPtr row) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto table =
        with_arrow::toCudfTable(row, pool_.get(), stream, get_output_mr());
    stream.synchronize();
    return std::make_shared<CudfVector>(
        pool_.get(), row->type(), row->size(), std::move(table), stream);
  }

  CudfVectorPtr makeCudfVector(const std::vector<int64_t>& keys) {
    return makeCudfVectorFromRow(
        makeRowVector({"c0"}, {makeFlatVector<int64_t>(keys)}));
  }

  InputChunk makeChunk(const std::vector<int64_t>& keys) {
    auto vector = makeCudfVector(keys);
    return InputChunk{
        vector->pool(),
        rowType_,
        vector->getTableView(),
        vector->stream(),
        std::move(vector),
        nullptr,
        InputChunkStorage::kOwned};
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

  std::vector<int64_t> toKeys(const CudfVectorPtr& output) {
    return toKeys(
        InputChunk{
            output->pool(),
            rowType_,
            output->getTableView(),
            output->stream(),
            output,
            nullptr,
            InputChunkStorage::kOwned});
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

TEST_F(
    PartitionedBufferedStateTest,
    materializedSiblingViewsOwnIndependentAllocations) {
  ASSERT_TRUE(output_mr_.has_value());
  auto trackingRoot = memory::memoryManager()->addRootPool();
  auto trackingPool = trackingRoot->addLeafChild("ownedGroupbyLeaves");
  CudfMemoryResource reportingResource{*output_mr_, trackingPool};
  auto reportingRef = rmm::device_async_resource_ref{reportingResource};
  ScopedCudfMemoryResources scopedResources{reportingRef, reportingRef};

  auto parentVector = makeCudfVector({1, 2, 3, 4});
  auto stream = parentVector->stream();
  auto parentOwner = std::shared_ptr<cudf::table>(parentVector->release());
  parentVector.reset();
  std::weak_ptr<cudf::table> parentWeak = parentOwner;

  std::vector<cudf::size_type> splitPoints{2};
  auto partitionViews = cudf::split(parentOwner->view(), splitPoints, stream);
  ASSERT_EQ(partitionViews.size(), 2);

  auto left = InputChunk{
      pool_.get(),
      rowType_,
      partitionViews[0],
      stream,
      nullptr,
      parentOwner,
      InputChunkStorage::kBorrowed};
  auto right = InputChunk{
      pool_.get(),
      rowType_,
      partitionViews[1],
      stream,
      nullptr,
      parentOwner,
      InputChunkStorage::kBorrowed};
  parentOwner.reset();

  auto leftOwned = std::move(left).materialize(reportingRef);
  auto rightOwned = std::move(right).materialize(reportingRef);
  stream.synchronize();

  EXPECT_TRUE(parentWeak.expired());
  ASSERT_TRUE(leftOwned.ownsFullTable());
  ASSERT_TRUE(rightOwned.ownsFullTable());
  ASSERT_EQ(leftOwned.owner.use_count(), 1);
  ASSERT_EQ(rightOwned.owner.use_count(), 1);
  EXPECT_NE(
      leftOwned.view.column(0).head<void>(),
      rightOwned.view.column(0).head<void>());

  const auto leftBytes = leftOwned.owner->estimateFlatSize();
  const auto rightBytes = rightOwned.owner->estimateFlatSize();
  ASSERT_GT(leftBytes, 0);
  ASSERT_GT(rightBytes, 0);
  EXPECT_EQ(trackingPool->usedBytes(), leftBytes + rightBytes);

  leftOwned = InputChunk{};
  stream.synchronize();
  EXPECT_EQ(trackingPool->usedBytes(), rightBytes);
  EXPECT_EQ(toKeys(rightOwned), (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(trackingPool->usedBytes(), rightBytes);

  rightOwned = InputChunk{};
  stream.synchronize();
  EXPECT_EQ(trackingPool->usedBytes(), 0);
}

TEST_F(PartitionedBufferedStateTest, spilledCudfVectorTracksAndRestoresBytes) {
  ASSERT_TRUE(output_mr_.has_value());
  auto gpuRoot = memory::memoryManager()->addRootPool();
  auto gpuPool = gpuRoot->addLeafChild("spilledCudfVectorGpu");
  CudfMemoryResource reportingResource{*output_mr_, gpuPool};
  auto reportingRef = rmm::device_async_resource_ref{reportingResource};
  ScopedCudfMemoryResources scopedResources{reportingRef, reportingRef};

  auto hostRoot = memory::memoryManager()->addRootPool();
  auto hostPool = hostRoot->addLeafChild("spilledCudfVectorHost");
  auto resident = makeCudfVector({1, 2, 3, 4});
  auto stream = resident->stream();
  const auto residentBytes = resident->estimateFlatSize();
  ASSERT_GT(residentBytes, 0);
  ASSERT_EQ(gpuPool->usedBytes(), residentBytes);
  ASSERT_EQ(hostPool->usedBytes(), 0);

  CudfVectorPtr restored;
  {
    auto spilled = SpilledCudfVector::spill(resident, hostPool.get());
    EXPECT_EQ(resident, nullptr);
    EXPECT_EQ(spilled.deviceBytes(), residentBytes);
    EXPECT_EQ(spilled.hostBytes(), residentBytes);
    EXPECT_EQ(gpuPool->usedBytes(), 0);
    EXPECT_EQ(hostPool->usedBytes(), residentBytes);

    restored = spilled.restore(reportingRef);
    EXPECT_EQ(gpuPool->usedBytes(), residentBytes);
    EXPECT_EQ(hostPool->usedBytes(), residentBytes);
    EXPECT_EQ(toKeys(restored), (std::vector<int64_t>{1, 2, 3, 4}));
  }

  EXPECT_EQ(hostPool->usedBytes(), 0);
  EXPECT_EQ(gpuPool->usedBytes(), residentBytes);
  restored.reset();
  stream.synchronize();
  EXPECT_EQ(gpuPool->usedBytes(), 0);
}

TEST_F(
    PartitionedBufferedStateTest,
    spilledCudfVectorPreservesStringsAndNulls) {
  ASSERT_TRUE(output_mr_.has_value());
  auto gpuRoot = memory::memoryManager()->addRootPool();
  auto gpuPool = gpuRoot->addLeafChild("spilledStringsGpu");
  CudfMemoryResource reportingResource{*output_mr_, gpuPool};
  auto reportingRef = rmm::device_async_resource_ref{reportingResource};
  ScopedCudfMemoryResources scopedResources{reportingRef, reportingRef};

  auto hostRoot = memory::memoryManager()->addRootPool();
  auto hostPool = hostRoot->addLeafChild("spilledStringsHost");
  auto expected = makeRowVector(
      {"c0", "c1"},
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4}),
       makeNullableFlatVector<std::string>(
           {"one", "a longer value", std::nullopt, ""})});
  auto resident = makeCudfVectorFromRow(expected);
  auto stream = resident->stream();
  const auto residentBytes = resident->estimateFlatSize();
  ASSERT_GT(residentBytes, 0);
  ASSERT_EQ(gpuPool->usedBytes(), residentBytes);

  CudfVectorPtr restored;
  {
    auto spilled = SpilledCudfVector::spill(resident, hostPool.get());
    EXPECT_EQ(spilled.deviceBytes(), residentBytes);
    EXPECT_EQ(spilled.hostBytes(), residentBytes);
    EXPECT_EQ(gpuPool->usedBytes(), 0);
    EXPECT_EQ(hostPool->usedBytes(), residentBytes);
    restored = spilled.restore(reportingRef);
  }

  EXPECT_EQ(hostPool->usedBytes(), 0);
  ASSERT_NE(restored, nullptr);
  auto actual = with_arrow::toVeloxColumn(
      restored->getTableView(),
      pool_.get(),
      expected->type(),
      restored->stream(),
      reportingRef);
  restored->stream().synchronize();
  assertEqualVectors(expected, actual);

  restored.reset();
  stream.synchronize();
  EXPECT_EQ(gpuPool->usedBytes(), 0);
}

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

TEST_F(
    PartitionedBufferedStateTest,
    reclaimsLargestLeavesFirstAndRestoresOnDrain) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  ops->enableSpilling();
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 10, 0);

  state.addInput(makeCudfVector({0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5}));
  EXPECT_EQ(state.reclaimableBytes(), 11);

  EXPECT_EQ(state.reclaim(1), 8);
  ASSERT_EQ(stats->spilledLeafRows.size(), 1);
  EXPECT_EQ(stats->spilledLeafRows[0], 8);
  EXPECT_GT(stats->pinnedHostBytes, 0);
  EXPECT_EQ(state.reclaimableBytes(), 3);

  EXPECT_EQ(state.reclaim(0), 3);
  EXPECT_EQ(stats->spilledLeafRows, (std::vector<size_t>{8, 3}));
  EXPECT_EQ(state.reclaimableBytes(), 0);

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{
          {0, 2, 4, 6, 8, 10, 12, 14}, {1, 3, 5}}));
  EXPECT_EQ(stats->restoredLeafRows, (std::vector<size_t>{8, 3}));
  EXPECT_TRUE(state.empty());
}

TEST_F(
    PartitionedBufferedStateTest,
    defaultPolicyRestoresSpilledLeafBeforeAddingInput) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  ops->enableSpilling();
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 10, 0);

  state.addInput(makeCudfVector({1, 2}));
  EXPECT_EQ(state.reclaim(0), 2);
  EXPECT_EQ(state.reclaimableBytes(), 0);

  state.addInput(makeCudfVector({3}));
  EXPECT_EQ(stats->restoredLeafRows, (std::vector<size_t>{2}));
  EXPECT_EQ(state.reclaimableBytes(), 3);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3}}));
}

TEST_F(PartitionedBufferedStateTest, reentrantReclaimExcludesTheActiveLeaf) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  auto* opsPtr = ops.get();
  ops->enableSpilling();
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 3, 0);
  state.addInput(makeCudfVector({0, 1, 2, 3}));
  ASSERT_EQ(state.reclaimableBytes(), 4);

  opsPtr->setLeafOperationCallback([&](const BufferedState& activeLeaf) {
    stats->activeLeafDuringCallback = &activeLeaf;
    EXPECT_EQ(state.reclaimableBytes(), 2);
    EXPECT_EQ(state.reclaim(0), 2);
  });
  state.addInput(makeCudfVector({4}));

  ASSERT_EQ(stats->spilledLeaves.size(), 1);
  EXPECT_NE(stats->spilledLeaves[0], stats->activeLeafDuringCallback);
  EXPECT_EQ(state.reclaimableBytes(), 3);
  EXPECT_EQ(
      drainAll(state), (std::vector<std::vector<int64_t>>{{0, 2, 4}, {1, 3}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(
    PartitionedBufferedStateTest,
    successfulSplitReleasesOldLeafBeforeCreatingChildren) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t /* seed */, int32_t numPartitions) {
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 3, 0);

  state.addInput(makeCudfVector({0, 1}));
  state.addInput(makeCudfVector({2, 3}));

  EXPECT_EQ(stats->originalLeafPartitionCalls, 1);
  EXPECT_TRUE(stats->oldLeafDestroyed);
  EXPECT_TRUE(stats->oldLeafDestroyedBeforeChildCreation);
  EXPECT_EQ(stats->createLeafFromInputsCalls, 2);
  EXPECT_EQ(stats->createLeafCallsAfterPartition, 0);

  state.addInput(makeCudfVector({4, 5}));
  EXPECT_EQ(stats->addInputCallsAfterPartition, 2);
  EXPECT_EQ(stats->createLeafCallsAfterPartition, 0);

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

TEST_F(PartitionedBufferedStateTest, noProgressSplitRetriesNewSeeds) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  std::vector<uint32_t> seeds;
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [&](int64_t key, uint32_t seed, int32_t numPartitions) {
        if (seeds.empty() || seeds.back() != seed) {
          seeds.push_back(seed);
        }
        if (seed < 2) {
          return 0;
        }
        return static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 2, 0);

  state.addInput(makeCudfVector({1, 2, 3}));

  EXPECT_EQ(seeds, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_EQ(stats->originalLeafPartitionCalls, 3);
  EXPECT_TRUE(stats->oldLeafAliveOnRetries);
  EXPECT_TRUE(stats->oldLeafDestroyed);
  EXPECT_TRUE(stats->oldLeafDestroyedBeforeChildCreation);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 3}, {2}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(
    PartitionedBufferedStateTest,
    noProgressSplitAggregatesOverlappingInputBeforeFailing) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  ops->enableDeduplicatingMerges();
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t /* key */, uint32_t /* seed */, int32_t /* partitions */) {
        return 0;
      });
  PartitionedBufferedState state(std::move(ops), 1, 0);

  state.addInput(makeCudfVector({7}));
  state.addInput(makeCudfVector({7}));

  EXPECT_EQ(stats->originalLeafPartitionCalls, 4);
  EXPECT_EQ(stats->addInputCalls, 1);
  EXPECT_FALSE(stats->oldLeafDestroyed);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{7}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(
    PartitionedBufferedStateTest,
    noProgressSplitPartitionsCompactedStateWhenItStillExceedsLimit) {
  auto stats = std::make_shared<IdentityBufferedStateOpsStats>();
  auto ops =
      std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_, stats);
  ops->setPartitioning(
      [&](const InputChunk& input) { return toKeys(input); },
      [&](const std::vector<int64_t>& keys) { return makeChunk(keys); },
      [](int64_t key, uint32_t seed, int32_t numPartitions) {
        return seed < 4 ? 0 : static_cast<int32_t>(key % numPartitions);
      });
  PartitionedBufferedState state(std::move(ops), 2, 0);

  state.addInput(makeCudfVector({1, 2}));
  state.addInput(makeCudfVector({3}));

  EXPECT_EQ(stats->originalLeafPartitionCalls, 5);
  EXPECT_EQ(stats->addInputCalls, 1);
  EXPECT_TRUE(stats->oldLeafDestroyed);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 3}, {2}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, flushableStateEmitsAtEnd) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  FlushableBufferedState state(
      std::move(ops), 10, std::numeric_limits<uint64_t>::max());

  EXPECT_TRUE(state.empty());
  EXPECT_EQ(state.getOutput(false), nullptr);

  state.addInput(makeCudfVector({1, 2}));

  EXPECT_FALSE(state.empty());
  EXPECT_EQ(state.getOutput(false), nullptr);

  auto output = state.getOutput(true);
  ASSERT_NE(output, nullptr);

  auto keys = toKeys(output);
  std::sort(keys.begin(), keys.end());
  EXPECT_EQ(keys, (std::vector<int64_t>{1, 2}));
  EXPECT_TRUE(state.empty());
  EXPECT_EQ(state.getOutput(true), nullptr);
}

TEST_F(PartitionedBufferedStateTest, flushableStateFlushesBeforeRowLimitMerge) {
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  FlushableBufferedState state(
      std::move(ops), 3, std::numeric_limits<uint64_t>::max());

  state.addInput(makeCudfVector({1, 2}));
  state.addInput(makeCudfVector({3, 4}));

  EXPECT_FALSE(state.empty());

  auto firstOutput = state.getOutput(false);
  ASSERT_NE(firstOutput, nullptr);
  auto firstKeys = toKeys(firstOutput);
  std::sort(firstKeys.begin(), firstKeys.end());
  EXPECT_EQ(firstKeys, (std::vector<int64_t>{1, 2}));

  EXPECT_FALSE(state.empty());
  EXPECT_EQ(state.getOutput(false), nullptr);

  auto secondOutput = state.getOutput(true);
  ASSERT_NE(secondOutput, nullptr);
  auto secondKeys = toKeys(secondOutput);
  std::sort(secondKeys.begin(), secondKeys.end());
  EXPECT_EQ(secondKeys, (std::vector<int64_t>{3, 4}));
  EXPECT_TRUE(state.empty());
}

} // namespace
} // namespace facebook::velox::cudf_velox
