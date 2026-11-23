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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

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
  // The size that a leaf reports for each of its rows. A fixed number, so that
  // the byte limit tests do not depend on how cuDF allocates a column.
  static constexpr uint64_t kRowBytes = 8;

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

  uint64_t estimatedMergedBytesUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return (asLeaf(leaf).chunk.size() + input.size()) * kRowBytes;
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
    return asLeaf(leaf).chunk.size() * kRowBytes;
  }

  // Routes on the host with the hash set by setPartitioning(), or on the
  // device with the hash of the keys when none is set.
  std::vector<InputChunk> routeInput(
      const InputChunk& input,
      const HashRouting& routing) override {
    ++numRouteCalls_;
    numRoutedRows_ += input.size();
    if (!hashFn_) {
      return routeChunkByHash(input, routing, get_output_mr());
    }

    std::vector<std::vector<int64_t>> buckets(routing.numTargets());
    for (auto key : extractKeys(input)) {
      const auto bucket =
          static_cast<int32_t>(hashFn_(key) % routing.numBuckets);
      VELOX_CHECK_GE(bucket, routing.bucketBounds.front());
      VELOX_CHECK_LT(bucket, routing.bucketBounds.back());
      const auto target = std::upper_bound(
                              routing.bucketBounds.begin(),
                              routing.bucketBounds.end(),
                              bucket) -
          routing.bucketBounds.begin() - 1;
      buckets[target].push_back(key);
    }

    std::vector<InputChunk> partitions(routing.numTargets());
    for (int32_t i = 0; i < routing.numTargets(); ++i) {
      if (!buckets[i].empty()) {
        partitions[i] = makeChunk_(buckets[i]);
      }
    }
    return partitions;
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const HashRouting& routing) override {
    auto partitions = routeInput(asLeaf(*leaf).chunk, routing);
    leaf.reset();

    std::vector<std::unique_ptr<BufferedState>> leaves(routing.numTargets());
    for (int32_t i = 0; i < routing.numTargets(); ++i) {
      if (!partitions[i].empty()) {
        leaves[i] = std::make_unique<TableLeafState>(std::move(partitions[i]));
      }
    }
    return leaves;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto chunk = std::move(asLeaf(*leaf).chunk);
    if (chunk.owner) {
      return chunk.owner;
    }
    // A chunk routed on the device is a slice of a shared table.
    auto table = std::make_unique<cudf::table>(
        chunk.view, chunk.stream, get_output_mr());
    const auto numRows = table->num_rows();
    return std::make_shared<CudfVector>(
        pool_, rowType_, numRows, std::move(table), chunk.stream);
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

  void setPartitioning(
      std::function<std::vector<int64_t>(const InputChunk&)> extractKeys,
      std::function<InputChunk(const std::vector<int64_t>&)> makeChunk,
      std::function<uint32_t(int64_t)> hashFn) {
    extractKeys_ = std::move(extractKeys);
    makeChunk_ = std::move(makeChunk);
    hashFn_ = std::move(hashFn);
  }

  size_t numRouteCalls() const {
    return numRouteCalls_;
  }

  size_t numRoutedRows() const {
    return numRoutedRows_;
  }

  void resetRouteCounters() {
    numRouteCalls_ = 0;
    numRoutedRows_ = 0;
  }

 private:
  memory::MemoryPool* pool_;
  RowTypePtr rowType_;
  std::vector<cudf::size_type> keyIndices_;

  std::function<std::vector<int64_t>(const InputChunk&)> extractKeys_;
  std::function<InputChunk(const std::vector<int64_t>&)> makeChunk_;
  std::function<uint32_t(int64_t)> hashFn_;
  size_t numRouteCalls_{0};
  size_t numRoutedRows_{0};

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
    std::vector<cuda::stream_ref> inputStreams{left.stream, right.stream};
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
    stream.sync();
    return std::make_shared<CudfVector>(
        pool_.get(), rowType_, row->size(), std::move(table), stream);
  }

  // Makes ops that route on the host by 'hashFn' of the key.
  std::unique_ptr<IdentityBufferedStateOps> makeHostOps(
      std::function<uint32_t(int64_t)> hashFn) {
    auto ops =
        std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
    ops->setPartitioning(
        [this](const InputChunk& input) { return toKeys(input); },
        [this](const std::vector<int64_t>& keys) { return makeChunk(keys); },
        std::move(hashFn));
    return ops;
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
    stream.sync();

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
            output});
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

// The tests that route on the host use eight buckets and a key's value as its
// hash, so key k is in bucket k % 8.
TEST_F(PartitionedBufferedStateTest, mergesLeafDirectlyBelowCap) {
  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  auto* opsPtr = ops.get();
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = 10, .numBuckets = 8});

  state.addInput(makeCudfVector({1, 2}));
  state.addInput(makeCudfVector({3, 4}));

  // Nothing is routed while one leaf holds everything.
  EXPECT_EQ(opsPtr->numRouteCalls(), 0);
  EXPECT_EQ(state.numLeaves(), 1);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3, 4}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, topLevelSplitKeepsRoutingStable) {
  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = 3, .numBuckets = 8});

  state.addInput(makeCudfVector({0, 1}));
  // Splits into the buckets [0, 4) and [4, 8).
  state.addInput(makeCudfVector({4, 5}));
  EXPECT_EQ(state.numLeaves(), 2);

  state.addInput(makeCudfVector({2, 6}));

  EXPECT_EQ(state.numLeaves(), 2);
  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 1, 2}, {4, 5, 6}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, overflowingChildSplitsAgain) {
  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = 2, .numBuckets = 8});

  state.addInput(makeCudfVector({0, 2}));
  // Splits into [0, 4) and [4, 8). The first child then holds {0, 2} and gets
  // 1, so it splits again into [0, 2) and [2, 4). The second child stays as
  // it is.
  state.addInput(makeCudfVector({1, 4}));
  EXPECT_EQ(state.numLeaves(), 3);

  // Reaches the leaf [2, 4) without passing through its parent.
  state.addInput(makeCudfVector({3}));

  EXPECT_EQ(state.numLeaves(), 3);
  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{{0, 1}, {2, 3}, {4}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, splitThatSeparatesNothingGoesOn) {
  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = 2, .numBuckets = 8});

  // All three rows are in the first half of the buckets, so the first split
  // leaves them together and the half is split again.
  state.addInput(makeCudfVector({0, 1, 3}));
  EXPECT_EQ(state.numLeaves(), 3);

  // The second half of the buckets is an empty leaf until a row arrives.
  state.addInput(makeCudfVector({5}));

  EXPECT_EQ(state.numLeaves(), 3);
  EXPECT_EQ(
      drainAll(state), (std::vector<std::vector<int64_t>>{{0, 1}, {3}, {5}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, singleBucketGrowsUpToHardLimit) {
  // Every key is in bucket 0.
  auto ops = makeHostOps([](int64_t /*key*/) { return 0U; });
  PartitionedBufferedState state(
      std::move(ops),
      {.maxRowsPerLeaf = 2, .hardMaxRowsPerLeaf = 4, .numBuckets = 8});

  state.addInput(makeCudfVector({1}));
  state.addInput(makeCudfVector({2}));
  EXPECT_EQ(state.numLeaves(), 1);

  // The third row does not fit. The leaf is split down to bucket 0 alone,
  // which cannot be split and takes the row.
  state.addInput(makeCudfVector({3}));
  EXPECT_EQ(state.numLeaves(), 4);
  state.addInput(makeCudfVector({4}));

  VELOX_ASSERT_THROW(
      state.addInput(makeCudfVector({5})),
      "Cannot partition buffered state any further");
  EXPECT_EQ(state.numLeaves(), 4);
  EXPECT_EQ(drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3, 4}}));
}

TEST_F(PartitionedBufferedStateTest, byteLimitSplitsLeaf) {
  const auto rowBytes = IdentityBufferedStateOps::kRowBytes;

  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  PartitionedBufferedState state(
      std::move(ops),
      {.maxRowsPerLeaf = 100,
       .maxBytesPerLeaf = 3 * rowBytes,
       .numBuckets = 8});

  state.addInput(makeCudfVector({0, 1}));
  EXPECT_EQ(state.numLeaves(), 1);
  // Four rows are over the byte limit, far below the row limit.
  state.addInput(makeCudfVector({4, 5}));
  EXPECT_EQ(state.numLeaves(), 2);
  // Three rows are not.
  state.addInput(makeCudfVector({2, 6}));
  EXPECT_EQ(state.numLeaves(), 2);

  // A batch far over the limit is divided into as many leaves as it needs.
  state.addInput(makeCudfVector({8, 9, 10, 11, 16, 17, 18, 19}));
  EXPECT_EQ(state.numLeaves(), 5);

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{
          {0, 8, 16}, {1, 9, 17}, {2, 10, 18}, {4, 5, 6}, {11, 19}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, byteLimitIsATarget) {
  const auto rowBytes = IdentityBufferedStateOps::kRowBytes;

  // Every key is in bucket 0.
  auto ops = makeHostOps([](int64_t /*key*/) { return 0U; });
  PartitionedBufferedState state(
      std::move(ops),
      {.maxRowsPerLeaf = 100,
       .maxBytesPerLeaf = 2 * rowBytes,
       .numBuckets = 8});

  // The leaf is split down to bucket 0 alone and then keeps growing: passing
  // the byte limit does not fail.
  for (int64_t key = 1; key <= 6; ++key) {
    state.addInput(makeCudfVector({key}));
  }
  EXPECT_EQ(state.numLeaves(), 4);
  EXPECT_EQ(
      drainAll(state), (std::vector<std::vector<int64_t>>{{1, 2, 3, 4, 5, 6}}));
  EXPECT_TRUE(state.empty());
}

TEST_F(PartitionedBufferedStateTest, splitLeavesAreRoutedOnce) {
  auto ops =
      makeHostOps([](int64_t key) { return static_cast<uint32_t>(key); });
  auto* opsPtr = ops.get();
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = 2, .numBuckets = 8});

  // Eight rows need four leaves of two rows, so the first split divides the
  // buckets in four.
  state.addInput(makeCudfVector({0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(state.numLeaves(), 4);

  // Keys 8 and 10 are in buckets 0 and 2. Their leaves overflow and split
  // into single buckets.
  state.addInput(makeCudfVector({8, 10}));
  EXPECT_EQ(state.numLeaves(), 6);

  // A batch for leaves that are the result of two splits is partitioned in
  // one pass over its rows, not once per split.
  opsPtr->resetRouteCounters();
  state.addInput(makeCudfVector({9, 11}));
  EXPECT_EQ(opsPtr->numRouteCalls(), 1);
  EXPECT_EQ(opsPtr->numRoutedRows(), 2);
  EXPECT_EQ(state.numLeaves(), 6);

  EXPECT_EQ(
      drainAll(state),
      (std::vector<std::vector<int64_t>>{
          {0, 8}, {1, 9}, {2, 10}, {3, 11}, {4, 5}, {6, 7}}));
  EXPECT_TRUE(state.empty());
}

// A row must be in the same bucket whenever it is routed: that is what lets a
// leaf split in place.
TEST_F(PartitionedBufferedStateTest, deviceRoutingIsConsistent) {
  std::vector<int64_t> keys(1'000);
  std::iota(keys.begin(), keys.end(), 0);
  auto input = makeChunk(keys);

  HashRouting routing;
  routing.keyIndices = {0};
  routing.seed = PartitionedBufferedState::kDefaultHashSeed;
  routing.numBuckets = 16;

  // Routes 'rows' and returns the target of each key, or -1 for a key that is
  // not in 'rows'.
  auto targetsOf = [&](cudf::table_view rows, const HashRouting& routing) {
    auto [table, offsets] =
        partitionByHashRouting(rows, routing, input.stream, get_output_mr());
    EXPECT_EQ(offsets.size(), routing.numTargets() + 1);
    EXPECT_EQ(table->num_rows(), rows.num_rows());
    const auto routedKeys = toKeys(
        InputChunk{
            pool_.get(), rowType_, table->view(), input.stream, nullptr});
    std::vector<int32_t> targets(keys.size(), -1);
    for (int32_t target = 0; target < routing.numTargets(); ++target) {
      for (auto row = offsets[target]; row < offsets[target + 1]; ++row) {
        targets[routedKeys[row]] = target;
      }
    }
    return targets;
  };

  routing.bucketBounds = {0, 8, 16};
  const auto halves = targetsOf(input.view, routing);

  routing.bucketBounds = {0, 4, 8, 12, 16};
  const auto quarters = targetsOf(input.view, routing);

  routing.bucketBounds = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const auto buckets = targetsOf(input.view, routing);

  std::vector<size_t> rowsPerBucket(16, 0);
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_GE(buckets[i], 0);
    ASSERT_EQ(buckets[i] / 4, quarters[i]) << "key " << keys[i];
    ASSERT_EQ(buckets[i] / 8, halves[i]) << "key " << keys[i];
    ++rowsPerBucket[buckets[i]];
  }
  // The hash spreads the keys over all buckets.
  for (auto rows : rowsPerBucket) {
    EXPECT_GT(rows, 20);
  }

  // Splitting the rows of one target, as a leaf split does, sends each row to
  // the part that holds its bucket.
  routing.bucketBounds = {0, 4, 8, 12, 16};
  auto [table, offsets] = partitionByHashRouting(
      input.view, routing, input.stream, get_output_mr());
  auto secondQuarter = cudf::slice(
      table->view(),
      std::vector<cudf::size_type>{offsets[1], offsets[2]},
      input.stream);
  routing.bucketBounds = {4, 6, 8};
  const auto parts = targetsOf(secondQuarter.front(), routing);
  for (size_t i = 0; i < keys.size(); ++i) {
    if (quarters[i] == 1) {
      ASSERT_EQ(parts[i], (buckets[i] - 4) / 2) << "key " << keys[i];
    } else {
      ASSERT_EQ(parts[i], -1) << "key " << keys[i];
    }
  }
}

TEST_F(PartitionedBufferedStateTest, routedChunksShareTheSizeOfTheirTable) {
  std::vector<int64_t> keys(1'000);
  std::iota(keys.begin(), keys.end(), 0);
  auto input = makeChunk(keys);

  HashRouting routing;
  routing.keyIndices = {0};
  routing.seed = PartitionedBufferedState::kDefaultHashSeed;
  routing.numBuckets = 16;
  routing.bucketBounds = {0, 4, 8, 12, 16};
  const auto chunks = routeChunkByHash(input, routing, get_output_mr());
  ASSERT_EQ(chunks.size(), 4);

  const auto tableBytes = chunks[0].tableOwner->alloc_size();
  ASSERT_GT(tableBytes, 0);
  size_t numRows = 0;
  uint64_t numBytes = 0;
  for (const auto& chunk : chunks) {
    ASSERT_GT(chunk.size(), 0);
    EXPECT_EQ(chunk.tableOwner, chunks[0].tableOwner);
    // Each chunk's size is its share of the table by rows.
    EXPECT_NEAR(
        chunk.flatSize,
        static_cast<double>(tableBytes) * chunk.size() / keys.size(),
        1);
    numRows += chunk.size();
    numBytes += chunk.flatSize;
  }
  EXPECT_EQ(numRows, keys.size());
  EXPECT_LE(numBytes, tableBytes);
}

TEST_F(PartitionedBufferedStateTest, deviceRoutingKeepsKeysTogether) {
  constexpr size_t kMaxRowsPerLeaf = 64;
  auto ops = std::make_unique<IdentityBufferedStateOps>(pool_.get(), rowType_);
  auto* opsPtr = ops.get();
  PartitionedBufferedState state(
      std::move(ops), {.maxRowsPerLeaf = kMaxRowsPerLeaf});

  // 300 distinct keys, each seen three or four times across the batches.
  size_t numRows = 0;
  for (int64_t batch = 0; batch < 10; ++batch) {
    std::vector<int64_t> keys(100);
    for (int64_t i = 0; i < 100; ++i) {
      keys[i] = (batch * 37 + i * 7) % 300;
    }
    numRows += keys.size();

    opsPtr->resetRouteCounters();
    const auto leavesBefore = state.numLeaves();
    state.addInput(makeCudfVector(keys));
    if (leavesBefore > 1 && state.numLeaves() == leavesBefore) {
      // No leaf split, so the batch was partitioned exactly once.
      EXPECT_EQ(opsPtr->numRouteCalls(), 1);
      EXPECT_EQ(opsPtr->numRoutedRows(), keys.size());
    }
  }
  EXPECT_GE(state.numLeaves(), numRows / kMaxRowsPerLeaf);

  std::unordered_map<int64_t, size_t> outputOfKey;
  size_t numOutputRows = 0;
  size_t numOutputs = 0;
  while (auto output = state.drainNextOutput()) {
    EXPECT_LE(output->size(), kMaxRowsPerLeaf);
    for (auto key : toKeys(output)) {
      auto [it, inserted] = outputOfKey.emplace(key, numOutputs);
      EXPECT_EQ(it->second, numOutputs) << "key " << key << " is in two leaves";
    }
    numOutputRows += output->size();
    ++numOutputs;
  }
  EXPECT_EQ(numOutputRows, numRows);
  EXPECT_EQ(outputOfKey.size(), 300);
  EXPECT_GE(numOutputs, numRows / kMaxRowsPerLeaf);
  EXPECT_TRUE(state.empty());
}

// Times how long it takes to bring one batch to its leaves as the number of
// leaves grows. Not a correctness test: run it by name with
// --gtest_also_run_disabled_tests.
TEST_F(PartitionedBufferedStateTest, DISABLED_routingCostByLeafCount) {
  constexpr cudf::size_type kNumRows = 20'000'000;
  constexpr int kNumPayloadColumns = 3;
  constexpr int kRepeats = 5;
  constexpr uint32_t kMaxSplits = 10;

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int i = 0; i <= kNumPayloadColumns; ++i) {
    cudf::numeric_scalar<int64_t> init(i * 1'000, true, stream, mr);
    columns.push_back(cudf::sequence(kNumRows, init, stream, mr));
  }
  cudf::table table(std::move(columns));
  const auto input = table.view();
  const std::vector<cudf::size_type> keyIndices{0};

  // Best of kRepeats, in milliseconds.
  auto time = [&](const std::function<void()>& fn) {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < kRepeats; ++i) {
      stream.sync();
      const auto start = std::chrono::steady_clock::now();
      fn();
      stream.sync();
      const std::chrono::duration<double, std::milli> elapsed =
          std::chrono::steady_clock::now() - start;
      best = std::min(best, elapsed.count());
    }
    return best;
  };

  // A tree with a hash seed per level: every level partitions the rows it
  // receives in two and copies them.
  std::function<void(cudf::table_view, uint32_t, uint32_t&)> routePerLevel =
      [&](cudf::table_view rows, uint32_t levels, uint32_t& seed) {
        if (levels == 0 || rows.num_rows() == 0) {
          return;
        }
        auto [partitioned, offsets] = cudf::hash_partition(
            rows,
            keyIndices,
            2,
            cudf::hash_id::HASH_MURMUR3,
            seed++,
            stream,
            mr);
        auto halves = cudf::split(
            partitioned->view(),
            std::vector<cudf::size_type>{offsets[1]},
            stream);
        for (auto& half : halves) {
          routePerLevel(half, levels - 1, seed);
        }
      };

  std::cout
      << "rows=" << kNumRows << " columns=" << input.num_columns()
      << " (int64), times in ms, best of " << kRepeats << "\n"
      << "splits leaves seedPerLevel bucketRanges hashPartitionToLeaves\n";
  for (uint32_t splits = 1; splits <= kMaxSplits; ++splits) {
    const auto numLeaves = int32_t{1} << splits;

    const auto seedPerLevel = time([&] {
      uint32_t seed = 0;
      routePerLevel(input, splits, seed);
    });

    // Leaves are equal ranges of the default number of buckets.
    HashRouting routing;
    routing.keyIndices = keyIndices;
    routing.seed = PartitionedBufferedState::kDefaultHashSeed;
    routing.numBuckets = PartitionedBufferedState::kDefaultNumBuckets;
    for (int32_t leaf = 0; leaf <= numLeaves; ++leaf) {
      routing.bucketBounds.push_back(leaf * (routing.numBuckets / numLeaves));
    }
    const auto bucketRanges = time([&] {
      auto [partitioned, offsets] =
          partitionByHashRouting(input, routing, stream, mr);
      offsets.erase(offsets.begin());
      offsets.pop_back();
      auto leaves = cudf::split(partitioned->view(), offsets, stream);
    });

    // cudf::hash_partition() with one partition per leaf.
    const auto hashPartitionToLeaves = time([&] {
      auto [partitioned, offsets] = cudf::hash_partition(
          input,
          keyIndices,
          numLeaves,
          cudf::hash_id::HASH_MURMUR3,
          routing.seed,
          stream,
          mr);
    });

    std::cout << splits << " " << numLeaves << " " << seedPerLevel << " "
              << bucketRanges << " " << hashPartitionToLeaves << "\n";
  }
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
