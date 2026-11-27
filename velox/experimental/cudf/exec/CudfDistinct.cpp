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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfDistinct.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/HashAggregationStatNames.h"

#include <cudf/stream_compaction.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

namespace {

// One leaf of the keys of a distinct.
struct DistinctLeafState final : public BufferedState {
  // The keys of the leaf. When 'merged' is set, this is one chunk that holds
  // every key once. Otherwise the chunks may repeat keys.
  std::vector<InputChunk> chunks;
  bool merged{false};
  // Totals over 'chunks'.
  size_t rows{0};
  uint64_t flatSize{0};
};

} // namespace

// Keeps the keys of a distinct in cuDF tables. A chunk holds the grouping keys
// in output order.
//
// A final or single step only buffers the chunks of a leaf. Their duplicates
// are removed when the leaf would pass a limit and when it is finalized, so
// that keys which fit in the limits cost one concatenate and distinct, however
// many batches they arrive in. A partial or intermediate step removes the
// duplicates with every batch, since it flushes on the size of the keys that
// are left.
class CudfDistinct::DistinctBufferedStateOps final : public BufferedStateOps {
 public:
  explicit DistinctBufferedStateOps(CudfDistinct& owner) : owner_(owner) {
    keyIndices_.reserve(owner_.groupingKeyOutputChannels_.size());
    for (auto keyIndex : owner_.groupingKeyOutputChannels_) {
      keyIndices_.push_back(static_cast<cudf::size_type>(keyIndex));
    }
  }

  InputChunk prepareInput(CudfVectorPtr rawInput) override {
    const auto keys = rawInput->getTableView().select(
        owner_.groupingKeyInputChannels_.begin(),
        owner_.groupingKeyInputChannels_.end());
    return makeBorrowedChunk(std::move(rawInput), owner_.outputType_, keys);
  }

  size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeafState(leaf).rows + input.size();
  }

  uint64_t estimatedMergedBytesUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const override {
    return asLeafState(leaf).flatSize + input.flatSize;
  }

  std::unique_ptr<BufferedState> createLeaf(InputChunk input) override {
    auto leaf = std::make_unique<DistinctLeafState>();
    addInputToLeaf(*leaf, std::move(input));
    if (leaf->chunks.empty()) {
      return nullptr;
    }
    return leaf;
  }

  void addInputToLeaf(BufferedState& leaf, InputChunk input) override {
    auto& state = asLeafState(leaf);
    append(state, std::move(input));
    if (owner_.isPartialOutput_) {
      merge(state);
    }
  }

  bool compactLeaf(BufferedState& leaf) override {
    auto& state = asLeafState(leaf);
    if (state.merged) {
      return false;
    }
    merge(state);
    return true;
  }

  size_t leafRowCount(const BufferedState& leaf) const override {
    return asLeafState(leaf).rows;
  }

  uint64_t leafFlatSize(const BufferedState& leaf) const override {
    return asLeafState(leaf).flatSize;
  }

  std::vector<InputChunk> routeInput(
      const InputChunk& input,
      const HashRouting& routing) override {
    return routeChunkByHash(input, routing, get_output_mr());
  }

  std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const HashRouting& routing) override {
    std::vector<std::unique_ptr<BufferedState>> children(routing.numTargets());

    auto& state = asLeafState(*leaf);
    merge(state);
    if (state.chunks.empty()) {
      return children;
    }
    auto partitions = routeInput(state.chunks.front(), routing);
    // The partitions are a copy of the leaf's keys.
    leaf.reset();

    for (int32_t i = 0; i < routing.numTargets(); ++i) {
      if (!partitions[i].empty()) {
        auto child = std::make_unique<DistinctLeafState>();
        append(*child, std::move(partitions[i]));
        // A part of keys that are all different.
        child->merged = true;
        children[i] = std::move(child);
      }
    }
    return children;
  }

  CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) override {
    auto& state = asLeafState(*leaf);
    merge(state);
    if (state.chunks.empty()) {
      return nullptr;
    }
    return releaseChunk(std::move(state.chunks.front()));
  }

  const std::vector<cudf::size_type>& keyIndices() const override {
    return keyIndices_;
  }

 private:
  CudfDistinct& owner_;
  std::vector<cudf::size_type> keyIndices_;

  DistinctLeafState& asLeafState(BufferedState& leaf) const {
    return static_cast<DistinctLeafState&>(leaf);
  }

  const DistinctLeafState& asLeafState(const BufferedState& leaf) const {
    return static_cast<const DistinctLeafState&>(leaf);
  }

  void append(DistinctLeafState& state, InputChunk chunk) const {
    state.rows += chunk.size();
    state.flatSize += chunk.flatSize;
    state.chunks.push_back(std::move(chunk));
    state.merged = false;
  }

  // Replaces the chunks of 'state' by one chunk that holds every key once.
  void merge(DistinctLeafState& state) const {
    if (state.merged) {
      return;
    }
    VELOX_CHECK(!state.chunks.empty());

    auto stream = state.chunks.front().stream;
    CudfVectorPtr keys;
    if (state.chunks.size() == 1) {
      keys = owner_.getDistinctKeys(
          state.chunks.front().view, owner_.groupingKeyOutputChannels_, stream);
      state.chunks.clear();
    } else {
      std::vector<cudf::table_view> views;
      std::vector<cuda::stream_ref> inputStreams;
      views.reserve(state.chunks.size());
      inputStreams.reserve(state.chunks.size());
      for (const auto& chunk : state.chunks) {
        views.push_back(chunk.view);
        inputStreams.push_back(chunk.stream);
      }
      auto concatenatedTable =
          concatenateViews(views, inputStreams, stream, get_temp_mr());
      // The concatenation owns a replacement for the chunks. Release them
      // before the distinct allocates its working and output buffers.
      // concatenateViews() made the chunks' streams wait for the
      // concatenation, so the frees cannot race the copy.
      state.chunks.clear();
      keys = owner_.getDistinctKeys(
          concatenatedTable->view(), owner_.groupingKeyOutputChannels_, stream);
    }

    state.rows = 0;
    state.flatSize = 0;
    if (keys) {
      append(state, makeOwnedChunk(std::move(keys), owner_.outputType_));
    }
    state.merged = true;
  }
};

CudfDistinct::CudfDistinct(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfDistinct"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfDistinct::initialize() {
  Operator::initialize();

  setupGroupingKeyChannelProjections(
      *aggregationNode_, groupingKeyInputChannels_, groupingKeyOutputChannels_);

  const auto& cudfConfig = CudfConfig::getInstance();
  const auto maxBufferedRows = cudfConfig.batchSizeMaxThreshold.value_or(
      std::numeric_limits<int32_t>::max());
  VELOX_CHECK_GT(maxBufferedRows, 0);
  if (isPartialOutput_) {
    flushableBufferedState_ = std::make_unique<FlushableBufferedState>(
        std::make_unique<DistinctBufferedStateOps>(*this),
        maxBufferedRows,
        maxPartialAggregationMemoryUsage_);
  } else {
    PartitionedBufferedState::Options options;
    options.maxRowsPerLeaf = maxBufferedRows;
    options.maxBytesPerLeaf = cudfConfig.groupbyMaxPartitionBytes.value_or(
        PartitionedBufferedState::kNoByteLimit);
    partitionedBufferedState_ = std::make_unique<PartitionedBufferedState>(
        std::make_unique<DistinctBufferedStateOps>(*this), options);
  }

  aggregationNode_.reset();
}

void CudfDistinct::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (isPartialOutput_) {
    flushableBufferedState_->addInput(std::move(cudfInput));
  } else {
    partitionedBufferedState_->addInput(std::move(cudfInput));
  }
}

CudfVectorPtr CudfDistinct::getDistinctKeys(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    cuda::stream_ref stream) {
  auto result = cudf::distinct(
      tableView.select(groupByKeys.begin(), groupByKeys.end()),
      {groupingKeyOutputChannels_.begin(), groupingKeyOutputChannels_.end()},
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      get_output_mr());

  auto numRows = result->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(result), stream);
}

void CudfDistinct::recordPartialFlushStats(const CudfVector& output) {
  const auto numOutputRows = output.size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregationStatNames::kFlushRowCount),
        RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregationStatNames::kFlushTimes),
        RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregationStatNames::kPartialAggregationPct),
        RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
}

RowVectorPtr CudfDistinct::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  if (isPartialOutput_) {
    if (auto output = flushableBufferedState_->getOutput(noMoreInput_)) {
      recordPartialFlushStats(*output);
      return output;
    }
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  if (!noMoreInput_) {
    // A final or single step waits for all of its input, so that every key
    // is returned once.
    return nullptr;
  }

  if (!statePartitionsRecorded_) {
    statePartitionsRecorded_ = true;
    stats_.wlock()->addRuntimeStat(
        std::string{kGroupbyStatePartitionsStat},
        RuntimeCounter(
            static_cast<int64_t>(partitionedBufferedState_->numLeaves())));
  }

  auto output = partitionedBufferedState_->drainNextOutput();
  if (!output) {
    finished_ = true;
  }
  return output;
}

void CudfDistinct::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfDistinct::doClose() {
  flushableBufferedState_.reset();
  partitionedBufferedState_.reset();
  Operator::close();
}

bool CudfDistinct::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
