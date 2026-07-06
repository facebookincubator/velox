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

#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <atomic>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string_view>

namespace facebook::velox::cudf_velox {
namespace {

// Avoid spinning forever when a single partition key cannot be split.
constexpr uint32_t kMaxSplitSeedAttempts = 64;

std::atomic<uint64_t> nextDiagnosticId{1};

bool partitionedStateDiagnosticsEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("VELOX_CUDF_STREAMING_GROUPBY_DIAGNOSTICS");
    return value != nullptr && std::string_view{value} != "0" &&
        std::string_view{value} != "false";
  }();
  return enabled;
}

std::string partitionSizes(
    const std::vector<std::unique_ptr<BufferedState>>& storedPartitions,
    const std::vector<InputChunk>& incomingPartitions,
    const BufferedStateOps& ops) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << i << ":stored="
        << (storedPartitions[i] ? ops.leafRowCount(*storedPartitions[i]) : 0)
        << "/incoming=" << incomingPartitions[i].size();
  }
  out << ']';
  return out.str();
}

bool nodeEmpty(const PartitionedBufferedState::Node& node) {
  if (node.isLeaf()) {
    return node.leafState == nullptr;
  }

  for (const auto& child : node.children) {
    if (child && !nodeEmpty(*child)) {
      return false;
    }
  }
  return true;
}

size_t countNonEmptyChildren(
    const std::vector<std::unique_ptr<BufferedState>>& storedPartitions,
    const std::vector<InputChunk>& incomingPartitions,
    const BufferedStateOps& ops) {
  VELOX_CHECK_EQ(storedPartitions.size(), incomingPartitions.size());

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if ((storedPartitions[i] && ops.leafRowCount(*storedPartitions[i]) > 0) ||
        !incomingPartitions[i].empty()) {
      ++nonEmptyChildren;
    }
  }
  return nonEmptyChildren;
}

struct SplitLeafAttempt {
  PartitionSpec spec;
  std::vector<std::unique_ptr<BufferedState>> storedPartitions;
  std::vector<InputChunk> incomingPartitions;
};

} // namespace

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t maxRowsPerLeaf,
    uint32_t initialHashSeed)
    : ops_(std::move(ops)),
      maxRowsPerLeaf_(maxRowsPerLeaf),
      root_(std::make_unique<Node>()),
      nextHashSeed_(initialHashSeed),
      diagnosticId_(nextDiagnosticId.fetch_add(1, std::memory_order_relaxed)) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(maxRowsPerLeaf_, 0);
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=create maxRowsPerLeaf=" << maxRowsPerLeaf_
                 << " initialHashSeed=" << initialHashSeed;
  }
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  const auto batch = ++inputBatchCount_;
  const auto rawRows = rawInput->size();
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=add_input_begin batch=" << batch
                 << " rawRows=" << rawRows;
  }
  auto compacted = ops_->prepareInput(std::move(rawInput));
  if (compacted.empty()) {
    if (partitionedStateDiagnosticsEnabled()) {
      LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                   << " event=add_input_empty batch=" << batch
                   << " rawRows=" << rawRows;
    }
    return;
  }

  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=add_input_prepared batch=" << batch
                 << " rawRows=" << rawRows
                 << " preparedRows=" << compacted.size()
                 << " preparedColumns=" << compacted.view.num_columns()
                 << " stream=" << compacted.stream.value();
  }
  insert(*root_, std::move(compacted));
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=add_input_end batch=" << batch;
  }
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput() {
  return drainNextOutput(*root_);
}

bool PartitionedBufferedState::empty() const {
  return nodeEmpty(*root_);
}

void PartitionedBufferedState::insert(Node& node, InputChunk bufferedInput) {
  if (bufferedInput.empty()) {
    return;
  }

  if (!node.isLeaf()) {
    if (partitionedStateDiagnosticsEnabled()) {
      LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                   << " event=route_internal inputRows=" << bufferedInput.size()
                   << " partitions=" << node.split->numPartitions
                   << " seed=" << node.split->seed;
    }
    auto partitions = partitionInput(bufferedInput, *node.split);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  if (!node.leafState) {
    if (partitionedStateDiagnosticsEnabled()) {
      LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                   << " event=create_leaf inputRows=" << bufferedInput.size();
    }
    node.leafState = ops_->createLeaf(std::move(bufferedInput));
    if (node.leafState) {
      node.leafRows = ops_->leafRowCount(*node.leafState);
      if (partitionedStateDiagnosticsEnabled()) {
        LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                     << " event=create_leaf_done leafRows=" << node.leafRows;
      }
      ensureLeafWithinLimit(node);
    }
    return;
  }

  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*node.leafState, bufferedInput);
  if (projectedRows > maxRowsPerLeaf_) {
    if (partitionedStateDiagnosticsEnabled()) {
      LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                   << " event=leaf_limit_crossed leafRows=" << node.leafRows
                   << " inputRows=" << bufferedInput.size()
                   << " projectedRows=" << projectedRows
                   << " maxRowsPerLeaf=" << maxRowsPerLeaf_;
    }
    splitLeaf(node, std::move(bufferedInput));
    return;
  }

  ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
  node.leafRows = ops_->leafRowCount(*node.leafState);
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=leaf_updated leafRows=" << node.leafRows
                 << " maxRowsPerLeaf=" << maxRowsPerLeaf_;
  }
  ensureLeafWithinLimit(node);
}

void PartitionedBufferedState::splitLeaf(Node& node) {
  splitLeaf(node, InputChunk{});
}

void PartitionedBufferedState::splitLeaf(Node& node, InputChunk bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(node.leafState || !bufferedInput.empty());

  const auto totalRows = node.leafRows + bufferedInput.size();
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=split_begin storedLeafRows=" << node.leafRows
                 << " incomingRows=" << bufferedInput.size()
                 << " totalRows=" << totalRows
                 << " maxRowsPerLeaf=" << maxRowsPerLeaf_;
  }
  auto splitAttempt = [&]() {
    for (uint32_t attempt = 1;; ++attempt) {
      auto spec = makePartitionSpec(totalRows);
      if (partitionedStateDiagnosticsEnabled()) {
        LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                     << " event=split_attempt_begin attempt=" << attempt
                     << " partitions=" << spec.numPartitions
                     << " seed=" << spec.seed
                     << " keyCount=" << spec.keyIndices.size();
      }
      auto storedPartitions = node.leafState
          ? ops_->repartitionLeaf(*node.leafState, spec)
          : std::vector<std::unique_ptr<BufferedState>>(spec.numPartitions);
      auto incomingPartitions = partitionInput(bufferedInput, spec);
      const auto nonEmptyChildren =
          countNonEmptyChildren(storedPartitions, incomingPartitions, *ops_);

      if (partitionedStateDiagnosticsEnabled()) {
        LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                     << " event=split_attempt_end attempt=" << attempt
                     << " partitions=" << spec.numPartitions
                     << " seed=" << spec.seed
                     << " nonEmptyChildren=" << nonEmptyChildren << " sizes="
                     << partitionSizes(
                            storedPartitions, incomingPartitions, *ops_);
      }

      if (nonEmptyChildren > 1) {
        return SplitLeafAttempt{
            std::move(spec),
            std::move(storedPartitions),
            std::move(incomingPartitions)};
      }

      VELOX_CHECK_LT(
          attempt,
          kMaxSplitSeedAttempts,
          "Partitioning buffered state made no progress after {} hash seed "
          "attempts: {} rows exceeded the per-leaf limit of {} rows using {} "
          "hash partitions. This can happen when every row has the same "
          "partition key.",
          kMaxSplitSeedAttempts,
          totalRows,
          maxRowsPerLeaf_,
          spec.numPartitions);
    }
  }();

  node.leafRows = 0;
  node.leafState.reset();
  node.split = splitAttempt.spec;
  node.children.clear();
  node.children.reserve(splitAttempt.spec.numPartitions);
  for (int32_t i = 0; i < splitAttempt.spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    if (splitAttempt.storedPartitions[i]) {
      auto& child = *node.children[i];
      child.leafRows = ops_->leafRowCount(*splitAttempt.storedPartitions[i]);
      child.leafState = std::move(splitAttempt.storedPartitions[i]);
      ensureLeafWithinLimit(child);
    }
    if (!splitAttempt.incomingPartitions[i].empty()) {
      insert(*node.children[i], std::move(splitAttempt.incomingPartitions[i]));
    }
  }
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=split_committed partitions="
                 << splitAttempt.spec.numPartitions
                 << " seed=" << splitAttempt.spec.seed;
  }
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput(Node& node) {
  if (!node.isLeaf()) {
    for (auto& child : node.children) {
      if (child) {
        auto output = drainNextOutput(*child);
        if (output) {
          return output;
        }
      }
    }
    return nullptr;
  }

  if (!node.leafState) {
    return nullptr;
  }

  const auto leafRows = node.leafRows;
  node.leafRows = 0;
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=drain_leaf_begin leafRows=" << leafRows;
  }
  auto output = ops_->finalizeLeaf(std::move(node.leafState));
  if (partitionedStateDiagnosticsEnabled()) {
    LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                 << " event=drain_leaf_end outputBatch="
                 << (output ? ++outputBatchCount_ : outputBatchCount_)
                 << " outputRows=" << (output ? output->size() : 0);
  }
  return output;
}

PartitionSpec PartitionedBufferedState::makePartitionSpec(size_t totalRows) {
  VELOX_CHECK_GT(totalRows, maxRowsPerLeaf_);

  const auto requiredPartitions =
      (totalRows + maxRowsPerLeaf_ - 1) / maxRowsPerLeaf_;
  const auto numPartitions =
      std::max<size_t>(2, std::min(requiredPartitions, totalRows));
  VELOX_CHECK_LE(numPartitions, std::numeric_limits<int32_t>::max());

  return PartitionSpec{
      static_cast<int32_t>(numPartitions),
      ops_->keyIndices(),
      cudf::hash_id::HASH_MURMUR3,
      nextHashSeed_++};
}

void PartitionedBufferedState::ensureLeafWithinLimit(Node& node) {
  if (node.isLeaf() && node.leafState && node.leafRows > maxRowsPerLeaf_) {
    if (partitionedStateDiagnosticsEnabled()) {
      LOG(ERROR) << "[SG_PBS_DIAG] state=" << diagnosticId_
                   << " event=post_update_limit_crossed leafRows="
                   << node.leafRows << " maxRowsPerLeaf=" << maxRowsPerLeaf_;
    }
    splitLeaf(node);
  }
}

std::vector<InputChunk> PartitionedBufferedState::partitionInput(
    const InputChunk& input,
    const PartitionSpec& spec) {
  return input.empty() ? std::vector<InputChunk>(spec.numPartitions)
                       : ops_->partitionInput(input, spec);
}

FlushableBufferedState::FlushableBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t flushRowLimit,
    uint64_t flushByteLimit)
    : ops_(std::move(ops)),
      flushRowLimit_(flushRowLimit),
      flushByteLimit_(flushByteLimit) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(flushRowLimit_, 0);
}

void FlushableBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto chunk = ops_->prepareInput(std::move(rawInput));
  if (chunk.empty()) {
    return;
  }

  if (!currentLeaf_) {
    currentLeaf_ = ops_->createLeaf(std::move(chunk));
    if (currentLeaf_) {
      currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
      if (currentLeafRows_ > flushRowLimit_) {
        finalizeActiveLeaf();
      }
    }
    return;
  }

  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*currentLeaf_, chunk);
  if (projectedRows > flushRowLimit_) {
    finalizeActiveLeaf();
    currentLeaf_ = ops_->createLeaf(std::move(chunk));
    if (currentLeaf_) {
      currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
      if (currentLeafRows_ > flushRowLimit_) {
        finalizeActiveLeaf();
      }
    }
    return;
  }

  ops_->addInputToLeaf(*currentLeaf_, std::move(chunk));
  currentLeafRows_ = ops_->leafRowCount(*currentLeaf_);
  if (currentLeafRows_ > flushRowLimit_) {
    finalizeActiveLeaf();
  }
}

bool FlushableBufferedState::shouldFlushActiveLeaf() const {
  return currentLeaf_ && ops_->leafFlatSize(*currentLeaf_) > flushByteLimit_;
}

CudfVectorPtr FlushableBufferedState::getOutput(bool noMoreInput) {
  if (auto output = popPendingOutput()) {
    return output;
  }

  if (shouldFlushActiveLeaf()) {
    finalizeActiveLeaf();
    return popPendingOutput();
  }

  if (noMoreInput && currentLeaf_) {
    finalizeActiveLeaf();
    return popPendingOutput();
  }

  return nullptr;
}

bool FlushableBufferedState::empty() const {
  return !currentLeaf_ && pendingOutputs_.empty();
}

CudfVectorPtr FlushableBufferedState::popPendingOutput() {
  if (pendingOutputs_.empty()) {
    return nullptr;
  }

  auto output = std::move(pendingOutputs_.front());
  pendingOutputs_.pop_front();
  return output;
}

void FlushableBufferedState::finalizeActiveLeaf() {
  if (!currentLeaf_) {
    return;
  }

  currentLeafRows_ = 0;
  auto output = ops_->finalizeLeaf(std::move(currentLeaf_));
  if (output) {
    pendingOutputs_.push_back(std::move(output));
  }
}

} // namespace facebook::velox::cudf_velox
