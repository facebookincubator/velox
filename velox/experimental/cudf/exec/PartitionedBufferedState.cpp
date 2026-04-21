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

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

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

} // namespace

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t maxRowsPerLeaf,
    uint32_t initialHashSeed)
    : ops_(std::move(ops)),
      maxRowsPerLeaf_(maxRowsPerLeaf),
      root_(std::make_unique<Node>()),
      nextHashSeed_(initialHashSeed) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(maxRowsPerLeaf_, 0);
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto compacted = ops_->prepareInput(std::move(rawInput));
  if (compacted.empty()) {
    return;
  }

  insert(*root_, std::move(compacted));
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
    auto partitions = partitionInput(std::move(bufferedInput), *node.split);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  if (!node.leafState) {
    node.leafState = ops_->createLeaf(std::move(bufferedInput));
    if (node.leafState) {
      node.leafRows = ops_->leafRowCount(*node.leafState);
      ensureLeafWithinLimit(node);
    }
    return;
  }

  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*node.leafState, bufferedInput);
  if (projectedRows > maxRowsPerLeaf_) {
    splitLeaf(node, std::move(bufferedInput));
    return;
  }

  ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
  node.leafRows = ops_->leafRowCount(*node.leafState);
  ensureLeafWithinLimit(node);
}

void PartitionedBufferedState::splitLeaf(Node& node) {
  splitLeaf(node, InputChunk{});
}

void PartitionedBufferedState::splitLeaf(Node& node, InputChunk bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(node.leafState || !bufferedInput.empty());

  const auto totalRows = node.leafRows + bufferedInput.size();
  auto spec = makePartitionSpec(totalRows);

  auto storedPartitions = node.leafState
      ? ops_->repartitionLeaf(std::move(node.leafState), spec)
      : std::vector<std::unique_ptr<BufferedState>>(spec.numPartitions);
  auto incomingPartitions = partitionInput(std::move(bufferedInput), spec);

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if ((storedPartitions[i] && ops_->leafRowCount(*storedPartitions[i]) > 0) ||
        !incomingPartitions[i].empty()) {
      ++nonEmptyChildren;
    }
  }

  VELOX_CHECK_GT(
      nonEmptyChildren,
      1,
      "Partitioning buffered state made no progress: {} rows exceeded the "
      "per-leaf limit of {} rows using {} hash partitions.",
      totalRows,
      maxRowsPerLeaf_,
      spec.numPartitions);

  node.leafRows = 0;
  node.split = spec;
  node.children.clear();
  node.children.reserve(spec.numPartitions);
  for (int32_t i = 0; i < spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    if (storedPartitions[i]) {
      auto& child = *node.children[i];
      child.leafRows = ops_->leafRowCount(*storedPartitions[i]);
      child.leafState = std::move(storedPartitions[i]);
      ensureLeafWithinLimit(child);
    }
    if (!incomingPartitions[i].empty()) {
      insert(*node.children[i], std::move(incomingPartitions[i]));
    }
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

  node.leafRows = 0;
  return ops_->finalizeLeaf(std::move(node.leafState));
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
    splitLeaf(node);
  }
}

std::vector<InputChunk> PartitionedBufferedState::partitionInput(
    InputChunk input,
    const PartitionSpec& spec) {
  return input.empty() ? std::vector<InputChunk>(spec.numPartitions)
                       : ops_->partitionInput(std::move(input), spec);
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
