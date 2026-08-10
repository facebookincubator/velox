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

// Avoid spinning forever when a single partition key cannot be split.
constexpr uint32_t kMaxSplitSeedAttempts = 4;

bool referencesSameColumn(
    const cudf::column_view& left,
    const cudf::column_view& right) {
  if (left.type() != right.type() || left.size() != right.size() ||
      left.offset() != right.offset() ||
      left.head<void>() != right.head<void>() ||
      left.null_mask() != right.null_mask() ||
      left.num_children() != right.num_children()) {
    return false;
  }

  for (cudf::size_type i = 0; i < left.num_children(); ++i) {
    if (!referencesSameColumn(left.child(i), right.child(i))) {
      return false;
    }
  }
  return true;
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
    const std::vector<InputChunk>& storedPartitions,
    const std::vector<InputChunk>& incomingPartitions) {
  VELOX_CHECK_EQ(storedPartitions.size(), incomingPartitions.size());

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if (!storedPartitions[i].empty() || !incomingPartitions[i].empty()) {
      ++nonEmptyChildren;
    }
  }
  return nonEmptyChildren;
}

struct SplitLeafAttempt {
  PartitionSpec spec;
  std::vector<InputChunk> storedPartitions;
  std::vector<InputChunk> incomingPartitions;
};

} // namespace

bool InputChunk::ownsFullTable() const {
  if (storage != InputChunkStorage::kOwned || owner == nullptr ||
      tableOwner != nullptr) {
    return false;
  }

  const auto ownerView = owner->getTableView();
  if (view.num_rows() != ownerView.num_rows() ||
      view.num_columns() != ownerView.num_columns()) {
    return false;
  }
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    if (!referencesSameColumn(view.column(i), ownerView.column(i))) {
      return false;
    }
  }
  return true;
}

InputChunk InputChunk::materialize(rmm::device_async_resource_ref mr) && {
  if (empty()) {
    return InputChunk{};
  }

  if (storage == InputChunkStorage::kOwned) {
    VELOX_CHECK(
        ownsFullTable(),
        "An owned InputChunk must reference its owner's complete table");
    if (owner.use_count() == 1) {
      return std::move(*this);
    }
  }

  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(type);
  VELOX_CHECK(
      owner != nullptr || tableOwner != nullptr,
      "A borrowed InputChunk must retain its source storage");

  auto materializedTable = std::make_unique<cudf::table>(view, stream, mr);
  auto materialized = std::make_shared<CudfVector>(
      pool,
      type,
      materializedTable->num_rows(),
      std::move(materializedTable),
      stream);
  auto result = InputChunk{
      materialized->pool(),
      type,
      materialized->getTableView(),
      materialized->stream(),
      std::move(materialized),
      nullptr,
      InputChunkStorage::kOwned};
  owner.reset();
  tableOwner.reset();
  return result;
}

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

  // If node is not a leaf, it is an internal node and any input has to be
  // partitioned and inserted into child nodes.
  if (!node.isLeaf()) {
    auto partitions = partitionInput(bufferedInput, *node.partitionSpec);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  // If node is a leaf but empty, initialize it with the input.
  if (!node.leafState) {
    node.leafState = ops_->createLeaf(std::move(bufferedInput));
    if (node.leafState) {
      node.leafRows = ops_->leafRowCount(*node.leafState);
      ensureLeafWithinLimit(node);
    }
    return;
  }

  // If node is a leaf and not empty, check if the input can be added to the
  // leaf. If not, split the leaf and insert the input into the new subtree.
  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*node.leafState, bufferedInput);
  if (projectedRows > maxRowsPerLeaf_) {
    splitLeafAndAddInput(node, std::move(bufferedInput));
    return;
  }

  ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
  node.leafRows = ops_->leafRowCount(*node.leafState);
  ensureLeafWithinLimit(node);
}

void PartitionedBufferedState::splitLeaf(Node& node) {
  splitLeafAndAddInput(node, InputChunk{});
}

void PartitionedBufferedState::splitLeafAndAddInput(
    Node& node,
    InputChunk bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(node.leafState || !bufferedInput.empty());

  const auto totalRows = node.leafRows + bufferedInput.size();
  auto splitAttempt = [&]() {
    for (uint32_t attempt = 1;; ++attempt) {
      auto spec = makePartitionSpec(totalRows);
      auto storedPartitions = node.leafState
          ? ops_->partitionLeaf(*node.leafState, spec)
          : std::vector<InputChunk>(spec.numPartitions);
      auto incomingPartitions = partitionInput(bufferedInput, spec);
      const auto nonEmptyChildren =
          countNonEmptyChildren(storedPartitions, incomingPartitions);

      if (nonEmptyChildren > 1) {
        // Found a partition spec that can split the leaf into more than one
        // child. If nonEmptyChildren = 1, then the partition spec could not
        // split and we'd have no meaningful progress.
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
  node.partitionSpec = splitAttempt.spec;
  node.children.clear();
  node.children.reserve(splitAttempt.spec.numPartitions);
  for (int32_t i = 0; i < splitAttempt.spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    auto& child = *node.children[i];
    auto& stored = splitAttempt.storedPartitions[i];
    auto& incoming = splitAttempt.incomingPartitions[i];
    if (!stored.empty() && !incoming.empty()) {
      child.leafState =
          ops_->createLeafFromInputs(std::move(stored), std::move(incoming));
    } else if (!stored.empty()) {
      child.leafState = ops_->createLeaf(std::move(stored));
    } else if (!incoming.empty()) {
      child.leafState = ops_->createLeaf(std::move(incoming));
    }

    if (child.leafState) {
      child.leafRows = ops_->leafRowCount(*child.leafState);
      ensureLeafWithinLimit(child);
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
