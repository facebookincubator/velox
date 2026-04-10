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

size_t rowCount(const CudfVectorPtr& table) {
  return table ? static_cast<size_t>(table->size()) : 0;
}

bool nodeEmpty(const PartitionedBufferedState::Node& node) {
  if (node.isLeaf()) {
    return node.leafData == nullptr;
  }

  for (const auto& child : node.children) {
    if (child && !nodeEmpty(*child)) {
      return false;
    }
  }
  return true;
}

class CudfHashPartitioner final
    : public PartitionedBufferedState::HashPartitioner {
 public:
  std::vector<CudfVectorPtr> partition(
      const CudfVectorPtr& input,
      const TypePtr& tableType,
      const PartitionedBufferedState::PartitionSpec& spec) const override {
    if (!input) {
      return std::vector<CudfVectorPtr>(spec.numPartitions);
    }
    return hashPartitionTable(
        input,
        tableType,
        spec.keyIndices,
        spec.numPartitions,
        spec.hashId,
        spec.seed,
        input->stream());
  }
};

} // namespace

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t maxRowsPerLeaf,
    std::unique_ptr<HashPartitioner> partitioner,
    uint32_t initialHashSeed)
    : ops_(std::move(ops)),
      partitioner_(
          partitioner ? std::move(partitioner)
                      : std::make_unique<CudfHashPartitioner>()),
      maxRowsPerLeaf_(maxRowsPerLeaf),
      root_(std::make_unique<Node>()),
      nextHashSeed_(initialHashSeed) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_NOT_NULL(partitioner_);
  VELOX_CHECK_GT(maxRowsPerLeaf_, 0);
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto compacted = ops_->compactInputBatch(std::move(rawInput));
  if (!compacted || compacted->size() == 0) {
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

void PartitionedBufferedState::insert(Node& node, CudfVectorPtr bufferedInput) {
  if (!bufferedInput || bufferedInput->size() == 0) {
    return;
  }

  if (!node.isLeaf()) {
    auto partitions = partitionInput(bufferedInput, *node.split);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (partitions[i]) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  const auto totalRows = rowCount(node.leafData) + rowCount(bufferedInput);
  if (totalRows <= maxRowsPerLeaf_) {
    node.leafData = node.leafData
        ? ops_->mergeBuffered(node.leafData, std::move(bufferedInput))
        : std::move(bufferedInput);
    return;
  }

  splitLeaf(node, std::move(bufferedInput));
}

void PartitionedBufferedState::splitLeaf(
    Node& node,
    CudfVectorPtr bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(bufferedInput);

  const auto totalRows = rowCount(node.leafData) + rowCount(bufferedInput);
  auto spec = makePartitionSpec(totalRows);

  auto stored = std::move(node.leafData);
  auto storedPartitions = partitionInput(stored, spec);
  auto incomingPartitions = partitionInput(bufferedInput, spec);

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if (rowCount(storedPartitions[i]) + rowCount(incomingPartitions[i]) > 0) {
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

  node.split = spec;
  node.children.clear();
  node.children.reserve(spec.numPartitions);
  for (int32_t i = 0; i < spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    if (storedPartitions[i]) {
      insert(*node.children[i], std::move(storedPartitions[i]));
    }
    if (incomingPartitions[i]) {
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

  if (!node.leafData) {
    return nullptr;
  }

  return ops_->finalizeLeaf(std::move(node.leafData));
}

PartitionedBufferedState::PartitionSpec
PartitionedBufferedState::makePartitionSpec(size_t totalRows) {
  VELOX_CHECK_GT(totalRows, maxRowsPerLeaf_);

  const auto requiredPartitions =
      (totalRows + maxRowsPerLeaf_ - 1) / maxRowsPerLeaf_;
  const auto numPartitions = std::max<size_t>(
      2, std::min(requiredPartitions, totalRows));
  VELOX_CHECK_LE(numPartitions, std::numeric_limits<int32_t>::max());

  return PartitionSpec{
      static_cast<int32_t>(numPartitions),
      ops_->keyIndices(),
      cudf::hash_id::HASH_MURMUR3,
      nextHashSeed_++};
}

std::vector<CudfVectorPtr> PartitionedBufferedState::partitionInput(
    const CudfVectorPtr& input,
    const PartitionSpec& spec) const {
  return input ? partitioner_->partition(input, ops_->bufferedType(), spec)
               : std::vector<CudfVectorPtr>(spec.numPartitions);
}

} // namespace facebook::velox::cudf_velox
