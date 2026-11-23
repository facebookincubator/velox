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
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"

#include <cudf/copying.hpp>

#include <algorithm>
#include <iterator>
#include <limits>

namespace facebook::velox::cudf_velox {

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
partitionByHashRouting(
    cudf::table_view input,
    const HashRouting& routing,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GE(routing.numTargets(), 1);
  VELOX_CHECK_GE(routing.bucketBounds.front(), 0);
  VELOX_CHECK_LE(routing.bucketBounds.back(), routing.numBuckets);

  // One pass groups the rows by bucket. A target is a range of buckets, so
  // its rows are contiguous in the result.
  // TODO: cudf::hash_partition_indices() (cudf#20891) returns the row indices
  // grouped by bucket instead of the reordered table. With it the rows of one
  // target at a time could be gathered here.
  auto [partitioned, bucketOffsets] = cudf::hash_partition(
      input,
      routing.keyIndices,
      routing.numBuckets,
      cudf::hash_id::HASH_MURMUR3,
      routing.seed,
      stream,
      mr);
  VELOX_CHECK_EQ(bucketOffsets.size(), routing.numBuckets + 1);

  std::vector<cudf::size_type> offsets;
  offsets.reserve(routing.bucketBounds.size());
  for (auto bucket : routing.bucketBounds) {
    offsets.push_back(bucketOffsets[bucket]);
  }
  VELOX_CHECK_EQ(
      offsets.front(), 0, "Rows below the first bucket of the routing");
  VELOX_CHECK_EQ(
      offsets.back(),
      input.num_rows(),
      "Rows above the last bucket of the routing");
  return {std::move(partitioned), std::move(offsets)};
}

std::vector<InputChunk> routeChunkByHash(
    const InputChunk& input,
    const HashRouting& routing,
    rmm::device_async_resource_ref mr) {
  std::vector<InputChunk> chunks(routing.numTargets());
  if (input.empty()) {
    return chunks;
  }

  auto [partitioned, offsets] =
      partitionByHashRouting(input.view, routing, input.stream, mr);
  auto tableOwner = std::shared_ptr<cudf::table>(std::move(partitioned));
  // cudf::split() takes the offsets between the parts only.
  const std::vector<cudf::size_type> splits(
      offsets.begin() + 1, offsets.end() - 1);
  auto views = cudf::split(tableOwner->view(), splits, input.stream);
  VELOX_CHECK_EQ(views.size(), chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (views[i].num_rows() > 0) {
      chunks[i] = InputChunk{
          input.pool, input.type, views[i], input.stream, nullptr, tableOwner};
    }
  }
  return chunks;
}

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    const Options& options)
    : ops_(std::move(ops)), options_(options) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(options_.maxRowsPerLeaf, 0);
  VELOX_CHECK_LE(options_.maxRowsPerLeaf, options_.hardMaxRowsPerLeaf);
  VELOX_CHECK_GE(options_.numBuckets, 2);
  // One leaf holds every bucket.
  leaves_.emplace_back();
  leaves_.back().bucketEnd = options_.numBuckets;
  routing_.keyIndices = ops_->keyIndices();
  routing_.seed = options_.hashSeed;
  routing_.numBuckets = options_.numBuckets;
  updateRouting();
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto prepared = ops_->prepareInput(std::move(rawInput));
  if (prepared.empty()) {
    return;
  }
  nextLeafToDrain_ = 0;

  if (leaves_.size() == 1) {
    insert(0, std::move(prepared));
    return;
  }

  if (routingStale_) {
    updateRouting();
  }
  // One hash partitioning pass takes the batch to its leaves, however often
  // they have been split.
  auto chunks = ops_->routeInput(prepared, routing_);
  VELOX_CHECK_EQ(chunks.size(), leaves_.size());
  prepared = InputChunk{};
  // A leaf that splits puts its children in its place, which moves the leaves
  // after it. Going backwards keeps the leaves that still wait for their
  // chunk where they were.
  for (auto i = chunks.size(); i-- > 0;) {
    insert(i, std::move(chunks[i]));
  }
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput() {
  for (; nextLeafToDrain_ < leaves_.size(); ++nextLeafToDrain_) {
    auto& leaf = leaves_[nextLeafToDrain_];
    if (!leaf.state) {
      continue;
    }
    leaf.rows = 0;
    if (auto output = ops_->finalizeLeaf(std::move(leaf.state))) {
      ++nextLeafToDrain_;
      return output;
    }
  }
  return nullptr;
}

bool PartitionedBufferedState::empty() const {
  return std::none_of(leaves_.begin(), leaves_.end(), [](const Leaf& leaf) {
    return leaf.state != nullptr;
  });
}

void PartitionedBufferedState::insert(
    size_t leafIndex,
    InputChunk bufferedInput) {
  if (bufferedInput.empty()) {
    return;
  }

  // If the leaf is empty, initialize it with the input.
  if (!leaves_[leafIndex].state) {
    auto state = ops_->createLeaf(std::move(bufferedInput));
    if (state) {
      auto& leaf = leaves_[leafIndex];
      leaf.rows = ops_->leafRowCount(*state);
      leaf.state = std::move(state);
      ensureLeafWithinLimit(leafIndex);
    }
    return;
  }

  // If the leaf is not empty, check if the input can be added to it. If not,
  // split the leaf and the input together.
  const auto projectedRows = ops_->estimatedMergedRowUpperBound(
      *leaves_[leafIndex].state, bufferedInput);
  if (projectedRows > options_.maxRowsPerLeaf) {
    splitLeafAndAddInput(leafIndex, std::move(bufferedInput));
    return;
  }

  auto& leaf = leaves_[leafIndex];
  ops_->addInputToLeaf(*leaf.state, std::move(bufferedInput));
  leaf.rows = ops_->leafRowCount(*leaf.state);
  ensureLeafWithinLimit(leafIndex);
}

void PartitionedBufferedState::splitLeafAndAddInput(
    size_t leafIndex,
    InputChunk bufferedInput) {
  VELOX_CHECK_NOT_NULL(leaves_[leafIndex].state);

  const auto bucketBegin = leaves_[leafIndex].bucketBegin;
  const auto bucketEnd = leaves_[leafIndex].bucketEnd;
  const auto numLeafBuckets = static_cast<size_t>(bucketEnd - bucketBegin);
  const auto totalRows = leaves_[leafIndex].rows + bufferedInput.size();

  if (numLeafBuckets == 1) {
    // A single bucket cannot be divided, so the leaf takes the input for as
    // long as that keeps it within the hard limit.
    auto& leaf = leaves_[leafIndex];
    VELOX_CHECK_LE(
        totalRows,
        options_.hardMaxRowsPerLeaf,
        "Cannot partition buffered state any further: {} rows exceeded the "
        "limit of {} rows in one of {} hash buckets. This can happen when "
        "many rows have the same partition key.",
        totalRows,
        options_.hardMaxRowsPerLeaf,
        routing_.numBuckets);
    if (!bufferedInput.empty()) {
      ops_->addInputToLeaf(*leaf.state, std::move(bufferedInput));
      leaf.rows = ops_->leafRowCount(*leaf.state);
    }
    return;
  }

  // Divide the leaf's buckets into as many equal parts as bring it within
  // the limit if its rows are spread evenly over them.
  const auto requiredChildren =
      (totalRows + options_.maxRowsPerLeaf - 1) / options_.maxRowsPerLeaf;
  const auto numChildren =
      std::max<size_t>(2, std::min(requiredChildren, numLeafBuckets));

  HashRouting split;
  split.keyIndices = routing_.keyIndices;
  split.seed = routing_.seed;
  split.numBuckets = routing_.numBuckets;
  split.bucketBounds.reserve(numChildren + 1);
  for (size_t child = 0; child <= numChildren; ++child) {
    split.bucketBounds.push_back(
        bucketBegin +
        static_cast<int32_t>(numLeafBuckets * child / numChildren));
  }

  auto storedPartitions =
      ops_->repartitionLeaf(std::move(leaves_[leafIndex].state), split);
  auto incomingPartitions = bufferedInput.empty()
      ? std::vector<InputChunk>(numChildren)
      : ops_->routeInput(bufferedInput, split);
  VELOX_CHECK_EQ(storedPartitions.size(), numChildren);
  VELOX_CHECK_EQ(incomingPartitions.size(), numChildren);
  bufferedInput = InputChunk{};

  // The children take the place of the leaf, in bucket order.
  std::vector<Leaf> children(numChildren);
  for (size_t child = 0; child < numChildren; ++child) {
    auto& childLeaf = children[child];
    childLeaf.bucketBegin = split.bucketBounds[child];
    childLeaf.bucketEnd = split.bucketBounds[child + 1];
    if (storedPartitions[child]) {
      childLeaf.rows = ops_->leafRowCount(*storedPartitions[child]);
      childLeaf.state = std::move(storedPartitions[child]);
    }
  }
  leaves_.erase(leaves_.begin() + leafIndex);
  leaves_.insert(
      leaves_.begin() + leafIndex,
      std::make_move_iterator(children.begin()),
      std::make_move_iterator(children.end()));
  routingStale_ = true;

  // A child that is still over the limit, as when all rows landed in it, is
  // split again. That moves the children after it, so go backwards.
  for (auto child = numChildren; child-- > 0;) {
    const auto childIndex = leafIndex + child;
    // insert() splits the child again if it does not fit together with its
    // share of the input, so the child is only checked here when it gets no
    // input.
    if (!incomingPartitions[child].empty()) {
      insert(childIndex, std::move(incomingPartitions[child]));
    } else {
      ensureLeafWithinLimit(childIndex);
    }
  }
}

void PartitionedBufferedState::ensureLeafWithinLimit(size_t leafIndex) {
  const auto& leaf = leaves_[leafIndex];
  if (leaf.state && leaf.rows > options_.maxRowsPerLeaf) {
    splitLeafAndAddInput(leafIndex, InputChunk{});
  }
}

void PartitionedBufferedState::updateRouting() {
  routing_.bucketBounds.clear();
  routing_.bucketBounds.reserve(leaves_.size() + 1);
  for (const auto& leaf : leaves_) {
    VELOX_DCHECK(
        routing_.bucketBounds.empty() ||
        routing_.bucketBounds.back() == leaf.bucketBegin);
    routing_.bucketBounds.push_back(leaf.bucketBegin);
  }
  routing_.bucketBounds.push_back(leaves_.back().bucketEnd);
  routingStale_ = false;
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
