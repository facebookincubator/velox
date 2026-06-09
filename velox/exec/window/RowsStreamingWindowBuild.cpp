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

#include "velox/exec/window/RowsStreamingWindowBuild.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/window/VectorWindowPartition.h"

#include <algorithm>

namespace facebook::velox::exec::window {

namespace {

bool hasRangeFrame(const std::shared_ptr<const core::WindowNode>& windowNode) {
  for (const auto& function : windowNode->windowFunctions()) {
    if (function.frame.type == core::WindowNode::WindowType::kRange) {
      return true;
    }
  }
  return false;
}

void appendUnique(
    std::vector<column_index_t>& channels,
    column_index_t channel) {
  if (std::find(channels.begin(), channels.end(), channel) == channels.end()) {
    channels.push_back(channel);
  }
}

// Returns the deduplicated input channels referenced by 'keyInfo', in
// first-seen order.
std::vector<column_index_t> keyChannels(
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(keyInfo.size());
  for (const auto& key : keyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  return channels;
}

// Returns the deduplicated input channels referenced by the partition and sort
// keys, in first-seen order.
std::vector<column_index_t> keyChannels(
    const std::vector<std::pair<column_index_t, core::SortOrder>>&
        partitionKeyInfo,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(partitionKeyInfo.size() + sortKeyInfo.size());
  for (const auto& key : partitionKeyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  for (const auto& key : sortKeyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  return channels;
}
} // namespace

RowsStreamingWindowBuild::RowsStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      hasRangeFrame_(hasRangeFrame(windowNode)),
      partitionKeyValues_(keyChannels(partitionKeyInfo_, inputChannels_), pool),
      peerKeyValues_(keyChannels(sortKeyInfo_, inputChannels_), pool),
      boundaryKeyChannels_(
          keyChannels(partitionKeyInfo_, sortKeyInfo_, inputChannels_)),
      pool_(pool) {
  VELOX_CHECK_NOT_NULL(pool_);
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::window::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      this);
}

bool RowsStreamingWindowBuild::needsInput() {
  // We need input if there is no or only partition.
  return windowPartitions_.size() < 2;
}

void RowsStreamingWindowBuild::ensureInputPartition() {
  if (windowPartitions_.empty() || windowPartitions_.back()->complete()) {
    windowPartitions_.emplace_back(
        std::make_shared<VectorWindowPartition>(
            inputChannels_, inversedInputChannels_, sortKeyInfo_, pool_));
  }
}

void RowsStreamingWindowBuild::addPartitionInputs(bool finished) {
  if (currentRanges_.empty()) {
    if (finished && !windowPartitions_.empty() &&
        !windowPartitions_.back()->complete()) {
      windowPartitions_.back()->setComplete();
    }
    return;
  }

  ensureInputPartition();
  auto partition =
      std::static_pointer_cast<VectorWindowPartition>(windowPartitions_.back());
  for (const auto& range : currentRanges_) {
    partition->addRows(range.input, range.startRow, range.endRow);
  }

  if (finished) {
    windowPartitions_.back()->setComplete();
  }

  currentRanges_.clear();
  pendingRowCount_ = 0;
}

void RowsStreamingWindowBuild::addInput(RowVectorPtr input) {
  loadBoundaryColumns(input);

  vector_size_t rangeStart = 0;
  for (auto row = 0; row < input->size(); ++row) {
    const bool hasPreviousRow = row > 0 || partitionKeyValues_.hasValue();
    if (isNewPartition(input, row)) {
      flushRange(input, rangeStart, row);
      addPartitionInputs(true);
      rangeStart = row;
    }
    if (hasPreviousRow && pendingRowCount_ >= numRowsPerOutput_) {
      // Needs to wait the peer group ready for range frame.
      if (hasRangeFrame_) {
        if (isNewPeerGroup(input, row)) {
          flushRange(input, rangeStart, row);
          addPartitionInputs(false);
          rangeStart = row;
        }
      } else {
        flushRange(input, rangeStart, row);
        addPartitionInputs(false);
        rangeStart = row;
      }
    }

    ++pendingRowCount_;
  }

  flushRange(input, rangeStart, input->size());
  if (input->size() > 0) {
    partitionKeyValues_.capture(input, input->size() - 1);
    peerKeyValues_.capture(input, input->size() - 1);
  }
}

void RowsStreamingWindowBuild::noMoreInput() {
  addPartitionInputs(true);
  partitionKeyValues_.reset();
  peerKeyValues_.reset();
}

std::shared_ptr<WindowPartition> RowsStreamingWindowBuild::nextPartition() {
  // Remove the processed output partition from the queue.
  //
  // NOTE: the window operator only calls this after processing a completed
  // partition.
  if (!windowPartitions_.empty() && windowPartitions_.front()->complete() &&
      windowPartitions_.front()->numRows() == 0) {
    windowPartitions_.pop_front();
  }

  VELOX_CHECK(hasNextPartition());
  return windowPartitions_.front();
}

bool RowsStreamingWindowBuild::hasNextPartition() {
  // Checks if there is a window partition that is either incomplete or
  // completed but has unconsumed rows.
  for (auto it = windowPartitions_.rbegin(); it != windowPartitions_.rend();
       ++it) {
    const auto& windowPartition = *it;
    if (!windowPartition->complete() || windowPartition->numRows() > 0) {
      return true;
    }
  }

  return false;
}

void RowsStreamingWindowBuild::flushRange(
    const RowVectorPtr& input,
    vector_size_t start,
    vector_size_t end) {
  if (start >= end) {
    return;
  }
  currentRanges_.emplace_back(input, start, end);
}

bool RowsStreamingWindowBuild::isNewPartition(
    const RowVectorPtr& input,
    vector_size_t row) const {
  if (row == 0 && !partitionKeyValues_.hasValue()) {
    return false;
  }
  return !compareRowsEqual(input, row, partitionKeyInfo_, partitionKeyValues_);
}

bool RowsStreamingWindowBuild::isNewPeerGroup(
    const RowVectorPtr& input,
    vector_size_t row) const {
  if (row == 0 && !peerKeyValues_.hasValue()) {
    return false;
  }
  return !compareRowsEqual(input, row, sortKeyInfo_, peerKeyValues_);
}

bool RowsStreamingWindowBuild::compareRowsEqual(
    const RowVectorPtr& input,
    vector_size_t row,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const SingleRowValues& previousValues) const {
  if (row == 0) {
    return previousValues.equals(input, row);
  }

  for (const auto& key : keyInfo) {
    const auto inputColumn = inputChannels_[key.first];
    if (!input->childAt(inputColumn)
             ->equalValueAt(input->childAt(inputColumn).get(), row - 1, row)) {
      return false;
    }
  }
  return true;
}

void RowsStreamingWindowBuild::loadBoundaryColumns(
    const RowVectorPtr& input) const {
  for (const auto channel : boundaryKeyChannels_) {
    input->childAt(channel)->loadedVector();
  }
}

} // namespace facebook::velox::exec::window
