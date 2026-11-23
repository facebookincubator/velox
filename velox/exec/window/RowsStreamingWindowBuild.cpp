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
    tsan_atomic<bool>* nonReclaimableSection,
    uint64_t maxRetainedBytes)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      hasRangeFrame_(hasRangeFrame(windowNode)),
      maxRetainedBytes_(std::max<uint64_t>(maxRetainedBytes, 1)),
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

int64_t RowsStreamingWindowBuild::numRetainedRows() const {
  int64_t numRows = pendingRowCount_;
  for (const auto& windowPartition : windowPartitions_) {
    numRows += windowPartition->numRows();
  }
  return numRows;
}

bool RowsStreamingWindowBuild::reachedRetainedBytesBudget() const {
  // RANGE frames must retain the whole incomplete peer group, so they are never
  // throttled on bytes.
  if (hasRangeFrame_ || !estimatedRowSize_.has_value()) {
    return false;
  }
  const uint64_t retainedBytes =
      static_cast<uint64_t>(numRetainedRows()) * estimatedRowSize_.value();
  return retainedBytes >= maxRetainedBytes_;
}

void RowsStreamingWindowBuild::updatePendingRowBudget() {
  if (!estimatedRowSize_.has_value()) {
    return;
  }
  const uint64_t rowBudget = maxRetainedBytes_ / estimatedRowSize_.value();
  maxPendingRows_ = static_cast<vector_size_t>(std::min<uint64_t>(
      std::max<uint64_t>(rowBudget, 1),
      std::numeric_limits<vector_size_t>::max()));
}

bool RowsStreamingWindowBuild::hasRowsToDrain() const {
  for (const auto& windowPartition : windowPartitions_) {
    if (windowPartition->numRows() > 0) {
      return true;
    }
  }
  return false;
}

bool RowsStreamingWindowBuild::needsInput() {
  // Stop accepting input once two partitions are buffered: the head can be
  // output while the tail keeps accepting input.
  if (windowPartitions_.size() >= 2) {
    return false;
  }

  // Within a single not-yet-complete partition, 'windowPartitions_.size()'
  // stays 1, so the check above never throttles and a large partition would
  // accumulate its entire input (held by reference) before any output. For
  // ROWS-frame functions (e.g. row_number/rank) every buffered row is
  // immediately emittable, so once the retained input reaches the byte budget,
  // ask the driver to drain output instead - draining calls
  // 'removeProcessedRows()', which releases the retained input vectors.
  // On the same condition 'addInput()' flushes the pending rows into a
  // partition, so a drainable partition always exists when this returns false.
  if (!reachedRetainedBytesBudget()) {
    return true;
  }

  // Only throttle while the operator still has rows it can emit. Refusing
  // input when nothing is emittable would leave the driver unable to either
  // feed or drain this operator, and the task would stall. Accepting input
  // past the budget grows memory, which is the lesser failure.
  return !hasRowsToDrain();
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

  // Skipped for RANGE frames, which are never throttled on bytes and so have no
  // use for the estimate: 'estimateFlatSize()' walks the vector on every batch.
  if (!hasRangeFrame_ && input->size() > 0) {
    // Floored at one byte: an encoded input can estimate to fewer bytes than
    // rows, and a zero row size would make the retained-byte total zero and
    // silently disable throttling.
    const int64_t rowSize = std::max<int64_t>(
        static_cast<int64_t>(input->estimateFlatSize()) / input->size(), 1);
    estimatedRowSize_ =
        std::max<int64_t>(estimatedRowSize_.value_or(0), rowSize);
    updatePendingRowBudget();
  }

  vector_size_t rangeStart = 0;
  for (auto row = 0; row < input->size(); ++row) {
    const bool hasPreviousRow = row > 0 || partitionKeyValues_.hasValue();
    if (isNewPartition(input, row)) {
      flushRange(input, rangeStart, row);
      addPartitionInputs(true);
      rangeStart = row;
    }
    // Flush pending rows into a partition once the output-row target is hit,
    // or once the retained-byte budget is reached with nothing yet drainable.
    // The byte-budget trigger exists only so 'needsInput()' never throttles a
    // single wide partition without leaving output for the driver to drain;
    // gating it on 'hasRowsToDrain()' keeps it to one flush, instead of
    // splitting every remaining row of this input into its own range while the
    // budget stays exceeded.
    if (hasPreviousRow &&
        (pendingRowCount_ >= numRowsPerOutput_ ||
         (pendingRowCount_ >= maxPendingRows_ && !hasRowsToDrain()))) {
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
