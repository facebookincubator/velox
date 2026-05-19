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
} // namespace

RowsStreamingWindowBuild::RowsStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      hasRangeFrame_(hasRangeFrame(windowNode)) {
  initializeRowContainer(pool);
  initializeDecodedInputVectors();
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
            inputChannels_, inversedInputChannels_, sortKeyInfo_));
  }
}

void RowsStreamingWindowBuild::addPartitionInputs(bool finished) {
  if (currentBlocks_.empty()) {
    if (finished && !windowPartitions_.empty() &&
        !windowPartitions_.back()->complete()) {
      windowPartitions_.back()->setComplete();
    }
    return;
  }

  ensureInputPartition();
  auto partition =
      std::static_pointer_cast<VectorWindowPartition>(windowPartitions_.back());
  for (const auto& block : currentBlocks_) {
    partition->addBlock(block.input, block.startRow, block.endRow);
  }

  if (finished) {
    windowPartitions_.back()->setComplete();
  }

  currentBlocks_.clear();
  pendingRowCount_ = 0;
}

void RowsStreamingWindowBuild::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }

  vector_size_t blockStart = 0;
  for (auto row = 0; row < input->size(); ++row) {
    const bool hasPreviousRow = row > 0 || previousRef_.isValid();
    if (isNewPartition(input, row)) {
      flushBlock(input, blockStart, row);
      addPartitionInputs(true);
      blockStart = row;
    }
    if (hasPreviousRow && pendingRowCount_ >= numRowsPerOutput_) {
      // Needs to wait the peer group ready for range frame.
      if (hasRangeFrame_) {
        if (isNewPeerGroup(input, row)) {
          flushBlock(input, blockStart, row);
          addPartitionInputs(false);
          blockStart = row;
        }
      } else {
        flushBlock(input, blockStart, row);
        addPartitionInputs(false);
        blockStart = row;
      }
    }

    ++pendingRowCount_;
  }

  flushBlock(input, blockStart, input->size());
  if (input->size() > 0) {
    previousRef_ = {input, input->size() - 1};
  }
}

void RowsStreamingWindowBuild::noMoreInput() {
  addPartitionInputs(true);
  previousRef_ = {};
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

void RowsStreamingWindowBuild::flushBlock(
    const RowVectorPtr& input,
    vector_size_t start,
    vector_size_t end) {
  if (start >= end) {
    return;
  }
  currentBlocks_.push_back({input, start, end});
}

bool RowsStreamingWindowBuild::isNewPartition(
    const RowVectorPtr& input,
    vector_size_t row) const {
  if (row == 0 && !previousRef_.isValid()) {
    return false;
  }
  return !compareRowsEqual(input, row, partitionKeyInfo_);
}

bool RowsStreamingWindowBuild::isNewPeerGroup(
    const RowVectorPtr& input,
    vector_size_t row) const {
  if (row == 0 && !previousRef_.isValid()) {
    return false;
  }
  return !compareRowsEqual(input, row, sortKeyInfo_);
}

bool RowsStreamingWindowBuild::compareRowsEqual(
    const RowVectorPtr& input,
    vector_size_t row,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo)
    const {
  const auto& previousInput = row == 0 ? previousRef_.input : input;
  const auto previousRow = row == 0 ? previousRef_.row : row - 1;

  for (const auto& key : keyInfo) {
    const auto inputColumn = inputChannels_[key.first];
    if (!previousInput->childAt(inputColumn)
             ->equalValueAt(
                 input->childAt(inputColumn).get(), previousRow, row)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::exec::window
