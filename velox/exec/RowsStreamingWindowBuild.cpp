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

#include "velox/exec/RowsStreamingWindowBuild.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/WindowFunction.h"

namespace facebook::velox::exec {

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
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::RowsStreamingWindowBuild::RowsStreamingWindowBuild",
      this);

  // Create the first WindowPartition.
  windowPartitions_.emplace_back(std::make_shared<WindowPartition>(
      data_.get(), inversedInputChannels_, sortKeyInfo_));
}

void RowsStreamingWindowBuild::addPartitionInputs(bool finished) {
  if (inputRows_.empty()) {
    return;
  }

  windowPartitions_.back()->addRows(inputRows_);

  if (finished) {
    windowPartitions_.back()->setComplete();
    ++inputPartition_;
    // Create a new partition for the next input.
    windowPartitions_.emplace_back(std::make_shared<WindowPartition>(
        data_.get(), inversedInputChannels_, sortKeyInfo_));
  }

  inputRows_.clear();
  inputRows_.shrink_to_fit();
}

void RowsStreamingWindowBuild::addInput(RowVectorPtr input) {
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  for (auto row = 0; row < input->size(); ++row) {
    char* newRow = data_->newRow();

    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }

    if (previousRow_ != nullptr &&
        compareRowsWithKeys(previousRow_, newRow, partitionKeyInfo_)) {
      addPartitionInputs(true);
    }

    if (previousRow_ != nullptr && inputRows_.size() >= numRowsPerOutput_) {
      // Needs to wait the peer group ready for range frame.
      if (hasRangeFrame_) {
        if (compareRowsWithKeys(previousRow_, newRow, sortKeyInfo_)) {
          addPartitionInputs(false);
        }
      } else {
        addPartitionInputs(false);
      }
    }

    inputRows_.push_back(newRow);
    previousRow_ = newRow;
  }
}

void RowsStreamingWindowBuild::noMoreInput() {
  addPartitionInputs(true);
}

std::shared_ptr<WindowPartition> RowsStreamingWindowBuild::nextPartition() {
  VELOX_CHECK(hasNextPartition());
  outputPartition_++;
  auto output = std::move(windowPartitions_.front());
  windowPartitions_.pop_front();
  return output;
}

bool RowsStreamingWindowBuild::hasNextPartition() {
  return !windowPartitions_.empty() && outputPartition_ + 1 <= inputPartition_;
}

} // namespace facebook::velox::exec
