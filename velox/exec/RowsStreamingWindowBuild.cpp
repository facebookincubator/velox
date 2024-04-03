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
#include "velox/exec/RowsStreamingWindowPartition.h"

namespace facebook::velox::exec {

RowsStreamingWindowBuild::RowsStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection) {}

void RowsStreamingWindowBuild::buildNextInputOrPartition(bool isFinished) {
  if (windowPartitions_.size() <= inputCurrentPartition_) {
    windowPartitions_.push_back(std::make_shared<RowsStreamingWindowPartition>(
        data_.get(),
        folly::Range<char**>(nullptr, nullptr),
        inversedInputChannels_,
        sortKeyInfo_));
  }

  windowPartitions_[inputCurrentPartition_]->addNewRows(inputRows_);

  if (isFinished) {
    windowPartitions_[inputCurrentPartition_]->setInputRowsFinished();
    inputCurrentPartition_++;
  }

  inputRows_.clear();
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
      buildNextInputOrPartition(true);
    }

    // Wait for the peers to be ready in single partition; these peers are the
    // rows that have identical values in the ORDER BY clause.
    if (previousRow_ != nullptr && inputRows_.size() >= numRowsPerOutput_ &&
        compareRowsWithKeys(previousRow_, newRow, sortKeyInfo_)) {
      buildNextInputOrPartition(false);
    }

    inputRows_.push_back(newRow);
    previousRow_ = newRow;
  }
}

void RowsStreamingWindowBuild::noMoreInput() {
  buildNextInputOrPartition(true);
}

std::shared_ptr<WindowPartition> RowsStreamingWindowBuild::nextPartition() {
  if (outputCurrentPartition_ > 0) {
    windowPartitions_[outputCurrentPartition_].reset();
  }

  return windowPartitions_[++outputCurrentPartition_];
}

bool RowsStreamingWindowBuild::hasNextPartition() {
  return windowPartitions_.size() > 0 &&
      outputCurrentPartition_ <= int(windowPartitions_.size() - 2);
}

} // namespace facebook::velox::exec
