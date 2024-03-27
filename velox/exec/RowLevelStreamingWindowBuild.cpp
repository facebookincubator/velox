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

#include "velox/exec/RowLevelStreamingWindowBuild.h"

namespace facebook::velox::exec {

RowLevelStreamingWindowBuild::RowLevelStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection) {}

void RowLevelStreamingWindowBuild::buildNextInputOrPartition(bool isFinished) {
  sortedRows_.push_back(inputRows_);
  if (windowPartitions_.size() <= inputCurrentPartition_) {
    auto partition =
        folly::Range(sortedRows_.back().data(), sortedRows_.back().size());

    windowPartitions_.push_back(std::make_shared<WindowPartition>(
        data_.get(),
        partition,
        inputColumns_,
        sortKeyInfo_,
        ProcessingUnit::kRow));
  }

  windowPartitions_[inputCurrentPartition_]->insertNewBatch(sortedRows_.back());

  if (isFinished) {
    windowPartitions_[inputCurrentPartition_]->setTotalNum(
        currentPartitionNum_ - 1);

    inputCurrentPartition_++;
    currentPartitionNum_ = 1;
  }

  inputRows_.clear();
}

void RowLevelStreamingWindowBuild::addInput(RowVectorPtr input) {
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  for (auto row = 0; row < input->size(); ++row) {
    currentPartitionNum_++;
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
    if (previousRow_ != nullptr && inputRows_.size() > 0 &&
        compareRowsWithKeys(previousRow_, newRow, sortKeyInfo_)) {
      buildNextInputOrPartition(false);
    }

    inputRows_.push_back(newRow);
    previousRow_ = newRow;
  }
}

void RowLevelStreamingWindowBuild::noMoreInput() {
  isFinished_ = true;
  buildNextInputOrPartition(true);
}

std::shared_ptr<WindowPartition> RowLevelStreamingWindowBuild::nextPartition() {
  return windowPartitions_[outputCurrentPartition_++];
}

bool RowLevelStreamingWindowBuild::hasNextPartition() {
  return windowPartitions_.size() > 0 &&
      outputCurrentPartition_ <= int(windowPartitions_.size() - 1);
}

} // namespace facebook::velox::exec
