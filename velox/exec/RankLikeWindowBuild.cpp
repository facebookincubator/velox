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

#include "velox/exec/RankLikeWindowBuild.h"

namespace facebook::velox::exec {

RankLikeWindowBuild::RankLikeWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection) {
  partitionOffsets_.push_back(0);
}

void RankLikeWindowBuild::addInput(RowVectorPtr input) {
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
      sortedRows_.push_back(inputRows_);
      partitionOffsets_.push_back(0);
      inputRows_.clear();
    }

    inputRows_.push_back(newRow);
    previousRow_ = newRow;
  }
  partitionOffsets_.push_back(inputRows_.size());
  sortedRows_.push_back(inputRows_);
  inputRows_.clear();
}

void RankLikeWindowBuild::noMoreInput() {
  isFinished_ = true;
  inputRows_.clear();
}

std::unique_ptr<WindowPartition> RankLikeWindowBuild::nextPartition() {
  currentPartition_++;

  if (currentPartition_ > 0) {
    // Erase data_ and sortedRows;
    data_->eraseRows(folly::Range<char**>(
        sortedRows_[currentPartition_ - 1].data(),
        sortedRows_[currentPartition_ - 1].size()));
    sortedRows_[currentPartition_ - 1].clear();
  }

  auto partition = folly::Range(
      sortedRows_[currentPartition_].data(),
      sortedRows_[currentPartition_].size());

  auto offset = 0;
  for (auto i = currentPartition_; partitionOffsets_[i] != 0; i--) {
    offset += partitionOffsets_[i];
  }
  return std::make_unique<WindowPartition>(
      data_.get(), partition, inputColumns_, sortKeyInfo_, offset);
}

bool RankLikeWindowBuild::hasNextPartition() {
  return sortedRows_.size() > 0 && currentPartition_ != sortedRows_.size() - 1;
}

} // namespace facebook::velox::exec
