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

#include "velox/exec/MultiGroupSortWindowBuild.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::exec {

MultiGroupSortWindowBuild::MultiGroupSortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& node,
    int32_t numRegions,
    velox::memory::MemoryPool* pool,
    common::PrefixSortConfig&& prefixSortConfig,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection,
    folly::Synchronized<OperatorStats>* opStats,
    folly::Synchronized<common::SpillStats>* spillStats)
    : WindowBuild(node, pool, spillConfig, nonReclaimableSection),
      numGroups_(numRegions),
      numPartitionKeys_{node->partitionKeys().size()},
      pool_(pool),
      spillStats_(spillStats) {
  VELOX_CHECK_NOT_NULL(pool_);
  data_.reset();

  std::vector<column_index_t> keyChannels(numPartitionKeys_);
  for (int i = 0; i < numPartitionKeys_; i++) {
    keyChannels[i] = inputChannels_[i];
  }
  groupPartitionFunction_ = std::make_unique<HashPartitionFunction>(
      false, numGroups_, node->inputType(), keyChannels);
  groupWindowBuilds_.resize(numGroups_);
  for (int i = 0; i < numGroups_; i++) {
    groupWindowBuilds_[i] = std::make_unique<SortWindowBuild>(
        node,
        pool,
        std::move(prefixSortConfig),
        spillConfig,
        nonReclaimableSection,
        opStats,
        spillStats);
  }
}

void MultiGroupSortWindowBuild::addInput(RowVectorPtr input) {
  VELOX_CHECK_LT(currentGroup_, 0);

  groupIdsBuffer_.resize(input->size());
  groupPartitionFunction_->partition(*input, groupIdsBuffer_);

  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  ensureInputFits(input);

  for (auto row = 0; row < input->size(); ++row) {
    auto& windowBuild = groupWindowBuilds_[groupIdsBuffer_[row]];
    windowBuild->addDecodedInputRow(decodedInputVectors_, row);
  }

  numRows_ += input->size();
}

bool MultiGroupSortWindowBuild::switchToNextGroup() {
  if (currentGroup_ >= numGroups_) {
    return false;
  }

  if (currentGroup_ >= 0) {
    groupWindowBuilds_[currentGroup_].reset();
  }
  currentGroup_++;
  if (currentGroup_ >= numGroups_) {
    return false;
  }

  VELOX_CHECK_NOT_NULL(groupWindowBuilds_[currentGroup_]);
  // WindowBuild starts processing the partitions when 'noMoreInput' is called,
  // which allocates additional memory. We want to defer the memory allocation
  // as late as possible to reduce memory usage, so we don't call 'noMoreInput'
  // until the group's data is to be consumed.
  groupWindowBuilds_[currentGroup_]->noMoreInput();
  return true;
}

void MultiGroupSortWindowBuild::ensureInputFits(const RowVectorPtr& input) {
  if (spillConfig_ == nullptr) {
    // Spilling is disabled.
    return;
  }

  if (numRows_ == 0) {
    // Nothing to spill.
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  VELOX_CHECK_LT(currentGroup_, 0);
  for (auto& windowBuild : groupWindowBuilds_) {
    windowBuild->ensureInputFits(input);
  }
}

void MultiGroupSortWindowBuild::spill() {
  VELOX_CHECK_LT(currentGroup_, 0);
  for (auto& windowBuild : groupWindowBuilds_) {
    windowBuild->spill();
  }
  spilled_ = true;
}

std::optional<common::SpillStats> MultiGroupSortWindowBuild::spilledStats()
    const {
  if (!spilled_) {
    return std::nullopt;
  }
  return {spillStats_->copy()};
}

void MultiGroupSortWindowBuild::noMoreInput() {
  if (numRows_ == 0) {
    return;
  }

  if (spilled_) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();
  }

  switchToNextGroup();

  VELOX_CHECK_EQ(currentGroup_, 0);
}

std::shared_ptr<WindowPartition> MultiGroupSortWindowBuild::nextPartition() {
  VELOX_CHECK_GE(currentGroup_, 0);
  VELOX_CHECK_LT(currentGroup_, numGroups_);
  VELOX_CHECK_NOT_NULL(groupWindowBuilds_[currentGroup_]);
  return groupWindowBuilds_[currentGroup_]->nextPartition();
}

std::optional<int64_t> MultiGroupSortWindowBuild::estimateRowSize() {
  auto region = std::max(currentGroup_, 0);
  if (region >= numGroups_) {
    return std::nullopt;
  }

  if (groupWindowBuilds_[region]) {
    return groupWindowBuilds_[region]->estimateRowSize();
  }

  return std::nullopt;
}

bool MultiGroupSortWindowBuild::hasNextPartition() {
  // Check if the build hasn't begun or has finished.
  if (currentGroup_ < 0 || currentGroup_ >= numGroups_) {
    return false;
  }

  VELOX_CHECK_NOT_NULL(groupWindowBuilds_[currentGroup_]);
  if (groupWindowBuilds_[currentGroup_]->hasNextPartition()) {
    return true;
  }

  if (switchToNextGroup()) {
    return hasNextPartition();
  }
  return false;
}
} // namespace facebook::velox::exec
