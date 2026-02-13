/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/SubPartitionedSortWindowBuild.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::exec {

SubPartitionedSortWindowBuild::SubPartitionedSortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& node,
    int32_t numSubPartitions,
    velox::memory::MemoryPool* pool,
    common::PrefixSortConfig&& prefixSortConfig,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection,
    folly::Synchronized<OperatorStats>* opStats,
    exec::SpillStats* spillStats)
    : WindowBuild(node, pool, spillConfig, nonReclaimableSection),
      numSubPartitions_(numSubPartitions),
      numPartitionKeys_{node->partitionKeys().size()},
      pool_(pool),
      spillStats_(spillStats) {
  VELOX_CHECK_NOT_NULL(pool_);
  data_.reset();

  std::vector<column_index_t> keyChannels(numPartitionKeys_);
  for (int i = 0; i < numPartitionKeys_; i++) {
    keyChannels[i] = inputChannels_[i];
  }
  subPartitioningFunction_ = std::make_unique<HashPartitionFunction>(
      false, numSubPartitions_, node->inputType(), keyChannels);
  subWindowBuilds_.resize(numSubPartitions_);
  for (int i = 0; i < numSubPartitions_; i++) {
    subWindowBuilds_[i] = std::make_unique<SortWindowBuild>(
        node,
        pool,
        common::PrefixSortConfig(prefixSortConfig),
        spillConfig,
        nonReclaimableSection,
        opStats,
        spillStats);
  }
}

void SubPartitionedSortWindowBuild::addInput(RowVectorPtr input) {
  VELOX_CHECK_LT(currentSubPartition_, 0);

  subPartitionIdsBuffer_.resize(input->size());
  subPartitioningFunction_->partition(*input, subPartitionIdsBuffer_);

  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  ensureInputFits(input);

  for (auto row = 0; row < input->size(); ++row) {
    auto& windowBuild = subWindowBuilds_[subPartitionIdsBuffer_[row]];
    windowBuild->addDecodedInputRow(decodedInputVectors_, row);
  }

  numRows_ += input->size();
}

bool SubPartitionedSortWindowBuild::switchToNextSubPartition() {
  if (currentSubPartition_ >= numSubPartitions_) {
    return false;
  }

  if (currentSubPartition_ >= 0) {
    subWindowBuilds_[currentSubPartition_].reset();
  }
  currentSubPartition_++;
  if (currentSubPartition_ >= numSubPartitions_) {
    return false;
  }

  VELOX_CHECK_NOT_NULL(subWindowBuilds_[currentSubPartition_]);
  // WindowBuild starts processing the partitions when 'noMoreInput' is called,
  // which allocates additional memory. We want to defer the memory allocation
  // as late as possible to reduce memory usage, so we don't call 'noMoreInput'
  // until the sub partition's data is to be consumed.
  subWindowBuilds_[currentSubPartition_]->noMoreInput();
  return true;
}

void SubPartitionedSortWindowBuild::ensureInputFits(const RowVectorPtr& input) {
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

  VELOX_CHECK_LT(currentSubPartition_, 0);
  for (auto& windowBuild : subWindowBuilds_) {
    windowBuild->ensureInputFits(input);
  }
}

void SubPartitionedSortWindowBuild::spill() {
  VELOX_CHECK_LT(currentSubPartition_, 0);
  for (auto& windowBuild : subWindowBuilds_) {
    windowBuild->spill();
  }
  spilled_ = true;
}

std::optional<exec::SpillStats> SubPartitionedSortWindowBuild::spilledStats()
    const {
  if (!spilled_) {
    return std::nullopt;
  }
  return {*spillStats_};
}

void SubPartitionedSortWindowBuild::noMoreInput() {
  if (numRows_ == 0) {
    return;
  }

  if (spilled_) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();
  }

  switchToNextSubPartition();

  VELOX_CHECK_EQ(currentSubPartition_, 0);
}

std::shared_ptr<WindowPartition>
SubPartitionedSortWindowBuild::nextPartition() {
  VELOX_CHECK_GE(currentSubPartition_, 0);
  VELOX_CHECK_LT(currentSubPartition_, numSubPartitions_);
  VELOX_CHECK_NOT_NULL(subWindowBuilds_[currentSubPartition_]);
  return subWindowBuilds_[currentSubPartition_]->nextPartition();
}

std::optional<int64_t> SubPartitionedSortWindowBuild::estimateRowSize() {
  auto subPartition = std::max(currentSubPartition_, 0);
  if (subPartition >= numSubPartitions_) {
    return std::nullopt;
  }

  if (subWindowBuilds_[subPartition]) {
    return subWindowBuilds_[subPartition]->estimateRowSize();
  }

  return std::nullopt;
}

bool SubPartitionedSortWindowBuild::hasNextPartition() {
  // Check if the build hasn't begun or has finished.
  if (currentSubPartition_ < 0 || currentSubPartition_ >= numSubPartitions_) {
    return false;
  }

  VELOX_CHECK_NOT_NULL(subWindowBuilds_[currentSubPartition_]);
  if (subWindowBuilds_[currentSubPartition_]->hasNextPartition()) {
    return true;
  }

  if (switchToNextSubPartition()) {
    return hasNextPartition();
  }
  return false;
}
} // namespace facebook::velox::exec
