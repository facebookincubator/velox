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

#include "velox/exec/RegionSortWindowBuild.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::exec {

RegionSortWindowBuild::RegionSortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& node,
    int32_t numRegions,
    velox::memory::MemoryPool* pool,
    common::PrefixSortConfig&& prefixSortConfig,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection,
    folly::Synchronized<OperatorStats>* opStats,
    folly::Synchronized<common::SpillStats>* spillStats)
    : WindowBuild(node, pool, spillConfig, nonReclaimableSection),
      numRegions_(numRegions),
      numPartitionKeys_{node->partitionKeys().size()},
      pool_(pool),
      prefixSortConfig_(prefixSortConfig),
      spillStats_(spillStats) {
  std::vector<column_index_t> keyChannels(numPartitionKeys_);
  for (int i = 0; i < numPartitionKeys_; i++) {
    keyChannels[i] = inputChannels_[i];
  }
  regionPartitionFunction_ = std::make_unique<HashPartitionFunction>(
      false, numRegions_, node->inputType(), keyChannels);
  windowBuilds_.resize(numRegions_);
  for (int i = 0; i < numRegions_; i++) {
    windowBuilds_[i] = std::make_unique<SortWindowBuild>(
        node,
        pool,
        prefixSortConfig_,
        spillConfig,
        nonReclaimableSection,
        opStats,
        spillStats);
  }
  data_.reset();
  VELOX_CHECK_NOT_NULL(pool_);
}

void RegionSortWindowBuild::addInput(RowVectorPtr input) {
  regionIdsBuffer_.resize(input->size());
  regionPartitionFunction_->partition(*input, regionIdsBuffer_);

  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  ensureInputFits(input);

  VELOX_CHECK_LT(currentRegion_, 0);
  for (auto row = 0; row < input->size(); ++row) {
    auto& windowBuild = windowBuilds_[regionIdsBuffer_[row]];
    windowBuild->addInputRow(decodedInputVectors_, row, input->childrenSize());
  }

  numRows_ += input->size();
}

bool RegionSortWindowBuild::switchToNextRegion() {
  if (currentRegion_ >= numRegions_) {
    return false;
  }

  if (currentRegion_ >= 0) {
    windowBuilds_[currentRegion_].reset();
  }
  currentRegion_++;
  if (currentRegion_ >= numRegions_) {
    return false;
  }

  VELOX_CHECK_NOT_NULL(windowBuilds_[currentRegion_]);
  windowBuilds_[currentRegion_]->noMoreInput();
  return true;
}

void RegionSortWindowBuild::ensureInputFits(const RowVectorPtr& input) {
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

  VELOX_CHECK_LT(currentRegion_, 0);
  for (auto& windowBuild : windowBuilds_) {
    windowBuild->ensureInputFits(input);
  }
}

void RegionSortWindowBuild::spill() {
  VELOX_CHECK_LT(currentRegion_, 0);
  for (auto& windowBuild : windowBuilds_) {
    windowBuild->spill();
  }
  spilled_ = true;
}

std::optional<common::SpillStats> RegionSortWindowBuild::spilledStats() const {
  if (!spilled_) {
    return std::nullopt;
  }
  return {spillStats_->copy()};
}

void RegionSortWindowBuild::noMoreInput() {
  if (numRows_ == 0) {
    return;
  }

  if (spilled_) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();
  }

  switchToNextRegion();

  VELOX_CHECK_EQ(currentRegion_, 0);
}

std::shared_ptr<WindowPartition> RegionSortWindowBuild::nextPartition() {
  VELOX_CHECK_GE(currentRegion_, 0);
  VELOX_CHECK_LT(currentRegion_, numRegions_);
  VELOX_CHECK_NOT_NULL(windowBuilds_[currentRegion_]);
  return windowBuilds_[currentRegion_]->nextPartition();
}

std::optional<int64_t> RegionSortWindowBuild::estimateRowSize() {
  auto region = std::max(currentRegion_, 0);
  if (region >= numRegions_) {
    return std::nullopt;
  }
  if (windowBuilds_[region]) {
    return windowBuilds_[region]->estimateRowSize();
  }
  return std::nullopt;
}

bool RegionSortWindowBuild::hasNextPartition() {
  // Check if the build hasn't begun or has finished.
  if (currentRegion_ < 0 || currentRegion_ >= numRegions_) {
    return false;
  }
  VELOX_CHECK_NOT_NULL(windowBuilds_[currentRegion_]);
  if (windowBuilds_[currentRegion_]->hasNextPartition()) {
    return true;
  }
  if (switchToNextRegion()) {
    return hasNextPartition();
  }
  return false;
}
} // namespace facebook::velox::exec
