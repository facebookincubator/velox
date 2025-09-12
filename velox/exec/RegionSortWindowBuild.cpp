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

namespace {
// todo: eliminate this. this is not used.
std::vector<CompareFlags> makeCompareFlags(
    int32_t numPartitionKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  std::vector<CompareFlags> compareFlags;
  compareFlags.reserve(numPartitionKeys + sortingOrders.size());

  for (auto i = 0; i < numPartitionKeys; ++i) {
    compareFlags.push_back({});
  }

  for (const auto& order : sortingOrders) {
    compareFlags.push_back(
        {order.isNullsFirst(), order.isAscending(), false /*equalsOnly*/});
  }

  return compareFlags;
}
} // namespace

RegionSortWindowBuild::RegionSortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& node,
    int numRegions,
    velox::memory::MemoryPool* pool,
    common::PrefixSortConfig&& prefixSortConfig,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection,
    folly::Synchronized<common::SpillStats>* spillStats)
    : WindowBuild(node, pool, spillConfig, nonReclaimableSection),
      numRegions_(numRegions),
      numPartitionKeys_{node->partitionKeys().size()},
      compareFlags_{makeCompareFlags(numPartitionKeys_, node->sortingOrders())},
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
        spillStats);
  }
  // todo: to make sure we won't use incorrect data, release the original
  // data container...
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
    // todo: check npe?
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
  // LOG(WARNING) << "CALL";
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

  // todo: would using the same pool cause any troule when calling
  // pool_->release()?
}

std::shared_ptr<WindowPartition> RegionSortWindowBuild::nextPartition() {
  VELOX_CHECK_GE(currentRegion_, 0);
  VELOX_CHECK_LT(currentRegion_, numRegions_);
  VELOX_CHECK_NOT_NULL(windowBuilds_[currentRegion_]);
  return windowBuilds_[currentRegion_]->nextPartition();
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
