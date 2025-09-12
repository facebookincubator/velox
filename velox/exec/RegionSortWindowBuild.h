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

#pragma once

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/SortWindowBuild.h"
#include "velox/exec/Spiller.h"
#include "velox/exec/WindowBuild.h"

namespace facebook::velox::exec {
// todo: rewrite the annos
// Sorts input data of the Window by {partition keys, sort keys}
// to identify window partitions. This sort fully orders
// rows as needed for window function computation.
class RegionSortWindowBuild : public WindowBuild {
 public:
  RegionSortWindowBuild(
      const std::shared_ptr<const core::WindowNode>& node,
      // todo: change to uint32_t
      int numRegions,
      velox::memory::MemoryPool* pool,
      common::PrefixSortConfig&& prefixSortConfig,
      const common::SpillConfig* spillConfig,
      tsan_atomic<bool>* nonReclaimableSection,
      folly::Synchronized<common::SpillStats>* spillStats);

  ~RegionSortWindowBuild() override {
    pool_->release();
  }

  bool needsInput() override {
    // todo: should we change this?
    // No partitions are available yet, so can consume input rows.
    return currentRegion_ < 0;
  }

  void addInput(RowVectorPtr input) override;

  void spill() override;

  std::optional<common::SpillStats> spilledStats() const override;

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::shared_ptr<WindowPartition> nextPartition() override;

  // todo: this must be taken good care of...
  std::optional<int64_t> estimateRowSize() override {
    LOG(WARNING) << "CALL";
    auto region = std::max(currentRegion_, 0);
    if (region >= numRegions_) {
      return std::nullopt;
    }
    if (windowBuilds_[region]) {
      return windowBuilds_[region]->estimateRowSize();
    }
    return std::nullopt;
  }

 private:
  bool switchToNextRegion();

  void ensureInputFits(const RowVectorPtr& input);

  const int numRegions_;

  const size_t numPartitionKeys_;

  std::unique_ptr<HashPartitionFunction> regionPartitionFunction_;

  std::vector<std::unique_ptr<SortWindowBuild>> windowBuilds_;

  bool spilled_{false};

  std::vector<uint32_t> regionIdsBuffer_;

  int currentRegion_ = -1;

  // Compare flags for partition and sorting keys. Compare flags for partition
  // keys are set to default values. Compare flags for sorting keys match
  // sorting order specified in the plan node.
  //
  // Used to sort 'data_' while spilling and in Prefix sort.
  const std::vector<CompareFlags> compareFlags_;

  memory::MemoryPool* const pool_;

  // Config for Prefix-sort.
  const common::PrefixSortConfig prefixSortConfig_;

  folly::Synchronized<common::SpillStats>* const spillStats_;
};
} // namespace facebook::velox::exec
