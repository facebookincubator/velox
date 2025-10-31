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

namespace facebook::velox::exec {
// Divides the input data into several groups by partition keys, then sorts
// input data of the Window by {partition keys, sort keys} to identify window
// partitions with SortWindowBuild.
class MultiGroupSortWindowBuild : public WindowBuild {
 public:
  MultiGroupSortWindowBuild(
      const std::shared_ptr<const core::WindowNode>& node,
      int32_t numGroups,
      velox::memory::MemoryPool* pool,
      common::PrefixSortConfig&& prefixSortConfig,
      const common::SpillConfig* spillConfig,
      tsan_atomic<bool>* nonReclaimableSection,
      folly::Synchronized<OperatorStats>* opStats,
      folly::Synchronized<common::SpillStats>* spillStats);

  ~MultiGroupSortWindowBuild() override {
    pool_->release();
  }

  bool needsInput() override {
    // No groups are available yet, so can consume input rows.
    return currentGroup_ < 0;
  }

  void addInput(RowVectorPtr input) override;

  void spill() override;

  std::optional<common::SpillStats> spilledStats() const override;

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::shared_ptr<WindowPartition> nextPartition() override;

  std::optional<int64_t> estimateRowSize() override;

 private:
  // The current group's WindowBuild has finished producing all the partitions.
  // Release all the memory of current group's WindowBuild, and then switch to
  // next group's WindowBuild as the new current one.
  bool switchToNextGroup();

  void ensureInputFits(const RowVectorPtr& input);

  const int32_t numGroups_;

  const size_t numPartitionKeys_;

  memory::MemoryPool* const pool_;

  folly::Synchronized<common::SpillStats>* const spillStats_;

  // Partition each row to the corresponding group.
  std::unique_ptr<HashPartitionFunction> groupPartitionFunction_;

  // WindowBuilds for each group.
  std::vector<std::unique_ptr<SortWindowBuild>> groupWindowBuilds_;

  bool spilled_{false};

  // Buffers the groupIds for each row. Reused across addInput callings.
  std::vector<uint32_t> groupIdsBuffer_;

  int32_t currentGroup_ = -1;
};
} // namespace facebook::velox::exec
