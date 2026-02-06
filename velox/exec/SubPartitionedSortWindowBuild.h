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
// Divides the input data into several sub partitions by partition keys, then
// sequentially sorts input data of each sub partition by {partition keys, sort
// keys} to identify window partitions with SortWindowBuild. As each sub
// partition has a smaller working set, the memory used by sorting is reduced.
// Besides, once a sub partition is completely consumed, its memory could be
// released immediately.
class SubPartitionedSortWindowBuild : public WindowBuild {
 public:
  SubPartitionedSortWindowBuild(
      const std::shared_ptr<const core::WindowNode>& node,
      int32_t numSubPartitions,
      velox::memory::MemoryPool* pool,
      common::PrefixSortConfig&& prefixSortConfig,
      const common::SpillConfig* spillConfig,
      tsan_atomic<bool>* nonReclaimableSection,
      folly::Synchronized<OperatorStats>* opStats,
      folly::Synchronized<common::SpillStats>* spillStats,
      filesystems::File::IoStats* spillFsStats);

  ~SubPartitionedSortWindowBuild() override {
    pool_->release();
  }

  bool needsInput() override {
    // No sub partitions are available yet, so can consume input rows.
    return currentSubPartition_ < 0;
  }

  void addInput(RowVectorPtr input) override;

  void spill() override;

  std::optional<common::SpillStats> spilledStats() const override;

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::shared_ptr<WindowPartition> nextPartition() override;

  std::optional<int64_t> estimateRowSize() override;

 private:
  // The current sub partition's WindowBuild has finished producing all the
  // data. Release all the memory of current sub partition's WindowBuild, and
  // then switch to next sub partition's WindowBuild as the new current one.
  bool switchToNextSubPartition();

  void ensureInputFits(const RowVectorPtr& input);

  const int32_t numSubPartitions_;

  const size_t numPartitionKeys_;

  memory::MemoryPool* const pool_;

  folly::Synchronized<common::SpillStats>* const spillStats_;

  filesystems::File::IoStats* const spillFsStats_{nullptr};

  // Divide input rows to the corresponding sub partitions.
  std::unique_ptr<HashPartitionFunction> subPartitioningFunction_;

  // WindowBuilds for each sub partition.
  std::vector<std::unique_ptr<SortWindowBuild>> subWindowBuilds_;

  bool spilled_{false};

  // Buffers the subPartitionIds for each row. Reused across addInput calls.
  std::vector<uint32_t> subPartitionIdsBuffer_;

  int32_t currentSubPartition_ = -1;
};
} // namespace facebook::velox::exec
