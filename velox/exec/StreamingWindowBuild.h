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

#include "velox/exec/Spiller.h"
#include "velox/exec/WindowBuild.h"

namespace facebook::velox::exec {

/// The StreamingWindowBuild is used when the input data is already sorted by
/// {partition keys + order by keys}. The logic identifies partition changes
/// when receiving input rows and splits out WindowPartitions for the Window
/// operator to process.
class StreamingWindowBuild : public WindowBuild {
 public:
  StreamingWindowBuild(
      const std::shared_ptr<const core::WindowNode>& windowNode,
      velox::memory::MemoryPool* pool,
      const common::SpillConfig* spillConfig,
      tsan_atomic<bool>* nonReclaimableSection);

  void addInput(RowVectorPtr input) override;

  void spill() override;

  std::optional<common::SpillStats> spilledStats() const override {
    if (spillers_.size() == 0) {
      return std::nullopt;
    }

    common::SpillStats stats;
    for (auto& spiller : spillers_) {
      if (spiller != nullptr) {
        stats += spiller->stats();
      }
    }
    return stats;
  }

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::unique_ptr<WindowPartition> nextPartition() override;

  bool needsInput() override {
    // No partitions are available or the currentPartition is the last available
    // one, so can consume input rows.
    return partitionStartRows_.size() == 0 ||
        currentPartition_ == partitionStartRows_.size() - 2;
  }

 private:
  void ensureInputFits(const RowVectorPtr& input);

  void setupSpiller();

  void buildNextPartition();

  // Reads next partition from spilled data into 'data_' and
  // 'spilledSortedRows_'.
  void loadNextPartitionFromSpill();

  // Vector of pointers to each input row in the data_ RowContainer.
  // Rows are erased from data_ when they are output from the
  // Window operator.
  std::vector<char*> sortedRows_;

  // Store the spilled sorted rows.
  std::vector<char*> spilledSortedRows_;

  // Holds input rows within the current partition.
  std::vector<char*> inputRows_;

  // Indices of  the start row (in sortedRows_) of each partition in
  // the RowContainer data_. This auxiliary structure helps demarcate
  // partitions.
  std::vector<vector_size_t> partitionStartRows_;

  // Record the rows in each window partition.
  vector_size_t partitionRows_ = -1;

  // Used to compare rows based on partitionKeys.
  char* previousRow_ = nullptr;

  // Current partition being output. Used to construct WindowPartitions
  // during resetPartition.
  vector_size_t currentPartition_ = -1;

  // Current spilled partition index. Used to construct spillers_ and merges_.
  vector_size_t currentSpilledPartition_ = 0;

  // Spiller for contents of the 'data_'.
  std::vector<std::unique_ptr<Spiller>> spillers_;

  // Used to sort-merge spilled data.
  std::vector<std::unique_ptr<TreeOfLosers<SpillMergeStream>>> merges_;

  // allKeyInfo_ is a combination of (partitionKeyInfo_ and sortKeyInfo_).
  // It is used to perform a full sorting of the input rows to be able to
  // separate partitions and sort the rows in it. The rows are output in
  // this order by the operator.
  std::vector<std::pair<column_index_t, core::SortOrder>> allKeyInfo_;

  bool lastRun = false;

  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox::exec
