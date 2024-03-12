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

#include "velox/exec/WindowBuild.h"

namespace facebook::velox::exec {

/// If the data has already been sorted according to the partition key and sort
/// key, there is no need to hold all the data of a partition in memory for Rank
/// and row_number functions. RankWindowBuild adopts a streaming method to
/// construct WindowPartition, which can reduce the occurrence of Out Of Memory
/// (OOM).
class RankLikeWindowBuild : public WindowBuild {
 public:
  RankLikeWindowBuild(
      const std::shared_ptr<const core::WindowNode>& windowNode,
      velox::memory::MemoryPool* pool,
      const common::SpillConfig* spillConfig,
      tsan_atomic<bool>* nonReclaimableSection);

  void addInput(RowVectorPtr input) override;

  void spill() override {
    VELOX_UNREACHABLE();
  }

  std::optional<common::SpillStats> spilledStats() const override {
    return std::nullopt;
  }

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::unique_ptr<WindowPartition> nextPartition() override;

  bool needsInput() override {
    return !isFinished_;
  }

 private:
  // Vector of pointers to each input row in the data_ RowContainer.
  // Rows are erased from data_ when they are output from the
  // Window operator.
  std::vector<std::vector<char*>> sortedRows_;

  // Holds input rows within the current partition.
  std::vector<char*> inputRows_;

  // Indices of  the start row (in sortedRows_) of each partition in
  // the RowContainer data_. This auxiliary structure helps demarcate
  // partitions.
  std::vector<vector_size_t> partitionOffsets_;

  // Used to compare rows based on partitionKeys.
  char* previousRow_ = nullptr;

  // Current partition being output. Used to construct WindowPartitions
  // during resetPartition.
  vector_size_t currentPartition_ = -1;

  bool isFinished_ = false;
};

} // namespace facebook::velox::exec
