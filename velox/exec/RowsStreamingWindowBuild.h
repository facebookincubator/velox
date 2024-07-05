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

/// Unlike StreamingWindowBuild, RowsStreamingWindowBuild is capable of
/// processing window functions as rows arrive within a single partition,
/// without the need to wait for the entire partition to be ready. This approach
/// can significantly reduce memory usage, especially when a single partition
/// contains a large amount of data. It is particularly suited for optimizing
/// rank and row_number functions, as well as aggregate window functions with a
/// default frame.
class RowsStreamingWindowBuild : public WindowBuild {
 public:
  RowsStreamingWindowBuild(
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

  std::shared_ptr<WindowPartition> nextPartition() override;

  bool needsInput() override {
    // No partitions are available or the currentPartition is the last available
    // one, so can consume input rows.
    return windowPartitions_.size() == 0 ||
        outputPartition_ == windowPartitions_.size() - 1;
  }

 private:
  void buildNextInputOrPartition(bool isFinished);

  // Holds input rows within the current partition.
  std::vector<char*> inputRows_;

  // Used to compare rows based on partitionKeys.
  char* previousRow_ = nullptr;

  // Current partition being output. Used to return the WidnowPartitions.
  vector_size_t outputPartition_ = -1;

  // Current partition when adding input. Used to construct WindowPartitions.
  vector_size_t inputPartition_ = 0;

  // Holds all the WindowPartitions.
  std::vector<std::shared_ptr<WindowPartition>> windowPartitions_;
};

} // namespace facebook::velox::exec
