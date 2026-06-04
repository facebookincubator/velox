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

#include "velox/exec/window/RowBlock.h"
#include "velox/exec/window/RowColumnsSnapshot.h"
#include "velox/exec/window/WindowBuild.h"

#include <deque>
#include <optional>
#include <vector>

namespace facebook::velox::exec::window {

/// Unlike PartitionStreamingWindowBuild, RowsStreamingWindowBuild is capable of
/// processing window functions as rows arrive within a single partition,
/// without the need to wait for the entire window partition to be ready. This
/// approach can significantly reduce memory usage, especially when a single
/// partition contains a large amount of data. It is particularly suited for
/// optimizing rank, dense_rank and row_number functions, as well as aggregate
/// window functions with a default frame.
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

  std::optional<exec::SpillStats> spilledStats() const override {
    return std::nullopt;
  }

  void noMoreInput() override;

  bool hasNextPartition() override;

  std::shared_ptr<WindowPartition> nextPartition() override;

  bool needsInput() override;

 private:
  // Flushes rows in [start, end) from 'input' as a vector block.
  void
  flushBlock(const RowVectorPtr& input, vector_size_t start, vector_size_t end);

  // Adds input rows to the current partition, or creates a new partition if it
  // does not exist.
  void addPartitionInputs(bool finished);

  // Invoked before add input to ensure there is an open (incomplete) partition
  // to accept new input. The function creates a new one at the tail of
  // 'windowPartitions_' if it is empty or the last partition is already
  // completed.
  void ensureInputPartition();

  // Returns true if 'row' starts a new partition relative to the previous row.
  bool isNewPartition(const RowVectorPtr& input, vector_size_t row) const;

  // Returns true if 'row' starts a new peer group relative to the previous row.
  bool isNewPeerGroup(const RowVectorPtr& input, vector_size_t row) const;

  // Compares 'row' with the previous row using the specified key columns.
  bool compareRowsEqual(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo)
      const;

  // Loads only key columns needed to detect partition and peer boundaries.
  void loadBoundaryColumns(const RowVectorPtr& input) const;

  // Sets to true if this window node has range frames.
  const bool hasRangeFrame_;

  // Blocks of input rows buffered for the current partition.
  std::vector<RowBlock> currentBlocks_;

  // Used to compare the first row of an input vector with the last row of the
  // previous input vector.
  RowColumnsSnapshot previousRow_;

  // Original input channels that must be copied to compare previous rows.
  std::vector<column_index_t> previousRowKeyChannels_;

  // Original input channels used to detect partition and peer boundaries.
  std::vector<column_index_t> boundaryKeyChannels_;

  // Pool used for copied previous-row key snapshots.
  memory::MemoryPool* const pool_;

  // Number of rows accumulated since the last partial flush.
  vector_size_t pendingRowCount_{0};

  // The output gets next partition from the head of 'windowPartitions_' and
  // input adds to the next partition from the tail of 'windowPartitions_'.
  std::deque<std::shared_ptr<WindowPartition>> windowPartitions_;
};

} // namespace facebook::velox::exec::window
