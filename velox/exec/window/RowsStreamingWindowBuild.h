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

#include "velox/exec/window/RowRange.h"
#include "velox/exec/window/SingleRowValues.h"
#include "velox/exec/window/WindowBuild.h"

#include <deque>
#include <limits>
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
      tsan_atomic<bool>* nonReclaimableSection,
      // Byte budget for input retained within a single not-yet-complete
      // partition before 'needsInput()' asks the driver to drain. The Window
      // operator passes 'preferredOutputBatchBytes'.
      uint64_t maxRetainedBytes);

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
  // Flushes rows in [start, end) from 'input' as a vector row range.
  void
  flushRange(const RowVectorPtr& input, vector_size_t start, vector_size_t end);

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
  // 'previousValues' holds the captured values of the previous input vector's
  // last row over the same keys; it is used only when 'row' is 0.
  bool compareRowsEqual(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
      const SingleRowValues& previousValues) const;

  // Loads only key columns needed to detect partition and peer boundaries.
  void loadBoundaryColumns(const RowVectorPtr& input) const;

  // Returns the number of buffered input rows not yet consumed by output:
  // rows pending in 'currentRanges_' plus unconsumed rows in
  // 'windowPartitions_'. Widened to 64 bits because it sums row counts across
  // partitions and so is not bounded by 'vector_size_t'.
  int64_t numRetainedRows() const;

  // Returns true for a ROWS-frame build whose retained (by-reference) input for
  // the current not-yet-complete partition has reached 'maxRetainedBytes_'.
  // Drives throttling in 'needsInput()', and in 'addInput()' the flush that
  // makes the pending rows drainable, so the build never stops requesting
  // input without leaving output to drain.
  bool reachedRetainedBytesBudget() const;

  // Recomputes 'maxPendingRows_' from the current row-size estimate. Called
  // once per input rather than per row.
  void updatePendingRowBudget();

  // Returns true if some partition holds rows the window operator can emit.
  // 'hasNextPartition()' is not sufficient for this: it also reports true for
  // an incomplete partition holding no rows, which the operator turns into a
  // null output.
  bool hasRowsToDrain() const;

  // Sets to true if this window node has range frames.
  const bool hasRangeFrame_;

  // Upper bound on the estimated bytes of input retained (by reference) for a
  // single not-yet-complete partition before 'needsInput()' asks the driver to
  // stop feeding and drain output. Bounds peak memory for large partitions.
  const uint64_t maxRetainedBytes_;

  // Largest per-batch average row size in bytes seen so far: the maximum over
  // each input's 'estimateFlatSize() / size()'. Converts the retained row count
  // into bytes. A maximum rather than a running average, so that a batch of
  // wide rows is not masked by earlier narrow ones, which would leave the build
  // accepting input well past the byte budget.
  std::optional<int64_t> estimatedRowSize_;

  // Ranges of input rows buffered for the current partition.
  std::vector<RowRange> currentRanges_;

  // Partition-key values from the last row of the previous input vector, used
  // to detect partition boundaries across vectors.
  SingleRowValues partitionKeyValues_;

  // Sort-key values from the last row of the previous input vector, used to
  // detect peer-group boundaries across vectors.
  SingleRowValues peerKeyValues_;

  // Original input channels used to detect partition and peer boundaries.
  std::vector<column_index_t> boundaryKeyChannels_;

  // Pool used to create window partitions.
  memory::MemoryPool* const pool_;

  // Number of rows accumulated since the last partial flush.
  vector_size_t pendingRowCount_{0};

  // 'maxRetainedBytes_' expressed in rows at the current 'estimatedRowSize_',
  // so the per-row check in 'addInput()' is an integer compare rather than a
  // walk of 'windowPartitions_'. Valid there only because that flush also
  // requires '!hasRowsToDrain()', i.e. every partition holds zero rows, which
  // makes 'pendingRowCount_' the whole retained-row count. Left at the maximum
  // for RANGE frames, which are never throttled on bytes.
  vector_size_t maxPendingRows_{std::numeric_limits<vector_size_t>::max()};

  // The output gets next partition from the head of 'windowPartitions_' and
  // input adds to the next partition from the tail of 'windowPartitions_'.
  std::deque<std::shared_ptr<WindowPartition>> windowPartitions_;
};

} // namespace facebook::velox::exec::window
