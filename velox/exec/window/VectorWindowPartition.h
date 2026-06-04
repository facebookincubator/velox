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

#include "velox/exec/WindowPartition.h"
#include "velox/exec/window/RowBlock.h"
#include "velox/exec/window/RowColumnsSnapshot.h"

#include <deque>
#include <utility>
#include <vector>

namespace facebook::velox::exec::window {

/// Provides WindowPartition accessors over retained input vectors.
class VectorWindowPartition : public WindowPartition {
 public:
  /// Constructs a partial window partition over retained input vector ranges.
  VectorWindowPartition(
      const std::vector<column_index_t>& inputChannels,
      const std::vector<column_index_t>& inputMapping,
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          sortKeyInfo,
      memory::MemoryPool* pool);

  /// Returns the number of retained rows in this partition.
  vector_size_t numRows() const override {
    return totalRows_;
  }

  /// Returns the number of retained rows available for processing.
  vector_size_t numRowsForProcessing(
      vector_size_t /*partitionOffset*/) const override {
    return totalRows_;
  }

  /// Rejects RowContainer rows because this partition stores vector ranges.
  void addRows(const std::vector<char*>& rows) override;

  /// Adds a retained input vector row range to this partition.
  void addRows(
      const RowVectorPtr& input,
      vector_size_t startRow,
      vector_size_t endRow);

  /// Removes processed rows from the front of this partition.
  void removeProcessedRows(vector_size_t numRows) override;

  /// Extracts a contiguous row range from retained input vectors.
  void extractColumn(
      int32_t columnIndex,
      vector_size_t partitionOffset,
      vector_size_t numRows,
      vector_size_t resultOffset,
      const VectorPtr& result) const override;

  /// Extracts arbitrary row positions from retained input vectors.
  void extractColumn(
      int32_t columnIndex,
      folly::Range<const vector_size_t*> rowNumbers,
      vector_size_t resultOffset,
      const VectorPtr& result) const override;

  /// Extracts null positions from a retained input vector column.
  void extractNulls(
      int32_t columnIndex,
      vector_size_t partitionOffset,
      vector_size_t numRows,
      const BufferPtr& nullsBuffer) const override;

  using WindowPartition::extractNulls;

  /// Computes peer group bounds for retained input vector rows.
  std::pair<vector_size_t, vector_size_t> computePeerBuffers(
      vector_size_t start,
      vector_size_t end,
      vector_size_t prevPeerStart,
      vector_size_t prevPeerEnd,
      vector_size_t* rawPeerStarts,
      vector_size_t* rawPeerEnds) override;

  /// Computes k range frame bounds for retained input vector rows.
  void computeKRangeFrameBounds(
      bool isStartBound,
      bool isPreceding,
      column_index_t frameColumn,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerStarts,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) const override;

 private:
  class VectorAccessor;

  // Returns true if the vector value at 'row' is NaN.
  template <typename T>
  bool isNanAt(const VectorPtr& vector, vector_size_t row) const;

  // Finds the retained block and local row for a partition-relative row.
  std::pair<size_t, vector_size_t> findBlock(vector_size_t row) const;

  // Rebuilds block prefix sums after processed rows are removed.
  void rebuildPrefixSums();

  // Retained input vector row ranges.
  std::deque<RowBlock> blocks_;

  // Prefix sums of retained row counts by block.
  std::vector<vector_size_t> blockPrefixSums_{0};

  // Number of retained rows in this partition.
  vector_size_t totalRows_{0};

  // Absolute partition offset of the first retained row.
  vector_size_t startRow_{0};

  // Last row from the previously processed range, if needed for peer grouping.
  RowColumnsSnapshot previousRow_;

  // Original input channels that must be copied to compare previous rows.
  std::vector<column_index_t> previousRowKeyChannels_;

  // Maps window input columns to retained input vector columns.
  const std::vector<column_index_t> inputChannels_;

  // Pool used for copied previous-row key snapshots. The owner must outlive
  // this partition.
  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox::exec::window
