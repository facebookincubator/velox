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

#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/SortBuffer.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
/// A utility class to accumulate data inside and output the sorted result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
///
/// Uses hybrid mode to sort input vectors, serializing only the sort key
/// columns and two additional index columns into the RowContainer. These index
/// columns are the vector indices and the row indices of each vector. After
/// sorting, rows are gathered and copied from the input vectors using these
/// indices.
class HybridSortBuffer final : public ISortBuffer {
 public:
  HybridSortBuffer(
      const RowTypePtr& input,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  ~HybridSortBuffer() override;

  void addInput(const VectorPtr& input) override;

  /// Indicates no more input and triggers either of:
  ///  - In-memory sorting on rows stored in 'data_' if spilling is not enabled.
  ///  - Finish spilling and setup the sort merge reader for the un-spilling
  ///  processing for the output.
  void noMoreInput() override;

  /// Returns the sorted output rows in batch.
  RowVectorPtr getOutput(vector_size_t maxOutputRows) override;

  /// Indicates if this sort buffer can spill or not.
  bool canSpill() const override {
    return spillConfig_ != nullptr;
  }

  /// Invoked to spill all the rows from 'data_'.
  void spill() override;

  memory::MemoryPool* pool() const {
    return pool_;
  }

  std::optional<uint64_t> estimateOutputRowSize() const override;

 private:
  // Ensures there is sufficient memory reserved to process 'input'.
  void ensureInputFits(const VectorPtr& input);

  // Reserves memory for output processing. If reservation cannot be increased,
  // spills enough to make output fit.
  void ensureOutputFits(vector_size_t outputBatchSize);

  // Reserves memory for sort. If reservation cannot be increased, spills enough
  // to make output fit.
  void ensureSortFits();

  void prepareOutputVector(
      RowVectorPtr& output,
      const RowTypePtr& outputType,
      vector_size_t outputBatchSize) const;

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize);

  void gatherCopyOutput(
      const RowVectorPtr& output,
      const RowVectorPtr& indexOutput,
      const std::vector<char*, memory::StlAllocator<char*>>& sortedRows,
      uint64_t offset) const;

  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  //
  // TODO: Adds offset to make it can do multiple round sorting.
  void sortInput(uint64_t numRows);

  // Invoked to initialize reader to read the spilled data from storage for
  // output processing.
  void prepareOutputWithSpill();

  void getOutputWithoutSpill();

  void getOutputWithSpill();

  // Spill during input stage.
  void spillInput();

  // Spill during output stage.
  void spillOutput();

  // Spill remaining sorted input vectors.
  void runSpill(
      NoRowContainerSpiller* spiller,
      int64_t numInputs,
      uint64_t offset) const;

  void finishInputSpill();

  // Finish output spill, there is only one spill partition;
  void finishOutputSpill();

  // Returns true if the sort buffer has spilled, regardless of during input or
  // output processing. If spilled() is true, it means the sort buffer is in
  // minimal memory mode and could not be spilled further.
  bool hasSpilled() const;

  const RowTypePtr input_;
  std::vector<RowVectorPtr> inputs_;
  const std::vector<SpillSortKey> sortingKeys_;
  const std::vector<CompareFlags> sortCompareFlags_;

  velox::memory::MemoryPool* const pool_;

  // The flag is passed from the associated operator such as OrderBy or
  // TableWriter to indicate if this sort buffer object is under non-reclaimable
  // execution section or not.
  tsan_atomic<bool>* const nonReclaimableSection_;

  // Configuration settings for prefix-sort.
  const common::PrefixSortConfig prefixSortConfig_;

  const common::SpillConfig* const spillConfig_;

  folly::Synchronized<common::SpillStats>* const spillStats_;

  // SpillFiles group for the input spills.
  std::vector<SpillFiles> inputSpillFileGroups_;

  // Two additional index columns the vector indices and the row indices of each
  // vector.
  const RowTypePtr indexType_{ROW({BIGINT(), BIGINT()})};

  // The column projection map between 'input_' and sort columns in 'data_'.
  std::vector<IdentityProjection> columnMap_;

  // The column projection map between 'data_' and 'indexOutput_', containing
  // two columns: 0 for vector indices and 1 for row indices.
  std::vector<IdentityProjection> indexColumnMap_;

  // Indicates no more input. Once it is set, addInput() can't be called on this
  // sort buffer object.
  bool noMoreInput_ = false;

  // The number of received input rows.
  uint64_t numInputRows_ = 0;

  // The number of received input bytes.
  uint64_t numInputBytes_ = 0;

  // Used to store the input data in row format.
  std::unique_ptr<RowContainer> data_;

  std::vector<char*, memory::StlAllocator<char*>> sortedRows_;
  std::unique_ptr<NoRowContainerSpiller> inputSpiller_;
  std::unique_ptr<NoRowContainerSpiller> outputSpiller_;

  SpillPartitionSet outputSpillPartitionSet_;

  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;

  // Used to read vectors sequentially from spill files, which have already been
  // sorted before spilling.
  std::unique_ptr<UnorderedStreamReader<BatchStream>> batchStreamReader_;

  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;

  std::vector<vector_size_t> spillSourceRows_;

  // Reusable output vector.
  RowVectorPtr output_;
  // Reusable indices vector.
  RowVectorPtr indexOutput_;

  // Estimated size of a single output row by using the max
  // 'data_->estimateRowSize()' across all accumulated data set.
  std::optional<uint64_t> estimatedOutputRowSize_{};

  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};
};
} // namespace facebook::velox::exec
