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
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
class SortBufferBase {
 public:
  SortBufferBase(
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  virtual ~SortBufferBase() = default;

  virtual void addInput(const VectorPtr& input) = 0;

  virtual void noMoreInput() = 0;

  virtual std::optional<uint64_t> estimateOutputRowSize() const = 0;

  /// Returns the sorted output rows in batch.
  virtual RowVectorPtr getOutput(vector_size_t maxOutputRows);

  /// Indicates if this sort buffer can spill or not.
  bool canSpill() const {
    return spillConfig_ != nullptr;
  }

  void spill();

  memory::MemoryPool* pool() const {
    return pool_;
  }

 protected:
  // Returns true if the sort buffer has spilled, regardless of during input or
  // output processing. If spilled() is true, it means the sort buffer is in
  // minimal memory mode and could not be spilled further.
  virtual bool hasSpilled() const {
    return false;
  }

  virtual int64_t estimateFlatInputBytes(const VectorPtr& input) const = 0;

  virtual int64_t estimateIncrementalBytes(
      const VectorPtr& input,
      uint64_t outOfLineBytes,
      int64_t flatInputBytes) const = 0;

  virtual void prepareOutput(vector_size_t batchSize) = 0;

  virtual void spillInput() {
    VELOX_UNSUPPORTED("Spill is not supported yet.");
  }

  virtual void spillOutput() {
    VELOX_UNSUPPORTED("Spill is not supported yet.");
  }

  virtual void getOutputWithoutSpill() {
    VELOX_UNSUPPORTED("Spill is not supported yet.");
  }

  virtual void getOutputWithSpill() {
    VELOX_UNSUPPORTED("Spill is not supported yet.");
  }

  void ensureSortFitsImpl() const;

  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  //
  // TODO: Adds offset to make it can do multiple round sorting.
  void sortInput(uint64_t numRows);

  // Ensures there is sufficient memory reserved to process 'input'.
  void ensureInputFits(const VectorPtr& input);

  // Reserves memory for output processing. If reservation cannot be increased,
  // spills enough to make output fit.
  void ensureOutputFits(vector_size_t batchSize);

  const RowTypePtr inputType_;

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

  std::vector<char*, memory::StlAllocator<char*>> sortedRows_;

  // The column projection map between 'input_' and 'spillerStoreType_' as sort
  // buffer stores the sort columns first in 'data_'.
  std::vector<IdentityProjection> columnMap_;

  // Indicates no more input. Once it is set, addInput() can't be called on this
  // sort buffer object.
  bool noMoreInput_ = false;

  // The number of received input rows.
  uint64_t numInputRows_ = 0;

  // Used to store the input data in row format.
  std::unique_ptr<RowContainer> data_;

  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;

  // Reusable output vector.
  RowVectorPtr output_;

  // Estimated size of a single output row by using the max
  // 'data_->estimateRowSize()' across all accumulated data set.
  std::optional<uint64_t> estimatedOutputRowSize_{};

  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;

  std::vector<vector_size_t> spillSourceRows_;

  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};
};
} // namespace facebook::velox::exec
