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

#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/SortBufferBase.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
class SortInputSpiller;
class SortOutputSpiller;

/// A utility class to accumulate data inside and output the sorted result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
class MaterializedSortBuffer final : public SortBufferBase {
 public:
  MaterializedSortBuffer(
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  ~MaterializedSortBuffer() override;

  void addInput(const VectorPtr& input) override;

  /// Indicates no more input and triggers either of:
  ///  - In-memory sorting on rows stored in 'data_' if spilling is not enabled.
  ///  - Finish spilling and setup the sort merge reader for the un-spilling
  ///  processing for the output.
  void noMoreInput() override;

  std::optional<uint64_t> estimateOutputRowSize() const override;

 private:
  // Reserves memory for sort. If reservation cannot be increased, spills enough
  // to make output fit.
  void ensureSortFits();

  void updateEstimatedOutputRowSize();

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize) override;

  // Invoked to initialize reader to read the spilled data from storage for
  // output processing.
  void prepareOutputWithSpill();

  void getOutputWithoutSpill() override;

  void getOutputWithSpill() override;

  // Spill during input stage.
  void spillInput() override;

  // Spill during output stage.
  void spillOutput() override;

  // Finish spill, and we shouldn't get any rows from non-spilled partition as
  // there is only one hash partition for SortBuffer.
  void finishSpill();

  // Returns true if the sort buffer has spilled, regardless of during input or
  // output processing. If spilled() is true, it means the sort buffer is in
  // minimal memory mode and could not be spilled further.
  bool hasSpilled() const override;

  int64_t estimateFlatInputBytes(const VectorPtr& input) const override;

  int64_t estimateIncrementalBytes(
      const VectorPtr& input,
      uint64_t outOfLineBytes,
      int64_t flatInputBytes) const override;

  // The data type of the rows stored in 'data_' and spilled on disk. The
  // sort key columns are stored first then the non-sorted data columns.
  RowTypePtr spillerStoreType_;

  std::unique_ptr<SortInputSpiller> inputSpiller_;

  std::unique_ptr<SortOutputSpiller> outputSpiller_;

  SpillPartitionSet spillPartitionSet_;
};
} // namespace facebook::velox::exec
