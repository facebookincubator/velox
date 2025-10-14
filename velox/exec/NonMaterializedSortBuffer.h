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

#include "velox/exec/MaterializedSortBuffer.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/SortBufferBase.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
/// A utility class to accumulate data inside and output the sorted result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
///
/// Uses non-materialize mode to sort input vectors, serializing only the sort
/// key columns and two additional index columns into the RowContainer. These
/// index columns are the vector indices and the row indices of each vector.
/// After sorting, rows are gathered and copied from the input vectors using
/// these indices.
class NonMaterializedSortBuffer final : public SortBufferBase {
 public:
  NonMaterializedSortBuffer(
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  ~NonMaterializedSortBuffer() override;

  void addInput(const VectorPtr& input) override;

  /// Indicates no more input and triggers either of:
  ///  - In-memory sorting on rows stored in 'data_' if spilling is not enabled.
  ///  - Finish spilling and setup the sort merge reader for the un-spilling
  ///  processing for the output.
  void noMoreInput() override;

  std::optional<uint64_t> estimateOutputRowSize() const override;

  RowVectorPtr getOutput(vector_size_t maxOutputRows) override;

 private:
  void prepareOutputVector(
      RowVectorPtr& output,
      const RowTypePtr& outputType,
      vector_size_t outputBatchSize) const;

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize) override;

  void gatherCopyOutput();

  int64_t estimateFlatInputBytes(const VectorPtr& input) const override;

  int64_t estimateIncrementalBytes(
      const VectorPtr& input,
      uint64_t outOfLineBytes,
      int64_t flatInputBytes) const override;

  const std::vector<SpillSortKey> sortingKeys_;

  // Two index columns materialized in the row container to index each input
  // row. The first column points to the vector in 'inputs_' and the second
  // points to the row in the pointed vector.
  const RowTypePtr indexType_{ROW({BIGINT(), BIGINT()})};

  std::vector<RowVectorPtr> inputs_;

  VectorPtr rowIndices_;

  // The column projection map between 'data_' and 'indexOutput_', containing
  // two columns: 0 for vector indices and 1 for row indices.
  std::vector<IdentityProjection> indexColumnMap_;

  // Reusable indices vector.
  RowVectorPtr indexOutput_;

  // The number of received input bytes.
  uint64_t numInputBytes_{0};
};
} // namespace facebook::velox::exec
