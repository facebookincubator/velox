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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// GPU-accelerated TopNRowNumber: partitioned top-N with row_number, rank,
/// or dense_rank. Used when the optimizer rewrites a ranking window with
/// a rank <= limit filter into a TopNRowNumber plan node.
///
/// Each input batch is locally reduced to rows with ranks <= limit per
/// partition, then merged with the running `candidates_` (which already holds
/// the qualifying rows per partition across all prior batches) and pruned
/// using the same rank limit. See reduceBatchToLocalCandidates() and
/// mergeAndPruneCandidates().
///
/// For row_number, retained state is bounded to O(limit * distinct partitions)
/// rather than the full input. rank and dense_rank retain all qualifying peers,
/// so their candidate sets can exceed `limit` rows per partition and grow to
/// the full input when all rows tie.
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  /// Checks whether the ranking function and key types are supported on GPU.
  static bool canRunOnGPU(const core::TopNRowNumberNode& node);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  // Reduces a single input batch to rows with ranks <= limit_ per partition:
  // sorts by partition+ordering keys, computes row_number, rank, or dense_rank
  // locally, filters the sort permutation to ranks <= limit_, and only then
  // gathers the full payload for the surviving rows. rank and dense_rank keep
  // all qualifying peers, even when their number exceeds limit_.
  CudfVectorPtr reduceBatchToLocalCandidates(
      const CudfVectorPtr& cudfInput,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  // Merges two candidate sets (each already sorted by partition+ordering
  // keys and containing rows with ranks <= limit_ per partition) via
  // cudf::merge, recomputes the selected ranking function over the merged
  // rows, and prunes using the same rank limit while retaining qualifying
  // peers. Adding rows cannot improve a row's rank, so previously pruned rows
  // cannot qualify after a merge.
  CudfVectorPtr mergeAndPruneCandidates(
      const CudfVectorPtr& previous,
      const CudfVectorPtr& incoming,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  const std::shared_ptr<const core::TopNRowNumberNode> node_;
  const int32_t limit_;
  const bool generateRowNumber_;
  const TypePtr inputType_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  // Combined partition+ordering sort/merge key (partition keys first, then
  // ordering keys), precomputed once and reused for every stable_sorted_order
  // / cudf::merge call.
  std::vector<cudf::size_type> allSortKeys_;
  std::vector<cudf::order> allOrders_;
  std::vector<cudf::null_order> allNullOrders_;
  // Positions of the partition keys within the narrower key-only table
  // selected via allSortKeys_ (always the prefix [0, partitionKeyIndices_.
  // size()), since partition keys are listed first in allSortKeys_).
  std::vector<cudf::size_type> localPartitionKeyIndices_;

  // Positions of ordering keys after the partition-key prefix of allSortKeys_.
  std::vector<cudf::size_type> localSortKeyIndices_;

  // Candidate state: rows with ranks <= limit_ per distinct partition seen so
  // far, sorted by partition+ordering keys, with the schema of inputType_
  // (the row_number, rank, or dense_rank column, if any, is only materialized
  // once in doGetOutput()). Updated incrementally after each input batch.
  // row_number retains at most limit_ rows per partition; rank and dense_rank
  // also retain all qualifying peers, which can require keeping the full input.
  CudfVectorPtr candidates_;
  bool finished_{false};
  std::unique_ptr<CudaEvent> cudaEvent_;
};

} // namespace facebook::velox::cudf_velox
