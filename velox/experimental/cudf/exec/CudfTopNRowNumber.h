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

/// GPU-accelerated TopNRowNumber: partitioned top-N with row_number.
/// Used when the optimizer rewrites ROW_NUMBER() OVER (...) WHERE rn <= limit
/// into a TopNRowNumber plan node.
///
/// Retained state is bounded to O(limit * distinct partitions) rather than
/// the full input: each input batch is locally reduced to its own top-`limit`
/// rows per partition, then merged with the running `candidates_` (which
/// already holds the top-`limit` rows per partition across all prior
/// batches) and pruned back down to `limit` rows per partition. See
/// reduceBatchToLocalCandidates() and mergeAndPruneCandidates().
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

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
  /// Reduces a single input batch to its own top-`limit_` rows per
  /// partition: sorts by partition+ordering keys, computes row_number
  /// locally, filters the sort permutation to row_number <= limit_, and only
  /// then gathers the full payload for the surviving rows.
  CudfVectorPtr reduceBatchToLocalCandidates(
      const CudfVectorPtr& cudfInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  /// Merges two candidate sets (each already sorted by partition+ordering
  /// keys and containing at most `limit_` rows per partition) via
  /// cudf::merge, recomputes row_number over the merged rows, and prunes
  /// back down to `limit_` rows per partition.
  CudfVectorPtr mergeAndPruneCandidates(
      const CudfVectorPtr& previous,
      const CudfVectorPtr& incoming,
      rmm::cuda_stream_view stream,
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

  // Bounded candidate state: at most `limit_` rows per distinct partition
  // seen so far, sorted by partition+ordering keys, with the schema of
  // inputType_ (the row_number column, if any, is only materialized once in
  // doGetOutput()). Updated incrementally after each input batch instead of
  // retaining the full input.
  CudfVectorPtr candidates_;
  bool finished_{false};
  std::unique_ptr<CudaEvent> cudaEvent_;
};

} // namespace facebook::velox::cudf_velox
