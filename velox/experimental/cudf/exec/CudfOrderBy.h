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

#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

class CudfOrderBy : public CudfOperatorBase {
 public:
  CudfOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::OrderByNode>& orderByNode);

  /// Builds the GPU merge appended after a kept UcxExchange operator. Together,
  /// the two operators implement MergeExchange on the GPU: the exchange hands
  /// on each received page as one vector, and every page is a sorted slice of
  /// one upstream stream, so the inputs are merged rather than sorted.
  CudfOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::MergeExchangeNode>& mergeExchangeNode);

  /// Largest number of sorted inputs that are merged rather than sorted;
  /// above it the rows are sorted, since the merge's cost grows with the input
  /// count while the sort's does not.
  static constexpr size_t kMaxMergeInputs = 32;

  /// Runtime stats of the merge-exchange variant: the number of sorted inputs
  /// that were merged, and the number of times (at most one) the rows were
  /// sorted because an input was not sorted after all.
  static constexpr const char* kMergedInputs = "mergedInputs";
  static constexpr const char* kSortFallbacks = "sortFallbacks";

  bool needsInput() const override {
    return !finished_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  /// Returns kOutput when built for a MergeExchangeNode, whose input the
  /// exchange operator sharing that node reports, and kBoth for an OrderByNode.
  core::PlanNode::Boundary planNodeBoundary() const override {
    return planNodeBoundary_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  // Initializes cuDF sort keys from the plan node's ordering.
  void initializeSortKeys(
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders);

  // Sorts 'rows' by the sort keys.
  std::unique_ptr<cudf::table> sortRows(
      const cudf::table_view& rows,
      cuda::stream_ref stream);

  // Merges the runs of 'rows' that 'splitPoints' delimit, each sorted by the
  // sort keys; sorts the rows instead when a run is not sorted.
  std::unique_ptr<cudf::table> mergeSortedRuns(
      const cudf::table_view& rows,
      const std::vector<cudf::size_type>& splitPoints,
      cuda::stream_ref stream);

  CudfVectorPtr outputTable_;
  std::vector<CudfVectorPtr> inputs_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::order> columnOrder_;
  std::vector<cudf::null_order> nullOrder_;
  bool finished_{false};

  // See planNodeBoundary().
  const core::PlanNode::Boundary planNodeBoundary_;

  // True when built for a MergeExchangeNode, whose inputs are sorted slices of
  // the upstream streams; see doNoMoreInput().
  const bool inputsAreSorted_;
};

} // namespace facebook::velox::cudf_velox
