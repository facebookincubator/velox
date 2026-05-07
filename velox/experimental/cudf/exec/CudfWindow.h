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
#include "velox/type/Type.h"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated Window operator using cuDF.
///
/// Incoming GPU batches are stored in addInput(). When noMoreInput() is called,
/// batches are concatenated and sorted. getOutput() then evaluates window
/// functions and returns one output batch.
///
/// inputsSorted fast path: when WindowNode::inputsSorted() is true, this
/// operator skips stable_sorted_order and the full-table gather (see
/// WindowNode::inputsSorted() for the ordering contract). The flag is taken
/// from the plan as-is; Velox does not infer it here. Connectors / optimizers
/// must only set it when a Sort or ordered exchange actually guarantees
/// partition keys then ORDER BY keys across concatenated input batches.
///
/// Memory: the sorted path peaks at roughly concat output plus gather copy plus
/// window result columns. Batch-wise / streaming evaluation would require a
/// larger redesign.
///
/// Rank-like functions (row_number, rank, dense_rank) use
/// cudf::groupby::scan with cudf::make_rank_aggregation.
/// Aggregate windows and lag/lead use cudf::grouped_rolling_window.
class CudfWindow : public CudfOperatorBase {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  /// Returns true if the window function is supported by CudfWindow.
  /// Supported functions:
  /// - Ranking: row_number, rank, dense_rank
  /// - Value: lag, lead (with up to 2 arguments), first_value, last_value
  /// - Aggregate: sum, min, max, count, avg
  /// Unsupported functions:
  /// - nth_value, ntile, cume_dist, percent_rank
  /// - lag/lead with default value (3rd argument)
  static bool isSupportedWindowFunction(
      const std::string& baseName,
      size_t numArgs) {
    static const std::unordered_set<std::string> kSupportedFuncs = {
        "lag",
        "lead",
        "row_number",
        "rank",
        "dense_rank",
        "first_value",
        "last_value",
        "sum",
        "min",
        "max",
        "count",
        "avg"};
    if (kSupportedFuncs.find(baseName) == kSupportedFuncs.end()) {
      return false;
    }
    // lag/lead only support up to 2 arguments (value, offset)
    if ((baseName == "lag" || baseName == "lead") && numArgs > 2) {
      return false;
    }
    return true;
  }

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
  // Resolve the input column index for a window function's first argument.
  cudf::size_type resolveInputColumn(
      const core::WindowNode::Function& func) const;

  // Compute row_number/rank/dense_rank via cudf::groupby::scan.
  std::unique_ptr<cudf::column> computeRankColumn(
      cudf::table_view const& sortedInput,
      const std::string& baseName,
      rmm::cuda_stream_view stream) const;

  // Compute LAG or LEAD via cudf::grouped_rolling_window.
  std::unique_ptr<cudf::column> computeLeadLagColumn(
      cudf::table_view const& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream) const;

  // Compute first_value or last_value via cudf::grouped_rolling_window.
  std::unique_ptr<cudf::column> computeNthValueColumn(
      cudf::table_view const& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream) const;

  // Compute aggregate window functions (sum, min, max, count, avg)
  // with frame bounds from the WindowNode.
  std::unique_ptr<cudf::column> computeAggregateColumn(
      cudf::table_view const& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream) const;

  std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputRowType_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;

  // Sorted and concatenated input data, prepared in doNoMoreInput().
  std::unique_ptr<cudf::table> sortedData_;
  rmm::cuda_stream_view stream_{};

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
