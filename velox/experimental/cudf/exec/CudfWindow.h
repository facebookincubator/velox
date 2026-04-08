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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated Window operator using cuDF.
///
/// Each incoming batch is immediately concatenated into an accumulated cudf
/// table on the GPU in addInput(). Once all input has arrived, getOutput()
/// sorts (if needed), evaluates the window functions, and returns the result.
///
/// Rank-like functions (row_number, rank, dense_rank) use
/// cudf::groupby::scan with cudf::make_rank_aggregation.
/// Aggregate windows and lag/lead use cudf::grouped_rolling_window.
class CudfWindow : public exec::Operator, public NvtxHelper {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  // Resolve the input column index for a window function's first argument.
  cudf::size_type resolveInputColumn(
      const core::WindowNode::Function& func) const;

  // Compute row_number/rank/dense_rank via cudf::groupby::scan.
  // When rankTieKeyAsDouble, single-key rank/dense_rank uses FLOAT64 for
  // libcudf tie resolution (small-window DECIMAL path; see getOutput()).
  std::unique_ptr<cudf::column> computeRankColumn(
      cudf::table_view const& sortedInput,
      const std::string& baseName,
      rmm::cuda_stream_view stream,
      bool rankTieKeyAsDouble) const;

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

  // Build a zero-copy STRUCT column_view over multiple sort key columns
  // for composite-key tie detection in rank functions.
  cudf::column_view multiSortKeyStructView(
      cudf::table_view const& sortedInput) const;

  std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputRowType_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<std::shared_ptr<CudfVector>> inputBatches_;

  // Scratch storage for multiSortKeyStructView children.
  mutable std::vector<cudf::column_view> sortKeyStructChildren_;

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
