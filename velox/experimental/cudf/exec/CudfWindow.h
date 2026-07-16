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

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

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
/// globally sorted input across concatenated batches with partition keys
/// ASCENDING / nulls-first, followed by the window ORDER BY keys (matching
/// sortOrders_/nullOrders_). Rank grouping with partition keys also assumes
/// that partition-key ordering when constructing the groupby grouper.
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

  /// Returns true if every window function and frame in the plan node is
  /// supported by CudfWindow. On failure, @p reason is populated with a
  /// human-readable explanation when a non-null pointer is provided.
  static bool canRunOnGPU(const core::WindowNode& windowNode);

  static bool canRunOnGPU(
      const core::WindowNode& windowNode,
      std::string* reason);

  /// Returns true if the window function is supported by CudfWindow.
  static bool isSupportedWindowFunction(
      const std::string& baseName,
      size_t numArgs);

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

  void doClose() override;

 private:
  // Compute row_number/rank/dense_rank via cudf::groupby::scan or cudf::scan.
  void computeRankColumnsBatch(
      const cudf::table_view& sortedInput,
      const std::vector<std::pair<size_t, std::string>>& pendingRanks,
      cudf::groupby::groupby* rankGrouper,
      std::vector<std::unique_ptr<cudf::column>>& windowResultCols,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeLeadLagColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute first_value or last_value via cudf rolling window APIs.
  std::unique_ptr<cudf::column> computeNthValueColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      bool isFullPartition,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute aggregate window functions (sum, min, max, count, avg)
  // with frame bounds from the WindowNode.
  std::unique_ptr<cudf::column> computeAggregateColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      bool isCountStar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Dispatch ROWS window frames to grouped_rolling_window. RANGE frames are
  // handled separately by the batched grouped_range_rolling_window path.
  std::unique_ptr<cudf::column> invokeGroupedRollingWindow(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      std::unique_ptr<cudf::rolling_aggregation> agg,
      bool isFullPartition,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputRowType_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;

  // Sorted and concatenated input data, prepared in doNoMoreInput().
  std::unique_ptr<cudf::table> sortedData_;
  cudf::size_type logicalRowCount_{0};
  rmm::cuda_stream_view stream_{};
  bool streamAcquired_{false};

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
