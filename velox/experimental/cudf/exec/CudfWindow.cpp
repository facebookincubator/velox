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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/core/Expressions.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <nvtx3/nvtx3.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

namespace {

cudf::size_type getLeadLagOffset(const core::WindowNode::Function& func) {
  const auto& args = func.functionCall->inputs();
  if (args.size() >= 2) {
    if (auto constExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(args[1])) {
      if (constExpr->hasValueVector()) {
        return constExpr->valueVector()->as<SimpleVector<int64_t>>()->valueAt(
            0);
      }
      return constExpr->value().value<int64_t>();
    }
  }
  return 1;
}

std::pair<cudf::window_bounds, cudf::window_bounds> toWindowBounds(
    const core::WindowNode::Frame& frame) {
  auto toBound = [](core::WindowNode::BoundType type,
                    const core::TypedExprPtr& value) -> cudf::window_bounds {
    switch (type) {
      case core::WindowNode::BoundType::kUnboundedPreceding:
      case core::WindowNode::BoundType::kUnboundedFollowing:
        return cudf::window_bounds::unbounded();
      case core::WindowNode::BoundType::kCurrentRow:
        return cudf::window_bounds::get(0);
      case core::WindowNode::BoundType::kPreceding:
      case core::WindowNode::BoundType::kFollowing: {
        if (value) {
          if (auto constExpr =
                  std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                      value)) {
            if (constExpr->hasValueVector()) {
              return cudf::window_bounds::get(constExpr->valueVector()
                                                  ->as<SimpleVector<int64_t>>()
                                                  ->valueAt(0));
            }
            return cudf::window_bounds::get(
                constExpr->value().value<int64_t>());
          }
        }
        return cudf::window_bounds::get(1);
      }
      default:
        return cudf::window_bounds::unbounded();
    }
  };
  return {
      toBound(frame.startType, frame.startValue),
      toBound(frame.endType, frame.endValue)};
}

} // namespace

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = get_output_mr();
  auto toRankMethod = [](const std::string& name) {
    if (name == "row_number") {
      return cudf::rank_method::FIRST;
    } else if (name == "rank") {
      return cudf::rank_method::MIN;
    }
    return cudf::rank_method::DENSE;
  };
  auto method = toRankMethod(baseName);

  // Build the "values" column for rank tie detection. For rank/dense_rank with
  // multiple sort keys, wrap them in a STRUCT for composite comparison.
  std::vector<cudf::column_view> structChildren;
  cudf::column_view valuesCol = [&]() -> cudf::column_view {
    if (sortKeyIndices_.empty()) {
      return sortedInput.column(0);
    }
    if (sortKeyIndices_.size() == 1 || baseName == "row_number") {
      return sortedInput.column(sortKeyIndices_[0]);
    }
    structChildren.reserve(sortKeyIndices_.size());
    for (auto idx : sortKeyIndices_) {
      structChildren.push_back(sortedInput.column(idx));
    }
    return cudf::column_view(
        cudf::data_type{cudf::type_id::STRUCT},
        sortedInput.num_rows(),
        nullptr,
        nullptr,
        0,
        0,
        structChildren);
  }();

  auto colOrder =
      sortKeyIndices_.empty() ? cudf::order::ASCENDING : sortOrders_[0];
  auto nullOrd =
      sortKeyIndices_.empty() ? cudf::null_order::BEFORE : nullOrders_[0];

  if (partitionKeyIndices_.empty()) {
    // libcudf's unpartitioned rank scan (cudf::scan) can disagree with SQL
    // engines on edge cases for global ordering. Use
    // the same groupby rank-scan path as non-empty partitions by ranking
    // within a synthetic single-key partition.
    auto const n = sortedInput.num_rows();
    auto zeroScalar = cudf::numeric_scalar<int32_t>(0, true, stream, mr);
    auto partCol = cudf::make_column_from_scalar(zeroScalar, n, stream, mr);
    cudf::groupby::groupby grouper(
        cudf::table_view({partCol->view()}),
        cudf::null_policy::INCLUDE,
        cudf::sorted::YES,
        std::vector<cudf::order>{cudf::order::ASCENDING},
        std::vector<cudf::null_order>{cudf::null_order::BEFORE});

    std::vector<cudf::groupby::scan_request> gbRequests(1);
    gbRequests[0].values = valuesCol;
    gbRequests[0].aggregations.push_back(
        cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            method, colOrder, cudf::null_policy::INCLUDE, nullOrd));
    auto scanResult = grouper.scan(gbRequests, stream, mr);
    auto& aggResults = scanResult.second;
    VELOX_CHECK_EQ(aggResults.size(), 1);
    VELOX_CHECK_EQ(aggResults[0].results.size(), 1);
    return std::move(aggResults[0].results[0]);
  }

  auto partCols = sortedInput.select(partitionKeyIndices_);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = valuesCol;
  requests[0].aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
          method, colOrder, cudf::null_policy::INCLUDE, nullOrd));

  cudf::groupby::groupby grouper(
      cudf::table_view(partCols),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyIndices_.size(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyIndices_.size(), cudf::null_order::BEFORE));

  auto scanResult = grouper.scan(requests, stream, mr);
  auto& aggResults = scanResult.second;
  VELOX_CHECK_EQ(aggResults.size(), 1);
  VELOX_CHECK_EQ(aggResults[0].results.size(), 1);
  return std::move(aggResults[0].results[0]);
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          windowNode->outputType(),
          windowNode->id(),
          "CudfWindow",
          nvtx3::rgb{255, 165, 0},
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput),
      windowNode_(windowNode),
      inputRowType_(asRowType(windowNode->inputType())) {
  const auto& inputType = windowNode->inputType();

  for (const auto& key : windowNode->partitionKeys()) {
    partitionKeyIndices_.push_back(inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(windowNode->sortingKeys()[i]->name()));
    const auto& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    // Velox isNullsFirst() is absolute; cuDF null_order is relative to sort
    // direction. BEFORE means nulls precede values in that direction.
    bool nullsBefore =
        (order.isNullsFirst() && order.isAscending()) ||
        (!order.isNullsFirst() && !order.isAscending());
    nullOrders_.push_back(
        nullsBefore ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
  }
}

void CudfWindow::doAddInput(RowVectorPtr input) {
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput, "CudfWindow expects CudfVector input");
    inputBatches_.push_back(std::move(cudfInput));
  }
}

cudf::size_type CudfWindow::resolveInputColumn(
    const core::WindowNode::Function& func) const {
  const auto& inputs = func.functionCall->inputs();
  // e.g. count(*) OVER (...) has no call arguments; use column 0 for row count.
  if (inputs.empty()) {
    return 0;
  }
  // Match exec::Window: resolve column via exprToChannel. Peel casts so we do
  // not default to column 0 (which broke nested/wrapped refs e.g. TPC-DS Q12
  // sum(sum(...)) / ratio over wrong column).
  const core::TypedExprPtr* arg = &inputs[0];
  while (auto cast =
             std::dynamic_pointer_cast<const core::CastTypedExpr>(*arg)) {
    VELOX_CHECK_EQ(cast->inputs().size(), 1u);
    arg = &cast->inputs()[0];
  }
  auto channel = exec::exprToChannel(arg->get(), windowNode_->inputType());
  VELOX_CHECK_NE(
      channel,
      kConstantChannel,
      "Constant window aggregate input not supported in CudfWindow");
  return static_cast<cudf::size_type>(channel);
}

std::unique_ptr<cudf::column> CudfWindow::computeLeadLagColumn(
    cudf::table_view const& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = get_output_mr();
  VELOX_CHECK_LE(
      func.functionCall->inputs().size(),
      2,
      "cudf {} does not support default value (3rd argument)",
      baseName);
  auto offset = getLeadLagOffset(func);

  if (baseName == "lag") {
    auto agg = cudf::make_lag_aggregation<cudf::rolling_aggregation>(offset);
    return cudf::grouped_rolling_window(
        partKeys, inputCol, offset + 1, 0, offset + 1, *agg, stream, mr);
  }
  auto agg = cudf::make_lead_aggregation<cudf::rolling_aggregation>(offset);
  return cudf::grouped_rolling_window(
      partKeys, inputCol, 0, offset + 1, offset + 1, *agg, stream, mr);
}

std::unique_ptr<cudf::column> CudfWindow::computeNthValueColumn(
    cudf::table_view const& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = get_output_mr();
  auto nullPolicy = func.ignoreNulls ? cudf::null_policy::EXCLUDE
                                     : cudf::null_policy::INCLUDE;
  if (baseName == "first_value") {
    auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
        0, nullPolicy);
    auto unbounded = cudf::window_bounds::unbounded();
    auto current = cudf::window_bounds::get(1);
    return cudf::grouped_rolling_window(
        partKeys, inputCol, unbounded, current, 1, *agg, stream, mr);
  }
  // last_value
  auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
      -1, nullPolicy);
  auto unbounded = cudf::window_bounds::unbounded();
  auto current = cudf::window_bounds::get(1);
  return cudf::grouped_rolling_window(
      partKeys, inputCol, current, unbounded, 1, *agg, stream, mr);
}

std::unique_ptr<cudf::column> CudfWindow::computeAggregateColumn(
    cudf::table_view const& partKeys,
    cudf::column_view inputCol,
    const core::WindowNode::Function& func,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = get_output_mr();
  std::unique_ptr<cudf::rolling_aggregation> agg;
  if (baseName == "sum") {
    agg = cudf::make_sum_aggregation<cudf::rolling_aggregation>();
  } else if (baseName == "min") {
    agg = cudf::make_min_aggregation<cudf::rolling_aggregation>();
  } else if (baseName == "max") {
    agg = cudf::make_max_aggregation<cudf::rolling_aggregation>();
  } else if (baseName == "count") {
    agg = cudf::make_count_aggregation<cudf::rolling_aggregation>(
        cudf::null_policy::EXCLUDE);
  } else {
    agg = cudf::make_mean_aggregation<cudf::rolling_aggregation>();
  }

  // For full-partition aggregation (UNBOUNDED...UNBOUNDED or
  // UNBOUNDED...CURRENT ROW with no sort keys), force unbounded on both
  // sides so grouped_rolling_window computes over the entire partition.
  bool isUnboundedPreceding =
      func.frame.startType == core::WindowNode::BoundType::kUnboundedPreceding;
  bool isUnboundedFollowing =
      func.frame.endType == core::WindowNode::BoundType::kUnboundedFollowing;
  bool isCurrentRowFollowing =
      func.frame.endType == core::WindowNode::BoundType::kCurrentRow;
  bool isFullPartition = isUnboundedPreceding &&
      (isUnboundedFollowing ||
       (isCurrentRowFollowing && sortKeyIndices_.empty()));

  if (isFullPartition) {
    return cudf::grouped_rolling_window(
        partKeys,
        inputCol,
        cudf::window_bounds::unbounded(),
        cudf::window_bounds::unbounded(),
        1,
        *agg,
        stream,
        mr);
  }

  auto [preceding, following] = toWindowBounds(func.frame);
  return cudf::grouped_rolling_window(
      partKeys, inputCol, preceding, following, 1, *agg, stream, mr);
}

void CudfWindow::doNoMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
    return;
  }

  // Verify total row count doesn't exceed cudf's int32 limit.
  int64_t totalRows = 0;
  for (const auto& batch : inputBatches_) {
    totalRows += batch->size();
  }
  VELOX_CHECK_LE(
      totalRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Total row count {} exceeds cudf int32 limit",
      totalRows);

  stream_ = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  // Concatenate all input batches into one table with proper stream sync.
  auto allData = getConcatenatedTable(
      std::exchange(inputBatches_, {}), inputRowType_, stream_, mr);

  // Sort by partition keys + sort keys if the plan is not already sorted.
  if (!windowNode_->inputsSorted()) {
    std::vector<cudf::size_type> allSortKeys;
    std::vector<cudf::order> allOrders;
    std::vector<cudf::null_order> allNullOrders;

    for (auto idx : partitionKeyIndices_) {
      allSortKeys.push_back(idx);
      allOrders.push_back(cudf::order::ASCENDING);
      allNullOrders.push_back(cudf::null_order::BEFORE);
    }
    for (size_t i = 0; i < sortKeyIndices_.size(); ++i) {
      allSortKeys.push_back(sortKeyIndices_[i]);
      allOrders.push_back(sortOrders_[i]);
      allNullOrders.push_back(nullOrders_[i]);
    }

    auto allView = allData->view();
    auto keyTable = allView.select(allSortKeys);
    sortedData_ = cudf::stable_sort_by_key(
        allView, keyTable, allOrders, allNullOrders, stream_, mr);
  } else {
    sortedData_ = std::move(allData);
  }
}

bool CudfWindow::isFinished() {
  return finished_;
}

RowVectorPtr CudfWindow::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (!sortedData_) {
    finished_ = true;
    return nullptr;
  }

  auto mr = get_output_mr();
  auto sortedView = sortedData_->view();

  // Build partition key table for grouped_rolling_window.
  auto partKeys = sortedView.select(partitionKeyIndices_);

  // Evaluate each window function and collect result columns.
  std::vector<std::unique_ptr<cudf::column>> windowResultCols;
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;

  for (const auto& func : windowNode_->windowFunctions()) {
    const auto baseName =
        stripFunctionPrefix(func.functionCall->name(), prefix);

    if (baseName == "row_number" || baseName == "rank" ||
        baseName == "dense_rank") {
      windowResultCols.push_back(
          computeRankColumn(sortedView, baseName, stream_));
    } else if (baseName == "lag" || baseName == "lead") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeLeadLagColumn(partKeys, inputCol, func, baseName, stream_));
    } else if (baseName == "first_value" || baseName == "last_value") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeNthValueColumn(partKeys, inputCol, func, baseName, stream_));
    } else if (
        baseName == "sum" || baseName == "min" || baseName == "max" ||
        baseName == "count" || baseName == "avg") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeAggregateColumn(partKeys, inputCol, func, baseName, stream_));
    } else {
      VELOX_FAIL("Unsupported window function for cudf: {}", baseName);
    }
  }

  // Build the output table: input columns + window result columns.
  // Cast window result columns to expected output types if needed.
  auto sortedCols = sortedData_->release();
  sortedData_.reset();
  const auto numInputCols = inputRowType_->size();
  for (size_t i = 0; i < windowResultCols.size(); ++i) {
    auto& wc = windowResultCols[i];
    auto expectedType =
        veloxToCudfDataType(outputType_->childAt(numInputCols + i));
    if (wc->type() != expectedType) {
      wc = cudf::cast(wc->view(), expectedType, stream_, mr);
    }
    sortedCols.push_back(std::move(wc));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(sortedCols));
  auto resultSize = resultTable->num_rows();

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool(), outputType_, resultSize, std::move(resultTable), stream_);
}

} // namespace facebook::velox::cudf_velox
