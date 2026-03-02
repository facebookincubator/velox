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
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {

// Extract the function name from a Presto-prefixed name like
// "presto.default.lag" → "lag".
std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

// Get the offset argument from a LAG/LEAD window function call.
// LAG(col) has offset=1 by default; LAG(col, 3) has offset=3.
cudf::size_type getLeadLagOffset(
    const core::WindowNode::Function& func) {
  const auto& args = func.functionCall->inputs();
  if (args.size() >= 2) {
    if (auto constExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                args[1])) {
      if (constExpr->hasValueVector()) {
        return constExpr->valueVector()
            ->as<SimpleVector<int64_t>>()
            ->valueAt(0);
      }
      return constExpr->value().value<int64_t>();
    }
  }
  return 1;
}

} // namespace

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : exec::Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "CudfWindow"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0}, // Orange
          operatorId,
          fmt::format("[{}]", windowNode->id())),
      windowNode_(windowNode) {
  const auto& inputType = windowNode->inputType();

  for (const auto& key : windowNode->partitionKeys()) {
    partitionKeyIndices_.push_back(
        inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(windowNode->sortingKeys()[i]->name()));
    const auto& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING
                            : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfWindow::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfWindow expects CudfVector input");
  inputBatches_.push_back(std::move(cudfInput));
}

void CudfWindow::noMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
  }
}

bool CudfWindow::isFinished() {
  return finished_;
}

RowVectorPtr CudfWindow::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (inputBatches_.empty()) {
    finished_ = true;
    return nullptr;
  }

  auto stream = inputBatches_[0]->stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto pool = inputBatches_[0]->pool();

  // 1. Concatenate all input batches into one table.
  std::vector<cudf::table_view> views;
  views.reserve(inputBatches_.size());
  for (const auto& batch : inputBatches_) {
    views.push_back(batch->getTableView());
  }
  std::unique_ptr<cudf::table> allData;
  if (views.size() == 1) {
    allData = std::make_unique<cudf::table>(views[0], stream, mr);
  } else {
    allData = cudf::concatenate(views, stream, mr);
  }
  inputBatches_.clear();

  auto allView = allData->view();
  const auto numInputCols = windowNode_->inputType()->size();
  const auto numRows = allView.num_rows();

  // 2. Sort by partition keys + sort keys if not already sorted.
  std::unique_ptr<cudf::table> sortedData;
  cudf::table_view sortedView;

  if (!windowNode_->inputsSorted()) {
    std::vector<cudf::size_type> allSortKeys;
    std::vector<cudf::order> allOrders;
    std::vector<cudf::null_order> allNullOrders;

    // Partition keys first (ascending), then sort keys.
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

    auto keyTable = allView.select(allSortKeys);
    auto indices = cudf::stable_sorted_order(
        keyTable, allOrders, allNullOrders, stream, mr);
    sortedData = cudf::detail::gather(
        allView,
        indices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::detail::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    sortedView = sortedData->view();
  } else {
    sortedView = allView;
  }

  // 3. Build partition key table for grouped_rolling_window.
  auto partKeys = sortedView.select(partitionKeyIndices_);

  // 4. Evaluate each window function and collect result columns.
  std::vector<std::unique_ptr<cudf::column>> windowResultCols;
  const auto& funcs = windowNode_->windowFunctions();

  for (const auto& func : funcs) {
    const auto baseName =
        getBaseFunctionName(func.functionCall->name());

    // Determine the input column for the window function.
    // For LAG/LEAD/FIRST_VALUE/LAST_VALUE, the first arg is the column.
    // For ROW_NUMBER/RANK/DENSE_RANK, there's no input column (use any).
    cudf::size_type inputColIdx = 0;
    if (!func.functionCall->inputs().empty()) {
      if (auto field =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  func.functionCall->inputs()[0])) {
        inputColIdx = windowNode_->inputType()->getChildIdx(field->name());
      }
    }
    auto inputCol = sortedView.column(inputColIdx);

    if (baseName == "lag") {
      auto offset = getLeadLagOffset(func);
      auto agg = cudf::make_lag_aggregation<cudf::rolling_aggregation>(offset);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, offset + 1, 0, offset + 1, *agg, stream, mr));
    } else if (baseName == "lead") {
      auto offset = getLeadLagOffset(func);
      auto agg =
          cudf::make_lead_aggregation<cudf::rolling_aggregation>(offset);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, 0, offset + 1, offset + 1, *agg, stream, mr));
    } else if (baseName == "first_value") {
      auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
          0, func.ignoreNulls ? cudf::null_policy::EXCLUDE
                              : cudf::null_policy::INCLUDE);
      auto unbounded = cudf::window_bounds::unbounded();
      auto current = cudf::window_bounds::get(1);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, unbounded, current, 1, *agg, stream, mr));
    } else if (baseName == "last_value") {
      auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
          -1, func.ignoreNulls ? cudf::null_policy::EXCLUDE
                               : cudf::null_policy::INCLUDE);
      auto unbounded = cudf::window_bounds::unbounded();
      auto current = cudf::window_bounds::get(1);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, current, unbounded, 1, *agg, stream, mr));
    } else if (baseName == "row_number") {
      auto agg =
          cudf::make_count_aggregation<cudf::rolling_aggregation>(
              cudf::null_policy::INCLUDE);
      auto unbounded = cudf::window_bounds::unbounded();
      auto current = cudf::window_bounds::get(1);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, unbounded, current, 1, *agg, stream, mr));
    } else if (
        baseName == "sum" || baseName == "min" || baseName == "max" ||
        baseName == "count" || baseName == "avg") {
      // Aggregate window functions with full partition frame.
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
      auto bounds = cudf::window_bounds::unbounded();
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, bounds, bounds, 1, *agg, stream, mr));
    } else {
      VELOX_FAIL("Unsupported window function for GPU: {}", baseName);
    }
  }

  // 5. Build the output table: input columns + window result columns.
  auto sortedCols = (sortedData ? sortedData : allData)->release();
  for (auto& wc : windowResultCols) {
    sortedCols.push_back(std::move(wc));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(sortedCols));
  auto resultSize = resultTable->num_rows();

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool, outputType_, resultSize, std::move(resultTable), stream);
}

} // namespace facebook::velox::cudf_velox
