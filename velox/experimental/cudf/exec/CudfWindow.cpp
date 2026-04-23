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
#include "velox/exec/WindowFunction.h"
#include "velox/type/Type.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <nvtx3/nvtx3.hpp>

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

cudf::rank_method toRankMethod(const std::string& baseName) {
  if (baseName == "row_number") {
    return cudf::rank_method::FIRST;
  } else if (baseName == "rank") {
    return cudf::rank_method::MIN;
  } else if (baseName == "dense_rank") {
    return cudf::rank_method::DENSE;
  }
  VELOX_FAIL("toRankMethod called on non-rank function: {}", baseName);
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

// Materialize a column without unary cast. libcudf::cast only supports
// fixed-width targets; STRING/LIST/STRUCT identity "casts" fail at runtime.
std::unique_ptr<cudf::column> materializeColumnSameType(
    cudf::column_view col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (col.size() == 0) {
    return cudf::empty_like(col);
  }
  auto indices = cudf::sequence(
      col.size(),
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);
  auto gathered = cudf::gather(
      cudf::table_view{{col}},
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  return std::move(gathered->release()[0]);
}

// Normalize each incoming GPU batch to the WindowNode's logical row type so
// libcudf::concatenate is well-defined. Invariant: every batch matches the
// WindowNode input *Velox* row type. That does not guarantee identical libcudf
// physical types across batches (e.g. width/precision mismatches on fixed types
// or Arrow import quirks); concatenate requires matching
// cudf::data_type per column. When types already match, we only materialize
// (gather) for a stable copy; variable-width columns must match exactly.
std::unique_ptr<cudf::table> normalizeTableToInputRowType(
    cudf::table_view view,
    const RowTypePtr& rowType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(
      static_cast<size_t>(view.num_columns()),
      rowType->size(),
      "CudfWindow: GPU batch column count does not match input row type");
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    auto expected = cudf_velox::veloxToCudfDataType(rowType->childAt(i));
    auto col = view.column(i);
    // libcudf unary cast rejects STRING/LIST/STRUCT targets regardless of
    // cudf::is_fixed_width() on some versions — never route those through cast.
    const bool typesMatch = (col.type() == expected);
    const bool stringCompat = !typesMatch &&
        col.type().id() == cudf::type_id::STRING &&
        expected.id() == cudf::type_id::STRING;
    if (typesMatch || stringCompat) {
      cols.push_back(materializeColumnSameType(col, stream, mr));
    } else {
      VELOX_CHECK(
          expected.id() != cudf::type_id::STRING &&
              expected.id() != cudf::type_id::LIST &&
              expected.id() != cudf::type_id::STRUCT,
          "CudfWindow: variable-width column types must match exactly for concatenate");
      VELOX_CHECK(
          cudf::is_fixed_width(col.type()) && cudf::is_fixed_width(expected),
          "CudfWindow: cannot align non-fixed-width column types for concatenate");
      cols.push_back(cudf::cast(col, expected, stream, mr));
    }
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

// True when every column's libcudf type already matches Velox input row type
// (same rules as normalizeTableToInputRowType); concat can use the view as-is.
bool tableViewMatchesInputRowType(
    cudf::table_view view,
    const RowTypePtr& rowType) {
  VELOX_CHECK_EQ(
      static_cast<size_t>(view.num_columns()),
      rowType->size(),
      "CudfWindow: column count does not match input row type");
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    auto expected = cudf_velox::veloxToCudfDataType(rowType->childAt(i));
    auto col = view.column(i);
    const bool typesMatch = (col.type() == expected);
    const bool stringCompat = !typesMatch &&
        col.type().id() == cudf::type_id::STRING &&
        expected.id() == cudf::type_id::STRING;
    if (!typesMatch && !stringCompat) {
      return false;
    }
  }
  return true;
}

} // namespace

cudf::column_view CudfWindow::multiSortKeyStructView(
    cudf::table_view const& sortedInput) const {
  VELOX_CHECK_GE(
      sortKeyIndices_.size(),
      2,
      "multiSortKeyStructView requires >= 2 sort keys");
  sortKeyStructChildren_.clear();
  sortKeyStructChildren_.reserve(sortKeyIndices_.size());
  for (auto ch : sortKeyIndices_) {
    sortKeyStructChildren_.push_back(sortedInput.column(ch));
  }
  return cudf::column_view(
      cudf::data_type{cudf::type_id::STRUCT},
      sortedInput.num_rows(),
      nullptr,
      nullptr,
      0,
      0,
      sortKeyStructChildren_);
}

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = get_output_mr();
  auto method = toRankMethod(baseName);

  // Build the "values" column for rank tie detection.
  cudf::column_view valuesCol = [&]() -> cudf::column_view {
    if (sortKeyIndices_.empty()) {
      return sortedInput.column(0);
    }
    if (sortKeyIndices_.size() == 1 || baseName == "row_number") {
      return sortedInput.column(sortKeyIndices_[0]);
    }
    return multiSortKeyStructView(sortedInput);
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
    : exec::Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "CudfWindow"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0},
          operatorId,
          fmt::format("[{}]", windowNode->id())),
      windowNode_(windowNode),
      inputRowType_(asRowType(windowNode->inputType())) {
  const auto& inputType = windowNode->inputType();

  // Validate window function signatures upfront using Velox's registry.
  // This ensures we produce proper error messages for unsupported signatures.
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
  const auto numInputCols = inputType->size();
  for (size_t i = 0; i < windowNode->windowFunctions().size(); ++i) {
    const auto& func = windowNode->windowFunctions()[i];
    const auto baseName =
        stripFunctionPrefix(func.functionCall->name(), prefix);

    // Gather argument types for signature validation.
    std::vector<TypePtr> argTypes;
    for (const auto& arg : func.functionCall->inputs()) {
      argTypes.push_back(arg->type());
    }

    // Validate signature and get expected return type.
    auto expectedReturnType = exec::resolveWindowResultType(baseName, argTypes);

    // Validate return type matches what the plan node expects.
    auto actualReturnType = outputType_->childAt(numInputCols + i);
    VELOX_USER_CHECK(
        expectedReturnType->equivalent(*actualReturnType),
        "Unexpected return type for window function {}. Expected {}. Got {}.",
        exec::toString(baseName, argTypes),
        expectedReturnType->toString(),
        actualReturnType->toString());
  }

  for (const auto& key : windowNode->partitionKeys()) {
    partitionKeyIndices_.push_back(inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(windowNode->sortingKeys()[i]->name()));
    const auto& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
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
  auto mr = get_output_mr();
  velox::memory::MemoryPool* const outPool = pool();

  // Concatenate all input batches into one table. Skip per-batch normalize when
  // libcudf types already match the Window input row (avoids identity gather).
  std::vector<std::unique_ptr<cudf::table>> normalizedBatches;
  normalizedBatches.reserve(inputBatches_.size());
  std::vector<cudf::table_view> views;
  views.reserve(inputBatches_.size());
  for (const auto& batch : inputBatches_) {
    auto batchView = batch->getTableView();
    if (tableViewMatchesInputRowType(batchView, inputRowType_)) {
      views.push_back(batchView);
    } else {
      normalizedBatches.push_back(
          normalizeTableToInputRowType(batchView, inputRowType_, stream, mr));
      views.push_back(normalizedBatches.back()->view());
    }
  }
  std::unique_ptr<cudf::table> allData;
  if (views.size() == 1) {
    if (!normalizedBatches.empty()) {
      allData = std::move(normalizedBatches[0]);
    } else {
      // Own a copy: input CudfVector batches are released below.
      allData = cudf::concatenate(
          std::vector<cudf::table_view>{views[0]}, stream, mr);
    }
  } else {
    allData = cudf::concatenate(views, stream, mr);
  }
  // Drop normalized intermediates; release input batches after concat copied
  // data.
  normalizedBatches.clear();
  inputBatches_.clear();

  auto allView = allData->view();

  // 1. Sort by partition keys + sort keys if the plan is not already sorted.
  std::unique_ptr<cudf::table> sortedData;
  cudf::table_view sortedView;

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

    {
      auto keyTable = allView.select(allSortKeys);
      auto indices = cudf::stable_sorted_order(
          keyTable, allOrders, allNullOrders, stream, mr);
      sortedData = cudf::gather(
          allView,
          indices->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
    }
    sortedView = sortedData->view();
    // sortedData is a full gather copy; release the unsorted table before rank
    // scans to cut peak GPU memory (large windows e.g. TPC-DS Q49).
    allData.reset();
    stream.synchronize();
  } else {
    sortedView = allView;
  }

  // 2. Build partition key table for grouped_rolling_window.
  auto partKeys = sortedView.select(partitionKeyIndices_);

  // 3. Evaluate each window function and collect result columns.
  std::vector<std::unique_ptr<cudf::column>> windowResultCols;
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;

  for (const auto& func : windowNode_->windowFunctions()) {
    const auto baseName =
        stripFunctionPrefix(func.functionCall->name(), prefix);

    if (baseName == "row_number" || baseName == "rank" ||
        baseName == "dense_rank") {
      windowResultCols.push_back(
          computeRankColumn(sortedView, baseName, stream));
    } else if (baseName == "lag" || baseName == "lead") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeLeadLagColumn(partKeys, inputCol, func, baseName, stream));
    } else if (baseName == "first_value" || baseName == "last_value") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeNthValueColumn(partKeys, inputCol, func, baseName, stream));
    } else if (
        baseName == "sum" || baseName == "min" || baseName == "max" ||
        baseName == "count" || baseName == "avg") {
      auto inputColIdx = resolveInputColumn(func);
      auto inputCol = sortedView.column(inputColIdx);
      windowResultCols.push_back(
          computeAggregateColumn(partKeys, inputCol, func, baseName, stream));
    } else {
      VELOX_FAIL("Unsupported window function for cudf: {}", baseName);
    }
  }

  // 4. Build the output table: input columns + window result columns.
  // Cast window result columns to expected output types if needed.
  auto& dataOwner = sortedData ? sortedData : allData;
  auto sortedCols = dataOwner->release();
  const auto numInputCols = inputRowType_->size();
  for (size_t i = 0; i < windowResultCols.size(); ++i) {
    auto& wc = windowResultCols[i];
    auto expectedType =
        veloxToCudfDataType(outputType_->childAt(numInputCols + i));
    if (wc->type() != expectedType) {
      wc = cudf::cast(wc->view(), expectedType, stream, mr);
    }
    sortedCols.push_back(std::move(wc));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(sortedCols));
  auto resultSize = resultTable->num_rows();

  finished_ = true;
  return std::make_shared<CudfVector>(
      outPool, outputType_, resultSize, std::move(resultTable), stream);
}

} // namespace facebook::velox::cudf_velox
