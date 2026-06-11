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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

namespace facebook::velox::cudf_velox {

namespace {

cudf::table_view makePartitionKeys(
    cudf::table_view sortedView,
    const std::vector<cudf::size_type>& partitionKeyIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::unique_ptr<cudf::column>& singlePartitionCol) {
  if (!partitionKeyIndices.empty()) {
    return sortedView.select(partitionKeyIndices);
  }
  auto zero = cudf::numeric_scalar<int8_t>(0, true, stream, mr);
  singlePartitionCol =
      cudf::make_column_from_scalar(zero, sortedView.num_rows(), stream, mr);
  return cudf::table_view{{singlePartitionCol->view()}};
}

} // namespace

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          node->outputType(),
          node->id(),
          "CudfTopNRowNumber",
          nvtx3::rgb{255, 200, 100},
          NvtxMethodFlag::kAll,
          std::nullopt,
          node),
      node_(node),
      limit_(node->limit()),
      generateRowNumber_(node->generateRowNumber()) {
  VELOX_CHECK_EQ(
      node->rankFunction(),
      core::TopNRowNumberNode::RankFunction::kRowNumber,
      "CudfTopNRowNumber only supports row_number");

  const auto inputType = node->sources()[0]->outputType();
  partitionKeyIndices_.reserve(node->partitionKeys().size());
  for (const auto& key : node->partitionKeys()) {
    partitionKeyIndices_.push_back(
        exec::exprToChannel(key.get(), inputType));
  }

  sortKeyIndices_.reserve(node->sortingKeys().size());
  sortOrders_.reserve(node->sortingKeys().size());
  nullOrders_.reserve(node->sortingKeys().size());
  for (size_t i = 0; i < node->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        exec::exprToChannel(node->sortingKeys()[i].get(), inputType));
    const auto& order = node->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING
                            : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfTopNRowNumber::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfTopNRowNumber expects CudfVector");
  inputBatches_.push_back(std::move(cudfInput));
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
  }
}

bool CudfTopNRowNumber::isFinished() {
  return finished_;
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (inputBatches_.empty()) {
    finished_ = true;
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  auto pool = inputBatches_[0]->pool();
  const auto inputType = node_->sources()[0]->outputType();

  auto allData = getConcatenatedTable(
      std::exchange(inputBatches_, {}), inputType, stream, mr);
  auto allView = allData->view();

  std::vector<cudf::size_type> allSortKeys;
  std::vector<cudf::order> allOrders;
  std::vector<cudf::null_order> allNullOrders;
  allSortKeys.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());
  allOrders.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());
  allNullOrders.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());

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
  auto sortedData = cudf::gather(
      allView,
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  auto sortedView = sortedData->view();

  std::unique_ptr<cudf::column> singlePartitionCol;
  auto partKeys = makePartitionKeys(
      sortedView, partitionKeyIndices_, stream, mr, singlePartitionCol);
  auto firstCol = sortedView.column(0);
  auto rowNumberAgg =
      cudf::make_row_number_aggregation<cudf::rolling_aggregation>();
  auto unbounded = cudf::window_bounds::unbounded();
  auto currentRow = cudf::window_bounds::get(0);
  auto rowNums = cudf::grouped_rolling_window(
      partKeys, firstCol, unbounded, currentRow, 1, *rowNumberAgg, stream, mr);

  auto limitScalar =
      cudf::numeric_scalar<int64_t>(limit_, true, stream, mr);
  auto mask = cudf::binary_operation(
      rowNums->view(),
      limitScalar,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto filteredTable =
      cudf::apply_boolean_mask(sortedView, mask->view(), stream, mr);

  if (generateRowNumber_) {
    auto filteredView = filteredTable->view();
    std::unique_ptr<cudf::column> filteredSinglePartitionCol;
    auto filtPartKeys = makePartitionKeys(
        filteredView,
        partitionKeyIndices_,
        stream,
        mr,
        filteredSinglePartitionCol);
    auto filtFirstCol = filteredView.column(0);
    auto filtRowNums = cudf::grouped_rolling_window(
        filtPartKeys,
        filtFirstCol,
        unbounded,
        currentRow,
        1,
        *rowNumberAgg,
        stream,
        mr);

    auto cols = filteredTable->release();
    cols.push_back(std::move(filtRowNums));
    filteredTable = std::make_unique<cudf::table>(std::move(cols));
  }

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool,
      outputType_,
      filteredTable->num_rows(),
      std::move(filteredTable),
      stream);
}

} // namespace facebook::velox::cudf_velox
