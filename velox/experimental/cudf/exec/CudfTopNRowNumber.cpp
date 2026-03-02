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
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : exec::Operator(
          driverCtx,
          node->outputType(),
          operatorId,
          node->id(),
          "CudfTopNRowNumber"),
      NvtxHelper(
          nvtx3::rgb{255, 200, 100}, // Light orange
          operatorId,
          fmt::format("[{}]", node->id())),
      node_(node),
      limit_(node->limit()),
      generateRowNumber_(node->generateRowNumber()) {
  const auto& inputType = node->sources()[0]->outputType();

  for (const auto& key : node->partitionKeys()) {
    partitionKeyIndices_.push_back(inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < node->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(node->sortingKeys()[i]->name()));
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

void CudfTopNRowNumber::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfTopNRowNumber expects CudfVector");
  inputBatches_.push_back(std::move(cudfInput));
}

void CudfTopNRowNumber::noMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
  }
}

bool CudfTopNRowNumber::isFinished() {
  return finished_;
}

RowVectorPtr CudfTopNRowNumber::getOutput() {
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
  const auto& inputType = node_->sources()[0]->outputType();

  // 1. Concatenate all input batches.
  std::vector<cudf::table_view> views;
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

  // 2. Sort by partition keys + sort keys.
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

  auto keyTable = allView.select(allSortKeys);
  auto indices = cudf::stable_sorted_order(
      keyTable, allOrders, allNullOrders, stream, mr);
  auto sortedData = cudf::detail::gather(
      allView,
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::detail::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  auto sortedView = sortedData->view();

  // 3. Compute per-partition row numbers using grouped rolling count.
  auto partKeys = sortedView.select(partitionKeyIndices_);
  auto firstCol = sortedView.column(0);
  auto countAgg = cudf::make_count_aggregation<cudf::rolling_aggregation>(
      cudf::null_policy::INCLUDE);
  auto unbounded = cudf::window_bounds::unbounded();
  auto current = cudf::window_bounds::get(1);
  auto rowNums = cudf::grouped_rolling_window(
      partKeys, firstCol, unbounded, current, 1, *countAgg, stream, mr);

  // 4. Filter rows where row_number <= limit.
  auto limitScalar = cudf::numeric_scalar<int64_t>(limit_, true, stream, mr);
  auto mask = cudf::binary_operation(
      rowNums->view(),
      limitScalar,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto filteredTable = cudf::apply_boolean_mask(
      sortedView, mask->view(), stream, mr);
  auto resultSize = filteredTable->num_rows();

  // 5. If generateRowNumber is true, append row number column.
  if (generateRowNumber_) {
    auto filteredView = filteredTable->view();
    auto filtPartKeys = filteredView.select(partitionKeyIndices_);
    auto filtFirstCol = filteredView.column(0);
    auto filtRowNums = cudf::grouped_rolling_window(
        filtPartKeys, filtFirstCol, unbounded, current, 1, *countAgg,
        stream, mr);

    auto cols = filteredTable->release();
    cols.push_back(std::move(filtRowNums));
    filteredTable = std::make_unique<cudf::table>(std::move(cols));
  }

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool, outputType_, resultSize, std::move(filteredTable), stream);
}

} // namespace facebook::velox::cudf_velox
