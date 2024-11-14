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
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

namespace facebook::velox::cudf_velox {

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderByNode>& orderByNode)
    : exec::Operator(
          driverCtx,
          orderByNode->outputType(),
          operatorId,
          orderByNode->id(),
          "CudfOrderBy"),
      orderByNode_(orderByNode) {
  maxOutputRows_ = outputBatchRows(std::nullopt);
  sort_keys_.reserve(orderByNode->sortingKeys().size());
  column_order_.reserve(orderByNode->sortingKeys().size());
  null_order_.reserve(orderByNode->sortingKeys().size());
  for (int i = 0; i < orderByNode->sortingKeys().size(); ++i) {
    const auto channel =
        exec::exprToChannel(orderByNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "OrderBy doesn't allow constant sorting keys");
    sort_keys_.push_back(channel);
    auto const& sorting_order = orderByNode->sortingOrders()[i];
    column_order_.push_back(
        sorting_order.isAscending() ? cudf::order::ASCENDING
                                    : cudf::order::DESCENDING);
    null_order_.push_back(
        sorting_order.isNullsFirst() ? cudf::null_order::BEFORE
                                     : cudf::null_order::AFTER);
  }
  if (cudfDebugEnabled()) {
    std::cout << "Number of Sort keys: " << sort_keys_.size() << std::endl;
  }
}

void CudfOrderBy::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    inputs_.push_back(std::move(input));
  }
}

void CudfOrderBy::noMoreInput() {
  exec::Operator::noMoreInput();
  // TODO: Get total row count, batch output
  // maxOutputRows_ = outputBatchRows(total_row_count);

  NVTX3_FUNC_RANGE();

  auto cudf_tables = std::vector<std::unique_ptr<cudf::table>>(inputs_.size());
  auto cudf_table_views = std::vector<cudf::table_view>(inputs_.size());
  for (int i = 0; i < inputs_.size(); i++) {
    VELOX_CHECK_NOT_NULL(inputs_[i]);
    cudf_tables[i] = with_arrow::to_cudf_table(inputs_[i], inputs_[i]->pool());
    cudf_table_views[i] = cudf_tables[i]->view();
  }
  auto tbl = cudf::concatenate(cudf_table_views);

  // Release input data
  cudf::get_default_stream().synchronize();
  cudf_table_views.clear();
  cudf_tables.clear();
  inputs_.clear();
  VELOX_CHECK_NOT_NULL(tbl);
  if (cudfDebugEnabled()) {
    std::cout << "Sort input table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "Sort input table number of rows: " << tbl->num_rows()
              << std::endl;
  }

  auto keys = tbl->view().select(sort_keys_);
  auto values = tbl->view();
  sortedTable_ = cudf::sort_by_key(values, keys, column_order_, null_order_);
}

RowVectorPtr CudfOrderBy::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  cudf::get_default_stream().synchronize();
  // TODO : batching later
  // RowVectorPtr output = sortBuffer_->getOutput(maxOutputRows_);
  RowVectorPtr output =
      with_arrow::to_velox_column(sortedTable_->view(), pool(), "");
  finished_ = noMoreInput_; //(output == nullptr);
  sortedTable_.reset();
  return output;
}

void CudfOrderBy::close() {
  exec::Operator::close();
  // TODO: Release stored inputs if needed
  // TODO: Release cudf memory resources
  sortedTable_.reset();
}
} // namespace facebook::velox::cudf_velox
