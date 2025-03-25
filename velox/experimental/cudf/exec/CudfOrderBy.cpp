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

#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
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
      NvtxHelper(nvtx3::rgb{64, 224, 208}, operatorId), // Turquoise
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
        (sorting_order.isNullsFirst() ^ !sorting_order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
  if (cudfDebugEnabled()) {
    std::cout << "Number of Sort keys: " << sort_keys_.size() << std::endl;
  }
}

void CudfOrderBy::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudf_input);
    inputs_.push_back(std::move(cudf_input));
  }
}

void CudfOrderBy::noMoreInput() {
  exec::Operator::noMoreInput();
  // TODO: Get total row count, batch output
  // maxOutputRows_ = outputBatchRows(total_row_count);

  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (inputs_.empty()) {
    return;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto tbl = getConcatenatedTable(inputs_, stream);

  // Release input data after synchronizing
  stream.synchronize();
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
  auto result =
      cudf::sort_by_key(values, keys, column_order_, null_order_, stream);
  auto const size = result->num_rows();
  outputTable_ = std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(result), stream);
}

RowVectorPtr CudfOrderBy::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  finished_ = noMoreInput_;
  return outputTable_;
}

void CudfOrderBy::close() {
  exec::Operator::close();
  // Release stored inputs
  // Release cudf memory resources
  inputs_.clear();
  outputTable_.reset();
}
} // namespace facebook::velox::cudf_velox
