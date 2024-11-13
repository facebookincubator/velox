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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {
CompareFlags fromSortOrderToCompareFlags(const core::SortOrder& sortOrder) {
  return {
      sortOrder.isNullsFirst(),
      sortOrder.isAscending(),
      false,
      CompareFlags::NullHandlingMode::kNullAsValue};
}
} // namespace

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderByNode>& orderByNode)
    : exec::Operator(
          driverCtx,
          orderByNode->outputType(),
          operatorId,
          orderByNode->id(),
          "CudfOrderBy",
          orderByNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt) {
  maxOutputRows_ = outputBatchRows(std::nullopt);
  VELOX_CHECK(pool()->trackUsage());
  std::vector<column_index_t> sortColumnIndices;
  std::vector<CompareFlags> sortCompareFlags;
  sortColumnIndices.reserve(orderByNode->sortingKeys().size());
  sortCompareFlags.reserve(orderByNode->sortingKeys().size());
  for (int i = 0; i < orderByNode->sortingKeys().size(); ++i) {
    const auto channel =
        exec::exprToChannel(orderByNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "OrderBy doesn't allow constant sorting keys");
    sortColumnIndices.push_back(channel);
    sortCompareFlags.push_back(
        fromSortOrderToCompareFlags(orderByNode->sortingOrders()[i]));
  }
}

void CudfOrderBy::addInput(RowVectorPtr input) {
  // TODO: Accumulate inputs
}

void CudfOrderBy::noMoreInput() {
  exec::Operator::noMoreInput();
  // TODO: Get total row count
  auto total_row_count = 0;
  maxOutputRows_ = outputBatchRows(total_row_count);
}

RowVectorPtr CudfOrderBy::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  RowVectorPtr output;
  // output = sortBuffer_->getOutput(maxOutputRows_);
  finished_ = (output == nullptr);
  return output;
}

void CudfOrderBy::close() {
  exec::Operator::close();
  // TODO: Release stored inputs if needed
  // TODO: Release cudf memory resources
}
} // namespace facebook::velox::exec
