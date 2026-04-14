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


#include "velox/experimental/cudf/exec/CudfExpand.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/expression/Expr.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

CudfExpand::CudfExpand(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode)
    : Operator(
          driverCtx,
          expandNode->outputType(),
          operatorId,
          expandNode->id(),
          "CudfExpand"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0}, // Orange
          operatorId,
          fmt::format("[{}]", expandNode->id())) {
  const auto& inputType = expandNode->inputType();
  const auto numRows = expandNode->projections().size();
  fieldProjections_.reserve(numRows);
  constantProjections_.reserve(numRows);
  constantOutputs_.reserve(numRows);
  const auto numColumns = expandNode->names().size();
  for (const auto& rowProjections : expandNode->projections()) {
    std::vector<column_index_t> rowProjection;
    rowProjection.reserve(numColumns);
    std::vector<std::shared_ptr<const core::ConstantTypedExpr>>
        constantProjection;
    constantProjection.reserve(numColumns);
    for (const auto& columnProjection : rowProjections) {
      if (auto field = core::TypedExprs::asFieldAccess(columnProjection)) {
        rowProjection.push_back(inputType->getChildIdx(field->name()));
        constantProjection.push_back(nullptr);
      } else if (
          auto constant = core::TypedExprs::asConstant(columnProjection)) {
        rowProjection.push_back(kConstantChannel);
        constantProjection.push_back(constant);
      } else {
        VELOX_USER_FAIL(
            "Expand operator doesn't support this expression. Only column references and constants are supported. {}",
            columnProjection->toString());
      }
    }

    fieldProjections_.emplace_back(std::move(rowProjection));
    constantProjections_.emplace_back(std::move(constantProjection));
  }
}

void CudfExpand::initialize() {
  if (constantProjections_.empty()) {
    return;
  }
  const auto numColumns = constantProjections_[0].size();
  for (const auto& projections : constantProjections_) {
    std::vector<std::unique_ptr<cudf::scalar>> constantOutput;
    constantOutput.reserve(numColumns);
    for (const auto& constant : projections) {
      if (constant) {
        VELOX_CHECK(!constant->hasValueVector());
        constantOutput.push_back(makeScalarFromVariant(constant->type(), constant->value()));
      } else {
        constantOutput.push_back(nullptr);
      }
    }
    constantOutputs_.emplace_back(std::move(constantOutput));
  }
}

bool CudfExpand::needsInput() const {
  return !noMoreInput_ && input_ == nullptr;
}

void CudfExpand::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfExpand::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (!input_) {
    return nullptr;
  }

  const auto numInput = input_->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(
      cudfInput, "CudfExpand expects CudfVector input, got regular RowVector");

  const auto& rowProjection = fieldProjections_[rowIndex_];
  const auto& constantProjection = constantOutputs_[rowIndex_];
  const auto numColumns = rowProjection.size();

  auto stream = cudfInput->stream();

  // Check if this is the last projection
  const bool isLastProjection = (rowIndex_ == fieldProjections_.size() - 1);

  // Build output columns
  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(numColumns);

  if (isLastProjection) {
    // Last projection: move columns from input table when possible
    auto inputTable = cudfInput->release();
    auto inputColumns = inputTable->release();
    
    // Count how many times each input column is used in this projection
    std::vector<int> columnUseCount(inputColumns.size(), 0);
    for (auto i = 0; i < numColumns; ++i) {
      if (rowProjection[i] != kConstantChannel) {
        columnUseCount[rowProjection[i]]++;
      }
    }
    
    // Track remaining uses for each column
    std::vector<int> columnRemainingUses = columnUseCount;
    
    for (auto i = 0; i < numColumns; ++i) {
      if (rowProjection[i] == kConstantChannel) {
        const auto& scalar = constantProjection[i];
        outputColumns.push_back(
            cudf::make_column_from_scalar(*scalar, numInput, stream, get_output_mr()));
      } else {
        auto colIdx = rowProjection[i];
        columnRemainingUses[colIdx]--;
        
        if (columnRemainingUses[colIdx] == 0) {
          // Last use of this column, can move
          outputColumns.push_back(std::move(inputColumns[colIdx]));
        } else {
          // Not the last use, must copy
          outputColumns.push_back(
              std::make_unique<cudf::column>(*inputColumns[colIdx], stream, get_output_mr()));
        }
      }
    }
  } else {
    // Not last projection: copy columns from table view
    auto inputTableView = cudfInput->getTableView();
    
    for (auto i = 0; i < numColumns; ++i) {
      if (rowProjection[i] == kConstantChannel) {
        const auto& scalar = constantProjection[i];
        outputColumns.push_back(
            cudf::make_column_from_scalar(*scalar, numInput, stream, get_output_mr()));
      } else {
        auto inputColumn = inputTableView.column(rowProjection[i]);
        outputColumns.push_back(
            std::make_unique<cudf::column>(inputColumn, stream, get_output_mr()));
      }
    }
  }

  ++rowIndex_;
  if (rowIndex_ == fieldProjections_.size()) {
    rowIndex_ = 0;
    input_ = nullptr;
  }

  // Create output table and wrap in CudfVector
  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numInput, std::move(outputTable), stream);
}

} // namespace facebook::velox::cudf_velox

