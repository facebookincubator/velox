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
#include "velox/experimental/cudf/exec/CudfExpand.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

CudfExpand::CudfExpand(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          expandNode->outputType(),
          expandNode->id(),
          "CudfExpand",
          nvtx3::rgb{255, 165, 0}, // Orange
          NvtxMethodFlag::kGetOutput | NvtxMethodFlag::kClose),
      numInputColumns_(expandNode->inputType()->size()) {
  const auto& inputType = expandNode->inputType();
  const auto numProjections = expandNode->projections().size();
  const auto numColumns = expandNode->names().size();
  fieldProjections_.reserve(numProjections);
  constantProjections_.reserve(numProjections);
  for (const auto& rowProjections : expandNode->projections()) {
    std::vector<column_index_t> fieldProjection;
    fieldProjection.reserve(numColumns);
    std::vector<core::ConstantTypedExprPtr> constantProjection;
    constantProjection.reserve(numColumns);
    for (const auto& columnProjection : rowProjections) {
      if (auto field = core::TypedExprs::asFieldAccess(columnProjection)) {
        fieldProjection.push_back(inputType->getChildIdx(field->name()));
        constantProjection.push_back(nullptr);
      } else if (
          auto constant = core::TypedExprs::asConstant(columnProjection)) {
        fieldProjection.push_back(kConstantChannel);
        constantProjection.push_back(constant);
      } else {
        // ExpandNode only accepts field accesses and constants.
        VELOX_UNREACHABLE(
            "Unexpected Expand projection: {}", columnProjection->toString());
      }
    }
    fieldProjections_.emplace_back(std::move(fieldProjection));
    constantProjections_.emplace_back(std::move(constantProjection));
  }
}

namespace {

// Mirrors the types makeScalarFromConstantExpr() can build.
bool isSupportedConstantType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::TIMESTAMP:
      return true;
    case TypeKind::INTEGER:
      return !type->isIntervalYearMonth();
    case TypeKind::HUGEINT:
      return type->isDecimal();
    default:
      return false;
  }
}

} // namespace

bool CudfExpand::canRunOnGPU(
    const core::ExpandNode& expandNode,
    std::string* reason) {
  for (const auto& rowProjections : expandNode.projections()) {
    for (const auto& columnProjection : rowProjections) {
      if (core::TypedExprs::isConstant(columnProjection) &&
          !isSupportedConstantType(columnProjection->type())) {
        if (reason != nullptr) {
          *reason = fmt::format(
              "Expand constant type is not supported: {}",
              columnProjection->type()->toString());
        }
        return false;
      }
    }
  }
  return true;
}

bool CudfExpand::needsInput() const {
  return !noMoreInput_ && input_ == nullptr;
}

void CudfExpand::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfExpand::doGetOutput() {
  if (!input_) {
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfExpand expects CudfVector input");

  const auto numRows = cudfInput->size();
  // Read before release() below, which invalidates the vector.
  const auto stream = cudfInput->stream();
  const auto outputMr = get_output_mr();

  if (constantScalars_.empty()) {
    constantScalars_.reserve(constantProjections_.size());
    for (const auto& constantProjection : constantProjections_) {
      std::vector<std::unique_ptr<cudf::scalar>> scalars;
      scalars.reserve(constantProjection.size());
      for (const auto& constant : constantProjection) {
        scalars.push_back(
            constant ? makeScalarFromConstantExpr(
                           constant, pool(), std::nullopt, stream)
                     : nullptr);
      }
      constantScalars_.emplace_back(std::move(scalars));
    }
  }

  const auto& fieldProjection = fieldProjections_[projectionIndex_];
  const auto& scalars = constantScalars_[projectionIndex_];
  const auto numColumns = fieldProjection.size();

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(numColumns);

  if (projectionIndex_ == fieldProjections_.size() - 1) {
    // The input is not needed after the last projection, so its columns can
    // be moved into the output instead of copied. A column projected more
    // than once is copied for all but its last use.
    auto inputColumns = cudfInput->release()->release();
    VELOX_CHECK_EQ(inputColumns.size(), numInputColumns_);

    std::vector<int32_t> remainingUses(inputColumns.size(), 0);
    for (const auto channel : fieldProjection) {
      if (channel != kConstantChannel) {
        ++remainingUses[channel];
      }
    }

    for (size_t i = 0; i < numColumns; ++i) {
      const auto channel = fieldProjection[i];
      if (channel == kConstantChannel) {
        outputColumns.push_back(
            cudf::make_column_from_scalar(
                *scalars[i], numRows, stream, outputMr));
      } else if (--remainingUses[channel] == 0) {
        outputColumns.push_back(std::move(inputColumns[channel]));
      } else {
        outputColumns.push_back(
            std::make_unique<cudf::column>(
                *inputColumns[channel], stream, outputMr));
      }
    }
  } else {
    // TODO: Avoid deep copies by letting the output reference the input
    // columns, copying only when a downstream operator needs to own them.
    auto inputTableView = cudfInput->getTableView();
    for (size_t i = 0; i < numColumns; ++i) {
      const auto channel = fieldProjection[i];
      if (channel == kConstantChannel) {
        outputColumns.push_back(
            cudf::make_column_from_scalar(
                *scalars[i], numRows, stream, outputMr));
      } else {
        outputColumns.push_back(
            std::make_unique<cudf::column>(
                inputTableView.column(channel), stream, outputMr));
      }
    }
  }

  ++projectionIndex_;
  if (projectionIndex_ == fieldProjections_.size()) {
    projectionIndex_ = 0;
    input_ = nullptr;
  }

  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  return std::make_shared<CudfVector>(
      pool(), outputType_, numRows, std::move(outputTable), stream);
}

void CudfExpand::doClose() {
  Operator::close();
  constantScalars_.clear();
}

} // namespace facebook::velox::cudf_velox
