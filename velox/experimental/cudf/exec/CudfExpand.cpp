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
#include "velox/experimental/cudf/exec/CudfProjectionCycle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {

// Null constants take their cuDF type from veloxToCudfDataType(), like input
// columns do, so the column type is the same in every projection. ARRAY and
// ROW are excluded because only scalar types are supported here.
bool isSupportedNullType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY:
    case TypeKind::ROW:
      return false;
    default:
      return tryVeloxToCudfDataType(type).has_value();
  }
}

// Mirrors the types makeScalarFromConstantExpr() can build for non-null
// values.
bool isSupportedConstantType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::TIMESTAMP:
      return true;
    case TypeKind::BIGINT:
      return !type->isIntervalDayTime();
    case TypeKind::INTEGER:
      return !type->isIntervalYearMonth();
    case TypeKind::HUGEINT:
      return type->isDecimal();
    default:
      return false;
  }
}

ScalarFactory makeConstantFactory(
    const core::ConstantTypedExprPtr& constant,
    memory::MemoryPool* pool) {
  return [constant, pool](cuda::stream_ref stream) {
    if (constant->isNull()) {
      return cudf::make_default_constructed_scalar(
          veloxToCudfDataType(constant->type()), stream, get_temp_mr());
    }
    return makeScalarFromConstantExpr(constant, pool, std::nullopt, stream);
  };
}

std::vector<std::vector<ProjectionColumn>> makeProjections(
    const core::ExpandNode& expandNode,
    memory::MemoryPool* pool) {
  const auto& inputType = expandNode.inputType();
  std::vector<std::vector<ProjectionColumn>> projections;
  projections.reserve(expandNode.projections().size());
  for (const auto& rowProjections : expandNode.projections()) {
    auto& columns = projections.emplace_back();
    columns.reserve(rowProjections.size());
    for (const auto& columnProjection : rowProjections) {
      if (auto field = core::TypedExprs::asFieldAccess(columnProjection)) {
        columns.push_back(
            ProjectionColumn::field(inputType->getChildIdx(field->name())));
      } else if (
          auto constant = core::TypedExprs::asConstant(columnProjection)) {
        columns.push_back(
            ProjectionColumn::constant(makeConstantFactory(constant, pool)));
      } else {
        // ExpandNode only accepts field accesses and constants.
        VELOX_UNREACHABLE(
            "Unexpected Expand projection: {}", columnProjection->toString());
      }
    }
  }
  return projections;
}

} // namespace

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
      projectionCycle_(
          makeProjections(*expandNode, pool()),
          expandNode->inputType()->size()) {}

bool CudfExpand::canRunOnGPU(
    const core::ExpandNode& expandNode,
    std::string* reason) {
  for (const auto& rowProjections : expandNode.projections()) {
    for (const auto& columnProjection : rowProjections) {
      const auto constant = core::TypedExprs::asConstant(columnProjection);
      if (constant == nullptr) {
        continue;
      }
      const auto supported = constant->isNull()
          ? isSupportedNullType(constant->type())
          : isSupportedConstantType(constant->type());
      if (!supported) {
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
  return !noMoreInput_ && !projectionCycle_.hasInput();
}

void CudfExpand::doAddInput(RowVectorPtr input) {
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfExpand expects CudfVector input");
  // Read before release(), which invalidates the vector.
  const auto numRows = static_cast<cudf::size_type>(cudfInput->size());
  const auto stream = cudfInput->stream();
  projectionCycle_.setInput(cudfInput->release(), numRows, stream);
}

RowVectorPtr CudfExpand::doGetOutput() {
  if (!projectionCycle_.hasInput()) {
    return nullptr;
  }
  const auto numRows = projectionCycle_.numRows();
  const auto stream = projectionCycle_.stream();
  auto outputTable = projectionCycle_.next(get_output_mr());
  return std::make_shared<CudfVector>(
      pool(), outputType_, numRows, std::move(outputTable), stream);
}

void CudfExpand::doClose() {
  Operator::close();
  projectionCycle_.clear();
}

} // namespace facebook::velox::cudf_velox
