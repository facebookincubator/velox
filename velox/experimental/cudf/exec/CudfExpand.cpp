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
#include "velox/experimental/cudf/exec/CudfProjectionOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Memory.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {

// Returns true for constants that cuDF could build but that would give the
// column a different type than the same column from CudfFromVelox, or whose
// behavior is unverified. Building the scalar cannot detect these.
bool isExcludedConstant(const core::ConstantTypedExprPtr& constant) {
  if (constant->isNull()) {
    // Null constants take their cuDF type from veloxToCudfDataType(), like
    // input columns do. Only scalar types are supported here.
    return constant->type()->kind() == TypeKind::ARRAY ||
        constant->type()->kind() == TypeKind::ROW;
  }
  // A day-time interval builds as a duration scalar, while the input column
  // is INT64.
  return constant->type()->isIntervalDayTime();
}

std::unique_ptr<cudf::scalar> makeConstantScalar(
    const core::ConstantTypedExprPtr& constant,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  if (constant->isNull()) {
    return cudf::make_default_constructed_scalar(
        veloxToCudfDataType(constant->type()), stream, get_temp_mr());
  }
  return makeScalarFromConstantExpr(constant, pool, std::nullopt, stream);
}

ScalarFactory makeConstantFactory(const core::ConstantTypedExprPtr& constant) {
  return [constant](cuda::stream_ref stream, memory::MemoryPool* pool) {
    return makeConstantScalar(constant, pool, stream);
  };
}

std::vector<std::vector<ProjectionColumn>> makeProjections(
    const core::ExpandNode& expandNode) {
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
            ProjectionColumn::constant(makeConstantFactory(constant)));
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
    : CudfProjectionOperator(
          operatorId,
          driverCtx,
          expandNode->outputType(),
          expandNode->id(),
          "CudfExpand",
          nvtx3::rgb{255, 165, 0}, // Orange
          NvtxMethodFlag::kGetOutput | NvtxMethodFlag::kClose,
          std::nullopt,
          makeProjections(*expandNode),
          expandNode->inputType()->size()) {}

bool CudfExpand::canRunOnGPU(
    const core::ExpandNode& expandNode,
    std::string* reason) {
  // Building each constant is the check, so it cannot disagree with what the
  // operator later builds. This runs once when the plan is adapted.
  auto pool = memory::memoryManager()->addLeafPool("CudfExpand::canRunOnGPU");
  const auto stream = getDefaultStreamForCurrentThread();
  for (const auto& rowProjections : expandNode.projections()) {
    for (const auto& columnProjection : rowProjections) {
      const auto constant = core::TypedExprs::asConstant(columnProjection);
      if (constant == nullptr) {
        continue;
      }
      std::string error;
      if (isExcludedConstant(constant)) {
        error = "excluded";
      } else {
        try {
          makeConstantScalar(constant, pool.get(), stream);
          continue;
        } catch (const std::exception& e) {
          error = e.what();
        }
      }
      if (reason != nullptr) {
        *reason = fmt::format(
            "Expand constant is not supported: {} ({})",
            constant->type()->toString(),
            error);
      }
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::cudf_velox
