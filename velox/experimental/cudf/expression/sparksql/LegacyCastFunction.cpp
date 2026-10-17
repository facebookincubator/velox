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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/sparksql/LegacyCastFunction.h"
#include "velox/experimental/cudf/expression/sparksql/LegacyCastKernel.h"

#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

bool isPlainNumeric(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return type->equivalent(*BOOLEAN());
    case TypeKind::TINYINT:
      return type->equivalent(*TINYINT());
    case TypeKind::SMALLINT:
      return type->equivalent(*SMALLINT());
    case TypeKind::INTEGER:
      return type->equivalent(*INTEGER());
    case TypeKind::BIGINT:
      return type->equivalent(*BIGINT());
    case TypeKind::REAL:
      return type->equivalent(*REAL());
    case TypeKind::DOUBLE:
      return type->equivalent(*DOUBLE());
    default:
      return false;
  }
}

bool isFloatingPoint(const TypePtr& type) {
  return type->isReal() || type->isDouble();
}

} // namespace

bool LegacyCastFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 1 || expr->inputs()[0]->isConstantKind()) {
    return false;
  }
  const auto& source = expr->inputs()[0]->type();
  const auto& target = expr->type();
  if ((!isPlainNumeric(source) && !source->isDecimal()) ||
      (!isPlainNumeric(target) && !target->isDecimal())) {
    return false;
  }
  // Decimal-to-numeric conversions need separate precision and overflow rules.
  if (source->isDecimal() && !target->isDecimal()) {
    return false;
  }
  return cudf::is_supported_cast(
      veloxToCudfDataType(source), veloxToCudfDataType(target));
}

LegacyCastFunction::LegacyCastFunction(const core::TypedExprPtr& expr)
    : targetType_(veloxToCudfDataType(expr->type())),
      targetPrecision_(
          expr->type()->isDecimal()
              ? getDecimalPrecisionScale(*expr->type()).first
              : 0),
      needsSparkKernel_(
          expr->type()->isDecimal() ||
          (isFloatingPoint(expr->inputs()[0]->type()) &&
           (expr->type()->isTinyint() || expr->type()->isSmallint() ||
            expr->type()->isInteger() || expr->type()->isBigint()))) {
  VELOX_CHECK(
      canEvaluate(expr), "Unsupported Spark legacy cast: {}", expr->toString());
}

ColumnOrView LegacyCastFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_EQ(inputColumns.size(), 1);
  const auto input = asView(inputColumns[0]);
  if (needsSparkKernel_) {
    return legacyCastNumeric(input, targetType_, targetPrecision_, stream, mr);
  }
  return cudf::cast(input, targetType_, stream, mr);
}

} // namespace facebook::velox::cudf_velox::sparksql
