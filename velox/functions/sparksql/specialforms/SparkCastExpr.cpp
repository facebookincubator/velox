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

#include "velox/functions/sparksql/specialforms/SparkCastExpr.h"

namespace {

bool isIntegralType(const facebook::velox::TypePtr& type) {
  return type == facebook::velox::TINYINT() ||
      type == facebook::velox::SMALLINT() ||
      type == facebook::velox::INTEGER() || type == facebook::velox::BIGINT();
}

bool isFloatingPointType(const facebook::velox::TypePtr& type) {
  return type == facebook::velox::REAL() || type == facebook::velox::DOUBLE();
}

bool isNumericType(const facebook::velox::TypePtr& type) {
  return isIntegralType(type) || isFloatingPointType(type);
}

} // namespace

namespace facebook::velox::functions::sparksql {
namespace {

bool isIntegralType(const TypePtr& type) {
  return type == TINYINT() || type == SMALLINT() || type == INTEGER() ||
      type == BIGINT();
}

} // namespace

bool SparkCastCallToSpecialForm::isAnsiSupported(
    const TypePtr& fromType,
    const TypePtr& toType) {
  // String to Boolean, Integer, or Date types support ANSI mode.
  if (fromType->isVarchar()) {
    if (toType->isBoolean() || toType->isDate()) {
      return true;
    }
    if (isIntegralType(toType)) {
      // For string-to-integral casts, ANSI mode rejects invalid formats (e.g.
      // decimal points) instead of returning NULL.
      return true;
    }
  }

  // Numeric types to integral types cast supports ANSI mode (overflow check).
  if (isNumericType(fromType) && isIntegralType(toType)) {
    return true;
  }

  // Timestamp to integral types cast supports ANSI mode (overflow check).
  if (fromType->isTimestamp() && isIntegralType(toType)) {
    return true;
  }

  return false;
}

exec::ExprPtr SparkCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());

  const auto& fromType = compiledChildren[0]->type();

  // In Spark SQL (with ANSI mode off), both CAST and TRY_CAST behave like
  // Velox's try_cast, so we set 'isTryCast' to true when ANSI is disabled or
  // the specific cast operation doesn't support ANSI mode.
  const bool isTryCast =
      !config.sparkAnsiEnabled() || !isAnsiSupported(fromType, type);

  // For ANSI-supported casts, CAST mirrors TRY_CAST when ANSI is disabled.
  // The distinction is controlled by the 'allowOverflow' flag in
  // SparkCastHooks.
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      isTryCast,
      std::make_shared<SparkCastHooks>(config, isTryCast));
}

exec::ExprPtr SparkTryCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "TRY_CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());

  // TRY_CAST always uses allowOverflow=false to return NULL on cast failures.
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      true,
      std::make_shared<SparkCastHooks>(config, false));
}

} // namespace facebook::velox::functions::sparksql
