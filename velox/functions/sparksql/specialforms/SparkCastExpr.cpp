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

namespace facebook::velox::functions::sparksql {
namespace {

bool isSparkIntegralType(const TypePtr& type) {
  if (type->isTinyint() || type->isSmallint()) {
    return true;
  }
  if (type->isInteger()) {
    return !type->isDate() && !type->isIntervalYearMonth();
  }
  if (type->isBigint()) {
    return !type->isIntervalDayTime() && !type->isTime();
  }
  return false;
}

} // namespace

bool SparkCastCallToSpecialForm::isAnsiSupported(
    const TypePtr& fromType,
    const TypePtr& toType) {
  // String to Boolean or Integer types support ANSI mode
  if (fromType->isVarchar()) {
    return toType->isBoolean() || isSparkIntegralType(toType);
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
  const bool ansiSupported = isAnsiSupported(fromType, type);

  // In Spark SQL (with ANSI mode off), both CAST and TRY_CAST behave like
  // Velox's try_cast, so we set 'isTryCast' to true when ANSI is disabled or
  // the specific cast operation doesn't support ANSI mode.
  const bool isTryCast = !config.sparkAnsiEnabled() || !ansiSupported;

  // For string-to-integer casts with ANSI enabled, we need to disallow
  // decimal points. This is controlled by allowOverflow: when false, it
  // uses SparkTryCastPolicy (truncate=false, no decimals allowed).
  // The distinction between CAST (ANSI off) and TRY_CAST is limited to
  // overflow handling, which is managed by the 'allowOverflow' flag in
  // SparkCastHooks.
  const bool allowOverflow = !(config.sparkAnsiEnabled() && ansiSupported);

  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      isTryCast,
      std::make_shared<SparkCastHooks>(config, allowOverflow));
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

  // TRY_CAST always uses allowOverflow=false to:
  // 1. Return NULL on overflow (instead of wrapping)
  // 2. Return NULL on invalid string formats like "125.5" for integers
  // This uses SparkTryCastPolicy which handles both cases correctly.
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      true,
      std::make_shared<SparkCastHooks>(config, false));
}

} // namespace facebook::velox::functions::sparksql
