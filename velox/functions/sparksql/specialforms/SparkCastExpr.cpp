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

bool SparkCastCallToSpecialForm::isAnsiSupported(
    const TypePtr& fromType,
    const TypePtr& toType) {
  // String to Boolean cast supports ANSI mode.
  if (fromType->isVarchar() && toType->isBoolean()) {
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
  
  // For ANSI-supported casts, check ANSI configuration to determine behavior.
  //
  // Exception: For string-to-boolean casts, Spark checks ANSI mode to determine
  // whether to throw on invalid input (ANSI on) or return NULL (ANSI off).
  const bool ansiEnabled = config.sparkAnsiEnabled();
  const bool isTryCast = isAnsiSupported(fromType, type) ? !ansiEnabled : true;
  
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      isTryCast,
      std::make_shared<SparkCastHooks>(config, true));
}

exec::ExprPtr SparkTryCastCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_CHECK_EQ(
      compiledChildren.size(),
      1,
      "TRY CAST statements expect exactly 1 argument, received {}.",
      compiledChildren.size());
  return std::make_shared<SparkCastExpr>(
      type,
      std::move(compiledChildren[0]),
      trackCpuUsage,
      true,
      std::make_shared<SparkCastHooks>(config, false));
}
} // namespace facebook::velox::functions::sparksql
