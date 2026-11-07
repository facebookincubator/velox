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
#include "velox/functions/prestosql/AtTimeZone.h"

#include <folly/String.h>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunctionListener.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::functions::prestosql {
namespace {

// Resolves 'functionName' for 'argTypes' as is when 'coercer' is null, or with
// coercions from 'coercer' otherwise. Of the value types, only TIME may be
// coerced. Throws a user error for argument types AT TIME ZONE rejects.
exec::SimpleFunctionRegistry::ResolvedSimpleFunction resolveAtTimeZone(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer* coercer) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 2, "AT TIME ZONE must have exactly two inputs");

  const auto& valueType = argTypes[0];
  VELOX_USER_CHECK(
      valueType->equivalent(*TIMESTAMP()) ||
          isTimestampWithTimeZoneType(valueType) ||
          isTimeWithTimeZone(valueType) ||
          (coercer != nullptr && valueType->equivalent(*TIME())),
      "AT TIME ZONE requires a time or timestamp value, but got: {}",
      valueType->toString());

  auto function = coercer == nullptr
      ? exec::simpleFunctions().resolveFunction(functionName, argTypes)
      : exec::simpleFunctions().resolveFunctionWithCoercions(
            functionName, argTypes, coercions, *coercer);
  VELOX_USER_CHECK(
      function.has_value(),
      "AT TIME ZONE requires a VARCHAR time zone, but got: {}",
      argTypes[1]->toString());
  return std::move(function).value();
}

} // namespace

AtTimeZoneCallToSpecialForm::AtTimeZoneCallToSpecialForm(
    std::string functionName)
    : functionName_{std::move(functionName)} {}

TypePtr AtTimeZoneCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  std::vector<TypePtr> coercions;
  return resolveAtTimeZone(
             functionName_, argTypes, coercions, /*coercer=*/nullptr)
      .type();
}

TypePtr AtTimeZoneCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  return resolveAtTimeZone(functionName_, argTypes, coercions, &coercer).type();
}

exec::ExprPtr AtTimeZoneCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  std::vector<TypePtr> argTypes;
  argTypes.reserve(compiledChildren.size());
  // A constant time zone is resolved once instead of per row.
  std::vector<VectorPtr> constantInputs;
  constantInputs.reserve(compiledChildren.size());
  for (const auto& child : compiledChildren) {
    argTypes.push_back(child->type());
    if (child->isConstant()) {
      constantInputs.push_back(child->as<exec::ConstantExpr>()->value());
    } else {
      constantInputs.push_back(nullptr);
    }
  }

  std::vector<TypePtr> coercions;
  auto function = resolveAtTimeZone(
      functionName_, argTypes, coercions, /*coercer=*/nullptr);
  VELOX_USER_CHECK(
      type->equivalent(*function.type()),
      "Found incompatible return types for '{}' ({} vs. {}) "
      "for input types ({}).",
      functionName_,
      function.type(),
      type,
      folly::join(", ", argTypes));

  const auto& metadata = function.metadata();
  return std::make_shared<exec::Expr>(
      type,
      std::move(compiledChildren),
      function.createFunction()->createVectorFunction(
          argTypes, constantInputs, config),
      metadata,
      functionName_,
      trackCpuUsage,
      exec::createVectorFunctionListeners(functionName_, metadata, config));
}

} // namespace facebook::velox::functions::prestosql
