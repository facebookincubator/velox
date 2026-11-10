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

// Requires final types so scalar overloads cannot widen the operator's inputs.
exec::SimpleFunctionRegistry::ResolvedSimpleFunction resolve(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 2, "AT TIME ZONE must have exactly two inputs");

  const auto& valueType = argTypes[0];
  VELOX_USER_CHECK(
      !valueType->equivalent(*TIME()),
      "AT TIME ZONE requires TIME to be coerced to TIME WITH TIME ZONE");
  VELOX_USER_CHECK(
      valueType->equivalent(*TIMESTAMP()) ||
          isTimestampWithTimeZoneType(valueType) ||
          isTimeWithTimeZone(valueType),
      "AT TIME ZONE requires a time or timestamp value, but got: {}",
      valueType);
  VELOX_USER_CHECK(
      argTypes[1]->equivalent(*VARCHAR()),
      "AT TIME ZONE requires a VARCHAR time zone, but got: {}",
      argTypes[1]);

  auto function =
      exec::simpleFunctions().resolveFunction(functionName, argTypes);
  VELOX_CHECK(
      function.has_value(),
      "AT TIME ZONE scalar is not registered for function and input types: "
      "{} ({})",
      functionName,
      folly::join(", ", argTypes));
  return std::move(function).value();
}

} // namespace

AtTimeZoneCallToSpecialForm::AtTimeZoneCallToSpecialForm(
    std::string functionName)
    : functionName_{std::move(functionName)} {}

TypePtr AtTimeZoneCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return resolve(functionName_, argTypes).type();
}

TypePtr AtTimeZoneCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 2, "AT TIME ZONE must have exactly two inputs");
  coercions.clear();
  coercions.resize(argTypes.size());
  auto coercedTypes = argTypes;
  if (argTypes[0]->equivalent(*TIME())) {
    VELOX_USER_CHECK(
        coercer.coerce(argTypes[0], TIME_WITH_TIME_ZONE()).has_value(),
        "AT TIME ZONE requires TIME to be coerced to TIME WITH TIME ZONE");
    coercions[0] = TIME_WITH_TIME_ZONE();
    coercedTypes[0] = coercions[0];
  }
  if (argTypes[1]->isUnknown()) {
    VELOX_USER_CHECK(
        coercer.coerce(argTypes[1], VARCHAR()).has_value(),
        "AT TIME ZONE cannot coerce UNKNOWN to VARCHAR");
    coercions[1] = VARCHAR();
    coercedTypes[1] = coercions[1];
  }
  return resolve(functionName_, coercedTypes).type();
}

exec::ExprPtr AtTimeZoneCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK_EQ(
      compiledChildren.size(), 2, "AT TIME ZONE must have exactly two inputs");

  std::vector<TypePtr> argTypes{
      compiledChildren[0]->type(),
      compiledChildren[1]->type(),
  };
  const auto& timeZone = compiledChildren[1];
  std::vector<VectorPtr> constantInputs{
      nullptr,
      timeZone->isConstant() ? timeZone->as<exec::ConstantExpr>()->value()
                             : nullptr,
  };

  auto function = resolve(functionName_, argTypes);
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
