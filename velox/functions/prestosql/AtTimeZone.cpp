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
#include "velox/expression/SimpleFunctionAdapter.h"
#include "velox/expression/VectorFunctionListener.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/DateTimeCalendarFunctions.h"
#include "velox/functions/prestosql/DateTimeFunctions.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::functions::prestosql {
namespace {

// Applies the type rules to final argument types and returns the result type.
TypePtr resolveResultType(
    const std::vector<TypePtr>& argTypes,
    bool ignoreTimeZoneLabel) {
  const auto& valueType = argTypes[0];
  VELOX_USER_CHECK(
      valueType->equivalent(*TIMESTAMP()) ||
          isTimestampWithTimeZoneType(valueType) ||
          isTimeWithTimeZone(valueType),
      "AT TIME ZONE value must be TIMESTAMP, TIMESTAMP WITH TIME ZONE or TIME "
      "WITH TIME ZONE, but got: {}",
      valueType);
  VELOX_USER_CHECK(
      argTypes[1]->equivalent(*VARCHAR()),
      "AT TIME ZONE time zone must be VARCHAR, but got: {}",
      argTypes[1]);

  if (isTimeWithTimeZone(valueType)) {
    return TIME_WITH_TIME_ZONE();
  }
  if (ignoreTimeZoneLabel) {
    return TIMESTAMP();
  }
  return TIMESTAMP_WITH_TIME_ZONE();
}

// A function factory and the metadata of the expression it evaluates.
struct Implementation {
  std::unique_ptr<exec::SimpleFunctionAdapterFactory> factory;
  exec::VectorFunctionMetadata metadata;
};

// Creates the simple function that registerFunction would register for 'Func'.
// Prefers the owner the function declares, as the function registry does.
template <template <class> typename Func, typename TReturn, typename... TArgs>
Implementation makeImplementation(std::string_view defaultOwner) {
  using Udf = Func<exec::VectorExec>;
  using Holder = SimpleFunctionHolder<Func, TReturn, TArgs...>;
  const std::string_view udfOwner = core::udf_owner<Udf>::value();
  return {
      std::make_unique<exec::SimpleFunctionAdapterFactoryImpl<Holder>>(),
      exec::VectorFunctionMetadataBuilder()
          .deterministic(core::udf_is_deterministic<Udf>())
          .defaultNullBehavior(Holder::is_default_null_behavior)
          .owner(udfOwner.empty() ? defaultOwner : udfOwner)
          .build(),
  };
}

// Returns the implementation for 'valueType' in the given mode.
Implementation selectImplementation(
    const TypePtr& valueType,
    bool ignoreTimeZoneLabel,
    std::string_view owner) {
  if (isTimeWithTimeZone(valueType)) {
    return makeImplementation<
        AtTimezoneTimeWithTimezoneFunction,
        TimeWithTimezone,
        TimeWithTimezone,
        Varchar>(owner);
  }
  if (isTimestampWithTimeZoneType(valueType)) {
    if (ignoreTimeZoneLabel) {
      return makeImplementation<
          AtTimezoneToTimestampFunction,
          Timestamp,
          TimestampWithTimezone,
          Varchar>(owner);
    }
    return makeImplementation<
        AtTimezoneFunction,
        TimestampWithTimezone,
        TimestampWithTimezone,
        Varchar>(owner);
  }
  if (ignoreTimeZoneLabel) {
    return makeImplementation<
        AtTimezoneFromTimestampToTimestampFunction,
        Timestamp,
        Timestamp,
        Varchar>(owner);
  }
  return makeImplementation<
      AtTimezoneFromTimestampFunction,
      TimestampWithTimezone,
      Timestamp,
      Varchar>(owner);
}

} // namespace

AtTimeZoneCallToSpecialForm::AtTimeZoneCallToSpecialForm(
    std::string name,
    bool ignoreTimeZoneLabel,
    std::string_view owner)
    : name_{std::move(name)},
      ignoreTimeZoneLabel_{ignoreTimeZoneLabel},
      owner_{owner} {}

TypePtr AtTimeZoneCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 2, "AT TIME ZONE must have exactly two inputs");
  return resolveResultType(argTypes, ignoreTimeZoneLabel_);
}

TypePtr AtTimeZoneCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& /*coercer*/) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 2, "AT TIME ZONE must have exactly two inputs");
  coercions.clear();
  coercions.resize(argTypes.size());
  auto coercedTypes = argTypes;
  if (argTypes[0]->equivalent(*TIME())) {
    coercions[0] = TIME_WITH_TIME_ZONE();
    coercedTypes[0] = coercions[0];
  }
  if (argTypes[1]->isUnknown()) {
    coercions[1] = VARCHAR();
    coercedTypes[1] = coercions[1];
  }
  return resolveResultType(coercedTypes, ignoreTimeZoneLabel_);
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
  const auto resultType = resolveResultType(argTypes, ignoreTimeZoneLabel_);
  VELOX_USER_CHECK(
      type->equivalent(*resultType),
      "AT TIME ZONE return type must match its input types: {} vs. {} for {}({})",
      type,
      resultType,
      name_,
      folly::join(", ", argTypes));

  const auto& timeZone = compiledChildren[1];
  std::vector<VectorPtr> constantInputs{
      nullptr,
      timeZone->isConstant() ? timeZone->as<exec::ConstantExpr>()->value()
                             : nullptr,
  };

  auto implementation =
      selectImplementation(argTypes[0], ignoreTimeZoneLabel_, owner_);
  return std::make_shared<exec::Expr>(
      type,
      std::move(compiledChildren),
      implementation.factory->createVectorFunction(
          argTypes, constantInputs, config),
      implementation.metadata,
      name_,
      trackCpuUsage,
      exec::createVectorFunctionListeners(
          name_, implementation.metadata, config));
}

} // namespace facebook::velox::functions::prestosql
