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
#include "velox/functions/sparksql/aggregates/SumAggregate.h"

#include "velox/functions/lib/aggregates/SumAggregateBase.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/aggregates/DecimalSumAggregate.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {

namespace {
// Spark's sum wraps around on integer overflow like Java long arithmetic.
// With ANSI mode enabled it throws instead.
template <
    typename TInput,
    typename TAccumulator,
    typename ResultType,
    bool throwOnOverflow>
using SumAggregate =
    SumAggregateBase<TInput, TAccumulator, ResultType, !throwOnOverflow>;

TypePtr getDecimalSumType(const TypePtr& resultType) {
  if (resultType->isRow()) {
    // If the resultType is ROW, then the type if sum is the type of the first
    // child of the ROW.
    return resultType->childAt(0);
  }
  return resultType;
}

void checkAccumulatorRowType(const TypePtr& type) {
  VELOX_CHECK_EQ(type->kind(), TypeKind::ROW);
  VELOX_CHECK(
      type->childAt(0)->isShortDecimal() || type->childAt(0)->isLongDecimal());
  VELOX_CHECK_EQ(type->childAt(1)->kind(), TypeKind::BOOLEAN);
}

template <bool throwOnOverflow>
std::unique_ptr<exec::Aggregate> constructDecimalSumAgg(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const TypePtr& inputType,
    const TypePtr& sumType) {
  if (inputType->isShortDecimal() && sumType->isShortDecimal()) {
    return std::make_unique<exec::SimpleAggregateAdapter<
        DecimalSumAggregate<int64_t, int64_t, throwOnOverflow>>>(
        step, argTypes, resultType);
  }
  if (inputType->isShortDecimal()) {
    return std::make_unique<exec::SimpleAggregateAdapter<
        DecimalSumAggregate<int64_t, int128_t, throwOnOverflow>>>(
        step, argTypes, resultType);
  }
  return std::make_unique<exec::SimpleAggregateAdapter<
      DecimalSumAggregate<int128_t, int128_t, throwOnOverflow>>>(
      step, argTypes, resultType);
}

template <bool throwOnOverflow>
std::unique_ptr<exec::Aggregate> createSumAggregate(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
  auto inputType = argTypes[0];
  switch (inputType->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<
          SumAggregate<int8_t, int64_t, int64_t, throwOnOverflow>>(BIGINT());
    case TypeKind::SMALLINT:
      return std::make_unique<
          SumAggregate<int16_t, int64_t, int64_t, throwOnOverflow>>(BIGINT());
    case TypeKind::INTEGER:
      return std::make_unique<
          SumAggregate<int32_t, int64_t, int64_t, throwOnOverflow>>(BIGINT());
    case TypeKind::BIGINT: {
      if (inputType->isShortDecimal()) {
        return constructDecimalSumAgg<throwOnOverflow>(
            step,
            argTypes,
            resultType,
            inputType,
            getDecimalSumType(resultType));
      }
      return std::make_unique<
          SumAggregate<int64_t, int64_t, int64_t, throwOnOverflow>>(BIGINT());
    }
    case TypeKind::HUGEINT: {
      VELOX_CHECK(inputType->isLongDecimal());
      // If inputType is long decimal,
      // its output type is always long decimal.
      return constructDecimalSumAgg<throwOnOverflow>(
          step, argTypes, resultType, inputType, getDecimalSumType(resultType));
    }
    case TypeKind::REAL:
      if (resultType->kind() == TypeKind::REAL) {
        return std::make_unique<
            SumAggregate<float, double, float, throwOnOverflow>>(resultType);
      }
      return std::make_unique<
          SumAggregate<float, double, double, throwOnOverflow>>(DOUBLE());
    case TypeKind::DOUBLE:
      if (resultType->kind() == TypeKind::REAL) {
        return std::make_unique<
            SumAggregate<double, double, float, throwOnOverflow>>(resultType);
      }
      return std::make_unique<
          SumAggregate<double, double, double, throwOnOverflow>>(DOUBLE());
    case TypeKind::ROW: {
      VELOX_DCHECK(!exec::isRawInput(step));
      checkAccumulatorRowType(inputType);
      // For the intermediate aggregation step, input intermediate sum
      // type is equal to final result sum type.
      return constructDecimalSumAgg<throwOnOverflow>(
          step,
          argTypes,
          resultType,
          inputType->childAt(0),
          inputType->childAt(0));
    }
    default:
      VELOX_UNREACHABLE(
          "Unknown input type for {} aggregation {}",
          name,
          inputType->kindName());
  }
}
} // namespace

std::unique_ptr<exec::Aggregate> createNonThrowingDecimalSum(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  VELOX_CHECK_EQ(argTypes.size(), 1);
  const auto& inputType = argTypes[0];
  if (inputType->isRow()) {
    checkAccumulatorRowType(inputType);
    return constructDecimalSumAgg<false>(
        step,
        argTypes,
        resultType,
        inputType->childAt(0),
        inputType->childAt(0));
  }
  VELOX_CHECK(inputType->isDecimal());
  return constructDecimalSumAgg<false>(
      step, argTypes, resultType, inputType, getDecimalSumType(resultType));
}

exec::AggregateRegistrationResult registerSum(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("double")
          .argumentType("real")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("double")
          .argumentType("double")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("r_precision", "min(38, a_precision + 10)")
          .integerVariable("r_scale", "min(38, a_scale)")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(r_precision, r_scale), boolean)")
          .returnType("DECIMAL(r_precision, r_scale)")
          .build(),
  };

  for (const auto& inputType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType("bigint")
            .intermediateType("bigint")
            .argumentType(inputType)
            .build());
  }

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        // Spark's sum throws on overflow when ANSI mode is enabled. Otherwise
        // it wraps around on integer overflow and returns null on decimal
        // overflow.
        if (functions::sparksql::SparkQueryConfig{config}.ansiEnabled()) {
          return createSumAggregate<true>(name, step, argTypes, resultType);
        }
        return createSumAggregate<false>(name, step, argTypes, resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
