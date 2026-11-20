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

#include "velox/functions/sparksql/aggregates/TrySumAggregate.h"

#include "velox/functions/sparksql/aggregates/SumAggregate.h"

namespace facebook::velox::functions::aggregate::sparksql {

namespace {

template <typename TInput>
std::unique_ptr<exec::Aggregate> createTrySumAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  return std::make_unique<
      exec::SimpleAggregateAdapter<TrySumAggregate<TInput>>>(
      step, argTypes, resultType);
}

} // namespace

exec::AggregateRegistrationResult registerTrySum(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  // Spark's try_sum on DECIMAL is its sum with null on overflow, so it shares
  // the decimal sum's signature and implementation.
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
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
            .intermediateType("row(bigint, boolean)")
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
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        const auto& inputType = argTypes[0];
        if (inputType->isDecimal() ||
            (inputType->isRow() && inputType->childAt(0)->isDecimal())) {
          return createNonThrowingDecimalSum(step, argTypes, resultType);
        }
        switch (inputType->kind()) {
          case TypeKind::TINYINT:
            return createTrySumAggregate<int8_t>(step, argTypes, resultType);
          case TypeKind::SMALLINT:
            return createTrySumAggregate<int16_t>(step, argTypes, resultType);
          case TypeKind::INTEGER:
            return createTrySumAggregate<int32_t>(step, argTypes, resultType);
          case TypeKind::BIGINT:
            return createTrySumAggregate<int64_t>(step, argTypes, resultType);
          case TypeKind::ROW:
            // Intermediate input, ROW(BIGINT sum, isEmpty).
            VELOX_DCHECK(!exec::isRawInput(step));
            return createTrySumAggregate<int64_t>(step, argTypes, resultType);
          default:
            VELOX_UNREACHABLE(
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
