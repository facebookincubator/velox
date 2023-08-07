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
#include "velox/functions/sparksql/aggregates/DecimalSumAggregate.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {

namespace {
template <typename TInput, typename TAccumulator, typename ResultType>
using SumAggregate = SumAggregateBase<TInput, TAccumulator, ResultType, true>;

TypePtr getDecimalSumType(
    const TypePtr& resultType,
    core::AggregationNode::Step step) {
  return exec::isPartialOutput(step) ? resultType->childAt(0) : resultType;
}
} // namespace

exec::AggregateRegistrationResult registerSum(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("real")
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
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
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
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        auto inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::TINYINT:
            return std::make_unique<SumAggregate<int8_t, int64_t, int64_t>>(
                BIGINT());
          case TypeKind::SMALLINT:
            return std::make_unique<SumAggregate<int16_t, int64_t, int64_t>>(
                BIGINT());
          case TypeKind::INTEGER:
            return std::make_unique<SumAggregate<int32_t, int64_t, int64_t>>(
                BIGINT());
          case TypeKind::BIGINT: {
            if (inputType->isShortDecimal()) {
              auto sumType = getDecimalSumType(resultType, step);
              if (sumType->isShortDecimal()) {
                return std::make_unique<DecimalSumAggregate<int64_t, int64_t>>(
                    resultType, sumType);
              } else if (sumType->isLongDecimal()) {
                return std::make_unique<DecimalSumAggregate<int64_t, int128_t>>(
                    resultType, sumType);
              }
            }
            return std::make_unique<SumAggregate<int64_t, int64_t, int64_t>>(
                BIGINT());
          }
          case TypeKind::HUGEINT: {
            if (inputType->isLongDecimal()) {
              auto sumType = getDecimalSumType(resultType, step);
              // If inputType is long decimal,
              // its output type always be long decimal.
              return std::make_unique<DecimalSumAggregate<int128_t, int128_t>>(
                  resultType, sumType);
            }
          }
          case TypeKind::REAL:
            if (resultType->kind() == TypeKind::REAL) {
              return std::make_unique<SumAggregate<float, double, float>>(
                  resultType);
            }
            return std::make_unique<SumAggregate<float, double, double>>(
                DOUBLE());
          case TypeKind::DOUBLE:
            if (resultType->kind() == TypeKind::REAL) {
              return std::make_unique<SumAggregate<double, double, float>>(
                  resultType);
            }
            return std::make_unique<SumAggregate<double, double, double>>(
                DOUBLE());
          case TypeKind::ROW: {
            VELOX_DCHECK(!exec::isRawInput(step));
            auto sumType = getDecimalSumType(resultType, step);
            // For intermediate input agg, input intermediate sum type
            // is equal to final result sum type.
            if (inputType->childAt(0)->isShortDecimal()) {
              return std::make_unique<DecimalSumAggregate<int64_t, int64_t>>(
                  resultType, sumType);
            } else if (inputType->childAt(0)->isLongDecimal()) {
              return std::make_unique<DecimalSumAggregate<int128_t, int128_t>>(
                  resultType, sumType);
            }
          }
          default:
            VELOX_CHECK(
                false,
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
