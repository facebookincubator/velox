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

#include "velox/functions/sparksql/aggregates/AverageAggregate.h"
#include "velox/functions/lib/aggregates/AverageAggregateBase.h"
#include "velox/functions/sparksql/aggregates/DecimalAverageAggregate.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

template <typename TInput, typename TAccumulator, typename TResult>
class AverageAggregate
    : public AverageAggregateBase<TInput, TAccumulator, TResult> {
 public:
  explicit AverageAggregate(TypePtr resultType)
      : AverageAggregateBase<TInput, TAccumulator, TResult>(resultType) {}

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TAccumulator>();
    auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();

    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);
    rowVector->clearAllNulls();

    int64_t* rawCounts = countVector->mutableRawValues();
    TAccumulator* rawSums = sumVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      // When all inputs are nulls, the partial result is (0, 0).
      char* group = groups[i];
      auto* sumCount = this->accumulator(group);
      rawCounts[i] = sumCount->count;
      rawSums[i] = sumCount->sum;
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResult>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = this->getRawNulls(vector);

    TResult* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto* sumCount = this->accumulator(group);
      if (sumCount->count == 0) {
        // In Spark, if all inputs are null, count will be 0,
        // and the result of final avg will be null.
        vector->setNull(i, true);
      } else {
        this->clearNull(rawNulls, i);
        rawValues[i] = static_cast<TResult>(sumCount->sum) / sumCount->count;
      }
    }
  }
};

TypePtr getAvgResultType(const TypePtr& rawInputType) {
  auto [p, s] = getDecimalPrecisionScale(*rawInputType.get());
  // In Spark,
  // final avg result precision = input precision + 4
  // final avg result scale = input scale + 4
  return DECIMAL(
      std::min<uint8_t>(LongDecimalType::kMaxPrecision, p + 4),
      std::min<uint8_t>(LongDecimalType::kMaxPrecision, s + 4));
}

std::unique_ptr<exec::Aggregate> getDecimalAverageAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  auto inputType = argTypes[0];
  VELOX_USER_CHECK(inputType->isDecimal());
  auto avgType = getAvgResultType(inputType);
  if (inputType->isShortDecimal()) {
    if (avgType->isShortDecimal()) {
      return std::make_unique<exec::SimpleAggregateAdapter<
          DecimalAverageAggregate<int64_t, int64_t>>>(
          step, argTypes, resultType);
    }
    return std::make_unique<exec::SimpleAggregateAdapter<
        DecimalAverageAggregate<int64_t, int128_t>>>(
        step, argTypes, resultType);
  }
  return std::make_unique<exec::SimpleAggregateAdapter<
      DecimalAverageAggregate<int128_t, int128_t>>>(step, argTypes, resultType);
}

} // namespace

/// Count is BIGINT() while sum and the final aggregates type depends on
/// the input types:
///       INPUT TYPE    |     SUM             |     AVG
///   ------------------|---------------------|------------------
///     DOUBLE          |     DOUBLE          |    DOUBLE
///     REAL            |     DOUBLE          |    DOUBLE
///     ALL INTs        |     DOUBLE          |    DOUBLE
///     DECIMAL         |     DECIMAL         |    DECIMAL
exec::AggregateRegistrationResult registerAverage(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  for (const auto& inputType :
       {"smallint", "integer", "bigint", "real", "double"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType("double")
            .intermediateType("row(double,bigint)")
            .argumentType(inputType)
            .build());
  }

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("i_precision", "min(38, a_precision + 10)")
          .integerVariable("r_precision", "min(38, a_precision + 4)")
          .integerVariable("r_scale", "min(38, a_scale + 4)")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(i_precision, a_scale), BIGINT)")
          .returnType("DECIMAL(r_precision, r_scale)")
          .build());

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (exec::isRawInput(step)) {
          switch (inputType->kind()) {
            case TypeKind::SMALLINT:
              return std::make_unique<
                  AverageAggregate<int16_t, double, double>>(resultType);
            case TypeKind::INTEGER:
              return std::make_unique<
                  AverageAggregate<int32_t, double, double>>(resultType);
            case TypeKind::BIGINT: {
              if (inputType->isShortDecimal()) {
                auto precision =
                    getDecimalPrecisionScale(*inputType.get()).first;
                // Spark has an optimization rule named DecimalAggregate that
                // converts input data to a Long for average calculation when
                // the input data is of type decimal and the precision is <= 11.
                // Therefore, the precision of the decimal inputs received by
                // the decimal average is guaranteed to be > 11. Since the
                // precision of the intermediate sum is input precision + 10, it
                // will definitely be > 21, which means the intermediate sum
                // will always be a long decimal. We can directly assume that
                // the physical type of the intermediate sum is int128_t.
                VELOX_USER_CHECK_GT(precision, 11);
                return getDecimalAverageAggregate(step, argTypes, resultType);
              }
              return std::make_unique<
                  AverageAggregate<int64_t, double, double>>(resultType);
            }
            case TypeKind::HUGEINT: {
              if (inputType->isLongDecimal()) {
                return getDecimalAverageAggregate(step, argTypes, resultType);
              }
              VELOX_NYI();
            }
            case TypeKind::REAL:
              return std::make_unique<AverageAggregate<float, double, double>>(
                  resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<AverageAggregate<double, double, double>>(
                  resultType);
            default:
              VELOX_FAIL(
                  "Unknown input type for {} aggregation {}",
                  name,
                  inputType->kindName());
          }
        } else {
          checkAvgIntermediateType(inputType);
          switch (resultType->kind()) {
            case TypeKind::REAL:
              return std::make_unique<AverageAggregate<int64_t, double, float>>(
                  resultType);
            case TypeKind::DOUBLE:
            case TypeKind::ROW:
              if (inputType->childAt(0)->isDecimal()) {
                return std::make_unique<
                    exec::SimpleAggregateAdapter<DecimalAverageAggregate<
                        int128_t /*unused*/,
                        int128_t /*unused*/>>>(step, argTypes, resultType);
              }
              return std::make_unique<
                  AverageAggregate<int64_t, double, double>>(resultType);
            case TypeKind::BIGINT:
              VELOX_USER_CHECK(resultType->isShortDecimal());
              return std::make_unique<exec::SimpleAggregateAdapter<
                  DecimalAverageAggregate<int64_t /*unused*/, int64_t>>>(
                  step, argTypes, resultType);
            case TypeKind::HUGEINT:
              VELOX_USER_CHECK(resultType->isLongDecimal());
              return std::make_unique<exec::SimpleAggregateAdapter<
                  DecimalAverageAggregate<int128_t /*unused*/, int128_t>>>(
                  step, argTypes, resultType);
            default:
              VELOX_FAIL(
                  "Unsupported result type for final aggregation: {}",
                  resultType->kindName());
          }
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
