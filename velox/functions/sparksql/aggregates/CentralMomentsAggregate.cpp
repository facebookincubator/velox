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

#include "velox/functions/sparksql/aggregates/CentralMomentsAggregate.h"
#include <limits>
#include "velox/functions/lib/aggregates/CentralMomentsAggregatesBase.h"

namespace facebook::velox::functions::aggregate::sparksql {

namespace {
template <bool nullOnDivideByZero>
struct SkewnessResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    if constexpr (nullOnDivideByZero) {
      return accumulator.count() >= 1 && accumulator.m2() != 0;
    }
    return accumulator.count() >= 1;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    if (accumulator.m2() == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(accumulator.count()) * accumulator.m3() /
        std::pow(accumulator.m2(), 1.5);
  }
};

template <bool nullOnDivideByZero>
struct KurtosisResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    if constexpr (nullOnDivideByZero) {
      return accumulator.count() >= 1 && accumulator.m2() != 0;
    }
    return accumulator.count() >= 1;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    if (accumulator.m2() == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double count = accumulator.count();
    double m2 = accumulator.m2();
    double m4 = accumulator.m4();
    return count * m4 / (m2 * m2) - 3.0;
  }
};

std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> getSignatures() {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  std::vector<std::string> inputTypes = {
      "smallint", "integer", "bigint", "real", "double"};
  for (const auto& inputType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType("double")
            .intermediateType(CentralMomentsIntermediateResult::type())
            .argumentType(inputType)
            .build());
  }
  return signatures;
}

exec::AggregateRegistrationResult registerSkewness(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures =
      getSignatures();

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (config.sparkLegacyStatisticalAggregate()) {
          if (exec::isRawInput(step)) {
            switch (inputType->kind()) {
              case TypeKind::SMALLINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int16_t,
                    SkewnessResultAccessor<false>>>(resultType);
              case TypeKind::INTEGER:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int32_t,
                    SkewnessResultAccessor<false>>>(resultType);
              case TypeKind::BIGINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int64_t,
                    SkewnessResultAccessor<false>>>(resultType);
              case TypeKind::DOUBLE:
                return std::make_unique<CentralMomentsAggregatesBase<
                    double,
                    SkewnessResultAccessor<false>>>(resultType);
              case TypeKind::REAL:
                return std::make_unique<CentralMomentsAggregatesBase<
                    float,
                    SkewnessResultAccessor<false>>>(resultType);
              default:
                VELOX_UNSUPPORTED(
                    "Unsupported input type: {}. "
                    "Expected SMALLINT, INTEGER, BIGINT, DOUBLE or REAL.",
                    inputType->toString());
            }
          } else {
            checkAccumulatorRowType(
                inputType,
                "Input type for final aggregation must be "
                "(count:bigint, m1:double, m2:double, m3:double, m4:double) struct");
            return std::make_unique<CentralMomentsAggregatesBase<
                int64_t /*unused*/,
                SkewnessResultAccessor<false>>>(resultType);
          }
        } else {
          if (exec::isRawInput(step)) {
            switch (inputType->kind()) {
              case TypeKind::SMALLINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int16_t,
                    SkewnessResultAccessor<true>>>(resultType);
              case TypeKind::INTEGER:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int32_t,
                    SkewnessResultAccessor<true>>>(resultType);
              case TypeKind::BIGINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int64_t,
                    SkewnessResultAccessor<true>>>(resultType);
              case TypeKind::DOUBLE:
                return std::make_unique<CentralMomentsAggregatesBase<
                    double,
                    SkewnessResultAccessor<true>>>(resultType);
              case TypeKind::REAL:
                return std::make_unique<CentralMomentsAggregatesBase<
                    float,
                    SkewnessResultAccessor<true>>>(resultType);
              default:
                VELOX_UNSUPPORTED(
                    "Unsupported input type: {}. "
                    "Expected SMALLINT, INTEGER, BIGINT, DOUBLE or REAL.",
                    inputType->toString());
            }
          } else {
            checkAccumulatorRowType(
                inputType,
                "Input type for final aggregation must be "
                "(count:bigint, m1:double, m2:double, m3:double, m4:double) struct");
            return std::make_unique<CentralMomentsAggregatesBase<
                int64_t /*unused*/,
                SkewnessResultAccessor<true>>>(resultType);
          }
        }
      },
      withCompanionFunctions,
      overwrite);
}

exec::AggregateRegistrationResult registerKurtosis(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures =
      getSignatures();

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (config.sparkLegacyStatisticalAggregate()) {
          if (exec::isRawInput(step)) {
            switch (inputType->kind()) {
              case TypeKind::SMALLINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int16_t,
                    KurtosisResultAccessor<false>>>(resultType);
              case TypeKind::INTEGER:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int32_t,
                    KurtosisResultAccessor<false>>>(resultType);
              case TypeKind::BIGINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int64_t,
                    KurtosisResultAccessor<false>>>(resultType);
              case TypeKind::DOUBLE:
                return std::make_unique<CentralMomentsAggregatesBase<
                    double,
                    KurtosisResultAccessor<false>>>(resultType);
              case TypeKind::REAL:
                return std::make_unique<CentralMomentsAggregatesBase<
                    float,
                    KurtosisResultAccessor<false>>>(resultType);
              default:
                VELOX_UNSUPPORTED(
                    "Unsupported input type: {}. "
                    "Expected SMALLINT, INTEGER, BIGINT, DOUBLE or REAL.",
                    inputType->toString());
            }
          } else {
            checkAccumulatorRowType(
                inputType,
                "Input type for final aggregation must be "
                "(count:bigint, m1:double, m2:double, m3:double, m4:double) struct");
            return std::make_unique<CentralMomentsAggregatesBase<
                int64_t /*unused*/,
                KurtosisResultAccessor<false>>>(resultType);
          }
        } else {
          if (exec::isRawInput(step)) {
            switch (inputType->kind()) {
              case TypeKind::SMALLINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int16_t,
                    KurtosisResultAccessor<true>>>(resultType);
              case TypeKind::INTEGER:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int32_t,
                    KurtosisResultAccessor<true>>>(resultType);
              case TypeKind::BIGINT:
                return std::make_unique<CentralMomentsAggregatesBase<
                    int64_t,
                    KurtosisResultAccessor<true>>>(resultType);
              case TypeKind::DOUBLE:
                return std::make_unique<CentralMomentsAggregatesBase<
                    double,
                    KurtosisResultAccessor<true>>>(resultType);
              case TypeKind::REAL:
                return std::make_unique<CentralMomentsAggregatesBase<
                    float,
                    KurtosisResultAccessor<true>>>(resultType);
              default:
                VELOX_UNSUPPORTED(
                    "Unsupported input type: {}. "
                    "Expected SMALLINT, INTEGER, BIGINT, DOUBLE or REAL.",
                    inputType->toString());
            }
          } else {
            checkAccumulatorRowType(
                inputType,
                "Input type for final aggregation must be "
                "(count:bigint, m1:double, m2:double, m3:double, m4:double) struct");
            return std::make_unique<CentralMomentsAggregatesBase<
                int64_t /*unused*/,
                KurtosisResultAccessor<true>>>(resultType);
          }
        }
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace

void registerCentralMomentsAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerSkewness(prefix + "skewness", withCompanionFunctions, overwrite);
  registerKurtosis(prefix + "kurtosis", withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
