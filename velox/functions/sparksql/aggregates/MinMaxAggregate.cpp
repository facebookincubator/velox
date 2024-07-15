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

#include "velox/functions/lib/aggregates/MinMaxAggregateBase.h"

namespace facebook::velox::functions::aggregate::sparksql {

namespace {

// Extracts timestamp value as microsecond precision in max aggregate.
class TimestampMicrosPrecisionMaxAggregate : public MaxAggregate<Timestamp> {
 public:
  explicit TimestampMicrosPrecisionMaxAggregate(TypePtr resultType)
      : MaxAggregate<Timestamp>(resultType) {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    using BaseAggregate =
        SimpleNumericAggregate<Timestamp, Timestamp, Timestamp>;
    BaseAggregate::template doExtractValues<Timestamp>(
        groups, numGroups, result, [&](char* group) {
          auto ts = *BaseAggregate::Aggregate::template value<Timestamp>(group);
          return Timestamp::fromMicros(ts.toMicros());
        });
  }
};

// Extracts timestamp value as microsecond precision in min aggregate.
class TimestampMicrosPrecisionMinAggregate : public MinAggregate<Timestamp> {
 public:
  explicit TimestampMicrosPrecisionMinAggregate(TypePtr resultType)
      : MinAggregate<Timestamp>(resultType) {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    using BaseAggregate =
        SimpleNumericAggregate<Timestamp, Timestamp, Timestamp>;
    BaseAggregate::template doExtractValues<Timestamp>(
        groups, numGroups, result, [&](char* group) {
          auto ts = *BaseAggregate::Aggregate::template value<Timestamp>(group);
          return Timestamp::fromMicros(ts.toMicros());
        });
  }
};

exec::AggregateRegistrationResult registerMin(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .orderableTypeVariable("T")
                           .returnType("T")
                           .intermediateType("T")
                           .argumentType("T")
                           .build());
  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          std::vector<TypePtr> argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        auto factory =
            getMinFunctionFactory<TimestampMicrosPrecisionMinAggregate>(
                name, true /*nestedNullAllowed*/, false /*mapTypeSupported*/);
        return factory(step, argTypes, resultType, config);
      },
      {false /*orderSensitive*/},
      withCompanionFunctions,
      overwrite);
}

exec::AggregateRegistrationResult registerMax(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .orderableTypeVariable("T")
                           .returnType("T")
                           .intermediateType("T")
                           .argumentType("T")
                           .build());
  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          std::vector<TypePtr> argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        auto factory =
            getMaxFunctionFactory<TimestampMicrosPrecisionMaxAggregate>(
                name, true /*nestedNullAllowed*/, false /*mapTypeSupported*/);
        return factory(step, argTypes, resultType, config);
      },
      {false /*orderSensitive*/},
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerMinMaxAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMin(prefix + "min", withCompanionFunctions, overwrite);
  registerMax(prefix + "max", withCompanionFunctions, overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
