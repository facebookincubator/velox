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
#include "velox/core/QueryConfig.h"
#include "velox/functions/lib/aggregates/SetBaseAggregate.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Empty arrays are returned for empty groups by setting 'nullForEmpty'
// as false.
template <typename T, bool ignoreNulls>
using SparkSetAggAggregate = SetAggAggregate<T, ignoreNulls, false>;

// NaN inputs are treated as distinct values.
template <typename T, bool ignoreNulls>
using FloatSetAggAggregateNaNUnaware = SetAggAggregate<
    T,
    ignoreNulls,
    false,
    velox::aggregate::prestosql::FloatSetAccumulatorNaNUnaware<T>>;

template <bool ignoreNulls>
std::unique_ptr<exec::Aggregate> createSetAgg(
    const TypeKind typeKind,
    const TypePtr& inputType,
    const TypePtr& resultType) {
  switch (typeKind) {
    case TypeKind::BOOLEAN:
      return std::make_unique<SparkSetAggAggregate<bool, ignoreNulls>>(
          resultType);
    case TypeKind::TINYINT:
      return std::make_unique<SparkSetAggAggregate<int8_t, ignoreNulls>>(
          resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<SparkSetAggAggregate<int16_t, ignoreNulls>>(
          resultType);
    case TypeKind::INTEGER:
      return std::make_unique<SparkSetAggAggregate<int32_t, ignoreNulls>>(
          resultType);
    case TypeKind::BIGINT:
      return std::make_unique<SparkSetAggAggregate<int64_t, ignoreNulls>>(
          resultType);
    case TypeKind::HUGEINT:
      VELOX_CHECK(
          inputType->isLongDecimal(),
          "Non-decimal use of HUGEINT is not supported");
      return std::make_unique<SparkSetAggAggregate<int128_t, ignoreNulls>>(
          resultType);
    case TypeKind::REAL:
      return std::make_unique<
          FloatSetAggAggregateNaNUnaware<float, ignoreNulls>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<
          FloatSetAggAggregateNaNUnaware<double, ignoreNulls>>(resultType);
    case TypeKind::TIMESTAMP:
      return std::make_unique<SparkSetAggAggregate<Timestamp, ignoreNulls>>(
          resultType);
    case TypeKind::VARBINARY:
      [[fallthrough]];
    case TypeKind::VARCHAR:
      return std::make_unique<SparkSetAggAggregate<StringView, ignoreNulls>>(
          resultType);
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::ROW:
      return std::make_unique<SparkSetAggAggregate<ComplexType, ignoreNulls>>(
          resultType);
    case TypeKind::UNKNOWN:
      return std::make_unique<SparkSetAggAggregate<UnknownValue, ignoreNulls>>(
          resultType);
    default:
      VELOX_UNSUPPORTED("Unsupported type {}", TypeKindName::toName(typeKind));
  }
}

} // namespace

void registerCollectSetAggAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures = {
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .intermediateType("array(T)")
          .argumentType("T")
          .build()};

  exec::registerAggregateFunction(
      prefix + "collect_set",
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1);

        const bool isRawInput = exec::isRawInput(step);
        const TypePtr& inputType =
            isRawInput ? argTypes[0] : argTypes[0]->childAt(0);
        const TypeKind typeKind = inputType->kind();

        if (config.sparkCollectSetIgnoreNulls()) {
          return createSetAgg<true>(typeKind, inputType, resultType);
        }
        return createSetAgg<false>(typeKind, inputType, resultType);
      },
      {.ignoreDuplicates = true},
      withCompanionFunctions,
      overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
