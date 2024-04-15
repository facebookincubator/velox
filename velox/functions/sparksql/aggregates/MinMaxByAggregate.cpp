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

#include "velox/functions/lib/aggregates/SimpleMinMaxByAggregatesBase.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {

namespace {

/// Returns compare result align with Spark's specific behavior,
/// which returns true if the value in 'index' row of 'newComparisons' is
/// greater than/equal or less than/equal the value in the 'accumulator'.
template <bool sparkGreaterThan, typename T, typename TAccumulator>
struct SparkComparator {
  FOLLY_ALWAYS_INLINE static bool compare(T accumulator, T newComparison) {
    if constexpr (sparkGreaterThan) {
      return newComparison >= accumulator;
    } else {
      return newComparison <= accumulator;
    }
  }

  static bool compare(
      const TAccumulator& accumulator,
      const DecodedVector& decoded,
      vector_size_t index) {
    static const CompareFlags kCompareFlags{
        true, // nullsFirst
        true, // ascending
        false, // equalsOnly
        CompareFlags::NullHandlingMode::kNullAsValue};

    int32_t result = accumulator.compare(decoded, index, kCompareFlags).value();
    if constexpr (sparkGreaterThan) {
      return result <= 0;
    }
    return result >= 0;
  }
};

std::string toString(const std::vector<TypePtr>& types) {
  std::ostringstream out;
  for (auto i = 0; i < types.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << types[i]->toString();
  }
  return out.str();
}

template <bool isMaxFunc>
exec::AggregateRegistrationResult registerMinMaxBy(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  // V, C -> row(V, C) -> V.
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .typeVariable("V")
                           .typeVariable("C")
                           .returnType("V")
                           .intermediateType("row(V,C)")
                           .argumentType("V")
                           .argumentType("C")
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
        const auto isRawInput = exec::isRawInput(step);
        const std::string errorMessage = fmt::format(
            "Unknown input types for {} ({}) aggregation: {}",
            name,
            mapAggregationStepToName(step),
            toString(argTypes));

        if (isRawInput) {
          // Input is: V, C.
          return createAll<SparkComparator, isMaxFunc, false>(
              resultType, argTypes[0], argTypes[1], errorMessage);
        } else {
          // Input is: ROW(V, C).
          const auto& rowType = argTypes[0];
          const auto& valueType = rowType->childAt(0);
          const auto& compareType = rowType->childAt(1);
          return createAll<SparkComparator, isMaxFunc, false>(
              resultType, valueType, compareType, errorMessage);
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerMinMaxByAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMinMaxBy<true>(prefix + "max_by", withCompanionFunctions, overwrite);
  registerMinMaxBy<false>(prefix + "min_by", withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
