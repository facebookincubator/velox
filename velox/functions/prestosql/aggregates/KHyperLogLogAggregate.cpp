/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/prestosql/aggregates/KHyperLogLogAggregate.h"
#include "velox/exec/Aggregate.h"

using namespace facebook::velox::aggregate;

namespace facebook::velox::aggregate::prestosql {
namespace {

template <TypeKind TValueKind, TypeKind TUiiKind>
std::unique_ptr<exec::Aggregate> createKHyperLogLogAggregate(
    const TypePtr& resultType) {
  using TValue = typename TypeTraits<TValueKind>::NativeType;
  using TUii = typename TypeTraits<TUiiKind>::NativeType;
  return std::make_unique<KHyperLogLogAggregate<TValue, TUii>>(resultType);
}

template <TypeKind TJoinKeyKind>
std::unique_ptr<exec::Aggregate> dispatchOnUiiType(
    TypeKind uiiKind,
    const TypePtr& resultType) {
  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createKHyperLogLogAggregate, TJoinKeyKind, uiiKind, resultType);
}

/// Registration for khyperloglog_agg aggregate function.
std::vector<exec::AggregateRegistrationResult> registerKHyperLogLogAgg(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  // Register all physical types for both JoinKey and UII.
  std::vector<std::string> inputTypes = {
      "boolean",
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real",
      "double",
      "varchar",
      "varbinary",
      "timestamp",
      "date"};

  // Generate all combinations of (joinKeyType, uiiType)
  for (const auto& valueType : inputTypes) {
    for (const auto& uiiType : inputTypes) {
      signatures.push_back(
          exec::AggregateFunctionSignatureBuilder()
              .returnType("khyperloglog")
              .intermediateType("khyperloglog")
              .argumentType(valueType)
              .argumentType(uiiType)
              .build());
    }
  }

  return exec::registerAggregateFunction(
      names,
      signatures,
      [names](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(),
            2,
            "{}: unexpected number of arguments",
            names.front());

        auto joinKeyKind = argTypes[0]->kind();
        auto uiiKind = argTypes[1]->kind();

        // First dispatch on JoinKey type, then on UII type.
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            dispatchOnUiiType, joinKeyKind, uiiKind, resultType);
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace

void registerKHyperLogLogAggregates(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  registerKHyperLogLogAgg(names, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
