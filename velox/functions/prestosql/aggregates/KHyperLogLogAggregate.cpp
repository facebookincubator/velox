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

#include "velox/functions/prestosql/aggregates/KHyperLogLogAggregate.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

using namespace facebook::velox::aggregate;

namespace facebook::velox::aggregate::prestosql {
namespace {

/// Registration for khyperloglog_agg aggregate function.
exec::AggregateRegistrationResult registerKHyperLogLogAgg(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  std::vector<std::string> inputTypes = {
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

  // Generate all combinations of (valueType, uiiType)
  // Both parameters can be any type. Both will be hashed to BIGINT internally
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

  for (const auto& uiiType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .returnType("khyperloglog")
            .intermediateType("khyperloglog")
            .argumentType("DECIMAL(a_precision, a_scale)")
            .argumentType(uiiType)
            .build());
  }

  for (const auto& valueType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .returnType("khyperloglog")
            .intermediateType("khyperloglog")
            .argumentType(valueType)
            .argumentType("DECIMAL(b_precision, b_scale)")
            .build());
  }

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .returnType("khyperloglog")
          .intermediateType("khyperloglog")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .argumentType("DECIMAL(b_precision, b_scale)")
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
        VELOX_CHECK_EQ(
            argTypes.size(), 2, "{}: unexpected number of arguments", name);

        auto valueKind = argTypes[0]->kind();
        auto uiiKind = argTypes[1]->kind();
        // hhhh should i do the type conversion here? - i think so

        // Double dispatch: first on value type, then on UII type
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            dispatchOnUiiType, valueKind, uiiKind, resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerKHyperLogLogAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerKHyperLogLogAgg(
      prefix + kKHyperLogLogAgg, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
