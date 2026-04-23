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

#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::cudf_velox {

void registerPrestoAggregateFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  unregisterAggregateFunctions();
  registerCommonAggregationFunctions(getGroupbyAggregationRegistry(), prefix);
  registerCommonAggregationFunctions(getReduceAggregationRegistry(), prefix);
  registerReduceOnlyAggregationFunctions(
      getReduceAggregationRegistry(), prefix);

  // Presto (default): SUM(REAL) -> REAL, AVG(REAL) -> REAL.
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build());
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kFinal,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("double")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kFinal,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("double")
          .build());
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kIntermediate,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("double")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kIntermediate,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("double")
          .build());

  appendGroupbyAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build());
  appendGroupbyAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kFinal,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("row(double,bigint)")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kFinal,
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("row(double,bigint)")
          .build());
}

} // namespace facebook::velox::cudf_velox
