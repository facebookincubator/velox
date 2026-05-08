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

namespace facebook::velox::cudf_velox {

namespace {

void appendAggregationFunctionForStep(
    StepAwareAggregationRegistry& registry,
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  registry[name][step].push_back(signature);
}

} // namespace

StepAwareAggregationRegistry& getGroupbyAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

StepAwareAggregationRegistry& getReduceAggregationRegistry() {
  static StepAwareAggregationRegistry registry;
  return registry;
}

bool registerAggregationFunctionForStep(
    StepAwareAggregationRegistry& registry,
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  if (!overwrite && registry.find(name) != registry.end() &&
      registry[name].find(step) != registry[name].end()) {
    return false;
  }

  registry[name][step] = signatures;
  return true;
}

void registerCommonAggregationFunctions(
    StepAwareAggregationRegistry& registry,
    const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  auto sumSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  registerAggregationFunctionForStep(
      registry,
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      sumSingleSignatures);

  auto sumPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      sumPartialSignatures);

  auto sumFinalIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  registerAggregationFunctionForStep(
      registry,
      prefix + "sum",
      core::AggregationNode::Step::kFinal,
      sumFinalIntermediateSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "sum",
      core::AggregationNode::Step::kIntermediate,
      sumFinalIntermediateSignatures);

  auto countSinglePartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("boolean")
          .build(),
      FunctionSignatureBuilder().returnType("bigint").build()};

  registerAggregationFunctionForStep(
      registry,
      prefix + "count",
      core::AggregationNode::Step::kSingle,
      countSinglePartialSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "count",
      core::AggregationNode::Step::kPartial,
      countSinglePartialSignatures);

  auto countFinalIntermediateSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("bigint")
                                                  .argumentType("bigint")
                                                  .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "count",
      core::AggregationNode::Step::kFinal,
      countFinalIntermediateSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "count",
      core::AggregationNode::Step::kIntermediate,
      countFinalIntermediateSignatures);

  auto minMaxSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("tinyint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("smallint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .build()};

  registerAggregationFunctionForStep(
      registry,
      prefix + "min",
      core::AggregationNode::Step::kSingle,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "min",
      core::AggregationNode::Step::kPartial,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "min",
      core::AggregationNode::Step::kFinal,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "min",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  registerAggregationFunctionForStep(
      registry,
      prefix + "max",
      core::AggregationNode::Step::kSingle,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "max",
      core::AggregationNode::Step::kPartial,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "max",
      core::AggregationNode::Step::kFinal,
      minMaxSignatures);
  registerAggregationFunctionForStep(
      registry,
      prefix + "max",
      core::AggregationNode::Step::kIntermediate,
      minMaxSignatures);

  auto avgSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .build()};

  registerAggregationFunctionForStep(
      registry,
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      avgSingleSignatures);

  auto avgPartialSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("double")
          .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "avg",
      core::AggregationNode::Step::kPartial,
      avgPartialSignatures);

  auto avgFinalSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("row(double,bigint)")
          .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "avg",
      core::AggregationNode::Step::kFinal,
      avgFinalSignatures);

  auto avgIntermediateSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("row(double,bigint)")
          .argumentType("row(double,bigint)")
          .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "avg",
      core::AggregationNode::Step::kIntermediate,
      avgIntermediateSignatures);
}

void registerReduceOnlyAggregationFunctions(
    StepAwareAggregationRegistry& registry,
    const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  auto approxDistinctSingleSignatures = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("tinyint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("smallint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("real")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("varbinary")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("timestamp")
          .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "approx_distinct",
      core::AggregationNode::Step::kSingle,
      approxDistinctSingleSignatures);

  auto approxDistinctPartialSignatures =
      std::vector<exec::FunctionSignaturePtr>{
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("tinyint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("smallint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("integer")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("bigint")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("real")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("double")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("varchar")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("varbinary")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("date")
              .build(),
          FunctionSignatureBuilder()
              .returnType("varbinary")
              .argumentType("timestamp")
              .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "approx_distinct",
      core::AggregationNode::Step::kPartial,
      approxDistinctPartialSignatures);

  auto approxDistinctIntermediateSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("varbinary")
                                                  .argumentType("varbinary")
                                                  .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "approx_distinct",
      core::AggregationNode::Step::kIntermediate,
      approxDistinctIntermediateSignatures);

  auto approxDistinctFinalSignatures =
      std::vector<exec::FunctionSignaturePtr>{FunctionSignatureBuilder()
                                                  .returnType("bigint")
                                                  .argumentType("varbinary")
                                                  .build()};
  registerAggregationFunctionForStep(
      registry,
      prefix + "approx_distinct",
      core::AggregationNode::Step::kFinal,
      approxDistinctFinalSignatures);
}

void appendGroupbyAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  appendAggregationFunctionForStep(
      getGroupbyAggregationRegistry(), name, step, signature);
}

void appendReduceAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  appendAggregationFunctionForStep(
      getReduceAggregationRegistry(), name, step, signature);
}

void unregisterAggregateFunctions() {
  getGroupbyAggregationRegistry().clear();
  getReduceAggregationRegistry().clear();
}

} // namespace facebook::velox::cudf_velox
