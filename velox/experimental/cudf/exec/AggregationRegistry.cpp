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

#include <algorithm>
#include <optional>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

namespace {

void appendAggregationFunctionForStep(
    StepAwareAggregationRegistry& registry,
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  registry[name][step].push_back(signature);
}

std::string makeSignatureKey(const exec::FunctionSignaturePtr& signature) {
  std::string key = signature->variableArity() ? "variadic|" : "fixed|";
  const auto& argumentTypes = signature->argumentTypes();
  const auto& constantArguments = signature->constantArguments();
  for (size_t i = 0; i < argumentTypes.size(); ++i) {
    const bool isConstantArg =
        i < constantArguments.size() && constantArguments[i];
    key.append(isConstantArg ? "constant:" : "argument:");
    key.append(argumentTypes[i].toString());
    key.push_back('|');
  }
  return key;
}

exec::AggregateFunctionSignaturePtr buildAggregateSignature(
    const exec::FunctionSignaturePtr& singleSignature,
    const exec::FunctionSignaturePtr& partialSignature) {
  exec::AggregateFunctionSignatureBuilder builder;

  // Preserve declared signature variables.
  for (const auto& [_, variable] : singleSignature->variables()) {
    if (variable.isTypeParameter()) {
      if (variable.knownTypesOnly()) {
        builder.knownTypeVariable(variable.name());
      } else if (variable.orderableTypesOnly()) {
        builder.orderableTypeVariable(variable.name());
      } else if (variable.comparableTypesOnly()) {
        builder.comparableTypeVariable(variable.name());
      } else {
        builder.typeVariable(variable.name());
      }
    } else if (variable.isIntegerParameter()) {
      builder.integerVariable(
          variable.name(),
          variable.constraint().empty()
              ? std::nullopt
              : std::make_optional(variable.constraint()));
    }
  }

  builder.returnType(singleSignature->returnType().toString());
  builder.intermediateType(partialSignature->returnType().toString());

  const auto& argumentTypes = singleSignature->argumentTypes();
  const auto& constantArguments = singleSignature->constantArguments();
  for (size_t argIndex = 0; argIndex < argumentTypes.size(); ++argIndex) {
    const auto argType = argumentTypes[argIndex].toString();
    const bool isConstantArg =
        argIndex < constantArguments.size() && constantArguments[argIndex];
    if (isConstantArg) {
      builder.constantArgumentType(argType);
    } else {
      builder.argumentType(argType);
    }
  }

  if (singleSignature->variableArity()) {
    builder.variableArity();
  }

  return builder.build();
}

void appendAggregateFunctionSignatures(
    const StepAwareAggregationRegistry& registry,
    exec::AggregateFunctionSignatureMap& result) {
  for (const auto& [name, stepMap] : registry) {
    const auto singleIt = stepMap.find(core::AggregationNode::Step::kSingle);
    const auto partialIt = stepMap.find(core::AggregationNode::Step::kPartial);

    // We need both single (for return type) and partial (for intermediate
    // type) signatures to build AggregateFunctionSignature entries.
    if (singleIt == stepMap.end() || partialIt == stepMap.end()) {
      continue;
    }

    const auto& singleSignatures = singleIt->second;
    const auto& partialSignatures = partialIt->second;
    if (singleSignatures.empty() || partialSignatures.empty()) {
      continue;
    }

    std::unordered_map<std::string, exec::FunctionSignaturePtr>
        singleSignatureIndex;
    for (const auto& signature : singleSignatures) {
      singleSignatureIndex.emplace(makeSignatureKey(signature), signature);
    }

    std::unordered_map<std::string, exec::FunctionSignaturePtr>
        partialSignatureIndex;
    for (const auto& signature : partialSignatures) {
      partialSignatureIndex.emplace(makeSignatureKey(signature), signature);
    }

    auto& aggregateSignatures = result[name];
    std::unordered_set<std::string> existingSignatures;
    for (const auto& signature : aggregateSignatures) {
      existingSignatures.insert(signature->toString());
    }

    for (const auto& [key, singleSignature] : singleSignatureIndex) {
      auto partialMatchIt = partialSignatureIndex.find(key);
      if (partialMatchIt == partialSignatureIndex.end()) {
        continue;
      }

      auto aggregateSignature =
          buildAggregateSignature(singleSignature, partialMatchIt->second);
      if (existingSignatures.insert(aggregateSignature->toString()).second) {
        aggregateSignatures.push_back(std::move(aggregateSignature));
      }
    }

    if (aggregateSignatures.empty()) {
      result.erase(name);
    }
  }
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

exec::AggregateFunctionSignatureMap getCudfAggregationFunctionSignatureMap() {
  exec::AggregateFunctionSignatureMap result;
  appendAggregateFunctionSignatures(getGroupbyAggregationRegistry(), result);
  appendAggregateFunctionSignatures(getReduceAggregationRegistry(), result);
  return result;
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
