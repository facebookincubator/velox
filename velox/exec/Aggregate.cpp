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

#include "velox/exec/Aggregate.h"

#include <unordered_map>
#include "velox/exec/AggregateFunctionAdapter.h"
#include "velox/exec/AggregateFunctionAdapterUtil.h"
#include "velox/exec/AggregateWindow.h"
#include "velox/expression/SignatureBinder.h"

namespace facebook::velox::exec {

bool isRawInput(core::AggregationNode::Step step) {
  return step == core::AggregationNode::Step::kPartial ||
      step == core::AggregationNode::Step::kSingle;
}

bool isPartialOutput(core::AggregationNode::Step step) {
  return step == core::AggregationNode::Step::kPartial ||
      step == core::AggregationNode::Step::kIntermediate;
}

AggregateFunctionMap& aggregateFunctions() {
  static AggregateFunctionMap functions;
  return functions;
}

std::optional<const AggregateFunctionEntry*> getAggregateFunctionEntry(
    const std::string& name) {
  auto sanitizedName = sanitizeName(name);

  auto& functionsMap = aggregateFunctions();
  auto it = functionsMap.find(sanitizedName);
  if (it != functionsMap.end()) {
    return &it->second;
  }

  return std::nullopt;
}

bool registerAggregateFunction(
    const std::string& name,
    std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures,
    AggregateFunctionFactory factory,
    AggregateFunctionMetadata metadata,
    bool registerCompanionFunctions) {
  auto sanitizedName = sanitizeName(name);

  aggregateFunctions()[sanitizedName] = {
      signatures, metadata, std::move(factory)};

  // Register the aggregate as a window function also.
  registerAggregateWindowFunction(sanitizedName);

  // Register companion function if needed.
  if (registerCompanionFunctions) {
    RegisterAdapter::registerPartialFunction(name, signatures);
    RegisterAdapter::registerMergeFunction(name, signatures);
    RegisterAdapter::registerExtractFunction(name, signatures);
    if (metadata.supportsRetract) {
      RegisterAdapter::registerRetractFunction(name, signatures);
    }
  }

  return true;
}

std::unordered_map<
    std::string,
    std::vector<std::shared_ptr<AggregateFunctionSignature>>>
getAggregateFunctionSignatures() {
  std::unordered_map<
      std::string,
      std::vector<std::shared_ptr<AggregateFunctionSignature>>>
      map;
  auto aggregateFunctions = exec::aggregateFunctions();
  for (const auto& aggregateFunction : aggregateFunctions) {
    map[aggregateFunction.first] = aggregateFunction.second.signatures;
  }
  return map;
}

std::optional<std::vector<std::shared_ptr<AggregateFunctionSignature>>>
getAggregateFunctionSignatures(const std::string& name) {
  if (auto func = getAggregateFunctionEntry(name)) {
    return func.value()->signatures;
  }

  return std::nullopt;
}

std::optional<
    std::unordered_map<CompanionType, std::vector<CompanionSignatureEntry>>>
getCompanionFunctionSignatures(const std::string& name) {
  auto entry = getAggregateFunctionEntry(name);
  if (!entry.has_value()) {
    return std::nullopt;
  }

  auto signatures = entry.value()->signatures;
  std::unordered_map<CompanionType, std::vector<CompanionSignatureEntry>>
      companionSignatures;

  auto partialSignatures =
      CompanionSignatures::partialFunctionSignatures(signatures);
  companionSignatures.emplace(
      CompanionType::kPartial,
      std::vector<CompanionSignatureEntry>{
          {CompanionSignatures::partialFunctionName(name),
           std::vector<FunctionSignaturePtr>{
               partialSignatures.begin(), partialSignatures.end()}}});

  auto mergeSignatures =
      CompanionSignatures::mergeFunctionSignatures(signatures);
  companionSignatures.emplace(
      CompanionType::kMerge,
      std::vector<CompanionSignatureEntry>{
          {CompanionSignatures::mergeFunctionName(name),
           std::vector<FunctionSignaturePtr>{
               mergeSignatures.begin(), mergeSignatures.end()}}});

  if (entry.value()->metadata.supportsRetract) {
    companionSignatures.emplace(
        CompanionType::kRetract,
        std::vector<CompanionSignatureEntry>{
            {CompanionSignatures::retractFunctionName(name),
             CompanionSignatures::retractFunctionSignatures(signatures)}});
  }

  if (CompanionSignatures::hasSameIntermediateTypesAcrossSignatures(
          signatures)) {
    std::vector<CompanionSignatureEntry> entries;
    for (const auto& signature : signatures) {
      entries.push_back(
          {CompanionSignatures::extractFunctionNameWithSuffix(
               name, signature->returnType()),
           {CompanionSignatures::extractFunctionSignature(signature)}});
    }
    companionSignatures.emplace(CompanionType::kExtract, std::move(entries));
  } else {
    companionSignatures.emplace(
        CompanionType::kExtract,
        std::vector<CompanionSignatureEntry>{
            {CompanionSignatures::extractFunctionName(name),
             CompanionSignatures::extractFunctionSignatures(signatures)}});
  }
  return companionSignatures;
}

std::unique_ptr<Aggregate> Aggregate::create(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  // Lookup the function in the new registry first.
  if (auto func = getAggregateFunctionEntry(name)) {
    return func.value()->factory(step, argTypes, resultType);
  }

  VELOX_USER_FAIL("Aggregate function not registered: {}", name);
}

// static
TypePtr Aggregate::intermediateType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  auto signatures = getAggregateFunctionSignatures(name);
  if (!signatures.has_value()) {
    VELOX_FAIL("Aggregate {} not registered", name);
  }
  for (auto& signature : signatures.value()) {
    SignatureBinder binder(*signature, argTypes);
    if (binder.tryBind()) {
      auto type = binder.tryResolveType(signature->intermediateType());
      VELOX_CHECK(type, "failed to resolve intermediate type for {}", name);
      return type;
    }
  }
  VELOX_FAIL("Could not infer intermediate type for aggregate {}", name);
}

int32_t Aggregate::combineAlignment(int32_t otherAlignment) const {
  auto thisAlignment = accumulatorAlignmentSize();
  VELOX_CHECK_EQ(
      __builtin_popcount(thisAlignment), 1, "Alignment can only be power of 2");
  VELOX_CHECK_EQ(
      __builtin_popcount(otherAlignment),
      1,
      "Alignment can only be power of 2");
  return std::max(thisAlignment, otherAlignment);
}

} // namespace facebook::velox::exec
