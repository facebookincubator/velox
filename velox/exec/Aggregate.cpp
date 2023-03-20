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
#include "velox/exec/AggregateWindow.h"
#include "velox/expression/FunctionSignature.h"
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

namespace {
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
} // namespace

bool registerAggregateFunction(
    const std::string& name,
    std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures,
    AggregateFunctionFactory factory) {
  auto sanitizedName = sanitizeName(name);

  aggregateFunctions()[sanitizedName] = {
      std::move(signatures), std::move(factory)};

  // Register the aggregate as a window function also.
  registerAggregateWindowFunction(sanitizedName);
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
