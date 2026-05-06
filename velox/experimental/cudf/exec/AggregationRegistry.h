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

#pragma once

#include "velox/core/PlanNode.h"
#include "velox/expression/FunctionSignature.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

// Step-aware aggregation function registry
// Map of function name -> Map of step -> signatures
using StepAwareAggregationRegistry = std::unordered_map<
    std::string,
    std::unordered_map<
        core::AggregationNode::Step,
        std::vector<exec::FunctionSignaturePtr>>>;

/// Runtime aggregation registries keyed by physical execution kind.
StepAwareAggregationRegistry& getGroupbyAggregationRegistry();
StepAwareAggregationRegistry& getReduceAggregationRegistry();

/// Shared registration helpers used to populate a target physical registry.
bool registerAggregationFunctionForStep(
    StepAwareAggregationRegistry& registry,
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true);

/// Shared common aggregation registration used by engine-specific builders.
void registerCommonAggregationFunctions(
    StepAwareAggregationRegistry& registry,
    const std::string& prefix);

/// Shared reduce-only registration used by engine-specific builders.
void registerReduceOnlyAggregationFunctions(
    StepAwareAggregationRegistry& registry,
    const std::string& prefix);

/// Append a single signature to the current groupby aggregation registry.
void appendGroupbyAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature);

/// Append a single signature to the current reduce aggregation registry.
void appendReduceAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature);

/// Clear all physical aggregation registries.
void unregisterAggregateFunctions();

} // namespace facebook::velox::cudf_velox
