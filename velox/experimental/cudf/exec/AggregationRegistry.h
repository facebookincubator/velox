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

/// Get the step-aware aggregation registry
StepAwareAggregationRegistry& getStepAwareAggregationRegistry();

/// Append a single signature to an existing aggregation function registration.
/// This is a utility function used by Spark and Presto aggregate function
/// registration to add engine-specific signatures.
void appendRegisterAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature);

/// Unregister all aggregation functions from the registry.
void unregisterAggregateFunctions();

/// Unregister aggregation functions whose names start with the given prefix.
void unregisterAggregateFunctionsWithPrefix(const std::string& prefix);

} // namespace facebook::velox::cudf_velox
