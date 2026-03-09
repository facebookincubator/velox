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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"
#include "velox/expression/FunctionSignature.h"

#include <cudf/groupby.hpp>

#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct Aggregator {
  core::AggregationNode::Step step;
  bool is_global;
  cudf::aggregation::Kind kind;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;

  virtual void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) = 0;

  virtual std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream) = 0;

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) = 0;

  virtual ~Aggregator() = default;

 protected:
  Aggregator(
      core::AggregationNode::Step step,
      cudf::aggregation::Kind kind,
      uint32_t inputIndex,
      VectorPtr constant,
      bool isGlobal,
      const TypePtr& _resultType)
      : step(step),
        is_global(isGlobal),
        kind(kind),
        inputIndex(inputIndex),
        constant(constant),
        resultType(_resultType) {}
};

// Factory functions for creating aggregators from plan nodes.
std::vector<std::unique_ptr<Aggregator>> toAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx);

std::vector<std::unique_ptr<Aggregator>> toIntermediateAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx);

bool hasFinalAggs(
    std::vector<core::AggregationNode::Aggregate> const& aggregates);

void setupGroupingKeyChannelProjections(
    const core::AggregationNode& aggregationNode,
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels);

// Step-aware aggregation function registry
// Map of function name -> Map of step -> signatures
using StepAwareAggregationRegistry = std::unordered_map<
    std::string,
    std::unordered_map<
        core::AggregationNode::Step,
        std::vector<exec::FunctionSignaturePtr>>>;

// Get the step-aware aggregation registry
StepAwareAggregationRegistry& getStepAwareAggregationRegistry();

// Register aggregation function signatures for a specific step
bool registerAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true);

// Register step-aware builtin aggregation functions
bool registerStepAwareBuiltinAggregationFunctions(const std::string& prefix);

// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

// Utility functions
core::TypedExprPtr expandFieldReference(
    const core::TypedExprPtr& expr,
    const core::PlanNode* sourceNode);

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode,
    core::QueryCtx* queryCtx);

bool matchTypedCallAgainstSignatures(
    const core::CallTypedExpr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs);

} // namespace facebook::velox::cudf_velox
