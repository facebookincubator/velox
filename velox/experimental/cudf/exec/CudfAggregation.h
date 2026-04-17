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

#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"
#include "velox/expression/FunctionSignature.h"

#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// \brief Convert companion function to step for the aggregation function
///
/// Companion functions are functions that are registered in velox along with
/// their main aggregation functions. These are designed to always function
/// with a fixed `step`. This is to allow spark style planNodes where `step` is
/// the property of the aggregation function rather than the planNode.
/// Companion functions allow us to override the planNode's step and use
/// aggregations of different steps in the same planNode
/// If an agg function name contains companionStep keyword, may cause error, now
/// it does not exist.
core::AggregationNode::Step getCompanionStep(
    std::string const& kind,
    core::AggregationNode::Step step);

std::string getOriginalName(const std::string& kind);

enum class CountInputKind {
  kColumn,
  kCountAll,
  kNullConstant,
};

CountInputKind getCountInputKind(
    const core::AggregationNode::Aggregate& aggregate,
    const VectorPtr& constant);

// Resolved per-aggregate parameters shared by groupby and reduce aggregator
// construction.  Computed once by resolveAggregateInfo and consumed by the
// type-specific factory in each module.
struct ResolvedAggregateInfo {
  core::AggregationNode::Step companionStep;
  std::string kind;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;
  std::optional<CountInputKind> countInputKind;
};

// Parse aggregate inputs from the aggregation node and resolve companion steps,
// original names, and result types. Returns one entry per aggregate.
std::vector<ResolvedAggregateInfo> resolveAggregateInfos(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants);

// Result of buildAggregationInputChannels: a channel permutation that places
// grouping keys first, followed by aggregate input columns in aggregate order,
// plus per-aggregate constants.
struct AggregationInputChannels {
  std::vector<column_index_t> channels;
  std::vector<VectorPtr> constants;
};

// Build the input channel permutation for an aggregation node.  The returned
// channels vector contains grouping key channels first, then one channel per
// aggregate (resolved from the aggregate's input expression).  Constants are
// stored in the parallel constants vector (nullptr when the aggregate uses a
// column, non-null when it uses a constant).
AggregationInputChannels buildAggregationInputChannels(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx,
    RowTypePtr const& inputRowSchema,
    std::vector<column_index_t> const& groupingKeyInputChannels);

bool isCountFunctionName(std::string_view kind);

bool hasOnlyConstantArguments(const core::CallTypedExpr& call);

// Returns true if the aggregation node contains companion aggregates
// (e.g. _merge, _partial, _merge_extract suffixes), which disables streaming.
bool hasCompanionAggregates(
    std::vector<core::AggregationNode::Aggregate> const& aggregates);

// Compute the intermediate ROW type used for buffered results in kFinal/kSingle
// streaming.  The key columns keep their original types but aggregate columns
// are replaced with the corresponding intermediate types.
RowTypePtr getBufferedResultType(core::AggregationNode const& aggregationNode);

bool hasFinalAggs(
    std::vector<core::AggregationNode::Aggregate> const& aggregates);

void setupGroupingKeyChannelProjections(
    const core::AggregationNode& aggregationNode,
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels);

// Top-level validation: dispatches to groupby or reduce validation based on
// whether the aggregation node is global.
bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

// Utility functions shared by groupby and reduce validation.
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

// Validate a single aggregation call against a given registry.
bool canAggregationBeEvaluatedByRegistry(
    const StepAwareAggregationRegistry& registry,
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

} // namespace facebook::velox::cudf_velox
