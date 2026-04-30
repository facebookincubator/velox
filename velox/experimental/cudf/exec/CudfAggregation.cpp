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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"

#include "velox/core/Expressions.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/expression/SignatureBinder.h"

#include <algorithm>
#include <numeric>

namespace facebook::velox::cudf_velox {

bool isCountFunctionName(std::string_view kind) {
  auto prefix = CudfConfig::getInstance().functionNamePrefix;
  return kind.rfind(prefix + "count", 0) == 0;
}

CountInputKind getCountInputKind(
    const core::AggregationNode::Aggregate& aggregate,
    const VectorPtr& constant) {
  if (aggregate.call->inputs().empty()) {
    return CountInputKind::kCountAll;
  }
  if (constant != nullptr) {
    return constant->isNullAt(0) ? CountInputKind::kNullConstant
                                 : CountInputKind::kCountAll;
  }
  return CountInputKind::kColumn;
}

bool hasOnlyConstantArguments(const core::CallTypedExpr& call) {
  return !call.inputs().empty() &&
      std::all_of(
          call.inputs().begin(), call.inputs().end(), [](const auto& arg) {
            return dynamic_cast<const core::ConstantTypedExpr*>(arg.get()) !=
                nullptr;
          });
}

core::AggregationNode::Step getCompanionStep(
    std::string const& kind,
    core::AggregationNode::Step step) {
  if (kind.ends_with("_merge")) {
    return core::AggregationNode::Step::kIntermediate;
  }

  if (kind.ends_with("_partial")) {
    return core::AggregationNode::Step::kPartial;
  }

  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (kind.find("_merge_extract") != std::string::npos) {
    return core::AggregationNode::Step::kFinal;
  }

  return step;
}

std::string getOriginalName(const std::string& kind) {
  if (kind.ends_with("_merge")) {
    return kind.substr(0, kind.size() - std::string("_merge").size());
  }

  if (kind.ends_with("_partial")) {
    return kind.substr(0, kind.size() - std::string("_partial").size());
  }
  // The format is count_merge_extract_BIGINT or count_merge_extract.
  if (auto pos = kind.find("_merge_extract"); pos != std::string::npos) {
    return kind.substr(0, pos);
  }

  return kind;
}

namespace {
bool isCompanionAggregateName(std::string const& kind) {
  return kind.ends_with("_merge") || kind.ends_with("_partial") ||
      kind.find("_merge_extract") != std::string::npos;
}

bool isSupportedZeroColumnAggregation(
    const core::AggregationNode& aggregationNode) {
  // Zero-column input: only global prefixed `count` aggregates (same as
  // createAggregator). No GROUP BY keys.
  return aggregationNode.groupingKeys().empty() &&
      !aggregationNode.aggregates().empty() &&
      std::all_of(
             aggregationNode.aggregates().begin(),
             aggregationNode.aggregates().end(),
             [](const auto& aggregate) {
               return isCountFunctionName(aggregate.call->name());
             });
}
} // namespace

bool hasCompanionAggregates(
    std::vector<core::AggregationNode::Aggregate> const& aggregates) {
  return std::any_of(aggregates.begin(), aggregates.end(), [](auto const& agg) {
    return isCompanionAggregateName(agg.call->name());
  });
}

std::vector<ResolvedAggregateInfo> resolveAggregateInfos(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants) {
  const auto numKeys = aggregationNode.groupingKeys().size();

  std::vector<ResolvedAggregateInfo> params;
  params.reserve(aggregationNode.aggregates().size());
  for (size_t i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    auto const companionStep = getCompanionStep(aggregate.call->name(), step);
    const auto originalName = getOriginalName(aggregate.call->name());
    const auto resultType = exec::isPartialOutput(companionStep)
        ? exec::resolveIntermediateType(originalName, aggregate.rawInputTypes)
        : outputType->childAt(numKeys + i);

    params.emplace_back(
        companionStep,
        aggregate.call->name(),
        static_cast<uint32_t>(numKeys + i),
        constants[i],
        resultType,
        isCountFunctionName(aggregate.call->name())
            ? std::make_optional(getCountInputKind(aggregate, constants[i]))
            : std::nullopt);
  }
  return params;
}

AggregationInputChannels buildAggregationInputChannels(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx,
    RowTypePtr const& inputRowSchema,
    std::vector<column_index_t> const& groupingKeyInputChannels) {
  AggregationInputChannels result;
  result.constants.resize(aggregationNode.aggregates().size());
  result.channels.reserve(
      groupingKeyInputChannels.size() + aggregationNode.aggregates().size());
  result.channels.insert(
      result.channels.end(),
      groupingKeyInputChannels.begin(),
      groupingKeyInputChannels.end());

  const auto fallbackChannel =
      groupingKeyInputChannels.empty() ? 0 : groupingKeyInputChannels.front();

  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    std::vector<column_index_t> aggInputs;
    for (auto const& arg : aggregate.call->inputs()) {
      if (auto const field =
              dynamic_cast<core::FieldAccessTypedExpr const*>(arg.get())) {
        aggInputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else if (
          auto constant =
              dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
        result.constants[i] = constant->toConstantVector(operatorCtx.pool());
        aggInputs.push_back(fallbackChannel);
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }

    VELOX_CHECK(aggInputs.size() <= 1);
    if (aggInputs.empty()) {
      aggInputs.push_back(fallbackChannel);
    }

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    result.channels.push_back(aggInputs[0]);
  }

  return result;
}

RowTypePtr getBufferedResultType(core::AggregationNode const& aggregationNode) {
  const auto outputRowType = asRowType(aggregationNode.outputType());
  const auto numKeys = aggregationNode.groupingKeys().size();

  std::vector<std::string> names = outputRowType->names();
  std::vector<TypePtr> types = outputRowType->children();

  VELOX_CHECK_EQ(names.size(), types.size());
  VELOX_CHECK_GE(types.size(), numKeys + aggregationNode.aggregates().size());

  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    const auto originalName = getOriginalName(aggregate.call->name());
    types[numKeys + i] =
        exec::resolveIntermediateType(originalName, aggregate.rawInputTypes);
  }

  return ROW(std::move(names), std::move(types));
}

bool hasFinalAggs(
    std::vector<core::AggregationNode::Aggregate> const& aggregates) {
  return std::any_of(aggregates.begin(), aggregates.end(), [](auto const& agg) {
    return agg.call->name().ends_with("_merge_extract");
  });
}

void setupGroupingKeyChannelProjections(
    const core::AggregationNode& aggregationNode,
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  auto const& inputType = aggregationNode.sources()[0]->outputType();
  auto const& groupingKeys = aggregationNode.groupingKeys();
  // The map from the grouping key output channel to the input channel.
  //
  // NOTE: grouping key output order is specified as 'groupingKeys' in
  // 'aggregationNode_'.
  std::vector<exec::IdentityProjection> groupingKeyProjections;
  groupingKeyProjections.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyProjections.emplace_back(
        exec::exprToChannel(groupingKeys[i].get(), inputType), i);
  }

  groupingKeyInputChannels.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyInputChannels.push_back(groupingKeyProjections[i].inputChannel);
  }

  groupingKeyOutputChannels.resize(groupingKeys.size());

  std::iota(
      groupingKeyOutputChannels.begin(), groupingKeyOutputChannels.end(), 0);
}

bool matchTypedCallAgainstSignatures(
    const core::CallTypedExpr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call.inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& input : call.inputs()) {
    argTypes.push_back(input->type());
  }
  for (const auto& sig : sigs) {
    std::vector<Coercion> coercions(n);
    exec::SignatureBinder binder(*sig, argTypes);
    if (!binder.tryBindWithCoercions(coercions)) {
      continue;
    }

    // For simplicity we skip checking for constant agruments, this may be added
    // in the future

    return true;
  }
  return false;
}

bool canAggregationBeEvaluatedByRegistry(
    const StepAwareAggregationRegistry& registry,
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  // Check against step-aware aggregation registry
  const auto companionStep = getCompanionStep(call.name(), step);
  const auto originalName = getOriginalName(call.name());
  auto funcIt = registry.find(originalName);
  if (funcIt == registry.end()) {
    return false;
  }

  auto stepIt = funcIt->second.find(companionStep);
  if (stepIt == funcIt->second.end()) {
    return false;
  }

  if (isCountFunctionName(call.name())) {
    return true;
  }

  if (hasOnlyConstantArguments(call)) {
    return false;
  }

  // Validate against step-specific signatures from registry.
  return matchTypedCallAgainstSignatures(call, stepIt->second);
}

bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  if (sourceNode && sourceNode->outputType()->size() == 0 &&
      !isSupportedZeroColumnAggregation(aggregationNode)) {
    return false;
  }

  bool isGlobal = aggregationNode.groupingKeys().empty();
  bool isDistinct = !isGlobal && aggregationNode.aggregates().empty();

  if (isDistinct) {
    return canGroupingKeysBeEvaluatedByCudf(
        aggregationNode.groupingKeys(), sourceNode, queryCtx);
  } else if (isGlobal) {
    return canReduceBeEvaluatedByCudf(aggregationNode, queryCtx);
  } else {
    return canGroupbyBeEvaluatedByCudf(aggregationNode, queryCtx);
  }
}

core::TypedExprPtr expandFieldReference(
    const core::TypedExprPtr& expr,
    const core::PlanNode* sourceNode) {
  // If this is a field reference and we have a source projection, expand it
  if (expr->kind() == core::ExprKind::kFieldAccess && sourceNode) {
    auto projectNode = dynamic_cast<const core::ProjectNode*>(sourceNode);
    if (projectNode) {
      auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr);
      if (fieldExpr) {
        // Find the corresponding projection expression
        const auto& projections = projectNode->projections();
        const auto& names = projectNode->names();
        for (size_t i = 0; i < names.size(); ++i) {
          if (names[i] == fieldExpr->name()) {
            return projections[i];
          }
        }
      }
    }
  }
  return expr;
}

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode,
    core::QueryCtx* queryCtx) {
  // Check grouping key expressions (with expansion)
  for (const auto& groupingKey : groupingKeys) {
    auto expandedKey = expandFieldReference(groupingKey, sourceNode);
    std::vector<core::TypedExprPtr> exprs = {expandedKey};
    if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
      return false;
    }
  }

  return true;
}

} // namespace facebook::velox::cudf_velox
