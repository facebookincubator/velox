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

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/expression/SignatureBinder.h"

#include <numeric>

namespace facebook::velox::cudf_velox {

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

namespace {
void appendToRegistry(
    StepAwareAggregationRegistry& registry,
    const std::string& name,
    core::AggregationNode::Step step,
    const exec::FunctionSignaturePtr& signature) {
  registry[name][step].push_back(signature);
}
} // namespace

void registerCommonAggregationFunctions(
    StepAwareAggregationRegistry& registry,
    const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // ===== Register sum function (split by aggregation step) =====
  // Common signatures for integer and double types (same for Spark and Presto).
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

  // Register count function (split by aggregation step)
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

  // Register min function (same signatures for all steps)
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

  // Register max function (same signatures for all steps)
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

  // Register avg function (different signatures for different steps)

  // Single step: avg(input_type) -> double
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

  // Final step: avg(row(double, bigint)) -> double
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

  // Intermediate step: avg(row(sum input_type, count bigint)) -> row(sum
  // input_type, count bigint)
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

  // ===== Engine-specific signatures for SUM(REAL) and AVG(REAL) =====
  // SUM(REAL):
  //   Spark:  single REAL->DOUBLE, partial REAL->DOUBLE,
  //           final/intermediate DOUBLE->DOUBLE (already covered above)
  //   Presto: single REAL->REAL,   partial REAL->DOUBLE,
  //           final/intermediate DOUBLE->REAL
  // AVG(REAL):
  //   Spark:  single REAL->DOUBLE,
  //           final row(DOUBLE,BIGINT)->DOUBLE (already covered above)
  //   Presto: single REAL->REAL,
  //           final row(DOUBLE,BIGINT)->REAL
  // AVG partial REAL->row(DOUBLE,BIGINT) and intermediate are the same for
  // both engines and are already registered above.

  if (CudfConfig::getInstance().functionEngine == "spark") {
    // Spark: SUM(REAL) -> DOUBLE, AVG(REAL) -> DOUBLE
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kSingle,
        FunctionSignatureBuilder()
            .returnType("double")
            .argumentType("real")
            .build());
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kPartial,
        FunctionSignatureBuilder()
            .returnType("double")
            .argumentType("real")
            .build());
    // SUM final/intermediate: DOUBLE->DOUBLE already registered.

    appendToRegistry(
        registry,
        prefix + "avg",
        core::AggregationNode::Step::kSingle,
        FunctionSignatureBuilder()
            .returnType("double")
            .argumentType("real")
            .build());
    // AVG final: row(DOUBLE,BIGINT)->DOUBLE already registered.
  } else {
    // Presto (default): SUM(REAL) -> REAL, AVG(REAL) -> REAL
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kSingle,
        FunctionSignatureBuilder()
            .returnType("real")
            .argumentType("real")
            .build());
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kPartial,
        FunctionSignatureBuilder()
            .returnType("double")
            .argumentType("real")
            .build());
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kFinal,
        FunctionSignatureBuilder()
            .returnType("real")
            .argumentType("double")
            .build());
    appendToRegistry(
        registry,
        prefix + "sum",
        core::AggregationNode::Step::kIntermediate,
        FunctionSignatureBuilder()
            .returnType("real")
            .argumentType("double")
            .build());

    appendToRegistry(
        registry,
        prefix + "avg",
        core::AggregationNode::Step::kSingle,
        FunctionSignatureBuilder()
            .returnType("real")
            .argumentType("real")
            .build());
    appendToRegistry(
        registry,
        prefix + "avg",
        core::AggregationNode::Step::kFinal,
        FunctionSignatureBuilder()
            .returnType("real")
            .argumentType("row(double,bigint)")
            .build());
  }
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

std::vector<ResolvedAggregateInfo> resolveAggregateInputs(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto const step = aggregationNode.step();
  auto const& inputRowSchema = aggregationNode.sources()[0]->outputType();
  const auto numKeys = aggregationNode.groupingKeys().size();
  const auto outputType = aggregationNode.outputType();

  std::vector<ResolvedAggregateInfo> result;
  for (auto i = 0; i < aggregationNode.aggregates().size(); ++i) {
    auto const& aggregate = aggregationNode.aggregates()[i];
    std::vector<column_index_t> aggInputs;
    std::vector<VectorPtr> aggConstants;
    for (auto const& arg : aggregate.call->inputs()) {
      if (auto const field =
              dynamic_cast<core::FieldAccessTypedExpr const*>(arg.get())) {
        aggInputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else if (
          auto constant =
              dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
        aggInputs.push_back(kConstantChannel);
        aggConstants.push_back(constant->toConstantVector(operatorCtx.pool()));
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }
    // The loop on aggregate.call->inputs() is taken from
    // AggregateInfo.cpp::toAggregateInfo(). It seems to suggest that there can
    // be multiple inputs to an aggregate.
    // We're postponing properly supporting this for now because the currently
    // supported aggregation functions in cudf_velox don't use it.
    VELOX_CHECK(aggInputs.size() <= 1);
    if (aggInputs.empty()) {
      aggInputs.push_back(0);
    }

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    auto const kind = aggregate.call->name();
    auto const inputIndex = aggInputs[0];
    auto const constant = aggConstants.empty() ? nullptr : aggConstants[0];
    auto const companionStep = getCompanionStep(kind, step);
    const auto originalName = getOriginalName(kind);
    const auto resultType = exec::isPartialOutput(companionStep)
        ? exec::resolveIntermediateType(originalName, aggregate.rawInputTypes)
        : outputType->childAt(numKeys + i);

    result.push_back({companionStep, kind, inputIndex, constant, resultType});
  }
  return result;
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

  // Validate against step-specific signatures from registry
  return matchTypedCallAgainstSignatures(call, stepIt->second);
}

bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  bool isGlobal = aggregationNode.groupingKeys().empty();
  bool isDistinct = !isGlobal && aggregationNode.aggregates().empty();

  if (isDistinct) {
    const core::PlanNode* sourceNode = aggregationNode.sources().empty()
        ? nullptr
        : aggregationNode.sources()[0].get();
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
