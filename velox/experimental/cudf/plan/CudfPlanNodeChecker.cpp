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
/*
 * Implementation for plan-node-level CUDF evaluation checks.
 */

#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/plan/CudfExpressionChecker.h"
#include "velox/experimental/cudf/plan/CudfPlanNodeChecker.h"

#include "velox/common/memory/Memory.h"
#include "velox/connectors/Connector.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/TypeCoercer.h"

namespace facebook::velox::cudf_velox {

namespace {

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

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode) {
  // Check grouping key expressions (with expansion)
  for (const auto& groupingKey : groupingKeys) {
    auto expandedKey = expandFieldReference(groupingKey, sourceNode);
    std::vector<core::TypedExprPtr> exprs = {expandedKey};
    if (!canBeEvaluatedByCudf(exprs)) {
      return false;
    }
  }

  return true;
}
} // namespace

bool isTableScanNodeSupported(const core::TableScanNode* tableScanNode) {
  auto const& connector = velox::connector::getConnector(
      tableScanNode->tableHandle()->connectorId());
  auto cudfHiveConnector = std::dynamic_pointer_cast<
      facebook::velox::cudf_velox::connector::hive::CudfHiveConnector>(
      connector);
  return cudfHiveConnector != nullptr;
}

bool isFilterNodeSupported(const core::FilterNode* filterNode) {
  return canBeEvaluatedByCudf(
      std::vector<velox::core::TypedExprPtr>{filterNode->filter()});
}

bool isProjectNodeSupported(const core::ProjectNode* projectNode) {
  // Check that source and output types are not empty
  if (projectNode->sources()[0]->outputType()->size() == 0 ||
      projectNode->outputType()->size() == 0) {
    return false;
  }

  return canBeEvaluatedByCudf(projectNode->projections());
}

// TODO: Fix tests for this helper function so it is not exposed outside of the
//   aggregation node plan checker.
// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step) {
  // Check against step-aware aggregation registry
  auto& stepAwareRegistry = getStepAwareAggregationRegistry();
  auto funcIt = stepAwareRegistry.find(call.name());
  if (funcIt == stepAwareRegistry.end()) {
    return false;
  }

  auto stepIt = funcIt->second.find(step);
  if (stepIt == funcIt->second.end()) {
    return false;
  }

  // Validate against step-specific signatures from registry
  return matchTypedCallAgainstSignatures(call, stepIt->second);
}

bool isAggregationNodeSupported(const core::AggregationNode* aggregationNode) {
  if (aggregationNode->sources()[0]->outputType()->size() == 0) {
    // We cannot handle RowVectors with a length but no data.
    // This is the case with count(*) global (without groupby)
    return false;
  }

  const core::PlanNode* sourceNode = aggregationNode->sources().empty()
      ? nullptr
      : aggregationNode->sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode->step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode->aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canAggregationBeEvaluatedByCudf(*aggregate.call, step)) {
      return false;
    }

    if (aggregate.distinct) {
      return false;
    }

    if (aggregate.mask) {
      return false;
    }

    // Check input expressions can be evaluated by CUDF, expand the input
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      std::vector<core::TypedExprPtr> exprs = {expandedInput};
      if (!canBeEvaluatedByCudf(exprs)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode->groupingKeys(), sourceNode)) {
    return false;
  }

  return true;
}

bool isHashJoinNodeSupported(const core::HashJoinNode* joinNode) {
  if (!joinNode) {
    return false;
  }

  if (!CudfHashJoinProbe::isSupportedJoinType(joinNode->joinType())) {
    return false;
  }

  // disabling null-aware anti join with filter until we implement it right
  if (joinNode->joinType() == core::JoinType::kAnti and
      joinNode->isNullAware() and joinNode->filter()) {
    return false;
  }

  if (joinNode->filter()) {
    if (!canBeEvaluatedByCudf(
            std::vector<velox::core::TypedExprPtr>{joinNode->filter()})) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::cudf_velox
