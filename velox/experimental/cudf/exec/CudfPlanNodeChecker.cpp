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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeChecker.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/connectors/ConnectorRegistry.h"

#include <fmt/format.h>

namespace facebook::velox::cudf_velox {

CudfNodeSupport isTableScanNodeSupported(
    const core::TableScanNode* tableScanNode) {
  const auto& connector = velox::connector::ConnectorRegistry::tryGet(
      tableScanNode->tableHandle()->connectorId());
  const bool supported =
      std::dynamic_pointer_cast<connector::hive::CudfHiveConnector>(
          connector) != nullptr ||
      std::dynamic_pointer_cast<connector::hive::iceberg::CudfIcebergConnector>(
          connector) != nullptr;
  if (!supported) {
    return {
        false,
        "table scan connector is not a cuDF connector "
        "(CudfHiveConnector or CudfIcebergConnector)"};
  }
  return {true, {}};
}

CudfNodeSupport isFilterNodeSupported(
    const core::FilterNode* filterNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  if (!canExprRunOnGpu(filterNode->filter(), queryCtx, pool)) {
    return {false, "filter expression cannot be evaluated by cuDF"};
  }
  return {true, {}};
}

CudfNodeSupport isProjectNodeSupported(
    const core::ProjectNode* projectNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  // A projection over a source with no columns (e.g. count(*) global) cannot be
  // represented in cuDF, so reject it unless there is nothing to project.
  if (projectNode->sources()[0]->outputType()->size() == 0 &&
      !projectNode->projections().empty()) {
    return {
        false,
        "projection over a source with no columns cannot be represented in cuDF"};
  }
  for (const auto& projection : projectNode->projections()) {
    if (!canExprRunOnGpu(projection, queryCtx, pool)) {
      return {false, "projection expression cannot be evaluated by cuDF"};
    }
  }
  return {true, {}};
}

CudfNodeSupport isHashJoinNodeSupported(
    const core::HashJoinNode* joinNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  if (!CudfHashJoinProbe::isSupportedJoinType(joinNode->joinType())) {
    return {
        false,
        fmt::format(
            "unsupported join type: {}",
            core::JoinTypeName::toName(joinNode->joinType()))};
  }
  // Null-aware anti join with a filter is disabled until it is implemented
  // correctly.
  if (joinNode->joinType() == core::JoinType::kAnti &&
      joinNode->isNullAware() && joinNode->filter()) {
    return {false, "null-aware anti join with filter is not implemented"};
  }
  if (joinNode->filter() &&
      !canExprRunOnGpu(joinNode->filter(), queryCtx, pool)) {
    return {false, "join filter cannot be evaluated by cuDF"};
  }
  return {true, {}};
}

CudfNodeSupport isNestedLoopJoinNodeSupported(
    const core::NestedLoopJoinNode* joinNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  if (!CudfNestedLoopJoinProbe::isSupportedJoinType(joinNode->joinType())) {
    return {
        false,
        fmt::format(
            "unsupported join type: {}",
            core::JoinTypeName::toName(joinNode->joinType()))};
  }
  if (joinNode->joinCondition() &&
      !canExprRunOnGpu(joinNode->joinCondition(), queryCtx, pool)) {
    return {false, "join condition cannot be evaluated by cuDF"};
  }
  return {true, {}};
}

CudfNodeSupport isAggregationNodeSupported(
    const core::AggregationNode* aggregationNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  if (!canBeEvaluatedByCudf(*aggregationNode, queryCtx, pool)) {
    return {false, "aggregation cannot be evaluated by cuDF"};
  }
  return {true, {}};
}

CudfNodeSupport isWindowNodeSupported(const core::WindowNode* windowNode) {
  std::string reason;
  if (!CudfWindow::canRunOnGPU(*windowNode, &reason)) {
    return {
        false,
        reason.empty() ? "window function or frame not supported by cuDF"
                       : reason};
  }
  return {true, {}};
}

CudfNodeSupport isTopNRowNumberNodeSupported(
    const core::TopNRowNumberNode* topNRowNumberNode) {
  if (topNRowNumberNode->rankFunction() !=
      core::TopNRowNumberNode::RankFunction::kRowNumber) {
    return {
        false, "unsupported ranking function (cuDF supports only row_number)"};
  }
  return {true, {}};
}

CudfNodeSupport isLocalPartitionNodeSupported(
    const core::LocalPartitionNode* localPartitionNode) {
  if (!CudfLocalPartition::shouldReplace(*localPartitionNode)) {
    return {false, "unsupported partitioning scheme for cuDF"};
  }
  return {true, {}};
}

} // namespace facebook::velox::cudf_velox
