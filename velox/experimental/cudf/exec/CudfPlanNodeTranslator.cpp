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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfDistinct.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfLimit.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeTranslator.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/connectors/ConnectorRegistry.h"

namespace facebook::velox::cudf_velox {
namespace {

bool isGpuTableScan(
    const std::shared_ptr<const core::TableScanNode>& tableScan) {
  if (!tableScan) {
    return false;
  }

  const auto connectorId = tableScan->tableHandle()->connectorId();
  auto connector =
      facebook::velox::connector::ConnectorRegistry::tryGet(connectorId);
  if (!connector) {
    return false;
  }

  return dynamic_cast<
             facebook::velox::cudf_velox::connector::hive::CudfHiveConnector*>(
             connector.get()) != nullptr;
}

} // namespace

std::unique_ptr<exec::Operator> CudfPlanNodeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto fromVelox =
          std::dynamic_pointer_cast<const CudfFromVeloxNode>(node)) {
    return std::make_unique<CudfFromVelox>(
        id, fromVelox->outputType(), ctx, fromVelox->id());
  }

  if (auto toVelox = std::dynamic_pointer_cast<const CudfToVeloxNode>(node)) {
    return std::make_unique<CudfToVelox>(
        id, toVelox->outputType(), ctx, toVelox->id());
  }

  if (auto filterProject =
          std::dynamic_pointer_cast<const CudfFilterProjectNode>(node)) {
    return std::make_unique<CudfFilterProject>(
        id, ctx, filterProject->filterNode(), filterProject->projectNode());
  }

  if (auto gpuAgg =
          std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
    const auto& aggregationNode = gpuAgg->aggregationNode();
    const bool isGlobal = aggregationNode->groupingKeys().empty();
    const bool isDistinct = !isGlobal && aggregationNode->aggregates().empty();
    if (isGlobal) {
      return std::make_unique<CudfReduce>(id, ctx, aggregationNode);
    }
    if (isDistinct) {
      return std::make_unique<CudfDistinct>(id, ctx, aggregationNode);
    }
    return std::make_unique<CudfGroupby>(id, ctx, aggregationNode);
  }

  if (auto orderBy = std::dynamic_pointer_cast<const CudfOrderByNode>(node)) {
    return std::make_unique<CudfOrderBy>(id, ctx, orderBy->planNode());
  }

  if (auto topN = std::dynamic_pointer_cast<const CudfTopNNode>(node)) {
    return std::make_unique<CudfTopN>(id, ctx, topN->planNode());
  }

  if (auto limit = std::dynamic_pointer_cast<const CudfLimitNode>(node)) {
    return std::make_unique<CudfLimit>(id, ctx, limit->planNode());
  }

  return nullptr;
}

std::optional<uint32_t> CudfPlanNodeTranslator::maxDrivers(
    const core::PlanNodePtr& node) {
  if (auto gpuAgg =
          std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
    return gpuAgg->preferredDriverCount();
  }

  if (auto gpuJoin = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
    return gpuJoin->preferredProbeDriverCount();
  }

  if (auto filterProject =
          std::dynamic_pointer_cast<const CudfFilterProjectNode>(node)) {
    return filterProject->preferredDriverCount();
  }

  if (auto nestedLoopJoin =
          std::dynamic_pointer_cast<const CudfNestedLoopJoinNode>(node)) {
    return nestedLoopJoin->preferredDriverCount();
  }

  if (auto orderBy = std::dynamic_pointer_cast<const CudfOrderByNode>(node)) {
    return orderBy->preferredDriverCount();
  }

  if (auto topN = std::dynamic_pointer_cast<const CudfTopNNode>(node)) {
    return topN->preferredDriverCount();
  }

  if (auto limit = std::dynamic_pointer_cast<const CudfLimitNode>(node)) {
    return limit->preferredDriverCount();
  }

  if (auto tableScan =
          std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
    if (isGpuTableScan(tableScan)) {
      return 1;
    }
  }

  return std::nullopt;
}

} // namespace facebook::velox::cudf_velox
