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

#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"

#include "velox/exec/RoundRobinPartitionFunction.h"

namespace facebook::velox::cudf_velox {

core::PlanNodePtr CudfPlanRewriter::rewrite(
    const core::PlanNodePtr& root,
    const Config& config) {
  // assuming output is to CPU
  return rewriteNode(root, ExecutionMode::kCpu, config);
}

bool CudfPlanRewriter::canUseGpuAggregation(
    const std::shared_ptr<const core::AggregationNode>& aggNode) {
  if (!aggNode) {
    return false;
  }

  if (!aggNode->isSingle()) {
    // problem with non-single aggs is that there's an exchange already between
    // partial/interm/final. we need to detect that an exchange is between two
    // cudf operators. if so then replace with cudf localpartition plan node and
    // don't add conversions.
    return false;
  }

  // TODO (dm): check with canBeEvaluated

  return true;
}

core::PlanNodePtr CudfPlanRewriter::convertToGpuAggregation(
    const std::shared_ptr<const core::AggregationNode>& aggNode,
    const core::PlanNodePtr& newSource,
    int preferredDriverCount) {
  // Reconstruct AggregationNode with new source
  auto newAggNode = std::make_shared<core::AggregationNode>(
      aggNode->id(),
      aggNode->step(),
      aggNode->groupingKeys(),
      aggNode->preGroupedKeys(),
      aggNode->aggregateNames(),
      aggNode->aggregates(),
      aggNode->ignoreNullKeys(),
      newSource);

  return std::make_shared<CudfAggregationNode>(
      newAggNode, preferredDriverCount);
}

CudfPlanRewriter::ExecutionMode CudfPlanRewriter::determineExecutionMode(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
    return ExecutionMode::kGpu;
  }

  if (auto aggNode =
          std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
    if (canUseGpuAggregation(aggNode)) {
      return ExecutionMode::kGpu;
    }
  }

  return ExecutionMode::kCpu;
}

core::PlanNodePtr CudfPlanRewriter::insertBoundary(
    const core::PlanNodePtr& source,
    ExecutionMode fromMode,
    ExecutionMode toMode,
    int targetPartitions) {
  if (fromMode == toMode || !source) {
    return source;
  }

  auto partitionType = core::LocalPartitionNode::Type::kRepartition;
  auto partitionSpec =
      std::make_shared<exec::RoundRobinPartitionFunctionSpec>();

  auto boundaryId = source->id() + "_boundary_" +
      (fromMode == ExecutionMode::kGpu ? "gpu_to_cpu" : "cpu_to_gpu");

  core::PlanNodePtr result = source;

  // cudftovelox and veloxtocudf conversion only within gpu pipeline
  if (fromMode == ExecutionMode::kGpu && toMode == ExecutionMode::kCpu) {
    // gpu -> cpu
    auto toVeloxId = source->id() + "_to_velox";
    result = std::make_shared<CudfToVeloxNode>(toVeloxId, result);

    result = std::make_shared<core::LocalPartitionNode>(
        boundaryId,
        partitionType,
        false,
        partitionSpec,
        std::vector<core::PlanNodePtr>{result});

  } else if (fromMode == ExecutionMode::kCpu && toMode == ExecutionMode::kGpu) {
    // cpu -> gpu
    auto fromVeloxId = boundaryId + "_from_velox";
    result = std::make_shared<CudfFromVeloxNode>(fromVeloxId, result);

    result = std::make_shared<core::LocalPartitionNode>(
        boundaryId,
        partitionType,
        false,
        partitionSpec,
        std::vector<core::PlanNodePtr>{result});
  }

  return result;
}

core::PlanNodePtr CudfPlanRewriter::rewriteNode(
    const core::PlanNodePtr& node,
    ExecutionMode parentMode,
    const Config& config) {
  if (!node) {
    return node;
  }

  ExecutionMode currentMode = determineExecutionMode(node);

  core::PlanNodePtr resultNode = node;
  bool converted = false;

  if (currentMode == ExecutionMode::kGpu) {
    if (auto aggNode =
            std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
      if (canUseGpuAggregation(aggNode)) {
        // First, recursively process children with GPU mode
        std::vector<core::PlanNodePtr> newSources;
        for (const auto& source : node->sources()) {
          auto rewrittenSource = rewriteNode(source, currentMode, config);

          auto sourceMode = determineExecutionMode(rewrittenSource);
          if (sourceMode != currentMode) {
            int targetPartitions = (currentMode == ExecutionMode::kGpu)
                ? config.gpuDriverCount
                : config.cpuDriverCount;
            rewrittenSource = insertBoundary(
                rewrittenSource, sourceMode, currentMode, targetPartitions);
          }

          newSources.push_back(rewrittenSource);
        }

        resultNode = convertToGpuAggregation(
            aggNode,
            newSources.empty() ? nullptr : newSources[0],
            config.gpuDriverCount);
        converted = true;
      }
    }
  }

  // if not converted, recursively process children
  if (!converted) {
    std::vector<core::PlanNodePtr> newSources;
    for (const auto& source : node->sources()) {
      auto rewrittenSource = rewriteNode(source, currentMode, config);

      auto sourceMode = determineExecutionMode(rewrittenSource);
      if (sourceMode != currentMode) {
        int targetPartitions = (currentMode == ExecutionMode::kGpu)
            ? config.gpuDriverCount
            : config.cpuDriverCount;
        rewrittenSource = insertBoundary(
            rewrittenSource, sourceMode, currentMode, targetPartitions);
      }

      newSources.push_back(rewrittenSource);
    }

    // reconstruct node with new sources.
    // TODO (dm): unsustainable to add one for each plan node type.
    // consider using templates and PlanNode::Builder
    if (!newSources.empty()) {
      if (auto projectNode =
              std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
        resultNode = std::make_shared<core::ProjectNode>(
            projectNode->id(),
            projectNode->names(),
            projectNode->projections(),
            newSources[0]);
      } else if (
          auto filterNode =
              std::dynamic_pointer_cast<const core::FilterNode>(node)) {
        resultNode = std::make_shared<core::FilterNode>(
            filterNode->id(), filterNode->filter(), newSources[0]);
      } else if (
          auto orderByNode =
              std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
        resultNode = std::make_shared<core::OrderByNode>(
            orderByNode->id(),
            orderByNode->sortingKeys(),
            orderByNode->sortingOrders(),
            orderByNode->isPartial(),
            newSources[0]);
      }
    }
  }

  return resultNode;
}

} // namespace facebook::velox::cudf_velox
