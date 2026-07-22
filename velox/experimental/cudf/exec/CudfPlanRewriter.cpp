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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"

namespace facebook::velox::cudf_velox {

namespace {

using Mode = CudfPlanRewriter::ExecutionMode;

template <typename T>
core::PlanNodePtr cloneWithNewSource(
    const core::PlanNodePtr& node,
    const core::PlanNodePtr& newSource) {
  if (auto typed = std::dynamic_pointer_cast<const T>(node)) {
    return typename T::Builder(*typed).source(newSource).build();
  }
  return nullptr;
}

struct RewriteResult {
  core::PlanNodePtr node;
  Mode mode;
};

std::shared_ptr<const CudfFilterProjectNode> makeFilterProjectNode(
    const std::shared_ptr<const core::FilterNode>& filterNode,
    const std::shared_ptr<const core::ProjectNode>& projectNode,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  VELOX_CHECK(filterNode || projectNode);
  return std::make_shared<CudfFilterProjectNode>(
      projectNode ? projectNode->id() : filterNode->id(),
      std::move(source),
      projectNode ? projectNode->outputType() : filterNode->outputType(),
      filterNode ? std::make_optional(filterNode->id()) : std::nullopt,
      projectNode ? std::make_optional(projectNode->id()) : std::nullopt,
      filterNode ? filterNode->filter() : nullptr,
      projectNode ? projectNode->projections()
                  : std::vector<core::TypedExprPtr>{},
      std::dynamic_pointer_cast<const core::LazyDereferenceNode>(projectNode) !=
          nullptr,
      preferredDriverCount);
}

std::shared_ptr<const CudfAggregationNode> makeAggregationNode(
    const std::shared_ptr<const core::AggregationNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfAggregationNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->step(),
      node->groupingKeys(),
      node->aggregates(),
      node->ignoreNullKeys(),
      preferredDriverCount);
}

std::shared_ptr<const CudfHashJoinNode> makeHashJoinNode(
    const std::shared_ptr<const core::HashJoinNode>& node,
    core::PlanNodePtr probeSource,
    core::PlanNodePtr buildSource,
    int preferredDriverCount) {
  return std::make_shared<CudfHashJoinNode>(
      node->id(),
      std::move(probeSource),
      std::move(buildSource),
      node->outputType(),
      node->joinType(),
      node->isNullAware(),
      node->leftKeys(),
      node->rightKeys(),
      node->filter(),
      preferredDriverCount,
      preferredDriverCount);
}

std::shared_ptr<const CudfNestedLoopJoinNode> makeNestedLoopJoinNode(
    const std::shared_ptr<const core::NestedLoopJoinNode>& node,
    core::PlanNodePtr probeSource,
    core::PlanNodePtr buildSource,
    int preferredDriverCount) {
  return std::make_shared<CudfNestedLoopJoinNode>(
      node->id(),
      std::move(probeSource),
      std::move(buildSource),
      node->outputType(),
      node->joinType(),
      node->joinCondition(),
      preferredDriverCount);
}

std::shared_ptr<const CudfOrderByNode> makeOrderByNode(
    const std::shared_ptr<const core::OrderByNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfOrderByNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->sortingKeys(),
      node->sortingOrders(),
      node->isPartial(),
      preferredDriverCount);
}

std::shared_ptr<const CudfTopNNode> makeTopNNode(
    const std::shared_ptr<const core::TopNNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfTopNNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->sortingKeys(),
      node->sortingOrders(),
      node->count(),
      node->isPartial(),
      preferredDriverCount);
}

std::shared_ptr<const CudfLimitNode> makeLimitNode(
    const std::shared_ptr<const core::LimitNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfLimitNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->offset(),
      node->count(),
      node->isPartial(),
      preferredDriverCount);
}

std::shared_ptr<const CudfAssignUniqueIdNode> makeAssignUniqueIdNode(
    const std::shared_ptr<const core::AssignUniqueIdNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfAssignUniqueIdNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->taskUniqueId(),
      preferredDriverCount);
}

std::shared_ptr<const CudfMarkDistinctNode> makeMarkDistinctNode(
    const std::shared_ptr<const core::MarkDistinctNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfMarkDistinctNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->distinctKeys(),
      preferredDriverCount);
}

std::shared_ptr<const CudfEnforceSingleRowNode> makeEnforceSingleRowNode(
    const std::shared_ptr<const core::EnforceSingleRowNode>& node,
    core::PlanNodePtr source) {
  return std::make_shared<CudfEnforceSingleRowNode>(
      node->id(), std::move(source), node->outputType());
}

std::shared_ptr<const CudfGroupIdNode> makeGroupIdNode(
    const std::shared_ptr<const core::GroupIdNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfGroupIdNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->groupingSets(),
      node->groupingKeyInfos(),
      node->aggregationInputs(),
      node->numGroupingKeys(),
      preferredDriverCount);
}

std::shared_ptr<const CudfWindowNode> makeWindowNode(
    const std::shared_ptr<const core::WindowNode>& node,
    core::PlanNodePtr source,
    int preferredDriverCount) {
  return std::make_shared<CudfWindowNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->partitionKeys(),
      node->sortingKeys(),
      node->sortingOrders(),
      node->windowFunctions(),
      node->inputsSorted(),
      preferredDriverCount);
}

class Rewriter {
 public:
  explicit Rewriter(const CudfPlanRewriter::Config& config)
      : config_(config),
        queryCtx_(
            config.queryCtx ? config.queryCtx : core::QueryCtx::create()) {}

  core::PlanNodePtr rewrite(const core::PlanNodePtr& root) {
    return rewriteNode(root, Mode::kCpu).node;
  }

 private:
  RewriteResult rewriteNode(const core::PlanNodePtr& node, Mode requestedMode) {
    auto result = rewriteImpl(node, requestedMode);
    if (result.mode != requestedMode) {
      result.node =
          addBoundary(std::move(result.node), result.mode, requestedMode);
      result.mode = requestedMode;
    }
    return result;
  }

  RewriteResult rewriteImpl(const core::PlanNodePtr& node, Mode requestedMode) {
    if (!node) {
      return {nullptr, requestedMode};
    }

    if (auto localPartition =
            std::dynamic_pointer_cast<const core::LocalPartitionNode>(node)) {
      return rewriteLocalPartition(localPartition, requestedMode);
    }

    if (auto aggregate =
            std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
      if (canUseGpuAggregation(aggregate)) {
        return rewriteAggregation(aggregate);
      }
    }

    if (auto project =
            std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
      if (canUseGpuProject(project)) {
        return rewriteProject(project);
      }
    }

    if (auto filter = std::dynamic_pointer_cast<const core::FilterNode>(node)) {
      if (canUseGpuFilter(filter)) {
        return rewriteFilter(filter);
      }
    }

    if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
      if (canUseGpuHashJoin(join)) {
        return rewriteHashJoin(join);
      }
    }

    if (auto join =
            std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
      if (canUseGpuNestedLoopJoin(join)) {
        return rewriteNestedLoopJoin(join);
      }
    }

    if (auto orderBy =
            std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
      return rewriteUnaryNode(orderBy, makeOrderByNode);
    }

    if (auto topN = std::dynamic_pointer_cast<const core::TopNNode>(node)) {
      return rewriteUnaryNode(topN, makeTopNNode);
    }

    if (auto limit = std::dynamic_pointer_cast<const core::LimitNode>(node)) {
      return rewriteUnaryNode(limit, makeLimitNode);
    }

    if (auto assignUniqueId =
            std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(node)) {
      return rewriteUnaryNode(assignUniqueId, makeAssignUniqueIdNode);
    }

    if (auto markDistinct =
            std::dynamic_pointer_cast<const core::MarkDistinctNode>(node)) {
      return rewriteUnaryNode(markDistinct, makeMarkDistinctNode);
    }

    if (auto enforceSingleRow =
            std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(node)) {
      return rewriteEnforceSingleRow(enforceSingleRow);
    }

    if (auto groupId =
            std::dynamic_pointer_cast<const core::GroupIdNode>(node)) {
      return rewriteUnaryNode(groupId, makeGroupIdNode);
    }

    if (auto window = std::dynamic_pointer_cast<const core::WindowNode>(node)) {
      if (CudfWindow::canRunOnGPU(*window)) {
        return rewriteUnaryNode(window, makeWindowNode);
      }
    }

    if (auto scan =
            std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
      if (isGpuTableScan(scan)) {
        return {scan, Mode::kGpu};
      }
    }

    return rewriteGenericNode(node);
  }

  RewriteResult rewriteAggregation(
      const std::shared_ptr<const core::AggregationNode>& aggNode) {
    auto source = aggNode->sources().empty() ? nullptr : aggNode->sources()[0];
    auto rewrittenSource = rewriteNode(source, Mode::kGpu);
    auto gpuSource = addBatchConcat(
        aggNode->id() + "_batch_concat", std::move(rewrittenSource.node));
    return {
        makeAggregationNode(
            aggNode, std::move(gpuSource), config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteProject(
      const std::shared_ptr<const core::ProjectNode>& projectNode) {
    VELOX_CHECK_EQ(projectNode->sources().size(), 1);

    if (auto filterNode = std::dynamic_pointer_cast<const core::FilterNode>(
            projectNode->sources()[0]);
        filterNode && canUseGpuFilter(filterNode)) {
      VELOX_CHECK_EQ(filterNode->sources().size(), 1);
      auto rewrittenSource = rewriteNode(filterNode->sources()[0], Mode::kGpu);
      return {
          makeFilterProjectNode(
              filterNode,
              projectNode,
              rewrittenSource.node,
              config_.gpuDriverCount),
          Mode::kGpu};
    }

    auto rewrittenSource = rewriteNode(projectNode->sources()[0], Mode::kGpu);
    return {
        makeFilterProjectNode(
            nullptr, projectNode, rewrittenSource.node, config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteFilter(
      const std::shared_ptr<const core::FilterNode>& filterNode) {
    VELOX_CHECK_EQ(filterNode->sources().size(), 1);
    auto rewrittenSource = rewriteNode(filterNode->sources()[0], Mode::kGpu);
    return {
        makeFilterProjectNode(
            filterNode, nullptr, rewrittenSource.node, config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteHashJoin(
      const std::shared_ptr<const core::HashJoinNode>& joinNode) {
    VELOX_CHECK_EQ(joinNode->sources().size(), 2);
    auto probeSource = joinNode->sources()[0];
    auto buildSource = joinNode->sources()[1];

    auto rewrittenProbe = rewriteNode(probeSource, Mode::kGpu);
    auto rewrittenBuild = rewriteNode(buildSource, Mode::kGpu);
    auto gpuProbe = addBatchConcat(
        joinNode->id() + "_batch_concat", std::move(rewrittenProbe.node));

    return {
        makeHashJoinNode(
            joinNode,
            std::move(gpuProbe),
            rewrittenBuild.node,
            config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteNestedLoopJoin(
      const std::shared_ptr<const core::NestedLoopJoinNode>& joinNode) {
    VELOX_CHECK_EQ(joinNode->sources().size(), 2);
    auto rewrittenProbe = rewriteNode(joinNode->sources()[0], Mode::kGpu);
    auto rewrittenBuild = rewriteNode(joinNode->sources()[1], Mode::kGpu);
    return {
        makeNestedLoopJoinNode(
            joinNode,
            rewrittenProbe.node,
            rewrittenBuild.node,
            config_.gpuDriverCount),
        Mode::kGpu};
  }

  template <typename CoreNode, typename Factory>
  RewriteResult rewriteUnaryNode(
      const std::shared_ptr<const CoreNode>& node,
      Factory factory) {
    VELOX_CHECK_EQ(node->sources().size(), 1);
    auto rewrittenSource = rewriteNode(node->sources()[0], Mode::kGpu);
    return {
        factory(node, rewrittenSource.node, config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteEnforceSingleRow(
      const std::shared_ptr<const core::EnforceSingleRowNode>& node) {
    VELOX_CHECK_EQ(node->sources().size(), 1);
    auto rewrittenSource = rewriteNode(node->sources()[0], Mode::kGpu);
    return {
        makeEnforceSingleRowNode(node, std::move(rewrittenSource.node)),
        Mode::kGpu};
  }

  core::PlanNodePtr addBatchConcat(
      const core::PlanNodeId& id,
      core::PlanNodePtr source) const {
    if (!CudfConfig::getInstance().concatOptimizationEnabled) {
      return source;
    }
    return std::make_shared<CudfBatchConcatNode>(
        id, std::move(source), config_.gpuDriverCount);
  }

  RewriteResult rewriteLocalPartition(
      const std::shared_ptr<const core::LocalPartitionNode>& node,
      Mode requestedMode) {
    const bool allowGpu = requestedMode == Mode::kGpu &&
        (node->type() == core::LocalPartitionNode::Type::kGather ||
         isHashPartition(node));
    const Mode childMode = allowGpu ? Mode::kGpu : Mode::kCpu;
    auto newSources = rewriteChildren(node->sources(), childMode);
    auto builder = core::LocalPartitionNode::Builder(*node);
    builder.sources(newSources);
    return {builder.build(), childMode};
  }

  RewriteResult rewriteGenericNode(const core::PlanNodePtr& node) {
    auto newSources = rewriteChildren(node->sources(), Mode::kCpu);
    if (newSources.empty()) {
      return {node, Mode::kCpu};
    }

    core::PlanNodePtr rebuilt = node;
    if (newSources.size() == 1) {
      if (auto aggregation =
              std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
        rebuilt = cloneAggregationNode(aggregation, newSources[0]);
      } else if (
          auto cloned =
              cloneWithNewSource<core::ProjectNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::FilterNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::OrderByNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::LimitNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::TopNNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned = cloneWithNewSource<core::EnforceSingleRowNode>(
              node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned = cloneWithNewSource<core::AssignUniqueIdNode>(
              node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::MarkDistinctNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::GroupIdNode>(node, newSources[0])) {
        rebuilt = cloned;
      } else if (
          auto cloned =
              cloneWithNewSource<core::WindowNode>(node, newSources[0])) {
        rebuilt = cloned;
      }
    } else if (newSources.size() == 2) {
      if (auto hashJoin =
              std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
        rebuilt = cloneHashJoinNode(hashJoin, newSources[0], newSources[1]);
      } else if (
          auto nestedLoopJoin =
              std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
        rebuilt = cloneNestedLoopJoinNode(
            nestedLoopJoin, newSources[0], newSources[1]);
      }
    }

    return {rebuilt, Mode::kCpu};
  }

  std::vector<core::PlanNodePtr> rewriteChildren(
      const std::vector<core::PlanNodePtr>& sources,
      Mode consumerMode) {
    std::vector<core::PlanNodePtr> result;
    result.reserve(sources.size());
    for (const auto& source : sources) {
      result.push_back(rewriteNode(source, consumerMode).node);
    }
    return result;
  }

  bool canUseGpuProject(
      const std::shared_ptr<const core::ProjectNode>& projectNode) const {
    if (!projectNode) {
      return false;
    }
    VELOX_CHECK_EQ(projectNode->sources().size(), 1);
    if (projectNode->sources()[0]->outputType()->size() == 0 &&
        !projectNode->projections().empty()) {
      return false;
    }
    return canBeEvaluatedByCudf(projectNode->projections(), queryCtx_.get());
  }

  bool canUseGpuFilter(
      const std::shared_ptr<const core::FilterNode>& filterNode) const {
    return filterNode &&
        canBeEvaluatedByCudf({filterNode->filter()}, queryCtx_.get());
  }

  bool canUseGpuAggregation(
      const std::shared_ptr<const core::AggregationNode>& aggNode) {
    return aggNode && canBeEvaluatedByCudf(*aggNode, queryCtx_.get());
  }

  bool canUseGpuHashJoin(
      const std::shared_ptr<const core::HashJoinNode>& joinNode) {
    if (!joinNode) {
      return false;
    }

    if (!CudfHashJoinProbe::isSupportedJoinType(joinNode->joinType())) {
      return false;
    }

    if (joinNode->joinType() == core::JoinType::kAnti &&
        joinNode->isNullAware() && joinNode->filter()) {
      return false;
    }

    if (joinNode->filter() &&
        !canBeEvaluatedByCudf({joinNode->filter()}, queryCtx_.get())) {
      return false;
    }

    return true;
  }

  bool canUseGpuNestedLoopJoin(
      const std::shared_ptr<const core::NestedLoopJoinNode>& joinNode) const {
    if (!joinNode ||
        !CudfNestedLoopJoinProbe::isSupportedJoinType(joinNode->joinType())) {
      return false;
    }
    return !joinNode->joinCondition() ||
        canBeEvaluatedByCudf({joinNode->joinCondition()}, queryCtx_.get());
  }

  static bool isGpuTableScan(
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
    return dynamic_cast<facebook::velox::cudf_velox::connector::hive::
                            CudfHiveConnector*>(connector.get()) != nullptr;
  }

  static bool isHashPartition(
      const std::shared_ptr<const core::LocalPartitionNode>& node) {
    if (!node || node->type() != core::LocalPartitionNode::Type::kRepartition) {
      return false;
    }
    const auto& spec = node->partitionFunctionSpec();
    return dynamic_cast<const exec::HashPartitionFunctionSpec*>(&spec) !=
        nullptr;
  }

  static std::shared_ptr<const core::AggregationNode> cloneAggregationNode(
      const std::shared_ptr<const core::AggregationNode>& node,
      const core::PlanNodePtr& newSource) {
    return core::AggregationNode::Builder(*node).source(newSource).build();
  }

  static std::shared_ptr<const core::HashJoinNode> cloneHashJoinNode(
      const std::shared_ptr<const core::HashJoinNode>& node,
      const core::PlanNodePtr& leftSource,
      const core::PlanNodePtr& rightSource) {
    core::HashJoinNode::Builder builder(*node);
    builder.left(leftSource);
    builder.right(rightSource);
    return builder.build();
  }

  static std::shared_ptr<const core::NestedLoopJoinNode>
  cloneNestedLoopJoinNode(
      const std::shared_ptr<const core::NestedLoopJoinNode>& node,
      const core::PlanNodePtr& leftSource,
      const core::PlanNodePtr& rightSource) {
    core::NestedLoopJoinNode::Builder builder(*node);
    builder.left(leftSource);
    builder.right(rightSource);
    return builder.build();
  }

  core::PlanNodePtr addBoundary(core::PlanNodePtr node, Mode from, Mode to) {
    if (!node || from == to) {
      return node;
    }

    auto partitionSpec =
        std::make_shared<exec::RoundRobinPartitionFunctionSpec>();
    auto boundaryId = node->id() + "_boundary_" +
        (from == Mode::kGpu ? "gpu_to_cpu" : "cpu_to_gpu");

    if (from == Mode::kGpu && to == Mode::kCpu) {
      auto toVeloxId = node->id() + "_to_velox";
      auto converted = std::make_shared<CudfToVeloxNode>(toVeloxId, node);
      if (!config_.addLocalPartitionBoundaries) {
        return converted;
      }
      return std::make_shared<core::LocalPartitionNode>(
          boundaryId,
          core::LocalPartitionNode::Type::kRepartition,
          false,
          partitionSpec,
          std::vector<core::PlanNodePtr>{converted});
    }

    auto fromVeloxId = boundaryId + "_from_velox";
    auto converted = std::make_shared<CudfFromVeloxNode>(fromVeloxId, node);
    if (!config_.addLocalPartitionBoundaries) {
      return converted;
    }
    return std::make_shared<core::LocalPartitionNode>(
        boundaryId,
        core::LocalPartitionNode::Type::kRepartition,
        false,
        partitionSpec,
        std::vector<core::PlanNodePtr>{converted});
  }

  const CudfPlanRewriter::Config& config_;
  const std::shared_ptr<core::QueryCtx> queryCtx_;
};

} // namespace

core::PlanNodePtr CudfPlanRewriter::rewrite(
    const core::PlanNodePtr& root,
    const Config& config) {
  Rewriter rewriter(config);
  return rewriter.rewrite(root);
}

core::PlanNodePtr CudfPlanRewriter::translateForAdapter(
    const core::PlanNodePtr& node,
    int gpuDriverCount) {
  VELOX_CHECK_NOT_NULL(node);

  if (auto aggregation =
          std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
    VELOX_CHECK_EQ(aggregation->sources().size(), 1);
    return makeAggregationNode(
        aggregation, aggregation->sources()[0], gpuDriverCount);
  }

  if (auto project = std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
    VELOX_CHECK_EQ(project->sources().size(), 1);
    auto filter = std::dynamic_pointer_cast<const core::FilterNode>(
        project->sources()[0]);
    if (filter) {
      VELOX_CHECK_EQ(filter->sources().size(), 1);
      return makeFilterProjectNode(
          filter, project, filter->sources()[0], gpuDriverCount);
    }
    return makeFilterProjectNode(
        nullptr, project, project->sources()[0], gpuDriverCount);
  }

  if (auto filter = std::dynamic_pointer_cast<const core::FilterNode>(node)) {
    VELOX_CHECK_EQ(filter->sources().size(), 1);
    return makeFilterProjectNode(
        filter, nullptr, filter->sources()[0], gpuDriverCount);
  }

  if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    VELOX_CHECK_EQ(join->sources().size(), 2);
    return makeHashJoinNode(
        join, join->sources()[0], join->sources()[1], gpuDriverCount);
  }

  if (auto join =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    VELOX_CHECK_EQ(join->sources().size(), 2);
    return makeNestedLoopJoinNode(
        join, join->sources()[0], join->sources()[1], gpuDriverCount);
  }

  if (auto orderBy = std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
    VELOX_CHECK_EQ(orderBy->sources().size(), 1);
    return makeOrderByNode(orderBy, orderBy->sources()[0], gpuDriverCount);
  }

  if (auto topN = std::dynamic_pointer_cast<const core::TopNNode>(node)) {
    VELOX_CHECK_EQ(topN->sources().size(), 1);
    return makeTopNNode(topN, topN->sources()[0], gpuDriverCount);
  }

  if (auto limit = std::dynamic_pointer_cast<const core::LimitNode>(node)) {
    VELOX_CHECK_EQ(limit->sources().size(), 1);
    return makeLimitNode(limit, limit->sources()[0], gpuDriverCount);
  }

  if (auto assignUniqueId =
          std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(node)) {
    VELOX_CHECK_EQ(assignUniqueId->sources().size(), 1);
    return makeAssignUniqueIdNode(
        assignUniqueId, assignUniqueId->sources()[0], gpuDriverCount);
  }

  if (auto markDistinct =
          std::dynamic_pointer_cast<const core::MarkDistinctNode>(node)) {
    VELOX_CHECK_EQ(markDistinct->sources().size(), 1);
    return makeMarkDistinctNode(
        markDistinct, markDistinct->sources()[0], gpuDriverCount);
  }

  if (auto enforceSingleRow =
          std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(node)) {
    VELOX_CHECK_EQ(enforceSingleRow->sources().size(), 1);
    return makeEnforceSingleRowNode(
        enforceSingleRow, enforceSingleRow->sources()[0]);
  }

  if (auto groupId = std::dynamic_pointer_cast<const core::GroupIdNode>(node)) {
    VELOX_CHECK_EQ(groupId->sources().size(), 1);
    return makeGroupIdNode(groupId, groupId->sources()[0], gpuDriverCount);
  }

  if (auto window = std::dynamic_pointer_cast<const core::WindowNode>(node)) {
    VELOX_CHECK_EQ(window->sources().size(), 1);
    return makeWindowNode(window, window->sources()[0], gpuDriverCount);
  }

  VELOX_FAIL("No cuDF PlanNode translation for {}", node->name());
}

core::PlanNodePtr CudfPlanRewriter::translateBatchConcatForAdapter(
    const core::PlanNodePtr& node,
    int gpuDriverCount) {
  VELOX_CHECK_NOT_NULL(node);
  if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    VELOX_CHECK_EQ(join->sources().size(), 2);
    return std::make_shared<CudfBatchConcatNode>(
        node->id() + "_batch_concat", join->sources()[0], gpuDriverCount);
  }
  VELOX_CHECK_EQ(
      node->sources().size(),
      1,
      "CudfBatchConcat expects a single-source plan node");
  return std::make_shared<CudfBatchConcatNode>(
      node->id() + "_batch_concat", node->sources()[0], gpuDriverCount);
}

} // namespace facebook::velox::cudf_velox
