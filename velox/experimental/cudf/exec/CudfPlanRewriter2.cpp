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
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter2.h"

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"

namespace facebook::velox::cudf_velox {

namespace {

using Mode = CudfPlanRewriter2::ExecutionMode;

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

class Rewriter {
 public:
  explicit Rewriter(const CudfPlanRewriter2::Config& config)
      : config_(config) {}

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

    if (auto cudfAgg =
            std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
      return rewriteCudfAggregation(cudfAgg);
    }

    if (auto cudfFrom =
            std::dynamic_pointer_cast<const CudfFromVeloxNode>(node)) {
      auto child = rewriteNode(cudfFrom->sources()[0], Mode::kCpu);
      return {
          std::make_shared<CudfFromVeloxNode>(cudfFrom->id(), child.node),
          Mode::kGpu};
    }

    if (auto cudfTo = std::dynamic_pointer_cast<const CudfToVeloxNode>(node)) {
      auto child = rewriteNode(cudfTo->sources()[0], Mode::kGpu);
      return {
          std::make_shared<CudfToVeloxNode>(cudfTo->id(), child.node),
          Mode::kCpu};
    }

    if (auto aggregate =
            std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
      if (canUseGpuAggregation(aggregate)) {
        return rewriteAggregation(aggregate);
      }
    }

    if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
      if (canUseGpuHashJoin(join)) {
        return rewriteHashJoin(join);
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
    auto newAggNode = cloneAggregationNode(aggNode, rewrittenSource.node);
    return {
        std::make_shared<CudfAggregationNode>(
            newAggNode, config_.gpuDriverCount),
        Mode::kGpu};
  }

  RewriteResult rewriteCudfAggregation(
      const std::shared_ptr<const CudfAggregationNode>& node) {
    auto source = node->sources().empty() ? nullptr : node->sources().front();
    auto rewrittenSource = rewriteNode(source, Mode::kGpu);
    auto rebuiltAgg =
        cloneAggregationNode(node->aggregationNode(), rewrittenSource.node);
    return {
        std::make_shared<CudfAggregationNode>(
            rebuiltAgg, node->preferredDriverCount()),
        Mode::kGpu};
  }

  RewriteResult rewriteHashJoin(
      const std::shared_ptr<const core::HashJoinNode>& joinNode) {
    VELOX_CHECK_EQ(joinNode->sources().size(), 2);
    auto probeSource = joinNode->sources()[0];
    auto buildSource = joinNode->sources()[1];

    auto rewrittenProbe = rewriteNode(probeSource, Mode::kGpu);
    auto rewrittenBuild = rewriteNode(buildSource, Mode::kGpu);

    auto rebuiltJoin =
        cloneHashJoinNode(joinNode, rewrittenProbe.node, rewrittenBuild.node);

    return {
        std::make_shared<CudfHashJoinNode>(
            rebuiltJoin, config_.gpuDriverCount, config_.gpuDriverCount),
        Mode::kGpu};
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
      if (auto cloned =
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

  static bool canUseGpuAggregation(
      const std::shared_ptr<const core::AggregationNode>& aggNode) {
    if (!aggNode) {
      return false;
    }
    return true;
  }

  static bool canUseGpuHashJoin(
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

    return true;
  }

  static bool isGpuTableScan(
      const std::shared_ptr<const core::TableScanNode>& tableScan) {
    if (!tableScan) {
      return false;
    }
    auto connectorId = tableScan->tableHandle()->connectorId();
    auto connector = connector::hive::getConnector(connectorId);
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
    return std::make_shared<core::AggregationNode>(
        node->id(),
        node->step(),
        node->groupingKeys(),
        node->preGroupedKeys(),
        node->aggregateNames(),
        node->aggregates(),
        node->ignoreNullKeys(),
        newSource);
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
      return std::make_shared<core::LocalPartitionNode>(
          boundaryId,
          core::LocalPartitionNode::Type::kRepartition,
          false,
          partitionSpec,
          std::vector<core::PlanNodePtr>{converted});
    }

    auto fromVeloxId = boundaryId + "_from_velox";
    auto converted = std::make_shared<CudfFromVeloxNode>(fromVeloxId, node);
    return std::make_shared<core::LocalPartitionNode>(
        boundaryId,
        core::LocalPartitionNode::Type::kRepartition,
        false,
        partitionSpec,
        std::vector<core::PlanNodePtr>{converted});
  }

  const CudfPlanRewriter2::Config& config_;
};

} // namespace

core::PlanNodePtr CudfPlanRewriter2::rewrite(
    const core::PlanNodePtr& root,
    const Config& config) {
  Rewriter rewriter(config);
  return rewriter.rewrite(root);
}

} // namespace facebook::velox::cudf_velox
