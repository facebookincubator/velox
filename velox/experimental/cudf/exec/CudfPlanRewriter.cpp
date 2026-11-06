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
#include "velox/experimental/cudf/exec/CudfPlanNodeConversion.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/CudfScanUtils.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/GpuPlacement.h"
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/core/FixedPointPlanNodes.h"

#include <atomic>
#include <typeinfo>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

namespace {

using Mode = CudfPlanRewriter::ExecutionMode;
std::atomic<uint64_t> nextRewriterId{0};

struct RewriteResult {
  core::PlanNodePtr node;
  Mode mode;
  bool hasOutput{true};
};

// Returns 'node' as a 'T' only when 'T' is its exact type. Rebuilding a
// subclass through the base class Builder would drop the subclass.
template <typename T>
std::shared_ptr<const T> asExactly(const core::PlanNodePtr& node) {
  if (typeid(*node) != typeid(T)) {
    return nullptr;
  }
  return std::static_pointer_cast<const T>(node);
}

template <typename T>
core::PlanNodePtr rebuildWithSource(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& sources) {
  if (auto typed = asExactly<T>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return typename T::Builder(*typed).source(sources[0]).build();
  }
  return nullptr;
}

template <typename T>
core::PlanNodePtr rebuildWithLeftAndRight(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& sources) {
  if (auto typed = asExactly<T>(node)) {
    VELOX_CHECK_EQ(sources.size(), 2);
    return
        typename T::Builder(*typed).left(sources[0]).right(sources[1]).build();
  }
  return nullptr;
}

template <typename T>
core::PlanNodePtr rebuildWithSources(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& sources) {
  if (auto typed = asExactly<T>(node)) {
    return typename T::Builder(*typed).sources(sources).build();
  }
  return nullptr;
}

// Rebuilds a core plan node that stays on the CPU with rewritten sources.
// Returns nullptr for any other node type.
core::PlanNodePtr rebuildCoreNode(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& sources) {
  using Rebuild = core::PlanNodePtr (*)(
      const core::PlanNodePtr&, const std::vector<core::PlanNodePtr>&);
  static const Rebuild kRebuilders[] = {
      rebuildWithSource<core::FilterNode>,
      rebuildWithSource<core::ProjectNode>,
      rebuildWithSource<core::AggregationNode>,
      rebuildWithSource<core::TableWriteNode>,
      rebuildWithSource<core::TableWriteMergeNode>,
      rebuildWithSource<core::ExpandNode>,
      rebuildWithSource<core::GroupIdNode>,
      rebuildWithSource<core::PartitionedOutputNode>,
      rebuildWithSource<core::OrderByNode>,
      rebuildWithSource<core::TopNNode>,
      rebuildWithSource<core::LimitNode>,
      rebuildWithSource<core::UnnestNode>,
      rebuildWithSource<core::EnforceSingleRowNode>,
      rebuildWithSource<core::AssignUniqueIdNode>,
      rebuildWithSource<core::WindowNode>,
      rebuildWithSource<core::RowNumberNode>,
      rebuildWithSource<core::MarkDistinctNode>,
      rebuildWithSource<core::MarkSortedNode>,
      rebuildWithSource<core::TopNRowNumberNode>,
      rebuildWithLeftAndRight<core::HashJoinNode>,
      rebuildWithLeftAndRight<core::MergeJoinNode>,
      rebuildWithLeftAndRight<core::IndexLookupJoinNode>,
      rebuildWithLeftAndRight<core::NestedLoopJoinNode>,
      rebuildWithLeftAndRight<core::SpatialJoinNode>,
      rebuildWithSources<core::LocalMergeNode>,
      rebuildWithSources<core::LocalPartitionNode>,
      rebuildWithSources<core::MixedUnionNode>,
  };
  for (auto rebuild : kRebuilders) {
    if (auto rebuilt = rebuild(node, sources)) {
      return rebuilt;
    }
  }

  // Core nodes without a Builder.
  if (auto lazy = asExactly<core::LazyDereferenceNode>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<core::LazyDereferenceNode>(
        lazy->id(), lazy->names(), lazy->projections(), sources[0]);
  }
  if (auto project = asExactly<core::ParallelProjectNode>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<core::ParallelProjectNode>(
        project->id(),
        project->exprNames(),
        project->exprGroups(),
        project->noLoadIdentities(),
        sources[0]);
  }
  if (auto distinct = asExactly<core::EnforceDistinctNode>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<core::EnforceDistinctNode>(
        distinct->id(),
        distinct->distinctKeys(),
        distinct->preGroupedKeys(),
        distinct->errorMessage(),
        sources[0]);
  }
  if (auto rpc = asExactly<core::RPCNode>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<core::RPCNode>(
        rpc->id(),
        sources[0],
        rpc->call(),
        rpc->outputColumn(),
        rpc->outputType(),
        rpc->streamingMode(),
        rpc->dispatchBatchSize());
  }
  if (auto join = asExactly<core::StateHashJoinNode>(node)) {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<core::StateHashJoinNode>(
        join->id(),
        join->stateName(),
        join->probeKeys(),
        join->outputType(),
        sources[0]);
  }
  return nullptr;
}

bool typesSupported(const core::PlanNodePtr& node, std::string* reason) {
  if (!isTypeSupportedByCudf(node->outputType())) {
    *reason = fmt::format(
        "{} output type is unsupported by cuDF: {}",
        node->name(),
        node->outputType()->toString());
    return false;
  }
  for (const auto& source : node->sources()) {
    if (!isTypeSupportedByCudf(source->outputType())) {
      *reason = fmt::format(
          "{} input type is unsupported by cuDF: {}",
          node->name(),
          source->outputType()->toString());
      return false;
    }
  }
  return true;
}

std::shared_ptr<core::QueryCtx> requireQueryCtx(
    const CudfPlanRewriter::Config& config) {
  VELOX_USER_CHECK_NOT_NULL(
      config.queryCtx, "cuDF planning requires the task's QueryCtx");
  return config.queryCtx;
}

class Rewriter {
 public:
  explicit Rewriter(const CudfPlanRewriter::Config& config)
      : config_(config),
        queryCtx_(requireQueryCtx(config)),
        pool_(queryCtx_->pool()->addLeafChild(
            fmt::format(
                "cudf-plan-rewriter-{}",
                nextRewriterId.fetch_add(1, std::memory_order_relaxed)))) {}

  core::PlanNodePtr rewrite(const core::PlanNodePtr& root) {
    VELOX_USER_CHECK_NOT_NULL(root);
    std::function<void(const core::PlanNodePtr&)> collect =
        [&](const core::PlanNodePtr& node) {
          VELOX_USER_CHECK(
              ids_.insert(node->id()).second,
              "Duplicate plan node ID: {}",
              node->id());
          for (const auto& source : node->sources()) {
            collect(source);
          }
        };
    collect(root);
    auto result = rewriteNode(
        root,
        config_.taskOutput == VectorRepresentation::kCudf ? Mode::kGpu
                                                          : Mode::kCpu);
    validateCudfPlan(result.node, *queryCtx_, config_.taskOutput);
    return result.node;
  }

 private:
  RewriteResult rewriteNode(const core::PlanNodePtr& node, Mode requestedMode) {
    auto result = rewriteImpl(node, requestedMode);
    if (result.hasOutput && result.mode != requestedMode) {
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

    if (auto traits =
            facebook::velox::cudf_velox::detail::customCudfPlanNodeTraits(
                node, *queryCtx_)) {
      VELOX_USER_CHECK_EQ(traits->inputs.size(), node->sources().size());
      VELOX_USER_CHECK(!(traits->output && traits->outputSource));
      std::vector<core::PlanNodePtr> sources;
      std::vector<Mode> modes;
      for (size_t i = 0; i < traits->inputs.size(); ++i) {
        const auto need = traits->inputs[i];
        auto child = need == InputRepresentation::kEither
            ? rewriteImpl(node->sources()[i], requestedMode)
            : rewriteNode(
                  node->sources()[i],
                  need == InputRepresentation::kCudf ? Mode::kGpu : Mode::kCpu);
        VELOX_USER_CHECK(
            child.hasOutput, "A terminal sink cannot feed another node");
        sources.push_back(std::move(child.node));
        modes.push_back(child.mode);
      }
      auto mode = traits->output == VectorRepresentation::kCudf ? Mode::kGpu
                                                                : Mode::kCpu;
      if (traits->outputSource) {
        VELOX_USER_CHECK_LT(*traits->outputSource, modes.size());
        mode = modes[*traits->outputSource];
      }
      return {
          facebook::velox::cudf_velox::detail::rebuildCustomCudfPlanNode(
              node, std::move(sources)),
          mode,
          traits->output.has_value() || traits->outputSource.has_value()};
    }

    if (auto output =
            std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
                node)) {
      const auto inputMode =
          output->transportKind() == core::TransportKind::kUcx ? Mode::kGpu
                                                               : Mode::kCpu;
      auto source = rewriteNode(output->sources()[0], inputMode);
      VELOX_USER_CHECK(source.hasOutput);
      return {
          core::PartitionedOutputNode::Builder(*output)
              .source(source.node)
              .build(),
          inputMode,
          false};
    }

    if (auto exchange =
            std::dynamic_pointer_cast<const core::ExchangeNode>(node)) {
      const bool device =
          exchange->transportKind() == core::TransportKind::kUcx;
      if (auto merge =
              std::dynamic_pointer_cast<const core::MergeExchangeNode>(node);
          device && merge) {
        // Preserve the exchange ID used by remote splits. UCX multiplexes its
        // inputs, so the ordering guarantee is an explicit physical sort.
        auto source = core::ExchangeNode::Builder(*exchange).build();
        return {
            std::make_shared<CudfOrderByNode>(
                newId(node->id() + "_ucx_order"),
                source,
                node->outputType(),
                merge->sortingKeys(),
                merge->sortingOrders(),
                false),
            Mode::kGpu};
      }
      return {node, device ? Mode::kGpu : Mode::kCpu};
    }

    if (auto scan =
            std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
      std::string reason;
      if (isGpuTableScan(*scan, &reason)) {
        return {node, Mode::kGpu};
      }
      if (gpuEnabled()) {
        LOG_FALLBACK("{}, PlanNode id: {}", reason, node->id());
      }
      return {node, Mode::kCpu};
    }

    if (!gpuEnabled()) {
      return rewriteGenericNode(node, "cuDF is disabled for the query");
    }

    // LocalPlanner fuses a Filter followed by a Project into one FilterProject
    // operator, so both go to the GPU together or neither does.
    if (auto project =
            std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
      if (auto filter = std::dynamic_pointer_cast<const core::FilterNode>(
              project->sources()[0])) {
        return rewriteFilterProject(filter, project);
      }
    }

    std::string reason;
    if (!typesSupported(node, &reason)) {
      return rewriteGenericNode(node, reason);
    }
    if (auto result = rewriteOnGpu(node, &reason)) {
      return std::move(*result);
    }
    return rewriteGenericNode(node, reason);
  }

  // Returns the GPU rewrite of 'node', or std::nullopt with the reason it has
  // to stay on the CPU.
  std::optional<RewriteResult> rewriteOnGpu(
      const core::PlanNodePtr& node,
      std::string* reason) {
    if (auto localPartition =
            std::dynamic_pointer_cast<const core::LocalPartitionNode>(node)) {
      // CudfLocalPartitionNode cannot carry scale-writer state.
      if (localPartition->scaleWriter()) {
        *reason = "LocalPartition scale writer is unsupported by cuDF";
        return std::nullopt;
      }
      if (!canRunLocalPartitionOnGpu(localPartition, reason)) {
        return std::nullopt;
      }
      return RewriteResult{
          facebook::velox::cudf_velox::detail::rebuildCudfLocalPartition(
              *localPartition,
              rewriteChildren(localPartition->sources(), Mode::kGpu)),
          Mode::kGpu};
    }

    if (auto aggregation =
            std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
      if (!canRunAggregationOnGpu(
              *aggregation, queryCtx_.get(), pool_.get(), reason)) {
        return std::nullopt;
      }
      auto source = rewriteNode(aggregation->sources()[0], Mode::kGpu).node;
      return toGpu(
          node,
          {addBatchConcat(
              newId(node->id() + "_batch_concat"), std::move(source))});
    }

    if (auto project =
            std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
      if (!canRunFilterProjectOnGpu(
              nullptr, project.get(), queryCtx_.get(), pool_.get(), reason)) {
        return std::nullopt;
      }
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    if (auto filter = std::dynamic_pointer_cast<const core::FilterNode>(node)) {
      if (!canRunFilterProjectOnGpu(
              filter.get(), nullptr, queryCtx_.get(), pool_.get(), reason)) {
        return std::nullopt;
      }
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
      if (!canRunHashJoinOnGpu(*join, queryCtx_.get(), pool_.get(), reason)) {
        return std::nullopt;
      }
      auto probe = rewriteNode(join->sources()[0], Mode::kGpu).node;
      auto build = rewriteNode(join->sources()[1], Mode::kGpu).node;
      return toGpu(
          node,
          {addBatchConcat(
               newId(node->id() + "_batch_concat"), std::move(probe)),
           std::move(build)});
    }

    if (auto join =
            std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
      if (!canRunNestedLoopJoinOnGpu(
              *join, queryCtx_.get(), pool_.get(), reason)) {
        return std::nullopt;
      }
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    if (auto topNRowNumber =
            std::dynamic_pointer_cast<const core::TopNRowNumberNode>(node)) {
      if (!canRunTopNRowNumberOnGpu(*topNRowNumber, reason)) {
        return std::nullopt;
      }
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    if (auto window = std::dynamic_pointer_cast<const core::WindowNode>(node)) {
      if (!CudfWindow::canRunOnGPU(*window, reason)) {
        return std::nullopt;
      }
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    if (node->is<core::OrderByNode>() || node->is<core::TopNNode>() ||
        node->is<core::LimitNode>() || node->is<core::LocalMergeNode>() ||
        node->is<core::AssignUniqueIdNode>() ||
        node->is<core::MarkDistinctNode>() ||
        node->is<core::EnforceSingleRowNode>() ||
        node->is<core::GroupIdNode>()) {
      return toGpu(node, rewriteChildren(node->sources(), Mode::kGpu));
    }

    *reason = fmt::format("{} has no cuDF implementation", node->name());
    return std::nullopt;
  }

  RewriteResult rewriteFilterProject(
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project) {
    // Do not absorb an application-selected implementation into our GPU
    // FilterProject. Keep the Project on CPU and visit the Filter normally so
    // its contract and rebuilding callback determine the input boundaries.
    // For a host Filter, LocalPlanner can still fuse the pair on CPU.
    if (facebook::velox::cudf_velox::detail::customCudfPlanNodeTraits(
            filter, *queryCtx_)) {
      return rewriteGenericNode(
          project, "Filter has an application-provided vector contract");
    }

    std::string reason;
    if (typesSupported(project, &reason) &&
        canRunFilterProjectOnGpu(
            filter.get(),
            project.get(),
            queryCtx_.get(),
            pool_.get(),
            &reason)) {
      return {
          rebuildCudfNode(
              toCudfPlanNode(project),
              rewriteChildren(filter->sources(), Mode::kGpu)),
          Mode::kGpu};
    }
    checkCpuFallback(project, reason);
    auto filterSources = rewriteChildren(filter->sources(), Mode::kCpu);
    if (filterSources == filter->sources()) {
      return {project, Mode::kCpu};
    }
    return {
        rebuildCpuNode(project, {rebuildCpuNode(filter, filterSources)}),
        Mode::kCpu};
  }

  RewriteResult toGpu(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) {
    return {
        rebuildCudfNode(toCudfPlanNode(node), std::move(sources)), Mode::kGpu};
  }

  static core::PlanNodePtr rebuildCudfNode(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) {
    return facebook::velox::cudf_velox::detail::rebuildCustomCudfPlanNode(
        node, std::move(sources));
  }

  core::PlanNodePtr addBatchConcat(
      const core::PlanNodeId& id,
      core::PlanNodePtr source) const {
    if (!CudfConfig::getInstance().concatOptimizationEnabled) {
      return source;
    }
    return std::make_shared<CudfBatchConcatNode>(id, std::move(source));
  }

  RewriteResult rewriteGenericNode(
      const core::PlanNodePtr& node,
      const std::string& reason) {
    if (!node->sources().empty()) {
      checkCpuFallback(node, reason);
    }
    auto newSources = rewriteChildren(node->sources(), Mode::kCpu);
    if (newSources == node->sources()) {
      return {node, Mode::kCpu};
    }
    return {rebuildCpuNode(node, newSources), Mode::kCpu};
  }

  static core::PlanNodePtr rebuildCpuNode(
      const core::PlanNodePtr& node,
      const std::vector<core::PlanNodePtr>& sources) {
    auto rebuilt = rebuildCoreNode(node, sources);
    VELOX_USER_CHECK_NOT_NULL(
        rebuilt,
        "No rebuilding support for {} ({}); register a CudfPlanNodeDescriptor",
        node->name(),
        node->id());
    VELOX_CHECK(rebuilt->sources() == sources);
    return rebuilt;
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

  bool gpuEnabled() const {
    return queryCtx_->queryConfig().get<bool>(
        CudfConfig::kCudfEnabled, CudfConfig::getInstance().enabled);
  }

  void checkCpuFallback(
      const core::PlanNodePtr& node,
      const std::string& reason) const {
    if (!gpuEnabled()) {
      return;
    }
    VELOX_USER_CHECK(
        queryCtx_->queryConfig().get<bool>(
            CudfConfig::kCudfAllowCpuFallback,
            CudfConfig::getInstance().allowCpuFallback),
        "CPU fallback is disabled for unsupported plan node {} ({}): {}",
        node->name(),
        node->id(),
        reason);
    LOG_FALLBACK("{}, PlanNode id: {}", reason, node->id());
  }

  core::PlanNodeId newId(const std::string& prefix) {
    core::PlanNodeId id;
    do {
      id = prefix + "_" + std::to_string(nextId_++);
    } while (!ids_.insert(id).second);
    return id;
  }

  core::PlanNodePtr addBoundary(core::PlanNodePtr node, Mode from, Mode to) {
    if (!node || from == to) {
      return node;
    }
    VELOX_USER_CHECK(
        isTypeSupportedByCudf(node->outputType()),
        "Cannot convert {} ({}) across a cuDF boundary: {}",
        node->name(),
        node->id(),
        node->outputType()->toString());
    if (to == Mode::kCpu) {
      return std::make_shared<CudfToVeloxNode>(
          newId(node->id() + "_to_velox"), std::move(node));
    }
    return std::make_shared<CudfFromVeloxNode>(
        newId(node->id() + "_from_velox"), std::move(node));
  }

  std::unordered_set<core::PlanNodeId> ids_;
  uint64_t nextId_{0};
  const CudfPlanRewriter::Config& config_;
  const std::shared_ptr<core::QueryCtx> queryCtx_;
  const std::shared_ptr<memory::MemoryPool> pool_;
};

} // namespace

core::PlanNodePtr CudfPlanRewriter::rewrite(
    const core::PlanNodePtr& root,
    const Config& config) {
  Rewriter rewriter(config);
  return rewriter.rewrite(root);
}

} // namespace facebook::velox::cudf_velox
