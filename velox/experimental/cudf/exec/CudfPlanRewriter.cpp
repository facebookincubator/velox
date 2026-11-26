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
#include "velox/experimental/cudf/exec/CudfPlanNodeDefinitions.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
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

struct RewriteResult {
  core::PlanNodePtr node;
  Mode mode;
  bool hasOutput{true};
};

struct RebuildContext : public core::PlanNodeVisitorContext {
  const core::PlanNodePtr& node;
  const std::vector<core::PlanNodePtr>& sources;
  core::PlanNodePtr rebuilt;

  RebuildContext(
      const core::PlanNodePtr& nodeIn,
      const std::vector<core::PlanNodePtr>& sourcesIn)
      : node(nodeIn), sources(sourcesIn) {}
};

// Rebuilds a core plan node that stays on the CPU with rewritten sources.
// Deriving from PlanNodeVisitor makes adding a new core PlanNode a compile
// error here rather than a runtime failure. Each overload verifies exact
// dynamic type so subclasses are not silently sliced to their base class.
class CoreNodeRebuilder : public core::PlanNodeVisitor {
 public:
  void visit(
      const core::AggregationNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::ArrowStreamNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::AssignUniqueIdNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::EnforceSingleRowNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::ExchangeNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::ExpandNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::FilterNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::GroupIdNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::HashJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithLeftAndRight(node, context);
  }

  void visit(
      const core::IndexLookupJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithLeftAndRight(node, context);
  }

  void visit(const core::LimitNode& node, core::PlanNodeVisitorContext& context)
      const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::LocalMergeNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSources(node, context);
  }

  void visit(
      const core::LocalPartitionNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSources(node, context);
  }

  void visit(
      const core::MarkDistinctNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::EnforceDistinctNode& node,
      core::PlanNodeVisitorContext& context) const override {
    if (!isExactly<core::EnforceDistinctNode>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 1);
    ctx.rebuilt = std::make_shared<core::EnforceDistinctNode>(
        node.id(),
        node.distinctKeys(),
        node.preGroupedKeys(),
        node.errorMessage(),
        ctx.sources[0]);
  }

  void visit(
      const core::MarkSortedNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::MergeExchangeNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::MergeJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithLeftAndRight(node, context);
  }

  void visit(
      const core::NestedLoopJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithLeftAndRight(node, context);
  }

  void visit(
      const core::SpatialJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithLeftAndRight(node, context);
  }

  void visit(
      const core::VectorSearchJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    if (!isExactly<core::VectorSearchJoinNode>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 2);
    ctx.rebuilt = std::make_shared<core::VectorSearchJoinNode>(
        node.id(),
        node.joinType(),
        node.leftKeys(),
        node.rightKeys(),
        node.indexSide(),
        node.embedding(),
        node.index(),
        node.format(),
        node.metric(),
        node.k(),
        node.searchOptions(),
        node.idName(),
        node.scoreName(),
        ctx.sources[0],
        ctx.sources[1],
        node.outputType());
  }

  void visit(
      const core::OrderByNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::PartitionedOutputNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::ProjectNode& node,
      core::PlanNodeVisitorContext& context) const override {
    if (isExactly<core::LazyDereferenceNode>(node)) {
      auto& ctx = static_cast<RebuildContext&>(context);
      VELOX_CHECK_EQ(ctx.sources.size(), 1);
      const auto& lazy = static_cast<const core::LazyDereferenceNode&>(node);
      ctx.rebuilt = std::make_shared<core::LazyDereferenceNode>(
          lazy.id(), lazy.names(), lazy.projections(), ctx.sources[0]);
      return;
    }
    rebuildWithSource(node, context);
  }

  void visit(
      const core::ParallelProjectNode& node,
      core::PlanNodeVisitorContext& context) const override {
    if (!isExactly<core::ParallelProjectNode>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 1);
    ctx.rebuilt = std::make_shared<core::ParallelProjectNode>(
        node.id(),
        node.exprNames(),
        node.exprGroups(),
        node.noLoadIdentities(),
        ctx.sources[0]);
  }

  void visit(
      const core::RowNumberNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::TableScanNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::TableWriteNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::TableWriteMergeNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(const core::TopNNode& node, core::PlanNodeVisitorContext& context)
      const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::TopNRowNumberNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::TraceScanNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::UnnestNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::ValuesNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::WindowNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSource(node, context);
  }

  void visit(
      const core::MixedUnionNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildWithSources(node, context);
  }

  void visit(
      const core::FixedPointNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::StateSourceNode& node,
      core::PlanNodeVisitorContext& context) const override {
    rebuildLeaf(node, context);
  }

  void visit(
      const core::StateHashJoinNode& node,
      core::PlanNodeVisitorContext& context) const override {
    if (!isExactly<core::StateHashJoinNode>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 1);
    ctx.rebuilt = std::make_shared<core::StateHashJoinNode>(
        node.id(),
        node.stateName(),
        node.probeKeys(),
        node.outputType(),
        ctx.sources[0]);
  }

  void visit(const core::PlanNode& node, core::PlanNodeVisitorContext& context)
      const override {
    // RPCNode does not override PlanNode::accept(), so it dispatches here.
    if (isExactly<core::RPCNode>(node)) {
      auto& ctx = static_cast<RebuildContext&>(context);
      VELOX_CHECK_EQ(ctx.sources.size(), 1);
      const auto& rpc = static_cast<const core::RPCNode&>(node);
      ctx.rebuilt = std::make_shared<core::RPCNode>(
          rpc.id(),
          ctx.sources[0],
          rpc.call(),
          rpc.outputColumn(),
          rpc.outputType(),
          rpc.streamingMode(),
          rpc.dispatchBatchSize(),
          rpc.executionPhase());
    }
  }

 private:
  template <typename T>
  static bool isExactly(const core::PlanNode& node) {
    return typeid(node) == typeid(T);
  }

  template <typename T>
  static void rebuildWithSource(
      const T& node,
      core::PlanNodeVisitorContext& context) {
    if (!isExactly<T>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 1);
    ctx.rebuilt = typename T::Builder(node).source(ctx.sources[0]).build();
  }

  template <typename T>
  static void rebuildWithLeftAndRight(
      const T& node,
      core::PlanNodeVisitorContext& context) {
    if (!isExactly<T>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK_EQ(ctx.sources.size(), 2);
    ctx.rebuilt = typename T::Builder(node)
                      .left(ctx.sources[0])
                      .right(ctx.sources[1])
                      .build();
  }

  template <typename T>
  static void rebuildWithSources(
      const T& node,
      core::PlanNodeVisitorContext& context) {
    if (!isExactly<T>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    ctx.rebuilt = typename T::Builder(node).sources(ctx.sources).build();
  }

  template <typename T>
  static void rebuildLeaf(
      const T& node,
      core::PlanNodeVisitorContext& context) {
    if (!isExactly<T>(node)) {
      return;
    }
    auto& ctx = static_cast<RebuildContext&>(context);
    VELOX_CHECK(ctx.sources.empty());
    ctx.rebuilt = ctx.node;
  }
};

core::PlanNodePtr rebuildCoreNode(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& sources) {
  static const CoreNodeRebuilder kRebuilder;
  RebuildContext context{node, sources};
  node->accept(kRebuilder, context);
  return context.rebuilt;
}

// Returns why cuDF does not support the input or output types of 'node', or
// std::nullopt if it supports them.
std::optional<std::string> unsupportedTypeReason(
    const core::PlanNodePtr& node) {
  if (!isTypeSupportedByCudf(node->outputType())) {
    return fmt::format(
        "{} output type is unsupported by cuDF: {}",
        node->name(),
        node->outputType()->toString());
  }
  for (const auto& source : node->sources()) {
    if (!isTypeSupportedByCudf(source->outputType())) {
      return fmt::format(
          "{} input type is unsupported by cuDF: {}",
          node->name(),
          source->outputType()->toString());
    }
  }
  return std::nullopt;
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
      : config_(config), queryCtx_(requireQueryCtx(config)) {}

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
    auto result = rewriteImpl(node);
    if (result.hasOutput && result.mode != requestedMode) {
      result.node =
          addBoundary(std::move(result.node), result.mode, requestedMode);
      result.mode = requestedMode;
    }
    return result;
  }

  // Places 'node' bottom-up. Only rewriteNode() looks at what the consumer
  // of 'node' needs.
  RewriteResult rewriteImpl(const core::PlanNodePtr& node) {
    if (auto traits = detail::customCudfPlanNodeTraits(node, *queryCtx_)) {
      VELOX_USER_CHECK_EQ(traits->inputs.size(), node->sources().size());
      VELOX_USER_CHECK(!(traits->output && traits->outputSource));
      std::vector<core::PlanNodePtr> sources;
      std::vector<Mode> modes;
      for (size_t i = 0; i < traits->inputs.size(); ++i) {
        const auto need = traits->inputs[i];
        auto child = need == InputRepresentation::kEither
            ? rewriteImpl(node->sources()[i])
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
          detail::rebuildCustomCudfPlanNode(node, std::move(sources)),
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
          source.node == output->sources()[0]
              ? node
              : rebuildCpuNode(node, {source.node}),
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
      if (isGpuTableScan(*scan, *queryCtx_, &reason)) {
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
    return rewriteAlone(node);
  }

  // Places 'node' by itself: on the GPU if a definition lowers it and its
  // placement rule allows it, otherwise on the CPU.
  RewriteResult rewriteAlone(const core::PlanNodePtr& node) {
    if (auto reason = unsupportedReason(node)) {
      return rewriteGenericNode(node, *reason);
    }
    return lowerOnGpu(node);
  }

  // Returns why 'node' has to stay on the CPU, or std::nullopt if it can run
  // on the GPU by itself.
  std::optional<std::string> unsupportedReason(
      const core::PlanNodePtr& node) const {
    if (auto reason = unsupportedTypeReason(node)) {
      return reason;
    }
    return detail::cudfUnsupportedReason(node, *queryCtx_);
  }

  // Returns the GPU rewrite of 'node', for which unsupportedReason() has no
  // reason.
  RewriteResult lowerOnGpu(const core::PlanNodePtr& node) {
    const auto* lowering = detail::findCudfPlanNodeLowering(node);
    VELOX_CHECK_NOT_NULL(lowering);
    auto sources = rewriteChildren(node->sources(), Mode::kGpu);
    if (lowering->concatFirstSource) {
      sources[0] = addBatchConcat(
          newId(node->id() + "_batch_concat"), std::move(sources[0]));
    }
    return {lowering->lower(node, std::move(sources)), Mode::kGpu};
  }

  RewriteResult rewriteFilterProject(
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project) {
    // Do not absorb an application-selected implementation into our GPU
    // FilterProject. Keep the Project on CPU and visit the Filter normally so
    // its contract and rebuilding callback determine the input boundaries.
    // For a host Filter, LocalPlanner can still fuse the pair on CPU.
    if (detail::customCudfPlanNodeTraits(filter, *queryCtx_)) {
      // Keeping the Project on the CPU is our choice, not something cuDF
      // cannot do, so it must not fail a query that disables CPU fallback.
      if (!config_.allowCpuFallback) {
        return rewriteAlone(project);
      }
      return rewriteGenericNode(
          project, "Filter has an application-provided vector contract");
    }

    // The fused cuDF FilterProject implements the nodes the definitions list.
    // A subclass of FilterNode or ProjectNode they do not list may add
    // behaviour the fused operator would drop.
    if (detail::findCudfPlanNodeLowering(filter) == nullptr ||
        detail::findCudfPlanNodeLowering(project) == nullptr) {
      return rewriteAlone(project);
    }

    auto reason = unsupportedTypeReason(project);
    if (!reason.has_value()) {
      reason =
          detail::filterProjectUnsupportedReason(filter, project, *queryCtx_);
    }
    if (!reason.has_value()) {
      return {
          detail::lowerFilterProject(
              filter,
              project,
              rewriteNode(filter->sources()[0], Mode::kGpu).node),
          Mode::kGpu};
    }
    checkCpuFallback(project, *reason);
    auto filterSources = rewriteChildren(filter->sources(), Mode::kCpu);
    if (filterSources == filter->sources()) {
      return {project, Mode::kCpu};
    }
    return {
        rebuildCpuNode(project, {rebuildCpuNode(filter, filterSources)}),
        Mode::kCpu};
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
    std::vector<core::PlanNodePtr> newSources;
    newSources.reserve(node->sources().size());
    for (size_t i = 0; i < node->sources().size(); ++i) {
      const auto& source = node->sources()[i];
      newSources.push_back(
          isIndexLookupSource(*node, i) ? source
                                        : rewriteNode(source, Mode::kCpu).node);
    }
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
        config_.allowCpuFallback,
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
};

} // namespace

core::PlanNodePtr CudfPlanRewriter::rewrite(
    const core::PlanNodePtr& root,
    const Config& config) {
  Rewriter rewriter(config);
  return rewriter.rewrite(root);
}

} // namespace facebook::velox::cudf_velox
