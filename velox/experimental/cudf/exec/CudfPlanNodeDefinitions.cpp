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
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfDistinct.h"
#include "velox/experimental/cudf/exec/CudfEnforceSingleRow.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupId.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfLimit.h"
#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfPlanNode.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeDefinitions.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/Task.h"

#include <typeinfo>

namespace facebook::velox::cudf_velox {
namespace {

// A definition is a struct that describes one node with static members:
//
//   using CoreNode = ...;  // The core plan node.
//   using CudfNode = ...;  // The cuDF plan node that runs it.
//
//   // Whether cuDF can run 'node'. Sets 'reason' when it cannot. A rule that
//   // depends only on the fields of the cuDF plan node is a static function
//   // of CudfNode, which its constructor checks too, so that a cuDF plan node
//   // an engine builds directly cannot skip it. A rule that needs 'queryCtx'
//   // and 'pool', such as whether cuDF can evaluate an expression, is here.
//   static bool canRunOnGpu(
//       const std::shared_ptr<const CoreNode>& node,
//       core::QueryCtx* queryCtx,
//       memory::MemoryPool* pool,
//       std::string* reason);
//
//   // The cuDF plan node for 'node' over 'source'. Does not rewrite 'source'.
//   static std::shared_ptr<const CudfNode> lower(
//       const std::shared_ptr<const CoreNode>& node,
//       core::PlanNodePtr source);
//
//   // The operator that runs 'node'.
//   static std::unique_ptr<exec::Operator> createOperator(
//       const std::shared_ptr<const CudfNode>& node,
//       exec::DriverCtx* driverCtx,
//       int32_t operatorId);
//
//   // Optional. Why cuDF cannot run the cuDF plan node 'node' with the session
//   // settings of 'queryCtx', or std::nullopt. Applies the rules of
//   // canRunOnGpu() that need 'queryCtx' and 'pool', for validateCudfPlan(),
//   // which starts from a cuDF plan node. canRunOnGpu() and this function
//   // share one implementation, so the two cannot drift apart.
//   static std::optional<std::string> unsupportedReason(
//       const CudfNode& node,
//       core::QueryCtx* queryCtx,
//       memory::MemoryPool* pool);
//
// CoreNode, canRunOnGpu() and lower() are for the plan rewriter and the cuDF
// driver adapter, which start from a core plan node. CudfNode and
// createOperator() are for CudfPlanNodeTranslator, which starts from a cuDF
// plan node. A definition has one group, the other, or both. definitions() at
// the end of this file lists every definition and says which groups it has, so
// a definition that lacks a member it is listed with does not compile.
//
// A node with more than one source takes them as a vector in lower().

// canRunOnGpu() of a node that has no rule of its own: cuDF runs it whenever
// it supports the input and output types.
template <typename TCoreNode>
bool noPlacementRule(
    const std::shared_ptr<const TCoreNode>& /*node*/,
    core::QueryCtx* /*queryCtx*/,
    memory::MemoryPool* /*pool*/,
    std::string* /*reason*/) {
  return true;
}

bool reject(std::string* reason, std::string message) {
  if (reason != nullptr) {
    *reason = std::move(message);
  }
  return false;
}

// Returns true if 'unsupported' has no reason. Otherwise sets 'reason' to it
// and returns false.
bool accept(
    const std::optional<std::string>& unsupported,
    std::string* reason) {
  return !unsupported.has_value() || reject(reason, *unsupported);
}

// Runs as CudfReduce without grouping keys, as CudfDistinct with grouping keys
// and no aggregates, and as CudfGroupby otherwise.
struct AggregationDefinition {
  using CoreNode = core::AggregationNode;
  using CudfNode = CudfAggregationNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return accept(unsupportedReason(*node, queryCtx, pool), reason);
  }

  // Serves CoreNode and CudfNode, which have the same accessors.
  template <typename TNode>
  static std::optional<std::string> unsupportedReason(
      const TNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (canBeEvaluatedByCudf(
            node.step(),
            node.groupingKeys(),
            node.aggregates(),
            node.sources().empty() ? nullptr : node.sources()[0].get(),
            queryCtx,
            pool)) {
      return std::nullopt;
    }
    return "Aggregation cannot be evaluated by cuDF";
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    std::vector<TypePtr> intermediateTypes;
    intermediateTypes.reserve(node->aggregates().size());
    for (const auto& aggregate : node->aggregates()) {
      intermediateTypes.push_back(
          exec::resolveIntermediateType(
              aggregate.call->name(), aggregate.rawInputTypes));
    }
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->step(),
        node->groupingKeys(),
        node->aggregates(),
        std::move(intermediateTypes),
        node->ignoreNullKeys());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    if (node->groupingKeys().empty()) {
      return std::make_unique<CudfReduce>(operatorId, driverCtx, node);
    }
    if (node->aggregates().empty()) {
      return std::make_unique<CudfDistinct>(operatorId, driverCtx, node);
    }
    return std::make_unique<CudfGroupby>(operatorId, driverCtx, node);
  }
};

// Returns why cuDF cannot run 'filter' followed by 'projections' over an input
// of 'inputType', or std::nullopt. 'filter' is null when there is no filter.
// 'projections' is empty when there is no projection.
std::optional<std::string> filterProjectUnsupportedReason(
    const RowTypePtr& inputType,
    const core::TypedExprPtr& filter,
    const std::vector<core::TypedExprPtr>& projections,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  // cuDF sizes an expression result from an input column. Over a zero-column
  // input a filter has nothing to evaluate against, and only constant
  // projections, which CudfFilterProject fills to the row count of each batch,
  // can run.
  if (inputType->size() == 0) {
    if (filter != nullptr) {
      return "FilterProject filter over a zero-column input";
    }
    if (!std::all_of(
            projections.begin(),
            projections.end(),
            [&](const auto& projection) {
              return isConstantProjectionOverZeroColumns(inputType, projection);
            })) {
      return "FilterProject non-constant projection over a zero-column input";
    }
  }
  if (filter != nullptr && !canExprRunOnGpu(filter, queryCtx, pool)) {
    return "FilterProject filter cannot be evaluated by cuDF";
  }
  for (const auto& projection : projections) {
    if (!canExprRunOnGpu(projection, queryCtx, pool)) {
      return "FilterProject projections cannot be evaluated by cuDF";
    }
  }
  return std::nullopt;
}

// One cuDF plan node and one operator run a Filter, a Project, or a Project
// over a Filter. FilterDefinition and ProjectDefinition lower one of the two
// alone. Callers that find a Project over a Filter decide the pair together
// with detail::canRunFilterProjectOnGpu() and lower it with
// detail::lowerFilterProject().
struct FilterProjectDefinition {
  using CudfNode = CudfFilterProjectNode;

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfFilterProject>(operatorId, driverCtx, node);
  }

  static std::optional<std::string> unsupportedReason(
      const CudfNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return filterProjectUnsupportedReason(
        node.sources()[0]->outputType(),
        node.filter(),
        node.projections(),
        queryCtx,
        pool);
  }
};

struct FilterDefinition {
  using CoreNode = core::FilterNode;
  using CudfNode = CudfFilterProjectNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return detail::canRunFilterProjectOnGpu(
        node.get(), nullptr, queryCtx, pool, reason);
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return detail::lowerFilterProject(node, nullptr, std::move(source));
  }
};

struct ProjectDefinition {
  using CoreNode = core::ProjectNode;
  using CudfNode = CudfFilterProjectNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return detail::canRunFilterProjectOnGpu(
        nullptr, node.get(), queryCtx, pool, reason);
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return detail::lowerFilterProject(nullptr, node, std::move(source));
  }
};

struct OrderByDefinition {
  using CoreNode = core::OrderByNode;
  using CudfNode = CudfOrderByNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfOrderBy>(operatorId, driverCtx, node);
  }
};

struct TopNDefinition {
  using CoreNode = core::TopNNode;
  using CudfNode = CudfTopNNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->count(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfTopN>(operatorId, driverCtx, node);
  }
};

struct TopNRowNumberDefinition {
  using CoreNode = core::TopNRowNumberNode;
  using CudfNode = CudfTopNRowNumberNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/,
      std::string* reason) {
    return accept(
        CudfNode::unsupportedReason(
            node->rankFunction(), node->partitionKeys(), node->sortingKeys()),
        reason);
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->rankFunction(),
        node->partitionKeys(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->limit(),
        node->generateRowNumber());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfTopNRowNumber>(operatorId, driverCtx, node);
  }
};

struct LimitDefinition {
  using CoreNode = core::LimitNode;
  using CudfNode = CudfLimitNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->offset(),
        node->count(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfLimit>(operatorId, driverCtx, node);
  }
};

struct AssignUniqueIdDefinition {
  using CoreNode = core::AssignUniqueIdNode;
  using CudfNode = CudfAssignUniqueIdNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->taskUniqueId());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfAssignUniqueId>(
        operatorId,
        driverCtx,
        node,
        driverCtx->task->planFragment().taskUniqueId.value_or(
            node->taskUniqueId()),
        driverCtx->task->uniqueRowIdPool());
  }
};

struct MarkDistinctDefinition {
  using CoreNode = core::MarkDistinctNode;
  using CudfNode = CudfMarkDistinctNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/,
      std::string* reason) {
    return node->masks().empty() ||
        reject(reason, "MarkDistinct masks are not supported by cuDF");
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    VELOX_CHECK(
        node->masks().empty(), "MarkDistinct masks are not supported by cuDF");
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->distinctKeys());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfMarkDistinct>(operatorId, driverCtx, node);
  }
};

struct EnforceSingleRowDefinition {
  using CoreNode = core::EnforceSingleRowNode;
  using CudfNode = CudfEnforceSingleRowNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(), std::move(source), node->outputType());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfEnforceSingleRow>(operatorId, driverCtx, node);
  }
};

struct GroupIdDefinition {
  using CoreNode = core::GroupIdNode;
  using CudfNode = CudfGroupIdNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->groupingSets(),
        node->groupingKeyInfos(),
        node->aggregationInputs(),
        node->numGroupingKeys());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfGroupId>(operatorId, driverCtx, node);
  }
};

struct WindowDefinition {
  using CoreNode = core::WindowNode;
  using CudfNode = CudfWindowNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/,
      std::string* reason) {
    return CudfWindow::canRunOnGPU(*node, reason);
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(source),
        node->outputType(),
        node->partitionKeys(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->windowFunctions(),
        node->inputsSorted());
  }

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfWindow>(operatorId, driverCtx, node);
  }
};

// The sources are the probe side and the build side. One join node becomes a
// probe operator, a build operator and the bridge between them, which
// CudfHashJoinBridgeTranslator creates.
struct HashJoinDefinition {
  using CoreNode = core::HashJoinNode;
  using CudfNode = CudfHashJoinNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    if (!accept(
            CudfNode::unsupportedReason(
                node->joinType(),
                node->isNullAware(),
                node->filter() != nullptr),
            reason)) {
      return false;
    }
    return accept(unsupportedReason(*node, queryCtx, pool), reason);
  }

  // Serves CoreNode and CudfNode, which have the same accessor.
  template <typename TNode>
  static std::optional<std::string> unsupportedReason(
      const TNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (node.filter() == nullptr ||
        canJoinFilterRunOnGpu(node.filter(), queryCtx, pool)) {
      return std::nullopt;
    }
    return "HashJoin join filter cannot be evaluated by cuDF";
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    VELOX_CHECK_EQ(sources.size(), 2);
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(sources[0]),
        std::move(sources[1]),
        node->outputType(),
        node->joinType(),
        node->isNullAware(),
        node->leftKeys(),
        node->rightKeys(),
        node->filter());
  }
};

// The sources are the probe side and the build side.
// CudfNestedLoopJoinBridgeTranslator creates the operators and the bridge.
struct NestedLoopJoinDefinition {
  using CoreNode = core::NestedLoopJoinNode;
  using CudfNode = CudfNestedLoopJoinNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    if (!accept(
            CudfNode::unsupportedReason(
                node->joinType(), node->joinCondition()),
            reason)) {
      return false;
    }
    return accept(unsupportedReason(*node, queryCtx, pool), reason);
  }

  // Serves CoreNode and CudfNode, which have the same accessor.
  template <typename TNode>
  static std::optional<std::string> unsupportedReason(
      const TNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (node.joinCondition() == nullptr ||
        canJoinFilterRunOnGpu(node.joinCondition(), queryCtx, pool)) {
      return std::nullopt;
    }
    return "NestedLoopJoin filter cannot be evaluated by cuDF";
  }

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    VELOX_CHECK_EQ(sources.size(), 2);
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(sources[0]),
        std::move(sources[1]),
        node->outputType(),
        node->joinType(),
        node->joinCondition());
  }
};

// LocalPlanner creates the operator of a local exchange itself, without asking
// a translator. CudfLocalBoundaryAdapter in ToCudf.cpp then replaces it with
// CudfLocalMerge or CudfLocalPartition.
struct LocalMergeDefinition {
  using CoreNode = core::LocalMergeNode;
  using CudfNode = CudfLocalMergeNode;

  static constexpr auto canRunOnGpu = noPlacementRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfNode>(
        node->id(),
        std::move(sources),
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders());
  }
};

// See LocalMergeDefinition for how the operator is created.
struct LocalPartitionDefinition {
  using CoreNode = core::LocalPartitionNode;

  static bool canRunOnGpu(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/,
      std::string* reason) {
    return accept(CudfLocalPartitionNode::unsupportedReason(*node), reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return CudfLocalPartitionNode::create(*node, std::move(sources));
  }
};

// The next three cuDF plan nodes have no core plan node. The plan rewriter
// adds them, and so can an engine that builds its own plan.

struct FromVeloxDefinition {
  using CudfNode = CudfFromVeloxNode;

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfFromVelox>(
        operatorId, node->outputType(), driverCtx, node->id());
  }
};

struct ToVeloxDefinition {
  using CudfNode = CudfToVeloxNode;

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfToVelox>(
        operatorId, node->outputType(), driverCtx, node->id());
  }
};

struct BatchConcatDefinition {
  using CudfNode = CudfBatchConcatNode;

  static std::unique_ptr<exec::Operator> createOperator(
      const std::shared_ptr<const CudfNode>& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId) {
    return std::make_unique<CudfBatchConcat>(operatorId, driverCtx, node);
  }
};

// One entry of definitions(): the functions of a definition behind signatures
// that do not name its node types.
struct Entry {
  // Returns whether a node is the core plan node of the definition. Null if
  // the definition has no core plan node.
  bool (*isCoreNode)(const core::PlanNodePtr& node){nullptr};

  // Meaningful only if 'isCoreNode' is set.
  detail::CudfPlanNodeLowering lowering{};

  // Returns the operator for a node, or nullptr if the node is not the cuDF
  // plan node of the definition. Null if the definition creates no operator.
  std::unique_ptr<exec::Operator> (*createOperator)(
      const core::PlanNodePtr& node,
      exec::DriverCtx* driverCtx,
      int32_t operatorId){nullptr};

  // Returns why cuDF cannot run a node with the session settings of
  // 'queryCtx'. Returns std::nullopt if it can, and if the node is not the
  // cuDF plan node of the definition. Null if the definition has no
  // unsupportedReason().
  std::optional<std::string> (*unsupportedReason)(
      const core::PlanNodePtr& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool){nullptr};
};

// Returns 'node' as the core plan node of a definition. The caller found the
// definition with isCoreNode().
template <typename TDefinition>
std::shared_ptr<const typename TDefinition::CoreNode> coreNode(
    const core::PlanNodePtr& node) {
  auto typed =
      std::dynamic_pointer_cast<const typename TDefinition::CoreNode>(node);
  VELOX_CHECK_NOT_NULL(typed);
  return typed;
}

// Returns whether 'node' is a 'T' and not a subclass of 'T'.
template <typename T>
bool isExactly(const core::PlanNode& node) {
  return typeid(node) == typeid(T);
}

// The part of lowering() and unaryLowering() that does not depend on how
// lower() takes the sources. The definition applies to its core plan node and
// to 'TSubclasses' only. Any other subclass of the core plan node may add
// behaviour that the cuDF plan node does not have, so it stays on the CPU.
template <typename TDefinition, typename... TSubclasses>
Entry coreNodeAndPlacement() {
  Entry entry;
  entry.isCoreNode = [](const core::PlanNodePtr& node) {
    return isExactly<typename TDefinition::CoreNode>(*node) ||
        (isExactly<TSubclasses>(*node) || ...);
  };
  entry.lowering.canRunOnGpu = [](const core::PlanNodePtr& node,
                                  core::QueryCtx* queryCtx,
                                  memory::MemoryPool* pool,
                                  std::string* reason) -> bool {
    return TDefinition::canRunOnGpu(
        coreNode<TDefinition>(node), queryCtx, pool, reason);
  };
  entry.lowering.concatFirstSource = false;
  return entry;
}

// Lists a definition that has CoreNode, canRunOnGpu() and lower(), where
// lower() takes all the sources as a vector.
template <typename TDefinition>
Entry lowering() {
  auto entry = coreNodeAndPlacement<TDefinition>();
  entry.lowering.lower =
      [](const core::PlanNodePtr& node,
         std::vector<core::PlanNodePtr> sources) -> core::PlanNodePtr {
    return TDefinition::lower(coreNode<TDefinition>(node), std::move(sources));
  };
  return entry;
}

// Same for a node with one source, where lower() takes that source.
template <typename TDefinition, typename... TSubclasses>
Entry unaryLowering() {
  auto entry = coreNodeAndPlacement<TDefinition, TSubclasses...>();
  entry.lowering.lower =
      [](const core::PlanNodePtr& node,
         std::vector<core::PlanNodePtr> sources) -> core::PlanNodePtr {
    VELOX_CHECK_EQ(sources.size(), 1);
    return TDefinition::lower(
        coreNode<TDefinition>(node), std::move(sources[0]));
  };
  return entry;
}

// Lists a definition that has CudfNode and createOperator().
template <typename TDefinition>
Entry operatorFactory() {
  Entry entry;
  entry.createOperator =
      [](const core::PlanNodePtr& node,
         exec::DriverCtx* driverCtx,
         int32_t operatorId) -> std::unique_ptr<exec::Operator> {
    auto cudfNode =
        std::dynamic_pointer_cast<const typename TDefinition::CudfNode>(node);
    if (cudfNode == nullptr) {
      return nullptr;
    }
    return TDefinition::createOperator(cudfNode, driverCtx, operatorId);
  };
  return entry;
}

// Lists a definition of a node with one source that has every member.
template <typename TDefinition>
Entry unaryNode() {
  auto entry = unaryLowering<TDefinition>();
  entry.createOperator = operatorFactory<TDefinition>().createOperator;
  return entry;
}

// Lists unsupportedReason() of a definition, which validateCudfPlan() then
// applies to its cuDF plan node.
template <typename TDefinition>
Entry withValidation(Entry entry) {
  entry.unsupportedReason =
      [](const core::PlanNodePtr& node,
         core::QueryCtx* queryCtx,
         memory::MemoryPool* pool) -> std::optional<std::string> {
    const auto* cudfNode =
        dynamic_cast<const typename TDefinition::CudfNode*>(node.get());
    if (cudfNode == nullptr) {
      return std::nullopt;
    }
    return TDefinition::unsupportedReason(*cudfNode, queryCtx, pool);
  };
  return entry;
}

// Makes the plan rewriter put a CudfBatchConcatNode below the first source of
// the node, so that its operator gets fewer, larger batches.
Entry withBatchConcat(Entry entry) {
  VELOX_CHECK_NOT_NULL(entry.isCoreNode);
  entry.lowering.concatFirstSource = true;
  return entry;
}

const std::vector<Entry>& definitions() {
  static const std::vector<Entry> kDefinitions{
      withValidation<AggregationDefinition>(
          withBatchConcat(unaryNode<AggregationDefinition>())),
      unaryNode<OrderByDefinition>(),
      unaryNode<TopNDefinition>(),
      unaryNode<TopNRowNumberDefinition>(),
      unaryNode<LimitDefinition>(),
      unaryNode<AssignUniqueIdDefinition>(),
      unaryNode<MarkDistinctDefinition>(),
      unaryNode<EnforceSingleRowDefinition>(),
      unaryNode<GroupIdDefinition>(),
      unaryNode<WindowDefinition>(),

      // Two core plan nodes that share a cuDF plan node and its operator.
      unaryLowering<FilterDefinition>(),
      // lowerFilterProject() keeps what a LazyDereferenceNode adds.
      unaryLowering<ProjectDefinition, core::LazyDereferenceNode>(),
      withValidation<FilterProjectDefinition>(
          operatorFactory<FilterProjectDefinition>()),

      // Joins and local exchanges. Their definitions say where their operators
      // are created. The first source of a hash join is the probe side.
      withValidation<HashJoinDefinition>(
          withBatchConcat(lowering<HashJoinDefinition>())),
      withValidation<NestedLoopJoinDefinition>(
          lowering<NestedLoopJoinDefinition>()),
      lowering<LocalMergeDefinition>(),
      lowering<LocalPartitionDefinition>(),

      // cuDF plan nodes that have no core plan node.
      operatorFactory<FromVeloxDefinition>(),
      operatorFactory<ToVeloxDefinition>(),
      operatorFactory<BatchConcatDefinition>(),
  };
  return kDefinitions;
}

} // namespace

namespace detail {

const CudfPlanNodeLowering* findCudfPlanNodeLowering(
    const core::PlanNodePtr& node) {
  for (const auto& entry : definitions()) {
    if (entry.isCoreNode != nullptr && entry.isCoreNode(node)) {
      return &entry.lowering;
    }
  }
  return nullptr;
}

bool canRunFilterProjectOnGpu(
    const core::FilterNode* filter,
    const core::ProjectNode* project,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  VELOX_CHECK(filter != nullptr || project != nullptr);
  // A FilterNode keeps its input type, so the source of the Project gives the
  // input type of a fused FilterProject too.
  const auto& inputType =
      (project != nullptr ? project->sources()[0] : filter->sources()[0])
          ->outputType();
  static const std::vector<core::TypedExprPtr> kNoProjections;
  return accept(
      filterProjectUnsupportedReason(
          inputType,
          filter != nullptr ? filter->filter() : nullptr,
          project != nullptr ? project->projections() : kNoProjections,
          queryCtx,
          pool),
      reason);
}

std::shared_ptr<const CudfFilterProjectNode> lowerFilterProject(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project,
    core::PlanNodePtr source) {
  VELOX_CHECK(filter || project);
  return std::make_shared<CudfFilterProjectNode>(
      project ? project->id() : filter->id(),
      std::move(source),
      project ? project->outputType() : filter->outputType(),
      filter ? std::make_optional(filter->id()) : std::nullopt,
      project ? std::make_optional(project->id()) : std::nullopt,
      filter ? filter->filter() : nullptr,
      project ? project->projections() : std::vector<core::TypedExprPtr>{},
      std::dynamic_pointer_cast<const core::LazyDereferenceNode>(project) !=
          nullptr);
}

std::optional<std::string> cudfPlanNodeUnsupportedReason(
    const core::PlanNodePtr& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  for (const auto& entry : definitions()) {
    if (entry.unsupportedReason == nullptr) {
      continue;
    }
    if (auto reason = entry.unsupportedReason(node, queryCtx, pool)) {
      return reason;
    }
  }
  return std::nullopt;
}

std::unique_ptr<exec::Operator> createCudfOperator(
    const core::PlanNodePtr& node,
    exec::DriverCtx* driverCtx,
    int32_t operatorId) {
  for (const auto& entry : definitions()) {
    if (entry.createOperator == nullptr) {
      continue;
    }
    if (auto op = entry.createOperator(node, driverCtx, operatorId)) {
      return op;
    }
  }
  return nullptr;
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
