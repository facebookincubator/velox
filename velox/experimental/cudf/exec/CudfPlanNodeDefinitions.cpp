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
//   // Why cuDF cannot run the core plan node 'node', or std::nullopt. A rule
//   // that depends only on the fields of the cuDF plan node is a static
//   // function of CudfNode, which its constructor checks too, so that a cuDF
//   // plan node an engine builds directly cannot skip it. A rule that needs
//   // 'queryCtx' and 'pool', such as whether cuDF can evaluate an expression,
//   // is here.
//   static std::optional<std::string> unsupportedReason(
//       const CoreNode& node,
//       core::QueryCtx* queryCtx,
//       memory::MemoryPool* pool);
//
//   // The cuDF plan node for 'node' over 'source'. Does not rewrite 'source'.
//   static std::shared_ptr<const CudfNode> lower(
//       const std::shared_ptr<const CoreNode>& node,
//       core::PlanNodePtr source);
//
//   // Optional. The operator that runs 'node'. Most operators are constructed
//   // from (operatorId, driverCtx, node); definitions() names the operator
//   // type of those, and they have no createOperator().
//   static std::unique_ptr<exec::Operator> createOperator(
//       const std::shared_ptr<const CudfNode>& node,
//       exec::DriverCtx* driverCtx,
//       int32_t operatorId);
//
//   // Optional. The same as the first function for the cuDF plan node 'node',
//   // for validateCudfPlan(). Applies the rules that need 'queryCtx' and
//   // 'pool'; the constructor of 'node' has checked the others. The two
//   // overloads share one implementation, so they cannot drift apart.
//   static std::optional<std::string> unsupportedReason(
//       const CudfNode& node,
//       core::QueryCtx* queryCtx,
//       memory::MemoryPool* pool);
//
// CoreNode, unsupportedReason() of a core plan node and lower() are for the
// plan rewriter and the cuDF driver adapter, which start from a core plan
// node. CudfNode and the operator are for CudfPlanNodeTranslator, which starts
// from a cuDF plan node. A definition has one group, the other, or both.
// definitions() at the end of this file lists every definition and says which
// groups it has, so a definition that lacks a member it is listed with does not
// compile.
//
// A node with more than one source takes them as a vector in lower().

// unsupportedReason() of a node that has no rule of its own: cuDF runs it
// whenever it supports the input and output types.
template <typename TCoreNode>
std::optional<std::string> noRule(
    const TCoreNode& /*node*/,
    core::QueryCtx* /*queryCtx*/,
    memory::MemoryPool* /*pool*/) {
  return std::nullopt;
}

// Runs as CudfReduce without grouping keys, as CudfDistinct with grouping keys
// and no aggregates, and as CudfGroupby otherwise.
struct AggregationDefinition {
  using CoreNode = core::AggregationNode;
  using CudfNode = CudfAggregationNode;

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
std::optional<std::string> filterProjectRule(
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
// with detail::filterProjectUnsupportedReason() and lower it with
// detail::lowerFilterProject().
struct FilterProjectDefinition {
  using CudfNode = CudfFilterProjectNode;

  static std::optional<std::string> unsupportedReason(
      const CudfNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return filterProjectRule(
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

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return detail::filterProjectUnsupportedReason(
        &node, nullptr, queryCtx, pool);
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

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return detail::filterProjectUnsupportedReason(
        nullptr, &node, queryCtx, pool);
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

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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
};

struct TopNDefinition {
  using CoreNode = core::TopNNode;
  using CudfNode = CudfTopNNode;

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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
};

struct TopNRowNumberDefinition {
  using CoreNode = core::TopNRowNumberNode;
  using CudfNode = CudfTopNRowNumberNode;

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/) {
    return CudfNode::unsupportedReason(
        node.rankFunction(), node.partitionKeys(), node.sortingKeys());
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
};

struct LimitDefinition {
  using CoreNode = core::LimitNode;
  using CudfNode = CudfLimitNode;

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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
};

struct AssignUniqueIdDefinition {
  using CoreNode = core::AssignUniqueIdNode;
  using CudfNode = CudfAssignUniqueIdNode;

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/) {
    if (node.masks().empty()) {
      return std::nullopt;
    }
    return "MarkDistinct masks are not supported by cuDF";
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
};

struct EnforceSingleRowDefinition {
  using CoreNode = core::EnforceSingleRowNode;
  using CudfNode = CudfEnforceSingleRowNode;

  static constexpr auto unsupportedReason = noRule<CoreNode>;

  static std::shared_ptr<const CudfNode> lower(
      const std::shared_ptr<const CoreNode>& node,
      core::PlanNodePtr source) {
    return std::make_shared<CudfNode>(
        node->id(), std::move(source), node->outputType());
  }
};

struct GroupIdDefinition {
  using CoreNode = core::GroupIdNode;
  using CudfNode = CudfGroupIdNode;

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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
};

struct WindowDefinition {
  using CoreNode = core::WindowNode;
  using CudfNode = CudfWindowNode;

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/) {
    std::string reason;
    if (CudfWindow::canRunOnGPU(node, &reason)) {
      return std::nullopt;
    }
    if (reason.empty()) {
      return "Window is not supported by cuDF";
    }
    return reason;
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
};

// The sources are the probe side and the build side. One join node becomes a
// probe operator, a build operator and the bridge between them, which
// CudfHashJoinBridgeTranslator creates.
struct HashJoinDefinition {
  using CoreNode = core::HashJoinNode;
  using CudfNode = CudfHashJoinNode;

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (auto reason = CudfNode::unsupportedReason(
            node.joinType(), node.isNullAware(), node.filter() != nullptr)) {
      return reason;
    }
    return filterUnsupportedReason(node.filter(), queryCtx, pool);
  }

  static std::optional<std::string> unsupportedReason(
      const CudfNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return filterUnsupportedReason(node.filter(), queryCtx, pool);
  }

  // 'filter' is null for a join that has none.
  static std::optional<std::string> filterUnsupportedReason(
      const core::TypedExprPtr& filter,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (filter == nullptr || canJoinFilterRunOnGpu(filter, queryCtx, pool)) {
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

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (auto reason = CudfNode::unsupportedReason(
            node.joinType(), node.joinCondition())) {
      return reason;
    }
    return conditionUnsupportedReason(node.joinCondition(), queryCtx, pool);
  }

  static std::optional<std::string> unsupportedReason(
      const CudfNode& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    return conditionUnsupportedReason(node.joinCondition(), queryCtx, pool);
  }

  // 'joinCondition' is null for a cross join.
  static std::optional<std::string> conditionUnsupportedReason(
      const core::TypedExprPtr& joinCondition,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool) {
    if (joinCondition == nullptr ||
        canJoinFilterRunOnGpu(joinCondition, queryCtx, pool)) {
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

  static constexpr auto unsupportedReason = noRule<CoreNode>;

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

  static std::optional<std::string> unsupportedReason(
      const CoreNode& node,
      core::QueryCtx* /*queryCtx*/,
      memory::MemoryPool* /*pool*/) {
    return CudfLocalPartitionNode::unsupportedReason(node);
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
  // unsupportedReason() for its cuDF plan node.
  std::optional<std::string> (*cudfNodeUnsupportedReason)(
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
  entry.lowering.unsupportedReason =
      [](const core::PlanNodePtr& node,
         core::QueryCtx* queryCtx,
         memory::MemoryPool* pool) -> std::optional<std::string> {
    return TDefinition::unsupportedReason(
        *coreNode<TDefinition>(node), queryCtx, pool);
  };
  entry.lowering.concatFirstSource = false;
  return entry;
}

// Lists a definition that has CoreNode, unsupportedReason() and lower(), where
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

// Lists a definition that has CudfNode and whose operator is 'TOperator',
// constructed from (operatorId, driverCtx, node).
template <typename TDefinition, typename TOperator>
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
    return std::make_unique<TOperator>(operatorId, driverCtx, cudfNode);
  };
  return entry;
}

// Lists a definition that has CudfNode and its own createOperator().
template <typename TDefinition>
Entry customOperatorFactory() {
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

// Lists a definition of a node with one source that has the lowering members
// and whose operator is 'TOperator'.
template <typename TDefinition, typename TOperator>
Entry unaryNode() {
  auto entry = unaryLowering<TDefinition>();
  entry.createOperator =
      operatorFactory<TDefinition, TOperator>().createOperator;
  return entry;
}

// Same for a definition that has its own createOperator().
template <typename TDefinition>
Entry customUnaryNode() {
  auto entry = unaryLowering<TDefinition>();
  entry.createOperator = customOperatorFactory<TDefinition>().createOperator;
  return entry;
}

// Lists unsupportedReason() of a definition for its cuDF plan node, which
// validateCudfPlan() then applies.
template <typename TDefinition>
Entry withValidation(Entry entry) {
  entry.cudfNodeUnsupportedReason =
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
          withBatchConcat(customUnaryNode<AggregationDefinition>())),
      unaryNode<OrderByDefinition, CudfOrderBy>(),
      unaryNode<TopNDefinition, CudfTopN>(),
      unaryNode<TopNRowNumberDefinition, CudfTopNRowNumber>(),
      unaryNode<LimitDefinition, CudfLimit>(),
      customUnaryNode<AssignUniqueIdDefinition>(),
      unaryNode<MarkDistinctDefinition, CudfMarkDistinct>(),
      unaryNode<EnforceSingleRowDefinition, CudfEnforceSingleRow>(),
      unaryNode<GroupIdDefinition, CudfGroupId>(),
      unaryNode<WindowDefinition, CudfWindow>(),

      // Two core plan nodes that share a cuDF plan node and its operator.
      unaryLowering<FilterDefinition>(),
      // lowerFilterProject() keeps what a LazyDereferenceNode adds.
      unaryLowering<ProjectDefinition, core::LazyDereferenceNode>(),
      withValidation<FilterProjectDefinition>(
          operatorFactory<FilterProjectDefinition, CudfFilterProject>()),

      // Joins and local exchanges. Their definitions say where their operators
      // are created. The first source of a hash join is the probe side.
      withValidation<HashJoinDefinition>(
          withBatchConcat(lowering<HashJoinDefinition>())),
      withValidation<NestedLoopJoinDefinition>(
          lowering<NestedLoopJoinDefinition>()),
      lowering<LocalMergeDefinition>(),
      lowering<LocalPartitionDefinition>(),

      // cuDF plan nodes that have no core plan node.
      customOperatorFactory<FromVeloxDefinition>(),
      customOperatorFactory<ToVeloxDefinition>(),
      operatorFactory<BatchConcatDefinition, CudfBatchConcat>(),
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

core::PlanNodePtr lowerInPlace(const core::PlanNodePtr& node) {
  VELOX_CHECK_NOT_NULL(node);
  const auto* lowering = findCudfPlanNodeLowering(node);
  VELOX_CHECK_NOT_NULL(lowering, "No cuDF plan node for {}", node->name());
  return lowering->lower(node, node->sources());
}

std::optional<std::string> filterProjectUnsupportedReason(
    const core::FilterNode* filter,
    const core::ProjectNode* project,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  VELOX_CHECK(filter != nullptr || project != nullptr);
  // A FilterNode keeps its input type, so the source of the Project gives the
  // input type of a fused FilterProject too.
  const auto& inputType =
      (project != nullptr ? project->sources()[0] : filter->sources()[0])
          ->outputType();
  static const std::vector<core::TypedExprPtr> kNoProjections;
  return filterProjectRule(
      inputType,
      filter != nullptr ? filter->filter() : nullptr,
      project != nullptr ? project->projections() : kNoProjections,
      queryCtx,
      pool);
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
    if (entry.cudfNodeUnsupportedReason == nullptr) {
      continue;
    }
    if (auto reason = entry.cudfNodeUnsupportedReason(node, queryCtx, pool)) {
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
