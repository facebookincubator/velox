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
#include "velox/experimental/cudf/exec/CudfLocalMerge.h"
#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/CudfNodeDefinitions.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeConversion.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/GpuPlacement.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/Task.h"
namespace facebook::velox::cudf_velox {
namespace {
struct DefinitionDefaults {
  static constexpr bool batchFirstInput = false;
  template <typename Node>
  static bool canRun(
      const std::shared_ptr<const Node>&,
      core::QueryCtx*,
      memory::MemoryPool*,
      std::string*) {
    return true;
  }
};
std::shared_ptr<const CudfAggregationNode> toCudfAggregationNode(
    const std::shared_ptr<const core::AggregationNode>& node,
    core::PlanNodePtr source) {
  std::vector<TypePtr> intermediateTypes;
  for (const auto& aggregate : node->aggregates()) {
    intermediateTypes.push_back(exec::resolveIntermediateType(
        getOriginalName(aggregate.call->name()), aggregate.rawInputTypes));
  }
  return std::make_shared<CudfAggregationNode>(
      node->id(),
      std::move(source),
      node->outputType(),
      node->step(),
      node->groupingKeys(),
      node->aggregates(),
      std::move(intermediateTypes),
      node->ignoreNullKeys());
}

struct AggregationDefinition : DefinitionDefaults {
  using CoreNode = core::AggregationNode;
  using CudfNode = CudfAggregationNode;
  static constexpr bool batchFirstInput = true;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunAggregationOnGpu(*node, queryCtx, pool, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return toCudfAggregationNode(node, sources.at(0));
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    const bool isGlobal = node->groupingKeys().empty();
    const bool isDistinct = !isGlobal && node->aggregates().empty();
    if (isGlobal) {
      return std::make_unique<CudfReduce>(id, ctx, node);
    }
    if (isDistinct) {
      return std::make_unique<CudfDistinct>(id, ctx, node);
    }
    return std::make_unique<CudfGroupby>(id, ctx, node);
  }
};

struct ProjectDefinition : DefinitionDefaults {
  using CoreNode = core::ProjectNode;
  using CudfNode = CudfFilterProjectNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunFilterProjectOnGpu(
        nullptr, node.get(), queryCtx, pool, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return toCudfFilterProjectNode(nullptr, node, sources.at(0));
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfFilterProject>(id, ctx, node);
  }
};

struct FilterDefinition : DefinitionDefaults {
  using CoreNode = core::FilterNode;
  using CudfNode = CudfFilterProjectNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunFilterProjectOnGpu(
        node.get(), nullptr, queryCtx, pool, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return toCudfFilterProjectNode(node, nullptr, sources.at(0));
  }
};

struct HashJoinDefinition : DefinitionDefaults {
  using CoreNode = core::HashJoinNode;
  using CudfNode = CudfHashJoinNode;
  static constexpr bool batchFirstInput = true;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunHashJoinOnGpu(*node, queryCtx, pool, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    VELOX_CHECK_EQ(node->sources().size(), 2);
    return std::make_shared<CudfHashJoinNode>(
        node->id(),
        sources.at(0),
        sources.at(1),
        node->outputType(),
        node->joinType(),
        node->isNullAware(),
        node->leftKeys(),
        node->rightKeys(),
        node->filter());
  }
};

struct NestedLoopJoinDefinition : DefinitionDefaults {
  using CoreNode = core::NestedLoopJoinNode;
  using CudfNode = CudfNestedLoopJoinNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunNestedLoopJoinOnGpu(*node, queryCtx, pool, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    VELOX_CHECK_EQ(node->sources().size(), 2);
    return std::make_shared<CudfNestedLoopJoinNode>(
        node->id(),
        sources.at(0),
        sources.at(1),
        node->outputType(),
        node->joinType(),
        node->joinCondition());
  }
};

struct OrderByDefinition : DefinitionDefaults {
  using CoreNode = core::OrderByNode;
  using CudfNode = CudfOrderByNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfOrderByNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfOrderBy>(id, ctx, node);
  }
};

struct TopNDefinition : DefinitionDefaults {
  using CoreNode = core::TopNNode;
  using CudfNode = CudfTopNNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfTopNNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->count(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfTopN>(id, ctx, node);
  }
};

struct TopNRowNumberDefinition : DefinitionDefaults {
  using CoreNode = core::TopNRowNumberNode;
  using CudfNode = CudfTopNRowNumberNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return canRunTopNRowNumberOnGpu(*node, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfTopNRowNumberNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->rankFunction(),
        node->partitionKeys(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->limit(),
        node->generateRowNumber());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfTopNRowNumber>(id, ctx, node);
  }
};

struct LocalMergeDefinition : DefinitionDefaults {
  using CoreNode = core::LocalMergeNode;
  using CudfNode = CudfLocalMergeNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfLocalMergeNode>(
        node->id(),
        sources,
        node->outputType(),
        node->sortingKeys(),
        node->sortingOrders());
  }
};

struct LimitDefinition : DefinitionDefaults {
  using CoreNode = core::LimitNode;
  using CudfNode = CudfLimitNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfLimitNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->offset(),
        node->count(),
        node->isPartial());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfLimit>(id, ctx, node);
  }
};

struct AssignUniqueIdDefinition : DefinitionDefaults {
  using CoreNode = core::AssignUniqueIdNode;
  using CudfNode = CudfAssignUniqueIdNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfAssignUniqueIdNode>(
        node->id(), sources.at(0), node->outputType(), node->taskUniqueId());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfAssignUniqueId>(
        id,
        ctx,
        node,
        ctx->task->planFragment().taskUniqueId.value_or(node->taskUniqueId()),
        ctx->task->uniqueRowIdPool());
  }
};

struct MarkDistinctDefinition : DefinitionDefaults {
  using CoreNode = core::MarkDistinctNode;
  using CudfNode = CudfMarkDistinctNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfMarkDistinctNode>(
        node->id(), sources.at(0), node->outputType(), node->distinctKeys());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfMarkDistinct>(id, ctx, node);
  }
};

struct EnforceSingleRowDefinition : DefinitionDefaults {
  using CoreNode = core::EnforceSingleRowNode;
  using CudfNode = CudfEnforceSingleRowNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfEnforceSingleRowNode>(
        node->id(), sources.at(0), node->outputType());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfEnforceSingleRow>(id, ctx, node);
  }
};

struct GroupIdDefinition : DefinitionDefaults {
  using CoreNode = core::GroupIdNode;
  using CudfNode = CudfGroupIdNode;
  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfGroupIdNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->groupingSets(),
        node->groupingKeyInfos(),
        node->aggregationInputs(),
        node->numGroupingKeys());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfGroupId>(id, ctx, node);
  }
};

struct WindowDefinition : DefinitionDefaults {
  using CoreNode = core::WindowNode;
  using CudfNode = CudfWindowNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      std::string* reason) {
    return CudfWindow::canRunOnGPU(*node, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return std::make_shared<CudfWindowNode>(
        node->id(),
        sources.at(0),
        node->outputType(),
        node->partitionKeys(),
        node->sortingKeys(),
        node->sortingOrders(),
        node->windowFunctions(),
        node->inputsSorted());
  }

  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfWindow>(id, ctx, node);
  }
};

struct LocalPartitionDefinition : DefinitionDefaults {
  using CoreNode = core::LocalPartitionNode;
  static bool canRun(
      const std::shared_ptr<const CoreNode>& node,
      core::QueryCtx*,
      memory::MemoryPool*,
      std::string* reason) {
    if (node->scaleWriter()) {
      *reason = "LocalPartition scale writer is unsupported by cuDF";
      return false;
    }
    return canRunLocalPartitionOnGpu(node, reason);
  }

  static core::PlanNodePtr lower(
      const std::shared_ptr<const CoreNode>& node,
      std::vector<core::PlanNodePtr> sources) {
    return detail::rebuildCudfLocalPartition(*node, std::move(sources));
  }
};

struct CudfFromVeloxNodeDefinition {
  using CudfNode = CudfFromVeloxNode;
  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfFromVelox>(
        id, node->outputType(), ctx, node->id());
  }
};

struct CudfToVeloxNodeDefinition {
  using CudfNode = CudfToVeloxNode;
  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfToVelox>(
        id, node->outputType(), ctx, node->id());
  }
};

struct CudfBatchConcatNodeDefinition {
  using CudfNode = CudfBatchConcatNode;
  static std::unique_ptr<exec::Operator> create(
      exec::DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const CudfNode>& node) {
    return std::make_unique<CudfBatchConcat>(id, ctx, node);
  }
};

template <typename D>
detail::CudfNodeLowering lowering() {
  return {
      [](const core::PlanNodePtr& node) {
        return std::dynamic_pointer_cast<const typename D::CoreNode>(node) !=
            nullptr;
      },
      [](const core::PlanNodePtr& node,
         core::QueryCtx* ctx,
         memory::MemoryPool* pool,
         std::string* reason) {
        return D::canRun(
            std::static_pointer_cast<const typename D::CoreNode>(node),
            ctx,
            pool,
            reason);
      },
      [](const core::PlanNodePtr& node,
         std::vector<core::PlanNodePtr> sources) {
        VELOX_CHECK_EQ(sources.size(), node->sources().size());
        return D::lower(
            std::static_pointer_cast<const typename D::CoreNode>(node),
            std::move(sources));
      },
      D::batchFirstInput};
}
template <typename D>
std::unique_ptr<exec::Operator>
create(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) {
  auto typed = std::dynamic_pointer_cast<const typename D::CudfNode>(node);
  return typed ? D::create(ctx, id, typed) : nullptr;
}
using Factory = std::unique_ptr<exec::Operator> (*)(
    exec::DriverCtx*,
    int32_t,
    const core::PlanNodePtr&);
struct DefinitionEntry {
  std::optional<detail::CudfNodeLowering> lowering;
  Factory factory{nullptr};
};
// Registration does not require both halves. Local boundaries and joins use
// their existing construction hooks; conversion nodes have no CPU counterpart.
template <typename D>
DefinitionEntry definition() {
  DefinitionEntry entry;
  if constexpr (requires { typename D::CoreNode; }) {
    entry.lowering = lowering<D>();
  }
  if constexpr (requires(
                    exec::DriverCtx* ctx,
                    int32_t id,
                    std::shared_ptr<const typename D::CudfNode> node) {
                  D::create(ctx, id, node);
                }) {
    entry.factory = create<D>;
  }
  return entry;
}
const auto& definitions() {
  static const DefinitionEntry entries[] = {
      definition<AggregationDefinition>(),
      definition<ProjectDefinition>(),
      definition<FilterDefinition>(),
      definition<HashJoinDefinition>(),
      definition<NestedLoopJoinDefinition>(),
      definition<OrderByDefinition>(),
      definition<TopNDefinition>(),
      definition<TopNRowNumberDefinition>(),
      definition<LocalMergeDefinition>(),
      definition<LimitDefinition>(),
      definition<AssignUniqueIdDefinition>(),
      definition<MarkDistinctDefinition>(),
      definition<EnforceSingleRowDefinition>(),
      definition<GroupIdDefinition>(),
      definition<WindowDefinition>(),
      definition<LocalPartitionDefinition>(),
      definition<CudfFromVeloxNodeDefinition>(),
      definition<CudfToVeloxNodeDefinition>(),
      definition<CudfBatchConcatNodeDefinition>()};
  return entries;
}
} // namespace

namespace detail {
const CudfNodeLowering* findCudfNodeLowering(const core::PlanNodePtr& node) {
  for (const auto& entry : definitions()) {
    if (entry.lowering && entry.lowering->matches(node)) {
      return &*entry.lowering;
    }
  }
  return nullptr;
}
std::unique_ptr<exec::Operator> createCudfOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  for (const auto& entry : definitions()) {
    if (entry.factory) {
      if (auto op = entry.factory(ctx, id, node)) {
        return op;
      }
    }
  }
  return nullptr;
}
} // namespace detail
} // namespace facebook::velox::cudf_velox
