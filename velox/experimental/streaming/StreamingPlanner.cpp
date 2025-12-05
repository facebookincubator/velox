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
#include "velox/core/PlanFragment.h"
#include "velox/exec/AssignUniqueId.h"
#include "velox/exec/CallbackSink.h"
#include "velox/exec/EnforceSingleRow.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Expand.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/GroupId.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/IndexLookupJoin.h"
#include "velox/exec/Limit.h"
#include "velox/exec/MarkDistinct.h"
#include "velox/exec/Merge.h"
#include "velox/exec/MergeJoin.h"
#include "velox/exec/NestedLoopJoinBuild.h"
#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/exec/OrderBy.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/exec/RowNumber.h"
#include "velox/exec/StreamingAggregation.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/TableWriteMerge.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/Task.h"
#include "velox/exec/TopN.h"
#include "velox/exec/TopNRowNumber.h"
#include "velox/exec/Unnest.h"
#include "velox/exec/Values.h"
#include "velox/exec/Window.h"
#include "velox/experimental/streaming/StreamingPlanner.h"
#include "velox/experimental/streaming/StreamingPlanNode.h"

namespace facebook::velox::streaming {

static std::atomic<int> opId = 0;

// static
StreamingOperatorPtr StreamingPlanner::plan(
    const core::PlanFragment& planFragment,
    exec::DriverCtx* ctx) {
  return nodeToStreamingOperator(planFragment.planNode, ctx);
}

//static
StreamingOperatorPtr StreamingPlanner::nodeToStreamingOperator(
    const core::PlanNodePtr& planNode,
    exec::DriverCtx* ctx) {
  auto streamingNode =
      std::dynamic_pointer_cast<const StreamingPlanNode>(planNode);
  VELOX_CHECK(streamingNode, "Not streaming node: {}", planNode->toString());
  std::vector<StreamingOperatorPtr> targets;
  std::unique_ptr<exec::Operator> op = std::move(nodeToOperator(streamingNode->node(), ctx));
  for (auto target : streamingNode->targets()) {
    targets.push_back(std::move(nodeToStreamingOperator(target, ctx)));
  }
  return std::make_unique<StreamingOperator>(std::move(op), std::move(targets));
}

//static
std::unique_ptr<exec::Operator> StreamingPlanner::nodeToOperator(
    const core::PlanNodePtr& planNode,
    exec::DriverCtx* ctx) {
  if (auto filterNode =
      std::dynamic_pointer_cast<const core::FilterNode>(planNode)) {
    if (planNode->sources().size() == 1) {
      auto next = planNode->sources()[0];
      if (auto projectNode =
          std::dynamic_pointer_cast<const core::ProjectNode>(next)) {
        return std::make_unique<exec::FilterProject>(
            opId.fetch_add(1),
            ctx,
            filterNode,
            projectNode);
      }
    }
    return std::make_unique<exec::FilterProject>(opId.fetch_add(1), ctx, filterNode, nullptr);
  } else if (
      auto projectNode =
          std::dynamic_pointer_cast<const core::ProjectNode>(planNode)) {
    return std::make_unique<exec::FilterProject>(opId.fetch_add(1), ctx, nullptr, projectNode);
  } else if (
      auto valuesNode =
          std::dynamic_pointer_cast<const core::ValuesNode>(planNode)) {
    return std::make_unique<exec::Values>(opId.fetch_add(1), ctx, valuesNode);
  } else if (
      auto tableScanNode =
          std::dynamic_pointer_cast<const core::TableScanNode>(planNode)) {
    return std::make_unique<exec::TableScan>(opId.fetch_add(1), ctx, tableScanNode);
  } else if (
      auto tableWriteNode =
          std::dynamic_pointer_cast<const core::TableWriteNode>(planNode)) {
      return std::make_unique<exec::TableWriter>(opId.fetch_add(1), ctx, tableWriteNode);
  } else if (
      auto tableWriteMergeNode =
          std::dynamic_pointer_cast<const core::TableWriteMergeNode>(planNode)) {
    return std::make_unique<exec::TableWriteMerge>(opId.fetch_add(1), ctx, tableWriteMergeNode);
  } else if (
      auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(planNode)) {
    return std::make_unique<exec::HashProbe>(opId.fetch_add(1), ctx, joinNode);
  } else if (
      auto joinNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(planNode)) {
    return std::make_unique<exec::NestedLoopJoinProbe>(opId.fetch_add(1), ctx, joinNode);
  } else if (
      auto joinNode =
          std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(planNode)) {
    return std::make_unique<exec::IndexLookupJoin>(opId.fetch_add(1), ctx, joinNode);
  } else if (
      auto aggregationNode =
          std::dynamic_pointer_cast<const core::AggregationNode>(planNode)) {
    if (aggregationNode->isPreGrouped()) {
      return std::make_unique<exec::StreamingAggregation>(opId.fetch_add(1), ctx, aggregationNode);
    } else {
      return std::make_unique<exec::HashAggregation>(opId.fetch_add(1), ctx, aggregationNode);
    }
  } else if (
      auto expandNode =
          std::dynamic_pointer_cast<const core::ExpandNode>(planNode)) {
    return std::make_unique<exec::Expand>(opId.fetch_add(1), ctx, expandNode);
  } else if (
      auto groupIdNode =
          std::dynamic_pointer_cast<const core::GroupIdNode>(planNode)) {
    return std::make_unique<exec::GroupId>(opId.fetch_add(1), ctx, groupIdNode);
  } else if (
      auto topNNode =
          std::dynamic_pointer_cast<const core::TopNNode>(planNode)) {
      return std::make_unique<exec::TopN>(opId.fetch_add(1), ctx, topNNode);
  } else if (
      auto limitNode =
          std::dynamic_pointer_cast<const core::LimitNode>(planNode)) {
    return std::make_unique<exec::Limit>(opId.fetch_add(1), ctx, limitNode);
  } else if (
      auto orderByNode =
          std::dynamic_pointer_cast<const core::OrderByNode>(planNode)) {
    return std::make_unique<exec::OrderBy>(opId.fetch_add(1), ctx, orderByNode);
  } else if (
      auto windowNode =
          std::dynamic_pointer_cast<const core::WindowNode>(planNode)) {
    return std::make_unique<exec::Window>(opId.fetch_add(1), ctx, windowNode);
  } else if (
      auto rowNumberNode =
          std::dynamic_pointer_cast<const core::RowNumberNode>(planNode)) {
    return std::make_unique<exec::RowNumber>(opId.fetch_add(1), ctx, rowNumberNode);
  } else if (
      auto topNRowNumberNode =
          std::dynamic_pointer_cast<const core::TopNRowNumberNode>(planNode)) {
    return std::make_unique<exec::TopNRowNumber>(opId.fetch_add(1), ctx, topNRowNumberNode);
  } else if (
      auto markDistinctNode =
          std::dynamic_pointer_cast<const core::MarkDistinctNode>(planNode)) {
    return std::make_unique<exec::MarkDistinct>(opId.fetch_add(1), ctx, markDistinctNode);
  } else if (
      auto mergeJoin =
          std::dynamic_pointer_cast<const core::MergeJoinNode>(planNode)) {
    auto mergeJoinOp = std::make_unique<exec::MergeJoin>(opId.fetch_add(1), ctx, mergeJoin);
    ctx->task->createMergeJoinSource(ctx->splitGroupId, mergeJoin->id());
    return std::move(mergeJoinOp);
  } else if (
      auto unnest =
          std::dynamic_pointer_cast<const core::UnnestNode>(planNode)) {
    return std::make_unique<exec::Unnest>(opId.fetch_add(1), ctx, unnest);
  } else if (
      auto enforceSingleRow =
          std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(planNode)) {
    return std::make_unique<exec::EnforceSingleRow>(opId.fetch_add(1), ctx, enforceSingleRow);
  } else if (
      auto assignUniqueIdNode =
          std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(planNode)) {
    return std::make_unique<exec::AssignUniqueId>(
        opId.fetch_add(1),
        ctx,
        assignUniqueIdNode,
        assignUniqueIdNode->taskUniqueId(),
        assignUniqueIdNode->uniqueIdCounter());
  } else {
    std::unique_ptr<exec::Operator> extended;
    extended = exec::Operator::fromPlanNode(ctx, opId.fetch_add(1), planNode);
    VELOX_CHECK(extended, "Unsupported plan node: {}", planNode->toString());
    return extended;
  }
}

} // namespace facebook::velox::streaming
