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
#include "velox/experimental/cudf/exec/CudfPlanNodeConversion.h"

#include "velox/exec/AggregateFunctionRegistry.h"

namespace facebook::velox::cudf_velox {
namespace {

core::PlanNodePtr onlySource(const core::PlanNodePtr& node) {
  VELOX_CHECK_EQ(node->sources().size(), 1, "{}", node->name());
  return node->sources()[0];
}

std::shared_ptr<const CudfAggregationNode> toCudfAggregationNode(
    const std::shared_ptr<const core::AggregationNode>& node) {
  std::vector<TypePtr> intermediateTypes;
  for (const auto& aggregate : node->aggregates()) {
    intermediateTypes.push_back(
        exec::resolveIntermediateType(
            getOriginalName(aggregate.call->name()), aggregate.rawInputTypes));
  }
  return std::make_shared<CudfAggregationNode>(
      node->id(),
      onlySource(node),
      node->outputType(),
      node->step(),
      node->groupingKeys(),
      node->aggregates(),
      std::move(intermediateTypes),
      node->ignoreNullKeys());
}

} // namespace

std::shared_ptr<const CudfFilterProjectNode> toCudfFilterProjectNode(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project) {
  VELOX_CHECK(filter || project);
  return std::make_shared<CudfFilterProjectNode>(
      project ? project->id() : filter->id(),
      filter ? onlySource(filter) : onlySource(project),
      project ? project->outputType() : filter->outputType(),
      filter ? std::make_optional(filter->id()) : std::nullopt,
      project ? std::make_optional(project->id()) : std::nullopt,
      filter ? filter->filter() : nullptr,
      project ? project->projections() : std::vector<core::TypedExprPtr>{},
      std::dynamic_pointer_cast<const core::LazyDereferenceNode>(project) !=
          nullptr);
}

std::shared_ptr<const CudfBatchConcatNode> toCudfBatchConcatNode(
    const core::PlanNodePtr& node) {
  VELOX_CHECK_NOT_NULL(node);
  if (node->is<core::HashJoinNode>()) {
    VELOX_CHECK_EQ(node->sources().size(), 2);
    return std::make_shared<CudfBatchConcatNode>(
        node->id(), node->sources()[0]);
  }
  return std::make_shared<CudfBatchConcatNode>(node->id(), onlySource(node));
}

core::PlanNodePtr toCudfPlanNode(const core::PlanNodePtr& node) {
  VELOX_CHECK_NOT_NULL(node);

  if (auto aggregation =
          std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
    return toCudfAggregationNode(aggregation);
  }

  if (auto project = std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
    return toCudfFilterProjectNode(
        std::dynamic_pointer_cast<const core::FilterNode>(onlySource(project)),
        project);
  }

  if (auto filter = std::dynamic_pointer_cast<const core::FilterNode>(node)) {
    return toCudfFilterProjectNode(filter, nullptr);
  }

  if (auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    VELOX_CHECK_EQ(join->sources().size(), 2);
    return std::make_shared<CudfHashJoinNode>(
        join->id(),
        join->sources()[0],
        join->sources()[1],
        join->outputType(),
        join->joinType(),
        join->isNullAware(),
        join->leftKeys(),
        join->rightKeys(),
        join->filter());
  }

  if (auto join =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    VELOX_CHECK_EQ(join->sources().size(), 2);
    return std::make_shared<CudfNestedLoopJoinNode>(
        join->id(),
        join->sources()[0],
        join->sources()[1],
        join->outputType(),
        join->joinType(),
        join->joinCondition());
  }

  if (auto orderBy = std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
    return std::make_shared<CudfOrderByNode>(
        orderBy->id(),
        onlySource(node),
        orderBy->outputType(),
        orderBy->sortingKeys(),
        orderBy->sortingOrders(),
        orderBy->isPartial());
  }

  if (auto topN = std::dynamic_pointer_cast<const core::TopNNode>(node)) {
    return std::make_shared<CudfTopNNode>(
        topN->id(),
        onlySource(node),
        topN->outputType(),
        topN->sortingKeys(),
        topN->sortingOrders(),
        topN->count(),
        topN->isPartial());
  }

  if (auto topNRowNumber =
          std::dynamic_pointer_cast<const core::TopNRowNumberNode>(node)) {
    VELOX_CHECK_EQ(
        topNRowNumber->rankFunction(),
        core::TopNRowNumberNode::RankFunction::kRowNumber,
        "CudfTopNRowNumber only supports row_number");
    return std::make_shared<CudfTopNRowNumberNode>(
        topNRowNumber->id(),
        onlySource(node),
        topNRowNumber->outputType(),
        topNRowNumber->partitionKeys(),
        topNRowNumber->sortingKeys(),
        topNRowNumber->sortingOrders(),
        topNRowNumber->limit(),
        topNRowNumber->generateRowNumber());
  }

  if (auto localMerge =
          std::dynamic_pointer_cast<const core::LocalMergeNode>(node)) {
    return std::make_shared<CudfLocalMergeNode>(
        localMerge->id(),
        localMerge->sources(),
        localMerge->outputType(),
        localMerge->sortingKeys(),
        localMerge->sortingOrders());
  }

  if (auto limit = std::dynamic_pointer_cast<const core::LimitNode>(node)) {
    return std::make_shared<CudfLimitNode>(
        limit->id(),
        onlySource(node),
        limit->outputType(),
        limit->offset(),
        limit->count(),
        limit->isPartial());
  }

  if (auto assignUniqueId =
          std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(node)) {
    return std::make_shared<CudfAssignUniqueIdNode>(
        assignUniqueId->id(),
        onlySource(node),
        assignUniqueId->outputType(),
        assignUniqueId->taskUniqueId());
  }

  if (auto markDistinct =
          std::dynamic_pointer_cast<const core::MarkDistinctNode>(node)) {
    return std::make_shared<CudfMarkDistinctNode>(
        markDistinct->id(),
        onlySource(node),
        markDistinct->outputType(),
        markDistinct->distinctKeys());
  }

  if (auto enforceSingleRow =
          std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(node)) {
    return std::make_shared<CudfEnforceSingleRowNode>(
        enforceSingleRow->id(),
        onlySource(node),
        enforceSingleRow->outputType());
  }

  if (auto groupId = std::dynamic_pointer_cast<const core::GroupIdNode>(node)) {
    return std::make_shared<CudfGroupIdNode>(
        groupId->id(),
        onlySource(node),
        groupId->outputType(),
        groupId->groupingSets(),
        groupId->groupingKeyInfos(),
        groupId->aggregationInputs(),
        groupId->numGroupingKeys());
  }

  if (auto window = std::dynamic_pointer_cast<const core::WindowNode>(node)) {
    return std::make_shared<CudfWindowNode>(
        window->id(),
        onlySource(node),
        window->outputType(),
        window->partitionKeys(),
        window->sortingKeys(),
        window->sortingOrders(),
        window->windowFunctions(),
        window->inputsSorted());
  }

  VELOX_FAIL("No cuDF plan node for {}", node->name());
}

} // namespace facebook::velox::cudf_velox
