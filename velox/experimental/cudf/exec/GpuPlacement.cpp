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
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/GpuPlacement.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

namespace facebook::velox::cudf_velox {
namespace {

bool reject(std::string* reason, std::string message) {
  if (reason != nullptr) {
    *reason = std::move(message);
  }
  return false;
}

bool containsCustomComparison(const TypePtr& type) {
  if (type->providesCustomComparison()) {
    return true;
  }
  for (size_t i = 0; i < type->size(); ++i) {
    if (containsCustomComparison(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

bool containsCustomComparison(const core::TypedExprPtr& expression) {
  if (containsCustomComparison(expression->type())) {
    return true;
  }
  return std::any_of(
      expression->inputs().begin(),
      expression->inputs().end(),
      [](const auto& input) { return containsCustomComparison(input); });
}

} // namespace

bool canRunFilterProjectOnGpu(
    const core::FilterNode* filter,
    const core::ProjectNode* project,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (project != nullptr && project->sources()[0]->outputType()->size() == 0 &&
      (filter != nullptr || !project->projections().empty())) {
    return reject(
        reason, "FilterProject empty input type with filter or projections");
  }
  if (filter != nullptr && !canExprRunOnGpu(filter->filter(), queryCtx, pool)) {
    return reject(reason, "FilterProject filter cannot be evaluated by cuDF");
  }
  if (project != nullptr) {
    for (const auto& projection : project->projections()) {
      if (!canExprRunOnGpu(projection, queryCtx, pool)) {
        return reject(
            reason, "FilterProject projections cannot be evaluated by cuDF");
      }
    }
  }
  return true;
}

bool canRunAggregationOnGpu(
    const core::AggregationNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!canBeEvaluatedByCudf(node, queryCtx, pool)) {
    return reject(
        reason, "Aggregation aggregation cannot be evaluated by cuDF");
  }
  return true;
}

bool canRunHashJoinOnGpu(
    const core::HashJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!CudfHashJoinProbe::isSupportedJoinType(node.joinType())) {
    return reject(reason, "HashJoin unsupported join type");
  }
  // Disabling null-aware anti join with filter until we implement it right
  if (node.joinType() == core::JoinType::kAnti && node.isNullAware() &&
      node.filter()) {
    return reject(
        reason, "HashJoin null-aware anti join with filter not implemented");
  }
  if (node.filter() && !canJoinFilterRunOnGpu(node.filter(), queryCtx, pool)) {
    return reject(reason, "HashJoin join filter cannot be evaluated by cuDF");
  }
  return true;
}

bool canRunNestedLoopJoinOnGpu(
    const core::NestedLoopJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason) {
  if (!CudfNestedLoopJoinProbe::isSupportedJoinType(node.joinType())) {
    return reject(
        reason,
        fmt::format(
            "NestedLoopJoin unsupported join type: {}",
            static_cast<int>(node.joinType())));
  }
  if (node.joinCondition()) {
    if (containsCustomComparison(node.joinCondition())) {
      return reject(
          reason, "NestedLoopJoin condition has custom comparison semantics");
    }
    if (!canJoinFilterRunOnGpu(node.joinCondition(), queryCtx, pool)) {
      return reject(
          reason, "NestedLoopJoin filter cannot be evaluated by cuDF");
    }
  }
  return true;
}

bool canRunTopNRowNumberOnGpu(
    const core::TopNRowNumberNode& node,
    std::string* reason) {
  if (node.rankFunction() !=
      core::TopNRowNumberNode::RankFunction::kRowNumber) {
    return reject(reason, "TopNRowNumber only supports row_number on cuDF");
  }
  return true;
}

bool canRunLocalPartitionOnGpu(
    const std::shared_ptr<const core::LocalPartitionNode>& node,
    std::string* reason) {
  if (!CudfLocalPartition::shouldReplace(node)) {
    return reject(
        reason, "LocalPartition partition spec is unsupported by cuDF");
  }
  return true;
}

} // namespace facebook::velox::cudf_velox
