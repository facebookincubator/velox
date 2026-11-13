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
#include "velox/experimental/cudf/exec/CudfNodeDefinitions.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeConversion.h"

namespace facebook::velox::cudf_velox {
namespace {
core::PlanNodePtr onlySource(const core::PlanNodePtr& node) {
  VELOX_CHECK_EQ(node->sources().size(), 1, "{}", node->name());
  return node->sources()[0];
}
} // namespace
std::shared_ptr<const CudfFilterProjectNode> toCudfFilterProjectNode(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project) {
  VELOX_CHECK(filter || project);
  return toCudfFilterProjectNode(
      filter, project, filter ? onlySource(filter) : onlySource(project));
}
std::shared_ptr<const CudfFilterProjectNode> toCudfFilterProjectNode(
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
  // Legacy callers have not rewritten sources. Preserve their fused pair.
  if (auto project = std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
    return toCudfFilterProjectNode(
        std::dynamic_pointer_cast<const core::FilterNode>(onlySource(project)),
        project);
  }
  const auto* definition = detail::findCudfNodeLowering(node);
  VELOX_CHECK_NOT_NULL(definition, "No cuDF plan node for {}", node->name());
  return definition->lower(node, node->sources());
}
} // namespace facebook::velox::cudf_velox
