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
#include "velox/experimental/cudf/exec/CudfPlanNodeConversion.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeDefinitions.h"

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
  return detail::lowerFilterProject(
      filter, project, filter ? onlySource(filter) : onlySource(project));
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
  const auto* lowering = detail::findCudfPlanNodeLowering(node);
  VELOX_CHECK_NOT_NULL(lowering, "No cuDF plan node for {}", node->name());
  return lowering->lower(node, node->sources());
}

} // namespace facebook::velox::cudf_velox
