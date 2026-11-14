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
#pragma once

#include "velox/experimental/cudf/exec/CudfPlanNodes.h"

namespace facebook::velox::cudf_velox {

/// Returns the cuDF plan node that runs a core plan node on the GPU. The result
/// keeps the core node's ID and sources. A ProjectNode whose source is a
/// FilterNode becomes one CudfFilterProjectNode over the filter's source, the
/// same way LocalPlanner fuses the two into one FilterProject operator. Check
/// the node against the rules in GpuPlacement.h first.
core::PlanNodePtr toCudfPlanNode(const core::PlanNodePtr& node);

template <typename CudfNode>
std::shared_ptr<const CudfNode> toCudfPlanNode(const core::PlanNodePtr& node) {
  auto converted =
      std::dynamic_pointer_cast<const CudfNode>(toCudfPlanNode(node));
  VELOX_CHECK_NOT_NULL(converted);
  return converted;
}

/// Returns the CudfFilterProjectNode for a Filter, a Project, or a Filter fused
/// under a Project.
std::shared_ptr<const CudfFilterProjectNode> toCudfFilterProjectNode(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project);

/// Returns a CudfBatchConcatNode with the ID of 'node' over the input of
/// 'node': its only source, or the probe side of a hash join.
std::shared_ptr<const CudfBatchConcatNode> toCudfBatchConcatNode(
    const core::PlanNodePtr& node);

} // namespace facebook::velox::cudf_velox
