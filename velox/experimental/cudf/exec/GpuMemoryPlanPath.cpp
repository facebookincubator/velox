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

#include "velox/experimental/cudf/exec/GpuMemoryPlanPath.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"

#include <array>

namespace facebook::velox::cudf_velox {

namespace {

/// Suffixes appended to plan node identifiers of synthetic conversion
/// operators, which have no plan node of their own.
constexpr std::array<std::string_view, 2> kConversionSuffixes{
    "-from-velox",
    "-to-velox"};

std::string displayPlanNodeType(const core::PlanNode& planNode) {
  auto type = std::string{planNode.name()};
  if (!type.ends_with("Node")) {
    type += "Node";
  }
  return type;
}

/// Depth first, in the order EXPLAIN prints.
///
/// 'nextOrder' advances for every node visited, so a subtree that does not
/// contain the target still shifts the index of everything after it. Returns as
/// soon as the target matches, leaving 'location' untouched when it does not.
bool findPlanLocation(
    const core::PlanNode& node,
    std::string_view planNodeId,
    int32_t& nextOrder,
    GpuMemoryPlanLocation& location) {
  const auto order = nextOrder++;
  location.path.push_back(
      GpuMemoryPlanPathEntry{
          std::string{node.id()}, displayPlanNodeType(node)});
  if (node.id() == planNodeId) {
    location.order = order;
    return true;
  }
  for (const auto& source : node.sources()) {
    if (source != nullptr &&
        findPlanLocation(*source, planNodeId, nextOrder, location)) {
      return true;
    }
  }
  location.path.pop_back();
  return false;
}

} // namespace

GpuMemoryPlanLocation gpuMemoryPlanLocationFromRoot(
    const core::PlanNode* planRoot,
    std::string_view planNodeId) {
  GpuMemoryPlanLocation location;
  if (planRoot == nullptr || planNodeId.empty()) {
    return location;
  }

  int32_t nextOrder = 0;
  if (findPlanLocation(*planRoot, planNodeId, nextOrder, location)) {
    return location;
  }

  for (const auto suffix : kConversionSuffixes) {
    if (planNodeId.ends_with(suffix)) {
      auto sourceId = planNodeId;
      sourceId.remove_suffix(suffix.size());
      location = GpuMemoryPlanLocation{};
      nextOrder = 0;
      if (findPlanLocation(*planRoot, sourceId, nextOrder, location)) {
        return location;
      }
      break;
    }
  }
  return GpuMemoryPlanLocation{};
}

GpuMemoryPlanLocation gpuMemoryPlanLocation(
    const exec::Task& task,
    std::string_view planNodeId) {
  return gpuMemoryPlanLocationFromRoot(
      task.planFragment().planNode.get(), planNodeId);
}

} // namespace facebook::velox::cudf_velox
