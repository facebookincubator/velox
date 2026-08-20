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

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::core {
class PlanNode;
}

namespace facebook::velox::exec {
class Task;
}

namespace facebook::velox::cudf_velox {

struct GpuMemoryPlanPathEntry {
  std::string planNodeId;
  /// From PlanNode::name(), with "Node" appended when absent.
  std::string planNodeType;
};

/// Where one plan node sits within its fragment. Empty path and order -1 when
/// the node does not resolve.
struct GpuMemoryPlanLocation {
  /// Fragment root to the node, inclusive.
  std::vector<GpuMemoryPlanPathEntry> path;
  /// Position in a depth-first pre-order walk, which is the order EXPLAIN
  /// prints.
  int32_t order{-1};

  /// Zero for the fragment root.
  int32_t depth() const {
    return path.empty() ? 0 : static_cast<int32_t>(path.size()) - 1;
  }

  const GpuMemoryPlanPathEntry* node() const {
    return path.empty() ? nullptr : &path.back();
  }
};

/// Locates 'planNodeId' within the task's plan fragment.
///
/// Synthetic conversion operators carry a "-from-velox" or "-to-velox" suffix,
/// which is stripped before one retry. Callers represent an unresolved id as an
/// unmapped position rather than dropping the operator.
GpuMemoryPlanLocation gpuMemoryPlanLocation(
    const exec::Task& task,
    std::string_view planNodeId);

/// Exposed so tests can supply a plan tree without building a Task.
GpuMemoryPlanLocation gpuMemoryPlanLocationFromRoot(
    const core::PlanNode* planRoot,
    std::string_view planNodeId);

} // namespace facebook::velox::cudf_velox
