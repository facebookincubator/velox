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

#include "velox/core/PlanNode.h"

#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

enum class VectorRepresentation { kVelox, kCudf };
enum class InputRepresentation { kVelox, kCudf, kEither };

/// Describes the implementation selected by a physical node, independently
/// of whether its operator launches GPU kernels. Exactly one output form is
/// used: a fixed representation, an input to preserve, or neither for a sink.
struct CudfPlanNodeTraits {
  std::vector<InputRepresentation> inputs;
  std::optional<VectorRepresentation> output;
  std::optional<size_t> outputSource;
};

/// Base class of plan nodes that run cuDF operators. Applications may also
/// implement it for their own nodes.
class CudfPlanNode : public core::PlanNode {
 public:
  using PlanNode::PlanNode;

  virtual CudfPlanNodeTraits cudfTraits() const = 0;

  /// Returns the same physical operation with the supplied sources. Retains
  /// its ID, dynamic type, expressions, and declared output schema.
  virtual core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const = 0;
};

namespace detail {
core::PlanNodePtr rebuildCudfLocalPartition(
    const core::LocalPartitionNode& node,
    std::vector<core::PlanNodePtr> sources);
} // namespace detail

} // namespace facebook::velox::cudf_velox
