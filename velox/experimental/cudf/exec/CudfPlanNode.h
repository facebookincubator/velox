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
#include "velox/core/QueryCtx.h"

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
/// implement it for their own nodes, or register a CudfPlanNodeDescriptor for
/// an existing node class. Neither interface chooses a replacement
/// implementation for the application's node.
class CudfPlanNode : public core::PlanNode {
 public:
  using PlanNode::PlanNode;

  virtual CudfPlanNodeTraits cudfTraits() const = 0;

  /// Returns the same physical operation with the supplied sources. Retains
  /// its ID, dynamic type, expressions, and declared output schema.
  virtual core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const = 0;
};

/// Supplies representation and rebuilding support for nodes owned by another
/// library. Operator construction remains registered with LocalPlanner through
/// Operator::PlanNodeTranslator. Descriptors must be registered before
/// planning.
class CudfPlanNodeDescriptor {
 public:
  virtual ~CudfPlanNodeDescriptor() = default;
  virtual bool canHandle(const core::PlanNode& node) const = 0;
  virtual CudfPlanNodeTraits traits(
      const core::PlanNode& node,
      const core::QueryCtx& queryCtx) const = 0;
  virtual core::PlanNodePtr withSources(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) const = 0;
};

/// Names identify registrations for targeted cleanup. Duplicate names fail;
/// intrinsic CudfPlanNode contracts take precedence over registered
/// descriptors.
void registerCudfPlanNodeDescriptor(
    std::string name,
    std::shared_ptr<const CudfPlanNodeDescriptor> descriptor);
void unregisterCudfPlanNodeDescriptor(const std::string& name);

/// Validates an engine-built or rewritten plan without changing it. A terminal
/// sink has no output and is independent of the requested task representation.
void validateCudfPlan(
    const core::PlanNodePtr& plan,
    const core::QueryCtx& queryCtx,
    VectorRepresentation taskOutput = VectorRepresentation::kVelox);

namespace detail {
core::PlanNodePtr rebuildCudfLocalPartition(
    const core::LocalPartitionNode& node,
    std::vector<core::PlanNodePtr> sources);
std::optional<CudfPlanNodeTraits> customCudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx);
CudfPlanNodeTraits cudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx);
core::PlanNodePtr rebuildCustomCudfPlanNode(
    const core::PlanNodePtr& node,
    std::vector<core::PlanNodePtr> sources);
} // namespace detail

} // namespace facebook::velox::cudf_velox
