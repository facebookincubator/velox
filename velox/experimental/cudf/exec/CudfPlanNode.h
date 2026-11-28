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

/// The kind of vectors a plan node produces: Velox vectors in host memory or
/// cuDF vectors in GPU memory.
enum class VectorRepresentation { kVelox, kCudf };

/// The kind of vectors a plan node takes from one source. kEither is for a
/// node that passes its input through without reading it.
enum class InputRepresentation { kVelox, kCudf, kEither };

/// Describes the implementation selected by a physical node, independently
/// of whether its operator launches GPU kernels. Exactly one output form is
/// used: a fixed representation, an input to preserve, or neither for a sink.
struct CudfPlanNodeTraits {
  /// What the node takes from each source, in source order.
  std::vector<InputRepresentation> inputs;
  /// What the node produces, when that does not depend on its sources.
  std::optional<VectorRepresentation> output;
  /// Index of the source whose representation the node produces, for a node
  /// that passes its input through.
  std::optional<size_t> outputSource;
};

/// Base class of plan nodes that run cuDF operators. Applications may also
/// implement it for their own nodes, or register a CudfPlanNodeDescriptor for
/// an existing node class. Neither interface chooses a replacement
/// implementation for the application's node.
class CudfPlanNode : public core::PlanNode {
 public:
  using PlanNode::PlanNode;

  /// cuDF plan nodes cannot be serialized yet. Throws rather than writing
  /// only the name, ID and sources that the base class knows.
  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("{} plan node is not serializable", name());
  }

  /// Returns what this node takes from each source and what it produces.
  virtual CudfPlanNodeTraits cudfTraits() const = 0;

  /// Returns the same physical operation with the supplied sources. Retains
  /// its ID, dynamic type, expressions, and declared output schema.
  virtual core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const = 0;

  /// Returns why this node cannot run with the session settings of
  /// 'queryCtx', or std::nullopt if it can. For example, a cast from VARCHAR
  /// cannot run on cuDF under legacy_cast. The constructor of a node checks
  /// the rules that need no session, so this covers the rest. The plan
  /// rewriter, the cuDF driver adapter and validateCudfPlan() all ask here,
  /// and the operators do not decide again. A plan is valid for the session
  /// it is checked with, which has to be the one the task runs with.
  virtual std::optional<std::string> unsupportedReasonForSession(
      const core::QueryCtx& /*queryCtx*/) const {
    return std::nullopt;
  }
};

/// Supplies representation and rebuilding support for nodes owned by another
/// library. Operator construction remains registered with LocalPlanner through
/// Operator::PlanNodeTranslator. Descriptors must be registered before
/// planning.
class CudfPlanNodeDescriptor {
 public:
  virtual ~CudfPlanNodeDescriptor() = default;

  /// Returns whether this descriptor describes 'node'.
  virtual bool canHandle(const core::PlanNode& node) const = 0;

  /// Returns what 'node' takes from each source and what it produces in the
  /// session of 'queryCtx'.
  virtual CudfPlanNodeTraits traits(
      const core::PlanNode& node,
      const core::QueryCtx& queryCtx) const = 0;

  /// Returns the operation of 'node' over 'sources'. The result must keep the
  /// ID, the dynamic type and the output schema of 'node'.
  virtual core::PlanNodePtr withSources(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) const = 0;
};

/// Registers 'descriptor' under 'name'. Throws if the name is taken. The
/// contract of a CudfPlanNode takes precedence over a registered descriptor,
/// and descriptors are asked in registration order.
void registerCudfPlanNodeDescriptor(
    std::string name,
    std::shared_ptr<const CudfPlanNodeDescriptor> descriptor);

/// Removes the descriptor registered under 'name', if there is one.
void unregisterCudfPlanNodeDescriptor(const std::string& name);

/// Validates an engine-built or rewritten plan without changing it. A terminal
/// sink has no output and is independent of the requested task representation.
///
/// Checks that both ends of every edge agree on Velox or cuDF vectors. Also
/// asks each cuDF plan node whether it can run with the session settings of
/// 'queryCtx', through CudfPlanNode::unsupportedReasonForSession(). Pass the
/// QueryCtx that the task will run with.
void validateCudfPlan(
    const core::PlanNodePtr& plan,
    const core::QueryCtx& queryCtx,
    VectorRepresentation taskOutput = VectorRepresentation::kVelox);

namespace detail {

/// Returns the traits that 'node' declares, as a CudfPlanNode, a local
/// boundary node or through a registered descriptor. Returns std::nullopt for
/// any other node.
std::optional<CudfPlanNodeTraits> customCudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx);

/// Returns the traits of 'node'. For a core plan node that declares none, a
/// GPU table scan and a UCX exchange produce cuDF vectors, a UCX partitioned
/// output takes them, and any other node takes and produces Velox vectors.
CudfPlanNodeTraits cudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx);

/// Returns 'node' over 'sources', or 'node' itself when the sources are
/// unchanged. Throws if nothing can rebuild the node, or if the result does
/// not keep its ID, dynamic type and output schema.
core::PlanNodePtr rebuildCustomCudfPlanNode(
    const core::PlanNodePtr& node,
    std::vector<core::PlanNodePtr> sources);

} // namespace detail

} // namespace facebook::velox::cudf_velox
