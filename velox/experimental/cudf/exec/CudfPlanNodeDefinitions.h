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

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Operator.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

/// CudfPlanNodeDefinitions.cpp describes each built-in cuDF plan node in one
/// place: when its core plan node can run on the GPU, how to build the cuDF
/// plan node from the core plan node, and how to create its operator. The plan
/// rewriter, the cuDF driver adapter and CudfPlanNodeTranslator reach those
/// definitions through the functions below, so none of them has a list of node
/// types of its own.
///
/// Building a cuDF plan node from a core plan node ("lowering") and creating
/// an operator from a cuDF plan node are independent. A plan that an engine
/// builds from cuDF plan nodes directly uses only the second.
///
/// Nodes owned by an application are not described here. They use
/// CudfPlanNodeDescriptor and their own Operator::PlanNodeTranslator.
namespace facebook::velox::cudf_velox::detail {

/// How one type of core plan node becomes a cuDF plan node.
struct CudfPlanNodeLowering {
  /// Returns why 'node' cannot become its cuDF plan node, or std::nullopt.
  /// Covers the rules the constructor of the cuDF plan node throws for, and
  /// what lowering does not carry over. Does not cover the rules that need
  /// the session; cudfUnsupportedReason() adds those.
  std::optional<std::string> (*unsupportedReason)(
      const core::PlanNodePtr& node);

  /// Returns the cuDF plan node that runs 'node', with the ID and output type
  /// of 'node' and with 'sources' in place of its sources. Does not rewrite
  /// 'sources' and does not call unsupportedReason().
  core::PlanNodePtr (*lower)(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources);

  /// True if the plan rewriter puts a CudfBatchConcatNode between the cuDF
  /// plan node and its first source.
  bool concatFirstSource;
};

/// Returns the lowering for the type of 'node'. Returns nullptr if 'node' is
/// not a core plan node that has a cuDF plan node.
const CudfPlanNodeLowering* findCudfPlanNodeLowering(
    const core::PlanNodePtr& node);

/// Returns the cuDF plan node that runs the core plan node 'node', over the
/// sources of 'node' as they are. The cuDF driver adapter uses this: it
/// replaces one operator at a time, so it has no rewritten sources. Fails if
/// 'node' has no lowering. Does not call unsupportedReason(); the constructor
/// of the cuDF plan node throws for a node cuDF cannot run.
core::PlanNodePtr lowerInPlace(const core::PlanNodePtr& node);

/// Same, with the result as 'TCudfNode'.
template <typename TCudfNode>
std::shared_ptr<const TCudfNode> lowerInPlace(const core::PlanNodePtr& node) {
  auto lowered = std::dynamic_pointer_cast<const TCudfNode>(lowerInPlace(node));
  VELOX_CHECK_NOT_NULL(lowered);
  return lowered;
}

/// Returns why cuDF cannot run the core plan node 'node' with the session
/// settings of 'queryCtx', or std::nullopt if it can. The plan rewriter and
/// the cuDF driver adapter both ask here, so the two place nodes the same way.
/// Asks, in order: whether 'node' has a lowering, the unsupportedReason() of
/// that lowering, and CudfPlanNode::unsupportedReasonForSession() of the cuDF
/// plan node that 'node' lowers to. Whether cuDF supports the input and output
/// types of 'node' is a separate check that is the same for every node:
/// isTypeSupportedByCudf().
std::optional<std::string> cudfUnsupportedReason(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx);

/// Same for 'filter', 'project', or 'project' over 'filter' as one
/// CudfFilterProjectNode. One of the two may be null. LocalPlanner fuses a
/// Filter followed by a Project into one operator, so the pair is decided
/// together.
std::optional<std::string> filterProjectUnsupportedReason(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project,
    const core::QueryCtx& queryCtx);

/// Returns the CudfFilterProjectNode over 'source' that runs 'filter',
/// 'project', or 'project' over 'filter'. One of the two may be null. The
/// lowering of a Filter or of a Project covers one of them alone. A caller
/// that finds a Project over a Filter passes both, after
/// filterProjectUnsupportedReason() accepts the pair.
std::shared_ptr<const CudfFilterProjectNode> lowerFilterProject(
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project,
    core::PlanNodePtr source);

/// Returns the operator that runs the cuDF plan node 'node'. Returns nullptr
/// for any other node, and for the joins and local exchanges, whose operators
/// their own translators create.
std::unique_ptr<exec::Operator> createCudfOperator(
    const core::PlanNodePtr& node,
    exec::DriverCtx* driverCtx,
    int32_t operatorId);

} // namespace facebook::velox::cudf_velox::detail
