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
#include "velox/tool/trace/OperatorReplayerBase.h"

namespace facebook::velox::tool::trace {
/// The replayer to replay the traced 'FilterProject' operator.
///
/// NOTE: For the plan fragment involving FilterNode->ProjectNode, users must
/// use the ProjectNode ID for tracing. This is because the planner will combine
/// these two operators into a single FilterProject operator. During replay,
/// the ProjectNode ID will be used to locate the trace data directory.
class FilterProjectReplayer : public OperatorReplayerBase {
 public:
  FilterProjectReplayer(
      const std::string& rootDir,
      const std::string& queryId,
      const std::string& taskId,
      const std::string& nodeId,
      const std::string& operatorType,
      const std::string& driverIds)
      : OperatorReplayerBase(
            rootDir,
            queryId,
            taskId,
            nodeId,
            operatorType,
            driverIds) {}

 private:
  // Create either a standalone FilterNode, a standalone ProjectNode, or a
  // ProjectNode with a FilterNode as its source.
  //
  // NOTE: If the target node is a FilterNode, it must be a standalone
  // FilterNode, without a ProjectNode as its parent.
  core::PlanNodePtr createPlanNode(
      const core::PlanNode* node,
      const core::PlanNodeId& nodeId,
      const core::PlanNodePtr& source) const override;

  // Checks whether the FilterNode is a source node of a ProjectNode.
  bool isFilterProject(const core::PlanNode* filterNode) const;
};
} // namespace facebook::velox::tool::trace
