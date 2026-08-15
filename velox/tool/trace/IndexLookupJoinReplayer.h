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
#include "velox/exec/Split.h"
#include "velox/tool/trace/OperatorReplayerBase.h"

namespace facebook::velox::tool::trace {
/// The replayer to replay the traced 'IndexLookupJoin' operator.
class IndexLookupJoinReplayer : public OperatorReplayerBase {
 public:
  IndexLookupJoinReplayer(
      const std::string& traceDir,
      const std::string& queryId,
      const std::string& taskId,
      const std::string& nodeId,
      const std::string& operatorType,
      const std::string& driverIds,
      uint64_t queryCapacity,
      folly::Executor* executor)
      : OperatorReplayerBase(
            traceDir,
            queryId,
            taskId,
            nodeId,
            operatorType,
            "",
            driverIds,
            queryCapacity,
            executor) {}

  /// Overrides to provide index splits to the task during replay.
  RowVectorPtr run(bool copyResults = true, bool cursorCopyResult = false)
      override;

 private:
  core::PlanNodePtr createPlanNode(
      const core::PlanNode* node,
      const core::PlanNodeId& nodeId,
      const core::PlanNodePtr& source) const override;

  // Reads traced index splits from the trace directory.
  std::vector<exec::Split> getIndexSplits() const;
};
} // namespace facebook::velox::tool::trace
