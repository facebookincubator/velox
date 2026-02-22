/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/tool/trace/MergeJoinReplayer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/trace/TraceUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::tool::trace {
core::PlanNodePtr MergeJoinReplayer::createPlanNode(
    const core::PlanNode* node,
    const core::PlanNodeId& nodeId,
    const core::PlanNodePtr& source) const {
  const auto* mergeJoinNode = dynamic_cast<const core::MergeJoinNode*>(node);
  return std::make_shared<core::MergeJoinNode>(
      nodeId,
      mergeJoinNode->joinType(),
      mergeJoinNode->leftKeys(),
      mergeJoinNode->rightKeys(),
      mergeJoinNode->filter(),
      source,
      PlanBuilder(planNodeIdGenerator_)
          .traceScan(
              nodeTraceDir_,
              pipelineIds_.at(1), // Right side
              driverIds_,
              exec::trace::getDataType(planFragment_, nodeId_, 1))
          .planNode(),
      mergeJoinNode->outputType());
}
} // namespace facebook::velox::tool::trace
