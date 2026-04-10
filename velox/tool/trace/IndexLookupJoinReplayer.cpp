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

#include "velox/tool/trace/IndexLookupJoinReplayer.h"
#include "velox/common/Casts.h"
#include "velox/connectors/Connector.h"
#include "velox/exec/OperatorTraceReader.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/trace/TraceUtil.h"
#include "velox/tool/trace/TraceReplayTaskRunner.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::tool::trace {

core::PlanNodePtr IndexLookupJoinReplayer::createPlanNode(
    const core::PlanNode* node,
    const core::PlanNodeId& nodeId,
    const core::PlanNodePtr& source) const {
  const auto* indexLookupJoinNode =
      checkedPointerCast<const core::IndexLookupJoinNode>(node);

  // Re-create the lookup source node with a unique node ID to avoid conflicts
  // with the replay plan's generated node IDs (e.g., TraceScanNode).
  const auto* lookupSource = checkedPointerCast<const core::TableScanNode>(
      indexLookupJoinNode->lookupSource().get());
  auto replayLookupSource = std::make_shared<core::TableScanNode>(
      planNodeIdGenerator_->next(),
      lookupSource->outputType(),
      lookupSource->tableHandle(),
      lookupSource->assignments());

  return std::make_shared<core::IndexLookupJoinNode>(
      nodeId,
      indexLookupJoinNode->joinType(),
      indexLookupJoinNode->leftKeys(),
      indexLookupJoinNode->rightKeys(),
      indexLookupJoinNode->joinConditions(),
      indexLookupJoinNode->filter(),
      indexLookupJoinNode->hasMarker(),
      source,
      replayLookupSource,
      indexLookupJoinNode->outputType());
}

RowVectorPtr IndexLookupJoinReplayer::run(
    bool copyResults,
    bool cursorCopyResult) {
  auto queryCtx = createQueryCtx();
  const auto plan = createPlan();

  // Find the IndexLookupJoinNode to get the index source node ID.
  const auto* replayNode =
      core::PlanNode::findNodeById(plan.get(), replayPlanNodeId_);
  VELOX_CHECK_NOT_NULL(replayNode);
  const auto* indexLookupJoinNode =
      checkedPointerCast<const core::IndexLookupJoinNode>(replayNode);
  const auto indexSourceNodeId = indexLookupJoinNode->lookupSource()->id();

  TraceReplayTaskRunner traceTaskRunner(plan, std::move(queryCtx));
  traceTaskRunner.maxDrivers(static_cast<int32_t>(driverIds_.size()))
      .cursorCopyResult(cursorCopyResult);

  // Provide the traced index splits if the lookup source needs them.
  const auto indexSplits = getIndexSplits();
  if (!indexSplits.empty()) {
    traceTaskRunner.splits(
        indexSourceNodeId,
        std::vector<exec::Split>(indexSplits.begin(), indexSplits.end()));
  }

  auto [task, result] = traceTaskRunner.run(copyResults);
  printStats(task);
  return result;
}

std::vector<exec::Split> IndexLookupJoinReplayer::getIndexSplits() const {
  std::vector<std::string> splitInfoDirs;
  splitInfoDirs.reserve(driverIds_.size());
  for (const auto driverId : driverIds_) {
    splitInfoDirs.push_back(
        exec::trace::getOpTraceDirectory(
            nodeTraceDir_, pipelineIds_.front(), driverId));
  }
  const auto serializedSplits =
      exec::trace::OperatorTraceSplitReader(
          splitInfoDirs, memory::MemoryManager::getInstance()->tracePool())
          .read();

  std::vector<exec::Split> splits;
  for (const auto& serializedSplit : serializedSplits) {
    folly::dynamic splitInfoObject{folly::parseJson(serializedSplit)};
    const auto split =
        ISerializable::deserialize<connector::ConnectorSplit>(splitInfoObject);
    splits.emplace_back(
        std::const_pointer_cast<connector::ConnectorSplit>(split));
  }
  return splits;
}

} // namespace facebook::velox::tool::trace
