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

#include "velox/exec/OperatorTraceCtx.h"

#include <re2/re2.h>
#include "velox/common/base/Exceptions.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorTraceWriter.h"
#include "velox/exec/TaskTraceWriter.h"
#include "velox/exec/trace/TraceUtil.h"

namespace facebook::velox::exec::trace {
namespace {

std::string setupTraceDirectory(
    const Operator& op,
    const std::string& queryTraceDir) {
  const auto* operatorCtx = op.operatorCtx();
  const auto pipelineId = operatorCtx->driverCtx()->pipelineId;
  const auto driverId = operatorCtx->driverCtx()->driverId;

  LOG(INFO) << "Trace input for operator type: " << op.operatorType()
            << ", operator id: " << op.operatorId()
            << ", pipeline: " << pipelineId << ", driver: " << driverId
            << ", task: " << op.taskId();

  const auto opTraceDirPath =
      getOpTraceDirectory(queryTraceDir, op.planNodeId(), pipelineId, driverId);

  createTraceDirectory(
      opTraceDirPath,
      operatorCtx->driverCtx()->queryConfig().opTraceDirectoryCreateConfig());
  return opTraceDirPath;
}

} // namespace

OperatorTraceCtx::OperatorTraceCtx(
    std::string queryNodeId,
    std::string queryTraceDir,
    UpdateAndCheckTraceLimitCB updateAndCheckTraceLimitCB,
    std::string taskRegExp,
    bool dryRun)
    : TraceCtx(dryRun),
      queryNodeId_(std::move(queryNodeId)),
      queryTraceDir_(std::move(queryTraceDir)),
      taskRegExp_(std::move(taskRegExp)),
      updateAndCheckTraceLimitCB_(std::move(updateAndCheckTraceLimitCB)) {
  VELOX_CHECK(!queryNodeId_.empty(), "The query trace node cannot be empty");
}

// static
std::unique_ptr<OperatorTraceCtx> OperatorTraceCtx::maybeCreate(
    core::QueryCtx& queryCtx,
    const core::PlanFragment& planFragment,
    const std::string& taskId) {
  const auto& queryConfig = queryCtx.queryConfig();

  VELOX_USER_CHECK(
      !queryConfig.queryTraceDir().empty(),
      "Query trace enabled but the trace dir is not set");

  VELOX_USER_CHECK(
      !queryConfig.queryTraceTaskRegExp().empty(),
      "Query trace enabled but the trace task regexp is not set");

  if (!RE2::FullMatch(taskId, queryConfig.queryTraceTaskRegExp())) {
    return nullptr;
  }

  const auto traceNodeId = queryConfig.queryTraceNodeId();
  VELOX_USER_CHECK(!traceNodeId.empty(), "Query trace node ID are not set");

  const auto traceDir = getTaskTraceDirectory(
      queryConfig.queryTraceDir(), queryCtx.queryId(), taskId);

  VELOX_USER_CHECK_NOT_NULL(
      core::PlanNode::findFirstNode(
          planFragment.planNode.get(),
          [traceNodeId](const core::PlanNode* node) -> bool {
            return node->id() == traceNodeId;
          }),
      "Trace plan node ID = '{}' not found from task '{}'",
      traceNodeId,
      taskId);

  LOG(INFO) << "Trace input for plan nodes '" << traceNodeId << "' from task '"
            << taskId << "'";

  UpdateAndCheckTraceLimitCB updateAndCheckTraceLimitCB = [&](uint64_t bytes) {
    queryCtx.updateTracedBytesAndCheckLimit(bytes);
  };

  return std::make_unique<OperatorTraceCtx>(
      traceNodeId,
      traceDir,
      std::move(updateAndCheckTraceLimitCB),
      queryConfig.queryTraceTaskRegExp(),
      queryConfig.queryTraceDryRun());
}

bool OperatorTraceCtx::shouldTrace(const Operator& op) const {
  const auto& nodeId = op.planNodeId();

  if (queryNodeId_.empty() || queryNodeId_ != nodeId) {
    return false;
  }

  auto& tracedOpMap = op.operatorCtx()->driverCtx()->tracedOperatorMap;
  if (const auto iter = tracedOpMap.find(op.operatorId());
      iter != tracedOpMap.end()) {
    LOG(WARNING) << "Operator " << iter->first << " with type of "
                 << op.operatorType() << ", plan node " << nodeId
                 << " might be the auxiliary operator of " << iter->second
                 << " which has the same operator id";
    return false;
  }
  tracedOpMap.emplace(op.operatorId(), op.operatorType());

  if (!canTrace(op.operatorType())) {
    VELOX_UNSUPPORTED("{} does not support tracing", op.operatorType());
  }
  return true;
}

std::unique_ptr<TraceInputWriter> OperatorTraceCtx::createInputTracer(
    Operator& op) const {
  return std::make_unique<OperatorTraceInputWriter>(
      &op,
      setupTraceDirectory(op, queryTraceDir_),
      memory::traceMemoryPool(),
      updateAndCheckTraceLimitCB_);
}

std::unique_ptr<TraceSplitWriter> OperatorTraceCtx::createSplitTracer(
    Operator& op) const {
  return std::make_unique<OperatorTraceSplitWriter>(
      &op, setupTraceDirectory(op, queryTraceDir_));
}

std::unique_ptr<TraceMetadataWriter> OperatorTraceCtx::createMetadataTracer()
    const {
  createTraceDirectory(queryTraceDir_);
  return std::make_unique<TaskTraceMetadataWriter>(
      queryTraceDir_, queryNodeId_, memory::traceMemoryPool());
}

} // namespace facebook::velox::exec::trace
