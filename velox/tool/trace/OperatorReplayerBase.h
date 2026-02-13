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

#pragma once

#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/parse/PlanNodeIdGenerator.h"

namespace facebook::velox::exec {
class Task;
}

namespace facebook::velox::tool::trace {
class OperatorReplayerBase {
 public:
  OperatorReplayerBase(
      const std::string& traceDir,
      const std::string& queryId,
      const std::string& taskId,
      const std::string& nodeId,
      const std::string& nodeName,
      const std::string& spillBaseDir,
      const std::string& driverIds,
      uint64_t queryCapacity,
      folly::Executor* executor);
  virtual ~OperatorReplayerBase() = default;

  OperatorReplayerBase(const OperatorReplayerBase& other) = delete;
  OperatorReplayerBase& operator=(const OperatorReplayerBase& other) = delete;
  OperatorReplayerBase(OperatorReplayerBase&& other) noexcept = delete;
  OperatorReplayerBase& operator=(OperatorReplayerBase&& other) noexcept =
      delete;

  /// @deprecated Use the two-parameter version instead. This overload exists
  ///        for backward compatibility and will be removed once all subclasses
  ///        migrate to the new interface.
  virtual RowVectorPtr run(bool copyResults) {
    return run(copyResults, /*cursorCopyResult=*/false);
  }

  /// Runs the replayer with control over both result copying and cursor
  /// per-batch copying.
  /// @param copyResults If true, copies and returns all results as a single
  ///        RowVector. If false, returns nullptr.
  /// @param cursorCopyResult If true, each output batch is deep copied as it's
  ///        consumed by the cursor. This can be expensive for complex nested
  ///        types. Default is false.
  virtual RowVectorPtr run(
      bool copyResults = true,
      bool cursorCopyResult = false);

 protected:
  virtual core::PlanNodePtr createPlanNode(
      const core::PlanNode* node,
      const core::PlanNodeId& nodeId,
      const core::PlanNodePtr& source) const = 0;

  core::PlanNodePtr createPlan();

  std::shared_ptr<core::QueryCtx> createQueryCtx();

  const std::string queryId_;
  const std::string taskId_;
  const std::string nodeId_;
  const std::string nodeName_;
  const std::string taskTraceDir_;
  const std::string nodeTraceDir_;
  const std::string spillBaseDir_;
  const std::shared_ptr<filesystems::FileSystem> fs_;
  const std::vector<uint32_t> pipelineIds_;
  const std::vector<uint32_t> driverIds_;
  const uint64_t queryCapacity_;
  const std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator_{
      std::make_shared<core::PlanNodeIdGenerator>()};
  folly::Executor* const executor_;

  std::unordered_map<std::string, std::string> queryConfigs_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      connectorConfigs_;
  core::PlanNodePtr planFragment_;
  core::PlanNodeId replayPlanNodeId_;

  void printStats(const std::shared_ptr<exec::Task>& task) const;

 private:
  std::function<core::PlanNodePtr(std::string, core::PlanNodePtr)>
  replayNodeFactory(const core::PlanNode* node) const;
};
} // namespace facebook::velox::tool::trace
