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

#include <folly/json/dynamic.h>
#include <stdint.h>
#include <velox/core/PlanNode.h>
#include <velox/exec/TaskStats.h>
#include <velox/vector/ComplexVector.h>
#include <memory>
#include <vector>

#include "velox4j/iterator/UpIterator.h"
#include "velox4j/memory/MemoryManager.h"
#include "velox4j/query/Query.h"

namespace facebook::velox4j {

/// SerialTaskStats is a JSON-able object that wraps the task stats
/// fetched from a Velox task.
class SerialTaskStats {
 public:
  SerialTaskStats(const facebook::velox::exec::TaskStats& taskStats);

  folly::dynamic toJson() const;

 private:
  facebook::velox::exec::TaskStats taskStats_;
};

/// An UpIterator implementation that is backed by a Velox task which is
/// executed in serial execution mode.
class SerialTask : public UpIterator {
 public:
  SerialTask(MemoryManager* memoryManager, std::shared_ptr<const Query> query);

  ~SerialTask() override;

  State advance() override;

  void wait() override;

  facebook::velox::RowVectorPtr get() override;

  void addSplit(
      const facebook::velox::core::PlanNodeId& planNodeId,
      int32_t groupId,
      std::shared_ptr<facebook::velox::connector::ConnectorSplit>
          connectorSplit);

  void noMoreSplits(const facebook::velox::core::PlanNodeId& planNodeId);

  std::unique_ptr<SerialTaskStats> collectStats();

 private:
  State initializeInternal(bool wait);

  void saveDrivers();

  MemoryManager* const memoryManager_;
  std::shared_ptr<const Query> query_;
  std::shared_ptr<facebook::velox::exec::Task> task_;
  std::vector<std::shared_ptr<facebook::velox::exec::Driver>> drivers_{};
  bool hasPendingState_{false};
  State pendingState_{State::BLOCKED};
  facebook::velox::RowVectorPtr pending_{nullptr};
};

class QueryExecutor {
 public:
  QueryExecutor(
      MemoryManager* memoryManager,
      std::shared_ptr<const Query> query);

  // Executes the query. A SerialTask will be returned.
  std::unique_ptr<SerialTask> execute() const;

 private:
  MemoryManager* const memoryManager_;
  const std::shared_ptr<const Query> query_;
};

} // namespace facebook::velox4j
