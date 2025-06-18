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
#include "velox4j/query/QueryExecutor.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <velox/common/base/Exceptions.h>
#include <velox/common/caching/AsyncDataCache.h>
#include <velox/common/future/VeloxPromise.h>
#include <velox/common/memory/MemoryPool.h>
#include <velox/core/PlanFragment.h>
#include <velox/core/QueryConfig.h>
#include <velox/core/QueryCtx.h>
#include <velox/exec/Driver.h>
#include <velox/exec/Operator.h>
#include <velox/exec/PlanNodeStats.h>
#include <velox/exec/Split.h>
#include <velox/exec/Task.h>
#include <velox/exec/TaskStructs.h>
#include <atomic>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_set>
#include <utility>

#include "velox4j/conf/Config.h"
#include "velox4j/memory/MemoryManager.h"
#include "velox4j/query/Query.h"

namespace facebook::velox4j {

using namespace facebook::velox;

SerialTaskStats::SerialTaskStats(const exec::TaskStats& taskStats)
    : taskStats_(taskStats) {}

folly::dynamic SerialTaskStats::toJson() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["planStats"] = exec::toPlanStatsJson(taskStats_);
  return obj;
}

SerialTask::SerialTask(
    MemoryManager* memoryManager,
    std::shared_ptr<const Query> query)
    : memoryManager_(memoryManager), query_(std::move(query)) {
  static std::atomic<uint32_t> nextExecutionId{
      0}; // Velox query ID, same with taskId.
  const uint32_t executionId = nextExecutionId++;
  core::PlanFragment planFragment{
      query_->plan(), core::ExecutionStrategy::kUngrouped, 1, {}};
  std::shared_ptr<core::QueryCtx> queryCtx = core::QueryCtx::create(
      nullptr,
      core::QueryConfig{query_->queryConfig()->toMap()},
      query_->connectorConfig()->toMap(),
      cache::AsyncDataCache::getInstance(),
      memoryManager_
          ->getVeloxPool(
              fmt::format(
                  "Query Memory Pool - Execution ID {}",
                  std::to_string(executionId)),
              memory::MemoryPool::Kind::kAggregate)
          ->shared_from_this(),
      nullptr,
      fmt::format(
          "Query Context - Execution ID {}", std::to_string(executionId)));

  auto task = exec::Task::create(
      fmt::format("Task - Execution ID {}", std::to_string(executionId)),
      std::move(planFragment),
      0,
      std::move(queryCtx),
      exec::Task::ExecutionMode::kSerial);

  task_ = task;

  VELOX_CHECK(
      task_->supportSerialExecutionMode(),
      "Task doesn't support single threaded execution: " + task->toString());
}

SerialTask::~SerialTask() {
  if (task_ != nullptr && task_->isRunning()) {
    // TODO: Calling .wait() may take no effect in single thread execution
    //  mode.
    task_->requestCancel().wait();
  }
}

UpIterator::State SerialTask::advance() {
  if (hasPendingState_) {
    hasPendingState_ = false;
    return pendingState_;
  }
  VELOX_CHECK_NULL(pending_);
  return initializeInternal(false);
}

void SerialTask::wait() {
  VELOX_CHECK(!hasPendingState_);
  VELOX_CHECK_NULL(pending_);
  pendingState_ = initializeInternal(true);
  hasPendingState_ = true;
}

RowVectorPtr SerialTask::get() {
  VELOX_CHECK(!hasPendingState_);
  VELOX_CHECK_NOT_NULL(
      pending_,
      "SerialTask: No pending row vector to return. Make sure the "
      "iterator is available via member function advance() first");
  const auto out = pending_;
  pending_ = nullptr;
  return out;
}

void SerialTask::addSplit(
    const core::PlanNodeId& planNodeId,
    int32_t groupId,
    std::shared_ptr<connector::ConnectorSplit> connectorSplit) {
  auto cs = connectorSplit;
  task_->addSplit(planNodeId, exec::Split{std::move(cs), groupId});
}

void SerialTask::noMoreSplits(const core::PlanNodeId& planNodeId) {
  task_->noMoreSplits(planNodeId);
}

std::unique_ptr<SerialTaskStats> SerialTask::collectStats() {
  const auto stats = task_->taskStats();
  return std::make_unique<SerialTaskStats>(stats);
}

UpIterator::State SerialTask::initializeInternal(bool wait) {
  while (true) {
    auto future = ContinueFuture::makeEmpty();
    auto out = task_->next(&future);
    saveDrivers();
    if (!future.valid()) {
      // Velox task is not blocked and a row vector should be gotten.
      if (out == nullptr) {
        return State::FINISHED;
      }
      pending_ = std::move(out);
      return State::AVAILABLE;
    }
    if (!wait) {
      return State::BLOCKED;
    }
    // Wait for Velox task to respond.
    VLOG(2) << "Velox task " << task_->taskId()
            << " is busy when ::next() is called. Will wait and try again. "
               "Task state: "
            << taskStateString(task_->state());
    VELOX_CHECK_NULL(
        out, "Expected to wait but still got non-null output from Velox task");
    // Avoid waiting forever because Velox doesn't propagate
    // Driver's async errors directly to Task::next.
    // https://github.com/facebookincubator/velox/blob/9a5946a09780020c1da86c37e8c377e2585d6800/velox/exec/Task.cpp#L3279
    std::move(future).wait(std::chrono::seconds(1));
  }
}

void SerialTask::saveDrivers() {
  if (drivers_.empty()) {
    // Save driver references in the first run.
    //
    // Doing this will prevent the planned operators in drivers from being
    // destroyed together with the drivers themselves in the last call to
    // Task::next (see
    // https://github.com/facebookincubator/velox/blob/4adec182144e23d7c7d6422e0090d5b59eb32b86/velox/exec/Driver.cpp#L727C13-L727C18),
    // so if a lazy vector is not loaded while the scan is drained, a meaning
    // error
    // (https://github.com/facebookincubator/velox/blob/7af0fce2c27424fbdec1974d96bb1a6d1296419d/velox/dwio/common/ColumnLoader.cpp#L32-L35)
    // can be thrown when the vector is being loaded rather than just crashing
    // the process with "pure virtual function call" or so.
    task_->testingVisitDrivers([&](exec::Driver* driver) -> void {
      drivers_.push_back(driver->shared_from_this());
    });
    VELOX_CHECK(!drivers_.empty());
  }
}

QueryExecutor::QueryExecutor(
    MemoryManager* memoryManager,
    std::shared_ptr<const Query> query)
    : memoryManager_(memoryManager), query_(query) {}

std::unique_ptr<SerialTask> QueryExecutor::execute() const {
  return std::make_unique<SerialTask>(memoryManager_, query_);
}
} // namespace facebook::velox4j
