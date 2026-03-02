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

#include "velox/python/runner/PyLocalRunner.h"

#include <pybind11/stl.h>
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Spill.h"
#include "velox/python/vector/PyVector.h"

namespace facebook::velox::py {
namespace {

std::list<std::weak_ptr<exec::Task>>& taskRegistry() {
  static std::list<std::weak_ptr<exec::Task>> registry;
  return registry;
}

std::mutex& taskRegistryLock() {
  static std::mutex lock;
  return lock;
}

exec::Split preparedSplit(
    std::shared_ptr<connector::ConnectorSplit> connectorSplit) {
  // Adds special $row_group_id column to the split.
  if (auto* hiveConnectorSplit =
          dynamic_cast<connector::hive::HiveConnectorSplit*>(
              connectorSplit.get())) {
    hiveConnectorSplit->infoColumns["$row_group_id"] =
        hiveConnectorSplit->getFileName();
  }
  return exec::Split{std::move(connectorSplit)};
}

} // namespace

namespace py = pybind11;

PyTaskIterator::PyTaskIterator(
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<exec::TaskCursor> cursor)
    : outputPool_(std::move(pool)), cursor_(std::move(cursor)) {
  if (outputPool_ == nullptr) {
    throw std::runtime_error(
        "Memory pool cannot be nullptr when constructing PyTaskIterator.");
  }
  if (cursor_ == nullptr) {
    throw std::runtime_error(
        "Cursor cannot be nullptr when constructing PyTaskIterator.");
  }
}

PyVector PyTaskIterator::next() {
  if (!cursor_->moveNext()) {
    vector_ = nullptr;
    throw py::stop_iteration(); // Raise StopIteration when done.
  }
  vector_ = cursor_->current();
  return PyVector{vector_, outputPool_};
}

PyVector PyTaskIterator::step() {
  if (!cursor_->moveStep()) {
    vector_ = nullptr;
    throw py::stop_iteration(); // Raise StopIteration when done.
  }
  vector_ = cursor_->current();
  return PyVector{vector_, outputPool_};
}

std::string PyTaskIterator::at() const {
  return cursor_->at();
}

PyLocalRunner::PyLocalRunner(
    const PyPlanNode& pyPlanNode,
    const std::shared_ptr<memory::MemoryPool>& pool,
    const std::shared_ptr<folly::CPUThreadPoolExecutor>& executor)
    : rootPool_(pool),
      outputPool_(memory::memoryManager()->addLeafPool()),
      executor_(executor),
      planNode_(pyPlanNode.planNode()),
      scanFiles_(pyPlanNode.scanFiles()),
      queryConfigs_(pyPlanNode.queryConfigs()) {}

void PyLocalRunner::addFileSplit(
    const PyFile& pyFile,
    const std::string& planId,
    const std::string& connectorId) {
  scanFiles_[planId].emplace_back(
      std::make_shared<connector::hive::HiveConnectorSplit>(
          connectorId, pyFile.filePath(), pyFile.fileFormat()));
}

void PyLocalRunner::addQueryConfig(
    const std::string& configName,
    const std::string& configValue) {
  queryConfigs_[configName] = configValue;
}

exec::CursorParameters PyLocalRunner::createCursorParameters(
    int32_t maxDrivers) {
  return exec::CursorParameters{
      .planNode = planNode_,
      .maxDrivers = maxDrivers,
      .queryCtx = core::QueryCtx::Builder()
                      .executor(executor_.get())
                      .queryConfig(core::QueryConfig(queryConfigs_))
                      .pool(rootPool_)
                      .build(),
      .outputPool = outputPool_,
  };
}

PyTaskIterator PyLocalRunner::execute(int32_t maxDrivers) {
  // Initialize task cursor and task.
  cursor_ = exec::TaskCursor::create(createCursorParameters(maxDrivers));

  // Add any files passed by the client during plan building.
  for (auto& [scanId, splits] : scanFiles_) {
    for (auto& split : splits) {
      cursor_->task()->addSplit(scanId, preparedSplit(std::move(split)));
    }
    cursor_->task()->noMoreSplits(scanId);
  }
  scanFiles_.clear();

  {
    std::lock_guard<std::mutex> guard(taskRegistryLock());
    taskRegistry().push_back(cursor_->task());
  }
  return PyTaskIterator{outputPool_, cursor_};
}

std::string PyLocalRunner::printPlanWithStats() const {
  return exec::printPlanWithStats(
      *planNode_, cursor_->task()->taskStats(), true);
}

void drainAllTasks() {
  auto& executor = folly::QueuedImmediateExecutor::instance();
  std::lock_guard<std::mutex> guard(taskRegistryLock());

  auto it = taskRegistry().begin();
  while (it != taskRegistry().end()) {
    // Try to acquire a shared_ptr from the weak_ptr (in case the task has
    // already finished).
    if (auto task = it->lock()) {
      if (!task->isFinished()) {
        task->requestAbort();
      }
      auto future = task->taskCompletionFuture()
                        .within(std::chrono::seconds(1))
                        .via(&executor);
      future.wait();
    }
    it = taskRegistry().erase(it);
  }
}

} // namespace facebook::velox::py
