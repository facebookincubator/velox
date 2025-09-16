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
#include "velox/experimental/streaming/StreamingPlanner.h"
#include "velox/experimental/streaming/StreamingTask.h"

#include <iostream>

namespace facebook::velox::streaming {

// static
std::shared_ptr<StreamingTask> StreamingTask::create(
    const std::string& taskId,
    core::PlanFragment planFragment,
    std::shared_ptr<core::QueryCtx> queryCtx) {
  VELOX_CHECK_NOT_NULL(planFragment.planNode);
  auto task = std::shared_ptr<StreamingTask>(new StreamingTask(
      taskId,
      std::move(planFragment),
      std::move(queryCtx)));
  task->initTaskPool();
  task->addToTaskList();
  return task;
}

StreamingTask::StreamingTask(
    const std::string& taskId,
    core::PlanFragment planFragment,
    std::shared_ptr<core::QueryCtx> queryCtx)
    : exec::Task(
          taskId,
          std::move(planFragment),
          0,
          std::move(queryCtx),
          exec::Task::ExecutionMode::kSerial,
          nullptr,
          0,
          nullptr) {
}

StreamingTask::~StreamingTask() {
}

void StreamingTask::init() {
  initOperators();
  operatorChain_->initialize();
}

void StreamingTask::initOperators() {
  auto self = shared_from_this();
  // Create the operators.
  VELOX_CHECK_NULL(operatorChain_);
  auto driverCtx = std::make_unique<exec::DriverCtx>(self, 0, 0, -1, 0);
  driver = exec::Driver::testingCreate(std::move(driverCtx));
  operatorChain_ =
      std::move(StreamingPlanner::plan(planFragment(), driver->driverCtx()));
}

StreamElementPtr StreamingTask::next(int32_t& retCode) {
  retCode = 0;

  if (!pendings_.empty()) {
    return std::move(popOutput());
  } else if (state() == exec::TaskState::kFinished) {
    // If the task is already finished, return null and 1 for retCode.
    retCode = 1;
    return nullptr;
  }

  // Run operators one by one. If an operator has output, run its downstream operators.
  // If the last operator has output, return the output.
  // If source operator has no output, check whether it is finished.
  // If source is finished, return null and 1 for rerCode, else return null and 0 for retCode.
  for (;;) {
    VELOX_CHECK_EQ(
        state(), exec::TaskState::kRunning, "Task has already finished processing.");

    operatorChain_->getOutput();
    if (pendings_.empty()) {
      if (operatorChain_->isFinished()) {
        finish();
        // finish may trigger window flush and generate output.
        if (pendings_.empty()) {
          retCode = 1;
          return nullptr;
        }
      } else if (operatorChain_->sourceEmpty()) {
        return nullptr;
      } else {
        continue;
      }
    }
    return std::move(popOutput());
  }
}

void StreamingTask::addOutput(StreamElementPtr output) {
  pendings_.push_back(std::move(output));
}

void StreamingTask::notifyWatermark(long watermark, int index) {
  operatorChain_->processWatermark(watermark, index);
}

void StreamingTask::snapshotState(long checkpointId) {
  operatorChain_->snapshotState(checkpointId);
}

void StreamingTask::notifyCheckpointComplete(long checkpointId) {
  operatorChain_->notifyCheckpointComplete(checkpointId);
}

void StreamingTask::notifyCheckpointAborted(long checkpointId) {
  operatorChain_->notifyCheckpointAborted(checkpointId);
}

StreamElementPtr StreamingTask::popOutput() {
  auto out = std::move(pendings_.front());
  pendings_.pop_front();
  return out;
}

void StreamingTask::finish() {
  VELOX_CHECK(
      pendings_.empty(),
      "Outputs have {} not been consumed before finishing the task.",
      pendings_.size());
  operatorChain_->close();
  // TODO: update operator stats

  // remove operators to release memory.
  operatorChain_.reset();
  driver.reset();
  testingFinish();
}

} // namespace facebook::velox::streaming
