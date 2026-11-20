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

#include "velox/exec/tests/utils/TaskMonitor.h"

#include <folly/String.h>
#include <folly/executors/QueuedImmediateExecutor.h>

#include "velox/exec/Cursor.h"

namespace facebook::velox::exec::test {

bool TaskMonitor::waitForFinish(
    Task& task,
    TaskState expectedState,
    uint64_t maxWaitMicros) {
  if (!waitForStateChange(task, expectedState, maxWaitMicros)) {
    return false;
  }
  return waitForTaskDriversToFinish(&task, maxWaitMicros);
}

bool TaskMonitor::waitForCompletion(Task& task, uint64_t maxWaitMicros) {
  return waitForFinish(task, TaskState::kFinished, maxWaitMicros);
}

bool TaskMonitor::waitForFailure(Task& task, uint64_t maxWaitMicros) {
  return waitForFinish(task, TaskState::kFailed, maxWaitMicros);
}

bool TaskMonitor::waitForAbort(Task& task, uint64_t maxWaitMicros) {
  return waitForFinish(task, TaskState::kAborted, maxWaitMicros);
}

bool TaskMonitor::waitForCancel(Task& task, uint64_t maxWaitMicros) {
  return waitForFinish(task, TaskState::kCanceled, maxWaitMicros);
}

bool TaskMonitor::waitForStateChange(
    Task& task,
    TaskState state,
    uint64_t maxWaitMicros) {
  if (task.state() != state) {
    auto& executor = folly::QueuedImmediateExecutor::instance();
    auto future = task.taskCompletionFuture()
                      .within(std::chrono::microseconds(maxWaitMicros))
                      .via(&executor);
    future.wait();
  }

  return task.state() == state;
}

void TaskMonitor::waitForAllTasksDeleted(uint64_t maxWaitMicros) {
  uint64_t waitMicros = 0;
  while (Task::numRunningTasks() != 0) {
    constexpr uint64_t kWaitIntervalMicros = 50'000;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitIntervalMicros));
    waitMicros += kWaitIntervalMicros;
    if (waitMicros >= maxWaitMicros) {
      break;
    }
  }
  std::vector<std::shared_ptr<Task>> pendingTasks = Task::getRunningTasks();
  if (pendingTasks.empty()) {
    return;
  }
  std::vector<std::string> pendingTaskStats;
  pendingTaskStats.reserve(pendingTasks.size());
  for (const auto& task : pendingTasks) {
    pendingTaskStats.push_back(task->toString());
  }
  VELOX_FAIL(
      "{} pending tasks\n{}",
      pendingTasks.size(),
      folly::join("\n", pendingTaskStats));
}

} // namespace facebook::velox::exec::test
