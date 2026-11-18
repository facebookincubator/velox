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

#include "velox/exec/Task.h"

namespace facebook::velox::exec::test {

/// Waits for tasks to reach a given state.
class TaskMonitor {
 public:
  /// Default time to wait for a task to reach a state.
  static constexpr uint64_t kDefaultMaxWaitMicros{1'000'000};

  /// Waits up to 'maxWaitMicros' for 'task' to reach 'expectedState' and for
  /// its drivers to finish. A task can return results before its drivers
  /// finish, so this ensures the drivers no longer update it. Returns true if
  /// both happen in time.
  static bool waitForFinish(
      Task& task,
      TaskState expectedState,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Same as waitForFinish with TaskState::kFinished.
  static bool waitForCompletion(
      Task& task,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Same as waitForFinish with TaskState::kFailed.
  static bool waitForFailure(
      Task& task,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Same as waitForFinish with TaskState::kAborted.
  static bool waitForAbort(
      Task& task,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Same as waitForFinish with TaskState::kCanceled.
  static bool waitForCancel(
      Task& task,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Waits up to 'maxWaitMicros' for 'task' to reach 'state'. Unlike
  /// waitForFinish, does not wait for the drivers. Returns true if 'task' is
  /// in 'state'.
  static bool waitForStateChange(
      Task& task,
      TaskState state,
      uint64_t maxWaitMicros = kDefaultMaxWaitMicros);

  /// Waits up to 'maxWaitMicros' for all tasks to be deleted. Fails with the
  /// list of remaining tasks if any are left. Assumes no new tasks are created
  /// during the wait.
  static void waitForAllTasksDeleted(uint64_t maxWaitMicros = 3'000'000);
};

} // namespace facebook::velox::exec::test
