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

#include <gtest/gtest.h>

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorAssertions.h"

namespace facebook::velox::exec::test {

bool assertEqualResults(
    const std::vector<RowVectorPtr>& expected,
    const std::vector<RowVectorPtr>& actual) {
  return velox::test::VectorAssertions::equalMultisets(expected, actual);
}

bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedType,
    const std::vector<RowVectorPtr>& actual) {
  return velox::test::VectorAssertions::equalMultisets(
      expectedRows, expectedType, actual);
}

namespace {
tsan_atomic<int32_t>& testingAbortPct() {
  static tsan_atomic<int32_t> abortPct = 0;
  return abortPct;
}

tsan_atomic<int32_t>& testingAbortCounter() {
  static tsan_atomic<int32_t> counter = 0;
  return counter;
}

std::function<void(Task*)>& testingAbortHook() {
  static std::function<void(Task*)> hook = nullptr;
  return hook;
}

// Aborts 'task' with the probability set by TestScopedAbortInjection.
void testingMaybeTriggerAbort(exec::Task* task) {
  if (testingAbortPct() <= 0 || testingAbortCounter() <= 0) {
    return;
  }

  if ((folly::Random::rand32() % 100) < testingAbortPct()) {
    if (testingAbortCounter()-- > 0) {
      if (testingAbortHook() != nullptr) {
        testingAbortHook()(task);
      }
      task->requestAbort();
    }
  }
}
} // namespace

TestScopedAbortInjection::TestScopedAbortInjection(
    int32_t abortPct,
    int32_t maxInjections,
    std::function<void(Task*)> hook) {
  testingAbortPct() = abortPct;
  testingAbortCounter() = maxInjections;
  testingAbortHook() = hook;
}

TestScopedAbortInjection::~TestScopedAbortInjection() {
  testingAbortPct() = 0;
  testingAbortCounter() = 0;
  testingAbortHook() = nullptr;
}

std::pair<std::unique_ptr<TaskCursor>, std::vector<RowVectorPtr>> readCursor(
    const CursorParameters& params,
    std::function<void(TaskCursor*)> addSplits,
    uint64_t maxWaitMicros) {
  auto cursor = TaskCursor::create(params);
  // Declared after 'cursor' so that it is destroyed first, also on exception.
  std::vector<RowVectorPtr> result;
  readCursorAsync(
      *cursor,
      [addSplitsVoid = std::move(addSplits)](TaskCursor* cursor) {
        addSplitsVoid(cursor);
        return ContinueFuture::makeEmpty();
      },
      maxWaitMicros,
      [&](const RowVectorPtr& vector) { result.push_back(vector); });
  return {std::move(cursor), std::move(result)};
}

void readCursorAsync(
    TaskCursor& cursor,
    std::function<ContinueFuture(TaskCursor*)> addSplits,
    uint64_t maxWaitMicros,
    const std::function<void(const RowVectorPtr&)>& onBatch) {
  auto* task = cursor.task().get();
  cursor.start();
  auto future = ContinueFuture::makeEmpty();
  while (!cursor.noMoreSplits()) {
    if (future.valid()) {
      future.wait();
    }
    future = addSplits(&cursor);
    while (cursor.moveNext()) {
      const auto& vector = cursor.current();
      vector->loadedVector();
      onBatch(vector);
      testingMaybeTriggerAbort(task);
    }
  }

  if (!TaskMonitor::waitForCompletion(*task, maxWaitMicros)) {
    // NOTE: there is async memory arbitration might fail the task after all the
    // results have been consumed and before the task finishes. So we might run
    // into the failed task state in some rare case such as exposed by
    // concurrent memory arbitration test.
    if (task->state() != TaskState::kFinished &&
        task->state() != TaskState::kRunning) {
      waitForTaskDriversToFinish(task, maxWaitMicros);
      std::rethrow_exception(task->error());
    } else {
      VELOX_FAIL(
          "Failed to wait for task to complete after {}, task: {}",
          succinctMicros(maxWaitMicros),
          task->toString());
    }
  }
}

} // namespace facebook::velox::exec::test
