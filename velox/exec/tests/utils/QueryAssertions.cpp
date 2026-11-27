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

std::shared_ptr<Task> assertFlatMapResults(
    CursorParameters params,
    const core::PlanNodePtr& mapPlan,
    const core::PlanNodePtr& flatMapPlan,
    const std::vector<column_index_t>& flatMapChannels) {
  const auto noMoreSplits = [](TaskCursor* cursor) {
    cursor->setNoMoreSplits();
    return ContinueFuture::makeEmpty();
  };
  params.copyResult = false;
  params.planNode = mapPlan;
  auto mapCursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> expected;
  readCursorAsync(
      *mapCursor, noMoreSplits, 5'000'000, [&](const RowVectorPtr& vector) {
        expected.push_back(vector);
      });
  params.planNode = flatMapPlan;
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> results;
  readCursorAsync(
      *cursor, noMoreSplits, 5'000'000, [&](const RowVectorPtr& vector) {
        results.push_back(vector);
      });

  vector_size_t expectedBatch{0};
  vector_size_t expectedRow{0};
  for (const auto& result : results) {
    for (auto channel : flatMapChannels) {
      EXPECT_EQ(
          result->childAt(channel)->wrappedVector()->encoding(),
          VectorEncoding::Simple::FLAT_MAP)
          << "Column " << channel;
    }
    for (vector_size_t i = 0; i < result->size(); ++i) {
      while (expectedBatch < expected.size() &&
             expectedRow == expected[expectedBatch]->size()) {
        ++expectedBatch;
        expectedRow = 0;
      }
      if (expectedBatch == expected.size()) {
        ADD_FAILURE() << "More rows than expected";
        return cursor->task();
      }
      const auto& expectedVector = expected[expectedBatch];
      EXPECT_TRUE(expectedVector->equalValueAt(result.get(), expectedRow, i))
          << "Expected " << expectedVector->toString(expectedRow) << ", got "
          << result->toString(i);
      ++expectedRow;
    }
  }
  while (expectedBatch < expected.size() &&
         expectedRow == expected[expectedBatch]->size()) {
    ++expectedBatch;
    expectedRow = 0;
  }
  EXPECT_EQ(expectedBatch, expected.size()) << "Fewer rows than expected";
  return cursor->task();
}

} // namespace facebook::velox::exec::test
