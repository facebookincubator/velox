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

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/DuckDbQueryRunner.h"
#include "velox/vector/ComplexVector.h"

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/vector/tests/utils/VectorAssertions.h"
#endif

namespace facebook::velox::exec::test {

/// Scoped abort percentage utility that allows user to trigger abort during the
///  query execution.
/// 'abortPct' specifies the probability of triggering abort. 100% means
/// abort will always be triggered.
/// 'maxInjections' indicates the max number of actual triggering, e.g. when
/// 'abortPct' is 20 and 'maxInjections' is 10, continuous calls to
/// testingMaybeTriggerAbort() will keep rolling the dice that has a chance of
/// 20% triggering until 10 triggers have been invoked.
/// 'hook' will be invoked whenever abort is triggered, if not nullptr.
class TestScopedAbortInjection {
 public:
  explicit TestScopedAbortInjection(
      int32_t abortPct,
      int32_t maxInjections = std::numeric_limits<int32_t>::max(),
      std::function<void(Task*)> hook = nullptr);

  ~TestScopedAbortInjection();
};

std::pair<std::unique_ptr<TaskCursor>, std::vector<RowVectorPtr>> readCursor(
    const CursorParameters& params,
    std::function<void(TaskCursor*)> addSplits =
        [](TaskCursor* taskCursor) {
          if (taskCursor->noMoreSplits()) {
            return;
          }
          taskCursor->setNoMoreSplits();
        },
    uint64_t maxWaitMicros = 5'000'000);

/// Runs the query of 'cursor', passing each output batch to 'onBatch' as it
/// arrives. The batches borrow memory from 'cursor', so any batch 'onBatch'
/// keeps must be released before 'cursor' is destroyed, including when this
/// function throws.
void readCursorAsync(
    TaskCursor& cursor,
    std::function<ContinueFuture(TaskCursor*)> addSplits,
    uint64_t maxWaitMicros,
    const std::function<void(const RowVectorPtr&)>& onBatch);

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
// TODO: Remove once Prestissimo migrates.
/// Legacy forms of TaskMonitor methods.
inline bool waitForTaskCompletion(
    exec::Task* task,
    uint64_t maxWaitMicros = TaskMonitor::kDefaultMaxWaitMicros) {
  return TaskMonitor::waitForCompletion(*task, maxWaitMicros);
}

inline bool waitForTaskFailure(
    exec::Task* task,
    uint64_t maxWaitMicros = TaskMonitor::kDefaultMaxWaitMicros) {
  return TaskMonitor::waitForFailure(*task, maxWaitMicros);
}

inline bool waitForTaskStateChange(
    exec::Task* task,
    TaskState state,
    uint64_t maxWaitMicros = TaskMonitor::kDefaultMaxWaitMicros) {
  return TaskMonitor::waitForStateChange(*task, state, maxWaitMicros);
}

inline void waitForAllTasksToBeDeleted(uint64_t maxWaitUs = 3'000'000) {
  TaskMonitor::waitForAllTasksDeleted(maxWaitUs);
}

/// Legacy form of VectorAssertions::equalMultisets against DuckDB results.
inline void assertResults(
    const std::vector<RowVectorPtr>& results,
    const RowTypePtr& resultType,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner) {
  velox::test::VectorAssertions::equalMultisets(
      duckDbQueryRunner.execute(duckDbSql, resultType), resultType, results);
}
#endif // VELOX_ENABLE_BACKWARD_COMPATIBILITY

/// Same as VectorAssertions::equalMultisets.
bool assertEqualResults(
    const std::vector<RowVectorPtr>& expected,
    const std::vector<RowVectorPtr>& actual);

/// Same as VectorAssertions::equalMultisets.
bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedRowType,
    const std::vector<RowVectorPtr>& actual);

} // namespace facebook::velox::exec::test
