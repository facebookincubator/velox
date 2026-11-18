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
#endif

namespace facebook::velox::exec::test {

/// Converts input 'RowVector' into a list of 'MaterializedRow's.
std::vector<MaterializedRow> materialize(const RowVectorPtr& vector);

/// Converts a list of 'RowVector's into 'MaterializedRowMultiset'.
MaterializedRowMultiset materialize(const std::vector<RowVectorPtr>& vectors);

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
#endif // VELOX_ENABLE_BACKWARD_COMPATIBILITY

void assertResults(
    const std::vector<RowVectorPtr>& results,
    const RowTypePtr& resultType,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner);

void assertResultsOrdered(
    const std::vector<RowVectorPtr>& results,
    const RowTypePtr& resultType,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner,
    const std::vector<uint32_t>& sortingKeys);

/// assertEqualResults() has limited support for results with floating-point
/// columns.
///   1. When there is one or more floating-point columns, we try to group the
///   rows in each set by the non-floating-point columns. If every group
///   contains only one row in both sets, we compare their rows
///   via the sort-merge algorithm with epsilon comparator.
///   2. If there is no floating-point column, we loop over every row in one set
///   and try to find it in the other via std::multiset::find(). Values are
///   compared directly without epsilon.
/// We made this difference because some operations, such as aggregation,
/// require tolerance to imprecision of floating-point computation in their
/// results, while epsilon comparator doesn't satisfy the strict weak ordering
/// requirement of std::multiset. Since aggregation results typically have one
/// row per group, we use sort-merge algorithm with epsilon comparator to verify
/// their results. This condition may be relaxed if there is only one
/// floating-point column in each set. Verifying arbitrary result sets with
/// epsilon comparison would require more advanced algorithms such as maximum
/// bipartite matching. Hence we leave them as future work. Check out
/// https://github.com/facebookincubator/velox/issues/3493 for more discussion.
bool assertEqualResults(
    const std::vector<RowVectorPtr>& expected,
    const std::vector<RowVectorPtr>& actual);

bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedRowType,
    const std::vector<RowVectorPtr>& actual);

bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedType,
    const MaterializedRowMultiset& actualRows,
    const TypePtr& actualType,
    const std::string& message);

} // namespace facebook::velox::exec::test
