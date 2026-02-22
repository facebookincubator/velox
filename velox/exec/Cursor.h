/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

/// Parameters for initializing a TaskCursor or RowCursor.
struct CursorParameters {
  /// Root node of the plan tree
  std::shared_ptr<const core::PlanNode> planNode;

  /// Partition number if task is expected to receive data from a remote data
  /// shuffle. Used to initialize ExchangeClient.
  int32_t destination{0};

  /// Maximum number of drivers per pipeline.
  int32_t maxDrivers{1};

  /// The max capacity of the query memory pool.
  int64_t maxQueryCapacity{memory::kMaxMemory};

  /// Maximum number of split groups processed concurrently.
  int32_t numConcurrentSplitGroups{1};

  /// Optional, created if not present.
  std::shared_ptr<core::QueryCtx> queryCtx;

  uint64_t bufferedBytes{512 * 1024};

  /// An optional memory pool to be used to allocate vectors returned by
  /// MultiThreadedTaskCursor. A new pool is created if not specified.
  ///
  /// Only used if serialExecution is false.
  std::shared_ptr<memory::MemoryPool> outputPool;

  /// Ungrouped (by default) or grouped (bucketed) execution.
  core::ExecutionStrategy executionStrategy{
      core::ExecutionStrategy::kUngrouped};

  /// Contains leaf plan nodes that need to be executed in the grouped mode.
  std::unordered_set<core::PlanNodeId> groupedExecutionLeafNodeIds;

  /// Number of splits groups the task will be processing. Must be 1 for
  /// ungrouped execution.
  int numSplitGroups{1};

  /// Spilling directory, if not empty, then the task's spilling directory
  /// would be built from it.
  std::string spillDirectory = "";

  /// Callback function to dynamically create or determine the spill directory
  /// path at runtime. If provided, this callback is invoked when spilling is
  /// needed and must return a valid directory path. This allows for dynamic
  /// spill directory creation or path resolution based on runtime conditions.
  std::function<std::string()> spillDirectoryCallback = nullptr;

  bool copyResult = true;

  /// If true, use serial execution mode. Use parallel execution mode
  /// otherwise.
  bool serialExecution = false;

  bool barrierExecution = false;

  /// If both 'queryConfigs' and 'queryCtx' are specified, the configurations
  /// in 'queryCtx' will be overridden by 'queryConfig'.
  std::unordered_map<std::string, std::string> queryConfigs = {};

  // Debugging related structures:

  /// Callback type for breakpoints.
  ///
  /// Called with the current vector when the breakpoint is hit. The return
  /// value semantics is "should block?"; returns true if the driver should stop
  /// and produce the vector, false to continue without stopping.
  ///
  /// If callback is not specified (nullptr), assume the partial vector should
  /// always be produced, which is the same as callback returning true (always
  /// stop).
  using BreakpointCallback = std::function<bool(const RowVectorPtr&)>;

  /// Map type for breakpoints: plan node ID to optional callback.
  using TBreakpointMap =
      std::unordered_map<core::PlanNodeId, BreakpointCallback>;

  /// Breakpoints enable step-by-step execution of a query plan, allowing users
  /// to inspect intermediate results at operator boundaries containing
  /// breakpoints. This is useful for debugging query execution and
  /// understanding data flow through operators.
  ///
  /// Maps plan node IDs to optional callbacks. When a breakpoint is hit, the
  /// callback (if non-null) is invoked with the current vector before the
  /// cursor pauses.
  TBreakpointMap breakpoints = {};
};

/// Abstract interface for iterating over query results. TaskCursor manages
/// task execution and provides batch-level access to output vectors.
///
/// Example usage:
/// @code
///
///   auto cursor = TaskCursor:create({
///     .planNode = node,
///   );
///
///   // Run through every output.
///   while (cursor->moveNext()) {
///     auto vector = cursor->current();
///   }
/// @endcode
///
/// If "breakpoints" are set in the CursorParameters input, then
/// `cursor->moveStep()` will move the cursor to the next breakpoint, which is
/// either the input of an operator with a breakpoint installed, or the next
/// task output.
///
/// `cursor->moveNext()` will always move the cursor to the next task output.
class TaskCursor {
 public:
  virtual ~TaskCursor() = default;

  static std::unique_ptr<TaskCursor> create(const CursorParameters& params);

  /// Starts the task if not started yet.
  virtual void start() = 0;

  /// Fetches another batch from the task queue. Starts the task if not started
  /// yet.
  ///
  /// @return Returns false is the task is done producing output.
  virtual bool moveNext() = 0;

  /// Steps through execution, returning either the input to the next operator
  /// with a breakpoint installed, or the next task output. If no breakpoints
  /// are set, then moveStep() == moveNext().
  ///
  /// If @planId is non-empty, only stops at a breakpoint whose plan node ID
  /// matches @planId; breakpoints for other plan nodes are skipped
  /// (unblocked) automatically. When empty (the default), stops at the next
  /// breakpoint regardless of plan node ID.
  ///
  /// @return Returns false is the task is done producing output.
  virtual bool moveStep(const core::PlanNodeId& planId = "") = 0;

  /// Returns the vector the cursor is currently on.
  virtual RowVectorPtr& current() = 0;

  /// If breakpoints are set, returns the plan node that generated the trace. If
  /// the cursor is at the task output or if there are no breakpoints,
  /// returns empty string.
  virtual core::PlanNodeId at() const = 0;

  virtual void setError(std::exception_ptr error) = 0;

  virtual bool noMoreSplits() const = 0;

  virtual void setNoMoreSplits() = 0;

  virtual const std::shared_ptr<Task>& task() = 0;
};

/// Row-level cursor that wraps a TaskCursor and provides access to individual
/// rows and column values within the result set.
class RowCursor {
 public:
  explicit RowCursor(CursorParameters& params) {
    cursor_ = TaskCursor::create(params);
  }

  bool isNullAt(int32_t columnIndex) const {
    checkOnRow();
    return decoded_[columnIndex]->isNullAt(currentRow_);
  }

  template <typename T>
  T valueAt(int32_t columnIndex) const {
    checkOnRow();
    return decoded_[columnIndex]->valueAt<T>(currentRow_);
  }

  bool next();

  std::shared_ptr<Task> task() const {
    return cursor_->task();
  }

 private:
  void checkOnRow() const {
    VELOX_CHECK(
        currentRow_ >= 0 && currentRow_ < numRows_, "Cursor not on row.");
  }

  std::unique_ptr<TaskCursor> cursor_;
  std::vector<std::unique_ptr<DecodedVector>> decoded_;
  SelectivityVector allRows_;
  vector_size_t currentRow_ = 0;
  vector_size_t numRows_ = 0;
};

/// Wait up to maxWaitMicros for all the task drivers to finish. The function
/// returns true if all the drivers have finished, otherwise false.
///
/// NOTE: user must call this on a finished or failed task.
bool waitForTaskDriversToFinish(
    exec::Task* task,
    uint64_t maxWaitMicros = 1'000'000);

} // namespace facebook::velox::exec
