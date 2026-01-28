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

#include <atomic>
#include <unordered_set>

#include "velox/core/PlanFragment.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Task.h"
#include "velox/exec/trace/TraceCtx.h"

namespace facebook::velox::exec {

/// A debugging cursor for interactive task execution.
///
/// TaskDebuggerCursor enables step-by-step execution of a query plan, allowing
/// users to inspect intermediate results at traced operator boundaries. This is
/// useful for debugging query execution and understanding data flow through
/// operators.
///
/// The cursor uses a custom tracing context that pauses execution at traced
/// operators, allowing inspection of input vectors before they are processed.
///
/// Example usage:
/// @code
///
///   TaskDebuggerCursor cursor(planFragment, {"planNodeId1", "plaNodeId10"});
///
///   auto vector = cursor.step(); // advances until the next "breakpoint";
///                                // either input of a traced operator or task
///                                // output.
///
///   auto vector = cursor.next(); // advances until the next task output.
///
///   // Get every result, from traced operators and output:
///   while (auto vector = cursor.step()) {
///   }
/// @endcode
///
/// @note This class assumes serial (single-threaded) execution mode.
/// @note The cursor maintains ownership of the underlying task and ensures
///       proper cleanup in the destructor.
class TaskDebuggerCursor {
 public:
  /// Constructs a TaskDebuggerCursor for the given plan fragment.
  ///
  /// @param planFragment The plan fragment to execute. This contains the
  ///        query plan and associated metadata.
  /// @param tracedPlanNodeIds A list of plan node IDs where execution should
  ///        pause to allow inspection of intermediate results. Only operators
  ///        corresponding to these node IDs will be traced.
  TaskDebuggerCursor(
      core::PlanFragment planFragment,
      const std::vector<core::PlanNodeId>& tracedPlanNodeIds) {
    static std::atomic_int32_t cursorId{0};
    taskId_ = fmt::format("debug_cursor_{}", ++cursorId);

    auto queryCtx =
        core::QueryCtx::Builder()
            .queryConfig(
                core::QueryConfig({
                    {core::QueryConfig::kQueryTraceEnabled, "true"},
                }))
            .traceCtxProvider([&](core::QueryCtx&, const core::PlanFragment&) {
              return std::make_unique<TaskDebuggerTraceCtx>(
                  tracedPlanNodeIds, traceState_);
            })
            .build();

    task_ = Task::create(
        taskId_,
        std::move(planFragment),
        0,
        std::move(queryCtx),
        Task::ExecutionMode::kSerial);
  }

  /// Ensures the task completes before cleanup.
  ~TaskDebuggerCursor() {
    if (task_) {
      waitForTaskDriversToFinish(task_.get());
    }
  }

  TaskDebuggerCursor(TaskDebuggerCursor&&) noexcept = default;
  TaskDebuggerCursor& operator=(TaskDebuggerCursor&&) noexcept = default;

  /// Retrieves the next complete result vector from the task.
  ///
  /// This method advances execution until a final output vector is produced,
  /// skipping over any intermediate traced results. Use this when you want
  /// to get the final query results without pausing at traced operators.
  ///
  /// @return The next result vector, or nullptr if execution is complete.
  RowVectorPtr next() {
    return advance(false);
  }

  /// Steps through execution, returning either a traced intermediate result
  /// or a final output vector.
  ///
  /// This method advances execution until either:
  /// - A traced operator produces an input vector (returns the traced input)
  /// - A final output vector is produced (returns the output)
  /// - Execution completes (returns nullptr)
  ///
  /// When a traced result is returned, execution is paused at that point.
  /// Call step() again to continue execution from where it left off.
  ///
  /// @return The next intermediate or final result vector, or nullptr if
  ///         execution is complete.
  RowVectorPtr step() {
    return advance(true);
  }

 private:
  // Advance to the next vector to produce. If `isStep` is true, move to the
  // next trace point or task output. If false, moves to the next task output.
  RowVectorPtr advance(bool isStep) {
    if (traceState_.traceData) {
      traceState_.traceData = nullptr;
      traceState_.tracePromise.setValue();
    }

    while (true) {
      ContinueFuture future = ContinueFuture::makeEmpty();

      if (auto vector = task_->next(&future)) {
        return vector;
      }

      // When we hit a tracing point, the driver will return nullptr, set a
      // future, and the trace implementation will capture state in traceState_.
      if (traceState_.traceData) {
        if (isStep) {
          return traceState_.traceData;
        }

        // Signal the task driver to unblock.
        traceState_.traceData = nullptr;
        traceState_.tracePromise.setValue();
      }

      // Wait until the task future is unblocked.
      if (future.valid()) {
        future.wait();
      } else {
        // When no vector was produced and the future is not valid, it's the
        // task signal that it has finished producing output.
        break;
      }
    }
    return nullptr;
  }

  /// Internal state for coordinating between the tracer and cursor.
  ///
  /// This struct manages the synchronization between the trace writer
  /// (which produces intermediate results) and the cursor (which consumes
  /// them).
  struct TraceState {
    /// Promise used to signal the tracer to continue after a partial result
    /// has been consumed.
    ContinuePromise tracePromise{ContinuePromise::makeEmpty()};

    /// The most recent intermediate result from a traced operator.
    RowVectorPtr traceData;
  } traceState_;

  // Custom trace context implementation for the debugger.
  //
  // This trace context pauses execution at traced operators by blocking
  // the trace writer until the cursor consumes the intermediate result.
  class TaskDebuggerTraceCtx : public trace::TraceCtx {
   public:
    // Constructs a trace context for the specified plan nodes.
    //
    // @param tracedIds The plan node IDs to trace.
    // @param traceState Reference to the shared trace state for coordination.
    TaskDebuggerTraceCtx(
        const std::vector<core::PlanNodeId>& tracedIds,
        TraceState& traceState)
        : TraceCtx(false),
          tracedIds_(tracedIds.begin(), tracedIds.end()),
          traceState_(traceState) {}

    // Determines whether a given operator should be traced.
    //
    // @param op The operator to check.
    // @return true if the operator's plan node ID is in the traced set.
    bool shouldTrace(const Operator& op) const override {
      return tracedIds_.contains(op.planNodeId());
    }

    // Creates an input trace writer for the given operator.
    //
    // @param op The operator to create a tracer for.
    // @return A unique pointer to the trace input writer.
    std::unique_ptr<trace::TraceInputWriter> createInputTracer(
        Operator& op) const override {
      return std::make_unique<TaskDebuggerTraceInputWriter>(
          op.planNodeId(), traceState_);
    }

   private:
    // Trace writer that captures input vectors and pauses execution.
    //
    // When an input vector is written, this writer stores it in the shared
    // trace state and blocks until the cursor signals to continue.
    class TaskDebuggerTraceInputWriter : public trace::TraceInputWriter {
     public:
      TaskDebuggerTraceInputWriter(
          const core::PlanNodeId& planId,
          TraceState& traceState)
          : planId_(planId), traceState_(traceState) {}

      // Writes an input vector and pauses execution.
      //
      // Stores the vector in the trace state and creates a future that
      // blocks until the cursor consumes the result and signals continuation.
      //
      // @param vector The input vector to trace.
      // @param future Output parameter set to a future that blocks until
      //        the cursor is ready to continue.
      // @return true to indicate the writer is blocked waiting for the future.
      bool write(const RowVectorPtr& vector, ContinueFuture* future) override {
        VELOX_CHECK(traceState_.tracePromise.isFulfilled());

        traceState_.traceData = vector;
        traceState_.tracePromise = ContinuePromise("TaskQueue::dequeue");
        *future = traceState_.tracePromise.getFuture();
        return true;
      }

      // Called when tracing is complete for this operator.
      void finish() override {}

     private:
      const core::PlanNodeId planId_;
      TraceState& traceState_;
    };

    std::unordered_set<core::PlanNodeId> tracedIds_;
    TraceState& traceState_;
  };

  std::shared_ptr<exec::Task> task_;
  std::string taskId_;
};

} // namespace facebook::velox::exec
