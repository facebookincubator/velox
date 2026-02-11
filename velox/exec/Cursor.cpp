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

#include "velox/exec/Cursor.h"

#include <folly/system/HardwareConcurrency.h>
#include <filesystem>

#include "velox/common/file/FileSystems.h"
#include "velox/vector/EncodedVectorCopy.h"

namespace facebook::velox::exec {
namespace {

class TaskQueue {
 public:
  struct TaskQueueEntry {
    RowVectorPtr vector;
    uint64_t bytes;
  };

  explicit TaskQueue(
      uint64_t maxBytes,
      const std::shared_ptr<memory::MemoryPool>& outputPool)
      : pool_(
            outputPool != nullptr ? outputPool
                                  : memory::memoryManager()->addLeafPool()),
        maxBytes_(maxBytes) {}

  void setNumProducers(int32_t n) {
    numProducers_ = n;
  }

  // Adds a batch of rows to the queue and returns kNotBlocked if the
  // producer may continue. Returns kWaitForConsumer if the queue is
  // full after the addition and sets '*future' to a future that is
  // realized when the producer may continue.
  exec::BlockingReason
  enqueue(RowVectorPtr vector, bool drained, velox::ContinueFuture* future);

  // Returns nullptr when all producers are at end. Otherwise blocks.
  RowVectorPtr dequeue();

  void close();

  bool hasNext();

  velox::memory::MemoryPool* pool() const {
    return pool_.get();
  }

  std::atomic<int32_t> producersDrainFinished_ = 0;

 private:
  // Owns the vectors in 'queue_', hence must be declared first.
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::deque<TaskQueueEntry> queue_;
  std::optional<int32_t> numProducers_;
  int32_t producersFinished_ = 0;
  uint64_t totalBytes_ = 0;
  // Blocks the producer if 'totalBytes' exceeds 'maxBytes' after
  // adding the result.
  uint64_t maxBytes_;
  std::mutex mutex_;
  std::vector<ContinuePromise> producerUnblockPromises_;
  bool consumerBlocked_ = false;
  ContinuePromise consumerPromise_{ContinuePromise::makeEmpty()};
  ContinueFuture consumerFuture_;
  bool closed_ = false;
};

exec::BlockingReason TaskQueue::enqueue(
    RowVectorPtr vector,
    bool drained,
    velox::ContinueFuture* future) {
  if (!vector) {
    std::lock_guard<std::mutex> l(mutex_);

    if (!drained) {
      ++producersFinished_;
    } else {
      ++producersDrainFinished_;
    }
    if (consumerBlocked_) {
      consumerBlocked_ = false;
      consumerPromise_.setValue();
    }
    return exec::BlockingReason::kNotBlocked;
  }

  auto bytes = vector->retainedSize();
  TaskQueueEntry entry{std::move(vector), bytes};

  std::lock_guard<std::mutex> l(mutex_);
  // Check inside 'mutex_'
  if (closed_) {
    throw std::runtime_error("Consumer cursor is closed");
  }
  queue_.push_back(std::move(entry));
  totalBytes_ += bytes;
  if (consumerBlocked_) {
    consumerBlocked_ = false;
    consumerPromise_.setValue();
  }
  if (totalBytes_ > maxBytes_) {
    auto [unblockPromise, unblockFuture] =
        makeVeloxContinuePromiseContract("TaskQueue::enqueue");
    producerUnblockPromises_.emplace_back(std::move(unblockPromise));
    *future = std::move(unblockFuture);
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr TaskQueue::dequeue() {
  for (;;) {
    RowVectorPtr vector;
    std::vector<ContinuePromise> mayContinue;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (closed_) {
        return nullptr;
      }

      if (!queue_.empty()) {
        auto result = std::move(queue_.front());
        queue_.pop_front();
        totalBytes_ -= result.bytes;
        vector = std::move(result.vector);
        if (totalBytes_ < maxBytes_ / 2) {
          mayContinue = std::move(producerUnblockPromises_);
        }
      } else if (
          numProducers_.has_value() && producersFinished_ == numProducers_) {
        return nullptr;
      } else if (producersDrainFinished_ == numProducers_) {
        return nullptr;
      }

      if (!vector) {
        consumerBlocked_ = true;
        consumerPromise_ = ContinuePromise("TaskQueue::dequeue");
        consumerFuture_ = consumerPromise_.getFuture();
      }
    }
    // outside of 'mutex_'
    for (auto& promise : mayContinue) {
      promise.setValue();
    }
    if (vector) {
      return vector;
    }
    consumerFuture_.wait();
  }
}

void TaskQueue::close() {
  std::lock_guard<std::mutex> l(mutex_);
  closed_ = true;
  // Unblock producers.
  for (auto& promise : producerUnblockPromises_) {
    promise.setValue();
  }
  producerUnblockPromises_.clear();

  // Unblock consumers.
  if (consumerBlocked_) {
    consumerBlocked_ = false;
    consumerPromise_.setValue();
  }
}

bool TaskQueue::hasNext() {
  std::lock_guard<std::mutex> l(mutex_);
  return !queue_.empty();
}

class TaskCursorBase : public TaskCursor {
 public:
  TaskCursorBase(
      const CursorParameters& params,
      const std::shared_ptr<folly::Executor>& executor) {
    static std::atomic<int32_t> cursorId;
    taskId_ = fmt::format("test_cursor_{}", ++cursorId);

    if (params.queryCtx) {
      queryCtx_ = params.queryCtx;
    } else {
      // NOTE: the destructor of 'executor_' will wait for all the async task
      // activities to finish on TaskCursor destruction.
      executor_ = executor;
      static std::atomic<uint64_t> cursorQueryId{0};
      queryCtx_ = core::QueryCtx::create(
          executor_.get(),
          core::QueryConfig({}),
          std::
              unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
          cache::AsyncDataCache::getInstance(),
          nullptr,
          nullptr,
          fmt::format("TaskCursorQuery_{}", cursorQueryId++));
    }

    // If query configs needs to be overwritten in queryCtx.
    if (!params.queryConfigs.empty() || !params.breakpoints.empty()) {
      auto configCopy = !params.queryConfigs.empty()
          ? params.queryConfigs
          : queryCtx_->queryConfig().rawConfigsCopy();

      if (!params.breakpoints.empty()) {
        configCopy.insert({core::QueryConfig::kQueryTraceEnabled, "true"});
      }
      queryCtx_->testingOverrideConfigUnsafe(std::move(configCopy));
    }

    planFragment_ = {
        params.planNode,
        params.executionStrategy,
        params.numSplitGroups,
        params.groupedExecutionLeafNodeIds};

    if (!params.spillDirectory.empty()) {
      taskSpillDirectory_ = params.spillDirectory + "/" + taskId_;
      taskSpillDirectoryCb_ = params.spillDirectoryCallback;
      if (taskSpillDirectoryCb_ == nullptr) {
        auto fileSystem =
            velox::filesystems::getFileSystem(taskSpillDirectory_, nullptr);
        VELOX_CHECK_NOT_NULL(fileSystem, "File System is null!");
        try {
          fileSystem->mkdir(taskSpillDirectory_);
        } catch (...) {
          LOG(ERROR) << "Faield to create task spill directory "
                     << taskSpillDirectory_ << " base director "
                     << params.spillDirectory << " exists["
                     << std::filesystem::exists(taskSpillDirectory_) << "]";

          std::rethrow_exception(std::current_exception());
        }

        LOG(INFO) << "Task spill directory[" << taskSpillDirectory_
                  << "] created";
      }
    }
  }

 protected:
  std::string taskId_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  core::PlanFragment planFragment_;
  std::string taskSpillDirectory_;
  std::function<std::string()> taskSpillDirectoryCb_;

 private:
  std::shared_ptr<folly::Executor> executor_;
};

class MultiThreadedTaskCursor : public TaskCursorBase {
 public:
  explicit MultiThreadedTaskCursor(const CursorParameters& params)
      : TaskCursorBase(
            params,
            std::make_shared<folly::CPUThreadPoolExecutor>(
                folly::hardware_concurrency())),
        maxDrivers_{params.maxDrivers},
        numConcurrentSplitGroups_{params.numConcurrentSplitGroups},
        numSplitGroups_{params.numSplitGroups} {
    VELOX_CHECK(!params.serialExecution);
    VELOX_CHECK(
        queryCtx_->isExecutorSupplied(),
        "Executor should be set in parallel task cursor");

    queue_ =
        std::make_shared<TaskQueue>(params.bufferedBytes, params.outputPool);

    // Captured as a shared_ptr by the consumer callback of task_.
    auto queueHolder = std::weak_ptr(queue_);
    std::optional<common::SpillDiskOptions> spillDiskOpts;
    if (!taskSpillDirectory_.empty()) {
      spillDiskOpts = common::SpillDiskOptions{
          .spillDirPath = taskSpillDirectory_,
          .spillDirCreated = taskSpillDirectoryCb_ == nullptr,
          .spillDirCreateCb = taskSpillDirectoryCb_};
    }
    task_ = Task::create(
        taskId_,
        std::move(planFragment_),
        params.destination,
        std::move(queryCtx_),
        Task::ExecutionMode::kParallel,
        // consumer
        [this, queueHolder, copyResult = params.copyResult, taskId = taskId_](
            const RowVectorPtr& vector,
            bool drained,
            velox::ContinueFuture* future) {
          auto queue = queueHolder.lock();
          if (queue == nullptr) {
            LOG(ERROR) << "TaskQueue has been destroyed, taskId: " << taskId;
            return exec::BlockingReason::kNotBlocked;
          }

          if (!vector || !copyResult) {
            return queue->enqueue(vector, drained, future);
          }
          VectorPtr copy = encodedVectorCopy(
              {.pool = queue->pool(), .reuseSource = false}, vector);
          return queue->enqueue(
              std::static_pointer_cast<RowVector>(std::move(copy)),
              drained,
              future);
        },
        0,
        std::move(spillDiskOpts),
        [queueHolder, taskId = taskId_](std::exception_ptr) {
          // onError close the queue to unblock producers and consumers.
          // moveNext will handle rethrowing the error once it's
          // unblocked.
          auto queue = queueHolder.lock();
          if (queue == nullptr) {
            LOG(ERROR) << "TaskQueue has been destroyed, taskId: " << taskId;
            return;
          }
          queue->close();
        });
  }

  ~MultiThreadedTaskCursor() override {
    queue_->close();
    if (task_ && !atEnd_) {
      task_->requestCancel();
    }
  }

  /// Starts the task if not started yet.
  void start() override {
    if (!started_) {
      started_ = true;
      try {
        task_->start(maxDrivers_, numConcurrentSplitGroups_);
        queue_->setNumProducers(numSplitGroups_ * task_->numOutputDrivers());
      } catch (const VeloxException& e) {
        // Could not find output pipeline, due to Task terminated before
        // start. Do not override the error.
        if (e.message().find("Output pipeline not found for task") ==
            std::string::npos) {
          throw;
        }
      }
    }
  }

  /// Fetches another batch from the task queue.
  /// Starts the task if not started yet.
  bool moveNext() override {
    start();
    if (error_) {
      std::rethrow_exception(error_);
    }

    // Task might be aborted before start.
    checkTaskError();
    current_ = queue_->dequeue();

    checkTaskError();
    if (!current_) {
      if (queue_->producersDrainFinished_ > 0) {
        queue_->producersDrainFinished_ = 0;
        return false;
      }
      atEnd_ = true;
    }
    return current_ != nullptr;
  }

  bool moveStep() override {
    return moveNext();
  }

  void setNoMoreSplits() override {
    VELOX_CHECK(!noMoreSplits_);
    noMoreSplits_ = true;
  }

  bool noMoreSplits() const override {
    return noMoreSplits_;
  }

  RowVectorPtr& current() override {
    return current_;
  }

  core::PlanNodeId at() const override {
    return ""; // always at task output.
  }

  void setError(std::exception_ptr error) override {
    error_ = error;
    if (task_) {
      task_->setError(error);
    }
  }

  const std::shared_ptr<Task>& task() override {
    return task_;
  }

 private:
  void checkTaskError() {
    if (!task_->error()) {
      return;
    }
    // Wait for the task to finish (there's' a small period of time between
    // when the error is set on the Task and terminate is called).
    task_->taskCompletionFuture()
        .within(std::chrono::microseconds(5'000'000))
        .wait();

    // Wait for all task drivers to finish to avoid destroying the executor_
    // before task_ finished using it and causing a crash.
    waitForTaskDriversToFinish(task_.get());
    std::rethrow_exception(task_->error());
  }

  const int32_t maxDrivers_;
  const int32_t numConcurrentSplitGroups_;
  const int32_t numSplitGroups_;

  bool started_{false};
  std::shared_ptr<TaskQueue> queue_;
  std::shared_ptr<exec::Task> task_;
  RowVectorPtr current_;
  bool atEnd_{false};
  tsan_atomic<bool> noMoreSplits_{false};
  std::exception_ptr error_;
};

class SingleThreadedTaskCursor : public TaskCursorBase {
 public:
  explicit SingleThreadedTaskCursor(const CursorParameters& params)
      : TaskCursorBase(params, nullptr) {
    VELOX_CHECK(params.serialExecution);
    VELOX_CHECK(
        !queryCtx_->isExecutorSupplied(),
        "Executor should not be set in serial task cursor");
    std::optional<common::SpillDiskOptions> spillDiskOpts;
    if (!taskSpillDirectory_.empty()) {
      spillDiskOpts = common::SpillDiskOptions{
          .spillDirPath = taskSpillDirectory_,
          .spillDirCreated = true,
          .spillDirCreateCb = taskSpillDirectoryCb_};
    }
    task_ = Task::create(
        taskId_,
        std::move(planFragment_),
        params.destination,
        std::move(queryCtx_),
        Task::ExecutionMode::kSerial,
        std::function<BlockingReason(RowVectorPtr, bool, ContinueFuture*)>{},
        0,
        std::move(spillDiskOpts));

    VELOX_CHECK(
        task_->supportSerialExecutionMode(),
        "Plan doesn't support serial execution mode");
  }

  ~SingleThreadedTaskCursor() override {
    if (task_) {
      task_->requestCancel().wait();
    }
  }

  void start() override {
    // no-op
  }

  void setNoMoreSplits() override {
    VELOX_CHECK(!noMoreSplits_);
    noMoreSplits_ = true;
  }

  bool noMoreSplits() const override {
    return noMoreSplits_;
  }

  bool moveNext() override {
    if (!task_->isRunning()) {
      return false;
    }

    while (true) {
      ContinueFuture future = ContinueFuture::makeEmpty();
      RowVectorPtr next = task_->next(&future);
      if (next != nullptr) {
        current_ = next;
        return true;
      }
      // When next is returned from task as a null pointer.
      if (!future.valid()) {
        VELOX_CHECK(!task_->isRunning() || !noMoreSplits_);
        return false;
      }
      // Task is blocked for some reason. Wait and try again.
      VELOX_CHECK_NULL(next);
      future.wait();
    }
    return false;
  }

  bool moveStep() override {
    return moveNext();
  }

  RowVectorPtr& current() override {
    return current_;
  }

  core::PlanNodeId at() const override {
    return ""; // always at task output.
  }

  void setError(std::exception_ptr error) override {
    error_ = error;
    if (task_) {
      task_->setError(error);
    }
  }

  const std::shared_ptr<Task>& task() override {
    return task_;
  }

 private:
  std::shared_ptr<exec::Task> task_;
  bool noMoreSplits_{false};
  RowVectorPtr current_;
  std::exception_ptr error_;
};

/// A debugging cursor for interactive task execution.
///
/// The cursor uses a custom tracing context that pauses execution at traced
/// operators, allowing inspection of input vectors before they are processed.
///
/// @note This class assumes serial (single-threaded) execution mode.
class TaskDebuggerCursor : public TaskCursorBase {
 public:
  explicit TaskDebuggerCursor(const CursorParameters& params)
      : TaskCursorBase(params, nullptr) {
    // Installs the required trace provider.
    queryCtx_->setTraceCtxProvider(
        [&](core::QueryCtx&, const core::PlanFragment&) {
          return std::make_unique<TaskDebuggerTraceCtx>(
              params.breakpoints, traceState_);
        });

    task_ = Task::create(
        taskId_,
        std::move(planFragment_),
        params.destination,
        std::move(queryCtx_),
        Task::ExecutionMode::kSerial);
  }

  /// Ensures the task completes before cleanup.
  ~TaskDebuggerCursor() {
    if (task_) {
      task_->requestCancel().wait();
    }
  }

  TaskDebuggerCursor(TaskDebuggerCursor&&) noexcept = default;
  TaskDebuggerCursor& operator=(TaskDebuggerCursor&&) noexcept = default;

  // no-op
  void start() override {}

  bool moveNext() override {
    return advance(false);
  }

  bool moveStep() override {
    return advance(true);
  }

  RowVectorPtr& current() override {
    return current_;
  }

  core::PlanNodeId at() const override {
    return traceState_.planId;
  }

  void setNoMoreSplits() override {
    VELOX_CHECK(!noMoreSplits_);
    noMoreSplits_ = true;
  }

  bool noMoreSplits() const override {
    return noMoreSplits_;
  }

  void setError(std::exception_ptr error) override {
    error_ = error;
    if (task_) {
      task_->setError(error);
    }
  }

  const std::shared_ptr<Task>& task() override {
    return task_;
  }

 private:
  // Advance to the next vector to produce, storing it in `current_`. If
  // `isStep` is true, move to the next trace point or task output. If false,
  // moves to the next task output.
  //
  // Returns false when the task is done producing output.
  bool advance(bool isStep) {
    if (error_) {
      std::rethrow_exception(error_);
    }

    if (traceState_.traceData) {
      traceState_.traceData = nullptr;
      traceState_.tracePromise.setValue();
    }

    while (true) {
      ContinueFuture future = ContinueFuture::makeEmpty();

      if (auto vector = task_->next(&future)) {
        current_ = vector;
        traceState_.planId.clear();
        return true;
      }

      // When we hit a tracing point, the driver will return nullptr, set a
      // future, and the trace implementation will capture state in traceState_.
      if (traceState_.traceData) {
        if (isStep) {
          current_ = traceState_.traceData;
          return true;
        }

        // Signal the task driver to unblock.
        traceState_.traceData = nullptr;
        traceState_.tracePromise.setValue();
        traceState_.planId.clear();
      }

      // Wait until the task future is unblocked.
      if (future.valid()) {
        future.wait();
      } else {
        // When no vector was produced and the future is not valid, it's the
        // task signal that it has finished producing output.
        VELOX_CHECK(!task_->isRunning() || !noMoreSplits_);
        break;
      }
    }
    return false;
  }

  // Internal state for coordinating between the tracer and cursor.
  //
  // This struct manages the synchronization between the trace writer
  // (which produces intermediate results) and the cursor (which consumes
  // them).
  struct TraceState {
    // Promise used to signal the tracer to continue after a partial result
    // has been consumed.
    ContinuePromise tracePromise{ContinuePromise::makeEmpty()};

    // The most recent intermediate result from a traced operator.
    RowVectorPtr traceData;

    // The plan id where this state came from.
    core::PlanNodeId planId;
  };

  TraceState traceState_;

  // Custom trace context implementation for the debugger.
  //
  // This trace context pauses execution at traced operators by blocking
  // the trace writer until the cursor consumes the intermediate result.
  class TaskDebuggerTraceCtx : public trace::TraceCtx {
   public:
    // Constructs a trace context for the specified plan nodes.
    //
    // @param breakpoints Map of plan node IDs to optional callbacks.
    // @param traceState Reference to the shared trace state for coordination.
    TaskDebuggerTraceCtx(
        const CursorParameters::TBreakpointMap& breakpoints,
        TraceState& traceState)
        : TraceCtx(false), breakpoints_(breakpoints), traceState_(traceState) {}

    // Determines whether a given operator should be traced.
    //
    // @param op The operator to check.
    // @return true if the operator's plan node ID is in the traced set.
    bool shouldTrace(const Operator& op) const override {
      return breakpoints_.contains(op.planNodeId());
    }

    // Creates an input trace writer for the given operator.
    //
    // @param op The operator to create a tracer for.
    // @return A unique pointer to the trace input writer.
    std::unique_ptr<trace::TraceInputWriter> createInputTracer(
        Operator& op) const override {
      auto it = breakpoints_.find(op.planNodeId());
      return std::make_unique<TaskDebuggerTraceInputWriter>(
          op.planNodeId(),
          it != breakpoints_.end() ? it->second : nullptr,
          traceState_);
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
          CursorParameters::BreakpointCallback callback,
          TraceState& traceState)
          : planId_(planId),
            callback_(std::move(callback)),
            traceState_(traceState) {}

      // Writes an input vector and potentially pauses execution.
      //
      // Invokes the callback if set. If the callback returns false, the writer
      // does not block and execution continues. If the callback returns true
      // (or is null), stores the vector in the trace state and creates a future
      // that blocks until the cursor consumes the result and signals
      // continuation.
      //
      // @param vector The input vector to trace.
      // @param future Output parameter set to a future that blocks until
      //        the cursor is ready to continue.
      // @return true if the writer is blocked waiting for the future, false
      //         if execution should continue without blocking.
      bool write(const RowVectorPtr& vector, ContinueFuture* future) override {
        // Invoke the callback if set. If it returns false, don't block.
        if (callback_ && !callback_(vector)) {
          return false;
        }

        VELOX_CHECK(traceState_.tracePromise.isFulfilled());

        traceState_.tracePromise = ContinuePromise("TaskQueue::dequeue");
        traceState_.traceData = vector;
        traceState_.planId = planId_;
        *future = traceState_.tracePromise.getFuture();
        return true;
      }

      // Called when tracing is complete for this operator.
      void finish() override {}

     private:
      const core::PlanNodeId planId_;
      const CursorParameters::BreakpointCallback callback_;
      TraceState& traceState_;
    };

    CursorParameters::TBreakpointMap breakpoints_;
    TraceState& traceState_;
  };

  std::shared_ptr<exec::Task> task_;
  bool noMoreSplits_{false};
  RowVectorPtr current_;
  std::exception_ptr error_;
};

} // namespace

bool RowCursor::next() {
  if (++currentRow_ < numRows_) {
    return true;
  }
  if (!cursor_->moveNext()) {
    return false;
  }
  auto vector = cursor_->current();
  numRows_ = vector->size();
  if (!numRows_) {
    return next();
  }
  currentRow_ = 0;
  if (decoded_.empty()) {
    decoded_.resize(vector->childrenSize());
    for (int32_t i = 0; i < vector->childrenSize(); ++i) {
      decoded_[i] = std::make_unique<DecodedVector>();
    }
  }
  allRows_.resize(vector->size());
  allRows_.setAll();
  for (int32_t i = 0; i < decoded_.size(); ++i) {
    decoded_[i]->decode(*vector->childAt(i), allRows_);
  }
  return true;
}

std::unique_ptr<TaskCursor> TaskCursor::create(const CursorParameters& params) {
  if (!params.breakpoints.empty()) {
    VELOX_CHECK(
        params.serialExecution,
        "Breakpoints are only supported in serial execution for now.");
    return std::make_unique<TaskDebuggerCursor>(params);
  }

  if (params.serialExecution) {
    return std::make_unique<SingleThreadedTaskCursor>(params);
  }
  return std::make_unique<MultiThreadedTaskCursor>(params);
}

bool waitForTaskDriversToFinish(exec::Task* task, uint64_t maxWaitMicros) {
  VELOX_USER_CHECK(!task->isRunning());
  uint64_t waitMicros = 0;
  while ((task->numFinishedDrivers() != task->numTotalDrivers()) &&
         (waitMicros < maxWaitMicros)) {
    const uint64_t kWaitMicros = 1000;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitMicros));
    waitMicros += kWaitMicros;
  }

  if (task->numFinishedDrivers() != task->numTotalDrivers()) {
    LOG(ERROR) << "Timed out waiting for all drivers of task " << task->taskId()
               << " to finish. Finished drivers: " << task->numFinishedDrivers()
               << ". Total drivers: " << task->numTotalDrivers();
  }

  return task->numFinishedDrivers() == task->numTotalDrivers();
}

} // namespace facebook::velox::exec
