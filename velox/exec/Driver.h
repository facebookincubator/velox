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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/portability/SysSyscall.h>
#include <memory>

#include "velox/common/future/VeloxPromise.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/connectors/Connector.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::exec {

class Driver;
class ExchangeClient;
class Operator;
struct OperatorStats;
class Task;

enum class StopReason {
  // Keep running.
  kNone,
  // Go off thread and do not schedule more activity.
  kPause,
  // Stop and free all. This is returned once and the thread that gets
  // this value is responsible for freeing the state associated with
  // the thread. Other threads will get kAlreadyTerminated after the
  // first thread has received kTerminate.
  kTerminate,
  kAlreadyTerminated,
  // Go off thread and then enqueue to the back of the runnable queue.
  kYield,
  // Must wait for external events.
  kBlock,
  // No more data to produce.
  kAtEnd,
  kAlreadyOnThread
};

std::string stopReasonString(StopReason reason);

std::ostream& operator<<(std::ostream& out, const StopReason& reason);

// Represents a Driver's state. This is used for cancellation, forcing
// release of and for waiting for memory. The fields are serialized on
// the mutex of the Driver's Task.
//
// The Driver goes through the following states:
// Not on thread. It is created and has not started. All flags are false.
//
// Enqueued - The Driver is added to an executor but does not yet have a thread.
// isEnqueued is true. Next states are terminated or on thread.
//
// On thread - 'thread' is set to the thread that is running the Driver. Next
// states are blocked, terminated, suspended, enqueued.
//
//  Blocked - The Driver is not on thread and is waiting for an external event.
//  Next states are terminated, enqueued.
//
// Suspended - The Driver is on thread, 'thread' and 'isSuspended' are set. The
// thread does not manipulate the Driver's state and is suspended as in waiting
// for memory or out of process IO. This is different from Blocked in that here
// we keep the stack so that when the wait is over the control stack is not
// lost. Next states are on thread or terminated.
//
//  Terminated - 'isTerminated' is set. The Driver cannot run after this and
// the state is final.
//
// CancelPool  allows terminating or pausing a set of Drivers. The Task API
// allows starting or resuming Drivers. When terminate is requested the request
// is successful when all Drivers are off thread, blocked or suspended. When
// pause is requested, we have success when all Drivers are either enqueued,
// suspended, off thread or blocked.
struct ThreadState {
  // The thread currently running this.
  std::atomic<std::thread::id> thread{};
  // The tid of 'thread'. Allows finding the thread in a debugger.
  std::atomic<int32_t> tid{0};
  // True if queued on an executor but not on thread.
  std::atomic<bool> isEnqueued{false};
  // True if being terminated or already terminated.
  std::atomic<bool> isTerminated{false};
  // True if there is a future outstanding that will schedule this on an
  // executor thread when some promise is realized.
  bool hasBlockingFuture{false};
  // True if on thread but in a section waiting for RPC or memory
  // strategy decision. The thread is not supposed to access its
  // memory, which a third party can revoke while the thread is in
  // this state.
  bool isSuspended{false};

  bool isOnThread() const {
    return thread != std::thread::id();
  }

  void setThread() {
    thread = std::this_thread::get_id();
#if !defined(__APPLE__)
    // This is a debugging feature disabled on the Mac since syscall
    // is deprecated on that platform.
    tid = syscall(FOLLY_SYS_gettid);
#endif
  }

  void clearThread() {
    thread = std::thread::id(); // no thread.
    tid = 0;
  }
};

enum class BlockingReason {
  kNotBlocked,
  kWaitForConsumer,
  kWaitForSplit,
  /// Some operators can get blocked due to the producer(s) (they are currently
  /// waiting data from) not having anything produced. Used by LocalExchange,
  /// LocalMergeExchange, Exchange and MergeExchange operators.
  kWaitForProducer,
  kWaitForJoinBuild,
  /// For a build operator, it is blocked waiting for the probe operators to
  /// finish probing before build the next hash table from one of the previously
  /// spilled partition data.
  /// For a probe operator, it is blocked waiting for all its peer probe
  /// operators to finish probing before notifying the build operators to build
  /// the next hash table from the previously spilled data.
  kWaitForJoinProbe,
  /// Used by MergeJoin operator, indicating that it was blocked by the right
  /// side input being unavailable.
  kWaitForMergeJoinRightSide,
  kWaitForMemory,
  kWaitForConnector,
  /// Build operator is blocked waiting for all its peers to stop to run group
  /// spill on all of them.
  kWaitForSpill,
};

std::string blockingReasonToString(BlockingReason reason);

class BlockingState {
 public:
  BlockingState(
      std::shared_ptr<Driver> driver,
      ContinueFuture&& future,
      Operator* FOLLY_NONNULL op,
      BlockingReason reason);

  ~BlockingState() {
    numBlockedDrivers_--;
  }

  static void setResume(std::shared_ptr<BlockingState> state);

  Operator* FOLLY_NONNULL op() {
    return operator_;
  }

  BlockingReason reason() {
    return reason_;
  }

  /// Moves out the blocking future stored inside. Can be called only once. Used
  /// in single-threaded execution.
  ContinueFuture future() {
    return std::move(future_);
  }

  /// Returns total number of drivers process wide that are currently in blocked
  /// state.
  static uint64_t numBlockedDrivers() {
    return numBlockedDrivers_;
  }

 private:
  std::shared_ptr<Driver> driver_;
  ContinueFuture future_;
  Operator* FOLLY_NONNULL operator_;
  BlockingReason reason_;
  uint64_t sinceMicros_;

  static std::atomic_uint64_t numBlockedDrivers_;
};

/// Special group id to reflect the ungrouped execution.
constexpr uint32_t kUngroupedGroupId{std::numeric_limits<uint32_t>::max()};

struct DriverCtx {
  const int driverId;
  const int pipelineId;
  /// Id of the split group this driver should process in case of grouped
  /// execution, kUngroupedGroupId otherwise.
  const uint32_t splitGroupId;
  /// Id of the partition to use by this driver. For local exchange, for
  /// instance.
  const uint32_t partitionId;

  std::shared_ptr<Task> task;
  memory::MemoryPool* FOLLY_NONNULL pool;
  Driver* FOLLY_NONNULL driver;

  explicit DriverCtx(
      std::shared_ptr<Task> _task,
      int _driverId,
      int _pipelineId,
      uint32_t _splitGroupId,
      uint32_t _partitionId);

  const core::QueryConfig& queryConfig() const;

  velox::memory::MemoryPool* FOLLY_NONNULL addOperatorPool(
      const core::PlanNodeId& planNodeId,
      const std::string& operatorType);
};

class Driver : public std::enable_shared_from_this<Driver> {
 public:
  Driver(
      std::unique_ptr<DriverCtx> driverCtx,
      std::vector<std::unique_ptr<Operator>> operators);

  static void enqueue(std::shared_ptr<Driver> instance);

  /// Run the pipeline until it produces a batch of data or gets blocked. Return
  /// the data produced or nullptr if pipeline finished processing and will not
  /// produce more data. Return nullptr and set 'blockingState' if pipeline got
  /// blocked.
  ///
  /// This API supports execution of a Task synchronously in the caller's
  /// thread. The caller must use either this API or 'enqueue', but not both.
  /// When using 'enqueue', the last operator in the pipeline (sink) must not
  /// return any data from Operator::getOutput(). When using 'next', the last
  /// operator must produce data that will be returned to caller.
  RowVectorPtr next(std::shared_ptr<BlockingState>& blockingState);

  bool isOnThread() const {
    return state_.isOnThread();
  }

  bool isTerminated() const {
    return state_.isTerminated;
  }

  std::string label() const;

  ThreadState& state() {
    return state_;
  }

  void initializeOperatorStats(std::vector<OperatorStats>& stats);

  void addStatsToTask();

  // Returns true if all operators between the source and 'aggregation' are
  // order-preserving and do not increase cardinality.
  bool mayPushdownAggregation(Operator* FOLLY_NONNULL aggregation) const;

  // Returns a subset of channels for which there are operators upstream from
  // filterSource that accept dynamically generated filters.
  std::unordered_set<column_index_t> canPushdownFilters(
      const Operator* FOLLY_NONNULL filterSource,
      const std::vector<column_index_t>& channels) const;

  // Returns the Operator with 'planNodeId.' or nullptr if not
  // found. For example, hash join probe accesses the corresponding
  // build by id.
  Operator* FOLLY_NULLABLE findOperator(std::string_view planNodeId) const;

  // Returns a list of all operators.
  std::vector<Operator*> operators() const;

  void setError(std::exception_ptr exception);

  std::string toString();

  DriverCtx* FOLLY_NONNULL driverCtx() const {
    return ctx_.get();
  }

  const std::shared_ptr<Task>& task() const {
    return ctx_->task;
  }

  // Updates the stats in Task and frees resources. Only called by Task for
  // closing non-running Drivers.
  void closeByTask();

  BlockingReason blockingReason() const {
    return blockingReason_;
  }

 private:
  void enqueueInternal();

  static void run(std::shared_ptr<Driver> self);

  StopReason runInternal(
      std::shared_ptr<Driver>& self,
      std::shared_ptr<BlockingState>& blockingState,
      RowVectorPtr& result);

  void close();

  // Push down dynamic filters produced by the operator at the specified
  // position in the pipeline.
  void pushdownFilters(int operatorIndex);

  /// If 'trackOperatorCpuUsage_' is true, returns initialized timer object to
  /// track cpu and wall time of an operation. Returns null otherwise.
  /// The delta CpuWallTiming object would be passes to 'func' upon destruction
  /// of the timer.
  template <typename F>
  std::unique_ptr<DeltaCpuWallTimer<F>> createDeltaCpuWallTimer(F&& func) {
    return trackOperatorCpuUsage_
        ? std::make_unique<DeltaCpuWallTimer<F>>(std::move(func))
        : nullptr;
  }

  std::unique_ptr<DriverCtx> ctx_;
  std::atomic_bool closed_{false};

  // Set via Task and serialized by Task's mutex.
  ThreadState state_;

  // Timer used to track down the time we are sitting in the driver queue.
  size_t queueTimeStartMicros_{0};
  // Index of the current operator to run (or the 1st one if we haven't started
  // yet). Used to determine which operator's queueTime we should update.
  size_t curOpIndex_{0};

  std::vector<std::unique_ptr<Operator>> operators_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};

  bool trackOperatorCpuUsage_;
};

using OperatorSupplier = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* FOLLY_NONNULL ctx)>;

using Consumer =
    std::function<BlockingReason(RowVectorPtr, ContinueFuture* FOLLY_NULLABLE)>;
using ConsumerSupplier = std::function<Consumer()>;

struct DriverFactory {
  std::vector<std::shared_ptr<const core::PlanNode>> planNodes;
  /// Function that will generate the final operator of a driver being
  /// constructed.
  OperatorSupplier consumerSupplier;
  /// Maximum number of drivers that can be run concurrently in this pipeline.
  uint32_t maxDrivers;
  /// Number of drivers that will be run concurrently in this pipeline for one
  /// split group (during grouped execution) or for the whole task (ungrouped
  /// execution).
  uint32_t numDrivers;
  /// Total number of drivers in this pipeline we expect to be run. In case of
  /// grouped execution it is 'numDrivers' * 'numSplitGroups', otherwise it is
  /// 'numDrivers'.
  uint32_t numTotalDrivers;
  /// The (local) node that will consume results supplied by this pipeline.
  /// Can be null. We use that to determine the max drivers.
  std::shared_ptr<const core::PlanNode> consumerNode;
  /// True if the drivers in this pipeline use grouped execution strategy.
  bool groupedExecution{false};
  /// True if 'planNodes' contains a source node for the task, e.g. TableScan or
  /// Exchange.
  bool inputDriver{false};
  /// True if 'planNodes' contains a sync node for the task, e.g.
  /// PartitionedOutput.
  bool outputDriver{false};
  /// Contains node ids for which Hash Join Bridges connect ungrouped execution
  /// and grouped execution and must be created in ungrouped execution pipeline
  /// and skipped in grouped execution pipeline.
  folly::F14FastSet<core::PlanNodeId> mixedExecutionModeHashJoinNodeIds;
  /// Same as 'mixedExecutionModeHashJoinNodeIds' but for Cross Joins.
  folly::F14FastSet<core::PlanNodeId> mixedExecutionModeCrossJoinNodeIds;

  std::shared_ptr<Driver> createDriver(
      std::unique_ptr<DriverCtx> ctx,
      std::shared_ptr<ExchangeClient> exchangeClient,
      std::function<int(int pipelineId)> numDrivers);

  bool supportsSingleThreadedExecution() const {
    return !needsPartitionedOutput() && !needsExchangeClient() &&
        !needsLocalExchange();
  }

  const core::PlanNodeId& leafNodeId() const {
    VELOX_CHECK(!planNodes.empty());
    return planNodes.front()->id();
  }

  const core::PlanNodeId& outputNodeId() const {
    VELOX_CHECK(!planNodes.empty());
    return planNodes.back()->id();
  }

  std::shared_ptr<const core::PartitionedOutputNode> needsPartitionedOutput()
      const {
    VELOX_CHECK(!planNodes.empty());
    if (auto partitionedOutputNode =
            std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
                planNodes.back())) {
      return partitionedOutputNode;
    }
    return nullptr;
  }

  /// Returns Exchange plan node ID if the pipeline receives data from an
  /// exchange.
  std::optional<core::PlanNodeId> needsExchangeClient() const {
    VELOX_CHECK(!planNodes.empty());
    const auto& leafNode = planNodes.front();
    if (leafNode->requiresExchangeClient()) {
      return leafNode->id();
    }
    return std::nullopt;
  }

  /// Returns LocalPartition plan node ID if the pipeline gets data from a local
  /// exchange.
  std::optional<core::PlanNodeId> needsLocalExchange() const {
    VELOX_CHECK(!planNodes.empty());
    if (auto exchangeNode =
            std::dynamic_pointer_cast<const core::LocalPartitionNode>(
                planNodes.front())) {
      return exchangeNode->id();
    }
    return std::nullopt;
  }

  /// Returns plan node IDs for which Hash Join Bridges must be created based on
  /// this pipeline.
  std::vector<core::PlanNodeId> needsHashJoinBridges() const;

  /// Returns plan node IDs for which Cross Join Bridges must be created based
  /// on this pipeline.
  std::vector<core::PlanNodeId> needsCrossJoinBridges() const;
};

// Begins and ends a section where a thread is running but not
// counted in its Task. Using this, a Driver thread can for
// example stop its own Task. For arbitrating memory overbooking,
// the contending threads go suspended and each in turn enters a
// global critical section. When running the arbitration strategy, a
// thread can stop and restart Tasks, including its own. When a Task
// is stopped, its drivers are blocked or suspended and the strategy thread
// can alter the Task's memory including spilling or killing the whole Task.
// Other threads waiting to run the arbitration, are in a suspended state
// which also means that they are instantaneously killable or spillable.
class SuspendedSection {
 public:
  explicit SuspendedSection(Driver* FOLLY_NONNULL driver);
  ~SuspendedSection();

 private:
  Driver* FOLLY_NONNULL driver_;
};

} // namespace facebook::velox::exec
