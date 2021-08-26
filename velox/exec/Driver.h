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
#include "velox/connectors/Connector.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::exec {

class Driver;
class ExchangeClient;
class Operator;
struct OperatorStats;
class Task;

enum class BlockingReason {
  kNotBlocked,
  kWaitForConsumer,
  kWaitForSplit,
  kWaitForExchange,
  kWaitForJoinBuild,
  kWaitForMemory
};

using ContinueFuture = folly::SemiFuture<bool>;

class BlockingState {
 public:
  BlockingState(
      std::shared_ptr<Driver> driver,
      ContinueFuture&& future,
      Operator* FOLLY_NONNULL op,
      BlockingReason reason);

  static void setResume(
      std::shared_ptr<BlockingState> state,
      folly::Executor* FOLLY_NULLABLE executor = nullptr);

  Operator* FOLLY_NONNULL op() {
    return operator_;
  }

  BlockingReason reason() {
    return reason_;
  }

 private:
  std::shared_ptr<Driver> driver_;
  ContinueFuture future_;
  Operator* FOLLY_NONNULL operator_;
  BlockingReason reason_;
  uint64_t sinceMicros_;
};

struct DriverCtx {
  std::shared_ptr<Task> task;
  std::unique_ptr<core::ExecCtx> execCtx;
  std::unique_ptr<connector::ExpressionEvaluator> expressionEvaluator;
  const int driverId;
  const int pipelineId;
  Driver* FOLLY_NONNULL driver;
  int32_t numDrivers;

  explicit DriverCtx(
      std::shared_ptr<Task> _task,
      int _driverId,
      int _pipelineId,
      int32_t numDrivers);

  velox::memory::MemoryPool* FOLLY_NONNULL addOperatorUserPool() {
    opMemPools_.push_back(execCtx->pool()->addScopedChild("operator_ctx"));
    return opMemPools_.back().get();
  }

  velox::memory::MemoryPool* FOLLY_NONNULL
  addOperatorSystemPool(velox::memory::MemoryPool* FOLLY_NONNULL userPool) {
    auto poolPtr = userPool->addScopedChild("operator_ctx");
    poolPtr->setMemoryUsageTracker(
        userPool->getMemoryUsageTracker()->addChild(true));
    auto pool = poolPtr.get();
    opMemPools_.push_back(std::move(poolPtr));

    return pool;
  }

  std::unique_ptr<connector::ConnectorQueryCtx> createConnectorQueryCtx(
      const std::string& connectorId) const;

 private:
  // Lifetime of operator memory pools is same as the driverCtx, since some
  // buffers allocated within an operator context may still be referenced by
  // other operators.
  std::vector<std::unique_ptr<velox::memory::MemoryPool>> opMemPools_;
};

class Driver {
 public:
  Driver(
      std::unique_ptr<DriverCtx> driverCtx,
      std::vector<std::unique_ptr<Operator>>&& operators);

  ~Driver() {
    close();
  }

  static folly::CPUThreadPoolExecutor* FOLLY_NONNULL
  executor(int32_t threads = 0);

  static void run(
      std::shared_ptr<Driver> self,
      folly::Executor* FOLLY_NULLABLE executor = nullptr);

  static void enqueue(
      std::shared_ptr<Driver> instance,
      folly::Executor* FOLLY_NULLABLE executor = nullptr);

  // Waits for activity on 'executor_' to finish and then makes a new
  // executor. Testing uses this to ensure that there are no live
  // references to memory pools before deleting the pools.
  static void testingJoinAndReinitializeExecutor(int32_t threads = 0);

  bool isOnThread() const {
    return state_.isOnThread();
  }

  bool isTerminated() const {
    return state_.isTerminated;
  }

  std::string label();

  core::ThreadState& state() {
    return state_;
  }

  core::CancelPool* cancelPool() const {
    return cancelPool_.get();
  }

  // Frees the resources associated with this if this is
  // off-thread. Returns true if resources are freed. If this is on
  // thread, returns false. In this case the Driver's thread will see
  // that the CancelPool is set to terminate and will free the
  // resources on the thread.
  bool terminate();

  void initializeOperatorStats(std::vector<OperatorStats>& stats);

  void addStatsToTask();

  // Returns true if all operators between the source and 'aggregation' are
  // order-preserving and do not increase cardinality.
  bool mayPushdownAggregation(Operator* FOLLY_NONNULL aggregation);

  // Returns the Operator with 'planNodeId.' or nullptr if not
  // found. For example, hash join probe accesses the corresponding
  // build by id.
  Operator* FOLLY_NULLABLE findOperator(std::string_view planNodeId) const;

  void setError(std::exception_ptr exception);

  std::string toString();

  DriverCtx* FOLLY_NONNULL driverCtx() const {
    return ctx_.get();
  }

  // Requests 'size' bytes of memory from the process memory manager,
  // to be added to the limit of 'tracker'. If tracker is
  // revocable and there is no ready supply, the reservation may be
  // unchanged. If tracker is not revocable, the allocation will be
  // made or an error is thrown. This must be called while on thread in a thread
  // of the Task of 'this'. Returns true if the
  // limit was increased.
  bool reserveMemory(memory::MemoryUsageTracker* tracker, int64_t size);

  // Tries to spill 'size' or more bytes of revocable memory from any
  // of the operators in 'this'. Returns the amount of memory
  // freed. 'this' must be off thread or in a cancel-free state.
  int64_t spill(int64_t size);

  // Attempts to increase the limit of 'tracker' by 'size'.  If 'type'
  // is kRecoverable then this may fail to increase the limit if the
  // limit is already high and there is no extra available. If 'type'
  // is not 'kRecoverable' then this will make a serious effort to
  // raise the limit, including spilling other queries and will fail
  // only if this really was not possible. Returns true if the limit
  // was raised by at least 'size'. The caller must be a thread of the Task of
  // 'this'.
  bool growTaskMemory(
      memory::UsageType type,
      int64_t size,
      memory::MemoryUsageTracker* tracker);

 private:
  core::StopReason runInternal(
      std::shared_ptr<Driver>& self,
      std::shared_ptr<BlockingState>* FOLLY_NONNULL blockingState);

  void close();

  std::unique_ptr<DriverCtx> ctx_;
  std::shared_ptr<Task> task_;
  core::CancelPoolPtr cancelPool_;

  // Set via 'cancelPool_' and serialized by ''cancelPool_'s  mutex.
  core::ThreadState state_;

  std::vector<std::unique_ptr<Operator>> operators_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
};

using OperatorSupplier = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* FOLLY_NONNULL ctx)>;

using Consumer =
    std::function<BlockingReason(RowVectorPtr, ContinueFuture* FOLLY_NULLABLE)>;
using ConsumerSupplier = std::function<Consumer()>;

struct DriverFactory {
  std::vector<std::shared_ptr<const core::PlanNode>> planNodes;
  // Function that will generate the final operator of a driver being
  // constructed.
  OperatorSupplier consumerSupplier;
  uint32_t maxDrivers;

  std::shared_ptr<Driver> createDriver(
      std::unique_ptr<DriverCtx> ctx,
      std::shared_ptr<ExchangeClient> exchangeClient,
      std::function<int(int pipelineId)> numDrivers);

  std::shared_ptr<const core::PartitionedOutputNode> needsPartitionedOutput() {
    VELOX_CHECK(!planNodes.empty());
    if (auto partitionedOutputNode =
            std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
                planNodes.back())) {
      return partitionedOutputNode;
    } else {
      return nullptr;
    }
  }

  bool needsExchangeClient() const {
    VELOX_CHECK(!planNodes.empty());
    if (auto exchangeNode = std::dynamic_pointer_cast<const core::ExchangeNode>(
            planNodes.front())) {
      return true;
    }

    return false;
  }

  /// Returns LocalPartition plan node ID if the pipeline gets data from a local
  /// exchange.
  std::optional<core::PlanNodeId> needsLocalExchangeSource() const {
    VELOX_CHECK(!planNodes.empty());
    if (auto exchangeNode =
            std::dynamic_pointer_cast<const core::LocalPartitionNode>(
                planNodes.front())) {
      return exchangeNode->id();
    }

    return std::nullopt;
  }
};

// Begins and ends a section where a thread is running but not
// counted in its CancelPool. Using this, a Driver thread can for
// example stop its own Task. For arbitrating memory overbooking,
// the contending threads go cancel-free and each in turn enter a
// global critical section. When running the arbitration strategy, a
// thread can stop and restart Tasks, including its own. When a Task
// is stopped, the strategy thread can alter its memory including
// spilling or killing the whole Task. Other threads waiting to run
// the arbitration (MemoryManagerStrategy::recover), are in a cancel
// free state which also means that they are not altering their own
// memory except via running recover.
class CancelFreeSection {
 public:
  explicit CancelFreeSection(Driver* driver);
  ~CancelFreeSection();

 private:
  Driver* driver_;
};

} // namespace facebook::velox::exec
