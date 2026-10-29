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
#include "velox/exec/FixedPointLoop.h"

#include <folly/coro/BlockingWait.h>
#include <folly/coro/FutureUtil.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/futures/Future.h>
#include <folly/lang/Bits.h>
#include <folly/stop_watch.h>

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>

#include "velox/common/time/Timer.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/ExchangeClient.h"
#include "velox/exec/FixedPointSplit.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/Split.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::exec {

using core::FixedPointNode;
using core::HashTableStateDeclaration;
using core::VectorStateDeclaration;

namespace {

// Deep-copies 'source' into 'pool'.
RowVectorPtr copyRowVector(
    const RowVectorPtr& source,
    memory::MemoryPool* pool) {
  auto target =
      BaseVector::create<RowVector>(source->type(), source->size(), pool);
  target->copy(source.get(), 0, 0, source->size());
  return target;
}

// Adds a leaf named 'name' to the pool of 'task', for memory that drivers of
// the loop's sub-tasks allocate.  When the task's pool has a reclaimer, so does
// the leaf: one that suspends the allocating driver while it waits for memory
// arbitration, as Task does for its exchange client pools.  A reclaim of the
// query waits for the drivers of each task it reclaims to pause, which a driver
// that is not suspended never does.
std::shared_ptr<memory::MemoryPool> addSubTaskLeafPool(
    exec::Task& task,
    const std::string& name) {
  auto* pool = task.pool();
  return pool->addLeafChild(
      name,
      /*threadSafe=*/true,
      pool->reclaimer() != nullptr ? exec::MemoryReclaimer::create() : nullptr);
}

uint64_t countRows(const std::vector<RowVectorPtr>& batches) {
  uint64_t numRows{0};
  for (const auto& batch : batches) {
    numRows += batch->size();
  }
  return numRows;
}

// Iterations whose sub-task TaskStats are kept, besides the latest one.
constexpr size_t kNumLeadingDetailedIterations{3};

void clearTaskStats(FixedPointChainExecutionStats& chain) {
  for (auto& plan : chain.plans) {
    plan.taskStats = TaskStats{};
  }
}

// Aborts 'tasks' and waits until none of their drivers is on a thread. A failed
// task resolves its completion future while drivers that are on thread keep
// running, and those drivers can still call a consumer that writes into the
// caller's frame.  For a task that runs a nested fixed point, this waits for
// the drivers above it; its loop stops at its own next wait, and the task holds
// the pool of the caller's state until it is destroyed.  Runs while an error
// unwinds, so it logs a failure to abort instead of throwing.
void abortAndWait(const std::vector<exec::TaskPtr>& tasks) {
  std::vector<ContinueFuture> futures;
  futures.reserve(tasks.size());
  for (const auto& task : tasks) {
    try {
      futures.push_back(task->requestAbort());
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to abort fixed-point sub-task " << task->taskId()
                 << ": " << e.what();
    }
  }
  for (auto& future : futures) {
    std::move(future).wait();
  }
}

// Returns the id of the first ExchangeNode found in the plan, if any.
std::optional<core::PlanNodeId> findExchangeNodeId(
    const core::PlanNodePtr& plan) {
  if (const auto* exchange = core::PlanNode::findFirstNode(
          plan.get(), [](const core::PlanNode* node) {
            return node->is<core::ExchangeNode>();
          })) {
    return exchange->id();
  }

  return std::nullopt;
}

// Collects the split sources in the plan, such as a TableScan or an Exchange,
// by id.  A fixed point nested in the plan, other than 'self', contributes the
// sources of its initial plans: the task running it passes their splits on.  A
// fixed point itself is not collected.
void collectSplitTargets(
    const core::PlanNodePtr& node,
    const core::PlanNode* self,
    folly::F14FastMap<core::PlanNodeId, const core::PlanNode*>& targets) {
  if (const auto* fixedPoint = node->as<FixedPointNode>()) {
    if (fixedPoint != self) {
      for (const auto& declaration : fixedPoint->stateDeclarations()) {
        if (declaration->initialPlan() != nullptr) {
          collectSplitTargets(declaration->initialPlan(), self, targets);
        }
      }
    }
  } else if (node->requiresSplits()) {
    targets.emplace(node->id(), node.get());
  }
  for (const auto& source : node->sources()) {
    collectSplitTargets(source, self, targets);
  }
}

// A plan must run in parallel mode if it sends partitioned output or consumes
// from an exchange; serial Task::next() rejects both.
bool planNeedsParallel(const core::PlanNodePtr& node) {
  return node->is<core::PartitionedOutputNode>() ||
      findExchangeNodeId(node).has_value();
}

// Waits for 'future', which the coordinator or a peer resolves, until
// 'ended' is requested: once the task ends nothing is bound to resolve it.
// The callback on 'ended' is unregistered when the wait returns, so waiting
// leaves nothing behind.
template <typename T>
folly::coro::Task<T> awaitWhileRunning(
    const folly::CancellationToken& ended,
    folly::SemiFuture<T> future,
    std::string_view waitingFor,
    std::string_view taskId) {
  folly::Promise<folly::Unit> stopped;
  folly::CancellationCallback onEnded(ended, [&] { stopped.setValue(); });
  std::vector<folly::SemiFuture<std::optional<T>>> futures;
  futures.push_back(std::move(future).deferValue([](T value) {
    return std::optional<T>{std::move(value)};
  }));
  futures.push_back(stopped.getSemiFuture().deferValue(
      [](auto&&) { return std::optional<T>{}; }));
  auto first = co_await folly::collectAny(std::move(futures));
  auto value = std::move(first.second).value();
  VELOX_CHECK(
      value.has_value(),
      "The task ended while its fixed point waited for {}: {}",
      waitingFor,
      taskId);
  co_return std::move(*value);
}

// Waits for each of 'tasks' to finish without holding a thread.  If one fails,
// or 'ended' is requested first, aborts the rest and rethrows the error.
// Callers pass a named vector: GCC 11 fails to compile a braced list in a
// co_await operand.
folly::coro::Task<void> awaitTasks(
    std::vector<exec::TaskPtr> tasks,
    const folly::CancellationToken& ended,
    std::string_view taskId) {
  for (const auto& task : tasks) {
    auto finished = co_await folly::coro::co_awaitTry(awaitWhileRunning(
        ended, task->taskCompletionFuture(), "its sub-tasks", taskId));
    if (finished.hasException()) {
      abortAndWait(tasks);
      finished.throwUnlessValue();
    }
    if (auto error = task->error()) {
      abortAndWait(tasks);
      std::rethrow_exception(error);
    }
  }
}

// Throws once 'ended' is requested.  Called before a step creates its
// sub-tasks, so a loop whose task has ended starts no more of them.
void checkRunning(
    const folly::CancellationToken& ended,
    std::string_view taskId) {
  VELOX_CHECK(
      !ended.isCancellationRequested(),
      "The fixed point's task ended before the loop created its next "
      "sub-tasks: {}",
      taskId);
}

} // namespace

FixedPointLoop::FixedPointLoop(
    exec::Task* owner,
    const core::FixedPointNode& node,
    const FixedPointOptions* fixedPointOptions,
    const NestedFixedPoint* nested)
    : options_(
          fixedPointOptions != nullptr ? *fixedPointOptions
                                       : FixedPointOptions{}),
      nested_(nested != nullptr ? *nested : NestedFixedPoint{}),
      executionMode_(owner->executionMode()),
      owner_(owner),
      node_(&node) {
  VELOX_CHECK_NOT_NULL(owner_);
  if (nested_.statePool != nullptr) {
    workers_ = nested_.workers;
    nesting_ = nested_.nesting;
    nesting_.push_back(
        {.planNodeId = node_->id(), .iteration = nested_.iteration});
  }

  const auto& convergence = node_->convergenceConfig();
  convergencePlans_ = convergence.plans;
  if (convergence.verdict != nullptr) {
    convergencePlans_.push_back(convergence.verdict);
  }
  if (executionMode_ == exec::Task::ExecutionMode::kParallel) {
    VELOX_CHECK_NOT_NULL(
        options_.orchestrationExecutor,
        "Parallel FixedPointLoop requires "
        "FixedPointOptions::orchestrationExecutor, an executor separate from "
        "the query executor its sub-tasks run on");
    if (node_->convergenceConfig().verdict != nullptr) {
      VELOX_CHECK_NOT_NULL(
          options_.publishVerdict,
          "A verdict plan requires FixedPointOptions::publishVerdict");
      VELOX_CHECK_NOT_NULL(
          options_.awaitVerdict,
          "A verdict plan requires FixedPointOptions::awaitVerdict");
    }
    // The plans of a chain are joined by shuffle.
    if (node_->plans().size() > 1 || convergencePlans_.size() > 1) {
      VELOX_CHECK_NOT_NULL(
          options_.producerTaskId,
          "A shuffling fixed point requires FixedPointOptions::producerTaskId");
      VELOX_CHECK_NOT_NULL(
          options_.producerSplit,
          "A shuffling fixed point requires FixedPointOptions::producerSplit");
    }
  }
  VELOX_CHECK_NOT_NULL(
      options_.subTaskId,
      "A fixed point requires FixedPointOptions::subTaskId");
  VELOX_USER_CHECK_GE(
      options_.iterationMaxDrivers,
      1,
      "FixedPointOptions::iterationMaxDrivers must be at least 1");
  // Each split group of the owning task's pipeline would re-read the whole
  // output state.
  VELOX_USER_CHECK(
      !owner_->planFragment().isGroupedExecution(),
      "A fixed point does not support grouped execution");
  workerIndex_ = owner_->destination();

  // One worker per output partition of whichever chain shuffles -- the body,
  // the convergence sequence, or a fixed point nested in either (see
  // FixedPointNode::numWorkers).  A loop whose chains only gather to worker 0
  // has a numWorkers() of 1, yet runs on as many workers as the coordinator
  // names peers.  A fixed point that reads no peers may run as any number of
  // tasks, each an independent loop over the rows its initial plans read for
  // its destination.
  const auto& plans = node_->plans();
  numWorkers_ = node_->numWorkers();
  if (numWorkers_ > 1) {
    VELOX_CHECK_LT(
        workerIndex_,
        numWorkers_,
        "Worker destination out of range; create exactly {} FixedPointLoops "
        "with destinations 0..{}",
        numWorkers_,
        numWorkers_ - 1);
  }

  // Every id this loop takes splits for.  The fixed point itself takes a peer
  // worker's remote split; everything else is a split-requiring source in an
  // initial plan.  The body and convergence chains wire their own exchanges,
  // so their ids are not here.
  splitTargets_.emplace(node_->id(), node_);

  for (const auto& declaration : node_->stateDeclarations()) {
    if (declaration->initialPlan() != nullptr) {
      collectSplitTargets(declaration->initialPlan(), node_, splitTargets_);
    }
  }

  // Strict mode inheritance: every sub-task runs in this task's execution mode,
  // so a serial fixed point cannot drive a shuffling body or convergence
  // sequence -- serial next() rejects PartitionedOutput/Exchange.  Either
  // requires parallel mode.
  if (executionMode_ == exec::Task::ExecutionMode::kSerial) {
    for (const auto& plan : plans) {
      VELOX_USER_CHECK(
          !planNeedsParallel(plan),
          "A serial fixed point cannot run a shuffling body "
          "(PartitionedOutput/Exchange); use parallel execution mode");
    }
    for (const auto& plan : convergencePlans_) {
      VELOX_USER_CHECK(
          !planNeedsParallel(plan),
          "A serial fixed point cannot run a shuffling convergence sequence "
          "(PartitionedOutput/Exchange); use parallel execution mode");
    }
  }
}

void FixedPointLoop::addSplit(
    const core::PlanNodeId& planNodeId,
    exec::Split&& split) {
  // A coordinator may add splits from several threads, so guard the captured
  // lists; the base takes its own lock for the same reason.
  std::lock_guard<std::mutex> l(splitsMutex_);

  // A FixedPointSplit on the FixedPointNode itself names the workers.  Every
  // other split is read by a source in an initial plan and goes to the sub-task
  // running it, held until that sub-task exists.
  if (split.hasConnectorSplit()) {
    if (const auto* fixedPointSplit =
            split.connectorSplit->as<exec::FixedPointSplit>()) {
      VELOX_USER_CHECK_EQ(
          planNodeId,
          node_->id(),
          "FixedPointSplit addressed to a node other than its FixedPointNode");
      VELOX_USER_CHECK(
          nested_.statePool == nullptr,
          "FixedPointSplit for a nested fixed point, which runs on the workers "
          "of the loop enclosing it: {}",
          node_->id());
      VELOX_USER_CHECK(
          !splitWorkers_.has_value(),
          "Duplicate FixedPointSplit for fixed point: {}",
          node_->id());
      splitWorkers_ = fixedPointSplit->workers;
      return;
    }
  }

  VELOX_USER_CHECK_NE(
      planNodeId,
      node_->id(),
      "A connector split must be addressed to the plan node that reads it -- "
      "an initial plan's TableScan, say -- not to the fixed point itself, "
      "which cannot tell which source it was meant for");

  // finish() stores the terminal phase before it takes splitsMutex_ to release
  // the splits held, so a split it does not find is released here.
  if (const Phase phase = phase_; phase == Phase::kFinalized ||
      phase == Phase::kFailed || phase == Phase::kAbandoned) {
    releaseRemoteSplitLocked(planNodeId, split);
    return;
  }

  if (auto it = splitConsumers_.find(planNodeId); it != splitConsumers_.end()) {
    if (auto task = it->second.lock()) {
      task->addSplit(planNodeId, std::move(split));
    } else {
      releaseRemoteSplitLocked(planNodeId, split);
    }
    return;
  }

  nodeSplits_[planNodeId].push_back(std::move(split));
}

void FixedPointLoop::noMoreSplits(const core::PlanNodeId& planNodeId) {
  if (planNodeId != node_->id()) {
    std::lock_guard<std::mutex> l(splitsMutex_);
    splitsComplete_.insert(planNodeId);
    if (auto it = splitConsumers_.find(planNodeId);
        it != splitConsumers_.end()) {
      if (auto task = it->second.lock()) {
        task->noMoreSplits(planNodeId);
      }
      splitConsumers_.erase(it);
    }
    return;
  }
  workersComplete_.setValue();
}

void FixedPointLoop::runSerially() {
  const Phase phase = leaveCreated(Phase::kStarted);
  if (phase == Phase::kFailed) {
    VELOX_FAIL("The fixed point failed on an earlier call and cannot resume");
  }
  if (phase == Phase::kAbandoned) {
    VELOX_FAIL("The fixed point's task ended before the loop ran");
  }
  VELOX_CHECK(
      phase == Phase::kCreated,
      "runSerially() cannot run a fixed point that has already been driven");
  try {
    folly::coro::blocking_wait(run());
  } catch (...) {
    // A throw part-way through leaves the state half-written.
    finish(Phase::kFailed);
    throw;
  }
}

void FixedPointLoop::start(
    uint32_t maxDrivers,
    std::function<void(std::exception_ptr)> onError) {
  // Task::start() already rejected a serial task before delegating here.
  // The owning task keeps this loop alive for its whole run.  Taken before the
  // loop leaves kCreated: from then on only the run ends the loop, so nothing
  // may throw before it is scheduled.
  auto ownerHandle = owner_->shared_from_this();
  const Phase phase = leaveCreated(Phase::kStarted);
  if (phase == Phase::kAbandoned) {
    return;
  }
  VELOX_USER_CHECK(
      phase == Phase::kCreated,
      "start() cannot run a fixed point that has already been driven");
  maxDrivers_ = maxDrivers;
  // Run the loop asynchronously on the orchestration executor so start()
  // returns immediately (Task::start's contract).  run() is the same coroutine
  // runSerially() waits on; here it suspends while its sub-tasks run on the
  // query executor.
  folly::coro::co_withExecutor(
      folly::getKeepAliveToken(options_.orchestrationExecutor), run())
      .start([this, ownerHandle, onError = std::move(onError)](
                 folly::Try<void>&& result) {
        if (result.hasException()) {
          // Fail the task before waking its FixedPointNode leaf, so the task
          // reports the loop's error.
          onError(result.exception().to_exception_ptr());
          finish(Phase::kFailed);
        }
      });
}

void FixedPointLoop::abandon() {
  if (leaveCreated(Phase::kAbandoned) == Phase::kCreated) {
    finish(Phase::kAbandoned);
  }
}

FixedPointLoop::Phase FixedPointLoop::leaveCreated(Phase next) {
  Phase phase = Phase::kCreated;
  phase_.compare_exchange_strong(phase, next);
  return phase;
}

void FixedPointLoop::finish(Phase terminal) {
  phase_ = terminal;
  releaseRemoteSplits();
  finished_.setValue();
}

void FixedPointLoop::releaseRemoteSplitLocked(
    const core::PlanNodeId& planNodeId,
    const exec::Split& split) {
  if (executionMode_ == exec::Task::ExecutionMode::kSerial ||
      !split.hasConnectorSplit()) {
    return;
  }
  const auto* remote = split.connectorSplit->as<RemoteConnectorSplit>();
  const auto target = splitTargets_.find(planNodeId);
  const auto* exchange = target != splitTargets_.end()
      ? target->second->as<core::ExchangeNode>()
      : nullptr;
  if (remote == nullptr || exchange == nullptr) {
    return;
  }
  // A closed client tells the producer of each remote task added to it that
  // its data is no longer needed (ExchangeClient::addRemoteTaskId()), as
  // Task::terminate() does for a task's own Exchanges.
  try {
    auto& closed = closedClients_[exchange->transportKind()];
    if (closed.client == nullptr) {
      if (closedClientPool_ == nullptr) {
        closedClientPool_ =
            owner_->pool()->addLeafChild("fixedpoint.closedExchange");
      }
      auto [client, entry] = owner_->makeExchangeClient(
          *exchange, /*numberOfConsumers=*/1, closedClientPool_.get());
      client->close();
      closed = ClosedClient{std::move(entry), std::move(client)};
    }
    closed.client->addRemoteTaskId(remote->taskId);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to release the producer of a remote split of "
                 << owner_->taskId() << ": " << e.what();
  }
}

void FixedPointLoop::releaseRemoteSplits() {
  std::lock_guard<std::mutex> l(splitsMutex_);
  for (const auto& [planNodeId, splits] : nodeSplits_) {
    for (const auto& split : splits) {
      releaseRemoteSplitLocked(planNodeId, split);
    }
  }
  nodeSplits_.clear();
  splitConsumers_.clear();
}

folly::coro::Task<void> FixedPointLoop::run() {
  // The handle shares the source's state, so the continuation may outlive this
  // loop.
  owner_->taskCompletionFuture()
      .via(&folly::InlineExecutor::instance())
      .thenTry([ended = cancellation_](auto&&) mutable {
        ended.requestCancellation();
      });
  co_await initialize();
  co_await awaitWorkers();

  bool hasConverged = false;
  for (int32_t iteration = 0; iteration < node_->maxIterations(); ++iteration) {
    co_await runIteration(iteration);
    const bool iterationConverged = co_await converged();
    // Copied, so the application runs outside the statistics lock.
    std::optional<FixedPointIterationExecutionStats> completed;
    {
      auto stats = executionStats_.wlock();
      auto& latest = stats->iterations.back();
      latest.converged = iterationConverged;
      if (options_.onIterationStats) {
        completed = latest;
      }
    }
    if (completed.has_value()) {
      try {
        options_.onIterationStats(nesting_, *completed);
      } catch (const std::exception& e) {
        LOG(ERROR) << "FixedPointOptions::onIterationStats threw for task "
                   << owner_->taskId() << ", iteration " << iteration << ": "
                   << e.what();
      }
    }
    if (iterationConverged) {
      hasConverged = true;
      break;
    }
  }
  // Running all maxIterations without converging fails the loop unless the plan
  // opted out (errorWhenMaxIterationReached=false, e.g. a fixed-count loop).
  // The node validated that the flag implies a convergence plan, so reaching
  // here with the flag set is a genuine non-convergence.
  VELOX_USER_CHECK(
      hasConverged || !node_->convergenceConfig().errorWhenMaxIterationReached,
      "Fixed point did not converge within {} iterations",
      node_->maxIterations());

  co_await finalize();
}

folly::coro::Task<void> FixedPointLoop::awaitWorkers() {
  // A nested loop got its workers from the enclosing loop, and a top-level
  // loop that reads no peers needs none.
  if (workersReady_ || nested_.statePool != nullptr || !node_->readsPeers()) {
    co_return;
  }
  co_await awaitWhileRunning(
      cancellation_.getToken(),
      workersComplete_.getSemiFuture(),
      "the FixedPointSplit naming its workers",
      owner_->taskId());
  std::lock_guard<std::mutex> l(splitsMutex_);
  VELOX_USER_CHECK(
      splitWorkers_.has_value(),
      "Missing FixedPointSplit for fixed point that reads its peers: {}",
      node_->id());
  workers_ = std::move(*splitWorkers_);
  // A chain that gathers to worker 0 gives the loop no width, so the number of
  // workers is checked only against one it has.
  if (numWorkers_ > 1) {
    VELOX_USER_CHECK_EQ(
        workers_.size(),
        numWorkers_,
        "Wrong number of workers in FixedPointSplit: {}",
        node_->id());
  }
  VELOX_USER_CHECK_LT(
      workerIndex_,
      workers_.size(),
      "Worker destination out of range of the workers the FixedPointSplit "
      "names: {}",
      node_->id());
  workersReady_ = true;
}

folly::coro::Task<void> FixedPointLoop::initialize() {
  VELOX_CHECK(
      phase_ == Phase::kStarted,
      "initialize() must run once, before any other phase");
  folly::stop_watch<std::chrono::nanoseconds> timer;

  // With more than one worker, every worker must reach the same verdict on the
  // same iteration or lockstep breaks (a converged worker stops producing while
  // peers' exchanges wait for it, deadlocking).  A verdict plan runs once, on
  // worker 0, and every worker receives its verdict.  Any other criterion runs
  // on every worker and must read the same value on each: state the body's
  // shuffle keeps identical, e.g. replicated.
  state_ = std::make_shared<PersistentState>(
      addSubTaskLeafPool(*owner_, "fixedpoint.state"));
  // Each sub-task receives the state and its pool when it is created -- no
  // global or QueryCtx registry.

  std::vector<FixedPointPlanExecutionStats> initializationPlans;
  co_await initializeState(initializationPlans);
  {
    auto stats = executionStats_.wlock();
    stats->initializationNanos = timer.elapsed().count();
    stats->initializationPlans = std::move(initializationPlans);
  }
  phase_ = Phase::kInitialized;
}

folly::coro::Task<void> FixedPointLoop::finalize() {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "finalize() must follow initialize()");
  // The output is the final contents of the output state entry, which the
  // owning task's pipeline reads (output shaping such as a recursive CTE's
  // UNION ALL is the plan's responsibility, e.g. accumulating into an
  // append-mode output entry).
  finish(Phase::kFinalized);
  co_return;

  // The state's pool is not released here.  A sub-task can still be tearing
  // down after its taskCompletionFuture() has resolved, and even after this
  // loop and its task are destroyed, holding vectors and hash tables from the
  // state.  So each sub-task holds the pool, which goes once the last of this
  // loop and its sub-tasks is destroyed.  (Waiting on each sub-task's
  // taskDeletionFuture() instead is not viable: a shuffle producer is read by
  // peer workers, so a peer must outlive it -- the wait would deadlock.)
}

folly::coro::Task<void> FixedPointLoop::initializeState(
    std::vector<FixedPointPlanExecutionStats>& initializationPlans) {
  for (const auto& declaration : node_->stateDeclarations()) {
    if (const auto* vector = declaration->as<VectorStateDeclaration>()) {
      // Register the entry's append mode (from the declaration) so writing it
      // accumulates or replaces accordingly.
      state_->declareVector(vector->name(), vector->append());
      if (vector->initialPlan() == nullptr) {
        state_->setVector(vector->name(), {});
        continue;
      }

      // The initial plan produces this worker's pre-partitioned shard, stored
      // as-is.  Its source may be local (Values), a table (TableScan) or a
      // shuffle (Exchange); drainPlan feeds the coordinator-assigned source
      // splits and runs it in this worker's execution mode.
      auto [batches, taskStats] = co_await drainPlan(
          vector->initialPlan(),
          /*iteration=*/0,
          state_->pool().get(),
          maxDrivers_);
      state_->setVector(vector->name(), batches);
      initializationPlans.push_back({
          .maxDrivers = maxDrivers_,
          .taskStats = std::move(taskStats),
      });
      continue;
    }

    if (const auto* hashTable = declaration->as<HashTableStateDeclaration>()) {
      initializationPlans.push_back(co_await buildHashTable(*hashTable));
      continue;
    }

    VELOX_UNREACHABLE(
        "Unknown state declaration type: {}", declaration->name());
  }
}

namespace {

// Creates an empty join table for 'declaration', owning its own key hashers.
std::unique_ptr<BaseHashTable> createHashTable(
    const HashTableStateDeclaration& declaration,
    const core::QueryConfig& queryConfig,
    memory::MemoryPool* pool) {
  const auto& schema = declaration.schema();
  const auto& keyColumns = declaration.keyColumns();
  const auto numKeys = static_cast<int32_t>(keyColumns.size());

  // Key columns are the leading columns, so a build input column index equals
  // its RowContainer column index (keys first, then dependents).
  std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
  hashers.reserve(numKeys);
  std::vector<TypePtr> dependentTypes;
  for (int32_t channel = 0; channel < schema->size(); ++channel) {
    if (channel < numKeys) {
      VELOX_CHECK_EQ(
          schema->nameOf(channel),
          keyColumns[channel],
          "HashTable build schema must list key columns first");
      hashers.push_back(
          exec::VectorHasher::create(schema->childAt(channel), channel));
    } else {
      dependentTypes.push_back(schema->childAt(channel));
    }
  }
  return exec::HashTable<true>::createForJoin(
      std::move(hashers),
      dependentTypes,
      /*allowDuplicates=*/true,
      /*hasProbedFlag=*/false,
      /*hasCountFlag=*/false,
      queryConfig.minTableRowsForParallelJoinBuild(),
      pool);
}

// Adds 'batch' to 'table' as HashBuild::addInput does for an inner join.  Rows
// with a null key are dropped, since they match nothing.
void addToHashTable(const RowVector& batch, BaseHashTable& table) {
  auto& hashers = table.hashers();
  SelectivityVector activeRows(batch.size());
  for (auto& hasher : hashers) {
    hasher->decode(
        *batch.childAt(hasher->channel())->loadedVector(), activeRows);
  }
  deselectRowsWithNulls(hashers, activeRows);
  if (!activeRows.hasSelections()) {
    return;
  }

  const auto numKeys = hashers.size();
  std::vector<DecodedVector> dependents;
  dependents.reserve(batch.childrenSize() - numKeys);
  for (auto column = numKeys; column < batch.childrenSize(); ++column) {
    dependents.emplace_back(*batch.childAt(column)->loadedVector(), activeRows);
  }
  auto* rows = table.rows();
  // With allowDuplicates the container chains same-key rows through a
  // next-link, which must start null on each new row.
  const auto nextOffset = rows->nextOffset();
  activeRows.applyToSelected([&](vector_size_t row) {
    char* newRow = rows->newRow();
    if (nextOffset > 0) {
      folly::storeUnaligned<char*>(
          newRow + nextOffset, static_cast<char*>(nullptr));
    }
    for (auto i = 0; i < numKeys; ++i) {
      rows->store(hashers[i]->decodedVector(), row, newRow, i);
    }
    for (auto i = 0; i < dependents.size(); ++i) {
      rows->store(dependents[i], row, newRow, numKeys + i);
    }
  });
}

} // namespace

folly::coro::Task<FixedPointPlanExecutionStats> FixedPointLoop::buildHashTable(
    const HashTableStateDeclaration& declaration) {
  const auto& queryCtx = *owner_->queryCtx();
  std::mutex partsMutex;
  std::vector<std::unique_ptr<BaseHashTable>> parts;
  // One table per output driver; HashTable merges at most 255 tables.
  const auto numDrivers =
      std::min<uint32_t>(maxDrivers_, std::numeric_limits<uint8_t>::max());
  auto taskStats = co_await consumePlan(
      declaration.initialPlan(),
      /*iteration=*/0,
      numDrivers,
      [&]() -> exec::Consumer {
        auto part = createHashTable(
            declaration, queryCtx.queryConfig(), state_->pool().get());
        auto* rawPart = part.get();
        {
          std::lock_guard<std::mutex> l(partsMutex);
          parts.push_back(std::move(part));
        }
        return [rawPart](
                   RowVectorPtr batch,
                   bool /*drained*/,
                   ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            addToHashTable(*batch, *rawPart);
          }
          return exec::BlockingReason::kNotBlocked;
        };
      });

  VELOX_CHECK(!parts.empty());
  auto table = std::move(parts.front());
  std::vector<std::unique_ptr<BaseHashTable>> otherTables(
      std::make_move_iterator(parts.begin() + 1),
      std::make_move_iterator(parts.end()));
  table->prepareJoinTable(
      std::move(otherTables),
      exec::BaseHashTable::kNoSpillInputStartPartitionBit,
      /*vectorHasherMaxNumDistinct=*/1'000'000,
      /*dropDuplicates=*/false,
      queryCtx.executor());

  state_->setHashTable(
      declaration.name(),
      HashTableEntry{
          std::shared_ptr<exec::BaseHashTable>(std::move(table)),
          declaration.schema(),
          static_cast<int32_t>(declaration.keyColumns().size()),
      });
  co_return FixedPointPlanExecutionStats{
      .maxDrivers = numDrivers, .taskStats = std::move(taskStats)};
}

void FixedPointLoop::routeSplits(
    const exec::TaskPtr& task,
    const core::PlanNodeId& sourceId) {
  std::lock_guard<std::mutex> l(splitsMutex_);
  const bool complete = splitsComplete_.contains(sourceId);
  if (executionMode_ == exec::Task::ExecutionMode::kSerial) {
    VELOX_USER_CHECK(
        complete,
        "A serial fixed point runs only once each source's splits are "
        "complete; call noMoreSplits before next(): {}",
        sourceId);
  }
  if (auto it = nodeSplits_.find(sourceId); it != nodeSplits_.end()) {
    for (auto& split : it->second) {
      task->addSplit(sourceId, std::move(split));
    }
    nodeSplits_.erase(it);
  }
  if (complete) {
    task->noMoreSplits(sourceId);
  } else {
    splitConsumers_[sourceId] = task;
  }
}

NestedFixedPoint FixedPointLoop::subTaskNesting(
    int32_t iteration,
    std::vector<std::string> workers) {
  return NestedFixedPoint{
      .state = state_,
      .statePool = state_->pool(),
      .workers = std::move(workers),
      .nesting = nesting_,
      .iteration = iteration,
  };
}

folly::coro::Task<FixedPointLoop::DrainedPlan> FixedPointLoop::drainPlan(
    core::PlanNodePtr plan,
    int32_t iteration,
    memory::MemoryPool* pool,
    uint32_t maxDrivers) {
  std::vector<RowVectorPtr> batches;
  std::mutex mutex;
  auto taskStats = co_await consumePlan(
      plan, iteration, maxDrivers, [&]() -> exec::Consumer {
        return [&](RowVectorPtr batch,
                   bool /*drained*/,
                   ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            auto copy = copyRowVector(batch, pool);
            std::lock_guard<std::mutex> l(mutex);
            batches.push_back(std::move(copy));
          }
          return exec::BlockingReason::kNotBlocked;
        };
      });
  co_return DrainedPlan{std::move(batches), std::move(taskStats)};
}

namespace {

// Creates a sub-task of 'owner': it runs on the owner's destination and query
// context, and 'nested' names the loop it belongs to.
exec::TaskPtr createSubTask(
    const exec::Task& owner,
    const std::string& taskId,
    const core::PlanNodePtr& plan,
    exec::Task::ExecutionMode mode,
    exec::ConsumerSupplier consumerSupplier,
    const FixedPointOptions& options,
    const NestedFixedPoint& nested) {
  return exec::Task::create(
      taskId,
      core::PlanFragment{plan},
      /*destination=*/owner.destination(),
      owner.queryCtx(),
      mode,
      std::move(consumerSupplier),
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/std::nullopt,
      /*onError=*/nullptr,
      &options,
      &nested);
}

} // namespace

folly::coro::Task<TaskStats> FixedPointLoop::consumePlan(
    core::PlanNodePtr plan,
    int32_t iteration,
    uint32_t maxDrivers,
    exec::ConsumerSupplier consumerSupplier) {
  // A loop nested in the plan runs on this loop's workers.
  if (core::PlanNode::findFirstNode<core::FixedPointNode>(plan.get()) !=
      nullptr) {
    co_await awaitWorkers();
  }
  // A serial task is drained on this thread, through one consumer.  In
  // parallel, the supplier gives each output driver its own consumer, called on
  // that driver's thread.
  const bool serial = executionMode_ == exec::Task::ExecutionMode::kSerial;
  if (!serial) {
    checkRunning(cancellation_.getToken(), owner_->taskId());
  }
  auto task = createSubTask(
      *owner_,
      subTaskId(),
      plan,
      executionMode_,
      serial ? exec::ConsumerSupplier{} : consumerSupplier,
      options_,
      subTaskNesting(iteration, workers_));
  if (serial) {
    SCOPE_FAIL {
      abortAndWait({task});
    };
    feedInitSplits(task, plan);
    auto consumer = consumerSupplier();
    while (auto batch = task->next()) {
      consumer(std::move(batch), /*drained=*/false, /*future=*/nullptr);
    }
    co_return task->taskStats();
  }

  {
    SCOPE_FAIL {
      abortAndWait({task});
    };
    task->start(maxDrivers);
    feedInitSplits(task, plan);
  }
  std::vector<std::shared_ptr<exec::Task>> tasks{task};
  co_await awaitTasks(
      std::move(tasks), cancellation_.getToken(), owner_->taskId());
  co_return task->taskStats();
}

void FixedPointLoop::feedInitSplits(
    const exec::TaskPtr& task,
    const core::PlanNodePtr& plan) {
  folly::F14FastMap<core::PlanNodeId, const core::PlanNode*> sources;
  collectSplitTargets(plan, node_, sources);
  // Splits are held under the id of the node that reads them, so each source
  // takes only its own: several sources in one initial plan, or several initial
  // plans that each scan a table, no longer share one list.
  for (const auto& [sourceId, _] : sources) {
    routeSplits(task, sourceId);
  }
}

namespace {

// Returns how many drivers each of 'numPlans' plans that run at the same time
// gets, out of 'budget' drivers in total
// (FixedPointOptions::iterationMaxDrivers).  Every plan gets one driver.  When
// 'budget' exceeds 'numPlans', the other 'budget - numPlans' drivers are dealt
// out one at a time, starting with the first plan.  So the total is
// max(budget, numPlans).
//
// TODO: The split ignores what each plan costs, and the extra drivers favor
// whichever plans come first.  Give every plan the full iterationMaxDrivers,
// as each stage of a distributed query gets its own, and let the executor bound
// how much runs at once.
std::vector<uint32_t> distributeDriverBudget(uint32_t budget, size_t numPlans) {
  std::vector<uint32_t> planDrivers(numPlans, 1);
  if (budget <= numPlans) {
    return planDrivers;
  }
  const size_t numExtraDrivers = budget - numPlans;
  for (size_t i = 0; i < numPlans; ++i) {
    planDrivers[i] += static_cast<uint32_t>(
        numExtraDrivers / numPlans + (i < numExtraDrivers % numPlans ? 1 : 0));
  }
  return planDrivers;
}

} // namespace

uint32_t FixedPointLoop::iterationDrivers() const {
  VELOX_CHECK_LE(
      options_.iterationMaxDrivers,
      maxDrivers_,
      "iterationMaxDrivers must be less than or equal to maxDrivers");
  return options_.iterationMaxDrivers;
}

// TODO: Reuse sub-tasks across iterations to avoid per-iteration planning and
// operator initialization (notably the body's HashBuild rebuilding a constant
// table every iteration).  Not possible via the public API today: Tasks are
// single-shot, barrier/drain reuse requires TableScan leaves with all nodes
// supporting barriers (HashJoin/Exchange/PartitionedOutput do not), and there
// is no DriverFactory-injection path.  The near-term mitigation is hash-table
// reuse via HashTable persistent state; full reuse needs Velox core support.
folly::coro::Task<std::vector<RowVectorPtr>> FixedPointLoop::runParallelChain(
    const std::vector<core::PlanNodePtr>& plans,
    size_t planIndexOffset,
    int32_t iteration,
    memory::MemoryPool* pool,
    bool producesVerdict,
    bool runLastPlan,
    FixedPointChainExecutionStats& stats) {
  checkRunning(cancellation_.getToken(), owner_->taskId());
  folly::stop_watch<std::chrono::nanoseconds> chainTimer;
  const size_t lastPlan = plans.size() - 1;
  const size_t numPlansToRun = runLastPlan ? plans.size() : lastPlan;

  // The last plan's output -- this iteration's rows for the output state entry,
  // collected through a consumer (copied into 'pool', which outlives the
  // sub-task) and returned for the caller to write back.
  std::vector<RowVectorPtr> output;
  std::mutex outputMutex;

  // This worker creates one sub-task per plan, all inheriting its destination
  // so StateSource resolves this worker's state and each Exchange pulls this
  // worker's partition.  Each producer is created under the task id from
  // producerTaskId(), which the peers' producerSplit() calls reach.
  std::vector<exec::TaskPtr> tasks;
  tasks.reserve(numPlansToRun);
  // The budget goes to the plans this worker runs, except a last plan that
  // produces the verdict: it runs on one driver of its own.
  const bool runsVerdict = producesVerdict && runLastPlan;
  auto planDrivers = distributeDriverBudget(
      iterationDrivers(), numPlansToRun - (runsVerdict ? 1 : 0));
  if (runsVerdict) {
    planDrivers.push_back(1);
  }
  {
    // Also covers a throw while later tasks are still being created or wired.
    SCOPE_FAIL {
      abortAndWait(tasks);
    };
    // A plan may itself contain a FixedPointNode, whose sub-task becomes a
    // nested FixedPointLoop running on this loop's workers.
    const bool hasVerdictPlan =
        producesVerdict && node_->convergenceConfig().verdict != nullptr;
    for (size_t p = 0; p < numPlansToRun; ++p) {
      auto subOptions = options_;
      // A loop nested in this plan runs its iterations within the plan's share.
      subOptions.iterationMaxDrivers = planDrivers[p];
      // Only worker 0 runs the verdict plan, so a loop nested in it runs on
      // worker 0 alone.
      auto nesting = subTaskNesting(
          iteration,
          hasVerdictPlan && p == lastPlan
              ? std::vector<std::string>{workers_.at(0)}
              : workers_);
      // The plans of a chain are joined only by shuffle, so the last plan has
      // no peer reading its output and its sub-task is internal.
      const auto createId = p == lastPlan
          ? subTaskId()
          : options_.producerTaskId(
                address(workerIndex_), iteration, plans[p]->id());
      // The last plan produces the rows written back to the output state entry;
      // collect them through a consumer.  Earlier plans shuffle to the next via
      // PartitionedOutput and need no consumer.
      exec::ConsumerSupplier consumerSupplier;
      if (p == lastPlan) {
        consumerSupplier = [&]() -> exec::Consumer {
          return [&](RowVectorPtr batch,
                     bool /*drained*/,
                     ContinueFuture* /*future*/) {
            if (batch != nullptr && batch->size() > 0) {
              folly::stop_watch<std::chrono::nanoseconds> copyTimer;
              auto copy = copyRowVector(batch, pool);
              const auto copyNanos = copyTimer.elapsed().count();
              std::lock_guard<std::mutex> l(outputMutex);
              output.push_back(std::move(copy));
              stats.outputCopyNanos += copyNanos;
            }
            return exec::BlockingReason::kNotBlocked;
          };
        };
      }
      tasks.push_back(createSubTask(
          *owner_,
          createId,
          plans[p],
          exec::Task::ExecutionMode::kParallel,
          std::move(consumerSupplier),
          subOptions,
          nesting));
    }

    // One sub-task per plan: the split wiring below indexes tasks by plan
    // index.
    VELOX_CHECK_EQ(tasks.size(), numPlansToRun);
    stats.taskCreationNanos = chainTimer.elapsed().count();

    {
      NanosecondWallTimer timer(&stats.taskStartNanos);
      for (size_t p = 0; p < tasks.size(); ++p) {
        tasks[p]->start(planDrivers[p]);
        // A broadcast buffer finishes only once it knows all its destinations:
        // every worker of this loop.
        const auto* output = plans[p]->as<core::PartitionedOutputNode>();
        if (output != nullptr && output->isBroadcast()) {
          tasks[p]->updateOutputBuffers(
              static_cast<int>(workers_.size()), /*noMoreBuffers=*/true);
        }
      }
    }

    // Wire each consumer plan's Exchange to plan p-1's producer of every
    // worker, through the split producerSplit() returns.  A peer's producer is
    // created by the peer's task, and its read is held until it exists (peers
    // run concurrently, near lockstep).
    NanosecondWallTimer timer(&stats.exchangeWiringNanos);
    for (size_t p = 1; p < numPlansToRun; ++p) {
      auto exchangeId = findExchangeNodeId(plans[p]);
      VELOX_CHECK(
          exchangeId.has_value(),
          "Non-first plan in a parallel chain must start with an Exchange");
      for (int32_t destination = 0;
           destination < static_cast<int32_t>(workers_.size());
           ++destination) {
        tasks[p]->addSplit(
            *exchangeId,
            exec::Split(options_.producerSplit(
                address(destination), iteration, plans[p - 1]->id())));
      }
      tasks[p]->noMoreSplits(*exchangeId);
    }
  }

  // Waiting on every task is an implicit cross-worker barrier: a producer task
  // finishes only once all the consumers wired to it have drained their
  // partition, so no worker advances to the next iteration until all peers in
  // the topology have shuffled.
  {
    NanosecondWallTimer timer(&stats.taskCompletionWaitNanos);
    co_await awaitTasks(tasks, cancellation_.getToken(), owner_->taskId());
  }
  stats.plans.reserve(tasks.size());
  for (size_t p = 0; p < tasks.size(); ++p) {
    stats.plans.push_back({
        .planIndex = planIndexOffset + p,
        .maxDrivers = planDrivers[p],
        .taskStats = tasks[p]->taskStats(),
    });
  }

  // Deliberately do NOT wait on these sub-tasks' owner_->taskDeletionFuture()
  // here.  A producer sub-task (PartitionedOutput) is read by peer workers, so
  // a peer must outlive it -- waiting for its full teardown would deadlock
  // against the peer.  Each sub-task instead holds the pool of the persistent
  // state it reads, which therefore outlives the sub-task's teardown.
  stats.outputRows = countRows(output);
  stats.totalNanos = chainTimer.elapsed().count();
  co_return output;
}

folly::coro::Task<void> FixedPointLoop::runIteration(int32_t iteration) {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "runIteration() must follow initialize()");
  // The iteration is encoded into the sub-task ids that form the cross-worker
  // barrier, so repeating or going back desynchronizes the peers and deadlocks.
  VELOX_CHECK_GT(
      iteration,
      lastIteration_,
      "Iterations must strictly increase; last was {}",
      lastIteration_);
  lastIteration_ = iteration;
  phase_ = Phase::kIterating;
  folly::stop_watch<std::chrono::nanoseconds> iterationTimer;
  FixedPointIterationExecutionStats iterationStats{.iteration = iteration};
  // The last plan produces this iteration's rows; the framework writes them
  // back into the output state entry, appending or replacing per its
  // declaration.  A scratch pool holds the captured rows until writeVector
  // deep-copies them into the stable state pool.
  auto scratchPool =
      addSubTaskLeafPool(*owner_, fmt::format("fixedpoint.body.{}", iteration));

  // Every sub-task inherits this task's execution mode (the constructor already
  // rejected a serial fixed point with a shuffling body).  Parallel runs the
  // plans as a shuffling chain on the executor and captures the last plan's
  // output; serial has no shuffle, hence a single plan run via next() on the
  // calling thread.
  std::vector<RowVectorPtr> output;
  if (executionMode_ == exec::Task::ExecutionMode::kParallel) {
    output = co_await runParallelChain(
        node_->plans(),
        /*planIndexOffset=*/0,
        iteration,
        scratchPool.get(),
        /*producesVerdict=*/false,
        /*runLastPlan=*/true,
        iterationStats.body);
  } else {
    folly::stop_watch<std::chrono::nanoseconds> bodyTimer;
    auto drained = co_await drainPlan(
        node_->plans().front(),
        iteration,
        scratchPool.get(),
        /*maxDrivers=*/1);
    output = std::move(drained.batches);
    iterationStats.body.totalNanos = bodyTimer.elapsed().count();
    iterationStats.body.outputRows = countRows(output);
    iterationStats.body.plans.push_back({
        .maxDrivers = 1,
        .taskStats = std::move(drained.taskStats),
    });
  }
  // An empty delta is the terminal state of a semi-naive recursion, and it is
  // free to observe here -- stopWhenDeltaEmpty reads it instead of running a
  // sub-task per iteration to recompute what this plan just produced.
  vector_size_t numDeltaRows{0};
  for (const auto& rows : output) {
    numDeltaRows += rows->size();
  }
  lastDeltaEmpty_ = numDeltaRows == 0;
  {
    NanosecondWallTimer timer(&iterationStats.stateWriteNanos);
    state_->writeVector(node_->outputStateEntry(), output);
  }
  iterationStats.totalNanos = iterationTimer.elapsed().count();
  {
    auto stats = executionStats_.wlock();
    if (stats->iterations.size() > kNumLeadingDetailedIterations) {
      auto& previous = stats->iterations.back();
      clearTaskStats(previous.body);
      if (previous.convergence.has_value()) {
        clearTaskStats(*previous.convergence);
      }
    }
    stats->iterations.push_back(std::move(iterationStats));
  }
  ++iterations_;
}

namespace {

// Returns the verdict in 'rows', which must hold exactly one non-NULL BOOLEAN
// row.
bool readVerdict(const std::vector<RowVectorPtr>& rows) {
  // The last convergence plan is contracted (and validated at node
  // construction) to produce exactly one BOOLEAN column.  The row count is a
  // runtime property the node cannot check, so enforce it here: several rows
  // would decide the loop on whichever batch arrived first, and no row leaves
  // no verdict.
  VELOX_USER_CHECK_EQ(
      countRows(rows),
      1u,
      "The convergence sequence must produce exactly one row; aggregate it to "
      "a single verdict");
  const auto& row =
      *std::find_if(rows.begin(), rows.end(), [](const auto& batch) {
        return batch->size() > 0;
      });
  const auto* flag = row->childAt(0)->as<SimpleVector<bool>>();
  VELOX_CHECK_NOT_NULL(flag, "Convergence column must be BOOLEAN");
  // Reading NULL as "not converged" would run out the whole iteration budget
  // and then report non-convergence, hiding that the criterion was malformed.
  VELOX_USER_CHECK(
      !flag->isNullAt(0),
      "The convergence criterion produced a NULL verdict; it must be TRUE or "
      "FALSE");
  return flag->valueAt(0);
}

} // namespace

folly::coro::Task<bool> FixedPointLoop::converged() {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "converged() must follow initialize()");

  const auto& config = node_->convergenceConfig();
  if (config.stopWhenDeltaEmpty) {
    // The node validated that this is a non-shuffling fixed point, so the
    // local delta is the whole delta and no peer is waiting on this worker.
    co_return lastDeltaEmpty_;
  }

  if (convergencePlans_.empty()) {
    // No convergence sequence: the loop is bounded only by maxIterations.
    co_return false;
  }

  auto convergencePool = addSubTaskLeafPool(
      *owner_, fmt::format("fixedpoint.convergence.{}", iterations_));

  // The sequence runs over the state the iteration just committed, chained by
  // shuffle exactly as the body is -- which is what lets a criterion reduce
  // across workers a statistic of that state (PageRank's post-update RMSE, say)
  // that has nowhere to go in the body.  Its producers are indexed after the
  // body's so their per-iteration ids do not collide.
  //
  // A verdict plan runs only on worker 0, whose verdict every worker shares.
  const auto iteration = lastIteration_;
  const bool hasVerdictPlan = config.verdict != nullptr;
  const bool computesVerdict = !hasVerdictPlan || workerIndex_ == 0;
  std::vector<RowVectorPtr> rows;
  FixedPointChainExecutionStats convergenceStats;
  if (executionMode_ == exec::Task::ExecutionMode::kParallel) {
    rows = co_await runParallelChain(
        convergencePlans_,
        /*planIndexOffset=*/node_->plans().size(),
        iteration,
        convergencePool.get(),
        /*producesVerdict=*/true,
        /*runLastPlan=*/computesVerdict,
        convergenceStats);
  } else {
    folly::stop_watch<std::chrono::nanoseconds> convergenceTimer;
    auto drained = co_await drainPlan(
        config.plans.front(),
        iteration,
        convergencePool.get(),
        /*maxDrivers=*/1);
    rows = std::move(drained.batches);
    convergenceStats.totalNanos = convergenceTimer.elapsed().count();
    convergenceStats.outputRows = countRows(rows);
    convergenceStats.plans.push_back({
        .planIndex = node_->plans().size(),
        .maxDrivers = 1,
        .taskStats = std::move(drained.taskStats),
    });
  }

  auto recordStats = [&]() {
    auto stats = executionStats_.wlock();
    auto& iterationStats = stats->iterations.back();
    iterationStats.totalNanos += convergenceStats.totalNanos;
    iterationStats.convergence = std::move(convergenceStats);
  };

  if (computesVerdict) {
    recordStats();
    const bool verdict = readVerdict(rows);
    if (hasVerdictPlan) {
      options_.publishVerdict(address(0), iteration, verdict);
    }
    co_return verdict;
  }

  folly::stop_watch<std::chrono::nanoseconds> waitTimer;
  auto received = co_await folly::coro::co_awaitTry(awaitWhileRunning(
      cancellation_.getToken(),
      options_.awaitVerdict(address(0), iteration, cancellation_.getToken()),
      "the convergence verdict",
      owner_->taskId()));
  convergenceStats.verdictWaitNanos = waitTimer.elapsed().count();
  convergenceStats.totalNanos += convergenceStats.verdictWaitNanos;
  recordStats();
  if (received.hasValue()) {
    co_return received.value();
  }
  const auto& error = received.exception();
  const auto* veloxError = error.get_exception<VeloxException>();
  VELOX_FAIL(
      "Failed to receive the convergence verdict from worker 0 for "
      "iteration {}: {}",
      iteration,
      veloxError != nullptr ? veloxError->message()
                            : error.what().toStdString());
}

FixedPointExecutionStats FixedPointLoop::executionStats() const {
  return *executionStats_.rlock();
}

FixedPointOptions::LoopAddress FixedPointLoop::address(
    int32_t destination) const {
  return FixedPointOptions::LoopAddress{workers_.at(destination), nesting_};
}

std::string FixedPointLoop::subTaskId() {
  // The application owns the id; the worker supplies only its task id and a
  // per-worker counter for uniqueness.
  return options_.subTaskId(owner_->taskId(), subTaskCounter_++);
}

} // namespace facebook::velox::exec
