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

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>

#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/HashTable.h"
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

uint64_t elapsedNanos(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
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
// caller's frame.  A task that runs a nested fixed point has no drivers, so
// this returns at once for it while its loop keeps running; see the
// cancellation TODO on FixedPointLoop.  Runs while an error unwinds, so it logs
// a failure to abort instead of throwing.
void abortAndWait(const std::vector<std::shared_ptr<exec::Task>>& tasks) {
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

// Waits for each of 'tasks' to finish without holding a thread.  If one fails,
// aborts the rest and rethrows its error.  Callers pass a named vector: GCC 11
// fails to compile a braced list in a co_await operand.
folly::coro::Task<void> awaitTasks(
    std::vector<std::shared_ptr<exec::Task>> tasks) {
  for (const auto& task : tasks) {
    co_await task->taskCompletionFuture();
    if (auto error = task->error()) {
      abortAndWait(tasks);
      std::rethrow_exception(error);
    }
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

// Collects the ids of the split sources in the plan, such as a TableScan or an
// Exchange.  A fixed point nested in the plan, other than 'self', contributes
// the sources of its initial plans: the task running it passes their splits
// on.  A fixed point's own id is not collected.
void collectSplitTargets(
    const core::PlanNodePtr& node,
    const core::PlanNode* self,
    folly::F14FastSet<core::PlanNodeId>& ids) {
  if (const auto* fixedPoint = node->as<FixedPointNode>()) {
    if (fixedPoint != self) {
      for (const auto& declaration : fixedPoint->stateDeclarations()) {
        if (declaration->initialPlan() != nullptr) {
          collectSplitTargets(declaration->initialPlan(), self, ids);
        }
      }
    }
  } else if (node->requiresSplits()) {
    ids.insert(node->id());
  }
  for (const auto& source : node->sources()) {
    collectSplitTargets(source, self, ids);
  }
}

// A plan must run in parallel mode if it sends partitioned output or consumes
// from an exchange; serial Task::next() rejects both.
bool planNeedsParallel(const core::PlanNodePtr& node) {
  return node->is<core::PartitionedOutputNode>() ||
      findExchangeNodeId(node).has_value();
}

// Returns the FixedPointNode in 'node's plan tree (the single leaf of the
// fragment), or nullptr if there is none.
core::FixedPointNodePtr findFixedPointNode(const core::PlanNodePtr& node) {
  if (auto fixedPoint = std::dynamic_pointer_cast<const FixedPointNode>(node)) {
    return fixedPoint;
  }
  for (const auto& source : node->sources()) {
    if (auto fixedPoint = findFixedPointNode(source)) {
      return fixedPoint;
    }
  }
  return nullptr;
}

} // namespace

// static
bool FixedPointLoop::claims(
    const core::PlanFragment& planFragment,
    const FixedPointOptions* fixedPointOptions) {
  // A caller can ask for the ordinary Driver pipeline over a plan this would
  // otherwise claim; that is how this loop runs a sub-task over its own
  // fragment without recursing.
  if (fixedPointOptions != nullptr && fixedPointOptions->nested.has_value() &&
      !fixedPointOptions->nested->dispatch) {
    return false;
  }
  // Any plan containing a FixedPointNode (as its single leaf), including one
  // with trailing downstream nodes above it.  (When such a loop runs its
  // trailing sub-task over the same fragment it calls Task::create with
  // NestedFixedPoint::dispatch=false, so that sub-task is a plain Task whose
  // FixedPointNode leaf reads the output state -- no recursion.)
  return findFixedPointNode(planFragment.planNode) != nullptr;
}

FixedPointLoop::FixedPointLoop(
    exec::Task* owner,
    const FixedPointOptions* fixedPointOptions)
    : options_(
          fixedPointOptions != nullptr ? *fixedPointOptions
                                       : FixedPointOptions{}),
      nested_(options_.nested.value_or(NestedFixedPoint{})),
      executionMode_(owner->executionMode()),
      owner_(owner) {
  VELOX_CHECK_NOT_NULL(owner_);
  node_ = findFixedPointNode(owner_->planFragment().planNode);
  VELOX_CHECK_NOT_NULL(
      node_, "FixedPointLoop requires a FixedPointNode in its plan");
  VELOX_USER_CHECK_GE(
      options_.iterationMaxDrivers,
      1,
      "FixedPointOptions::iterationMaxDrivers must be at least 1");
  // The trailing sub-task reuses this fragment, so each split group would
  // re-read the whole output state.
  VELOX_USER_CHECK(
      !owner_->planFragment().isGroupedExecution(),
      "A fixed point does not support grouped execution");
  // Trailing nodes are present when the fragment root is not the
  // FixedPointNode.
  hasTrailing_ = owner_->planFragment().planNode.get() != node_.get();
  workerIndex_ = owner_->destination();

  // One worker per output partition of whichever chain shuffles -- the body,
  // the convergence sequence, or a fixed point nested in either (see
  // FixedPointNode::numWorkers).
  const auto& plans = node_->plans();
  numWorkers_ = node_->numWorkers();
  VELOX_CHECK_LT(
      workerIndex_,
      numWorkers_,
      "Worker destination out of range; create exactly {} FixedPointLoops "
      "with destinations 0..{}",
      numWorkers_,
      numWorkers_ - 1);

  // Every id a caller may address a split to.  The fixed point itself takes a
  // peer worker's remote split; everything else is a split-requiring source in
  // a plan this loop feeds -- an initial plan's scan or exchange, or the
  // trailing plan's scan or the exchange an enclosing loop feeds.  The body and
  // convergence chains wire their own exchanges, so their ids are not here.
  collectSplitTargets(
      owner_->planFragment().planNode, node_.get(), trailingSplitSources_);

  splitTargets_ = trailingSplitSources_;
  splitTargets_.insert(node_->id());

  for (const auto& declaration : node_->stateDeclarations()) {
    if (declaration->initialPlan() != nullptr) {
      collectSplitTargets(
          declaration->initialPlan(), node_.get(), splitTargets_);
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
    for (const auto& plan : node_->convergenceConfig().plans) {
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

  // A RemoteConnectorSplit on the FixedPointNode itself names a peer worker
  // for this loop's shuffle; its per-iteration producer location is derived by
  // producerLocation.  Every other split is read by a specific plan node --
  // a TableScan in an initial plan or the trailing plan, or an Exchange in the
  // trailing plan -- and goes to the sub-task running it, held until that
  // sub-task exists.
  if (split.hasConnectorSplit()) {
    if (const auto* remoteSplit =
            split.connectorSplit->as<exec::RemoteConnectorSplit>()) {
      if (planNodeId == node_->id()) {
        peerParentIds_.push_back(remoteSplit->taskId);
        return;
      }
    }
  }

  VELOX_USER_CHECK_NE(
      planNodeId,
      node_->id(),
      "A connector split must be addressed to the plan node that reads it -- "
      "an initial plan's TableScan, say -- not to the fixed point itself, "
      "which cannot tell which source it was meant for");

  // The driver path rejects an id that takes no splits; do the same here rather
  // than file the split where nothing will read it and return an empty result.
  VELOX_USER_CHECK(
      splitTargets_.contains(planNodeId),
      "Plan node takes no splits: {}",
      planNodeId);

  if (auto it = splitConsumers_.find(planNodeId); it != splitConsumers_.end()) {
    if (auto task = it->second.lock()) {
      task->addSplit(planNodeId, std::move(split));
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
  // The fixed point's own peer splits are captured eagerly in addSplit().
}

RowVectorPtr FixedPointLoop::next(ContinueFuture* /*future*/) {
  switch (phase_) {
    case Phase::kCreated:
      // Mark failed before rethrowing rather than after: a throw part-way
      // through leaves the state half-written, and a retried next() must not
      // re-seed it and rebuild hash tables on top of a half-finished run.
      try {
        folly::coro::blocking_wait(run());
      } catch (...) {
        phase_ = Phase::kFailed;
        throw;
      }
      break;
    case Phase::kFinalized:
      break;
    case Phase::kFailed:
      VELOX_FAIL("The fixed point failed on an earlier call and cannot resume");
    case Phase::kInitialized:
    case Phase::kIterating:
      // run() drives these phases synchronously inside next(), and Task rejects
      // next() on a parallel task, so no caller can observe a mid-run phase.
      VELOX_UNREACHABLE("Fixed point observed mid-run by next()");
  }
  while (current_ < outputBuffer_.size()) {
    auto batch = outputBuffer_[current_++];
    if (batch != nullptr && batch->size() > 0) {
      return batch;
    }
  }
  return nullptr;
}

void FixedPointLoop::start(
    uint32_t maxDrivers,
    uint32_t /*concurrentSplitGroups*/,
    std::function<void(std::exception_ptr)> onComplete) {
  // Task::start() already rejected a serial task before delegating here.
  maxDrivers_ = maxDrivers;
  // Parallel execution: run the loop asynchronously on the orchestration
  // executor so start() returns immediately (Task::start's contract), deliver
  // this worker's shard to the consumer (as a driver pipeline's CallbackSink
  // would), and drive the task to a terminal state so
  // owner_->taskCompletionFuture() / owner_->error() behave as for a
  // driver-based task.  run() is the same coroutine the serial path waits on;
  // here it suspends while its sub-tasks run on the query executor.
  auto* executor = options_.orchestrationExecutor;
  VELOX_CHECK_NOT_NULL(
      executor,
      "Parallel FixedPointLoop requires "
      "FixedPointOptions::orchestrationExecutor, an executor separate from "
      "the query executor its sub-tasks run on");
  // The owning task keeps this loop alive for its whole run:
  // the task owns it, so holding a shared_ptr to the owner is what
  // shared_from_this() did before this was a Task itself.
  VELOX_USER_CHECK(
      phase_ == Phase::kCreated,
      "start() cannot run a fixed point that has already been driven");
  auto ownerHandle = owner_->shared_from_this();
  folly::coro::co_withExecutor(
      folly::getKeepAliveToken(executor), runAndDeliver())
      .start([this, ownerHandle, onComplete = std::move(onComplete)](
                 folly::Try<void>&& result) {
        if (result.hasException()) {
          phase_ = Phase::kFailed;
          onComplete(result.exception().to_exception_ptr());
        } else {
          onComplete(nullptr);
        }
      });
}

folly::coro::Task<void> FixedPointLoop::runAndDeliver() {
  co_await run();
  auto supplier = owner_->consumerSupplier();
  Consumer consumer = supplier ? supplier() : nullptr;
  if (consumer == nullptr) {
    co_return;
  }
  for (const auto& batch : outputBuffer_) {
    if (batch == nullptr || batch->size() == 0) {
      continue;
    }
    ContinueFuture future = ContinueFuture::makeEmpty();
    const auto reason = consumer(batch, /*drained=*/false, &future);
    // Honor back-pressure: a blocked consumer returns a future that resolves
    // when it can take more (as a Driver would wait on it).
    if (reason != exec::BlockingReason::kNotBlocked && future.valid()) {
      co_await std::move(future);
    }
  }
  // Signal end of data, mirroring CallbackSink::close().
  consumer(nullptr, /*drained=*/false, /*future=*/nullptr);
}

folly::coro::Task<void> FixedPointLoop::run() {
  co_await initialize();

  bool hasConverged = false;
  for (int32_t iteration = 0; iteration < node_->maxIterations(); ++iteration) {
    co_await runIteration(iteration);
    const bool iterationConverged = co_await converged();
    executionStats_.wlock()->iterations.back().converged = iterationConverged;
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

folly::coro::Task<void> FixedPointLoop::initialize() {
  VELOX_CHECK(
      phase_ == Phase::kCreated,
      "initialize() must run once, before any other phase");
  const auto start = std::chrono::steady_clock::now();

  // Output batches are copied into a leaf child of the task pool (the task's
  // own pool is an aggregate pool and cannot allocate directly); it outlives
  // the loop so the batches streamed by next() stay valid for the task's
  // lifetime.
  outputPool_ = owner_->pool()->addLeafChild("fixedpoint.output");

  // converged() inspects only this worker's local shard.  With more than one
  // worker, every worker must reach the same verdict on the same iteration or
  // lockstep breaks (a converged worker stops producing while peers' exchanges
  // wait for it, deadlocking).  Making the verdict globally consistent is the
  // plan's responsibility: the convergence-deciding state must be synchronized
  // across workers by the body's shuffle (e.g. replicated), so each worker's
  // local read returns the same value.  There is intentionally no cross-worker
  // convergence reduction here -- that would add an all-reduce shuffle.
  // Keyed by task id, not by query and worker index: one query may run several
  // fixed points at the same worker index -- siblings, or one nested inside
  // another's body -- and this pool is a child of the process-wide root, where
  // a repeated name is an error.
  statePool_ = memory::memoryManager()->addLeafPool(
      fmt::format("fixedpoint.state.{}", owner_->taskId()));
  state_ = std::make_shared<PersistentState>(statePool_);
  // The state lives on this task (state_); its sub-tasks reach it via the
  // parentFixedPoint() link set when they are created -- no global or QueryCtx
  // registry.

  std::vector<FixedPointPlanExecutionStats> initializationPlans;
  co_await initializeState(initializationPlans);
  {
    auto stats = executionStats_.wlock();
    stats->initializationNanos = elapsedNanos(start);
    stats->initializationPlans = std::move(initializationPlans);
  }
  phase_ = Phase::kInitialized;
}

folly::coro::Task<void> FixedPointLoop::finalize() {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "finalize() must follow initialize()");
  const auto start = std::chrono::steady_clock::now();
  // Emit the output.  Without trailing nodes, it is the final contents of the
  // output state entry (output shaping such as a recursive CTE's UNION ALL is
  // the plan's responsibility, e.g. accumulating into an append-mode output
  // entry).  With trailing nodes above the FixedPointNode, run them over that
  // state and emit their result.
  if (hasTrailing_) {
    co_await runTrailingPlan();
  } else {
    appendOutput(node_->outputStateEntry());
  }
  executionStats_.wlock()->finalizationNanos = elapsedNanos(start);
  // A next() after this -- including one after a parallel run -- streams the
  // buffered output instead of running the loop again.
  phase_ = Phase::kFinalized;

  // The state pool is intentionally NOT released here.  A sub-task driver can
  // still be tearing down asynchronously after its
  // owner_->taskCompletionFuture() has resolved, holding persistent-state
  // vectors; releasing the pool now would race that teardown and trip
  // ~MemoryPoolImpl.  state_/statePool_ are members released only in the
  // FixedPointLoop destructor, by which point every sub-task created during
  // the loop is long gone.  (Waiting on each sub-task's
  // owner_->taskDeletionFuture() before releasing is not viable: a shuffle
  // producer is read by peer workers, so a peer must outlive it -- the wait
  // would deadlock.)
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
          vector->initialPlan(), outputPool_.get(), maxDrivers_);
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
      *reinterpret_cast<char**>(newRow + nextOffset) = nullptr;
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
      declaration.initialPlan(), numDrivers, [&]() -> exec::Consumer {
        auto part = createHashTable(
            declaration, queryCtx.queryConfig(), state_->pool());
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

void FixedPointLoop::appendOutput(const std::string& stateEntry) {
  for (const auto& batch : state_->getVector(stateEntry)) {
    if (batch != nullptr && batch->size() > 0) {
      outputBuffer_.push_back(copyRowVector(batch, outputPool_.get()));
    }
  }
}

void FixedPointLoop::routeSplits(
    const std::shared_ptr<exec::Task>& task,
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

FixedPointOptions FixedPointLoop::subTaskOptions() {
  FixedPointOptions subOptions = options_;
  // Rebuilt rather than inherited: the enclosing nesting state describes where
  // *this* loop sits, and handing it down would give the sub-task an address
  // and a trailing id belonging to the level above.
  subOptions.nested.emplace();
  subOptions.nested->parent = this;
  return subOptions;
}

folly::coro::Task<void> FixedPointLoop::runTrailingPlan() {
  // Sub-tasks run over this same fragment, so they must not recurse into
  // another FixedPointLoop (NestedFixedPoint::dispatch=false); the
  // FixedPointNode leaf instead compiles to a state-reading operator (see the
  // operator translator).
  auto trailingOptions = subTaskOptions();
  trailingOptions.nested->dispatch = false;
  // The enclosing loop sets trailingTaskId when this fragment is one of its
  // shuffle producers: the PartitionedOutput runs here, in the trailing
  // sub-task, so this is the id its consumers are wired to.
  std::string trailingTaskId = nested_.trailingTaskId;
  if (trailingTaskId.empty()) {
    trailingTaskId = subTaskId();
  }
  if (executionMode_ == exec::Task::ExecutionMode::kSerial) {
    // An empty ConsumerSupplier selects the supplier overload; the trailing
    // plan's output is drained via next() below.
    auto task = exec::Task::create(
        trailingTaskId,
        owner_->planFragment(),
        /*destination=*/workerIndex_,
        owner_->queryCtx(),
        exec::Task::ExecutionMode::kSerial,
        exec::ConsumerSupplier{},
        /*memoryArbitrationPriority=*/0,
        /*spillDiskOpts=*/std::nullopt,
        /*onError=*/nullptr,
        &trailingOptions);
    SCOPE_FAIL {
      abortAndWait({task});
    };
    // The FixedPointNode is a split source when the body shuffles; in the
    // trailing sub-task it is read as state, so close its (empty) split source.
    if (node_->requiresSplits()) {
      task->noMoreSplits(node_->id());
    }
    for (const auto& sourceId : trailingSplitSources_) {
      routeSplits(task, sourceId);
    }
    while (auto batch = task->next()) {
      outputBuffer_.push_back(copyRowVector(batch, outputPool_.get()));
    }
    executionStats_.wlock()->trailingPlan = FixedPointPlanExecutionStats{
        .maxDrivers = 1,
        .taskStats = task->taskStats(),
    };
    co_return;
  }

  // The trailing plan re-shuffles; run it in parallel and collect through a
  // consumer (guarded, as it runs on the sub-task's driver thread).
  std::mutex mutex;
  auto consumer =
      [&](RowVectorPtr batch, bool /*drained*/, ContinueFuture* /*future*/) {
        if (batch != nullptr && batch->size() > 0) {
          auto copy = copyRowVector(batch, outputPool_.get());
          std::lock_guard<std::mutex> l(mutex);
          outputBuffer_.push_back(std::move(copy));
        }
        return exec::BlockingReason::kNotBlocked;
      };
  auto task = exec::Task::create(
      trailingTaskId,
      owner_->planFragment(),
      /*destination=*/workerIndex_,
      owner_->queryCtx(),
      exec::Task::ExecutionMode::kParallel,
      ConsumerSupplier{[consumer]() -> Consumer { return consumer; }},
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/std::nullopt,
      /*onError=*/nullptr,
      &trailingOptions);
  {
    SCOPE_FAIL {
      abortAndWait({task});
    };
    task->start(iterationDrivers());
    if (node_->requiresSplits()) {
      task->noMoreSplits(node_->id());
    }
    for (const auto& sourceId : trailingSplitSources_) {
      routeSplits(task, sourceId);
    }
  }
  std::vector<std::shared_ptr<exec::Task>> tasks{task};
  co_await awaitTasks(std::move(tasks));
  executionStats_.wlock()->trailingPlan = FixedPointPlanExecutionStats{
      .maxDrivers = iterationDrivers(),
      .taskStats = task->taskStats(),
  };
}

folly::coro::Task<FixedPointLoop::DrainedPlan> FixedPointLoop::drainPlan(
    core::PlanNodePtr plan,
    memory::MemoryPool* pool,
    uint32_t maxDrivers) {
  std::vector<RowVectorPtr> batches;
  std::mutex mutex;
  auto taskStats =
      co_await consumePlan(plan, maxDrivers, [&]() -> exec::Consumer {
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

folly::coro::Task<TaskStats> FixedPointLoop::consumePlan(
    core::PlanNodePtr plan,
    uint32_t maxDrivers,
    exec::ConsumerSupplier consumerSupplier) {
  const auto subOptions = subTaskOptions();
  if (executionMode_ == exec::Task::ExecutionMode::kSerial) {
    auto task = exec::Task::create(
        subTaskId(),
        core::PlanFragment{plan},
        /*destination=*/workerIndex_,
        owner_->queryCtx(),
        exec::Task::ExecutionMode::kSerial,
        exec::ConsumerSupplier{},
        /*memoryArbitrationPriority=*/0,
        /*spillDiskOpts=*/std::nullopt,
        /*onError=*/nullptr,
        &subOptions);
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

  // The supplier gives each output driver its own consumer, called on that
  // driver's thread.
  auto task = exec::Task::create(
      subTaskId(),
      core::PlanFragment{plan},
      /*destination=*/workerIndex_,
      owner_->queryCtx(),
      exec::Task::ExecutionMode::kParallel,
      consumerSupplier,
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/std::nullopt,
      /*onError=*/nullptr,
      &subOptions);
  {
    SCOPE_FAIL {
      abortAndWait({task});
    };
    task->start(maxDrivers);
    feedInitSplits(task, plan);
  }
  std::vector<std::shared_ptr<exec::Task>> tasks{task};
  co_await awaitTasks(std::move(tasks));
  co_return task->taskStats();
}

void FixedPointLoop::feedInitSplits(
    const std::shared_ptr<exec::Task>& task,
    const core::PlanNodePtr& plan) {
  // An Exchange source reads the upstream producer the coordinator launched and
  // named via options_.upstreamExchangeUri (keyed by the Exchange node id); the
  // worker pulls its own partition by task destination.
  if (auto exchangeId = findExchangeNodeId(plan)) {
    VELOX_CHECK(
        options_.upstreamExchangeUri,
        "Initial plan reads via Exchange {}; the coordinator must set "
        "FixedPointOptions::upstreamExchangeUri",
        *exchangeId);
    // This branch feeds the Exchange and nothing else, so a connector split the
    // coordinator also assigned has no source to go to.  Fail rather than drop
    // it silently, which would just produce missing rows.
    {
      // Scoped to this plan's own source.  nodeSplits_ is task-wide and
      // legitimately holds other initial plans' splits -- a fixed point may mix
      // an Exchange-fed state with a scanned one -- so only a split addressed
      // to this Exchange is a mistake.
      std::lock_guard<std::mutex> l(splitsMutex_);
      auto it = nodeSplits_.find(*exchangeId);
      VELOX_USER_CHECK(
          it == nodeSplits_.end() || it->second.empty(),
          "Initial plan reads via Exchange {}, whose upstream comes from "
          "FixedPointOptions::upstreamExchangeUri, but the coordinator also "
          "assigned it {} connector splits",
          *exchangeId,
          it->second.size());
    }
    task->addSplit(
        *exchangeId,
        exec::Split(
            std::make_shared<exec::RemoteConnectorSplit>(
                options_.upstreamExchangeUri(*exchangeId))));
    task->noMoreSplits(*exchangeId);
    return;
  }
  // Any other split source (e.g. a TableScan) reads the coordinator-assigned
  // source splits; a local source (e.g. Values) needs none.
  folly::F14FastSet<core::PlanNodeId> sourceIds;
  collectSplitTargets(plan, node_.get(), sourceIds);
  // Splits are held under the id of the node that reads them, so each source
  // takes only its own: several sources in one initial plan, or several initial
  // plans that each scan a table, no longer share one list.
  for (const auto& sourceId : sourceIds) {
    routeSplits(task, sourceId);
  }
}

namespace {

// Splits 'budget' drivers across 'numPlans' plans that run at the same time.
// Every plan gets at least one driver, so the total exceeds 'budget' when there
// are more plans than drivers.  With 'singleDriverLastPlan' the last plan gets
// exactly one and the others share the rest.
std::vector<uint32_t> distributeDriverBudget(
    uint32_t budget,
    size_t numPlans,
    bool singleDriverLastPlan) {
  std::vector<uint32_t> planDrivers(numPlans, 1);
  const size_t numScalablePlans = numPlans - (singleDriverLastPlan ? 1 : 0);
  if (numScalablePlans == 0 || budget <= numPlans) {
    return planDrivers;
  }
  const size_t numExtraDrivers = budget - numPlans;
  for (size_t i = 0; i < numScalablePlans; ++i) {
    planDrivers[i] += static_cast<uint32_t>(
        numExtraDrivers / numScalablePlans +
        (i < numExtraDrivers % numScalablePlans ? 1 : 0));
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
    bool singleDriverLastPlan,
    FixedPointChainExecutionStats& stats) {
  const auto chainStart = std::chrono::steady_clock::now();
  const size_t lastPlan = plans.size() - 1;

  // The last plan's output -- this iteration's rows for the output state entry,
  // collected through a consumer (copied into 'pool', which outlives the
  // sub-task) and returned for the caller to write back.
  std::vector<RowVectorPtr> output;
  std::mutex outputMutex;

  // This worker creates one sub-task per plan, all inheriting its destination
  // so StateSource resolves this worker's state and each Exchange pulls this
  // worker's partition.  Each producer is created under the task id from
  // producerLocation() -- a pure function of (this worker's task id, iteration,
  // plan) -- so a peer can address it knowing only this worker's task id.
  std::vector<std::shared_ptr<exec::Task>> tasks;
  tasks.reserve(plans.size());
  const auto planDrivers = distributeDriverBudget(
      iterationDrivers(), plans.size(), singleDriverLastPlan);
  {
    // Also covers a throw while later tasks are still being created or wired.
    SCOPE_FAIL {
      abortAndWait(tasks);
    };
    // A plan may itself contain a FixedPointNode, whose sub-task becomes a
    // nested FixedPointLoop.  It cannot address peers by task id, so hand it
    // the worker addresses extended by the (iteration, plan) this loop is
    // running -- every peer extends its own the same way, so the nested loops
    // find each other.
    const auto peers = peerAddresses();
    for (size_t p = 0; p < plans.size(); ++p) {
      auto subOptions = subTaskOptions();
      // A loop nested in this plan runs its iterations within the plan's share.
      subOptions.iterationMaxDrivers = planDrivers[p];
      subOptions.nested->workerAddress =
          nestedAddress(workerAddress(), iteration, planIndexOffset + p);
      subOptions.nested->peerAddresses.reserve(peers.size());
      for (const auto& peer : peers) {
        subOptions.nested->peerAddresses.push_back(
            nestedAddress(peer, iteration, planIndexOffset + p));
      }
      const auto producerId =
          producerLocation(workerAddress(), iteration, planIndexOffset + p)
              .taskId;
      auto createId = producerId;
      if (findFixedPointNode(plans[p]) != nullptr) {
        // This plan nests a fixed point, so its sub-task is a FixedPointLoop
        // and the plan's own nodes run in that task's trailing sub-task.  Both
        // ends of this loop's shuffle therefore have to be redirected there.
        //
        // Producing: only the trailing sub-task can carry the id peers read, so
        // the wrapper takes a derived one -- sharing would collide on the
        // per-task memory pool.
        if (plans[p]->is<core::PartitionedOutputNode>()) {
          createId = fmt::format("{}.fp", producerId);
          subOptions.nested->trailingTaskId = producerId;
        }
      }
      // The last plan produces the rows written back to the output state entry;
      // collect them through a consumer.  Earlier plans shuffle to the next via
      // PartitionedOutput and need no consumer.
      if (p == lastPlan) {
        auto consumer = [&](RowVectorPtr batch,
                            bool /*drained*/,
                            ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            const auto copyStart = std::chrono::steady_clock::now();
            auto copy = copyRowVector(batch, pool);
            const auto copyNanos = elapsedNanos(copyStart);
            std::lock_guard<std::mutex> l(outputMutex);
            output.push_back(std::move(copy));
            stats.outputCopyNanos += copyNanos;
          }
          return exec::BlockingReason::kNotBlocked;
        };
        tasks.push_back(
            exec::Task::create(
                createId,
                core::PlanFragment{plans[p]},
                /*destination=*/workerIndex_,
                owner_->queryCtx(),
                exec::Task::ExecutionMode::kParallel,
                consumer,
                /*memoryArbitrationPriority=*/0,
                /*spillDiskOpts=*/std::nullopt,
                /*onError=*/nullptr,
                &subOptions));
      } else {
        tasks.push_back(
            exec::Task::create(
                createId,
                core::PlanFragment{plans[p]},
                /*destination=*/workerIndex_,
                owner_->queryCtx(),
                exec::Task::ExecutionMode::kParallel,
                exec::ConsumerSupplier{},
                /*memoryArbitrationPriority=*/0,
                /*spillDiskOpts=*/std::nullopt,
                /*onError=*/nullptr,
                &subOptions));
      }
    }

    // One sub-task per plan: the split wiring below indexes tasks by plan
    // index.
    VELOX_CHECK_EQ(tasks.size(), plans.size());
    stats.taskCreationNanos = elapsedNanos(chainStart);
    const auto taskStartBegin = std::chrono::steady_clock::now();
    for (size_t p = 0; p < tasks.size(); ++p) {
      tasks[p]->start(planDrivers[p]);
    }
    stats.taskStartNanos = elapsedNanos(taskStartBegin);

    // Wire each consumer plan's Exchange to plan p-1's producer of every peer
    // the coordinator named, reaching each by the exchange URI
    // producerLocation() derives from that peer's task id.  A peer's producer
    // is created by the peer's task; the Exchange polls until it registers
    // (peers run concurrently, near lockstep).
    const auto wiringStart = std::chrono::steady_clock::now();
    for (size_t p = 1; p < plans.size(); ++p) {
      auto exchangeId = findExchangeNodeId(plans[p]);
      VELOX_CHECK(
          exchangeId.has_value(),
          "Non-first plan in a parallel chain must start with an Exchange");
      for (const auto& peer : peerAddresses()) {
        tasks[p]->addSplit(
            *exchangeId,
            exec::Split(
                std::make_shared<exec::RemoteConnectorSplit>(
                    producerLocation(peer, iteration, planIndexOffset + p - 1)
                        .exchangeUri)));
      }
      tasks[p]->noMoreSplits(*exchangeId);
    }
    stats.exchangeWiringNanos = elapsedNanos(wiringStart);
  }

  // Waiting on every task is an implicit cross-worker barrier: a producer task
  // finishes only once all the consumers wired to it have drained their
  // partition, so no worker advances to the next iteration until all peers in
  // the topology have shuffled.
  const auto waitStart = std::chrono::steady_clock::now();
  co_await awaitTasks(tasks);
  stats.plans.reserve(tasks.size());
  for (size_t p = 0; p < tasks.size(); ++p) {
    stats.plans.push_back({
        .planIndex = planIndexOffset + p,
        .maxDrivers = planDrivers[p],
        .taskStats = tasks[p]->taskStats(),
    });
  }
  stats.taskCompletionWaitNanos = elapsedNanos(waitStart);

  // Deliberately do NOT wait on these sub-tasks' owner_->taskDeletionFuture()
  // here.  A producer sub-task (PartitionedOutput) is read by peer workers, so
  // a peer must outlive it -- waiting for its full teardown would deadlock
  // against the peer.  Their persistent-state references are instead kept safe
  // by the state pool living for the whole FixedPointLoop lifetime (it is
  // released in the destructor, not eagerly in run()).
  stats.outputRows = countRows(output);
  stats.totalNanos = elapsedNanos(chainStart);
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
  const auto iterationStart = std::chrono::steady_clock::now();
  FixedPointIterationExecutionStats iterationStats{.iteration = iteration};
  // The last plan produces this iteration's rows; the framework writes them
  // back into the output state entry, appending or replacing per its
  // declaration.  A scratch pool holds the captured rows until writeVector
  // deep-copies them into the stable state pool.
  auto scratchPool = owner_->pool()->addLeafChild(
      fmt::format("fixedpoint.body.{}", iteration));

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
        /*singleDriverLastPlan=*/false,
        iterationStats.body);
  } else {
    const auto bodyStart = std::chrono::steady_clock::now();
    auto drained = co_await drainPlan(
        node_->plans().front(), scratchPool.get(), /*maxDrivers=*/1);
    output = std::move(drained.batches);
    iterationStats.body.totalNanos = elapsedNanos(bodyStart);
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
  const auto writeStart = std::chrono::steady_clock::now();
  state_->writeVector(node_->outputStateEntry(), output);
  iterationStats.stateWriteNanos = elapsedNanos(writeStart);
  iterationStats.totalNanos = elapsedNanos(iterationStart);
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
  if (config.plans.empty()) {
    // No convergence sequence: the loop is bounded only by maxIterations.
    co_return false;
  }
  auto convergencePool = owner_->pool()->addLeafChild(
      fmt::format("fixedpoint.convergence.{}", iterations_));
  // The sequence runs over the state the iteration just committed, chained by
  // shuffle exactly as the body is -- which is what lets a criterion reduce
  // across workers a statistic of that state (PageRank's post-update RMSE, say)
  // that has nowhere to go in the body.  Its producers are indexed after the
  // body's so their per-iteration ids do not collide.
  std::vector<RowVectorPtr> rows;
  FixedPointChainExecutionStats convergenceStats;
  if (executionMode_ == exec::Task::ExecutionMode::kParallel) {
    rows = co_await runParallelChain(
        config.plans,
        /*planIndexOffset=*/node_->plans().size(),
        static_cast<int32_t>(iterations_),
        convergencePool.get(),
        /*singleDriverLastPlan=*/true,
        convergenceStats);
  } else {
    const auto convergenceStart = std::chrono::steady_clock::now();
    auto drained = co_await drainPlan(
        config.plans.front(), convergencePool.get(), /*maxDrivers=*/1);
    rows = std::move(drained.batches);
    convergenceStats.totalNanos = elapsedNanos(convergenceStart);
    convergenceStats.outputRows = countRows(rows);
    convergenceStats.plans.push_back({
        .planIndex = node_->plans().size(),
        .maxDrivers = 1,
        .taskStats = std::move(drained.taskStats),
    });
  }
  {
    auto stats = executionStats_.wlock();
    auto& iterationStats = stats->iterations.back();
    iterationStats.totalNanos += convergenceStats.totalNanos;
    iterationStats.convergence = std::move(convergenceStats);
  }
  // The last convergence plan is contracted (and validated at node
  // construction) to produce exactly one BOOLEAN column.  The row count is a
  // runtime property the node cannot check, so enforce it here: a sequence
  // emitting several rows would otherwise decide the loop on whichever batch
  // arrived first, silently and nondeterministically.
  vector_size_t numRows{0};
  for (const auto& row : rows) {
    numRows += row->size();
  }
  VELOX_USER_CHECK_LE(
      numRows,
      1,
      "The convergence sequence must produce at most one row, but produced {}; "
      "aggregate it to a single verdict",
      numRows);
  for (const auto& row : rows) {
    if (row->size() == 0) {
      continue;
    }
    auto flag = row->childAt(0)->asFlatVector<bool>();
    VELOX_CHECK_NOT_NULL(flag, "Convergence column must be a flat BOOLEAN");
    // Reading NULL as "not converged" would run out the whole iteration budget
    // and then report non-convergence, hiding that the criterion was malformed.
    VELOX_USER_CHECK(
        !flag->isNullAt(0),
        "The convergence criterion produced a NULL verdict; it must be TRUE or "
        "FALSE");
    co_return flag->valueAt(0);
  }
  // No rows means an empty terminal state, which is converged by definition.
  co_return true;
}

FixedPointExecutionStats FixedPointLoop::executionStats() const {
  return *executionStats_.rlock();
}

WorkerAddress FixedPointLoop::workerAddress() const {
  if (!nested_.workerAddress.rootTaskId.empty()) {
    return nested_.workerAddress;
  }
  // A top-level worker is its own root, and its address is its task id.
  return WorkerAddress{owner_->taskId(), owner_->taskId()};
}

std::vector<WorkerAddress> FixedPointLoop::peerAddresses() {
  if (!nested_.peerAddresses.empty()) {
    return nested_.peerAddresses;
  }
  // The coordinator names top-level peers by task id, which is also their
  // address.
  std::lock_guard<std::mutex> l(splitsMutex_);
  std::vector<WorkerAddress> peers;
  peers.reserve(peerParentIds_.size());
  for (const auto& peerTaskId : peerParentIds_) {
    peers.push_back(WorkerAddress{peerTaskId, peerTaskId});
  }
  return peers;
}

// static
WorkerAddress FixedPointLoop::nestedAddress(
    const WorkerAddress& worker,
    int32_t iteration,
    size_t planIndex) {
  return WorkerAddress{
      worker.rootTaskId,
      fmt::format("{}/it{}.p{}", worker.address, iteration, planIndex)};
}

std::string FixedPointLoop::subTaskId() {
  // The coordinator owns the id; the worker supplies only its task id and a
  // per-worker counter for uniqueness.
  VELOX_CHECK(
      options_.subTaskId,
      "A fixed point requires FixedPointOptions::subTaskId");
  return options_.subTaskId(owner_->taskId(), subTaskCounter_++);
}

ProducerLocation FixedPointLoop::producerLocation(
    const WorkerAddress& worker,
    int32_t iteration,
    size_t planIndex) const {
  // The coordinator owns all ids/URIs; the fixed point generates none.  The ids
  // MUST encode the iteration: per-iteration sub-tasks are re-created each
  // round, and reusing a prior round's ids desynchronizes the cross-worker
  // barrier and deadlocks.
  VELOX_CHECK(
      options_.producerLocation,
      "A shuffling fixed point requires "
      "FixedPointOptions::producerLocation");
  return options_.producerLocation(
      worker.rootTaskId, worker.address, iteration, planIndex);
}

} // namespace facebook::velox::exec
