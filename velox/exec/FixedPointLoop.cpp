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

#include <mutex>

#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/HashTable.h"
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

// Returns the id of the first ExchangeNode found in the plan, if any.
std::optional<core::PlanNodeId> findExchangeNodeId(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const core::ExchangeNode>(node) != nullptr) {
    return node->id();
  }
  for (const auto& source : node->sources()) {
    if (auto id = findExchangeNodeId(source)) {
      return id;
    }
  }
  return std::nullopt;
}

// Collects the ids of every split-requiring source node in the plan (e.g.
// TableScan or Exchange).  An initial plan whose source needs splits (reads a
// table or receives a shuffle) has one; a local source (Values) has none.
void collectSplitSourceNodeIds(
    const core::PlanNodePtr& node,
    std::vector<core::PlanNodeId>& ids) {
  if (node->requiresSplits()) {
    ids.push_back(node->id());
    return;
  }
  for (const auto& source : node->sources()) {
    collectSplitSourceNodeIds(source, ids);
  }
}

// Collects the ids of every split-requiring node in the plan.  Unlike
// collectSplitSourceNodeIds this does not stop at the first one: a fragment can
// hold both the FixedPointNode (itself a split source when the body shuffles)
// and a trailing Exchange above it.
void collectSplitTargets(
    const core::PlanNodePtr& node,
    folly::F14FastSet<core::PlanNodeId>& ids) {
  if (node->requiresSplits()) {
    ids.insert(node->id());
  }
  for (const auto& source : node->sources()) {
    collectSplitTargets(source, ids);
  }
}

bool rootIsPartitionedOutput(const core::PlanNodePtr& node) {
  return std::dynamic_pointer_cast<const core::PartitionedOutputNode>(node) !=
      nullptr;
}

// A plan must run in parallel mode if it sends partitioned output or consumes
// from an exchange; serial Task::next() rejects both.
bool planNeedsParallel(const core::PlanNodePtr& node) {
  return rootIsPartitionedOutput(node) || findExchangeNodeId(node).has_value();
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
  // trailing plan's exchange when an enclosing loop feeds it.  The body and
  // convergence chains wire their own exchanges, so their ids are not here.
  splitTargets_.insert(node_->id());
  collectSplitTargets(owner_->planFragment().planNode, splitTargets_);
  for (const auto& declaration : node_->stateDeclarations()) {
    if (declaration->initialPlan() != nullptr) {
      collectSplitTargets(declaration->initialPlan(), splitTargets_);
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
  // an initial plan's TableScan, or an Exchange in the trailing plan -- and is
  // held under that node's id until the sub-task running it exists.
  auto remoteSplit = std::dynamic_pointer_cast<exec::RemoteConnectorSplit>(
      split.connectorSplit);
  if (remoteSplit != nullptr && planNodeId == node_->id()) {
    peerParentIds_.push_back(remoteSplit->taskId);
    return;
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
  nodeSplits_[planNodeId].push_back(std::move(split));
}

void FixedPointLoop::noMoreSplits(const core::PlanNodeId& planNodeId) {
  if (planNodeId != node_->id()) {
    // Whoever assigns a node's splits is done with it; runTrailingPlan waits
    // for this before running a trailing plan an enclosing loop feeds.
    {
      std::lock_guard<std::mutex> l(splitsMutex_);
      splitsComplete_.insert(planNodeId);
    }
    trailingSplitsCv_.notify_all();
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
        run();
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
  // The Phase-1 initial plans run at the parent's maxDrivers; the body,
  // convergence, and trailing sub-tasks always run single-driver.
  maxDrivers_ = maxDrivers;
  // Parallel execution: run the loop asynchronously on the orchestration
  // executor so start() returns immediately (Task::start's contract), deliver
  // this worker's shard to the consumer (as a driver pipeline's CallbackSink
  // would), and drive the task to a terminal state so
  // owner_->taskCompletionFuture() / owner_->error() behave as for a
  // driver-based task.  run() itself is unchanged from the serial path; it
  // creates the per-iteration sub-tasks on this same QueryCtx, whose drivers
  // run on the query executor -- which is why the loop must not occupy a thread
  // of that executor itself (see FixedPointOptions::orchestrationExecutor).
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
  executor->add([this, ownerHandle, onComplete = std::move(onComplete)]() {
    try {
      run();
      auto supplier = owner_->consumerSupplier();
      Consumer consumer = supplier ? supplier() : nullptr;
      if (consumer != nullptr) {
        for (const auto& batch : outputBuffer_) {
          if (batch == nullptr || batch->size() == 0) {
            continue;
          }
          ContinueFuture future = ContinueFuture::makeEmpty();
          const auto reason = consumer(batch, /*drained=*/false, &future);
          // Honor back-pressure: a blocked consumer returns a future that
          // resolves when it can take more (as a Driver would wait on it).
          if (reason != exec::BlockingReason::kNotBlocked && future.valid()) {
            std::move(future).wait();
          }
        }
        // Signal end of data, mirroring CallbackSink::close().
        consumer(nullptr, /*drained=*/false, /*future=*/nullptr);
      }
      onComplete(nullptr);
    } catch (...) {
      phase_ = Phase::kFailed;
      onComplete(std::current_exception());
    }
  });
}

void FixedPointLoop::run() {
  initialize();

  bool hasConverged = false;
  for (int32_t iteration = 0; iteration < node_->maxIterations(); ++iteration) {
    runIteration(iteration);
    if (converged()) {
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

  finalize();
}

void FixedPointLoop::initialize() {
  VELOX_CHECK(
      phase_ == Phase::kCreated,
      "initialize() must run once, before any other phase");
  const auto queryId = owner_->queryCtx()->queryId();

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

  initializeState();
  phase_ = Phase::kInitialized;
}

void FixedPointLoop::finalize() {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "finalize() must follow initialize()");
  // Emit the output.  Without trailing nodes, it is the final contents of the
  // output state entry (output shaping such as a recursive CTE's UNION ALL is
  // the plan's responsibility, e.g. accumulating into an append-mode output
  // entry).  With trailing nodes above the FixedPointNode, run them over that
  // state and emit their result.
  if (hasTrailing_) {
    runTrailingPlan();
  } else {
    appendOutput(node_->outputStateEntry());
  }
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

void FixedPointLoop::initializeState() {
  for (const auto& declaration : node_->stateDeclarations()) {
    if (auto vector = std::dynamic_pointer_cast<const VectorStateDeclaration>(
            declaration)) {
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
      state_->setVector(
          vector->name(),
          drainPlan(vector->initialPlan(), outputPool_.get(), maxDrivers_));
      continue;
    }
    if (auto hashTable =
            std::dynamic_pointer_cast<const HashTableStateDeclaration>(
                declaration)) {
      buildHashTable(*hashTable);
      continue;
    }
    VELOX_UNREACHABLE(
        "Unknown state declaration type: {}", declaration->name());
  }
}

void FixedPointLoop::buildHashTable(
    const HashTableStateDeclaration& declaration) {
  const auto& schema = declaration.schema();
  const auto& keyColumns = declaration.keyColumns();
  const auto numKeys = static_cast<int32_t>(keyColumns.size());

  // Prototype requirement: key columns are the leading columns, so a build
  // input column index equals its RowContainer column index (keys first, then
  // dependents), and a probe input shares the same key channels.
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

  auto table = exec::HashTable<true>::createForJoin(
      std::move(hashers),
      dependentTypes,
      /*allowDuplicates=*/true,
      /*hasProbedFlag=*/false,
      /*hasCountFlag=*/false,
      /*minTableSizeForParallelJoinBuild=*/1'000,
      state_->pool());

  auto buildPool = owner_->pool()->addLeafChild("fixedpoint.hashbuild");
  auto batches =
      drainPlan(declaration.initialPlan(), buildPool.get(), maxDrivers_);
  auto* rowContainer = table->rows();
  // With allowDuplicates the container chains same-key rows through a
  // next-link; it must start null on each new row so chain insertion is
  // well-formed.
  const auto nextOffset = rowContainer->nextOffset();
  for (const auto& batch : batches) {
    std::vector<DecodedVector> decoded;
    decoded.reserve(batch->childrenSize());
    for (const auto& child : batch->children()) {
      decoded.emplace_back(*child);
    }
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      char* newRow = rowContainer->newRow();
      if (nextOffset > 0) {
        *reinterpret_cast<char**>(newRow + nextOffset) = nullptr;
      }
      for (int32_t column = 0; column < static_cast<int32_t>(decoded.size());
           ++column) {
        rowContainer->store(decoded[column], row, newRow, column);
      }
    }
  }
  table->prepareJoinTable(
      {}, exec::BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  state_->setHashTable(
      declaration.name(),
      HashTableEntry{
          std::shared_ptr<exec::BaseHashTable>(std::move(table)),
          schema,
          numKeys,
      });
}

void FixedPointLoop::appendOutput(const std::string& stateEntry) {
  for (const auto& batch : state_->getVector(stateEntry)) {
    if (batch != nullptr && batch->size() > 0) {
      outputBuffer_.push_back(copyRowVector(batch, outputPool_.get()));
    }
  }
}

void FixedPointLoop::feedTrailingSplits(exec::Task& task) {
  if (nested_.trailingExchangeId.empty()) {
    return;
  }
  const auto& exchangeId = nested_.trailingExchangeId;
  std::vector<exec::Split> splits;
  {
    // The enclosing loop starts this task before assigning the splits, so they
    // may not have arrived yet; running the trailing plan now would read an
    // empty Exchange and emit nothing.  Wait for its noMoreSplits.
    std::unique_lock<std::mutex> l(splitsMutex_);
    trailingSplitsCv_.wait(
        l, [&] { return splitsComplete_.count(exchangeId) > 0; });
    auto it = nodeSplits_.find(exchangeId);
    if (it != nodeSplits_.end()) {
      splits = std::move(it->second);
      nodeSplits_.erase(it);
    }
  }
  for (auto& split : splits) {
    task.addSplit(exchangeId, std::move(split));
  }
  task.noMoreSplits(exchangeId);
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

void FixedPointLoop::runTrailingPlan() {
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
    // The FixedPointNode is a split source when the body shuffles; in the
    // trailing sub-task it is read as state, so close its (empty) split source.
    if (node_->requiresSplits()) {
      task->noMoreSplits(node_->id());
    }
    feedTrailingSplits(*task);
    while (auto batch = task->next()) {
      outputBuffer_.push_back(copyRowVector(batch, outputPool_.get()));
    }
    return;
  }

  // The trailing plan re-shuffles; run it in parallel and collect through a
  // consumer (guarded, as it runs on the sub-task's driver thread).
  std::mutex mutex;
  auto consumer =
      [&](RowVectorPtr batch, bool /*drained*/, ContinueFuture* /*future*/) {
        if (batch != nullptr && batch->size() > 0) {
          std::lock_guard<std::mutex> l(mutex);
          outputBuffer_.push_back(copyRowVector(batch, outputPool_.get()));
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
  task->start(/*maxDrivers=*/1);
  if (node_->requiresSplits()) {
    task->noMoreSplits(node_->id());
  }
  feedTrailingSplits(*task);
  auto future = task->taskCompletionFuture();
  std::move(future).wait();
  if (auto error = task->error()) {
    std::rethrow_exception(error);
  }
}

std::vector<RowVectorPtr> FixedPointLoop::drainPlan(
    const core::PlanNodePtr& plan,
    memory::MemoryPool* pool,
    uint32_t maxDrivers) {
  std::vector<RowVectorPtr> batches;
  if (executionMode_ == exec::Task::ExecutionMode::kSerial) {
    const auto subOptions = subTaskOptions();
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
    feedInitSplits(*task, plan);
    while (auto batch = task->next()) {
      batches.push_back(copyRowVector(batch, pool));
    }
    return batches;
  }

  // Parallel: start on the executor and collect through a consumer (guarded, as
  // it runs on the sub-task's driver thread).  The sub-task's pool is released
  // on completion, so each batch is copied into 'pool'.
  std::mutex mutex;
  auto consumer =
      [&](RowVectorPtr batch, bool /*drained*/, ContinueFuture* /*future*/) {
        if (batch != nullptr && batch->size() > 0) {
          std::lock_guard<std::mutex> l(mutex);
          batches.push_back(copyRowVector(batch, pool));
        }
        return exec::BlockingReason::kNotBlocked;
      };
  const auto subOptions = subTaskOptions();
  auto task = exec::Task::create(
      subTaskId(),
      core::PlanFragment{plan},
      /*destination=*/workerIndex_,
      owner_->queryCtx(),
      exec::Task::ExecutionMode::kParallel,
      consumer,
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/std::nullopt,
      /*onError=*/nullptr,
      &subOptions);
  task->start(maxDrivers);
  feedInitSplits(*task, plan);
  auto future = task->taskCompletionFuture();
  std::move(future).wait();
  if (auto error = task->error()) {
    std::rethrow_exception(error);
  }
  return batches;
}

void FixedPointLoop::feedInitSplits(
    exec::Task& task,
    const core::PlanNodePtr& plan) {
  // splitsMutex_ is taken per lookup below, not held across the addSplit calls
  // into the sub-task.
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
    task.addSplit(
        *exchangeId,
        exec::Split(
            std::make_shared<exec::RemoteConnectorSplit>(
                options_.upstreamExchangeUri(*exchangeId))));
    task.noMoreSplits(*exchangeId);
    return;
  }
  // Any other split source (e.g. a TableScan) reads the coordinator-assigned
  // source splits; a local source (e.g. Values) needs none.
  std::vector<core::PlanNodeId> sourceIds;
  collectSplitSourceNodeIds(plan, sourceIds);
  if (sourceIds.empty()) {
    return;
  }
  // Splits are held under the id of the node that reads them, so each source
  // takes only its own: several sources in one initial plan, or several initial
  // plans that each scan a table, no longer share one list.
  for (const auto& sourceId : sourceIds) {
    std::vector<exec::Split> splits;
    {
      std::lock_guard<std::mutex> l(splitsMutex_);
      auto it = nodeSplits_.find(sourceId);
      VELOX_USER_CHECK(
          it != nodeSplits_.end() && !it->second.empty(),
          "Initial plan source {} requires splits; the coordinator must assign "
          "them to that node (e.g. a TableScan file split)",
          sourceId);
      splits = it->second;
    }
    for (auto& split : splits) {
      task.addSplit(sourceId, std::move(split));
    }
    task.noMoreSplits(sourceId);
  }
}

// TODO: Reuse sub-tasks across iterations to avoid per-iteration planning and
// operator initialization (notably the body's HashBuild rebuilding a constant
// table every iteration).  Not possible via the public API today: Tasks are
// single-shot, barrier/drain reuse requires TableScan leaves with all nodes
// supporting barriers (HashJoin/Exchange/PartitionedOutput do not), and there
// is no DriverFactory-injection path.  The near-term mitigation is hash-table
// reuse via HashTable persistent state; full reuse needs Velox core support.
std::vector<RowVectorPtr> FixedPointLoop::runParallelChain(
    const std::vector<core::PlanNodePtr>& plans,
    size_t planIndexOffset,
    int32_t iteration,
    memory::MemoryPool* pool) {
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
  // A plan may itself contain a FixedPointNode, whose sub-task becomes a nested
  // FixedPointLoop.  It cannot address peers by task id, so hand it the
  // worker addresses extended by the (iteration, plan) this loop is running --
  // every peer extends its own the same way, so the nested loops find each
  // other.
  const auto peers = peerAddresses();
  for (size_t p = 0; p < plans.size(); ++p) {
    auto subOptions = subTaskOptions();
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
      if (rootIsPartitionedOutput(plans[p])) {
        createId = fmt::format("{}.fp", producerId);
        subOptions.nested->trailingTaskId = producerId;
      }
      // Consuming: the Exchange this loop feeds below lives in the trailing
      // plan, so name it for the nested task to route the splits to.
      if (p > 0) {
        auto exchangeId = findExchangeNodeId(plans[p]);
        VELOX_CHECK(
            exchangeId.has_value(),
            "Non-first plan in a parallel chain must start with an Exchange");
        subOptions.nested->trailingExchangeId = *exchangeId;
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
          std::lock_guard<std::mutex> l(outputMutex);
          output.push_back(copyRowVector(batch, pool));
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

  // One sub-task per plan: the split wiring below indexes tasks by plan index.
  VELOX_CHECK_EQ(tasks.size(), plans.size());
  for (auto& task : tasks) {
    task->start(/*maxDrivers=*/1);
  }

  // Wire each consumer plan's Exchange to plan p-1's producer of every peer the
  // coordinator named, reaching each by the exchange URI producerLocation()
  // derives from that peer's task id.  A peer's producer is created by the
  // peer's task; the Exchange polls until it registers (peers run concurrently,
  // near lockstep).
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

  // Waiting on every task is an implicit cross-worker barrier: a producer task
  // finishes only once all the consumers wired to it have drained their
  // partition, so no worker advances to the next iteration until all peers in
  // the topology have shuffled.
  for (auto& task : tasks) {
    auto future = task->taskCompletionFuture();
    std::move(future).wait();
    if (auto error = task->error()) {
      std::rethrow_exception(error);
    }
  }

  // Deliberately do NOT wait on these sub-tasks' owner_->taskDeletionFuture()
  // here.  A producer sub-task (PartitionedOutput) is read by peer workers, so
  // a peer must outlive it -- waiting for its full teardown would deadlock
  // against the peer.  Their persistent-state references are instead kept safe
  // by the state pool living for the whole FixedPointLoop lifetime (it is
  // released in the destructor, not eagerly in run()).
  return output;
}

void FixedPointLoop::runIteration(int32_t iteration) {
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
    output = runParallelChain(
        node_->plans(), /*planIndexOffset=*/0, iteration, scratchPool.get());
  } else {
    output =
        drainPlan(node_->plans().front(), scratchPool.get(), /*maxDrivers=*/1);
  }
  // An empty delta is the terminal state of a semi-naive recursion, and it is
  // free to observe here -- stopWhenDeltaEmpty reads it instead of running a
  // sub-task per iteration to recompute what this plan just produced.
  vector_size_t numDeltaRows{0};
  for (const auto& rows : output) {
    numDeltaRows += rows->size();
  }
  lastDeltaEmpty_ = numDeltaRows == 0;
  state_->writeVector(node_->outputStateEntry(), output);
  ++iterations_;
}

bool FixedPointLoop::converged() {
  VELOX_CHECK(
      phase_ == Phase::kInitialized || phase_ == Phase::kIterating,
      "converged() must follow initialize()");
  const auto& config = node_->convergenceConfig();
  if (config.stopWhenDeltaEmpty) {
    // The node validated that this is a non-shuffling fixed point, so the
    // local delta is the whole delta and no peer is waiting on this worker.
    return lastDeltaEmpty_;
  }
  if (config.plans.empty()) {
    // No convergence sequence: the loop is bounded only by maxIterations.
    return false;
  }
  auto convergencePool = owner_->pool()->addLeafChild(
      fmt::format("fixedpoint.convergence.{}", iterations_));
  // The sequence runs over the state the iteration just committed, chained by
  // shuffle exactly as the body is -- which is what lets a criterion reduce
  // across workers a statistic of that state (PageRank's post-update RMSE, say)
  // that has nowhere to go in the body.  Its producers are indexed after the
  // body's so their per-iteration ids do not collide.
  std::vector<RowVectorPtr> rows;
  if (executionMode_ == exec::Task::ExecutionMode::kParallel) {
    rows = runParallelChain(
        config.plans,
        /*planIndexOffset=*/node_->plans().size(),
        static_cast<int32_t>(iterations_),
        convergencePool.get());
  } else {
    rows = drainPlan(
        config.plans.front(), convergencePool.get(), /*maxDrivers=*/1);
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
    return flag->valueAt(0);
  }
  // No rows means an empty terminal state, which is converged by definition.
  return true;
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
