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

#include <folly/CancellationToken.h>
#include <folly/Executor.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/coro/Task.h>
#include <folly/futures/SharedPromise.h>

#include <optional>

#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/FixedPointOptions.h"
#include "velox/exec/PersistentState.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

class FixedPointLoop;

/// The nesting protocol's internal state, set by an enclosing FixedPointLoop on
/// a sub-task it creates.  A coordinator never fills this in: it describes
/// where a sub-task sits inside a loop that is already running, which only that
/// loop knows.
struct NestedFixedPoint {
  /// The loop that created the sub-task.  Its State* operators reach that
  /// loop's persistent state by walking Task::parentFixedPoint() outward.
  FixedPointLoop* parent{nullptr};

  /// The workers the nested loop runs on, by destination: the enclosing
  /// loop's, or worker 0 alone for a loop nested in a verdict plan.
  std::vector<std::string> workers{};

  /// The enclosing loop's nesting.  The nested loop appends a step naming
  /// itself and 'iteration'.
  std::vector<FixedPointOptions::NestingStep> nesting{};

  /// The enclosing loop's iteration that runs the sub-task; 0 for an initial
  /// plan.
  int32_t iteration{0};
};

/// Experimental and subject to change, like FixedPointExecutionStats.
/// Statistics of one sub-task a fixed point ran.
struct FixedPointPlanExecutionStats {
  /// Index of the plan in the body plans followed by the convergence plans; 0
  /// for an initial plan.
  size_t planIndex{0};

  /// Driver limit the sub-task was started with.
  uint32_t maxDrivers{0};

  /// The sub-task's statistics.  Empty for a plan that nests a fixed point,
  /// whose work runs in that nested loop's sub-tasks.
  /// TODO: Report the nested loop's own FixedPointExecutionStats.
  TaskStats taskStats{};
};

/// Experimental and subject to change, like FixedPointExecutionStats.
/// Statistics of one run of a body or convergence plan chain.
/// Serial mode runs a single plan per chain, so it reports only totalNanos,
/// outputRows and plans; the other timings stay zero.
struct FixedPointChainExecutionStats {
  /// Time to create the chain's sub-tasks.
  uint64_t taskCreationNanos{0};

  /// Time to start them.
  uint64_t taskStartNanos{0};

  /// Time to wire each consumer plan to its peers' producers.
  uint64_t exchangeWiringNanos{0};

  /// Time from the end of wiring until every sub-task completed.
  uint64_t taskCompletionWaitNanos{0};

  /// Time spent copying the last plan's output, summed across its drivers, so
  /// it can exceed totalNanos.
  uint64_t outputCopyNanos{0};

  /// Time a worker other than worker 0 waited for worker 0's verdict, whether
  /// or not it arrived.  Part of totalNanos.
  uint64_t verdictWaitNanos{0};

  /// Wall time of the whole chain.
  uint64_t totalNanos{0};

  /// Rows the last plan produced.
  uint64_t outputRows{0};

  /// One entry per plan, in chain order.
  std::vector<FixedPointPlanExecutionStats> plans{};
};

/// Experimental and subject to change, like FixedPointExecutionStats.
/// Statistics of one loop iteration.
struct FixedPointIterationExecutionStats {
  /// Zero-based iteration number.
  int32_t iteration{0};

  /// The body chain.
  FixedPointChainExecutionStats body{};

  /// Time to write the body's output into the output state entry.
  uint64_t stateWriteNanos{0};

  /// The convergence chain, when the node has one.
  std::optional<FixedPointChainExecutionStats> convergence{};

  /// Whether this iteration converged.
  bool converged{false};

  /// Wall time of the body, the state write and the convergence chain.
  uint64_t totalNanos{0};
};

/// Experimental: this struct and the structs it holds may change without
/// notice, so projects outside Velox should not depend on them yet.
/// Statistics of a fixed point's run so far.
struct FixedPointExecutionStats {
  /// Time to create the persistent state and run the initial plans.
  uint64_t initializationNanos{0};

  /// One entry per initial plan, in declaration order.
  std::vector<FixedPointPlanExecutionStats> initializationPlans{};

  /// One entry per completed iteration.  To bound memory, only the first
  /// three iterations and the latest one keep their sub-tasks' TaskStats; the
  /// others keep empty TaskStats.
  std::vector<FixedPointIterationExecutionStats> iterations{};
};

/// Runs the iterations of the FixedPointNode leaf of its owning task's plan.
/// Instead of a Driver pipeline it orchestrates the iteration's sub-tasks
/// itself -- above the Operator layer, at task scope (the loop runs once per
/// task, not once per driver) -- which keeps task orchestration out of
/// operators and bounds the parallelism.
///
/// One FixedPointLoop is one worker: its owning task's destination is the
/// worker / shard index.  A fixed point that reads its peers
/// (FixedPointNode::readsPeers()) runs as N peer tasks; the application creates
/// them like any partitioned stage (via Task::create) and names every worker to
/// each of them with a FixedPointSplit.  A fixed point that reads no peers may
/// also run as several tasks, each an independent loop over the rows its
/// initial plans read.
///
/// It orchestrates ordinary Tasks rather than being another kind of Task: the
/// per-iteration sub-tasks it creates are plain driver-based tasks.  The
/// owning task runs its own plan as a Driver pipeline, in which the
/// FixedPointNode leaf waits for the loop to finish and then reads the output
/// state entry.  The nodes above the leaf, including a PartitionedOutput, run
/// in that pipeline like in any task.
///
/// What the owning Task's API means for a fixed point:
///
///  - start() starts the loop on the orchestration executor alongside the
///    task's drivers; serial next() runs the loop on the caller's thread before
///    its first batch.
///  - addSplit() hands the loop the splits it routes -- the FixedPointSplit
///    on the FixedPointNode, or an initial plan's source splits -- and
///    the task's own drivers every other split.
///  - A failed loop fails the task.
///  - taskStats() covers the task's own pipeline.  The experimental
///    Task::fixedPointExecutionStats() reports the loop's sub-tasks.
///
/// In parallel mode, when the owning task ends -- requestCancel(),
/// requestAbort(), or a failure -- the loop stops at its next wait: it aborts
/// the sub-tasks it is running, waits for them, and fails.  The loop waits on
/// its sub-tasks, its peers and the verdict, so it stops within one step of the
/// work it was doing.  A serial loop runs to completion inside the caller's
/// next().  A nested loop holds the enclosing loop's task, so the state it
/// reads outlives a failed enclosing loop until the nested loop stops.
/// requestPause() does not end the task, and the loop keeps running through a
/// pause.
class FixedPointLoop {
 public:
  /// 'owner' is the fully constructed Task this loop runs the fragment for; it
  /// owns the loop and so outlives it. 'node' is the FixedPointNode in the
  /// owner's plan fragment.
  /// 'fixedPointOptions' supplies the hooks; the constructor checks that the
  /// ones this fixed point needs are set.
  FixedPointLoop(
      exec::Task* owner,
      const core::FixedPointNode& node,
      const FixedPointOptions* fixedPointOptions,
      const NestedFixedPoint* nested);

  exec::Task* owner() const {
    return owner_;
  }

  /// Whether splits for 'planNodeId' go to this loop: the FixedPointNode, whose
  /// split names the workers, or a source in an initial plan.
  bool takesSplits(const core::PlanNodeId& planNodeId) const {
    return splitTargets_.contains(planNodeId);
  }

  /// Captures the application's splits -- the FixedPointSplit naming the
  /// workers, or an initial plan's source splits.
  void addSplit(const core::PlanNodeId& planNodeId, exec::Split&& split);

  void noMoreSplits(const core::PlanNodeId& planNodeId);

  /// Serial execution: runs the whole fixed point on the caller's thread.
  void runSerially();

  /// Parallel execution: runs the loop asynchronously on the orchestration
  /// executor.  'maxDrivers' is used for the Phase-1 initial plans, which read
  /// real sources (TableScan/Exchange) that parallelize naturally.  The body
  /// and convergence plans run with up to
  /// FixedPointOptions::iterationMaxDrivers drivers, which must not exceed
  /// 'maxDrivers'.  'onError' runs on the orchestration executor if the loop
  /// fails.
  void start(
      uint32_t maxDrivers,
      std::function<void(std::exception_ptr)> onError);

  /// Resolves when the loop has finished, successfully or not.
  ContinueFuture finishedFuture() {
    return finished_.getSemiFuture();
  }

  /// True once the loop has finished successfully, so the output state entry
  /// holds its final contents.
  bool succeeded() const {
    return succeeded_;
  }

  /// Number of iterations the loop ran (converged early or hit maxIterations).
  int64_t iterations() const {
    return iterations_;
  }

  /// Returns a snapshot of the execution statistics so far.  Safe to call from
  /// any thread, during the run or after it ends, including after a failure.
  FixedPointExecutionStats executionStats() const;

  /// This worker's persistent state, owned by this loop.  Its sub-tasks'
  /// StateSource/StateHashJoin operators reach it through
  /// Task::parentFixedPoint(). Created
  /// at the start of run() and held for the rest of the task's lifetime (it is
  /// released in the destructor, not eagerly at loop end, so a sub-task driver
  /// still tearing down asynchronously cannot outlive the state pool it
  /// references).  Null before run().
  PersistentState* state() const {
    return state_.get();
  }

 private:
  // The three phases run() composes, in order.  Private: runSerially() and
  // start() are the only ways to drive a fixed point.
  //
  // Phase 1: creates the persistent state and seeds each declared entry from
  // its initial plan.
  folly::coro::Task<void> initialize();

  // Phase 2: runs one iteration's plans, writing the result back into the
  // output state entry.  'iteration' must strictly increase -- it is encoded
  // into the sub-task ids that form the cross-worker barrier.
  folly::coro::Task<void> runIteration(int32_t iteration);

  // Returns true if the convergence criterion signals convergence; false when
  // there is no criterion.  Evaluated over this worker's local state: either
  // the last iteration's delta was empty (stopWhenDeltaEmpty) or the
  // convergence sequence produced true.
  folly::coro::Task<bool> converged();

  // Phase 3: marks the output state entry final and resolves
  // finishedFuture().
  folly::coro::Task<void> finalize();

  // True when the last iteration wrote no rows into the output state entry.
  // Read only for a ConvergenceConfig with stopWhenDeltaEmpty.
  bool lastDeltaEmpty_{false};

  // Composes initialize() + the runIteration() / converged() loop (bounded by
  // maxIterations) + finalize().  Called by runSerially() and start().
  folly::coro::Task<void> run();

  // Waits for the FixedPointSplit naming the workers of a top-level loop that
  // reads its peers, and checks it.  Called before the first plan that needs
  // the workers: the first iteration, or an initial plan hosting a nested
  // loop (see consumePlan()).  The other initial plans run while the
  // application names the workers.
  folly::coro::Task<void> awaitWorkers();

  // Gives 'task' the splits held for 'sourceId' and forwards later ones to it
  // as they arrive, through noMoreSplits. A serial loop requires the splits to
  // be complete, since its caller cannot add splits while the loop runs.
  void routeSplits(const exec::TaskPtr& task, const core::PlanNodeId& sourceId);

  // Nesting state for a sub-task this loop creates in 'iteration' (0 for an
  // initial plan), naming this loop as the parent.  A loop nested in the
  // sub-task runs on 'workers'.
  NestedFixedPoint subTaskNesting(
      int32_t iteration,
      std::vector<std::string> workers);

  // Phase 1: runs each state's initial plan and stores this worker's shard,
  // recording each plan into 'initializationPlans'.
  folly::coro::Task<void> initializeState(
      std::vector<FixedPointPlanExecutionStats>& initializationPlans);

  // Builds a join hash table once from its initial plan and stores it.  Each
  // output driver of the initial plan fills its own table as rows arrive, and
  // the tables are merged at the end, in parallel on the query executor for a
  // large table.  Returns the initial plan's statistics.
  folly::coro::Task<FixedPointPlanExecutionStats> buildHashTable(
      const core::HashTableStateDeclaration& declaration);

  // Runs this worker's column of per-plan sub-tasks over the chain 'plans' for
  // 'iteration', wiring each consumer plan's Exchange to the previous-plan
  // producer of every worker.  Returns the last plan's output copied into
  // 'pool'.  Serves both chains a fixed point runs: 'planIndexOffset' is added
  // to the plan index in the statistics, so the body's plans (offset 0) and the
  // convergence sequence's (offset node_->plans().size()) are told apart.  The
  // plans this worker runs share iterationDrivers().
  // 'producesVerdict' says the last plan produces the convergence verdict, a
  // single row, so it runs on one driver of its own, outside the budget.
  // 'runLastPlan' false skips the last plan on this worker; the chain's other
  // plans still run and feed the peers that run it.  Records the run into
  // 'stats'.
  folly::coro::Task<std::vector<RowVectorPtr>> runParallelChain(
      const std::vector<core::PlanNodePtr>& plans,
      size_t planIndexOffset,
      int32_t iteration,
      memory::MemoryPool* pool,
      bool producesVerdict,
      bool runLastPlan,
      FixedPointChainExecutionStats& stats);

  // Driver budget for the plans run after initialization; see
  // FixedPointOptions::iterationMaxDrivers.
  uint32_t iterationDrivers() const;

  // The output of a sub-task drainPlan ran, and the sub-task's statistics.
  struct DrainedPlan {
    std::vector<RowVectorPtr> batches;
    TaskStats taskStats;
  };

  // Runs 'plan' as a sub-task in this worker's execution mode and returns its
  // output batches, copied into 'pool'.  Serial mode drains via next() on the
  // calling thread (single-driver); parallel mode starts the sub-task on the
  // executor with 'maxDrivers' drivers and collects through a consumer.  Feeds
  // the plan's split source (TableScan files or an Exchange upstream) when it
  // has one.  Used for the initial plans (with the parent maxDrivers) and, in
  // serial mode, the body and convergence plans.  'iteration' is 0 for an
  // initial plan.
  folly::coro::Task<DrainedPlan> drainPlan(
      core::PlanNodePtr plan,
      int32_t iteration,
      memory::MemoryPool* pool,
      uint32_t maxDrivers);

  // Runs 'plan' as a sub-task in this worker's execution mode and feeds its
  // split source, handing its output to consumers from 'consumerSupplier': one
  // per output driver in parallel mode, a single one on the calling thread in
  // serial mode.  Returns the sub-task's statistics.
  folly::coro::Task<TaskStats> consumePlan(
      core::PlanNodePtr plan,
      int32_t iteration,
      uint32_t maxDrivers,
      exec::ConsumerSupplier consumerSupplier);

  // Gives 'task' the coordinator-assigned splits of each split source in 'plan'
  // -- a TableScan, or an Exchange reading an upstream stage.  No-op when the
  // plan has no split source (e.g. Values).
  void feedInitSplits(const exec::TaskPtr& task, const core::PlanNodePtr& plan);

  // Label for an internal sub-task of this worker that no peer addresses (an
  // initial, convergence, serial-body, or last-in-chain sub-task).
  std::string subTaskId();

  // The address of this loop on the worker at 'destination'.
  FixedPointOptions::LoopAddress address(int32_t destination) const;

  // The application's hooks (see FixedPointOptions).
  FixedPointOptions options_;

  // The nesting state an enclosing loop handed down, default-constructed when a
  // coordinator created this task directly.
  NestedFixedPoint nested_;

  // The task of the enclosing loop, for a nested loop.  Holding it keeps the
  // enclosing loop and the state this loop's sub-tasks read alive after the
  // enclosing loop fails and unwinds, until this loop stops.
  std::shared_ptr<exec::Task> enclosingTask_;

  // This worker's execution mode (from construction).  Every sub-task inherits
  // it: serial drains on the calling thread, parallel runs on the executor.  A
  // serial fixed point therefore cannot run a shuffling body (rejected in the
  // constructor).
  const exec::Task::ExecutionMode executionMode_;

  // The task that owns this loop; outlives it by construction.
  exec::Task* const owner_;

  // The parent task's maxDrivers (from start()), used for the Phase-1 initial
  // plans -- they read real sources (TableScan / Exchange / upstream) that
  // parallelize naturally -- and as a cap on iterationDrivers().  Serial mode
  // (runSerially()) is always single-driver, so this stays 1.
  uint32_t maxDrivers_{1};

  // Lives in the owning task's plan fragment, which outlives this loop.
  const core::FixedPointNode* const node_;

  // The convergence plans in the order they run: the node's convergence
  // 'plans', then its 'verdict' plan, when set.
  std::vector<core::PlanNodePtr> convergencePlans_;

  int32_t workerIndex_{0};
  int32_t numWorkers_{1};

  // Guards the splits an application adds, from several threads perhaps,
  // until the loop reads them.
  std::mutex splitsMutex_;

  // The workers this loop runs on, by destination: from the FixedPointSplit
  // of a top-level loop, taken by awaitWorkers(), or from the enclosing loop.
  // Empty for a top-level loop that reads no peers.  Written only on the
  // loop's thread, before any plan reads it.
  std::vector<std::string> workers_;

  // Where this loop sits within its worker (see
  // FixedPointOptions::LoopAddress).
  std::vector<FixedPointOptions::NestingStep> nesting_;

  // Resolved by noMoreSplits() on the fixed point itself (see
  // awaitWorkers()).
  folly::SharedPromise<folly::Unit> workersComplete_;

  // The workers the FixedPointSplit names, from addSplit(), which an
  // application may call from any thread.  Guarded by splitsMutex_.
  std::optional<std::vector<std::string>> splitWorkers_;

  // Set once awaitWorkers() has received and checked the workers.
  bool workersReady_{false};

  // Requested once the owning task ends: the loop stops waiting on its peers
  // and the application, which then no longer need to answer.
  folly::CancellationSource cancellation_;

  // Every other coordinator-assigned split that arrived before the sub-task
  // reading it started, keyed by the plan node that reads it: a TableScan or
  // an Exchange in an initial plan.  Keyed rather than pooled so two initial
  // plans that each scan a table receive only their own splits.
  folly::F14FastMap<core::PlanNodeId, std::vector<exec::Split>> nodeSplits_;

  // The sub-task that reads each such node, until its noMoreSplits.  Weak, so
  // a sub-task that failed or was cancelled is not kept alive.
  folly::F14FastMap<core::PlanNodeId, std::weak_ptr<exec::Task>>
      splitConsumers_;

  // Nodes the coordinator (or an enclosing loop) has finished assigning splits
  // to.  Guarded by splitsMutex_, as are nodeSplits_ and splitConsumers_.
  folly::F14FastSet<core::PlanNodeId> splitsComplete_;

  // Stable pool for persistent state, independent of sub-task pools.
  std::shared_ptr<memory::MemoryPool> statePool_;
  std::shared_ptr<PersistentState> state_;

  int64_t subTaskCounter_{0};
  int64_t iterations_{0};

  // Written by the loop thread as each phase completes; read by
  // executionStats() from any thread.
  folly::Synchronized<FixedPointExecutionStats> executionStats_;

  // Resolved when the loop finishes, successfully or not.
  folly::SharedPromise<folly::Unit> finished_;
  std::atomic_bool succeeded_{false};

  // Where the loop is in the phase sequence.  run() checks each transition,
  // because getting the order wrong is silent -- initialize() twice re-seeds
  // the state mid-run, and runIteration() first reads state that was never
  // created.
  enum class Phase {
    // Nothing run yet.  runSerially() and start() run the whole loop from here.
    kCreated,
    // initialize() done; runIteration(), converged() or finalize() may follow.
    kInitialized,
    // At least one runIteration() done.
    kIterating,
    // finalize() done: the output state entry is final.
    kFinalized,
    // A run threw.  The state is half-written, so nothing may run again.
    kFailed,
  };
  Phase phase_{Phase::kCreated};

  // The last iteration passed to runIteration(), so a repeated or decreasing
  // one is rejected: the iteration is encoded into the sub-task ids that form
  // the cross-worker barrier, and reusing one desynchronizes the peers.
  int32_t lastIteration_{-1};

  // Plan node ids a caller may address splits to.  The driver path rejects an
  // unknown id in getPlanNodeSplitsStateLocked; without the same check here a
  // split would be filed under an id nothing ever reads, and the query would
  // return nothing rather than fail.
  folly::F14FastSet<core::PlanNodeId> splitTargets_;
};

} // namespace facebook::velox::exec
