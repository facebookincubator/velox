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

#include <folly/Executor.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include <condition_variable>
#include <optional>

#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/PersistentState.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

class FixedPointLoop;

/// Where a producer sub-task lives: the bare task id it is created under (for
/// Task::create and its output buffer) and the exchange URI a consumer reaches
/// it through (a RemoteConnectorSplit -> ExchangeSource, selected by scheme).
/// In-process the two are the same string; a distributed coordinator's URI also
/// embeds the producer's endpoint, e.g. "thrift://{host:port}/{taskId}".
struct ProducerLocation {
  /// Task id the producer is created under.
  std::string taskId;
  /// Exchange URI a consumer reaches the producer through.
  std::string exchangeUri;
};

/// A worker's shuffle identity.  'rootTaskId' names the top-level worker task,
/// and so the process it runs in -- nesting never relocates execution, so a
/// nested loop shares its root's endpoint.  'address' additionally locates the
/// loop within that worker: equal to 'rootTaskId' for a top-level fixed point,
/// extended per nesting level below it.
struct WorkerAddress {
  std::string rootTaskId;
  std::string address;
};

/// The nesting protocol's internal state, set by an enclosing FixedPointLoop on
/// a sub-task it creates.  A coordinator never fills this in: it describes
/// where a sub-task sits inside a loop that is already running, which only that
/// loop knows.
struct NestedFixedPoint {
  /// The loop that created the sub-task.  Its State* operators reach that
  /// loop's persistent state by walking Task::parentFixedPoint() outward.
  FixedPointLoop* parent{nullptr};

  /// A shuffling fixed point addresses its producers by worker: its own
  /// through 'workerAddress', a peer's through the matching entry of
  /// 'peerAddresses'.  For a top-level worker those are just task ids -- its
  /// own, and its peers' as named by the coordinator's remote splits.  A
  /// nested fixed point cannot use task ids: its own comes from subTaskId()
  /// and is worker-local, so a peer could never derive it.  The enclosing loop
  /// therefore hands down addresses extended by the (iteration, planIndex) it
  /// is running, which every peer extends identically because they are in
  /// lockstep on the same iteration of the same plan.  Nesting composes: each
  /// level appends its own suffix.
  WorkerAddress workerAddress;
  std::vector<WorkerAddress> peerAddresses;

  /// When the sub-task's fragment ends in a PartitionedOutput, the enclosing
  /// loop's consumers read the id it created the sub-task under -- but the
  /// PartitionedOutput actually runs in that sub-task's own trailing sub-task,
  /// so that one must carry the id.  The enclosing loop therefore creates the
  /// sub-task under a derived id and puts the id peers read here.  Empty means
  /// the trailing sub-task names itself through subTaskId(), as it does when
  /// nothing reads it remotely.
  std::string trailingTaskId;

  /// Set only when the sub-task's fragment consumes the enclosing loop's
  /// shuffle: the id of the Exchange in the trailing plan that loop assigns
  /// splits to.  Those splits arrive after the sub-task has already started, so
  /// the trailing plan cannot just take whatever has shown up -- it waits for
  /// the enclosing loop's noMoreSplits, or it would read a still-empty Exchange
  /// and silently produce nothing.  Empty means no enclosing shuffle feeds this
  /// fragment.
  core::PlanNodeId trailingExchangeId;

  /// When false, a fragment containing a FixedPointNode compiles to an ordinary
  /// Driver pipeline instead of dispatching to a FixedPointLoop -- which is
  /// what a loop asks for when it runs its trailing sub-task over its own
  /// fragment and must not recurse into itself.
  bool dispatch{true};
};

/// Hooks a coordinator passes to Task::create for a plan containing a
/// FixedPointNode, which Task composes a FixedPointLoop into.  A FixedPointLoop
/// generates no task ids or exchange URIs itself -- the coordinator owns its
/// addressing protocol and supplies them all through these hooks.
struct FixedPointOptions {
  /// Returns where a worker's producer for ('iteration', 'planIndex') lives:
  /// the task id the worker creates it under, and the exchange URI a peer
  /// consumer reaches it through.  Required for a shuffling fixed point (one
  /// whose plans chain by Exchange).  The ids MUST encode the iteration:
  /// per-iteration sub-tasks are re-created each round, so reusing the previous
  /// round's ids desynchronizes the cross-worker barrier and deadlocks.
  ///
  /// The worker is named twice.  'rootWorkerTaskId' is the top-level worker
  /// task id, which identifies the process and is what an endpoint lookup
  /// should key on; it is unaffected by nesting, since a nested loop runs in
  /// its root's process.  'workerAddress' additionally locates the loop within
  /// that worker and is what the returned id must be unique in -- for a
  /// top-level fixed point the two are equal, so a coordinator that ignores
  /// nesting can just use 'workerAddress'.
  std::function<ProducerLocation(
      const std::string& rootWorkerTaskId,
      const std::string& workerAddress,
      int32_t iteration,
      size_t planIndex)>
      producerLocation;

  /// Returns the task id for an internal sub-task this worker creates -- an
  /// initial, convergence, serial-body, or trailing sub-task that no peer reads
  /// -- from the worker's task id and a per-worker counter (for uniqueness).
  /// Required: the fixed point creates no task ids itself.
  std::function<std::string(const std::string& workerTaskId, int64_t counter)>
      subTaskId;

  /// Returns the exchange URI of the upstream producer feeding an initial
  /// plan's Exchange, keyed by that Exchange's plan node id.  Required when an
  /// initial plan reads its shard through an Exchange (the coordinator launches
  /// the upstream); unused otherwise.
  std::function<std::string(const core::PlanNodeId& exchangeNodeId)>
      upstreamExchangeUri;

  /// Runs the iteration loop of a FixedPointLoop started with start().
  /// Required in parallel mode; unused in serial mode, which runs the loop on
  /// the caller's thread via next().  Two requirements, both deadlocks if
  /// broken:
  ///
  ///  - It MUST NOT be the query executor.  The loop blocks waiting on the
  ///    sub-tasks it creates, whose drivers run on the query executor, so a
  ///    loop holding a query executor thread starves the very sub-tasks it
  ///    waits for -- and with one such loop per thread, nothing progresses.
  ///  - It MUST have a thread for every peer worker of the same fixed point
  ///    that runs concurrently.  Peers barrier with each other each iteration
  ///    (one's shuffle producer feeds the others' consumers), so a peer whose
  ///    loop is still queued blocks the peers that are running.
  ///
  /// Sizing it for the largest concurrent fixed point satisfies both.
  folly::Executor* orchestrationExecutor{nullptr};

  /// Set by an enclosing FixedPointLoop on the sub-tasks it creates; left empty
  /// by a coordinator, which owns none of what it carries.
  std::optional<NestedFixedPoint> nested;
};

/// Runs a plan containing a FixedPointNode as its single leaf, optionally with
/// trailing (non-fixed-point) nodes above it.  Instead of a Driver pipeline it
/// orchestrates the iteration's sub-tasks itself -- above the Operator layer,
/// at task scope (the loop runs once per task, not once per driver) -- which
/// keeps task orchestration out of operators and bounds the parallelism.
///
/// One FixedPointLoop is one worker: its owning task's destination is the
/// worker / shard index.  A shuffling fixed point runs as N peer tasks; the
/// coordinator creates them like any partitioned stage (via Task::create) and
/// wires their shuffle topology with remote splits.  Drive via Task::next().
///
/// It orchestrates ordinary Tasks rather than being another kind of Task: the
/// per-iteration sub-tasks it creates are plain driver-based tasks, and it is
/// composed into its owning Task, which delegates start(), next(),
/// addSplit() and noMoreSplits() to it.
///
/// What the owning Task's API means for a fixed point:
///
///  - start() / next() are the two ways in, and the execution mode picks which:
///    parallel runs the loop on the orchestration executor, serial runs it on
///    the caller's thread inside the first next().  Task checks the mode before
///    delegating, so the wrong one fails rather than half-runs.
///  - taskCompletionFuture() resolves when the loop has finished and the task
///    has reached a terminal state -- on the parallel path when the loop's
///    completion callback fires, on the serial path when next() has been
///    drained to nullptr.  error() then carries whatever the loop threw.
///  - addSplit() takes only the splits the loop itself routes: a peer worker's
///    remote split on the FixedPointNode, an initial plan's source splits, or
///    an enclosing loop's trailing Exchange splits.  Any other plan node id is
///    rejected.
///  - taskStats() covers none of the work -- see the TODO below.
///
/// TODO: Cancellation does not reach the loop.  Task::requestCancel(),
/// requestAbort() and requestPause() move the owning task to a terminal state
/// and resolve its completion future, but the loop keeps creating sub-tasks
/// until it converges or exhausts maxIterations.  Nothing crashes -- the loop
/// holds a reference to its owner for its whole run -- but the work continues
/// after the coordinator believes the query is dead, the state pools stay
/// unreclaimable while it does, and peers of a partially cancelled shuffling
/// fixed point lose the producer they barrier on each iteration.  Wiring this
/// up requires the loop to retain handles to the sub-tasks it is running;
/// today they are local to the phase that creates them.
///
/// TODO: taskStats() on the owning task reports no drivers and no operator
/// stats, because all the work happens in sub-tasks whose stats are discarded
/// when each phase ends.  Aggregating them needs a way to fold per-iteration
/// pipelines into one TaskStats, whose pipelineStats are indexed by a
/// pipeline id that is only unique within a single task.
class FixedPointLoop {
 public:
  /// Whether 'planFragment' contains a FixedPointNode as its leaf, whether or
  /// not trailing nodes sit above it.  Task::create consults this to decide
  /// whether to build a loop for the fragment instead of compiling it into a
  /// Driver pipeline.
  static bool claims(
      const core::PlanFragment& planFragment,
      const FixedPointOptions* fixedPointOptions);

  /// 'owner' is the fully constructed Task this loop runs the fragment for; it
  /// owns the loop and so outlives it.
  /// 'fixedPointOptions' may be null: a fixed point that needs no hooks -- a
  /// serial one, say -- leaves them empty.
  FixedPointLoop(exec::Task* owner, const FixedPointOptions* fixedPointOptions);

  exec::Task* owner() const {
    return owner_;
  }

  /// Captures the coordinator-assigned splits -- a peer worker's task id for
  /// the body shuffle, or the initial plan's source splits -- rather than
  /// feeding a Driver pipeline.
  void addSplit(const core::PlanNodeId& planNodeId, exec::Split&& split);

  void noMoreSplits(const core::PlanNodeId& planNodeId);

  /// Runs the whole fixed point on the first call, then streams this worker's
  /// shard one batch per call (nullptr when exhausted).
  RowVectorPtr next(ContinueFuture* future);

  /// Parallel execution: runs the loop asynchronously on the query executor and
  /// delivers this worker's shard to the task's consumer, completing via
  /// taskCompletionFuture() (as a coordinator running the fixed point as a
  /// parallel stage would).  Use next() for serial, in-process driving.
  /// 'maxDrivers' is used for the Phase-1 initial plans, which read real
  /// sources (TableScan/Exchange) that parallelize naturally; the iteration
  /// body, convergence, and trailing sub-tasks always run single-driver
  /// (multi-driver body support is a future extension -- see the design doc's
  /// open questions).
  /// 'onComplete' runs on the orchestration executor when the loop finishes:
  /// with a null exception on success, or the one it failed with.  The owning
  /// Task uses it to reach a terminal state, which keeps terminate() private to
  /// Task -- the loop decides when the work is done, not what that means for
  /// the task's state machine.
  void start(
      uint32_t maxDrivers,
      uint32_t concurrentSplitGroups,
      std::function<void(std::exception_ptr)> onComplete);

  /// Number of iterations the loop ran (converged early or hit maxIterations).
  int64_t iterations() const {
    return iterations_;
  }

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
  // The three phases run() composes, in order.  Private: next() and start() are
  // the only ways to drive a fixed point.
  //
  // Phase 1: creates the persistent state and seeds each declared entry from
  // its initial plan.
  void initialize();

  // Phase 2: runs one iteration's plans, writing the result back into the
  // output state entry.  'iteration' must strictly increase -- it is encoded
  // into the sub-task ids that form the cross-worker barrier.
  void runIteration(int32_t iteration);

  // Returns true if the convergence criterion signals convergence; false when
  // there is no criterion.  Evaluated over this worker's local state: either
  // the last iteration's delta was empty (stopWhenDeltaEmpty) or the
  // convergence sequence produced true.
  bool converged();

  // Phase 3: materializes the fixed point's output -- the output state entry's
  // final contents, or the trailing plan run over it -- into the output buffer
  // that next() streams.
  void finalize();

  // True when the last iteration wrote no rows into the output state entry.
  // Read only for a ConvergenceConfig with stopWhenDeltaEmpty.
  bool lastDeltaEmpty_{false};

  // Returns where the producer for 'workerTaskId' (this worker's or a peer's
  // task id), 'iteration', and 'planIndex' lives, via options_.producerLocation
  // (which the coordinator must set for a shuffling fixed point).  This worker
  // creates its own producers under '.taskId' and reaches peers' producers via
  // '.exchangeUri'; a peer is reachable because the location is a pure function
  // of (peer task id, iteration, plan), with no shared registry.  The ids MUST
  // encode the iteration, or the per-iteration cross-worker barrier desyncs and
  // deadlocks.
  ProducerLocation producerLocation(
      const WorkerAddress& worker,
      int32_t iteration,
      size_t planIndex) const;

  // The generic in-Velox driver: composes initialize() + the runIteration() /
  // converged() loop (bounded by maxIterations) + finalize().  Called by next()
  // and start().  An engine that wants to own the loop drives the public phase
  // methods instead.
  void run();

  // Runs the trailing (non-fixed-point) nodes -- the plan above the
  // FixedPointNode -- over the loop's output state, buffering their result. The
  // FixedPointNode leaf reads the output state entry (via its operator
  // translator); the sub-task is built without re-dispatching to a
  // FixedPointLoop.
  // Waits for the enclosing loop to finish assigning the trailing plan's
  // Exchange splits, then hands them to 'task', the sub-task running it.
  // A no-op unless an enclosing loop's shuffle feeds this fragment.
  void feedTrailingSplits(exec::Task& task);

  void runTrailingPlan();

  // Options for a sub-task this loop creates: the coordinator's hooks, plus a
  // fresh NestedFixedPoint naming this loop as the parent.  The nesting state
  // is rebuilt per sub-task rather than inherited, so a sub-task never picks up
  // an address or trailing id belonging to the loop above it.
  FixedPointOptions subTaskOptions();

  // Phase 1: runs each state's initial plan and stores this worker's shard.
  void initializeState();

  // Builds a join hash table once from its initial plan and stores it.
  void buildHashTable(const core::HashTableStateDeclaration& declaration);

  // Runs this worker's column of per-plan sub-tasks over the chain 'plans' for
  // 'iteration', wiring each consumer plan's Exchange to the previous-plan
  // producer of every peer worker named by the coordinator's splits.  Returns
  // the last plan's output copied into 'pool'.  Serves both chains a fixed
  // point runs: 'planIndexOffset' is added to the plan index passed to
  // producerLocation, so the body's producers (offset 0) and the convergence
  // sequence's (offset node_->plans().size()) get distinct ids in the same
  // iteration.
  std::vector<RowVectorPtr> runParallelChain(
      const std::vector<core::PlanNodePtr>& plans,
      size_t planIndexOffset,
      int32_t iteration,
      memory::MemoryPool* pool);

  // Appends a copy (in outputPool_) of the named Vector state entry to the
  // output buffer.
  void appendOutput(const std::string& stateEntry);

  // Runs 'plan' as a sub-task in this worker's execution mode and returns its
  // output batches, copied into 'pool'.  Serial mode drains via next() on the
  // calling thread (single-driver); parallel mode starts the sub-task on the
  // executor with 'maxDrivers' drivers and collects through a consumer.  Feeds
  // the plan's split source (TableScan files or an Exchange upstream) when it
  // has one.  Used for the initial and hash-build plans (with the parent
  // maxDrivers) and the convergence plan (single-driver).
  std::vector<RowVectorPtr> drainPlan(
      const core::PlanNodePtr& plan,
      memory::MemoryPool* pool,
      uint32_t maxDrivers);

  // Feeds 'plan's split source on 'task': an Exchange source reads the upstream
  // producer named by options_.upstreamExchangeUri (keyed by the Exchange node
  // id);
  // any other split source (e.g. a TableScan) reads the coordinator-assigned
  // source splits.  No-op when the plan has no split source (e.g. Values).
  void feedInitSplits(exec::Task& task, const core::PlanNodePtr& plan);

  // Label for an internal sub-task of this worker that no peer addresses (an
  // initial, convergence, serial-body, last-in-chain, or trailing sub-task).
  std::string subTaskId();

  // This worker's shuffle address, and its peers'.  A top-level worker uses
  // task ids: its own, and the peers' the coordinator named through remote
  // splits.  A nested one uses the addresses its enclosing loop handed down
  // (see FixedPointOptions::workerAddress).
  WorkerAddress workerAddress() const;
  std::vector<WorkerAddress> peerAddresses();

  // The address a sub-task running plan 'planIndex' of this loop's chain at
  // 'iteration' presents to its own peers.  Only the address deepens; the root
  // task id is carried through, since the sub-task runs in the same process.
  // Every peer derives the same suffix, which is what lets nested loops
  // shuffle.
  static WorkerAddress nestedAddress(
      const WorkerAddress& worker,
      int32_t iteration,
      size_t planIndex);

  // Coordinator-supplied hooks for sub-task id / upstream id generation (see
  // FixedPointOptions); empty hooks fall back to the fixed point's defaults.
  FixedPointOptions options_;

  // The nesting state an enclosing loop handed down, default-constructed when a
  // coordinator created this task directly.
  NestedFixedPoint nested_;

  // This worker's execution mode (from construction).  Every sub-task inherits
  // it: serial drains on the calling thread, parallel runs on the executor.  A
  // serial fixed point therefore cannot run a shuffling body (rejected in the
  // constructor).
  const exec::Task::ExecutionMode executionMode_;

  // The task that owns this loop; outlives it by construction.
  exec::Task* const owner_;

  // The parent task's maxDrivers (from start()), used for the Phase-1 initial
  // plans -- they read real sources (TableScan / Exchange / upstream) that
  // parallelize naturally.  Serial mode (next()) is always single-driver, so
  // this stays 1.  The iteration body, convergence, and trailing sub-tasks
  // always run single-driver for now (multi-driver body support is a future
  // extension -- see the design doc's open questions).
  uint32_t maxDrivers_{1};

  core::FixedPointNodePtr node_;

  // True when the plan has trailing nodes above the FixedPointNode (the
  // fragment root is not the FixedPointNode itself).
  bool hasTrailing_{false};

  int32_t workerIndex_{0};
  int32_t numWorkers_{1};

  // Task ids of the peer workers this worker reads from each body shuffle, and
  // the initial plan's source splits, captured from the coordinator's splits
  // via addSplit().  A peer's producer location is derived from its task id by
  // producerLocation(), so naming the peer here suffices.  Guarded by
  // splitsMutex_ while the coordinator is still adding: it may do so from
  // several threads, and the loop reads them once it starts.
  std::mutex splitsMutex_;
  std::vector<std::string> peerParentIds_;

  // Every other coordinator-assigned split, keyed by the plan node that reads
  // it: an initial plan's TableScan, or an Exchange in the trailing plan that
  // an enclosing loop feeds.  Keyed rather than pooled so two initial plans
  // that each scan a table receive only their own splits.
  folly::F14FastMap<core::PlanNodeId, std::vector<exec::Split>> nodeSplits_;

  // Nodes the coordinator (or an enclosing loop) has finished assigning splits
  // to.  runTrailingPlan waits on 'trailingSplitsCv_' for its Exchange to
  // appear here before running, since an enclosing loop assigns splits only
  // after starting this task.  Guarded by splitsMutex_, as is nodeSplits_.
  folly::F14FastSet<core::PlanNodeId> splitsComplete_;
  std::condition_variable trailingSplitsCv_;

  // Stable pool for persistent state, independent of sub-task pools.
  std::shared_ptr<memory::MemoryPool> statePool_;
  std::shared_ptr<PersistentState> state_;

  // Leaf pool holding the output batches (outputBuffer_).  The task's own
  // pool() is an aggregate pool and cannot allocate; this leaf child can, and
  // lives as long as the task so the streamed batches stay valid.
  std::shared_ptr<memory::MemoryPool> outputPool_;

  int64_t subTaskCounter_{0};
  int64_t iterations_{0};

  // Output rows, copied into outputPool_, emitted one batch per next() (or
  // pushed to the consumer by start()).
  std::vector<RowVectorPtr> outputBuffer_;
  size_t current_{0};

  // Where the loop is in the phase sequence.  next() reads it to decide whether
  // to run the loop, drain the buffered output, or refuse; run() checks each
  // transition, because getting the order wrong is silent -- initialize() twice
  // re-seeds the state mid-run, and runIteration() first reads state that was
  // never created.
  enum class Phase {
    // Nothing run yet.  next() and start() run the whole loop from here.
    kCreated,
    // initialize() done; runIteration(), converged() or finalize() may follow.
    kInitialized,
    // At least one runIteration() done.
    kIterating,
    // finalize() done: the output is buffered and next() may drain it.
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
