/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include <folly/futures/Future.h>

#include <compare>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::connector {
struct ConnectorSplit;
} // namespace facebook::velox::connector

namespace facebook::velox::exec {

/// Defined in FixedPointLoop.h.
struct FixedPointIterationExecutionStats;

/// Hooks an application passes to Task::create for a plan containing a
/// FixedPointNode.  The fixed point generates no task ids and reaches no
/// worker by itself: the application names the workers of a loop with a
/// FixedPointSplit and supplies the rest through these hooks.
struct FixedPointOptions {
  /// One level of nesting: the id of the nested FixedPointNode, and the
  /// iteration of the enclosing loop that runs it.  The iteration is 0 for a
  /// loop nested in an initial plan, which runs once.
  struct NestingStep {
    std::string planNodeId;
    int32_t iteration;

    auto operator<=>(const NestingStep& other) const = default;
  };

  /// Identifies one loop of one worker.  'worker' is the application's name
  /// for the worker, from the top-level loop's FixedPointSplit.  'nesting'
  /// locates the loop within the worker: empty for a top-level fixed point, one
  /// step per enclosing loop, outermost first.  Every worker runs a copy of the
  /// same plan, so the loops of one worker and of its peers that run in
  /// lockstep have equal nesting.
  struct LoopAddress {
    std::string worker;
    std::vector<NestingStep> nesting;
  };

  /// Returns the task id under which this worker creates the producer whose
  /// root is the PartitionedOutput 'planNodeId', in 'iteration' of the loop at
  /// 'self'.  Required for a fixed point whose plans chain by shuffle.  The id
  /// MUST encode the iteration: producers are re-created every iteration, and
  /// an id reused across iterations lets a consumer read a stale producer.
  std::function<std::string(
      const LoopAddress& self,
      int32_t iteration,
      const std::string& planNodeId)>
      producerTaskId;

  /// Returns the split through which a consumer reads its partition of the
  /// producer whose root is 'planNodeId', created by 'peer' in 'iteration'. The
  /// split is added to the consumer's Exchange, so the application returns
  /// whatever split its ExchangeSource expects.  Required with
  /// 'producerTaskId'.  The two hooks must agree: the split
  /// producerSplit(peer, i, id) returns reads the task that 'peer' created
  /// under producerTaskId(peer, i, id).
  std::function<std::shared_ptr<connector::ConnectorSplit>(
      const LoopAddress& peer,
      int32_t iteration,
      const std::string& planNodeId)>
      producerSplit;

  /// Returns the task id for a sub-task this worker creates that no peer reads
  /// -- an initial plan, a serial body, or the last plan of a chain -- from
  /// the worker's task id and a per-worker counter (for uniqueness).
  /// Required: the fixed point creates no task ids itself.
  std::function<std::string(const std::string& workerTaskId, int64_t counter)>
      subTaskId;

  /// A verdict plan (ConvergenceConfig::verdict) runs only on worker 0, the
  /// task whose destination is 0.  Worker 0 reports each iteration's verdict
  /// through 'publishVerdict', and every other worker of the loop receives it
  /// through 'awaitVerdict', so all of them stop on the same iteration.
  ///
  /// The application carries the verdict between workers; the fixed point has
  /// no transport of its own.  The transport the application's shuffles already
  /// use between workers can carry it: worker 0 already reaches every worker
  /// through it, and a worker's failure surfaces there as it does for the
  /// shuffles.  Both are required for a verdict plan in parallel mode.
  ///
  /// Both hooks receive worker 0's address, so an application keys verdicts on
  /// it directly.  An application can rely on:
  ///  - Worker 0 publishes each iteration's verdict once, in order.
  ///  - It publishes iteration t+1 only after every worker has received t, so
  ///    holding the latest verdict of each loop is enough.
  ///  - Both hooks run on the orchestration executor, which serves every loop,
  ///    so neither may block.
  std::function<
      void(const LoopAddress& workerZero, int32_t iteration, bool converged)>
      publishVerdict;

  /// Resolves to the verdict worker 0 published for 'iteration'.  Worker 0
  /// publishes nothing when it fails, so once any worker of the query fails,
  /// the application must, within bounded time, fail a pending or later call
  /// or end the caller's task, as aborting the query's tasks does.  A failed
  /// call fails the caller's loop.  The loop stops waiting once its own task
  /// ends; 'cancel' is then requested, so an implementation that polls for the
  /// verdict stops.
  std::function<folly::SemiFuture<bool>(
      const LoopAddress& workerZero,
      int32_t iteration,
      folly::CancellationToken cancel)>
      awaitVerdict;

  /// Runs the iteration loop of a fixed point in a parallel task.
  /// Required in parallel mode; unused in serial mode, which runs the loop on
  /// the caller's thread inside the owning task's first next().  A loop
  /// suspends while its sub-tasks run, so one thread serves any number of
  /// loops, nested or peers.  It MUST NOT be the query executor: on failure, a
  /// loop blocks until the sub-tasks it aborts finish, and their drivers run on
  /// the query executor.
  folly::Executor* orchestrationExecutor{nullptr};

  /// Drivers that the plans of one chain share in each iteration of a parallel
  /// loop: the body plans, or the convergence plans.  The plans of a chain run
  /// at the same time, and each gets at least one driver, so a chain with more
  /// plans than this uses more.  Must not exceed the parent task's maxDrivers.
  /// A loop nested in a plan gets that plan's share as its own
  /// iterationMaxDrivers.  With more than one driver these plans follow
  /// ordinary Velox multi-driver semantics, so an aggregation that must see
  /// every row needs a local exchange in front of it.  The last convergence
  /// plan runs on one driver of its own, outside this budget, because it must
  /// produce a single verdict row.  The default of 1 runs all of them
  /// single-driver.
  uint32_t iterationMaxDrivers{1};

  /// Experimental, like FixedPointIterationExecutionStats.  Optional.  Called
  /// with the statistics of each iteration a loop completes, after its
  /// convergence check.  They include every plan's TaskStats, which
  /// Task::fixedPointExecutionStats() keeps only for the first iterations and
  /// the latest one.  A TaskListener also receives each sub-task's TaskStats,
  /// but for every task in the process, and a sub-task named by 'subTaskId',
  /// such as the last plan of a chain, carries no iteration.  Only the loop
  /// joins a sub-task's TaskStats to its iteration, verdict and nesting, in a
  /// record that it trims and that, for a nested loop, ends with the internal
  /// sub-task running it.
  ///
  /// A loop nested in a plan inherits these options, so 'nesting' tells the
  /// loops of the task apart, as in LoopAddress: empty for the top-level loop.
  /// An application that shares one FixedPointOptions between tasks binds each
  /// task's identity into its own copy.  Runs on the loop's thread -- the
  /// orchestration executor, or the caller of next() in serial mode -- outside
  /// the statistics lock, so it may read Task::fixedPointExecutionStats(), and
  /// must not block.  Calls for one loop come in iteration order, but calls
  /// from different loops -- loops nested in the plans of one chain, or peer
  /// tasks whose copies share state -- may overlap, so it must be thread-safe.
  /// It must own, or share ownership of, whatever it captures: a nested loop
  /// runs in an internal sub-task that can outlive the task, and may call it
  /// after the task has ended, even after the task is destroyed.  Once the
  /// task ends, a parallel loop stops at its next wait, while a serial loop
  /// runs to completion and reports every iteration.  An iteration whose body
  /// or convergence check fails is not reported.  A std::exception it throws
  /// is logged and ignored, as for a TaskListener.
  std::function<void(
      const std::vector<NestingStep>& nesting,
      const FixedPointIterationExecutionStats& stats)>
      onIterationStats;
};

} // namespace facebook::velox::exec
