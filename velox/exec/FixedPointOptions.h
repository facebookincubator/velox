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

#include <functional>
#include <string>

namespace facebook::velox::exec {

/// Hooks a coordinator passes to Task::create for a plan containing a
/// FixedPointNode.  The fixed point generates no task ids or exchange URIs
/// itself -- the coordinator owns its addressing protocol and supplies them
/// through these hooks and the remote splits it adds.
struct FixedPointOptions {
  /// Where a producer sub-task lives: the bare task id it is created under (for
  /// Task::create and its output buffer) and the exchange URI a consumer
  /// reaches it through (a RemoteConnectorSplit -> ExchangeSource, selected by
  /// scheme). In-process the two are the same string; a distributed
  /// coordinator's URI also embeds the producer's endpoint, e.g.
  /// "thrift://{host:port}/{taskId}".
  struct ProducerLocation {
    /// Task id the producer is created under.
    std::string taskId;
    /// Exchange URI a consumer reaches the producer through.
    std::string exchangeUri;
  };

  /// Identifies one loop of one worker.  'rootTaskId' names the worker's
  /// top-level task, and so the process it runs in -- nesting never relocates
  /// execution, so a nested loop shares its root's endpoint.  'address' locates
  /// the loop within that task: equal to 'rootTaskId' for a top-level fixed
  /// point, and 'rootTaskId' followed by a suffix per nesting level below it.
  struct LoopAddress {
    std::string rootTaskId;
    std::string address;
  };

  /// Returns where a worker's producer for ('iteration', 'planIndex') lives:
  /// the task id the worker creates it under, and the exchange URI a peer
  /// consumer reaches it through.  Required for a shuffling fixed point (one
  /// whose plans chain by Exchange).  The ids MUST encode the iteration:
  /// per-iteration sub-tasks are re-created each round, so reusing the previous
  /// round's ids desynchronizes the cross-worker barrier and deadlocks.
  ///
  /// The loop calls this for its own worker and for each peer.  For its own,
  /// 'worker.rootTaskId' is its task id, and the loop creates the producer
  /// under the returned '.taskId'.  For a peer, it is the name the coordinator
  /// gave that peer in its peer split, and the loop reads the producer at the
  /// returned '.exchangeUri'.  The coordinator keeps the two consistent: the
  /// producer a worker creates is the one its peers reach.  An endpoint lookup
  /// keys on 'worker.rootTaskId', and the returned id must be unique in
  /// 'worker.address'.
  std::function<ProducerLocation(
      const LoopAddress& worker,
      int32_t iteration,
      size_t planIndex)>
      producerLocation;

  /// Returns the task id for an internal sub-task this worker creates -- an
  /// initial, convergence, serial-body, or one-plan parallel-body sub-task
  /// that no peer reads -- from the worker's task id and a per-worker counter
  /// (for uniqueness).
  /// Required: the fixed point creates no task ids itself.
  std::function<std::string(const std::string& workerTaskId, int64_t counter)>
      subTaskId;

  /// A verdict plan (ConvergenceConfig::verdict) runs only on worker 0, the
  /// task whose destination is 0.  Worker 0 reports each iteration's verdict
  /// through 'publishVerdict', and every other worker of the loop receives it
  /// through 'awaitVerdict', so all of them stop on the same iteration.  The
  /// coordinator carries the verdict between workers; the fixed point has no
  /// transport of its own.  Both are required for a verdict plan in parallel
  /// mode.
  ///
  /// A coordinator can rely on:
  ///  - Worker 0 publishes each iteration's verdict once, in order.
  ///  - It publishes iteration t+1 only after every worker has received t, so
  ///    holding the latest verdict of each loop is enough.
  ///  - Both hooks run on the orchestration executor, which serves every loop,
  ///    so neither may block.
  ///
  /// 'self' is the caller's address.  The coordinator finds worker 0 of the
  /// caller's loop: the peer whose task has destination 0.  For a nested loop,
  /// every peer's address is its root task's address extended by the same
  /// suffix, so worker 0's address is worker 0's root with the caller's
  /// suffix.
  std::function<
      void(const LoopAddress& self, int32_t iteration, bool converged)>
      publishVerdict;

  /// Resolves to the verdict worker 0 published for 'iteration'.  Worker 0
  /// publishes nothing when it fails, so once any worker of the query fails,
  /// the coordinator must, within bounded time, fail a pending or later call or
  /// end the caller's task, as aborting the query's tasks does.  A failed call
  /// fails the caller's loop.  The loop stops waiting once its own task ends;
  /// 'cancel' is then requested, so an implementation that polls for the
  /// verdict stops.
  std::function<folly::SemiFuture<bool>(
      const LoopAddress& self,
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
};

} // namespace facebook::velox::exec
