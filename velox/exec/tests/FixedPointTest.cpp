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
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <functional>
#include <mutex>
#include <thread>
#include <tuple>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/FixedPointPlanNodes.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/FixedPointLoop.h"
#include "velox/exec/FixedPointOperators.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Split.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
namespace {

using core::ConvergenceConfig;
using core::FixedPointNode;
using core::FixedPointNodePtr;
using core::HashTableStateDeclaration;
using core::PlanNodePtr;
using core::StateDeclarationPtr;
using core::VectorStateDeclaration;
using exec::test::PlanBuilder;

// Partitions rows by the first (BIGINT) column modulo the partition count.
// Unlike hash partitioning, this pins a key to a known partition, so a test can
// guarantee balanced, non-empty shards (key k -> worker k % numWorkers).
class ModuloPartitionFunction : public core::PartitionFunction {
 public:
  explicit ModuloPartitionFunction(int numPartitions)
      : numPartitions_{numPartitions} {}

  std::optional<uint32_t> partition(
      const RowVector& input,
      std::vector<uint32_t>& partitions) override {
    auto keys = input.childAt(0)->as<SimpleVector<int64_t>>();
    partitions.resize(input.size());
    for (vector_size_t i = 0; i < input.size(); ++i) {
      const auto key = keys->valueAt(i);
      partitions[i] = static_cast<uint32_t>(
          ((key % numPartitions_) + numPartitions_) % numPartitions_);
    }
    return std::nullopt;
  }

 private:
  const int numPartitions_;
};

class ModuloPartitionFunctionSpec : public core::PartitionFunctionSpec {
 public:
  std::unique_ptr<core::PartitionFunction> create(
      int numPartitions,
      bool /*localExchange*/) const override {
    return std::make_unique<ModuloPartitionFunction>(numPartitions);
  }

  std::string toString() const override {
    return "modulo";
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "ModuloPartitionFunctionSpec";
    return obj;
  }
};

class FixedPointTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    registerFixedPoint();
    // Route local:// task ids to the in-process exchange source so
    // PartitionedOutput -> Exchange works without networking.
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(
        exec::test::createLocalExchangeSource);
  }

  void TearDown() override {
    exec::Operator::unregisterAllOperators();
    HiveConnectorTestBase::TearDown();
  }

  // Number of top-level worker tasks for 'node': the partition count of the
  // first plan of whichever chain shuffles -- the body, or the convergence
  // sequence when the body stays local -- or 1 when neither does.
  static int32_t numWorkersOf(const FixedPointNodePtr& node) {
    return node->numWorkers();
  }

  // Runs the iteration loops, kept off the query executor the sub-tasks'
  // drivers run on.  Needs a thread per concurrently blocked loop, which with
  // nesting is peers x nesting depth, not just peers (see
  // FixedPointOptions::orchestrationExecutor).
  static folly::Executor* orchestrationExecutor() {
    static auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(16);
    return executor.get();
  }

  // FixedPointOptions wiring sub-task ids/URIs in the in-process "local://"
  // scheme.  A FixedPointLoop provides no defaults, so the coordinator
  // (here the test) supplies them: producerLocation gives each peer's
  // per-iteration shuffle producer (task id + exchange URI), subTaskId names
  // this worker's internal sub-tasks.  Callers add upstreamExchangeUri when an
  // initial plan reads via Exchange.
  static const FixedPointOptions& localOptions() {
    static const FixedPointOptions options = [] {
      FixedPointOptions opts;
      opts.orchestrationExecutor = orchestrationExecutor();
      opts.producerLocation = [](const std::string& rootWorkerTaskId,
                                 const std::string& workerAddress,
                                 int32_t iteration,
                                 size_t planIndex) {
        // In-process the exchange URI is the task id (LocalExchangeSource keys
        // by it), so the root task id goes unused here.  A distributed
        // coordinator would key its worker -> endpoint map on
        // 'rootWorkerTaskId' -- nesting never moves a loop off its root
        // worker's process -- and return "{scheme}://{endpoint}/{id}".
        (void)rootWorkerTaskId;
        auto id =
            fmt::format("{}.it{}.p{}", workerAddress, iteration, planIndex);
        return ProducerLocation{.taskId = id, .exchangeUri = id};
      };
      opts.subTaskId = [](const std::string& workerTaskId, int64_t counter) {
        return fmt::format("{}.sub{}", workerTaskId, counter);
      };
      return opts;
    }();
    return options;
  }

  // A single-worker, non-shuffling fixed point: seed val=0, increment each
  // iteration, converge at val=3.  For tests about the Task contract rather
  // than about a particular plan shape.
  FixedPointNodePtr countingNode() {
    auto schema = ROW({"key", "val"}, BIGINT());
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto seed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
    auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

    PlanBuilder bodyBuilder(idGenerator);
    bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});

    PlanBuilder convergenceBuilder(idGenerator);
    convergenceBuilder.stateSource("vals", schema)
        .project({"val >= 3 AS converged"});

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
    ConvergenceConfig convergence{
        .plans = {convergenceBuilder.planNode()}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergence),
        /*outputStateEntry=*/"vals");
  }

  // Creates the task 'node' runs as, named after 'label'.  'options' defaults
  // to the in-process coordinator hooks.
  std::shared_ptr<exec::Task> makeTask(
      const FixedPointNodePtr& node,
      exec::Task::ExecutionMode mode,
      const std::string& label,
      const FixedPointOptions* options = nullptr) {
    auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());
    return exec::Task::create(
        fmt::format("local://fixedpoint-{}-{}", label, queryCtx->queryId()),
        core::PlanFragment{node},
        /*destination=*/0,
        queryCtx,
        mode,
        exec::Consumer{},
        /*memoryArbitrationPriority=*/0,
        /*spillDiskOpts=*/std::nullopt,
        /*onError=*/nullptr,
        options != nullptr ? options : &localOptions());
  }

  using NodeFactory = std::function<FixedPointNodePtr(int32_t)>;

  // One worker's result: its shard rows and the number of iterations its loop
  // ran (read from the worker's FixedPointLoop).
  struct WorkerRun {
    std::vector<std::pair<int64_t, int64_t>> rows;
    int64_t iterations;
  };

  // Drives a fixed point as N peer top-level worker tasks, the way a
  // coordinator launches shuffle fragments.  'makeWorkerNode(d)' builds worker
  // d's node: the same plans for every worker, but its own pre-partitioned seed
  // (the operator never shards; the coordinator partitions the input).  N is
  // taken from worker 0's first plan.  The coordinator (this method) decides
  // the shuffle topology and distributes it via remote splits on the parent
  // tasks; here it chooses all-to-all (every worker reads every worker,
  // including itself).  If 'upstreamPlan' is set, the coordinator also launches
  // an upstream producer task (with the deterministic upstream id) so that a
  // worker's Exchange-based initial plan receives its shard from it via
  // shuffle.  If 'initSplitsFor' is set, the coordinator assigns worker d the
  // source splits it returns (e.g. a TableScan's file split) — its initial plan
  // reads its shard from them.  Workers run concurrently (each on its own
  // thread) and rendezvous through the shuffle.  Returns each worker's shard
  // rows and iteration count.
  std::vector<WorkerRun> runWorkers(
      const NodeFactory& makeWorkerNode,
      exec::Task::ExecutionMode mode,
      const core::PlanNodePtr& upstreamPlan = nullptr,
      const std::function<std::vector<std::pair<core::PlanNodeId, exec::Split>>(
          int32_t)>& initSplitsFor = nullptr) {
    auto node0 = makeWorkerNode(0);
    const int32_t numWorkers = numWorkersOf(node0);
    const auto fixedPointNodeId = node0->id();
    // The in-process exchange's data-size probe waits up to
    // request_data_sizes_max_wait_sec (default 10s) when a partition is empty
    // (e.g. a hash shuffle that leaves one worker's partition empty), which
    // otherwise dominates these tests' runtime.  Use a short wait.
    auto queryCtx = core::QueryCtx::create(
        cpuExecutor_.get(),
        core::QueryConfig{
            {{core::QueryConfig::kRequestDataSizesMaxWaitSec, "1"}}});

    // Upstream producer task feeding the workers' initial plans.  Its
    // partitioned output is read (by partition = worker destination) by each
    // worker's initial-plan Exchange.
    std::shared_ptr<exec::Task> upstream;
    if (upstreamPlan != nullptr) {
      // The coordinator names the upstream task (the operator cannot); workers
      // learn it via options.upstreamExchangeUri below.  "local://" routes it
      // to the in-process exchange source.
      upstream = exec::Task::create(
          fmt::format("local://fixedpoint-upstream-{}", queryCtx->queryId()),
          core::PlanFragment{upstreamPlan},
          /*destination=*/0,
          queryCtx,
          exec::Task::ExecutionMode::kParallel);
      upstream->start(/*maxDrivers=*/1);
    }

    // The coordinator owns sub-task addressing (the fixed point provides no
    // defaults): localOptions() supplies the producer ids and internal sub-task
    // ids in the in-process scheme; add the upstream's task id when an initial
    // plan reads via Exchange.
    auto options = localOptions();
    if (upstream != nullptr) {
      options.upstreamExchangeUri =
          [id = upstream->taskId()](const core::PlanNodeId&) { return id; };
    }

    // Each worker collects its shard rows into perWorker[d] -- a serial worker
    // from next() below, a parallel worker from this consumer (guarded by
    // consumerMutexes[d], as it runs on the worker's executor thread).
    std::vector<std::vector<std::pair<int64_t, int64_t>>> perWorker(numWorkers);
    std::vector<std::mutex> consumerMutexes(numWorkers);

    std::vector<std::shared_ptr<exec::Task>> workers;
    workers.reserve(numWorkers);
    for (int32_t d = 0; d < numWorkers; ++d) {
      auto node = (d == 0) ? node0 : makeWorkerNode(d);
      exec::Consumer consumer;
      if (mode == exec::Task::ExecutionMode::kParallel) {
        auto* rows = &perWorker[d];
        auto* mutex = &consumerMutexes[d];
        consumer = [rows, mutex](
                       RowVectorPtr batch,
                       bool /*drained*/,
                       ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            std::lock_guard<std::mutex> l(*mutex);
            auto first = batch->childAt(0)->as<SimpleVector<int64_t>>();
            auto second = batch->childAt(1)->as<SimpleVector<int64_t>>();
            for (vector_size_t i = 0; i < batch->size(); ++i) {
              rows->emplace_back(first->valueAt(i), second->valueAt(i));
            }
          }
          return exec::BlockingReason::kNotBlocked;
        };
      }
      workers.push_back(
          exec::Task::create(
              fmt::format(
                  "local://fixedpoint-worker-{}-{}", queryCtx->queryId(), d),
              core::PlanFragment{node},
              /*destination=*/d,
              queryCtx,
              mode,
              consumer,
              /*memoryArbitrationPriority=*/0,
              /*spillDiskOpts=*/std::nullopt,
              /*onError=*/nullptr,
              &options));
    }

    // The coordinator wires the body-shuffle topology by adding one remote
    // split per peer worker's task to each parent (all-to-all here); the fixed
    // point derives each peer's per-iteration producer location from its task
    // id via the producerLocation hook.  'initSplitsFor(d)' splits are worker
    // d's initial-plan source splits (e.g. a TableScan file split), forwarded
    // verbatim; an Exchange initial plan instead uses
    // options.upstreamExchangeUri.
    if (node0->requiresSplits()) {
      for (int32_t d = 0; d < numWorkers; ++d) {
        for (int32_t e = 0; e < numWorkers; ++e) {
          workers[d]->addSplit(
              fixedPointNodeId,
              exec::Split(
                  std::make_shared<exec::RemoteConnectorSplit>(
                      workers[e]->taskId())));
        }
        if (initSplitsFor != nullptr) {
          // Addressed to the node that reads them, not to the fixed point:
          // that is how a fixed point with several scanned initial plans keeps
          // each one's splits apart.
          for (auto& [sourceId, split] : initSplitsFor(d)) {
            workers[d]->addSplit(sourceId, std::move(split));
            workers[d]->noMoreSplits(sourceId);
          }
        }
        workers[d]->noMoreSplits(fixedPointNodeId);
      }
    }

    // Drive each worker in 'mode'.  Serial: next() on its own thread, so peers
    // rendezvous through the shuffle (next() runs the whole loop
    // synchronously). Parallel: start() on the executor (run() runs there and
    // blocks on its sub-tasks), with the shard delivered to the consumer above.
    std::vector<std::exception_ptr> errors(numWorkers);
    if (mode == exec::Task::ExecutionMode::kSerial) {
      std::vector<std::thread> threads;
      threads.reserve(numWorkers);
      for (int32_t d = 0; d < numWorkers; ++d) {
        threads.emplace_back([&, d]() {
          try {
            while (auto batch = workers[d]->next()) {
              auto first = batch->childAt(0)->as<SimpleVector<int64_t>>();
              auto second = batch->childAt(1)->as<SimpleVector<int64_t>>();
              for (vector_size_t i = 0; i < batch->size(); ++i) {
                perWorker[d].emplace_back(
                    first->valueAt(i), second->valueAt(i));
              }
            }
          } catch (...) {
            errors[d] = std::current_exception();
          }
        });
      }
      for (auto& thread : threads) {
        thread.join();
      }
    } else {
      for (auto& worker : workers) {
        worker->start(/*maxDrivers=*/1);
      }
      for (int32_t d = 0; d < numWorkers; ++d) {
        auto future = workers[d]->taskCompletionFuture();
        std::move(future).wait();
        errors[d] = workers[d]->error();
      }
    }
    for (auto& error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }

    // All workers have finished Phase 1, so the upstream's partitions are
    // drained; let it complete and surface any error.
    if (upstream != nullptr) {
      auto future = upstream->taskCompletionFuture();
      std::move(future).wait();
      if (auto error = upstream->error()) {
        std::rethrow_exception(error);
      }
    }

    std::vector<WorkerRun> results;
    results.reserve(numWorkers);
    for (int32_t d = 0; d < numWorkers; ++d) {
      // Each worker task runs a FixedPointLoop (Task::create composes one
      // onto a FixedPointNode plan); read its iteration count directly.
      auto* fixedPoint = workers[d]->testingFixedPoint();
      VELOX_CHECK_NOT_NULL(fixedPoint, "Worker task has no FixedPointLoop");
      results.push_back({std::move(perWorker[d]), fixedPoint->iterations()});
    }
    return results;
  }

  // Unions every worker's shard rows into sorted (col0, col1) BIGINT pairs.
  static std::vector<std::pair<int64_t, int64_t>> unionRows(
      const std::vector<WorkerRun>& runs) {
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto& worker : runs) {
      pairs.insert(pairs.end(), worker.rows.begin(), worker.rows.end());
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
  }

  // Runs the workers in 'mode' and returns the union of their shard rows as
  // sorted (col0, col1) BIGINT pairs.
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const NodeFactory& makeWorkerNode,
      exec::Task::ExecutionMode mode) {
    return unionRows(runWorkers(makeWorkerNode, mode));
  }

  // Same, but with an upstream producer task feeding the workers' initial plans
  // through shuffle.
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const NodeFactory& makeWorkerNode,
      const core::PlanNodePtr& upstreamPlan,
      exec::Task::ExecutionMode mode) {
    return unionRows(runWorkers(makeWorkerNode, mode, upstreamPlan));
  }

  // Convenience overload: every worker runs the same node (used for N=1 tests,
  // and shuffles where replicating the seed is acceptable).
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const FixedPointNodePtr& node,
      exec::Task::ExecutionMode mode) {
    return runViaTask(
        [&](int32_t /*worker*/) -> FixedPointNodePtr { return node; }, mode);
  }

  // Runs a non-shuffling fixed point under BOTH parent modes and asserts each
  // yields 'expected'.  Sub-tasks inherit the parent's mode, so this exercises
  // both the serial (next()) and parallel (start()) sub-task drives.
  void expectBothModes(
      const FixedPointNodePtr& node,
      const std::vector<std::pair<int64_t, int64_t>>& expected) {
    EXPECT_EQ(runViaTask(node, exec::Task::ExecutionMode::kSerial), expected)
        << "serial parent";
    EXPECT_EQ(runViaTask(node, exec::Task::ExecutionMode::kParallel), expected)
        << "parallel parent";
  }

  // Like runViaTask, but tags each row with its worker's iteration count as a
  // third column (col0, col1, iterations), so a test can verify the loop ran
  // the expected number of iterations.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> runViaTaskWithIterations(
      const NodeFactory& makeWorkerNode,
      exec::Task::ExecutionMode mode) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> rows;
    for (const auto& worker : runWorkers(makeWorkerNode, mode)) {
      for (const auto& [key, value] : worker.rows) {
        rows.emplace_back(key, value, worker.iterations);
      }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  }

  // Drives a plan whose root is NOT a FixedPointNode but which contains one as
  // its leaf, with trailing nodes above it (e.g. a Project over a fixed point),
  // in 'mode'.  Task::create yields a FixedPointLoop that runs the loop,
  // then runs the trailing plan over the result.  Returns the rows as sorted
  // (col0, col1) BIGINT pairs.
  std::vector<std::pair<int64_t, int64_t>> runViaTrailing(
      const core::PlanNodePtr& plan,
      exec::Task::ExecutionMode mode) {
    auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());
    std::vector<std::pair<int64_t, int64_t>> rows;
    std::mutex mutex;
    auto collect = [&](const RowVectorPtr& batch) {
      auto first = batch->childAt(0)->as<SimpleVector<int64_t>>();
      auto second = batch->childAt(1)->as<SimpleVector<int64_t>>();
      for (vector_size_t i = 0; i < batch->size(); ++i) {
        rows.emplace_back(first->valueAt(i), second->valueAt(i));
      }
    };
    const auto taskId =
        fmt::format("local://fixedpoint-trailing-{}", queryCtx->queryId());
    if (mode == exec::Task::ExecutionMode::kSerial) {
      auto task = exec::Task::create(
          taskId,
          core::PlanFragment{plan},
          /*destination=*/0,
          queryCtx,
          exec::Task::ExecutionMode::kSerial,
          exec::Consumer{},
          /*memoryArbitrationPriority=*/0,
          /*spillDiskOpts=*/std::nullopt,
          /*onError=*/nullptr,
          &localOptions());
      while (auto batch = task->next()) {
        collect(batch);
      }
    } else {
      exec::Consumer consumer = [&](RowVectorPtr batch,
                                    bool /*drained*/,
                                    ContinueFuture* /*future*/) {
        if (batch != nullptr && batch->size() > 0) {
          std::lock_guard<std::mutex> l(mutex);
          collect(batch);
        }
        return exec::BlockingReason::kNotBlocked;
      };
      auto task = exec::Task::create(
          taskId,
          core::PlanFragment{plan},
          /*destination=*/0,
          queryCtx,
          exec::Task::ExecutionMode::kParallel,
          consumer,
          /*memoryArbitrationPriority=*/0,
          /*spillDiskOpts=*/std::nullopt,
          /*onError=*/nullptr,
          &localOptions());
      task->start(/*maxDrivers=*/1);
      auto future = task->taskCompletionFuture();
      std::move(future).wait();
      if (auto error = task->error()) {
        std::rethrow_exception(error);
      }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  }

  // Runs the workers' sub-tasks.  A shuffling fixed point keeps up to
  // numWorkers * numPlans sub-task drivers live at once (producers stay up
  // while peers' consumers drain them), so this must comfortably exceed that.
  // The run() loops that block on these sub-tasks run on a separate executor
  // (see orchestrationExecutor()).
  std::shared_ptr<folly::CPUThreadPoolExecutor> cpuExecutor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(16)};
};

// Transitive reachability over an acyclic graph, modeled as a recursive CTE:
//
//   WITH RECURSIVE reach(id, depth) AS (
//     SELECT 1, 0
//     UNION ALL
//     SELECT e.dst, r.depth + 1 FROM reach r JOIN edges e ON r.id = e.src)
//
// The single body plan runs serially each iteration (no exchange): it reads the
// frontier (the rows the append-mode result entry accumulated last iteration),
// joins it with the static edges, and appends the next frontier back to result.
// The FixedPointNode is driven through the Task interface and emits result --
// the union of every iteration's frontier.
TEST_F(FixedPointTest, recursiveCte) {
  auto schema = ROW({"id", "depth"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Base case seed: {(id=1, depth=0)}.
  auto seed = makeRowVector(
      {"id", "depth"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      {"src", "dst"},
      {makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
       makeFlatVector<int64_t>({2, 3, 5, 4, 6})});
  auto buildSide = PlanBuilder(idGenerator).values({edges}).planNode();

  // A single body plan: read the frontier (an in-loop StateSource over the
  // append-mode result entry returns the rows appended last iteration), probe
  // the edges, and append the next depth back to result.  The frontier read and
  // the accumulation are the same entry, and the iteration is one local
  // pipeline (no shuffle), so it is a single plan -- no state-chained second
  // plan.
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("result", schema)
      .hashJoin({"id"}, {"src"}, buildSide, "", {"dst", "depth"})
      .project({"dst AS id", "depth + 1 AS depth"});
  auto body = bodyBuilder.planNode();

  // Converged when the last iteration appended no rows (the frontier is empty).
  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("result", schema)
      .singleAggregation({}, {"count(1)"})
      .project({"a0 = 0 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  // result accumulates every iteration's frontier (an append-mode entry) and
  // also provides the frontier each iteration reads; seeded with the base case.
  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "result", schema, initialPlan, /*append=*/true)};
  std::vector<core::PlanNodePtr> plans{body};
  ConvergenceConfig convergence{
      .plans = {convergencePlan}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"result");

  // The exact 6-row union both verifies reachability and proves the iteration
  // terminated at the empty frontier (a wrong convergence would diverge or
  // hit maxIterations).
  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  std::sort(expected.begin(), expected.end());
  expectBothModes(node, expected);
}

// The same recursive CTE, terminated by whenDeltaEmpty instead of a
// count(1) == 0 convergence plan.  Producing the identical 6-row result proves
// the delta row count is an equivalent verdict, reached without running a
// convergence sub-task per iteration.
TEST_F(FixedPointTest, whenDeltaEmptyRecursiveCte) {
  auto schema = ROW({"id", "depth"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      {"id", "depth"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      {"src", "dst"},
      {makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
       makeFlatVector<int64_t>({2, 3, 5, 4, 6})});
  auto buildSide = PlanBuilder(idGenerator).values({edges}).planNode();

  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("result", schema)
      .hashJoin({"id"}, {"src"}, buildSide, "", {"dst", "depth"})
      .project({"dst AS id", "depth + 1 AS depth"});
  auto body = bodyBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "result", schema, initialPlan, /*append=*/true)};
  std::vector<core::PlanNodePtr> plans{body};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      ConvergenceConfig::whenDeltaEmpty(100),
      /*outputStateEntry=*/"result");

  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  std::sort(expected.begin(), expected.end());
  expectBothModes(node, expected);
}

// A body that never runs dry hits maxIterations, and whenDeltaEmpty sets
// errorWhenMaxIterationReached, so the loop fails rather than returning a
// non-terminal result.
TEST_F(FixedPointTest, whenDeltaEmptyFailsWithoutEmptyDelta) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({1})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Every iteration rewrites the single row, so the delta is never empty.
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("result", schema)
      .project({"id + 1 AS id", "val AS val"});

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("result", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      ConvergenceConfig::whenDeltaEmpty(5),
      /*outputStateEntry=*/"result");

  VELOX_ASSERT_THROW(
      runViaTask(node, exec::Task::ExecutionMode::kSerial),
      "did not converge within");
}

// Same recursive CTE, but the edges are a HashTable state entry built once and
// probed by a StateHashJoin every iteration (hash-table reuse, ask #7) instead
// of a HashJoin that rebuilds the edges table each iteration.  Produces the
// same result, demonstrating the reuse path is correct.
TEST_F(FixedPointTest, recursiveCteHashTableReuse) {
  auto schema = ROW({"id", "depth"}, BIGINT());
  auto edgesSchema = ROW({"src", "dst"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      {"id", "depth"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({0})});
  auto seedPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      {"src", "dst"},
      {makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
       makeFlatVector<int64_t>({2, 3, 5, 4, 6})});
  auto edgesPlan = PlanBuilder(idGenerator).values({edges}).planNode();

  // A single body plan: probe the prebuilt edges table with the frontier id (an
  // in-loop StateSource over the append-mode result entry returns the rows
  // appended last iteration), emit the next depth, and append it back to
  // result. StateHashJoin output is the probe columns (id, depth) plus the
  // table's dependent column (dst).
  auto joinOutput = ROW({"id", "depth", "dst"}, BIGINT());
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("result", schema);
  bodyBuilder.stateHashJoin("edges_ht", {"id"}, joinOutput)
      .project({"dst AS id", "depth + 1 AS depth"});
  auto body = bodyBuilder.planNode();

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("result", schema)
      .singleAggregation({}, {"count(1)"})
      .project({"a0 = 0 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<HashTableStateDeclaration>(
          "edges_ht", edgesSchema, std::vector<std::string>{"src"}, edgesPlan),
      std::make_shared<VectorStateDeclaration>(
          "result", schema, seedPlan, /*append=*/true)};
  std::vector<core::PlanNodePtr> plans{body};
  ConvergenceConfig convergence{
      .plans = {convergencePlan}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"result");

  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  std::sort(expected.begin(), expected.end());
  expectBothModes(node, expected);
}

// Strict mode inheritance: a serial fixed point runs every sub-task on the
// calling thread, so it cannot drive a shuffling body (PartitionedOutput /
// Exchange require parallel mode).  The constructor rejects that combination.
TEST_F(FixedPointTest, serialModeRejectsShuffle) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({1})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder producerBuilder(idGenerator);
  auto producer = producerBuilder.stateSource("frontier", schema)
                      .partitionedOutput({}, 1)
                      .planNode();
  PlanBuilder consumerBuilder(idGenerator);
  consumerBuilder.exchange(schema, "Presto");
  auto consumer = consumerBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "frontier", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{producer, consumer};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      ConvergenceConfig{
          .plans = {},
          .maxIterations = 3,
          .errorWhenMaxIterationReached = false},
      /*outputStateEntry=*/"frontier");

  // Creating this shuffling body as a serial FixedPointLoop is rejected.
  auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());
  VELOX_ASSERT_THROW(
      exec::Task::create(
          "local://fixedpoint-serial-shuffle",
          core::PlanFragment{node},
          /*destination=*/0,
          queryCtx,
          exec::Task::ExecutionMode::kSerial,
          exec::Consumer{},
          /*memoryArbitrationPriority=*/0,
          /*spillDiskOpts=*/std::nullopt,
          /*onError=*/nullptr,
          &localOptions()),
      "A serial fixed point cannot run a shuffling body");
}

// Two-plan iteration that shuffles between sub-plans: plan 0 reads the frontier
// and gathers it through a PartitionedOutput; plan 1 receives it via an
// Exchange (a separate parallel task), halves each value, and writes the
// frontier back.  Converges when the values reach zero.  Driven through the
// Task interface; emits the final frontier.
TEST_F(FixedPointTest, shuffleBetweenSubplans) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({8, 5})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Plan 0: gather the frontier to a single destination (producer task).
  PlanBuilder producerBuilder(idGenerator);
  auto producer = producerBuilder.stateSource("frontier", schema)
                      .partitionedOutput({}, 1)
                      .planNode();

  // Plan 1: receive via exchange, halve values, write back (consumer task).
  PlanBuilder consumerBuilder(idGenerator);
  consumerBuilder.exchange(schema, "Presto").project({"id", "val / 2 AS val"});
  auto consumer = consumerBuilder.planNode();

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("frontier", schema)
      .singleAggregation({}, {"sum(val)"})
      .project({"a0 = 0 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "frontier", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{producer, consumer};
  ConvergenceConfig convergence{
      .plans = {convergencePlan}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"frontier");

  // 8 -> 4 -> 2 -> 1 -> 0 halvings converge to all-zero values.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 0}, {1, 0}};
  EXPECT_EQ(runViaTask(node, exec::Task::ExecutionMode::kParallel), expected);
}

// Two shuffles per iteration (a three-plan chain): the body gathers, halves,
// gathers again, and halves again before writing the frontier back, so every
// iteration crosses two PartitionedOutput -> Exchange hops.  Exercises the
// multi-hop parallel chain (the middle plan is both an Exchange consumer and a
// PartitionedOutput producer) recurring across iterations.
TEST_F(FixedPointTest, shuffleDuringIteration) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({8, 12})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Plan 0: gather the frontier (producer).
  PlanBuilder firstBuilder(idGenerator);
  auto first = firstBuilder.stateSource("frontier", schema)
                   .partitionedOutput({}, 1)
                   .planNode();

  // Plan 1: receive, halve, gather again (consumer + producer).
  PlanBuilder middleBuilder(idGenerator);
  auto middle = middleBuilder.exchange(schema, "Presto")
                    .project({"id", "val / 2 AS val"})
                    .partitionedOutput({}, 1)
                    .planNode();

  // Plan 2: receive, halve again, write the frontier back (consumer).
  PlanBuilder lastBuilder(idGenerator);
  lastBuilder.exchange(schema, "Presto").project({"id", "val / 2 AS val"});
  auto last = lastBuilder.planNode();

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("frontier", schema)
      .singleAggregation({}, {"sum(val)"})
      .project({"a0 = 0 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "frontier", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{first, middle, last};
  ConvergenceConfig convergence{
      .plans = {convergencePlan}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"frontier");

  // Each iteration divides by four (two halvings):
  //   (8, 12) -> (2, 3) -> (0, 0).
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 0}, {1, 0}};
  EXPECT_EQ(runViaTask(node, exec::Task::ExecutionMode::kParallel), expected);
}

// A real by-key shuffle across two peer top-level tasks.  The fixed point runs
// as two workers (destinations 0 and 1); the coordinator (runViaTask) wires
// them all-to-all via remote splits and they shuffle directly with each other.
//
// The coordinator pre-partitions the input across the workers (modeling a
// partitioned source): worker 0 gets {(0,1),(1,10)}, worker 1 gets
// {(0,2),(1,20)} -- each key's rows start split across both workers.  Every
// iteration each worker partitions its shard by `key` into two partitions and
// reads its partition from BOTH workers, so a key's rows co-locate on one
// worker before that key is summed.  This is a genuine redistribution -- the
// per-key sum is correct only if the shuffle co-located each key -- not a
// single-task scatter.  After the first iteration each key is owned by a single
// worker, so the per-key sums are stable; the loop is bounded by maxIterations
// (no convergence plan) and the union of the workers' shards gives the final
// sums.
TEST_F(FixedPointTest, shuffleByKey) {
  auto schema = ROW({"key", "val"}, BIGINT());

  // key=0 sums to 3 ({1, 2}); key=1 sums to 30 ({10, 20}).  Each worker's shard
  // holds some rows of each key, so neither can aggregate a key alone.
  std::vector<RowVectorPtr> seeds{
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({1, 10})}),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({2, 20})})};

  // Builds worker d's node: the same plans for every worker, but its own
  // pre-partitioned shard as the seed.  Plan 0 partitions this worker's shard
  // by key into two partitions (the partition count, 2, is the worker count);
  // plan 1 reads this worker's partition from the wired peers, sums each key,
  // and writes this worker's shard back.
  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto initialPlan =
        PlanBuilder(idGenerator).values({seeds[worker]}).planNode();

    PlanBuilder producerBuilder(idGenerator);
    auto producer = producerBuilder.stateSource("vals", schema)
                        .partitionedOutput({"key"}, 2)
                        .planNode();

    PlanBuilder consumerBuilder(idGenerator);
    consumerBuilder.exchange(schema, "Presto")
        .singleAggregation({"key"}, {"sum(val)"})
        .project({"key", "a0 AS val"});
    auto consumer = consumerBuilder.planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    // No convergence plan: the loop runs exactly maxIterations times.
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        ConvergenceConfig{
            .plans = {},
            .maxIterations = 3,
            .errorWhenMaxIterationReached = false},
        /*outputStateEntry=*/"vals");
  };

  // After every iteration the by-key sums are stable: key 0 -> 3, key 1 -> 30.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_EQ(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// Convergence with two workers (N=2).  A convergence plan reads only the local
// shard, so the verdict must be globally consistent or lockstep breaks; the
// framework adds no cross-worker reduction, so the plan must synchronize it.
// Here synchronization is achieved by construction: a modulo(key, 2) shuffle
// pins key k to worker k % 2, so both shards stay non-empty, and both keys
// start at the same value and halve identically -- so every worker's local sum
// crosses the threshold on the SAME iteration and the verdicts always agree. If
// the verdicts ever disagreed (e.g. an empty shard converging early), the
// shuffle would deadlock; reaching the result instead proves the workers stayed
// in lockstep.
TEST_F(FixedPointTest, convergenceTwoWorkers) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  // Pre-partitioned, balanced seeds: worker 0 owns key 0, worker 1 owns key 1
  // (matching key % 2).  Both start at 8.
  std::vector<RowVectorPtr> seeds{
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({8})}),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({8})})};

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto initialPlan =
        PlanBuilder(idGenerator).values({seeds[worker]}).planNode();

    // Plan 0 (produce): partition by key % 2 (stable: each key stays on its
    // worker).  Plan 1 (consume): sum each key and halve it.
    PlanBuilder producerBuilder(idGenerator);
    auto producer =
        producerBuilder.stateSource("vals", schema)
            .partitionedOutput(
                {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec)
            .planNode();

    PlanBuilder consumerBuilder(idGenerator);
    consumerBuilder.exchange(schema, "Presto")
        .singleAggregation({"key"}, {"sum(val)"})
        .project({"key", "a0 / 2 AS val"});
    auto consumer = consumerBuilder.planNode();

    // Converge when this worker's shard has decayed to <= 1.  Globally
    // consistent here because both keys decay in lockstep.  The threshold is 1
    // (not 0) so the converged result (value 1) differs from the
    // max-iterations result (value 0), proving the loop actually stopped on
    // convergence.
    PlanBuilder convergenceBuilder(idGenerator);
    convergenceBuilder.stateSource("vals", schema)
        .singleAggregation({}, {"sum(val)"})
        .project({"a0 <= 1 AS converged"});
    auto convergencePlan = convergenceBuilder.planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    ConvergenceConfig convergence{
        .plans = {convergencePlan}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergence),
        /*outputStateEntry=*/"vals");
  };

  // 8 -> 4 -> 2 -> 1: converges at value 1 after exactly three iterations.  The
  // third column is each worker's actual loop-iteration count; both must be 3.
  // Had convergence not fired, the loop would run to maxIterations (100) and
  // decay to 0 -- so value 1 and iteration count 3 (not 0 and 100) confirm the
  // loop stopped on the synchronized convergence verdict.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 1, 3}, {1, 1, 3}};
  EXPECT_EQ(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// An all-reduce built from existing nodes, with REPLICATED (not sharded) state.
// Every worker holds the same one-row entry.  Each iteration a worker
// replicates its local contribution once per destination and shuffles it, so
// every worker receives every peer's contribution and aggregates them into the
// identical new value -- reduce + broadcast, i.e. an all-reduce, from
// PartitionedOutput + Exchange + Aggregation alone, with no dedicated reduction
// node.
//
// Because the resulting state is identical on every worker, convergence is a
// purely local test on it and is globally consistent by construction; no
// cross-worker reduction is needed for the verdict either.
TEST_F(FixedPointTest, allReduceOverReplicatedState) {
  constexpr int32_t kNumWorkers{2};
  auto schema = ROW({"key", "val"}, BIGINT());
  auto shuffleType = ROW({"destination", "val"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  auto makeWorkerNode = [&](int32_t /*worker*/) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    // Every worker starts from the same replicated row.
    auto seed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({1})});
    // One row per destination; the cross join below turns a worker's single
    // contribution into one copy addressed to each peer.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});
    auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

    PlanBuilder producerBuilder(idGenerator);
    auto producer =
        producerBuilder.stateSource("total", schema)
            .nestedLoopJoin(
                PlanBuilder(idGenerator).values({destinations}).planNode(),
                {"destination", "val"})
            .partitionedOutput(
                {"destination"},
                kNumWorkers,
                /*replicateNullsAndAny=*/false,
                moduloSpec)
            .planNode();

    // Sums every peer's contribution: the same total on every worker.
    PlanBuilder consumerBuilder(idGenerator);
    auto consumer = consumerBuilder.exchange(shuffleType, "Presto")
                        .singleAggregation({}, {"sum(val)"})
                        .project({"CAST(0 AS BIGINT) AS key", "a0 AS val"})
                        .planNode();

    PlanBuilder convergenceBuilder(idGenerator);
    auto convergencePlan = convergenceBuilder.stateSource("total", schema)
                               .project({"val >= 8 AS converged"})
                               .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("total", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    ConvergenceConfig convergence{
        .plans = {convergencePlan}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergence),
        /*outputStateEntry=*/"total");
  };

  // Each iteration sums both workers' equal values, doubling the replicated
  // total: 1 -> 2 -> 4 -> 8, converging after exactly three iterations.  Both
  // workers must report the same value and the same iteration count -- the
  // property that keeps a replicated fixed point in lockstep.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 8, 3}, {0, 8, 3}};
  EXPECT_EQ(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// A convergence criterion that shuffles.  The body here needs no exchange, so
// each worker's shard stays private and the two workers hold different values;
// the criterion is a global sum, which neither can evaluate alone.  This is the
// shape PageRank's post-update RMSE needs: a statistic of the state the
// iteration just committed, reduced across workers.  It cannot be folded into
// the body, because the last body plan's output *is* the committed state, so a
// reducing stage placed after it would replace the state with the scalar.  The
// convergence sequence therefore chains like the body -- replicate-by-expansion
// producer, then an Exchange that sums every peer's partial -- and every worker
// evaluates the same predicate on the same iteration, which is what keeps them
// in lockstep.
TEST_F(FixedPointTest, convergenceShuffle) {
  constexpr int32_t kNumWorkers{2};
  auto schema = ROW({"key", "val"}, BIGINT());
  auto shuffleType = ROW({"destination", "partial"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});

    // Worker 0 starts at 0 and worker 1 at 10, so no local value ever equals
    // the global sum and a local-only criterion could not agree across workers.
    auto seed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({worker}),
         makeFlatVector<int64_t>({worker * 10})});
    auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

    PlanBuilder bodyBuilder(idGenerator);
    auto body = bodyBuilder.stateSource("vals", schema)
                    .project({"key", "val + 1 AS val"})
                    .planNode();

    // C0: this worker's partial, replicated to every peer by expansion.
    PlanBuilder partialBuilder(idGenerator);
    auto partial =
        partialBuilder.stateSource("vals", schema)
            .singleAggregation({}, {"sum(val)"})
            .project({"a0 AS partial"})
            .nestedLoopJoin(
                PlanBuilder(idGenerator).values({destinations}).planNode(),
                {"destination", "partial"})
            .partitionedOutput(
                {"destination"},
                kNumWorkers,
                /*replicateNullsAndAny=*/false,
                moduloSpec)
            .planNode();

    // C1: every worker sums every peer's partial and emits the same verdict.
    PlanBuilder verdictBuilder(idGenerator);
    auto verdict = verdictBuilder.exchange(shuffleType, "Presto")
                       .singleAggregation({}, {"sum(partial)"})
                       .project({"a0 >= 16 AS converged"})
                       .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{body};
    ConvergenceConfig convergence{
        .plans = {partial, verdict}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergence),
        /*outputStateEntry=*/"vals");
  };

  // Values move 0,10 -> 1,11 -> 2,12 -> 3,13; the global sum reaches 16 on the
  // third iteration.  Both workers stop there, on values (3 and 13) that differ
  // from each other and from the total.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 3, 3}, {1, 13, 3}};
  EXPECT_EQ(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// Both chains shuffle: the body re-partitions each iteration and the
// convergence criterion runs its own reduction on top of the state that body
// just committed.  The two chains address their per-iteration producers out of
// the same coordinator hook, so this is what proves their plan indices do not
// collide -- a collision would have one chain's Exchange find the other's
// producer and hang.
TEST_F(FixedPointTest, shuffleInBodyAndConvergence) {
  constexpr int32_t kNumWorkers{2};
  auto schema = ROW({"key", "val"}, BIGINT());
  auto shuffleType = ROW({"destination", "partial"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  // Worker 0 owns key 0, worker 1 owns key 1; both start at 8.
  std::vector<RowVectorPtr> seeds{
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({8})}),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({8})})};

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});
    auto initialPlan =
        PlanBuilder(idGenerator).values({seeds[worker]}).planNode();

    // Body: re-partition by key % 2 (each key stays on its worker), then halve.
    auto producer = PlanBuilder(idGenerator)
                        .stateSource("vals", schema)
                        .partitionedOutput(
                            {"key"},
                            kNumWorkers,
                            /*replicateNullsAndAny=*/false,
                            moduloSpec)
                        .planNode();
    auto consumer = PlanBuilder(idGenerator)
                        .exchange(schema, "Presto")
                        .singleAggregation({"key"}, {"sum(val)"})
                        .project({"key", "a0 / 2 AS val"})
                        .planNode();

    // Convergence: reduce the committed state across workers and stop when the
    // global sum reaches 2 (1 per worker).  No worker can see that locally.
    auto partial =
        PlanBuilder(idGenerator)
            .stateSource("vals", schema)
            .singleAggregation({}, {"sum(val)"})
            .project({"a0 AS partial"})
            .nestedLoopJoin(
                PlanBuilder(idGenerator).values({destinations}).planNode(),
                {"destination", "partial"})
            .partitionedOutput(
                {"destination"},
                kNumWorkers,
                /*replicateNullsAndAny=*/false,
                moduloSpec)
            .planNode();
    auto verdict = PlanBuilder(idGenerator)
                       .exchange(shuffleType, "Presto")
                       .singleAggregation({}, {"sum(partial)"})
                       .project({"a0 <= 2 AS converged"})
                       .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    ConvergenceConfig convergence{
        .plans = {partial, verdict}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergence),
        /*outputStateEntry=*/"vals");
  };

  // 8 -> 4 -> 2 -> 1 per worker; the global sum reaches 2 after three
  // iterations, and both workers stop on the same one.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 1, 3}, {1, 1, 3}};
  EXPECT_EQ(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// A complete KMeans, expressed with ordinary relational nodes -- no imperative
// operator and no dedicated reduction node.  Six points form two obvious
// clusters; from deliberately off-centre seeds the loop converges to their
// means.  One iteration:
//
//   assign  -- cross join points x centroids, keep each point's nearest slot;
//   update  -- sum and count per slot, then RIGHT join the old centroids so a
//              slot that drew no points keeps its previous position;
//   measure -- record each slot's squared movement, which drives convergence.
//
// The trailing Project scales the coordinates to integers so the exact fixed
// point is asserted rather than a float tolerance.
TEST_F(FixedPointTest, kMeans) {
  auto pointsType =
      ROW({{"point_id", BIGINT()}, {"x", DOUBLE()}, {"y", DOUBLE()}});
  auto centroidsType = ROW(
      {{"slot", BIGINT()},
       {"cx", DOUBLE()},
       {"cy", DOUBLE()},
       {"movement", DOUBLE()}});
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Two tight clusters around (1.33, 1.33) and (8.33, 8.33).
  auto pointRows = makeRowVector(
      {"point_id", "x", "y"},
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5}),
       makeFlatVector<double>({1.0, 2.0, 1.0, 8.0, 9.0, 8.0}),
       makeFlatVector<double>({1.0, 1.0, 2.0, 8.0, 8.0, 9.0})});
  // Seeds sit off-centre so the centroids have to move.  Slot 2 is far from
  // every point and so draws none, exercising the empty-cluster path: it must
  // keep its seeded position instead of becoming null.  The seed movement is
  // never read: convergence runs only after an iteration has overwritten it.
  auto centroidRows = makeRowVector(
      {"slot", "cx", "cy", "movement"},
      {makeFlatVector<int64_t>({0, 1, 2}),
       makeFlatVector<double>({2.0, 8.0, 100.0}),
       makeFlatVector<double>({1.0, 8.0, 100.0}),
       makeFlatVector<double>({0.0, 0.0, 0.0})});

  auto pointsInitial = PlanBuilder(idGenerator).values({pointRows}).planNode();
  auto centroidsInitial =
      PlanBuilder(idGenerator).values({centroidRows}).planNode();

  // Old centroids, renamed so the RIGHT join below can carry both sides.
  auto oldCentroids =
      PlanBuilder(idGenerator)
          .stateSource("centroids", centroidsType)
          .project({"slot AS old_slot", "cx AS old_cx", "cy AS old_cy"})
          .planNode();

  auto body =
      PlanBuilder(idGenerator)
          .stateSource("points", pointsType)
          .nestedLoopJoin(
              PlanBuilder(idGenerator)
                  .stateSource("centroids", centroidsType)
                  .planNode(),
              {"point_id", "x", "y", "slot", "cx", "cy"})
          .project(
              {"point_id",
               "x",
               "y",
               "slot",
               "(x - cx) * (x - cx) + (y - cy) * (y - cy) AS d2"})
          // Nearest slot per point; x and y are constant within a point_id.
          .singleAggregation(
              {"point_id"},
              {"min_by(slot, d2) AS slot", "min(x) AS x", "min(y) AS y"})
          .singleAggregation(
              {"slot"}, {"sum(x) AS sx", "sum(y) AS sy", "count(1) AS cnt"})
          // RIGHT join keeps every declared slot: an empty cluster has a null
          // count, and the coalesce below then retains its old centroid.
          .hashJoin(
              {"slot"},
              {"old_slot"},
              oldCentroids,
              "",
              {"old_slot", "old_cx", "old_cy", "sx", "sy", "cnt"},
              core::JoinType::kRight)
          // Velox has no divide(DOUBLE, BIGINT); cast the count once.
          .project(
              {"old_slot",
               "old_cx",
               "old_cy",
               "sx",
               "sy",
               "CAST(cnt AS DOUBLE) AS n"})
          .project(
              {"old_slot AS slot",
               "coalesce(sx / n, old_cx) AS cx",
               "coalesce(sy / n, old_cy) AS cy",
               "coalesce("
               "(sx / n - old_cx) * (sx / n - old_cx) + "
               "(sy / n - old_cy) * (sy / n - old_cy), 0.0) AS movement"})
          .planNode();

  // Converged once no centroid moved.
  auto convergencePlan = PlanBuilder(idGenerator)
                             .stateSource("centroids", centroidsType)
                             .singleAggregation({}, {"max(movement)"})
                             .project({"a0 <= 1e-9 AS converged"})
                             .planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>(
          "points", pointsType, pointsInitial),
      std::make_shared<VectorStateDeclaration>(
          "centroids", centroidsType, centroidsInitial)};
  auto kMeansNode = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::vector<core::PlanNodePtr>{body},
      ConvergenceConfig{.plans = {convergencePlan}, .maxIterations = 20},
      /*outputStateEntry=*/"centroids");

  auto plan = PlanBuilder(kMeansNode, idGenerator)
                  .project(
                      {"CAST(round(cx * 100.0) AS BIGINT) AS cx100",
                       "CAST(round(cy * 100.0) AS BIGINT) AS cy100"})
                  .planNode();

  // Means of the two clusters: (4/3, 4/3) and (25/3, 25/3), scaled by 100; the
  // point-less slot keeps its seed.  errorWhenMaxIterationReached defaults to
  // true, so a loop that never converged would throw rather than reach these.
  std::vector<std::pair<int64_t, int64_t>> expectedCentroids{
      {133, 133}, {833, 833}, {10000, 10000}};
  EXPECT_EQ(
      runViaTrailing(plan, exec::Task::ExecutionMode::kSerial),
      expectedCentroids)
      << "serial parent";
  EXPECT_EQ(
      runViaTrailing(plan, exec::Task::ExecutionMode::kParallel),
      expectedCentroids)
      << "parallel parent";
}

// The same KMeans across two workers: the points are sharded, the centroids are
// replicated, and each iteration reduces the per-slot statistics across workers
// so that every worker installs the identical next centroid set.  The reduction
// is an all-reduce assembled from existing nodes -- each worker replicates its
// local statistics once per destination and shuffles them, so plan 1 sees every
// worker's contribution and aggregates them globally (see
// allReduceOverReplicatedState for the mechanism on its own).
//
// Coordinates are scaled integers rather than doubles.  Integer addition is
// associative, so the reduction is exact regardless of the order rows arrive
// from the exchange; every worker therefore computes bit-identical centroids
// and reaches the same convergence verdict on the same iteration.  With doubles
// the reduction order would decide the last bits, and two workers could
// disagree at the epsilon boundary -- which would break lockstep.
TEST_F(FixedPointTest, kMeansTwoWorkers) {
  constexpr int32_t kNumWorkers{2};
  auto pointsType = ROW({"point_id", "x", "y"}, BIGINT());
  // Coordinates first, so the harness (which reads the first two BIGINT
  // columns) collects the centroid positions.
  auto centroidsType = ROW({"cx", "cy", "slot", "movement"}, BIGINT());
  auto shuffleType = ROW({"destination", "slot", "sx", "sy", "cnt"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  // The coordinator pre-shards the points: worker 0 owns the cluster near
  // (100, 100), worker 1 the cluster near (800, 800).  Coordinates are scaled
  // by 100, so the means below land on exact integers.
  std::vector<RowVectorPtr> pointShards{
      makeRowVector(
          {"point_id", "x", "y"},
          {makeFlatVector<int64_t>({0, 1, 2}),
           makeFlatVector<int64_t>({100, 200, 100}),
           makeFlatVector<int64_t>({100, 100, 200})}),
      makeRowVector(
          {"point_id", "x", "y"},
          {makeFlatVector<int64_t>({3, 4, 5}),
           makeFlatVector<int64_t>({800, 900, 800}),
           makeFlatVector<int64_t>({800, 800, 900})})};

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    // Replicated seeds, identical on every worker; slot 2 again draws no
    // points.
    auto centroidRows = makeRowVector(
        {"cx", "cy", "slot", "movement"},
        {makeFlatVector<int64_t>({200, 800, 10000}),
         makeFlatVector<int64_t>({100, 800, 10000}),
         makeFlatVector<int64_t>({0, 1, 2}),
         makeFlatVector<int64_t>({0, 0, 0})});
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});
    auto pointsInitial =
        PlanBuilder(idGenerator).values({pointShards[worker]}).planNode();
    auto centroidsInitial =
        PlanBuilder(idGenerator).values({centroidRows}).planNode();

    // Plan 0: assign this worker's points, reduce to per-slot local statistics,
    // then replicate them once per destination so every peer receives them.
    auto producer =
        PlanBuilder(idGenerator)
            .stateSource("points", pointsType)
            .nestedLoopJoin(
                PlanBuilder(idGenerator)
                    .stateSource("centroids", centroidsType)
                    .planNode(),
                {"point_id", "x", "y", "slot", "cx", "cy"})
            .project(
                {"point_id",
                 "x",
                 "y",
                 "slot",
                 "(x - cx) * (x - cx) + (y - cy) * (y - cy) AS d2"})
            .singleAggregation(
                {"point_id"},
                {"min_by(slot, d2) AS slot", "min(x) AS x", "min(y) AS y"})
            .singleAggregation(
                {"slot"}, {"sum(x) AS sx", "sum(y) AS sy", "count(1) AS cnt"})
            .nestedLoopJoin(
                PlanBuilder(idGenerator).values({destinations}).planNode(),
                {"destination", "slot", "sx", "sy", "cnt"})
            .partitionedOutput(
                {"destination"},
                kNumWorkers,
                /*replicateNullsAndAny=*/false,
                moduloSpec)
            .planNode();

    auto oldCentroids =
        PlanBuilder(idGenerator)
            .stateSource("centroids", centroidsType)
            .project({"slot AS old_slot", "cx AS old_cx", "cy AS old_cy"})
            .planNode();

    // Plan 1: sum every worker's statistics -- the same totals on each worker
    // -- then install the new centroids, keeping the old one where a slot drew
    // no points anywhere.
    auto consumer =
        PlanBuilder(idGenerator)
            .exchange(shuffleType, "Presto")
            .singleAggregation(
                {"slot"},
                {"sum(sx) AS gsx", "sum(sy) AS gsy", "sum(cnt) AS gcnt"})
            .hashJoin(
                {"slot"},
                {"old_slot"},
                oldCentroids,
                "",
                {"old_slot", "old_cx", "old_cy", "gsx", "gsy", "gcnt"},
                core::JoinType::kRight)
            .project(
                {"coalesce(gsx / gcnt, old_cx) AS cx",
                 "coalesce(gsy / gcnt, old_cy) AS cy",
                 "old_slot AS slot",
                 "coalesce("
                 "(gsx / gcnt - old_cx) * (gsx / gcnt - old_cx) + "
                 "(gsy / gcnt - old_cy) * (gsy / gcnt - old_cy), 0) AS "
                 "movement"})
            .planNode();

    // Exact integer test: no centroid moved.
    auto convergencePlan = PlanBuilder(idGenerator)
                               .stateSource("centroids", centroidsType)
                               .singleAggregation({}, {"max(movement)"})
                               .project({"a0 = 0 AS converged"})
                               .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>(
            "points", pointsType, pointsInitial),
        std::make_shared<VectorStateDeclaration>(
            "centroids", centroidsType, centroidsInitial)};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::vector<core::PlanNodePtr>{producer, consumer},
        ConvergenceConfig{.plans = {convergencePlan}, .maxIterations = 20},
        /*outputStateEntry=*/"centroids");
  };

  // Neither worker can reach these alone: worker 0 holds no point near
  // (833, 833) and worker 1 none near (133, 133), so each centroid proves the
  // cross-worker reduction ran.  Both workers emit the same three centroids
  // after the same two iterations -- the replicated state stayed in lockstep.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {133, 133, 2},
      {133, 133, 2},
      {833, 833, 2},
      {833, 833, 2},
      {10000, 10000, 2},
      {10000, 10000, 2}};
  EXPECT_EQ(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// Initial plan that receives its shard through a shuffle from an UPSTREAM task,
// then iterates while shuffling among peers.  The coordinator launches an
// upstream producer task that emits the full seed and partitions it by key;
// each worker's initial plan is an Exchange that reads its partition (Phase 1
// shuffle, fed by a kUpstreamSplitPrefix-tagged split the coordinator adds).
// The workers then iterate, shuffling by key with each other (Phase 2).  This
// exercises the full path: receive data via shuffle from an upstream task, then
// iterate while shuffling with each other.
TEST_F(FixedPointTest, initialPlanReceivesShuffle) {
  auto schema = ROW({"key", "val"}, BIGINT());

  // Upstream producer: the full seed, partitioned by key across the workers.
  auto upstreamIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0, 0, 1, 1}),
       makeFlatVector<int64_t>({1, 2, 10, 20})});
  auto upstreamPlan = PlanBuilder(upstreamIdGenerator)
                          .values({seed})
                          .partitionedOutput({"key"}, 2)
                          .planNode();

  // Every worker runs the same node; its shard arrives from the upstream, not a
  // local seed.
  auto makeWorkerNode = [&](int32_t /*worker*/) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Initial plan: receive this worker's shard via shuffle from the upstream.
    auto initialPlan =
        PlanBuilder(idGenerator).exchange(schema, "Presto").planNode();

    // Body: shuffle by key among peers and sum each key (as in shuffleByKey).
    PlanBuilder producerBuilder(idGenerator);
    auto producer = producerBuilder.stateSource("vals", schema)
                        .partitionedOutput({"key"}, 2)
                        .planNode();
    PlanBuilder consumerBuilder(idGenerator);
    consumerBuilder.exchange(schema, "Presto")
        .singleAggregation({"key"}, {"sum(val)"})
        .project({"key", "a0 AS val"});
    auto consumer = consumerBuilder.planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    // No convergence plan: the loop runs a few iterations of peer shuffle.
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        ConvergenceConfig{
            .plans = {},
            .maxIterations = 3,
            .errorWhenMaxIterationReached = false},
        /*outputStateEntry=*/"vals");
  };

  // The upstream delivers the seed; the by-key peer shuffle sums each key:
  // key 0 -> 1+2 = 3, key 1 -> 10+20 = 30, regardless of the upstream's split.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_EQ(
      runViaTask(
          makeWorkerNode, upstreamPlan, exec::Task::ExecutionMode::kParallel),
      expected);
}

// Initial plan that reads its shard from a TABLE SCAN — the canonical
// partitioned source.  The coordinator pre-partitions the input across files
// (one per worker) and assigns each worker its file split; each worker's
// initial plan is a TableScan that reads its file (Phase 1).  The workers then
// iterate, shuffling by key with each other (Phase 2).  This exercises the
// realistic path: a partitioned source feeds the fixed point, then it iterates
// while shuffling with peers.
TEST_F(FixedPointTest, initialPlanTableScan) {
  auto schema = ROW({"key", "val"}, BIGINT());

  // Coordinator pre-partitions the input across two files (its split
  // assignment): worker 0 scans file 0, worker 1 scans file 1.  Each key's rows
  // are split across the files, so only the by-key peer shuffle co-locates
  // them.
  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({1, 10})}));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({2, 20})}));

  // Every worker runs the same node; its shard comes from the table, not a
  // local seed.
  core::PlanNodeId scanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Initial plan: read this worker's shard from its assigned file split.
    auto initialPlan = PlanBuilder(idGenerator).tableScan(schema).planNode();
    scanId = initialPlan->id();

    // Body: shuffle by key among peers and sum each key.
    PlanBuilder producerBuilder(idGenerator);
    auto producer = producerBuilder.stateSource("vals", schema)
                        .partitionedOutput({"key"}, 2)
                        .planNode();
    PlanBuilder consumerBuilder(idGenerator);
    consumerBuilder.exchange(schema, "Presto")
        .singleAggregation({"key"}, {"sum(val)"})
        .project({"key", "a0 AS val"});
    auto consumer = consumerBuilder.planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        ConvergenceConfig{
            .plans = {},
            .maxIterations = 3,
            .errorWhenMaxIterationReached = false},
        /*outputStateEntry=*/"vals");
  };

  // Each worker scans its own file via TableScan; the by-key peer shuffle sums
  // each key: key 0 -> 1+2 = 3, key 1 -> 10+20 = 30.
  auto initSplitsFor = [&](int32_t worker)
      -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
    return {
        {scanId,
         exec::Split(makeHiveConnectorSplit(filePaths[worker]->getPath()))}};
  };

  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_EQ(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          /*upstreamPlan=*/nullptr,
          initSplitsFor)),
      expected);
}

// Two initial plans that each scan a table.  Splits are held under the id of
// the node that reads them, so each scan takes only its own file; pooling them
// on the fixed point would feed both files to both scans and seed each state
// with the other's rows.
TEST_F(FixedPointTest, twoScannedInitialPlans) {
  auto valsSchema = ROW({"key", "val"}, BIGINT());
  auto stepsSchema = ROW({"skey", "delta"}, BIGINT());

  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({1})}));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          {"skey", "delta"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({10})}));

  core::PlanNodeId valsScanId;
  core::PlanNodeId stepsScanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto valsInitial =
        PlanBuilder(idGenerator).tableScan(valsSchema).planNode();
    valsScanId = valsInitial->id();
    auto stepsInitial =
        PlanBuilder(idGenerator).tableScan(stepsSchema).planNode();
    stepsScanId = stepsInitial->id();

    // Each iteration adds the per-key step, which only exists if 'steps' was
    // seeded from its own file.
    auto body = PlanBuilder(idGenerator)
                    .stateSource("vals", valsSchema)
                    .hashJoin(
                        {"key"},
                        {"skey"},
                        PlanBuilder(idGenerator)
                            .stateSource("steps", stepsSchema)
                            .planNode(),
                        "",
                        {"key", "val", "delta"})
                    .project({"key", "val + delta AS val"})
                    .planNode();
    auto convergence = PlanBuilder(idGenerator)
                           .stateSource("vals", valsSchema)
                           .project({"val >= 21 AS converged"})
                           .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>(
            "vals", valsSchema, valsInitial),
        std::make_shared<VectorStateDeclaration>(
            "steps", stepsSchema, stepsInitial)};
    std::vector<core::PlanNodePtr> plans{body};
    ConvergenceConfig convergenceConfig{
        .plans = {convergence}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        std::move(convergenceConfig),
        /*outputStateEntry=*/"vals");
  };

  auto initSplitsFor = [&](int32_t /*worker*/)
      -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
    return {
        {valsScanId,
         exec::Split(makeHiveConnectorSplit(filePaths[0]->getPath()))},
        {stepsScanId,
         exec::Split(makeHiveConnectorSplit(filePaths[1]->getPath()))}};
  };

  // val walks 1 -> 11 -> 21 in steps of delta=10, so both states were seeded
  // from the right file.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 21}};
  EXPECT_EQ(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          /*upstreamPlan=*/nullptr,
          initSplitsFor)),
      expected);
}

// One initial state fed by Exchange, another by TableScan.  Splits are held
// per reading node, so the Exchange-fed plan must ignore the scan's splits:
// a task-wide "this plan takes no connector splits" check would see the scan's
// file and reject a legitimate plan.
TEST_F(FixedPointTest, initialPlanMixesExchangeAndScan) {
  auto valsSchema = ROW({"key", "val"}, BIGINT());
  auto stepsSchema = ROW({"skey", "delta"}, BIGINT());
  // Modulo, not hash: key K must land on worker K, the one whose step file
  // holds skey=K, or the join finds no match.
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  // Upstream producer feeding the Exchange-backed state.
  auto upstreamIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto upstreamPlan =
      PlanBuilder(upstreamIdGenerator)
          .values({makeRowVector(
              {"key", "val"},
              {makeFlatVector<int64_t>({0, 0, 1, 1}),
               makeFlatVector<int64_t>({1, 2, 10, 20})})})
          .partitionedOutput(
              {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec)
          .planNode();

  // Each worker scans its own per-key step file.
  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          {"skey", "delta"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({100})}));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          {"skey", "delta"},
          {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({200})}));

  core::PlanNodeId stepsScanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto valsInitial =
        PlanBuilder(idGenerator).exchange(valsSchema, "Presto").planNode();
    auto stepsInitial =
        PlanBuilder(idGenerator).tableScan(stepsSchema).planNode();
    stepsScanId = stepsInitial->id();

    auto producer =
        PlanBuilder(idGenerator)
            .stateSource("vals", valsSchema)
            .partitionedOutput(
                {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec)
            .planNode();
    auto consumer = PlanBuilder(idGenerator)
                        .exchange(valsSchema, "Presto")
                        .singleAggregation({"key"}, {"sum(val)"})
                        .project({"key", "a0 AS val"})
                        .hashJoin(
                            {"key"},
                            {"skey"},
                            PlanBuilder(idGenerator)
                                .stateSource("steps", stepsSchema)
                                .planNode(),
                            "",
                            {"key", "val", "delta"})
                        .project({"key", "val + delta AS val"})
                        .planNode();

    std::vector<StateDeclarationPtr> stateDeclarations{
        std::make_shared<VectorStateDeclaration>(
            "vals", valsSchema, valsInitial),
        std::make_shared<VectorStateDeclaration>(
            "steps", stepsSchema, stepsInitial)};
    std::vector<core::PlanNodePtr> plans{producer, consumer};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(stateDeclarations),
        std::move(plans),
        ConvergenceConfig{
            .plans = {},
            .maxIterations = 1,
            .errorWhenMaxIterationReached = false},
        /*outputStateEntry=*/"vals");
  };

  auto initSplitsFor = [&](int32_t worker)
      -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
    return {
        {stepsScanId,
         exec::Split(makeHiveConnectorSplit(filePaths[worker]->getPath()))}};
  };

  // key 0: 1+2 = 3, plus its step 100 -> 103.  key 1: 10+20 = 30, plus 200.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 103}, {1, 230}};
  EXPECT_EQ(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          upstreamPlan,
          initSplitsFor)),
      expected);
}

// Drives a fixed point through the PARALLEL Task::start() path instead of
// serial next(): the FixedPointLoop runs its loop asynchronously on the
// executor and delivers the final state to a consumer, completing via
// taskCompletionFuture(). A single-worker counting fixed point keeps the focus
// on the start()/consumer/ completion machinery: the state holds one row
// (key=0, val) and the body increments val each iteration until it reaches 3.
TEST_F(FixedPointTest, countingLoopBothModes) {
  // 0 -> 1 -> 2 -> 3: converges at val 3.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}};
  expectBothModes(countingNode(), expected);
}

// The contract a coordinator depends on: the owning task reaches a terminal
// state when the loop is done, in both modes.  Nothing else moves a fixed point
// off kRunning -- it has no drivers -- so without this taskCompletionFuture()
// would never resolve.
TEST_F(FixedPointTest, taskReachesTerminalStateWhenLoopFinishes) {
  {
    auto task = makeTask(
        countingNode(), exec::Task::ExecutionMode::kSerial, "terminal-serial");
    while (task->next() != nullptr) {
    }
    EXPECT_EQ(task->state(), exec::TaskState::kFinished);
    EXPECT_TRUE(task->taskCompletionFuture().isReady());
    EXPECT_EQ(task->error(), nullptr);
  }
  {
    auto task = makeTask(
        countingNode(),
        exec::Task::ExecutionMode::kParallel,
        "terminal-parallel");
    task->start(/*maxDrivers=*/1);
    task->taskCompletionFuture().wait();
    EXPECT_EQ(task->state(), exec::TaskState::kFinished);
    EXPECT_EQ(task->error(), nullptr);
  }
}

// A parallel fixed point needs an orchestration executor separate from the
// query executor.  Rejecting that throws on the caller's thread, before the
// loop schedules anything -- so the task has to record the failure itself or it
// stays kRunning for ever with no error to report.
TEST_F(FixedPointTest, startWithoutOrchestrationExecutorFails) {
  FixedPointOptions options = localOptions();
  options.orchestrationExecutor = nullptr;
  auto task = makeTask(
      countingNode(),
      exec::Task::ExecutionMode::kParallel,
      "no-orchestrator",
      &options);

  VELOX_ASSERT_THROW(task->start(/*maxDrivers=*/1), "orchestrationExecutor");
  EXPECT_EQ(task->state(), exec::TaskState::kFailed);
  EXPECT_NE(task->error(), nullptr);
}

// A failing loop reports through the task, not only to the caller.
TEST_F(FixedPointTest, loopFailureReachesTheTask) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});
  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"val >= 100 AS converged"});
  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  // Never reaches 100 within 2 iterations, so the loop fails the query.
  ConvergenceConfig convergence{
      .plans = {convergenceBuilder.planNode()}, .maxIterations = 2};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  auto task =
      makeTask(node, exec::Task::ExecutionMode::kSerial, "loop-failure");
  VELOX_ASSERT_THROW(task->next(), "did not converge within");
  EXPECT_EQ(task->state(), exec::TaskState::kFailed);
  EXPECT_NE(task->error(), nullptr);
  // The state is half-written, so a retried next() must not re-seed it and
  // rebuild hash tables on top of a half-finished run.
  VELOX_ASSERT_THROW(task->next(), "cannot resume");
}

// A split addressed to a node that reads none is a coordinator bug.  The driver
// path rejects it; so must this one, or the split is filed where nothing reads
// it and the query quietly returns nothing.
TEST_F(FixedPointTest, addSplitRejectsPlanNodeThatTakesNoSplits) {
  auto node = countingNode();
  auto task = makeTask(node, exec::Task::ExecutionMode::kSerial, "bad-split");
  // The body's Project takes no splits.
  VELOX_ASSERT_THROW(
      task->addSplit(
          "no-such-node",
          exec::Split(std::make_shared<exec::RemoteConnectorSplit>("peer"))),
      "Plan node takes no splits");
}

// A NULL verdict is a malformed criterion, not a "no".  Read as "not
// converged" it would burn the whole iteration budget and then surface as a
// non-convergence failure, pointing at the wrong thing.
TEST_F(FixedPointTest, convergenceVerdictMustNotBeNull) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"CAST(NULL AS BOOLEAN) AS converged"});

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  ConvergenceConfig convergence{
      .plans = {convergenceBuilder.planNode()}, .maxIterations = 10};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  VELOX_ASSERT_THROW(
      runViaTask(node, exec::Task::ExecutionMode::kSerial),
      "produced a NULL verdict");
}

// A plan with trailing (non-fixed-point) nodes above the FixedPointNode: a
// Project over a counting fixed point.  Task::create yields a
// FixedPointLoop even though the FixedPointNode is the fragment's leaf (not
// its root); the task runs the loop, then runs the trailing Project over the
// loop's output state.
TEST_F(FixedPointTest, trailingNodes) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});
  auto body = bodyBuilder.planNode();

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"val >= 3 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{body};
  ConvergenceConfig convergence{
      .plans = {convergencePlan}, .maxIterations = 100};
  auto fixedPoint = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  // Trailing Project sitting on top of the fixed point: scale val by 10.
  auto plan =
      PlanBuilder(idGenerator)
          .addNode(
              [fixedPoint](
                  const core::PlanNodeId& /*id*/,
                  const core::PlanNodePtr& /*input*/) -> core::PlanNodePtr {
                return fixedPoint;
              })
          .project({"key", "val * 10 AS val"})
          .planNode();

  // Counting fixed point converges at val 3; the trailing Project scales to 30.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 30}};
  EXPECT_EQ(runViaTrailing(plan, exec::Task::ExecutionMode::kSerial), expected);
  EXPECT_EQ(
      runViaTrailing(plan, exec::Task::ExecutionMode::kParallel), expected);
}

// A fixed point nested inside another one's body: the shape Hierarchical
// Affinity clustering needs, where each outer level runs an inner clustering
// loop to convergence.  A body plan's primary leaf must be a StateSource, so
// the nested FixedPointNode sits on a non-primary branch -- here the build side
// of the outer body's join.
TEST_F(FixedPointTest, nestedFixedPoint) {
  auto outerSchema = ROW({"key", "val"}, BIGINT());
  auto innerSchema = ROW({"ikey", "ival"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Inner fixed point: counts ival up to 3, emitting {(0, 3)}.
  auto innerSeed = makeRowVector(
      {"ikey", "ival"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto innerInitialPlan =
      PlanBuilder(idGenerator).values({innerSeed}).planNode();

  PlanBuilder innerBodyBuilder(idGenerator);
  innerBodyBuilder.stateSource("inner", innerSchema)
      .project({"ikey", "ival + 1 AS ival"});

  PlanBuilder innerConvergenceBuilder(idGenerator);
  innerConvergenceBuilder.stateSource("inner", innerSchema)
      .project({"ival >= 3 AS converged"});

  std::vector<StateDeclarationPtr> innerStates{
      std::make_shared<VectorStateDeclaration>(
          "inner", innerSchema, innerInitialPlan)};
  std::vector<core::PlanNodePtr> innerPlans{innerBodyBuilder.planNode()};
  ConvergenceConfig innerConvergence{
      .plans = {innerConvergenceBuilder.planNode()}, .maxIterations = 100};
  core::PlanNodePtr innerFixedPoint = std::make_shared<FixedPointNode>(
      "inner-fixed-point",
      std::move(innerStates),
      std::move(innerPlans),
      std::move(innerConvergence),
      /*outputStateEntry=*/"inner");

  // Outer fixed point: each iteration adds the inner loop's result to val,
  // reaching 3 then 6.
  auto outerSeed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto outerInitialPlan =
      PlanBuilder(idGenerator).values({outerSeed}).planNode();

  PlanBuilder outerBodyBuilder(idGenerator);
  outerBodyBuilder.stateSource("outer", outerSchema)
      .hashJoin({"key"}, {"ikey"}, innerFixedPoint, "", {"key", "val", "ival"})
      .project({"key", "val + ival AS val"});

  PlanBuilder outerConvergenceBuilder(idGenerator);
  outerConvergenceBuilder.stateSource("outer", outerSchema)
      .project({"val >= 6 AS converged"});

  std::vector<StateDeclarationPtr> outerStates{
      std::make_shared<VectorStateDeclaration>(
          "outer", outerSchema, outerInitialPlan)};
  std::vector<core::PlanNodePtr> outerPlans{outerBodyBuilder.planNode()};
  ConvergenceConfig outerConvergence{
      .plans = {outerConvergenceBuilder.planNode()}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "outer-fixed-point",
      std::move(outerStates),
      std::move(outerPlans),
      std::move(outerConvergence),
      /*outputStateEntry=*/"outer");

  expectBothModes(node, {{0, 6}});
}

// Nesting where the inner loop shuffles but the outer does not.  Each outer
// worker hosts one inner task, and those inner tasks are each other's shuffle
// peers, so the fixed point is two-worker even though nothing in the outer
// node's own chains says so.
TEST_F(FixedPointTest, nestedInnerShuffle) {
  constexpr int32_t kNumWorkers{2};
  auto outerSchema = ROW({"key", "val"}, BIGINT());
  auto innerSchema = ROW({"ikey", "ival"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey each iteration, counting ival up to 3.
    auto innerSeed = makeRowVector(
        {"ikey", "ival"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto innerInitialPlan =
        PlanBuilder(idGenerator).values({innerSeed}).planNode();
    auto innerProducer = PlanBuilder(idGenerator)
                             .stateSource("inner", innerSchema)
                             .partitionedOutput(
                                 {"ikey"},
                                 kNumWorkers,
                                 /*replicateNullsAndAny=*/false,
                                 moduloSpec)
                             .planNode();
    auto innerConsumer = PlanBuilder(idGenerator)
                             .exchange(innerSchema, "Presto")
                             .singleAggregation({"ikey"}, {"sum(ival)"})
                             .project({"ikey", "a0 + 1 AS ival"})
                             .planNode();
    auto innerConvergence = PlanBuilder(idGenerator)
                                .stateSource("inner", innerSchema)
                                .project({"ival >= 3 AS converged"})
                                .planNode();
    std::vector<StateDeclarationPtr> innerStates{
        std::make_shared<VectorStateDeclaration>(
            "inner", innerSchema, innerInitialPlan)};
    std::vector<core::PlanNodePtr> innerPlans{innerProducer, innerConsumer};
    ConvergenceConfig innerConvergenceConfig{
        .plans = {innerConvergence}, .maxIterations = 100};
    core::PlanNodePtr innerFixedPoint = std::make_shared<FixedPointNode>(
        "inner-fixed-point",
        std::move(innerStates),
        std::move(innerPlans),
        std::move(innerConvergenceConfig),
        /*outputStateEntry=*/"inner");

    // Outer loop: local, adds the inner result to its own value each round.
    auto outerSeed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto outerInitialPlan =
        PlanBuilder(idGenerator).values({outerSeed}).planNode();
    auto outerBody =
        PlanBuilder(idGenerator)
            .stateSource("outer", outerSchema)
            .hashJoin(
                {"key"}, {"ikey"}, innerFixedPoint, "", {"key", "val", "ival"})
            .project({"key", "val + ival AS val"})
            .planNode();
    auto outerConvergence = PlanBuilder(idGenerator)
                                .stateSource("outer", outerSchema)
                                .project({"val >= 6 AS converged"})
                                .planNode();
    std::vector<StateDeclarationPtr> outerStates{
        std::make_shared<VectorStateDeclaration>(
            "outer", outerSchema, outerInitialPlan)};
    std::vector<core::PlanNodePtr> outerPlans{outerBody};
    ConvergenceConfig outerConvergenceConfig{
        .plans = {outerConvergence}, .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "outer-fixed-point",
        std::move(outerStates),
        std::move(outerPlans),
        std::move(outerConvergenceConfig),
        /*outputStateEntry=*/"outer");
  };

  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_EQ(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// Nesting where both loops shuffle.  The outer re-partitions its own state
// each round while the inner runs a full shuffling loop inside every outer
// iteration, so the two levels' producers coexist and must stay addressable
// apart: the inner's address carries the outer's (iteration, plan), which is
// what keeps round j of the inner under outer round i distinct from round j
// under outer round i+1.
TEST_F(FixedPointTest, nestedOuterAndInnerShuffle) {
  constexpr int32_t kNumWorkers{2};
  auto outerSchema = ROW({"key", "val"}, BIGINT());
  auto innerSchema = ROW({"ikey", "ival"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey, counting ival up to 3.
    auto innerSeed = makeRowVector(
        {"ikey", "ival"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto innerProducer = PlanBuilder(idGenerator)
                             .stateSource("inner", innerSchema)
                             .partitionedOutput(
                                 {"ikey"},
                                 kNumWorkers,
                                 /*replicateNullsAndAny=*/false,
                                 moduloSpec)
                             .planNode();
    auto innerConsumer = PlanBuilder(idGenerator)
                             .exchange(innerSchema, "Presto")
                             .singleAggregation({"ikey"}, {"sum(ival)"})
                             .project({"ikey", "a0 + 1 AS ival"})
                             .planNode();
    std::vector<StateDeclarationPtr> innerStates{
        std::make_shared<VectorStateDeclaration>(
            "inner",
            innerSchema,
            PlanBuilder(idGenerator).values({innerSeed}).planNode())};
    std::vector<core::PlanNodePtr> innerPlans{innerProducer, innerConsumer};
    ConvergenceConfig innerConvergenceConfig{
        .plans = {PlanBuilder(idGenerator)
                      .stateSource("inner", innerSchema)
                      .project({"ival >= 3 AS converged"})
                      .planNode()},
        .maxIterations = 100};
    core::PlanNodePtr innerFixedPoint = std::make_shared<FixedPointNode>(
        "inner-fixed-point",
        std::move(innerStates),
        std::move(innerPlans),
        std::move(innerConvergenceConfig),
        /*outputStateEntry=*/"inner");

    // Outer loop: adds the inner loop's result to its own value, then shuffles
    // by key (each key stays on its worker).  The nest sits in the *producing*
    // plan, so that plan's sub-task is itself a FixedPointLoop and its
    // PartitionedOutput runs in that task's trailing plan -- which therefore
    // has to carry the producer id peers read.
    auto outerSeed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto outerProducer =
        PlanBuilder(idGenerator)
            .stateSource("outer", outerSchema)
            .hashJoin(
                {"key"}, {"ikey"}, innerFixedPoint, "", {"key", "val", "ival"})
            .project({"key", "val + ival AS val"})
            .partitionedOutput(
                {"key"},
                kNumWorkers,
                /*replicateNullsAndAny=*/false,
                moduloSpec)
            .planNode();
    auto outerConsumer = PlanBuilder(idGenerator)
                             .exchange(outerSchema, "Presto")
                             .singleAggregation({"key"}, {"sum(val)"})
                             .project({"key", "a0 AS val"})
                             .planNode();
    std::vector<StateDeclarationPtr> outerStates{
        std::make_shared<VectorStateDeclaration>(
            "outer",
            outerSchema,
            PlanBuilder(idGenerator).values({outerSeed}).planNode())};
    std::vector<core::PlanNodePtr> outerPlans{outerProducer, outerConsumer};
    ConvergenceConfig outerConvergenceConfig{
        .plans = {PlanBuilder(idGenerator)
                      .stateSource("outer", outerSchema)
                      .project({"val >= 6 AS converged"})
                      .planNode()},
        .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "outer-fixed-point",
        std::move(outerStates),
        std::move(outerPlans),
        std::move(outerConvergenceConfig),
        /*outputStateEntry=*/"outer");
  };

  // The inner loop yields 3 every outer round, so each worker goes 0 -> 3 -> 6
  // and stops on the second.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_EQ(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// The other placement: both loops shuffle, but the nest sits in the outer's
// *consuming* plan rather than its producing one.  That plan's sub-task is a
// FixedPointLoop, and the enclosing loop wires its Exchange splits to that
// task -- so this covers the split-routing side, where the producer-plan test
// covers the output-buffer side.
TEST_F(FixedPointTest, nestedInConsumerPlan) {
  constexpr int32_t kNumWorkers{2};
  auto outerSchema = ROW({"key", "val"}, BIGINT());
  auto innerSchema = ROW({"ikey", "ival"}, BIGINT());
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  auto makeWorkerNode = [&](int32_t worker) -> FixedPointNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey, counting ival up to 3.
    auto innerSeed = makeRowVector(
        {"ikey", "ival"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto innerProducer = PlanBuilder(idGenerator)
                             .stateSource("inner", innerSchema)
                             .partitionedOutput(
                                 {"ikey"},
                                 kNumWorkers,
                                 /*replicateNullsAndAny=*/false,
                                 moduloSpec)
                             .planNode();
    auto innerConsumer = PlanBuilder(idGenerator)
                             .exchange(innerSchema, "Presto")
                             .singleAggregation({"ikey"}, {"sum(ival)"})
                             .project({"ikey", "a0 + 1 AS ival"})
                             .planNode();
    std::vector<StateDeclarationPtr> innerStates{
        std::make_shared<VectorStateDeclaration>(
            "inner",
            innerSchema,
            PlanBuilder(idGenerator).values({innerSeed}).planNode())};
    std::vector<core::PlanNodePtr> innerPlans{innerProducer, innerConsumer};
    ConvergenceConfig innerConvergenceConfig{
        .plans = {PlanBuilder(idGenerator)
                      .stateSource("inner", innerSchema)
                      .project({"ival >= 3 AS converged"})
                      .planNode()},
        .maxIterations = 100};
    core::PlanNodePtr innerFixedPoint = std::make_shared<FixedPointNode>(
        "inner-fixed-point",
        std::move(innerStates),
        std::move(innerPlans),
        std::move(innerConvergenceConfig),
        /*outputStateEntry=*/"inner");

    // Outer loop: shuffles by key first, then adds the inner loop's result in
    // the consuming plan.
    auto outerSeed = makeRowVector(
        {"key", "val"},
        {makeFlatVector<int64_t>({worker}), makeFlatVector<int64_t>({0})});
    auto outerProducer = PlanBuilder(idGenerator)
                             .stateSource("outer", outerSchema)
                             .partitionedOutput(
                                 {"key"},
                                 kNumWorkers,
                                 /*replicateNullsAndAny=*/false,
                                 moduloSpec)
                             .planNode();
    auto outerConsumer =
        PlanBuilder(idGenerator)
            .exchange(outerSchema, "Presto")
            .singleAggregation({"key"}, {"sum(val)"})
            .project({"key", "a0 AS val"})
            .hashJoin(
                {"key"}, {"ikey"}, innerFixedPoint, "", {"key", "val", "ival"})
            .project({"key", "val + ival AS val"})
            .planNode();
    std::vector<StateDeclarationPtr> outerStates{
        std::make_shared<VectorStateDeclaration>(
            "outer",
            outerSchema,
            PlanBuilder(idGenerator).values({outerSeed}).planNode())};
    std::vector<core::PlanNodePtr> outerPlans{outerProducer, outerConsumer};
    ConvergenceConfig outerConvergenceConfig{
        .plans = {PlanBuilder(idGenerator)
                      .stateSource("outer", outerSchema)
                      .project({"val >= 6 AS converged"})
                      .planNode()},
        .maxIterations = 100};
    return std::make_shared<FixedPointNode>(
        "outer-fixed-point",
        std::move(outerStates),
        std::move(outerPlans),
        std::move(outerConvergenceConfig),
        /*outputStateEntry=*/"outer");
  };

  // The inner loop yields 3 every outer round, so each worker goes 0 -> 3 -> 6
  // and stops on the second.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_EQ(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      expected);
}

// FixedPointNode validates the sub-plan chaining convention in its constructor:
// the first plan starts with StateSource, the last produces the rows written
// back (so it must not end with PartitionedOutput), and adjacent plans are
// linked only by PartitionedOutput -> Exchange (plans are chained by shuffle,
// never through shared state).
TEST_F(FixedPointTest, validatesPlanStructure) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});

  auto makeNode = [&](std::vector<core::PlanNodePtr> plans) {
    std::vector<StateDeclarationPtr> declarations{
        std::make_shared<VectorStateDeclaration>("s", schema)};
    return std::make_shared<FixedPointNode>(
        "fixed-point",
        std::move(declarations),
        std::move(plans),
        ConvergenceConfig{
            .plans = {},
            .maxIterations = 10,
            .errorWhenMaxIterationReached = false},
        /*outputStateEntry=*/"s");
  };

  // Valid: a single plan reading the state and projecting the rows written
  // back (the framework performs the write).
  {
    PlanBuilder builder(idGen);
    builder.stateSource("s", schema);
    EXPECT_NO_THROW(makeNode({builder.project({"id", "val"}).planNode()}));
  }

  // At least one plan is required.
  VELOX_ASSERT_THROW(makeNode({}), "requires at least one plan");

  // outputStateEntry must name a declared vector entry (here it names none).
  {
    PlanBuilder builder(idGen);
    builder.stateSource("s", schema);
    auto plan = builder.project({"id", "val"}).planNode();
    std::vector<StateDeclarationPtr> declarations{
        std::make_shared<VectorStateDeclaration>("s", schema)};
    VELOX_ASSERT_THROW(
        std::make_shared<FixedPointNode>(
            "fixed-point",
            std::move(declarations),
            std::vector<core::PlanNodePtr>{plan},
            ConvergenceConfig{
                .plans = {},
                .maxIterations = 10,
                .errorWhenMaxIterationReached = false},
            /*outputStateEntry=*/"missing"),
        "outputStateEntry must name a declared vector state entry");
  }

  // First plan must start with StateSource (here it starts with Values).
  {
    PlanBuilder builder(idGen);
    VELOX_ASSERT_THROW(
        makeNode({builder.values({seed}).planNode()}),
        "must start with a StateSourceNode");
  }

  // Last plan must not end with PartitionedOutput (it produces the rows written
  // back, it does not shuffle).
  {
    PlanBuilder builder(idGen);
    auto plan =
        builder.stateSource("s", schema).partitionedOutput({}, 1).planNode();
    VELOX_ASSERT_THROW(
        makeNode({plan}), "not shuffle through a PartitionedOutput");
  }

  // A non-last plan must end with PartitionedOutput: here plan 0 ends with a
  // Project (state-chaining, not a shuffle) -- the bug the validation forbids.
  {
    PlanBuilder builder0(idGen);
    auto plan0 =
        builder0.stateSource("s", schema).project({"id", "val"}).planNode();
    PlanBuilder builder1(idGen);
    auto plan1 = builder1.exchange(schema, "Presto").planNode();
    VELOX_ASSERT_THROW(
        makeNode({plan0, plan1}), "must end with a PartitionedOutput");
  }

  // A non-first plan must start with Exchange: here plan 1 starts with a
  // StateSource instead of receiving plan 0's shuffle.
  {
    PlanBuilder builder0(idGen);
    auto plan0 =
        builder0.stateSource("s", schema).partitionedOutput({}, 1).planNode();
    PlanBuilder builder1(idGen);
    auto plan1 =
        builder1.stateSource("s", schema).project({"id", "val"}).planNode();
    VELOX_ASSERT_THROW(makeNode({plan0, plan1}), "must start with an Exchange");
  }
}

// errorWhenMaxIterationReached (default true) requires a convergence plan: a
// null plan never converges, so combining it with the default flag would always
// fail -- the FixedPointNode constructor rejects that combination up front.
TEST_F(FixedPointTest, errorWhenMaxIterationReachedRequiresPlan) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>();
  PlanBuilder builder(idGen);
  builder.stateSource("s", schema);
  std::vector<StateDeclarationPtr> declarations{
      std::make_shared<VectorStateDeclaration>("s", schema)};
  std::vector<core::PlanNodePtr> plans{
      builder.project({"id", "val"}).planNode()};
  // Null convergence plan + the default errorWhenMaxIterationReached=true.
  VELOX_ASSERT_THROW(
      std::make_shared<FixedPointNode>(
          "fixed-point",
          std::move(declarations),
          std::move(plans),
          ConvergenceConfig{.plans = {}, .maxIterations = 10},
          /*outputStateEntry=*/"s"),
      "errorWhenMaxIterationReached requires a convergence criterion");
}

// The delta row count is already the verdict, so pairing it with a convergence
// sequence is a contradiction rather than a refinement.
TEST_F(FixedPointTest, whenDeltaEmptyRejectsConvergencePlan) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>();
  PlanBuilder builder(idGen);
  builder.stateSource("s", schema);
  PlanBuilder convergenceBuilder(idGen);
  convergenceBuilder.stateSource("s", schema)
      .singleAggregation({}, {"count(1)"})
      .project({"a0 = 0 AS converged"});

  std::vector<StateDeclarationPtr> declarations{
      std::make_shared<VectorStateDeclaration>("s", schema)};
  std::vector<core::PlanNodePtr> plans{
      builder.project({"id", "val"}).planNode()};
  auto convergence = ConvergenceConfig::whenDeltaEmpty(10);
  convergence.plans = {convergenceBuilder.planNode()};
  VELOX_ASSERT_THROW(
      std::make_shared<FixedPointNode>(
          "fixed-point",
          std::move(declarations),
          std::move(plans),
          std::move(convergence),
          /*outputStateEntry=*/"s"),
      "stopWhenDeltaEmpty and a convergence sequence are mutually exclusive");
}

// The delta is a worker's local row count, so with peers one worker's frontier
// can empty while others still read its output.  The node rejects the
// combination rather than letting the loop strand its peers.
TEST_F(FixedPointTest, whenDeltaEmptyRejectsShuffle) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"id", "val"},
      {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({8, 5})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder producerBuilder(idGenerator);
  auto producer = producerBuilder.stateSource("frontier", schema)
                      .partitionedOutput({"id"}, 2)
                      .planNode();
  PlanBuilder consumerBuilder(idGenerator);
  consumerBuilder.exchange(schema, "Presto").project({"id", "val / 2 AS val"});

  std::vector<StateDeclarationPtr> declarations{
      std::make_shared<VectorStateDeclaration>(
          "frontier", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{producer, consumerBuilder.planNode()};
  VELOX_ASSERT_THROW(
      std::make_shared<FixedPointNode>(
          "fixed-point",
          std::move(declarations),
          std::move(plans),
          ConvergenceConfig::whenDeltaEmpty(10),
          /*outputStateEntry=*/"frontier"),
      "stopWhenDeltaEmpty requires a non-shuffling fixed point");
}

// With errorWhenMaxIterationReached (default true), running all maxIterations
// without the convergence plan firing fails the loop.
// start() is the parallel entry point; a serial fixed point is driven by
// next().  Task enforces that before delegating to the executor -- the same
// check, and the same error, as for a driver-based task -- so serial-mode
// misuse cannot silently launch the loop on the orchestration executor.
TEST_F(FixedPointTest, startRequiresParallelMode) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"val >= 3 AS converged"});

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  ConvergenceConfig convergence{
      .plans = {convergenceBuilder.planNode()}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  auto task = makeTask(node, exec::Task::ExecutionMode::kSerial, "serialstart");
  VELOX_ASSERT_THROW(
      task->start(/*maxDrivers=*/1), "Inconsistent task execution mode");
}

// The node validates that the convergence sequence emits one BOOLEAN *column*,
// but the row count is a runtime property it cannot see.  A sequence that emits
// several rows would otherwise decide the loop on whichever batch arrived
// first, so the task rejects it instead.
TEST_F(FixedPointTest, convergenceMustProduceOneRow) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  // Two state rows, and a convergence plan that projects per row rather than
  // aggregating -- two verdicts for one loop.
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({0, 0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"val >= 3 AS converged"});

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  ConvergenceConfig convergence{
      .plans = {convergenceBuilder.planNode()}, .maxIterations = 100};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  VELOX_ASSERT_THROW(
      runViaTask(node, exec::Task::ExecutionMode::kSerial),
      "must produce at most one row");
}

// What one iteration costs when the body does nothing: a Task created, planned,
// run and torn down per iteration, plus the state write.  Measured at 35-43 us
// per iteration (2'000 iterations in 69-85 ms), which is the floor on how deep
// a recursion can practically go.  Disabled because it is a measurement, not an
// assertion; run it with
//
//   buck2 run @fbcode//mode/opt fbcode//velox/exec/tests:velox_fixed_point_test
//   \
//     -- --gtest_filter='*perIterationCost' --gtest_also_run_disabled_tests
//
// and use mode/opt: a sanitized build reports ~40x this.
TEST_F(FixedPointTest, DISABLED_perIterationCost) {
  auto schema = ROW({"key", "val"}, BIGINT());
  constexpr int32_t kIterations = 2000;
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});
  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{bodyBuilder.planNode()};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      ConvergenceConfig::withMaxIterations(kIterations),
      /*outputStateEntry=*/"vals");
  auto start = std::chrono::steady_clock::now();
  auto rows = runViaTask(node, exec::Task::ExecutionMode::kSerial);
  auto elapsed = std::chrono::steady_clock::now() - start;
  const auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cout << kIterations << " iterations in " << us
            << " us = " << static_cast<double>(us) / kIterations
            << " us/iteration" << std::endl;
  EXPECT_EQ(rows.size(), 1);
}

TEST_F(FixedPointTest, errorWhenMaxIterationReachedThrows) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      {"key", "val"},
      {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})});
  auto initialPlan = PlanBuilder(idGenerator).values({seed}).planNode();

  // Body increments val; convergence wants val >= 1000, never reached in 3.
  PlanBuilder bodyBuilder(idGenerator);
  bodyBuilder.stateSource("vals", schema).project({"key", "val + 1 AS val"});
  auto body = bodyBuilder.planNode();

  PlanBuilder convergenceBuilder(idGenerator);
  convergenceBuilder.stateSource("vals", schema)
      .project({"val >= 1000 AS converged"});
  auto convergencePlan = convergenceBuilder.planNode();

  std::vector<StateDeclarationPtr> stateDeclarations{
      std::make_shared<VectorStateDeclaration>("vals", schema, initialPlan)};
  std::vector<core::PlanNodePtr> plans{body};
  ConvergenceConfig convergence{.plans = {convergencePlan}, .maxIterations = 3};
  auto node = std::make_shared<FixedPointNode>(
      "fixed-point",
      std::move(stateDeclarations),
      std::move(plans),
      std::move(convergence),
      /*outputStateEntry=*/"vals");

  // val reaches 3 after maxIterations, below the threshold -> the loop fails in
  // both execution modes.
  VELOX_ASSERT_THROW(
      runViaTask(node, exec::Task::ExecutionMode::kSerial),
      "did not converge within");
  VELOX_ASSERT_THROW(
      runViaTask(node, exec::Task::ExecutionMode::kParallel),
      "did not converge within");
}

} // namespace
} // namespace facebook::velox::exec
