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
#include <folly/synchronization/Baton.h>
#include <gmock/gmock.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <tuple>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/FixedPointPlanNodes.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/FixedPointLoop.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Split.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TestIndexStorageConnector.h"

namespace facebook::velox::exec {
namespace {

using core::ConvergenceConfig;
using core::FixedPointNode;
using core::HashTableStateDeclaration;
using core::PlanNodePtr;
using core::StateDeclarationPtr;
using core::VectorStateDeclaration;
using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

std::vector<std::pair<int64_t, int64_t>> readRows(TaskCursor& cursor) {
  std::vector<std::pair<int64_t, int64_t>> rows;
  while (cursor.moveNext()) {
    const auto& result = cursor.current();
    auto first = result->childAt(0)->as<SimpleVector<int64_t>>();
    auto second = result->childAt(1)->as<SimpleVector<int64_t>>();
    for (vector_size_t i = 0; i < result->size(); ++i) {
      rows.emplace_back(first->valueAt(i), second->valueAt(i));
    }
  }
  return rows;
}

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

// Returns the id of the Exchange through which an initial plan of the fixed
// point 'node' reads its upstream.
core::PlanNodeId initialExchangeId(const core::PlanNodePtr& node) {
  for (const auto& declaration :
       node->as<FixedPointNode>()->stateDeclarations()) {
    if (declaration->initialPlan() == nullptr) {
      continue;
    }
    if (const auto* exchange = core::PlanNode::findFirstNode(
            declaration->initialPlan().get(),
            [](const core::PlanNode* planNode) {
              return planNode->is<core::ExchangeNode>();
            })) {
      return exchange->id();
    }
  }
  VELOX_FAIL("No initial plan of the fixed point reads through an Exchange");
}

class FixedPointTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    // Route local:// task ids to the in-process exchange source so
    // PartitionedOutput -> Exchange works without networking.
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(
        exec::test::createLocalExchangeSource);
  }

  // Number of top-level worker tasks for 'node': the partition count of the
  // first plan of whichever chain shuffles -- the body, or the convergence
  // sequence when the body stays local -- or 1 when neither does.
  static int32_t numWorkersOf(const core::PlanNodePtr& node) {
    return node->as<FixedPointNode>()->numWorkers();
  }

  // Runs the iteration loops, kept off the query executor the sub-tasks'
  // drivers run on.  One thread serves every loop, nested or peer: a loop
  // suspends while its sub-tasks run.
  static folly::Executor* orchestrationExecutor() {
    static auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(1);
    return executor.get();
  }

  // FixedPointOptions wiring sub-task ids/URIs in the in-process "local://"
  // scheme.  A FixedPointLoop provides no defaults, so the coordinator
  // (here the test) supplies them: producerLocation gives each peer's
  // per-iteration shuffle producer (task id + exchange URI), subTaskId names
  // this worker's internal sub-tasks.
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
        auto id = fmt::format(
            "local://{}.it{}.p{}", workerAddress, iteration, planIndex);
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
  core::PlanNodePtr countingNode() {
    return countingNodeTo(/*target=*/3, /*maxIterations=*/100);
  }

  // Same, but converges at val='target', after 'target' iterations, unless
  // 'maxIterations' runs out first.
  core::PlanNodePtr countingNodeTo(int64_t target, int32_t maxIterations) {
    auto schema = ROW({"key", "val"}, BIGINT());
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto seed = makeRowVector(
        schema->names(),
        {
            makeFlatVector<int64_t>({0}),
            makeFlatVector<int64_t>({0}),
        });

    return PlanBuilder(idGenerator)
        .fixedPoint(
            std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                PlanBuilder(idGenerator).values({seed}).planNode()),
            PlanBuilder(idGenerator)
                .stateSource("vals", schema)
                .project({"key AS next_key", "val + 1 AS next_val"}),
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("vals", schema)
                    .project({fmt::format("val >= {} AS converged", target)})
                    .planNode(),
                maxIterations),
            /*outputStateEntry=*/"vals")
        .planNode();
  }

  // Creates the task 'node' runs as, named after 'label'.  'options' defaults
  // to the in-process coordinator hooks.
  exec::TaskPtr makeTask(
      const core::PlanNodePtr& node,
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
        // A parallel task delivers its rows to a consumer; these tests read
        // the task's state and statistics instead.
        mode == exec::Task::ExecutionMode::kParallel
            ? exec::Consumer{[](RowVectorPtr, bool, ContinueFuture*) {
                return exec::BlockingReason::kNotBlocked;
              }}
            : exec::Consumer{},
        /*memoryArbitrationPriority=*/0,
        /*spillDiskOpts=*/std::nullopt,
        /*onError=*/nullptr,
        options != nullptr ? options : &localOptions());
  }

  // Starts a task running 'plan', a producer whose partitioned output a fixed
  // point's initial plan reads through an Exchange.
  exec::TaskPtr startUpstreamTask(
      const PlanBuilder& plan,
      const std::shared_ptr<core::QueryCtx>& queryCtx) {
    auto task = exec::Task::create(
        fmt::format(
            "local://fixedpoint-upstream-{}-{}",
            queryCtx->queryId(),
            numUpstreamTasks_++),
        core::PlanFragment{plan.planNode()},
        /*destination=*/0,
        queryCtx,
        exec::Task::ExecutionMode::kParallel);
    task->start(/*maxDrivers=*/1);
    return task;
  }

  using NodeFactory = std::function<core::PlanNodePtr(int32_t)>;

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
  // an upstream producer task and adds a remote split for it to each worker's
  // initial-plan Exchange, so the worker's initial plan receives its shard via
  // shuffle.  If 'initSplitsFor' is set, the coordinator assigns worker d the
  // source splits it returns (e.g. a TableScan's file split) — its initial plan
  // reads its shard from them.  Workers run concurrently (each on its own
  // thread) and rendezvous through the shuffle.  Returns each worker's shard
  // rows and iteration count.
  std::vector<WorkerRun> runWorkers(
      const NodeFactory& makeWorkerNode,
      exec::Task::ExecutionMode mode,
      const PlanBuilder* upstreamPlan = nullptr,
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
    exec::TaskPtr upstream;
    if (upstreamPlan != nullptr) {
      // The coordinator names the upstream task (the operator cannot); workers
      // learn it from the remote splits added below.  "local://" routes it to
      // the in-process exchange source.
      upstream = startUpstreamTask(*upstreamPlan, queryCtx);
    }

    // Each worker collects its shard rows into perWorker[d].
    std::vector<std::vector<std::pair<int64_t, int64_t>>> perWorker(numWorkers);

    std::vector<std::unique_ptr<exec::TaskCursor>> cursors;
    cursors.reserve(numWorkers);
    for (int32_t d = 0; d < numWorkers; ++d) {
      auto node = (d == 0) ? node0 : makeWorkerNode(d);
      cursors.push_back(
          TaskCursor::create({
              .planNode = std::move(node),
              .destination = d,
              .queryCtx = mode == exec::Task::ExecutionMode::kParallel
                  ? queryCtx
                  : nullptr,
              .copyResult = false,
              .serialExecution = mode == exec::Task::ExecutionMode::kSerial,
              .fixedPointOptions = &localOptions(),
          }));
    }

    // The coordinator wires the body-shuffle topology by adding one remote
    // split per peer worker's task to each parent (all-to-all here); the fixed
    // point derives each peer's per-iteration producer location from its task
    // id via the producerLocation hook.  'initSplitsFor(d)' splits are worker
    // d's initial-plan source splits (e.g. a TableScan file split), forwarded
    // verbatim; an initial-plan Exchange gets a remote split naming the
    // upstream task.
    if (node0->requiresSplits()) {
      for (int32_t d = 0; d < numWorkers; ++d) {
        for (int32_t e = 0; e < numWorkers; ++e) {
          cursors[d]->task()->addSplit(
              fixedPointNodeId, remoteSplit(cursors[e]->task()));
        }
        if (upstream != nullptr) {
          const auto exchangeId = initialExchangeId(node0);
          cursors[d]->task()->addSplit(exchangeId, remoteSplit(upstream));
          cursors[d]->task()->noMoreSplits(exchangeId);
        }
        if (initSplitsFor != nullptr) {
          // Addressed to the node that reads them, not to the fixed point:
          // that is how a fixed point with several scanned initial plans keeps
          // each one's splits apart.
          for (auto& [sourceId, split] : initSplitsFor(d)) {
            cursors[d]->task()->addSplit(sourceId, std::move(split));
            cursors[d]->task()->noMoreSplits(sourceId);
          }
        }
        cursors[d]->task()->noMoreSplits(fixedPointNodeId);
      }
    }

    // Drive each cursor on its own thread so peer workers can rendezvous
    // through the shuffle in both serial and parallel execution.
    std::vector<std::exception_ptr> errors(numWorkers);
    std::vector<std::thread> threads;
    threads.reserve(numWorkers);
    for (int32_t d = 0; d < numWorkers; ++d) {
      threads.emplace_back([&, d]() {
        try {
          perWorker[d] = readRows(*cursors[d]);
        } catch (...) {
          errors[d] = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
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
      auto* fixedPoint = cursors[d]->task()->fixedPoint();
      VELOX_CHECK_NOT_NULL(fixedPoint, "Worker task has no FixedPointLoop");
      results.push_back({std::move(perWorker[d]), fixedPoint->iterations()});
    }
    return results;
  }

  // Unions every worker's shard rows into (col0, col1) BIGINT pairs.
  static std::vector<std::pair<int64_t, int64_t>> unionRows(
      const std::vector<WorkerRun>& runs) {
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto& worker : runs) {
      pairs.insert(pairs.end(), worker.rows.begin(), worker.rows.end());
    }
    return pairs;
  }

  // Runs the workers in 'mode' and returns the union of their shard rows as
  // (col0, col1) BIGINT pairs.
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const NodeFactory& makeWorkerNode,
      exec::Task::ExecutionMode mode) {
    return unionRows(runWorkers(makeWorkerNode, mode));
  }

  // Same, but with an upstream producer task feeding the workers' initial plans
  // through shuffle.
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const NodeFactory& makeWorkerNode,
      const PlanBuilder& upstreamPlan,
      exec::Task::ExecutionMode mode) {
    return unionRows(runWorkers(makeWorkerNode, mode, &upstreamPlan));
  }

  // Convenience overload: every worker runs the same node (used for N=1 tests,
  // and shuffles where replicating the seed is acceptable).
  std::vector<std::pair<int64_t, int64_t>> runViaTask(
      const core::PlanNodePtr& node,
      exec::Task::ExecutionMode mode) {
    return runViaTask(
        [&](int32_t /*worker*/) -> core::PlanNodePtr { return node; }, mode);
  }

  // Runs a non-shuffling fixed point under BOTH parent modes and asserts each
  // yields 'expected'.  Sub-tasks inherit the parent's mode, so this exercises
  // both the serial (next()) and parallel (start()) sub-task drives.
  void expectBothModes(
      const core::PlanNodePtr& node,
      const std::vector<std::pair<int64_t, int64_t>>& expected) {
    EXPECT_THAT(
        runViaTask(node, exec::Task::ExecutionMode::kSerial),
        testing::UnorderedElementsAreArray(expected))
        << "serial parent";
    EXPECT_THAT(
        runViaTask(node, exec::Task::ExecutionMode::kParallel),
        testing::UnorderedElementsAreArray(expected))
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
    return rows;
  }

  // Returns a split that reads 'file' through the Hive connector.
  exec::Split hiveSplit(const std::shared_ptr<test::TempFilePath>& file) {
    return exec::Split(makeHiveConnectorSplit(file->getPath()));
  }

  exec::Split remoteSplit(const exec::TaskPtr& task) {
    return exec::Split(
        std::make_shared<exec::RemoteConnectorSplit>(task->taskId()));
  }

  // Runs 'plan' in parallel with up to 'maxDrivers' drivers per pipeline and
  // FixedPointOptions::iterationMaxDrivers set to 'iterationMaxDrivers'.  A
  // shuffling fixed point at the root runs as one worker.  Returns the rows as
  // (col0, col1) BIGINT pairs.
  std::vector<std::pair<int64_t, int64_t>> runWithDrivers(
      const core::PlanNodePtr& plan,
      int32_t maxDrivers,
      uint32_t iterationMaxDrivers) {
    auto options = localOptions();
    options.iterationMaxDrivers = iterationMaxDrivers;
    auto cursor = TaskCursor::create({
        .planNode = plan,
        .maxDrivers = maxDrivers,
        .queryCtx = core::QueryCtx::create(cpuExecutor_.get()),
        .copyResult = false,
        .fixedPointOptions = &options,
    });
    if (plan->requiresSplits()) {
      cursor->task()->addSplit(plan->id(), remoteSplit(cursor->task()));
      cursor->task()->noMoreSplits(plan->id());
    }
    auto rows = readRows(*cursor);
    return rows;
  }

  // Returns 'numBatches' single-row (key, val) batches with key 0 and val
  // 0..numBatches-1.
  std::vector<RowVectorPtr> makeOneRowBatches(int32_t numBatches) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (int32_t i = 0; i < numBatches; ++i) {
      batches.push_back(makeRowVector(
          {"key", "val"},
          {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({i})}));
    }
    return batches;
  }

  // Runs the workers' sub-tasks.  A shuffling fixed point keeps up to
  // numWorkers * numPlans sub-task drivers live at once (producers stay up
  // while peers' consumers drain them), so this must comfortably exceed that.
  // The run() loops that block on these sub-tasks run on a separate executor
  // (see orchestrationExecutor()).
  std::shared_ptr<folly::CPUThreadPoolExecutor> cpuExecutor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(16)};

  // Makes each upstream task id unique within a query.
  int32_t numUpstreamTasks_{0};
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
// The FixedPointNode is driven through TaskCursor and emits result --
// the union of every iteration's frontier.
TEST_F(FixedPointTest, recursiveCte) {
  auto schema = ROW({"id", "depth"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Base case seed: {(id=1, depth=0)}.
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({0}),
      });

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      {"src", "dst"},
      {makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
       makeFlatVector<int64_t>({2, 3, 5, 4, 6})});

  // The exact 6-row union both verifies reachability and proves the iteration
  // terminated at the empty frontier (a wrong convergence would diverge or
  // hit maxIterations).
  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              // result accumulates every iteration's frontier (an append-mode
              // entry) and also provides the frontier each iteration reads;
              // seeded with the base case.
              std::make_shared<VectorStateDeclaration>(
                  "result",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode(),
                  /*append=*/true),
              // A single body plan: read the frontier (an in-loop StateSource
              // over the append-mode result entry returns the rows appended
              // last iteration), probe the edges, and append the next depth
              // back to result.  The frontier read and the accumulation are
              // the same entry, and the iteration is one local pipeline (no
              // shuffle), so it is a single plan -- no state-chained second
              // plan.
              PlanBuilder(idGenerator)
                  .stateSource("result", schema)
                  .hashJoin(
                      {"id"},
                      {"src"},
                      PlanBuilder(idGenerator).values({edges}).planNode(),
                      "",
                      {"dst", "depth"})
                  .project({"dst AS id", "depth + 1 AS depth"}),
              ConvergenceConfig::converging(
                  // Converged when the last iteration appended no rows (the
                  // frontier is empty).
                  PlanBuilder(idGenerator)
                      .stateSource("result", schema)
                      .singleAggregation({}, {"count(1)"})
                      .project({"a0 = 0 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"result")
          .planNode(),
      expected);
}

// The same recursive CTE, terminated by whenDeltaEmpty instead of a
// count(1) == 0 convergence plan.  Producing the identical 6-row result proves
// the delta row count is an equivalent verdict, reached without running a
// convergence sub-task per iteration.
TEST_F(FixedPointTest, whenDeltaEmptyRecursiveCte) {
  auto schema = ROW({"id", "depth"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({0}),
      });

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      {"src", "dst"},
      {makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
       makeFlatVector<int64_t>({2, 3, 5, 4, 6})});

  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "result",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode(),
                  /*append=*/true),
              PlanBuilder(idGenerator)
                  .stateSource("result", schema)
                  .hashJoin(
                      {"id"},
                      {"src"},
                      PlanBuilder(idGenerator).values({edges}).planNode(),
                      "",
                      {"dst", "depth"})
                  .project({"dst AS id", "depth + 1 AS depth"}),
              ConvergenceConfig::whenDeltaEmpty(100),
              /*outputStateEntry=*/"result")
          .planNode(),
      expected);
}

// A body that never runs dry hits maxIterations, and whenDeltaEmpty sets
// errorWhenMaxIterationReached, so the loop fails rather than returning a
// non-terminal result.
TEST_F(FixedPointTest, whenDeltaEmptyFailsWithoutEmptyDelta) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({1}),
      });

  VELOX_ASSERT_THROW(
      runViaTask(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  std::make_shared<VectorStateDeclaration>(
                      "result",
                      schema,
                      PlanBuilder(idGenerator).values({seed}).planNode()),
                  // Every iteration rewrites the single row, so the delta is
                  // never empty.
                  PlanBuilder(idGenerator)
                      .stateSource("result", schema)
                      .project({"id + 1 AS id", "val AS val"}),
                  ConvergenceConfig::whenDeltaEmpty(5),
                  /*outputStateEntry=*/"result")
              .planNode(),
          exec::Task::ExecutionMode::kSerial),
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
      schema->names(),
      {
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({0}),
      });

  // Static edges: 1->2, 2->3, 2->5, 3->4, 4->6.
  auto edges = makeRowVector(
      edgesSchema->names(),
      {
          makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
          makeFlatVector<int64_t>({2, 3, 5, 4, 6}),
      });

  // A single body plan: probe the prebuilt edges table with the frontier id (an
  // in-loop StateSource over the append-mode result entry returns the rows
  // appended last iteration), emit the next depth, and append it back to
  // result. StateHashJoin output is the probe columns (id, depth) plus the
  // table's dependent column (dst).
  auto joinOutput = ROW({"id", "depth", "dst"}, BIGINT());
  auto bodyBuilder = PlanBuilder(idGenerator).stateSource("result", schema);
  bodyBuilder.stateHashJoin("edges_ht", {"id"}, joinOutput)
      .project({"dst AS id", "depth + 1 AS depth"});
  auto body = bodyBuilder.planNode();

  std::vector<std::pair<int64_t, int64_t>> expected{
      {1, 0}, {2, 1}, {3, 2}, {5, 2}, {4, 3}, {6, 4}};
  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              {std::make_shared<HashTableStateDeclaration>(
                   "edges_ht",
                   edgesSchema,
                   std::vector<std::string>{"src"},
                   PlanBuilder(idGenerator).values({edges}).planNode()),
               std::make_shared<VectorStateDeclaration>(
                   "result",
                   schema,
                   PlanBuilder(idGenerator).values({seed}).planNode(),
                   /*append=*/true)},
              {body},
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("result", schema)
                      .singleAggregation({}, {"count(1)"})
                      .project({"a0 = 0 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"result")
          .planNode(),
      expected);
}

// Each output driver of a hash-table state's initial plan fills its own table,
// and the tables are merged.
TEST_F(FixedPointTest, hashTableStateIsBuiltPerDriver) {
  auto lookupSchema = ROW({"key", "payload"}, BIGINT());
  auto probeSchema = ROW({"key", "tag"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<RowVectorPtr> lookupBatches;
  for (int64_t batch = 0; batch < 4; ++batch) {
    lookupBatches.push_back(makeRowVector(
        lookupSchema->names(),
        {
            makeFlatVector<int64_t>(
                1'024, [batch](auto row) { return batch * 1'024 + row; }),
            makeFlatVector<int64_t>(
                1'024,
                [batch](auto row) { return (batch * 1'024 + row) * 10; }),
        }));
  }

  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::vector<StateDeclarationPtr>{
                          std::make_shared<HashTableStateDeclaration>(
                              "lookup",
                              lookupSchema,
                              std::vector<std::string>{"key"},
                              PlanBuilder(idGenerator)
                                  .values(lookupBatches)
                                  .localPartitionRoundRobin()
                                  .planNode()),
                          std::make_shared<VectorStateDeclaration>(
                              "probe",
                              probeSchema,
                              PlanBuilder(idGenerator)
                                  .values({makeRowVector(
                                      probeSchema->names(),
                                      {
                                          makeFlatVector<int64_t>({5, 4'000}),
                                          makeFlatVector<int64_t>({-1, -1}),
                                      })})
                                  .planNode())},
                      PlanBuilder(idGenerator)
                          .stateSource("probe", probeSchema)
                          .stateHashJoin(
                              "lookup",
                              {"key"},
                              ROW({"key", "tag", "payload"}, BIGINT()))
                          .project({"key", "payload AS tag"}),
                      ConvergenceConfig::withMaxIterations(1),
                      /*outputStateEntry=*/"probe")
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({5, 4'000}),
      makeFlatVector<int64_t>({50, 40'000}),
  });
  auto task = AssertQueryBuilder(plan)
                  .maxDrivers(4)
                  .fixedPointOptions(&localOptions())
                  .assertResults(expected);
  auto entry = task->fixedPoint()->state()->getHashTable("lookup");
  ASSERT_TRUE(entry.has_value());
  EXPECT_THAT(entry->table->allRows(), testing::SizeIs(4));
}

// Keys that cannot map to value ids leave the merged table in hash mode, which
// merges the per-driver tables in parallel on the query executor.
TEST_F(FixedPointTest, hashTableStateMergesInParallel) {
  constexpr int32_t kRowsPerBatch{4'096};
  auto lookupSchema = ROW({{"key", DOUBLE()}, {"payload", BIGINT()}});
  auto probeSchema = ROW({{"key", DOUBLE()}, {"id", BIGINT()}});
  auto outputSchema = ROW({"id", "payload"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<RowVectorPtr> lookupBatches;
  for (int64_t batch = 0; batch < 4; ++batch) {
    const auto offset = batch * kRowsPerBatch;
    lookupBatches.push_back(makeRowVector(
        {"key", "payload"},
        {makeFlatVector<double>(
             kRowsPerBatch, [offset](auto row) { return offset + row + 0.5; }),
         makeFlatVector<int64_t>(
             kRowsPerBatch, [offset](auto row) { return offset + row; })}));
  }

  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::vector<StateDeclarationPtr>{
                          std::make_shared<HashTableStateDeclaration>(
                              "lookup",
                              lookupSchema,
                              std::vector<std::string>{"key"},
                              PlanBuilder(idGenerator)
                                  .values(lookupBatches)
                                  .localPartitionRoundRobin()
                                  .planNode()),
                          std::make_shared<VectorStateDeclaration>(
                              "probe",
                              probeSchema,
                              PlanBuilder(idGenerator)
                                  .values({makeRowVector(
                                      {"key", "id"},
                                      {makeFlatVector<double>({7.5, 16'000.5}),
                                       makeFlatVector<int64_t>({1, 2})})})
                                  .planNode()),
                          std::make_shared<VectorStateDeclaration>(
                              "out", outputSchema)},
                      PlanBuilder(idGenerator)
                          .stateSource("probe", probeSchema)
                          .stateHashJoin(
                              "lookup",
                              {"key"},
                              ROW(
                                  {{"key", DOUBLE()},
                                   {"id", BIGINT()},
                                   {"payload", BIGINT()}}))
                          .project({"id", "payload"}),
                      ConvergenceConfig::withMaxIterations(1),
                      /*outputStateEntry=*/"out")
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<int64_t>({7, 16'000}),
  });
  auto task = AssertQueryBuilder(plan)
                  .maxDrivers(4)
                  .fixedPointOptions(&localOptions())
                  .assertResults(expected);
  auto entry = task->fixedPoint()->state()->getHashTable("lookup");
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->table->hashMode(), BaseHashTable::HashMode::kHash);
  EXPECT_THAT(
      entry->table->parallelJoinBuildStats().buildTimings,
      testing::Not(testing::IsEmpty()));
}

// HashTable merges at most 255 tables in parallel, so a hash-table state built
// by more output drivers than that must still build.
TEST_F(FixedPointTest, hashTableStateCapsBuildDrivers) {
  constexpr int32_t kNumDrivers{300};
  constexpr int32_t kRowsPerBatch{4'096};
  auto lookupSchema = ROW({{"key", DOUBLE()}, {"payload", BIGINT()}});
  auto probeSchema = ROW({{"key", DOUBLE()}, {"id", BIGINT()}});
  auto outputSchema = ROW({"id", "payload"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<RowVectorPtr> lookupBatches;
  for (int64_t batch = 0; batch < 100; ++batch) {
    const auto offset = batch * kRowsPerBatch;
    lookupBatches.push_back(makeRowVector(
        {"key", "payload"},
        {makeFlatVector<double>(
             kRowsPerBatch, [offset](auto row) { return offset + row + 0.5; }),
         makeFlatVector<int64_t>(
             kRowsPerBatch, [offset](auto row) { return offset + row; })}));
  }

  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::vector<StateDeclarationPtr>{
                          std::make_shared<HashTableStateDeclaration>(
                              "lookup",
                              lookupSchema,
                              std::vector<std::string>{"key"},
                              PlanBuilder(idGenerator)
                                  .values(lookupBatches)
                                  .localPartitionRoundRobin()
                                  .planNode()),
                          std::make_shared<VectorStateDeclaration>(
                              "probe",
                              probeSchema,
                              PlanBuilder(idGenerator)
                                  .values({makeRowVector(
                                      {"key", "id"},
                                      {makeFlatVector<double>({7.5}),
                                       makeFlatVector<int64_t>({1})})})
                                  .planNode()),
                          std::make_shared<VectorStateDeclaration>(
                              "out", outputSchema)},
                      PlanBuilder(idGenerator)
                          .stateSource("probe", probeSchema)
                          .stateHashJoin(
                              "lookup",
                              {"key"},
                              ROW(
                                  {{"key", DOUBLE()},
                                   {"id", BIGINT()},
                                   {"payload", BIGINT()}}))
                          .project({"id", "payload"}),
                      ConvergenceConfig::withMaxIterations(1),
                      /*outputStateEntry=*/"out")
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeFlatVector<int64_t>({7}),
  });
  auto task = AssertQueryBuilder(plan)
                  .maxDrivers(kNumDrivers)
                  .fixedPointOptions(&localOptions())
                  .assertResults(expected);
  auto entry = task->fixedPoint()->state()->getHashTable("lookup");
  ASSERT_TRUE(entry.has_value());
  EXPECT_THAT(entry->table->allRows(), testing::SizeIs(255));
}

// A NULL build key matches nothing, as in a hash join.  The NULL row's key slot
// holds 0, so storing it would let a probe key of 0 match.
TEST_F(FixedPointTest, hashTableStateDropsNullKeys) {
  auto lookupSchema = ROW({"key", "payload"}, BIGINT());
  auto probeSchema = ROW({"key", "tag"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  const std::vector<std::pair<int64_t, int64_t>> expected{{1, 10}};
  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::vector<StateDeclarationPtr>{
                  std::make_shared<HashTableStateDeclaration>(
                      "lookup",
                      lookupSchema,
                      std::vector<std::string>{"key"},
                      PlanBuilder(idGenerator)
                          .values({makeRowVector(
                              lookupSchema->names(),
                              {
                                  makeNullableFlatVector<int64_t>(
                                      {std::nullopt, 1}),
                                  makeFlatVector<int64_t>({99, 10}),
                              })})
                          .planNode()),
                  std::make_shared<VectorStateDeclaration>(
                      "probe",
                      probeSchema,
                      PlanBuilder(idGenerator)
                          .values({makeRowVector(
                              probeSchema->names(),
                              {
                                  makeFlatVector<int64_t>({0, 1}),
                                  makeFlatVector<int64_t>({-1, -1}),
                              })})
                          .planNode())},
              PlanBuilder(idGenerator)
                  .stateSource("probe", probeSchema)
                  .stateHashJoin(
                      "lookup",
                      {"key"},
                      ROW({"key", "tag", "payload"}, BIGINT()))
                  .project({"key", "payload AS tag"}),
              ConvergenceConfig::withMaxIterations(1),
              /*outputStateEntry=*/"probe")
          .planNode(),
      expected);
}

// Strict mode inheritance: a serial fixed point runs every sub-task on the
// calling thread, so it cannot drive a shuffling body (PartitionedOutput /
// Exchange require parallel mode).  The constructor rejects that combination.
TEST_F(FixedPointTest, serialModeRejectsShuffle) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
      });

  // Creating this shuffling body as a serial FixedPointLoop is rejected.
  auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());
  VELOX_ASSERT_THROW(
      exec::Task::create(
          "local://fixedpoint-serial-shuffle",
          core::PlanFragment{
              PlanBuilder(idGenerator)
                  .fixedPoint(
                      {std::make_shared<VectorStateDeclaration>(
                          "frontier",
                          schema,
                          PlanBuilder(idGenerator).values({seed}).planNode())},
                      {PlanBuilder(idGenerator)
                           .stateSource("frontier", schema)
                           .partitionedOutput({}, 1)
                           .planNode(),
                       PlanBuilder(idGenerator)
                           .exchange(schema, "Presto")
                           .planNode()},
                      ConvergenceConfig::withMaxIterations(3),
                      /*outputStateEntry=*/"frontier")
                  .planNode()},
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
// frontier back. Converges when the values reach zero. TaskCursor emits the
// final frontier.
TEST_F(FixedPointTest, shuffleBetweenSubplans) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({8, 5}),
      });

  // 8 -> 4 -> 2 -> 1 -> 0 halvings converge to all-zero values.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 0}, {1, 0}};
  EXPECT_THAT(
      runViaTask(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  {std::make_shared<VectorStateDeclaration>(
                      "frontier",
                      schema,
                      PlanBuilder(idGenerator).values({seed}).planNode())},
                  {// Plan 0: gather the frontier to a single destination
                   // (producer task).
                   PlanBuilder(idGenerator)
                       .stateSource("frontier", schema)
                       .partitionedOutput({}, 1)
                       .planNode(),
                   // Plan 1: receive via exchange, halve values, write back
                   // (consumer task).
                   PlanBuilder(idGenerator)
                       .exchange(schema, "Presto")
                       .project({"id", "val / 2 AS val"})
                       .planNode()},
                  ConvergenceConfig::converging(
                      PlanBuilder(idGenerator)
                          .stateSource("frontier", schema)
                          .singleAggregation({}, {"sum(val)"})
                          .project({"a0 = 0 AS converged"})
                          .planNode(),
                      100),
                  /*outputStateEntry=*/"frontier")
              .planNode(),
          exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({8, 12}),
      });

  // Each iteration divides by four (two halvings):
  //   (8, 12) -> (2, 3) -> (0, 0).
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 0}, {1, 0}};
  EXPECT_THAT(
      runViaTask(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  {std::make_shared<VectorStateDeclaration>(
                      "frontier",
                      schema,
                      PlanBuilder(idGenerator).values({seed}).planNode())},
                  {// Plan 0: gather the frontier (producer).
                   PlanBuilder(idGenerator)
                       .stateSource("frontier", schema)
                       .partitionedOutput({}, 1)
                       .planNode(),
                   // Plan 1: receive, halve, gather again (consumer +
                   // producer).
                   PlanBuilder(idGenerator)
                       .exchange(schema, "Presto")
                       .project({"id", "val / 2 AS val"})
                       .partitionedOutput({}, 1)
                       .planNode(),
                   // Plan 2: receive, halve again, write the frontier back
                   // (consumer).
                   PlanBuilder(idGenerator)
                       .exchange(schema, "Presto")
                       .project({"id", "val / 2 AS val"})
                       .planNode()},
                  ConvergenceConfig::converging(
                      PlanBuilder(idGenerator)
                          .stateSource("frontier", schema)
                          .singleAggregation({}, {"sum(val)"})
                          .project({"a0 = 0 AS converged"})
                          .planNode(),
                      100),
                  /*outputStateEntry=*/"frontier")
              .planNode(),
          exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
}

// A real by-key shuffle across two peer top-level tasks.  The fixed point runs
// as two workers (destinations 0 and 1); the test harness wires
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
          schema->names(),
          {
              makeFlatVector<int64_t>({0, 1}),
              makeFlatVector<int64_t>({1, 10}),
          }),
      makeRowVector(
          schema->names(),
          {
              makeFlatVector<int64_t>({0, 1}),
              makeFlatVector<int64_t>({2, 20}),
          })};

  // Builds worker d's node: the same plans for every worker, but its own
  // pre-partitioned shard as the seed.  Plan 0 partitions this worker's shard
  // by key into two partitions (the partition count, 2, is the worker count);
  // plan 1 reads this worker's partition from the wired peers, sums each key,
  // and writes this worker's shard back.
  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // No convergence plan: the loop runs exactly maxIterations times.
    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                PlanBuilder(idGenerator).values({seeds[worker]}).planNode())},
            {PlanBuilder(idGenerator)
                 .stateSource("vals", schema)
                 .partitionedOutput({"key"}, 2)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(schema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 AS val"})
                 .planNode()},
            ConvergenceConfig::withMaxIterations(3),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // After every iteration the by-key sums are stable: key 0 -> 3, key 1 -> 30.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_THAT(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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
          schema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({8}),
          }),
      makeRowVector(
          schema->names(),
          {
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({8}),
          })};

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                PlanBuilder(idGenerator).values({seeds[worker]}).planNode())},
            {// Plan 0 (produce): partition by key % 2 (stable: each key stays
             // on its worker).  Plan 1 (consume): sum each key and halve it.
             PlanBuilder(idGenerator)
                 .stateSource("vals", schema)
                 .partitionedOutput(
                     {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(schema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 / 2 AS val"})
                 .planNode()},
            ConvergenceConfig::converging(
                // Converge when this worker's shard has decayed to <= 1.
                // Globally consistent here because both keys decay in lockstep.
                // The threshold is 1 (not 0) so the converged result (value 1)
                // differs from the max-iterations result (value 0), proving the
                // loop actually stopped on convergence.
                PlanBuilder(idGenerator)
                    .stateSource("vals", schema)
                    .singleAggregation({}, {"sum(val)"})
                    .project({"a0 <= 1 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // 8 -> 4 -> 2 -> 1: converges at value 1 after exactly three iterations.  The
  // third column is each worker's actual loop-iteration count; both must be 3.
  // Had convergence not fired, the loop would run to maxIterations (100) and
  // decay to 0 -- so value 1 and iteration count 3 (not 0 and 100) confirm the
  // loop stopped on the synchronized convergence verdict.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 1, 3}, {1, 1, 3}};
  EXPECT_THAT(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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

  auto makeWorkerNode = [&](int32_t /*worker*/) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    // Every worker starts from the same replicated row.
    auto seed = makeRowVector(
        schema->names(),
        {
            makeFlatVector<int64_t>({0}),
            makeFlatVector<int64_t>({1}),
        });
    // One row per destination; the cross join below turns a worker's single
    // contribution into one copy addressed to each peer.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "total",
                schema,
                PlanBuilder(idGenerator).values({seed}).planNode())},
            {PlanBuilder(idGenerator)
                 .stateSource("total", schema)
                 .nestedLoopJoin(
                     PlanBuilder(idGenerator).values({destinations}).planNode(),
                     {"destination", "val"})
                 .partitionedOutput(
                     {"destination"},
                     kNumWorkers,
                     /*replicateNullsAndAny=*/false,
                     moduloSpec)
                 .planNode(),
             // Sums every peer's contribution: the same total on every worker.
             PlanBuilder(idGenerator)
                 .exchange(shuffleType, "Presto")
                 .singleAggregation({}, {"sum(val)"})
                 .project({"CAST(0 AS BIGINT) AS key", "a0 AS val"})
                 .planNode()},
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("total", schema)
                    .project({"val >= 8 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"total")
        .planNode();
  };

  // Each iteration sums both workers' equal values, doubling the replicated
  // total: 1 -> 2 -> 4 -> 8, converging after exactly three iterations.  Both
  // workers must report the same value and the same iteration count -- the
  // property that keeps a replicated fixed point in lockstep.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 8, 3}, {0, 8, 3}};
  EXPECT_THAT(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});

    // Worker 0 starts at 0 and worker 1 at 10, so no local value ever equals
    // the global sum and a local-only criterion could not agree across workers.
    auto seed = makeRowVector(
        schema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({worker * 10}),
        });

    return PlanBuilder(idGenerator)
        .fixedPoint(
            std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                PlanBuilder(idGenerator).values({seed}).planNode()),
            PlanBuilder(idGenerator)
                .stateSource("vals", schema)
                .project({"key", "val + 1 AS val"}),
            ConvergenceConfig::converging(
                {// C0: this worker's partial, replicated to every peer by
                 // expansion.
                 PlanBuilder(idGenerator)
                     .stateSource("vals", schema)
                     .singleAggregation({}, {"sum(val)"})
                     .project({"a0 AS partial"})
                     .nestedLoopJoin(
                         PlanBuilder(idGenerator)
                             .values({destinations})
                             .planNode(),
                         {"destination", "partial"})
                     .partitionedOutput(
                         {"destination"},
                         kNumWorkers,
                         /*replicateNullsAndAny=*/false,
                         moduloSpec)
                     .planNode(),
                 // C1: every worker sums every peer's partial and emits the
                 // same verdict.
                 PlanBuilder(idGenerator)
                     .exchange(shuffleType, "Presto")
                     .singleAggregation({}, {"sum(partial)"})
                     .project({"a0 >= 16 AS converged"})
                     .planNode()},
                100),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // Values move 0,10 -> 1,11 -> 2,12 -> 3,13; the global sum reaches 16 on the
  // third iteration.  Both workers stop there, on values (3 and 13) that differ
  // from each other and from the total.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 3, 3}, {1, 13, 3}};
  EXPECT_THAT(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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
          schema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({8}),
          }),
      makeRowVector(
          schema->names(),
          {
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({8}),
          })};

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                PlanBuilder(idGenerator).values({seeds[worker]}).planNode())},
            {// Body: re-partition by key % 2 (each key stays on its worker),
             // then halve.
             PlanBuilder(idGenerator)
                 .stateSource("vals", schema)
                 .partitionedOutput(
                     {"key"},
                     kNumWorkers,
                     /*replicateNullsAndAny=*/false,
                     moduloSpec)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(schema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 / 2 AS val"})
                 .planNode()},
            ConvergenceConfig::converging(
                {// Convergence: reduce the committed state across workers and
                 // stop when the global sum reaches 2 (1 per worker).  No
                 // worker can see that locally.
                 PlanBuilder(idGenerator)
                     .stateSource("vals", schema)
                     .singleAggregation({}, {"sum(val)"})
                     .project({"a0 AS partial"})
                     .nestedLoopJoin(
                         PlanBuilder(idGenerator)
                             .values({destinations})
                             .planNode(),
                         {"destination", "partial"})
                     .partitionedOutput(
                         {"destination"},
                         kNumWorkers,
                         /*replicateNullsAndAny=*/false,
                         moduloSpec)
                     .planNode(),
                 PlanBuilder(idGenerator)
                     .exchange(shuffleType, "Presto")
                     .singleAggregation({}, {"sum(partial)"})
                     .project({"a0 <= 2 AS converged"})
                     .planNode()},
                100),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // 8 -> 4 -> 2 -> 1 per worker; the global sum reaches 2 after three
  // iterations, and both workers stop on the same one.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> expected{
      {0, 1, 3}, {1, 1, 3}};
  EXPECT_THAT(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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

  auto plan =
      PlanBuilder(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  {std::make_shared<VectorStateDeclaration>(
                       "points",
                       pointsType,
                       PlanBuilder(idGenerator).values({pointRows}).planNode()),
                   std::make_shared<VectorStateDeclaration>(
                       "centroids",
                       centroidsType,
                       PlanBuilder(idGenerator)
                           .values({centroidRows})
                           .planNode())},
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
                      // Nearest slot per point; x and y are constant within
                      // a point_id.
                      .singleAggregation(
                          {"point_id"},
                          {"min_by(slot, d2) AS slot",
                           "min(x) AS x",
                           "min(y) AS y"})
                      .singleAggregation(
                          {"slot"},
                          {"sum(x) AS sx", "sum(y) AS sy", "count(1) AS cnt"})
                      // RIGHT join keeps every declared slot: an empty
                      // cluster has a null count, and the coalesce below
                      // then retains its old centroid.
                      .hashJoin(
                          {"slot"},
                          {"old_slot"},
                          // Old centroids, renamed so the RIGHT join below
                          // can carry both sides.
                          PlanBuilder(idGenerator)
                              .stateSource("centroids", centroidsType)
                              .project(
                                  {"slot AS old_slot",
                                   "cx AS old_cx",
                                   "cy AS old_cy"})
                              .planNode(),
                          "",
                          {"old_slot", "old_cx", "old_cy", "sx", "sy", "cnt"},
                          core::JoinType::kRight)
                      // Velox has no divide(DOUBLE, BIGINT); cast the count
                      // once.
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
                           "(sy / n - old_cy) * (sy / n - old_cy), 0.0) AS movement"}),
                  ConvergenceConfig::converging(
                      // Converged once no centroid moved.
                      PlanBuilder(idGenerator)
                          .stateSource("centroids", centroidsType)
                          .singleAggregation({}, {"max(movement)"})
                          .project({"a0 <= 1e-9 AS converged"})
                          .planNode(),
                      20),
                  /*outputStateEntry=*/"centroids")
              .planNode(),
          idGenerator)
          .project(
              {"CAST(round(cx * 100.0) AS BIGINT) AS cx100",
               "CAST(round(cy * 100.0) AS BIGINT) AS cy100"})
          .planNode();

  // Means of the two clusters: (4/3, 4/3) and (25/3, 25/3), scaled by 100; the
  // point-less slot keeps its seed.  errorWhenMaxIterationReached defaults to
  // true, so a loop that never converged would throw rather than reach these.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({133, 833, 10000}),
      makeFlatVector<int64_t>({133, 833, 10000}),
  });
  for (const bool serial : {true, false}) {
    SCOPED_TRACE(serial ? "serial" : "parallel");
    AssertQueryBuilder(plan)
        .serialExecution(serial)
        .fixedPointOptions(&localOptions())
        .assertResults(expected);
  }
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
          pointsType->names(),
          {
              makeFlatVector<int64_t>({0, 1, 2}),
              makeFlatVector<int64_t>({100, 200, 100}),
              makeFlatVector<int64_t>({100, 100, 200}),
          }),
      makeRowVector(
          pointsType->names(),
          {
              makeFlatVector<int64_t>({3, 4, 5}),
              makeFlatVector<int64_t>({800, 900, 800}),
              makeFlatVector<int64_t>({800, 800, 900}),
          })};

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Built per worker: one RowVector shared by concurrently running
    // workers races on the lazy-load memo RowVector::copy() touches.
    // Replicated seeds, identical on every worker; slot 2 again draws no
    // points.
    auto centroidRows = makeRowVector(
        centroidsType->names(),
        {
            makeFlatVector<int64_t>({200, 800, 10000}),
            makeFlatVector<int64_t>({100, 800, 10000}),
            makeFlatVector<int64_t>({0, 1, 2}),
            makeFlatVector<int64_t>({0, 0, 0}),
        });
    auto destinations =
        makeRowVector({"destination"}, {makeFlatVector<int64_t>({0, 1})});

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                 "points",
                 pointsType,
                 PlanBuilder(idGenerator)
                     .values({pointShards[worker]})
                     .planNode()),
             std::make_shared<VectorStateDeclaration>(
                 "centroids",
                 centroidsType,
                 PlanBuilder(idGenerator).values({centroidRows}).planNode())},
            std::vector<core::PlanNodePtr>{
                // Plan 0: assign this worker's points, reduce to per-slot local
                // statistics, then replicate them once per destination so every
                // peer receives them.
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
                        {"min_by(slot, d2) AS slot",
                         "min(x) AS x",
                         "min(y) AS y"})
                    .singleAggregation(
                        {"slot"},
                        {"sum(x) AS sx", "sum(y) AS sy", "count(1) AS cnt"})
                    .nestedLoopJoin(
                        PlanBuilder(idGenerator)
                            .values({destinations})
                            .planNode(),
                        {"destination", "slot", "sx", "sy", "cnt"})
                    .partitionedOutput(
                        {"destination"},
                        kNumWorkers,
                        /*replicateNullsAndAny=*/false,
                        moduloSpec)
                    .planNode(),
                // Plan 1: sum every worker's statistics -- the same totals on
                // each worker -- then install the new centroids, keeping the
                // old one where a slot drew no points anywhere.
                PlanBuilder(idGenerator)
                    .exchange(shuffleType, "Presto")
                    .singleAggregation(
                        {"slot"},
                        {"sum(sx) AS gsx",
                         "sum(sy) AS gsy",
                         "sum(cnt) AS gcnt"})
                    .hashJoin(
                        {"slot"},
                        {"old_slot"},
                        PlanBuilder(idGenerator)
                            .stateSource("centroids", centroidsType)
                            .project(
                                {"slot AS old_slot",
                                 "cx AS old_cx",
                                 "cy AS old_cy"})
                            .planNode(),
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
                    .planNode()},
            ConvergenceConfig::converging(
                // Exact integer test: no centroid moved.
                PlanBuilder(idGenerator)
                    .stateSource("centroids", centroidsType)
                    .singleAggregation({}, {"max(movement)"})
                    .project({"a0 = 0 AS converged"})
                    .planNode(),
                20),
            /*outputStateEntry=*/"centroids")
        .planNode();
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
  EXPECT_THAT(
      runViaTaskWithIterations(
          makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
}

// Initial plan that receives its shard through a shuffle from an UPSTREAM task,
// then iterates while shuffling among peers.  The coordinator launches an
// upstream producer task that emits the full seed and partitions it by key;
// each worker's initial plan is an Exchange that reads its partition (Phase 1
// shuffle, fed by a remote split the coordinator adds).
// The workers then iterate, shuffling by key with each other (Phase 2).  This
// exercises the full path: receive data via shuffle from an upstream task, then
// iterate while shuffling with each other.
TEST_F(FixedPointTest, initialPlanReceivesShuffle) {
  auto schema = ROW({"key", "val"}, BIGINT());

  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({1, 2, 10, 20}),
      });

  // Every worker runs the same node; its shard arrives from the upstream, not a
  // local seed.
  auto makeWorkerNode = [&](int32_t /*worker*/) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // No convergence plan: the loop runs a few iterations of peer shuffle.
    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                // Initial plan: receive this worker's shard via shuffle from
                // the upstream.
                PlanBuilder(idGenerator)
                    .exchange(schema, "Presto")
                    .planNode())},
            {// Body: shuffle by key among peers and sum each key (as in
             // shuffleByKey).
             PlanBuilder(idGenerator)
                 .stateSource("vals", schema)
                 .partitionedOutput({"key"}, 2)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(schema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 AS val"})
                 .planNode()},
            ConvergenceConfig::withMaxIterations(3),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // The upstream delivers the seed; the by-key peer shuffle sums each key:
  // key 0 -> 1+2 = 3, key 1 -> 10+20 = 30, regardless of the upstream's split.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_THAT(
      runViaTask(
          makeWorkerNode,
          // Upstream producer: the full seed, partitioned by key across the
          // workers.
          PlanBuilder().values({seed}).partitionedOutput({"key"}, 2),
          exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
}

// Initial plan whose Exchange reads several upstream tasks.  The coordinator
// adds one remote split per upstream task to the Exchange, as it would outside
// a fixed point.
TEST_F(FixedPointTest, upstreamSplits) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());

  std::vector<exec::TaskPtr> upstreams;
  for (int64_t i = 0; i < 2; ++i) {
    auto part = makeRowVector(
        schema->names(),
        {
            makeFlatVector<int64_t>({i}),
            makeFlatVector<int64_t>({10 * i}),
        });
    upstreams.push_back(startUpstreamTask(
        PlanBuilder().values({part}).partitionedOutput({}, 1), queryCtx));
  }

  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId exchangeId;
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .exchange(schema, "Presto")
                              .capturePlanNodeId(exchangeId)
                              .planNode()),
                      PlanBuilder(idGenerator).stateSource("vals", schema),
                      ConvergenceConfig::withMaxIterations(1))
                  .planNode();

  auto expected = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({0, 10}),
      });

  AssertQueryBuilder(plan)
      .queryCtx(queryCtx)
      .fixedPointOptions(&localOptions())
      .splits(
          exchangeId, {remoteSplit(upstreams[0]), remoteSplit(upstreams[1])})
      .assertResults(expected);
}

// A fixed point whose fragment ends in a PartitionedOutput: consumers read the
// owning task, which finishes once they have consumed its output.
TEST_F(FixedPointTest, partitionedOutputAboveLoop) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({0, 10}),
      });

  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto fixedPoint =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::withMaxIterations(2))
          .planNode();
  auto plan = PlanBuilder(idGenerator)
                  .addNode([&](const core::PlanNodeId& /*id*/,
                               const core::PlanNodePtr& /*input*/) {
                    return fixedPoint;
                  })
                  .partitionedOutput({}, 1)
                  .planNode();

  auto task = makeTask(plan, exec::Task::ExecutionMode::kParallel, "output");
  task->start(/*maxDrivers=*/1);

  auto expected = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({2, 12}),
      });
  AssertQueryBuilder(PlanBuilder().exchange(schema, "Presto").planNode())
      .split(remoteSplit(task))
      .assertResults(expected);

  task->taskCompletionFuture().wait();
  EXPECT_EQ(task->state(), exec::TaskState::kFinished);
}

// A fixed point that does not shuffle runs as several independent tasks, each
// a loop over the partition its initial plan reads for its destination.
TEST_F(FixedPointTest, independentTasks) {
  auto queryCtx = core::QueryCtx::create(
      cpuExecutor_.get(),
      core::QueryConfig{
          {{core::QueryConfig::kRequestDataSizesMaxWaitSec, "1"}}});

  auto schema = ROW({"key", "val"}, BIGINT());
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeFlatVector<int64_t>({0, 10, 20, 30}),
      });

  auto upstream = startUpstreamTask(
      PlanBuilder().values({seed}).partitionedOutput({"key"}, 2), queryCtx);

  // Each loop adds one to 'val' twice.
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId exchangeId;
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .exchange(schema, "Presto")
                              .capturePlanNodeId(exchangeId)
                              .planNode()),
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"key", "val + 1 AS val"}),
                      ConvergenceConfig::withMaxIterations(2))
                  .planNode();

  std::vector<RowVectorPtr> results;
  for (int32_t destination = 0; destination < 2; ++destination) {
    results.push_back(AssertQueryBuilder(plan)
                          .destination(destination)
                          .queryCtx(queryCtx)
                          .fixedPointOptions(&localOptions())
                          .split(exchangeId, remoteSplit(upstream))
                          .copyResults(pool()));
  }

  auto expected = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeFlatVector<int64_t>({2, 12, 22, 32}),
      });
  exec::test::assertEqualResults({expected}, results);
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
          schema->names(),
          {
              makeFlatVector<int64_t>({0, 1}),
              makeFlatVector<int64_t>({1, 10}),
          }));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          schema->names(),
          {
              makeFlatVector<int64_t>({0, 1}),
              makeFlatVector<int64_t>({2, 20}),
          }));

  // Every worker runs the same node; its shard comes from the table, not a
  // local seed.
  core::PlanNodeId scanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "vals",
                schema,
                // Initial plan: read this worker's shard from its assigned file
                // split.
                PlanBuilder(idGenerator)
                    .tableScan(schema)
                    .capturePlanNodeId(scanId)
                    .planNode())},
            {// Body: shuffle by key among peers and sum each key.
             PlanBuilder(idGenerator)
                 .stateSource("vals", schema)
                 .partitionedOutput({"key"}, 2)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(schema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 AS val"})
                 .planNode()},
            ConvergenceConfig::withMaxIterations(3),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // Each worker scans its own file via TableScan; the by-key peer shuffle sums
  // each key: key 0 -> 1+2 = 3, key 1 -> 10+20 = 30.
  auto initSplitsFor = [&](int32_t worker)
      -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
    return {{scanId, hiveSplit(filePaths[worker])}};
  };

  std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}, {1, 30}};
  EXPECT_THAT(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          /*upstreamPlan=*/nullptr,
          initSplitsFor)),
      testing::UnorderedElementsAreArray(expected));
}

// Two initial plans that each scan a table.  Splits are held under the id of
// the node that reads them, so each scan takes only its own file; pooling them
// on the fixed point would feed both files to both scans and seed each state
// with the other's rows.
TEST_F(FixedPointTest, twoScannedInitialPlans) {
  auto valuesSchema = ROW({"key", "val"}, BIGINT());
  auto stepsSchema = ROW({"skey", "delta"}, BIGINT());

  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          valuesSchema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({1}),
          }));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          stepsSchema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({10}),
          }));

  core::PlanNodeId valuesScanId;
  core::PlanNodeId stepsScanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                 "vals",
                 valuesSchema,
                 PlanBuilder(idGenerator)
                     .tableScan(valuesSchema)
                     .capturePlanNodeId(valuesScanId)
                     .planNode()),
             std::make_shared<VectorStateDeclaration>(
                 "steps",
                 stepsSchema,
                 PlanBuilder(idGenerator)
                     .tableScan(stepsSchema)
                     .capturePlanNodeId(stepsScanId)
                     .planNode())},
            // Each iteration adds the per-key step, which only exists if
            // 'steps' was seeded from its own file.
            PlanBuilder(idGenerator)
                .stateSource("vals", valuesSchema)
                .hashJoin(
                    {"key"},
                    {"skey"},
                    PlanBuilder(idGenerator)
                        .stateSource("steps", stepsSchema)
                        .planNode(),
                    "",
                    {"key", "val", "delta"})
                .project({"key", "val + delta AS val"}),
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("vals", valuesSchema)
                    .project({"val >= 21 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  auto initSplitsFor = [&](int32_t /*worker*/)
      -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
    return {
        {valuesScanId, hiveSplit(filePaths[0])},
        {stepsScanId, hiveSplit(filePaths[1])}};
  };

  // val walks 1 -> 11 -> 21 in steps of delta=10, so both states were seeded
  // from the right file.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 21}};
  EXPECT_THAT(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          /*upstreamPlan=*/nullptr,
          initSplitsFor)),
      testing::UnorderedElementsAreArray(expected));
}

// One initial state fed by Exchange, another by TableScan.  Splits are held
// per reading node, so the Exchange-fed plan must ignore the scan's splits:
// a task-wide "this plan takes no connector splits" check would see the scan's
// file and reject a legitimate plan.
TEST_F(FixedPointTest, initialPlanMixesExchangeAndScan) {
  auto valuesSchema = ROW({"key", "val"}, BIGINT());
  auto stepsSchema = ROW({"skey", "delta"}, BIGINT());

  // Each worker scans its own per-key step file.
  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          stepsSchema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({100}),
          }));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector(
          stepsSchema->names(),
          {
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({200}),
          }));

  // Modulo, not hash: key K must land on worker K, the one whose step file
  // holds skey=K, or the join finds no match.
  auto moduloSpec = std::make_shared<ModuloPartitionFunctionSpec>();

  core::PlanNodeId stepsScanId;
  auto makeWorkerNode = [&](int32_t /*worker*/) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    return PlanBuilder(idGenerator)
        .fixedPoint(
            {
                std::make_shared<VectorStateDeclaration>(
                    "vals",
                    valuesSchema,
                    PlanBuilder(idGenerator)
                        .exchange(valuesSchema, "Presto")
                        .planNode()),
                std::make_shared<VectorStateDeclaration>(
                    "steps",
                    stepsSchema,
                    PlanBuilder(idGenerator)
                        .tableScan(stepsSchema)
                        .capturePlanNodeId(stepsScanId)
                        .planNode()),
            },
            {
                PlanBuilder(idGenerator)
                    .stateSource("vals", valuesSchema)
                    .partitionedOutput(
                        {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec)
                    .planNode(),
                PlanBuilder(idGenerator)
                    .exchange(valuesSchema, "Presto")
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
                    .planNode(),
            },
            ConvergenceConfig::withMaxIterations(1),
            /*outputStateEntry=*/"vals")
        .planNode();
  };

  // key 0: 1+2 = 3, plus its step 100 -> 103.  key 1: 10+20 = 30, plus 200.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 103}, {1, 230}};

  // Upstream producer feeding the Exchange-backed state.
  auto seed = makeRowVector(
      valuesSchema->names(),
      {
          makeFlatVector<int64_t>({0, 0, 1, 1}),
          makeFlatVector<int64_t>({1, 2, 10, 20}),
      });
  auto upstreamPlan = PlanBuilder().values({seed}).partitionedOutput(
      {"key"}, 2, /*replicateNullsAndAny=*/false, moduloSpec);

  EXPECT_THAT(
      unionRows(runWorkers(
          makeWorkerNode,
          exec::Task::ExecutionMode::kParallel,
          &upstreamPlan,
          [&](int32_t worker)
              -> std::vector<std::pair<core::PlanNodeId, exec::Split>> {
            return {{stepsScanId, hiveSplit(filePaths[worker])}};
          })),
      testing::UnorderedElementsAreArray(expected));
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

// iterationMaxDrivers lets the body run several drivers, which split the state
// between them: four drivers each count one of the four batches.  The default
// keeps the body single-driver, which counts all four.
TEST_F(FixedPointTest, iterationMaxDriversScalesTheBody) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      core::VectorState("vals", schema)
                          .initial(PlanBuilder(idGenerator)
                                       .values(makeOneRowBatches(4))
                                       .planNode()),
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .partialAggregation({"key"}, {"count(1) AS val"}),
                      ConvergenceConfig::withMaxIterations(1))
                  .planNode();

  const std::vector<std::pair<int64_t, int64_t>> perDriver{
      {0, 1}, {0, 1}, {0, 1}, {0, 1}};
  EXPECT_THAT(
      runWithDrivers(plan, /*maxDrivers=*/4, /*iterationMaxDrivers=*/4),
      testing::UnorderedElementsAreArray(perDriver));
  const std::vector<std::pair<int64_t, int64_t>> singleDriver{{0, 4}};
  EXPECT_THAT(
      runWithDrivers(plan, /*maxDrivers=*/4, /*iterationMaxDrivers=*/1),
      testing::UnorderedElementsAreArray(singleDriver));
}

// The plans of a chain run at the same time and share the driver budget: with
// five drivers the producer gets three and the consumer two.  Each producer
// driver counts the batches it read, so the counts show the split: one driver
// reads batches 0 and 3, the others one batch each.
TEST_F(FixedPointTest, chainSharesIterationDrivers) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  const std::vector<std::pair<int64_t, int64_t>> expected{
      {0, 1}, {0, 1}, {0, 2}};
  EXPECT_THAT(
      runWithDrivers(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  std::vector<StateDeclarationPtr>{
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .values(makeOneRowBatches(4))
                              .planNode())},
                  std::vector<core::PlanNodePtr>{
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .partialAggregation({"key"}, {"count(1) AS val"})
                          .partitionedOutput({}, 1)
                          .planNode(),
                      PlanBuilder(idGenerator)
                          .exchange(schema, "Presto")
                          .planNode()},
                  ConvergenceConfig::withMaxIterations(1),
                  /*outputStateEntry=*/"vals")
              .planNode(),
          /*maxDrivers=*/5,
          /*iterationMaxDrivers=*/5),
      testing::UnorderedElementsAreArray(expected));
}

// The last convergence plan runs single-driver even when the other plans scale,
// so a global aggregation still produces one verdict.
TEST_F(FixedPointTest, convergenceVerdictStaysSingleDriver) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  const std::vector<std::pair<int64_t, int64_t>> expected{
      {0, 0}, {0, 1}, {0, 2}, {0, 3}};
  EXPECT_THAT(
      runWithDrivers(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  core::VectorState("vals", schema)
                      .initial(PlanBuilder(idGenerator)
                                   .values(makeOneRowBatches(4))
                                   .planNode()),
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"key", "val"}),
                  ConvergenceConfig::converging(
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .singleAggregation({}, {"count(1)"})
                          .project({"a0 = 4 AS converged"})
                          .planNode(),
                      /*maxIterations=*/3))
              .planNode(),
          /*maxDrivers=*/4,
          /*iterationMaxDrivers=*/4),
      testing::UnorderedElementsAreArray(expected));
}

// The nodes above the fixed point run in the task's own pipeline, with the
// task's drivers, which split the output state.
TEST_F(FixedPointTest, trailingNodesUseTaskDrivers) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  const std::vector<std::pair<int64_t, int64_t>> perDriver{
      {0, 1}, {0, 1}, {0, 1}, {0, 1}};
  EXPECT_THAT(
      runWithDrivers(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  core::VectorState("vals", schema)
                      .initial(PlanBuilder(idGenerator)
                                   .values(makeOneRowBatches(4))
                                   .planNode()),
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"key", "val"}),
                  ConvergenceConfig::withMaxIterations(1))
              .partialAggregation({"key"}, {"count(1) AS val"})
              .planNode(),
          /*maxDrivers=*/4,
          /*iterationMaxDrivers=*/1),
      testing::UnorderedElementsAreArray(perDriver));
}

// Several drivers probe one hash-table state at once, each with its own
// hashers.
TEST_F(FixedPointTest, stateHashJoinAcrossDrivers) {
  constexpr int64_t kNumBatches{16};
  constexpr int64_t kRowsPerBatch{256};
  constexpr int64_t kNumKeys{kNumBatches * kRowsPerBatch};
  auto probeSchema = ROW({"key", "tag"}, BIGINT());
  auto lookupSchema = ROW({"key", "payload"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto lookup = makeRowVector(
      lookupSchema->names(),
      {
          makeFlatVector<int64_t>(kNumKeys, [](auto row) { return row; }),
          makeFlatVector<int64_t>(kNumKeys, [](auto row) { return row * 10; }),
      });
  std::vector<RowVectorPtr> probeBatches;
  for (int64_t batch = 0; batch < kNumBatches; ++batch) {
    const auto offset = batch * kRowsPerBatch;
    probeBatches.push_back(makeRowVector(
        probeSchema->names(),
        {
            makeFlatVector<int64_t>(
                kRowsPerBatch, [offset](auto row) { return offset + row; }),
            makeFlatVector<int64_t>(
                kRowsPerBatch, [](auto /*row*/) { return -1; }),
        }));
  }

  auto node =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::vector<StateDeclarationPtr>{
                  std::make_shared<HashTableStateDeclaration>(
                      "lookup",
                      lookupSchema,
                      std::vector<std::string>{"key"},
                      PlanBuilder(idGenerator).values({lookup}).planNode()),
                  std::make_shared<VectorStateDeclaration>(
                      "probe",
                      probeSchema,
                      PlanBuilder(idGenerator)
                          .values(probeBatches)
                          .planNode())},
              PlanBuilder(idGenerator)
                  .stateSource("probe", probeSchema)
                  .stateHashJoin(
                      "lookup",
                      {"key"},
                      ROW({"key", "tag", "payload"}, BIGINT()))
                  .project({"key", "payload AS tag"}),
              ConvergenceConfig::withMaxIterations(1),
              /*outputStateEntry=*/"probe")
          .planNode();

  std::vector<std::pair<int64_t, int64_t>> expected;
  expected.reserve(kNumKeys);
  for (int64_t key = 0; key < kNumKeys; ++key) {
    expected.emplace_back(key, key * 10);
  }
  for (uint32_t iterationMaxDrivers : {1, 4}) {
    EXPECT_THAT(
        runWithDrivers(node, /*maxDrivers=*/4, iterationMaxDrivers),
        testing::UnorderedElementsAreArray(expected))
        << iterationMaxDrivers;
  }
}

// A fixed point nested in a plan runs within that plan's share of the
// iteration driver budget. The last convergence plan always gets one driver,
// so a loop nested there runs its own iterations on one driver too.
TEST_F(FixedPointTest, nestedFixedPointRunsWithinItsPlansDrivers) {
  auto outerSchema = ROW({"key", "val"}, BIGINT());
  auto innerSchema = ROW({"ikey", "ival"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Inner fixed point: counts ival up to 3, emitting {(0, 3)}.
  auto innerSeed = makeRowVector(
      innerSchema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });
  core::PlanNodePtr innerFixedPoint =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "inner",
                  innerSchema,
                  PlanBuilder(idGenerator).values({innerSeed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("inner", innerSchema)
                  .project({"ikey", "ival + 1 AS ival"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("inner", innerSchema)
                      .project({"ival >= 3 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"inner")
          .planNode();

  // Outer fixed point: val grows by one per iteration, and the convergence
  // plan joins the inner loop's 3, so the loop stops once val reaches 3.
  auto outerSeed = makeRowVector(
      outerSchema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });
  auto node =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "outer",
                  outerSchema,
                  PlanBuilder(idGenerator).values({outerSeed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("outer", outerSchema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("outer", outerSchema)
                      .hashJoin(
                          {"key"},
                          {"ikey"},
                          innerFixedPoint,
                          "",
                          {"val", "ival"})
                      .project({"val + ival >= 6 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"outer")
          .planNode();

  const std::vector<std::pair<int64_t, int64_t>> expected{{0, 3}};
  for (uint32_t iterationMaxDrivers : {1, 4}) {
    EXPECT_THAT(
        runWithDrivers(node, /*maxDrivers=*/4, iterationMaxDrivers),
        testing::UnorderedElementsAreArray(expected))
        << iterationMaxDrivers;
  }
}

// A StateHashJoin finds its probe keys by name.  Here the key is the second
// probe column, after a column of the same type that must not be joined on.
TEST_F(FixedPointTest, stateHashJoinFindsProbeKeyByName) {
  auto probeSchema = ROW({"tag", "key"}, BIGINT());
  auto lookupSchema = ROW({"key", "payload"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::vector<StateDeclarationPtr>{
                  std::make_shared<HashTableStateDeclaration>(
                      "lookup",
                      lookupSchema,
                      std::vector<std::string>{"key"},
                      PlanBuilder(idGenerator)
                          .values({makeRowVector(
                              lookupSchema->names(),
                              {
                                  makeFlatVector<int64_t>({0, 1, 2}),
                                  makeFlatVector<int64_t>({0, 10, 20}),
                              })})
                          .planNode()),
                  std::make_shared<VectorStateDeclaration>(
                      "probe",
                      probeSchema,
                      PlanBuilder(idGenerator)
                          .values({makeRowVector(
                              probeSchema->names(),
                              {
                                  makeFlatVector<int64_t>({-1, -1}),
                                  makeFlatVector<int64_t>({1, 2}),
                              })})
                          .planNode())},
              PlanBuilder(idGenerator)
                  .stateSource("probe", probeSchema)
                  .stateHashJoin(
                      "lookup",
                      {"key"},
                      ROW({"tag", "key", "payload"}, BIGINT()))
                  .project({"payload AS tag", "key"}),
              ConvergenceConfig::withMaxIterations(1),
              /*outputStateEntry=*/"probe")
          .planNode(),
      {{10, 1}, {20, 2}});
}

TEST_F(FixedPointTest, groupedExecutionIsRejected) {
  auto queryCtx = core::QueryCtx::create(cpuExecutor_.get());
  VELOX_ASSERT_THROW(
      exec::Task::create(
          fmt::format("local://fixedpoint-grouped-{}", queryCtx->queryId()),
          core::PlanFragment{
              countingNode(),
              core::ExecutionStrategy::kGrouped,
              /*numberOfSplitGroups=*/1,
              /*groupedExecLeafNodeIds=*/{}},
          /*destination=*/0,
          queryCtx,
          exec::Task::ExecutionMode::kParallel,
          exec::Consumer{},
          /*memoryArbitrationPriority=*/0,
          /*spillDiskOpts=*/std::nullopt,
          /*onError=*/nullptr,
          &localOptions()),
      "A fixed point does not support grouped execution");
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

// The owning task reports what each phase of the loop ran: the initial plan,
// one entry per iteration with its body and convergence chains, and the
// verdicts.  Timings are not checked; they may be zero on a coarse clock.
TEST_F(FixedPointTest, reportsExecutionStats) {
  const std::vector<bool> expectedVerdicts{false, false, true};
  for (auto mode :
       {exec::Task::ExecutionMode::kSerial,
        exec::Task::ExecutionMode::kParallel}) {
    SCOPED_TRACE(executionModeString(mode));
    auto task = makeTask(countingNode(), mode, "stats");
    if (mode == exec::Task::ExecutionMode::kSerial) {
      while (task->next() != nullptr) {
      }
    } else {
      task->start(/*maxDrivers=*/1);
      task->taskCompletionFuture().wait();
    }
    ASSERT_EQ(task->error(), nullptr);

    auto stats = task->fixedPointExecutionStats();
    ASSERT_TRUE(stats.has_value());
    ASSERT_THAT(stats->initializationPlans, testing::SizeIs(1));
    EXPECT_THAT(
        stats->initializationPlans[0].taskStats.pipelineStats,
        testing::Not(testing::IsEmpty()));
    std::vector<bool> verdicts;
    for (size_t i = 0; i < stats->iterations.size(); ++i) {
      const auto& iteration = stats->iterations[i];
      EXPECT_EQ(iteration.iteration, i);
      EXPECT_EQ(iteration.body.outputRows, 1);
      ASSERT_THAT(iteration.body.plans, testing::SizeIs(1));
      ASSERT_TRUE(iteration.convergence.has_value());
      ASSERT_THAT(iteration.convergence->plans, testing::SizeIs(1));
      EXPECT_EQ(iteration.convergence->plans[0].planIndex, 1);
      verdicts.push_back(iteration.converged);
    }
    EXPECT_EQ(verdicts, expectedVerdicts);
  }

  auto plainTask = makeTask(
      PlanBuilder()
          .values({makeRowVector({"key"}, {makeFlatVector<int64_t>({0})})})
          .planNode(),
      exec::Task::ExecutionMode::kSerial,
      "plain");
  EXPECT_FALSE(plainTask->fixedPointExecutionStats().has_value());
  while (plainTask->next() != nullptr) {
  }
}

// The statistics stay readable after the loop fails, which is when a caller
// needs them most.
TEST_F(FixedPointTest, executionStatsSurviveFailure) {
  auto node = countingNodeTo(/*target=*/100, /*maxIterations=*/2);
  auto task = makeTask(node, exec::Task::ExecutionMode::kParallel, "failed");
  task->start(/*maxDrivers=*/1);
  task->taskCompletionFuture().wait();
  ASSERT_EQ(task->state(), exec::TaskState::kFailed);

  auto stats = task->fixedPointExecutionStats();
  ASSERT_TRUE(stats.has_value());
  ASSERT_THAT(stats->iterations, testing::SizeIs(2));
  EXPECT_FALSE(stats->iterations[0].converged);
  EXPECT_FALSE(stats->iterations[1].converged);
}

// Only the first three iterations and the latest one keep their sub-tasks'
// TaskStats, which bounds memory for long-running loops.
TEST_F(FixedPointTest, executionStatsBoundDetailedIterations) {
  auto node = countingNodeTo(/*target=*/6, /*maxIterations=*/10);
  auto task = makeTask(node, exec::Task::ExecutionMode::kSerial, "bounded");
  while (task->next() != nullptr) {
  }

  auto stats = task->fixedPointExecutionStats();
  ASSERT_TRUE(stats.has_value());
  std::vector<bool> detailed;
  for (const auto& iteration : stats->iterations) {
    detailed.push_back(
        !iteration.body.plans[0].taskStats.pipelineStats.empty() &&
        !iteration.convergence->plans[0].taskStats.pipelineStats.empty());
  }
  const std::vector<bool> expected{true, true, true, false, false, true};
  EXPECT_EQ(detailed, expected);
}

// A chain records each plan's driver limit, which the sub-task's own TaskStats
// confirm.  The nodes above the fixed point run with the task's drivers.
TEST_F(FixedPointTest, executionStatsRecordDrivers) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto fixedPoint = PlanBuilder(idGenerator)
                        .fixedPoint(
                            std::vector<StateDeclarationPtr>{
                                std::make_shared<VectorStateDeclaration>(
                                    "vals",
                                    schema,
                                    PlanBuilder(idGenerator)
                                        .values(makeOneRowBatches(4))
                                        .planNode())},
                            std::vector<core::PlanNodePtr>{
                                PlanBuilder(idGenerator)
                                    .stateSource("vals", schema)
                                    .partitionedOutput({}, 1)
                                    .planNode(),
                                PlanBuilder(idGenerator)
                                    .exchange(schema, "Presto")
                                    .planNode()},
                            ConvergenceConfig::withMaxIterations(1),
                            /*outputStateEntry=*/"vals")
                        .planNode();

  auto options = localOptions();
  options.iterationMaxDrivers = 5;
  auto cursor = TaskCursor::create({
      .planNode = PlanBuilder(idGenerator)
                      .addNode([&](const core::PlanNodeId& /*id*/,
                                   const core::PlanNodePtr& /*input*/) {
                        return fixedPoint;
                      })
                      .project({"key", "val"})
                      .planNode(),
      .maxDrivers = 5,
      .queryCtx = core::QueryCtx::create(cpuExecutor_.get()),
      .copyResult = false,
      .fixedPointOptions = &options,
  });
  cursor->task()->addSplit(fixedPoint->id(), remoteSplit(cursor->task()));
  cursor->task()->noMoreSplits(fixedPoint->id());
  EXPECT_THAT(readRows(*cursor), testing::SizeIs(4));

  auto stats = cursor->task()->fixedPointExecutionStats();
  ASSERT_TRUE(stats.has_value());
  ASSERT_THAT(stats->iterations, testing::SizeIs(1));
  const auto& plans = stats->iterations[0].body.plans;
  ASSERT_THAT(plans, testing::SizeIs(2));
  EXPECT_EQ(plans[0].maxDrivers, 3);
  EXPECT_EQ(plans[0].taskStats.numTotalDrivers, 3);
  EXPECT_EQ(plans[1].maxDrivers, 2);
  EXPECT_EQ(plans[1].taskStats.numTotalDrivers, 2);
  // The Project above the fixed point runs in the task's own pipeline.
  EXPECT_EQ(cursor->task()->taskStats().numTotalDrivers, 5);
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto task = makeTask(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              // Never reaches 100 within 2 iterations, so the loop fails the
              // query.
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"val >= 100 AS converged"})
                      .planNode(),
                  2),
              /*outputStateEntry=*/"vals")
          .planNode(),
      exec::Task::ExecutionMode::kSerial,
      "loop-failure");
  VELOX_ASSERT_THROW(task->next(), "did not converge within");
  EXPECT_EQ(task->state(), exec::TaskState::kFailed);
  EXPECT_NE(task->error(), nullptr);
  // The state is half-written, so a retried next() must not re-seed it and
  // rebuild hash tables on top of a half-finished run.
  VELOX_ASSERT_THROW(task->next(), "cannot resume");
}

// A driver the test holds on its thread while another driver fails. Shared
// with the injection callback, which can outlive the test body when the loop
// stops the owning task too early.
struct HeldDriver {
  folly::Baton<> onThread;
  folly::Baton<> done;
  std::atomic_bool ownerStoppedFirst{false};
};

// Waits up to five seconds for 'baton' to be posted.
void waitFor(folly::Baton<>& baton) {
  if (!baton.try_wait_for(std::chrono::seconds(5))) {
    ADD_FAILURE() << "Timed out waiting for a held driver";
  }
}

// Holds the calling driver on its thread until 'owner' stops running or a
// second passes, and records whether 'owner' stopped first.
void holdUntilStopped(exec::Task& owner, HeldDriver& held) {
  held.onThread.post();
  owner.taskCompletionFuture().wait(std::chrono::seconds(1));
  held.ownerStoppedFirst = !owner.isRunning();
  held.done.post();
}

// A failed sub-task resolves its completion future while its other drivers can
// still be on thread, feeding a consumer that writes into the loop's buffers.
// The loop must wait for those drivers before it fails the owning task. Driver
// 0 of the initial plan fails once driver 1 is on thread, and driver 1 records
// whether the owning task stopped while it was there.
DEBUG_ONLY_TEST_F(FixedPointTest, failedInitialPlanStopsDriversFirst) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto task = makeTask(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator)
                      .values({seed}, /*parallelizable=*/true)
                      .planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::withMaxIterations(1),
              /*outputStateEntry=*/"vals")
          .planNode(),
      exec::Task::ExecutionMode::kParallel,
      "stop-initial");
  auto held = std::make_shared<HeldDriver>();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::getOutput",
      std::function<void(Operator*)>([task, held](Operator* op) {
        if (op->operatorType() != "Values") {
          return;
        }
        if (op->operatorCtx()->driverCtx()->driverId == 0) {
          waitFor(held->onThread);
          // Lets the loop start waiting on the sub-task before it fails.
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          VELOX_FAIL("Injected initial-plan failure");
        }
        holdUntilStopped(*task, *held);
        VELOX_FAIL("Injected failure of the held driver");
      }));
  task->start(/*maxDrivers=*/2);
  task->taskCompletionFuture().wait();
  waitFor(held->done);

  ASSERT_TRUE(held->done.ready());
  EXPECT_FALSE(held->ownerStoppedFirst);
  VELOX_ASSERT_THROW(
      std::rethrow_exception(task->error()), "Injected initial-plan failure");
}

// The same contract for a body chain: the producer plan fails while the
// consumer plan's driver is on thread with the batch the producer sent first.
DEBUG_ONLY_TEST_F(FixedPointTest, failedBodyPlanStopsChainFirst) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto batch = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({1}),
      });
  auto node =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::vector<
                  StateDeclarationPtr>{std::make_shared<VectorStateDeclaration>(
                  "frontier",
                  schema,
                  PlanBuilder(idGenerator).values({batch, batch}).planNode())},
              std::vector<core::PlanNodePtr>{
                  PlanBuilder(idGenerator)
                      .stateSource("frontier", schema)
                      .partitionedOutput({}, 1)
                      .planNode(),
                  PlanBuilder(idGenerator)
                      .exchange(schema, "Presto")
                      .project({"id", "val"})
                      .planNode()},
              ConvergenceConfig::withMaxIterations(1),
              /*outputStateEntry=*/"frontier")
          .planNode();

  // Eager flush sends the producer's first batch before it reads the second.
  auto queryCtx = core::QueryCtx::create(
      cpuExecutor_.get(),
      core::QueryConfig{
          {{core::QueryConfig::kPartitionedOutputEagerFlush, "true"}}});
  auto task = exec::Task::create(
      fmt::format("local://fixedpoint-stop-body-{}", queryCtx->queryId()),
      core::PlanFragment{node},
      /*destination=*/0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel,
      exec::Consumer{},
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/std::nullopt,
      /*onError=*/nullptr,
      &localOptions());
  task->addSplit(node->id(), remoteSplit(task));
  task->noMoreSplits(node->id());

  auto held = std::make_shared<HeldDriver>();
  auto numProducerBatches = std::make_shared<std::atomic_int>(0);
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::getOutput",
      std::function<void(Operator*)>(
          [task, held, numProducerBatches](Operator* op) {
            if (op->operatorType() == "FixedPointStateSource" &&
                ++*numProducerBatches == 2) {
              waitFor(held->onThread);
              VELOX_FAIL("Injected producer failure");
            }
            if (op->operatorType() == "Exchange") {
              holdUntilStopped(*task, *held);
              VELOX_FAIL("Injected failure of the held consumer");
            }
          }));
  task->start(/*maxDrivers=*/1);
  task->taskCompletionFuture().wait();
  waitFor(held->done);

  ASSERT_TRUE(held->done.ready());
  EXPECT_FALSE(held->ownerStoppedFirst);
  VELOX_ASSERT_THROW(
      std::rethrow_exception(task->error()), "Injected producer failure");
}

// A split addressed to a node that reads none is a coordinator bug, rejected as
// in any task rather than filed where nothing reads it.
TEST_F(FixedPointTest, addSplitRejectsPlanNodeThatTakesNoSplits) {
  auto node = countingNode();
  auto task = makeTask(node, exec::Task::ExecutionMode::kSerial, "bad-split");
  VELOX_ASSERT_THROW(
      task->addSplit(
          "no-such-node",
          exec::Split(std::make_shared<exec::RemoteConnectorSplit>("peer"))),
      "Splits can be associated only with leaf plan nodes which require "
      "splits");
  task->requestCancel().wait();
}

// The loop feeds splits only to its initial plans and to the Exchange linking a
// plan to its predecessor, so a body or convergence plan that reads a table is
// rejected rather than left waiting for splits.
TEST_F(FixedPointTest, stepTableScan) {
  auto schema = ROW("x", BIGINT());
  auto tableSchema = ROW("y", BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto declaration = std::make_shared<VectorStateDeclaration>(
      "n", schema, /*initialPlan=*/nullptr, /*append=*/true);
  const auto noConvergence = ConvergenceConfig::withMaxIterations(3);

  // A body that joins a table.
  {
    auto scan = PlanBuilder(idGenerator).tableScan(tableSchema).planNode();
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(idGenerator)
            .fixedPoint(
                declaration,
                PlanBuilder(idGenerator)
                    .stateSource("n", schema)
                    .hashJoin({"x"}, {"y"}, scan, "", {"x"}),
                noConvergence),
        fmt::format(
            "FixedPointNode: only an initial plan can read splits; other "
            "plans must read their rows from state: plan 0, TableScan {}",
            scan->id()));
  }

  // A convergence plan that joins a table.
  {
    auto scan = PlanBuilder(idGenerator).tableScan(tableSchema).planNode();
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(idGenerator)
            .fixedPoint(
                declaration,
                PlanBuilder(idGenerator).stateSource("n", schema),
                ConvergenceConfig::converging(
                    PlanBuilder(idGenerator)
                        .stateSource("n", schema, /*delta=*/false)
                        .hashJoin({"x"}, {"y"}, scan, "", {"x"})
                        .singleAggregation({}, {"count(1)"})
                        .project({"a0 = 0"})
                        .planNode(),
                    3)),
        fmt::format(
            "FixedPointNode: only an initial plan can read splits; other "
            "plans must read their rows from state: convergence plan 0, "
            "TableScan {}",
            scan->id()));
  }

  // An index lookup reads splits for its lookup side only if the connector
  // asks for them.
  auto lookupTable = [&](bool needsIndexSplit) {
    auto lookup = std::make_shared<core::TableScanNode>(
        idGenerator->next(),
        tableSchema,
        std::make_shared<test::TestIndexTableHandle>(
            "test-index",
            /*indexTable=*/nullptr,
            /*asyncLookup=*/false,
            needsIndexSplit),
        connector::ColumnHandleMap{});
    return PlanBuilder(idGenerator)
        .stateSource("n", schema)
        .indexLookupJoin(
            {"x"},
            {"y"},
            lookup,
            /*joinConditions=*/{},
            /*filter=*/"",
            /*hasMarker=*/false,
            {"x"});
  };

  {
    ASSERT_NO_THROW(PlanBuilder(idGenerator)
                        .fixedPoint(
                            declaration,
                            lookupTable(/*needsIndexSplit=*/false),
                            noConvergence));
  }

  {
    auto body = lookupTable(/*needsIndexSplit=*/true);
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(idGenerator).fixedPoint(declaration, body, noConvergence),
        fmt::format(
            "FixedPointNode: only an initial plan can read splits; other "
            "plans must read their rows from state: plan 0, IndexLookupJoin {}",
            body.planNode()->id()));
  }

  // A loop nested in the body gets no splits for its initial plans.
  {
    auto scan = PlanBuilder(idGenerator).tableScan(tableSchema).planNode();
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(idGenerator)
            .fixedPoint(
                declaration,
                PlanBuilder(idGenerator)
                    .stateSource("n", schema)
                    .hashJoin(
                        {"x"},
                        {"y"},
                        PlanBuilder(idGenerator)
                            .fixedPoint(
                                std::make_shared<VectorStateDeclaration>(
                                    "m", tableSchema, scan, /*append=*/true),
                                PlanBuilder(idGenerator)
                                    .stateSource("m", tableSchema),
                                noConvergence)
                            .planNode(),
                        "",
                        {"x"}),
                noConvergence),
        fmt::format(
            "FixedPointNode: only an initial plan can read splits; other "
            "plans must read their rows from state: plan 0, TableScan {}",
            scan->id()));
  }
}

// A NULL verdict is a malformed criterion, not a "no".  Read as "not
// converged" it would burn the whole iteration budget and then surface as a
// non-convergence failure, pointing at the wrong thing.
TEST_F(FixedPointTest, convergenceVerdictMustNotBeNull) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  VELOX_ASSERT_THROW(
      runViaTask(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  std::make_shared<VectorStateDeclaration>(
                      "vals",
                      schema,
                      PlanBuilder(idGenerator).values({seed}).planNode()),
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"key", "val + 1 AS val"}),
                  ConvergenceConfig::converging(
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"CAST(NULL AS BOOLEAN) AS converged"})
                          .planNode(),
                      10),
                  /*outputStateEntry=*/"vals")
              .planNode(),
          exec::Task::ExecutionMode::kSerial),
      "produced a NULL verdict");
}

// A plan with trailing (non-fixed-point) nodes above the FixedPointNode: a
// Project over a counting fixed point.  The task runs the Project in its own
// pipeline over the loop's output state.
TEST_F(FixedPointTest, trailingNodes) {
  auto schema = ROW({"key", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto fixedPoint =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"val >= 3 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"vals")
          .planNode();

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
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0}),
      makeFlatVector<int64_t>({30}),
  });
  for (const bool serial : {true, false}) {
    SCOPED_TRACE(serial ? "serial" : "parallel");
    AssertQueryBuilder(plan)
        .serialExecution(serial)
        .fixedPointOptions(&localOptions())
        .assertResults(expected);
  }
}

// Nodes above the fixed point that read a table: the task's own pipeline reads
// the scan's splits.
TEST_F(FixedPointTest, trailingTableScan) {
  auto schema = ROW({"k", "v"}, BIGINT());

  auto filePaths = makeFilePaths(1);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          {"tk", "w"},
          {makeFlatVector<int64_t>({0, 1}), makeFlatVector<int64_t>({7, 8})}));

  core::PlanNodeId scanId;
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .values({makeRowVector(
                                  schema->names(),
                                  {
                                      makeFlatVector<int64_t>({0}),
                                      makeFlatVector<int64_t>({0}),
                                  })})
                              .planNode(),
                          /*append=*/true),
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"k", "v + 1 AS v"})
                          .filter("v < 3"),
                      ConvergenceConfig::whenDeltaEmpty(10))
                  .hashJoin(
                      {"k"},
                      {"tk"},
                      PlanBuilder(idGenerator)
                          .tableScan(ROW({"tk", "w"}, BIGINT()))
                          .capturePlanNodeId(scanId)
                          .planNode(),
                      "",
                      {"w", "v"})
                  .planNode();

  // The loop outputs the seed (0, 0) and one row per iteration, (0, 1) and
  // (0, 2). The table maps k = 0 to w = 7.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({7, 7, 7}),
      makeFlatVector<int64_t>({0, 1, 2}),
  });

  for (const bool serial : {true, false}) {
    SCOPED_TRACE(serial ? "serial" : "parallel");
    AssertQueryBuilder(plan)
        .serialExecution(serial)
        .fixedPointOptions(&localOptions())
        .split(scanId, hiveSplit(filePaths[0]))
        .assertResults(expected);
  }
}

// A fixed point whose body and convergence plans are one plan each shuffles
// nothing, so it runs in parallel without FixedPointOptions::producerLocation.
TEST_F(FixedPointTest, parallelWithoutProducerLocation) {
  FixedPointOptions options = localOptions();
  options.producerLocation = nullptr;

  // 0 -> 1 -> 2 -> 3: converges at val 3.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0}),
      makeFlatVector<int64_t>({3}),
  });
  AssertQueryBuilder(countingNode())
      .fixedPointOptions(&options)
      .assertResults(expected);
}

// A split sent again with the sequence id it had is ignored, as for any scan,
// when the scan is in a fixed point's initial plan.
TEST_F(FixedPointTest, resentInitialPlanSplit) {
  auto schema = ROW({"k", "v"}, BIGINT());
  auto data = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({7, 8}),
      });
  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), data);

  core::PlanNodeId scanId;
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .tableScan(schema)
                              .capturePlanNodeId(scanId)
                              .planNode()),
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"k", "v"}),
                      ConvergenceConfig::withMaxIterations(1))
                  .planNode();

  auto task = makeTask(plan, exec::Task::ExecutionMode::kSerial, "resent");
  EXPECT_TRUE(task->addSplitWithSequence(
      scanId, hiveSplit(filePaths[0]), /*sequenceId=*/1));
  task->setMaxSplitSequenceId(scanId, /*maxSequenceId=*/1);
  EXPECT_FALSE(task->addSplitWithSequence(
      scanId, hiveSplit(filePaths[0]), /*sequenceId=*/1));
  task->noMoreSplits(scanId);

  std::vector<RowVectorPtr> results;
  while (auto batch = task->next()) {
    results.push_back(batch);
  }
  test::assertEqualResults({data}, results);
}

// A fixed point whose initial plan joins another fixed point with a table, and
// the inner fixed point reads a table too: the outer loop passes both scans'
// splits to the task running the inner one.
TEST_F(FixedPointTest, nestedInitialPlanTableScan) {
  auto schema = ROW({"k", "v"}, BIGINT());

  auto filePaths = makeFilePaths(2);
  writeToFile(
      filePaths[0]->getPath(),
      makeRowVector(
          schema->names(),
          {
              makeFlatVector<int64_t>({0}),
              makeFlatVector<int64_t>({0}),
          }));
  writeToFile(
      filePaths[1]->getPath(),
      makeRowVector({"tk"}, {makeFlatVector<int64_t>({0})}));

  core::PlanNodeId scanId;
  core::PlanNodeId joinScanId;
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "outer",
                          schema,
                          PlanBuilder(idGenerator)
                              .fixedPoint(
                                  std::make_shared<VectorStateDeclaration>(
                                      "inner",
                                      schema,
                                      PlanBuilder(idGenerator)
                                          .tableScan(schema)
                                          .capturePlanNodeId(scanId)
                                          .planNode(),
                                      /*append=*/true),
                                  PlanBuilder(idGenerator)
                                      .stateSource("inner", schema)
                                      .project({"k", "v + 1 AS v"})
                                      .filter("v < 3"),
                                  ConvergenceConfig::whenDeltaEmpty(10))
                              .hashJoin(
                                  {"k"},
                                  {"tk"},
                                  PlanBuilder(idGenerator)
                                      .tableScan(ROW("tk", BIGINT()))
                                      .capturePlanNodeId(joinScanId)
                                      .planNode(),
                                  "",
                                  {"k", "v"})
                              .planNode(),
                          /*append=*/true),
                      PlanBuilder(idGenerator)
                          .stateSource("outer", schema)
                          .project({"k", "v + 10 AS v"})
                          .filter("v < 20"),
                      ConvergenceConfig::whenDeltaEmpty(10))
                  .planNode();

  // The inner loop counts the table's v = 0 up to 2; the outer loop adds 10 to
  // each once.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 0, 0, 0}),
      makeFlatVector<int64_t>({0, 1, 2, 10, 11, 12}),
  });

  for (const bool serial : {true, false}) {
    SCOPED_TRACE(serial ? "serial" : "parallel");
    AssertQueryBuilder(plan)
        .serialExecution(serial)
        .fixedPointOptions(&localOptions())
        .split(scanId, hiveSplit(filePaths[0]))
        .split(joinScanId, hiveSplit(filePaths[1]))
        .assertResults(expected);
  }
}

// An initial plan over a table with no splits seeds an empty state. A serial
// loop runs only once each source's splits are complete.
TEST_F(FixedPointTest, initialPlanSplitsComplete) {
  auto schema = ROW({"k", "v"}, BIGINT());

  core::PlanNodeId scanId;
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator)
                              .tableScan(schema)
                              .capturePlanNodeId(scanId)
                              .planNode()),
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"k", "v + 1 AS v"})
                          .filter("v < 3"),
                      ConvergenceConfig::whenDeltaEmpty(10))
                  .planNode();

  for (const bool serial : {true, false}) {
    SCOPED_TRACE(serial ? "serial" : "parallel");
    AssertQueryBuilder(plan)
        .serialExecution(serial)
        .fixedPointOptions(&localOptions())
        .splits(scanId, std::vector<exec::Split>{})
        .assertEmptyResults();
  }

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .copyResult = false,
      .serialExecution = true,
      .fixedPointOptions = &localOptions(),
  });
  VELOX_ASSERT_USER_THROW(
      readRows(*cursor),
      fmt::format(
          "A serial fixed point runs only once each source's splits are "
          "complete; call noMoreSplits before next(): {}",
          scanId));
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
      innerSchema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  core::PlanNodePtr innerFixedPoint =
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "inner",
                  innerSchema,
                  PlanBuilder(idGenerator).values({innerSeed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("inner", innerSchema)
                  .project({"ikey", "ival + 1 AS ival"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("inner", innerSchema)
                      .project({"ival >= 3 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"inner")
          .planNode();

  // Outer fixed point: each iteration adds the inner loop's result to val,
  // reaching 3 then 6.
  auto outerSeed = makeRowVector(
      outerSchema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  expectBothModes(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "outer",
                  outerSchema,
                  PlanBuilder(idGenerator).values({outerSeed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("outer", outerSchema)
                  .hashJoin(
                      {"key"},
                      {"ikey"},
                      innerFixedPoint,
                      "",
                      {"key", "val", "ival"})
                  .project({"key", "val + ival AS val"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("outer", outerSchema)
                      .project({"val >= 6 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"outer")
          .planNode(),
      {{0, 6}});
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

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey each iteration, counting ival up to 3.
    auto innerSeed = makeRowVector(
        innerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    core::PlanNodePtr innerFixedPoint =
        PlanBuilder(idGenerator)
            .fixedPoint(
                {std::make_shared<VectorStateDeclaration>(
                    "inner",
                    innerSchema,
                    PlanBuilder(idGenerator).values({innerSeed}).planNode())},
                {PlanBuilder(idGenerator)
                     .stateSource("inner", innerSchema)
                     .partitionedOutput(
                         {"ikey"},
                         kNumWorkers,
                         /*replicateNullsAndAny=*/false,
                         moduloSpec)
                     .planNode(),
                 PlanBuilder(idGenerator)
                     .exchange(innerSchema, "Presto")
                     .singleAggregation({"ikey"}, {"sum(ival)"})
                     .project({"ikey", "a0 + 1 AS ival"})
                     .planNode()},
                ConvergenceConfig::converging(
                    PlanBuilder(idGenerator)
                        .stateSource("inner", innerSchema)
                        .project({"ival >= 3 AS converged"})
                        .planNode(),
                    100),
                /*outputStateEntry=*/"inner")
            .planNode();

    // Outer loop: local, adds the inner result to its own value each round.
    auto outerSeed = makeRowVector(
        outerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    return PlanBuilder(idGenerator)
        .fixedPoint(
            std::make_shared<VectorStateDeclaration>(
                "outer",
                outerSchema,
                PlanBuilder(idGenerator).values({outerSeed}).planNode()),
            PlanBuilder(idGenerator)
                .stateSource("outer", outerSchema)
                .hashJoin(
                    {"key"},
                    {"ikey"},
                    innerFixedPoint,
                    "",
                    {"key", "val", "ival"})
                .project({"key", "val + ival AS val"}),
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("outer", outerSchema)
                    .project({"val >= 6 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"outer")
        .planNode();
  };

  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_THAT(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey, counting ival up to 3.
    auto innerSeed = makeRowVector(
        innerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    core::PlanNodePtr innerFixedPoint =
        PlanBuilder(idGenerator)
            .fixedPoint(
                {std::make_shared<VectorStateDeclaration>(
                    "inner",
                    innerSchema,
                    PlanBuilder(idGenerator).values({innerSeed}).planNode())},
                {PlanBuilder(idGenerator)
                     .stateSource("inner", innerSchema)
                     .partitionedOutput(
                         {"ikey"},
                         kNumWorkers,
                         /*replicateNullsAndAny=*/false,
                         moduloSpec)
                     .planNode(),
                 PlanBuilder(idGenerator)
                     .exchange(innerSchema, "Presto")
                     .singleAggregation({"ikey"}, {"sum(ival)"})
                     .project({"ikey", "a0 + 1 AS ival"})
                     .planNode()},
                ConvergenceConfig::converging(
                    PlanBuilder(idGenerator)
                        .stateSource("inner", innerSchema)
                        .project({"ival >= 3 AS converged"})
                        .planNode(),
                    100),
                /*outputStateEntry=*/"inner")
            .planNode();

    // Outer loop: adds the inner loop's result to its own value, then shuffles
    // by key (each key stays on its worker).  The nest sits in the *producing*
    // plan, so that plan's sub-task runs a FixedPointLoop, and its
    // PartitionedOutput runs in that sub-task's own pipeline, under the
    // producer id peers read.
    auto outerSeed = makeRowVector(
        outerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "outer",
                outerSchema,
                PlanBuilder(idGenerator).values({outerSeed}).planNode())},
            {PlanBuilder(idGenerator)
                 .stateSource("outer", outerSchema)
                 .hashJoin(
                     {"key"},
                     {"ikey"},
                     innerFixedPoint,
                     "",
                     {"key", "val", "ival"})
                 .project({"key", "val + ival AS val"})
                 .partitionedOutput(
                     {"key"},
                     kNumWorkers,
                     /*replicateNullsAndAny=*/false,
                     moduloSpec)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(outerSchema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 AS val"})
                 .planNode()},
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("outer", outerSchema)
                    .project({"val >= 6 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"outer")
        .planNode();
  };

  // The inner loop yields 3 every outer round, so each worker goes 0 -> 3 -> 6
  // and stops on the second.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_THAT(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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

  auto makeWorkerNode = [&](int32_t worker) -> core::PlanNodePtr {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Inner loop: shuffles by ikey, counting ival up to 3.
    auto innerSeed = makeRowVector(
        innerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    core::PlanNodePtr innerFixedPoint =
        PlanBuilder(idGenerator)
            .fixedPoint(
                {std::make_shared<VectorStateDeclaration>(
                    "inner",
                    innerSchema,
                    PlanBuilder(idGenerator).values({innerSeed}).planNode())},
                {PlanBuilder(idGenerator)
                     .stateSource("inner", innerSchema)
                     .partitionedOutput(
                         {"ikey"},
                         kNumWorkers,
                         /*replicateNullsAndAny=*/false,
                         moduloSpec)
                     .planNode(),
                 PlanBuilder(idGenerator)
                     .exchange(innerSchema, "Presto")
                     .singleAggregation({"ikey"}, {"sum(ival)"})
                     .project({"ikey", "a0 + 1 AS ival"})
                     .planNode()},
                ConvergenceConfig::converging(
                    PlanBuilder(idGenerator)
                        .stateSource("inner", innerSchema)
                        .project({"ival >= 3 AS converged"})
                        .planNode(),
                    100),
                /*outputStateEntry=*/"inner")
            .planNode();

    // Outer loop: shuffles by key first, then adds the inner loop's result in
    // the consuming plan.
    auto outerSeed = makeRowVector(
        outerSchema->names(),
        {
            makeFlatVector<int64_t>({worker}),
            makeFlatVector<int64_t>({0}),
        });
    return PlanBuilder(idGenerator)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>(
                "outer",
                outerSchema,
                PlanBuilder(idGenerator).values({outerSeed}).planNode())},
            {PlanBuilder(idGenerator)
                 .stateSource("outer", outerSchema)
                 .partitionedOutput(
                     {"key"},
                     kNumWorkers,
                     /*replicateNullsAndAny=*/false,
                     moduloSpec)
                 .planNode(),
             PlanBuilder(idGenerator)
                 .exchange(outerSchema, "Presto")
                 .singleAggregation({"key"}, {"sum(val)"})
                 .project({"key", "a0 AS val"})
                 .hashJoin(
                     {"key"},
                     {"ikey"},
                     innerFixedPoint,
                     "",
                     {"key", "val", "ival"})
                 .project({"key", "val + ival AS val"})
                 .planNode()},
            ConvergenceConfig::converging(
                PlanBuilder(idGenerator)
                    .stateSource("outer", outerSchema)
                    .project({"val >= 6 AS converged"})
                    .planNode(),
                100),
            /*outputStateEntry=*/"outer")
        .planNode();
  };

  // The inner loop yields 3 every outer round, so each worker goes 0 -> 3 -> 6
  // and stops on the second.
  std::vector<std::pair<int64_t, int64_t>> expected{{0, 6}, {1, 6}};
  EXPECT_THAT(
      runViaTask(makeWorkerNode, exec::Task::ExecutionMode::kParallel),
      testing::UnorderedElementsAreArray(expected));
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto makeNode = [&](std::vector<core::PlanNodePtr> plans) {
    return PlanBuilder(idGen)
        .fixedPoint(
            {std::make_shared<VectorStateDeclaration>("s", schema)},
            std::move(plans),
            ConvergenceConfig::withMaxIterations(10),
            /*outputStateEntry=*/"s")
        .planNode();
  };

  // Valid: a single plan reading the state and projecting the rows written
  // back (the framework performs the write).
  {
    EXPECT_NO_THROW(makeNode({PlanBuilder(idGen)
                                  .stateSource("s", schema)
                                  .project({"id", "val"})
                                  .planNode()}));
  }

  // At least one plan is required.
  VELOX_ASSERT_THROW(makeNode({}), "requires at least one plan");

  // outputStateEntry must name a declared vector entry (here it names none).
  {
    VELOX_ASSERT_THROW(
        PlanBuilder(idGen)
            .fixedPoint(
                std::make_shared<VectorStateDeclaration>("s", schema),
                PlanBuilder(idGen)
                    .stateSource("s", schema)
                    .project({"id", "val"}),
                ConvergenceConfig::withMaxIterations(10),
                /*outputStateEntry=*/"missing")
            .planNode(),
        "outputStateEntry must name a declared vector state entry");
  }

  // First plan must start with StateSource (here it starts with Values).
  {
    VELOX_ASSERT_THROW(
        makeNode({PlanBuilder(idGen).values({seed}).planNode()}),
        "must start with a StateSourceNode");
  }

  // Last plan must not end with PartitionedOutput (it produces the rows written
  // back, it does not shuffle).
  {
    VELOX_ASSERT_THROW(
        makeNode({PlanBuilder(idGen)
                      .stateSource("s", schema)
                      .partitionedOutput({}, 1)
                      .planNode()}),
        "not shuffle through a PartitionedOutput");
  }

  // A non-last plan must end with PartitionedOutput: here plan 0 ends with a
  // Project (state-chaining, not a shuffle) -- the bug the validation forbids.
  {
    VELOX_ASSERT_THROW(
        makeNode(
            {PlanBuilder(idGen)
                 .stateSource("s", schema)
                 .project({"id", "val"})
                 .planNode(),
             PlanBuilder(idGen).exchange(schema, "Presto").planNode()}),
        "must end with a PartitionedOutput");
  }

  // A non-first plan must start with Exchange: here plan 1 starts with a
  // StateSource instead of receiving plan 0's shuffle.
  {
    VELOX_ASSERT_THROW(
        makeNode(
            {PlanBuilder(idGen)
                 .stateSource("s", schema)
                 .partitionedOutput({}, 1)
                 .planNode(),
             PlanBuilder(idGen)
                 .stateSource("s", schema)
                 .project({"id", "val"})
                 .planNode()}),
        "must start with an Exchange");
  }
}

// errorWhenMaxIterationReached (default true) requires a convergence plan: a
// null plan never converges, so combining it with the default flag would always
// fail -- the FixedPointNode constructor rejects that combination up front.
TEST_F(FixedPointTest, errorWhenMaxIterationReachedRequiresPlan) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> plans{PlanBuilder(idGen)
                                           .stateSource("s", schema)
                                           .project({"id", "val"})
                                           .planNode()};
  // Null convergence plan + the default errorWhenMaxIterationReached=true.
  VELOX_ASSERT_THROW(
      PlanBuilder(idGen)
          .fixedPoint(
              {std::make_shared<VectorStateDeclaration>("s", schema)},
              std::move(plans),
              ConvergenceConfig{.plans = {}, .maxIterations = 10},
              /*outputStateEntry=*/"s")
          .planNode(),
      "errorWhenMaxIterationReached requires a convergence criterion");
}

// The delta row count is already the verdict, so pairing it with a convergence
// sequence is a contradiction rather than a refinement.
TEST_F(FixedPointTest, whenDeltaEmptyRejectsConvergencePlan) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodePtr> plans{PlanBuilder(idGen)
                                           .stateSource("s", schema)
                                           .project({"id", "val"})
                                           .planNode()};
  auto convergence = ConvergenceConfig::whenDeltaEmpty(10);
  convergence.plans = {PlanBuilder(idGen)
                           .stateSource("s", schema)
                           .singleAggregation({}, {"count(1)"})
                           .project({"a0 = 0 AS converged"})
                           .planNode()};
  VELOX_ASSERT_THROW(
      PlanBuilder(idGen)
          .fixedPoint(
              {std::make_shared<VectorStateDeclaration>("s", schema)},
              std::move(plans),
              std::move(convergence),
              /*outputStateEntry=*/"s")
          .planNode(),
      "stopWhenDeltaEmpty and a convergence sequence are mutually exclusive");
}

// The delta is a worker's local row count, so with peers one worker's frontier
// can empty while others still read its output.  The node rejects the
// combination rather than letting the loop strand its peers.
TEST_F(FixedPointTest, whenDeltaEmptyRejectsShuffle) {
  auto schema = ROW({"id", "val"}, BIGINT());
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto seed = makeRowVector(
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({8, 5}),
      });

  VELOX_ASSERT_THROW(
      PlanBuilder(idGenerator)
          .fixedPoint(
              {std::make_shared<VectorStateDeclaration>(
                  "frontier",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode())},
              {PlanBuilder(idGenerator)
                   .stateSource("frontier", schema)
                   .partitionedOutput({"id"}, 2)
                   .planNode(),
               PlanBuilder(idGenerator)
                   .exchange(schema, "Presto")
                   .project({"id", "val / 2 AS val"})
                   .planNode()},
              ConvergenceConfig::whenDeltaEmpty(10),
              /*outputStateEntry=*/"frontier")
          .planNode(),
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto task = makeTask(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::converging(
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"val >= 3 AS converged"})
                      .planNode(),
                  100),
              /*outputStateEntry=*/"vals")
          .planNode(),
      exec::Task::ExecutionMode::kSerial,
      "serialstart");
  VELOX_ASSERT_THROW(
      task->start(/*maxDrivers=*/1), "Inconsistent task execution mode");
  task->requestCancel().wait();
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0, 1}),
          makeFlatVector<int64_t>({0, 0}),
      });

  VELOX_ASSERT_THROW(
      runViaTask(
          PlanBuilder(idGenerator)
              .fixedPoint(
                  std::make_shared<VectorStateDeclaration>(
                      "vals",
                      schema,
                      PlanBuilder(idGenerator).values({seed}).planNode()),
                  PlanBuilder(idGenerator)
                      .stateSource("vals", schema)
                      .project({"key", "val + 1 AS val"}),
                  ConvergenceConfig::converging(
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"val >= 3 AS converged"})
                          .planNode(),
                      100),
                  /*outputStateEntry=*/"vals")
              .planNode(),
          exec::Task::ExecutionMode::kSerial),
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });
  auto start = std::chrono::steady_clock::now();
  auto rows = runViaTask(
      PlanBuilder(idGenerator)
          .fixedPoint(
              std::make_shared<VectorStateDeclaration>(
                  "vals",
                  schema,
                  PlanBuilder(idGenerator).values({seed}).planNode()),
              PlanBuilder(idGenerator)
                  .stateSource("vals", schema)
                  .project({"key", "val + 1 AS val"}),
              ConvergenceConfig::withMaxIterations(kIterations),
              /*outputStateEntry=*/"vals")
          .planNode(),
      exec::Task::ExecutionMode::kSerial);
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
      schema->names(),
      {
          makeFlatVector<int64_t>({0}),
          makeFlatVector<int64_t>({0}),
      });

  auto node = PlanBuilder(idGenerator)
                  .fixedPoint(
                      std::make_shared<VectorStateDeclaration>(
                          "vals",
                          schema,
                          PlanBuilder(idGenerator).values({seed}).planNode()),
                      // Body increments val; convergence wants val >= 1000,
                      // never reached in 3.
                      PlanBuilder(idGenerator)
                          .stateSource("vals", schema)
                          .project({"key", "val + 1 AS val"}),
                      ConvergenceConfig::converging(
                          PlanBuilder(idGenerator)
                              .stateSource("vals", schema)
                              .project({"val >= 1000 AS converged"})
                              .planNode(),
                          3),
                      /*outputStateEntry=*/"vals")
                  .planNode();

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
