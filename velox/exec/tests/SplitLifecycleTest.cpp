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

#include <chrono>
#include <thread>

#include <folly/OperationCancelled.h>
#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;

namespace facebook::velox::exec {
namespace {

// Tests split accounting when consumers close with unfinished work.
class SplitLifecycleTest : public test::HiveConnectorTestBase {};

// Builds a round-robin local union of two table scans. Applies a partial LIMIT
// of two rows to the first scan. Sets 'sourceId' to the limited scan's ID and
// 'siblingId' to the other scan's ID so callers can supply splits
// independently.
core::PlanNodePtr makeScanUnion(
    const RowTypePtr& rowType,
    core::PlanNodeId& sourceId,
    core::PlanNodeId& siblingId) {
  const auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  return test::PlanBuilder(ids)
      .localPartitionRoundRobin({
          test::PlanBuilder(ids)
              .tableScan(rowType)
              .capturePlanNodeId(sourceId)
              .limit(0, 2, true)
              .planNode(),
          test::PlanBuilder(ids)
              .tableScan(rowType)
              .capturePlanNodeId(siblingId)
              .planNode(),
      })
      .planNode();
}

// Creates a weighted Hive split for a DWRF file, optionally in a split group.
exec::Split
makeSplit(const std::string& path, int64_t weight, int32_t group = -1) {
  return exec::Split(
      connector::hive::HiveConnectorSplitBuilder(path)
          .connectorId(test::kHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::DWRF)
          .splitWeight(weight)
          .build(),
      group);
}

// Consumes exactly 'numRows' rows and leaves the cursor open for task checks.
void readRows(TaskCursor& cursor, vector_size_t numRows) {
  vector_size_t numRead{0};
  while (numRead < numRows) {
    VELOX_CHECK(cursor.moveNext());
    numRead += cursor.current()->size();
  }
  VELOX_CHECK_EQ(numRead, numRows);
}

// Waits up to five seconds for at least 'numDrivers' drivers to finish.
void waitForDrivers(Task& task, uint32_t numDrivers) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (task.numFinishedDrivers() < numDrivers) {
    VELOX_CHECK(
        std::chrono::steady_clock::now() < deadline,
        "Timed out waiting for drivers to finish: {}",
        numDrivers);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

// Verifies that neither acquired nor queued splits retain counts or weights.
void expectNoOutstandingSplits(const TaskStats& stats) {
  EXPECT_EQ(stats.numRunningSplits, 0);
  EXPECT_EQ(stats.numQueuedSplits, 0);
  EXPECT_EQ(stats.numRunningTableScanSplits, 0);
  EXPECT_EQ(stats.numQueuedTableScanSplits, 0);
  EXPECT_EQ(stats.runningTableScanSplitWeights, 0);
  EXPECT_EQ(stats.queuedTableScanSplitWeights, 0);
}

// Supplies the sibling scan's final split and returns the remaining query
// output. Returned vectors borrow the cursor's memory pool; the caller must
// keep the cursor alive while using them.
std::vector<RowVectorPtr>
finishScan(TaskCursor& cursor, const core::PlanNodeId& scanId, Split split) {
  const auto task = cursor.task();
  task->addSplit(scanId, std::move(split));
  task->noMoreSplits(scanId);
  std::vector<RowVectorPtr> results;
  while (cursor.moveNext()) {
    const auto result = cursor.current();
    result->loadedVector();
    results.push_back(result);
  }
  VELOX_CHECK(test::waitForTaskCompletion(task.get()));
  return results;
}

TEST_F(SplitLifecycleTest, earlyClosure) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  for (const auto numDrivers : {1, 4}) {
    for (const auto weight : {0, 100}) {
      SCOPED_TRACE(fmt::format("drivers={}, weight={}", numDrivers, weight));
      core::PlanNodeId sourceId;
      core::PlanNodeId siblingId;
      CursorParameters params;
      params.planNode = makeScanUnion(data->rowType(), sourceId, siblingId);
      params.maxDrivers = numDrivers;
      const auto cursor = TaskCursor::create(params);
      const auto task = cursor->task();
      const auto numSplits = numDrivers + 2;
      for (int32_t i = 0; i < numSplits; ++i) {
        task->addSplit(sourceId, makeSplit(file->getPath(), weight));
      }

      readRows(*cursor, 2 * numDrivers);
      waitForDrivers(*task, numDrivers);
      ASSERT_TRUE(task->isRunning());

      {
        const auto stats = task->taskStats();
        expectNoOutstandingSplits(stats);
        EXPECT_EQ(stats.numFinishedSplits, 0);
        EXPECT_EQ(stats.numAbandonedSplits, numSplits);
      }

      task->addSplit(sourceId, makeSplit(file->getPath(), weight));
      EXPECT_FALSE(task->addSplitWithSequence(
          sourceId, makeSplit(file->getPath(), weight), 100));

      EXPECT_EQ(task->taskStats().numTotalSplits, numSplits);

      task->noMoreSplits(sourceId);
      test::assertEqualResults(
          {data},
          finishScan(*cursor, siblingId, makeSplit(file->getPath(), 100)));

      {
        const auto stats = task->taskStats();
        expectNoOutstandingSplits(stats);
        EXPECT_EQ(stats.numFinishedSplits, 1);
      }
    }
  }
}

// Adds 'split' once the task's queued and running scan split weight is below
// 'maxWeight', waiting for task state changes as a coordinator with split
// back-pressure does. Fails after five seconds without capacity.
void admitSplit(
    Task& task,
    const core::PlanNodeId& scanId,
    Split split,
    int64_t maxWeight) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (true) {
    auto stateChange = task.stateChangeFuture(0);
    const auto stats = task.taskStats();
    if (stats.queuedTableScanSplitWeights + stats.runningTableScanSplitWeights <
        maxWeight) {
      break;
    }
    VELOX_CHECK(
        std::move(stateChange)
            .wait(
                std::chrono::duration_cast<folly::HighResDuration>(
                    deadline - std::chrono::steady_clock::now())),
        "Timed out waiting for split admission capacity");
  }
  task.addSplit(scanId, std::move(split));
}

// A coordinator that admits splits one plan node at a time, at most two
// weighted splits outstanding, keeps admitting after a LIMIT branch stops
// reading: the drivers that close release their splits' weight. The sibling
// scan then gets its split and the query completes.
TEST_F(SplitLifecycleTest, admissionResumesAfterLimit) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  const auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId sourceId;
  core::PlanNodeId siblingId;
  CursorParameters params;
  params.planNode = test::PlanBuilder(ids)
                        .localPartitionRoundRobin({
                            test::PlanBuilder(ids)
                                .tableScan(data->rowType())
                                .capturePlanNodeId(sourceId)
                                .limit(0, 2, true)
                                .localGather()
                                .limit(0, 2, false)
                                .planNode(),
                            test::PlanBuilder(ids)
                                .tableScan(data->rowType())
                                .capturePlanNodeId(siblingId)
                                .planNode(),
                        })
                        .planNode();
  params.maxDrivers = 4;
  const auto cursor = TaskCursor::create(params);
  cursor->start();
  const auto task = cursor->task();

  for (int32_t i = 0; i < 20; ++i) {
    admitSplit(*task, sourceId, makeSplit(file->getPath(), 100), 200);
  }
  task->noMoreSplits(sourceId);
  admitSplit(*task, siblingId, makeSplit(file->getPath(), 100), 200);
  task->noMoreSplits(siblingId);

  readRows(*cursor, data->size() + 2);
  ASSERT_FALSE(cursor->moveNext());
  ASSERT_TRUE(test::waitForTaskCompletion(task.get()));
  expectNoOutstandingSplits(task->taskStats());
}

TEST_F(SplitLifecycleTest, remainingConsumerAcceptsSplits) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  core::PlanNodeId sourceId;
  core::PlanNodeId siblingId;
  CursorParameters params;
  params.planNode = makeScanUnion(data->rowType(), sourceId, siblingId);
  params.maxDrivers = 2;
  const auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();

  task->addSplit(sourceId, makeSplit(file->getPath(), 100));
  readRows(*cursor, 2);
  waitForDrivers(*task, 1);

  EXPECT_EQ(task->taskStats().numAbandonedSplits, 1);

  EXPECT_TRUE(task->addSplitWithSequence(
      sourceId, makeSplit(file->getPath(), 100), 100));
  readRows(*cursor, 2);
  waitForDrivers(*task, 2);

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numAbandonedSplits, 2);
  }

  task->noMoreSplits(sourceId);
  test::assertEqualResults(
      {data}, finishScan(*cursor, siblingId, makeSplit(file->getPath(), 100)));

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numFinishedSplits, 1);
  }
}

TEST_F(SplitLifecycleTest, closureIsScopedToGroup) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  core::PlanNodeId sourceId;
  core::PlanNodeId siblingId;
  CursorParameters params;
  params.planNode = makeScanUnion(data->rowType(), sourceId, siblingId);
  params.executionStrategy = core::ExecutionStrategy::kGrouped;
  params.groupedExecutionLeafNodeIds = {sourceId, siblingId};
  params.numSplitGroups = 2;
  const auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();

  for (const auto group : {0, 1}) {
    for (int32_t i = 0; i < 3; ++i) {
      task->addSplit(sourceId, makeSplit(file->getPath(), 100, group));
    }
  }
  auto progress = task->stateChangeFuture(0);
  readRows(*cursor, 2);
  waitForDrivers(*task, 1);
  EXPECT_TRUE(progress.isReady());

  {
    const auto stats = task->taskStats();
    EXPECT_EQ(stats.numAbandonedSplits, 3);
    EXPECT_EQ(stats.numQueuedTableScanSplits, 3);
    EXPECT_EQ(stats.queuedTableScanSplitWeights, 300);
  }

  EXPECT_FALSE(task->addSplitWithSequence(
      sourceId, makeSplit(file->getPath(), 100, 0), 100));
  EXPECT_TRUE(task->addSplitWithSequence(
      sourceId, makeSplit(file->getPath(), 100, 1), 100));

  task->addSplit(siblingId, makeSplit(file->getPath(), 100, 0));
  task->noMoreSplitsForGroup(siblingId, 0);
  readRows(*cursor, data->size() + 2);
  waitForDrivers(*task, 4);

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numAbandonedSplits, 7);
  }

  task->noMoreSplits(sourceId);
  test::assertEqualResults(
      {data},
      finishScan(*cursor, siblingId, makeSplit(file->getPath(), 100, 1)));

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numFinishedSplits, 2);
  }
}

TEST_F(SplitLifecycleTest, cancellationDiscardsUnstartedGroup) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  core::PlanNodeId sourceId;
  core::PlanNodeId siblingId;
  CursorParameters params;
  params.planNode = makeScanUnion(data->rowType(), sourceId, siblingId);
  params.maxDrivers = 2;
  params.executionStrategy = core::ExecutionStrategy::kGrouped;
  params.groupedExecutionLeafNodeIds = {sourceId, siblingId};
  params.numSplitGroups = 2;
  const auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();

  task->addSplit(sourceId, makeSplit(file->getPath(), 100, 0));
  for (int32_t i = 0; i < 3; ++i) {
    task->addSplit(sourceId, makeSplit(file->getPath(), 100, 1));
  }
  readRows(*cursor, 2);
  waitForDrivers(*task, 1);

  task->requestCancel();
  ASSERT_TRUE(test::waitForTaskCancelled(task.get()));
  const auto stats = task->taskStats();
  expectNoOutstandingSplits(stats);
  EXPECT_EQ(stats.numAbandonedSplits, 4);
}

// Builds a table scan with a partial LIMIT of two rows and sets 'sourceId'
// to the scan's ID.
core::PlanNodePtr makeLimitedScan(
    const RowTypePtr& rowType,
    core::PlanNodeId& sourceId) {
  return test::PlanBuilder()
      .tableScan(rowType)
      .capturePlanNodeId(sourceId)
      .limit(0, 2, true)
      .planNode();
}

TEST_F(SplitLifecycleTest, lastConsumerDiscardsQueue) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);

  core::PlanNodeId sourceId;
  const auto plan = makeLimitedScan(data->rowType(), sourceId);
  const auto task =
      test::AssertQueryBuilder(plan)
          .maxDrivers(4)
          .beforeTaskStart([&](Task& task) {
            for (int32_t i = 0; i < 6; ++i) {
              task.addSplit(sourceId, makeSplit(file->getPath(), 100));
            }
          })
          .assertResults(
              std::vector<RowVectorPtr>(
                  4, makeRowVector({makeFlatVector({0, 1})})));

  const auto stats = task->taskStats();
  expectNoOutstandingSplits(stats);
  EXPECT_EQ(stats.numFinishedSplits, 0);
  EXPECT_EQ(stats.numAbandonedSplits, 6);
}

TEST_F(SplitLifecycleTest, closesPreloadedSplits) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  core::PlanNodeId sourceId;
  const auto plan = makeLimitedScan(data->rowType(), sourceId);
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
  const auto task = test::AssertQueryBuilder(plan)
                        .config(core::QueryConfig::kMaxSplitPreloadPerDriver, 3)
                        .beforeTaskStart([&](Task& task) {
                          for (int32_t i = 0; i < 3; ++i) {
                            auto split = makeSplit(file->getPath(), 100);
                            splits.push_back(split.connectorSplit);
                            task.addSplit(sourceId, std::move(split));
                          }
                        })
                        .assertResults(makeRowVector({makeFlatVector({0, 1})}));
  for (const auto& split : splits) {
    ASSERT_NE(split->dataSource, nullptr);
    EXPECT_EQ(split->dataSource->move(), nullptr);
  }
  const auto stats = task->taskStats();
  expectNoOutstandingSplits(stats);
  EXPECT_EQ(stats.numAbandonedSplits, 3);
}

DEBUG_ONLY_TEST_F(SplitLifecycleTest, cancellationDiscardsAcquiredSplit) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  const auto plan = test::PlanBuilder().tableScan(data->rowType()).planNode();
  std::shared_ptr<Task> task;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::TableScan::getOutput::gotSplit",
      std::function<void(const TableScan*)>([](const TableScan* scan) {
        scan->operatorCtx()->task()->requestCancel();
      }));

  EXPECT_THAT(
      [&] {
        test::AssertQueryBuilder(plan)
            .beforeTaskStart([&](Task& startedTask) {
              task = startedTask.shared_from_this();
              for (int32_t i = 0; i < 3; ++i) {
                startedTask.addSplit(
                    plan->id(), makeSplit(file->getPath(), 100));
              }
            })
            .copyResults(pool());
      },
      testing::ThrowsMessage<folly::OperationCancelled>(
          testing::StrEq("coroutine operation cancelled")));
  ASSERT_TRUE(test::waitForTaskCancelled(task.get()));
  const auto stats = task->taskStats();
  expectNoOutstandingSplits(stats);
  EXPECT_EQ(stats.numAbandonedSplits, 3);
}

TEST_F(SplitLifecycleTest, exchangeSource) {
  const auto data = makeRowVector({makeFlatIdentityVector<int32_t>(10'000)});
  const auto file = writeToFile(data);
  ExchangeSource::registerFactory(test::createLocalExchangeSource);
  const auto upstream = Task::create(
      "local://split-lifecycle",
      test::PlanBuilder()
          .values({data})
          .partitionedOutput({}, 1)
          .planFragment(),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  upstream->start(1);

  const auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId sourceId;
  core::PlanNodeId siblingId;
  CursorParameters params;
  params.maxDrivers = 1;
  params.planNode = test::PlanBuilder(ids)
                        .localPartitionRoundRobin({
                            test::PlanBuilder(ids)
                                .exchange(data->rowType(), "Presto")
                                .capturePlanNodeId(sourceId)
                                .limit(0, 2, true)
                                .planNode(),
                            test::PlanBuilder(ids)
                                .tableScan(data->rowType())
                                .capturePlanNodeId(siblingId)
                                .planNode(),
                        })
                        .planNode();
  const auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();
  auto remoteSplit = [&] {
    return exec::Split(
        std::make_shared<RemoteConnectorSplit>(upstream->taskId()));
  };
  task->addSplit(sourceId, remoteSplit());
  readRows(*cursor, 2);
  waitForDrivers(*task, 1);

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numAbandonedSplits, 1);
  }

  EXPECT_FALSE(task->addSplitWithSequence(sourceId, remoteSplit(), 100));
  task->noMoreSplits(sourceId);
  test::assertEqualResults(
      {data}, finishScan(*cursor, siblingId, makeSplit(file->getPath(), 100)));

  {
    const auto stats = task->taskStats();
    expectNoOutstandingSplits(stats);
    EXPECT_EQ(stats.numFinishedSplits, 1);
  }

  ASSERT_TRUE(test::waitForTaskCompletion(upstream.get()));
}

} // namespace
} // namespace facebook::velox::exec
