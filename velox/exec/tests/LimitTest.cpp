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
#include <folly/executors/ManualExecutor.h>

#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {
class LimitTest : public HiveConnectorTestBase {
 protected:
  void testParallelBarrier(
      int32_t numDrivers,
      int32_t numSplits,
      int32_t limit,
      bool finishBeforeBarrier,
      bool abortAfterBarrier) {
    SCOPED_TRACE(
        fmt::format(
            "drivers={} splits={} limit={} finishBeforeBarrier={} abort={}",
            numDrivers,
            numSplits,
            limit,
            finishBeforeBarrier,
            abortAfterBarrier));
    auto data = makeRowVector(
        {makeFlatVector<int32_t>(1'000, [](auto row) { return row; })});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), {data});

    folly::ManualExecutor executor;
    core::PlanNodeId scanId;
    CursorParameters params;
    params.planNode = PlanBuilder()
                          .tableScan(asRowType(data->type()))
                          .capturePlanNodeId(scanId)
                          .limit(0, limit, true)
                          .planNode();
    params.maxDrivers = numDrivers;
    if (finishBeforeBarrier) {
      params.queryCtx = core::QueryCtx::create(&executor);
    }
    auto cursor = TaskCursor::create(params);
    auto task = cursor->task();
    SCOPE_EXIT {
      auto finished = task->requestAbort();
      executor.drain();
      finished.wait();
    };
    cursor->start();
    ASSERT_EQ(task->numTotalDrivers(), numDrivers);

    std::vector<RowVectorPtr> results;
    auto readRound = [&]() {
      for (;;) {
        ContinueFuture future = ContinueFuture::makeEmpty();
        if (cursor->moveNext(&future)) {
          results.push_back(cursor->current());
        } else if (!future.valid()) {
          return;
        } else {
          executor.drain();
          ASSERT_TRUE(std::move(future).wait(std::chrono::seconds(5)))
              << task->toString();
        }
      }
    };

    for (int32_t i = 0; i < numSplits; ++i) {
      task->addSplit(
          scanId, exec::Split(makeHiveConnectorSplit(file->getPath())));
      if (finishBeforeBarrier) {
        executor.drain();
        ASSERT_GT(task->numFinishedDrivers(), 0);
      }
      auto barrier = task->requestBarrier();
      executor.drain();
      ASSERT_NO_FATAL_FAILURE(readRound());
      ASSERT_TRUE(std::move(barrier).wait(std::chrono::seconds(5)))
          << task->toString();
      ASSERT_FALSE(task->underBarrier());
      if (abortAfterBarrier) {
        ASSERT_TRUE(task->isRunning());
        auto finished = task->requestAbort();
        executor.drain();
        ASSERT_TRUE(std::move(finished).wait(std::chrono::seconds(5)));
        ASSERT_EQ(task->state(), TaskState::kAborted);
        return;
      }
      if (!task->isRunning()) {
        break;
      }
    }

    task->noMoreSplits(scanId);
    executor.drain();
    ASSERT_NO_FATAL_FAILURE(readRound());
    ASSERT_TRUE(waitForTaskCompletion(task.get()));
    const auto expectedSplits =
        limit == 1'000 ? std::min(numSplits, numDrivers) : numSplits;
    assertEqualResults(
        std::vector<RowVectorPtr>(expectedSplits, data), results);
  }
};

// https://github.com/facebookincubator/velox/issues/19866
TEST_F(LimitTest, parallelBarrierEarlyFinish) {
  for (const auto numDrivers : {2, 4}) {
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 1, 1'000, false, false));
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 5, 5'000, false, false));
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 8, 1'000, false, false));
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 5, 6'000, false, false));
  }
}

TEST_F(LimitTest, parallelBarrierFinishBeforeRequest) {
  for (const auto numDrivers : {2, 4}) {
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 8, 1'000, true, false));
  }
}

TEST_F(LimitTest, parallelBarrierFinishAfterDrain) {
  for (const auto numDrivers : {2, 4}) {
    for (const auto consumeRound : {false, true}) {
      SCOPED_TRACE(
          fmt::format("drivers={} consumeRound={}", numDrivers, consumeRound));
      folly::ManualExecutor executor;
      core::PlanNodeId scanId;
      CursorParameters params;
      params.planNode = PlanBuilder()
                            .tableScan(ROW({"c0"}, {INTEGER()}))
                            .capturePlanNodeId(scanId)
                            .limit(0, 1'000, true)
                            .planNode();
      params.maxDrivers = numDrivers;
      params.queryCtx = core::QueryCtx::create(&executor);
      auto cursor = TaskCursor::create(params);
      auto task = cursor->task();
      SCOPE_EXIT {
        auto finished = task->requestAbort();
        executor.drain();
        finished.wait();
      };
      cursor->start();
      ASSERT_EQ(task->numTotalDrivers(), numDrivers);
      auto barrier = task->requestBarrier();
      if (consumeRound) {
        executor.drain();
        ASSERT_TRUE(barrier.isReady());
        ContinueFuture future = ContinueFuture::makeEmpty();
        ASSERT_FALSE(cursor->moveNext(&future));
        ASSERT_FALSE(future.valid());
      } else {
        // One producer reports drained before EOF, without advancing the
        // cursor to the next round. Its EOF must not count as another arrival.
        ASSERT_EQ(executor.step(), 1);
        ASSERT_FALSE(barrier.isReady());
      }
      task->noMoreSplits(scanId);
      executor.drain();
      ASSERT_TRUE(barrier.isReady());
      ContinueFuture future = ContinueFuture::makeEmpty();
      ASSERT_FALSE(cursor->moveNext(&future));
      ASSERT_FALSE(future.valid());
      ASSERT_EQ(task->state(), TaskState::kFinished);
    }
  }
}

TEST_F(LimitTest, parallelBarrierCancelAfterEarlyFinish) {
  for (const auto numDrivers : {2, 4}) {
    ASSERT_NO_FATAL_FAILURE(
        testParallelBarrier(numDrivers, 1, 1'000, false, true));
  }
}

TEST_F(LimitTest, basic) {
  vector_size_t batchSize = 1'000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 3; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](auto row) { return batchSize * i + row; }, nullEvery(5));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [&](auto row) { return row; }, nullEvery(7));
    auto c2 = makeFlatVector<double>(
        batchSize, [](auto row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  auto makePlan = [&](int64_t offset, int64_t limit) {
    return PlanBuilder().values(vectors).limit(offset, limit, true).planNode();
  };

  assertQuery(makePlan(0, 10), "SELECT * FROM tmp LIMIT 10");
  int64_t limit = (int64_t)(std::numeric_limits<int32_t>::max()) + 1000000;
  int64_t offset = (int64_t)(std::numeric_limits<int32_t>::max()) + 1000;
  assertQuery(
      makePlan(0, limit), fmt::format("SELECT * FROM tmp LIMIT {}", limit));
  assertQuery(makePlan(0, 1'234), "SELECT * FROM tmp LIMIT 1234");

  assertQuery(makePlan(17, 10), "SELECT * FROM tmp OFFSET 17 LIMIT 10");
  assertQuery(makePlan(17, 983), "SELECT * FROM tmp OFFSET 17 LIMIT 983");
  assertQuery(makePlan(17, 2'000), "SELECT * FROM tmp OFFSET 17 LIMIT 2000");
  assertQuery(
      makePlan(offset, limit),
      fmt::format("SELECT * FROM tmp OFFSET {} LIMIT {}", offset, limit));

  assertQuery(
      makePlan(offset, 2000),
      fmt::format("SELECT * FROM tmp OFFSET {} LIMIT 2000", offset));

  assertQuery(makePlan(1'000, 145), "SELECT * FROM tmp OFFSET 1000 LIMIT 145");
  assertQuery(
      makePlan(1'000, 1'000), "SELECT * FROM tmp OFFSET 1000 LIMIT 1000");
  assertQuery(
      makePlan(1'000, 1'234), "SELECT * FROM tmp OFFSET 1000 LIMIT 1234");

  assertQuery(makePlan(1'234, 10), "SELECT * FROM tmp OFFSET 1234 LIMIT 10");
  assertQuery(makePlan(1'234, 983), "SELECT * FROM tmp OFFSET 1234 LIMIT 983");
  assertQuery(
      makePlan(1'234, 1'000), "SELECT * FROM tmp OFFSET 1234 LIMIT 1000");
  assertQuery(
      makePlan(1'234, 2'000), "SELECT * FROM tmp OFFSET 1234 LIMIT 2000");

  assertQueryReturnsEmptyResult(makePlan(12'345, 10));
}

TEST_F(LimitTest, limitOverLocalExchange) {
  auto data = makeRowVector(
      {makeFlatVector<int32_t>(1'000, [](auto row) { return row; })});

  auto file = TempFilePath::create();
  writeToFile(file->getPath(), {data});

  core::PlanNodeId scanNodeId;

  CursorParameters params;
  params.planNode = PlanBuilder()
                        .tableScan(asRowType(data->type()))
                        .capturePlanNodeId(scanNodeId)
                        .localPartition(std::vector<std::string>{})
                        .limit(0, 20, true)
                        .planNode();

  auto cursor = TaskCursor::create(params);
  cursor->task()->addSplit(
      scanNodeId, exec::Split(makeHiveConnectorSplit(file->getPath())));

  int32_t numRead = 0;
  while (cursor->moveNext()) {
    auto vector = cursor->current();
    numRead += vector->size();
  }

  // Do not send no-more-splits message. Expect the task to finish without
  // receiving that message.

  ASSERT_EQ(20, numRead);
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
}

TEST_F(LimitTest, partialLimitEagerFlush) {
  std::vector<RowVectorPtr> batches(
      10, makeRowVector({makeFlatVector(std::vector<int64_t>(1, 0))}));
  auto test = [&](bool projectInBetween) {
    SCOPED_TRACE(fmt::format("projectInBetween={}", projectInBetween));
    CursorParameters params;
    auto builder = PlanBuilder().values(batches).limit(0, 10, true);
    if (projectInBetween) {
      builder = builder.project({"c0 + 1 as c0"});
    }
    params.planNode = builder.partitionedOutput({}, 1).planNode();
    auto cursor = TaskCursor::create(params);
    ASSERT_FALSE(cursor->moveNext());
    auto bufferManager = exec::DefaultOutputBufferManager::getInstanceRef();
    auto [numPagesPromise, numPagesFuture] = folly::makePromiseContract<int>();
    ASSERT_TRUE(bufferManager->getData(
        cursor->task()->taskId(),
        0,
        INT32_MAX,
        0,
        [numPagesPromise =
             std::make_shared<folly::Promise<int>>(std::move(numPagesPromise))](
            std::vector<std::unique_ptr<folly::IOBuf>> pages,
            int64_t /*sequence*/,
            std::vector<int64_t> /*remainingBytes*/) {
          numPagesPromise->setValue(pages.size());
        }));
    ASSERT_GE(std::move(numPagesFuture).get(std::chrono::seconds(1)), 10);
    cursor->task()->requestCancel();
  };
  test(true);
  test(false);
}

TEST_F(LimitTest, DISABLED_barrier) {
  std::vector<RowVectorPtr> vectors;
  std::vector<std::shared_ptr<TempFilePath>> tempFiles;
  const int numSplits{5};
  const int numRowsPerSplit{1'000};
  for (int32_t i = 0; i < numSplits; ++i) {
    vectors.push_back(makeRowVector({makeFlatVector<int32_t>(
        numRowsPerSplit, [](auto row) { return row; })}));
    tempFiles.push_back(TempFilePath::create());
  }
  writeToFiles(toFilePaths(tempFiles), vectors);
  createDuckDbTable(vectors);

  struct {
    bool serialExecution;
    bool barrierExecution;
    int offset;
    int limit;
    int numExpectedBarriers;
    int numExpectedOutputRows;
    int numExpectedFinishedSplits;
    int numExpectedOutputBatches;

    std::string toString() const {
      return fmt::format(
          "serialExecution {}, barrierExecution {}, offset: {}, limit: {}, numExpectedBarriers: {}, numExpectedOutputRows: {}, numExpectedFinishedSplits: {}, numExpectedOutputBatches: {}",
          serialExecution,
          barrierExecution,
          offset,
          limit,
          numExpectedBarriers,
          numExpectedOutputRows,
          numExpectedFinishedSplits,
          numExpectedOutputBatches);
    }
  } testSettings[] = {
      // Test the case where the limit covers all the input rows
      // with barrier and not.
      {false,
       false,
       0,
       numRowsPerSplit * numSplits,
       numSplits,
       numRowsPerSplit * numSplits,
       numSplits - 1,
       numSplits},
      {true,
       false,
       0,
       numRowsPerSplit * numSplits,
       0,
       numRowsPerSplit * numSplits,
       numSplits - 1,
       numSplits},
      {false,
       true,
       0,
       numRowsPerSplit * numSplits,
       numSplits,
       numRowsPerSplit * numSplits,
       numSplits - 1,
       numSplits},
      {false,
       false,
       0,
       numRowsPerSplit * numSplits,
       0,
       numRowsPerSplit * numSplits,
       numSplits - 1,
       numSplits},
      // Test the cases where the limit covers the first entire
      // split rows with barrier or not. with barrier and not.
      {
          true,
          true,
          0,
          numRowsPerSplit,
          1,
          numRowsPerSplit,
          0,
          1,
      },
      {
          true,
          false,
          0,
          numRowsPerSplit,
          0,
          numRowsPerSplit,
          0,
          1,
      },
      {
          false,
          true,
          0,
          numRowsPerSplit,
          1,
          numRowsPerSplit,
          0,
          1,
      },
      {
          false,
          false,
          0,
          numRowsPerSplit,
          0,
          numRowsPerSplit,
          0,
          1,
      },
      // Test the case where the limit covers the one and half
      // split rows with barrier or not.
      {true,
       true,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       2,
       numRowsPerSplit + numRowsPerSplit / 2,
       1,
       2},
      {true,
       false,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       1,
       2},
      {false,
       true,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       2,
       numRowsPerSplit + numRowsPerSplit / 2,
       1,
       2},
      {false,
       false,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       0,
       numRowsPerSplit + numRowsPerSplit / 2,
       1,
       2},
      {true,
       true,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       numRowsPerSplit + numRowsPerSplit - 1,
       1,
       2},
      {true,
       false,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       1,
       2},
      {false,
       true,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       numRowsPerSplit + numRowsPerSplit - 1,
       1,
       2},
      {false,
       false,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       1,
       2},
      // Test the case where the limit set to cover more than
      // all the input rows with barrier or not.
      {true,
       true,
       0,
       numRowsPerSplit * (numSplits + 1),
       numSplits,
       numRowsPerSplit * numSplits,
       numSplits,
       numSplits},
      {true,
       false,
       0,
       numRowsPerSplit * (numSplits + 1),
       0,
       numRowsPerSplit * numSplits,
       numSplits,
       numSplits},
      {false,
       true,
       0,
       numRowsPerSplit * (numSplits + 1),
       numSplits,
       numRowsPerSplit * numSplits,
       numSplits,
       numSplits},
      {false,
       false,
       0,
       numRowsPerSplit * (numSplits + 1),
       0,
       numRowsPerSplit * numSplits,
       numSplits,
       numSplits},
      // Test the cases where the limit set to cover partial
      // input rows in the middle with barrier or not.
      {true,
       true,
       numRowsPerSplit,
       numRowsPerSplit + numRowsPerSplit - 1,
       3,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       2},
      {true,
       false,
       numRowsPerSplit,
       numRowsPerSplit + numRowsPerSplit - 1,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       2},
      {false,
       true,
       numRowsPerSplit,
       numRowsPerSplit + numRowsPerSplit - 1,
       3,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       2},
      {false,
       false,
       numRowsPerSplit,
       numRowsPerSplit + numRowsPerSplit - 1,
       0,
       numRowsPerSplit + numRowsPerSplit - 1,
       2,
       2},
      {true,
       true,
       numRowsPerSplit,
       numRowsPerSplit * (numSplits - 1),
       numSplits,
       numRowsPerSplit * (numSplits - 1),
       numSplits - 1,
       numSplits - 1},
      {true,
       false,
       numRowsPerSplit,
       numRowsPerSplit * (numSplits - 1),
       0,
       numRowsPerSplit * (numSplits - 1),
       numSplits - 1,
       numSplits - 1},
      {false,
       true,
       numRowsPerSplit,
       numRowsPerSplit * (numSplits - 1),
       numSplits,
       numRowsPerSplit * (numSplits - 1),
       numSplits - 1,
       numSplits - 1},
      {false,
       false,
       numRowsPerSplit,
       numRowsPerSplit * (numSplits - 1),
       0,
       numRowsPerSplit * (numSplits - 1),
       numSplits - 1,
       numSplits - 1},
      {true,
       true,
       numRowsPerSplit,
       numRowsPerSplit / 2,
       2,
       numRowsPerSplit / 2,
       1,
       1},
      {true,
       false,
       numRowsPerSplit,
       numRowsPerSplit / 2,
       0,
       numRowsPerSplit / 2,
       1,
       1},
      {false,
       true,
       numRowsPerSplit,
       numRowsPerSplit / 2,
       2,
       numRowsPerSplit / 2,
       1,
       1},
      {false,
       false,
       numRowsPerSplit,
       numRowsPerSplit / 2,
       0,
       numRowsPerSplit / 2,
       1,
       1},
      {true,
       true,
       numRowsPerSplit / 2,
       numRowsPerSplit * numSplits,
       numSplits,
       numRowsPerSplit * numSplits - numRowsPerSplit / 2,
       numSplits,
       numSplits},
      {true,
       false,
       numRowsPerSplit / 2,
       numRowsPerSplit * numSplits,
       0,
       numRowsPerSplit * numSplits - numRowsPerSplit / 2,
       numSplits,
       numSplits},
      {false,
       true,
       numRowsPerSplit / 2,
       numRowsPerSplit * numSplits,
       numSplits,
       numRowsPerSplit * numSplits - numRowsPerSplit / 2,
       numSplits,
       numSplits},
      {false,
       false,
       numRowsPerSplit / 2,
       numRowsPerSplit * numSplits,
       0,
       numRowsPerSplit * numSplits - numRowsPerSplit / 2,
       numSplits,
       numSplits},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.toString());
    core::PlanNodeId limitPlanNodeId;
    auto plan = PlanBuilder()
                    .tableScan(asRowType(vectors.back()->type()))
                    .limit(testData.offset, testData.limit, true)
                    .capturePlanNodeId(limitPlanNodeId)
                    .planNode();
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .splits(makeHiveConnectorSplits(tempFiles))
                    .serialExecution(testData.serialExecution)
                    .barrierExecution(testData.barrierExecution)
                    .assertResults(
                        fmt::format(
                            "SELECT * FROM tmp LIMIT {} OFFSET {}",
                            testData.limit,
                            testData.offset));
    const auto taskStats = task->taskStats();
    ASSERT_EQ(taskStats.numBarriers, testData.numExpectedBarriers);
    ASSERT_EQ(
        exec::toPlanStats(taskStats).at(limitPlanNodeId).outputRows,
        testData.numExpectedOutputRows);
    ASSERT_EQ(
        exec::toPlanStats(taskStats).at(limitPlanNodeId).outputVectors,
        testData.numExpectedOutputBatches);
  }
}
} // namespace
