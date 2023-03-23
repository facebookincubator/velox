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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class LocalPartitionTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, T max, vector_size_t size) {
    return makeFlatVector<T>(
        size, [start, max](auto row) { return (start + row) % max; });
  }

  std::vector<std::shared_ptr<TempFilePath>> writeToFiles(
      const std::vector<RowVectorPtr>& vectors) {
    auto filePaths = makeFilePaths(vectors.size());
    for (auto i = 0; i < vectors.size(); i++) {
      writeToFile(filePaths[i]->path, vectors[i]);
    }
    return filePaths;
  }

  void verifyExchangeSourceOperatorStats(
      const std::shared_ptr<exec::Task>& task,
      int expectedPositions,
      int expectedVectors) {
    auto stats = task->taskStats().pipelineStats[0].operatorStats.front();
    ASSERT_EQ(stats.inputPositions, expectedPositions);
    ASSERT_EQ(stats.inputVectors, expectedVectors);
    ASSERT_TRUE(stats.inputBytes > 0);

    ASSERT_EQ(stats.outputPositions, stats.inputPositions);
    ASSERT_EQ(stats.outputVectors, stats.inputVectors);
    ASSERT_EQ(stats.inputBytes, stats.outputBytes);
  }

  void assertTaskReferenceCount(
      const std::shared_ptr<exec::Task>& task,
      int expected) {
    // Make sure there is only one reference to Task left, i.e. no Driver is
    // blocked forever. Wait for a bit if that's not immediately the case.
    if (task.use_count() > expected) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    ASSERT_EQ(expected, task.use_count());
  }

  void waitForTaskState(
      const std::shared_ptr<exec::Task>& task,
      exec::TaskState expected) {
    if (task->state() != expected) {
      auto& executor = folly::QueuedImmediateExecutor::instance();
      auto future = task->stateChangeFuture(1'000'000).via(&executor);
      future.wait();
      EXPECT_EQ(expected, task->state());
    }
  }
};

TEST_F(LocalPartitionTest, gather) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto valuesNode = [&](int index) {
    return PlanBuilder(planNodeIdGenerator).values({vectors[index]}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {},
                    {
                        valuesNode(0),
                        valuesNode(1),
                        valuesNode(2),
                    })
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
                .planNode();

  auto task = assertQuery(op, "SELECT 300, -71, 152");
  verifyExchangeSourceOperatorStats(task, 300, 3);

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  std::vector<core::PlanNodeId> scanNodeIds;

  auto tableScanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  op = PlanBuilder(planNodeIdGenerator)
           .localPartition(
               {},
               {
                   tableScanNode(),
                   tableScanNode(),
                   tableScanNode(),
               })
           .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
           .planNode();

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->path));
  }

  task = queryBuilder.assertResults("SELECT 300, -71, 152");
  verifyExchangeSourceOperatorStats(task, 300, 3);
}

TEST_F(LocalPartitionTest, partition) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanAggNode = [&]() {
    auto builder = PlanBuilder(planNodeIdGenerator);
    auto scanNode = builder.tableScan(rowType).planNode();
    scanNodeIds.push_back(scanNode->id());
    return builder.partialAggregation({"c0"}, {"count(1)"}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {
                        scanAggNode(),
                        scanAggNode(),
                        scanAggNode(),
                    })
                .partialAggregation({"c0"}, {"count(1)"})
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  queryBuilder.maxDrivers(2);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->path));
  }

  auto task =
      queryBuilder.assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");
  verifyExchangeSourceOperatorStats(task, 300, 6);
}

TEST_F(LocalPartitionTest, maxBufferSizeGather) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto valuesNode = [&](int start, int end) {
    return PlanBuilder(planNodeIdGenerator)
        .values(std::vector<RowVectorPtr>(
            vectors.begin() + start, vectors.begin() + end))
        .planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {},
                    {
                        valuesNode(0, 7),
                        valuesNode(7, 14),
                        valuesNode(14, 21),
                    })
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
                .planNode();

  auto task = AssertQueryBuilder(op, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxLocalExchangeBufferSize, "100")
                  .assertResults("SELECT 2100, -71, 228");

  verifyExchangeSourceOperatorStats(task, 2100, 21);
}

TEST_F(LocalPartitionTest, maxBufferSizePartition) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  createDuckDbTable(vectors);

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {
                        scanNode(),
                        scanNode(),
                        scanNode(),
                    })
                .partialAggregation({"c0"}, {"count(1)"})
                .planNode();

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  queryBuilder.maxDrivers(2);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i % 3], makeHiveConnectorSplit(filePaths[i]->path));
  }

  // Set an artificially low buffer size limit to trigger blocking behavior.
  queryBuilder.config(core::QueryConfig::kMaxLocalExchangeBufferSize, "100");

  auto task =
      queryBuilder.assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");
  verifyExchangeSourceOperatorStats(task, 2100, 42);

  // Re-run with higher memory limit (enough to hold ~10 vectors at a time).
  queryBuilder.config(core::QueryConfig::kMaxLocalExchangeBufferSize, "10240");

  task = queryBuilder.assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");
  verifyExchangeSourceOperatorStats(task, 2100, 42);
}

TEST_F(LocalPartitionTest, multipleExchanges) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatSequence<int32_t>(0, 100),
          makeFlatSequence<int64_t>(0, 7, 100),
      }),
      makeRowVector({
          makeFlatSequence<int32_t>(53, 100),
          makeFlatSequence<int64_t>(0, 11, 100),
      }),
      makeRowVector({
          makeFlatSequence<int32_t>(-71, 100),
          makeFlatSequence<int64_t>(0, 13, 100),
      }),
  };

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodeId> scanNodeIds;

  auto tableScanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  // Make a plan with 2 local exchanges. UNION ALL results of 3 table scans.
  // Group by 0, 1 and compute counts. Group by 0 and compute counts and sums.
  // First exchange re-partitions the results of table scan on two keys. Second
  // exchange re-partitions the results on just the first key.
  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {PlanBuilder(planNodeIdGenerator)
                         .localPartition(
                             {"c0", "c1"},
                             {
                                 tableScanNode(),
                                 tableScanNode(),
                                 tableScanNode(),
                             })
                         .partialAggregation({"c0", "c1"}, {"count(1)"})
                         .planNode()})
                .partialAggregation({"c0"}, {"count(1)", "sum(a0)"})
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->path));
  }

  queryBuilder.maxDrivers(2).assertResults(
      "SELECT c0, count(1), sum(cnt) FROM ("
      "   SELECT c0, c1, count(1) as cnt FROM tmp GROUP BY 1, 2"
      ") t GROUP BY 1");
}

TEST_F(LocalPartitionTest, earlyCompletion) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(11, 100)}),
      makeRowVector({makeFlatSequence(13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {}, {PlanBuilder(planNodeIdGenerator).values(data).planNode()})
          .limit(0, 2, true)
          .planNode();

  auto task = assertQuery(plan, "VALUES (3), (4)");

  verifyExchangeSourceOperatorStats(task, 100, 1);

  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, earlyCancelation) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(11, 100)}),
      makeRowVector({makeFlatSequence(13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {}, {PlanBuilder(planNodeIdGenerator).values(data).planNode()})
          .limit(0, 2'000, true)
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  // Make sure results are queued one batch at a time.
  params.bufferedBytes = 100;

  auto cursor = std::make_unique<TaskCursor>(params);
  const auto& task = cursor->task();

  // Fetch first batch of data.
  ASSERT_TRUE(cursor->moveNext());
  ASSERT_EQ(100, cursor->current()->size());

  // Cancel the task.
  task->requestCancel();

  // Fetch the remaining results. This will throw since only one vector can be
  // buffered in the cursor.
  try {
    while (cursor->moveNext()) {
      ;
      FAIL() << "Expected a throw due to cancellation";
    }
  } catch (const std::exception& e) {
  }

  // Wait for task to transition to final state.
  waitForTaskState(task, exec::kCanceled);

  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, producerError) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(-11, 100)}),
      makeRowVector({makeFlatSequence(-13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .values(data)
                           .project({"7 / c0"})
                           .planNode()})
                  .limit(0, 2'000, true)
                  .planNode();

  CursorParameters params;
  params.planNode = plan;

  auto cursor = std::make_unique<TaskCursor>(params);
  const auto& task = cursor->task();

  // Expect division by zero error.
  ASSERT_THROW(
      while (cursor->moveNext()) { ; }, VeloxException);

  // Wait for task to transition to failed state.
  waitForTaskState(task, exec::kFailed);

  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, unionAll) {
  auto data1 = makeRowVector(
      {"d0", "d1"},
      {makeFlatVector<int32_t>({10, 11}),
       makeFlatVector<StringView>({"x", "y"})});
  auto data2 = makeRowVector(
      {"e0", "e1"},
      {makeFlatVector<int32_t>({20, 21}),
       makeFlatVector<StringView>({"z", "w"})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data1})
                           .project({"d0 as c0", "d1 as c1"})
                           .planNode(),
                       PlanBuilder(planNodeIdGenerator)
                           .values({data2})
                           .project({"e0 as c0", "e1 as c1"})
                           .planNode()})
                  .planNode();

  assertQuery(
      plan,
      "WITH t1 AS (VALUES (10, 'x'), (11, 'y')), "
      "t2 AS (VALUES (20, 'z'), (21, 'w')) "
      "SELECT * FROM t1 UNION ALL SELECT * FROM t2");
}

TEST_F(LocalPartitionTest, unionAllLocalExchange) {
  auto data1 = makeRowVector({"d0"}, {makeFlatVector<StringView>({"x"})});
  auto data2 = makeRowVector({"e0"}, {makeFlatVector<StringView>({"y"})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartitionRoundRobin(
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data1})
                           .project({"d0 as c0"})
                           .planNode(),
                       PlanBuilder(planNodeIdGenerator)
                           .values({data2})
                           .project({"e0 as c0"})
                           .planNode()})
                  .project({"length(c0)"})
                  .planNode();

  assertQuery(
      plan,
      "SELECT length(c0) FROM ("
      "   SELECT * FROM (VALUES ('x')) as t1(c0) UNION ALL "
      "   SELECT * FROM (VALUES ('y')) as t2(c0)"
      ")");
}
