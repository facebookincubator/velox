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
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

static const std::string kWriter = "LocalPartitionTest.Writer";

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
      writeToFile(filePaths[i]->path, kWriter, vectors[i]);
    }
    return filePaths;
  }

  static RowTypePtr getRowType(const RowVectorPtr& rowVector) {
    return std::dynamic_pointer_cast<const RowType>(rowVector->type());
  }

  void verifyExchangeSourceOperatorStats(
      const std::shared_ptr<exec::Task>& task,
      int expectedPositions) {
    auto stats = task->taskStats().pipelineStats[0].operatorStats.front();
    ASSERT_EQ(stats.inputPositions, expectedPositions);
    ASSERT_EQ(stats.outputPositions, expectedPositions);
    ASSERT_TRUE(stats.inputBytes > 0);
    ASSERT_EQ(stats.inputBytes, stats.outputBytes);
  }
};

TEST_F(LocalPartitionTest, gather) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

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

  auto task = assertQuery(
      op, std::vector<std::shared_ptr<TempFilePath>>{}, "SELECT 300, -71, 152");
  verifyExchangeSourceOperatorStats(task, 300);

  auto filePaths = writeToFiles(vectors);

  auto rowType = getRowType(vectors[0]);

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

  int32_t fileIndex = 0;
  task = ::assertQuery(
      op,
      [&](exec::Task* task) {
        while (fileIndex < filePaths.size()) {
          auto planNodeId = scanNodeIds[fileIndex];
          addSplit(task, planNodeId, makeHiveSplit(filePaths[fileIndex]->path));
          task->noMoreSplits(planNodeId);
          ++fileIndex;
        }
      },
      "SELECT 300, -71, 152",
      duckDbQueryRunner_);
  verifyExchangeSourceOperatorStats(task, 300);
}

TEST_F(LocalPartitionTest, partition) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto filePaths = writeToFiles(vectors);

  auto rowType = getRowType(vectors[0]);

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanAggNode = [&]() {
    auto builder = PlanBuilder(planNodeIdGenerator);
    auto scanNode = builder.tableScan(rowType).planNode();
    scanNodeIds.push_back(scanNode->id());
    return builder.partialAggregation({0}, {"count(1)"}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {0},
                    {
                        scanAggNode(),
                        scanAggNode(),
                        scanAggNode(),
                    })
                .partialAggregation({0}, {"count(1)"})
                .planNode();

  createDuckDbTable(vectors);

  CursorParameters params;
  params.planNode = op;
  params.maxDrivers = 2;

  uint32_t fileIndex = 0;
  auto task = ::assertQuery(
      params,
      [&](exec::Task* task) {
        while (fileIndex < filePaths.size()) {
          auto planNodeId = scanNodeIds[fileIndex];
          addSplit(task, planNodeId, makeHiveSplit(filePaths[fileIndex]->path));
          task->noMoreSplits(planNodeId);
          ++fileIndex;
        }
      },
      "SELECT c0, count(1) FROM tmp GROUP BY 1",
      duckDbQueryRunner_);
  verifyExchangeSourceOperatorStats(task, 300);
}

TEST_F(LocalPartitionTest, maxBufferSizeGather) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

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

  CursorParameters params;
  params.planNode = op;
  params.queryCtx = core::QueryCtx::createForTest();

  // Set an artificially low buffer size limit to trigger blocking behavior.
  params.queryCtx->setConfigOverridesUnsafe({
      {core::QueryConfig::kMaxLocalExchangeBufferSize, "100"},
  });

  auto task = ::assertQuery(
      params,
      [&](exec::Task* /*task*/) {},
      "SELECT 2100, -71, 228",
      duckDbQueryRunner_);
  verifyExchangeSourceOperatorStats(task, 2100);
}

TEST_F(LocalPartitionTest, maxBufferSizePartition) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  createDuckDbTable(vectors);

  auto filePaths = writeToFiles(vectors);

  auto rowType = getRowType(vectors[0]);

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {0},
                    {
                        scanNode(),
                        scanNode(),
                        scanNode(),
                    })
                .partialAggregation({0}, {"count(1)"})
                .planNode();

  CursorParameters params;
  params.planNode = op;
  params.maxDrivers = 2;
  params.queryCtx = core::QueryCtx::createForTest();

  // Set an artificially low buffer size limit to trigger blocking behavior.
  params.queryCtx->setConfigOverridesUnsafe({
      {core::QueryConfig::kMaxLocalExchangeBufferSize, "100"},
  });

  uint32_t fileIndex = 0;
  auto addSplits = [&](exec::Task* task) {
    while (fileIndex < filePaths.size()) {
      auto planNodeId = scanNodeIds[fileIndex % 3];
      addSplit(task, planNodeId, makeHiveSplit(filePaths[fileIndex]->path));
      task->noMoreSplits(planNodeId);
      ++fileIndex;
    }
  };

  auto task = ::assertQuery(
      params,
      addSplits,
      "SELECT c0, count(1) FROM tmp GROUP BY 1",
      duckDbQueryRunner_);
  verifyExchangeSourceOperatorStats(task, 2100);

  // Re-run with higher memory limit (enough to hold ~10 vectors at a time).
  params.queryCtx->setConfigOverridesUnsafe({
      {core::QueryConfig::kMaxLocalExchangeBufferSize, "10240"},
  });

  fileIndex = 0;
  task = ::assertQuery(
      params,
      addSplits,
      "SELECT c0, count(1) FROM tmp GROUP BY 1",
      duckDbQueryRunner_);
  verifyExchangeSourceOperatorStats(task, 2100);
}

TEST_F(LocalPartitionTest, outputLayoutGather) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return row; }),
          makeFlatVector<int32_t>(100, [](auto row) { return row / 2; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return 53 + row; }),
          makeFlatVector<int32_t>(100, [](auto row) { return 53 + row / 2; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return -71 + row; }),
          makeFlatVector<int32_t>(100, [](auto row) { return -71 + row / 2; }),
      }),
  };

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
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
                    },
                    // Change column order: (c0, c1) -> (c1, c0).
                    {1, 0})
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c1)"})
                .planNode();

  auto task = assertQuery(op, "SELECT 300, -71, 102");
  verifyExchangeSourceOperatorStats(task, 300);

  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .localPartition(
               {},
               {
                   valuesNode(0),
                   valuesNode(1),
                   valuesNode(2),
               },
               // Drop column: (c0, c1) -> (c1).
               {1})
           .singleAggregation({}, {"count(1)", "min(c1)", "max(c1)"})
           .planNode();

  task = assertQuery(op, "SELECT 300, -71, 102");
  verifyExchangeSourceOperatorStats(task, 300);

  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .localPartition(
               {},
               {
                   valuesNode(0),
                   valuesNode(1),
                   valuesNode(2),
               },
               // Drop all columns.
               {})
           .singleAggregation({}, {"count(1)"})
           .planNode();

  task = assertQuery(op, "SELECT 300");
  verifyExchangeSourceOperatorStats(task, 300);
}

TEST_F(LocalPartitionTest, outputLayoutPartition) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return row; }),
          makeFlatVector<int32_t>(100, [](auto /*row*/) { return 123; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return 53 + row; }),
          makeFlatVector<int32_t>(100, [](auto /*row*/) { return 123; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(100, [](auto row) { return -71 + row; }),
          makeFlatVector<int32_t>(100, [](auto /*row*/) { return 123; }),
      }),
  };

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

  auto valuesNode = [&](int index) {
    return PlanBuilder(planNodeIdGenerator).values({vectors[index]}).planNode();
  };

  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {0},
              {
                  valuesNode(0),
                  valuesNode(1),
                  valuesNode(2),
              },
              // Change column order: (c0, c1) -> (c1, c0).
              {1, 0})
          .partialAggregation({}, {"count(1)", "min(c0)", "max(c1)"})
          .planNode();

  auto task = OperatorTestBase::assertQuery(
      params, "VALUES (146, -71, 123), (154, -70, 123)");
  verifyExchangeSourceOperatorStats(task, 300);

  planNodeIdGenerator->reset();
  params.planNode =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {0},
              {
                  valuesNode(0),
                  valuesNode(1),
                  valuesNode(2),
              },
              // Drop column: (c0, c1) -> (c1).
              {1})
          .partialAggregation({}, {"count(1)", "min(c1)", "max(c1)"})
          .planNode();

  task = OperatorTestBase::assertQuery(
      params, "VALUES (146, 123, 123), (154, 123, 123)");
  verifyExchangeSourceOperatorStats(task, 300);

  planNodeIdGenerator->reset();
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .localPartition(
                            {0},
                            {
                                valuesNode(0),
                                valuesNode(1),
                                valuesNode(2),
                            },
                            // Drop all columns.
                            {})
                        .partialAggregation({}, {"count(1)"})
                        .planNode();

  task = OperatorTestBase::assertQuery(params, "VALUES (146), (154)");
  verifyExchangeSourceOperatorStats(task, 300);
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

  auto rowType = getRowType(vectors[0]);

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
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
                    {0},
                    {PlanBuilder(planNodeIdGenerator)
                         .localPartition(
                             {0, 1},
                             {
                                 tableScanNode(),
                                 tableScanNode(),
                                 tableScanNode(),
                             })
                         .partialAggregation({0, 1}, {"count(1)"})
                         .planNode()})
                .partialAggregation({0}, {"count(1)", "sum(a0)"})
                .planNode();

  createDuckDbTable(vectors);

  CursorParameters params;
  params.planNode = op;
  params.maxDrivers = 2;

  uint32_t fileIndex = 0;
  auto task = ::assertQuery(
      params,
      [&](exec::Task* task) {
        while (fileIndex < filePaths.size()) {
          auto planNodeId = scanNodeIds[fileIndex];
          addSplit(task, planNodeId, makeHiveSplit(filePaths[fileIndex]->path));
          task->noMoreSplits(planNodeId);
          ++fileIndex;
        }
      },
      "SELECT c0, count(1), sum(cnt) FROM ("
      "   SELECT c0, c1, count(1) as cnt FROM tmp GROUP BY 1, 2"
      ") t GROUP BY 1",
      duckDbQueryRunner_);
}
