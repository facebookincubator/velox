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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
using namespace facebook::velox::common::testutil;
namespace {

class LocalPartitionTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::registerCudf();
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
      writeToFile(filePaths[i]->getPath(), vectors[i]);
    }
    return filePaths;
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
                .singleAggregation({}, {"min(c0)", "max(c0)"})
                .planNode();

  auto task = assertQuery(op, "SELECT -71, 152");

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
           .singleAggregation({}, {"min(c0)", "max(c0)"})
           .planNode();

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  task = queryBuilder.assertResults("SELECT -71, 152");
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
    return builder.partialAggregation({"c0"}, {"max(c0)"}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {
                        scanAggNode(),
                        scanAggNode(),
                        scanAggNode(),
                    })
                .finalAggregation()
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  queryBuilder.maxDrivers(2);
  queryBuilder.config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2");

  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  auto task =
      queryBuilder.assertResults("SELECT c0, max(c0) FROM tmp GROUP BY 1");
}

TEST_F(LocalPartitionTest, unionAllLocalExchange) {
  auto data1 = makeRowVector({"d0"}, {makeFlatVector<StringView>({"x"})});
  auto data2 = makeRowVector({"e0"}, {makeFlatVector<StringView>({"y"})});
  for (bool serialExecutionMode : {false, true}) {
    SCOPED_TRACE(fmt::format("serialExecutionMode {}", serialExecutionMode));
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    // applyTestParameters(queryBuilder);
    queryBuilder.serialExecution(serialExecutionMode)
        .plan(PlanBuilder(planNodeIdGenerator)
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
                  .planNode())
        .assertResults(
            "SELECT length(c0) FROM ("
            "   SELECT * FROM (VALUES ('x')) as t1(c0) UNION ALL "
            "   SELECT * FROM (VALUES ('y')) as t2(c0)"
            ")");
  }
}

TEST_F(LocalPartitionTest, roundRobinMultipleBatches) {
  // Test round robin with multiple input batches to verify counter continuity
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 5)}), // Batch 1: 5 rows
      makeRowVector({makeFlatSequence<int32_t>(5, 5)}), // Batch 2: 5 rows
      makeRowVector({makeFlatSequence<int32_t>(10, 5)}), // Batch 3: 5 rows
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
          .planNode();

  // Total 15 rows, 3 partitions: each gets 5 rows
  auto task = assertQuery(op, "SELECT 15, 0, 14");

  auto stats = task->taskStats();
  ASSERT_EQ(stats.numTotalDrivers, 2);
}

TEST_F(LocalPartitionTest, roundRobinEmptyInput) {
  // Test with empty input
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatVector<int32_t>({})}), // Empty vector
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .singleAggregation({}, {"count(1)"})
          .planNode();

  // Should handle empty input gracefully
  auto task = assertQuery(op, "SELECT 0");

  auto stats = task->taskStats();
  ASSERT_EQ(stats.numTotalDrivers, 2);
}

TEST_F(LocalPartitionTest, roundRobinMultipleSources) {
  // Test round robin with multiple input sources
  std::vector<RowVectorPtr> vectors1 = {
      makeRowVector({makeFlatSequence<int32_t>(0, 5)}),
  };

  std::vector<RowVectorPtr> vectors2 = {
      makeRowVector({makeFlatSequence<int32_t>(5, 5)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin({
              PlanBuilder(planNodeIdGenerator).values(vectors1).planNode(),
              PlanBuilder(planNodeIdGenerator).values(vectors2).planNode(),
          })
          .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
          .planNode();

  // Total 10 rows from 2 sources, 3 partitions
  auto task = assertQuery(op, "SELECT 10, 0, 9");

  auto stats = task->taskStats();
  ASSERT_EQ(stats.numTotalDrivers, 3);
}

TEST_F(LocalPartitionTest, roundRobinWithAggregation) {
  // Test round robin followed by aggregation
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatSequence<int32_t>(0, 20), // 20 rows
          makeFlatSequence<int32_t>(0, 4, 20), // Values 0,1,2,3 repeating
      }),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .singleAggregation({}, {"count(1)", "sum(c0)"})
          .planNode();

  auto task = assertQuery(op, "SELECT 20, 190");

  auto stats = task->taskStats();
  ASSERT_EQ(stats.numTotalDrivers, 2);
}

TEST_F(LocalPartitionTest, roundRobinWithTableScan) {
  // Test round robin with table scan sources
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 30)}),
      makeRowVector({makeFlatSequence<int32_t>(0, 12)}),
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

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartitionRoundRobin({
                    tableScanNode(),
                    tableScanNode(),
                })
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  queryBuilder.maxDrivers(3); // 3 partitions
  queryBuilder.config(core::QueryConfig::kMaxLocalExchangePartitionCount, "3");

  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  // 1st partition gets all rows
  // 2nd partition gets zero rows
  // 3rd partition gets zero rows

  auto task = queryBuilder.assertResults(
      "SELECT 42, 0, 29 UNION ALL SELECT 0, NULL, NULL UNION ALL SELECT 0, NULL, NULL");

  auto stats = task->taskStats();
  ASSERT_EQ(stats.numTotalDrivers, 9);
}

TEST_F(LocalPartitionTest, roundRobinAllCombinations) {
  // 6 value vectors in different scenarios.
  // Test all possible round robin distribution configurations:
  for (auto [num_sources, num_partitions] :
       {std::pair{3, 1},
        {2, 3},
        {3, 2},
        {1, 3},
        {2, 2},
        {2, 5},
        {1, 5},
        {5, 2},
        {5, 3}}) {
    // Test to verify actual distribution pattern
    std::vector<RowVectorPtr> vectors = {
        makeRowVector({makeFlatSequence<int32_t>(0, 12)}), // 12 rows
        makeRowVector({makeFlatSequence<int32_t>(0, 4)}), // 4 rows
        makeRowVector({makeFlatSequence<int32_t>(0, 40)}), // 40 rows
        makeRowVector({makeFlatSequence<int32_t>(0, 8)}), // 8 rows
        makeRowVector({makeFlatSequence<int32_t>(0, 6)}), // 6 rows
        makeRowVector({makeFlatSequence<int32_t>(0, 25)}), // 25 rows
    };
    auto expectedTotalRows = std::accumulate(
        vectors.begin(),
        vectors.end(),
        0,
        [](int sum, const RowVectorPtr& vector) {
          return sum + vector->size();
        });
    std::vector<int> expected_partition_counts(num_partitions, 0);
    for (int i = 0; i < vectors.size(); ++i) {
      expected_partition_counts[i % num_partitions] += 1;
    }

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto make_op = [&](int num_sources) {
      std::vector<core::PlanNodePtr> sources;
      for (size_t i = 0; i < num_sources; ++i) {
        auto num_vectors = (vectors.size() + num_sources - 1) / num_sources;
        auto start = std::min(i * num_vectors, vectors.size());
        auto end = std::min(start + num_vectors, vectors.size());
        std::vector<RowVectorPtr> source_vectors(
            vectors.begin() + start, vectors.begin() + end);
        if (source_vectors.empty()) {
          source_vectors.push_back(
              makeRowVector({makeFlatSequence<int32_t>(0, 0)}));
        }
        sources.push_back(
            PlanBuilder(planNodeIdGenerator).values(source_vectors).planNode());
      }
      return PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(sources)
          .planNode();
    };

    auto op = make_op(num_sources);

    // Use cursor to examine actual distribution
    CursorParameters params;
    params.queryConfigs.insert(
        {cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1"});
    params.queryConfigs.insert(
        {core::QueryConfig::kMaxLocalExchangePartitionCount,
         std::to_string(num_partitions)});

    params.planNode = op;
    params.maxDrivers = num_partitions;
    params.copyResult = false;

    auto cursor = TaskCursor::create(params);

    std::vector<int> partitionCounts(num_partitions, 0);
    int partition = 0;
    int totalRows = 0;

    while (cursor->moveNext()) {
      auto* batch = cursor->current()->as<RowVector>();
      partitionCounts[partition++ % num_partitions] += 1;
      totalRows += batch->size();
    }

    EXPECT_EQ(totalRows, expectedTotalRows)
        << "for {" << num_sources << ", " << num_partitions << "}";
    // With round robin, each batch should go into separate partitions.
    for (int i = 0; i < num_partitions; ++i) {
      auto expected_count = expected_partition_counts[i];
      ASSERT_EQ(partitionCounts[i], expected_count)
          << "for i " << i << ", {" << num_sources << ", " << num_partitions
          << "}";
    }
  }
}

TEST_F(LocalPartitionTest, roundRobinDistributionVerification) {
  // Test to verify actual distribution pattern
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 12)}), // 12 rows
      makeRowVector({makeFlatSequence<int32_t>(0, 4)}), // 4 rows
      makeRowVector({makeFlatSequence<int32_t>(0, 40)}), // 40 rows
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .planNode();

  // Use cursor to examine actual distribution
  CursorParameters params;
  params.queryConfigs.insert(
      {cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1"});
  params.queryConfigs.insert(
      {core::QueryConfig::kMaxLocalExchangePartitionCount, "3"});

  params.planNode = op;
  params.maxDrivers = 3;
  params.copyResult = false;

  auto cursor = TaskCursor::create(params);

  std::vector<int> partitionCounts(3, 0);
  int partition = 0;
  int totalRows = 0;

  while (cursor->moveNext()) {
    auto* batch = cursor->current()->as<RowVector>();
    partitionCounts[partition++ % 3] += 1;
    totalRows += batch->size();
  }

  ASSERT_EQ(totalRows, 56);
  // With round robin, each batch should go into separate partitions.
  ASSERT_EQ(partitionCounts[0], 1);
  ASSERT_EQ(partitionCounts[1], 1);
  ASSERT_EQ(partitionCounts[2], 1);
}

} // namespace
} // namespace facebook::velox::exec::test
