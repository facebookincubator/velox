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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class TopNRowNumberTest : public OperatorTestBase {
 public:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

 protected:
  static bool wasCudfTopNRowNumberUsed(
      const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfTopNRowNumber") {
          return true;
        }
      }
    }
    return false;
  }

  static bool wasCpuTopNRowNumberUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "TopNRowNumber") {
          return true;
        }
      }
    }
    return false;
  }

  void assertGpuTopNRowNumber(
      const core::PlanNodePtr& plan,
      const std::string& duckDbSql) {
    auto task = assertQuery(plan, duckDbSql);
    ASSERT_TRUE(wasCudfTopNRowNumberUsed(task));
    ASSERT_FALSE(wasCpuTopNRowNumberUsed(task));
  }
};

TEST_F(TopNRowNumberTest, basic) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 1, 2, 1}),
      makeFlatVector<int64_t>({77, 66, 55, 44, 33, 22, 11}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70}),
  });
  createDuckDbTable({data});

  auto testLimit = [&](int32_t limit) {
    SCOPED_TRACE(fmt::format("limit={}", limit));

    auto plan = PlanBuilder()
                    .values({data})
                    .topNRowNumber({"c0"}, {"c1"}, limit, true)
                    .planNode();
    assertGpuTopNRowNumber(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
            "WHERE row_number <= {}",
            limit));

    plan = PlanBuilder()
               .values({data})
               .topNRowNumber({"c0"}, {"c1"}, limit, false)
               .planNode();
    assertGpuTopNRowNumber(
        plan,
        fmt::format(
            "SELECT c0, c1, c2 FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
            "WHERE row_number <= {}",
            limit));

    plan = PlanBuilder()
               .values({data})
               .topNRowNumber({}, {"c1"}, limit, true)
               .planNode();
    assertGpuTopNRowNumber(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over (order by c1) as row_number FROM tmp) "
            "WHERE row_number <= {}",
            limit));
  };

  testLimit(1);
  testLimit(2);
  testLimit(3);
  testLimit(5);
}

TEST_F(TopNRowNumberTest, basicWithPeers) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1}),
      makeFlatVector<int64_t>({33, 11, 55, 44, 11, 22, 11, 11, 11, 33, 33}),
      makeFlatVector<int64_t>({10, 50, 30, 40, 50, 60, 50, 50, 50, 10, 10}),
  });
  createDuckDbTable({data});

  auto testLimit = [&](int32_t limit) {
    SCOPED_TRACE(fmt::format("limit={}", limit));

    auto plan = PlanBuilder()
                    .values({data})
                    .topNRowNumber({"c0"}, {"c1"}, limit, true)
                    .planNode();
    assertGpuTopNRowNumber(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
            "WHERE row_number <= {}",
            limit));
  };

  testLimit(1);
  testLimit(2);
  testLimit(3);
  testLimit(5);
}

TEST_F(TopNRowNumberTest, descendingSort) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 2, 2, 2}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500, 600}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"c0"}, {"c1 DESC"}, 2, true)
                  .planNode();
  assertGpuTopNRowNumber(
      plan,
      "SELECT * FROM (SELECT *, row_number() over (partition by c0 order by c1 DESC) as row_number FROM tmp) "
      "WHERE row_number <= 2");
}

TEST_F(TopNRowNumberTest, multiBatch) {
  const vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 3; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            batchSize,
            [&](vector_size_t row) { return (batch * batchSize + row) % 5; }),
        makeFlatVector<int64_t>(
            batchSize,
            [&](vector_size_t row) { return batch * batchSize + row; }),
        makeFlatVector<int64_t>(
            batchSize, [&](vector_size_t row) { return row; }),
    }));
  }
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .topNRowNumber({"c0"}, {"c1"}, 3, false)
                  .planNode();
  assertGpuTopNRowNumber(
      plan,
      "SELECT c0, c1, c2 FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
      "WHERE row_number <= 3");
}

TEST_F(TopNRowNumberTest, multiBatchWithRowNumber) {
  // Same shape as multiBatch, but with generateRowNumber=true so the
  // row_number column must be recomputed correctly across the incremental
  // merge/prune steps in CudfTopNRowNumber::mergeAndPruneCandidates, not just
  // the filtered row set.
  const vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 3; ++batch) {
    vectors.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            batchSize,
            [&](vector_size_t row) { return (batch * batchSize + row) % 5; }),
        makeFlatVector<int64_t>(
            batchSize,
            [&](vector_size_t row) { return batch * batchSize + row; }),
        makeFlatVector<int64_t>(
            batchSize, [&](vector_size_t row) { return row; }),
    }));
  }
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .topNRowNumber({"c0"}, {"c1"}, 3, true)
                  .planNode();
  assertGpuTopNRowNumber(
      plan,
      "SELECT * FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
      "WHERE row_number <= 3");
}

TEST_F(TopNRowNumberTest, manySmallBatchesStaggeredPartitions) {
  // Exercises the per-batch merge/prune path many times (one merge per
  // batch) with partitions that only start appearing partway through the
  // stream, and with candidate state from earlier batches getting displaced
  // by later, better-ranked rows within the same partition.
  const vector_size_t batchSize = 20;
  const int32_t numBatches = 15;
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < numBatches; ++batch) {
    // Partition 'batch % 4' only starts contributing rows once 'batch' is at
    // least that value, e.g. partition 3 first appears in batch 3.
    vectors.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            batchSize, [&](vector_size_t row) { return (batch + row) % 4; }),
        // Descending c1 so later batches (smaller multiplier applied via
        // batch-dependent offset) sometimes outrank earlier candidates,
        // forcing candidates_ to be displaced during merges.
        makeFlatVector<int64_t>(
            batchSize,
            [&](vector_size_t row) {
              return (numBatches - batch) * 1000 + row;
            }),
        makeFlatVector<int64_t>(
            batchSize, [&](vector_size_t row) { return row; }),
    }));
  }
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .topNRowNumber({"c0"}, {"c1"}, 4, true)
                  .planNode();
  assertGpuTopNRowNumber(
      plan,
      "SELECT * FROM (SELECT *, row_number() over (partition by c0 order by c1) as row_number FROM tmp) "
      "WHERE row_number <= 4");
}

TEST_F(TopNRowNumberTest, rankFallsBackToCpu) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeFlatVector<int64_t>({100, 200, 300, 400}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .topNRank("rank", {"c0"}, {"c1"}, 2, true)
                  .planNode();
  auto task = assertQuery(
      plan,
      "SELECT * FROM (SELECT *, rank() over (partition by c0 order by c1) as row_number FROM tmp) "
      "WHERE row_number <= 2");
  ASSERT_FALSE(wasCudfTopNRowNumberUsed(task));
  ASSERT_TRUE(wasCpuTopNRowNumberUsed(task));
}
