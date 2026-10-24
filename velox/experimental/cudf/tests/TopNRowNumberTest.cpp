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
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class TopNRowNumberTest : public OperatorTestBase {
 public:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
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

  void assertCpuAndGpuResults(const core::PlanNodePtr& plan) {
    cudf_velox::unregisterCudf();
    auto expected = AssertQueryBuilder(plan).copyResults(pool());
    cudf_velox::registerCudf();
    auto task = AssertQueryBuilder(plan).assertResults(expected);
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

TEST_F(TopNRowNumberTest, rank) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, 1, 2, std::nullopt, 1, 2, std::nullopt, 1, 2, 3, 3, 1}),
      makeNullableFlatVector<int64_t>(
          {30, 20, 30, 10, 10, 20, 10, 10, 20, std::nullopt, std::nullopt, 20}),
      makeFlatVector<int64_t>(12, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  // Better keys and peers arrive in later batches. Keep every peer at the
  // cutoff, even when their number exceeds the limit.
  for (const auto& function : {"rank", "dense_rank"}) {
    for (bool partitioned : {false, true}) {
      for (bool generateRowNumber : {false, true}) {
        for (int32_t limit : {1, 2, 3, 20}) {
          for (auto numBatches : {1, 4}) {
            SCOPED_TRACE(
                fmt::format(
                    "function={}, partitioned={}, generate={}, limit={}, batches={}",
                    function,
                    partitioned,
                    generateRowNumber,
                    limit,
                    numBatches));
            auto plan = PlanBuilder()
                            .values(split(data, numBatches))
                            .topNRank(
                                function,
                                partitioned ? std::vector<std::string>{"c0"}
                                            : std::vector<std::string>{},
                                {"c1"},
                                limit,
                                generateRowNumber)
                            .planNode();
            assertGpuTopNRowNumber(
                plan,
                fmt::format(
                    "SELECT {} FROM (SELECT *, {}() OVER ({} ORDER BY c1) "
                    "AS row_number FROM tmp) WHERE row_number <= {}",
                    generateRowNumber ? "*" : "c0, c1, c2",
                    function,
                    partitioned ? "PARTITION BY c0" : "",
                    limit));
          }
        }
      }
    }
  }
}

TEST_F(TopNRowNumberTest, rankMultipleKeys) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, 2, 1, 1, std::nullopt, 2, 1, std::nullopt, 2, 1, 2, 1}),
      makeNullableFlatVector<int64_t>(
          {10, 20, 10, std::nullopt, 10, 20, 10, 10, 20, std::nullopt, 30, 10}),
      makeNullableFlatVector<std::string>(
          {"b",
           "b",
           "a",
           "a",
           "a",
           "b",
           "a",
           "a",
           std::nullopt,
           "a",
           "c",
           "c"}),
      makeFlatVector<int64_t>(12, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  for (const auto& function : {"rank", "dense_rank"}) {
    for (bool partitioned : {false, true}) {
      for (bool ascending : {false, true}) {
        for (bool nullsFirst : {false, true}) {
          for (int32_t limit : {1, 2, 3}) {
            SCOPED_TRACE(
                fmt::format(
                    "function={}, partitioned={}, ascending={}, nullsFirst={}, limit={}",
                    function,
                    partitioned,
                    ascending,
                    nullsFirst,
                    limit));
            auto ordering = fmt::format(
                "c1 {} NULLS {}",
                ascending ? "ASC" : "DESC",
                nullsFirst ? "FIRST" : "LAST");
            auto plan = PlanBuilder()
                            .values(split(data, 4))
                            .topNRank(
                                function,
                                partitioned ? std::vector<std::string>{"c0"}
                                            : std::vector<std::string>{},
                                {ordering, "c2 DESC NULLS LAST"},
                                limit,
                                true)
                            .planNode();
            assertGpuTopNRowNumber(
                plan,
                fmt::format(
                    "SELECT * FROM (SELECT *, {}() OVER ({} ORDER BY {}, "
                    "c2 DESC NULLS LAST) AS row_number FROM tmp) "
                    "WHERE row_number <= {}",
                    function,
                    partitioned ? "PARTITION BY c0" : "",
                    ordering,
                    limit));
          }
        }
      }
    }

    auto plan = PlanBuilder()
                    .values(split(data, 4))
                    .topNRank(function, {"c0", "c2"}, {"c1 DESC"}, 2, true)
                    .planNode();
    assertGpuTopNRowNumber(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, {}() OVER (PARTITION BY c0, c2 "
            "ORDER BY c1 DESC) AS row_number FROM tmp) WHERE row_number <= 2",
            function));
  }
}

TEST_F(TopNRowNumberTest, rankSpecialValues) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto floatingData = makeRowVector({
      makeNullableFlatVector<double>(
          {nan, 0.0, 1.0, -nan, -0.0, 1.0, nan, -0.0, -nan, std::nullopt}),
      makeNullableFlatVector<double>(
          {nan, 0.0, nan, -nan, -0.0, 3.0, 2.0, std::nullopt, -nan, 0.0}),
      makeFlatVector<int64_t>(10, [](auto row) { return row; }),
  });

  auto sortingKeys = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, 1, std::nullopt, 2, 1, 2, 1, 2, std::nullopt, 1}),
      makeNullableFlatVector<std::string>(
          {"a", "a", "b", "c", std::nullopt, "c", "a", "b", "b", "a"}),
  });
  sortingKeys->setNull(2, true);
  sortingKeys->setNull(8, true);
  auto nestedData = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 1, 1, 1, 2, 1, 2, 1}),
      sortingKeys,
      makeFlatVector<int64_t>(10, [](auto row) { return row; }),
  });

  for (const auto& data : {floatingData, nestedData}) {
    for (const auto& function : {"rank", "dense_rank"}) {
      for (const auto& ordering :
           {"c1 ASC NULLS FIRST", "c1 DESC NULLS LAST"}) {
        for (int32_t limit : {1, 2, 5}) {
          SCOPED_TRACE(
              fmt::format(
                  "type={}, function={}, ordering={}, limit={}",
                  data->type()->toString(),
                  function,
                  ordering,
                  limit));
          assertCpuAndGpuResults(
              PlanBuilder()
                  .values(split(data, 5))
                  .topNRank(function, {"c0"}, {ordering}, limit, true)
                  .planNode());
        }
      }
    }
  }
}

TEST_F(TopNRowNumberTest, allPeers) {
  constexpr vector_size_t kNumRows = 10'000;
  auto data = makeRowVector({
      makeFlatVector<int64_t>(kNumRows, [](auto /*row*/) { return 1; }),
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; }),
  });
  for (const auto& function : {"rank", "dense_rank"}) {
    for (bool generateRowNumber : {false, true}) {
      SCOPED_TRACE(
          fmt::format("function={}, generate={}", function, generateRowNumber));
      auto plan = PlanBuilder()
                      .values(split(data, 10))
                      .topNRank(function, {}, {"c0"}, 1, generateRowNumber)
                      .planNode();
      auto task = AssertQueryBuilder(plan).assertTypeAndNumRows(
          plan->outputType(), kNumRows);
      ASSERT_TRUE(wasCudfTopNRowNumberUsed(task));
      ASSERT_FALSE(wasCpuTopNRowNumberUsed(task));
      assertCpuAndGpuResults(plan);
    }
  }
}

TEST_F(TopNRowNumberTest, emptyInput) {
  auto data = makeRowVector({makeFlatVector<int64_t>({})});
  for (const auto& function : {"row_number", "rank", "dense_rank"}) {
    for (bool generateRowNumber : {false, true}) {
      SCOPED_TRACE(
          fmt::format("function={}, generate={}", function, generateRowNumber));
      auto plan = PlanBuilder()
                      .values({data})
                      .topNRank(function, {}, {"c0"}, 1, generateRowNumber)
                      .planNode();
      auto task = AssertQueryBuilder(plan).assertEmptyResults();
      ASSERT_TRUE(wasCudfTopNRowNumberUsed(task));
      ASSERT_FALSE(wasCpuTopNRowNumberUsed(task));
    }
  }
}

TEST_F(TopNRowNumberTest, unsupportedRankKeys) {
  const std::vector<TypePtr> unsupportedSortingTypes = {
      ARRAY(BIGINT()),
      MAP(BIGINT(), BIGINT()),
      ROW("nested", ARRAY(BIGINT())),
      TIMESTAMP_WITH_TIME_ZONE(),
      ROW("nested", TIMESTAMP_WITH_TIME_ZONE()),
  };
  for (const auto& function : {"rank", "dense_rank"}) {
    for (const auto& type : unsupportedSortingTypes) {
      auto data = BaseVector::create<RowVector>(
          ROW({{"c0", BIGINT()}, {"c1", type}}), 0, pool());
      for (bool multipleKeys : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "function={}, type={}, multipleKeys={}",
                function,
                type->toString(),
                multipleKeys));
        auto plan = PlanBuilder()
                        .values({data})
                        .topNRank(
                            function,
                            {},
                            multipleKeys ? std::vector<std::string>{"c0", "c1"}
                                         : std::vector<std::string>{"c1"},
                            1,
                            true)
                        .planNode();
        ASSERT_FALSE(
            cudf_velox::CudfTopNRowNumber::canRunOnGPU(
                *std::dynamic_pointer_cast<const core::TopNRowNumberNode>(
                    plan)));
      }
    }

    for (const auto& type : std::vector<TypePtr>{
             TIMESTAMP_WITH_TIME_ZONE(),
             ROW("nested", TIMESTAMP_WITH_TIME_ZONE())}) {
      auto data = BaseVector::create<RowVector>(
          ROW({{"c0", type}, {"c1", BIGINT()}}), 0, pool());
      auto plan = PlanBuilder()
                      .values({data})
                      .topNRank(function, {"c0"}, {"c1"}, 1, true)
                      .planNode();
      ASSERT_FALSE(
          cudf_velox::CudfTopNRowNumber::canRunOnGPU(
              *std::dynamic_pointer_cast<const core::TopNRowNumberNode>(plan)));
    }
  }

  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 1}),
      makeArrayVector<int64_t>({{2}, {1}, {2}, {3}}),
  });
  createDuckDbTable({data});
  for (const auto& function : {"rank", "dense_rank"}) {
    auto plan = PlanBuilder()
                    .values({data})
                    .topNRank(function, {"c0"}, {"c1"}, 2, true)
                    .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).copyResults(pool()),
        "Replacement with cuDF operator failed");
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = true;
    cudf_velox::registerCudf();
    auto task = assertQuery(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, {}() OVER (PARTITION BY c0 "
            "ORDER BY c1) AS row_number FROM tmp) WHERE row_number <= 2",
            function));
    ASSERT_FALSE(wasCudfTopNRowNumberUsed(task));
    ASSERT_TRUE(wasCpuTopNRowNumberUsed(task));
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }
}
