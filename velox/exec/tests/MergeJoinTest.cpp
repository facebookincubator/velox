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

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/tests/FilterBuilder.h"

#if __has_include("filesystem")
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class MergeJoinTest : public HiveConnectorTestBase {
 protected:
  using OperatorTestBase::assertQuery;
  static constexpr const char* kWriter = "MergeJoinTest.Writer";

  static CursorParameters makeCursorParameters(
      const std::shared_ptr<const core::PlanNode>& planNode,
      uint32_t preferredOutputBatchSize) {
    auto queryCtx = core::QueryCtx::createForTest();
    queryCtx->setConfigOverridesUnsafe(
        {{core::QueryConfig::kCreateEmptyFiles, "true"}});

    CursorParameters params;
    params.planNode = planNode;
    params.queryCtx = core::QueryCtx::createForTest();
    params.queryCtx->setConfigOverridesUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchSize,
          std::to_string(preferredOutputBatchSize)}});
    return params;
  }

  template <typename T>
  void testJoin(
      std::function<T(vector_size_t /*row*/)> leftKeyAt,
      std::function<T(vector_size_t /*row*/)> rightKeyAt) {
    // Single batch on the left and right sides of the join.
    {
      auto leftKeys = makeFlatVector<T>(1'234, leftKeyAt);
      auto rightKeys = makeFlatVector<T>(1'234, rightKeyAt);

      testJoin({leftKeys}, {rightKeys});
    }

    // Multiple batches on one side. Single batch on the other side.
    {
      std::vector<VectorPtr> leftKeys = {
          makeFlatVector<T>(1024, leftKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return leftKeyAt(1024 + row); }),
      };
      std::vector<VectorPtr> rightKeys = {makeFlatVector<T>(2048, rightKeyAt)};

      testJoin(leftKeys, rightKeys);

      // Swap left and right side keys.
      testJoin(rightKeys, leftKeys);
    }

    // Multiple batches on each side.
    {
      std::vector<VectorPtr> leftKeys = {
          makeFlatVector<T>(512, leftKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return leftKeyAt(512 + row); }),
          makeFlatVector<T>(
              16, [&](auto row) { return leftKeyAt(512 + 1024 + row); }),
      };
      std::vector<VectorPtr> rightKeys = {
          makeFlatVector<T>(123, rightKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return rightKeyAt(123 + row); }),
          makeFlatVector<T>(
              1234, [&](auto row) { return rightKeyAt(123 + 1024 + row); }),
      };

      testJoin(leftKeys, rightKeys);

      // Swap left and right side keys.
      testJoin(rightKeys, leftKeys);
    }
  }

  void testJoin(
      const std::vector<VectorPtr>& leftKeys,
      const std::vector<VectorPtr>& rightKeys) {
    std::vector<RowVectorPtr> left;
    left.reserve(leftKeys.size());
    vector_size_t startRow = 0;
    for (const auto& key : leftKeys) {
      auto payload = makeFlatVector<int32_t>(
          key->size(), [startRow](auto row) { return (startRow + row) * 10; });
      left.push_back(makeRowVector({key, payload}));
      startRow += key->size();
    }

    std::vector<RowVectorPtr> right;
    right.reserve(rightKeys.size());
    startRow = 0;
    for (const auto& key : rightKeys) {
      auto payload = makeFlatVector<int32_t>(
          key->size(), [startRow](auto row) { return (startRow + row) * 20; });
      right.push_back(makeRowVector({key, payload}));
      startRow += key->size();
    }

    createDuckDbTable("t", left);
    createDuckDbTable("u", right);

    // Test INNER join.
    auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values(left)
                    .mergeJoin(
                        {"c0"},
                        {"u_c0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values(right)
                            .project({"c0 AS u_c0", "c1 AS u_c1"})
                            .planNode(),
                        "",
                        {"c0", "c1", "u_c1"},
                        core::JoinType::kInner)
                    .planNode();

    // Use very small output batch size.
    assertQuery(
        makeCursorParameters(plan, 16),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Use regular output batch size.
    assertQuery(
        makeCursorParameters(plan, 1024),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Use very large output batch size.
    assertQuery(
        makeCursorParameters(plan, 10'000),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Test LEFT join.
    planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
    plan = PlanBuilder(planNodeIdGenerator)
               .values(left)
               .mergeJoin(
                   {"c0"},
                   {"u_c0"},
                   PlanBuilder(planNodeIdGenerator)
                       .values(right)
                       .project({"c0 as u_c0", "c1 as u_c1"})
                       .planNode(),
                   "",
                   {"c0", "c1", "u_c1"},
                   core::JoinType::kLeft)
               .planNode();

    // Use very small output batch size.
    assertQuery(
        makeCursorParameters(plan, 16),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");

    // Use regular output batch size.
    assertQuery(
        makeCursorParameters(plan, 1024),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");

    // Use very large output batch size.
    assertQuery(
        makeCursorParameters(plan, 10'000),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");
  }
};

TEST_F(MergeJoinTest, oneToOneAllMatch) {
  testJoin<int32_t>([](auto row) { return row; }, [](auto row) { return row; });
}

TEST_F(MergeJoinTest, someDontMatch) {
  testJoin<int32_t>(
      [](auto row) { return row % 5 == 0 ? row - 1 : row; },
      [](auto row) { return row % 7 == 0 ? row - 1 : row; });
}

TEST_F(MergeJoinTest, fewMatch) {
  testJoin<int32_t>(
      [](auto row) { return row * 5; }, [](auto row) { return row * 7; });
}

TEST_F(MergeJoinTest, duplicateMatch) {
  testJoin<int32_t>(
      [](auto row) { return row / 2; }, [](auto row) { return row / 3; });
}

TEST_F(MergeJoinTest, allRowsMatch) {
  std::vector<VectorPtr> leftKeys = {
      makeFlatVector<int32_t>(2, [](auto /* row */) { return 5; }),
      makeFlatVector<int32_t>(3, [](auto /* row */) { return 5; }),
      makeFlatVector<int32_t>(4, [](auto /* row */) { return 5; }),
  };
  std::vector<VectorPtr> rightKeys = {
      makeFlatVector<int32_t>(7, [](auto /* row */) { return 5; })};

  testJoin(leftKeys, rightKeys);

  testJoin(rightKeys, leftKeys);
}

TEST_F(MergeJoinTest, aggregationOverJoin) {
  auto left =
      makeRowVector({"t_c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto right = makeRowVector({"u_c0"}, {makeFlatVector<int32_t>({2, 4, 6})});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left})
          .mergeJoin(
              {"t_c0"},
              {"u_c0"},
              PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
              "",
              {"t_c0", "u_c0"},
              core::JoinType::kInner)
          .singleAggregation({}, {"count(1)"})
          .planNode();

  auto result = readSingleValue(plan);
  ASSERT_FALSE(result.isNull());
  ASSERT_EQ(2, result.value<int64_t>());
}

TEST_F(MergeJoinTest, nonFirstJoinKeys) {
  auto left = makeRowVector(
      {"t_data", "t_key"},
      {
          makeFlatVector<int32_t>({50, 40, 30, 20, 10}),
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      });
  auto right = makeRowVector(
      {"u_data", "u_key"},
      {
          makeFlatVector<int32_t>({23, 22, 21}),
          makeFlatVector<int32_t>({2, 4, 6}),
      });

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left})
          .mergeJoin(
              {"t_key"},
              {"u_key"},
              PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
              "",
              {"t_key", "t_data", "u_data"},
              core::JoinType::kInner)
          .planNode();

  assertQuery(plan, "VALUES (2, 40, 23), (4, 20, 22)");
}

TEST_F(MergeJoinTest, innerJoinFilter) {
  vector_size_t size = 1'000;
  // Join keys on the left side: 0, 10, 20,..
  // Payload on the left side: 0, 1, 2, 3,..
  auto left = makeRowVector(
      {"t_c0", "t_c1"},
      {
          makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
          makeFlatVector<int64_t>(
              size, [](auto row) { return row; }, nullEvery(13)),
      });

  // Join keys on the right side: 0, 5, 10, 15, 20,..
  // Payload on the right side: 0, 1, 2, 3, 4, 5, 6, 0, 1, 2,..
  auto right = makeRowVector(
      {"u_c0", "u_c1"},
      {
          makeFlatVector<int32_t>(size, [](auto row) { return row * 5; }),
          makeFlatVector<int64_t>(
              size, [](auto row) { return row % 7; }, nullEvery(17)),
      });

  createDuckDbTable("t", {left});
  createDuckDbTable("u", {right});

  auto plan = [&](const std::string& filter) {
    auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({left})
        .mergeJoin(
            {"t_c0"},
            {"u_c0"},
            PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
            filter,
            {"t_c0", "u_c0", "u_c1"},
            core::JoinType::kInner)
        .planNode();
  };

  assertQuery(
      plan("(t_c1 + u_c1) % 2 = 0"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c1) % 2 = 0");

  assertQuery(
      plan("(t_c1 + u_c1) % 2 = 1"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c1) % 2 = 1");

  // No rows pass filter.
  assertQuery(
      plan("(t_c1 + u_c1) % 2 < 0"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c1) % 2 < 0");

  // All rows pass filter.
  assertQuery(
      plan("(t_c1 + u_c1) % 2 >= 0"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c1) % 2 >= 0");

  // Filter expressions over join keys.
  assertQuery(
      plan("(t_c0 + u_c1) % 2 = 0"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c0 + u_c1) % 2 = 0");

  assertQuery(
      plan("(t_c1 + u_c0) % 2 = 0"),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c0) % 2 = 0");

  // Very small output batch size.
  assertQuery(
      makeCursorParameters(plan("(t_c1 + u_c1) % 2 = 0"), 16),
      "SELECT t_c0, u_c0, u_c1 FROM t, u WHERE t_c0 = u_c0 AND (t_c1 + u_c1) % 2 = 0");
}

TEST_F(MergeJoinTest, leftJoinFilter) {
  // Each row on the left side has at most one match on the right side.
  auto left = makeRowVector(
      {"t_c0", "t_c1"},
      {
          makeFlatVector<int32_t>({0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}),
          makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
      });

  auto right = makeRowVector(
      {"u_c0", "u_c1"},
      {
          makeFlatVector<int32_t>({0, 10, 20, 30, 40, 50}),
          makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5}),
      });

  createDuckDbTable("t", {left});
  createDuckDbTable("u", {right});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan = [&](const std::string& filter) {
    return PlanBuilder(planNodeIdGenerator)
        .values({left})
        .mergeJoin(
            {"t_c0"},
            {"u_c0"},
            PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
            filter,
            {"t_c0", "t_c1", "u_c1"},
            core::JoinType::kLeft)
        .planNode();
  };

  // Test with different output batch sizes.
  for (auto batchSize : {1, 3, 16}) {
    assertQuery(
        makeCursorParameters(plan("(t_c1 + u_c1) % 2 = 0"), batchSize),
        "SELECT t_c0, t_c1, u_c1 FROM t LEFT JOIN u ON t_c0 = u_c0 AND (t_c1 + u_c1) % 2 = 0");
  }

  // A left-side row with multiple matches on the right side.
  left = makeRowVector(
      {"t_c0", "t_c1"},
      {
          makeFlatVector<int32_t>({5, 10}),
          makeFlatVector<int32_t>({0, 0}),
      });

  right = makeRowVector(
      {"u_c0", "u_c1"},
      {
          makeFlatVector<int32_t>({10, 10, 10, 10, 10, 10}),
          makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5}),
      });

  createDuckDbTable("t", {left});
  createDuckDbTable("u", {right});

  // Test with different filters and output batch sizes.
  for (auto batchSize : {1, 3, 16}) {
    for (auto filter :
         {"t_c1 + u_c1 > 3",
          "t_c1 + u_c1 < 3",
          "t_c1 + u_c1 > 100",
          "t_c1 + u_c1 < 100"}) {
      assertQuery(
          makeCursorParameters(plan(filter), batchSize),
          fmt::format(
              "SELECT t_c0, t_c1, u_c1 FROM t LEFT JOIN u ON t_c0 = u_c0 AND {}",
              filter));
    }
  }
}

// Verify that both left-side and right-side pipelines feeding the merge join
// always run single-threaded.
TEST_F(MergeJoinTest, numDrivers) {
  auto left = makeRowVector({"t_c0"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto right = makeRowVector({"u_c0"}, {makeFlatVector<int32_t>({0, 2, 5})});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left}, true)
          .mergeJoin(
              {"t_c0"},
              {"u_c0"},
              PlanBuilder(planNodeIdGenerator).values({right}, true).planNode(),
              "",
              {"t_c0", "u_c0"},
              core::JoinType::kInner)
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 5;
  params.numResultDrivers = 1;
  auto task = assertQuery(params, "SELECT 2, 2");
  // We have two pipelines in the task and each must have 1 driver.
  EXPECT_EQ(2, task->numTotalDrivers());
  EXPECT_EQ(2, task->numFinishedDrivers());
}

TEST_F(MergeJoinTest, lazyVectors) {
  // a dataset of multiple row groups with multiple columns. We create
  // different dictionary wrappings for different columns and load the
  // rows in scope at different times.  We make 11000 repeats of 300
  // followed by ascending rows. These will hits one 300 from the
  // right side and cover more than one batch, so that we test lazy
  // loading where we buffer multiple batches of input.
  auto leftVectors = makeRowVector(
      {makeFlatVector<int32_t>(
           30'000, [](auto row) { return row < 11000 ? 300 : row; }),
       makeFlatVector<int64_t>(30'000, [](auto row) { return row % 23; }),
       makeFlatVector<int32_t>(30'000, [](auto row) { return row % 31; }),
       makeFlatVector<StringView>(30'000, [](auto row) {
         return StringView(fmt::format("{}   string", row % 43));
       })});

  auto rightVectors = makeRowVector(
      {"rc0", "rc1"},
      {makeFlatVector<int32_t>(10'000, [](auto row) { return row * 3; }),
       makeFlatVector<int64_t>(10'000, [](auto row) { return row % 31; })});

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->path, kWriter, leftVectors);
  createDuckDbTable("t", {leftVectors});

  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->path, kWriter, rightVectors);
  createDuckDbTable("u", {rightVectors});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId leftScanId;
  core::PlanNodeId rightScanId;
  auto op = PlanBuilder(planNodeIdGenerator)
                .tableScan(
                    ROW({"c0", "c1", "c2", "c3"},
                        {INTEGER(), BIGINT(), INTEGER(), VARCHAR()}))
                .capturePlanNodeId(leftScanId)
                .mergeJoin(
                    {"c0"},
                    {"rc0"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(ROW({"rc0", "rc1"}, {INTEGER(), BIGINT()}))
                        .capturePlanNodeId(rightScanId)
                        .planNode(),
                    "c1 + rc1 < 30",
                    {"c0", "rc0", "c1", "rc1", "c2", "c3"})
                .planNode();

  HiveConnectorTestBase::assertQuery(
      op,
      {{rightScanId, {rightFile}}, {leftScanId, {leftFile}}},
      "SELECT c0, rc0, c1, rc1, c2, c3  FROM t, u WHERE t.c0 = u.rc0 and c1 + rc1 < 30");
}

TEST_F(MergeJoinTest, xxx) {
  std::vector<std::string> paths = {
      "/Users/mbasmanova/test_data/day1.dwrf",
      "/Users/mbasmanova/test_data/day2.dwrf",
      "/Users/mbasmanova/test_data/day3.dwrf",
      "/Users/mbasmanova/test_data/day4.dwrf",
      "/Users/mbasmanova/test_data/day5.dwrf",
      "/Users/mbasmanova/test_data/day6.dwrf",
      "/Users/mbasmanova/test_data/day7.dwrf",
  };

  std::vector<std::string> ds = {
      "2022-02-14",
      "2022-02-15",
      "2022-02-16",
      "2022-02-17",
      "2022-02-18",
      "2022-02-19",
      "2022-02-20"};
  std::vector<int64_t> times = {
      1644796800,
      1644883200,
      1644969600,
      1645056000,
      1645142400,
      1645228800,
      1645315200};

  std::vector<core::PlanNodeId> scanIds(paths.size());

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

  auto tableScan = [&](int i) {
    return PlanBuilder(planNodeIdGenerator)
        .tableScan(
            ROW({"user_id", "total_request_cpu_us"}, {BIGINT(), BIGINT()}))
        .capturePlanNodeId(scanIds[i])
        //        .limit(0, 10'000'000, true)
        .project({
            "user_id as metric_unitid",
            "total_request_cpu_us",
            fmt::format("{} as metric_time", times[i]),
                        fmt::format("'{}' as ds", ds[i]),
        })
        .planNode();
  };

  std::vector<std::shared_ptr<const core::PlanNode>> scans;
  for (auto i = 0; i < paths.size(); ++i) {
    scans.push_back(tableScan(i));
  }

  std::unordered_map<core::PlanNodeId, std::vector<std::string>> splits;
  for (auto i = 0; i < paths.size(); ++i) {
    splits.insert({scanIds[i], {paths[i]}});
  }

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localMerge({"metric_unitid", "metric_time"}, scans)
                  .singleAggregation({}, {"count(1)"})
                  .planNode();

  auto task = HiveConnectorTestBase::assertQuery(plan, splits, "SELECT null");
  std::cout << printPlanWithStats(*plan, task->taskStats()) << std::endl;
}

TEST_F(MergeJoinTest, yyy) {
  std::string path = "/Users/mbasmanova/test_data/exp.dwrf";

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  core::PlanNodeId buildScanId;

  auto rowType =
      ROW({"userid", "time", "condition", "experiment"},
          {BIGINT(), BIGINT(), VARCHAR(), VARCHAR()});
  ColumnHandleMap buildAssignments = allRegularColumns(rowType);

  ColumnHandleMap assignments;
  assignments["metric_unitid"] = regularColumn("user_id", BIGINT());
  assignments["total_request_cpu_us"] =
      regularColumn("total_request_cpu_us", BIGINT());
  assignments["ds"] = partitionKey("ds", VARCHAR());
  assignments["metric_time"] = partitionKey("metric_time", BIGINT());

  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "experiment",
              common::test::equal("fb4a_pre_capture_timer_v289_backtest"))
          .build();
  auto buildTableHandle = makeTableHandle(std::move(filters));

  std::vector<std::string> paths = {
      "/Users/mbasmanova/test_data/day1.dwrf",
      "/Users/mbasmanova/test_data/day2.dwrf",
      "/Users/mbasmanova/test_data/day3.dwrf",
      "/Users/mbasmanova/test_data/day4.dwrf",
      "/Users/mbasmanova/test_data/day5.dwrf",
      "/Users/mbasmanova/test_data/day6.dwrf",
      "/Users/mbasmanova/test_data/day7.dwrf",
  };

  std::vector<std::string> ds = {
      "2022-02-14",
      "2022-02-15",
      "2022-02-16",
      "2022-02-17",
      "2022-02-18",
      "2022-02-19",
      "2022-02-20"};
  std::vector<int64_t> times = {
      1644796800,
      1644883200,
      1644969600,
      1645056000,
      1645142400,
      1645228800,
      1645315200};

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(
              ROW({"metric_unitid",
                   "total_request_cpu_us",
                   "metric_time",
                   "ds"},
                  {BIGINT(), BIGINT(), BIGINT(), VARCHAR()}),
              makeTableHandle(common::test::SubfieldFiltersBuilder().build()),
              assignments)
          .capturePlanNodeId(buildScanId)
          .hashJoin(
              {"metric_unitid"},
              {"userid"},
              PlanBuilder(planNodeIdGenerator)
                  .tableScan(
                      ROW({"userid", "time", "condition"},
                          {BIGINT(), BIGINT(), VARCHAR()}),
                      buildTableHandle,
                      buildAssignments)
                  .capturePlanNodeId(scanId)
                  .planNode(),
              "time < metric_time",
              {"userid", "condition", "total_request_cpu_us"},
              core::JoinType::kInner)
          .singleAggregation({0, 1}, {"sum(total_request_cpu_us) as sum_cpu"})
          .singleAggregation(
              {1}, {"avg(sum_cpu)", "variance(sum_cpu)", "count(sum_cpu)"})
          .planNode();

  std::vector<exec::Split> splits;
  for (auto i = 0; i < paths.size(); ++i) {
    auto split = std::make_shared<connector::hive::HiveConnectorSplit>(
        kHiveConnectorId,
        paths[i],
        facebook::velox::dwio::common::FileFormat::ORC,
        0,
        fs::file_size(paths[i]),
        std::unordered_map<std::string, std::optional<std::string>>{
            {"ds", {ds[i]}},
            {"metric_time", {std::to_string(times[i])}},
        });
    splits.push_back(exec::Split(split));
  }

  bool noMoreSplits = false;
  auto task = ::assertQuery(
      plan,
      [&](Task* task) {
        if (noMoreSplits) {
          return;
        }
        task->addSplit(scanId, exec::Split(makeHiveConnectorSplit(path)));
        task->noMoreSplits(scanId);
        for (auto& split : splits) {
          task->addSplit(buildScanId, std::move(split));
        }
        task->noMoreSplits(buildScanId);
        noMoreSplits = true;
      },
      "SELECT null, null, null, null",
      duckDbQueryRunner_);

  std::cout << printPlanWithStats(*plan, task->taskStats(), true) << std::endl;
}