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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/DriverStats.h"
#include "velox/exec/HashProbeStatNames.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using facebook::velox::core::QueryConfig;

namespace facebook::velox::exec::test {
namespace {

// Tests filters that a hash join pushes into the table scan on its probe side.
class DynamicFilterPushdownTest : public HiveConnectorTestBase {};

// A join to run and the DuckDB query that gives its expected result.
struct JoinCase {
  core::JoinType joinType;
  std::vector<std::string> outputLayout;
  std::string sql;
};

struct JoinPlan {
  core::PlanNodePtr plan;
  core::PlanNodeId probeScanId;
  core::PlanNodeId buildScanId;
  core::PlanNodeId joinId;
};

// Joins a scan of 'probeType' with a scan of 'buildType'.
JoinPlan makeJoinPlan(
    const RowTypePtr& probeType,
    const RowTypePtr& buildType,
    const std::string& condition,
    const JoinCase& joinCase,
    bool nullAsValue) {
  JoinPlan joinPlan;
  PlanBuilder planBuilder;
  joinPlan.plan = planBuilder.tableScan(probeType)
                      .capturePlanNodeId(joinPlan.probeScanId)
                      .hashJoin(
                          condition,
                          planBuilder.newPlan()
                              .tableScan(buildType)
                              .capturePlanNodeId(joinPlan.buildScanId)
                              .planNode(),
                          joinCase.outputLayout,
                          joinCase.joinType,
                          /*nullAware=*/false,
                          nullAsValue)
                      .capturePlanNodeId(joinPlan.joinId)
                      .planNode();
  return joinPlan;
}

TEST_F(DynamicFilterPushdownTest, mergeFiltersOnSameColumn) {
  // Make sure filters on same column from multiple downstream operators are
  // merged properly without overwriting each other.
  auto aVector =
      makeNamedRowVector("a", makeFlatIdentityVector<int64_t>(20'000));
  auto bVector =
      makeNamedRowVector("b", makeFlatIdentityVector<int64_t>(10'000));
  auto cVector = makeNamedRowVector(
      "c", makeFlatVector<int64_t>(10'000, [](auto i) { return i + 10'000; }));

  std::shared_ptr<TempFilePath> files[3];
  files[0] = writeToFile(aVector);
  files[1] = writeToFile(bVector);
  files[2] = writeToFile(cVector);

  PlanBuilder builder;
  core::PlanNodeId aScanId;
  core::PlanNodeId bScanId;
  core::PlanNodeId cScanId;
  auto plan = builder.tableScan(ROW("a", BIGINT()))
                  .capturePlanNodeId(aScanId)
                  .hashJoin(
                      "a = b",
                      builder.newPlan()
                          .tableScan(ROW("b", BIGINT()))
                          .capturePlanNodeId(bScanId)
                          .planNode(),
                      {"a"})
                  .hashJoin(
                      "a = c",
                      builder.newPlan()
                          .tableScan(ROW("c", BIGINT()))
                          .capturePlanNodeId(cScanId)
                          .planNode(),
                      {"a"})
                  .planNode();
  AssertQueryBuilder(plan)
      .split(aScanId, makeHiveConnectorSplit(files[0]))
      .split(bScanId, makeHiveConnectorSplit(files[1]))
      .split(cScanId, makeHiveConnectorSplit(files[2]))
      .assertResults(makeRowVector({makeFlatVector<int64_t>(0)}));
}

TEST_F(DynamicFilterPushdownTest, rowIndexColumn) {
  // This test ensures dynamic filters can be mapped to correct field when
  // there is row_index column.
  auto aVector = makeNamedRowVector("a", makeFlatIdentityVector<int64_t>(10));
  auto bVector =
      makeNamedRowVector("b", makeFlatVector<int64_t>(10, [](auto i) {
                           if (i < 5) {
                             return i;
                           } else {
                             return 10 + i;
                           }
                         }));
  auto resVector = makeRowVector(
      {"row_index", "a"},
      {makeFlatIdentityVector<int64_t>(5), makeFlatIdentityVector<int64_t>(5)});
  connector::ColumnHandleMap assignments;
  assignments["a"] = std::make_shared<connector::hive::HiveColumnHandle>(
      "a",
      connector::FileColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT());
  assignments["row_index"] =
      std::make_shared<connector::hive::HiveColumnHandle>(
          "row_index",
          connector::FileColumnHandle::ColumnType::kRowIndex,
          BIGINT(),
          BIGINT());
  std::shared_ptr<TempFilePath> files[2];
  files[0] = writeToFile(aVector);
  files[1] = writeToFile(bVector);
  PlanBuilder builder;
  core::PlanNodeId aScanId;
  core::PlanNodeId bScanId;
  auto plan =
      builder
          .tableScan(
              ROW({"row_index", "a"}, BIGINT()), {}, "", nullptr, assignments)
          .capturePlanNodeId(aScanId)
          .hashJoin(
              "a = b",
              builder.newPlan()
                  .tableScan(ROW("b", BIGINT()))
                  .capturePlanNodeId(bScanId)
                  .planNode(),
              {"row_index", "a"})
          .planNode();
  AssertQueryBuilder(plan)
      .split(aScanId, makeHiveConnectorSplit(files[0]))
      .split(bScanId, makeHiveConnectorSplit(files[1]))
      .assertResults(resVector);
}

// A join with more distinct keys than an exact filter holds pushes a bloom
// filter per key into its probe scan. Joins that drop the probe rows that match
// nothing push one. A full join emits those rows, so it pushes none. Two wide
// keys put the table in kHash mode, which tracks no distinct values, so a
// narrow key there pushes no filter.
TEST_F(DynamicFilterPushdownTest, bloomFilter) {
  constexpr int64_t kStep = 1'000'000'000'000;
  const vector_size_t numBuildRows = 10'001 + VectorHasher::kMaxDistinct;
  // Both wide build keys hold the same values, as do both wide probe keys.
  // Probe row 2 * i matches build row i on all three keys.
  const auto buildKey = makeFlatVector<int64_t>(
      numBuildRows, [](auto i) { return kStep * i; }, nullEvery(100));
  const auto probeKey = makeFlatVector<int64_t>(
      2 * numBuildRows, [](auto i) { return kStep / 2 * i; });
  auto build = makeNamedRowVector(
      {{"b1", buildKey},
       {"b2", buildKey},
       {"b3",
        makeFlatVector<int64_t>(numBuildRows, [](auto i) { return i % 10; })}});
  auto probe = makeNamedRowVector(
      {{"a1", probeKey},
       {"a2", probeKey},
       {"a3", makeFlatVector<int64_t>(2 * numBuildRows, [](auto i) {
          return (i / 2) % 10;
        })}});
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});
  const auto probeFile = writeToFile(probe);
  const auto buildFile = writeToFile(build);

  struct {
    std::string condition;
    BaseHashTable::HashMode hashMode;
    int32_t numFilters;
  } keyCases[] = {
      {
          "a1 = b1",
          BaseHashTable::HashMode::kNormalizedKey,
          1,
      },
      {
          "a1 = b1 AND a2 = b2 AND a3 = b3",
          BaseHashTable::HashMode::kHash,
          2,
      },
  };

  // 'sql' takes the join condition.
  const JoinCase joinCases[] = {
      {
          core::JoinType::kInner,
          {"a1", "b1"},
          "SELECT a1, b1 FROM t JOIN u ON {}",
      },
      {
          core::JoinType::kRight,
          {"a1", "b1"},
          "SELECT a1, b1 FROM t RIGHT JOIN u ON {}",
      },
      {
          core::JoinType::kRightSemiProject,
          {"b1", "match"},
          "SELECT b1, EXISTS (SELECT 1 FROM t WHERE {}) FROM u",
      },
      {
          core::JoinType::kRightAnti,
          {"b1"},
          "SELECT b1 FROM u WHERE NOT EXISTS (SELECT 1 FROM t WHERE {})",
      },
      {
          core::JoinType::kFull,
          {"a1", "b1"},
          "SELECT a1, b1 FROM t FULL JOIN u ON {}",
      },
  };
  for (const auto& keyCase : keyCases) {
    for (const auto& joinCase : joinCases) {
      for (bool parallelBuild : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "{}, {}, parallelBuild={}",
                keyCase.condition,
                joinCase.joinType,
                parallelBuild));
        const auto joinPlan = makeJoinPlan(
            ROW({"a1", "a2", "a3"}, BIGINT()),
            ROW({"b1", "b2", "b3"}, BIGINT()),
            keyCase.condition,
            joinCase,
            /*nullAsValue=*/false);
        AssertQueryBuilder builder(joinPlan.plan, duckDbQueryRunner_);
        builder
            .config(
                QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                4 * numBuildRows)
            .split(joinPlan.probeScanId, makeHiveConnectorSplit(probeFile))
            .split(joinPlan.buildScanId, makeHiveConnectorSplit(buildFile));
        if (parallelBuild) {
          builder.serialExecution(false).maxDrivers(2).config(
              QueryConfig::kMinTableRowsForParallelJoinBuild, 1);
        }
        auto task = builder.assertResults(
            fmt::format(fmt::runtime(joinCase.sql), keyCase.condition));

        const auto joinStats =
            PlanNodeStats::extract(task->taskStats(), joinPlan.joinId);
        const auto& hashMode =
            joinStats.operatorStatsFor(OperatorType::kHashBuild)
                .customStat(BaseHashTable::kHashMode);
        ASSERT_EQ(hashMode.min, static_cast<int64_t>(keyCase.hashMode));
        ASSERT_EQ(hashMode.max, static_cast<int64_t>(keyCase.hashMode));

        if (joinCase.joinType == core::JoinType::kFull) {
          // A full join emits the probe rows that match nothing, so it pushes
          // no filter.
          ASSERT_FALSE(
              joinStats.hasCustomStat(DriverStats::kDynamicFiltersProduced));
          ASSERT_FALSE(
              joinStats.hasCustomStat(HashProbeStatNames::kBloomFilterSize));
        } else {
          // Each probe driver pushes one bloom filter per wide key.
          ASSERT_EQ(
              joinStats.customSum(DriverStats::kDynamicFiltersProduced),
              keyCase.numFilters * (parallelBuild ? 2 : 1));
          ASSERT_GT(
              joinStats.customSum(HashProbeStatNames::kBloomFilterSize), 0);
        }
      }
    }
  }
}

// A join that matches null keys to each other keeps the probe rows with null
// keys, so it pushes no filter, which would drop them. Covers an exact filter
// (few keys) and a bloom filter (more keys than an exact filter holds).
TEST_F(DynamicFilterPushdownTest, nullAsValue) {
  for (const vector_size_t numRows :
       {1'000, 10'001 + VectorHasher::kMaxDistinct}) {
    auto probe = makeNamedRowVector(
        "a",
        makeFlatVector<int64_t>(
            numRows, [](auto i) { return i; }, nullEvery(100)));
    auto build = makeNamedRowVector(
        "b",
        makeFlatVector<int64_t>(
            numRows, [](auto i) { return 2 * i; }, nullEvery(100)));
    createDuckDbTable("t", {probe});
    createDuckDbTable("u", {build});
    const auto probeFile = writeToFile(probe);
    const auto buildFile = writeToFile(build);

    const JoinCase joinCases[] = {
        {
            core::JoinType::kLeftSemiFilter,
            {"a"},
            "SELECT a FROM t WHERE EXISTS "
            "(SELECT 1 FROM u WHERE a IS NOT DISTINCT FROM b)",
        },
        {
            core::JoinType::kRightAnti,
            {"b"},
            "SELECT b FROM u WHERE NOT EXISTS "
            "(SELECT 1 FROM t WHERE a IS NOT DISTINCT FROM b)",
        },
    };
    for (const auto& joinCase : joinCases) {
      SCOPED_TRACE(fmt::format("{}, numRows={}", joinCase.joinType, numRows));
      const auto joinPlan = makeJoinPlan(
          ROW("a", BIGINT()),
          ROW("b", BIGINT()),
          "a = b",
          joinCase,
          /*nullAsValue=*/true);
      auto task =
          AssertQueryBuilder(joinPlan.plan, duckDbQueryRunner_)
              .config(
                  QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                  4 * numRows)
              .split(joinPlan.probeScanId, makeHiveConnectorSplit(probeFile))
              .split(joinPlan.buildScanId, makeHiveConnectorSplit(buildFile))
              .assertResults(joinCase.sql);
      ASSERT_FALSE(
          PlanNodeStats::extract(task->taskStats(), joinPlan.joinId)
              .hasCustomStat(DriverStats::kDynamicFiltersProduced));
    }
  }
}

} // namespace
} // namespace facebook::velox::exec::test
