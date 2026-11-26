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
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/tests/utils/CudfPlanBuilder.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <unordered_set>

namespace facebook::velox::cudf_velox::test {
namespace {

using ::facebook::velox::exec::test::AssertQueryBuilder;
using ::facebook::velox::exec::test::PlanBuilder;

// Rewrites plans that read through the cuDF Hive connector.
class PlanRewriterHiveTest
    : public cudf_velox::exec::test::CudfHiveConnectorTestBase {
 protected:
  // A scan the cuDF data source turns down reads with the CPU Parquet reader.
  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
    parquet::registerParquetReaderFactory();
  }

  static void TearDownTestCase() {
    parquet::unregisterParquetReaderFactory();
    CudfHiveConnectorTestBase::TearDownTestCase();
  }

  void SetUp() override {
    savedDriverAdapter_ = CudfConfig::getInstance().enableDriverAdapter;
    CudfConfig::getInstance().enableDriverAdapter = false;
    CudfHiveConnectorTestBase::SetUp();
  }

  void TearDown() override {
    CudfHiveConnectorTestBase::TearDown();
    CudfConfig::getInstance().enableDriverAdapter = savedDriverAdapter_;
  }

 private:
  bool savedDriverAdapter_;
};

TEST_F(PlanRewriterHiveTest, gpuScanFeedsGpuNodesDirectly) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  auto vectors = makeVectors(rowType, 4, 1'000);
  auto file = common::testutil::TempFilePath::create();
  writeToFile(file->getPath(), vectors);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(cudf_velox::exec::test::kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      ::facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .filter("c0 > 0")
                  .singleAggregation({}, {"count(c0)", "min(c1)", "max(c1)"})
                  .planNode();
  auto physical = rewriteToCudfPlan(plan);
  ASSERT_TRUE(physical->is<CudfToVeloxNode>());
  auto aggregation = physical->sources()[0];
  ASSERT_TRUE(aggregation->is<CudfAggregationNode>());
  auto filter = aggregation->sources()[0];
  ASSERT_TRUE(filter->is<CudfFilterProjectNode>());
  EXPECT_TRUE(filter->sources()[0]->is<core::TableScanNode>());

  auto task = ::facebook::velox::exec::test::AssertQueryBuilder(
                  physical, duckDbQueryRunner_)
                  .splits(makeCudfHiveConnectorSplits({file}))
                  .assertResults(
                      "SELECT count(c0), min(c1), max(c1) FROM tmp "
                      "WHERE c0 > 0");
  std::unordered_set<std::string> operators;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      operators.insert(op.operatorType);
    }
  }
  EXPECT_FALSE(operators.contains("CudfFromVelox"));
  EXPECT_TRUE(operators.contains("CudfFilterProject"));
  EXPECT_TRUE(operators.contains("CudfReduceSINGLE"));
}

// The lookup side of an index lookup join is not a pipeline: the join reads
// the table handle of that scan. The rewriter leaves the scan as it is, even
// when the scan would run on the GPU by itself.
TEST_F(PlanRewriterHiveTest, indexLookupSourceStaysAsIs) {
  auto lookupType = ROW({"u0", "u1"}, BIGINT());
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto lookupScan = std::dynamic_pointer_cast<const core::TableScanNode>(
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(cudf_velox::exec::test::kCudfHiveConnectorId)
          .outputType(lookupType)
          .dataColumns(lookupType)
          .indexColumns({"u0"})
          .assignments(
              ::facebook::velox::exec::test::HiveConnectorTestBase::
                  allRegularColumns(lookupType))
          .endTableScan()
          .planNode());
  ASSERT_NE(lookupScan, nullptr);
  ASSERT_TRUE(rewriteToCudfPlan(lookupScan)->is<CudfToVeloxNode>());

  auto plan =
      PlanBuilder(ids, pool_.get())
          .values(makeVectors(ROW({"c0", "c1"}, BIGINT()), 1, 10))
          .project({"c0 + 1 AS k", "c1"})
          .indexLookupJoin(
              {"k"}, {"u0"}, lookupScan, {}, "", false, {"k", "c1", "u1"})
          .planNode();
  // The join itself has no cuDF plan node.
  auto physical = rewriteToCudfPlan(
      plan,
      core::QueryCtx::create(
          nullptr,
          core::QueryConfig({{CudfConfig::kCudfAllowCpuFallback, "true"}})));
  const auto* join = physical->as<core::IndexLookupJoinNode>();
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->lookupSource(), lookupScan);
  ASSERT_TRUE(join->sources()[0]->is<CudfToVeloxNode>());
  EXPECT_TRUE(join->sources()[0]->sources()[0]->is<CudfFilterProjectNode>());
}

// A right or full join emits the build rows that no probe row matched. With a
// grouped probe and an ungrouped build, every split group would emit them
// again, so the task refuses such a join, as it does for a core hash join.
TEST_F(PlanRewriterHiveTest, hashJoinWithMixedGroupedExecution) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  auto file = common::testutil::TempFilePath::create();
  writeToFile(file->getPath(), makeVectors(rowType, 2, 100));

  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto scan = [&](core::PlanNodeId& scanId) {
    return PlanBuilder(ids, pool_.get())
        .startTableScan()
        .connectorId(cudf_velox::exec::test::kCudfHiveConnectorId)
        .outputType(rowType)
        .dataColumns(rowType)
        .assignments(
            ::facebook::velox::exec::test::HiveConnectorTestBase::
                allRegularColumns(rowType))
        .endTableScan()
        .capturePlanNodeId(scanId);
  };

  // Runs the join with two split groups on the probe side and an ungrouped
  // build side, and returns the number of rows.
  auto numJoinedRows = [&](core::JoinType joinType) {
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    auto plan =
        scan(probeScanId)
            .hashJoin(
                {"c0"},
                {"u0"},
                scan(buildScanId).project({"c0 AS u0", "c1 AS u1"}).planNode(),
                "",
                {"c0", "u1"},
                joinType)
            .planNode();
    auto physical = rewriteToCudfPlan(plan);
    EXPECT_TRUE(physical->sources()[0]->is<CudfHashJoinNode>());

    constexpr int32_t kNumSplitGroups{2};
    std::vector<::facebook::velox::exec::Split> probeSplits;
    for (int32_t group = 0; group < kNumSplitGroups; ++group) {
      probeSplits.emplace_back(
          makeCudfHiveConnectorSplit(file->getPath()), group);
    }
    return AssertQueryBuilder(physical)
        .splits(probeScanId, std::move(probeSplits))
        .splits(buildScanId, {makeCudfHiveConnectorSplit(file->getPath())})
        .executionStrategy(core::ExecutionStrategy::kGrouped)
        .groupedExecutionLeafNodeIds({probeScanId})
        .numSplitGroups(kNumSplitGroups)
        .numConcurrentSplitGroups(1)
        .countResults();
  };

  EXPECT_GT(numJoinedRows(core::JoinType::kInner), 0);
  VELOX_ASSERT_THROW(
      numJoinedRows(core::JoinType::kRight),
      "Hash join currently does not support mixed grouped execution for join type RIGHT");
  VELOX_ASSERT_THROW(
      numJoinedRows(core::JoinType::kFull),
      "Hash join currently does not support mixed grouped execution for join type FULL");
}

TEST_F(PlanRewriterHiveTest, scanWithUnsupportedRemainingFilterStaysOnCpu) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  auto vectors = makeVectors(rowType, 4, 1'000);
  auto file = common::testutil::TempFilePath::create();
  writeToFile(file->getPath(), vectors);
  createDuckDbTable(vectors);

  // The cuDF data source cannot evaluate to_big_endian_64, so the connector
  // creates the CPU Hive data source for this scan.
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(cudf_velox::exec::test::kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      ::facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .remainingFilter(
                      "to_big_endian_64(c0) <> "
                      "to_big_endian_64(CAST(1 AS BIGINT))")
                  .endTableScan()
                  .singleAggregation({}, {"count(c0)", "min(c1)", "max(c1)"})
                  .planNode();
  auto physical = rewriteToCudfPlan(plan);
  ASSERT_TRUE(physical->is<CudfToVeloxNode>());
  auto aggregation = physical->sources()[0];
  ASSERT_TRUE(aggregation->is<CudfAggregationNode>());
  auto conversion = aggregation->sources()[0];
  ASSERT_TRUE(conversion->is<CudfFromVeloxNode>());
  EXPECT_TRUE(conversion->sources()[0]->is<core::TableScanNode>());

  ::facebook::velox::exec::test::AssertQueryBuilder(
      physical, duckDbQueryRunner_)
      .splits(makeCudfHiveConnectorSplits({file}))
      .assertResults(
          "SELECT count(c0), min(c1), max(c1) FROM tmp WHERE c0 <> 1");
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
