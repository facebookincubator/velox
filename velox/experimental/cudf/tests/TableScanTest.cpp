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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/io/parquet.hpp>

#include <fmt/ranges.h>

using namespace facebook::velox;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::tests::utils;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {
struct StatsFilterMetrics {
  cudf::size_type inputRowGroups{0};
  std::optional<cudf::size_type> rowGroupsAfterStats;
  cudf::size_type outputRows{0};
};

StatsFilterMetrics readParquetWithStatsFilter(
    const std::string& filePath,
    const RowTypePtr& rowType,
    const common::SubfieldFilters& filters,
    bool useJitFilter) {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  auto const& expr =
      createAstFromSubfieldFilters(filters, tree, scalars, rowType);

  auto options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filePath))
          .use_jit_filter(useJitFilter)
          .build();
  options.set_filter(expr);

  auto result = cudf::io::read_parquet(options);
  return {
      result.metadata.num_input_row_groups,
      result.metadata.num_row_groups_after_stats_filter,
      result.tbl->num_rows()};
}
} // namespace

class TableScanTest : public virtual CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    ExchangeSource::factories().clear();
    ExchangeSource::registerFactory(createLocalExchangeSource);
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  Split makeCudfHiveSplit(std::string path, int64_t splitWeight = 0) {
    return Split(makeCudfHiveConnectorSplit(std::move(path), splitWeight));
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>&
          parquetSplit,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {parquetSplit}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const Split&& split,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {split}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& duckDbSql) {
    return CudfHiveConnectorTestBase::assertQuery(plan, filePaths, duckDbSql);
  }

  // Run query with spill enabled.
  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& spillDirectory,
      const std::string& duckDbSql) {
    return AssertQueryBuilder(plan, duckDbQueryRunner_)
        .spillDirectory(spillDirectory)
        .config(core::QueryConfig::kSpillEnabled, false)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults(duckDbSql);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    auto tableHandle = makeTableHandle();
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .endTableScan()
        .planNode();
  }

  static PlanNodeStats getTableScanStats(const std::shared_ptr<Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  static std::unordered_map<std::string, RuntimeMetric>
  getTableScanRuntimeStats(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  static int64_t getSkippedStridesStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedStrides"].sum;
  }

  static int64_t getSkippedSplitsStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedSplits"].sum;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<Task>& task,
      uint32_t n) {
    // Limit wait to 10 seconds.
    size_t iteration{0};
    while (task->numFinishedDrivers() < n and iteration < 100) {
      /* sleep override */
      usleep(100'000); // 0.1 second.
      ++iteration;
    }
    ASSERT_EQ(n, task->numFinishedDrivers());
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           TINYINT(),
           DOUBLE(),
           BIGINT(),
           VARCHAR(),
           REAL()})};
};

TEST_F(TableScanTest, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");

  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const std::string duckDbSql = "SELECT * FROM tmp";

  // Helper to test scan all columns for the given splits
  auto testScanAllColumns =
      [&](const std::vector<std::shared_ptr<
              facebook::velox::connector::ConnectorSplit>>& splits) {
        auto task = AssertQueryBuilder(duckDbQueryRunner_)
                        .plan(plan)
                        .splits(splits)
                        .assertResults(duckDbSql);

        // A quick sanity check for memory usage reporting. Check that peak
        // total memory usage for the project node is > 0.
        auto planStats = toPlanStats(task->taskStats());
        auto scanNodeId = plan->id();
        auto it = planStats.find(scanNodeId);
        ASSERT_TRUE(it != planStats.end());
        // TODO (dm): enable this test once we start to track gpu memory
        // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

        //  Verifies there is no dynamic filter stats.
        ASSERT_TRUE(it->second.dynamicFilterStats.empty());

        // TODO: We are not writing any customStats yet so disable this check
        // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
      };

  // Test scan all columns with CudfHiveConnectorSplits
  {
    auto splits = makeCudfHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }

  // Test scan all columns with HiveConnectorSplits
  {
    // Lambda to create HiveConnectorSplits from file paths
    auto makeHiveConnectorSplits =
        [&](const std::vector<std::shared_ptr<
                facebook::velox::exec::test::TempFilePath>>& filePaths) {
          std::vector<
              std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
              splits;
          for (const auto& filePath : filePaths) {
            splits.push_back(
                facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                    filePath->getPath())
                    .connectorId(kCudfHiveConnectorId)
                    .fileFormat(dwio::common::FileFormat::PARQUET)
                    .build());
          }
          return splits;
        };

    auto splits = makeHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }
}

TEST_F(TableScanTest, allColumnsUsingFileDataSource) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");

  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const std::string duckDbSql = "SELECT * FROM tmp";

  // Reset the CudfHiveConnector config to not buffered input data source
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));
  auto splits = makeCudfHiveConnectorSplits({filePath});
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(splits)
                  .assertResults(duckDbSql);

  // A quick sanity check for memory usage reporting. Check that peak
  // total memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  // TODO (dm): enable this test once we start to track gpu memory
  // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

  //  Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());

  // TODO: We are not writing any customStats yet so disable this check
  // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
}

TEST_F(TableScanTest, allColumnsUsingExperimentalReader) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");

  createDuckDbTable(vectors);
  const std::string duckDbSql =
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp";

  auto splits = makeCudfHiveConnectorSplits(
      {filePath, filePath, filePath, filePath, filePath});

  // Helper to test scan all columns for the given splits
  auto testScanAllColumnsUsingExperimentalReader =
      [&](const core::PlanNodePtr& plan) {
        auto task = AssertQueryBuilder(duckDbQueryRunner_)
                        .plan(plan)
                        .splits(splits)
                        .assertResults(duckDbSql);

        // A quick sanity check for memory usage reporting. Check that peak
        // total memory usage for the project node is > 0.
        auto planStats = toPlanStats(task->taskStats());
        auto scanNodeId = plan->id();
        auto it = planStats.find(scanNodeId);
        ASSERT_TRUE(it != planStats.end());
        // TODO (dm): enable this test once we start to track gpu memory
        // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

        //  Verifies there is no dynamic filter stats.
        ASSERT_TRUE(it->second.dynamicFilterStats.empty());

        // TODO: We are not writing any customStats yet so disable this check
        // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
      };

  // Reset the CudfHiveConnector config to use the experimental cudf reader
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseExperimentalCudfReader,
       "true"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  // Test scan all columns with buffered input datasource(s)
  {
    auto plan = tableScanNode();
    testScanAllColumnsUsingExperimentalReader(plan);
  }

  // Test scan all columns with kvikIO datasource(s)
  {
    config.insert(
        {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
             kUseBufferedInput,
         "false"});
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(std::move(config)));
    auto plan = tableScanNode();
    testScanAllColumnsUsingExperimentalReader(plan);
  }
}

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector}, "c");

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(ROW({"c0", "c2"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs = {};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(std::move(config)),
      connectorConfigs,
      nullptr);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(makeCudfHiveConnectorSplits({filePath}))
                  .queryCtx(queryCtx)
                  .assertResults("SELECT c0, c2 FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  // Reduced from 500 to 400 as cudf CudfHive writer seems to be writing smaller
  // files.
  ASSERT_GE(rawInputBytes, 400);

  // TableScan runtime stats not available with CudfHive connector yet
#if 0
  auto overreadBytes =
  getTableScanRuntimeStats(task).at("overreadBytes").sum;
  ASSERT_EQ(overreadBytes, 13);
  ASSERT_EQ(
      getTableScanRuntimeStats(task).at("storageReadBytes").sum,
      rawInputBytes + overreadBytes);
  ASSERT_GT(getTableScanRuntimeStats(task)["totalScanTime"].sum, 0);
  ASSERT_GT(getTableScanRuntimeStats(task)["ioWaitWallNanos"].sum, 0);
#endif
}

TEST_F(TableScanTest, columnAliases) {
  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  std::string tableName = "t";
  std::unordered_map<std::string, std::string> aliases = {{"a", "c0"}};
  auto outputType = ROW({"a"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto op = PlanBuilder(pool_.get())
                .startTableScan()
                .tableHandle(tableHandle)
                .tableName(tableName)
                .outputType(outputType)
                .columnAliases(aliases)
                .endTableScan()
                .planNode();
  assertQuery(op, {filePath}, "SELECT c0 FROM tmp");
}

TEST_F(TableScanTest, filterPushdown) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3"}, {TINYINT(), BIGINT(), DOUBLE(), BOOLEAN()});
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // c1 >= 0 or null and c3 is true
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), true))
          .add("c3", std::make_unique<common::BoolValue>(true, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, true, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto task = assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c1", "c3", "c0"}, {BIGINT(), BOOLEAN(), TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0 ) AND c3");

  auto tableScanStats = getTableScanStats(task);
  // EXPECT_EQ(tableScanStats.rawInputRows, 10'000);
  // EXPECT_LT(tableScanStats.inputRows, tableScanStats.rawInputRows);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);

#if 0
  // Repeat the same but do not project out the filtered columns.
  assignments.clear();
  assignments["c0"] =
      facebook::velox::exec::test::HiveConnectorTestBase::regularColumn(
          "c0", TINYINT());
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c0"}, {TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c0 FROM tmp WHERE (c1 >= 0 ) AND c3");

  // TODO: zero column non-empty table is not possible in cudf, need to implement.
  // Do the same for count, no columns projected out.
  assignments.clear();
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp WHERE (c1 >= 0 ) AND c3");

  // Do the same for count, no filter, no projections.
  assignments.clear();
  // subfieldFilters.clear(); // Explicitly clear this.
  tableHandle = makeTableHandle(
      "parquet_table",
      rowType,
      false,
      nullptr,
      nullptr);
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp");
#endif
}

// Disable this test and the one below for now, pending a CUDF fix.
// simoneves 2/25/26
// @TODO simoneves/mattgara re-enable once fixed.

TEST_F(TableScanTest, DISABLED_decimalFilterPushdown) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});

  auto vector = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>(
              {123, 500, -250, 300, 400, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {int128_t{200},
               int128_t{200},
               int128_t{700},
               int128_t{700},
               int128_t{900},
               int128_t{-100}},
              DECIMAL(20, 2)),
      });

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // c0 between 1.00 and 4.00 and c1 in (2.00, 7.00)
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{100}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              common::createHugeintValues(
                  {int128_t{200}, int128_t{700}}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, true, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 BETWEEN CAST('1.00' AS DECIMAL(12, 2)) "
      "AND CAST('4.00' AS DECIMAL(12, 2)) "
      "AND c1 IN (CAST('2.00' AS DECIMAL(20, 2)), "
      "CAST('7.00' AS DECIMAL(20, 2)))");
}

TEST_F(TableScanTest, DISABLED_decimalStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{1000}, int128_t{2000}}, DECIMAL(20, 2))});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({300, 400}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{3000}, int128_t{4000}}, DECIMAL(20, 2))});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({500, 600}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{5000}, int128_t{6000}}, DECIMAL(20, 2))});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{300}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::HugeintRange>(
                  int128_t{3000}, int128_t{4000}, /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, doubleStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DOUBLE(), DOUBLE()});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({1.0, 2.0}),
       makeFlatVector<double>({10.0, 20.0})});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({3.0, 4.0}),
       makeFlatVector<double>({30.0, 40.0})});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({5.0, 6.0}),
       makeFlatVector<double>({50.0, 60.0})});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::DoubleRange>(
                  3.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  4.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::DoubleRange>(
                  30.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  40.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, splitOffsetAndLength) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // Note that the number of row groups selected within `halfFileSize` may
  // change in the future and this test may start failing. In such a case,
  // just adjust the duckdb sql string accordingly.
  const auto halfFileSize = fs::file_size(filePath->getPath()) / 2;

  // First half of file - OFFSET 0 LIMIT 6000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, halfFileSize),
      "SELECT * FROM tmp OFFSET 0 LIMIT 6000");

  // Second half of file - OFFSET 6000 LIMIT 4000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), halfFileSize),
      "SELECT * FROM tmp OFFSET 6000 LIMIT 4000");

  const auto fileSize = fs::file_size(filePath->getPath());

  // All row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, fileSize),
      "SELECT * FROM tmp");

  // No row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), fileSize),
      "SELECT * FROM tmp LIMIT 0");
}
