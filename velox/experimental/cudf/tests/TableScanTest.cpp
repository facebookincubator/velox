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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/io/parquet.hpp>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>

#include <atomic>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
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
  struct BatchedFiles {
    std::vector<std::shared_ptr<TempFilePath>> files;
    std::vector<RowVectorPtr> vectors;
    std::vector<std::string> paths;
  };

  class SuccessfulBufferedInputBuilder final
      : public facebook::velox::connector::hive::BufferedInputBuilder {
   public:
    explicit SuccessfulBufferedInputBuilder(
        std::shared_ptr<facebook::velox::connector::hive::BufferedInputBuilder>
            delegate)
        : delegate_(std::move(delegate)) {}

    std::unique_ptr<dwio::common::BufferedInput> create(
        const FileHandle& fileHandle,
        const dwio::common::ReaderOptions& readerOptions,
        const ConnectorQueryCtx* connectorQueryCtx,
        std::shared_ptr<io::IoStatistics> ioStatistics,
        std::shared_ptr<IoStats> ioStats,
        folly::Executor* executor,
        const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
        override {
      auto input = delegate_->create(
          fileHandle,
          readerOptions,
          connectorQueryCtx,
          std::move(ioStatistics),
          std::move(ioStats),
          executor,
          fileReadOps);
      if (input) {
        successfulCreateCount_.fetch_add(1, std::memory_order_relaxed);
      }
      return input;
    }

    uint64_t successfulCreateCount() const {
      return successfulCreateCount_.load(std::memory_order_relaxed);
    }

   private:
    std::shared_ptr<facebook::velox::connector::hive::BufferedInputBuilder>
        delegate_;
    std::atomic_uint64_t successfulCreateCount_{0};
  };

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

  BatchedFiles makeBatchedFiles(int32_t count, int32_t rowsPerFile = 1'000) {
    BatchedFiles data;
    data.files = makeFilePaths(count);
    data.vectors.reserve(count);
    data.paths.reserve(count);
    for (size_t fileIndex = 0; fileIndex < data.files.size(); ++fileIndex) {
      const auto& file = data.files[fileIndex];
      auto vector = makeVectors(1, rowsPerFile).front();
      vector->children()[0] = makeFlatVector<int32_t>(
          rowsPerFile, [fileIndex, rowsPerFile](auto row) {
            const auto magnitude =
                static_cast<int32_t>(fileIndex) * rowsPerFile + row + 1;
            return row % 2 == 0 ? -magnitude : magnitude;
          });
      writeToFile(file->getPath(), vector);
      data.vectors.push_back(std::move(vector));
      data.paths.push_back(file->getPath());
    }
    createDuckDbTable(data.vectors);
    return data;
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

  void assertDecimalScanRoundTrip(
      const RowVectorPtr& vector,
      const RowTypePtr& rowType) {
    auto filePath = TempFilePath::create();
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool =
        rootPool_->addAggregateChild("TableScanTest.ParquetWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    auto parquetOptions = std::make_shared<parquet::ParquetWriterOptions>();
    parquetOptions->enableStoreDecimalAsInteger = true;
    options.formatSpecificOptions = std::move(parquetOptions);
    parquet::Writer writer(std::move(sink), options, writerPool, rowType);
    writer.write(vector);
    writer.close();
    createDuckDbTable({vector});

    auto assignments =
        facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
            rowType);
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();

    assertQuery(plan, {filePath}, "SELECT * FROM tmp");
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

class TableScanTestParameterized : public TableScanTest,
                                   public testing::WithParamInterface<bool> {};

TEST_P(TableScanTestParameterized, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

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

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  // Test scan all columns with CudfHiveConnectorSplits
  {
    auto splits = makeCudfHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }

  // Test scan all columns with HiveConnectorSplits
  {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    splits.push_back(
        facebook::velox::connector::hive::HiveConnectorSplitBuilder(
            filePath->getPath())
            .connectorId(kCudfHiveConnectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .build());
    testScanAllColumns(splits);
  }
}

TEST_P(TableScanTestParameterized, allColumnsUsingExperimentalReader) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  const std::string duckDbSql =
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp";

  auto splits = makeCudfHiveConnectorSplits(
      {filePath, filePath, filePath, filePath, filePath});

  auto useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseExperimentalCudfReader,
       "true"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  auto plan = tableScanNode();
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

INSTANTIATE_TEST_SUITE_P(
    ,
    TableScanTestParameterized,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "FileDataSource";
    });

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector});

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
  writeToFile(filePath->getPath(), vectors);
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
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

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
                  {int128_t{200}, int128_t{700}},
                  /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

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

// Verify that extractFiltersFromRemainingFilter extracts simple single-column
// filters from the remaining filter into subfield filters for pushdown.
// When a filter like "c0 = 1" is fully extracted,
// cudfRemainingFilterExpression_ is null and totalRemainingFilterWallNanos is
// 0. Without extraction, the filter runs post-read on the GPU and the stat is
// > 0.
TEST_F(TableScanTest, remainingFilterExtraction) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  // "c0 = 1" is a single-column equality that should be fully extracted into
  // a subfield filter, leaving no remaining filter to evaluate post-read.
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = 1")
                  .endTableScan()
                  .planNode();

  auto task = assertQuery(plan, {filePath}, "SELECT * FROM tmp WHERE c0 = 1");

  // Verify the filter was fully extracted: no post-read remaining filter ran.
  auto planStats = toPlanStats(task->taskStats());
  const auto& scanStats = planStats.at(plan->id());
  auto it = scanStats.customStats.find("totalRemainingFilterWallNanos");
  ASSERT_NE(it, scanStats.customStats.end());
  EXPECT_EQ(it->second.sum, 0)
      << "Expected no remaining filter time when filter is fully extracted";
}

TEST_F(TableScanTest, batchedMultiFileSplitReaderModes) {
  auto& config = CudfConfig::getInstance();
  const auto oldConfig = config;
  SCOPE_EXIT {
    config = oldConfig;
  };
  config.batchSplitsEnabled = true;
  auto data = makeBatchedFiles(5);
  auto originalBuilder =
      facebook::velox::connector::hive::BufferedInputBuilder::getInstance();
  auto countingBuilder =
      std::make_shared<SuccessfulBufferedInputBuilder>(originalBuilder);
  facebook::velox::connector::hive::BufferedInputBuilder::registerBuilder(
      countingBuilder);
  SCOPE_EXIT {
    facebook::velox::connector::hive::BufferedInputBuilder::registerBuilder(
        originalBuilder);
  };

  for (const bool useExperimentalReader : {false, true}) {
    for (const bool useBufferedInput : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "experimental={}, buffered={}",
              useExperimentalReader,
              useBufferedInput));
      resetCudfHiveConnector(
          std::make_shared<
              config::ConfigBase>(std::unordered_map<std::string, std::string>{
              {cudf_velox::connector::hive::CudfHiveConfig::
                   kUseExperimentalCudfReader,
               useExperimentalReader ? "true" : "false"},
              {cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput,
               useBufferedInput ? "true" : "false"},
              {cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit,
               "16384"}}));

      std::vector<Split> splits;
      for (const auto& path : data.paths) {
        splits.emplace_back(
            facebook::velox::connector::hive::HiveConnectorSplitBuilder(path)
                .connectorId(kCudfHiveConnectorId)
                .fileFormat(dwio::common::FileFormat::PARQUET)
                .build());
      }
      const auto successfulCreatesBefore =
          countingBuilder->successfulCreateCount();
      auto plan = tableScanNode();
      auto task = AssertQueryBuilder(duckDbQueryRunner_)
                      .plan(plan)
                      .beforeTaskStart([&](Task& task) {
                        task.addSplit(plan->id(), std::move(splits));
                        task.noMoreSplits(plan->id());
                      })
                      .assertResults("SELECT * FROM tmp");
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_GT(stats.at(plan->id()).outputVectors, 2);
      const auto& metrics = stats.at(plan->id()).customStats;
      const auto multiFileReaders =
          metrics.find("parquet.cudfMultiFileReaders");
      if (useExperimentalReader) {
        ASSERT_NE(multiFileReaders, metrics.end());
        EXPECT_EQ(multiFileReaders->second.sum, 1);
        EXPECT_EQ(metrics.at("parquet.cudfBatchedFiles").sum, 5);
      } else {
        EXPECT_EQ(multiFileReaders, metrics.end());
      }
      EXPECT_EQ(
          countingBuilder->successfulCreateCount() - successfulCreatesBefore,
          useBufferedInput ? data.paths.size() : 0);
    }
  }
}

TEST_F(TableScanTest, batchedMultiFileSplitWithFilter) {
  auto& config = CudfConfig::getInstance();
  const auto oldConfig = config;
  SCOPE_EXIT {
    config = oldConfig;
  };
  config.batchSplitsEnabled = true;
  auto data = makeBatchedFiles(4);
  auto outputType = ROW({"c0", "c4"}, {INTEGER(), BIGINT()});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add("c0", greaterThan(0, false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", nullptr, std::move(filters));
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType_))
                  .endTableScan()
                  .planNode();
  for (const bool useExperimentalReader : {false, true}) {
    for (const bool useBufferedInput : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "experimental={}, buffered={}",
              useExperimentalReader,
              useBufferedInput));
      resetCudfHiveConnector(
          std::make_shared<
              config::ConfigBase>(std::unordered_map<std::string, std::string>{
              {cudf_velox::connector::hive::CudfHiveConfig::
                   kUseExperimentalCudfReader,
               useExperimentalReader ? "true" : "false"},
              {cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput,
               useBufferedInput ? "true" : "false"}}));

      std::vector<Split> splits;
      for (const auto& path : data.paths) {
        splits.emplace_back(
            facebook::velox::connector::hive::HiveConnectorSplitBuilder(path)
                .connectorId(kCudfHiveConnectorId)
                .fileFormat(dwio::common::FileFormat::PARQUET)
                .build());
      }
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .beforeTaskStart([&](Task& task) {
            task.addSplit(plan->id(), std::move(splits));
            task.noMoreSplits(plan->id());
          })
          .assertResults("SELECT c0, c4 FROM tmp WHERE c0 > 0");
    }
  }
}

TEST_F(TableScanTest, splitVectorPreservesByteRangesAndDisabledMode) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  auto& config = CudfConfig::getInstance();
  const auto oldConfig = config;
  SCOPE_EXIT {
    config = oldConfig;
  };
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2})});
  auto firstFile = TempFilePath::create();
  auto secondFile = TempFilePath::create();
  writeToFile(firstFile->getPath(), data);
  writeToFile(secondFile->getPath(), data);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(asRowType(data->type()))
                  .tableHandle(makeTableHandle("parquet_table"))
                  .endTableScan()
                  .planNode();
  for (const bool enabled : {false, true}) {
    config.batchSplitsEnabled = enabled;
    for (const bool wholeFile : {false, true}) {
      SCOPED_TRACE(fmt::format("enabled={}, wholeFile={}", enabled, wholeFile));
      std::vector<Split> splits;
      for (const auto& file : {firstFile, secondFile}) {
        splits.emplace_back(
            facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                file->getPath())
                .connectorId(kCudfHiveConnectorId)
                .fileFormat(dwio::common::FileFormat::PARQUET)
                .length(wholeFile ? std::numeric_limits<uint64_t>::max() : 1)
                .build());
      }
      std::vector<RowVectorPtr> expected = wholeFile
          ? std::vector<RowVectorPtr>{data, data}
          : std::vector<RowVectorPtr>{makeRowVector(
                {makeFlatVector<int64_t>(std::vector<int64_t>{})})};
      auto task = AssertQueryBuilder(plan)
                      .connectorSessionProperty(
                          kCudfHiveConnectorId,
                          CudfHiveConfig::kUseExperimentalCudfReaderSession,
                          "true")
                      .beforeTaskStart([&](Task& task) {
                        task.addSplit(plan->id(), std::move(splits));
                        task.noMoreSplits(plan->id());
                      })
                      .assertResults(expected);
      EXPECT_EQ(task->taskStats().numTotalSplits, enabled ? 1 : 2);
      EXPECT_EQ(
          toPlanStats(task->taskStats())
              .at(plan->id())
              .customStats.contains("parquet.cudfMultiFileReaders"),
          enabled && wholeFile);
    }
  }
}

TEST_F(TableScanTest, multiFileHybridEmptyAndPrunedSources) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using cudf_velox::connector::hive::CudfHiveConnectorSplit;
  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<std::string> paths;
  for (const auto& values : {std::vector<int64_t>{}, {1, 2, 3}, {}, {10, 11}}) {
    auto file = TempFilePath::create();
    writeToFile(
        file->getPath(), makeRowVector({makeFlatVector<int64_t>(values)}));
    paths.push_back(file->getPath());
    files.push_back(std::move(file));
  }
  auto type = ROW({"c0"}, {BIGINT()});
  for (const auto threshold : {5, 100}) {
    SCOPED_TRACE(threshold);
    auto filters = common::test::SubfieldFiltersBuilder()
                       .add("c0", greaterThan(threshold, false))
                       .build();
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(type)
                    .tableHandle(makeTableHandle(
                        "parquet_table", nullptr, std::move(filters)))
                    .endTableScan()
                    .planNode();
    auto expected = makeRowVector({makeFlatVector<int64_t>(
        threshold == 5 ? std::vector<int64_t>{10, 11}
                       : std::vector<int64_t>{})});
    auto task = AssertQueryBuilder(plan)
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        CudfHiveConfig::kUseExperimentalCudfReaderSession,
                        "true")
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        CudfHiveConfig::kMaxChunkReadLimitSession,
                        "1024")
                    .split(Split(
                        CudfHiveConnectorSplit::makeBatch(
                            kCudfHiveConnectorId, paths, 0)))
                    .assertResults({expected});
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_EQ(
        stats.at(plan->id()).customStats.at("parquet.cudfMultiFileReaders").sum,
        1);
  }
}

TEST_F(TableScanTest, batchedSplitValidationAndSerialization) {
  using cudf_velox::connector::hive::CudfHiveConnectorSplit;
  using cudf_velox::connector::hive::CudfHiveConnectorSplitBuilder;
  using facebook::velox::connector::ConnectorSplitBatch;

  CudfHiveConnectorSplit singleSplit(
      kCudfHiveConnectorId, {"file:/tmp/single.parquet"}, 123);
  EXPECT_EQ(singleSplit.filePath, "/tmp/single.parquet");
  EXPECT_EQ(singleSplit.start, 123);
  auto builtSingle = CudfHiveConnectorSplitBuilder{"file:/tmp/single.parquet"}
                         .connectorId(kCudfHiveConnectorId)
                         .start(123)
                         .build();
  EXPECT_EQ(builtSingle->filePath, "/tmp/single.parquet");
  EXPECT_EQ(builtSingle->start, 123);

  EXPECT_THROW(
      CudfHiveConnectorSplit::makeBatch(
          kCudfHiveConnectorId, std::vector<std::string>{}, 0),
      VeloxUserError);
  EXPECT_THROW(
      CudfHiveConnectorSplitBuilder::forFilePaths(
          std::vector<std::string>{"a", "b"})
          .connectorId(kCudfHiveConnectorId)
          .start(1)
          .build(),
      VeloxUserError);

  const auto makeWeightedChild = [](std::string path, int64_t weight) {
    return CudfHiveConnectorSplitBuilder(std::move(path))
        .connectorId(kCudfHiveConnectorId)
        .splitWeight(weight)
        .build();
  };
  std::vector<std::shared_ptr<ConnectorSplit>> weightedChildren{
      makeWeightedChild("a", std::numeric_limits<int64_t>::max() - 1),
      makeWeightedChild("b", 2),
      makeWeightedChild("c", -1)};
  EXPECT_THROW(
      ConnectorSplitBatch(kCudfHiveConnectorId, std::move(weightedChildren)),
      VeloxUserError);

  auto split = CudfHiveConnectorSplit::makeBatch(
      kCudfHiveConnectorId,
      std::vector<std::string>{"file:/tmp/a.parquet", "s3a://bucket/b.parquet"},
      7);
  auto copy = CudfHiveConnectorSplit::create(split->serialize());
  EXPECT_EQ(copy->filePaths, split->filePaths);
  EXPECT_EQ(copy->splitWeight, 7);
  EXPECT_EQ(copy->start, 0);
  EXPECT_EQ(copy->size(), std::numeric_limits<uint64_t>::max());
}

TEST_F(TableScanTest, decimalSubfieldFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{-500}, int64_t{-500}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);
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
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, decimalRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = CAST('-5.00' AS DECIMAL(5, 2))")
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

// Velox's parquet writer stores DECIMAL(7, 2) as INT32 when
// enableStoreDecimalAsInteger is true, and cuDF's reader maps INT32 decimals
// to DECIMAL32. Velox short decimals are always DECIMAL64, so the scan output
// must be cast from DECIMAL32 to DECIMAL64.
TEST_F(TableScanTest, lowPrecisionDecimalScan) {
  auto rowType = ROW({"d"}, {DECIMAL(7, 2)});
  auto vector = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {12345, std::nullopt, -2500, 300}, DECIMAL(7, 2))});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, lowPrecisionDecimalScanNoCast) {
  auto rowType = ROW({"d"}, {DECIMAL(12, 4)});
  auto vector = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {123456789, std::nullopt, -999999}, DECIMAL(12, 4))});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, nestedDecimalScan) {
  auto rowType = ROW({"s"}, {ROW({"x", "d"}, {INTEGER(), DECIMAL(7, 2)})});
  auto vector = makeRowVector(
      {"s"},
      {makeRowVector(
          {"x", "d"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2))})});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, arrayDecimalScan) {
  auto rowType = ROW({"a"}, {ARRAY(DECIMAL(7, 2))});
  auto elements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300}, DECIMAL(7, 2));
  auto vector = makeRowVector({"a"}, {makeArrayVector({0, 2}, elements)});
  assertDecimalScanRoundTrip(vector, rowType);
}

// Exercises the recursive cast through struct -> struct -> list nesting so a
// decimal buried several levels deep is normalized. Schema:
// struct<int, decimal, struct<int, list<decimal>>>.
TEST_F(TableScanTest, multiLevelNestedDecimalScan) {
  auto rowType =
      ROW({"s"},
          {ROW(
              {"x", "d", "nested"},
              {INTEGER(),
               DECIMAL(7, 2),
               ROW({"y", "a"}, {INTEGER(), ARRAY(DECIMAL(7, 2))})})});
  auto listElements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300, 400}, DECIMAL(7, 2));
  auto vector = makeRowVector(
      {"s"},
      {makeRowVector(
          {"x", "d", "nested"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2)),
           makeRowVector(
               {"y", "a"},
               {makeNullableFlatVector<int32_t>({10, std::nullopt, 30}),
                makeArrayVector({0, 2, 4}, listElements)})})});
  assertDecimalScanRoundTrip(vector, rowType);
}
