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
#include "velox/experimental/cudf/connectors/hive/CudfDynamicFilter.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>

#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <unordered_set>

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
    return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  // Runtime stats omit zero-valued counters, so a missing stat reads as zero.
  static int64_t getTableScanRuntimeStat(
      const std::shared_ptr<Task>& task,
      std::string_view name) {
    const auto runtimeStats = getTableScanRuntimeStats(task);
    const auto it = runtimeStats.find(std::string(name));
    return it == runtimeStats.end() ? 0 : it->second.sum;
  }

  // Verifies I/O is bounded by one footer and one data read of the unique file.
  static void assertStorageReadStats(
      const std::unordered_map<std::string, RuntimeMetric>& runtimeStats,
      int64_t fileSize) {
    for (const auto key : {
             io::kStorageReadBytes,
             cudf_velox::connector::hive::CudfHiveDataSource::
                 kDwioStorageReadBytes,
         }) {
      const auto& metric = runtimeStats.at(std::string(key));
      EXPECT_GT(metric.sum, 0);
      EXPECT_LE(metric.sum, 2 * fileSize);
    }
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

    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .splits(makeCudfHiveConnectorSplits({filePath}))
        .assertResults("SELECT * FROM tmp");
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

// Reads several splits of a multi-row-group file with chunk and pass read
// limits small enough that each split is read as multiple row group passes,
// each yielding multiple table chunks.
TEST_P(TableScanTestParameterized, allColumnsWithRowGroupPasses) {
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

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxChunkReadLimit,
       "8192"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxPassReadLimit,
       "32768"},
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

  auto planStats = toPlanStats(task->taskStats());
  auto it = planStats.find(plan->id());
  ASSERT_TRUE(it != planStats.end());

  // Reading in chunks must not change the number of rows returned.
  const auto& scanStats = it->second.operatorStatsFor("TableScan");
  EXPECT_EQ(scanStats.outputRows, 5 * 10 * 1'000);

  // Splitting the read into chunks must produce more than one output vector
  // per split.
  EXPECT_GT(scanStats.outputVectors, splits.size());

  //  Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());
}

// Splits prepared in the background by the preloader must produce the same
// results as splits prepared on the driver thread.
TEST_F(TableScanTest, preloadSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "10")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(plan->id()).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_EQ(
      customStats.at(std::string(TableScan::kPreloadedSplits)).sum,
      filePaths.size());
}

// A busy IO thread pool never runs the queued preload tasks, so every split is
// prepared inline by the driver that comes to read it.
TEST_F(TableScanTest, preloadingSplitClose) {
  auto filePaths = makeFilePaths(20);
  auto vectors = makeVectors(20, 100);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto* ioExecutor = ioExecutor_.get();
  folly::Latch latch(ioExecutor->numThreads());
  std::vector<folly::Baton<>> batons(ioExecutor->numThreads());
  // Simulate a busy IO thread pool by blocking all its threads.
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  ASSERT_EQ(Task::numRunningTasks(), 0);
  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "4")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(
      planStats.at(plan->id())
          .customStats.at(std::string(TableScan::kPreloadedSplits))
          .sum,
      1);

  task.reset();
  // Once all task references are cleared, all the tasks should be destroyed.
  ASSERT_EQ(Task::numRunningTasks(), 0);
  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A query that stops early leaves the splits the preloader has already prepared
// unread, so their readers are destroyed with a payload fetch outstanding.
TEST_F(TableScanTest, abandonPreloadedSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // Consume all but one IO threads so the preloader can post column chunk fetch
  // tasks queuing behind the blockers and stay pending. Driver reads complete
  // inline via work-stealing.
  auto* ioExecutor = ioExecutor_.get();
  const auto numBlocked = ioExecutor->numThreads() - 1;
  ASSERT_GE(numBlocked, 1);
  folly::Latch latch(numBlocked);
  std::vector<folly::Baton<>> batons(numBlocked);
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  // The limit is reached partway into the second split, so the splits the
  // preloader prepared behind it are never read. Which rows are returned is
  // unspecified, so only the row count can be asserted.
  constexpr int32_t kLimit = 1'500;
  core::PlanNodeId scanNodeId;
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(rowType_)
                  .tableHandle(makeTableHandle())
                  .endTableScan()
                  .capturePlanNodeId(scanNodeId)
                  .limit(0, kLimit, false)
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "8")
                    // Enable column chunk fetch during preload
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kPreloadColumnChunksSession,
                        "true")
                    .splits(makeCudfHiveConnectorSplits(filePaths))
                    .copyResults(pool_.get(), task);
  EXPECT_EQ(result->size(), kLimit);

  // The first split is read before the preloader runs, so only the splits
  // after it are preloaded. The stat counts the preloaded splits that were
  // read, so it confirms preloading was on but cannot measure how many were
  // abandoned.
  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(scanNodeId).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_GE(customStats.at(std::string(TableScan::kPreloadedSplits)).sum, 1);

  // Tear down while abandoned splits' payload fetches are still queued.
  task.reset();
  ASSERT_EQ(Task::numRunningTasks(), 0);

  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A filter that no row group can satisfy prunes every row group of the split,
// leaving no row group passes to read.
TEST_F(TableScanTest, filterPrunesAllRowGroups) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  // One row group per vector, all holding values well below the filter bound.
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({4, 5, 6})}),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({7, 8, 9})}),
  };
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  constexpr int64_t kUnmatchedValue = 1'000;
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  kUnmatchedValue, kUnmatchedValue, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .splits(makeCudfHiveConnectorSplits({filePath}))
          .assertResults(
              fmt::format("SELECT c0 FROM tmp WHERE c0 = {}", kUnmatchedValue));

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_EQ(planStats.at(plan->id()).outputRows, 0);
}

// Table schemas use lowercase names while the file keeps mixed-case names. The
// filter-only column must still be read so the pushed-down filter can find it.
TEST_F(TableScanTest, mixedCaseFileColumnNames) {
  auto vector = makeRowVector(
      {"Filter_Col", "Value_Col"},
      {makeFlatVector<std::string>({"a", "b", "c", "d"}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {vector});
  createDuckDbTable({vector});

  auto rowType = ROW({"filter_col", "value_col"}, {VARCHAR(), BIGINT()});
  auto outputType = ROW({"value_col"}, {BIGINT()});
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "filter_col",
              std::make_unique<common::BytesRange>(
                  "b",
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  "",
                  /*upperUnbounded*/ true,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();
  auto plan =
      PlanBuilder()
          .startTableScan()
          .outputType(outputType)
          .tableHandle(makeTableHandle(
              "parquet_table", rowType, std::move(subfieldFilters), nullptr))
          .assignments(
              facebook::velox::exec::test::HiveConnectorTestBase::
                  allRegularColumns(outputType))
          .endTableScan()
          .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .connectorSessionProperty(
          kCudfHiveConnectorId,
          facebook::velox::connector::hive::HiveConfig::
              kFileColumnNamesReadAsLowerCaseSession,
          "true")
      .splits(makeCudfHiveConnectorSplits({filePath}))
      .assertResults("SELECT Value_Col FROM tmp WHERE Filter_Col >= 'b'");
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TableScanTestParameterized,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "FileDataSource";
    });

TEST_F(TableScanTest, mixedCaseDynamicIntegerFilter) {
  auto probe = makeRowVector(
      {"Key_Col"},
      {makeNullableFlatVector<int64_t>({1, 2, 100000, std::nullopt})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), probe);
  auto build = makeRowVector({"b"}, {makeFlatVector<int64_t>({1, 100000})});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
  auto rowType = ROW({"key_col"}, {BIGINT()});
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .tableHandle(makeTableHandle("t", rowType))
          .outputType(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"key_col"},
              {"b"},
              buildPlan,
              "",
              {"key_col"},
              core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  auto expected =
      makeRowVector({"key_col"}, {makeFlatVector<int64_t>({1, 100000})});
  auto task = AssertQueryBuilder(plan)
                  .config("hash_probe_dynamic_filter_pushdown_enabled", "true")
                  .connectorSessionProperty(
                      kCudfHiveConnectorId,
                      facebook::velox::connector::hive::HiveConfig::
                          kFileColumnNamesReadAsLowerCaseSession,
                      "true")
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({file}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(scanId).outputRows, 2);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  EXPECT_EQ(
      stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum, 2);
}

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatIdentityVector<int64_t>(kSize),
      makeFlatIdentityVector<int64_t>(kSize),
      makeFlatIdentityVector<int64_t>(kSize),
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

  const auto runtimeStats = getTableScanRuntimeStats(task);
  assertStorageReadStats(runtimeStats, filePath->fileSize());
  ASSERT_GT(runtimeStats.at("totalScanTime").sum, 0);
  ASSERT_GT(runtimeStats.at("ioWaitWallNanos").sum, 0);
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

TEST_F(TableScanTest, dynamicFilterUsesPhysicalNamesWithoutDataColumns) {
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);

  auto build = makeRowVector({"u_key"}, {makeFlatVector<int64_t>({1})});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide =
      PlanBuilder(planNodeIdGenerator, pool_.get()).values({build}).planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto outputType = ROW({"c1", "c0"}, {BIGINT(), BIGINT()});
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .tableHandle(makeTableHandle())
                  .outputType(outputType)
                  .columnAliases({{"c1", "c0"}, {"c0", "c1"}})
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c1"},
                      {"u_key"},
                      buildSide,
                      "",
                      {"c1", "c0"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();
  auto expected = makeRowVector(
      {"c1", "c0"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto task = AssertQueryBuilder(plan)
                  .config(CudfConfig::kCudfEnabled, "true")
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
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
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  auto tableScanStats = getTableScanStats(task);
  // EXPECT_EQ(tableScanStats.rawInputRows, 10'000);
  // EXPECT_LT(tableScanStats.inputRows, tableScanStats.rawInputRows);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);

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
      "SELECT c0 FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  // Count with no projections, however the filter columns c1 and c3 are still
  // read.
  assignments.clear();
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"count(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  // Do the same for count, no filter, no projections. Only the footer is read.
  assignments.clear();
  tableHandle = makeTableHandle("parquet_table", rowType);
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"count(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp");
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

// Verify that a `count(*)` scan with no projected columns returns the correct
// row count from the Parquet footer
TEST_F(TableScanTest, countStarNoProjectedColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  // Scans no columns, the shape a `SELECT count(*)` plan produces.
  auto countStarPlan = PlanBuilder(pool_.get())
                           .startTableScan()
                           .outputType(ROW({}, {}))
                           .tableHandle(makeTableHandle())
                           .endTableScan()
                           .singleAggregation({}, {"count(1)"})
                           .planNode();

  // Counts the same rows through a decoded column, the oracle for the
  // footer-derived counts below.
  auto countColumnPlan = PlanBuilder(pool_.get())
                             .startTableScan()
                             .outputType(ROW({"c0"}, {INTEGER()}))
                             .tableHandle(makeTableHandle())
                             .endTableScan()
                             .singleAggregation({}, {"count(1)"})
                             .planNode();

  const auto fileSize = fs::file_size(filePath->getPath());
  const auto halfFileSize = fileSize / 2;

  const auto countRows = [&](const core::PlanNodePtr& plan,
                             uint64_t start,
                             uint64_t length) {
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .split(Split(makeCudfHiveConnectorSplit(
                          filePath->getPath(), start, length)))
                      .copyResults(pool_.get(), task);
    EXPECT_EQ(result->size(), 1);
    // A footer-only scan reads no column chunks.
    if (plan == countStarPlan) {
      EXPECT_LT(
          getTableScanRuntimeStat(task, io::kStorageReadBytes), fileSize / 4);
    }
    return result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0);
  };

  // A split owns only the row groups that start inside its byte range, so a
  // whole-file footer count would over-count either half.
  const auto firstHalf = countRows(countStarPlan, 0, halfFileSize);
  const auto secondHalf =
      countRows(countStarPlan, halfFileSize, fileSize - halfFileSize);
  EXPECT_EQ(firstHalf, countRows(countColumnPlan, 0, halfFileSize));
  EXPECT_EQ(
      secondHalf,
      countRows(countColumnPlan, halfFileSize, fileSize - halfFileSize));
  EXPECT_EQ(firstHalf + secondHalf, 10'000);

  // A split that starts past the last row group owns no rows.
  EXPECT_EQ(countRows(countStarPlan, fileSize, 1), 0);
}

// A remaining filter that folds to a constant must either keep the row count
// from the footer or skip the split without reading any columns.
TEST_F(TableScanTest, constantRemainingFilter) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  using CountAndSkippedSplits = std::pair<int64_t, int64_t>;
  const auto countRows = [&](const core::TypedExprPtr& remainingFilter) {
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(ROW({}, {}))
                    .tableHandle(makeTableHandle(
                        "parquet_table", rowType_, {}, remainingFilter))
                    .endTableScan()
                    .singleAggregation({}, {"count(1)"})
                    .planNode();
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .split(makeCudfHiveSplit(filePath->getPath()))
                      .copyResults(pool_.get(), task);
    EXPECT_EQ(result->size(), 1);
    return CountAndSkippedSplits{
        result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0),
        getTableScanRuntimeStat(task, "skippedSplits")};
  };

  EXPECT_EQ(
      countRows(std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true)),
      CountAndSkippedSplits(10'000, 0));
  EXPECT_EQ(
      countRows(std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)),
      CountAndSkippedSplits(0, 1));
  EXPECT_EQ(
      countRows(
          std::make_shared<core::ConstantTypedExpr>(
              BOOLEAN(), Variant::null(TypeKind::BOOLEAN))),
      CountAndSkippedSplits(0, 1));

  // A non-constant filter that references no column is rejected.
  auto randomFilter = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::CallTypedExpr>(
              DOUBLE(), std::vector<core::TypedExprPtr>{}, "rand"),
          std::make_shared<core::ConstantTypedExpr>(DOUBLE(), 0.5),
      },
      "gt");
  VELOX_ASSERT_USER_THROW(
      countRows(randomFilter), "references no column is not supported");
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

TEST_F(TableScanTest, dynamicFilterPrunesReaderRowGroups) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  std::vector<RowVectorPtr> vectors;
  for (int group = 0; group < 3; ++group) {
    vectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(
             1'000,
             [group](auto row) { return group * 10'000 + row * 5; },
             nullEvery(13)),
         makeFlatVector<int64_t>(
             1'000, [group](auto row) { return group * 10'000 + row; })}));
  }
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable("t", vectors);
  common::SubfieldFilters staticFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{12505}, int64_t{12505}, false))
          .build();
  auto staticMetrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, staticFilters, true);
  EXPECT_EQ(staticMetrics.inputRowGroups, 3);
  ASSERT_TRUE(staticMetrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(*staticMetrics.rowGroupsAfterStats, 1);

  auto runTest = [&](const std::vector<int64_t>& values,
                     common::FilterKind expectedKind,
                     bool expectMatch = true,
                     bool includeBuildKey = false) {
    SCOPED_TRACE(fmt::format("values: {}", fmt::join(values, ", ")));
    ASSERT_EQ(common::createBigintValues(values, false)->kind(), expectedKind);
    auto build = makeRowVector({"c0"}, {makeFlatVector<int64_t>(values)});
    createDuckDbTable("u", {build});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    auto scanType = ROW({"c1", "c0"}, BIGINT());
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(scanType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                buildSide,
                "",
                includeBuildKey ? std::vector<std::string>{"c0", "c1", "u_key"}
                                : std::vector<std::string>{"c0", "c1"},
                core::JoinType::kInner)
            .planNode();
    std::atomic<int32_t> selected{-1};
    const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
    common::testutil::TestValue::enable();
    SCOPE_EXIT {
      if (!testValuesWereEnabled) {
        common::testutil::TestValue::disable();
      }
    };
    common::testutil::ScopedTestValue selectedCounter(
        "facebook::velox::cudf_velox::connector::hive::CudfSplitReader::selectedRowGroups",
        std::function<void(void*)>(
            [&](void* value) { selected = *static_cast<size_t*>(value); }));
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(
                        fmt::format(
                            "SELECT t.c0, t.c1{} FROM t JOIN u ON t.c0 = u.c0",
                            includeBuildKey ? ", u.c0" : ""));
#ifndef NDEBUG
    EXPECT_EQ(selected, expectMatch ? 1 : 0);
#endif
    EXPECT_EQ(
        toPlanStats(task->taskStats()).at(scanId).outputRows,
        expectMatch ? (includeBuildKey ? 1000 : values.size()) : 0);
  };
  for (bool includeBuildKey : {false, true}) {
    runTest({12505}, common::FilterKind::kBigintRange, true, includeBuildKey);
    runTest(
        {10005, 10015},
        common::FilterKind::kBigintValuesUsingBitmask,
        true,
        includeBuildKey);
    runTest(
        {10005, 14005},
        common::FilterKind::kBigintValuesUsingHashTable,
        true,
        includeBuildKey);
  }
  auto nullKeys = makeRowVector(
      {"c0", "c1"},
      {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
       makeFlatVector<int64_t>({1, 2})});
  filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), nullKeys);
  createDuckDbTable("t", {nullKeys});
  runTest({1}, common::FilterKind::kBigintRange, false);
}

TEST_F(TableScanTest, integerDynamicFilterFromHashJoin) {
  constexpr vector_size_t kNumBuildRows{
      ::facebook::velox::exec::VectorHasher::kMaxDistinct + 1};
  auto probeType = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}});
  auto probe = makeRowVector(
      {makeFlatVector<int32_t>(1'000, folly::identity),
       makeFlatVector<int64_t>(1'000, [](auto row) {
         // Both per-column filters accept this row, but the pair must
         // still be checked by the join.
         return static_cast<int64_t>(row == 150 ? 151 : row) * 10;
       })});
  auto buildKey = makeFlatVector<int32_t>(
      kNumBuildRows, [](auto row) { return 100 + row; });
  auto secondBuildKey = makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
    return static_cast<int64_t>(100 + row) * 10;
  });
  for (vector_size_t row = 100; row < kNumBuildRows; ++row) {
    buildKey->setNull(row, true);
    secondBuildKey->setNull(row, true);
  }
  auto build = makeRowVector({buildKey, secondBuildKey});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key0", "c1 AS u_key1"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(probeType)
                  .dataColumns(probeType)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(probeType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .project({"c1 AS key1", "c0 AS key0"})
                  .hashJoin(
                      {"key0", "key1"},
                      {"u_key0", "u_key1"},
                      buildSide,
                      "",
                      {"key0", "key1"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();

  std::atomic<int32_t> numFiltersBuilt{0};
  const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
  common::testutil::TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesWereEnabled) {
      common::testutil::TestValue::disable();
    }
  };
  common::testutil::ScopedTestValue filterBuildCounter(
      "facebook::velox::cudf_velox::CudfHashJoinProbe::makeIntegerDynamicFilter",
      std::function<void(void*)>([&](void*) { ++numFiltersBuilt; }));
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(4)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(
                      "SELECT t.c0, t.c1 FROM t JOIN u "
                      "ON t.c0 = u.c0 AND t.c1 = u.c1");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_GE(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 2);
  EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
  EXPECT_EQ(stats.at(scanId).outputRows, probe->size());
  EXPECT_EQ(
      stats.at(scanId).dynamicFilterStats.producerNodeIds,
      std::unordered_set<core::PlanNodeId>{joinId});
#ifndef NDEBUG
  EXPECT_EQ(numFiltersBuilt, 2);
#endif

  auto disabledTask =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .config("hash_probe_dynamic_filter_pushdown_enabled", "false")
          .maxDrivers(4)
          .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
          .assertResults(
              "SELECT t.c0, t.c1 FROM t JOIN u "
              "ON t.c0 = u.c0 AND t.c1 = u.c1");
  const auto disabledStats = toPlanStats(disabledTask->taskStats());
  EXPECT_EQ(disabledStats.at(scanId).outputRows, probe->size());
  EXPECT_EQ(
      disabledStats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
}

TEST_F(TableScanTest, integerDynamicFilterPreservesKeyWidth) {
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto probe = makeRowVector(
        {"c0"}, {makeNullableFlatVector<T>({lo, 0, hi, std::nullopt})});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), probe);
    // Unique extreme keys exercise sparse filter production and native width.
    for (int count :
         {3, 2 * ::facebook::velox::exec::VectorHasher::kMaxDistinct}) {
      std::vector<std::optional<T>> values;
      int64_t matches = 0;
      for (int i = 0; i < count; ++i) {
        values.push_back(
            i % 3 ? std::optional<T>{i % 2 ? lo : hi} : std::nullopt);
        matches += values.back().has_value();
      }
      auto build = makeRowVector({"b"}, {makeNullableFlatVector<T>(values)});
      auto ids = std::make_shared<core::PlanNodeIdGenerator>();
      auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
      core::PlanNodeId scanId;
      auto plan =
          PlanBuilder(ids, pool_.get())
              .startTableScan()
              .connectorId(kCudfHiveConnectorId)
              .tableHandle(makeTableHandle("t", probe->rowType()))
              .outputType(probe->rowType())
              .assignments(
                  HiveConnectorTestBase::allRegularColumns(probe->rowType()))
              .endTableScan()
              .capturePlanNodeId(scanId)
              .hashJoin(
                  {"c0"}, {"b"}, buildPlan, "", {"c0"}, core::JoinType::kInner)
              .singleAggregation({}, {"count(1) AS n"})
              .planNode();
      auto expected =
          makeRowVector({"n"}, {makeFlatVector<int64_t>({matches})});
      auto task = AssertQueryBuilder(plan)
                      .maxDrivers(1)
                      .splits(scanId, makeCudfHiveConnectorSplits({file}))
                      .assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      const auto& scanStats = stats.at(scanId);
      if (matches <= ::facebook::velox::exec::VectorHasher::kMaxDistinct) {
        EXPECT_EQ(scanStats.customStats.at("dynamicFiltersAccepted").sum, 1);
        EXPECT_EQ(scanStats.outputRows, 2);
      } else {
        EXPECT_EQ(scanStats.customStats.count("dynamicFiltersAccepted"), 0);
        EXPECT_EQ(scanStats.outputRows, probe->size());
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, integerMembershipMask) {
  using facebook::velox::cudf_velox::connector::hive::applyIntegerBitmapToMask;
  using facebook::velox::cudf_velox::connector::hive::applyIntegerRangeToMask;
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto copyColumn = [&]<typename T>(const std::vector<T>& values) {
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_to_id<T>()},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().template data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.get()));
    stream.sync();
    return column;
  };
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto input = copyColumn(
        std::vector<T>{
            lo, static_cast<T>(lo + 1), static_cast<T>(lo + 2), -1, 0, hi, 0});
    auto nulls =
        cudf::create_null_mask(7, cudf::mask_state::ALL_VALID, stream, mr);
    const cudf::bitmask_type validity = ~(cudf::bitmask_type{1} << 6);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        nulls.data(),
        &validity,
        sizeof(validity),
        cudaMemcpyHostToDevice,
        stream.get()));
    stream.sync();
    input->set_null_mask(std::move(nulls), 1);
    auto bitmap = copyColumn(std::vector<uint32_t>{5});
    auto values = copyColumn(std::vector<T>{static_cast<T>(lo + 2), hi});
    cudf_velox::connector::hive::CudfIntegerHashSet hash(
        values->view(), lo, stream, mr);

    // Each representation initializes the mask once. Their intersection keeps
    // only lo + 2; NULL passes both sets but fails the range.
    for (int first = 0; first < 3; ++first) {
      std::unique_ptr<cudf::column> mask;
      for (int i = 0; i < 3; ++i) {
        switch ((first + i) % 3) {
          case 0:
            applyIntegerRangeToMask(
                input->view(), lo, 0, false, mask, stream, mr);
            break;
          case 1:
            applyIntegerBitmapToMask(
                input->view(), bitmap->view(), lo, true, mask, stream, mr);
            break;
          case 2:
            hash.apply(input->view(), true, mask, stream, mr);
            break;
        }
      }
      std::array<bool, 7> actual;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          actual.data(),
          mask->view().data<bool>(),
          sizeof(actual),
          cudaMemcpyDeviceToHost,
          stream.get()));
      stream.sync();
      EXPECT_EQ(
          actual,
          (std::array<bool, 7>{
              false, false, true, false, false, false, false}));
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, cachedIntegerHashMembership) {
  using cudf_velox::connector::hive::applyIntegerRangeToMask;
  using cudf_velox::connector::hive::CudfIntegerHashSet;
  auto buildStream = cudfGlobalStreamPool().get_stream();
  auto probeStream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto values = makeNullableFlatVector<T>(
        {hi,
         lo,
         static_cast<T>(lo + 1),
         static_cast<T>(lo + 2),
         0,
         hi,
         std::nullopt,
         lo,
         0});
    // Initialize the complete validity word directly, including padding bits.
    auto copyColumn = [&](const std::vector<T>& host) {
      auto column = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_to_id<T>()},
          host.size(),
          cudf::mask_state::ALL_VALID,
          buildStream,
          mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          column->mutable_view().template data<T>(),
          host.data(),
          host.size() * sizeof(T),
          cudaMemcpyHostToDevice,
          buildStream.get()));
      buildStream.sync();
      return column;
    };
    auto input = copyColumn(
        {hi,
         lo,
         static_cast<T>(lo + 1),
         static_cast<T>(lo + 2),
         0,
         hi,
         0,
         lo,
         0});
    const cudf::bitmask_type validity = ~(cudf::bitmask_type{1} << 6);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        input->mutable_view().null_mask(),
        &validity,
        sizeof(validity),
        cudaMemcpyHostToDevice,
        buildStream.get()));
    input->set_null_count(1);
    auto keys =
        copyColumn({lo, static_cast<T>(lo + 2), hi, static_cast<T>(lo + 2)});
    CudfIntegerHashSet set(keys->view(), lo + 1, buildStream, mr);
    buildStream.sync();
    auto slices = cudf::slice(input->view(), {1, 8}, probeStream);
    const auto probe = slices.front();
    const std::unordered_set<T> expectedKeys{lo, static_cast<T>(lo + 2), hi};
    for (bool nullAllowed : {false, true}) {
      for (bool precedingRange : {false, true}) {
        std::unique_ptr<cudf::column> mask;
        if (precedingRange) {
          applyIntegerRangeToMask(probe, lo, 0, false, mask, probeStream, mr);
        }
        set.apply(probe, nullAllowed, mask, probeStream, mr);
        // Reuse both the table and an existing mask; false rows stay false.
        set.apply(probe, nullAllowed, mask, probeStream, mr);
        std::vector<uint8_t> actual(probe.size());
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            actual.data(),
            mask->view().data<bool>(),
            actual.size(),
            cudaMemcpyDeviceToHost,
            probeStream.get()));
        probeStream.sync();
        for (int i = 0; i < probe.size(); ++i) {
          const bool isNull = values->isNullAt(i + 1);
          const bool expected = isNull
              ? nullAllowed && !precedingRange
              : expectedKeys.contains(values->valueAt(i + 1)) &&
                  (!precedingRange || values->valueAt(i + 1) <= 0);
          EXPECT_EQ(actual[i] != 0, expected) << "row " << i;
        }
      }
    }
    auto emptyKeys = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_to_id<T>()},
        0,
        cudf::mask_state::UNALLOCATED,
        buildStream,
        mr);
    CudfIntegerHashSet empty(emptyKeys->view(), lo + 1, buildStream, mr);
    buildStream.sync();
    for (bool nullAllowed : {false, true}) {
      std::unique_ptr<cudf::column> mask;
      empty.apply(probe, nullAllowed, mask, probeStream, mr);
      std::vector<uint8_t> actual(probe.size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          actual.data(),
          mask->view().data<bool>(),
          actual.size(),
          cudaMemcpyDeviceToHost,
          probeStream.get()));
      probeStream.sync();
      for (int i = 0; i < probe.size(); ++i) {
        EXPECT_EQ(actual[i] != 0, nullAllowed && values->isNullAt(i + 1));
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, exactDynamicFilterUsesInputType) {
  using cudf_velox::connector::hive::makeRowGroupOnlyFilter;
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto input = makeRowVector(
        {"c0", "c1", "c2", "c3"},
        {makeNullableFlatVector<T>({lo, 0, 1, hi, std::nullopt, lo, hi}),
         makeFlatVector<T>(
             {static_cast<T>(lo + 10),
              static_cast<T>(lo + 20),
              static_cast<T>(lo + 30),
              static_cast<T>(lo + 40),
              static_cast<T>(lo + 50),
              lo,
              static_cast<T>(lo + 51)}),
         makeNullableFlatVector<int64_t>(
             {std::nullopt, 1, 2, 3, std::nullopt, 1, 3}),
         makeFlatVector<int64_t>(7, [](auto) { return 0; })});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), input);
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    ::facebook::velox::connector::ConnectorQueryCtx ctx(
        pool_.get(),
        pool_.get(),
        properties.get(),
        nullptr,
        common::PrefixSortConfig{},
        nullptr,
        nullptr,
        "query",
        "task",
        "scan",
        0,
        "");
    auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
        kCudfHiveConnectorId);
    for (bool lateRange : {false, true}) {
      for (bool nullAllowed : {false, true}) {
        auto source = connector->createDataSource(
            asRowType(input->type()),
            makeTableHandle("t", asRowType(input->type())),
            HiveConnectorTestBase::allRegularColumns(asRowType(input->type())),
            &ctx);
        const std::vector<int64_t> values{
            std::numeric_limits<int64_t>::min(),
            lo,
            hi,
            std::numeric_limits<int64_t>::max()};
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        source->addDynamicFilter(
            0, common::createBigintValues(sorted, nullAllowed));
        source->addDynamicFilter(
            1,
            common::createBigintValues(
                {lo + 10, lo + 20, lo + 40, lo + 50}, false));
        if (lateRange) {
          std::vector<std::unique_ptr<common::BigintRange>> ranges;
          for (int64_t key : {1, 3}) {
            ranges.push_back(
                std::make_unique<common::BigintRange>(key, key, false));
          }
          source->addDynamicFilter(
              2,
              std::make_shared<common::BigintMultiRange>(
                  std::move(ranges), nullAllowed));
        } else {
          source->addDynamicFilter(
              2, common::createBigintValues({1, 3}, nullAllowed));
        }
        for (int split = 0; split < 2; ++split) {
          if (lateRange) {
            source->addDynamicFilter(
                0, std::make_shared<common::BigintRange>(lo, hi, true));
          }
          source->addDynamicFilter(
              3,
              makeRowGroupOnlyFilter(
                  std::make_unique<common::BigintRange>(0, 2, false)));
          source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
          // Late pruning bounds must not become a row predicate, including
          // when another column requires the MultiRange AST path.
          source->addDynamicFilter(
              3,
              makeRowGroupOnlyFilter(
                  std::make_unique<common::BigintRange>(1, 2, false)));
          if (lateRange) {
            // A prepared reader still owns the old, wider Range snapshot.
            source->addDynamicFilter(
                0, std::make_shared<common::BigintRange>(lo, 0, nullAllowed));
          }
          std::vector<RowVectorPtr> actual;
          ContinueFuture future;
          while (auto next = source->next(100, future)) {
            if (!*next) {
              break;
            }
            auto gpu = std::dynamic_pointer_cast<CudfVector>(*next);
            actual.push_back(
                with_arrow::toVeloxColumn(
                    gpu->getTableView(),
                    pool_.get(),
                    asRowType(input->type()),
                    gpu->stream(),
                    cudf::get_current_device_resource_ref()));
            gpu->stream().sync();
          }
          const T second = lateRange ? 0 : hi;
          const T secondC1 = lo + (lateRange ? 20 : 40);
          const int64_t secondC2 = lateRange ? 1 : 3;
          auto expected = makeRowVector(
              {"c0", "c1", "c2", "c3"},
              {nullAllowed
                   ? makeNullableFlatVector<T>({lo, second, std::nullopt})
                   : makeNullableFlatVector<T>({second}),
               nullAllowed ? makeFlatVector<T>(
                                 {static_cast<T>(lo + 10),
                                  secondC1,
                                  static_cast<T>(lo + 50)})
                           : makeFlatVector<T>({secondC1}),
               nullAllowed ? makeNullableFlatVector<int64_t>(
                                 {std::nullopt, secondC2, std::nullopt})
                           : makeNullableFlatVector<int64_t>({secondC2}),
               makeFlatVector<int64_t>(
                   nullAllowed ? 3 : 1, [](auto) { return 0; })});
          ::facebook::velox::exec::test::assertEqualResults({expected}, actual);
        }
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, rejectsDynamicFiltersItCannotApply) {
  auto physicalProbe = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({100, 200}, DECIMAL(18, 2))});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), physicalProbe);

  auto checkJoin = [&](const TypePtr& scanType,
                       bool useGpuJoin,
                       const char* expectedError) {
    auto rowType = ROW("c0", scanType);
    auto build =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(rowType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                buildSide,
                "",
                {"c0"},
                useGpuJoin ? core::JoinType::kInner
                           : core::JoinType::kCountingLeftSemiFilter,
                /*nullAware=*/false,
                /*nullAsValue=*/false)
            .capturePlanNodeId(joinId)
            .planNode();

    AssertQueryBuilder query(plan);
    query.config(CudfConfig::kCudfEnabled, "true")
        .splits(scanId, makeCudfHiveConnectorSplits({filePath}));
    if (expectedError) {
      VELOX_ASSERT_THROW(query.copyResults(pool_.get()), expectedError);
    } else {
      auto expected =
          makeRowVector({"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
      auto task = query.assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_TRUE(stats.at(joinId).operatorStats.count(
          useGpuJoin ? "CudfHashJoinProbe" : "HashProbe"));
      EXPECT_TRUE(stats.at(scanId).dynamicFilterStats.producerNodeIds.empty());
    }
  };

  {
    auto& config = CudfConfig::getInstance();
    const auto previousFallback = config.allowCpuFallback;
    unregisterCudf();
    config.allowCpuFallback = true;
    registerCudf();
    SCOPE_EXIT {
      unregisterCudf();
      config.allowCpuFallback = previousFallback;
      registerCudf();
    };
    checkJoin(DECIMAL(18, 2), false, nullptr);
  }
  checkJoin(DECIMAL(18, 2), true, nullptr);
  checkJoin(BIGINT(), true, "column type does not match dynamic filter type");
}

TEST_F(TableScanTest, exactDynamicFilterCanReplaceInnerJoin) {
  for (int64_t keyScale : {1, 1'000'000}) {
    auto probe = makeRowVector(
        {"p_key", "p_value"},
        {makeNullableFlatVector<int64_t>(
             {keyScale,
              2 * keyScale,
              2 * keyScale,
              3 * keyScale,
              5 * keyScale,
              std::nullopt}),
         makeFlatVector<int64_t>({6, 5, 4, 3, 2, 1})});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), probe);
    createDuckDbTable("t", {probe});
    for (bool duplicate : {false, true}) {
      auto build = makeRowVector(
          {"b_key", "b_value"},
          {makeNullableFlatVector<int64_t>(
               {2 * keyScale, (duplicate ? 2 : 5) * keyScale, std::nullopt}),
           makeFlatVector<int64_t>({5, 1, 0})});
      createDuckDbTable("u", {build});
      for (int mode = 0; mode < 6; ++mode) {
        SCOPED_TRACE(
            fmt::format(
                "keyScale: {}, duplicate: {}, mode: {}",
                keyScale,
                duplicate,
                mode));
        auto ids = std::make_shared<core::PlanNodeIdGenerator>();
        auto buildBuilder = PlanBuilder(ids, pool_.get()).values({build});
        if (mode == 5) {
          buildBuilder.filter("b_key IS NOT NULL");
        }
        auto buildPlan = buildBuilder.planNode();
        core::PlanNodeId scanId;
        core::PlanNodeId joinId;
        const std::vector<std::vector<std::string>> layouts{
            {"p_value", "p_key"},
            {"p_key", "b_value"},
            {},
            {"p_key"},
            {"p_key", "p_key"},
            {"p_key"}};
        const std::vector<std::string> selections{
            "t.p_value, t.p_key",
            "t.p_key, u.b_value",
            "count(*)",
            "t.p_key",
            "t.p_key, t.p_key",
            "t.p_key"};
        auto builder = PlanBuilder(ids, pool_.get());
        builder.startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .tableHandle(makeTableHandle("t", probe->rowType()))
            .outputType(probe->rowType())
            .assignments(
                HiveConnectorTestBase::allRegularColumns(probe->rowType()))
            .endTableScan()
            .capturePlanNodeId(scanId);
        if (mode == 5) {
          builder.filter("p_key IS NOT NULL");
        }
        builder
            .hashJoin(
                {"p_key"},
                {"b_key"},
                buildPlan,
                mode == 3 ? "p_value < b_value" : "",
                layouts[mode],
                core::JoinType::kInner,
                false,
                mode == 5)
            .capturePlanNodeId(joinId);
        if (mode == 2) {
          builder.partialAggregation({}, {"count(1) AS n"})
              .localPartition({})
              .finalAggregation();
        }
        auto task =
            AssertQueryBuilder(builder.planNode(), duckDbQueryRunner_)
                .maxDrivers(4)
                .splits(scanId, makeCudfHiveConnectorSplits({file}))
                .assertResults(
                    fmt::format(
                        "SELECT {} FROM t JOIN u ON t.p_key=u.b_key {}",
                        selections[mode],
                        mode == 3 ? "WHERE t.p_value < u.b_value" : ""));
        const auto stats = toPlanStats(task->taskStats());
        EXPECT_EQ(
            stats.at(joinId).customStats.count(
                "replacedWithDynamicFilterRows") != 0,
            !duplicate && (mode == 0 || mode == 2));
        if (mode == 5) {
          EXPECT_EQ(
              stats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
        } else {
          EXPECT_GT(
              stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 0);
        }
      }
    }
  }
}

TEST_F(TableScanTest, rowGroupFilterKeepsFullJoin) {
  auto probe = makeRowVector(
      {"p_key"}, {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt})});
  auto build = makeRowVector(
      {"b_key", "b_value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  auto probeFile = TempFilePath::create();
  auto buildFile = TempFilePath::create();
  writeToFile(probeFile->getPath(), probe);
  writeToFile(buildFile->getPath(), build);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    common::SubfieldFilters filters;
    if (mode == 1) {
      filters.emplace(
          common::Subfield("b_key"),
          std::make_shared<common::BigintRange>(1, 1, false));
    }
    core::PlanNodeId buildScanId;
    auto buildBuilder = PlanBuilder(ids, pool_.get());
    buildBuilder.startTableScan()
        .connectorId(kCudfHiveConnectorId)
        .tableHandle(makeTableHandle("u", build->rowType(), std::move(filters)))
        .outputType(build->rowType())
        .assignments(HiveConnectorTestBase::allRegularColumns(build->rowType()))
        .endTableScan()
        .capturePlanNodeId(buildScanId)
        .project({"b_key", "b_value"})
        .localPartition({});
    if (mode == 3) {
      buildBuilder.filter("b_key = 1");
    }
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    const bool probeOnly = mode == 2;
    auto plan =
        PlanBuilder(ids, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .tableHandle(makeTableHandle("t", probe->rowType()))
            .outputType(probe->rowType())
            .assignments(
                HiveConnectorTestBase::allRegularColumns(probe->rowType()))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"p_key"},
                {"b_key"},
                buildBuilder.planNode(),
                "",
                probeOnly ? std::vector<std::string>{"p_key"}
                          : std::vector<std::string>{"p_key", "b_value"},
                core::JoinType::kInner)
            .capturePlanNodeId(joinId)
            .planNode();
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .splits(scanId, makeCudfHiveConnectorSplits({probeFile}))
            .splits(buildScanId, makeCudfHiveConnectorSplits({buildFile}))
            .assertResults(
                fmt::format(
                    "SELECT t.p_key{} FROM t JOIN u ON t.p_key=u.b_key{}",
                    probeOnly ? "" : ", u.b_value",
                    mode == 1 || mode == 3 ? " WHERE u.b_key=1" : ""));
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_GT(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 0);
    EXPECT_EQ(stats.at(scanId).outputRows, probeOnly ? 2 : 4);
  }
}

TEST_F(TableScanTest, rowGroupFilterMergesWithExactFilter) {
  auto rowType = ROW("c0", BIGINT());
  auto probe = makeRowVector(
      {"c0"}, {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt})});
  auto firstBuild = makeRowVector(
      {"c0", "payload"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {firstBuild});
  for (int64_t key : {2, 3}) {
    SCOPED_TRACE(key);
    auto secondBuild = makeRowVector({"c0"}, {makeFlatVector<int64_t>({key})});
    createDuckDbTable("v", {secondBuild});
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    auto firstBuildSide = PlanBuilder(ids, pool_.get())
                              .values({firstBuild})
                              .project({"c0 AS u_key", "payload"})
                              .planNode();
    auto secondBuildSide = PlanBuilder(ids, pool_.get())
                               .values({secondBuild})
                               .project({"c0 AS v_key"})
                               .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan =
        PlanBuilder(ids, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(rowType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                firstBuildSide,
                "",
                {"c0", "payload"},
                core::JoinType::kInner)
            .hashJoin(
                {"c0"},
                {"v_key"},
                secondBuildSide,
                "",
                {"c0", "payload"},
                core::JoinType::kInner)
            .capturePlanNodeId(joinId)
            .planNode();
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
            .assertResults(
                "SELECT t.c0, u.payload FROM t JOIN u ON t.c0=u.c0 JOIN v ON t.c0=v.c0");
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
    // The second join can be replaced only after the merged predicate is fully
    // enforced. Keeping the pruning-only marker would leak keys 1 and 3.
    EXPECT_EQ(stats.at(scanId).outputRows, key == 2 ? 1 : 0);
    if (key == 2) {
      EXPECT_GT(
          stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum,
          0);
    }
  }
}

TEST_F(TableScanTest, dynamicFilterStopsAtCpuScanConversion) {
  const std::string cpuConnectorId{"cpu-hive-dynamic-filter"};
  facebook::velox::connector::hive::HiveConnectorFactory factory;
  auto cpuConnector = factory.newConnector(
      cpuConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}),
      ioExecutor_.get());
  ConnectorRegistry::global().insert(cpuConnectorId, cpuConnector);
  parquet::registerParquetReaderFactory();
  SCOPE_EXIT {
    parquet::unregisterParquetReaderFactory();
    ConnectorRegistry::global().erase(cpuConnectorId);
  };

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({101, 102, 103, 104}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto build = makeRowVector({"c0"}, {makeFlatVector<int64_t>({2, 4})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .startTableScan()
          .connectorId(cpuConnectorId)
          .outputType(rowType)
          .dataColumns(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c1"},
              {"u_key"},
              buildSide,
              "",
              {"c0", "c1"},
              core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  std::vector<std::shared_ptr<ConnectorSplit>> splits{
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(
          filePath->getPath())
          .connectorId(cpuConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build()};

  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2)
          .splits(scanId, splits)
          .assertResults("SELECT t.c0, t.c1 FROM t JOIN u ON t.c1 = u.c0");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_TRUE(stats.at(joinId).operatorStats.count("CudfFromVelox"));
  EXPECT_EQ(stats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
  EXPECT_EQ(stats.at(scanId).customStats.count("dynamicFiltersAccepted"), 0);
  EXPECT_EQ(stats.at(scanId).outputRows, probe->size());
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
