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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/io/parquet.hpp>

#include <cuda_runtime_api.h>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>

#include <functional>

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
    parquet::registerParquetReaderFactory();
  }

  static void TearDownTestCase() {
    parquet::unregisterParquetReaderFactory();
    CudfHiveConnectorTestBase::TearDownTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  std::shared_ptr<TempFilePath> writeUnsupportedMapInput() {
    auto input = makeRowVector(
        {"k", "m"},
        {makeFlatVector<int64_t>({1, 2}),
         makeMapVector<int64_t, int64_t>({{{10, 100}}, {{20, 200}}})});
    auto filePath = TempFilePath::create();
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool =
        rootPool_->addAggregateChild("TableScanTest.UnsupportedMapWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    parquet::Writer writer(
        std::move(sink), options, writerPool, asRowType(input->type()));
    writer.write(input);
    writer.close();
    return filePath;
  }

  Split makeCudfHiveSplit(std::string path, int64_t splitWeight = 0) {
    return Split(makeCudfHiveConnectorSplit(std::move(path), splitWeight));
  }

  // Returns a split of 'filePath' that carries 'partitionKeys'.
  static std::shared_ptr<facebook::velox::connector::ConnectorSplit>
  makePartitionedSplit(
      const std::string& filePath,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys) {
    auto builder = hive::HiveConnectorSplitBuilder(filePath)
                       .connectorId(kCudfHiveConnectorId)
                       .fileFormat(dwio::common::FileFormat::PARQUET);
    for (const auto& [name, value] : partitionKeys) {
      builder.partitionKey(name, value);
    }
    return builder.build();
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

  void writeCompactDecimalParquet(
      const std::shared_ptr<TempFilePath>& filePath,
      const RowVectorPtr& vector) {
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
    parquet::Writer writer(
        std::move(sink), options, writerPool, vector->rowType());
    writer.write(vector);
    writer.close();
  }

  void assertDecimalScanRoundTrip(
      const RowVectorPtr& vector,
      const RowTypePtr& rowType) {
    auto filePath = TempFilePath::create();
    writeCompactDecimalParquet(filePath, vector);
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

TEST_P(TableScanTestParameterized, ignoreMissingFiles) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using hive::HiveConfig;
  auto directory = TempDirectoryPath::create();
  const auto missingPath = directory->getPath() + "/missing.parquet";
  auto missingSplit = [&]() {
    return hive::HiveConnectorSplitBuilder(missingPath)
        .connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .length(123)
        .build();
  };
  auto vectors = makeVectors(1, 100);
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), vectors);
  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const auto useBufferedInput = GetParam() ? "true" : "false";
  EXPECT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              CudfHiveConfig::kUseBufferedInputSession,
              useBufferedInput)
          .split(missingSplit())
          .assertEmptyResults(),
      std::exception);

  for (const auto preload : {0, 4}) {
    std::vector<std::shared_ptr<ConnectorSplit>> splits{missingSplit()};
    auto existingSplits = makeCudfHiveConnectorSplits({file});
    splits.insert(splits.end(), existingSplits.begin(), existingSplits.end());
    splits.push_back(missingSplit());
    auto moreExistingSplits = makeCudfHiveConnectorSplits({file});
    splits.insert(
        splits.end(), moreExistingSplits.begin(), moreExistingSplits.end());
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .maxDrivers(1)
            .config(
                QueryConfig::kMaxSplitPreloadPerDriver, std::to_string(preload))
            .connectorSessionProperty(
                kCudfHiveConnectorId,
                CudfHiveConfig::kUseBufferedInputSession,
                useBufferedInput)
            .connectorSessionProperty(
                kCudfHiveConnectorId,
                HiveConfig::kIgnoreMissingFilesSession,
                "true")
            .splits(splits)
            .assertResults("SELECT * FROM tmp UNION ALL SELECT * FROM tmp");
    const auto stats = getTableScanRuntimeStats(task);
    EXPECT_EQ(stats.count("skippedSplits"), 0);
    EXPECT_EQ(stats.count("skippedSplitBytes"), 0);
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
  }
}

TEST_F(TableScanTest, ignoreMissingFilesPreservesReadErrors) {
  // Preserve the actual open failure instead of retrying through KvikIO.
  filesystems::registerFileSystem(
      [](std::string_view path) { return path.starts_with("denied:"); },
      [](std::shared_ptr<const config::ConfigBase>,
         std::string_view) -> std::shared_ptr<filesystems::FileSystem> {
        VELOX_FAIL("Injected permission denied at open");
      });
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(tableScanNode())
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              cudf_velox::connector::hive::CudfHiveConfig::
                  kUseBufferedInputSession,
              "true")
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kIgnoreMissingFilesSession,
              "true")
          .split(makeCudfHiveConnectorSplit("denied:/missing.parquet"))
          .assertEmptyResults(),
      "Injected permission denied at open");
}

TEST_F(TableScanTest, ignoreMissingFilesPreservesCorruption) {
  // An existing empty file is invalid Parquet, not a missing split.
  auto file = TempFilePath::create();
  EXPECT_THROW(
      AssertQueryBuilder(tableScanNode())
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kIgnoreMissingFilesSession,
              "true")
          .splits(makeCudfHiveConnectorSplits({file}))
          .assertEmptyResults(),
      std::exception);
}

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

// The split's size differs from the file's so that the assertion tells which
// one the scan reports. Local opens take the size from the file itself.
TEST_F(TableScanTest, rawInputBytesFromSplitFileSize) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  auto vectors = makeVectors(2, 100);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);
  const int64_t fileSize = filePath->fileSize();
  const int64_t splitFileSize = fileSize + 4'096;

  struct FileSizeSource {
    std::string name;
    std::function<void(hive::HiveConnectorSplitBuilder&)> setFileSize;
    int64_t expectedRawInputBytes;
  };
  const std::vector<FileSizeSource> fileSizeSources = {
      {"none", [](auto&) {}, fileSize},
      {"$file_size",
       [&](auto& builder) {
         builder.infoColumn("$file_size", std::to_string(splitFileSize));
       },
       splitFileSize},
      {"properties",
       [&](auto& builder) {
         builder.fileProperties(FileProperties{.fileSize = splitFileSize});
       },
       splitFileSize},
  };

  for (const auto* useBufferedInput : {"true", "false"}) {
    for (const auto& source : fileSizeSources) {
      SCOPED_TRACE(
          fmt::format(
              "use_buffered_input={} file size from {}",
              useBufferedInput,
              source.name));
      // A scan consumes its split, so every case builds a fresh one.
      hive::HiveConnectorSplitBuilder builder(filePath->getPath());
      builder.connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET);
      source.setFileSize(builder);
      auto task = AssertQueryBuilder(tableScanNode(), duckDbQueryRunner_)
                      .connectorSessionProperty(
                          kCudfHiveConnectorId,
                          CudfHiveConfig::kUseBufferedInputSession,
                          useBufferedInput)
                      .split(builder.build())
                      .assertResults("SELECT * FROM tmp");
      EXPECT_EQ(
          getTableScanStats(task).rawInputBytes, source.expectedRawInputBytes);
    }
  }
}

TEST_P(TableScanTestParameterized, scanOnNonDefaultGpu) {
  int deviceCount = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }

  int originalDevice = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&originalDevice));
  const int selectedDevice = originalDevice == 0 ? 1 : 0;
  ensureCudaContextForThread();
  unregisterCudf();
  SCOPE_EXIT {
    unregisterCudf();
    EXPECT_EQ(cudaSuccess, cudaSetDevice(originalDevice));
    registerCudf();
  };
  ASSERT_EQ(cudaSuccess, cudaSetDevice(selectedDevice));
  registerCudf();
  // Verify re-registration rebinds a thread still using the previous device.
  ASSERT_EQ(cudaSuccess, cudaSetDevice(originalDevice));
  ensureCudaContextForThread();
  int boundDevice = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&boundDevice));
  ASSERT_EQ(selectedDevice, boundDevice);
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       GetParam() ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

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

// Partition keys are not in the data file: the scan materializes each from
// the split's partition value, typed per its column handle, wherever the
// output puts it.
TEST_F(TableScanTest, partitionedTable) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(10, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  const auto scan =
      [&](const RowTypePtr& outputType,
          const facebook::velox::connector::ColumnHandleMap& assignments) {
        return PlanBuilder(pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(outputType)
            .assignments(assignments)
            .endTableScan()
            .planNode();
      };

  // Scans a split in the partition 'partitionValue' of a key of
  // 'partitionType', expecting 'expectedValueSql' for the key. A split is
  // consumed by the task it is added to, so each scan gets its own.
  const auto assertPartitionedScan =
      [&](const TypePtr& partitionType,
          const std::optional<std::string>& partitionValue,
          const std::string& expectedValueSql) {
        SCOPED_TRACE(
            fmt::format(
                "{} key {}",
                partitionType->toString(),
                partitionValue.value_or("null")));
        const auto makeSplit = [&]() {
          return makePartitionedSplit(
              filePath->getPath(), {{"pkey", partitionValue}});
        };
        const facebook::velox::connector::ColumnHandleMap assignments = {
            {"pkey",
             HiveConnectorTestBase::partitionKey("pkey", partitionType)},
            {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
            {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())}};

        // The key at every output position.
        assertQuery(
            scan(
                ROW({"pkey", "c0", "c1"}, {partitionType, BIGINT(), DOUBLE()}),
                assignments),
            makeSplit(),
            fmt::format("SELECT {}, c0, c1 FROM tmp", expectedValueSql));
        assertQuery(
            scan(
                ROW({"c0", "pkey", "c1"}, {BIGINT(), partitionType, DOUBLE()}),
                assignments),
            makeSplit(),
            fmt::format("SELECT c0, {}, c1 FROM tmp", expectedValueSql));
        assertQuery(
            scan(
                ROW({"c0", "c1", "pkey"}, {BIGINT(), DOUBLE(), partitionType}),
                assignments),
            makeSplit(),
            fmt::format("SELECT c0, c1, {} FROM tmp", expectedValueSql));
        // The key alone: nothing is read from the file but its row count.
        assertQuery(
            scan(ROW({"pkey"}, {partitionType}), assignments),
            makeSplit(),
            fmt::format("SELECT {} FROM tmp", expectedValueSql));
      };

  // A value is parsed per the key's type.
  assertPartitionedScan(VARCHAR(), "2020-11-01", "'2020-11-01'");
  assertPartitionedScan(BIGINT(), "123456789123456789", "'123456789123456789'");
  assertPartitionedScan(INTEGER(), "123456789", "'123456789'");
  assertPartitionedScan(SMALLINT(), "1", "'1'");
  assertPartitionedScan(TINYINT(), "1", "'1'");
  assertPartitionedScan(BOOLEAN(), "0", "'0'");
  assertPartitionedScan(REAL(), "3.5", "'3.5'");
  assertPartitionedScan(DOUBLE(), "3.5", "'3.5'");
  assertPartitionedScan(
      DECIMAL(10, 2), "1234.56", "CAST(1234.56 AS DECIMAL(10, 2))");
  assertPartitionedScan(DATE(), "2023-10-27", "'2023-10-27'");
  // A NULL value, whatever the type.
  for (const auto& type :
       std::vector<TypePtr>{VARCHAR(), BIGINT(), DATE(), TIMESTAMP()}) {
    assertPartitionedScan(type, std::nullopt, "null");
  }

  {
    // A split read in several batches carries the key in each of them.
    SCOPED_TRACE("several batches");
    auto plan = scan(
        ROW({"pkey", "c0", "c1"}, {VARCHAR(), BIGINT(), DOUBLE()}),
        {{"pkey", HiveConnectorTestBase::partitionKey("pkey", VARCHAR())},
         {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
         {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())}});
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kMaxChunkReadLimitSession,
                        "8192")
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kMaxPassReadLimitSession,
                        "32768")
                    .splits({makePartitionedSplit(
                        filePath->getPath(), {{"pkey", "a"}})})
                    .assertResults("SELECT 'a', c0, c1 FROM tmp");
    const auto planStats = toPlanStats(task->taskStats());
    EXPECT_GT(
        planStats.at(plan->id()).operatorStatsFor("TableScan").outputVectors,
        1);
  }

  {
    // A DATE value is a count of days since the epoch when its handle says so.
    SCOPED_TRACE("days since epoch");
    const facebook::velox::connector::ColumnHandleMap assignments = {
        {"pkey",
         std::make_shared<hive::HiveColumnHandle>(
             "pkey",
             hive::HiveColumnHandle::ColumnType::kPartitionKey,
             DATE(),
             DATE(),
             std::vector<common::Subfield>{},
             hive::HiveColumnHandle::ColumnParseParameters{
                 hive::HiveColumnHandle::ColumnParseParameters::
                     kDaysSinceEpoch})},
        {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}};
    assertQuery(
        scan(ROW({"pkey", "c0"}, {DATE(), BIGINT()}), assignments),
        makePartitionedSplit(filePath->getPath(), {{"pkey", "19657"}}),
        "SELECT '2023-10-27', c0 FROM tmp");
  }

  {
    // A TIMESTAMP value is local time or UTC, as the session says.
    SCOPED_TRACE("local time");
    const std::string partitionValue = "2023-10-27 00:12:35";
    auto plan = scan(
        ROW({"pkey", "c0"}, {TIMESTAMP(), BIGINT()}),
        {{"pkey", HiveConnectorTestBase::partitionKey("pkey", TIMESTAMP())},
         {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}});
    const auto utcTimestamp =
        util::fromTimestampString(
            StringView(partitionValue), util::TimestampParseMode::kPrestoCast)
            .thenOrThrow(folly::identity, [&](const Status& status) {
              VELOX_USER_FAIL("{}", status.message());
            });
    auto localTimestamp = utcTimestamp;
    localTimestamp.toGMT(Timestamp::defaultTimezone());
    for (const auto& [asLocalTime, expected] :
         {std::pair{false, utcTimestamp}, std::pair{true, localTimestamp}}) {
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kReadTimestampPartitionValueAsLocalTimeSession,
              asLocalTime ? "true" : "false")
          .splits({makePartitionedSplit(
              filePath->getPath(), {{"pkey", partitionValue}})})
          .assertResults(
              fmt::format("SELECT '{}', c0 FROM tmp", expected.toString()));
    }
  }

  {
    // An output name need not be the key's name, a key may be projected under
    // several names, and a key the split carries but no handle requests is
    // ignored.
    SCOPED_TRACE("alias");
    const auto makeSplit = [&]() {
      return makePartitionedSplit(filePath->getPath(), {{"ds", "2021-12-02"}});
    };
    assertQuery(
        scan(
            ROW({"a", "ds_alias"}, {BIGINT(), VARCHAR()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
             {"ds_alias",
              HiveConnectorTestBase::partitionKey("ds", VARCHAR())}}),
        makeSplit(),
        "SELECT c0, '2021-12-02' FROM tmp");
    assertQuery(
        scan(
            ROW({"ds_alias", "a", "ds"}, {VARCHAR(), BIGINT(), VARCHAR()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
             {"ds_alias", HiveConnectorTestBase::partitionKey("ds", VARCHAR())},
             {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}}),
        makeSplit(),
        "SELECT '2021-12-02', c0, '2021-12-02' FROM tmp");
    assertQuery(
        scan(
            ROW({"a"}, {BIGINT()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())}}),
        makeSplit(),
        "SELECT c0 FROM tmp");
  }
}

// A TIMESTAMP partition key takes the type the reader gives file timestamps,
// whatever unit cuDF expressions default to, so an expression can combine the
// two.
TEST_F(TableScanTest, partitionTimestampMixedWithFileTimestamp) {
  auto& config = CudfConfig::getInstance();
  const auto originalUnit = config.timestampUnit;
  SCOPE_EXIT {
    config.timestampUnit = originalUnit;
  };

  auto input = makeRowVector(
      {"flag", "file_ts"},
      {makeFlatVector<bool>({true, false}),
       makeFlatVector<Timestamp>(
           {Timestamp(1'700'000'000, 0), Timestamp(1'700'000'001, 0)})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), input);
  const auto outputType =
      ROW({"flag", "file_ts", "partition_ts"},
          {BOOLEAN(), TIMESTAMP(), TIMESTAMP()});
  const auto fileValues = makeRowVector({input->childAt(1)});
  const auto caseValues = makeRowVector({makeFlatVector<Timestamp>(
      {Timestamp(1'577'836'800, 0), Timestamp(1'700'000'001, 0)})});

  // The reader's timestamp type is pinned to a unit distinct from both default
  // units below, so each scan proves the key follows the reader rather than a
  // fixed unit. The connector and session settings take the enumerator's
  // integer value.
  const auto readerTimestampType = std::to_string(
      static_cast<int32_t>(cudf::type_id::TIMESTAMP_MICROSECONDS));

  // Projects 'projection' over the file and partition timestamps of a split
  // in the partition 'partitionValue', expecting 'expected'. The session pins
  // the reader's timestamp type to 'sessionTimestampType' when that is set.
  const auto check = [&](const std::string& projection,
                         const std::optional<std::string>& partitionValue,
                         const RowVectorPtr& expected,
                         const std::optional<std::string>&
                             sessionTimestampType) {
    SCOPED_TRACE(projection);
    auto plan =
        PlanBuilder(pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(outputType)
            .assignments(
                {{"flag",
                  HiveConnectorTestBase::regularColumn("flag", BOOLEAN())},
                 {"file_ts",
                  HiveConnectorTestBase::regularColumn("file_ts", TIMESTAMP())},
                 {"partition_ts",
                  HiveConnectorTestBase::partitionKey(
                      "partition_ts", TIMESTAMP())}})
            .endTableScan()
            .project({projection})
            .planNode();
    AssertQueryBuilder builder(plan, duckDbQueryRunner_);
    builder.connectorSessionProperty(
        kCudfHiveConnectorId,
        hive::HiveConfig::kReadTimestampPartitionValueAsLocalTimeSession,
        "false");
    if (sessionTimestampType.has_value()) {
      builder.connectorSessionProperty(
          kCudfHiveConnectorId,
          cudf_velox::connector::hive::CudfHiveConfig::kTimestampTypeSession,
          *sessionTimestampType);
    }
    builder
        .splits({makePartitionedSplit(
            filePath->getPath(), {{"partition_ts", partitionValue}})})
        .assertResults(expected);
  };

  for (const auto& [unit, label] :
       {std::pair{cudf::type_id::TIMESTAMP_MILLISECONDS, "ms default unit"},
        std::pair{cudf::type_id::TIMESTAMP_NANOSECONDS, "ns default unit"}}) {
    SCOPED_TRACE(label);
    config.timestampUnit = unit;
    check("file_ts", std::nullopt, fileValues, readerTimestampType);
    check(
        "coalesce(partition_ts, file_ts)",
        std::nullopt,
        fileValues,
        readerTimestampType);
    check(
        "CASE WHEN flag THEN partition_ts ELSE file_ts END",
        "2020-01-01 00:00:00",
        caseValues,
        readerTimestampType);
  }

  // Without a session timestamp type, the key follows the connector's
  // parquet.reader.timestamp-type.
  {
    SCOPED_TRACE("connector timestamp type");
    config.timestampUnit = cudf::type_id::TIMESTAMP_MILLISECONDS;
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {cudf_velox::connector::hive::CudfHiveConfig::kTimestampType,
                 readerTimestampType}}));
    check(
        "coalesce(partition_ts, file_ts)",
        /*partitionValue=*/std::nullopt,
        fileValues,
        /*sessionTimestampType=*/std::nullopt);
  }
}

// A pushed-down filter on a partition key is decided once per split from the
// split's partition value: a split it rejects is skipped without being read.
TEST_F(TableScanTest, partitionKeyFilter) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  using PartitionKeys =
      std::unordered_map<std::string, std::optional<std::string>>;
  // The same file in the partitions 'a', 'b' and NULL. A split is consumed by
  // the task it is added to, so each scan gets fresh ones.
  std::vector<PartitionKeys> partitions = {
      PartitionKeys{{"ds", "a"}},
      PartitionKeys{{"ds", "b"}},
      PartitionKeys{{"ds", std::nullopt}}};
  const auto makeSplits = [&]() {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    for (const auto& partition : partitions) {
      splits.push_back(makePartitionedSplit(filePath->getPath(), partition));
    }
    return splits;
  };
  facebook::velox::connector::ColumnHandleMap assignments = {
      {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
      {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}};

  // Scans the partitions with 'filter' on the key, checking the rows and how
  // many splits were skipped. The table schema lists the file columns only.
  const auto assertFilteredScan = [&](std::unique_ptr<common::Filter> filter,
                                      const RowTypePtr& outputType,
                                      const std::string& expectedSql,
                                      int64_t expectedSkippedSplits) {
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(outputType)
                    .tableHandle(makeTableHandle(
                        "parquet_table",
                        rowType,
                        common::test::SubfieldFiltersBuilder()
                            .add("ds", std::move(filter))
                            .build()))
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();
    auto task = OperatorTestBase::assertQuery(plan, makeSplits(), expectedSql);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "skippedSplits"), expectedSkippedSplits);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "processedSplits"),
        static_cast<int64_t>(partitions.size()) - expectedSkippedSplits);
  };

  {
    SCOPED_TRACE("ds = 'a'");
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"a"}, false),
        ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
        "SELECT c0, 'a' FROM tmp",
        2);
  }
  {
    SCOPED_TRACE("ds IS NULL");
    assertFilteredScan(
        std::make_unique<common::IsNull>(),
        ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
        "SELECT c0, null FROM tmp",
        2);
  }
  {
    // The key need not be projected.
    SCOPED_TRACE("ds IS NOT NULL");
    assertFilteredScan(
        std::make_unique<common::IsNotNull>(),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp UNION ALL SELECT c0 FROM tmp",
        1);
  }
  {
    SCOPED_TRACE("ds = 'b' OR ds IS NULL");
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"b"}, true),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp UNION ALL SELECT c0 FROM tmp",
        1);
  }
  {
    // A key the split does not carry is NULL.
    SCOPED_TRACE("missing key");
    partitions = {PartitionKeys{}};
    assertFilteredScan(
        std::make_unique<common::IsNull>(),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp",
        0);
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"a"}, false),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp WHERE false",
        1);
  }
  {
    // A typed key with a range filter: ds > DATE '2020-09-01' over day counts.
    SCOPED_TRACE("date range");
    partitions = {
        PartitionKeys{{"ds", "19657"}}, PartitionKeys{{"ds", "18000"}}};
    assignments["ds"] = std::make_shared<hive::HiveColumnHandle>(
        "ds",
        hive::HiveColumnHandle::ColumnType::kPartitionKey,
        DATE(),
        DATE(),
        std::vector<common::Subfield>{},
        hive::HiveColumnHandle::ColumnParseParameters{
            hive::HiveColumnHandle::ColumnParseParameters::kDaysSinceEpoch});
    assertFilteredScan(
        std::make_unique<common::BigintRange>(
            18506, std::numeric_limits<int64_t>::max(), false),
        ROW({"c0", "ds"}, {BIGINT(), DATE()}),
        "SELECT c0, '2023-10-27' FROM tmp",
        1);
  }
  {
    // A key only the filter references comes with the table handle's filter
    // column handles.
    SCOPED_TRACE("filter column handle");
    partitions = {PartitionKeys{{"ds", "a"}}, PartitionKeys{{"ds", "b"}}};
    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        kCudfHiveConnectorId,
        "parquet_table",
        common::test::SubfieldFiltersBuilder()
            .add(
                "ds",
                std::make_unique<common::BytesValues>(
                    std::vector<std::string>{"a"}, false))
            .build(),
        nullptr,
        rowType,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{},
        std::vector<hive::HiveColumnHandlePtr>{
            HiveConnectorTestBase::partitionKey("ds", VARCHAR())});
    auto plan =
        PlanBuilder(pool_.get())
            .startTableScan()
            .outputType(ROW({"c0"}, {BIGINT()}))
            .tableHandle(tableHandle)
            .assignments(
                {{"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}})
            .endTableScan()
            .planNode();
    auto task =
        OperatorTestBase::assertQuery(plan, makeSplits(), "SELECT c0 FROM tmp");
    EXPECT_EQ(getTableScanRuntimeStat(task, "skippedSplits"), 1);
  }
  {
    // The decision on a split prepared ahead of time carries over to the
    // driver that adopts it.
    SCOPED_TRACE("preloaded splits");
    partitions.clear();
    for (int i = 0; i < 3; ++i) {
      partitions.insert(
          partitions.end(),
          {PartitionKeys{{"ds", "a"}},
           PartitionKeys{{"ds", "b"}},
           PartitionKeys{{"ds", std::nullopt}}});
    }
    assignments["ds"] = HiveConnectorTestBase::partitionKey("ds", VARCHAR());
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}))
                    .tableHandle(makeTableHandle(
                        "parquet_table",
                        rowType,
                        common::test::SubfieldFiltersBuilder()
                            .add(
                                "ds",
                                std::make_unique<common::BytesValues>(
                                    std::vector<std::string>{"a"}, false))
                            .build()))
                    .assignments(assignments)
                    .endTableScan()
                    .singleAggregation({"ds"}, {"count(1)"})
                    .planNode();
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "4")
                    .splits(makeSplits())
                    .assertResults("SELECT 'a', 3 * count(*) FROM tmp");
    EXPECT_EQ(getTableScanRuntimeStat(task, "skippedSplits"), 6);
    EXPECT_EQ(getTableScanRuntimeStat(task, "processedSplits"), 3);
  }
}

// The remaining filter sees a partition key as a constant column of the split,
// whether or not the key is projected and whether or not a file column is
// read along with it.
TEST_F(TableScanTest, partitionKeyRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // The same file in the partitions 'a' and 'bb'. A split is consumed by the
  // task it is added to, so each scan gets fresh ones.
  const auto makeSplits = [&]() {
    return std::vector<
        std::shared_ptr<facebook::velox::connector::ConnectorSplit>>{
        makePartitionedSplit(filePath->getPath(), {{"ds", "a"}}),
        makePartitionedSplit(filePath->getPath(), {{"ds", "bb"}})};
  };
  const facebook::velox::connector::ColumnHandleMap assignments = {
      {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
      {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())},
      {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}};
  // The filter is parsed against the table schema, which lists the key here.
  const auto dataColumns =
      ROW({"c0", "c1", "ds"}, {BIGINT(), DOUBLE(), VARCHAR()});

  const auto assertFilteredScan = [&](const RowTypePtr& outputType,
                                      const std::string& remainingFilter,
                                      const std::string& expectedSql,
                                      int64_t expectedSkippedSplits) {
    SCOPED_TRACE(remainingFilter);
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(outputType)
                    .dataColumns(dataColumns)
                    .assignments(assignments)
                    .remainingFilter(remainingFilter)
                    .endTableScan()
                    .planNode();
    auto task = OperatorTestBase::assertQuery(plan, makeSplits(), expectedSql);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "skippedSplits"), expectedSkippedSplits);
  };

  // A filter over the key and a file column runs after the read, with the key
  // projected or not.
  assertFilteredScan(
      ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
      "ds = 'a' OR c0 > 500",
      "SELECT c0, 'a' FROM tmp UNION ALL "
      "SELECT c0, 'bb' FROM tmp WHERE c0 > 500",
      0);
  assertFilteredScan(
      ROW({"c1"}, {DOUBLE()}),
      "ds = 'a' OR c0 > 500",
      "SELECT c1 FROM tmp UNION ALL SELECT c1 FROM tmp WHERE c0 > 500",
      0);
  // A comparison of the key alone is decided per split.
  assertFilteredScan(
      ROW({"c0"}, {BIGINT()}), "ds = 'bb'", "SELECT c0 FROM tmp", 1);
  // Any other filter on the key alone runs over the materialized key, even
  // when no file column is read.
  assertFilteredScan(
      ROW({"c0"}, {BIGINT()}), "length(ds) = 2", "SELECT c0 FROM tmp", 0);
  assertFilteredScan(
      ROW({"ds"}, {VARCHAR()}), "length(ds) = 2", "SELECT 'bb' FROM tmp", 0);
}

// A split round-trips through its serialized form with its partition keys,
// NULL ones included.
TEST_F(TableScanTest, partitionedSplitSerde) {
  auto split = CudfHiveConnectorSplitBuilder("/path/to/file.parquet")
                   .infoColumn("$file_size", "123")
                   .partitionKey("ds", "2021-12-02")
                   .partitionKey("region", std::nullopt)
                   .build();
  auto copy = cudf_velox::connector::hive::CudfHiveConnectorSplit::create(
      split->serialize());
  EXPECT_EQ(copy->filePath, split->filePath);
  EXPECT_EQ(copy->infoColumns, split->infoColumns);
  EXPECT_EQ(copy->partitionKeys, split->partitionKeys);
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

TEST_F(TableScanTest, decimalFilterPushdown) {
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

TEST_F(TableScanTest, decimalStatsFilterIoPruning) {
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

// A split carries the file size in its properties or, as Presto sends it, in
// the $file_size info column; both survive a serialization round trip.
TEST_F(TableScanTest, splitCarriesFileSize) {
  using cudf_velox::connector::hive::CudfHiveConnectorSplit;
  auto split =
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(
              FileProperties{.fileSize = 1'234, .modificationTime = 5'678})
          .build();
  auto copy = CudfHiveConnectorSplit::create(split->serialize());
  ASSERT_TRUE(copy->properties.has_value());
  EXPECT_EQ(copy->properties->fileSize, 1'234);
  EXPECT_EQ(copy->properties->modificationTime, 5'678);
  EXPECT_EQ(copy->fileSize(), 1'234);

  auto fromInfoColumn = CudfHiveConnectorSplitBuilder("/data/file.parquet")
                            .infoColumn("$file_size", "1234")
                            .build();
  EXPECT_FALSE(fromInfoColumn->properties.has_value());
  EXPECT_EQ(fromInfoColumn->fileSize(), 1'234);
  EXPECT_EQ(
      CudfHiveConnectorSplit::create(fromInfoColumn->serialize())->fileSize(),
      1'234);

  auto bare = CudfHiveConnectorSplitBuilder("/data/file.parquet").build();
  EXPECT_FALSE(bare->fileSize().has_value());
  EXPECT_FALSE(
      CudfHiveConnectorSplit::create(bare->serialize())
          ->properties.has_value());

  // A negative size in the properties is no size, so the info column stands
  // in for it, and a valid size in the properties wins over the info column.
  EXPECT_EQ(
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(FileProperties{.fileSize = -1})
          .infoColumn("$file_size", "1234")
          .build()
          ->fileSize(),
      1'234);
  EXPECT_FALSE(CudfHiveConnectorSplitBuilder("/data/file.parquet")
                   .fileProperties(FileProperties{.fileSize = -1})
                   .build()
                   ->fileSize()
                   .has_value());
  EXPECT_EQ(
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(FileProperties{.fileSize = 1'234})
          .infoColumn("$file_size", "5678")
          .build()
          ->fileSize(),
      1'234);
  for (const auto* notASize : {"", "abc", "-1", "12.5"}) {
    EXPECT_FALSE(CudfHiveConnectorSplitBuilder("/data/file.parquet")
                     .infoColumn("$file_size", notASize)
                     .build()
                     ->fileSize()
                     .has_value())
        << notASize;
  }
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

TEST_F(TableScanTest, mixedCaseDecimalFilterUsesSplitPhysicalType) {
  auto fileVector = makeRowVector(
      {"Price", "Value"},
      {makeFlatVector<int64_t>({100, -500, -700}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3})});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);
  createDuckDbTable({fileVector});

  auto rowType = ROW({{"price", DECIMAL(18, 2)}, {"value", BIGINT()}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "price",
                         std::make_unique<common::BigintRange>(
                             -500, -500, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
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
      .assertResults(
          "SELECT Price, Value FROM tmp "
          "WHERE Price = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, decimalFilterUsesSplitPhysicalType) {
  // Both file schemas fit the table's wider logical precision.
  auto rowType = ROW({{"c0", DECIMAL(18, 2)}, {"c1", BIGINT()}});
  auto decimal32Vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3})});
  auto decimal64Vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({200, -500, -800}, DECIMAL(18, 2)),
       makeFlatVector<int64_t>({4, 5, 6})});

  auto decimal32Path = TempFilePath::create();
  auto decimal64Path = TempFilePath::create();
  writeCompactDecimalParquet(decimal32Path, decimal32Vector);
  writeToFile(decimal64Path->getPath(), {decimal64Vector});
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(
           {100, -500, -700, 200, -500, -800}, DECIMAL(18, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             int64_t{-500},
                             int64_t{-500},
                             /*nullAllowed*/ false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  const auto expected =
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))";
  for (const auto& paths :
       {std::vector{decimal32Path, decimal64Path},
        std::vector{decimal64Path, decimal32Path}}) {
    // A missing file is skipped before its footer is read for the filter.
    auto splits = makeCudfHiveConnectorSplits(paths);
    splits.insert(
        splits.begin() + 1,
        makeCudfHiveConnectorSplit(paths.front()->getPath() + ".missing"));
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .maxDrivers(1)
        .connectorSessionProperty(
            kCudfHiveConnectorId,
            hive::HiveConfig::kIgnoreMissingFilesSession,
            "true")
        .splits(splits)
        .assertResults(expected);
  }
}

TEST_F(TableScanTest, decimalFilterUsesSplitScale) {
  auto rowType = ROW({{"c0", DECIMAL(12, 4)}, {"c1", BIGINT()}});
  auto fileVector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({30000000, 123}, DECIMAL(9, 2)),
       makeFlatVector<int64_t>({1, 2})});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({3000000000, 12300}, DECIMAL(12, 4)),
       makeFlatVector<int64_t>({1, 2})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             int64_t{3000000000},
                             int64_t{3000000000},
                             /*nullAllowed*/ false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT * FROM tmp WHERE c0 = CAST('300000.0000' AS DECIMAL(12, 4))");
}

TEST_F(TableScanTest, decimalFilterRejectsFileScaleAboveTableScale) {
  auto fileVector =
      makeRowVector({makeFlatVector<int64_t>({12399}, DECIMAL(7, 4))});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);

  auto rowType = ROW({{"c0", DECIMAL(5, 2)}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             123, 123, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_THROW(
      assertQuery(plan, {filePath}, "SELECT CAST(1.23 AS DECIMAL(5, 2))"),
      "Parquet decimal scale of 'c0' exceeds the table scale");
}

TEST_F(TableScanTest, decimalFilterUsesRawIntegerStorage) {
  auto rowType = ROW({{"c0", DECIMAL(10, 0)}, {"c1", BIGINT()}});
  auto fileVector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>({123, 456}), makeFlatVector<int64_t>({1, 2})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {fileVector});
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({123, 456}, DECIMAL(10, 0)),
       makeFlatVector<int64_t>({1, 2})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             123, 123, /*nullAllowed=*/false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT * FROM tmp WHERE c0 = CAST(123 AS DECIMAL(10, 0))");
}

TEST_F(TableScanTest, decimalFilterUsesAnnotatedIntegerStorage) {
  auto fileVector =
      makeRowVector({makeFlatVector<int8_t>({-100, 0, 100}, TINYINT())});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {fileVector});
  createDuckDbTable({fileVector});

  auto rowType = ROW({{"c0", DECIMAL(10, 0)}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             -100, -100, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT CAST(c0 AS DECIMAL(10, 0)) FROM tmp WHERE c0 = -100");
}

TEST_F(TableScanTest, nullableDecimalFilterUsesSplitPhysicalType) {
  // The file retains precision 5; the table schema widens it to precision 12.
  auto rowType = ROW({{"c0", DECIMAL(12, 2)}, {"c1", BIGINT()}});
  auto outputType = ROW("c1", BIGINT());
  auto mixedPath = TempFilePath::create();
  auto allNullPath = TempFilePath::create();
  auto nonNullPath = TempFilePath::create();
  writeCompactDecimalParquet(
      mixedPath,
      makeRowVector(
          {"c0", "c1"},
          {makeNullableFlatVector<int64_t>(
               {std::nullopt, -500, 200}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({1, 2, 3})}));
  writeCompactDecimalParquet(
      allNullPath,
      makeRowVector(
          {"c0", "c1"},
          {makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({4, 5, 6})}));
  writeCompactDecimalParquet(
      nonNullPath,
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>({100, -500, 200}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({7, 8, 9})}));

  for (const bool nullAllowed : {false, true}) {
    auto check = [&](std::unique_ptr<common::Filter> filter,
                     bool matchesNonNull) {
      SCOPED_TRACE(filter->toString());
      common::SubfieldFilters filters;
      filters.emplace(common::Subfield("c0"), std::move(filter));
      auto tableHandle = makeTableHandle(
          "parquet_table", rowType, std::move(filters), nullptr);
      auto plan = PlanBuilder()
                      .startTableScan()
                      .outputType(outputType)
                      .tableHandle(tableHandle)
                      .assignments(
                          facebook::velox::exec::test::HiveConnectorTestBase::
                              allRegularColumns(outputType))
                      .endTableScan()
                      .planNode();
      std::vector<int64_t> expectedIds;
      if (nullAllowed) {
        expectedIds = {1, 4, 5, 6};
      }
      if (matchesNonNull) {
        expectedIds.insert(expectedIds.end(), {2, 8});
      }
      auto expected =
          makeRowVector({"c1"}, {makeFlatVector<int64_t>(expectedIds)});
      AssertQueryBuilder(plan)
          .maxDrivers(1)
          .splits(makeCudfHiveConnectorSplits(
              {mixedPath, allNullPath, nonNullPath}))
          .assertResults(expected);
    };
    check(std::make_unique<common::BigintRange>(-500, -500, nullAllowed), true);
    // These bounds fit the logical decimal, but no DECIMAL32 value matches.
    check(
        std::make_unique<common::BigintRange>(
            3'000'000'000, 4'000'000'000, nullAllowed),
        false);
    check(common::createBigintValues({-500, 3'000'000'000}, nullAllowed), true);
    check(
        common::createBigintValues({3'000'000'000, 3'000'000'002}, nullAllowed),
        false);
  }
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

// The data source cannot apply the session's cast settings, so it rejects a
// remaining filter that casts from VARCHAR.
TEST_F(TableScanTest, remainingFilterCastFromVarchar) {
  auto rowType = ROW({"c0"}, {VARCHAR()});
  auto filePath = TempFilePath::create();
  writeToFile(
      filePath->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<std::string>({"1.5", "x"})}));

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .remainingFilter("try_cast(c0 as double) > 1.0")
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_UNSUPPORTED_THROW(
      AssertQueryBuilder(plan)
          .splits(makeCudfHiveConnectorSplits({filePath}))
          .copyResults(pool_.get()),
      "Remaining filter with a cast from VARCHAR is not supported");
}

TEST_F(TableScanTest, unsupportedScanColumnProjectedOut) {
  auto rowType = ROW({"k", "m"}, {BIGINT(), MAP(BIGINT(), BIGINT())});
  auto filePath = writeUnsupportedMapInput();

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
                  .project({"k"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, true)
                    .split(makeCudfHiveConnectorSplit(filePath->getPath()))
                    .copyResults(pool(), task);
  auto expected = makeRowVector({"k"}, {makeFlatVector<int64_t>({1, 2})});
  facebook::velox::test::assertEqualVectors(expected, result);

  const auto operatorStats = toOperatorStats(task->taskStats());
  EXPECT_EQ(operatorStats.count("CudfFilterProject"), 0);
  EXPECT_EQ(operatorStats.count("FilterProject"), 1);
}

TEST_F(TableScanTest, unsupportedScanOutputFallsBack) {
  auto rowType = ROW({"k", "m"}, {BIGINT(), MAP(BIGINT(), BIGINT())});
  auto filePath = writeUnsupportedMapInput();

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

  auto expected = makeRowVector(
      {"k", "m"},
      {makeFlatVector<int64_t>({1, 2}),
       makeMapVector<int64_t, int64_t>({{{10, 100}}, {{20, 200}}})});
  AssertQueryBuilder(plan)
      .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, true)
      .split(makeCudfHiveConnectorSplit(filePath->getPath()))
      .assertResults(expected);
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
