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
#include <atomic>
#include <shared_mutex>

#include <fmt/ranges.h>
#include <folly/experimental/EventCount.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"

#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetReaderConfig.h"
#include "velox/experimental/cudf/connectors/parquet/tests/ParquetConnectorTestBase.h"

#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::common::test;
using namespace facebook::velox::tests::utils;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

class TableScanTest : public virtual ParquetConnectorTestBase {
 protected:
  void SetUp() override {
    ParquetConnectorTestBase::SetUp();
    facebook::velox::exec::ExchangeSource::factories().clear();
    facebook::velox::exec::ExchangeSource::registerFactory(
        facebook::velox::exec::test::createLocalExchangeSource);
  }

  static void SetUpTestCase() {
    ParquetConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return ParquetConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  facebook::velox::exec::Split makeParquetSplit(
      std::string path,
      int64_t splitWeight = 0) {
    return facebook::velox::exec::Split(
        makeParquetConnectorSplit(std::move(path), splitWeight));
  }

  std::shared_ptr<facebook::velox::exec::Task> assertQuery(
      const PlanNodePtr& plan,
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>&
          parquetSplit,
      const std::string& duckDbSql) {
    return facebook::velox::exec::test::OperatorTestBase::assertQuery(
        plan, {parquetSplit}, duckDbSql);
  }

  std::shared_ptr<facebook::velox::exec::Task> assertQuery(
      const PlanNodePtr& plan,
      const facebook::velox::exec::Split&& split,
      const std::string& duckDbSql) {
    return facebook::velox::exec::test::OperatorTestBase::assertQuery(
        plan, {split}, duckDbSql);
  }

  std::shared_ptr<facebook::velox::exec::Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<
          std::shared_ptr<facebook::velox::exec::test::TempFilePath>>&
          filePaths,
      const std::string& duckDbSql) {
    return ParquetConnectorTestBase::assertQuery(plan, filePaths, duckDbSql);
  }

  // Run query with spill enabled.
  std::shared_ptr<facebook::velox::exec::Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<
          std::shared_ptr<facebook::velox::exec::test::TempFilePath>>&
          filePaths,
      const std::string& spillDirectory,
      const std::string& duckDbSql) {
    return facebook::velox::exec::test::AssertQueryBuilder(
               plan, duckDbQueryRunner_)
        .spillDirectory(spillDirectory)
        .config(core::QueryConfig::kSpillEnabled, false)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .splits(makeParquetConnectorSplits(filePaths))
        .assertResults(duckDbSql);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    return facebook::velox::exec::test::PlanBuilder(pool_.get())
        .tableScan(outputType)
        .planNode();
  }

  static facebook::velox::exec::PlanNodeStats getTableScanStats(
      const std::shared_ptr<facebook::velox::exec::Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  static std::unordered_map<std::string, RuntimeMetric>
  getTableScanRuntimeStats(
      const std::shared_ptr<facebook::velox::exec::Task>& task) {
    return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  static int64_t getSkippedStridesStat(
      const std::shared_ptr<facebook::velox::exec::Task>& task) {
    return getTableScanRuntimeStats(task)["skippedStrides"].sum;
  }

  static int64_t getSkippedSplitsStat(
      const std::shared_ptr<facebook::velox::exec::Task>& task) {
    return getTableScanRuntimeStats(task)["skippedSplits"].sum;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<facebook::velox::exec::Task>& task,
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
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           REAL(),
           DOUBLE(),
           VARCHAR(),
           TINYINT()})};
};

TEST_F(TableScanTest, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = facebook::velox::exec::test::TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto task = assertQuery(plan, {filePath}, "SELECT * FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
  // Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());
}

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = facebook::velox::exec::test::TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector});

  auto plan = facebook::velox::exec::test::PlanBuilder(pool_.get())
                  .startTableScan()
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

  auto task =
      facebook::velox::exec::test::AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .splits(makeParquetConnectorSplits({filePath}))
          .queryCtx(queryCtx)
          .assertResults("SELECT c0, c2 FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = facebook::velox::exec::toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  auto overreadBytes = getTableScanRuntimeStats(task).at("overreadBytes").sum;
  ASSERT_GE(rawInputBytes, 500);
  ASSERT_EQ(overreadBytes, 13);
  ASSERT_EQ(
      getTableScanRuntimeStats(task).at("storageReadBytes").sum,
      rawInputBytes + overreadBytes);
  ASSERT_GT(getTableScanRuntimeStats(task)["totalScanTime"].sum, 0);
  ASSERT_GT(getTableScanRuntimeStats(task)["ioWaitWallNanos"].sum, 0);
}
