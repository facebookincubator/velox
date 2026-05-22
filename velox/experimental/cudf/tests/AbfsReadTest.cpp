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
#include "velox/experimental/cudf/connectors/hive/storage_adapters/CudfDataSourceRegistry.h"
#include "velox/experimental/cudf/connectors/hive/storage_adapters/abfs/RegisterCudfAbfsDataSource.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"
#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <folly/ScopeGuard.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

constexpr const char* kCudfHiveConnectorId{"test-cudf-hive"};
constexpr int kAzuritePort{12'345};
constexpr int64_t kIntParquetRows{10};

namespace velox_filesystems = ::facebook::velox::filesystems;

class AbfsReadTest : public ::testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

    velox_filesystems::registerLocalFileSystem();
    velox_filesystems::registerAbfsFileSystem();
    cudf_velox::filesystems::registerCudfAbfsDataSource();
    cudf_velox::registerCudf();
  }

  static void TearDownTestCase() {
    cudf_velox::filesystems::unregisterCudfDataSources();
    cudf_velox::unregisterCudf();
  }

  void SetUp() override {
    ioExecutor_ = std::make_shared<folly::IOThreadPoolExecutor>(3);

    azuriteServer_ =
        std::make_unique<velox_filesystems::AzuriteServer>(kAzuritePort);
    azuriteServer_->start();

    // Force the cudf hive connector to use the BufferedInput data source to
    // read using the upstream AbfsFileSystem instead of KvikIO. Also declare
    // the Azurite account's auth type so registerAzureClientProvider picks it
    // up and registers a SharedKey factory for the Azurite account.
    auto hiveConfig = azuriteServer_->hiveConfig(
        {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput,
          "true"},
         {"fs.azure.account.auth.type.test.dfs.core.windows.net",
          "SharedKey"}});

    velox_filesystems::registerAzureClientProvider(*hiveConfig);

    cudf_velox::connector::hive::CudfHiveConnectorFactory factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId, hiveConfig, ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
  }

  void TearDown() override {
    connector::ConnectorRegistry::global().erase(kCudfHiveConnectorId);
    if (azuriteServer_) {
      azuriteServer_->stop();
      azuriteServer_.reset();
    }
    ioExecutor_.reset();
  }

  std::string uploadSourceFile() {
    const auto sourceFilePath = test::getDataFilePath(
        "velox/experimental/cudf/tests",
        "../../../dwio/parquet/tests/examples/int.parquet");
    azuriteServer_->addFile(sourceFilePath);
    return azuriteServer_->fileURI();
  }

  core::PlanNodePtr tableScanNode() {
    auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
    auto tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kCudfHiveConnectorId, "int_table", common::SubfieldFilters{}, nullptr);
    return PlanBuilder(pool())
        .startTableScan()
        .tableHandle(tableHandle)
        .outputType(rowType)
        .endTableScan()
        .planNode();
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& abfsPath) const {
    return connector::hive::HiveConnectorSplitBuilder(abfsPath)
        .connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .build();
  }

  RowVectorPtr expectedVector() {
    return makeRowVector(
        {makeFlatVector<int32_t>(
             kIntParquetRows, [](auto row) { return row + 100; }),
         makeFlatVector<int64_t>(
             kIntParquetRows, [](auto row) { return row + 1000; })});
  }

  std::unique_ptr<velox_filesystems::AzuriteServer> azuriteServer_;
  std::shared_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

} // namespace

// Reads a Parquet file from an ABFS path
TEST_F(AbfsReadTest, readWithBufferedInput) {
  const auto abfsFilePath = uploadSourceFile();
  auto plan = tableScanNode();
  auto split = makeSplit(abfsFilePath);

  auto actual = AssertQueryBuilder(plan).split(split).copyResults(pool());

  assertEqualResults({expectedVector()}, {actual});
}

// Reads a Parquet file from an ABFS path via the zero-copy
// CudfAbfsDataSource, selected by the session property.
TEST_F(AbfsReadTest, readWithCudfAbfsDataSource) {
  const auto abfsFilePath = uploadSourceFile();
  auto plan = tableScanNode();
  auto split = makeSplit(abfsFilePath);

  auto actual = AssertQueryBuilder(plan)
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kUseBufferedInputSession,
                        "false")
                    .split(split)
                    .copyResults(pool());
  assertEqualResults({expectedVector()}, {actual});
}

// An invalid ABFS split throws instead of falling back to the KvikIO data
// source
TEST_F(AbfsReadTest, nonExistentBlobBufferedInput) {
  const auto missingPath = azuriteServer_->URI() + "does_not_exist.parquet";
  auto plan = tableScanNode();
  auto split = makeSplit(missingPath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "encountered azure storage exception");
}

// Non existent blob with CudfAbfsDataSource
TEST_F(AbfsReadTest, nonExistentBlobCudfAbfsDatasource) {
  const auto missingPath = azuriteServer_->URI() + "does_not_exist.parquet";
  auto plan = tableScanNode();
  auto split = makeSplit(missingPath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              cudf_velox::connector::hive::CudfHiveConfig::
                  kUseBufferedInputSession,
              "false")
          .split(split)
          .copyResults(pool()),
      "encountered azure storage exception");
}

// Buffered input fails to open non existent blob and the fallback
// is also not available
TEST_F(AbfsReadTest, nonExistentBlobNoFallback) {
  cudf_velox::filesystems::unregisterCudfDataSources();
  SCOPE_EXIT {
    cudf_velox::filesystems::registerCudfAbfsDataSource();
  };

  const auto missingPath = azuriteServer_->URI() + "does_not_exist.parquet";
  auto plan = tableScanNode();
  auto split = makeSplit(missingPath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Failed to generate file handle cache for ABFS path");
}

// No available datasource for ABFS path
TEST_F(AbfsReadTest, noAvailableDataSource) {
  cudf_velox::filesystems::unregisterCudfDataSources();
  SCOPE_EXIT {
    cudf_velox::filesystems::registerCudfAbfsDataSource();
  };

  const auto abfsFilePath = uploadSourceFile();
  auto plan = tableScanNode();
  auto split = makeSplit(abfsFilePath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              cudf_velox::connector::hive::CudfHiveConfig::
                  kUseBufferedInputSession,
              "false")
          .split(split)
          .copyResults(pool()),
      "ABFS path requires a registered native cuDF datasource");
}

// Multiple independent queries reading the same ABFS blob in parallel must
// all succeed and produce identical results
TEST_F(AbfsReadTest, concurrentSplits) {
  const auto abfsFilePath = uploadSourceFile();
  auto plan = tableScanNode();
  const auto expected = expectedVector();

  constexpr int kNumThreads{8};
  std::atomic<bool> startThreads{false};
  std::vector<std::thread> threads;
  std::vector<RowVectorPtr> results(kNumThreads);
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i] {
      while (!startThreads.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto split = makeSplit(abfsFilePath);
      results[i] = AssertQueryBuilder(plan)
                       .split(split)
                       .connectorSessionProperty(
                           kCudfHiveConnectorId,
                           cudf_velox::connector::hive::CudfHiveConfig::
                               kUseBufferedInputSession,
                           i % 2 == 0 ? "true" : "false")
                       .copyResults(pool());
    });
  }
  startThreads.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < kNumThreads; ++i) {
    ASSERT_NE(results[i], nullptr) << "thread " << i << " produced no result";
    assertEqualResults({expected}, {results[i]});
  }
}
