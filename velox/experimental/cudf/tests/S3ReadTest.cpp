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
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/StatsReporter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/base/tests/StatsReporterUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;
namespace {

constexpr const char* kBucketName = "data";
constexpr const char* kIntParquet = "int.parquet";

// Counts the S3 requests the scans make.
folly::Singleton<BaseStatsReporter> reporter([]() {
  return new facebook::velox::test::TestReporter();
});

// Resolves int.parquet, falling back to a working-directory lookup when the
// build-time path baked into the binary is unreachable (velox-cudf CI builds
// and runs the binaries on different hosts).
std::string resolveIntParquetPath() {
  const std::string relativePath =
      "../../../dwio/parquet/tests/examples/int.parquet";
  auto path =
      test::getDataFilePath("velox/experimental/cudf/tests", relativePath);
  if (std::filesystem::exists(path)) {
    return path;
  }
  return (std::filesystem::current_path() / relativePath)
      .lexically_normal()
      .string();
}

class S3ReadTest : public S3Test, public ::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    // The stats macros record only once a reporter is registered.
    BaseStatsReporter::registered = true;
    // The S3 SDK initializes once per process, so the file system is
    // registered and finalized once for all tests.
    filesystems::registerS3FileSystem();
    filesystems::registerS3Metrics();
  }

  static void TearDownTestCase() {
    filesystems::finalizeS3FileSystem();
  }

  void SetUp() override {
    S3Test::SetUp();
    // Register cudf to enable the CudfDatasource creation from
    // CudfHiveConnector
    facebook::velox::cudf_velox::CudfConfig::getInstance().allowCpuFallback =
        false;
    facebook::velox::cudf_velox::registerCudf();
    reporter_ = std::dynamic_pointer_cast<facebook::velox::test::TestReporter>(
        folly::Singleton<BaseStatsReporter>::try_get());

    // Register Hive connector. Without the file handle cache every open
    // reaches S3, so the tests can count the requests.
    facebook::velox::cudf_velox::connector::hive::CudfHiveConnectorFactory
        factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId,
        siloServer_->s3Config({
            {facebook::velox::connector::hive::HiveConfig::
                 kEnableFileHandleCache,
             "false"},
        }),
        ioExecutor_.get());
    facebook::velox::connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
  }

  void TearDown() override {
    if (kvikioPointedAtTestServer_) {
      for (const auto* name :
           {"AWS_ENDPOINT_URL",
            "AWS_DEFAULT_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY"}) {
        unsetenv(name);
      }
    }
    facebook::velox::connector::ConnectorRegistry::global().erase(
        kCudfHiveConnectorId);
    facebook::velox::cudf_velox::unregisterCudf();
    S3Test::TearDown();
  }

  // Uploads int.parquet to the test bucket and returns its size.
  int64_t uploadIntParquet() {
    const auto sourceFile = resolveIntParquetPath();
    siloServer_->addBucket(kBucketName);
    std::ifstream sourceStream(sourceFile, std::ios::binary);
    const std::string content(
        (std::istreambuf_iterator<char>(sourceStream)),
        std::istreambuf_iterator<char>());
    EXPECT_GT(content.size(), 0) << "Unable to read source " << sourceFile;

    // Upload via the S3 API; getFileSystem initializes S3 on first creation.
    // The server only serves objects it stores.
    const auto path = filesystems::s3URI(kBucketName, kIntParquet);
    auto s3fs = facebook::velox::filesystems::getFileSystem(
        path, siloServer_->s3Config());
    auto writeFile = s3fs->openFileForWrite(path, {{}, pool(), std::nullopt});
    writeFile->append(content);
    writeFile->close();
    return static_cast<int64_t>(content.size());
  }

  // Plans a scan of int.parquet's two columns.
  core::PlanNodePtr intParquetScanPlan() {
    auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
    auto tableHandle =
        std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
            kCudfHiveConnectorId,
            "int_table",
            common::SubfieldFilters{},
            nullptr);
    return PlanBuilder(pool())
        .startTableScan()
        .tableHandle(tableHandle)
        .outputType(rowType)
        .endTableScan()
        .planNode();
  }

  // Scans int.parquet through 'split' and checks the rows.
  void scanIntParquet(
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>& split,
      bool useBufferedInput) {
    using facebook::velox::cudf_velox::connector::hive::CudfHiveConfig;
    auto copy = AssertQueryBuilder(intParquetScanPlan())
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        CudfHiveConfig::kUseBufferedInputSession,
                        useBufferedInput ? "true" : "false")
                    .split(split)
                    .copyResults(pool());

    // expectedResults is the data in int.parquet file.
    const int64_t kExpectedRows = 10;
    auto expectedResults = makeRowVector(
        {makeFlatVector<int32_t>(
             kExpectedRows, [](auto row) { return row + 100; }),
         makeFlatVector<int64_t>(
             kExpectedRows, [](auto row) { return row + 1000; })});
    assertEqualResults({expectedResults}, {copy});
  }

  // Where the split carries the file size, if anywhere. Presto sends it as
  // the $file_size info column.
  enum class FileSizeSource { kNone, kInfoColumn, kProperties };

  std::shared_ptr<facebook::velox::connector::ConnectorSplit>
  makeIntParquetSplit(FileSizeSource source, int64_t fileSize) {
    facebook::velox::connector::hive::HiveConnectorSplitBuilder builder(
        filesystems::s3URI(kBucketName, kIntParquet));
    builder.connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET);
    switch (source) {
      case FileSizeSource::kNone:
        break;
      case FileSizeSource::kInfoColumn:
        builder.infoColumn("$file_size", std::to_string(fileSize));
        break;
      case FileSizeSource::kProperties:
        builder.fileProperties(FileProperties{.fileSize = fileSize});
        break;
    }
    return builder.build();
  }

  // KvikIO reads S3 through libcurl and finds the server and the credentials
  // in the AWS environment variables.
  void pointKvikioAtTestServer() {
    const auto endpoint =
        siloServer_->s3Config()->get<std::string>("s3.endpoint").value();
    setenv("AWS_ENDPOINT_URL", ("http://" + endpoint).c_str(), 1);
    setenv("AWS_DEFAULT_REGION", "us-east-1", 1);
    setenv("AWS_ACCESS_KEY_ID", kSiloAccessKey, 1);
    setenv("AWS_SECRET_ACCESS_KEY", kSiloSecretKey, 1);
    kvikioPointedAtTestServer_ = true;
  }

  size_t s3MetadataCalls() {
    return reporter_->getCounterValue(
        std::string{facebook::velox::filesystems::kMetricS3MetadataCalls});
  }

  std::shared_ptr<facebook::velox::test::TestReporter> reporter_;
  bool kvikioPointedAtTestServer_{false};
};
} // namespace

TEST_F(S3ReadTest, s3ReadTest) {
  const auto fileSize = uploadIntParquet();
  scanIntParquet(
      makeIntParquetSplit(FileSizeSource::kNone, fileSize),
      /*useBufferedInput=*/true);
}

// With the size in the split the open sends no request, so a missing file
// shows at the first read; ignore_missing_files still skips the split.
TEST_F(S3ReadTest, missingFileWithSizeIsIgnored) {
  using facebook::velox::connector::hive::HiveConfig;
  using facebook::velox::cudf_velox::connector::hive::CudfHiveConfig;
  siloServer_->addBucket(kBucketName);
  auto missingSplit = [&]() {
    facebook::velox::connector::hive::HiveConnectorSplitBuilder builder(
        filesystems::s3URI(kBucketName, "missing.parquet"));
    return builder.connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .infoColumn("$file_size", "1234")
        .build();
  };
  auto plan = intParquetScanPlan();

  // KvikIO reports a missing remote file as a generic error that the reader
  // does not ignore yet, so the buffered input alone is checked.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              CudfHiveConfig::kUseBufferedInputSession,
              "true")
          .split(missingSplit())
          .copyResults(pool()),
      "missing.parquet");
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kCudfHiveConnectorId,
          CudfHiveConfig::kUseBufferedInputSession,
          "true")
      .connectorSessionProperty(
          kCudfHiveConnectorId, HiveConfig::kIgnoreMissingFilesSession, "true")
      .split(missingSplit())
      .assertEmptyResults();
}

// A split that carries the file size lets the connector open the file without
// asking S3 for the size.
TEST_F(S3ReadTest, fileSizeInSplitSkipsHeadRequests) {
  const auto fileSize = uploadIntParquet();
  pointKvikioAtTestServer();

  for (const bool useBufferedInput : {true, false}) {
    SCOPED_TRACE(fmt::format("use_buffered_input={}", useBufferedInput));

    // Without the size, the buffered input opens the file for the data and the
    // data source opens it again to account the bytes. KvikIO opens its own
    // data source outside Velox, so only the accounting open shows here.
    reporter_->clear();
    scanIntParquet(
        makeIntParquetSplit(FileSizeSource::kNone, fileSize), useBufferedInput);
    EXPECT_EQ(s3MetadataCalls(), useBufferedInput ? 2 : 1);

    for (const auto source :
         {FileSizeSource::kInfoColumn, FileSizeSource::kProperties}) {
      SCOPED_TRACE(
          fmt::format(
              "file size from {}",
              source == FileSizeSource::kInfoColumn ? "$file_size"
                                                    : "properties"));
      reporter_->clear();
      scanIntParquet(makeIntParquetSplit(source, fileSize), useBufferedInput);
      EXPECT_EQ(s3MetadataCalls(), 0);
    }
  }
}
