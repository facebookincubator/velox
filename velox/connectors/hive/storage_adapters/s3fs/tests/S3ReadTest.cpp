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

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::filesystems {
namespace {

class S3ReadTest : public S3Test, public ::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    S3Test::SetUp();
    filesystems::registerS3FileSystem();
    connector::hive::HiveConnectorFactory factory;
    auto hiveConnector =
        factory.newConnector(kHiveConnectorId, siloServer_->s3Config());
    connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
    parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    parquet::unregisterParquetReaderFactory();
    filesystems::finalizeS3FileSystem();
    connector::ConnectorRegistry::global().erase(kHiveConnectorId);
    S3Test::TearDown();
  }
};
} // namespace

TEST_F(S3ReadTest, s3ReadTest) {
  const auto sourceFile = test::getDataFilePath(
      "velox/connectors/hive/storage_adapters/s3fs/tests",
      "../../../../../dwio/parquet/tests/examples/int.parquet");
  const char* bucketName = "data";
  addBucket(bucketName);
  const auto s3File = s3URI(bucketName, "int.parquet");
  std::ifstream src(sourceFile, std::ios::binary);
  const std::string content(
      (std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
  ASSERT_GT(content.size(), 0) << "Unable to read source " << sourceFile;
  src.close();

  // Upload the source file to the S3 bucket via the S3 API; the server
  // only serves objects it stores.
  auto s3Config = siloServer_->s3Config();
  {
    // Upload via the S3 API; getFileSystem initializes S3 on first creation.
    auto s3fs = filesystems::getFileSystem(s3File, s3Config);
    auto writeFile = s3fs->openFileForWrite(s3File, {{}, pool(), std::nullopt});
    writeFile->append(content);
    writeFile->close();
  }

  // Read the parquet file via the S3 bucket.
  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto plan = PlanBuilder().tableScan(rowType).planNode();
  auto split = HiveConnectorSplitBuilder(s3URI(bucketName, "int.parquet"))
                   .fileFormat(dwio::common::FileFormat::PARQUET)
                   .build();
  auto copy = AssertQueryBuilder(plan).split(split).copyResults(pool());

  // expectedResults is the data in int.parquet file.
  const int64_t kExpectedRows = 10;
  auto expectedResults = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});
  assertEqualResults({expectedResults}, {copy});
}
} // namespace facebook::velox::filesystems

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
