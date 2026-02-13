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
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;
namespace {

class S3ReadTest : public S3Test, public ::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    S3Test::SetUp();
    // Register cudf to enable the CudfDatasource creation from
    // CudfHiveConnector
    facebook::velox::cudf_velox::registerCudf();
    filesystems::registerS3FileSystem();

    // Register Hive connector
    facebook::velox::cudf_velox::connector::hive::CudfHiveConnectorFactory
        factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId, minioServer_->hiveConfig(), ioExecutor_.get());
    facebook::velox::connector::registerConnector(hiveConnector);
  }

  void TearDown() override {
    filesystems::finalizeS3FileSystem();
    facebook::velox::connector::unregisterConnector(kCudfHiveConnectorId);
    S3Test::TearDown();
  }
};
} // namespace

TEST_F(S3ReadTest, s3ReadTest) {
  const auto sourceFile = test::getDataFilePath(
      "velox/experimental/cudf/tests",
      "../../../dwio/parquet/tests/examples/int.parquet");
  const char* bucketName = "data";
  const auto destinationFile = S3Test::localPath(bucketName) + "/int.parquet";
  minioServer_->addBucket(bucketName);
  std::ifstream src(sourceFile, std::ios::binary);
  std::ofstream dest(destinationFile, std::ios::binary);
  // Copy source file to destination bucket.
  dest << src.rdbuf();
  ASSERT_GT(dest.tellp(), 0) << "Unable to copy from source " << sourceFile;
  dest.close();

  // Read the parquet file via the S3 bucket.
  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          false,
          common::SubfieldFilters{},
          nullptr,
          nullptr);
  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  auto split = facebook::velox::connector::hive::HiveConnectorSplitBuilder(
                   filesystems::s3URI(bucketName, "int.parquet"))
                   .connectorId(kCudfHiveConnectorId)
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
