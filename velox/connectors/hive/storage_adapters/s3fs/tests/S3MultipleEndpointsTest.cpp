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

#include "gtest/gtest.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/exec/TableWriter.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::test;
using namespace facebook::velox::filesystems;

static const std::string kConnectorId1 = "test-hive1";
static const std::string kConnectorId2 = "test-hive2";

class S3MultipleEndpoints : public S3Test {
 public:
  void SetUp() override {
    S3Test::SetUp();
    filesystems::registerS3FileSystem();
    auto hiveConnector1 =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kConnectorId1, minioServer_->hiveConfig(), ioExecutor_.get());
    connector::registerConnector(hiveConnector1);
    auto port = facebook::velox::exec::test::getFreePort();
    minioSecondServer_ =
        std::make_unique<MinioServer>(fmt::format("127.0.0.1:{}", port));
    minioSecondServer_->start();
    auto hiveConnector2 =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kConnectorId2,
                minioSecondServer_->hiveConfig(),
                ioExecutor_.get());
    connector::registerConnector(hiveConnector2);
  }

  void TearDown() override {
    connector::unregisterConnector(kConnectorId1);
    connector::unregisterConnector(kConnectorId2);
    S3Test::TearDown();
    filesystems::finalizeS3FileSystem();
  }

  folly::dynamic writeData(
      const RowVectorPtr input,
      const std::string& outputDirectory,
      const std::string& connectorId) {
    auto plan = PlanBuilder()
                    .values({input})
                    .tableWrite(
                        outputDirectory.data(),
                        {},
                        0,
                        {},
                        {},
                        dwio::common::FileFormat::PARQUET,
                        {},
                        connectorId)
                    .planNode();
    // Execute the write plan.
    auto results = AssertQueryBuilder(plan).copyResults(pool());
    // Second column contains details about written files.
    auto details = results->childAt(TableWriteTraits::kFragmentChannel)
                       ->as<FlatVector<StringView>>();
    folly::dynamic obj = folly::parseJson(details->valueAt(1));
    return obj["fileWriteInfos"];
  }

  std::shared_ptr<HiveConnectorSplit> createSplit(
      folly::dynamic tableWriteInfo,
      std::string outputDirectory,
      std::string connectorId) {
    auto writeFileName = tableWriteInfo[0]["writeFileName"].asString();
    auto filePath = fmt::format("{}{}", outputDirectory, writeFileName);
    const int64_t fileSize = tableWriteInfo[0]["fileSize"].asInt();

    return HiveConnectorSplitBuilder(filePath)
        .connectorId(connectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .length(fileSize)
        .build();
  }

  std::unique_ptr<MinioServer> minioSecondServer_;
};

TEST_F(S3MultipleEndpoints, s3Join) {
  const int64_t kExpectedRows = 1'000;
  const std::string_view kOutputDirectory{"s3://writedata/"};

  auto rowType1 = ROW(
      {"a0", "a1", "a2", "a3"}, {BIGINT(), INTEGER(), SMALLINT(), DOUBLE()});
  auto rowType2 = ROW(
      {"b0", "b1", "b2", "b3"}, {BIGINT(), INTEGER(), SMALLINT(), DOUBLE()});

  auto input1 = makeRowVector(
      rowType1->names(),
      {makeFlatVector<int64_t>(kExpectedRows, [](auto row) { return row; }),
       makeFlatVector<int32_t>(kExpectedRows, [](auto row) { return row; }),
       makeFlatVector<int16_t>(kExpectedRows, [](auto row) { return row; }),
       makeFlatVector<double>(kExpectedRows, [](auto row) { return row; })});
  auto input2 = makeRowVector(rowType2->names(), input1->children());
  minioServer_->addBucket("writedata");
  minioSecondServer_->addBucket("writedata");

  // Insert input data into both tables.
  auto table1WriteInfo =
      writeData(input1, kOutputDirectory.data(), kConnectorId1);
  auto table2WriteInfo =
      writeData(input2, kOutputDirectory.data(), kConnectorId2);

  // Inner Join both the tables.
  PlanNodeId scan1, scan2;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto table1Scan = PlanBuilder(planNodeIdGenerator, pool())
                        .startTableScan()
                        .tableName("hive_table1")
                        .outputType(rowType1)
                        .connectorId(kConnectorId1)
                        .endTableScan()
                        .capturePlanNodeId(scan1)
                        .planNode();
  auto join =
      PlanBuilder(planNodeIdGenerator, pool())
          .startTableScan()
          .tableName("hive_table1")
          .outputType(rowType2)
          .connectorId(kConnectorId2)
          .endTableScan()
          .capturePlanNodeId(scan2)
          .hashJoin({"b0"}, {"a0"}, table1Scan, "", {"a0", "a1", "a2", "a3"})
          .planNode();

  auto split1 =
      createSplit(table1WriteInfo, kOutputDirectory.data(), kConnectorId1);
  auto split2 =
      createSplit(table2WriteInfo, kOutputDirectory.data(), kConnectorId2);
  auto results = AssertQueryBuilder(join)
                     .split(scan1, split1)
                     .split(scan2, split2)
                     .copyResults(pool());
  assertEqualResults({input1}, {results});
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
