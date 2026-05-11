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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {
namespace {

/// End-to-end tests for writing and reading Iceberg tables using the DWRF file
/// format. Exercises the full write path (IcebergDataSink -> DWRF writer) and
/// the full read path (IcebergSplitReader -> DWRF reader), verifying data
/// round-trip correctness.
class IcebergDwrfInsertTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    dwrf::registerDwrfReaderFactory();
    dwrf::registerDwrfWriterFactory();
    fileFormat_ = dwio::common::FileFormat::DWRF;
  }

  /// Write test data using DWRF format, then read it back and verify results.
  void test(const RowTypePtr& rowType, double nullRatio = 0.0) {
    const auto outputDirectory = TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    constexpr int32_t numBatches = 10;
    constexpr int32_t vectorSize = 5'000;
    const auto vectors =
        createTestData(rowType, numBatches, vectorSize, nullRatio);
    const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
    const auto commitTasks = dataSink->close();

    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_EQ(splits.size(), commitTasks.size());
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
};

TEST_F(IcebergDwrfInsertTest, basic) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           BOOLEAN(),
           REAL(),
           VARCHAR(),
           VARBINARY(),
           DOUBLE()});
  test(rowType, 0.2);
}

TEST_F(IcebergDwrfInsertTest, mapAndArray) {
  auto rowType =
      ROW({"c1", "c2"}, {MAP(INTEGER(), VARCHAR()), ARRAY(VARCHAR())});
  test(rowType);
}

/// Verify the commit message maps DWRF format to "ORC" per Iceberg SDK
/// convention (Iceberg has no DWRF enum; DWRF files use the ORC identifier).
TEST_F(IcebergDwrfInsertTest, commitMessageFormat) {
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  const auto vectors = createTestData(rowType, 2, 100);
  const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
  const auto commitTasks = dataSink->close();

  ASSERT_GT(commitTasks.size(), 0);
  for (const auto& task : commitTasks) {
    auto taskJson = folly::parseJson(task);
    ASSERT_TRUE(taskJson.count("fileFormat") > 0);
    ASSERT_EQ(taskJson["fileFormat"].asString(), "ORC");
  }
}

/// Verify TIMESTAMP values round-trip unchanged through the DWRF write path
/// even when the session is configured with a non-UTC timezone and
/// adjustTimestampToTimezone=true. The Iceberg spec requires timestamps NOT
/// be adjusted to UTC; the DataSink enforces this by overriding the DWRF
/// WriterOptions fields. If that override regresses, timestamps would
/// silently shift by the session-timezone offset and assertResults() would
/// fail.
TEST_F(IcebergDwrfInsertTest, timestampRoundTrip) {
  recreateConnectorQueryCtx(
      /*sessionTimezone=*/"America/Los_Angeles",
      /*adjustTimestampToTimezone=*/true);
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), TIMESTAMP()});
  test(rowType);
}

/// End-to-end test for partitioned DWRF writes. Mirrors the identity-
/// transform partition coverage that exists for Parquet and exercises the
/// commitPartitionValue_ accounting on the DWRF code path.
TEST_F(IcebergDwrfInsertTest, partitioned) {
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  const auto vectors = createTestData(rowType, 2, 50, 0.2);

  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}};
  const auto dataSink =
      createDataSinkAndAppendData(vectors, dataPath, partitionTransforms);
  const auto commitTasks = dataSink->close();
  ASSERT_GT(commitTasks.size(), 0);

  for (const auto& task : commitTasks) {
    auto taskJson = folly::parseJson(task);
    ASSERT_EQ(taskJson["fileFormat"].asString(), "ORC");
    EXPECT_GT(taskJson.count("partitionDataJson"), 0);
  }

  auto splits = createSplitsForDirectory(dataPath);
  ASSERT_EQ(splits.size(), commitTasks.size());
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
}

/// Regression test for the isPartitioned() guard added to ensureWriter().
/// Without the guard, calling ensureWriter() on a non-partitioned table
/// invoked makeCommitPartitionValue(), which dereferences
/// partitionIdGenerator_ — null for unpartitioned tables — causing a crash.
/// Exercises the unpartitioned write path explicitly so any future
/// regression is caught with a named test.
TEST_F(IcebergDwrfInsertTest, ensureWriterNonPartitioned) {
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  const auto vectors = createTestData(rowType, 1, 50);

  // No partitionFields => unpartitioned table, partitionIdGenerator_ stays
  // null inside the sink. appendData triggers ensureWriter().
  const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
  const auto commitTasks = dataSink->close();

  ASSERT_EQ(commitTasks.size(), 1);
  auto taskJson = folly::parseJson(commitTasks[0]);
  // Unpartitioned tables must not emit partitionDataJson.
  EXPECT_EQ(taskJson.count("partitionDataJson"), 0);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
