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
                    .startTableScan(test::kIcebergConnectorId)
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

/// Regression test for the DWRF/ORC recordCount=0 manifest bug. Before the
/// fix, IcebergDataSink::closeWriterAndCollectStats emplaced
/// IcebergDataFileStatistics::empty() (numRecords = 0) on every non-Parquet
/// commit, which makes the iceberg manifest report DataFile.recordCount() == 0
/// and silently breaks DELETE/UPDATE/MERGE planning (planner treats files as
/// empty and skips them).
///
/// Writes a single batch of N rows to an unpartitioned table and asserts the
/// resulting commit message reports metrics.recordCount = N. Without the fix
/// the assertion fails with 0.
TEST_F(IcebergDwrfInsertTest, recordCountUnpartitioned) {
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  constexpr int32_t kNumRows = 100;
  const auto vectors = createTestData(rowType, /*numBatches=*/1, kNumRows);
  const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
  const auto commitTasks = dataSink->close();

  ASSERT_EQ(commitTasks.size(), 1);
  auto taskJson = folly::parseJson(commitTasks[0]);
  ASSERT_GT(taskJson.count("metrics"), 0);
  ASSERT_GT(taskJson["metrics"].count("recordCount"), 0);
  EXPECT_EQ(taskJson["metrics"]["recordCount"].asInt(), kNumRows);
}

/// Multi-batch single-writer accumulation. closeWriterAndCollectStats reads
/// writerInfo_->numWrittenRows, which the FileDataSink accumulates across
/// every appendData call to the same writer. The per-file delta should equal
/// the SUM of all batches' rows when only one file is produced.
TEST_F(IcebergDwrfInsertTest, recordCountMultiBatchAccumulated) {
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  constexpr int32_t kNumBatches = 4;
  constexpr int32_t kRowsPerBatch = 25;
  const auto vectors = createTestData(rowType, kNumBatches, kRowsPerBatch);
  const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
  const auto commitTasks = dataSink->close();

  ASSERT_EQ(commitTasks.size(), 1);
  auto taskJson = folly::parseJson(commitTasks[0]);
  EXPECT_EQ(
      taskJson["metrics"]["recordCount"].asInt(), kNumBatches * kRowsPerBatch);
}

/// Multi-partition per-writer delta. Two partitions => two distinct writer
/// indices => two commit messages, each with its own DataFileStatistics
/// derived from the reportedRowsPerWriter_[index] delta. If the index
/// bookkeeping is wrong (e.g., shared counter across writers), totals would
/// double-count or under-count.
///
/// Asserts: (a) exactly 2 commits, (b) every commit reports a positive
/// recordCount, (c) the sum across commits equals the total rows written.
TEST_F(IcebergDwrfInsertTest, recordCountPartitionedPerWriter) {
  auto rowType = ROW({"c1", "c2"}, {BIGINT(), VARCHAR()});
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  constexpr int32_t kNumBatches = 2;
  constexpr int32_t kRowsPerBatch = 60;
  const auto vectors = createTestData(rowType, kNumBatches, kRowsPerBatch, 0.0);

  // Partition by c2 (VARCHAR). createTestData populates strings from a small
  // randomized pool, so distinct partition values are produced reliably for
  // 60 rows; assertions below tolerate any partition count >= 2.
  std::vector<test::PartitionField> partitionTransforms = {
      {1, TransformType::kIdentity, std::nullopt}};
  const auto dataSink =
      createDataSinkAndAppendData(vectors, dataPath, partitionTransforms);
  const auto commitTasks = dataSink->close();

  ASSERT_GE(commitTasks.size(), 2)
      << "Expected at least 2 partition files; got " << commitTasks.size();

  int64_t totalRecords = 0;
  for (const auto& task : commitTasks) {
    auto taskJson = folly::parseJson(task);
    const int64_t recordCount = taskJson["metrics"]["recordCount"].asInt();
    EXPECT_GT(recordCount, 0)
        << "Per-partition recordCount must be positive; task: " << task;
    totalRecords += recordCount;
  }
  EXPECT_EQ(totalRecords, kNumBatches * kRowsPerBatch);
}

/// Round-trips TIMESTAMP values through the DWRF write path with the session
/// configured for non-UTC timezone and adjustTimestampToTimezone=true. The
/// Iceberg spec requires timestamps NOT be adjusted to UTC; the DataSink
/// enforces this by overriding the DWRF WriterOptions fields via
/// DwrfWriterOptionsAdapter::applyPostConfigs.
///
/// TODO: This test is a symmetric Velox-only round-trip and cannot, by
/// itself, detect a regression where the DataSink stops overriding the DWRF
/// timezone fields — any write-side shift is exactly cancelled by the
/// matching read-side shift. The adapter's override contract is locked down
/// at the unit level by
/// WriterOptionsAdapterTest::dwrfPostConfigsOverridesTimestampFields. A
/// true cross-engine validation (e.g., reading Velox-written Iceberg files
/// with a Java Spark reader) is needed to verify the on-disk timestamp
/// matches the spec.
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
                  .startTableScan(test::kIcebergConnectorId)
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
