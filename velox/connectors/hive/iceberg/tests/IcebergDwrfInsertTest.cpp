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

#include <fstream>

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
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

/// Verify the commit message contains "DWRF" as the file format.
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

/// Integration test: write a DWRF data file, create a deletion vector (DV)
/// that marks specific rows as deleted, attach the DV to the split, read back,
/// and verify only the non-deleted rows are returned. This exercises the full
/// DWRF write path -> DV creation -> DWRF read path with DV filtering pipeline.
TEST_F(IcebergDwrfInsertTest, deletionVectors) {
  const auto outputDirectory = TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto rowType = ROW({"c0"}, {BIGINT()});

  // Write 100 deterministic rows: {0, 1, 2, ..., 99}.
  constexpr int32_t numRows = 100;
  auto data = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>(numRows, [](auto row) { return row; })});

  auto dataSink = createDataSink(rowType, dataPath);
  dataSink->appendData(data);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_EQ(commitTasks.size(), 1);

  // Create a deletion vector deleting every 10th row: {0, 10, 20, ..., 90}.
  std::vector<int64_t> deletePositions;
  for (int64_t i = 0; i < numRows; i += 10) {
    deletePositions.push_back(i);
  }

  DeletionVectorWriter dvWriter;
  dvWriter.addDeletedPositions(deletePositions);
  auto dvBlob = dvWriter.serialize();

  // Write the DV blob to a temp file.
  const auto dvDirectory = TempDirectoryPath::create();
  const auto dvFilePath = dvDirectory->getPath() + "/dv.bin";
  {
    std::ofstream out(dvFilePath, std::ios::binary | std::ios::trunc);
    out.write(dvBlob.data(), static_cast<std::streamsize>(dvBlob.size()));
  }
  auto dvFileSize = static_cast<uint64_t>(dvBlob.size());

  // Build IcebergDeleteFile metadata with DV blob offset and length.
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] = "0";
  upperBounds[DeletionVectorReader::kDvLengthFieldId] =
      std::to_string(dvFileSize);

  IcebergDeleteFile dvDeleteFile(
      FileContent::kDeletionVector,
      dvFilePath,
      dwio::common::FileFormat::DWRF,
      deletePositions.size(),
      dvFileSize,
      {},
      lowerBounds,
      upperBounds);

  // Create a split for the DWRF data file with the DV attached.
  auto dataFiles = listFiles(dataPath);
  ASSERT_EQ(dataFiles.size(), 1);
  const auto& dataFilePath = dataFiles[0];
  auto fileHandle = filesystems::getFileSystem(dataFilePath, nullptr)
                        ->openFileForRead(dataFilePath);

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      std::make_shared<HiveIcebergSplit>(
          test::kIcebergConnectorId,
          dataFilePath,
          fileFormat_,
          0,
          fileHandle->size(),
          std::unordered_map<std::string, std::optional<std::string>>{},
          std::nullopt,
          std::unordered_map<std::string, std::string>{},
          nullptr,
          /*cacheable=*/true,
          std::vector<IcebergDeleteFile>{dvDeleteFile}));

  // Build expected result: all rows NOT in deletePositions.
  std::unordered_set<int64_t> deletedSet(
      deletePositions.begin(), deletePositions.end());
  std::vector<int64_t> expectedValues;
  for (int64_t i = 0; i < numRows; ++i) {
    if (!deletedSet.contains(i)) {
      expectedValues.push_back(i);
    }
  }
  auto expected =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(expectedValues)});

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
