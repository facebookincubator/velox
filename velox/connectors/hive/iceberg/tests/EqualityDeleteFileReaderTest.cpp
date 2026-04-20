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

#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox::exec::test;

namespace {

const std::string kIcebergConnectorId = "test-iceberg-eq-delete";

} // namespace

/// End-to-end tests for equality deletes via the IcebergSplitReader.
/// These tests write DWRF data files and delete files, then execute
/// table scans verifying that matching rows are filtered out.
class EqualityDeleteFileReaderTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    IcebergConnectorFactory icebergFactory;
    auto icebergConnector = icebergFactory.newConnector(
        kIcebergConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()),
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        icebergConnector->connectorId(), icebergConnector);
  }

  void TearDown() override {
    connector::ConnectorRegistry::global().erase(kIcebergConnectorId);
    HiveConnectorTestBase::TearDown();
  }

  uint64_t getFileSize(const std::string& path) {
    return filesystems::getFileSystem(path, nullptr)
        ->openFileForRead(path)
        ->size();
  }

  /// Writes a DWRF data file containing the given vectors.
  std::shared_ptr<common::testutil::TempFilePath> writeDataFile(
      const std::vector<RowVectorPtr>& data) {
    auto file = common::testutil::TempFilePath::create();
    writeToFile(file->getPath(), data);
    return file;
  }

  /// Writes a DWRF delete file containing the equality delete rows.
  std::shared_ptr<common::testutil::TempFilePath> writeEqDeleteFile(
      const std::vector<RowVectorPtr>& deleteData) {
    auto file = common::testutil::TempFilePath::create();
    writeToFile(file->getPath(), deleteData);
    return file;
  }

  /// Creates splits with equality delete files attached.
  std::vector<std::shared_ptr<ConnectorSplit>> makeSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      int64_t dataSequenceNumber = 0) {
    auto fileSize = getFileSize(dataFilePath);
    return {std::make_shared<HiveIcebergSplit>(
        kIcebergConnectorId,
        dataFilePath,
        dwio::common::FileFormat::DWRF,
        0,
        fileSize,
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        /*cacheable=*/true,
        deleteFiles,
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        dataSequenceNumber)};
  }

  /// Builds a table scan plan node with the given schema.
  core::PlanNodePtr makeTableScanPlan(const RowTypePtr& rowType) {
    return PlanBuilder()
        .startTableScan()
        .connectorId(kIcebergConnectorId)
        .outputType(rowType)
        .dataColumns(rowType)
        .endTableScan()
        .planNode();
  }
};

/// Verifies that base rows matching the equality delete file are removed.
TEST_F(EqualityDeleteFileReaderTest, basicSingleColumnDelete) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete rows where id == 3 or id == 7.
  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({3, 7}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1}); // field ID 1 = column 0 = "id"

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 4, 5, 6, 8, 9}),
          makeFlatVector<std::string>({"a", "b", "c", "e", "f", "g", "i", "j"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies multi-column equality deletes (both columns must match).
TEST_F(EqualityDeleteFileReaderTest, multiColumnDelete) {
  auto rowType = ROW({"a", "b", "c"}, {INTEGER(), VARCHAR(), BIGINT()});

  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"x", "y", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete rows where (a=2, b="y") — matches row index 1.
  // Also (a=5, b="y") — matches row index 4.
  // But (a=1, b="y") — no match (a=1 has b="x").
  auto deleteData = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int32_t>({2, 5, 1}),
          makeFlatVector<std::string>({"y", "y", "y"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2}); // field IDs 1,2 = columns "a","b"

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Rows 0, 2, 3 survive (rows 1 and 4 deleted).
  auto expected = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({1, 3, 4}),
          makeFlatVector<std::string>({"x", "z", "x"}),
          makeFlatVector<int64_t>({10, 30, 40}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that when no rows match, all rows survive.
TEST_F(EqualityDeleteFileReaderTest, noMatchingDeletes) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete file has values not present in base data.
  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({100, 200}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that all rows are deleted when every base row matches.
TEST_F(EqualityDeleteFileReaderTest, allRowsDeleted) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  EXPECT_EQ(result->size(), 0);
}

/// Verifies equality deletes with VARCHAR columns.
TEST_F(EqualityDeleteFileReaderTest, stringColumnDelete) {
  auto rowType = ROW({"name", "age"}, {VARCHAR(), INTEGER()});

  auto baseData = makeRowVector(
      {"name", "age"},
      {
          makeFlatVector<std::string>({"alice", "bob", "charlie", "dave"}),
          makeFlatVector<int32_t>({25, 30, 35, 40}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete rows where name is "bob" or "dave".
  auto deleteData = makeRowVector(
      {"name"},
      {
          makeFlatVector<std::string>({"bob", "dave"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1}); // field ID 1 = "name"

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"name", "age"},
      {
          makeFlatVector<std::string>({"alice", "charlie"}),
          makeFlatVector<int32_t>({25, 35}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies equality deletes on a non-first column (field ID 2).
TEST_F(EqualityDeleteFileReaderTest, deleteOnSecondColumn) {
  auto rowType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"A", "B", "A", "C", "B"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete rows where category == "B".
  auto deleteData = makeRowVector(
      {"category"},
      {
          makeFlatVector<std::string>({"B"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{2}); // field ID 2 = column 1 = "category"

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Rows with category="B" (indices 1,4) deleted.
  auto expected = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 3, 4}),
          makeFlatVector<std::string>({"A", "A", "C"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that equality deletes apply when the delete file has a higher
/// sequence number than the data file (per the Iceberg V2+ spec).
TEST_F(EqualityDeleteFileReaderTest, sequenceNumberDeleteApplies) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2, 4}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  // Delete file has sequence number 5, data file has sequence number 3.
  // Since deleteSeq (5) > dataSeq (3), the delete should apply.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*dataSequenceNumber=*/3);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Rows with id=2 and id=4 are deleted.
  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 3, 5}),
          makeFlatVector<std::string>({"a", "c", "e"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that equality deletes are skipped when the delete file has a
/// lower or equal sequence number compared to the data file.
TEST_F(EqualityDeleteFileReaderTest, sequenceNumberDeleteSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  // Delete file has sequence number 2, data file has sequence number 5.
  // Since deleteSeq (2) <= dataSeq (5), the delete should be skipped.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // All rows survive because the delete file is skipped.
  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that equality deletes are skipped when the delete file has the
/// same sequence number as the data file (edge case of the <= check).
TEST_F(EqualityDeleteFileReaderTest, sequenceNumberEqualSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  // Delete file and data file have the same sequence number (5).
  // Since deleteSeq (5) <= dataSeq (5), the delete should be skipped.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // All rows survive because the delete file is skipped (equal seq#).
  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that when either sequence number is 0 (unassigned/legacy V1),
/// the delete file is always applied (filtering is disabled).
TEST_F(EqualityDeleteFileReaderTest, sequenceNumberZeroAlwaysApplies) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  // Delete file has sequence number 0 (legacy), data file has sequence 10.
  // Since deleteSeq is 0, filtering is disabled and the delete applies.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0);

  auto splits = makeSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*dataSequenceNumber=*/10);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Row id=2 is deleted because sequence number filtering is disabled.
  auto expected = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 3}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that when multiple delete files have different sequence numbers,
/// only those with higher sequence numbers than the data file are applied.
TEST_F(EqualityDeleteFileReaderTest, mixedSequenceNumbers) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = writeDataFile({baseData});

  // First delete file: seqNum=10 (higher than data seqNum=5) → applied.
  auto deleteData1 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2}),
      });
  auto eqDeleteFile1 = writeEqDeleteFile({deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/10);

  // Second delete file: seqNum=3 (lower than data seqNum=5) → skipped.
  auto deleteData2 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({4}),
      });
  auto eqDeleteFile2 = writeEqDeleteFile({deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeSplits(
      dataFile->getPath(),
      {icebergDeleteFile1, icebergDeleteFile2},
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Only id=2 is deleted (from delete file 1 with seqNum=10).
  // id=4 survives because delete file 2 (seqNum=3) is skipped.
  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "c", "d", "e"}),
      });

  assertEqualResults({expected}, {result});
}

// TODO: Add a Parquet-format equality delete test. Currently all equality
// delete tests use DWRF because writeToFile() (from HiveConnectorTestBase)
// only supports DWRF. Adding a Parquet test requires adding Parquet writer
// dependencies to this test target's BUCK file and a Parquet write helper.

} // namespace facebook::velox::connector::hive::iceberg
