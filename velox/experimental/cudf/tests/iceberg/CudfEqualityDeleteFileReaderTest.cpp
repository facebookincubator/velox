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

#include "velox/experimental/cudf/tests/iceberg/CudfIcebergTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive::iceberg;
using facebook::velox::common::testutil::TempFilePath;

namespace facebook::velox::cudf_velox::exec::test {

/// End-to-end tests for equality deletes via the CudfIcebergSplitReader.
/// These tests write DWRF and Parquet data files and DWRF delete files,
/// then execute table scans verifying that matching rows are filtered out.
class CudfEqualityDeleteFileReaderTest
    : public CudfIcebergTestBase,
      public ::testing::WithParamInterface<DeleteFileFormat> {};

/// Basic single-column equality delete.
TEST_P(CudfEqualityDeleteFileReaderTest, basicSingleColumnDelete) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({3, 7}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
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
TEST_P(CudfEqualityDeleteFileReaderTest, multiColumnDelete) {
  auto rowType = ROW({"a", "b", "c"}, {INTEGER(), VARCHAR(), BIGINT()});

  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"x", "y", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Delete rows where (a=2, b="y") — matches row index 1.
  // Also (a=5, b="y") — matches row index 4.
  // But (a=1, b="y") — no match (a=1 has b="x").
  auto deleteData = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int32_t>({2, 5, 1}),
          makeFlatVector<std::string>({"y", "y", "y"}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2}); // field IDs 1,2 = columns "a","b"

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
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
TEST_P(CudfEqualityDeleteFileReaderTest, noMatchingDeletes) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({100, 200}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
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
TEST_P(CudfEqualityDeleteFileReaderTest, allRowsDeleted) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  EXPECT_EQ(result->size(), 0);
}

/// Verifies equality deletes with VARCHAR columns.
TEST_P(CudfEqualityDeleteFileReaderTest, stringColumnDelete) {
  auto rowType = ROW({"name", "age"}, {VARCHAR(), INTEGER()});

  auto baseData = makeRowVector(
      {"name", "age"},
      {
          makeFlatVector<std::string>({"alice", "bob", "charlie", "dave"}),
          makeFlatVector<int32_t>({25, 30, 35, 40}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"name"},
      {
          makeFlatVector<std::string>({"bob", "dave"}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
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
TEST_P(CudfEqualityDeleteFileReaderTest, deleteOnSecondColumn) {
  auto rowType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"A", "B", "A", "C", "B"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Delete rows where category == "B".
  auto deleteData = makeRowVector(
      {"category"},
      {
          makeFlatVector<std::string>({"B"}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{2}); // field ID 2 = column 1 = "category"

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
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
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteApplies) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2, 4}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  // Delete file has sequence number 5, data file has sequence number 3.
  // Since deleteSeq (5) > dataSeq (3), the delete should apply.

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
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
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  // Delete file has sequence number 2, data file has sequence number 5.
  // Since deleteSeq (2) <= dataSeq (5), the delete should be skipped.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
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
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberEqualSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  // Delete file and data file have the same sequence number (5).
  // Since deleteSeq (5) <= dataSeq (5), the delete should be skipped.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
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
TEST_P(CudfEqualityDeleteFileReaderTest, sequenceNumberZeroAlwaysApplies) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2}),
      });
  auto eqDeleteFile = TempFilePath::create();
  const auto eqDeleteFileFormat = GetParam();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile->getPath(), {deleteData});

  // Delete file has sequence number 0 (legacy), data file has sequence 10.
  // Since deleteSeq is 0, filtering is disabled and the delete applies.
  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
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
TEST_P(CudfEqualityDeleteFileReaderTest, mixedSequenceNumbers) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);
  const auto eqDeleteFileFormat = GetParam();

  // First delete file: seqNum=10 (higher than data seqNum=5) → applied.
  auto deleteData1 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2}),
      });
  auto eqDeleteFile1 = TempFilePath::create();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile1->getPath(), {deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      toDwioFormat(eqDeleteFileFormat),
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
  auto eqDeleteFile2 = TempFilePath::create();
  writeDeleteFile(eqDeleteFileFormat, eqDeleteFile2->getPath(), {deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      toDwioFormat(eqDeleteFileFormat),
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile1, icebergDeleteFile2},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
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

INSTANTIATE_TEST_SUITE_P(
    DeleteFormats,
    CudfEqualityDeleteFileReaderTest,
    ::testing::Values(DeleteFileFormat::DWRF, DeleteFileFormat::PARQUET),
    [](const auto& info) {
      return info.param == DeleteFileFormat::PARQUET ? "Parquet" : "Dwrf";
    });

/// Cudf-specific: mixed DWRF + Parquet delete files targeting the same data.
class CudfMixedFormatEqualityDeleteTest : public CudfIcebergTestBase {};

TEST_F(CudfMixedFormatEqualityDeleteTest, mixedFormatDeleteFiles) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData1 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2}),
      });
  auto eqDeleteFile1 = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF, eqDeleteFile1->getPath(), {deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1});

  auto deleteData2 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({4}),
      });
  auto eqDeleteFile2 = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile2->getPath(), {deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {icebergDeleteFile1, icebergDeleteFile2});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 3, 5}),
          makeFlatVector<std::string>({"a", "c", "e"}),
      });

  assertEqualResults({expected}, {result});
}

} // namespace facebook::velox::cudf_velox::exec::test
