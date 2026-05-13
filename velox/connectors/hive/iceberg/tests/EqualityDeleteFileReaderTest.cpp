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
    return makeSplits(
        dataFilePath,
        /*partitionKeys=*/{},
        deleteFiles,
        dataSequenceNumber);
  }

  /// Creates splits with equality delete files and partition keys attached.
  /// Use this overload to exercise the equality-delete augmentation for
  /// partition columns missing from the user's projection.
  std::vector<std::shared_ptr<ConnectorSplit>> makeSplits(
      const std::string& dataFilePath,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys,
      const std::vector<IcebergDeleteFile>& deleteFiles,
      int64_t dataSequenceNumber = 0) {
    auto fileSize = getFileSize(dataFilePath);
    return {std::make_shared<HiveIcebergSplit>(
        kIcebergConnectorId,
        dataFilePath,
        dwio::common::FileFormat::DWRF,
        0,
        fileSize,
        partitionKeys,
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
    return makeTableScanPlan(rowType, rowType);
  }

  /// Builds a table scan plan node with separate output and table column
  /// schemas. Use this when the user's projection ('outputType') does not
  /// contain every column referenced by an equality delete file
  /// ('dataColumns' must contain the full table schema so the equality
  /// column resolution can map field IDs to names).
  core::PlanNodePtr makeTableScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns) {
    return PlanBuilder()
        .startTableScan()
        .connectorId(kIcebergConnectorId)
        .outputType(outputType)
        .dataColumns(dataColumns)
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

/// Regression test for the bug where IcebergSplitReader fails with
/// "Column not found in row: <name>" when an equality-delete column is not
/// part of the user's projection. The reader must augment its scan spec to
/// physically read the equality-delete column, apply the delete, and then
/// project the column away from the output before returning to the operator.
TEST_F(EqualityDeleteFileReaderTest, equalityColumnNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  // The user only selects 'value'. The equality delete is on 'id', which is
  // NOT in the projection — this is the case that previously failed.
  auto outputType = ROW({"value"}, {VARCHAR()});

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
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // The 'id' column must not appear in the output; only 'value' is projected.
  // Rows with id=3 ("d") and id=7 ("h") are removed by the equality delete.
  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "b", "c", "e", "f", "g", "i", "j"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that two equality-delete files referencing the SAME column not in
/// the user's projection only augment 'scanSpec_' once. Exercises the
/// de-duplication branch in 'IcebergSplitReader::prepareSplit'.
TEST_F(EqualityDeleteFileReaderTest, multipleDeleteFilesSameMissingColumn) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Two delete files, both targeting 'id' (which is NOT in the projection).
  auto deleteData1 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({2, 5}),
      });
  auto eqDeleteFile1 = writeEqDeleteFile({deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1});

  auto deleteData2 = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({0, 9}),
      });
  auto eqDeleteFile2 = writeEqDeleteFile({deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits =
      makeSplits(dataFile->getPath(), {icebergDeleteFile1, icebergDeleteFile2});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Rows with id=0, 2, 5, 9 are removed (across both delete files).
  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"b", "d", "e", "g", "h", "i"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies a multi-column equality-delete file where some columns ARE in the
/// user's projection and some are NOT. Both must end up in the read output for
/// the equality probe to succeed, while only the projected columns appear in
/// the operator-visible result.
TEST_F(EqualityDeleteFileReaderTest, equalityMixedInAndOutOfProjection) {
  auto tableType = ROW({"a", "b", "c"}, {INTEGER(), VARCHAR(), BIGINT()});
  // User selects only 'b' and 'c'. 'a' is referenced by the equality delete
  // but not part of the projection.
  auto outputType = ROW({"b", "c"}, {VARCHAR(), BIGINT()});

  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"x", "y", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete rows where (a=2, b="y") -- removes row 1.
  // Also (a=1, b="y") -- no match (row with a=1 has b="x").
  auto deleteData = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int32_t>({2, 1}),
          makeFlatVector<std::string>({"y", "y"}),
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
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Row 1 (a=2, b="y", c=20) is deleted. The remaining rows project to
  // (b, c).
  auto expected = makeRowVector(
      {"b", "c"},
      {
          makeFlatVector<std::string>({"x", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 30, 40, 50}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies that an equality delete on a partition column that IS in the data
/// file (Iceberg-style) but NOT in the user's projection works correctly. The
/// augmentation should set the partition value as a constant; the file-read
/// path should then leave the constant in place because the column is present
/// in 'fileType'.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnInFileNotInProjection) {
  auto tableType = ROW({"part", "value"}, {INTEGER(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  // Data file contains both 'part' and 'value', all rows in partition 2.
  auto baseData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({2, 2, 2, 2}),
          makeFlatVector<std::string>({"a", "b", "c", "d"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({2, 2}),
          makeFlatVector<std::string>({"b", "d"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{{"part", std::optional<std::string>{"2"}}},
      {icebergDeleteFile});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Same as above but the partition value does NOT match the equality-delete
/// value, so no rows should be removed.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnNonMatchingPartition) {
  auto tableType = ROW({"part", "value"}, {INTEGER(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  // Data file holds rows in partition 2.
  auto baseData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({2, 2, 2}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete (part=99, value="b"). No file row matches part=99.
  auto deleteData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({99}),
          makeFlatVector<std::string>({"b"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{{"part", std::optional<std::string>{"2"}}},
      {icebergDeleteFile});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "b", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Multi-column equality delete where ONE column is a partition column not in
/// the projection and ANOTHER is a regular data column not in the projection.
/// Both must be augmented; the partition column gets a constant value, the
/// regular column is read from the file.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityMixedPartitionAndRegularNotInProjection) {
  auto tableType =
      ROW({"part", "id", "value"}, {INTEGER(), BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  // Data file contains all three columns, all rows in partition 7.
  auto baseData = makeRowVector(
      {"part", "id", "value"},
      {
          makeFlatVector<int32_t>({7, 7, 7, 7}),
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<std::string>({"a", "b", "c", "d"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete (part=7, id=20) and (part=7, id=40). Should remove "b" and "d".
  auto deleteData = makeRowVector(
      {"part", "id"},
      {
          makeFlatVector<int32_t>({7, 7}),
          makeFlatVector<int64_t>({20, 40}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2}); // field IDs 1,2 = part, id

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{{"part", std::optional<std::string>{"7"}}},
      {icebergDeleteFile});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Verifies equality delete on a DATE partition column not in projection.
/// Iceberg encodes DATE partition values as days-since-epoch (e.g. "19345").
/// This exercises the type-derived 'isDaysSinceEpoch' flag in
/// 'configureEqualityDeleteColumns' — for DATE columns the partition string
/// must be parsed as an integer day count, NOT as an ISO-8601 date string.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityDatePartitionColumnNotInProjection) {
  auto tableType = ROW({"part_date", "value"}, {DATE(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  // 19345 days since 1970-01-01 == 2022-12-22. All file rows belong to that
  // partition.
  constexpr int32_t kPartitionDays = 19345;
  auto baseData = makeRowVector(
      {"part_date", "value"},
      {
          makeFlatVector<int32_t>(
              {kPartitionDays, kPartitionDays, kPartitionDays}, DATE()),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Delete (part_date=19345, value="b").
  auto deleteData = makeRowVector(
      {"part_date", "value"},
      {
          makeFlatVector<int32_t>({kPartitionDays}, DATE()),
          makeFlatVector<std::string>({"b"}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/
      {{"part_date",
        std::optional<std::string>{std::to_string(kPartitionDays)}}},
      {icebergDeleteFile});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Exercises the filter-only column upgrade path in
/// 'configureEqualityDeleteColumns'. The equality-delete column 'id' is
/// referenced by a WHERE predicate (so the planner installs a scan-spec
/// child with 'projectOut=false') but is NOT in the user's SELECT
/// projection. The augmentation must upgrade the existing scan-spec child
/// from filter-only to 'projectOut=true' and assign a non-conflicting
/// channel so the equality-delete reader can probe by name.
TEST_F(EqualityDeleteFileReaderTest, equalityFilterOnlyColumnNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Equality delete removes id == 4 and id == 8.
  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({4, 8}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1}); // field ID 1 = "id"

  auto splits = makeSplits(dataFile->getPath(), {icebergDeleteFile});
  // WHERE id >= 3 keeps rows {3,4,5,6,7,8,9} from the file; the equality
  // delete then removes id=4 and id=8, leaving values {d->skipped} no, we
  // expect surviving values for ids {3,5,6,7,9}, projected as 'value' only.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .subfieldFilter("id >= 3")
                  .endTableScan()
                  .planNode();
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"d", "f", "g", "h", "j"}),
      });

  assertEqualResults({expected}, {result});
}

/// Regression test for the schema-evolution +
/// partition-column-not-in-projection scenario surfaced by the Presto Iceberg
/// integration test 'testEqualityDeleteWithPartitionColumnMissingInSelect'.
///
/// Setup (mirrors the Presto test for the older data file):
///   - Full table schema: (a, b, c, d) with a, c, d as partition columns.
///   - The data file under test was written BEFORE 'd' was added, so it
///     physically contains only (a, b, c).
///   - User projection: (a, b, d). 'd' must be NULL-filled (not in file);
///     'c' is NOT in the projection but IS referenced by the equality
///     delete and IS the file's identity-partition column.
///
/// The equality delete (a=6, c=2, b=1006) targets the file row
/// (6, '1006', 2). The augmentation must:
///   1. Add 'c' to 'scanSpec_' / 'readerOutputType_' so the eq-delete
///      probe can find it by name.
///   2. Leave 'd' alone — 'd' is in the user projection and gets the
///      standard schema-evolution NULL-fill from 'adaptColumns'.
///   3. Honour the existing partition-key constant on 'c' regardless of
///      whether the file physically contains 'c'.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnNotInProjectionWithEvolvedSchema) {
  // Full evolved table schema (after 'ALTER TABLE ADD COLUMN d').
  auto tableType =
      ROW({"a", "b", "c", "d"}, {INTEGER(), VARCHAR(), INTEGER(), VARCHAR()});
  // User selects 'a', 'b', 'd'. Note 'c' is NOT projected.
  auto outputType = ROW({"a", "b", "d"}, {INTEGER(), VARCHAR(), VARCHAR()});

  // Data file contains only (a, b, c) — written before 'd' was added.
  // Both rows are in the (a=6, c=2) partition.
  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({6, 6}),
          makeFlatVector<std::string>({"1006", "1009"}),
          makeFlatVector<int32_t>({2, 2}),
      });
  auto dataFile = writeDataFile({baseData});

  // Equality delete on (a, b, c) with values (6, '1006', 2). Field IDs
  // are in field-id order = [1, 2, 3].
  auto deleteData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({6}),
          makeFlatVector<std::string>({"1006"}),
          makeFlatVector<int32_t>({2}),
      });
  auto eqDeleteFile = writeEqDeleteFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2, 3});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/
      {{"a", std::optional<std::string>{"6"}},
       {"c", std::optional<std::string>{"2"}}},
      {icebergDeleteFile});
  auto plan = makeTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Row (6, '1006', 2) is deleted by the equality delete; (6, '1009', 2)
  // survives. 'd' is NULL because the data file was written before 'd'
  // was added.
  auto expected = makeRowVector(
      {"a", "b", "d"},
      {
          makeFlatVector<int32_t>({6}),
          makeFlatVector<std::string>({"1009"}),
          makeNullableFlatVector<std::string>({std::nullopt}),
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
