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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"

namespace facebook::velox::connector::hive::iceberg::test {

using exec::test::assertEqualResults;
using exec::test::AssertQueryBuilder;

// ---------------------------------------------------------------------------
// FileWriteMode — parameterizes file format and whether iceberg.id field IDs
// are stamped in the written files.
//
// Valid combinations:
//   {DWRF,    withFieldIds=false}  — plain DWRF, positional-fallback path
//   {DWRF,    withFieldIds=true }  — DWRF with iceberg.id attrs, kFieldId path
//   {PARQUET, withFieldIds=true }  — Parquet with field IDs (only valid mode)
// ---------------------------------------------------------------------------
struct FileWriteMode {
  dwio::common::FileFormat format;
  bool withFieldIds;

  std::string toString() const {
    const std::string fmt =
        format == dwio::common::FileFormat::PARQUET ? "Parquet" : "Dwrf";
    return fmt + (withFieldIds ? "_FieldId" : "_Positional");
  }
};

// ---------------------------------------------------------------------------
// EqualityDeleteFileReaderTest
//
// Non-parameterized base fixture.  Partition-column and evolved-schema tests
// live here as TEST_F.  The parameterized class inherits this.
// ---------------------------------------------------------------------------
class EqualityDeleteFileReaderTest : public IcebergTestBase {
 protected:
  /// Writes a file in the format described by 'mode'. When
  /// Writes a file according to 'mode':
  ///   DWRF  + withFieldIds=false  — plain DWRF, no iceberg.id attributes
  ///                                  (positional fallback in DwrfReader)
  ///   DWRF  + withFieldIds=true   — DWRF with iceberg.id footer attributes
  ///                                  (kFieldId / renameByFieldId path)
  ///   PARQUET + withFieldIds=true — Parquet with field_id metadata stamped
  ///                                  (kParquetFieldId path)
  ///   PARQUET + withFieldIds=false— Parquet without field_id metadata;
  ///                                  buildFieldIds() returns empty so
  ///                                  kParquetFieldId is not activated and the
  ///                                  reader uses kPosition (ordinal binding).
  std::shared_ptr<common::testutil::TempFilePath> writeFile(
      const std::vector<RowVectorPtr>& data,
      const std::vector<int32_t>& fieldIds,
      const FileWriteMode& mode) {
    if (mode.format == dwio::common::FileFormat::PARQUET) {
#ifdef VELOX_ENABLE_PARQUET
      // Pass fieldIds only when the mode wants them stamped; empty vector means
      // no field_id metadata in the Parquet schema -> kPosition mode.
      return writeParquetFile(
          data, mode.withFieldIds ? fieldIds : std::vector<int32_t>{});
#else
      VELOX_FAIL("Parquet support is not enabled");
#endif
    }
    return mode.withFieldIds ? writeDwrfFileWithFieldIds(data, fieldIds)
                             : writeDataFile(data);
  }

  /// Creates splits for the data file using the format in 'mode', attaching
  /// delete files and (optionally) partition keys.
  std::vector<std::shared_ptr<ConnectorSplit>> makeSplits(
      const std::string& dataFilePath,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys,
      const FileWriteMode& mode,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      int64_t dataSequenceNumber = 0,
      const std::unordered_map<int32_t, std::optional<std::string>>&
          identityPartitionKeys = {}) {
    fileFormat_ = mode.format; // makeIcebergSplits uses fileFormat_
    return makeIcebergSplits(
        dataFilePath,
        deleteFiles,
        partitionKeys,
        /*splitCount=*/1,
        /*infoColumns=*/{},
        dataSequenceNumber,
        identityPartitionKeys);
  }

  /// Builds an IcebergDeleteFile descriptor.
  IcebergDeleteFile makeDeleteFile(
      const std::string& path,
      const std::vector<int32_t>& equalityFieldIds,
      dwio::common::FileFormat format,
      int64_t recordCount = 2,
      int64_t deleteSeqNum = 0) {
    return IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        path,
        format,
        recordCount,
        getFileSize(path),
        equalityFieldIds,
        /*lowerBounds=*/{},
        /*upperBounds=*/{},
        deleteSeqNum);
  }
};

// ---------------------------------------------------------------------------
// EqualityDeleteFileReaderTestP — parameterized over FileWriteMode.
//
// Every TEST_P runs three times:
//   1. Dwrf_Positional — DWRF, no field IDs     (kPosition fallback)
//   2. Dwrf_FieldId    — DWRF, iceberg.id attrs (kFieldId path)
//   3. Parquet_FieldId — Parquet field_id       (kParquetFieldId path)
// ---------------------------------------------------------------------------
class EqualityDeleteFileReaderTestP
    : public EqualityDeleteFileReaderTest,
      public ::testing::WithParamInterface<FileWriteMode> {
 protected:
  void SetUp() override {
#ifndef VELOX_ENABLE_PARQUET
    if (GetParam().format == dwio::common::FileFormat::PARQUET) {
      GTEST_SKIP() << "Parquet support not enabled";
    }
#endif
    EqualityDeleteFileReaderTest::SetUp();
    fileFormat_ = GetParam().format;
  }

  std::shared_ptr<common::testutil::TempFilePath> writeDataFileP(
      const std::vector<RowVectorPtr>& data,
      const std::vector<int32_t>& fieldIds) {
    return writeFile(data, fieldIds, GetParam());
  }

  std::vector<std::shared_ptr<ConnectorSplit>> makeSplitsP(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      int64_t dataSequenceNumber = 0,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {}) {
    return makeSplits(
        dataFilePath,
        partitionKeys,
        GetParam(),
        deleteFiles,
        dataSequenceNumber,
        {});
  }
};

// Three FileWriteMode combinations:
//
//   Dwrf_Positional {DWRF, false}   — no iceberg.id attributes; DwrfReader
//                                     falls back to positional name mapping.
//
//   Dwrf_FieldId    {DWRF, true}    — "iceberg.id" footer attributes;
//                                     DwrfReader uses renameByFieldId.
//
//   Parquet_FieldId {PARQUET, true} — Parquet field_id metadata present;
//                                     IcebergSplitReader activates
//                                     kParquetFieldId mapping.
//
// {PARQUET, false} does not work end-to-end: without field IDs,
// buildFieldIds() returns empty, IcebergSplitReader does not set a fileSchema
// on baseReaderOpts_, and the Parquet reader has no requested-type to match
// positions against — physical columns cannot be bound to the scan-spec names,
// yielding all-null output.  All production Iceberg Parquet files carry field
// IDs; this combination is therefore not a valid use case.
INSTANTIATE_TEST_SUITE_P(
    Formats,
    EqualityDeleteFileReaderTestP,
    ::testing::Values(
        FileWriteMode{dwio::common::FileFormat::DWRF, /*withFieldIds=*/false},
        FileWriteMode{dwio::common::FileFormat::DWRF, /*withFieldIds=*/true},
        FileWriteMode{
            dwio::common::FileFormat::PARQUET,
            /*withFieldIds=*/true}),
    [](const ::testing::TestParamInfo<FileWriteMode>& info) {
      return info.param.toString();
    });

// ===========================================================================
// Parameterized tests
// ===========================================================================

/// Verifies that base rows matching the equality delete file are removed.
TEST_P(EqualityDeleteFileReaderTestP, basicSingleColumnDelete) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({3, 7})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 4, 5, 6, 8, 9}),
          makeFlatVector<std::string>({"a", "b", "c", "e", "f", "g", "i", "j"}),
      });
  assertEqualResults({expected}, {result});
}

/// Regression test: equality-delete column absent from the user's projection.
TEST_P(EqualityDeleteFileReaderTestP, equalityColumnNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({3, 7})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {makeFlatVector<std::string>({"a", "b", "c", "e", "f", "g", "i", "j"})});
  assertEqualResults({expected}, {result});
}

// Regression test: a column only a later split needs. The stable child order
// used to be fixed when the first reader tree was built, so the second
// split's tree had no reader for 'id' and its delete went unapplied.
TEST_P(EqualityDeleteFileReaderTestP, equalityDeleteColumnAddedBySecondSplit) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  // 'id' is outside the projection, so only the delete file puts it in the
  // scan spec.
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto firstData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto firstDataFile = writeDataFileP({firstData}, {1, 2});

  auto secondData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({5, 6, 7, 8, 9}),
          makeFlatVector<std::string>({"f", "g", "h", "i", "j"}),
      });
  auto secondDataFile = writeDataFileP({secondData}, {1, 2});

  auto thirdData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({10, 11, 12, 13, 14}),
          makeFlatVector<std::string>({"k", "l", "m", "n", "o"}),
      });
  auto thirdDataFile = writeDataFileP({thirdData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({6, 8})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});
  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(firstDataFile->getPath());
  auto secondSplits =
      makeSplitsP(secondDataFile->getPath(), {icebergDeleteFile});
  splits.insert(splits.end(), secondSplits.begin(), secondSplits.end());
  auto thirdSplits = makeSplitsP(thirdDataFile->getPath());
  splits.insert(splits.end(), thirdSplits.begin(), thirdSplits.end());

  // One driver and no preloading make the splits share one ScanSpec, which is
  // what carries the stale order forward.
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan)
                    .splits(splits)
                    .maxDrivers(1)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "0")
                    .copyResults(pool());

  // Only the second split loses rows, id=6 and id=8.
  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>(
              {"a",
               "b",
               "c",
               "d",
               "e",
               "f",
               "h",
               "j",
               "k",
               "l",
               "m",
               "n",
               "o"}),
      });

  assertEqualResults({expected}, {result});
}

/// Two delete files targeting the same column not in the projection.
TEST_P(EqualityDeleteFileReaderTestP, multipleDeleteFilesSameMissingColumn) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData1 = makeRowVector({"id"}, {makeFlatVector<int64_t>({2, 5})});
  auto eqDeleteFile1 = writeDataFileP({deleteData1}, {1});
  auto icebergDeleteFile1 =
      makeDeleteFile(eqDeleteFile1->getPath(), {1}, GetParam().format);

  auto deleteData2 = makeRowVector({"id"}, {makeFlatVector<int64_t>({0, 9})});
  auto eqDeleteFile2 = writeDataFileP({deleteData2}, {1});
  auto icebergDeleteFile2 =
      makeDeleteFile(eqDeleteFile2->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(
      dataFile->getPath(), {icebergDeleteFile1, icebergDeleteFile2});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"}, {makeFlatVector<std::string>({"b", "d", "e", "g", "h", "i"})});
  assertEqualResults({expected}, {result});
}

/// Multi-column equality delete: some delete columns in projection, some not.
TEST_P(EqualityDeleteFileReaderTestP, equalityMixedInAndOutOfProjection) {
  auto tableType = ROW({"a", "b", "c"}, {INTEGER(), VARCHAR(), BIGINT()});
  auto outputType = ROW({"b", "c"}, {VARCHAR(), BIGINT()});

  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"x", "y", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2, 3});

  // Delete (a=2, b="y") => row 1; (a=1, b="y") => no match.
  auto deleteData = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int32_t>({2, 1}),
          makeFlatVector<std::string>({"y", "y"}),
      });
  auto eqDeleteFile = writeDataFileP({deleteData}, {1, 2});

  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1, 2}, GetParam().format, /*recordCount=*/3);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"b", "c"},
      {
          makeFlatVector<std::string>({"x", "z", "x", "y"}),
          makeFlatVector<int64_t>({10, 30, 40, 50}),
      });
  assertEqualResults({expected}, {result});
}

/// Filter-only column upgrade: 'id' in WHERE but not SELECT, also the
/// equality-delete column.
TEST_P(EqualityDeleteFileReaderTestP, equalityFilterOnlyColumnNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({4, 8})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  // WHERE id >= 3 => {3,4,5,6,7,8,9}; delete id=4,8 => {3,5,6,7,9}.
  auto plan = makeIcebergTableScanPlan(
      outputType, tableType, {}, /*subfieldFilters=*/{"id >= 3"});
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"}, {makeFlatVector<std::string>({"d", "f", "g", "h", "j"})});
  assertEqualResults({expected}, {result});
}

/// Multi-column equality delete: both columns must match simultaneously.
TEST_P(EqualityDeleteFileReaderTestP, multiColumnDelete) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({3}),
          makeFlatVector<std::string>({"c"}),
      });
  auto eqDeleteFile = writeDataFileP({deleteData}, {1, 2});

  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1, 2}, GetParam().format, /*recordCount=*/1);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"}, {makeFlatVector<std::string>({"a", "b", "d", "e"})});
  assertEqualResults({expected}, {result});
}

/// Two separate delete files both apply (multi-reader path).
TEST_P(EqualityDeleteFileReaderTestP, twoDeleteFiles) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<std::string>(
              {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData1 = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 5})});
  auto eqDeleteFile1 = writeDataFileP({deleteData1}, {1});
  auto icebergDelete1 =
      makeDeleteFile(eqDeleteFile1->getPath(), {1}, GetParam().format);

  auto deleteData2 = makeRowVector({"id"}, {makeFlatVector<int64_t>({3, 8})});
  auto eqDeleteFile2 = writeDataFileP({deleteData2}, {1});
  auto icebergDelete2 =
      makeDeleteFile(eqDeleteFile2->getPath(), {1}, GetParam().format);

  auto splits =
      makeSplitsP(dataFile->getPath(), {icebergDelete1, icebergDelete2});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Deleted: 1,3,5,8. Surviving: 0,2,4,6,7,9.
  auto expected = makeRowVector(
      {"value"}, {makeFlatVector<std::string>({"a", "c", "e", "g", "h", "j"})});
  assertEqualResults({expected}, {result});
}

/// No rows match the delete — all rows survive.
TEST_P(EqualityDeleteFileReaderTestP, noMatchingDeletes) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile = writeDataFileP({baseData}, {1});

  auto deleteData =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({100, 200})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})})}, {result});
}

/// Every row is deleted.
TEST_P(EqualityDeleteFileReaderTestP, allRowsDeleted) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile = writeDataFileP({baseData}, {1});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1}, GetParam().format, /*recordCount=*/3);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  EXPECT_EQ(result->size(), 0);
}

/// VARCHAR column equality delete.
TEST_P(EqualityDeleteFileReaderTestP, stringColumnDelete) {
  auto rowType = ROW({"name", "age"}, {VARCHAR(), INTEGER()});

  auto baseData = makeRowVector(
      {"name", "age"},
      {
          makeFlatVector<std::string>({"alice", "bob", "charlie", "dave"}),
          makeFlatVector<int32_t>({25, 30, 35, 40}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData =
      makeRowVector({"name"}, {makeFlatVector<std::string>({"bob", "dave"})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  auto icebergDeleteFile =
      makeDeleteFile(eqDeleteFile->getPath(), {1}, GetParam().format);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"name", "age"},
      {
          makeFlatVector<std::string>({"alice", "charlie"}),
          makeFlatVector<int32_t>({25, 35}),
      });
  assertEqualResults({expected}, {result});
}

/// Verifies equality deletes after a field has been dropped, leaving sparse
/// top-level field IDs.
TEST_F(EqualityDeleteFileReaderTest, nonSequentialEqualityFieldId) {
  auto tableType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});
  const std::vector<int32_t> fieldIds{1, 3};

  auto baseData = makeRowVector(
      {"id", "dropped", "category"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
          makeFlatVector<std::string>({"A", "B", "A", "C", "B"}),
      });
  auto dataFile = writeDwrfFileWithFieldIds({baseData}, {1, 2, 3});

  auto deleteData = makeRowVector(
      {"category"},
      {
          makeFlatVector<std::string>({"B"}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{3});

  auto splits = makeSplits(
      dataFile->getPath(),
      {},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(tableType, tableType, fieldIds);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 3, 4}),
          makeFlatVector<std::string>({"A", "A", "C"}),
      });
  assertEqualResults({expected}, {result});
}

TEST_F(
    EqualityDeleteFileReaderTest,
    nonSequentialEqualityFieldIdNotInProjection) {
  auto tableType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"id"}, {BIGINT()});
  const std::vector<int32_t> fieldIds{1, 3};

  auto baseData = makeRowVector(
      {"id", "dropped", "category"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
          makeFlatVector<std::string>({"A", "B", "A", "C", "B"}),
      });
  auto dataFile = writeDwrfFileWithFieldIds({baseData}, {1, 2, 3});

  auto deleteData = makeRowVector(
      {"category"},
      {
          makeFlatVector<std::string>({"B"}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{3});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(outputType, tableType, fieldIds);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({1, 3, 4}),
      });
  assertEqualResults({expected}, {result});
}

/// Verifies the ordinal fallback on a non-first column when full-schema field
/// IDs are unavailable.
TEST_P(EqualityDeleteFileReaderTestP, deleteOnSecondColumn) {
  auto rowType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"A", "B", "A", "C", "B"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData =
      makeRowVector({"category"}, {makeFlatVector<std::string>({"B"})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {2});

  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {2}, GetParam().format, /*recordCount=*/1);

  auto splits = makeSplitsP(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "category"},
      {
          makeFlatVector<int64_t>({1, 3, 4}),
          makeFlatVector<std::string>({"A", "A", "C"}),
      });
  assertEqualResults({expected}, {result});
}

/// Delete applies when deleteSeqNum > dataSeqNum.
TEST_P(EqualityDeleteFileReaderTestP, sequenceNumberDeleteApplies) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({2, 4})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  // deleteSeqNum=5 > dataSeqNum=3 => applies.
  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1}, GetParam().format, 2, /*deleteSeqNum=*/5);

  auto splits = makeSplitsP(
      dataFile->getPath(), {icebergDeleteFile}, /*dataSequenceNumber=*/3);
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 3, 5}),
          makeFlatVector<std::string>({"a", "c", "e"}),
      });
  assertEqualResults({expected}, {result});
}

/// Delete skipped when deleteSeqNum <= dataSeqNum.
TEST_P(EqualityDeleteFileReaderTestP, sequenceNumberDeleteSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  // deleteSeqNum=2 <= dataSeqNum=5 => skipped.
  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1}, GetParam().format, 3, /*deleteSeqNum=*/2);

  auto splits = makeSplitsP(
      dataFile->getPath(), {icebergDeleteFile}, /*dataSequenceNumber=*/5);
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  assertEqualResults({expected}, {result});
}

/// Delete skipped when deleteSeqNum == dataSeqNum (edge case of <=).
TEST_P(EqualityDeleteFileReaderTestP, sequenceNumberEqualSkipped) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  // deleteSeqNum=5 == dataSeqNum=5 => skipped.
  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1}, GetParam().format, 3, /*deleteSeqNum=*/5);

  auto splits = makeSplitsP(
      dataFile->getPath(), {icebergDeleteFile}, /*dataSequenceNumber=*/5);
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  assertEqualResults({expected}, {result});
}

/// deleteSeqNum=0 disables filtering — delete always applies.
TEST_P(EqualityDeleteFileReaderTestP, sequenceNumberZeroAlwaysApplies) {
  auto rowType = ROW({"id"}, {BIGINT()});

  auto baseData = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile = writeDataFileP({baseData}, {1});

  auto deleteData = makeRowVector({"id"}, {makeFlatVector<int64_t>({2})});
  auto eqDeleteFile = writeDataFileP({deleteData}, {1});

  // deleteSeqNum=0 => filtering disabled, applies despite dataSeqNum=10.
  auto icebergDeleteFile = makeDeleteFile(
      eqDeleteFile->getPath(), {1}, GetParam().format, 1, /*deleteSeqNum=*/0);

  auto splits = makeSplitsP(
      dataFile->getPath(), {icebergDeleteFile}, /*dataSequenceNumber=*/10);
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 3})})}, {result});
}

/// Only delete files with higher sequence numbers than the data file apply.
TEST_P(EqualityDeleteFileReaderTestP, mixedSequenceNumbers) {
  auto rowType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });
  auto dataFile = writeDataFileP({baseData}, {1, 2});

  // seqNum=10 > dataSeqNum=5 => applied.
  auto deleteData1 = makeRowVector({"id"}, {makeFlatVector<int64_t>({2})});
  auto eqDeleteFile1 = writeDataFileP({deleteData1}, {1});
  auto icebergDeleteFile1 = makeDeleteFile(
      eqDeleteFile1->getPath(), {1}, GetParam().format, 1, /*deleteSeqNum=*/10);

  // seqNum=3 <= dataSeqNum=5 => skipped.
  auto deleteData2 = makeRowVector({"id"}, {makeFlatVector<int64_t>({4})});
  auto eqDeleteFile2 = writeDataFileP({deleteData2}, {1});
  auto icebergDeleteFile2 = makeDeleteFile(
      eqDeleteFile2->getPath(), {1}, GetParam().format, 1, /*deleteSeqNum=*/3);

  auto splits = makeSplitsP(
      dataFile->getPath(),
      {icebergDeleteFile1, icebergDeleteFile2},
      /*dataSequenceNumber=*/5);
  auto plan = makeIcebergTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // id=2 deleted; id=4 survives.
  auto expected = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({1, 3, 4, 5}),
          makeFlatVector<std::string>({"a", "c", "d", "e"}),
      });
  assertEqualResults({expected}, {result});
}

// ===========================================================================
// Non-parameterized tests -- partition columns and evolved schema (DWRF only).
// ===========================================================================

/// Equality delete on a partition column in the data file but not projected.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnInFileNotInProjection) {
  auto tableType = ROW({"part", "value"}, {INTEGER(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

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
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      // Source field ID 1 ('part') is an explicit identity partition field.
      /*identityPartitionKeys=*/{{1, std::optional<std::string>{"2"}}});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector({"value"}, {makeFlatVector<std::string>({"a", "c"})})},
      {result});
}

/// Same as above but partition value does not match -- no rows deleted.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnNonMatchingPartition) {
  auto tableType = ROW({"part", "value"}, {INTEGER(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({2, 2, 2}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"part", "value"},
      {
          makeFlatVector<int32_t>({99}),
          makeFlatVector<std::string>({"b"}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      /*identityPartitionKeys=*/{{1, std::optional<std::string>{"2"}}});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector(
          {"value"}, {makeFlatVector<std::string>({"a", "b", "c"})})},
      {result});
}

/// One partition column + one regular column, both absent from the projection.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityMixedPartitionAndRegularNotInProjection) {
  auto tableType =
      ROW({"part", "id", "value"}, {INTEGER(), BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"part", "id", "value"},
      {
          makeFlatVector<int32_t>({7, 7, 7, 7}),
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<std::string>({"a", "b", "c", "d"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"part", "id"},
      {
          makeFlatVector<int32_t>({7, 7}),
          makeFlatVector<int64_t>({20, 40}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      {},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      /*identityPartitionKeys=*/{{1, std::optional<std::string>{"7"}}});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector({"value"}, {makeFlatVector<std::string>({"a", "c"})})},
      {result});
}

/// DATE partition column not in projection (days-since-epoch encoding).
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityDatePartitionColumnNotInProjection) {
  auto tableType = ROW({"part_date", "value"}, {DATE(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  constexpr int32_t kPartitionDays = 19345; // 2022-12-22
  auto baseData = makeRowVector(
      {"part_date", "value"},
      {
          makeFlatVector<int32_t>(
              {kPartitionDays, kPartitionDays, kPartitionDays}, DATE()),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"part_date", "value"},
      {
          makeFlatVector<int32_t>({kPartitionDays}, DATE()),
          makeFlatVector<std::string>({"b"}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      /*identityPartitionKeys=*/
      {{1, std::optional<std::string>{std::to_string(kPartitionDays)}}});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  assertEqualResults(
      {makeRowVector({"value"}, {makeFlatVector<std::string>({"a", "c"})})},
      {result});
}

/// Regression for transformed partition fields whose derived partition-field
/// name collides with a real source column name.
///
/// A partition field may be named anything, so a 'bucket[4]' field on source
/// column 'id' can itself be named "id". The split's name-keyed
/// 'partitionKeys' then maps "id" to the *bucket ordinal*, not to any row's
/// 'id'. Substituting that value for the source column would corrupt the
/// equality-delete probe. Only a field the spec marks 'identity' may be
/// substituted, and no identity metadata is supplied here, so the reader must
/// read the physical 'id' column from the data file.
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityBucketPartitionNameCollisionNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<std::string>({"a", "b", "c", "d"}),
      });
  auto dataFile = writeDataFile({baseData});

  // Equality delete removes the row whose physical id is 20.
  auto deleteData = makeRowVector(
      {"id"},
      {
          makeFlatVector<int64_t>({20}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeSplits(
      dataFile->getPath(),
      // The bucket ordinal, stored under a name that collides with the
      // source column.
      /*partitionKeys=*/{{"id", std::optional<std::string>{"2"}}},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      // 'bucket[4]' is not identity, so nothing is substitutable.
      /*identityPartitionKeys=*/{});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Only the physically matching row is deleted. Substituting the bucket
  // ordinal would make every row's id 2 and delete nothing.
  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "c", "d"}),
      });

  assertEqualResults({expected}, {result});
}

/// Regression for the 'void' transform, which always stores a null partition
/// value and — unlike bucket/truncate/temporal — keeps the source column's
/// name by default. A name-keyed lookup therefore finds an entry for "id"
/// and would install a constant null over every row, making the null
/// equality-delete key match everything. With identity metadata absent, the
/// physical non-null 'id' values must survive instead.
TEST_F(EqualityDeleteFileReaderTest, equalityVoidPartitionNotInProjection) {
  auto tableType = ROW({"id", "value"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"value"}, {VARCHAR()});

  auto baseData = makeRowVector(
      {"id", "value"},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<std::string>({"a", "b", "c"}),
      });
  auto dataFile = writeDataFile({baseData});

  // The delete key is null, matching the null the void transform stores.
  auto deleteData = makeRowVector(
      {"id"},
      {
          makeNullableFlatVector<int64_t>({std::nullopt}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeSplits(
      dataFile->getPath(),
      /*partitionKeys=*/{{"id", std::nullopt}},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      /*dataSequenceNumber=*/0,
      // 'void' is not identity, so its null is not substitutable.
      /*identityPartitionKeys=*/{});
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector(
      {"value"},
      {
          makeFlatVector<std::string>({"a", "b", "c"}),
      });

  assertEqualResults({expected}, {result});
}

/// Schema evolution + partition column not in projection (Presto regression).
TEST_F(
    EqualityDeleteFileReaderTest,
    equalityPartitionColumnNotInProjectionWithEvolvedSchema) {
  auto tableType =
      ROW({"a", "b", "c", "d"}, {INTEGER(), VARCHAR(), INTEGER(), VARCHAR()});
  auto outputType = ROW({"a", "b", "d"}, {INTEGER(), VARCHAR(), VARCHAR()});

  // Data file was written before 'd' was added -- contains only (a, b, c).
  auto baseData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({6, 6}),
          makeFlatVector<std::string>({"1006", "1009"}),
          makeFlatVector<int32_t>({2, 2}),
      });
  auto dataFile = writeDataFile({baseData});

  auto deleteData = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int32_t>({6}),
          makeFlatVector<std::string>({"1006"}),
          makeFlatVector<int32_t>({2}),
      });
  auto eqDeleteFile = writeDataFile({deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2, 3});

  auto splits = makeSplits(
      dataFile->getPath(),
      {{"a", std::optional<std::string>{"6"}},
       {"c", std::optional<std::string>{"2"}}},
      {dwio::common::FileFormat::DWRF, false},
      {icebergDeleteFile},
      0);
  auto plan = makeIcebergTableScanPlan(outputType, tableType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // (6,'1006',2) deleted; (6,'1009',2) survives. 'd' is NULL-filled.
  auto expected = makeRowVector(
      {"a", "b", "d"},
      {
          makeFlatVector<int32_t>({6}),
          makeFlatVector<std::string>({"1009"}),
          makeNullableFlatVector<std::string>({std::nullopt}),
      });
  assertEqualResults({expected}, {result});
}

} // namespace facebook::velox::connector::hive::iceberg::test
