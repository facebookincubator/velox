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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <algorithm>

#include <folly/Singleton.h>
#include <folly/lang/Bits.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

using TempFilePath = common::testutil::TempFilePath;

class IcebergReadTest : public test::IcebergTestBase {
 protected:
  struct AssignmentSpec {
    std::string outputName;
    std::string sourceName;
    TypePtr type;
    int fieldId;
    std::optional<std::string> defaultValue = std::nullopt;
  };

  struct RowLineageTestCase {
    std::vector<int64_t> values;
    std::optional<std::vector<std::optional<int64_t>>> storedRowIds =
        std::nullopt;
    std::optional<std::vector<std::optional<int64_t>>> storedSequenceNumbers =
        std::nullopt;
    std::optional<int64_t> firstRowId = std::nullopt;
    std::optional<int64_t> dataSequenceNumber = std::nullopt;
    std::vector<int64_t> deletePositions;
    std::string subfieldFilter;
    std::vector<RowVectorPtr> expectedVectors;
  };

  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    fileFormat_ = dwio::common::FileFormat::DWRF;
  }

  std::shared_ptr<IcebergColumnHandle> makeIcebergHandle(
      const std::string& name,
      const TypePtr& type,
      int fieldId,
      std::optional<std::string> defaultValue = std::nullopt) {
    return std::make_shared<IcebergColumnHandle>(
        name,
        HiveColumnHandle::ColumnType::kRegular,
        type,
        parquet::ParquetFieldId(fieldId),
        std::vector<common::Subfield>{},
        defaultValue);
  }

  void assertDefaultValues(
      const RowTypePtr& outputType,
      const RowTypePtr& scanSpecType,
      const ColumnHandleMap& assignments,
      const std::vector<RowVectorPtr>& data,
      const std::vector<RowVectorPtr>& expected,
      const std::unordered_map<std::string, std::string>& sessionProperties =
          {}) {
    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), data);
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(outputType)
                    .dataColumns(scanSpecType)
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();

    exec::test::AssertQueryBuilder(plan)
        .connectorSessionProperties(
            {{test::kIcebergConnectorId, sessionProperties}})
        .splits(makeIcebergSplits(dataFilePath->getPath()))
        .assertResults(expected);
  }

  std::vector<RowVectorPtr> makeSingleBigintData(
      const std::vector<int64_t>& values) {
    return {makeRowVector({makeFlatVector<int64_t>(values)})};
  }

  ColumnHandleMap makeAssignments(std::initializer_list<AssignmentSpec> specs) {
    ColumnHandleMap assignments;
    for (const auto& spec : specs) {
      assignments[spec.outputName] = spec.defaultValue.has_value()
          ? makeIcebergHandle(
                spec.sourceName, spec.type, spec.fieldId, *spec.defaultValue)
          : makeIcebergHandle(spec.sourceName, spec.type, spec.fieldId);
    }
    return assignments;
  }

  std::vector<RowVectorPtr> makeBigintAndVarcharExpected(
      const std::vector<std::string>& names,
      const RowVectorPtr& data,
      const std::vector<std::string>& values) {
    return {makeRowVector(
        names, {data->childAt(0), makeFlatVector<std::string>(values)})};
  }

  void assertRowLineage(const RowLineageTestCase& tc) {
    VELOX_CHECK_EQ(
        tc.storedRowIds.has_value(),
        tc.storedSequenceNumbers.has_value(),
        "rowIds and sequenceNumbers must both be set or both absent.");

    std::vector<RowVectorPtr> inputVectors;
    if (!tc.storedRowIds.has_value()) {
      inputVectors = {makeRowVector({makeFlatVector<int64_t>(tc.values)})};
    } else {
      static const std::vector<std::string> kFileColumns = {
          "c0", "_row_id", "_last_updated_sequence_number"};
      inputVectors = {makeRowVector(
          kFileColumns,
          {
              makeFlatVector<int64_t>(tc.values),
              makeNullableFlatVector<int64_t>(*tc.storedRowIds),
              makeNullableFlatVector<int64_t>(*tc.storedSequenceNumbers),
          })};
    }

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), inputVectors);

    std::vector<IcebergDeleteFile> deleteFiles;
    std::shared_ptr<TempFilePath> deleteFilePath;
    if (!tc.deletePositions.empty()) {
      auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
      auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
      deleteFilePath = TempFilePath::create();
      writeToFile(
          deleteFilePath->getPath(),
          {makeRowVector(
              {pathColumn->name, posColumn->name},
              {
                  makeFlatVector<std::string>(
                      static_cast<vector_size_t>(tc.deletePositions.size()),
                      [&](vector_size_t) { return dataFilePath->getPath(); }),
                  makeFlatVector<int64_t>(tc.deletePositions),
              })});

      std::unordered_map<int32_t, std::string> upperBounds;
      const uint64_t upperBound = static_cast<uint64_t>(*std::max_element(
          tc.deletePositions.begin(), tc.deletePositions.end()));
      const auto upperBoundLE = folly::Endian::little(upperBound);
      upperBounds[posColumn->id] = encoding::Base64::encode(
          std::string_view(
              reinterpret_cast<const char*>(&upperBoundLE),
              sizeof(upperBoundLE)));

      deleteFiles.push_back(IcebergDeleteFile(
          FileContent::kPositionalDeletes,
          deleteFilePath->getPath(),
          fileFormat_,
          static_cast<int64_t>(tc.deletePositions.size()),
          this->getFileSize(deleteFilePath->getPath()),
          {},
          {},
          upperBounds,
          0));
    }

    std::unordered_map<std::string, std::string> infoColumns;
    if (tc.firstRowId.has_value()) {
      infoColumns[IcebergMetadataColumn::kFirstRowIdInfoColumn] =
          std::to_string(*tc.firstRowId);
    }
    if (tc.dataSequenceNumber.has_value()) {
      infoColumns[IcebergMetadataColumn::kDataSequenceNumberInfoColumn] =
          std::to_string(*tc.dataSequenceNumber);
    }

    const auto outputType =
        ROW({"c0", "_row_id", "_last_updated_sequence_number"},
            {BIGINT(), BIGINT(), BIGINT()});
    const auto tableDataColumns = ROW({"c0"}, {BIGINT()});
    exec::test::PlanBuilder planBuilder;
    auto& tableScanBuilder =
        planBuilder.startTableScan(test::kIcebergConnectorId)
            .outputType(outputType)
            .dataColumns(tableDataColumns);
    if (!tc.subfieldFilter.empty()) {
      tableScanBuilder.subfieldFilter(tc.subfieldFilter);
    }
    auto plan = tableScanBuilder.endTableScan().planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits({makeIcebergSplitWithInfoColumns(
            dataFilePath->getPath(), infoColumns, deleteFiles)})
        .assertResults(tc.expectedVectors);
  }
};

TEST_F(IcebergReadTest, schemaEvolutionRemoveColumn) {
  // Write data file with old schema (c0, c1, c2).
  auto oldRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});
  auto newRowType = ROW({"c0", "c2"}, {BIGINT(), VARCHAR()});

  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      oldRowType->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      })};
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  std::vector<RowVectorPtr> expectedVectors = {makeRowVector(
      newRowType->names(),
      {dataVectors[0]->childAt(0), dataVectors[0]->childAt(2)})};

  // Read with new schema (c0 and c2 only, c1 removed).
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(newRowType)
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults(expectedVectors);
}

TEST_F(IcebergReadTest, schemaEvolutionAddColumns) {
  // Write data file with old schema (only c0).
  auto oldRowType = ROW({"c0"}, {BIGINT()});
  auto newRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});

  std::vector<RowVectorPtr> dataVectors = {
      makeRowVector({makeFlatVector<int64_t>({100, 200, 300})})};
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  std::vector<RowVectorPtr> expectedVectors = {makeRowVector(
      {dataVectors[0]->childAt(0),
       makeNullConstant(TypeKind::INTEGER, 3),
       makeNullConstant(TypeKind::VARCHAR, 3)})};

  // Read with new schema (c0, c1, and c2).
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(newRowType)
                  .dataColumns(newRowType)
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults(expectedVectors);
}

TEST_F(IcebergReadTest, addColumnWithDefault) {
  // Test Iceberg V3 initial-default: a column added after data files were
  // written should return its initial-default value, not NULL.
  auto newRowType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto dataVectors = makeSingleBigintData({1, 2, 3});
  auto assignments = makeAssignments(
      {{"c0", "c0", BIGINT(), 1}, {"country", "country", VARCHAR(), 2, "IN"}});
  auto expectedVectors = makeBigintAndVarcharExpected(
      newRowType->names(), dataVectors[0], {"IN", "IN", "IN"});

  assertDefaultValues(
      newRowType, newRowType, assignments, dataVectors, expectedVectors);
}

TEST_F(IcebergReadTest, addColumnWithDefaultAndAlias) {
  auto outputType = ROW({"c0", "region"}, {BIGINT(), VARCHAR()});
  auto dataColumns = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto dataVectors = makeSingleBigintData({1, 2, 3});
  auto assignments = makeAssignments({{"c0", "c0", BIGINT(), 1}});
  // Key is "region" (alias), but handle refers to "country" (table column).
  assignments["region"] = makeIcebergHandle("country", VARCHAR(), 2, "IN");
  auto expectedVectors = makeBigintAndVarcharExpected(
      outputType->names(), dataVectors[0], {"IN", "IN", "IN"});

  assertDefaultValues(
      outputType, dataColumns, assignments, dataVectors, expectedVectors);
}

TEST_F(IcebergReadTest, fileValueOverridesDefault) {
  auto rowType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});

  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"US", "UK", "CA"})})};

  ColumnHandleMap assignments;
  assignments["c0"] = makeIcebergHandle("c0", BIGINT(), 1);
  assignments["country"] = makeIcebergHandle("country", VARCHAR(), 2, "IN");

  // Expected: file values ("US", "UK", "CA"), not the default "IN".
  std::vector<RowVectorPtr> expectedVectors = {makeRowVector(
      rowType->names(),
      {dataVectors[0]->childAt(0), dataVectors[0]->childAt(1)})};

  assertDefaultValues(
      rowType, rowType, assignments, dataVectors, expectedVectors);
}

TEST_F(IcebergReadTest, addColumnWithDefaultAllTypes) {
  auto newRowType =
      ROW({"c0",
           "tiny_val",
           "small_val",
           "int_val",
           "big_val",
           "real_val",
           "double_val",
           "bool_val",
           "str_val",
           "short_decimal",
           "long_decimal",
           "date_val",
           "timestamp_val"},
          {BIGINT(),
           TINYINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           BOOLEAN(),
           VARCHAR(),
           DECIMAL(10, 2),
           DECIMAL(38, 10),
           DATE(),
           TIMESTAMP()});

  std::vector<RowVectorPtr> dataVectors = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};

  ColumnHandleMap assignments;
  assignments["c0"] = makeIcebergHandle("c0", BIGINT(), 1);
  assignments["tiny_val"] = makeIcebergHandle("tiny_val", TINYINT(), 2, "10");
  assignments["small_val"] =
      makeIcebergHandle("small_val", SMALLINT(), 3, "100");
  assignments["int_val"] = makeIcebergHandle("int_val", INTEGER(), 4, "1000");
  assignments["big_val"] = makeIcebergHandle("big_val", BIGINT(), 5, "10000");
  assignments["real_val"] = makeIcebergHandle("real_val", REAL(), 6, "3.14");
  assignments["double_val"] =
      makeIcebergHandle("double_val", DOUBLE(), 7, "2.718");
  assignments["bool_val"] = makeIcebergHandle("bool_val", BOOLEAN(), 8, "true");
  assignments["str_val"] =
      makeIcebergHandle("str_val", VARCHAR(), 9, "default_string");
  assignments["short_decimal"] =
      makeIcebergHandle("short_decimal", DECIMAL(10, 2), 10, "99.99");
  assignments["long_decimal"] = makeIcebergHandle(
      "long_decimal",
      DECIMAL(38, 10),
      11,
      "123456789012345678901234567.8901234567");
  assignments["date_val"] =
      makeIcebergHandle("date_val", DATE(), 12, "2024-01-15");
  assignments["timestamp_val"] = makeIcebergHandle(
      "timestamp_val", TIMESTAMP(), 13, "2024-01-15 10:30:00");

  std::vector<RowVectorPtr> expectedVectors = {makeRowVector(
      newRowType->names(),
      {dataVectors[0]->childAt(0),
       makeFlatVector<int8_t>({10, 10, 10}),
       makeFlatVector<int16_t>({100, 100, 100}),
       makeFlatVector<int32_t>({1000, 1000, 1000}),
       makeFlatVector<int64_t>({10000, 10000, 10000}),
       makeFlatVector<float>({3.14F, 3.14F, 3.14F}),
       makeFlatVector<double>({2.718, 2.718, 2.718}),
       makeFlatVector<bool>({true, true, true}),
       makeFlatVector<std::string>(
           {"default_string", "default_string", "default_string"}),
       makeFlatVector<int64_t>({9999, 9999, 9999}, DECIMAL(10, 2)),
       makeFlatVector<int128_t>(
           {HugeInt::parse("1234567890123456789012345678901234567"),
            HugeInt::parse("1234567890123456789012345678901234567"),
            HugeInt::parse("1234567890123456789012345678901234567")},
           DECIMAL(38, 10)),
       makeFlatVector<int32_t>({19737, 19737, 19737}, DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1705314600, 0),
            Timestamp(1705314600, 0),
            Timestamp(1705314600, 0)})})};

  assertDefaultValues(
      newRowType,
      newRowType,
      assignments,
      dataVectors,
      expectedVectors,
      {{HiveConfig::kReadTimestampPartitionValueAsLocalTimeSession, "false"}});
}

TEST_F(IcebergReadTest, addColumnWithInvalidDefault) {
  auto newRowType = ROW({"c0", "age"}, {BIGINT(), INTEGER()});

  std::vector<RowVectorPtr> dataVectors = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  ColumnHandleMap assignments;
  assignments["c0"] = makeIcebergHandle("c0", BIGINT(), 1);
  assignments["age"] = makeIcebergHandle("age", INTEGER(), 2, "IN");

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(
          exec::test::PlanBuilder()
              .startTableScan(test::kIcebergConnectorId)
              .outputType(newRowType)
              .dataColumns(newRowType)
              .assignments(assignments)
              .endTableScan()
              .planNode())
          .splits(makeIcebergSplits(dataFilePath->getPath()))
          .assertResults(std::vector<RowVectorPtr>{}),
      "Invalid");
}

TEST_F(IcebergReadTest, addColumnWithEmptyStringDefault) {
  auto newRowType = ROW({"c0", "name"}, {BIGINT(), VARCHAR()});
  auto dataVectors = makeSingleBigintData({1, 2, 3});
  auto assignments = makeAssignments(
      {{"c0", "c0", BIGINT(), 1}, {"name", "name", VARCHAR(), 2, ""}});
  auto expectedVectors = makeBigintAndVarcharExpected(
      newRowType->names(), dataVectors[0], {"", "", ""});

  assertDefaultValues(
      newRowType, newRowType, assignments, dataVectors, expectedVectors);
}

TEST_F(IcebergReadTest, defaultValueWithDeletesAndFilters) {
  auto newRowType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})})};
  auto dataFilePath = TempFilePath::create();
  // Write data file with old schema (only c0) containing rows 1-10.
  writeToFile(dataFilePath->getPath(), dataVectors);

  auto deleteFilePath = TempFilePath::create();
  // Create delete file that deletes positions 1, 3, 5 (rows 2, 4, 6).
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  writeToFile(
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  static_cast<vector_size_t>(3),
                  [&](vector_size_t) { return dataFilePath->getPath(); }),
              makeFlatVector<int64_t>({1, 3, 5}),
          })});
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      fileFormat_,
      3,
      this->getFileSize(deleteFilePath->getPath()));

  ColumnHandleMap assignments;
  assignments = makeAssignments(
      {{"c0", "c0", BIGINT(), 1}, {"country", "country", VARCHAR(), 2, "IN"}});

  const auto makeSplits = [&]() {
    return makeIcebergSplits(dataFilePath->getPath(), {deleteFile});
  };
  const auto makeExpected = [&](const std::vector<int64_t>& values) {
    return std::vector<RowVectorPtr>{makeRowVector(
        newRowType->names(),
        {makeFlatVector<int64_t>(values),
         makeFlatVector<std::string>(
             static_cast<vector_size_t>(values.size()),
             [](vector_size_t) { return "IN"; })})};
  };

  {
    // Test 1: No filter. After deletes, rows 1, 3, 5, 7, 8, 9, 10 remain.
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(makeSplits())
        .assertResults(makeExpected({1, 3, 5, 7, 8, 9, 10}));
  }
  {
    // Test 2: Filter on file column (c0 > 5) with deletes.
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .endTableScan()
                    .filter("c0 > 5")
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(makeSplits())
        .assertResults(makeExpected({7, 8, 9, 10}));
  }
  {
    // Test 3: Filter on default value column (country = 'IN') with deletes.
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .endTableScan()
                    .filter("country = 'IN'")
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(makeSplits())
        .assertResults(makeExpected({1, 3, 5, 7, 8, 9, 10}));
  }
  {
    // Test 4: Combined filter (c0 > 3 AND country = 'IN') with deletes.
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .endTableScan()
                    .filter("c0 > 3 AND country = 'IN'")
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(makeSplits())
        .assertResults(makeExpected({5, 7, 8, 9, 10}));
  }
}

// Test filter pushdown (remainingFilter) with initial-default columns.
// This test validates that when a filter is pushed down to the split reader,
// files with missing columns that have initial-defaults are correctly handled
// during checkIfSplitIsEmpty().
TEST_F(IcebergReadTest, filterPushdownWithInitialDefault) {
  auto newRowType =
      ROW({"c0", "country", "status"}, {BIGINT(), VARCHAR(), VARCHAR()});

  // Write data file with old schema (only c0) containing rows 1-5.
  std::vector<RowVectorPtr> dataVectors;
  dataVectors.push_back(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4, 5})}));
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  ColumnHandleMap assignments;
  assignments["c0"] = makeIcebergHandle("c0", BIGINT(), 1);
  assignments["country"] = makeIcebergHandle("country", VARCHAR(), 2, "IN");
  assignments["status"] = makeIcebergHandle("status", VARCHAR(), 3);

  // Test 1: Filter pushdown on initial-default column (matching value)
  // Without the fix, checkIfSplitIsEmpty() would incorrectly skip this file
  // because it treats missing 'country' column as NULL, and NULL != 'IN'.
  std::vector<RowVectorPtr> allRowsExpected;
  allRowsExpected.push_back(makeRowVector(
      newRowType->names(),
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<std::string>({"IN", "IN", "IN", "IN", "IN"}),
       makeNullableFlatVector<std::string>(
           {std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt})}));

  auto filteredExpected = std::vector<RowVectorPtr>{makeRowVector(
      newRowType->names(),
      {makeFlatVector<int64_t>({3, 4, 5}),
       makeFlatVector<std::string>({"IN", "IN", "IN"}),
       makeNullableFlatVector<std::string>(
           {std::nullopt, std::nullopt, std::nullopt})})};

  auto assertFilter = [&](const std::string& filter,
                          const std::vector<RowVectorPtr>& expected,
                          int32_t numSplitsSkipped = 0) {
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .remainingFilter(filter)
                    .endTableScan()
                    .planNode();
    auto task = exec::test::AssertQueryBuilder(plan)
                    .splits(makeIcebergSplits(dataFilePath->getPath()))
                    .assertResults(expected);
    ASSERT_EQ(
        task->taskStats()
            .pipelineStats[0]
            .operatorStats[0]
            .runtimeStats["skippedSplits"]
            .sum,
        numSplitsSkipped);
  };

  assertFilter("country = 'IN'", allRowsExpected);
  assertFilter("country IS NOT NULL", allRowsExpected);
  assertFilter("status IS NULL", allRowsExpected);
  assertFilter("status IS NOT NULL", {}, 1);
  assertFilter("c0 > 2 AND country = 'IN'", filteredExpected);
  assertFilter("country = 'US'", {}, 1);
}

// Test filter pushdown with non-VARCHAR initial-default columns (INTEGER,
// REAL). This validates that the type casting in testFilterOnConstantVector()
// works correctly for numeric types.
TEST_F(IcebergReadTest, filterPushdownWithNumericInitialDefaults) {
  auto newRowType = ROW({"c0", "age", "score"}, {BIGINT(), INTEGER(), REAL()});

  // Write data file with old schema (only c0) containing rows 1-5.
  std::vector<RowVectorPtr> dataVectors;
  dataVectors.push_back(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4, 5})}));
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  ColumnHandleMap assignments;
  assignments["c0"] = makeIcebergHandle("c0", BIGINT(), 1);
  assignments["age"] = makeIcebergHandle("age", INTEGER(), 2, "25");
  assignments["score"] = makeIcebergHandle("score", REAL(), 3, "3.14");

  std::vector<RowVectorPtr> allRowsExpected;
  allRowsExpected.push_back(makeRowVector(
      newRowType->names(),
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int32_t>({25, 25, 25, 25, 25}),
       makeFlatVector<float>({3.14f, 3.14f, 3.14f, 3.14f, 3.14f})}));

  auto assertFilter = [&](const std::string& filter,
                          const std::vector<RowVectorPtr>& expected,
                          int32_t numSplitsSkipped = 0) {
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(newRowType)
                    .dataColumns(newRowType)
                    .assignments(assignments)
                    .remainingFilter(filter)
                    .endTableScan()
                    .planNode();
    auto task = exec::test::AssertQueryBuilder(plan)
                    .splits(makeIcebergSplits(dataFilePath->getPath()))
                    .assertResults(expected);
    ASSERT_EQ(
        task->taskStats()
            .pipelineStats[0]
            .operatorStats[0]
            .runtimeStats["skippedSplits"]
            .sum,
        numSplitsSkipped);
  };

  assertFilter("age = cast(25 as INTEGER)", allRowsExpected);
  assertFilter("age = cast(30 as INTEGER)", {}, 1);
  assertFilter("score = cast(3.14 as REAL)", allRowsExpected);
  assertFilter("score > cast(5.0 as REAL)", {}, 1);
  assertFilter(
      "age = cast(25 as INTEGER) AND score = cast(3.14 as REAL)",
      allRowsExpected);
  assertFilter(
      "age = cast(25 as INTEGER) AND score > cast(5.0 as REAL)", {}, 1);
}

TEST_F(IcebergReadTest, partitionColumnsFromHive) {
  // Test reading partition columns from Hive-migrated tables.
  // This tests the adaptColumns method handling partition columns that are not
  // stored in the data file but provided via partitionKeys map.
  // This scenario occurs when reading Hive-written data files where partition
  // column values are stored in partition metadata rather than in the data
  // file.
  auto fileRowType = ROW({"c0", "c1"}, {BIGINT(), INTEGER()});
  auto tableRowType =
      ROW({"c0", "c1", "region", "year"},
          {BIGINT(), INTEGER(), VARCHAR(), INTEGER()});

  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int32_t>({10, 20, 30})})};
  auto dataFilePath = TempFilePath::create();
  // Write data file with only non-partition columns (c0, c1).
  writeToFile(dataFilePath->getPath(), dataVectors);

  // Set partition keys for region and year.
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"region", "US"},
      {"year", "2025"},
  };

  std::vector<RowVectorPtr> expectedVectors = {makeRowVector(
      tableRowType->names(),
      {
          dataVectors[0]->childAt(0),
          dataVectors[0]->childAt(1),
          makeFlatVector<std::string>({"US", "US", "US"}),
          makeFlatVector<int32_t>({2025, 2025, 2025}),
      })};

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(tableRowType)
                  .dataColumns(tableRowType)
                  .assignments(makeColumnHandles(tableRowType, {2, 3}))
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys))
      .assertResults(expectedVectors);
}

TEST_F(IcebergReadTest, rowLineage) {
  // Row lineage scenarios for _row_id and _last_updated_sequence_number:
  //   1. Pre-V3: no info columns, no physical columns → both null.
  //   2. V3 new insert: no physical columns; derived from info columns.
  //   3. V3 rewrite: physical values take precedence over info columns.
  //   4. Physical columns all null: falls back to info column derivation.
  //   5. Mixed null/non-null: null slots derived, non-null slots preserved.
  //   6. first_row_id = 0 is a valid value.
  //   7. Positional deletes: _row_id uses file-absolute positions.
  //   8. Subfield filter: _row_id uses file-absolute positions, not output
  //   indices.
  //   9. data_sequence_number without first_row_id: both _row_id and
  //      _last_updated_sequence_number are null.
  //  10. Physical lineage columns present with data_sequence_number but no
  //      first_row_id: both columns are null (spec requires null when
  //      first_row_id is absent, regardless of physical storage).
  static const std::vector<std::string> kOutputNames = {
      "c0", "_row_id", "_last_updated_sequence_number"};

  // 1. Pre-V3.
  assertRowLineage({
      .values = {1, 2, 3},
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });

  // 2. V3 new insert.
  assertRowLineage({
      .values = {10, 20, 30},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeFlatVector<int64_t>({100, 101, 102}),
              makeFlatVector<int64_t>({7, 7, 7}),
          })},
  });

  // 3. V3 rewrite: physical values must not be overridden by info columns.
  assertRowLineage({
      .values = {1, 2, 3},
      .storedRowIds = {{500, 501, 502}},
      .storedSequenceNumbers = {{3, 5, 3}},
      .firstRowId = 999,
      .dataSequenceNumber = 99,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeFlatVector<int64_t>({500, 501, 502}),
              makeFlatVector<int64_t>({3, 5, 3}),
          })},
  });

  // 4. Physical columns all null: falls back to info column derivation.
  assertRowLineage({
      .values = {1, 2, 3},
      .storedRowIds = {{std::nullopt, std::nullopt, std::nullopt}},
      .storedSequenceNumbers = {{std::nullopt, std::nullopt, std::nullopt}},
      .firstRowId = 50,
      .dataSequenceNumber = 42,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeFlatVector<int64_t>({50, 51, 52}),
              makeFlatVector<int64_t>({42, 42, 42}),
          })},
  });

  // 5. Mixed null/non-null: null slots derived from info columns, non-null
  // preserved.
  assertRowLineage({
      .values = {10, 20, 30, 40},
      .storedRowIds = {{std::nullopt, 99, std::nullopt, 77}},
      .storedSequenceNumbers = {{std::nullopt, 5, std::nullopt, 10}},
      .firstRowId = 10,
      .dataSequenceNumber = 42,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30, 40}),
              makeFlatVector<int64_t>({10, 99, 12, 77}),
              makeFlatVector<int64_t>({42, 5, 42, 10}),
          })},
  });

  // 6. first_row_id = 0 is a valid value; _row_id starts at zero.
  assertRowLineage({
      .values = {5, 6, 7},
      .firstRowId = 0,
      .dataSequenceNumber = 5,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({5, 6, 7}),
              makeFlatVector<int64_t>({0, 1, 2}),
              makeFlatVector<int64_t>({5, 5, 5}),
          })},
  });

  // 7. Positional deletes: _row_id uses file-absolute positions.
  assertRowLineage({
      .values = {10, 20, 30, 40, 50},
      .firstRowId = 200,
      .dataSequenceNumber = 42,
      .deletePositions = {1, 3},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({10, 30, 50}),
              makeFlatVector<int64_t>({200, 202, 204}),
              makeFlatVector<int64_t>({42, 42, 42}),
          })},
  });

  // 8. Subfield filter: _row_id uses file-absolute positions, not output
  // indices.
  assertRowLineage({
      .values = {10, 20, 30, 40, 50},
      .firstRowId = 100,
      .dataSequenceNumber = 15,
      .deletePositions = {},
      .subfieldFilter = "c0 > 20",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({30, 40, 50}),
              makeFlatVector<int64_t>({102, 103, 104}),
              makeFlatVector<int64_t>({15, 15, 15}),
          })},
  });

  // 9. data_sequence_number without first_row_id: _last_updated_sequence_number
  // must be null because _row_id is null (no first_row_id to anchor it).
  assertRowLineage({
      .values = {1, 2, 3},
      .dataSequenceNumber = 7,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });

  // 10. Physical lineage columns present, data_sequence_number set,
  // first_row_id absent. Per spec, first_row_id absent means null for both
  // _row_id and _last_updated_sequence_number regardless of what is
  // physically stored in the file.
  assertRowLineage({
      .values = {10, 20, 30},
      .storedRowIds = {{500, 501, 502}},
      .storedSequenceNumbers = {{3, 5, 3}},
      .dataSequenceNumber = 7,
      .deletePositions = {},
      .subfieldFilter = "",
      .expectedVectors = {makeRowVector(
          kOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });
}

// Tests Iceberg MERGE INTO row-id synthesis: the projection of the synthetic
// $target_table_row_id ROW column produced at read time from the split's
// infoColumns ($path, $spec_id, partition_data) plus the file row positions.
// Mirrors the IcebergPageSourceProvider Java path that backs
// MERGE_TARGET_ROW_ID_DATA.
TEST_F(IcebergReadTest, targetTableRowIdSynthesis) {
  static const std::string kPartitionDataJson =
      R"({"partitionValues":["2024-01-01"]})";

  std::vector<RowVectorPtr> inputVectors = {
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30})})};
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), inputVectors);

  const auto rowIdType =
      ROW({"file_path", "row_position", "spec_id", "partition_data"},
          {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()});
  const auto outputType =
      ROW({"c0", IcebergMetadataColumn::kTargetTableRowIdColumnName},
          {BIGINT(), rowIdType});

  auto expected = makeRowVector(
      {"c0", IcebergMetadataColumn::kTargetTableRowIdColumnName},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
          makeRowVector(
              {"file_path", "row_position", "spec_id", "partition_data"},
              {
                  makeFlatVector<std::string>(
                      static_cast<vector_size_t>(3),
                      [&](vector_size_t) { return dataFilePath->getPath(); }),
                  makeFlatVector<int64_t>({0, 1, 2}),
                  makeFlatVector<int32_t>({7, 7, 7}),
                  makeFlatVector<std::string>(
                      static_cast<vector_size_t>(3),
                      [&](vector_size_t) { return kPartitionDataJson; }),
              }),
      });

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits({makeIcebergSplitWithInfoColumns(
          dataFilePath->getPath(),
          {
              {IcebergMetadataColumn::kSpecIdInfoColumn, "7"},
              {IcebergMetadataColumn::kPartitionDataInfoColumn,
               kPartitionDataJson},
          })})
      .assertResults({expected});
}

TEST_F(IcebergReadTest, flatMapAsStruct) {
  // Write a DWRF file with a MAP<BIGINT, DOUBLE> column.
  auto mapType = MAP(BIGINT(), DOUBLE());
  auto dataSchema = ROW({"id", "features"}, {BIGINT(), mapType});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector(
          {"id", "features"},
          {makeFlatVector<int64_t>({1, 2}),
           makeMapVector(
               {0, 3},
               makeFlatVector<int64_t>({1, 2, 3, 1, 2, 3}),
               makeFlatVector<double>(
                   {10.0, 20.0, 30.0, 100.0, 200.0, 300.0}))})});

  // Build struct-encoded column handle for "features": keys {1, 2} as
  // struct fields {"1", "2"}.
  auto structType = ROW({"1", "2"}, {DOUBLE(), DOUBLE()});
  ColumnHandleMap assignments;
  assignments["id"] = std::shared_ptr<HiveColumnHandle>(
      exec::test::HiveConnectorTestBase::makeColumnHandle(
          "id", BIGINT(), std::vector<std::string>{})
          .release());
  assignments["features"] = std::shared_ptr<HiveColumnHandle>(
      exec::test::HiveConnectorTestBase::makeColumnHandle(
          "features",
          mapType,
          mapType,
          std::vector<std::string>{"features.1", "features.2"})
          .release());

  auto expected = makeRowVector(
      {"id", "features"},
      {makeFlatVector<int64_t>({1, 2}),
       makeRowVector(
           {"1", "2"},
           {makeFlatVector<double>({10.0, 100.0}),
            makeFlatVector<double>({20.0, 200.0})})});

  // Output type has ROW for the struct-encoded column.
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"id", "features"}, {BIGINT(), structType}))
                  .dataColumns(dataSchema)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({expected});
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
