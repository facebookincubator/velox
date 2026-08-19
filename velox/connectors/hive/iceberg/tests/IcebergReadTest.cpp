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
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

using TempFilePath = common::testutil::TempFilePath;

class IcebergReadTest : public test::IcebergTestBase {
 protected:
  static inline const std::vector<std::string> kRowLineageOutputNames{
      "c0",
      "_row_id",
      "_last_updated_sequence_number"};

  struct AssignmentSpec {
    std::string outputName;
    std::string sourceName;
    TypePtr type;
    int fieldId;
    std::optional<std::string> defaultValue = std::nullopt;
  };

  struct RowLineageTestCase {
    // Names the case in the failure message.
    std::string name{};
    std::vector<int64_t> values{};
    std::optional<std::vector<std::optional<int64_t>>> storedRowIds =
        std::nullopt;
    std::optional<std::vector<std::optional<int64_t>>> storedSequenceNumbers =
        std::nullopt;
    std::optional<int64_t> firstRowId = std::nullopt;
    std::optional<int64_t> dataSequenceNumber = std::nullopt;
    std::vector<int64_t> deletePositions{};
    std::string subfieldFilter{};
    // Filter keyed by column name. 'subfieldFilter' cannot name the lineage
    // columns, which are not in 'tableDataColumns'. A pair rather than a
    // SubfieldFilters because common::Subfield is move-only.
    std::optional<std::pair<std::string, common::FilterPtr>> directFilter =
        std::nullopt;
    // Values an equality-delete file on 'c0' deletes. No file when empty.
    std::vector<int64_t> equalityDeleteValues{};
    // Empty means the query is expected to return no rows.
    std::vector<RowVectorPtr> expectedVectors{};
  };

  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    fileFormat_ = dwio::common::FileFormat::DWRF;
  }

  std::shared_ptr<IcebergColumnHandle> makeIcebergHandle(
      const std::string& name,
      const TypePtr& type,
      parquet::ParquetFieldId fieldId,
      std::optional<std::string> defaultValue = std::nullopt,
      std::vector<common::Subfield> requiredSubfields = {}) {
    return std::make_shared<IcebergColumnHandle>(
        name,
        HiveColumnHandle::ColumnType::kRegular,
        type,
        std::move(fieldId),
        std::move(requiredSubfields),
        defaultValue);
  }

  std::shared_ptr<IcebergColumnHandle> makeIcebergHandle(
      const std::string& name,
      const TypePtr& type,
      int fieldId,
      std::optional<std::string> defaultValue = std::nullopt) {
    return makeIcebergHandle(
        name, type, parquet::ParquetFieldId{fieldId, {}}, defaultValue);
  }

#ifdef VELOX_ENABLE_PARQUET
  parquet::ParquetFieldId makeFieldId(int32_t fieldId) {
    return parquet::ParquetFieldId{fieldId, {}};
  }

  parquet::ParquetFieldId makeFieldId(
      int32_t fieldId,
      std::vector<parquet::ParquetFieldId> children) {
    return parquet::ParquetFieldId{fieldId, std::move(children)};
  }

  struct FieldIdColumnSpec {
    std::string outputName;
    std::string dataName;
    TypePtr type;
    parquet::ParquetFieldId fieldId;
    std::vector<std::string> requiredSubfields{};
  };

  ColumnHandleMap makeFieldIdAssignments(
      const std::vector<FieldIdColumnSpec>& columns) {
    ColumnHandleMap assignments;
    for (const auto& column : columns) {
      std::vector<common::Subfield> requiredSubfields;
      requiredSubfields.reserve(column.requiredSubfields.size());
      for (const auto& subfield : column.requiredSubfields) {
        requiredSubfields.emplace_back(subfield);
      }
      assignments[column.outputName] = makeIcebergHandle(
          column.dataName,
          column.type,
          column.fieldId,
          std::nullopt,
          std::move(requiredSubfields));
    }
    return assignments;
  }

  void assertParquetFieldIdRead(
      const std::string& outputDirectory,
      const RowTypePtr& outputType,
      const RowTypePtr& scanSpecType,
      const std::vector<FieldIdColumnSpec>& columns,
      const std::vector<RowVectorPtr>& expected,
      const std::optional<std::string>& subfieldFilter = std::nullopt) {
    exec::test::PlanBuilder planBuilder;
    auto& tableScanBuilder = planBuilder.startTableScan()
                                 .connectorId(test::kIcebergConnectorId)
                                 .outputType(outputType)
                                 .dataColumns(scanSpecType)
                                 .assignments(makeFieldIdAssignments(columns));
    if (subfieldFilter.has_value()) {
      tableScanBuilder.subfieldFilter(*subfieldFilter);
    }
    auto plan = tableScanBuilder.endTableScan().planNode();

    exec::test::AssertQueryBuilder(plan)
        .splits(createSplitsForDirectory(outputDirectory))
        .assertResults(expected);
  }

  std::shared_ptr<test::TempDirectoryPath> writeParquetData(
      const std::vector<RowVectorPtr>& data) {
    fileFormat_ = dwio::common::FileFormat::PARQUET;
    auto outputDirectory = test::TempDirectoryPath::create();
    const auto dataSink =
        createDataSinkAndAppendData(data, outputDirectory->getPath());
    dataSink->close();
    return outputDirectory;
  }

  struct FlatParquetFieldIdData {
    RowTypePtr writeType;
    std::shared_ptr<test::TempDirectoryPath> outputDirectory;
  };

  FlatParquetFieldIdData writeFlatParquetFieldIdData() {
    const auto writeType =
        ROW({"id", "flag", "status"}, {BIGINT(), BOOLEAN(), VARCHAR()});
    const std::vector<RowVectorPtr> data{makeRowVector(
        writeType->names(),
        {makeFlatVector<int64_t>({10, 20, 30}),
         makeFlatVector<bool>({true, false, true}),
         makeFlatVector<std::string>({"old-a", "old-b", "old-c"})})};
    return {writeType, writeParquetData(data)};
  }
#endif

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

  // Writes a positional delete file listing 'positions' of 'dataFilePath'. The
  // returned temp file handle must outlive the read.
  std::pair<std::shared_ptr<TempFilePath>, IcebergDeleteFile>
  makePositionalDeleteFile(
      const std::string& dataFilePath,
      const std::vector<int64_t>& positions) {
    const auto pathColumn =
        IcebergMetadataColumn::icebergDeleteFilePathColumn();
    const auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
    auto deleteFilePath = TempFilePath::create();
    writeToFile(
        deleteFilePath->getPath(),
        {makeRowVector(
            {pathColumn->name, posColumn->name},
            {
                makeFlatVector<std::string>(
                    static_cast<vector_size_t>(positions.size()),
                    [&](vector_size_t) { return dataFilePath; }),
                makeFlatVector<int64_t>(positions),
            })});

    const auto upperBound = folly::Endian::little(
        static_cast<uint64_t>(
            *std::max_element(positions.begin(), positions.end())));
    std::unordered_map<int32_t, std::string> upperBounds;
    upperBounds[posColumn->id] = encoding::Base64::encode(
        std::string_view(
            reinterpret_cast<const char*>(&upperBound), sizeof(upperBound)));

    return {
        deleteFilePath,
        IcebergDeleteFile(
            FileContent::kPositionalDeletes,
            deleteFilePath->getPath(),
            fileFormat_,
            static_cast<int64_t>(positions.size()),
            this->getFileSize(deleteFilePath->getPath()),
            {},
            {},
            upperBounds,
            0)};
  }

  void assertRowLineage(const RowLineageTestCase& tc) {
    SCOPED_TRACE(tc.name);
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
      auto [path, deleteFile] =
          makePositionalDeleteFile(dataFilePath->getPath(), tc.deletePositions);
      deleteFilePath = std::move(path);
      deleteFiles.push_back(std::move(deleteFile));
    }

    std::shared_ptr<TempFilePath> equalityDeleteFilePath;
    if (!tc.equalityDeleteValues.empty()) {
      equalityDeleteFilePath = TempFilePath::create();
      writeToFile(
          equalityDeleteFilePath->getPath(),
          {makeRowVector(
              {"c0"}, {makeFlatVector<int64_t>(tc.equalityDeleteValues)})});
      deleteFiles.push_back(IcebergDeleteFile(
          FileContent::kEqualityDeletes,
          equalityDeleteFilePath->getPath(),
          fileFormat_,
          static_cast<int64_t>(tc.equalityDeleteValues.size()),
          this->getFileSize(equalityDeleteFilePath->getPath()),
          // Field ID 1 is 'c0', the first column of the table schema.
          /*equalityFieldIds=*/{1}));
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
    if (tc.directFilter.has_value()) {
      common::SubfieldFilters directFilters;
      directFilters.emplace(
          common::Subfield(tc.directFilter->first), tc.directFilter->second);
      tableScanBuilder.subfieldFiltersMap(directFilters);
    }
    auto plan = tableScanBuilder.endTableScan().planNode();
    exec::test::AssertQueryBuilder queryBuilder(plan);
    queryBuilder.splits({makeIcebergSplitWithInfoColumns(
        dataFilePath->getPath(), infoColumns, deleteFiles)});
    if (tc.expectedVectors.empty()) {
      queryBuilder.assertEmptyResults();
    } else {
      queryBuilder.assertResults(tc.expectedVectors);
    }
  }

  // Three full output batches.
  static constexpr vector_size_t kMultiBatchNumRows = 300;
  static constexpr vector_size_t kMultiBatchRowsPerBatch = 100;
  static constexpr int64_t kMultiBatchFirstRowId = 1'000;
  static constexpr int64_t kMultiBatchSequenceNumber = 7;

  // Reads one split with a range filter on '_row_id' keeping the file rows
  // 'firstSelected' through 'lastSelected', and asserts the output holds
  // exactly those. A 'lastSelected' past the last row of the file keeps fewer,
  // down to none at all. Every row read counts as raw input, including the ones
  // dropped after the reader returned them.
  void assertRowLineageFilterOverBatches(
      vector_size_t firstSelected,
      vector_size_t lastSelected) {
    SCOPED_TRACE(
        fmt::format(
            "selected file rows [{}, {}]", firstSelected, lastSelected));
    auto dataFilePath = TempFilePath::create();
    writeToFile(
        dataFilePath->getPath(),
        {makeRowVector({makeFlatVector<int64_t>(
            kMultiBatchNumRows, [](vector_size_t row) { return row; })})});
    const std::unordered_map<std::string, std::string> infoColumns{
        {IcebergMetadataColumn::kFirstRowIdInfoColumn,
         std::to_string(kMultiBatchFirstRowId)},
        {IcebergMetadataColumn::kDataSequenceNumberInfoColumn,
         std::to_string(kMultiBatchSequenceNumber)}};

    common::SubfieldFilters directFilters;
    directFilters.emplace(
        common::Subfield(IcebergMetadataColumn::kRowIdColumnName),
        std::make_shared<common::BigintRange>(
            kMultiBatchFirstRowId + firstSelected,
            kMultiBatchFirstRowId + lastSelected,
            false));
    core::PlanNodeId scanId;
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(
                        ROW(std::vector<std::string>(kRowLineageOutputNames),
                            {BIGINT(), BIGINT(), BIGINT()}))
                    .dataColumns(ROW({"c0"}, {BIGINT()}))
                    .subfieldFiltersMap(directFilters)
                    .endTableScan()
                    .capturePlanNodeId(scanId)
                    .planNode();

    const vector_size_t numSelected = std::max<vector_size_t>(
        0, std::min(lastSelected, kMultiBatchNumRows - 1) - firstSelected + 1);
    exec::test::AssertQueryBuilder queryBuilder(plan);
    queryBuilder.maxDrivers(1)
        .config(
            core::QueryConfig::kPreferredOutputBatchRows,
            std::to_string(kMultiBatchRowsPerBatch))
        .config(
            core::QueryConfig::kMaxOutputBatchRows,
            std::to_string(kMultiBatchRowsPerBatch))
        .splits({makeIcebergSplitWithInfoColumns(
            dataFilePath->getPath(), infoColumns)});
    auto task = numSelected == 0
        ? queryBuilder.assertEmptyResults()
        : queryBuilder.assertResults({makeRowVector(
              kRowLineageOutputNames,
              {
                  makeFlatVector<int64_t>(
                      numSelected,
                      [&](vector_size_t row) { return firstSelected + row; }),
                  makeFlatVector<int64_t>(
                      numSelected,
                      [&](vector_size_t row) {
                        return kMultiBatchFirstRowId + firstSelected + row;
                      }),
                  makeFlatVector<int64_t>(
                      numSelected,
                      [](vector_size_t /*row*/) {
                        return kMultiBatchSequenceNumber;
                      }),
              })});

    const auto planStats = exec::toPlanStats(task->taskStats());
    EXPECT_EQ(planStats.at(scanId).rawInputRows, kMultiBatchNumRows);
    EXPECT_EQ(planStats.at(scanId).outputRows, numSelected);
  }

  // Reads two splits of one data source with 'rowIdFilter' on '_row_id' and
  // asserts the output is 'expected'. The second split gets a first_row_id only
  // when 'secondFirstRowId' is set. Runs with preloading off, where the splits
  // share one scan spec, and on, where each gets its own.
  void assertRowLineageAcrossSplits(
      const common::FilterPtr& rowIdFilter,
      std::optional<int64_t> secondFirstRowId,
      const RowVectorPtr& expected) {
    auto firstFile = TempFilePath::create();
    writeToFile(
        firstFile->getPath(),
        {makeRowVector({makeFlatVector<int64_t>({10, 20, 30})})});
    const std::unordered_map<std::string, std::string> firstInfoColumns{
        {IcebergMetadataColumn::kFirstRowIdInfoColumn, "100"},
        {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}};

    auto secondFile = TempFilePath::create();
    writeToFile(
        secondFile->getPath(),
        {makeRowVector({makeFlatVector<int64_t>({40, 50, 60})})});
    std::unordered_map<std::string, std::string> secondInfoColumns{
        {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "9"}};
    if (secondFirstRowId.has_value()) {
      secondInfoColumns[IcebergMetadataColumn::kFirstRowIdInfoColumn] =
          std::to_string(*secondFirstRowId);
    }

    common::SubfieldFilters directFilters;
    directFilters.emplace(
        common::Subfield(IcebergMetadataColumn::kRowIdColumnName), rowIdFilter);
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(
                        ROW(std::vector<std::string>(kRowLineageOutputNames),
                            {BIGINT(), BIGINT(), BIGINT()}))
                    .dataColumns(ROW({"c0"}, {BIGINT()}))
                    .subfieldFiltersMap(directFilters)
                    .endTableScan()
                    .planNode();

    for (const auto* maxSplitPreload : {"0", "2"}) {
      SCOPED_TRACE(fmt::format("maxSplitPreload: {}", maxSplitPreload));
      exec::test::AssertQueryBuilder(plan)
          .maxDrivers(1)
          .config(core::QueryConfig::kMaxSplitPreloadPerDriver, maxSplitPreload)
          .splits(
              {makeIcebergSplitWithInfoColumns(
                   firstFile->getPath(), firstInfoColumns),
               makeIcebergSplitWithInfoColumns(
                   secondFile->getPath(), secondInfoColumns)})
          .assertResults({expected});
    }
  }

  // Spec id and partition data the split carries for the MERGE INTO composite.
  static constexpr int32_t kTargetTableSpecId = 7;
  static inline const std::string kTargetTablePartitionData =
      R"({"partitionValues":["2024-01-01"]})";

  // Reads one split of 'values' with '$target_table_row_id' projected next to
  // 'c0', 'deletePositions' deleted by position and 'filter' on the composite,
  // and asserts the surviving rows are 'expectedValues' carrying
  // 'expectedPositions' as their file row positions. 'filterField' names the
  // field of the composite the filter applies to; empty targets the composite
  // itself, which can only take a null filter.
  void assertTargetTableRowId(
      const std::vector<int64_t>& values,
      const std::vector<int64_t>& deletePositions,
      const std::vector<int64_t>& expectedValues,
      const std::vector<int64_t>& expectedPositions,
      const common::FilterPtr& filter,
      const std::string& filterField) {
    SCOPED_TRACE(
        fmt::format(
            "{} deleted positions, filter: {} on {}",
            deletePositions.size(),
            filter == nullptr ? "none" : filter->toString(),
            filterField.empty() ? "the composite" : filterField));
    VELOX_CHECK_EQ(expectedValues.size(), expectedPositions.size());

    auto dataFilePath = TempFilePath::create();
    writeToFile(
        dataFilePath->getPath(),
        {makeRowVector({"c0"}, {makeFlatVector<int64_t>(values)})});

    std::vector<IcebergDeleteFile> deleteFiles;
    std::shared_ptr<TempFilePath> deleteFilePath;
    if (!deletePositions.empty()) {
      auto [path, deleteFile] =
          makePositionalDeleteFile(dataFilePath->getPath(), deletePositions);
      deleteFilePath = std::move(path);
      deleteFiles.push_back(std::move(deleteFile));
    }

    const auto rowIdType =
        ROW({"file_path", "row_position", "spec_id", "partition_data"},
            {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()});
    exec::test::PlanBuilder planBuilder;
    auto& tableScanBuilder =
        planBuilder.startTableScan(test::kIcebergConnectorId)
            .outputType(
                ROW({"c0", IcebergMetadataColumn::kTargetTableRowIdColumnName},
                    {BIGINT(), rowIdType}))
            .dataColumns(ROW({"c0"}, {BIGINT()}));
    if (filter != nullptr) {
      // Built from path elements rather than from a string: the composite's
      // name starts with '$', which the subfield parser does not accept.
      std::vector<std::unique_ptr<common::Subfield::PathElement>> path;
      path.push_back(
          std::make_unique<common::Subfield::NestedField>(
              IcebergMetadataColumn::kTargetTableRowIdColumnName));
      if (!filterField.empty()) {
        path.push_back(
            std::make_unique<common::Subfield::NestedField>(filterField));
      }
      common::SubfieldFilters filters;
      filters.emplace(common::Subfield(std::move(path)), filter);
      tableScanBuilder.subfieldFiltersMap(filters);
    }

    const auto numExpected = static_cast<vector_size_t>(expectedValues.size());
    auto expected = makeRowVector(
        {"c0", IcebergMetadataColumn::kTargetTableRowIdColumnName},
        {
            makeFlatVector<int64_t>(expectedValues),
            makeRowVector(
                rowIdType->names(),
                {
                    makeFlatVector<std::string>(
                        numExpected,
                        [&](vector_size_t) { return dataFilePath->getPath(); }),
                    makeFlatVector<int64_t>(expectedPositions),
                    makeFlatVector<int32_t>(
                        numExpected,
                        [](vector_size_t) { return kTargetTableSpecId; }),
                    makeFlatVector<std::string>(
                        numExpected,
                        [](vector_size_t) {
                          return kTargetTablePartitionData;
                        }),
                }),
        });

    exec::test::AssertQueryBuilder(tableScanBuilder.endTableScan().planNode())
        .splits({makeIcebergSplitWithInfoColumns(
            dataFilePath->getPath(),
            {{IcebergMetadataColumn::kSpecIdInfoColumn,
              std::to_string(kTargetTableSpecId)},
             {IcebergMetadataColumn::kPartitionDataInfoColumn,
              kTargetTablePartitionData}},
            deleteFiles)})
        .assertResults({expected});
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

#ifdef VELOX_ENABLE_PARQUET
TEST_F(IcebergReadTest, readParquetFlatSchemaEvolutionByFieldId) {
  const auto testData = writeFlatParquetFieldIdData();
  struct ReadCase {
    std::string name;
    RowTypePtr outputType;
    RowTypePtr scanSpecType;
    std::vector<FieldIdColumnSpec> columns;
    RowVectorPtr expected;
  };

  const std::vector<ReadCase> cases{
      {
          "rename and reorder columns",
          ROW({"enabled", "id"}, {BOOLEAN(), BIGINT()}),
          ROW({"id", "flag", "status"}, {BIGINT(), BOOLEAN(), VARCHAR()}),
          {
              {"id", "id", BIGINT(), makeFieldId(1), {}},
              {"enabled", "flag", BOOLEAN(), makeFieldId(2), {}},
              {"status", "status", VARCHAR(), makeFieldId(3), {}},
          },
          makeRowVector(
              {"enabled", "id"},
              {makeFlatVector<bool>({true, false, true}),
               makeFlatVector<int64_t>({10, 20, 30})}),
      },
      {
          "add column",
          ROW({"id", "flag", "status", "score"},
              {BIGINT(), BOOLEAN(), VARCHAR(), INTEGER()}),
          ROW({"id", "flag", "status", "score"},
              {BIGINT(), BOOLEAN(), VARCHAR(), INTEGER()}),
          {
              {"id", "id", BIGINT(), makeFieldId(1), {}},
              {"flag", "flag", BOOLEAN(), makeFieldId(2), {}},
              {"status", "status", VARCHAR(), makeFieldId(3), {}},
              {"score", "score", INTEGER(), makeFieldId(4), {}},
          },
          makeRowVector(
              {"id", "flag", "status", "score"},
              {makeFlatVector<int64_t>({10, 20, 30}),
               makeFlatVector<bool>({true, false, true}),
               makeFlatVector<std::string>({"old-a", "old-b", "old-c"}),
               makeNullableFlatVector<int32_t>(
                   {std::nullopt, std::nullopt, std::nullopt})}),
      },
      {
          "delete column",
          ROW({"id", "status"}, {BIGINT(), VARCHAR()}),
          ROW({"id", "status"}, {BIGINT(), VARCHAR()}),
          {
              {"id", "id", BIGINT(), makeFieldId(1), {}},
              {"status", "status", VARCHAR(), makeFieldId(3), {}},
          },
          makeRowVector(
              {"id", "status"},
              {makeFlatVector<int64_t>({10, 20, 30}),
               makeFlatVector<std::string>({"old-a", "old-b", "old-c"})}),
      },
      {
          "project only added column",
          ROW({"score"}, {INTEGER()}),
          ROW({"score"}, {INTEGER()}),
          {
              {"score", "score", INTEGER(), makeFieldId(4), {}},
          },
          makeRowVector(
              {"score"},
              {makeNullableFlatVector<int32_t>(
                  {std::nullopt, std::nullopt, std::nullopt})}),
      },
      {
          "delete and add back column with same name but different type",
          ROW({"id", "status"}, {BIGINT(), BOOLEAN()}),
          ROW({"id", "status"}, {BIGINT(), BOOLEAN()}),
          {
              {"id", "id", BIGINT(), makeFieldId(1), {}},
              {"status", "status", BOOLEAN(), makeFieldId(4), {}},
          },
          makeRowVector(
              {"id", "status"},
              {makeFlatVector<int64_t>({10, 20, 30}),
               makeNullableFlatVector<bool>(
                   {std::nullopt, std::nullopt, std::nullopt})}),
      },
  };

  for (const auto& readCase : cases) {
    SCOPED_TRACE(readCase.name);
    assertParquetFieldIdRead(
        testData.outputDirectory->getPath(),
        readCase.outputType,
        readCase.scanSpecType,
        readCase.columns,
        {readCase.expected});
  }
}

TEST_F(IcebergReadTest, readParquetFilterOnlyColumnByFieldId) {
  const auto testData = writeFlatParquetFieldIdData();

  auto filterOnlyColumnPlan =
      exec::test::PlanBuilder()
          .startTableScan(test::kIcebergConnectorId)
          .outputType(ROW({"id"}, {BIGINT()}))
          .dataColumns(testData.writeType)
          .assignments(makeFieldIdAssignments({
              {"id", "id", BIGINT(), makeFieldId(1), {}},
          }))
          .filterColumnHandles({
              makeIcebergHandle("status", VARCHAR(), makeFieldId(3)),
          })
          .remainingFilter("status = 'old-b'")
          .endTableScan()
          .planNode();
  exec::test::AssertQueryBuilder(filterOnlyColumnPlan)
      .splits(createSplitsForDirectory(testData.outputDirectory->getPath()))
      .assertResults({makeRowVector({makeFlatVector<int64_t>({20})})});
}

TEST_F(IcebergReadTest, readParquetNestedStructByFieldId) {
  const auto addressWriteType = ROW({"city", "zip"}, {VARCHAR(), VARCHAR()});
  const auto profileWriteType =
      ROW({"name", "address"}, {VARCHAR(), addressWriteType});
  const auto profileWriteData = makeRowVector(
      profileWriteType->names(),
      {makeFlatVector<std::string>({"Ada", "Ben", "Cy"}),
       makeRowVector(
           addressWriteType->names(),
           {makeFlatVector<std::string>({"New York", "Boston", "New York"}),
            makeFlatVector<std::string>({"10001", "02108", "10001"})})});
  const auto profileTableType = ROW({"profile"}, {profileWriteType});
  const std::vector<RowVectorPtr> profileData{
      makeRowVector(profileTableType->names(), {profileWriteData})};
  const auto profileOutputDirectory = writeParquetData(profileData);
  common::SubfieldFilters profileFilters;
  profileFilters.emplace(
      common::Subfield("profile.address.zip"),
      std::make_shared<common::BytesRange>(
          "10001", false, false, "10001", false, false, false));
  exec::test::PlanBuilder profilePlanBuilder;
  auto& profileTableScanBuilder =
      profilePlanBuilder.startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(profileTableType)
          .dataColumns(profileTableType)
          .assignments(makeFieldIdAssignments({
              {"profile",
               "profile",
               profileWriteType,
               makeFieldId(
                   1,
                   {makeFieldId(2),
                    makeFieldId(3, {makeFieldId(4), makeFieldId(5)})}),
               {"profile.name", "profile.address.city"}},
          }))
          .subfieldFiltersMap(profileFilters);
  auto profilePlan = profileTableScanBuilder.endTableScan()
                         .project({"profile.name as name", "profile.address"})
                         .project({"name", "address.city"})
                         .planNode();
  exec::test::AssertQueryBuilder(profilePlan)
      .splits(createSplitsForDirectory(profileOutputDirectory->getPath()))
      .assertResults({makeRowVector(
          {makeFlatVector<std::string>({"Ada", "Cy"}),
           makeFlatVector<std::string>({"New York", "New York"})})});
}

TEST_F(IcebergReadTest, readParquetArrayByFieldId) {
  const auto nestedWriteType =
      ROW({"items"}, {ARRAY(ROW({"name", "quantity"}, {VARCHAR(), BIGINT()}))});
  const auto nestedElements = makeRowVector(
      {"name", "quantity"},
      {makeFlatVector<std::string>({"apple", "banana", "pear"}),
       makeFlatVector<int64_t>({5, 7, 11})});
  const std::vector<RowVectorPtr> nestedData{makeRowVector(
      nestedWriteType->names(), {makeArrayVector({0, 2}, nestedElements)})};
  const auto nestedOutputDirectory = writeParquetData(nestedData);

  const auto nestedReadType =
      ROW({"items"}, {ARRAY(ROW({"amount", "label"}, {BIGINT(), VARCHAR()}))});
  const auto nestedExpectedElements = makeRowVector(
      {"amount", "label"},
      {makeFlatVector<int64_t>({5, 7, 11}),
       makeFlatVector<std::string>({"apple", "banana", "pear"})});
  auto nestedExpected = makeRowVector(
      nestedReadType->names(),
      {makeArrayVector({0, 2}, nestedExpectedElements)});

  {
    SCOPED_TRACE("full array element row");
    assertParquetFieldIdRead(
        nestedOutputDirectory->getPath(),
        nestedReadType,
        nestedReadType,
        {
            {"items",
             "items",
             nestedReadType->childAt(0),
             makeFieldId(1, {makeFieldId(2, {makeFieldId(4), makeFieldId(3)})}),
             {}},
        },
        {nestedExpected});
  }

  const auto nestedProjectedReadType =
      ROW({"items"}, {ARRAY(ROW({"amount"}, {BIGINT()}))});
  const auto nestedProjectedElements =
      makeRowVector({"amount"}, {makeFlatVector<int64_t>({5, 7, 11})});
  auto nestedProjectedExpected = makeRowVector(
      nestedProjectedReadType->names(),
      {makeArrayVector({0, 2}, nestedProjectedElements)});
  {
    SCOPED_TRACE("projected array element row");
    assertParquetFieldIdRead(
        nestedOutputDirectory->getPath(),
        nestedProjectedReadType,
        nestedProjectedReadType,
        {
            {"items",
             "items",
             nestedProjectedReadType->childAt(0),
             makeFieldId(1, {makeFieldId(2, {makeFieldId(4)})}),
             {}},
        },
        {nestedProjectedExpected});
  }
}

TEST_F(IcebergReadTest, readParquetMapByFieldId) {
  const auto mapValueWriteType = ROW({"name", "score"}, {VARCHAR(), BIGINT()});
  const auto mapWriteType =
      ROW({"attributes"}, {MAP(VARCHAR(), mapValueWriteType)});
  const auto mapValueElements = makeRowVector(
      mapValueWriteType->names(),
      {makeFlatVector<std::string>({"silver", "gold"}),
       makeFlatVector<int64_t>({11, 17})});
  const std::vector<RowVectorPtr> mapData{makeRowVector(
      mapWriteType->names(),
      {makeMapVector(
          {0, 1},
          makeFlatVector<std::string>({"left", "right"}),
          mapValueElements)})};
  const auto mapOutputDirectory = writeParquetData(mapData);

  const auto mapValueReadType = ROW({"points", "label"}, {BIGINT(), VARCHAR()});
  const auto mapReadType =
      ROW({"attributes"}, {MAP(VARCHAR(), mapValueReadType)});
  const auto mapExpectedValues = makeRowVector(
      mapValueReadType->names(),
      {makeFlatVector<int64_t>({11, 17}),
       makeFlatVector<std::string>({"silver", "gold"})});
  auto mapExpected = makeRowVector(
      mapReadType->names(),
      {makeMapVector(
          {0, 1},
          makeFlatVector<std::string>({"left", "right"}),
          mapExpectedValues)});

  {
    SCOPED_TRACE("full map value row");
    assertParquetFieldIdRead(
        mapOutputDirectory->getPath(),
        mapReadType,
        mapReadType,
        {
            {"attributes",
             "attributes",
             mapReadType->childAt(0),
             makeFieldId(
                 1,
                 {makeFieldId(2),
                  makeFieldId(3, {makeFieldId(5), makeFieldId(4)})}),
             {}},
        },
        {mapExpected});
  }

  const auto mapValueProjectedReadType = ROW({"points"}, {BIGINT()});
  const auto mapProjectedReadType =
      ROW({"attributes"}, {MAP(VARCHAR(), mapValueProjectedReadType)});
  const auto mapProjectedExpectedValues =
      makeRowVector({"points"}, {makeFlatVector<int64_t>({11, 17})});
  auto mapProjectedExpected = makeRowVector(
      mapProjectedReadType->names(),
      {makeMapVector(
          {0, 1},
          makeFlatVector<std::string>({"left", "right"}),
          mapProjectedExpectedValues)});
  {
    SCOPED_TRACE("projected map value row");
    assertParquetFieldIdRead(
        mapOutputDirectory->getPath(),
        mapProjectedReadType,
        mapProjectedReadType,
        {
            {"attributes",
             "attributes",
             mapProjectedReadType->childAt(0),
             makeFieldId(1, {makeFieldId(2), makeFieldId(3, {makeFieldId(5)})}),
             {}},
        },
        {mapProjectedExpected});
  }
}
#endif

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

// Synthesis of _row_id and _last_updated_sequence_number from the split's info
// columns and from the values stored in the file. Filters on either column are
// covered by rowLineageWithFilter.
TEST_F(IcebergReadTest, rowLineage) {
  assertRowLineage({
      .name = "pre-V3: no info columns and no stored columns",
      .values = {1, 2, 3},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });

  assertRowLineage({
      .name = "V3 insert: derived from the info columns",
      .values = {10, 20, 30},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeFlatVector<int64_t>({100, 101, 102}),
              makeFlatVector<int64_t>({7, 7, 7}),
          })},
  });

  assertRowLineage({
      .name =
          "V3 rewrite: stored values are not overridden by the info columns",
      .values = {1, 2, 3},
      .storedRowIds = {{500, 501, 502}},
      .storedSequenceNumbers = {{3, 5, 3}},
      .firstRowId = 999,
      .dataSequenceNumber = 99,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeFlatVector<int64_t>({500, 501, 502}),
              makeFlatVector<int64_t>({3, 5, 3}),
          })},
  });

  assertRowLineage({
      .name = "stored columns all null: derived from the info columns",
      .values = {1, 2, 3},
      .storedRowIds = {{std::nullopt, std::nullopt, std::nullopt}},
      .storedSequenceNumbers = {{std::nullopt, std::nullopt, std::nullopt}},
      .firstRowId = 50,
      .dataSequenceNumber = 42,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({1, 2, 3}),
              makeFlatVector<int64_t>({50, 51, 52}),
              makeFlatVector<int64_t>({42, 42, 42}),
          })},
  });

  assertRowLineage({
      .name = "stored columns partly null: only the null slots are derived",
      .values = {10, 20, 30, 40},
      .storedRowIds = {{std::nullopt, 99, std::nullopt, 77}},
      .storedSequenceNumbers = {{std::nullopt, 5, std::nullopt, 10}},
      .firstRowId = 10,
      .dataSequenceNumber = 42,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30, 40}),
              makeFlatVector<int64_t>({10, 99, 12, 77}),
              makeFlatVector<int64_t>({42, 5, 42, 10}),
          })},
  });

  assertRowLineage({
      .name = "first_row_id 0 is a value, not an absent info column",
      .values = {5, 6, 7},
      .firstRowId = 0,
      .dataSequenceNumber = 5,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({5, 6, 7}),
              makeFlatVector<int64_t>({0, 1, 2}),
              makeFlatVector<int64_t>({5, 5, 5}),
          })},
  });

  assertRowLineage({
      .name = "positional deletes: _row_id uses file-absolute positions",
      .values = {10, 20, 30, 40, 50},
      .firstRowId = 200,
      .dataSequenceNumber = 42,
      .deletePositions = {1, 3},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 30, 50}),
              makeFlatVector<int64_t>({200, 202, 204}),
              makeFlatVector<int64_t>({42, 42, 42}),
          })},
  });

  // The reader drops rows for the filter, so an output position is no longer
  // the file position.
  assertRowLineage({
      .name = "filter on a data column",
      .values = {10, 20, 30, 40, 50},
      .firstRowId = 100,
      .dataSequenceNumber = 15,
      .subfieldFilter = "c0 > 20",
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({30, 40, 50}),
              makeFlatVector<int64_t>({102, 103, 104}),
              makeFlatVector<int64_t>({15, 15, 15}),
          })},
  });

  // Per the spec, an absent first_row_id means null for both columns, whatever
  // the file stores and whatever data_sequence_number says.
  assertRowLineage({
      .name = "stored columns and data_sequence_number without first_row_id",
      .values = {10, 20, 30},
      .storedRowIds = {{500, 501, 502}},
      .storedSequenceNumbers = {{3, 5, 3}},
      .dataSequenceNumber = 7,
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });
}

// A filter on either lineage column has to see the value the scan produces,
// which the reader never sees.
TEST_F(IcebergReadTest, rowLineageWithFilter) {
  assertRowLineage({
      .name = "IS NOT NULL keeps every synthesized _row_id",
      .values = {10, 20, 30},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .directFilter = {{"_row_id", std::make_shared<common::IsNotNull>()}},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeFlatVector<int64_t>({100, 101, 102}),
              makeFlatVector<int64_t>({7, 7, 7}),
          })},
  });

  // A file an UPDATE rewrote stores the lineage columns only for the rows it
  // carried over. The cases below synthesize _row_id {100, 555, 102} and
  // sequence numbers {7, 3, 7}: one stored row between two inherited ones.
  const std::vector<std::optional<int64_t>> storedRowIds = {
      std::nullopt, 555, std::nullopt};
  const std::vector<std::optional<int64_t>> storedSequenceNumbers = {
      std::nullopt, 3, std::nullopt};

  assertRowLineage({
      .name = "range on _row_id over an inherited and a stored value",
      .values = {10, 20, 30},
      .storedRowIds = storedRowIds,
      .storedSequenceNumbers = storedSequenceNumbers,
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .directFilter =
          {{"_row_id", std::make_shared<common::BigintRange>(102, 555, false)}},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({20, 30}),
              makeFlatVector<int64_t>({555, 102}),
              makeFlatVector<int64_t>({3, 7}),
          })},
  });

  // Both sides at once: the stored row is the one the range leaves out, so a
  // value wrongly inherited for it would show up here.
  assertRowLineage({
      .name = "_last_updated_sequence_number selecting the inherited values",
      .values = {10, 20, 30},
      .storedRowIds = storedRowIds,
      .storedSequenceNumbers = storedSequenceNumbers,
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .directFilter =
          {{"_last_updated_sequence_number",
            std::make_shared<common::BigintRange>(7, 7, false)}},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 30}),
              makeFlatVector<int64_t>({100, 102}),
              makeFlatVector<int64_t>({7, 7}),
          })},
  });

  // Nothing fills the nulls in later, so the row reader keeps the filter.
  assertRowLineage({
      .name = "_row_id without first_row_id",
      .values = {10, 20, 30},
      .dataSequenceNumber = 7,
      .directFilter = {{"_row_id", std::make_shared<common::IsNotNull>()}},
      .expectedVectors = {},
  });

  // A deleted position leaves a gap, so the filter has to line up with the
  // surviving rows.
  assertRowLineage({
      .name = "_row_id with positional deletes",
      .values = {10, 20, 30, 40},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .deletePositions = {1},
      .directFilter =
          {{"_row_id", std::make_shared<common::BigintRange>(100, 102, false)}},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 30}),
              makeFlatVector<int64_t>({100, 102}),
              makeFlatVector<int64_t>({7, 7}),
          })},
  });

  // Both compact the batch, the filter first: the range drops _row_id 100 and
  // the equality delete then drops c0 = 20.
  assertRowLineage({
      .name = "_row_id with an equality delete",
      .values = {10, 20, 30, 40},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .directFilter =
          {{"_row_id", std::make_shared<common::BigintRange>(101, 103, false)}},
      .equalityDeleteValues = {20},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({30, 40}),
              makeFlatVector<int64_t>({102, 103}),
              makeFlatVector<int64_t>({7, 7}),
          })},
  });

  assertRowLineage({
      .name = "_row_id leaving the equality deletes an empty batch",
      .values = {10, 20, 30, 40},
      .firstRowId = 100,
      .dataSequenceNumber = 7,
      .directFilter =
          {{"_row_id",
            std::make_shared<common::BigintRange>(1000, 2000, false)}},
      .equalityDeleteValues = {20},
      .expectedVectors = {},
  });

  // The next two store the lineage columns but leave first_row_id out, so the
  // spec requires null for every row. The filter has to see that null, not the
  // stored values.
  assertRowLineage({
      .name = "_row_id rejecting null, stored columns without first_row_id",
      .values = {10, 20, 30},
      .storedRowIds = storedRowIds,
      .storedSequenceNumbers = storedSequenceNumbers,
      .dataSequenceNumber = 7,
      .directFilter = {{"_row_id", std::make_shared<common::IsNotNull>()}},
      .expectedVectors = {},
  });

  // Every row survives, so the split must not be pruned on the stored values.
  assertRowLineage({
      .name = "_row_id accepting null, stored columns without first_row_id",
      .values = {10, 20, 30},
      .storedRowIds = storedRowIds,
      .storedSequenceNumbers = storedSequenceNumbers,
      .dataSequenceNumber = 7,
      .directFilter = {{"_row_id", std::make_shared<common::IsNull>()}},
      .expectedVectors = {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, std::nullopt, std::nullopt}),
          })},
  });
}

// A filter on a lineage column the query does not select must still see the
// synthesized value. Such a column is left out of the reader output, so there
// is no slot to synthesize into until the split reader adds one back.
TEST_F(IcebergReadTest, rowLineageFilterOnUnprojectedColumn) {
  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({10, 20, 30})})});
  const std::unordered_map<std::string, std::string> infoColumns{
      {IcebergMetadataColumn::kFirstRowIdInfoColumn, "100"},
      {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}};

  // _row_id synthesizes to {100, 101, 102} and the sequence numbers to
  // {7, 7, 7}. Only 'c0' is selected, so both are filter-only columns.
  auto assertFilter = [&](const RowTypePtr& dataColumns,
                          const std::string& columnName,
                          const common::FilterPtr& filter,
                          const std::vector<int64_t>& expectedValues) {
    SCOPED_TRACE(fmt::format("{}: {}", columnName, filter->toString()));
    common::SubfieldFilters filters;
    filters.emplace(common::Subfield(columnName), filter);
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .dataColumns(dataColumns)
                    .subfieldFiltersMap(filters)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder queryBuilder(plan);
    queryBuilder.splits({makeIcebergSplitWithInfoColumns(
        dataFilePath->getPath(), infoColumns)});
    if (expectedValues.empty()) {
      queryBuilder.assertEmptyResults();
    } else {
      queryBuilder.assertResults(
          {makeRowVector({"c0"}, {makeFlatVector<int64_t>(expectedValues)})});
    }
  };

  // A filter-only column has to be in the table schema for makeScanSpec() to
  // give it a scan-spec child at all, so this is the only shape the split
  // reader ever sees one in.
  const auto dataColumns =
      ROW(std::vector<std::string>(kRowLineageOutputNames),
          {BIGINT(), BIGINT(), BIGINT()});
  assertFilter(
      dataColumns,
      IcebergMetadataColumn::kRowIdColumnName,
      std::make_shared<common::BigintRange>(101, 102, false),
      {20, 30});
  // A dropped filter would return all three rows.
  assertFilter(
      dataColumns,
      IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName,
      std::make_shared<common::IsNull>(),
      {});
}

// Extraction pushdown gives FileDataSource a second cached output type,
// 'readerProducedType_', which is what next() allocates the output from. A
// filter-only lineage column is appended to the split reader's output type
// after that type was built, so both caches have to learn about the appended
// column or the appended channel indexes past the allocated RowVector.
TEST_F(IcebergReadTest, rowLineageFilterOnUnprojectedColumnWithExtraction) {
  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeMapVector<StringView, int64_t>(
          {{{"a", 1}},
           {{"b", 2}, {"c", 3}},
           {{"d", 4}, {"e", 5}, {"f", 6}}})})});
  const std::unordered_map<std::string, std::string> infoColumns{
      {IcebergMetadataColumn::kFirstRowIdInfoColumn, "100"},
      {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}};

  const auto mapType = MAP(VARCHAR(), BIGINT());
  const auto keysType = ARRAY(VARCHAR());
  // Reading only the map's keys sets ExtractionType::kKeys on the ScanSpec,
  // which is what makes 'readerProducedType_' differ from the output type.
  std::vector<NamedExtraction> extractions{
      {"c0",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       keysType}};
  auto handle = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      keysType,
      mapType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  // _row_id synthesizes to {100, 101, 102}; it is filtered on but not
  // selected, so it is appended to the split reader's output type.
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield(IcebergMetadataColumn::kRowIdColumnName),
      std::make_shared<common::BigintRange>(101, 102, false));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0"}, {keysType}))
                  .dataColumns(
                      ROW(std::vector<std::string>(kRowLineageOutputNames),
                          {mapType, BIGINT(), BIGINT()}))
                  .assignments({{"c0", handle}})
                  .subfieldFiltersMap(filters)
                  .endTableScan()
                  .planNode();

  auto result = exec::test::AssertQueryBuilder(plan)
                    .splits({makeIcebergSplitWithInfoColumns(
                        dataFilePath->getPath(), infoColumns)})
                    .copyResults(pool_.get());

  ASSERT_EQ(result->size(), 2);
  auto* keys = result->childAt(0)->as<ArrayVector>();
  ASSERT_EQ(keys->sizeAt(0), 2);
  ASSERT_EQ(keys->sizeAt(1), 3);
}

// The remaining filter is a second way onto the scan spec: extraction merges
// it into the same filter map, so it needs the same deferral.
TEST_F(IcebergReadTest, rowLineageFilterFromRemainingFilter) {
  // Stored _row_id {null, 555, null} with first_row_id 100 synthesizes
  // {100, 555, 102}. Filtered before synthesis, only 555 survives.
  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, 555, std::nullopt}),
              makeNullableFlatVector<int64_t>({std::nullopt, 3, std::nullopt}),
          })});
  const std::unordered_map<std::string, std::string> infoColumns{
      {IcebergMetadataColumn::kFirstRowIdInfoColumn, "100"},
      {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}};

  // A single-column comparison is the shape extraction pushes down.
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0", "_row_id"}, {BIGINT(), BIGINT()}))
                  .dataColumns(
                      ROW(std::vector<std::string>(kRowLineageOutputNames),
                          {BIGINT(), BIGINT(), BIGINT()}))
                  .remainingFilter("_row_id >= 102")
                  .endTableScan()
                  .planNode();

  exec::test::AssertQueryBuilder(plan)
      .splits({makeIcebergSplitWithInfoColumns(
          dataFilePath->getPath(), infoColumns)})
      .assertResults(makeRowVector(
          {"c0", "_row_id"},
          {
              makeFlatVector<int64_t>({20, 30}),
              makeFlatVector<int64_t>({555, 102}),
          }));
}

// What extraction leaves behind becomes a MetadataFilter, whose leaf prunes
// strides by the stored statistics. Those say nothing about what the null
// slots synthesize to, so pruning by them drops live rows.
TEST_F(IcebergReadTest, rowLineageMetadataFilterPruning) {
  // Stored _row_id {101, null, null} with first_row_id 10000 synthesizes
  // {101, 10001, 10002}, against statistics of min = max = 101.
  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {101, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>({3, std::nullopt, std::nullopt}),
          })});
  const std::unordered_map<std::string, std::string> infoColumns{
      {IcebergMetadataColumn::kFirstRowIdInfoColumn, "10000"},
      {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}};

  // 'c0 < 0' matches nothing, leaving the '_row_id' predicate alone. It is
  // there because extraction gives up on a disjunction spanning two columns,
  // which is how the predicate survives into the residual.
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0", "_row_id"}, {BIGINT(), BIGINT()}))
                  .dataColumns(
                      ROW(std::vector<std::string>(kRowLineageOutputNames),
                          {BIGINT(), BIGINT(), BIGINT()}))
                  .remainingFilter("_row_id > 9999 OR c0 < 0")
                  .endTableScan()
                  .planNode();

  exec::test::AssertQueryBuilder(plan)
      .splits({makeIcebergSplitWithInfoColumns(
          dataFilePath->getPath(), infoColumns)})
      .assertResults(makeRowVector(
          {"c0", "_row_id"},
          {
              makeFlatVector<int64_t>({20, 30}),
              makeFlatVector<int64_t>({10'001, 10'002}),
          }));
}

// A join key filter pushed onto '_row_id' at run time must see the synthesized
// value too. The split is prepared before the filter arrives, so the deferral
// cannot be conditioned on a filter already being there.
TEST_F(IcebergReadTest, rowLineageDynamicFilterOnRowId) {
  // The join alone would produce the same rows, so the scan statistics are what
  // tell us the filter reached the scan.
  core::PlanNodeId scanId;
  auto assertJoinOnRowId =
      [&](const std::vector<std::shared_ptr<ConnectorSplit>>& splits,
          const std::vector<int64_t>& joinKeys,
          const std::vector<int64_t>& expectedValues,
          const std::vector<int64_t>& expectedRowIds) {
        auto planNodeIdGenerator =
            std::make_shared<core::PlanNodeIdGenerator>();
        auto plan =
            exec::test::PlanBuilder(planNodeIdGenerator)
                .startTableScan(test::kIcebergConnectorId)
                .outputType(ROW({"c0", "_row_id"}, {BIGINT(), BIGINT()}))
                .dataColumns(ROW({"c0"}, {BIGINT()}))
                .endTableScan()
                .capturePlanNodeId(scanId)
                .hashJoin(
                    {"_row_id"},
                    {"u0"},
                    exec::test::PlanBuilder(planNodeIdGenerator)
                        .values({makeRowVector(
                            {"u0"}, {makeFlatVector<int64_t>(joinKeys)})})
                        .planNode(),
                    /*filter=*/"",
                    {"c0", "_row_id"})
                .planNode();

        auto task =
            exec::test::AssertQueryBuilder(plan)
                .maxDrivers(1)
                .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "2")
                .splits(scanId, splits)
                .assertResults(makeRowVector(
                    {"c0", "_row_id"},
                    {
                        makeFlatVector<int64_t>(expectedValues),
                        makeFlatVector<int64_t>(expectedRowIds),
                    }));

        const auto planStats = exec::toPlanStats(task->taskStats());
        const auto& scanStats = planStats.at(scanId);
        EXPECT_FALSE(scanStats.dynamicFilterStats.empty());
        EXPECT_EQ(scanStats.outputRows, expectedValues.size());
        return task;
      };

  // Stored _row_id {null, 555, null} with first_row_id 100 synthesizes
  // {100, 555, 102}. The join key 102 belongs to a row whose stored value is
  // null, so a filter run before synthesis matches nothing.
  auto storedFile = TempFilePath::create();
  writeToFile(
      storedFile->getPath(),
      {makeRowVector(
          {"c0", "_row_id", "_last_updated_sequence_number"},
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeNullableFlatVector<int64_t>(
                  {std::nullopt, 555, std::nullopt}),
              makeNullableFlatVector<int64_t>({std::nullopt, 3, std::nullopt}),
          })});
  auto splitAt = [&](const std::string& path, const std::string& firstRowId) {
    return makeIcebergSplitWithInfoColumns(
        path,
        {{IcebergMetadataColumn::kFirstRowIdInfoColumn, firstRowId},
         {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "7"}});
  };
  assertJoinOnRowId(
      {splitAt(storedFile->getPath(), "100")}, {102}, {30}, {102});

  // Neither file stores the lineage columns, so '_row_id' is a null constant on
  // the scan spec, the shape in which a filter is easiest to lose: nothing in
  // the reader filters a constant. Both splits are preloaded, each on its own
  // scan spec, and synthesize {100, 101, 102} and {200, 201, 202}.
  auto firstFile = TempFilePath::create();
  writeToFile(
      firstFile->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({10, 20, 30})})});
  auto secondFile = TempFilePath::create();
  writeToFile(
      secondFile->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({40, 50, 60})})});
  auto task = assertJoinOnRowId(
      {splitAt(firstFile->getPath(), "100"),
       splitAt(secondFile->getPath(), "200")},
      {102, 200},
      {30, 40},
      {102, 200});
  const auto planStats = exec::toPlanStats(task->taskStats());
  EXPECT_EQ(planStats.at(scanId).customStats.at("preloadedSplits").sum, 2);
}

// A filter on '_row_id' must be enforced on every split of a data source, which
// share one scan spec, and only for as long as a split synthesizes the column.
TEST_F(IcebergReadTest, rowLineageFilterAcrossSplits) {
  // The splits synthesize _row_id {100, 101, 102} and {200, 201, 202}. If the
  // filter is lost on the second split, rows 201 and 202 leak into the result.
  assertRowLineageAcrossSplits(
      std::make_shared<common::BigintRange>(102, 200, false),
      /*secondFirstRowId=*/200,
      makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({30, 40}),
              makeFlatVector<int64_t>({102, 200}),
              makeFlatVector<int64_t>({7, 9}),
          }));

  // Without first_row_id the second split's _row_id is null for every row and
  // nothing fills it in, so the filter has to move back to the row reader.
  assertRowLineageAcrossSplits(
      std::make_shared<common::IsNotNull>(),
      /*secondFirstRowId=*/std::nullopt,
      makeRowVector(
          kRowLineageOutputNames,
          {
              makeFlatVector<int64_t>({10, 20, 30}),
              makeFlatVector<int64_t>({100, 101, 102}),
              makeFlatVector<int64_t>({7, 7, 7}),
          }));
}

// A deferred filter runs per output batch, on rows the reader has already read.
TEST_F(IcebergReadTest, rowLineageFilterOverBatches) {
  // The first two batches come back empty. An empty batch must not be reported
  // as the end of the split, or the rows after it are lost.
  assertRowLineageFilterOverBatches(
      /*firstSelected=*/2 * kMultiBatchRowsPerBatch,
      /*lastSelected=*/kMultiBatchNumRows - 1);

  // A range ending inside the second batch leaves that one partially compacted.
  assertRowLineageFilterOverBatches(
      /*firstSelected=*/0,
      /*lastSelected=*/kMultiBatchRowsPerBatch + kMultiBatchRowsPerBatch / 2 -
          1);

  // A range past the last '_row_id' the split synthesizes keeps nothing. The
  // filter is deferred, so it prunes no split and skips no row group: every row
  // is read and then dropped.
  assertRowLineageFilterOverBatches(
      /*firstSelected=*/kMultiBatchNumRows,
      /*lastSelected=*/2 * kMultiBatchNumRows);
}

// A join can push a filter onto a data column after the split has started
// reading. The reader then drops rows partway through, so a batch's start
// offset no longer says where a row sat in the file.
DEBUG_ONLY_TEST_F(IcebergReadTest, rowIdWithDynamicFilterMidSplit) {
  constexpr vector_size_t kNumRows = 300;
  constexpr vector_size_t kRowsPerBatch = 100;
  constexpr int64_t kFirstRowId = 1'000;
  // Lands inside the second batch, so the rows before it survive.
  constexpr int64_t kSmallestValueKept = 150;

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>(
          kNumRows, [](vector_size_t row) { return row; })})});

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0", "_row_id"}, {BIGINT(), BIGINT()}))
                  .dataColumns(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();

  common::testutil::TestValue::enable();
  int numBatches{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::connector::hive::FileDataSource::next",
      std::function<void(FileDataSource*)>([&](FileDataSource* dataSource) {
        if (numBatches++ == 1) {
          // 'c0' is output channel 0.
          dataSource->addDynamicFilter(
              0,
              std::make_shared<common::BigintRange>(
                  kSmallestValueKept, kNumRows - 1, false));
        }
      }));

  auto result = exec::test::AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config(
                        core::QueryConfig::kPreferredOutputBatchRows,
                        std::to_string(kRowsPerBatch))
                    .config(
                        core::QueryConfig::kMaxOutputBatchRows,
                        std::to_string(kRowsPerBatch))
                    .splits({makeIcebergSplitWithInfoColumns(
                        dataFilePath->getPath(),
                        {{IcebergMetadataColumn::kFirstRowIdInfoColumn,
                          std::to_string(kFirstRowId)}})})
                    .copyResults(pool());

  // Some rows were read before the filter arrived and some were dropped after,
  // so the surviving positions have a gap in them.
  ASSERT_GT(result->size(), 0);
  ASSERT_LT(result->size(), kNumRows);
  auto* values = result->childAt(0)->asFlatVector<int64_t>();
  auto* rowIds = result->childAt(1)->asFlatVector<int64_t>();
  for (vector_size_t row = 0; row < result->size(); ++row) {
    EXPECT_EQ(rowIds->valueAt(row), kFirstRowId + values->valueAt(row))
        << "at output row " << row;
  }
}

// Tests Iceberg MERGE INTO row-id synthesis: the projection of the synthetic
// $target_table_row_id ROW column produced at read time from the split's
// infoColumns ($path, $spec_id, partition_data) plus the file row positions.
// Mirrors the IcebergPageSourceProvider Java path that backs
// MERGE_TARGET_ROW_ID_DATA.
TEST_F(IcebergReadTest, targetTableRowIdSynthesis) {
  // The column is a null placeholder constant until next() synthesizes it, so a
  // filter the reader evaluates would prune the whole split.
  for (const auto& filter : std::vector<common::FilterPtr>{
           nullptr, std::make_shared<common::IsNotNull>()}) {
    assertTargetTableRowId(
        /*values=*/{10, 20, 30},
        /*deletePositions=*/{},
        /*expectedValues=*/{10, 20, 30},
        /*expectedPositions=*/{0, 1, 2},
        filter,
        /*filterField=*/"");
  }

  // Positional deletes leave the reader returning fewer rows than it scanned,
  // so the composite has to be sized, and its row positions taken, from the
  // rows that came back rather than the rows read: the survivors keep the file
  // positions 0 and 3.
  assertTargetTableRowId(
      /*values=*/{10, 20, 30, 40},
      /*deletePositions=*/{1, 2},
      /*expectedValues=*/{10, 40},
      /*expectedPositions=*/{0, 3},
      /*filter=*/nullptr,
      /*filterField=*/"");

  // A range on 'row_position' has to select from the positions next() puts in
  // the composite. Deleting positions 1 and 2 leaves 0, 3 and 4, of which the
  // range keeps 3 alone. Run before synthesis the range would match nothing:
  // the reader sees a null placeholder.
  assertTargetTableRowId(
      /*values=*/{10, 20, 30, 40, 50},
      /*deletePositions=*/{1, 2},
      /*expectedValues=*/{40},
      /*expectedPositions=*/{3},
      std::make_shared<common::BigintRange>(1, 3, false),
      /*filterField=*/"row_position");
}

// Info columns arrive as strings on the split and are parsed at read time.
// A value the coordinator could not have produced means the split metadata is
// corrupt, so the reader must fail loudly rather than silently substituting a
// default: $first_row_id and $data_sequence_number both feed V3 row lineage,
// and a wrong value there mislabels every row in the file.
class IcebergInfoColumnValidationTest : public IcebergReadTest {
 protected:
  // Runs a scan of a single BIGINT column with 'infoColumns' attached to the
  // split, projecting 'outputType' (which decides whether the row-lineage or
  // MERGE row-id parsing paths run at all).
  void assertScanFails(
      const std::unordered_map<std::string, std::string>& infoColumns,
      const RowTypePtr& outputType,
      const std::string& expectedMessage) {
    std::vector<RowVectorPtr> inputVectors = {
        makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30})})};
    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), inputVectors);

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(outputType)
                    .dataColumns(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();

    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .splits({makeIcebergSplitWithInfoColumns(
                dataFilePath->getPath(), infoColumns)})
            .copyResults(pool()),
        expectedMessage);
  }

  // Projecting _row_id is what makes the reader parse $first_row_id.
  RowTypePtr rowLineageOutputType() const {
    return ROW(
        {"c0", IcebergMetadataColumn::kRowIdColumnName}, {BIGINT(), BIGINT()});
  }

  // Projecting $target_table_row_id is what makes the reader parse $spec_id.
  RowTypePtr targetRowIdOutputType() const {
    return ROW(
        {"c0", IcebergMetadataColumn::kTargetTableRowIdColumnName},
        {BIGINT(),
         ROW({"file_path", "row_position", "spec_id", "partition_data"},
             {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()})});
  }
};

TEST_F(IcebergInfoColumnValidationTest, rejectsNonNumericFirstRowId) {
  assertScanFails(
      {{IcebergMetadataColumn::kFirstRowIdInfoColumn, "not-a-number"}},
      rowLineageOutputType(),
      "Invalid $first_row_id value in split info columns");
}

TEST_F(IcebergInfoColumnValidationTest, rejectsNegativeFirstRowId) {
  // Parses cleanly but is out of range: row ids are file-absolute offsets.
  assertScanFails(
      {{IcebergMetadataColumn::kFirstRowIdInfoColumn, "-1"}},
      rowLineageOutputType(),
      "First row ID must be non-negative");
}

TEST_F(IcebergInfoColumnValidationTest, rejectsNonNumericDataSequenceNumber) {
  assertScanFails(
      {{IcebergMetadataColumn::kFirstRowIdInfoColumn, "0"},
       {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "abc"}},
      rowLineageOutputType(),
      "Invalid $data_sequence_number value in split info columns");
}

TEST_F(IcebergInfoColumnValidationTest, rejectsNegativeDataSequenceNumber) {
  assertScanFails(
      {{IcebergMetadataColumn::kFirstRowIdInfoColumn, "0"},
       {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "-5"}},
      rowLineageOutputType(),
      "Data sequence number must be non-negative");
}

TEST_F(IcebergInfoColumnValidationTest, rejectsNonNumericSpecId) {
  assertScanFails(
      {{IcebergMetadataColumn::kSpecIdInfoColumn, "spec-seven"}},
      targetRowIdOutputType(),
      "Invalid $spec_id value in split info columns");
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

TEST_F(IcebergReadTest, filterPushdownWithInitialDefaultInFilterColumnHandles) {
  // Test's a scenario where filter on default value column is used in the query
  // TABLE = [id int , country varchar(defaultValue='IN')]
  // QUERY = SELECT id FROM table WHERE country = 'IN'
  // When filter pushdown is enabled, column handle for 'country' is present
  // only in filterColumnHandles_ of HiveTableHandle. adaptColumns was searching
  // only in columnHandles_ for the default value, missing it and creating null
  // vector.

  // Old data file: only the 'id' column present.
  std::vector<RowVectorPtr> dataVectors = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVectors);

  auto outputType = ROW({"id"}, {BIGINT()});

  ColumnHandleMap assignments;
  assignments["id"] = makeIcebergHandle("id", BIGINT(), 1);

  // filterColumnHandles carries country WITH initialDefaultValue="IN".
  std::vector<HiveColumnHandlePtr> filterHandles = {
      makeIcebergHandle("country", VARCHAR(), 2, "IN")};

  // Expected: all 3 rows (country filter passes via default constant).
  std::vector<RowVectorPtr> allRowsIN = {
      makeRowVector(outputType->names(), {makeFlatVector<int64_t>({1, 2, 3})})};

  // Full schema used for subfieldFilter expression parsing (country must be
  // reachable even though it is not in outputType).
  auto fullSchema = ROW({"id", "country"}, {BIGINT(), VARCHAR()});

  auto assertFilter =
      [&](const std::string& subfieldFilter,
          const std::vector<RowVectorPtr>& expected,
          const std::vector<std::shared_ptr<ConnectorSplit>>& splits,
          int32_t numSplitsSkipped = 0) {
        auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(test::kIcebergConnectorId)
                        .outputType(outputType)
                        .dataColumns(fullSchema)
                        .assignments(assignments)
                        .filterColumnHandles(filterHandles)
                        .subfieldFilter(subfieldFilter)
                        .endTableScan()
                        .planNode();
        auto task =
            exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(
                expected);
        ASSERT_EQ(
            task->taskStats()
                .pipelineStats[0]
                .operatorStats[0]
                .runtimeStats["skippedSplits"]
                .sum,
            numSplitsSkipped)
            << "Unexpected skipped splits for filter: " << subfieldFilter;
      };

  // Bug scenario: country absent from file, assignments handle has no default.
  // Without fix: adaptColumns finds no default → sets NULL → testFilters skips
  //              file (NULL != 'IN') → 0 rows, numSplitsSkipped=1. WRONG.
  // With fix:    adaptColumns finds default 'IN' in filterColumnHandles →
  //              constant 'IN' → testFilters passes → 3 rows. CORRECT.
  // Note: splits must be recreated for each assertFilter call — ConnectorSplit
  // objects have their dataSource set during execution and cannot be reused.
  assertFilter(
      "country = 'IN'",
      allRowsIN,
      makeIcebergSplits(dataFilePath->getPath()),
      /*numSplitsSkipped=*/0);

  // Non-matching default: constant 'IN' != 'US' → file skipped regardless.
  assertFilter(
      "country = 'US'",
      {},
      makeIcebergSplits(dataFilePath->getPath()),
      /*numSplitsSkipped=*/1);

  // New file written AFTER ALTER TABLE: country physically present = 'US'.
  // Output still only has {id} — country is filter-only.
  std::vector<RowVectorPtr> newData = {makeRowVector(
      {"id", "country"},
      {makeFlatVector<int64_t>({4, 5}),
       makeFlatVector<std::string>({"US", "US"})})};
  auto newFilePath = TempFilePath::create();
  writeToFile(newFilePath->getPath(), newData);

  auto makeTwoSplits = [&]() {
    auto s1 = makeIcebergSplits(dataFilePath->getPath());
    auto s2 = makeIcebergSplits(newFilePath->getPath());
    s1.insert(s1.end(), s2.begin(), s2.end());
    return s1;
  };

  // country='IN': old file passes (constant 'IN'), new file skipped ('US').
  // 1 split skipped, rows {1,2,3} from old file.
  assertFilter(
      "country = 'IN'", allRowsIN, makeTwoSplits(), /*numSplitsSkipped=*/1);

  // country='US': old file skipped (constant 'IN'!='US'), new file passes.
  // 1 split skipped, rows {4,5} from new file.
  std::vector<RowVectorPtr> newRowsUS = {
      makeRowVector(outputType->names(), {makeFlatVector<int64_t>({4, 5})})};
  assertFilter(
      "country = 'US'", newRowsUS, makeTwoSplits(), /*numSplitsSkipped=*/1);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
