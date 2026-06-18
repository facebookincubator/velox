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

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;

namespace facebook::velox::connector::hive::iceberg {

namespace {

class IcebergChangelogE2ETest : public test::IcebergTestBase {
 protected:
  static constexpr int32_t kDefaultNumBatches = 2;
  static constexpr int32_t kDefaultRowsPerBatch = 100;

  std::vector<RowVectorPtr> createChangelogTestData() {
    std::vector<RowVectorPtr> batches;

    for (auto i = 0; i < kDefaultNumBatches; i++) {
      auto idVector = makeFlatVector<int64_t>(
          kDefaultRowsPerBatch,
          [i](auto row) { return i * kDefaultRowsPerBatch + row; });
      auto nameVector =
          makeFlatVector<std::string>(kDefaultRowsPerBatch, [i](auto row) {
            return "name_" + std::to_string(i * kDefaultRowsPerBatch + row);
          });
      batches.push_back(makeRowVector({"id", "name"}, {idVector, nameVector}));
    }
    return batches;
  }

  std::shared_ptr<ConnectorSplit> makeChangelogSplit(
      const std::string& dataFilePath,
      ChangelogOperation operation,
      int64_t ordinal,
      int64_t snapshotId) {
    auto changelogInfo =
        std::make_shared<ChangelogSplitInfo>(operation, ordinal, snapshotId);

    // Get actual file size
    const auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                          ->openFileForRead(dataFilePath);

    return IcebergSplitBuilder(dataFilePath)
        .connectorId(test::kIcebergConnectorId)
        .fileFormat(fileFormat_)
        .start(0)
        .length(file->size())
        .changelogSplitInfo(changelogInfo)
        .build();
  }

  RowTypePtr createChangelogOutputType(const RowTypePtr& dataRowType) {
    return ROW(
        {"operation", "ordinal", "snapshotid", "rowdata"},
        {VARCHAR(), BIGINT(), BIGINT(), dataRowType});
  }

  ColumnHandleMap createChangelogColumnHandles(const RowTypePtr& dataType) {
    ColumnHandleMap handles;
    handles["operation"] = std::make_shared<HiveColumnHandle>(
        "operation",
        HiveColumnHandle::ColumnType::kRegular,
        VARCHAR(),
        VARCHAR());
    handles["ordinal"] = std::make_shared<HiveColumnHandle>(
        "ordinal", HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
    handles["snapshotid"] = std::make_shared<HiveColumnHandle>(
        "snapshotid",
        HiveColumnHandle::ColumnType::kRegular,
        BIGINT(),
        BIGINT());
    handles["rowdata"] = std::make_shared<HiveColumnHandle>(
        "rowdata", HiveColumnHandle::ColumnType::kRegular, dataType, dataType);
    return handles;
  }

  ColumnHandleMap createDataColumnHandles(const RowTypePtr& dataType) {
    ColumnHandleMap handles;
    for (size_t i = 0; i < dataType->size(); ++i) {
      const auto& name = dataType->nameOf(i);
      const auto& type = dataType->childAt(i);
      handles[name] = std::make_shared<HiveColumnHandle>(
          name, HiveColumnHandle::ColumnType::kRegular, type, type);
    }
    return handles;
  }

  std::shared_ptr<HiveTableHandle> createChangelogTableHandle(
      const RowTypePtr& dataType) {
    auto dataColumnHandles = createDataColumnHandles(dataType);
    return std::make_shared<HiveTableHandle>(
        test::kIcebergConnectorId,
        "test_table",
        common::SubfieldFilters{},
        nullptr, // remainingFilter
        dataType, // dataColumns
        std::vector<std::string>{}, // indexColumns
        std::unordered_map<std::string, std::string>{}, // tableParameters
        std::vector<HiveColumnHandlePtr>{}, // filterColumnHandles
        1.0, // sampleRate
        "", // dbName
        true, // isChangelogQuery
        dataColumnHandles);
  }

  void verifyChangelogSchema(const RowTypePtr& outputType) {
    ASSERT_EQ(outputType->size(), 4);
    ASSERT_EQ(outputType->nameOf(0), "operation");
    ASSERT_EQ(outputType->childAt(0)->kind(), TypeKind::VARCHAR);
    ASSERT_EQ(outputType->nameOf(1), "ordinal");
    ASSERT_EQ(outputType->childAt(1)->kind(), TypeKind::BIGINT);
    ASSERT_EQ(outputType->nameOf(2), "snapshotid");
    ASSERT_EQ(outputType->childAt(2)->kind(), TypeKind::BIGINT);
    ASSERT_EQ(outputType->nameOf(3), "rowdata");
    ASSERT_EQ(outputType->childAt(3)->kind(), TypeKind::ROW);
  }

  void verifyChangelogRecord(
      const RowVectorPtr& resultVector,
      int32_t rowIndex,
      const std::string& expectedOperation,
      int64_t expectedOrdinal,
      int64_t expectedSnapshotId) {
    // Verify operation
    auto operationVector =
        resultVector->childAt(0)->as<SimpleVector<StringView>>();
    ASSERT_EQ(
        operationVector->valueAt(rowIndex), StringView(expectedOperation));

    // Verify ordinal
    auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();
    ASSERT_EQ(ordinalVector->valueAt(rowIndex), expectedOrdinal);

    // Verify snapshotid
    auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();
    ASSERT_EQ(snapshotVector->valueAt(rowIndex), expectedSnapshotId);
  }
};

TEST_F(IcebergChangelogE2ETest, basicChangelogQuery) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  // Write test data using IcebergDataSink (writes Parquet)
  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  // Get the data file path
  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 12345);

  // Use changelog output schema (operation, ordinal, snapshotid, rowdata)
  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  // Build query plan with changelog schema and table handle
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .planNode();

  // Execute query
  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);

  // Verify changelog schema
  verifyChangelogSchema(
      std::dynamic_pointer_cast<const RowType>(resultVector->type()));

  // Verify row count
  int32_t totalRows = 0;
  for (const auto& batch : batches) {
    totalRows += batch->size();
  }
  ASSERT_EQ(resultVector->size(), totalRows);

  // Verify all rows have correct changelog metadata
  for (auto i = 0; i < resultVector->size(); i++) {
    verifyChangelogRecord(resultVector, i, "INSERT", 1, 12345);
  }

  // Verify rowdata contains the original data
  auto rowdataVector = resultVector->childAt(3)->as<RowVector>();
  ASSERT_NE(rowdataVector, nullptr);
  ASSERT_EQ(rowdataVector->childrenSize(), 2);

  auto idVector = rowdataVector->childAt(0)->as<SimpleVector<int64_t>>();
  auto nameVector = rowdataVector->childAt(1)->as<SimpleVector<StringView>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_FALSE(idVector->isNullAt(i));
    ASSERT_FALSE(nameVector->isNullAt(i));
    ASSERT_EQ(idVector->valueAt(i), i);
    std::string expectedName = "name_" + std::to_string(i);
    ASSERT_EQ(nameVector->valueAt(i), StringView(expectedName));
  }
}

TEST_F(IcebergChangelogE2ETest, differentOperations) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  std::vector<std::pair<ChangelogOperation, std::string>> operations = {
      {ChangelogOperation::INSERT, "INSERT"},
      {ChangelogOperation::DELETE, "DELETE"},
      {ChangelogOperation::UPDATE_BEFORE, "UPDATE_BEFORE"},
      {ChangelogOperation::UPDATE_AFTER, "UPDATE_AFTER"}};

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .planNode();

  // Verify all operation types produce correct changelog records
  for (const auto& [operation, operationStr] : operations) {
    auto changelogSplit = makeChangelogSplit(dataFilePath, operation, 1, 12345);

    auto resultVector =
        AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

    ASSERT_TRUE(resultVector != nullptr);
    ASSERT_GT(resultVector->size(), 0);

    // Verify changelog schema
    verifyChangelogSchema(
        std::dynamic_pointer_cast<const RowType>(resultVector->type()));

    // Verify all rows have the correct operation
    for (auto i = 0; i < resultVector->size(); i++) {
      verifyChangelogRecord(resultVector, i, operationStr, 1, 12345);
    }
  }
}

TEST_F(IcebergChangelogE2ETest, multipleSnapshots) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_BEFORE, 2, 200));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .planNode();

  // Verify multiple changelog splits can be read
  auto resultVector =
      AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);

  // Verify changelog schema
  verifyChangelogSchema(
      std::dynamic_pointer_cast<const RowType>(resultVector->type()));

  int32_t expectedRowsPerSplit = 0;
  for (const auto& batch : batches) {
    expectedRowsPerSplit += batch->size();
  }
  int32_t expectedTotalRows = expectedRowsPerSplit * 3; // 3 splits

  ASSERT_EQ(resultVector->size(), expectedTotalRows);

  // Verify changelog records from each split
  // First split: INSERT with ordinal=1, snapshotId=100
  for (auto i = 0; i < expectedRowsPerSplit; i++) {
    verifyChangelogRecord(resultVector, i, "INSERT", 1, 100);
  }

  // Second split: UPDATE_BEFORE with ordinal=2, snapshotId=200
  for (auto i = expectedRowsPerSplit; i < expectedRowsPerSplit * 2; i++) {
    verifyChangelogRecord(resultVector, i, "UPDATE_BEFORE", 2, 200);
  }

  // Third split: UPDATE_AFTER with ordinal=3, snapshotId=200
  for (auto i = expectedRowsPerSplit * 2; i < expectedTotalRows; i++) {
    verifyChangelogRecord(resultVector, i, "UPDATE_AFTER", 3, 200);
  }
}

TEST_F(IcebergChangelogE2ETest, nonChangelogSplitPassthrough) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto regularSplit = std::make_shared<HiveIcebergSplit>(
      test::kIcebergConnectorId,
      dataFilePath,
      fileFormat_,
      0,
      std::numeric_limits<uint64_t>::max(),
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      nullptr,
      true,
      std::vector<IcebergDeleteFile>{});

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(dataRowType)
                  .endTableScan()
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(regularSplit).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);

  // Verify we got regular data schema, not changelog schema
  auto rowType = std::dynamic_pointer_cast<const RowType>(resultVector->type());
  ASSERT_EQ(rowType->size(), 2);
  ASSERT_EQ(rowType->nameOf(0), "id");
  ASSERT_EQ(rowType->nameOf(1), "name");
}

TEST_F(IcebergChangelogE2ETest, selectMetadataColumnsOnly) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 5, 99999);

  // Select only metadata columns (operation, ordinal, snapshotid) without
  // rowdata
  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto metadataOnlyType = ROW(
      {"operation", "ordinal", "snapshotid"}, {VARCHAR(), BIGINT(), BIGINT()});

  ColumnHandleMap metadataHandles;
  metadataHandles["operation"] = std::make_shared<HiveColumnHandle>(
      "operation",
      HiveColumnHandle::ColumnType::kRegular,
      VARCHAR(),
      VARCHAR());
  metadataHandles["ordinal"] = std::make_shared<HiveColumnHandle>(
      "ordinal", HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
  metadataHandles["snapshotid"] = std::make_shared<HiveColumnHandle>(
      "snapshotid", HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());

  auto tableHandle = createChangelogTableHandle(dataRowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(metadataOnlyType)
                  .tableHandle(tableHandle)
                  .assignments(metadataHandles)
                  .endTableScan()
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);
  ASSERT_EQ(resultVector->childrenSize(), 3);

  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }
  ASSERT_EQ(resultVector->size(), expectedRows);

  // Verify all rows have correct metadata
  auto operationVector =
      resultVector->childAt(0)->as<SimpleVector<StringView>>();
  auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();
  auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_EQ(operationVector->valueAt(i), StringView("INSERT"));
    ASSERT_EQ(ordinalVector->valueAt(i), 5);
    ASSERT_EQ(snapshotVector->valueAt(i), 99999);
  }
}

TEST_F(IcebergChangelogE2ETest, filterOnMetadataColumns) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  // Create splits with different operations and ordinals
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  // Filter: operation = 'INSERT' OR ordinal > 2
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .filter("operation = 'INSERT' OR ordinal > 2")
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);

  int32_t expectedRowsPerSplit = 0;
  for (const auto& batch : batches) {
    expectedRowsPerSplit += batch->size();
  }
  // Should get rows from split 1 (INSERT) and split 3 (ordinal=3)
  int32_t expectedTotalRows = expectedRowsPerSplit * 2;
  ASSERT_EQ(resultVector->size(), expectedTotalRows);

  // Verify filtered results
  auto operationVector =
      resultVector->childAt(0)->as<SimpleVector<StringView>>();
  auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    auto op = operationVector->valueAt(i);
    auto ord = ordinalVector->valueAt(i);
    // Each row should match the filter condition
    ASSERT_TRUE(op == StringView("INSERT") || ord > 2);
  }
}

TEST_F(IcebergChangelogE2ETest, filterOnRowdataNestedColumns) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 12345);

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  // Filter on nested rowdata column: rowdata.id < 50
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .filter("rowdata.id < 50")
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);
  ASSERT_GT(resultVector->size(), 0);
  ASSERT_LT(resultVector->size(), kDefaultNumBatches * kDefaultRowsPerBatch);

  // Verify all filtered rows have id < 50
  auto rowdataVector = resultVector->childAt(3)->as<RowVector>();
  ASSERT_NE(rowdataVector, nullptr);
  auto idVector = rowdataVector->childAt(0)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_LT(idVector->valueAt(i), 50);
  }
}

TEST_F(IcebergChangelogE2ETest, selectRowdataSubfieldsOnly) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto changelogSplit = makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 7, 54321);

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  // Project only rowdata subfields (no metadata columns)
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles)
                  .endTableScan()
                  .project({"rowdata.id", "rowdata.name"})
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_TRUE(resultVector != nullptr);
  ASSERT_EQ(resultVector->childrenSize(), 2);

  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }
  ASSERT_EQ(resultVector->size(), expectedRows);

  // Verify projected columns - only data columns, no metadata
  auto rowType = std::dynamic_pointer_cast<const RowType>(resultVector->type());
  ASSERT_EQ(rowType->nameOf(0), "id");
  ASSERT_EQ(rowType->nameOf(1), "name");

  // Verify data columns
  auto idVector = resultVector->childAt(0)->as<SimpleVector<int64_t>>();
  auto nameVector = resultVector->childAt(1)->as<SimpleVector<StringView>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_FALSE(idVector->isNullAt(i));
    ASSERT_FALSE(nameVector->isNullAt(i));
    ASSERT_EQ(idVector->valueAt(i), i);
    std::string expectedName = "name_" + std::to_string(i);
    ASSERT_EQ(nameVector->valueAt(i), StringView(expectedName));
  }
}

TEST_F(IcebergChangelogE2ETest, aggregations) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  int32_t expectedRowsPerSplit = 0;
  for (const auto& batch : batches) {
    expectedRowsPerSplit += batch->size();
  }

  // Test 1: Count all changelog records (count(*))
  // Create fresh splits for this query
  std::vector<std::shared_ptr<ConnectorSplit>> countSplits;
  countSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  countSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100));
  countSplits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  auto countPlan = PlanBuilder()
                       .startTableScan()
                       .connectorId(test::kIcebergConnectorId)
                       .outputType(changelogOutputType)
                       .tableHandle(tableHandle)
                       .assignments(changelogColumnHandles)
                       .endTableScan()
                       .singleAggregation({}, {"count(1)"})
                       .planNode();

  auto countResult =
      AssertQueryBuilder(countPlan).splits(countSplits).copyResults(pool());

  ASSERT_TRUE(countResult != nullptr);
  ASSERT_EQ(countResult->size(), 1);

  int64_t expectedTotal = expectedRowsPerSplit * 3; // 3 splits
  auto countVector = countResult->childAt(0)->as<SimpleVector<int64_t>>();
  ASSERT_EQ(countVector->valueAt(0), expectedTotal);

  // Test 2: Group by operation and count
  // Create fresh splits for this query
  std::vector<std::shared_ptr<ConnectorSplit>> groupBySplits;
  groupBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  groupBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100));
  groupBySplits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  auto groupByPlan = PlanBuilder()
                         .startTableScan()
                         .connectorId(test::kIcebergConnectorId)
                         .outputType(changelogOutputType)
                         .tableHandle(tableHandle)
                         .assignments(changelogColumnHandles)
                         .endTableScan()
                         .singleAggregation({"operation"}, {"count(1)"})
                         .planNode();

  auto groupByResult =
      AssertQueryBuilder(groupByPlan).splits(groupBySplits).copyResults(pool());

  ASSERT_TRUE(groupByResult != nullptr);
  ASSERT_EQ(groupByResult->size(), 3); // 3 different operations

  // Verify each operation has the correct count
  auto operationVector =
      groupByResult->childAt(0)->as<SimpleVector<StringView>>();
  auto groupCountVector =
      groupByResult->childAt(1)->as<SimpleVector<int64_t>>();

  std::unordered_map<std::string, int64_t> operationCounts;
  for (auto i = 0; i < groupByResult->size(); i++) {
    std::string op(
        operationVector->valueAt(i).data(), operationVector->valueAt(i).size());
    operationCounts[op] = groupCountVector->valueAt(i);
  }

  ASSERT_EQ(operationCounts["INSERT"], expectedRowsPerSplit);
  ASSERT_EQ(operationCounts["DELETE"], expectedRowsPerSplit);
  ASSERT_EQ(operationCounts["UPDATE_AFTER"], expectedRowsPerSplit);

  // Test 3: Group by rowdata nested column (rowdata.id) and count
  // Project the nested field first, then group by it
  std::vector<std::shared_ptr<ConnectorSplit>> groupByNestedSplits;
  groupByNestedSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));

  auto groupByNestedPlan = PlanBuilder()
                               .startTableScan()
                               .connectorId(test::kIcebergConnectorId)
                               .outputType(changelogOutputType)
                               .tableHandle(tableHandle)
                               .assignments(changelogColumnHandles)
                               .endTableScan()
                               .project({"rowdata.id AS id"})
                               .singleAggregation({"id"}, {"count(1)"})
                               .planNode();

  auto groupByNestedResult = AssertQueryBuilder(groupByNestedPlan)
                                 .splits(groupByNestedSplits)
                                 .copyResults(pool());

  ASSERT_TRUE(groupByNestedResult != nullptr);
  ASSERT_EQ(groupByNestedResult->size(), expectedRowsPerSplit);

  // Verify each id has count of 1 (each id is unique)
  auto idGroupVector =
      groupByNestedResult->childAt(0)->as<SimpleVector<int64_t>>();
  auto idCountVector =
      groupByNestedResult->childAt(1)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < groupByNestedResult->size(); i++) {
    ASSERT_EQ(idCountVector->valueAt(i), 1);
  }

  // Test 4: Group by metadata column (operation) and aggregate rowdata
  // Select operation (group by key) and aggregated rowdata
  std::vector<std::shared_ptr<ConnectorSplit>> groupByMetadataSplits;
  groupByMetadataSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  groupByMetadataSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100));

  auto groupByMetadataPlan =
      PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType)
          .tableHandle(tableHandle)
          .assignments(changelogColumnHandles)
          .endTableScan()
          .singleAggregation({"operation"}, {"count(rowdata)"})
          .planNode();

  auto groupByMetadataResult = AssertQueryBuilder(groupByMetadataPlan)
                                   .splits(groupByMetadataSplits)
                                   .copyResults(pool());

  ASSERT_TRUE(groupByMetadataResult != nullptr);
  ASSERT_EQ(groupByMetadataResult->size(), 2); // 2 different operations

  // Verify we got operation (group key) and count of rowdata
  auto opGroupVector =
      groupByMetadataResult->childAt(0)->as<SimpleVector<StringView>>();
  auto rowdataCountVector =
      groupByMetadataResult->childAt(1)->as<SimpleVector<int64_t>>();

  std::unordered_map<std::string, int64_t> operationRowdataCounts;
  for (auto i = 0; i < groupByMetadataResult->size(); i++) {
    std::string op(
        opGroupVector->valueAt(i).data(), opGroupVector->valueAt(i).size());
    operationRowdataCounts[op] = rowdataCountVector->valueAt(i);
  }

  // Each operation should have expectedRowsPerSplit rows
  ASSERT_EQ(operationRowdataCounts["INSERT"], expectedRowsPerSplit);
  ASSERT_EQ(operationRowdataCounts["DELETE"], expectedRowsPerSplit);
}

TEST_F(IcebergChangelogE2ETest, orderAndGroup) {
  auto batches = createChangelogTestData();
  ASSERT_FALSE(batches.empty());

  auto outputDirectory = test::TempDirectoryPath::create();
  auto dataSink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  dataSink->close();

  auto files = listFiles(outputDirectory->getPath());
  ASSERT_FALSE(files.empty());
  std::string dataFilePath = files[0];

  auto dataRowType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto changelogOutputType = createChangelogOutputType(dataRowType);
  auto changelogColumnHandles = createChangelogColumnHandles(dataRowType);
  auto tableHandle = createChangelogTableHandle(dataRowType);

  int32_t expectedRowsPerSplit = 0;
  for (const auto& batch : batches) {
    expectedRowsPerSplit += batch->size();
  }

  // Test 1: Order by ordinal ascending
  // Create fresh splits for this query
  std::vector<std::shared_ptr<ConnectorSplit>> orderBySplits;
  orderBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300));
  orderBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100));
  orderBySplits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200));

  auto orderByPlan = PlanBuilder()
                         .startTableScan()
                         .connectorId(test::kIcebergConnectorId)
                         .outputType(changelogOutputType)
                         .tableHandle(tableHandle)
                         .assignments(changelogColumnHandles)
                         .endTableScan()
                         .orderBy({"ordinal ASC"}, false)
                         .planNode();

  auto orderByResult =
      AssertQueryBuilder(orderByPlan).splits(orderBySplits).copyResults(pool());

  ASSERT_TRUE(orderByResult != nullptr);
  ASSERT_EQ(orderByResult->size(), expectedRowsPerSplit * 3);

  // Verify ordering: first expectedRowsPerSplit rows should have ordinal=1,
  // next should have ordinal=2, last should have ordinal=3
  auto ordinalVector = orderByResult->childAt(1)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < expectedRowsPerSplit; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 1);
  }
  for (auto i = expectedRowsPerSplit; i < expectedRowsPerSplit * 2; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 2);
  }
  for (auto i = expectedRowsPerSplit * 2; i < expectedRowsPerSplit * 3; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 3);
  }

  // Test 2: Group by snapshotid, count, and order by snapshotid
  // Create fresh splits for this query
  std::vector<std::shared_ptr<ConnectorSplit>> groupByOrderBySplits;
  groupByOrderBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300));
  groupByOrderBySplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100));
  groupByOrderBySplits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200));

  auto groupByOrderByPlan = PlanBuilder()
                                .startTableScan()
                                .connectorId(test::kIcebergConnectorId)
                                .outputType(changelogOutputType)
                                .tableHandle(tableHandle)
                                .assignments(changelogColumnHandles)
                                .endTableScan()
                                .singleAggregation({"snapshotid"}, {"count(1)"})
                                .orderBy({"snapshotid ASC"}, false)
                                .planNode();

  auto groupByOrderByResult = AssertQueryBuilder(groupByOrderByPlan)
                                  .splits(groupByOrderBySplits)
                                  .copyResults(pool());

  ASSERT_TRUE(groupByOrderByResult != nullptr);
  ASSERT_EQ(groupByOrderByResult->size(), 3); // 3 different snapshot IDs

  // Verify results are ordered by snapshotid
  auto snapshotVector =
      groupByOrderByResult->childAt(0)->as<SimpleVector<int64_t>>();
  auto countVector =
      groupByOrderByResult->childAt(1)->as<SimpleVector<int64_t>>();

  ASSERT_EQ(snapshotVector->valueAt(0), 100);
  ASSERT_EQ(countVector->valueAt(0), expectedRowsPerSplit);

  ASSERT_EQ(snapshotVector->valueAt(1), 200);
  ASSERT_EQ(countVector->valueAt(1), expectedRowsPerSplit);

  ASSERT_EQ(snapshotVector->valueAt(2), 300);
  ASSERT_EQ(countVector->valueAt(2), expectedRowsPerSplit);

  // Test 3: Order by metadata column (ordinal) but select only rowdata
  std::vector<std::shared_ptr<ConnectorSplit>> orderByMetadataSplits;
  orderByMetadataSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300));
  orderByMetadataSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100));
  orderByMetadataSplits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200));

  auto orderByMetadataPlan = PlanBuilder()
                                 .startTableScan()
                                 .connectorId(test::kIcebergConnectorId)
                                 .outputType(changelogOutputType)
                                 .tableHandle(tableHandle)
                                 .assignments(changelogColumnHandles)
                                 .endTableScan()
                                 .orderBy({"ordinal ASC"}, false)
                                 .project({"rowdata"})
                                 .planNode();

  auto orderByMetadataResult = AssertQueryBuilder(orderByMetadataPlan)
                                   .splits(orderByMetadataSplits)
                                   .copyResults(pool());

  ASSERT_TRUE(orderByMetadataResult != nullptr);
  ASSERT_EQ(orderByMetadataResult->size(), expectedRowsPerSplit * 3);
  ASSERT_EQ(orderByMetadataResult->childrenSize(), 1); // Only rowdata

  // Verify we got rowdata (ordered by ordinal, but ordinal not in output)
  auto orderedRowdata = orderByMetadataResult->childAt(0)->as<RowVector>();
  ASSERT_NE(orderedRowdata, nullptr);
  ASSERT_EQ(orderedRowdata->childrenSize(), 2); // id and name

  // Test 4: Order by rowdata subfield (rowdata.id) and select only metadata
  // Project the field first, then order by it, then project only metadata
  std::vector<std::shared_ptr<ConnectorSplit>> orderByRowdataSplits;
  orderByRowdataSplits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));

  auto orderByRowdataPlan =
      PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType)
          .tableHandle(tableHandle)
          .assignments(changelogColumnHandles)
          .endTableScan()
          .project({"operation", "ordinal", "snapshotid", "rowdata.id AS id"})
          .orderBy({"id ASC"}, false)
          .project({"operation", "ordinal", "snapshotid"})
          .planNode();

  auto orderByRowdataResult = AssertQueryBuilder(orderByRowdataPlan)
                                  .splits(orderByRowdataSplits)
                                  .copyResults(pool());

  ASSERT_TRUE(orderByRowdataResult != nullptr);
  ASSERT_EQ(orderByRowdataResult->size(), expectedRowsPerSplit);
  ASSERT_EQ(orderByRowdataResult->childrenSize(), 3); // Only metadata columns

  // Verify we got metadata columns (ordered by rowdata.id, but id not in
  // output)
  auto orderedOpVector =
      orderByRowdataResult->childAt(0)->as<SimpleVector<StringView>>();
  auto orderedOrdinalVector =
      orderByRowdataResult->childAt(1)->as<SimpleVector<int64_t>>();
  auto orderedSnapshotVector =
      orderByRowdataResult->childAt(2)->as<SimpleVector<int64_t>>();

  // All rows should have same metadata since they're from same split
  for (auto i = 0; i < orderByRowdataResult->size(); i++) {
    ASSERT_EQ(orderedOpVector->valueAt(i), StringView("INSERT"));
    ASSERT_EQ(orderedOrdinalVector->valueAt(i), 1);
    ASSERT_EQ(orderedSnapshotVector->valueAt(i), 100);
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
