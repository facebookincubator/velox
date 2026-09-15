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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;

namespace facebook::velox::connector::hive::iceberg {

namespace {

class IcebergChangelogE2ETest : public test::IcebergTestBase {
 protected:
  static constexpr int32_t kDefaultNumBatches = 2;
  static constexpr int32_t kDefaultRowsPerBatch = 100;

  void SetUp() override {
    IcebergTestBase::SetUp();

    dataRowType_ = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
    changelogOutputType_ = makeChangelogOutputType(dataRowType_);
    changelogColumnHandles_ = makeChangelogColumnHandles(dataRowType_);
    tableHandle_ = makeChangelogTableHandle(dataRowType_);
  }

  /// Creates kDefaultNumBatches * kDefaultRowsPerBatch rows with sequential
  /// {id, name} values, writes them to a temp directory, and returns the
  /// single resulting file path.
  std::string writeTestFile() {
    auto batches = makeTestBatches();
    outputDirectory_ = test::TempDirectoryPath::create();
    auto dataSink =
        createDataSinkAndAppendData(batches, outputDirectory_->getPath(), {});
    dataSink->close();
    return getOnlyDataFilePath(outputDirectory_->getPath());
  }

  /// Returns the total number of rows in makeTestBatches().
  int32_t expectedRows() const {
    return kDefaultNumBatches * kDefaultRowsPerBatch;
  }

  core::PlanNodePtr makeChangelogScanPlan() {
    return PlanBuilder()
        .startTableScan()
        .connectorId(test::kIcebergConnectorId)
        .outputType(changelogOutputType_)
        .tableHandle(tableHandle_)
        .assignments(changelogColumnHandles_)
        .endTableScan()
        .planNode();
  }

  void verifyChangelogRecord(
      const RowVectorPtr& resultVector,
      int32_t rowIndex,
      std::string_view expectedOperation,
      int64_t expectedOrdinal,
      int64_t expectedSnapshotId) {
    auto operationVector =
        resultVector->childAt(0)->as<SimpleVector<StringView>>();
    ASSERT_EQ(
        operationVector->valueAt(rowIndex), StringView(expectedOperation));

    auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();
    ASSERT_EQ(ordinalVector->valueAt(rowIndex), expectedOrdinal);

    auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();
    ASSERT_EQ(snapshotVector->valueAt(rowIndex), expectedSnapshotId);
  }

  RowTypePtr dataRowType_;
  RowTypePtr changelogOutputType_;
  ColumnHandleMap changelogColumnHandles_;
  std::shared_ptr<IcebergTableHandle> tableHandle_;
  std::shared_ptr<test::TempDirectoryPath> outputDirectory_;
};

TEST_F(IcebergChangelogE2ETest, differentOperations) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::pair<ChangelogOperation, std::string_view>> operations = {
      {ChangelogOperation::INSERT, kChangelogOpInsert},
      {ChangelogOperation::DELETE, kChangelogOpDelete},
      {ChangelogOperation::UPDATE_BEFORE, kChangelogOpUpdateBefore},
      {ChangelogOperation::UPDATE_AFTER, kChangelogOpUpdateAfter}};

  auto plan = makeChangelogScanPlan();

  for (const auto& [operation, operationStr] : operations) {
    auto changelogSplit = makeChangelogSplit(dataFilePath, operation, 1, 12345);

    auto resultVector =
        AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

    ASSERT_NE(resultVector, nullptr);
    ASSERT_EQ(resultVector->size(), expectedRows());
    ASSERT_EQ(
        *std::dynamic_pointer_cast<const RowType>(resultVector->type()),
        *changelogOutputType_);

    for (auto i = 0; i < resultVector->size(); i++) {
      verifyChangelogRecord(resultVector, i, operationStr, 1, 12345);
    }
  }
}

TEST_F(IcebergChangelogE2ETest, multipleSnapshots) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_BEFORE, 2, 200));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  auto plan = makeChangelogScanPlan();

  auto resultVector =
      AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  ASSERT_EQ(
      *std::dynamic_pointer_cast<const RowType>(resultVector->type()),
      *changelogOutputType_);

  const int32_t rowsPerSplit = expectedRows();
  ASSERT_EQ(resultVector->size(), rowsPerSplit * 3);

  for (auto i = 0; i < rowsPerSplit; i++) {
    verifyChangelogRecord(resultVector, i, kChangelogOpInsert, 1, 100);
  }
  for (auto i = rowsPerSplit; i < rowsPerSplit * 2; i++) {
    verifyChangelogRecord(resultVector, i, kChangelogOpUpdateBefore, 2, 200);
  }
  for (auto i = rowsPerSplit * 2; i < rowsPerSplit * 3; i++) {
    verifyChangelogRecord(resultVector, i, kChangelogOpUpdateAfter, 3, 200);
  }
}

TEST_F(IcebergChangelogE2ETest, selectMetadataColumnsOnly) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 5, 99999);

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

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(metadataOnlyType)
                  .tableHandle(tableHandle_)
                  .assignments(metadataHandles)
                  .endTableScan()
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  ASSERT_EQ(resultVector->childrenSize(), 3);
  ASSERT_EQ(resultVector->size(), expectedRows());

  auto operationVector =
      resultVector->childAt(0)->as<SimpleVector<StringView>>();
  auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();
  auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_EQ(operationVector->valueAt(i), StringView(kChangelogOpInsert));
    ASSERT_EQ(ordinalVector->valueAt(i), 5);
    ASSERT_EQ(snapshotVector->valueAt(i), 99999);
  }
}

TEST_F(IcebergChangelogE2ETest, filterOnMetadataColumns) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100));
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200));

  // Filter: operation = 'INSERT' OR ordinal > 2
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .filter("operation = 'INSERT' OR ordinal > 2")
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  // Should get rows from split 1 (INSERT) and split 3 (ordinal=3).
  ASSERT_EQ(resultVector->size(), expectedRows() * 2);

  auto operationVector =
      resultVector->childAt(0)->as<SimpleVector<StringView>>();
  auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_TRUE(
        operationVector->valueAt(i) == StringView(kChangelogOpInsert) ||
        ordinalVector->valueAt(i) > 2);
  }
}

TEST_F(IcebergChangelogE2ETest, filterOnRowdataNestedColumns) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 12345);

  // Filter on nested rowdata column: rowdata.id < 50
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .filter("rowdata.id < 50")
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  ASSERT_GT(resultVector->size(), 0);
  ASSERT_LT(resultVector->size(), expectedRows());

  auto rowdataVector = resultVector->childAt(3)->as<RowVector>();
  ASSERT_NE(rowdataVector, nullptr);
  auto idVector = rowdataVector->childAt(0)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_LT(idVector->valueAt(i), 50);
  }
}

TEST_F(IcebergChangelogE2ETest, rowdataSubfieldFilterThrows) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 12345);

  // Subfield filter pushdown on rowdata.* is not supported for changelog
  // queries — verify that it throws VeloxUserError rather than silently
  // dropping the predicate and returning incorrect results.
  common::SubfieldFilters filters;
  filters[common::Subfield("rowdata.id")] = lessThan(static_cast<int64_t>(50));
  auto tableHandle = makeChangelogTableHandle(dataRowType_, std::move(filters));

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool()),
      "Subfield filter pushdown on rowdata columns is not supported for "
      "changelog queries");
}

TEST_F(IcebergChangelogE2ETest, selectRowdataSubfieldsOnly) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit = makeChangelogSplit(
      dataFilePath, ChangelogOperation::UPDATE_AFTER, 7, 54321);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .project({"rowdata.id", "rowdata.name"})
                  .planNode();

  auto resultVector =
      AssertQueryBuilder(plan).split(changelogSplit).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  ASSERT_EQ(resultVector->childrenSize(), 2);
  ASSERT_EQ(resultVector->size(), expectedRows());

  auto rowType = std::dynamic_pointer_cast<const RowType>(resultVector->type());
  ASSERT_EQ(rowType->nameOf(0), "id");
  ASSERT_EQ(rowType->nameOf(1), "name");

  auto idVector = resultVector->childAt(0)->as<SimpleVector<int64_t>>();
  auto nameVector = resultVector->childAt(1)->as<SimpleVector<StringView>>();

  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_FALSE(idVector->isNullAt(i));
    ASSERT_FALSE(nameVector->isNullAt(i));
    ASSERT_EQ(idVector->valueAt(i), i);
    const std::string expectedName = "name_" + std::to_string(i);
    ASSERT_EQ(nameVector->valueAt(i), StringView(expectedName));
  }
}

// --- Aggregation tests -------------------------------------------------------

TEST_F(IcebergChangelogE2ETest, countAllRows) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .singleAggregation({}, {"count(1)"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 1);
  auto countVector = result->childAt(0)->as<SimpleVector<int64_t>>();
  ASSERT_EQ(countVector->valueAt(0), expectedRows() * 3);
}

TEST_F(IcebergChangelogE2ETest, groupByOperation) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::UPDATE_AFTER, 3, 200),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .singleAggregation({"operation"}, {"count(1)"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 3);

  auto operationVector = result->childAt(0)->as<SimpleVector<StringView>>();
  auto groupCountVector = result->childAt(1)->as<SimpleVector<int64_t>>();

  std::unordered_map<std::string, int64_t> counts;
  for (auto i = 0; i < result->size(); i++) {
    std::string op(
        operationVector->valueAt(i).data(), operationVector->valueAt(i).size());
    counts[op] = groupCountVector->valueAt(i);
  }

  ASSERT_EQ(counts[std::string(kChangelogOpInsert)], expectedRows());
  ASSERT_EQ(counts[std::string(kChangelogOpDelete)], expectedRows());
  ASSERT_EQ(counts[std::string(kChangelogOpUpdateAfter)], expectedRows());
}

TEST_F(IcebergChangelogE2ETest, groupByRowdataNestedColumn) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .project({"rowdata.id AS id"})
                  .singleAggregation({"id"}, {"count(1)"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows());

  auto idCountVector = result->childAt(1)->as<SimpleVector<int64_t>>();
  for (auto i = 0; i < result->size(); i++) {
    ASSERT_EQ(idCountVector->valueAt(i), 1);
  }
}

TEST_F(IcebergChangelogE2ETest, groupByOperationCountRowdata) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 2, 100),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .singleAggregation({"operation"}, {"count(rowdata)"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 2);

  auto opGroupVector = result->childAt(0)->as<SimpleVector<StringView>>();
  auto rowdataCountVector = result->childAt(1)->as<SimpleVector<int64_t>>();

  std::unordered_map<std::string, int64_t> counts;
  for (auto i = 0; i < result->size(); i++) {
    std::string op(
        opGroupVector->valueAt(i).data(), opGroupVector->valueAt(i).size());
    counts[op] = rowdataCountVector->valueAt(i);
  }

  ASSERT_EQ(counts[std::string(kChangelogOpInsert)], expectedRows());
  ASSERT_EQ(counts[std::string(kChangelogOpDelete)], expectedRows());
}

// --- Ordering tests ----------------------------------------------------------

TEST_F(IcebergChangelogE2ETest, orderByOrdinal) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .orderBy({"ordinal ASC"}, false)
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows() * 3);

  auto ordinalVector = result->childAt(1)->as<SimpleVector<int64_t>>();

  const int32_t n = expectedRows();
  for (auto i = 0; i < n; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 1);
  }
  for (auto i = n; i < n * 2; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 2);
  }
  for (auto i = n * 2; i < n * 3; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 3);
  }
}

TEST_F(IcebergChangelogE2ETest, groupBySnapshotIdOrderBySnapshotId) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .singleAggregation({"snapshotid"}, {"count(1)"})
                  .orderBy({"snapshotid ASC"}, false)
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 3);

  auto snapshotVector = result->childAt(0)->as<SimpleVector<int64_t>>();
  auto countVector = result->childAt(1)->as<SimpleVector<int64_t>>();

  ASSERT_EQ(snapshotVector->valueAt(0), 100);
  ASSERT_EQ(countVector->valueAt(0), expectedRows());

  ASSERT_EQ(snapshotVector->valueAt(1), 200);
  ASSERT_EQ(countVector->valueAt(1), expectedRows());

  ASSERT_EQ(snapshotVector->valueAt(2), 300);
  ASSERT_EQ(countVector->valueAt(2), expectedRows());
}

TEST_F(IcebergChangelogE2ETest, orderByOrdinalSelectRowdata) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::DELETE, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::UPDATE_AFTER, 2, 200),
  };

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .orderBy({"ordinal ASC"}, false)
                  .project({"rowdata"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows() * 3);
  ASSERT_EQ(result->childrenSize(), 1);

  auto orderedRowdata = result->childAt(0)->as<RowVector>();
  ASSERT_NE(orderedRowdata, nullptr);
  ASSERT_EQ(orderedRowdata->childrenSize(), 2);
}

TEST_F(IcebergChangelogE2ETest, orderByRowdataSubfieldSelectMetadata) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::INSERT, 1, 100),
  };

  auto plan =
      PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType_)
          .tableHandle(tableHandle_)
          .assignments(changelogColumnHandles_)
          .endTableScan()
          .project({"operation", "ordinal", "snapshotid", "rowdata.id AS id"})
          .orderBy({"id ASC"}, false)
          .project({"operation", "ordinal", "snapshotid"})
          .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows());
  ASSERT_EQ(result->childrenSize(), 3);

  auto orderedOpVector = result->childAt(0)->as<SimpleVector<StringView>>();
  auto orderedOrdinalVector = result->childAt(1)->as<SimpleVector<int64_t>>();
  auto orderedSnapshotVector = result->childAt(2)->as<SimpleVector<int64_t>>();

  for (auto i = 0; i < result->size(); i++) {
    ASSERT_EQ(orderedOpVector->valueAt(i), StringView(kChangelogOpInsert));
    ASSERT_EQ(orderedOrdinalVector->valueAt(i), 1);
    ASSERT_EQ(orderedSnapshotVector->valueAt(i), 100);
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
