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
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

class IcebergChangelogE2ETest : public test::IcebergTestBase {
 protected:
  static constexpr int32_t kDefaultNumBatches = 2;
  static constexpr int32_t kDefaultRowsPerBatch = 100;

  void SetUp() override {
    IcebergTestBase::SetUp();

    dataRowType_ = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});
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
    return exec::test::PlanBuilder()
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
      {ChangelogOperation::kInsert, kChangelogOpInsert},
      {ChangelogOperation::kDelete, kChangelogOpDelete},
      {ChangelogOperation::kUpdateBefore, kChangelogOpUpdateBefore},
      {ChangelogOperation::kUpdateAfter, kChangelogOpUpdateAfter},
  };

  auto plan = makeChangelogScanPlan();

  for (const auto& [operation, operationStr] : operations) {
    auto changelogSplit =
        makeChangelogSplit(dataFilePath, operation, 1, 12'345);

    auto resultVector = exec::test::AssertQueryBuilder(plan)
                            .split(changelogSplit)
                            .copyResults(pool());

    ASSERT_NE(resultVector, nullptr);
    ASSERT_EQ(resultVector->size(), expectedRows());
    ASSERT_EQ(
        *std::dynamic_pointer_cast<const RowType>(resultVector->type()),
        *changelogOutputType_);

    for (auto i = 0; i < resultVector->size(); i++) {
      verifyChangelogRecord(resultVector, i, operationStr, 1, 12'345);
    }
  }
}

TEST_F(IcebergChangelogE2ETest, multipleSnapshots) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::kUpdateBefore, 2, 200));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::kUpdateAfter, 3, 200));

  auto plan = makeChangelogScanPlan();

  auto resultVector =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

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
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 5, 99'999);

  auto metadataOnlyType =
      ROW({std::string(kChangelogColOperation),
           std::string(kChangelogColOrdinal),
           std::string(kChangelogColSnapshotId)},
          {VARCHAR(), BIGINT(), BIGINT()});

  // Use only the three metadata handles from makeChangelogColumnHandles.
  auto allHandles = makeChangelogColumnHandles(dataRowType_);
  ColumnHandleMap metadataHandles;
  metadataHandles[std::string(kChangelogColOperation)] =
      allHandles.at(std::string(kChangelogColOperation));
  metadataHandles[std::string(kChangelogColOrdinal)] =
      allHandles.at(std::string(kChangelogColOrdinal));
  metadataHandles[std::string(kChangelogColSnapshotId)] =
      allHandles.at(std::string(kChangelogColSnapshotId));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(metadataOnlyType)
                  .tableHandle(tableHandle_)
                  .assignments(metadataHandles)
                  .endTableScan()
                  .planNode();

  auto resultVector = exec::test::AssertQueryBuilder(plan)
                          .split(changelogSplit)
                          .copyResults(pool());

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
    ASSERT_EQ(snapshotVector->valueAt(i), 99'999);
  }
}

// Verifies that a snapshotid filter baked into the table handle is applied at
// split-level: only the split with snapshotid=200 contributes rows.
TEST_F(IcebergChangelogE2ETest, filterOnSnapshotIdThroughHandle) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100));
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 200));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::kUpdateAfter, 3, 300));

  // Build a table handle with snapshotid = 200 baked in as a filter.
  common::SubfieldFilters handleFilters;
  handleFilters[common::Subfield(std::string(kChangelogColSnapshotId))] =
      std::make_shared<common::BigintRange>(200, 200, false);
  auto filteredHandle =
      makeChangelogTableHandle(dataRowType_, std::move(handleFilters));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(filteredHandle)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .planNode();

  auto resultVector =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(resultVector, nullptr);
  // Only the DELETE split (snapshotid=200) should pass.
  ASSERT_EQ(resultVector->size(), expectedRows());

  auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();
  for (auto i = 0; i < resultVector->size(); i++) {
    ASSERT_EQ(snapshotVector->valueAt(i), 200);
  }
}

TEST_F(IcebergChangelogE2ETest, filterOnMetadataColumns) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100));
  splits.push_back(
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 100));
  splits.push_back(makeChangelogSplit(
      dataFilePath, ChangelogOperation::kUpdateAfter, 3, 200));

  // Filter: operation = 'INSERT' OR ordinal > 2
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .filter("operation = 'INSERT' OR ordinal > 2")
                  .planNode();

  auto resultVector =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

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
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 12'345);

  // Filter on nested rowdata column: rowdata.id < 50
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .filter("rowdata.id < 50")
                  .planNode();

  auto resultVector = exec::test::AssertQueryBuilder(plan)
                          .split(changelogSplit)
                          .copyResults(pool());

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
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 12'345);

  // Subfield filter pushdown on rowdata.* is not supported for changelog
  // queries — verify that it throws VeloxUserError rather than silently
  // dropping the predicate and returning incorrect results.
  common::SubfieldFilters filters;
  filters[common::Subfield("rowdata.id")] =
      exec::lessThan(static_cast<int64_t>(50));
  auto tableHandle = makeChangelogTableHandle(dataRowType_, std::move(filters));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .split(changelogSplit)
          .copyResults(pool()),
      "Subfield filter pushdown on 'rowdata' is not supported for changelog "
      "queries");
}

TEST_F(IcebergChangelogE2ETest, selectRowdataSubfieldsOnly) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit = makeChangelogSplit(
      dataFilePath, ChangelogOperation::kUpdateAfter, 7, 54'321);

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .project({"rowdata.id", "rowdata.name"})
                  .planNode();

  auto resultVector = exec::test::AssertQueryBuilder(plan)
                          .split(changelogSplit)
                          .copyResults(pool());

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

TEST_F(IcebergChangelogE2ETest, countAllRows) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 3, 200),
  };

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .singleAggregation({}, {"count(1)"})
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 1);
  auto countVector = result->childAt(0)->as<SimpleVector<int64_t>>();
  ASSERT_EQ(countVector->valueAt(0), expectedRows() * 3);
}

TEST_F(IcebergChangelogE2ETest, groupByOperation) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 3, 200),
  };

  exec::test::AssertQueryBuilder(
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType_)
          .tableHandle(tableHandle_)
          .assignments(changelogColumnHandles_)
          .endTableScan()
          .singleAggregation({"operation"}, {"count(1)"})
          .planNode())
      .splits(splits)
      .assertResults(makeRowVector(
          {"operation", "count(1)"},
          {makeFlatVector<std::string>(
               {std::string(kChangelogOpInsert),
                std::string(kChangelogOpDelete),
                std::string(kChangelogOpUpdateAfter)}),
           makeFlatVector<int64_t>(
               {expectedRows(), expectedRows(), expectedRows()})}));
}

TEST_F(IcebergChangelogE2ETest, groupByRowdataNestedColumn) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
  };

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .project({"rowdata.id AS id"})
                  .singleAggregation({"id"}, {"count(1)"})
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

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
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 100),
  };

  exec::test::AssertQueryBuilder(
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType_)
          .tableHandle(tableHandle_)
          .assignments(changelogColumnHandles_)
          .endTableScan()
          .singleAggregation({"operation"}, {"count(rowdata)"})
          .planNode())
      .splits(splits)
      .assertResults(makeRowVector(
          {"operation", "count(rowdata)"},
          {makeFlatVector<std::string>(
               {std::string(kChangelogOpInsert),
                std::string(kChangelogOpDelete)}),
           makeFlatVector<int64_t>({expectedRows(), expectedRows()})}));
}

TEST_F(IcebergChangelogE2ETest, orderByOrdinal) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 2, 200),
  };

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .orderBy({"ordinal ASC"}, false)
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows() * 3);

  auto ordinalVector = result->childAt(1)->as<SimpleVector<int64_t>>();

  const int32_t rowsPerSplit = expectedRows();
  for (auto i = 0; i < rowsPerSplit; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 1);
  }
  for (auto i = rowsPerSplit; i < rowsPerSplit * 2; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 2);
  }
  for (auto i = rowsPerSplit * 2; i < rowsPerSplit * 3; i++) {
    ASSERT_EQ(ordinalVector->valueAt(i), 3);
  }
}

TEST_F(IcebergChangelogE2ETest, groupBySnapshotIdOrderBySnapshotId) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 2, 200),
  };

  exec::test::AssertQueryBuilder(
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
          .outputType(changelogOutputType_)
          .tableHandle(tableHandle_)
          .assignments(changelogColumnHandles_)
          .endTableScan()
          .singleAggregation({"snapshotid"}, {"count(1)"})
          .orderBy({"snapshotid ASC"}, false)
          .planNode())
      .splits(splits)
      .assertResults(makeRowVector(
          {"snapshotid", "count(1)"},
          {makeFlatVector<int64_t>({100, 200, 300}),
           makeFlatVector<int64_t>(
               {expectedRows(), expectedRows(), expectedRows()})}));
}

TEST_F(IcebergChangelogE2ETest, orderByOrdinalSelectRowdata) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 3, 300),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 1, 100),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 2, 200),
  };

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .orderBy({"ordinal ASC"}, false)
                  .project({"rowdata"})
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

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
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
  };

  auto plan =
      exec::test::PlanBuilder()
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

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

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

// Verifies that dynamic filters injected at runtime by HashProbe on a
// changelog constant column (snapshotid) are applied at split-skipping time.
//
// HashProbe sets canReplaceWithDynamicFilter_ (single unique key, no
// build-side output columns, no post-join filter), which means it pushes the
// dynamic filter to the scan AND skips the join key comparison entirely —
// passing all probe rows through fillOutput() on the assumption the scan
// filtered them. IcebergDataSource::addDynamicFilter() intercepts the filter,
// accumulates it in changelogDynamicFilters_, and IcebergChangelogSplitReader
// evaluates it in prepareSplit() to skip non-matching splits.
TEST_F(IcebergChangelogE2ETest, dynamicFilterOnSnapshotIdApplied) {
  std::string dataFilePath = writeTestFile();

  // Three splits with distinct snapshotids.
  std::vector<std::shared_ptr<ConnectorSplit>> changelogSplits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 200),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 3, 300),
  };

  // Build side: a single-row table with snapshotid=100. The HashJoin will
  // inject a dynamic filter snapshotid IN (100) into the probe scan.
  auto buildVector =
      makeRowVector({"build_snapshotid"}, {makeFlatVector<int64_t>({100})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId changelogScanId;

  auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .capturePlanNodeId(changelogScanId)
                  .hashJoin(
                      {"snapshotid"},
                      {"build_snapshotid"},
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values({buildVector})
                          .planNode(),
                      /*filter=*/"",
                      {"snapshotid"})
                  .planNode();

  auto result = exec::test::AssertQueryBuilder(plan)
                    .splits(changelogScanId, changelogSplits)
                    .copyResults(pool());

  ASSERT_NE(result, nullptr);
  // With the fix: only the snapshotid=100 split passes the dynamic filter.
  ASSERT_EQ(result->size(), expectedRows());
  auto snapshotVector = result->childAt(0)->as<SimpleVector<int64_t>>();
  for (auto i = 0; i < result->size(); i++) {
    ASSERT_EQ(snapshotVector->valueAt(i), 100);
  }
}

// Verifies that dynamic filters from two sequential hash joins on the same
// changelog column (snapshotid) are merged (ANDed) and applied correctly.
//
// Two downstream HashProbe operators each push a dynamic filter onto the
// changelog scan's snapshotid channel.  addDynamicFilter() is called twice for
// the same subfield: the first call inserts it; the second hits the merge path
// (Filter::merge) so the accumulated filter is the intersection of both.
// Build1 = {100, 200}, build2 = {200, 300} → only snapshotid=200 survives.
TEST_F(IcebergChangelogE2ETest, mergedDynamicFiltersOnSnapshotId) {
  std::string dataFilePath = writeTestFile();

  std::vector<std::shared_ptr<ConnectorSplit>> changelogSplits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 200),
      makeChangelogSplit(
          dataFilePath, ChangelogOperation::kUpdateAfter, 3, 300),
  };

  // Build side 1: {100, 200} — pushes snapshotid IN (100, 200).
  auto build1 =
      makeRowVector({"b1_snapshotid"}, {makeFlatVector<int64_t>({100, 200})});
  // Build side 2: {200, 300} — pushes snapshotid IN (200, 300).
  auto build2 =
      makeRowVector({"b2_snapshotid"}, {makeFlatVector<int64_t>({200, 300})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId changelogScanId;

  // Two chained joins on snapshotid with no build-side output columns, so
  // canReplaceWithDynamicFilter_ fires on both and both filters are pushed.
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(tableHandle_)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .capturePlanNodeId(changelogScanId)
                  .hashJoin(
                      {"snapshotid"},
                      {"b1_snapshotid"},
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values({build1})
                          .planNode(),
                      /*filter=*/"",
                      {"snapshotid"})
                  .hashJoin(
                      {"snapshotid"},
                      {"b2_snapshotid"},
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values({build2})
                          .planNode(),
                      /*filter=*/"",
                      {"snapshotid"})
                  .planNode();

  auto result = exec::test::AssertQueryBuilder(plan)
                    .splits(changelogScanId, changelogSplits)
                    .copyResults(pool());

  ASSERT_NE(result, nullptr);
  // Intersection of {100,200} and {200,300} = {200}: only the DELETE split.
  ASSERT_EQ(result->size(), expectedRows());
  auto snapshotVector = result->childAt(0)->as<SimpleVector<int64_t>>();
  for (auto i = 0; i < result->size(); i++) {
    ASSERT_EQ(snapshotVector->valueAt(i), 200);
  }
}

// Verifies that a subfield filter rooted on a column that is not one of the
// four changelog columns (operation/ordinal/snapshotid/rowdata) is rejected
// with a clear VeloxUserError at DataSource construction time.
// Such a filter (e.g. on "$path") would otherwise pass the old rowdata-only
// denylist, get forwarded into the base-table scan spec, and crash with a
// confusing "Field not found" error.
TEST_F(IcebergChangelogE2ETest, unknownColumnFilterIsRejected) {
  std::string dataFilePath = writeTestFile();

  // Bake a filter on "$path" — not one of the four changelog columns —
  // directly into the table handle.
  common::SubfieldFilters filters;
  filters[common::Subfield("$path")] = std::make_shared<common::BytesValues>(
      std::vector<std::string>{"nonexistent/path"}, false);
  auto filteredHandle =
      makeChangelogTableHandle(dataRowType_, std::move(filters));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(filteredHandle)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .planNode();

  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100);

  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .split(changelogSplit)
          .copyResults(pool()),
      "Subfield filter pushdown on '$path' is not supported for changelog "
      "queries");
}

TEST_F(IcebergChangelogE2ETest, rowdataRemainingFilterSilentlyDropped) {
  std::string dataFilePath = writeTestFile();
  auto changelogSplit =
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 12'345);

  // Compile "rowdata.id < 50" against the changelog output type so that
  // extractFiltersFromRemainingFilter can parse it as a subfield filter.
  auto parsedExpr =
      parse::DuckSqlExpressionsParser().parseExpr("rowdata.id < 50");
  auto typedExpr =
      core::Expressions::inferTypes(parsedExpr, changelogOutputType_, pool());

  // Build a changelog table handle that carries the expression as a
  // remainingFilter (not a subfieldFilter), mimicking a scan-level pushdown.
  auto handleWithRemainingFilter = std::make_shared<IcebergTableHandle>(
      test::kIcebergConnectorId,
      "test_table",
      /*subfieldFilters=*/common::SubfieldFilters{},
      /*remainingFilter=*/typedExpr,
      /*dataColumns=*/dataRowType_,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      test::IcebergTestBase::makeDataColumnHandles(dataRowType_));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(changelogOutputType_)
                  .tableHandle(handleWithRemainingFilter)
                  .assignments(changelogColumnHandles_)
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .split(changelogSplit)
          .copyResults(pool()),
      "Subfield filter pushdown on 'rowdata' is not supported for changelog "
      "queries");
}

TEST_F(IcebergChangelogE2ETest, filterOnlyColumnMissingFromReaderOutputType) {
  std::string dataFilePath = writeTestFile();

  // Two splits with distinct snapshotids; only snapshotid=200 should survive.
  std::vector<std::shared_ptr<ConnectorSplit>> splits = {
      makeChangelogSplit(dataFilePath, ChangelogOperation::kInsert, 1, 100),
      makeChangelogSplit(dataFilePath, ChangelogOperation::kDelete, 2, 200),
  };

  // Project only 'operation' — snapshotid is NOT in the output type.
  auto opOnlyType = ROW({{"operation", VARCHAR()}});
  auto opOnlyHandles =
      ColumnHandleMap{{"operation", changelogColumnHandles_.at("operation")}};

  // A non-extractable remaining filter that references snapshotid:
  // cast(snapshotid as varchar) cannot be reduced to a simple subfield filter
  // by ExprToSubfieldFilterParser, so it stays as a remainingFilter and causes
  // FileDataSource to append snapshotid to readerOutputType_ as a filter-only
  // column.
  auto parsedExpr = parse::DuckSqlExpressionsParser().parseExpr(
      "cast(snapshotid as varchar) = '200'");
  auto typedExpr =
      core::Expressions::inferTypes(parsedExpr, changelogOutputType_, pool());

  auto handleWithRemainingFilter = std::make_shared<IcebergTableHandle>(
      test::kIcebergConnectorId,
      "test_table",
      /*subfieldFilters=*/common::SubfieldFilters{},
      /*remainingFilter=*/typedExpr,
      /*dataColumns=*/dataRowType_,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      test::IcebergTestBase::makeDataColumnHandles(dataRowType_));

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(opOnlyType)
                  .tableHandle(handleWithRemainingFilter)
                  .assignments(opOnlyHandles)
                  .endTableScan()
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  ASSERT_NE(result, nullptr);
  // Only the DELETE split (snapshotid=200) passes cast(snapshotid as
  // varchar)='200'.
  ASSERT_EQ(result->size(), expectedRows());

  auto operationVector = result->childAt(0)->as<SimpleVector<StringView>>();
  for (auto i = 0; i < result->size(); i++) {
    ASSERT_EQ(operationVector->valueAt(i), StringView(kChangelogOpDelete));
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
