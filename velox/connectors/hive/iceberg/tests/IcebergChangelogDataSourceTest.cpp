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

#include "velox/connectors/hive/iceberg/IcebergChangelogDataSource.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergChangelogSplitInfo.h"
#include "velox/connectors/hive/iceberg/IcebergDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::iceberg;

namespace facebook::velox::connector::hive::iceberg::test {

// Mock IcebergDataSource that returns predefined static row batches
class MockIcebergDataSource : public connector::DataSource {
 public:
  MockIcebergDataSource(
      const RowTypePtr& outputType,
      std::vector<RowVectorPtr> batches,
      memory::MemoryPool* pool)
      : outputType_(outputType),
        batches_(std::move(batches)),
        pool_(pool),
        currentBatch_(0) {}

  void addSplit(std::shared_ptr<ConnectorSplit> split) override {
    currentSplit_ = split;
  }

  std::optional<RowVectorPtr> next(
      uint64_t /*size*/,
      velox::ContinueFuture& /*future*/) override {
    if (currentBatch_ >= batches_.size()) {
      return std::nullopt;
    }
    return batches_[currentBatch_++];
  }

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {}

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return currentBatch_ * (batches_.empty() ? 0 : batches_[0]->size());
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override {
    return {};
  }

  bool allPrefetchIssued() const override {
    return true;
  }

  void setFromDataSource(
      std::unique_ptr<DataSource> /*sourceUnique*/) override {}

  const common::SubfieldFilters* getFilters() const override {
    return nullptr;
  }

 private:
  RowTypePtr outputType_;
  std::vector<RowVectorPtr> batches_;
  memory::MemoryPool* pool_;
  size_t currentBatch_;
  std::shared_ptr<ConnectorSplit> currentSplit_;
};

class IcebergChangelogDataSourceTest : public IcebergTestBase {
 protected:
  RowVectorPtr createTestBatch(
      const RowTypePtr& rowType,
      vector_size_t size,
      int32_t startValue) {
    std::vector<VectorPtr> children;
    for (auto i = 0; i < rowType->size(); ++i) {
      auto type = rowType->childAt(i);
      if (type->isBigint()) {
        auto flat = makeFlatVector<int64_t>(
            size, [startValue](auto row) { return startValue + row; });
        children.push_back(flat);
      } else if (type->isVarchar()) {
        std::vector<std::string> values;
        for (int j = 0; j < size; j++) {
          values.push_back("value_" + std::to_string(startValue + j));
        }
        children.push_back(makeFlatVector<std::string>(values));
      }
    }
    return makeRowVector(rowType->names(), children);
  }

  // Helper to create changelog output type
  RowTypePtr createChangelogOutputType(const RowTypePtr& dataRowType) {
    return ROW(
        {"operation", "ordinal", "snapshotid", "rowdata"},
        {VARCHAR(), BIGINT(), BIGINT(), dataRowType});
  }

  // Helper to create column handles for changelog schema
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

  // Helper to create a changelog split
  std::shared_ptr<HiveIcebergSplit> createChangelogSplit(
      const std::shared_ptr<ChangelogSplitInfo>& changelogInfo) {
    return IcebergSplitBuilder("file:///test/path")
        .connectorId("test-connector")
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .start(0)
        .length(1000)
        .changelogSplitInfo(changelogInfo)
        .build();
  }

  // Helper method to test changelog transformation for a given operation
  void testChangelogTransformation(
      ChangelogOperation operation,
      const std::string& expectedOperationStr,
      int64_t ordinal,
      int64_t snapshotId,
      vector_size_t batchSize,
      int32_t startValue) {
    auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
    auto batch = createTestBatch(dataType, batchSize, startValue);

    std::vector<RowVectorPtr> batches = {batch};
    auto mockSource =
        std::make_unique<MockIcebergDataSource>(dataType, batches, pool());

    auto outputType = createChangelogOutputType(dataType);
    auto columnHandles = createChangelogColumnHandles(dataType);
    auto changelogSource = std::make_unique<IcebergChangelogDataSource>(
        std::move(mockSource), outputType, columnHandles);

    auto changelogInfo =
        std::make_shared<ChangelogSplitInfo>(operation, ordinal, snapshotId);
    auto split = createChangelogSplit(changelogInfo);
    changelogSource->addSplit(split);

    velox::ContinueFuture future;
    auto result = changelogSource->next(1000, future);

    ASSERT_TRUE(result.has_value());
    auto resultVector = result.value();
    ASSERT_NE(resultVector, nullptr);

    // Verify schema: (operation, ordinal, snapshotid, rowdata)
    ASSERT_EQ(resultVector->childrenSize(), 4);
    ASSERT_EQ(resultVector->size(), batchSize);

    // Verify operation column
    auto operationVector =
        resultVector->childAt(0)->as<SimpleVector<StringView>>();
    for (int i = 0; i < batchSize; i++) {
      ASSERT_EQ(operationVector->valueAt(i), StringView(expectedOperationStr));
    }

    // Verify ordinal column
    auto ordinalVector = resultVector->childAt(1)->as<SimpleVector<int64_t>>();
    for (int i = 0; i < batchSize; i++) {
      ASSERT_EQ(ordinalVector->valueAt(i), ordinal);
    }

    // Verify snapshotid column
    auto snapshotVector = resultVector->childAt(2)->as<SimpleVector<int64_t>>();
    for (int i = 0; i < batchSize; i++) {
      ASSERT_EQ(snapshotVector->valueAt(i), snapshotId);
    }

    // Verify rowdata column (should be a RowVector)
    auto rowdataVector = resultVector->childAt(3)->as<RowVector>();
    ASSERT_NE(rowdataVector, nullptr);
    ASSERT_EQ(rowdataVector->childrenSize(), 2);

    auto idVector = rowdataVector->childAt(0)->as<SimpleVector<int64_t>>();
    auto nameVector = rowdataVector->childAt(1)->as<SimpleVector<StringView>>();

    for (int i = 0; i < batchSize; i++) {
      ASSERT_EQ(idVector->valueAt(i), startValue + i);
      std::string expected = "value_" + std::to_string(startValue + i);
      ASSERT_EQ(nameVector->valueAt(i), StringView(expected));
    }
  }
};

TEST_F(IcebergChangelogDataSourceTest, allOperations) {
  // Test INSERT operation
  testChangelogTransformation(
      ChangelogOperation::INSERT, "INSERT", 1L, 100L, 5, 0);

  // Test DELETE operation
  testChangelogTransformation(
      ChangelogOperation::DELETE, "DELETE", 2L, 200L, 3, 10);

  // Test UPDATE_BEFORE operation
  testChangelogTransformation(
      ChangelogOperation::UPDATE_BEFORE, "UPDATE_BEFORE", 3L, 300L, 2, 20);

  // Test UPDATE_AFTER operation
  testChangelogTransformation(
      ChangelogOperation::UPDATE_AFTER, "UPDATE_AFTER", 4L, 400L, 2, 30);
}

TEST_F(IcebergChangelogDataSourceTest, multipleBatches) {
  auto dataType = ROW({"id"}, {BIGINT()});

  // Create multiple batches
  auto batch1 = createTestBatch(dataType, 3, 0);
  auto batch2 = createTestBatch(dataType, 2, 10);
  auto batch3 = createTestBatch(dataType, 4, 20);

  std::vector<RowVectorPtr> batches = {batch1, batch2, batch3};
  auto mockSource =
      std::make_unique<MockIcebergDataSource>(dataType, batches, pool());

  auto outputType = createChangelogOutputType(dataType);
  auto columnHandles = createChangelogColumnHandles(dataType);
  auto changelogSource = std::make_unique<IcebergChangelogDataSource>(
      std::move(mockSource), outputType, columnHandles);

  auto changelogInfo = std::make_shared<ChangelogSplitInfo>(
      ChangelogOperation::INSERT, 1L, 100L);
  auto split = createChangelogSplit(changelogInfo);
  changelogSource->addSplit(split);

  // Read first batch
  velox::ContinueFuture future;
  auto result1 = changelogSource->next(1000, future);
  ASSERT_TRUE(result1.has_value());
  ASSERT_EQ(result1.value()->size(), 3);

  // Read second batch
  auto result2 = changelogSource->next(1000, future);
  ASSERT_TRUE(result2.has_value());
  ASSERT_EQ(result2.value()->size(), 2);

  // Read third batch
  auto result3 = changelogSource->next(1000, future);
  ASSERT_TRUE(result3.has_value());
  ASSERT_EQ(result3.value()->size(), 4);

  // No more data
  auto result4 = changelogSource->next(1000, future);
  ASSERT_FALSE(result4.has_value());
}

TEST_F(IcebergChangelogDataSourceTest, emptyBatch) {
  auto dataType = ROW({"id"}, {BIGINT()});

  // Empty batches
  std::vector<RowVectorPtr> batches = {};
  auto mockSource =
      std::make_unique<MockIcebergDataSource>(dataType, batches, pool());

  auto outputType = createChangelogOutputType(dataType);
  auto columnHandles = createChangelogColumnHandles(dataType);
  auto changelogSource = std::make_unique<IcebergChangelogDataSource>(
      std::move(mockSource), outputType, columnHandles);

  auto changelogInfo = std::make_shared<ChangelogSplitInfo>(
      ChangelogOperation::INSERT, 1L, 100L);
  auto split = createChangelogSplit(changelogInfo);
  changelogSource->addSplit(split);

  // Should return nullopt immediately
  velox::ContinueFuture future;
  auto result = changelogSource->next(1000, future);
  ASSERT_FALSE(result.has_value());
}

TEST_F(IcebergChangelogDataSourceTest, delegatedMethods) {
  auto dataType = ROW({"id"}, {BIGINT()});
  auto batch = createTestBatch(dataType, 3, 0);

  std::vector<RowVectorPtr> batches = {batch};
  auto mockSource =
      std::make_unique<MockIcebergDataSource>(dataType, batches, pool());

  auto outputType = createChangelogOutputType(dataType);
  auto columnHandles = createChangelogColumnHandles(dataType);
  auto changelogSource = std::make_unique<IcebergChangelogDataSource>(
      std::move(mockSource), outputType, columnHandles);

  auto changelogInfo = std::make_shared<ChangelogSplitInfo>(
      ChangelogOperation::INSERT, 1L, 100L);
  auto split = createChangelogSplit(changelogInfo);
  changelogSource->addSplit(split);

  // Test delegated methods before reading data
  ASSERT_EQ(changelogSource->getCompletedBytes(), 0);
  ASSERT_EQ(changelogSource->getCompletedRows(), 0);
  ASSERT_TRUE(changelogSource->allPrefetchIssued());
  ASSERT_TRUE(changelogSource->getRuntimeStats().empty());

  // Read data to update completed rows
  velox::ContinueFuture future;
  auto result = changelogSource->next(1000, future);
  ASSERT_TRUE(result.has_value());

  // After reading, completed rows should be updated
  ASSERT_EQ(changelogSource->getCompletedRows(), 3);
}

} // namespace facebook::velox::connector::hive::iceberg::test
