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

#include <folly/init/Init.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/lakehouse/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::lakehouse::iceberg::test {
class IcebergInsertTest : public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    rowType_ =
        ROW({"c1", "c2", "c3", "c4", "c5", "c6"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             DECIMAL(18, 5),
             BOOLEAN(),
             VARCHAR()});
  }
};

TEST_F(IcebergInsertTest, testIcebergTableWrite) {
  const auto outputDirectory = exec::test::TempDirectoryPath::create();
  const auto dataPath = fmt::format("{}", outputDirectory->getPath());
  constexpr int32_t numBatches = 10;
  constexpr int32_t vectorSize = 5'000;
  const auto vectors = createTestData(numBatches, vectorSize, 0.5);
  auto dataSink =
      createIcebergDataSink(rowType_, outputDirectory->getPath(), {});

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  const auto commitTasks = dataSink->close();
  createDuckDbTable(vectors);
  auto splits = createSplitsForDirectory(dataPath);
  ASSERT_EQ(splits.size(), commitTasks.size());
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

TEST_F(IcebergInsertTest, testSingleColumnAsPartition) {
  for (auto colIndex = 0; colIndex < rowType_->size() - 1; colIndex++) {
    const auto& colName = rowType_->nameOf(colIndex);
    const auto colType = rowType_->childAt(colIndex);

    const bool isDecimal = colType->isDecimal();
    const bool isVarbinary = colType->isVarbinary();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto outputDirectory = exec::test::TempDirectoryPath::create();

    if (isDecimal || isVarbinary) {
      const auto vectors = createTestData(numBatches, vectorSize, 0.5);
      std::vector<std::string> partitionTransforms = {colName};
      auto dataSink = createIcebergDataSink(
          rowType_, outputDirectory->getPath(), partitionTransforms);
      for (const auto& vector : vectors) {
        if (isDecimal) {
          VELOX_ASSERT_THROW(
              dataSink->appendData(vector),
              "Partition on decimal column is not supported yet.");
        } else if (isVarbinary) {
          VELOX_ASSERT_THROW(
              dataSink->appendData(vector),
              "Partition on varbinary column is not supported yet.");
        }
      }
      continue;
    }
    const auto dataPath = fmt::format("{}", outputDirectory->getPath());
    const auto vectors = createTestData(numBatches, vectorSize, 0.5);
    std::vector<std::string> partitionTransforms = {colName};
    auto dataSink = createIcebergDataSink(
        rowType_, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    createDuckDbTable(vectors);
    auto splits = createSplitsForDirectory(dataPath);

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      ASSERT_TRUE(taskJson.count("partitionDataJson") > 0);
      ASSERT_FALSE(taskJson["partitionDataJson"].empty());
    }

    connector::ColumnHandleMap assignments;
    for (auto i = 0; i < rowType_->size(); i++) {
      const auto& name = rowType_->nameOf(i);
      if (i != colIndex) {
        assignments.insert(
            {name,
             std::make_shared<common::HiveColumnHandle>(
                 name,
                 common::HiveColumnHandle::ColumnType::kRegular,
                 rowType_->childAt(i),
                 rowType_->childAt(i))});
      }
    }

    // Add partition column.
    assignments.insert(
        {colName,
         std::make_shared<common::HiveColumnHandle>(
             colName,
             common::HiveColumnHandle::ColumnType::kPartitionKey,
             rowType_->childAt(colIndex),
             rowType_->childAt(colIndex))});

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .tableScan(rowType_, {}, "", nullptr, assignments)
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

TEST_F(IcebergInsertTest, testPartitionNullColumn) {
  for (auto colIndex = 0; colIndex < rowType_->size() - 1; colIndex++) {
    const auto& colName = rowType_->nameOf(colIndex);
    const auto colType = rowType_->childAt(colIndex);
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    const auto dataPath = fmt::format("{}", outputDirectory->getPath());
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 100;
    const bool isDecimal = colType->isDecimal();
    const bool isVarbinary = colType->isVarbinary();

    if (isDecimal || isVarbinary) {
      const auto vectors = createTestData(numBatches, vectorSize, 0.5);
      std::vector<std::string> partitionTransforms = {colName};
      auto dataSink = createIcebergDataSink(
          rowType_, outputDirectory->getPath(), partitionTransforms);
      for (const auto& vector : vectors) {
        if (isDecimal) {
          VELOX_ASSERT_THROW(
              dataSink->appendData(vector),
              "Partition on decimal column is not supported yet.");
        } else if (isVarbinary) {
          VELOX_ASSERT_THROW(
              dataSink->appendData(vector),
              "Partition on varbinary column is not supported yet.");
        }
      }
      continue;
    }

    const auto vectors = createTestData(numBatches, vectorSize, 1.0);
    std::vector<std::string> partitionTransforms = {colName};
    auto dataSink = createIcebergDataSink(
        rowType_, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();

    auto files = listFiles(dataPath);
    ASSERT_EQ(files.size(), 1);

    for (const auto& file : files) {
      std::vector<std::string> pathComponents;
      folly::split("/", file, pathComponents);
      bool foundPartitionDir = false;
      for (const auto& component : pathComponents) {
        if (component.find('=') != std::string::npos) {
          foundPartitionDir = true;
          std::vector<std::string> parts;
          folly::split('=', component, parts);
          ASSERT_EQ(parts.size(), 2);
          ASSERT_EQ(parts[0], colName);
          ASSERT_EQ(parts[1], "null");
        }
      }
      ASSERT_TRUE(foundPartitionDir)
          << "No partition directory found in path: " << file;
    }
  }
}

TEST_F(IcebergInsertTest, testColumnCombinationsAsPartition) {
  std::vector<std::vector<int32_t>> columnCombinations = {
      {0, 1}, // BIGINT, INTEGER.
      {2, 1}, // SMALLINT, INTEGER.
      {2, 0}, // SMALLINT, BIGINT.
      {0, 2, 1} // BIGINT, SMALLINT, INTEGER.
  };

  for (const auto& combination : columnCombinations) {
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    const auto dataPath = fmt::format("{}", outputDirectory->getPath());
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(numBatches, vectorSize);
    std::vector<std::string> partitionTransforms;
    for (auto colIndex : combination) {
      partitionTransforms.push_back(rowType_->nameOf(colIndex));
    }

    auto dataSink = createIcebergDataSink(
        rowType_, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    createDuckDbTable(vectors);
    auto splits = createSplitsForDirectory(dataPath);

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    connector::ColumnHandleMap assignments;
    std::unordered_set<int32_t> partitionColumns(
        combination.begin(), combination.end());

    for (auto i = 0; i < rowType_->size(); i++) {
      const auto& name = rowType_->nameOf(i);
      auto columnType = partitionColumns.count(i) > 0
          ? common::HiveColumnHandle::ColumnType::kPartitionKey
          : common::HiveColumnHandle::ColumnType::kRegular;

      assignments.insert(
          {name,
           std::make_shared<common::HiveColumnHandle>(
               name, columnType, rowType_->childAt(i), rowType_->childAt(i))});
    }

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .tableScan(rowType_, {}, "", nullptr, assignments)
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

} // namespace facebook::velox::connector::lakehouse::iceberg::test
