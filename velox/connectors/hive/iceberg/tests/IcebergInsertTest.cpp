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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {
namespace {

class IcebergInsertTest : public test::IcebergTestBase {
 protected:
  void test(const RowTypePtr& rowType, double nullRatio = 0.0) {
    const auto outputDirectory = TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    constexpr int32_t numBatches = 10;
    constexpr int32_t vectorSize = 5'000;
    const auto vectors =
        createTestData(rowType, numBatches, vectorSize, nullRatio);
    auto dataSink =
        createIcebergDataSink(rowType, outputDirectory->getPath(), {});

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    createDuckDbTable(vectors);
    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_EQ(splits.size(), commitTasks.size());
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    assertQuery(plan, splits, "SELECT * FROM tmp");
  }
};

TEST_F(IcebergInsertTest, basic) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           BOOLEAN(),
           REAL(),
           DECIMAL(18, 5),
           VARCHAR(),
           VARBINARY(),
           DATE(),
           TIMESTAMP(),
           ROW({"id", "name"}, {INTEGER(), VARCHAR()})});
  test(rowType, 0.2);
}

TEST_F(IcebergInsertTest, mapAndArray) {
  auto rowType =
      ROW({"c1", "c2"}, {MAP(INTEGER(), VARCHAR()), ARRAY(VARCHAR())});
  test(rowType);
}

#ifdef VELOX_ENABLE_PARQUET
TEST_F(IcebergInsertTest, bigDecimal) {
  auto rowType = ROW({"c1"}, {DECIMAL(38, 5)});
  fileFormat_ = dwio::common::FileFormat::PARQUET;
  test(rowType);
}

TEST_F(IcebergInsertTest, maxTargetFileSizeRotation) {
  setConnectorSessionProperty(HiveConfig::kMaxTargetFileSizeSession, "4KB");

  const auto outputPath = TempDirectoryPath::create()->getPath();
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  const auto vectors = createTestData(rowType, 10, 1'000);
  auto dataSink = createIcebergDataSink(rowType, outputPath, {});

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  const auto commitTasks = dataSink->close();

  ASSERT_EQ(listFiles(outputPath).size(), 5);

  createDuckDbTable(vectors);
  auto splits = createSplitsForDirectory(outputPath);
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  assertQuery(plan, splits, "SELECT * FROM tmp");
}
#endif

TEST_F(IcebergInsertTest, testSingleColumnAsPartition) {
  auto rowType = ROW(
      {"c1", "c2", "c3", "c4", "c5", "c6"},
      {BIGINT(), INTEGER(), SMALLINT(), DECIMAL(18, 5), BOOLEAN(), VARCHAR()});
  for (auto colIndex = 0; colIndex < rowType->size() - 1; colIndex++) {
    const auto& colName = rowType->nameOf(colIndex);
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(rowType, numBatches, vectorSize, 0.5);
    std::vector<test::PartitionField> partitionTransforms = {
        {colIndex, TransformType::kIdentity, std::nullopt}};
    auto dataSink = createIcebergDataSink(
        rowType, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    createDuckDbTable(vectors);
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      ASSERT_TRUE(taskJson.count("partitionDataJson") > 0);
      ASSERT_FALSE(taskJson["partitionDataJson"].empty());
    }

    connector::ColumnHandleMap assignments;
    for (auto i = 0; i < rowType->size(); i++) {
      const auto& name = rowType->nameOf(i);
      if (i != colIndex) {
        assignments.insert(
            {name,
             std::make_shared<HiveColumnHandle>(
                 name,
                 HiveColumnHandle::ColumnType::kRegular,
                 rowType->childAt(i),
                 rowType->childAt(i))});
      }
    }

    // Add partition column.
    assignments.insert(
        {colName,
         std::make_shared<HiveColumnHandle>(
             colName,
             HiveColumnHandle::ColumnType::kPartitionKey,
             rowType->childAt(colIndex),
             rowType->childAt(colIndex))});

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .assignments(assignments)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

TEST_F(IcebergInsertTest, testPartitionNullColumn) {
  auto rowType = ROW(
      {"c1", "c2", "c3", "c4", "c5", "c6"},
      {BIGINT(), INTEGER(), SMALLINT(), DECIMAL(18, 5), BOOLEAN(), VARCHAR()});
  for (auto colIndex = 0; colIndex < rowType->size() - 1; colIndex++) {
    const auto& colName = rowType->nameOf(colIndex);
    const auto colType = rowType->childAt(colIndex);
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 100;

    const auto vectors = createTestData(rowType, numBatches, vectorSize, 1.0);
    std::vector<test::PartitionField> partitionTransforms = {
        {colIndex, TransformType::kIdentity, std::nullopt}};
    auto dataSink = createIcebergDataSink(
        rowType, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    ASSERT_EQ(1, commitTasks.size());
    auto taskJson = folly::parseJson(commitTasks.at(0));
    ASSERT_EQ(1, taskJson.count("partitionDataJson"));
    auto partitionDataStr = taskJson["partitionDataJson"].asString();
    auto partitionData = folly::parseJson(partitionDataStr);
    ASSERT_EQ(1, partitionData.count("partitionValues"));
    auto partitionValues = partitionData["partitionValues"];
    ASSERT_TRUE(partitionValues.isArray());
    ASSERT_TRUE(partitionValues[0].isNull());

    auto files = listFiles(outputDirectory->getPath());
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
  auto rowType = ROW(
      {"c1", "c2", "c3", "c4", "c5", "c6"},
      {BIGINT(), INTEGER(), SMALLINT(), DECIMAL(18, 5), BOOLEAN(), VARCHAR()});
  std::vector<std::vector<int32_t>> columnCombinations = {
      {0, 1}, // BIGINT, INTEGER.
      {2, 1}, // SMALLINT, INTEGER.
      {2, 3}, // SMALLINT, DECIMAL.
      {0, 2, 1} // BIGINT, SMALLINT, INTEGER.
  };

  for (const auto& combination : columnCombinations) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(rowType, numBatches, vectorSize);
    std::vector<test::PartitionField> partitionTransforms;
    for (auto colIndex : combination) {
      partitionTransforms.push_back(
          {colIndex, TransformType::kIdentity, std::nullopt});
    }

    auto dataSink = createIcebergDataSink(
        rowType, outputDirectory->getPath(), partitionTransforms);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    const auto commitTasks = dataSink->close();
    createDuckDbTable(vectors);
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    connector::ColumnHandleMap assignments;
    std::unordered_set<int32_t> partitionColumns(
        combination.begin(), combination.end());

    for (auto i = 0; i < rowType->size(); i++) {
      const auto& name = rowType->nameOf(i);
      auto columnType = partitionColumns.count(i) > 0
          ? HiveColumnHandle::ColumnType::kPartitionKey
          : HiveColumnHandle::ColumnType::kRegular;

      assignments.insert(
          {name,
           std::make_shared<HiveColumnHandle>(
               name, columnType, rowType->childAt(i), rowType->childAt(i))});
    }

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .assignments(assignments)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

TEST_F(IcebergInsertTest, testInfinityValues) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto realVector = makeFlatVector<float>(
      {std::numeric_limits<float>::max(),
       -std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::min(),
       std::numeric_limits<float>::lowest(),
       std::numeric_limits<float>::quiet_NaN()});

  auto doubleVector = makeFlatVector<double>(
      {std::numeric_limits<double>::max(),
       -std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::min(),
       std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::quiet_NaN()});

  auto idVector = makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5});

  auto rowType =
      ROW({"id", "real_col", "double_col"}, {BIGINT(), REAL(), DOUBLE()});
  auto vector = makeRowVector(
      {"id", "real_col", "double_col"}, {idVector, realVector, doubleVector});

  auto dataSink =
      createIcebergDataSink(rowType, outputDirectory->getPath(), {});
  dataSink->appendData(vector);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  createDuckDbTable({vector});
  auto splits = createSplitsForDirectory(outputDirectory->getPath());

  auto plan = exec::test::PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp ORDER BY id");
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
