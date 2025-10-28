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
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

class IcebergInsertTest : public test::IcebergTestBase {
 protected:
  void test(const RowTypePtr& rowType, double nullRatio = 0.0) {
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    constexpr int32_t numBatches = 10;
    constexpr int32_t vectorSize = 5'000;
    const auto vectors =
        createTestData(rowType, numBatches, vectorSize, nullRatio);
    const auto& dataSink =
        createDataSinkAndAppendData(rowType, vectors, dataPath);
    const auto commitTasks = dataSink->close();

    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_EQ(splits.size(), commitTasks.size());
    auto plan = exec::test::PlanBuilder().tableScan(rowType).planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
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
#endif

TEST_F(IcebergInsertTest, singleColumnPartition) {
  auto rowType = ROW(
      {"c1", "c2", "c3", "c4", "c5", "c6"},
      {BIGINT(), INTEGER(), SMALLINT(), DECIMAL(18, 5), BOOLEAN(), VARCHAR()});
  for (auto colIndex = 0; colIndex < rowType->size(); colIndex++) {
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(rowType, numBatches, vectorSize, 0.5);
    std::vector<test::PartitionField> partitionTransforms = {
        {colIndex, TransformType::kIdentity, std::nullopt}};
    const auto& dataSink = createDataSinkAndAppendData(
        rowType, vectors, outputDirectory->getPath(), partitionTransforms);
    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      ASSERT_TRUE(taskJson.count("partitionDataJson") > 0);
    }

    auto plan = exec::test::PlanBuilder().tableScan(rowType).planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

TEST_F(IcebergInsertTest, partitionNullColumn) {
  auto rowType = ROW(
      {"c1", "c2", "c3", "c4", "c5", "c6"},
      {BIGINT(), INTEGER(), SMALLINT(), DECIMAL(18, 5), BOOLEAN(), VARCHAR()});
  for (auto colIndex = 0; colIndex < rowType->size(); colIndex++) {
    const auto& colName = rowType->nameOf(colIndex);
    const auto colType = rowType->childAt(colIndex);
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 100;

    // nullRatio = 1.0
    const auto vectors = createTestData(rowType, numBatches, vectorSize, 1.0);

    std::vector<test::PartitionField> partitionTransforms = {
        {colIndex, TransformType::kIdentity, std::nullopt}};
    const auto& dataSink = createDataSinkAndAppendData(
        rowType, vectors, outputDirectory->getPath(), partitionTransforms);

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

TEST_F(IcebergInsertTest, partitionMultiColumns) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4", "c5", "c6"},
          {
              BIGINT(),
              INTEGER(),
              SMALLINT(),
              DECIMAL(18, 5),
              BOOLEAN(),
              VARCHAR(),
          });
  std::vector<std::vector<int32_t>> columnCombinations = {
      {0, 1}, // BIGINT, INTEGER.
      {2, 1}, // SMALLINT, INTEGER.
      {2, 3}, // SMALLINT, DECIMAL.
      {0, 2, 1} // BIGINT, SMALLINT, INTEGER.
  };

  for (const auto& combination : columnCombinations) {
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(rowType, numBatches, vectorSize);
    std::vector<test::PartitionField> partitionTransforms;
    for (auto colIndex : combination) {
      partitionTransforms.push_back(
          {colIndex, TransformType::kIdentity, std::nullopt});
    }

    const auto& dataSink = createDataSinkAndAppendData(
        rowType, vectors, outputDirectory->getPath(), partitionTransforms);

    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    auto plan = exec::test::PlanBuilder().tableScan(rowType).planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
