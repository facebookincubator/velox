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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg::test {
class IcebergInsertTest
    : public testing::WithParamInterface<dwio::common::FileFormat>,
      public IcebergTestBase {
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

TEST_P(IcebergInsertTest, testIcebergTableWrite) {
  const auto& format = GetParam();
  fileFormat_ = format;
  const auto outputDirectory = exec::test::TempDirectoryPath::create();
  const auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
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

TEST_P(IcebergInsertTest, testSingleColumnAsPartition) {
  const auto& format = GetParam();
  fileFormat_ = format;

  for (int32_t colIndex = 0; colIndex < rowType_->size() - 1; colIndex++) {
    const auto& colName = rowType_->nameOf(colIndex);
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    const auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;

    const auto vectors = createTestData(numBatches, vectorSize);
    std::vector<std::string> partitionTransforms = {fmt::format("{}", colName)};
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

    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
        assignments;
    for (int32_t i = 0; i < rowType_->size(); i++) {
      const auto& name = rowType_->nameOf(i);
      if (i != colIndex) {
        assignments.insert(
            {name,
             std::make_shared<HiveColumnHandle>(
                 name,
                 HiveColumnHandle::ColumnType::kRegular,
                 rowType_->childAt(i),
                 rowType_->childAt(i))});
      }
    }

    // Add partition column
    assignments.insert(
        {colName,
         std::make_shared<HiveColumnHandle>(
             colName,
             HiveColumnHandle::ColumnType::kPartitionKey,
             rowType_->childAt(colIndex),
             rowType_->childAt(colIndex))});

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .tableScan(rowType_, {}, "", nullptr, assignments)
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

TEST_P(IcebergInsertTest, testColumnCombinationsAsPartition) {
  const auto& format = GetParam();
  fileFormat_ = format;
  std::vector<std::vector<int32_t>> columnCombinations = {
      {0, 1}, // BIGINT, INTEGER
      {2, 1}, // SMALLINT, INTEGER
      {2, 3}, // SMALLINT, DECIMAL
      {0, 2, 1} // BIGINT, SMALLINT, INTEGER
  };

  for (const auto& combination : columnCombinations) {
    const auto outputDirectory = exec::test::TempDirectoryPath::create();
    const auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    const auto vectors = createTestData(numBatches, vectorSize);
    std::vector<std::string> partitionTransforms;
    for (int32_t colIndex : combination) {
      partitionTransforms.push_back(
          fmt::format("{}", rowType_->nameOf(colIndex)));
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

    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
        assignments;
    std::unordered_set<int32_t> partitionColumns(
        combination.begin(), combination.end());

    for (int32_t i = 0; i < rowType_->size(); i++) {
      const auto& name = rowType_->nameOf(i);
      auto columnType = partitionColumns.count(i) > 0
          ? HiveColumnHandle::ColumnType::kPartitionKey
          : HiveColumnHandle::ColumnType::kRegular;

      assignments.insert(
          {name,
           std::make_shared<HiveColumnHandle>(
               name, columnType, rowType_->childAt(i), rowType_->childAt(i))});
    }

    auto plan = exec::test::PlanBuilder(pool_.get())
                    .tableScan(rowType_, {}, "", nullptr, assignments)
                    .planNode();

    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

INSTANTIATE_TEST_SUITE_P(
    IcebergInsertTest,
    IcebergInsertTest,
    testing::Values(
        dwio::common::FileFormat::DWRF
#ifdef VELOX_ENABLE_PARQUET
        ,
        dwio::common::FileFormat::PARQUET
#endif
        ));

} // namespace facebook::velox::connector::hive::iceberg::test

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
