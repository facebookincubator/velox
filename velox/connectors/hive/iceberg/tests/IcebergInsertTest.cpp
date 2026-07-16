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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {
namespace {

#ifdef VELOX_ENABLE_PARQUET

class IcebergInsertTest : public test::IcebergTestBase {
 protected:
  void test(const RowTypePtr& rowType, double nullRatio = 0.0) {
    const auto outputDirectory = TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    constexpr int32_t numBatches = 10;
    constexpr int32_t vectorSize = 5'000;
    const auto vectors =
        createTestData(rowType, numBatches, vectorSize, nullRatio);
    const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
    const auto commitTasks = dataSink->close();

    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_EQ(splits.size(), commitTasks.size());
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
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

TEST_F(IcebergInsertTest, bigDecimal) {
  auto rowType = ROW({"c1"}, {DECIMAL(38, 5)});
  fileFormat_ = dwio::common::FileFormat::PARQUET;
  test(rowType);
}

TEST_F(IcebergInsertTest, singleColumnPartition) {
  struct TestCase {
    std::string name;
    TypePtr type;
  };

  std::vector<TestCase> testCases = {
      {"c1", BIGINT()},
      {"c2", INTEGER()},
      {"c3", SMALLINT()},
      {"c4", DECIMAL(18, 5)},
      {"c5", BOOLEAN()},
      {"c6", VARCHAR()},
      {"c7", DATE()},
      {"c8", TIMESTAMP()}};

  for (const auto& testCase : testCases) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    auto rowType = ROW({testCase.name}, {testCase.type});

    const auto vectors = createTestData(rowType, numBatches, vectorSize, 0.5);
    std::vector<test::PartitionField> partitionTransforms = {
        {0, TransformType::kIdentity, std::nullopt}};
    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);
    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      ASSERT_TRUE(taskJson.count("partitionDataJson") > 0);
    }

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

TEST_F(IcebergInsertTest, partitionNullColumn) {
  struct TestCase {
    std::string name;
    TypePtr type;
  };

  std::vector<TestCase> testCases = {
      {"c1", BIGINT()},
      {"c2", INTEGER()},
      {"c3", SMALLINT()},
      {"c4", DECIMAL(18, 5)},
      {"c5", BOOLEAN()},
      {"c6", VARCHAR()},
      {"c7", DATE()},
      {"c8", TIMESTAMP()}};

  for (const auto& testCase : testCases) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 100;
    auto rowType = ROW({testCase.name}, {testCase.type});
    // nullRatio = 1.0
    const auto vectors = createTestData(rowType, numBatches, vectorSize, 1.0);

    std::vector<test::PartitionField> partitionTransforms = {
        {0, TransformType::kIdentity, std::nullopt}};
    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);

    const auto commitTasks = dataSink->close();
    ASSERT_EQ(1, commitTasks.size());
    auto taskJson = folly::parseJson(commitTasks.at(0));
    ASSERT_EQ(1, taskJson.count("partitionDataJson"));
    auto partitionData =
        folly::parseJson(taskJson["partitionDataJson"].asString());
    ASSERT_EQ(1, partitionData.count("partitionValues"));
    auto partitionValues = partitionData["partitionValues"];
    ASSERT_TRUE(partitionValues.isArray());
    ASSERT_TRUE(partitionValues[0].isNull());

    auto files = listFiles(outputDirectory->getPath());
    ASSERT_EQ(files.size(), 1);

    for (const auto& file : files) {
      auto partitionKeys = extractPartitionKeys(file);
      ASSERT_EQ(partitionKeys.size(), 1);
      ASSERT_TRUE(partitionKeys.contains(testCase.name));
      ASSERT_FALSE(partitionKeys.at(testCase.name).has_value());
    }
  }
}

TEST_F(IcebergInsertTest, partitionMultiColumns) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4"},
          {
              BIGINT(),
              INTEGER(),
              SMALLINT(),
              DECIMAL(18, 5),
          });
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

    std::vector<RowVectorPtr> vectors;
    vectors.reserve(numBatches);
    for (int32_t batch = 0; batch < numBatches; ++batch) {
      vectors.push_back(makeRowVector(
          rowType->names(),
          {
              makeFlatVector<int64_t>(
                  vectorSize, [](auto row) { return row * 100; }),
              makeFlatVector<int32_t>(
                  vectorSize, [](auto row) { return row * 10; }),
              makeFlatVector<int16_t>(vectorSize, [](auto row) { return row; }),
              makeFlatVector<int64_t>(
                  vectorSize,
                  [](auto row) { return (row * 1000); },
                  nullptr,
                  DECIMAL(18, 5)),
          }));
    }

    std::vector<test::PartitionField> partitionTransforms;
    for (auto colIndex : combination) {
      partitionTransforms.push_back(
          {colIndex, TransformType::kIdentity, std::nullopt});
    }

    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);

    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_EQ(commitTasks.size(), vectorSize);
    ASSERT_EQ(splits.size(), commitTasks.size());

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

TEST_F(IcebergInsertTest, maxTargetFileSizeRotation) {
  constexpr int32_t kNumBatches = 10;
  constexpr vector_size_t kRowsPerBatch = 100;
  constexpr int32_t kPayloadSize = 96;

  // Generate fixed-size, per-row-varying strings for predictable size
  // accounting without relying on fuzzed VARCHAR lengths.
  auto makePayload = [](int64_t value) {
    std::string payload;
    payload.reserve(kPayloadSize);
    for (auto i = 0; i < kPayloadSize; ++i) {
      payload.push_back(static_cast<char>('a' + ((value + i) % 26)));
    }
    return payload;
  };

  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(kNumBatches);
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    const auto batchOffset = batch * kRowsPerBatch;
    vectors.push_back(makeRowVector(
        rowType->names(),
        {
            makeFlatVector<int64_t>(
                kRowsPerBatch,
                [batchOffset](auto row) { return batchOffset + row; }),
            makeFlatVector<std::string>(
                kRowsPerBatch,
                [&, batchOffset](auto row) {
                  return makePayload(batchOffset + row);
                }),
        }));
  }

  auto writeAndRead = [&](const std::string& maxTargetFileSize) {
    setConnectorSessionProperty(
        HiveConfig::kParquetMaxTargetFileSizeSession, maxTargetFileSize);

    const auto outputDirectory = TempDirectoryPath::create();
    const auto outputPath = outputDirectory->getPath();
    const auto dataSink = createDataSinkAndAppendData(vectors, outputPath);
    const auto commitTasks = dataSink->close();
    const auto files = listFiles(outputPath);
    EXPECT_EQ(files.size(), commitTasks.size());

    auto splits = createSplitsForDirectory(outputPath);
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);

    return files.size();
  };

  ASSERT_EQ(writeAndRead("1KB"), kNumBatches);
  ASSERT_EQ(writeAndRead("10MB"), 1);
}

#endif

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
