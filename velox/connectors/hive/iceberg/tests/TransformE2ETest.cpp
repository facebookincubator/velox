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

#include <filesystem>

#include <folly/json.h>

#include "velox/common/encode/Base64.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/iceberg/Murmur3Hash32.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox::exec::test;

namespace {

#ifdef VELOX_ENABLE_PARQUET

class TransformE2ETest : public test::IcebergTestBase {
 protected:
  static constexpr int32_t kDefaultNumBatches = 2;
  static constexpr int32_t kDefaultRowsPerBatch = 100;

  std::vector<RowVectorPtr> createTimestampTestData() {
    std::vector<RowVectorPtr> batches;
    // 1. Aligned to hour boundaries (no minutes/seconds) since hour is the
    //    finest granularity transform tested.
    // 2. Include negative epoch values (pre-1970) to test edge cases.
    // 3. Span multiple years and months.
    static const std::vector<Timestamp> timestamps = {
        // 1969-01-01 00:00:00 (negative epoch)
        Timestamp(-31536000, 0),
        // 1969-12-31 00:00:00 (day before epoch)
        Timestamp(-86400, 0),
        // 1970-01-01 00:00:00
        Timestamp(0, 0),
        // 1970-01-01 01:00:00
        Timestamp(3600, 0),
        // 1970-01-02 00:00:00
        Timestamp(86400, 0),
        // 1970-01-31 00:00:00
        Timestamp(2592000, 0),
        // 1971-01-01 00:00:00
        Timestamp(31536000, 0),
        // 2021-01-01 00:00:00
        Timestamp(1609459200, 0),
        // 2021-01-02 00:00:00
        Timestamp(1609545600, 0),
        // 2021-02-01 00:00:00
        Timestamp(1612224000, 0),
        // 2022-01-01 00:00:00
        Timestamp(1640995200, 0),
        // 2023-01-01 00:00:00
        Timestamp(1672531200, 0),
        // 2100-01-01 00:00:00
        Timestamp(4102444800, 0),
    };

    for (auto i = 0; i < kDefaultNumBatches; i++) {
      auto timestampVector = makeFlatVector<Timestamp>(
          kDefaultRowsPerBatch,
          [](auto row) { return timestamps[row % timestamps.size()]; });
      batches.push_back(makeRowVector({timestampVector}));
    }
    return batches;
  }

  std::shared_ptr<TempDirectoryPath> writeBatchesWithTransforms(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<test::PartitionField>& partitionFields) {
    VELOX_CHECK(!batches.empty(), "input cannot be empty");

    int64_t expectedRowCount = 0;
    for (const auto& batch : batches) {
      expectedRowCount += batch->size();
    }

    auto rowType = batches.front()->rowType();
    auto outputDirectory = TempDirectoryPath::create();
    const auto dataSink = createDataSinkAndAppendData(
        batches, outputDirectory->getPath(), partitionFields);
    dataSink->close();
    verifyTotalRowCount(rowType, outputDirectory->getPath(), expectedRowCount);
    return outputDirectory;
  }

  // Generate a key from a timestamp and transform type.
  // The key format depends on the transform type:
  // - kMonth: "YYYY-MM"
  // - kDay: "YYYY-MM-DD"
  // - kHour: "YYYY-MM-DD-HH"
  static std::string timestampToKey(
      const Timestamp& ts,
      TransformType transformType) {
    std::tm tm;
    if (!Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
      return "";
    }

    int32_t year = tm.tm_year + 1900;
    int32_t month = tm.tm_mon + 1;
    int32_t day = tm.tm_mday;
    int32_t hour = tm.tm_hour;

    switch (transformType) {
      case TransformType::kMonth:
        return fmt::format("{:04d}-{:02d}", year, month);
      case TransformType::kDay:
        return fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
      case TransformType::kHour:
        return fmt::format(
            "{:04d}-{:02d}-{:02d}-{:02d}", year, month, day, hour);
      default:
        VELOX_UNREACHABLE();
    }
  }

  // Helper function to build expected counts map from timestamp batches.
  // The key format depends on the transform type:
  // - kMonth: "YYYY-MM"
  // - kDay: "YYYY-MM-DD"
  // - kHour: "YYYY-MM-DD-HH"
  static std::unordered_map<std::string, int32_t>
  buildExpectedCountsFromTimestamps(
      const std::vector<RowVectorPtr>& batches,
      TransformType transformType) {
    std::unordered_map<std::string, int32_t> expectedCounts;

    for (const auto& batch : batches) {
      auto timestampVector = batch->childAt(0)->as<SimpleVector<Timestamp>>();
      for (auto i = 0; i < batch->size(); i++) {
        Timestamp ts = timestampVector->valueAt(i);
        std::string key = timestampToKey(ts, transformType);
        expectedCounts[key]++;
      }
    }
    return expectedCounts;
  }

  static std::string dirName(const std::string& path) {
    return std::filesystem::path(path).filename().string();
  }

  static std::vector<std::string> firstLevelDirectories(
      const std::string& basePath) {
    std::vector<std::string> directories;
    for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
      if (entry.is_directory()) {
        directories.push_back(entry.path().string());
      }
    }
    return directories;
  }

  static std::vector<std::string> listDirectoriesRecursively(
      const std::string& path) {
    std::vector<std::string> directories;
    auto firstLevelDirs = firstLevelDirectories(path);

    for (const auto& dir : firstLevelDirs) {
      directories.push_back(dirName(dir));
      auto subDirs = listDirectoriesRecursively(dir);
      directories.insert(directories.end(), subDirs.begin(), subDirs.end());
    }

    return directories;
  }

  static std::vector<std::string> verifyPartitionCount(
      const std::string& outputPath,
      int32_t expectedCount) {
    const auto partitionDirs = firstLevelDirectories(outputPath);
    EXPECT_EQ(partitionDirs.size(), expectedCount);
    for (const auto& dir : partitionDirs) {
      const auto name = dirName(dir);
      EXPECT_TRUE(name.find('=') != std::string::npos)
          << "Partition directory " << name
          << " does not follow Iceberg naming convention";
    }
    return partitionDirs;
  }

  // Verify the total row count across all partitions.
  void verifyTotalRowCount(
      const RowTypePtr& rowType,
      const std::string& outputPath,
      int32_t expectedRowCount) {
    auto splits = createSplitsForDirectory(outputPath);

    const auto plan = PlanBuilder()
                          .startTableScan()
                          .connectorId(test::kIcebergConnectorId)
                          .outputType(rowType)
                          .endTableScan()
                          .planNode();

    const auto actualRowCount =
        AssertQueryBuilder(plan).splits(splits).countResults();

    ASSERT_EQ(actualRowCount, expectedRowCount);
  }

  // Verify data in a specific partition.
  void verifyPartitionData(
      const RowTypePtr& rowType,
      const std::string& partitionPath,
      const std::string& partitionFilter,
      const int32_t expectedRowCount) {
    auto scanPlan = PlanBuilder()
                        .startTableScan()
                        .connectorId(test::kIcebergConnectorId)
                        .outputType(rowType)
                        .endTableScan()
                        .planNode();

    const auto actualRowCount =
        AssertQueryBuilder(scanPlan)
            .splits(createSplitsForDirectory(partitionPath))
            .countResults();

    ASSERT_EQ(actualRowCount, expectedRowCount);

    const auto filterPlan = PlanBuilder()
                                .startTableScan()
                                .connectorId(test::kIcebergConnectorId)
                                .outputType(rowType)
                                .endTableScan()
                                .filter(partitionFilter)
                                .planNode();
    const auto filteredRowCount =
        AssertQueryBuilder(filterPlan)
            .splits(createSplitsForDirectory(partitionPath))
            .countResults();
    ASSERT_EQ(expectedRowCount, filteredRowCount);
  }

  static std::pair<std::string, std::string> parsePartitionDirName(
      const std::string& name) {
    auto eq = name.find('=');
    VELOX_CHECK(eq != std::string::npos);
    auto us = name.rfind('_', eq - 1);
    auto columnName = name.substr(0, us);
    auto value = name.substr(eq + 1);
    return {columnName, value};
  }

  static int32_t computeBucketHash(
      const StringView& value,
      int32_t numBuckets) {
    int32_t hash = functions::iceberg::Murmur3Hash32::hashBytes(
        value.data(), value.size());
    return ((hash & 0x7FFFFFFF) % numBuckets);
  }

  folly::dynamic getPartitionValuesFromCommitMessage(
      const RowVectorPtr& rowVector) {
    auto outputDirectory = TempDirectoryPath::create();
    std::vector<test::PartitionField> partitionTransforms = {
        {0, TransformType::kIdentity, std::nullopt},
    };
    const auto dataSink = createDataSinkAndAppendData(
        {rowVector}, outputDirectory->getPath(), partitionTransforms);

    auto commitMessages = dataSink->commitMessage();
    VELOX_CHECK_EQ(commitMessages.size(), 1);
    auto commitData = folly::parseJson(commitMessages[0]);
    auto partitionDataJson =
        folly::parseJson(commitData["partitionDataJson"].asString());
    auto partitionValues = partitionDataJson["partitionValues"];
    VELOX_CHECK_EQ(partitionValues.size(), 1);
    dataSink->close();
    return partitionValues[0];
  }
};

TEST_F(TransformE2ETest, identity) {
  constexpr auto rowsPerBatch = 10;
  constexpr auto duplicates = 5;
  auto rowType = ROW({"c0"}, {INTEGER()});
  auto baseVectors = createTestData(rowType, kDefaultNumBatches, rowsPerBatch);

  // Duplicate each row to create multiple rows with the same partition key.
  std::vector<RowVectorPtr> vectors;
  for (const auto& baseVector : baseVectors) {
    auto duplicatedColumn = wrapInDictionary(
        makeIndices(
            baseVector->size() * duplicates,
            [duplicates](auto row) { return row / duplicates; }),
        baseVector->size() * duplicates,
        baseVector->childAt(0));
    vectors.push_back(makeRowVector({duplicatedColumn}));
  }

  auto outputDirectory = writeBatchesWithTransforms(
      vectors, {{0, TransformType::kIdentity, std::nullopt}});

  auto partitionDirs = verifyPartitionCount(
      outputDirectory->getPath(), kDefaultNumBatches * rowsPerBatch);

  for (const auto& dir : partitionDirs) {
    const auto name = dirName(dir);
    // Each partition should have duplicates rows.
    verifyPartitionData(rowType, dir, name, duplicates);
  }
}

TEST_F(TransformE2ETest, partitionNamingConventions) {
  auto rowVector = makeRowVector(
      {
          "c_int",
          "c_bigint",
          "c_varchar",
          "c_varchar2",
          "c_decimal",
          "c_varbinary",
      },
      {
          makeConstant(42, 1, INTEGER()),
          makeConstant(static_cast<int64_t>(9'876'543'210), 1, BIGINT()),
          makeConstant("test string", 1, VARCHAR()),
          makeNullConstant(TypeKind::VARCHAR, 1),
          makeConstant(static_cast<int64_t>(1'234'567'890), 1, DECIMAL(18, 3)),
          makeConstant("binarydata\1\2\3", 1, VARBINARY()),
      });

  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}, // c_int.
      {1, TransformType::kIdentity, std::nullopt}, // c_bigint.
      {2, TransformType::kIdentity, std::nullopt}, // c_varchar.
      {4, TransformType::kIdentity, std::nullopt}, // c_decimal.
      {5, TransformType::kIdentity, std::nullopt}, // c_varbinary.
      {3, TransformType::kIdentity, std::nullopt}, // c_varchar2.
  };

  auto outputDirectory =
      writeBatchesWithTransforms({rowVector}, partitionTransforms);

  const auto actualPartitionNames =
      listDirectoriesRecursively(outputDirectory->getPath());

  // Build expected partition folder names.
  std::vector<std::string> expectedPartitionNames = {
      "c_int=42",
      "c_bigint=9876543210",
      "c_varchar=test+string",
      "c_decimal=1234567.890",
      "c_varbinary=YmluYXJ5ZGF0YQECAw%3D%3D",
      "c_varchar2=null",
  };

  ASSERT_EQ(actualPartitionNames, expectedPartitionNames)
      << "Partition folder names do not match expected values";
}

TEST_F(TransformE2ETest, varbinaryPartitionCommitMessage) {
  std::vector<std::string> testData = {
      "binarydata\x01\x02\x03",
      "",
      ".-_*/?%#&=+,;@!$\\\"'()<>:|[]{}^~`\n\r\t\u00A9",
  };

  for (const auto& binaryData : testData) {
    SCOPED_TRACE(testing::Message() << "binaryData: " << binaryData);
    auto rowVector = makeRowVector({
        makeConstant(binaryData, 1, VARBINARY()),
    });

    auto partitionValue = getPartitionValuesFromCommitMessage(rowVector);
    ASSERT_EQ(
        partitionValue.asString(),
        encoding::Base64::encode(binaryData.data(), binaryData.size()));
  }
}

TEST_F(TransformE2ETest, nullPartitionValue) {
  const std::vector<TypeKind> typeKinds = {
      TypeKind::BOOLEAN,
      TypeKind::INTEGER,
      TypeKind::BIGINT,
      TypeKind::VARCHAR,
      TypeKind::VARBINARY,
      TypeKind::TIMESTAMP,
  };

  for (const auto& typeKind : typeKinds) {
    auto rowVector = makeRowVector({
        makeNullConstant(typeKind, 1),
    });

    auto partitionValue = getPartitionValuesFromCommitMessage(rowVector);
    ASSERT_TRUE(partitionValue.isNull());
  }
}

TEST_F(TransformE2ETest, bucket) {
  constexpr int32_t numBuckets = 4;
  auto rowType = ROW({"c_varchar"}, {VARCHAR()});
  auto vectors =
      createTestData(rowType, kDefaultNumBatches, kDefaultRowsPerBatch);

  auto outputDirectory = writeBatchesWithTransforms(
      vectors, {{0, TransformType::kBucket, numBuckets}});

  const auto partitionDirs =
      verifyPartitionCount(outputDirectory->getPath(), numBuckets);

  int32_t totalRows = 0;
  for (const auto& dir : partitionDirs) {
    auto splits = createSplitsForDirectory(dir);
    auto countPlan = PlanBuilder()
                         .startTableScan()
                         .connectorId(test::kIcebergConnectorId)
                         .outputType(rowType)
                         .endTableScan()
                         .planNode();
    auto partitionRowCount =
        AssertQueryBuilder(countPlan).splits(splits).countResults();

    totalRows += partitionRowCount;
    ASSERT_GT(partitionRowCount, 0);
  }

  ASSERT_EQ(totalRows, kDefaultNumBatches * kDefaultRowsPerBatch);

  std::unordered_map<std::string, int32_t> valueToExpectedBucket;

  for (const auto& dir : partitionDirs) {
    const auto name = dirName(dir);
    const auto [k, v] = parsePartitionDirName(name);
    const int32_t expectedBucket = std::stoi(v);

    auto dataPlan = PlanBuilder()
                        .startTableScan()
                        .connectorId(test::kIcebergConnectorId)
                        .outputType(rowType)
                        .endTableScan()
                        .project({"c_varchar"})
                        .planNode();
    const auto& dataResult = AssertQueryBuilder(dataPlan)
                                 .splits(createSplitsForDirectory(dir))
                                 .copyResults(opPool_.get());

    auto varcharColumn = dataResult->childAt(0)->asFlatVector<StringView>();
    for (auto i = 0; i < dataResult->size(); i++) {
      auto value = varcharColumn->valueAt(i);

      auto computedBucket = computeBucketHash(value, numBuckets);
      ASSERT_EQ(computedBucket, expectedBucket);
    }
  }
}

TEST_F(TransformE2ETest, truncate) {
  auto rowType = ROW({"c_int"}, {INTEGER()});

  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < kDefaultNumBatches; i++) {
    std::vector<VectorPtr> columns;
    columns.push_back(
        makeFlatVector<int32_t>(50, [](auto row) { return row % 100; }));
    auto vectors = makeRowVector(rowType->names(), columns);
    batches.push_back(vectors);
  }

  auto outputDirectory =
      writeBatchesWithTransforms(batches, {{0, TransformType::kTruncate, 10}});

  auto partitionDirs = verifyPartitionCount(outputDirectory->getPath(), 5);

  for (const auto& dir : partitionDirs) {
    const std::string name = dirName(dir);
    auto [c, v] = parsePartitionDirName(name);
    const std::string filter = fmt::format(
        "{}>={} AND {}<{}", c, v, c, std::to_string(std::stoi(v) + 10));

    verifyPartitionData(
        rowType, dir, filter, 20); // 10 values per batch * 2 batches.
  }
}

TEST_F(TransformE2ETest, year) {
  auto rowType = ROW({"c_date"}, {DATE()});
  static const std::vector<int32_t> dates = {
      18'262, 18'628, 18'993, 19'358, 19'723, 20'181};
  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < kDefaultNumBatches; i++) {
    auto dateVector = makeFlatVector<int32_t>(
        kDefaultRowsPerBatch,
        [](auto row) { return dates[row % dates.size()]; },
        nullptr,
        DATE());
    batches.emplace_back(makeRowVector(rowType->names(), {dateVector}));
  }

  auto outputDirectory = writeBatchesWithTransforms(
      batches, {{0, TransformType::kYear, std::nullopt}});

  auto partitionDirs = verifyPartitionCount(outputDirectory->getPath(), 6);

  for (int32_t year = 2020; year <= 2025; year++) {
    const auto expectedDirName = fmt::format("c_date_year={}", year);
    bool foundPartition = false;
    auto yearFilter = [](int32_t year) -> std::string {
      return fmt::format("YEAR(DATE '{}-01-01')={}", year, year);
    };

    for (const auto& dir : partitionDirs) {
      SCOPED_TRACE(year);
      const auto name = dirName(dir);
      if (name == expectedDirName) {
        foundPartition = true;
        auto datePlan = PlanBuilder()
                            .startTableScan()
                            .connectorId(test::kIcebergConnectorId)
                            .outputType(rowType)
                            .endTableScan()
                            .filter(yearFilter(year))
                            .planNode();

        auto partitionRowCount = AssertQueryBuilder(datePlan)
                                     .splits(createSplitsForDirectory(dir))
                                     .countResults();

        auto countPlan = PlanBuilder()
                             .startTableScan()
                             .connectorId(test::kIcebergConnectorId)
                             .outputType(rowType)
                             .endTableScan()
                             .planNode();
        auto totalPartitionCount = AssertQueryBuilder(countPlan)
                                       .splits(createSplitsForDirectory(dir))
                                       .countResults();
        ASSERT_EQ(partitionRowCount, totalPartitionCount);
        break;
      }
    }
    ASSERT_TRUE(foundPartition);
  }
}

TEST_F(TransformE2ETest, month) {
  auto batches = createTimestampTestData();

  auto outputDirectory = writeBatchesWithTransforms(
      batches, {{0, TransformType::kMonth, std::nullopt}});

  auto expectedCounts =
      buildExpectedCountsFromTimestamps(batches, TransformType::kMonth);
  const auto partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto name = dirName(dir);
    auto [c, v] = parsePartitionDirName(name);
    size_t dashPos = v.find('-');
    ASSERT_NE(dashPos, std::string::npos) << "Invalid month format: " << v;

    int32_t year = std::stoi(v.substr(0, dashPos));
    int32_t month = std::stoi(v.substr(dashPos + 1));
    std::string filter =
        fmt::format("YEAR(c0) = {} AND MONTH(c0) = {}", year, month);
    std::string monthKey = fmt::format("{:04d}-{:02d}", year, month);
    verifyPartitionData(
        ROW({"c0"}, {TIMESTAMP()}), dir, filter, expectedCounts[monthKey]);
  }
}

TEST_F(TransformE2ETest, day) {
  auto batches = createTimestampTestData();

  auto outputDirectory = writeBatchesWithTransforms(
      batches, {{0, TransformType::kDay, std::nullopt}});

  auto expectedCounts =
      buildExpectedCountsFromTimestamps(batches, TransformType::kDay);
  const auto partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto name = dirName(dir);
    auto [c, v] = parsePartitionDirName(name);
    std::vector<std::string> dateParts;
    folly::split('-', v, dateParts);
    ASSERT_EQ(dateParts.size(), 3) << "Invalid day format: " << v;

    int32_t year = std::stoi(dateParts[0]);
    int32_t month = std::stoi(dateParts[1]);
    int32_t day = std::stoi(dateParts[2]);

    std::string filter = fmt::format(
        "YEAR(c0) = {} AND MONTH(c0) = {} AND DAY(c0) = {}", year, month, day);
    std::string dayKey = fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
    verifyPartitionData(
        ROW({"c0"}, {TIMESTAMP()}), dir, filter, expectedCounts[dayKey]);
  }
}

TEST_F(TransformE2ETest, hour) {
  auto batches = createTimestampTestData();

  auto outputDirectory = writeBatchesWithTransforms(
      batches, {{0, TransformType::kHour, std::nullopt}});

  auto expectedCounts =
      buildExpectedCountsFromTimestamps(batches, TransformType::kHour);
  const auto partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto name = dirName(dir);
    auto [c, v] = parsePartitionDirName(name);
    std::vector<std::string> dateParts;
    folly::split('-', v, dateParts);
    ASSERT_EQ(dateParts.size(), 4) << "Invalid hour format: " << v;

    int32_t year = std::stoi(dateParts[0]);
    int32_t month = std::stoi(dateParts[1]);
    int32_t day = std::stoi(dateParts[2]);
    int32_t hour = std::stoi(dateParts[3]);

    std::string filter = fmt::format(
        "YEAR(c0) = {} AND MONTH(c0) = {} AND "
        "DAY(c0) = {} AND HOUR(c0) = {}",
        year,
        month,
        day,
        hour);
    std::string hourKey =
        fmt::format("{:04d}-{:02d}-{:02d}-{:02d}", year, month, day, hour);
    verifyPartitionData(
        ROW({"c0"}, {TIMESTAMP()}), dir, filter, expectedCounts[hourKey]);
  }
}

TEST_F(TransformE2ETest, multipleTransformsOnSameColumn) {
  auto rowType = ROW(
      {
          "c_int",
          "c_bigint",
      },
      {
          INTEGER(),
          BIGINT(),
      });

  auto vectors = createTestData(rowType, 2, 20);
  auto outputDirectory = writeBatchesWithTransforms(
      vectors,
      {
          {0, TransformType::kIdentity, std::nullopt}, // c_int.
          {0, TransformType::kTruncate, 10}, // truncate(c_int, 10).
          {0, TransformType::kBucket, 4}, // bucket(c_int, 4).
      });

  auto firstLevelDirs = firstLevelDirectories(outputDirectory->getPath());
  ASSERT_GT(firstLevelDirs.size(), 0);
  for (const auto& dir : firstLevelDirs) {
    const auto name = dirName(dir);
    ASSERT_TRUE(name.find("c_int=") != std::string::npos)
        << "First level directory " << name << " should use identity transform";

    auto secondLevelDirs = firstLevelDirectories(dir);
    ASSERT_GT(secondLevelDirs.size(), 0)
        << "No second level directories found in " << dir;

    for (const auto& secondDir : secondLevelDirs) {
      const auto secondName = dirName(secondDir);
      ASSERT_TRUE(secondName.find("c_int_trunc=") != std::string::npos)
          << "Second level directory " << secondName
          << " should use truncate transform";

      auto thirdLevelDirs = firstLevelDirectories(secondDir);
      ASSERT_GT(thirdLevelDirs.size(), 0)
          << "No third level directories found in " << secondDir;

      for (const auto& thirdDir : thirdLevelDirs) {
        const auto thirdName = dirName(thirdDir);
        ASSERT_TRUE(thirdName.find("c_int_bucket=") != std::string::npos)
            << "Third level directory " << thirdName
            << " should use bucket transform";

        // Verify the partition has data.
        auto splits = createSplitsForDirectory(thirdDir);
        auto countPlan = PlanBuilder()
                             .startTableScan()
                             .connectorId(test::kIcebergConnectorId)
                             .outputType(rowType)
                             .endTableScan()
                             .planNode();
        auto rowCount =
            AssertQueryBuilder(countPlan).splits(splits).countResults();
        ASSERT_GT(rowCount, 0)
            << "Leaf partition directory " << thirdDir << " has no data";
      }
    }
  }
}

TEST_F(TransformE2ETest, dateIdentityPartitionWithFilter) {
  auto rowType = ROW({"c_date", "c_value"}, {DATE(), INTEGER()});

  static const std::vector<int32_t> dates = {20147, 19816};
  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < kDefaultNumBatches; i++) {
    batches.emplace_back(makeRowVector(
        rowType->names(),
        {makeFlatVector<int32_t>(
             kDefaultRowsPerBatch,
             [](auto row) { return dates[row % dates.size()]; },
             nullptr,
             DATE()),
         makeFlatVector<int32_t>(
             kDefaultRowsPerBatch, [](auto row) { return row; })}));
  }

  auto outputDirectory = writeBatchesWithTransforms(
      batches, {{0, TransformType::kIdentity, std::nullopt}});

  auto partitionDirs = verifyPartitionCount(outputDirectory->getPath(), 2);

  std::unordered_map<std::string, std::string> customSplitInfo{
      {"table_format", "hive-iceberg"}};
  std::vector<std::shared_ptr<ConnectorSplit>> splits;

  for (const auto& dir : partitionDirs) {
    const auto daysSinceEpoch =
        dirName(dir) == "c_date=2025-02-28" ? "20147" : "19816";

    for (const auto& filePath : listFiles(dir)) {
      const auto file = filesystems::getFileSystem(filePath, nullptr)
                            ->openFileForRead(filePath);
      splits.push_back(
          std::make_shared<HiveIcebergSplit>(
              test::kIcebergConnectorId,
              filePath,
              fileFormat_,
              0,
              file->size(),
              std::unordered_map<std::string, std::optional<std::string>>{
                  {"c_date", daysSinceEpoch}},
              std::nullopt,
              customSplitInfo,
              nullptr,
              /*cacheable=*/true,
              std::vector<IcebergDeleteFile>()));
    }
  }

  ColumnHandleMap assignments{
      {"c_date",
       std::make_shared<IcebergColumnHandle>(
           "c_date",
           HiveColumnHandle::ColumnType::kPartitionKey,
           DATE(),
           parquet::ParquetFieldId{0, {}},
           std::vector<common::Subfield>{})},
      {"c_value",
       std::make_shared<IcebergColumnHandle>(
           "c_value",
           HiveColumnHandle::ColumnType::kRegular,
           INTEGER(),
           parquet::ParquetFieldId{1, {}})},
  };

  auto filterPlan = PlanBuilder()
                        .startTableScan()
                        .connectorId(test::kIcebergConnectorId)
                        .outputType(rowType)
                        .assignments(assignments)
                        .remainingFilter("c_date = DATE '2025-02-28'")
                        .endTableScan()
                        .planNode();

  const auto filteredRowCount =
      AssertQueryBuilder(filterPlan).splits(splits).countResults();

  ASSERT_EQ(filteredRowCount, kDefaultRowsPerBatch);
}
#endif

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
