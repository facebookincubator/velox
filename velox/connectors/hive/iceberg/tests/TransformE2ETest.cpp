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

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/iceberg/Murmur3Hash32.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox::exec::test;

namespace {

class TransformE2ETest : public test::IcebergTestBase {
 protected:
  static constexpr int32_t kDefaultNumBatches = 2;
  static constexpr int32_t kDefaultRowsPerBatch = 100;

  std::vector<RowVectorPtr> createTimestampTestData(
      int32_t numBatches,
      int32_t rowsPerBatch,
      RowTypePtr rowType) {
    std::vector<RowVectorPtr> batches;
    static const std::vector<Timestamp> timestamps = {
        Timestamp(0, 0), // 1970-01-01 00:00:00
        Timestamp(3600, 0), // 1970-01-01 01:00:00
        Timestamp(86400, 0), // 1970-01-02 00:00:00
        Timestamp(2592000, 0), // 1970-01-31 00:00:00
        Timestamp(31536000, 0), // 1971-01-01 00:00:00
        Timestamp(1609459200, 0), // 2021-01-01 00:00:00
        Timestamp(1609545600, 0), // 2021-01-02 00:00:00
        Timestamp(1612224000, 0), // 2021-02-01 00:00:00
        Timestamp(1640995200, 0), // 2022-01-01 00:00:00
        Timestamp(1672531200, 0) // 2023-01-01 00:00:00
    };

    for (auto i = 0; i < numBatches; i++) {
      auto timestampVector = BaseVector::create<FlatVector<Timestamp>>(
          TIMESTAMP(), rowsPerBatch, opPool_.get());
      for (auto j = 0; j < rowsPerBatch; j++) {
        timestampVector->set(j, timestamps[j % timestamps.size()]);
      }
      batches.push_back(makeRowVector(rowType->names(), {timestampVector}));
    }
    return batches;
  }

  std::shared_ptr<TempDirectoryPath> writeBatchesWithTransforms(
      const std::vector<RowVectorPtr>& batches,
      RowTypePtr rowType,
      const std::vector<test::PartitionField>& partitionFields) {
    int64_t expectedRowCount = 0;
    for (const auto& batch : batches) {
      expectedRowCount += batch->size();
    }

    auto outputDirectory = TempDirectoryPath::create();
    const auto& dataSink = createDataSinkAndAppendData(
        rowType, batches, outputDirectory->getPath(), partitionFields);
    dataSink->close();
    verifyTotalRowCount(rowType, outputDirectory->getPath(), expectedRowCount);
    return outputDirectory;
  }

  std::shared_ptr<TempDirectoryPath> writeTimestampBatchesWithTransform(
      const std::vector<RowVectorPtr>& batches,
      TransformType transformType) {
    auto rowType = ROW({"c_timestamp"}, {TIMESTAMP()});
    return writeBatchesWithTransforms(
        batches, rowType, {{0, transformType, std::nullopt}});
  }

  // Helper function to build expected counts map from timestamp batches.
  // The key format depends on the granularity:
  // - Month: "YYYY-MM"
  // - Day: "YYYY-MM-DD"
  // - Hour: "YYYY-MM-DD-HH"
  std::unordered_map<std::string, int32_t> buildExpectedCountsFromTimestamps(
      const std::vector<RowVectorPtr>& batches,
      const std::string& granularity) {
    std::unordered_map<std::string, int32_t> expectedCounts;

    for (const auto& batch : batches) {
      auto timestampVector = batch->childAt(0)->as<SimpleVector<Timestamp>>();
      for (auto i = 0; i < batch->size(); i++) {
        if (!timestampVector->isNullAt(i)) {
          Timestamp ts = timestampVector->valueAt(i);
          std::tm tm;
          if (Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
            int32_t year = tm.tm_year + 1900;
            int32_t month = tm.tm_mon + 1;
            int32_t day = tm.tm_mday;
            int32_t hour = tm.tm_hour;

            std::string key;
            if (granularity == "month") {
              key = fmt::format("{:04d}-{:02d}", year, month);
            } else if (granularity == "day") {
              key = fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
            } else if (granularity == "hour") {
              key = fmt::format(
                  "{:04d}-{:02d}-{:02d}-{:02d}", year, month, day, hour);
            }
            expectedCounts[key]++;
          }
        }
      }
    }
    return expectedCounts;
  }

  std::vector<std::string> firstLevelDirectories(const std::string& basePath) {
    std::vector<std::string> partitionDirs;
    for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
      if (entry.is_directory()) {
        partitionDirs.push_back(entry.path().string());
      }
    }
    return partitionDirs;
  }

  std::vector<std::string> listDirectoriesRecursively(const std::string& path) {
    std::vector<std::string> allPartitionDirs;
    auto firstLevelDirs = firstLevelDirectories(path);

    for (const auto& dir : firstLevelDirs) {
      if (std::filesystem::is_directory(dir)) {
        allPartitionDirs.push_back(
            std::filesystem::path(dir).filename().string());
        auto subDirs = listDirectoriesRecursively(dir);
        allPartitionDirs.insert(
            allPartitionDirs.end(), subDirs.begin(), subDirs.end());
      }
    }

    return allPartitionDirs;
  }

  // Verify the number of partitions and their naming convention.
  void verifyPartitionCount(
      const std::string& outputPath,
      const std::vector<std::string>& partitionTransforms,
      int32_t expectedPartitionCount) {
    const auto partitionDirs = firstLevelDirectories(outputPath);

    if (partitionTransforms.empty()) {
      ASSERT_EQ(partitionDirs.size(), 1)
          << "Expected 1 directory for no partitioning, got "
          << partitionDirs.size();
    } else {
      ASSERT_EQ(partitionDirs.size(), expectedPartitionCount)
          << "Expected " << expectedPartitionCount << " partitions, got "
          << partitionDirs.size();

      for (const auto& dir : partitionDirs) {
        const auto dirName = std::filesystem::path(dir).filename().string();
        ASSERT_TRUE(dirName.find('=') != std::string::npos)
            << "Partition directory " << dirName
            << " does not follow Iceberg naming convention";
      }
    }
  }

  // Verify the total row count across all partitions.
  void verifyTotalRowCount(
      RowTypePtr rowType,
      const std::string& outputPath,
      int32_t expectedRowCount) {
    auto splits = createSplitsForDirectory(outputPath);

    const auto plan = PlanBuilder()
                          .tableScan(rowType)
                          .singleAggregation({}, {"count(1)"})
                          .planNode();

    const auto result =
        AssertQueryBuilder(plan).splits(splits).copyResults(opPool_.get());

    ASSERT_EQ(result->size(), 1);
    ASSERT_EQ(
        result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
        expectedRowCount);
  }

  // Verify data in a specific partition.
  void verifyPartitionData(
      RowTypePtr rowType,
      const std::string& partitionPath,
      const std::string& partitionFilter,
      const int32_t expectedRowCount,
      bool skipRowCountCheck = false) {
    auto splits = createSplitsForDirectory(partitionPath);

    auto countPlan = PlanBuilder()
                         .tableScan(rowType)
                         .singleAggregation({}, {"count(1)"})
                         .planNode();

    const auto& countResult =
        AssertQueryBuilder(countPlan).splits(splits).copyResults(opPool_.get());

    ASSERT_EQ(countResult->size(), 1);
    const auto actualRowCount =
        countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);

    if (!skipRowCountCheck) {
      ASSERT_EQ(actualRowCount, expectedRowCount);
    } else {
      // Just verify that we have some data.
      ASSERT_GT(actualRowCount, 0);
    }

    const auto dataPlan = PlanBuilder()
                              .tableScan(rowType)
                              .filter(partitionFilter)
                              .singleAggregation({}, {"count(1)"})
                              .planNode();
    const auto& dataResult =
        AssertQueryBuilder(dataPlan).splits(splits).copyResults(opPool_.get());
    ASSERT_EQ(dataResult->size(), 1);
    const auto filteredRowCount =
        dataResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
    if (!skipRowCountCheck) {
      ASSERT_EQ(filteredRowCount, expectedRowCount);
    } else {
      ASSERT_EQ(filteredRowCount, actualRowCount);
    }
  }

  std::pair<std::string, std::string> buildFilter(
      const std::string& partitionDir) {
    auto eq = partitionDir.find('=');
    auto us = partitionDir.rfind('_', eq - 1);
    auto column = partitionDir.substr(0, us);
    auto value = partitionDir.substr(eq + 1);
    return {column, value};
  }

  int32_t extractBucketNumber(const std::string& dirName) {
    const auto& eq = dirName.find('=');
    VELOX_CHECK_NE(
        eq, std::string::npos, "Invalid partition directory name: {}", dirName);
    const auto& bucketStr = dirName.substr(eq + 1);
    return std::stoi(bucketStr);
  }

  int32_t computeBucketHash(const StringView& value, int32_t numBuckets) {
    int32_t hash = functions::iceberg::Murmur3Hash32::hashBytes(
        value.data(), value.size());
    return ((hash & 0x7FFFFFFF) % numBuckets);
  }
};

TEST_F(TransformE2ETest, identity) {
  constexpr auto rowsPerBatch = 50;
  auto rowType = ROW({"c_int"}, {INTEGER()});
  auto vectors = createTestData(rowType, kDefaultNumBatches, rowsPerBatch);

  auto outputDirectory = writeBatchesWithTransforms(
      vectors, rowType, {{0, TransformType::kIdentity, std::nullopt}});

  verifyPartitionCount(
      outputDirectory->getPath(), {"c_int"}, kDefaultNumBatches * rowsPerBatch);

  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());
  for (const auto& dir : partitionDirs) {
    const auto& dirName = std::filesystem::path(dir).filename().string();
    verifyPartitionData(rowType, dir, dirName, 1);
  }
}

TEST_F(TransformE2ETest, partitionNamingConventions) {
  std::string binaryData = "binary\0data\1\2\3";
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
          makeConstant(StringView(binaryData), 1, VARBINARY()),
      });

  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}, // c_int.
      {1, TransformType::kIdentity, std::nullopt}, // c_bigint.
      {2, TransformType::kIdentity, std::nullopt}, // c_varchar.
      {4, TransformType::kIdentity, std::nullopt}, // c_decimal.
      {5, TransformType::kIdentity, std::nullopt}, // c_varbinary.
      {3, TransformType::kIdentity, std::nullopt}, // c_varchar2.
  };

  auto outputDirectory = writeBatchesWithTransforms(
      {rowVector}, asRowType(rowVector->type()), partitionTransforms);

  const auto& actualPartitionNames =
      listDirectoriesRecursively(outputDirectory->getPath());

  // Build expected partition folder names
  std::vector<std::string> expectedPartitionNames = {
      "c_int=42",
      "c_bigint=9876543210",
      "c_varchar=test+string",
      "c_decimal=1234567.890",
      "c_varbinary=" +
          encoding::Base64::encode(binaryData.data(), binaryData.size()),
      "c_varchar2=null",
  };

  ASSERT_EQ(actualPartitionNames, expectedPartitionNames)
      << "Partition folder names do not match expected values";
}

TEST_F(TransformE2ETest, bucket) {
  constexpr int32_t numBuckets = 4;
  auto rowType = ROW({"c_varchar"}, {VARCHAR()});
  auto vectors =
      createTestData(rowType, kDefaultNumBatches, kDefaultRowsPerBatch);

  auto outputDirectory = writeBatchesWithTransforms(
      vectors, rowType, {{0, TransformType::kBucket, numBuckets}});

  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());
  ASSERT_LE(partitionDirs.size(), numBuckets);

  int32_t totalRowsInPartitions = 0;
  for (const auto& dir : partitionDirs) {
    auto splits = createSplitsForDirectory(dir);
    auto countPlan = PlanBuilder()
                         .tableScan(rowType)
                         .singleAggregation({}, {"count(1)"})
                         .planNode();
    auto countResult =
        AssertQueryBuilder(countPlan).splits(splits).copyResults(opPool_.get());

    auto partitionRowCount =
        countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
    totalRowsInPartitions += partitionRowCount;
    ASSERT_GT(partitionRowCount, 0);
  }

  ASSERT_EQ(totalRowsInPartitions, kDefaultNumBatches * kDefaultRowsPerBatch);

  std::unordered_map<std::string, int32_t> valueToExpectedBucket;

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    const int32_t expectedBucket = extractBucketNumber(dirName);

    auto dataPlan =
        PlanBuilder().tableScan(rowType).project({"c_varchar"}).planNode();
    const auto& dataResult = AssertQueryBuilder(dataPlan)
                                 .splits(createSplitsForDirectory(dir))
                                 .copyResults(opPool_.get());

    auto varcharColumn = dataResult->childAt(0)->asFlatVector<StringView>();
    for (auto i = 0; i < dataResult->size(); i++) {
      StringView value = varcharColumn->valueAt(i);

      auto computedBucket = computeBucketHash(value, numBuckets);
      ASSERT_EQ(computedBucket, expectedBucket)
          << "Value '" << value << "' in partition " << dirName
          << " has computed bucket " << computedBucket
          << " but expected bucket " << expectedBucket;

      // Track the bucket for this value to verify consistency.
      if (valueToExpectedBucket.find(value) == valueToExpectedBucket.end()) {
        valueToExpectedBucket[value] = expectedBucket;
      } else {
        // Verify that the same value always hashes to the same bucket.
        ASSERT_EQ(valueToExpectedBucket[value], expectedBucket)
            << "Value '" << value
            << "' hashed to different buckets: " << valueToExpectedBucket[value]
            << " and " << expectedBucket;
      }
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

  auto outputDirectory = writeBatchesWithTransforms(
      batches, rowType, {{0, TransformType::kTruncate, 10}});

  verifyPartitionCount(outputDirectory->getPath(), {"truncate(c_int, 10)"}, 5);
  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const std::string dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    const std::string filter =
        c + ">=" + v + " AND " + c + "<" + std::to_string(std::stoi(v) + 10);
    verifyPartitionData(
        rowType, dir, filter, 20); // 10 values per batch * 2 batches.
  }
}

TEST_F(TransformE2ETest, year) {
  auto rowType = ROW({"c_date"}, {DATE()});

  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < kDefaultNumBatches; i++) {
    auto dateVector = BaseVector::create<FlatVector<int32_t>>(
        DATE(), kDefaultRowsPerBatch, opPool_.get());
    for (auto i = 0; i < kDefaultRowsPerBatch; i++) {
      static const std::vector<int32_t> dates = {
          18'262, 18'628, 18'993, 19'358, 19'723, 20'181};
      dateVector->set(i, dates[i % dates.size()]);
    }
    batches.emplace_back(makeRowVector(rowType->names(), {dateVector}));
  }

  auto outputDirectory = writeBatchesWithTransforms(
      batches, rowType, {{0, TransformType::kYear, std::nullopt}});

  verifyPartitionCount(outputDirectory->getPath(), {"year(c_date)"}, 6);

  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());
  for (int32_t year = 2020; year <= 2025; year++) {
    const auto expectedDirName = fmt::format("c_date_year={}", year);
    bool foundPartition = false;
    auto yearFilter = [](const int32_t year) -> std::string {
      return fmt::format(
          "YEAR(DATE '{}-01-01')={}",
          std::to_string(year),
          std::to_string(year));
    };

    for (const auto& dir : partitionDirs) {
      const auto dirName = std::filesystem::path(dir).filename().string();
      if (dirName == expectedDirName) {
        foundPartition = true;
        auto datePlan = PlanBuilder()
                            .tableScan(rowType)
                            .filter(yearFilter(year))
                            .singleAggregation({}, {"count(1)"})
                            .planNode();

        auto dateResult = AssertQueryBuilder(datePlan)
                              .splits(createSplitsForDirectory(dir))
                              .copyResults(opPool_.get());

        auto partitionRowCount =
            dateResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
        auto countPlan = PlanBuilder()
                             .tableScan(rowType)
                             .singleAggregation({}, {"count(1)"})
                             .planNode();
        auto countResult = AssertQueryBuilder(countPlan)
                               .splits(createSplitsForDirectory(dir))
                               .copyResults(opPool_.get());
        auto totalPartitionCount =
            countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
        ASSERT_EQ(partitionRowCount, totalPartitionCount);
        break;
      }
    }
    ASSERT_TRUE(foundPartition)
        << "Partition for year " << year << " not found";
  }
}

TEST_F(TransformE2ETest, month) {
  auto rowType = ROW({"c_timestamp"}, {TIMESTAMP()});
  auto batches = createTimestampTestData(
      kDefaultNumBatches, kDefaultRowsPerBatch, rowType);

  auto outputDirectory =
      writeTimestampBatchesWithTransform(batches, TransformType::kMonth);

  auto expectedCounts = buildExpectedCountsFromTimestamps(batches, "month");
  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    size_t dashPos = v.find('-');
    ASSERT_NE(dashPos, std::string::npos) << "Invalid month format: " << v;

    int32_t year = std::stoi(v.substr(0, dashPos));
    int32_t month = std::stoi(v.substr(dashPos + 1));
    std::string filter = fmt::format(
        "YEAR(c_timestamp) = {} AND MONTH(c_timestamp) = {}", year, month);
    std::string monthKey = fmt::format("{:04d}-{:02d}", year, month);
    verifyPartitionData(rowType, dir, filter, expectedCounts[monthKey]);
  }
}

TEST_F(TransformE2ETest, day) {
  auto rowType = ROW({"c_timestamp"}, {TIMESTAMP()});
  auto batches = createTimestampTestData(
      kDefaultNumBatches, kDefaultRowsPerBatch, rowType);

  auto outputDirectory =
      writeTimestampBatchesWithTransform(batches, TransformType::kDay);

  auto expectedCounts = buildExpectedCountsFromTimestamps(batches, "day");
  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    std::vector<std::string> dateParts;
    folly::split('-', v, dateParts);
    ASSERT_EQ(dateParts.size(), 3) << "Invalid day format: " << v;

    int32_t year = std::stoi(dateParts[0]);
    int32_t month = std::stoi(dateParts[1]);
    int32_t day = std::stoi(dateParts[2]);

    std::string filter = fmt::format(
        "YEAR(c_timestamp) = {} AND MONTH(c_timestamp) = {} AND DAY(c_timestamp) = {}",
        year,
        month,
        day);
    std::string dayKey = fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
    verifyPartitionData(rowType, dir, filter, expectedCounts[dayKey]);
  }
}

TEST_F(TransformE2ETest, hour) {
  auto rowType = ROW({"c_timestamp"}, {TIMESTAMP()});
  auto batches = createTimestampTestData(
      kDefaultNumBatches, kDefaultRowsPerBatch, rowType);

  auto outputDirectory =
      writeTimestampBatchesWithTransform(batches, TransformType::kHour);

  auto expectedCounts = buildExpectedCountsFromTimestamps(batches, "hour");
  const auto& partitionDirs = firstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    std::vector<std::string> dateParts;
    folly::split('-', v, dateParts);
    ASSERT_EQ(dateParts.size(), 4) << "Invalid hour format: " << v;

    int32_t year = std::stoi(dateParts[0]);
    int32_t month = std::stoi(dateParts[1]);
    int32_t day = std::stoi(dateParts[2]);
    int32_t hour = std::stoi(dateParts[3]);

    std::string filter = fmt::format(
        "YEAR(c_timestamp) = {} AND MONTH(c_timestamp) = {} AND "
        "DAY(c_timestamp) = {} AND HOUR(c_timestamp) = {}",
        year,
        month,
        day,
        hour);
    std::string hourKey =
        fmt::format("{:04d}-{:02d}-{:02d}-{:02d}", year, month, day, hour);
    verifyPartitionData(rowType, dir, filter, expectedCounts[hourKey]);
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
      rowType,
      {
          {0, TransformType::kIdentity, std::nullopt}, // c_int.
          {0, TransformType::kTruncate, 10}, // truncate(c_int, 10).
          {0, TransformType::kBucket, 4}, // bucket(c_int, 4)
      });

  auto firstLevelDirs = firstLevelDirectories(outputDirectory->getPath());
  ASSERT_GT(firstLevelDirs.size(), 0);
  for (const auto& dir : firstLevelDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    ASSERT_TRUE(dirName.find("c_int=") != std::string::npos)
        << "First level directory " << dirName
        << " should use identity transform";

    auto secondLevelDirs = firstLevelDirectories(dir);
    ASSERT_GT(secondLevelDirs.size(), 0)
        << "No second level directories found in " << dir;

    for (const auto& secondDir : secondLevelDirs) {
      const auto secondDirName =
          std::filesystem::path(secondDir).filename().string();
      ASSERT_TRUE(secondDirName.find("c_int_trunc=") != std::string::npos)
          << "Second level directory " << secondDirName
          << " should use truncate transform";

      auto thirdLevelDirs = firstLevelDirectories(secondDir);
      ASSERT_GT(thirdLevelDirs.size(), 0)
          << "No third level directories found in " << secondDir;

      for (const auto& thirdDir : thirdLevelDirs) {
        const auto thirdDirName =
            std::filesystem::path(thirdDir).filename().string();
        ASSERT_TRUE(thirdDirName.find("c_int_bucket=") != std::string::npos)
            << "Third level directory " << thirdDirName
            << " should use bucket transform";

        // Verify the partition has data.
        auto splits = createSplitsForDirectory(thirdDir);
        auto countPlan = PlanBuilder()
                             .tableScan(rowType)
                             .singleAggregation({}, {"count(1)"})
                             .planNode();
        auto countResult =
            AssertQueryBuilder(countPlan).splits(splits).copyResults(
                opPool_.get());
        ASSERT_GT(
            countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 0)
            << "Leaf partition directory " << thirdDir << " has no data";
      }
    }
  }
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
