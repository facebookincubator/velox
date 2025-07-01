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

#include <exec/tests/utils/AssertQueryBuilder.h>
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::connector::hive::iceberg::test {
class IcebergTransformE2ETest : public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    rowType_ =
        ROW({"c_int",
             "c_bigint",
             "c_varchar",
             "c_date",
             "c_decimal",
             "c_varbinary",
             "c_timestamp"},
            {INTEGER(),
             BIGINT(),
             VARCHAR(),
             DATE(),
             DECIMAL(18, 3),
             VARBINARY(),
             TIMESTAMP()});
    rng_.seed(1);
  }

  std::pair<std::string, std::string> buildFilter(
      const std::string& partitionDir) {
    const auto eq = partitionDir.find('=');
    const auto us = partitionDir.rfind('_', eq - 1);
    const auto column = partitionDir.substr(0, us);
    const auto value = partitionDir.substr(eq + 1);
    return {column, value};
  }

  std::vector<RowVectorPtr> createTestData(
      int32_t numBatches,
      int32_t rowsPerBatch) {
    std::vector<RowVectorPtr> batches;
    for (auto batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
      std::vector<VectorPtr> columns;
      columns.push_back(makeFlatVector<int32_t>(
          rowsPerBatch, [](auto row) { return row % 100; }));
      columns.push_back(makeFlatVector<int64_t>(
          rowsPerBatch, [](auto row) { return row * 1'000; }));
      auto varcharVector = BaseVector::create<FlatVector<StringView>>(
          VARCHAR(), rowsPerBatch, opPool_.get());
      for (auto i = 0; i < rowsPerBatch; i++) {
        std::string s =
            fmt::format("string_long_data_test__{}__{}", i % 10, i % 100);
        varcharVector->set(i, StringView(s));
      }
      columns.push_back(varcharVector);

      auto dateVector = BaseVector::create<FlatVector<int32_t>>(
          DATE(), rowsPerBatch, opPool_.get());
      for (auto i = 0; i < rowsPerBatch; i++) {
        static const std::vector<int32_t> dates = {
            18'262, 18'628, 18'993, 19'358, 19'723, 20'181};
        dateVector->set(i, dates[i % dates.size()]);
      }
      columns.push_back(dateVector);

      auto decimalVector = BaseVector::create<FlatVector<int64_t>>(
          DECIMAL(18, 3), rowsPerBatch, opPool_.get());
      for (auto i = 0; i < rowsPerBatch; i++) {
        decimalVector->set(i, (i % 10 + 1) * 123'456);
      }
      columns.push_back(decimalVector);

      auto varbinaryVector = BaseVector::create<FlatVector<StringView>>(
          VARBINARY(), rowsPerBatch, opPool_.get());
      std::vector<uint8_t> binaryData;
      for (auto i = 0; i < rowsPerBatch; i++) {
        if (i % 5 == 0) {
          for (int32_t j = 0; j < 40; j++) {
            binaryData.push_back(static_cast<uint8_t>(j + (i % 10)));
          }
        } else if (i % 5 == 1) {
          for (int32_t j = 0; j < 40; j++) {
            binaryData.push_back(static_cast<uint8_t>(i % 256));
          }
        } else if (i % 5 == 2) {
          for (int32_t j = 0; j < 40; j++) {
            binaryData.push_back(
                static_cast<uint8_t>(j % 2 == 0 ? 0xAA : 0x55));
          }
        } else {
          for (int32_t j = 0; j < 34; j++) {
            binaryData.push_back(static_cast<uint8_t>(255 - j - (i % 10)));
          }
        }

        varbinaryVector->set(
            i,
            StringView(
                reinterpret_cast<const char*>(binaryData.data()),
                binaryData.size()));
      }
      columns.push_back(varbinaryVector);

      // Add timestamp column with different time values.
      auto timestampVector = BaseVector::create<FlatVector<Timestamp>>(
          TIMESTAMP(), rowsPerBatch, opPool_.get());
      for (auto i = 0; i < rowsPerBatch; i++) {
        // Create timestamps for different years, months, days, and hours
        // to test various transforms.
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
        timestampVector->set(i, timestamps[i % timestamps.size()]);
      }
      columns.push_back(timestampVector);

      batches.push_back(makeRowVector(rowType_->names(), columns));
    }
    return batches;
  }

  std::vector<std::string> listFirstLevelDirectories(
      const std::string& basePath) {
    std::vector<std::string> partitionDirs;
    for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
      if (entry.is_directory()) {
        partitionDirs.push_back(entry.path().string());
      }
    }
    return partitionDirs;
  }

  std::vector<std::string> listDirectoriesRecursively(const std::string& path) {
    std::vector<std::string> allDirs;
    auto firstLevelDirs = listFirstLevelDirectories(path);
    allDirs.insert(allDirs.end(), firstLevelDirs.begin(), firstLevelDirs.end());

    for (const auto& dir : firstLevelDirs) {
      if (std::filesystem::is_directory(dir)) {
        auto subDirs = listDirectoriesRecursively(dir);
        allDirs.insert(allDirs.end(), subDirs.begin(), subDirs.end());
      }
    }

    return allDirs;
  }

  // Verify the number of partitions and their naming convention.
  void verifyPartitionCount(
      const std::string& outputPath,
      const std::vector<std::string>& partitionTransforms,
      const int32_t expectedPartitionCount) {
    const auto partitionDirs = listFirstLevelDirectories(outputPath);

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
    const auto splits = createSplitsForDirectory(partitionPath);

    const auto countPlan = PlanBuilder()
                               .tableScan(rowType)
                               .singleAggregation({}, {"count(1)"})
                               .planNode();

    const auto countResult =
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
    const auto dataResult =
        AssertQueryBuilder(dataPlan).splits(splits).copyResults(opPool_.get());
    ASSERT_EQ(dataResult->size(), 1);
    const auto filteredRowCount =
        dataResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
    if (!skipRowCountCheck) {
      ASSERT_EQ(filteredRowCount, expectedRowCount);
    } else {
      // Just verify that the filter matches all rows in the partition.
      ASSERT_EQ(filteredRowCount, actualRowCount);
    }
  }

  folly::Random::DefaultGenerator rng_;
};

TEST_F(IcebergTransformE2ETest, identityPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyPartitionCount(outputDirectory->getPath(), {"c_int"}, rowsPerBatch);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    verifyPartitionData(rowType_, dir, dirName, numBatches);
  }
}

TEST_F(IcebergTransformE2ETest, truncatePartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kTruncate, 10}});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyPartitionCount(outputDirectory->getPath(), {"truncate(c_int, 10)"}, 10);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);
  const auto partitionDirs =
      listFirstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const std::string dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    const std::string filter =
        c + ">=" + v + " AND " + c + "<" + std::to_string(std::stoi(v) + 10);
    verifyPartitionData(
        rowType_, dir, filter, 20); // 10 values per batch * 2 batches.
  }
}

TEST_F(IcebergTransformE2ETest, bucketPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), {{2, TransformType::kBucket, 4}});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  // Verify the number of partitions (should be at most 4 buckets).
  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  ASSERT_EQ(partitionDirs.size(), 4);

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  int32_t totalRowsInPartitions = 0;
  for (const auto& dir : partitionDirs) {
    auto splits = createSplitsForDirectory(dir);
    auto countPlan = PlanBuilder()
                         .tableScan(rowType_)
                         .singleAggregation({}, {"count(1)"})
                         .planNode();
    auto countResult =
        AssertQueryBuilder(countPlan).splits(splits).copyResults(opPool_.get());

    auto partitionRowCount =
        countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
    totalRowsInPartitions += partitionRowCount;
    ASSERT_GE(partitionRowCount, 0);
  }

  ASSERT_EQ(totalRowsInPartitions, numBatches * rowsPerBatch);

  // Verify that each partition contains only rows with the same bucket value.
  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    const auto equalsPos = dirName.find('=');
    ASSERT_NE(equalsPos, std::string::npos);
    auto dataPlan =
        PlanBuilder().tableScan(rowType_).project({"c_varchar"}).planNode();
    auto dataResult = AssertQueryBuilder(dataPlan)
                          .splits(createSplitsForDirectory(dir))
                          .copyResults(opPool_.get());
    // Verify that all rows in this partition have the same bucket hash value.
    auto varcharColumn = dataResult->childAt(0)->asFlatVector<StringView>();
    for (auto i = 0; i < dataResult->size(); i++) {
      StringView value = varcharColumn->valueAt(i);
      auto valuePlan = PlanBuilder()
                           .tableScan(rowType_)
                           .filter(fmt::format("c_varchar = '{}'", value.str()))
                           .project({"c_varchar"})
                           .planNode();

      auto valueResult =
          AssertQueryBuilder(valuePlan)
              .splits(createSplitsForDirectory(outputDirectory->getPath()))
              .copyResults(opPool_.get());
      auto valueCount = valueResult->size();
      auto partitionValuePlan =
          PlanBuilder()
              .tableScan(rowType_)
              .filter(fmt::format("c_varchar = '{}'", value.str()))
              .project({"c_varchar"})
              .planNode();
      auto partitionValueResult = AssertQueryBuilder(partitionValuePlan)
                                      .splits(createSplitsForDirectory(dir))
                                      .copyResults(opPool_.get());

      ASSERT_EQ(partitionValueResult->size(), valueCount);
    }
  }
}

TEST_F(IcebergTransformE2ETest, yearPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{3, TransformType::kYear, std::nullopt}});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  // Verify the number of partitions (should be 6 for years 2020-2025).
  verifyPartitionCount(outputDirectory->getPath(), {"year(c_date)"}, 6);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());

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
                            .tableScan(rowType_)
                            .filter(yearFilter(year))
                            .singleAggregation({}, {"count(1)"})
                            .planNode();

        auto dateResult = AssertQueryBuilder(datePlan)
                              .splits(createSplitsForDirectory(dir))
                              .copyResults(opPool_.get());

        auto partitionRowCount =
            dateResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0);
        auto countPlan = PlanBuilder()
                             .tableScan(rowType_)
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

TEST_F(IcebergTransformE2ETest, varbinaryTruncatePartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();

  std::vector<PartitionField> partitionTransforms = {
      {5, TransformType::kTruncate, 36}};
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    // For binary data, we need to use a different approach for filtering.
    const auto filter = c + " IS NOT NULL";
    // Verify the partition has data.
    verifyPartitionData(rowType_, dir, filter, 0, true);
  }
}

TEST_F(IcebergTransformE2ETest, multipleTransformsOnSameColumn) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto vectors = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}, // c_int.
      {0, TransformType::kTruncate, 10}, // truncate(c_int, 10).
      {0, TransformType::kBucket, 4}}; // bucket(c_int, 4).
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto firstLevelDirs = listFirstLevelDirectories(outputDirectory->getPath());
  ASSERT_GT(firstLevelDirs.size(), 0);
  for (const auto& dir : firstLevelDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    ASSERT_TRUE(dirName.find("c_int=") != std::string::npos)
        << "First level directory " << dirName
        << " should use identity transform";

    auto secondLevelDirs = listFirstLevelDirectories(dir);
    ASSERT_GT(secondLevelDirs.size(), 0)
        << "No second level directories found in " << dir;

    for (const auto& secondDir : secondLevelDirs) {
      const auto secondDirName =
          std::filesystem::path(secondDir).filename().string();
      ASSERT_TRUE(secondDirName.find("c_int_trunc=") != std::string::npos)
          << "Second level directory " << secondDirName
          << " should use truncate transform";

      auto thirdLevelDirs = listFirstLevelDirectories(secondDir);
      ASSERT_GT(thirdLevelDirs.size(), 0)
          << "No third level directories found in " << secondDir;

      for (const auto& thirdDir : thirdLevelDirs) {
        const auto thirdDirName =
            std::filesystem::path(thirdDir).filename().string();
        ASSERT_TRUE(thirdDirName.find("c_int_bucket=") != std::string::npos)
            << "Third level directory " << thirdDirName
            << " should use bucket transform";

        auto leafDir = thirdDir;
        auto intValue =
            std::stoi(std::filesystem::path(dir).filename().string().substr(
                6)); // c_int=X.
        auto truncValue = std::stoi(
            std::filesystem::path(secondDir).filename().string().substr(
                12)); // c_int_trunc=X.
        std::string filter = fmt::format(
            "c_int = {} AND c_int >= {} AND c_int < {}",
            intValue,
            truncValue,
            truncValue + 10);

        // Verify the partition has data.
        auto splits = createSplitsForDirectory(leafDir);
        auto countPlan = PlanBuilder()
                             .tableScan(rowType_)
                             .singleAggregation({}, {"count(1)"})
                             .planNode();
        auto countResult =
            AssertQueryBuilder(countPlan).splits(splits).copyResults(
                opPool_.get());
        ASSERT_GT(
            countResult->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 0)
            << "Leaf partition directory " << leafDir << " has no data";
      }
    }
  }
}

TEST_F(IcebergTransformE2ETest, timestampYearPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto batches = createTestData(numBatches, rowsPerBatch);

  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {6, TransformType::kYear, std::nullopt}}; // c_timestamp column.
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);

  for (const auto& vector : batches) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  std::unordered_map<int32_t, int32_t> yearToExpectedCount;

  for (const auto& batch : batches) {
    auto timestampVector = batch->childAt(6)->as<SimpleVector<Timestamp>>();
    for (auto i = 0; i < batch->size(); i++) {
      if (!timestampVector->isNullAt(i)) {
        Timestamp ts = timestampVector->valueAt(i);
        std::tm tm;
        if (Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
          int32_t year = tm.tm_year + 1900;
          yearToExpectedCount[year]++;
        }
      }
    }
  }

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();
    auto [c, v] = buildFilter(dirName);
    auto year = std::stoi(v);
    std::string filter = fmt::format("YEAR(c_timestamp) = {}", year);
    auto expectedRowCount = yearToExpectedCount.at(year);
    verifyPartitionData(rowType_, dir, filter, expectedRowCount);
  }
}

TEST_F(IcebergTransformE2ETest, timestampMonthPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;

  auto batches = createTestData(numBatches, rowsPerBatch);

  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {6, TransformType::kMonth, std::nullopt}}; // c_timestamp column.
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);

  for (const auto& vector : batches) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  std::unordered_map<std::string, int32_t> monthToExpectedCount;

  for (const auto& batch : batches) {
    auto timestampVector = batch->childAt(6)->as<SimpleVector<Timestamp>>();
    for (auto i = 0; i < batch->size(); i++) {
      if (!timestampVector->isNullAt(i)) {
        Timestamp ts = timestampVector->valueAt(i);
        std::tm tm;
        if (Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
          int32_t year = tm.tm_year + 1900;
          int32_t month = tm.tm_mon + 1;
          std::string monthKey = fmt::format("{:04d}-{:02d}", year, month);
          monthToExpectedCount[monthKey]++;
        }
      }
    }
  }

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
    auto expectedCount = monthToExpectedCount[monthKey];
    verifyPartitionData(rowType_, dir, filter, expectedCount);
  }
}

TEST_F(IcebergTransformE2ETest, timestampDayPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto batches = createTestData(numBatches, rowsPerBatch);
  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {6, TransformType::kDay, std::nullopt}}; // c_timestamp column
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);

  for (const auto& vector : batches) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  std::unordered_map<std::string, int32_t> dayToExpectedCount;
  for (const auto& batch : batches) {
    auto timestampVector = batch->childAt(6)->as<SimpleVector<Timestamp>>();
    for (auto i = 0; i < batch->size(); i++) {
      if (!timestampVector->isNullAt(i)) {
        Timestamp ts = timestampVector->valueAt(i);
        std::tm tm;
        if (Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
          int32_t year = tm.tm_year + 1900;
          int32_t month = tm.tm_mon + 1;
          int32_t day = tm.tm_mday;
          std::string dayKey =
              fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
          dayToExpectedCount[dayKey]++;
        }
      }
    }
  }

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

    // Get expected count for this day.
    std::string dayKey = fmt::format("{:04d}-{:02d}-{:02d}", year, month, day);
    auto expectedCount = dayToExpectedCount[dayKey];
    verifyPartitionData(rowType_, dir, filter, expectedCount);
  }
}

TEST_F(IcebergTransformE2ETest, timestampHourPartitioning) {
  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  auto batches = createTestData(numBatches, rowsPerBatch);

  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {6, TransformType::kHour, std::nullopt}}; // c_timestamp column.
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);

  for (const auto& vector : batches) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto partitionDirs = listFirstLevelDirectories(outputDirectory->getPath());
  std::unordered_map<std::string, int32_t> hourToExpectedCount;

  for (const auto& batch : batches) {
    auto timestampVector = batch->childAt(6)->as<SimpleVector<Timestamp>>();
    for (auto i = 0; i < batch->size(); i++) {
      if (!timestampVector->isNullAt(i)) {
        Timestamp ts = timestampVector->valueAt(i);
        std::tm tm;
        if (Timestamp::epochToCalendarUtc(ts.getSeconds(), tm)) {
          int32_t year = tm.tm_year + 1900;
          int32_t month = tm.tm_mon + 1;
          int32_t day = tm.tm_mday;
          int32_t hour = tm.tm_hour;
          std::string hourKey = fmt::format(
              "{:04d}-{:02d}-{:02d}-{:02d}", year, month, day, hour);
          hourToExpectedCount[hourKey]++;
        }
      }
    }
  }

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
    auto expectedCount = hourToExpectedCount[hourKey];
    verifyPartitionData(rowType_, dir, filter, expectedCount);
  }
}

TEST_F(IcebergTransformE2ETest, partitionFolderNamingConventions) {
  auto intVector = makeFlatVector<int32_t>(1, [](auto) { return 42; });
  auto bigintVector =
      makeFlatVector<int64_t>(1, [](auto) { return 9'876'543'210; });
  auto varcharVector =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 1, opPool_.get());
  varcharVector->set(0, StringView("test string"));

  auto varcharVector2 =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 1, opPool_.get());
  varcharVector2->setNull(0, true);

  auto decimalVector =
      BaseVector::create<FlatVector<int64_t>>(DECIMAL(18, 3), 1, opPool_.get());
  decimalVector->set(0, 1'234'567'890);

  auto varbinaryVector =
      BaseVector::create<FlatVector<StringView>>(VARBINARY(), 1, opPool_.get());
  std::string binaryData = "binary\0data\1\2\3";
  varbinaryVector->set(0, StringView(binaryData));

  auto rowVector = makeRowVector(
      {"c_int",
       "c_bigint",
       "c_varchar",
       "c_varchar2",
       "c_decimal",
       "c_varbinary"},
      {intVector,
       bigintVector,
       varcharVector,
       varcharVector2,
       decimalVector,
       varbinaryVector});
  auto outputDirectory = TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}, // c_int.
      {1, TransformType::kIdentity, std::nullopt}, // c_bigint.
      {2, TransformType::kIdentity, std::nullopt}, // c_varchar.
      {4, TransformType::kIdentity, std::nullopt}, // c_decimal.
      {5, TransformType::kIdentity, std::nullopt}, // c_varbinary.
      {3, TransformType::kIdentity, std::nullopt} // c_varchar2.
  };
  auto dataSink = createIcebergDataSink(
      asRowType(rowVector->type()),
      outputDirectory->getPath(),
      partitionTransforms);

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      asRowType(rowVector->type()), outputDirectory->getPath(), 1);
  auto partitionDirs = listDirectoriesRecursively(outputDirectory->getPath());

  const std::string expectedIntFolder = "c_int=42";
  const std::string expectedBigintFolder = "c_bigint=9876543210";
  const std::string expectedVarcharFolder = "c_varchar=test+string";
  const std::string expectedVarcharFolder2 = "c_varchar2=null";
  const std::string expectedDecimalFolder = "c_decimal=1234567.890";
  const std::string expectedVarbinary = "c_varbinary=" +
      encoding::Base64::encode(binaryData.data(), binaryData.size());

  bool foundIntPartition = false;
  bool foundBigintPartition = false;
  bool foundVarcharPartition = false;
  bool foundVarcharPartition2 = false;
  bool foundDecimalPartition = false;
  bool foundVarbinaryPartition = false;

  for (const auto& dir : partitionDirs) {
    const auto dirName = std::filesystem::path(dir).filename().string();

    if (dirName == expectedIntFolder) {
      foundIntPartition = true;
      verifyPartitionData(asRowType(rowVector->type()), dir, "c_int = 42", 1);
    } else if (dirName == expectedBigintFolder) {
      foundBigintPartition = true;
      verifyPartitionData(
          asRowType(rowVector->type()), dir, "c_bigint = 9876543210", 1);
    } else if (dirName == expectedVarcharFolder) {
      foundVarcharPartition = true;
      verifyPartitionData(
          asRowType(rowVector->type()), dir, "c_varchar = 'test string'", 1);
    } else if (dirName == expectedVarcharFolder2) {
      foundVarcharPartition2 = true;
      verifyPartitionData(
          asRowType(rowVector->type()), dir, "c_varchar2 IS NULL", 1);
    } else if (dirName == expectedDecimalFolder) {
      foundDecimalPartition = true;
      verifyPartitionData(
          asRowType(rowVector->type()),
          dir,
          "c_decimal = DECIMAL '1234567.890'",
          1);
    } else if (dirName.find(expectedVarbinary) == 0) {
      foundVarbinaryPartition = true;
      verifyPartitionData(
          asRowType(rowVector->type()), dir, "c_varbinary IS NOT NULL", 1);
    }
  }

  ASSERT_TRUE(foundIntPartition)
      << "Integer partition folder not found: " << expectedIntFolder;
  ASSERT_TRUE(foundBigintPartition)
      << "Bigint partition folder not found: " << expectedBigintFolder;
  ASSERT_TRUE(foundVarcharPartition)
      << "Varchar partition folder not found: " << expectedVarcharFolder;
  ASSERT_TRUE(foundVarcharPartition2)
      << "Varchar2 partition folder not found: " << expectedVarcharFolder2;
  ASSERT_TRUE(foundDecimalPartition)
      << "Decimal partition folder not found: " << expectedDecimalFolder;
  ASSERT_TRUE(foundVarbinaryPartition)
      << "Varbinary partition folder not found with prefix: "
      << expectedVarbinary;
}

} // namespace facebook::velox::connector::hive::iceberg::test
