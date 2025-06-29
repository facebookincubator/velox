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
    FLAGS_velox_exception_user_stacktrace_enabled = true;
    FLAGS_velox_exception_system_stacktrace_enabled = true;
    FLAGS_velox_exception_user_stacktrace_rate_limit_ms = 0;
    FLAGS_velox_exception_system_stacktrace_rate_limit_ms = 0;
    IcebergTestBase::SetUp();
    rowType_ = ROW(
        {"c_int",
         "c_bigint",
         "c_varchar",
         "c_date",
         "c_decimal",
         "c_varbinary"},
        {INTEGER(), BIGINT(), VARCHAR(), DATE(), DECIMAL(18, 3), VARBINARY()});
    rng_.seed(1);
  }

  static std::pair<std::string, std::string> buildFilter(
      const std::string& partitionDir) {
    const auto eq = partitionDir.find('=');
    const auto us = partitionDir.rfind('_', eq - 1);
    const auto column = partitionDir.substr(0, us);
    const auto value = partitionDir.substr(eq + 1);
    return {column, value};
  }

  std::vector<RowVectorPtr> createTestData(int numBatches, int rowsPerBatch) {
    std::vector<RowVectorPtr> batches;
    for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
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
      for (vector_size_t i = 0; i < rowsPerBatch; i++) {
        static const std::vector<int32_t> dates = {
            18'262, 18'628, 18'993, 19'358, 19'723, 20'181};
        dateVector->set(i, dates[i % dates.size()]);
      }
      columns.push_back(dateVector);

      auto decimalVector = BaseVector::create<FlatVector<int64_t>>(
          DECIMAL(18, 3), rowsPerBatch, opPool_.get());
      for (vector_size_t i = 0; i < rowsPerBatch; i++) {
        decimalVector->set(i, (i % 10 + 1) * 123'456);
      }
      columns.push_back(decimalVector);

      auto varbinaryVector = BaseVector::create<FlatVector<StringView>>(
          VARBINARY(), rowsPerBatch, opPool_.get());
      std::vector<uint8_t> binaryData;
      for (vector_size_t i = 0; i < rowsPerBatch; i++) {
        if (i % 5 == 0) {
          for (int j = 0; j < 40; j++) {
            binaryData.push_back(static_cast<uint8_t>(j + (i % 10)));
          }
        } else if (i % 5 == 1) {
          for (int j = 0; j < 40; j++) {
            binaryData.push_back(static_cast<uint8_t>(i % 256));
          }
        } else if (i % 5 == 2) {
          for (int j = 0; j < 40; j++) {
            binaryData.push_back(
                static_cast<uint8_t>(j % 2 == 0 ? 0xAA : 0x55));
          }
        } else {
          for (int j = 0; j < 34; j++) {
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

      batches.push_back(makeRowVector(rowType_->names(), columns));
    }
    return batches;
  }

  static std::vector<std::string> listFirstLevelDirectories(
      const std::string& basePath) {
    std::vector<std::string> partitionDirs;
    for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
      if (entry.is_directory()) {
        partitionDirs.push_back(entry.path().string());
      }
    }
    return partitionDirs;
  }

  // Verify the number of partitions and their naming convention.
  void verifyPartitionCount(
      const std::string& outputPath,
      const std::vector<std::string>& partitionTransforms,
      const int expectedPartitionCount) {
    const auto dataPath = fmt::format("{}/data", outputPath);
    const auto partitionDirs = listFirstLevelDirectories(dataPath);

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
      int expectedRowCount) {
    auto dataPath = fmt::format("{}/data", outputPath);
    auto splits = createSplitsForDirectory(dataPath);

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
      const int expectedRowCount,
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
  auto dataSink =
      createIcebergDataSink(rowType_, outputDirectory->getPath(), {"c_int"});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyPartitionCount(outputDirectory->getPath(), {"c_int"}, rowsPerBatch);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto partitionDirs = listFirstLevelDirectories(dataPath);
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
      rowType_, outputDirectory->getPath(), {"truncate(c_int, 10)"});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyPartitionCount(outputDirectory->getPath(), {"truncate(c_int, 10)"}, 10);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  const auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  const auto partitionDirs = listFirstLevelDirectories(dataPath);

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
      rowType_, outputDirectory->getPath(), {"bucket(c_varchar, 4)"});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  // Verify the number of partitions (should be at most 4 buckets).
  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto partitionDirs = listFirstLevelDirectories(dataPath);
  ASSERT_EQ(partitionDirs.size(), 4);

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  int totalRowsInPartitions = 0;
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
    for (vector_size_t i = 0; i < dataResult->size(); i++) {
      StringView value = varcharColumn->valueAt(i);
      // Create a plan to find all rows with this c_varchar value.
      auto valuePlan = PlanBuilder()
                           .tableScan(rowType_)
                           .filter(fmt::format("c_varchar = '{}'", value.str()))
                           .project({"c_varchar"})
                           .planNode();

      auto valueResult = AssertQueryBuilder(valuePlan)
                             .splits(createSplitsForDirectory(dataPath))
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
      rowType_, outputDirectory->getPath(), {"year(c_date)"});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  // Verify the number of partitions (should be 6 for years 2020-2025).
  verifyPartitionCount(outputDirectory->getPath(), {"year(c_date)"}, 6);
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto partitionDirs = listFirstLevelDirectories(dataPath);
  std::unordered_map<int, int> yearToYearsSince1970 = {
      {2020, 50}, {2021, 51}, {2022, 52}, {2023, 53}, {2024, 54}, {2025, 55}};

  for (const auto& [year, yearsSince1970] : yearToYearsSince1970) {
    const auto expectedDirName = fmt::format("c_date_year={}", yearsSince1970);
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

        int partitionRowCount =
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

  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), {"truncate(c_varbinary, 36)"});

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto partitionDirs = listFirstLevelDirectories(dataPath);

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
  std::vector<std::string> partitionTransforms = {
      "c_int", "truncate(c_int, 10)", "bucket(c_int, 4)"};
  auto dataSink = createIcebergDataSink(
      rowType_, outputDirectory->getPath(), partitionTransforms);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      rowType_, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto firstLevelDirs = listFirstLevelDirectories(dataPath);
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
        int intValue =
            std::stoi(std::filesystem::path(dir).filename().string().substr(
                6)); // c_int=X.
        int truncValue = std::stoi(
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

TEST_F(IcebergTransformE2ETest, structSubfieldPartitioning) {
  auto structType = ROW({"id", "name"}, {INTEGER(), BIGINT()});
  auto testRowType =
      ROW({"c_int", "c_bigint", "c_struct"}, {INTEGER(), BIGINT(), structType});
  const int32_t numRows = 10'000;
  auto idVector =
      makeFlatVector<int32_t>(numRows, [](auto row) { return row; });
  auto nameVector =
      makeFlatVector<int64_t>(numRows, [](auto row) { return row; });
  auto structVector =
      makeRowVector(structType->names(), {idVector, nameVector});
  auto intVector =
      makeFlatVector<int32_t>(numRows, [](auto row) { return row * 10; });
  auto bigintVector =
      makeFlatVector<int64_t>(numRows, [](auto row) { return row * 100; });

  auto rowVector = makeRowVector(
      {"c_int", "c_bigint", "c_struct"},
      {intVector, bigintVector, structVector});

  std::vector<RowVectorPtr> vectors;
  vectors.emplace_back(rowVector);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      testRowType, outputDirectory->getPath(), {"bucket(c_struct.id, 4)"});
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  verifyTotalRowCount(testRowType, outputDirectory->getPath(), numRows);
}

TEST_F(IcebergTransformE2ETest, multipleStructSubfieldPartitioning) {
  auto addressType =
      ROW({"city", "state", "zip"}, {VARCHAR(), VARCHAR(), INTEGER()});
  auto personType =
      ROW({"id", "name", "age"}, {INTEGER(), VARCHAR(), INTEGER()});
  auto testRowType = ROW(
      {"c_int", "c_person", "c_address"}, {INTEGER(), personType, addressType});

  constexpr auto numBatches = 2;
  constexpr auto rowsPerBatch = 100;
  std::vector<RowVectorPtr> vectors;

  for (auto batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    std::vector<VectorPtr> columns;
    columns.push_back(makeFlatVector<int32_t>(
        rowsPerBatch, [](auto row) { return row % 100; }));

    std::vector<VectorPtr> personFields;
    personFields.push_back(makeFlatVector<int32_t>(
        rowsPerBatch, [](auto row) { return (row % 3) + 1; }));

    auto nameVector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), rowsPerBatch, opPool_.get());
    for (auto i = 0; i < rowsPerBatch; i++) {
      std::string name = fmt::format("person{}", (i % 3) + 1);
      nameVector->set(i, StringView(name));
    }
    personFields.push_back(nameVector);
    personFields.push_back(makeFlatVector<int32_t>(
        rowsPerBatch, [](auto row) { return 20 + (row % 10); }));

    auto personVector = std::make_shared<RowVector>(
        opPool_.get(),
        personType,
        BufferPtr(nullptr),
        rowsPerBatch,
        personFields);
    columns.push_back(personVector);

    std::vector<VectorPtr> addressFields;
    auto cityVector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), rowsPerBatch, opPool_.get());
    for (auto i = 0; i < rowsPerBatch; i++) {
      std::string city = fmt::format("City{}", (i % 4) + 1);
      cityVector->set(i, StringView(city));
    }
    addressFields.push_back(cityVector);
    auto stateVector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), rowsPerBatch, opPool_.get());
    for (auto i = 0; i < rowsPerBatch; i++) {
      std::string state = fmt::format("State{}", (i % 2) + 1);
      stateVector->set(i, StringView(state));
    }
    addressFields.push_back(stateVector);
    addressFields.push_back(makeFlatVector<int32_t>(
        rowsPerBatch, [](auto row) { return 10000 + (row % 5) * 1000; }));

    auto addressVector = std::make_shared<RowVector>(
        opPool_.get(),
        addressType,
        BufferPtr(nullptr),
        rowsPerBatch,
        addressFields);
    columns.push_back(addressVector);

    vectors.push_back(makeRowVector(testRowType->names(), columns));
  }

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      testRowType,
      outputDirectory->getPath(),
      {"c_person.id", "c_address.state"});

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  verifyTotalRowCount(
      testRowType, outputDirectory->getPath(), numBatches * rowsPerBatch);

  auto dataPath = fmt::format("{}/data", outputDirectory->getPath());
  auto firstLevelDirs = listFirstLevelDirectories(dataPath);
  ASSERT_EQ(firstLevelDirs.size(), 3);
  for (int personId = 1; personId <= 3; personId++) {
    const auto expectedFirstLevelDir = fmt::format("c_person.id={}", personId);
    bool foundFirstLevel = false;

    for (const auto& firstDir : firstLevelDirs) {
      const auto firstDirName =
          std::filesystem::path(firstDir).filename().string();
      if (firstDirName == expectedFirstLevelDir) {
        foundFirstLevel = true;

        auto secondLevelDirs = listFirstLevelDirectories(firstDir);
        ASSERT_EQ(secondLevelDirs.size(), 2);

        for (int stateNum = 1; stateNum <= 2; stateNum++) {
          const auto expectedSecondLevelDir =
              fmt::format("c_address.state=State{}", stateNum);
          bool foundSecondLevel = false;

          for (const auto& secondDir : secondLevelDirs) {
            const auto secondDirName =
                std::filesystem::path(secondDir).filename().string();
            if (secondDirName == expectedSecondLevelDir) {
              foundSecondLevel = true;

              std::string filter = fmt::format(
                  "c_person.id = {} AND c_address.state = 'State{}'",
                  personId,
                  stateNum);

              verifyPartitionData(testRowType, secondDir, filter, 0, true);
              break;
            }
          }

          ASSERT_TRUE(foundSecondLevel) << "Second-level partition for state "
                                        << stateNum << " not found";
        }
        break;
      }
    }

    ASSERT_TRUE(foundFirstLevel)
        << "First-level partition for person.id = " << personId << " not found";
  }
}

} // namespace facebook::velox::connector::hive::iceberg::test
