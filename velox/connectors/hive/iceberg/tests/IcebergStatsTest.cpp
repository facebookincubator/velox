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

#include <gtest/gtest.h>
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg::test {
class IcebergStatsTest : public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    rowType_ =
        ROW({"c_int", "c_bigint", "c_varchar", "c_date", "c_decimal"},
            {INTEGER(), BIGINT(), VARCHAR(), DATE(), DECIMAL(18, 3)});
  }

  void TearDown() override {
    IcebergTestBase::TearDown();
  }
};

TEST_F(IcebergStatsTest, mixedNullTest) {
  auto rowType = ROW({"int_col"}, {INTEGER()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedIntNulls = 34;

  auto rowVector = makeRowVector({makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * 10; }, nullEvery(3))});
  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t intColId = 1;
  EXPECT_EQ(stats->valueCounts.at(intColId), size)
      << "Int column value count incorrect";
  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";

  EXPECT_EQ(stats->nullValueCounts.at(intColId), expectedIntNulls)
      << "Int column null count incorrect";
  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(intColId));
  auto lb = *reinterpret_cast<int32_t*>(lowerBounds.data());
  EXPECT_EQ(lb, 10);
  EXPECT_FALSE(stats->lowerBounds.at(intColId).empty())
      << "Int column should have non-empty lower bound";
  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(intColId));
  auto ub = *reinterpret_cast<int32_t*>(upperBounds.data());
  EXPECT_EQ(ub, 980);
  EXPECT_FALSE(stats->upperBounds.at(intColId).empty())
      << "Int column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, bigintStatsTest) {
  auto rowType = ROW({"bigint_col"}, {BIGINT()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 25;

  auto rowVector = makeRowVector({makeFlatVector<int64_t>(
      size,
      [](vector_size_t row) { return row * 1'000'000'000LL; },
      nullEvery(4))});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t bigintColId = 1;
  EXPECT_EQ(stats->valueCounts.at(bigintColId), size)
      << "Bigint column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(bigintColId), expectedNulls)
      << "Bigint column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(bigintColId));
  auto lb = *reinterpret_cast<int64_t*>(lowerBounds.data());
  EXPECT_EQ(lb, 1'000'000'000LL);

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(bigintColId));
  auto ub = *reinterpret_cast<int64_t*>(upperBounds.data());
  EXPECT_EQ(ub, 99'000'000'000LL);
  folly::dynamic json = stats->toJson();
  std::string jsonstring = folly::toJson(json);
}

TEST_F(IcebergStatsTest, decimalStatsTest) {
  auto rowType = ROW({"decimal_col"}, {DECIMAL(38, 3)});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 20;

  auto rowVector = makeRowVector({makeFlatVector<int128_t>(
      size,
      [](vector_size_t row) { return HugeInt::build(row, row * 123); },
      nullEvery(5),
      DECIMAL(38, 3))});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t decimalColId = 1;
  EXPECT_EQ(stats->valueCounts.at(decimalColId), size)
      << "Decimal column value count incorrect";
  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(decimalColId), expectedNulls)
      << "Decimal column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";
  EXPECT_FALSE(stats->lowerBounds.at(decimalColId).empty())
      << "Decimal column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(decimalColId).empty())
      << "Decimal column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, varcharStatsTest) {
  auto rowType = ROW({"varchar_col"}, {VARCHAR()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (auto i = 0; i < size; ++i) {
    if (i % 6 == 0) {
      flatVarcharVector->setNull(i, true);
      expectedNulls++;
    } else {
      std::string value =
          "Customer#00000" + std::to_string(i) + "_" + std::string(i % 10, 'a');
      flatVarcharVector->set(i, StringView(value));
    }
  }

  auto rowVector = makeRowVector({varcharVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t varcharColId = 1;
  EXPECT_EQ(stats->valueCounts.at(varcharColId), size)
      << "Varchar column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(varcharColId), expectedNulls)
      << "Varchar column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";
  EXPECT_FALSE(stats->lowerBounds.at(varcharColId).empty())
      << "Varchar column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(varcharColId).empty())
      << "Varchar column should have non-empty upper bound";

  // Decode and verify string bounds.
  std::string lowerBound =
      encoding::Base64::decode(stats->lowerBounds.at(varcharColId));
  std::string upperBound =
      encoding::Base64::decode(stats->upperBounds.at(varcharColId));
  EXPECT_TRUE(lowerBound.find("Customer#00000") != std::string::npos)
      << "Lower bound should contain 'Customer#00000'";
  EXPECT_TRUE(upperBound.find("Customer#000009") != std::string::npos)
      << "Upper bound should contain 'Customer#000009'";
}

TEST_F(IcebergStatsTest, varbinaryStatsTest) {
  auto rowType = ROW({"varbinary_col"}, {VARBINARY()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  auto varbinaryVector = BaseVector::create(VARBINARY(), size, opPool_.get());
  auto flatVarbinaryVector = varbinaryVector->asFlatVector<StringView>();
  for (auto i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatVarbinaryVector->setNull(i, true);
      expectedNulls++;
    } else {
      // Create binary values with varying content.
      std::string value(17, 11);
      value[0] = static_cast<char>(i % 256);
      value[1] = static_cast<char>((i * 3) % 256);
      value[2] = static_cast<char>((i * 7) % 256);
      value[3] = static_cast<char>((i * 11) % 256);
      flatVarbinaryVector->set(i, StringView(value));
    }
  }

  auto rowVector = makeRowVector({varbinaryVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t varbinaryColId = 1;
  EXPECT_EQ(stats->valueCounts.at(varbinaryColId), size)
      << "Varbinary column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(varbinaryColId), expectedNulls)
      << "Varbinary column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";
  EXPECT_FALSE(stats->lowerBounds.at(varbinaryColId).empty())
      << "Varbinary column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(varbinaryColId).empty())
      << "Varbinary column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, varbinaryStatsTest2) {
  auto rowType = ROW({"varbinary_col"}, {VARBINARY()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kBucket, 4}};
  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);
  constexpr vector_size_t size = 10;

  auto varbinaryVector = BaseVector::create(VARBINARY(), size, opPool_.get());
  auto flatVarbinaryVector = varbinaryVector->asFlatVector<StringView>();
  std::string values[] = {
      "01020304",
      "05060708",
      "090A0B0C",
      "0D0E0F10",
      "11121314",
      "15161718",
      "191A1B1C",
      "1D1E1F20",
      "21222324",
      "25262728"};
  for (auto i = 0; i < size; ++i) {
    flatVarbinaryVector->set(i, StringView(values[i]));
  }

  auto rowVector = makeRowVector({"varbinary_col"}, {varbinaryVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 3) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, 5) << "Record count should match input size";

  constexpr int32_t varbinaryColId = 1;
  EXPECT_EQ(stats->valueCounts.at(varbinaryColId), 5);
}

TEST_F(IcebergStatsTest, multipleDataTypesTest) {
  auto rowType = ROW(
      {"int_col", "bigint_col", "decimal_col", "varchar_col", "varbinary_col"},
      {INTEGER(), BIGINT(), DECIMAL(38, 3), VARCHAR(), VARBINARY()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;

  int32_t expectedIntNulls = 34;
  int32_t expectedBigintNulls = 25;
  int32_t expectedDecimalNulls = 20;
  int32_t expectedVarcharNulls = 0;
  int32_t expectedVarbinaryNulls = 0;
  // Create columns with different null patterns
  auto intVector = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * 10; }, nullEvery(3));

  auto bigintVector = makeFlatVector<int64_t>(
      size,
      [](vector_size_t row) { return row * 1'000'000'000LL; },
      nullEvery(4));

  auto decimalVector = makeFlatVector<int128_t>(
      size,
      [](vector_size_t row) { return HugeInt::build(row, row * 12'345); },
      nullEvery(5),
      DECIMAL(38, 3));

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (auto i = 0; i < size; ++i) {
    if (i % 6 == 0) {
      flatVarcharVector->setNull(i, true);
      expectedVarcharNulls++;
    } else {
      std::string value = "str_" + std::to_string(i);
      flatVarcharVector->set(i, StringView(value));
    }
  }

  auto varbinaryVector = BaseVector::create(VARBINARY(), size, opPool_.get());
  auto flatVarbinaryVector = varbinaryVector->asFlatVector<StringView>();
  for (auto i = 0; i < size; ++i) {
    if (i % 7 == 0) {
      flatVarbinaryVector->setNull(i, true);
      expectedVarbinaryNulls++;
    } else {
      std::string value(4, 0);
      value[0] = static_cast<char>(i % 256);
      value[1] = static_cast<char>((i * 3) % 256);
      value[2] = static_cast<char>((i * 7) % 256);
      value[3] = static_cast<char>((i * 11) % 256);
      flatVarbinaryVector->set(i, StringView(value));
    }
  }

  auto rowVector = makeRowVector(
      {intVector, bigintVector, decimalVector, varcharVector, varbinaryVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";

  constexpr int32_t intColId = 1;
  constexpr int32_t bigintColId = 2;
  constexpr int32_t decimalColId = 3;
  constexpr int32_t varcharColId = 4;
  constexpr int32_t varbinaryColId = 5;

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(intColId), expectedIntNulls)
      << "Int column null count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(bigintColId), expectedBigintNulls)
      << "Bigint column null count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(decimalColId), expectedDecimalNulls)
      << "Decimal column null count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(varcharColId), expectedVarcharNulls)
      << "Varchar column null count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(varbinaryColId), expectedVarbinaryNulls)
      << "Varbinary column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  // Verify all columns have non-empty bounds.
  EXPECT_FALSE(stats->lowerBounds.at(intColId).empty())
      << "Int column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(intColId).empty())
      << "Int column should have non-empty upper bound";

  EXPECT_FALSE(stats->lowerBounds.at(bigintColId).empty())
      << "Bigint column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(bigintColId).empty())
      << "Bigint column should have non-empty upper bound";

  EXPECT_FALSE(stats->lowerBounds.at(decimalColId).empty())
      << "Decimal column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(decimalColId).empty())
      << "Decimal column should have non-empty upper bound";

  EXPECT_FALSE(stats->lowerBounds.at(varcharColId).empty())
      << "Varchar column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(varcharColId).empty())
      << "Varchar column should have non-empty upper bound";

  EXPECT_FALSE(stats->lowerBounds.at(varbinaryColId).empty())
      << "Varbinary column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(varbinaryColId).empty())
      << "Varbinary column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, dateStatsTest) {
  auto rowType = ROW({"date_col"}, {DATE()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 20;

  auto rowVector = makeRowVector({makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return 18262 + row; },
      nullEvery(5),
      DATE())});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t dateColId = 1;
  EXPECT_EQ(stats->valueCounts.at(dateColId), size)
      << "Date column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(dateColId), expectedNulls)
      << "Date column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";
  EXPECT_FALSE(stats->lowerBounds.at(dateColId).empty())
      << "Date column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(dateColId).empty())
      << "Date column should have non-empty upper bound";

  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(dateColId));
  auto lb = *reinterpret_cast<int32_t*>(lowerBounds.data());
  EXPECT_EQ(lb, 18263) << "Lower bound should be 2020-01-02";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(dateColId));
  auto ub = *reinterpret_cast<int32_t*>(upperBounds.data());
  EXPECT_EQ(ub, 18262 + 99) << "Upper bound should be 2020-04-09";
}

TEST_F(IcebergStatsTest, booleanStatsTest) {
  auto rowType = ROW({"boolean_col"}, {BOOLEAN()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 10;

  auto rowVector = makeRowVector({makeFlatVector<bool>(
      size,
      [](vector_size_t row) { return row % 2 == 1; },
      nullEvery(10),
      BOOLEAN())});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t boolColId = 1;
  EXPECT_EQ(stats->valueCounts.at(boolColId), size)
      << "Boolean column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(boolColId), expectedNulls)
      << "Boolean column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";
  EXPECT_FALSE(stats->lowerBounds.at(boolColId).empty())
      << "Boolean column should have non-empty lower bound";
  EXPECT_FALSE(stats->upperBounds.at(boolColId).empty())
      << "Boolean column should have non-empty upper bound";

  // For boolean, the lower bound should be false (0) and upper bound should be
  // true (1) if both values are present.
  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(boolColId));
  auto lb = *reinterpret_cast<bool*>(lowerBounds.data());
  EXPECT_FALSE(lb) << "Lower bound should be false";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(boolColId));
  auto ub = *reinterpret_cast<bool*>(upperBounds.data());
  EXPECT_TRUE(ub) << "Upper bound should be true";
}

TEST_F(IcebergStatsTest, emptyStatsTest) {
  auto rowType = ROW({"int_col", "varchar_col"}, {INTEGER(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  // Create an empty row vector (0 rows)
  constexpr vector_size_t size = 0;
  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(0), makeFlatVector<StringView>(0)});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, size) << "Record count should be 0";
  ASSERT_TRUE(stats->valueCounts.empty())
      << "Should no value counts for columns";
}

TEST_F(IcebergStatsTest, nullValuesTest) {
  auto rowType = ROW({"int_col", "varchar_col"}, {INTEGER(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  // Create an empty row vector (0 rows)
  constexpr vector_size_t size = 100;
  auto rowVector = makeRowVector(
      {makeNullConstant(TypeKind::INTEGER, size),
       makeNullConstant(TypeKind::VARCHAR, size)});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, size) << "Record count should be 0";
  ASSERT_EQ(stats->nullValueCounts.at(1), size) << "All values is NULL.";
  // Do not collect lower and upper bounds for NULLs.
  ASSERT_EQ(stats->lowerBounds.size(), 0);
  ASSERT_EQ(stats->upperBounds.size(), 0);
}

TEST_F(IcebergStatsTest, realStatsTest) {
  auto rowType = ROW({"real_col"}, {REAL()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 20;
  auto expectedNaNs = 0;

  auto rowVector = makeRowVector({makeFlatVector<float>(
      size,
      [&](vector_size_t row) {
        if (row % 6 == 0) {
          expectedNaNs++;
          return std::numeric_limits<float>::quiet_NaN();
        }
        return row * 1.5f;
      },
      nullEvery(5),
      REAL())});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t realColId = 1;
  EXPECT_EQ(stats->valueCounts.at(realColId), size)
      << "Real column value count incorrect";

  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(realColId), expectedNulls)
      << "Real column null count incorrect";
  EXPECT_EQ(stats->nanValueCounts.at(realColId), expectedNaNs)
      << "Real column null count incorrect";

  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(realColId));
  auto lb = *reinterpret_cast<float*>(lowerBounds.data());
  EXPECT_FLOAT_EQ(lb, 1.5f) << "Lower bound should be 1.5";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(realColId));
  auto ub = *reinterpret_cast<float*>(upperBounds.data());
  EXPECT_FLOAT_EQ(ub, 148.5f) << "Upper bound should be 148.5";
}

TEST_F(IcebergStatsTest, doubleStatsTest) {
  auto rowType = ROW({"double_col"}, {DOUBLE()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 15;
  auto expectedNaNs = 0;

  auto rowVector = makeRowVector({makeFlatVector<double>(
      size,
      [&](vector_size_t row) {
        if (row % 3 == 0) {
          expectedNaNs++;
          return std::numeric_limits<double>::quiet_NaN();
        }
        if (row % 4 == 0) {
          return std::numeric_limits<double>::infinity();
        }
        if (row % 5 == 0) {
          return -std::numeric_limits<double>::infinity();
        }
        return row * 2.5;
      },
      nullEvery(7),
      DOUBLE())});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t doubleColId = 1;
  EXPECT_EQ(stats->valueCounts.at(doubleColId), size)
      << "Double column value count incorrect";
  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(doubleColId), expectedNulls)
      << "Double column null count incorrect";
  EXPECT_EQ(stats->nanValueCounts.at(doubleColId), expectedNaNs)
      << "Double column null count incorrect";
  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  // Verify bounds are set correctly and NaN/infinity values don't affect
  // min/max incorrectly.
  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(doubleColId));
  auto lb = *reinterpret_cast<double*>(lowerBounds.data());
  EXPECT_DOUBLE_EQ(lb, -std::numeric_limits<double>::infinity())
      << "Lower bound should be -infinity";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(doubleColId));
  auto ub = *reinterpret_cast<double*>(upperBounds.data());
  EXPECT_DOUBLE_EQ(ub, std::numeric_limits<double>::infinity())
      << "Upper bound should be infinity";
}

TEST_F(IcebergStatsTest, MixedDoubleFloatStatsTest) {
  std::vector<std::string> names = {"id", "data1", "data2", "data3"};
  auto rowType = ROW(names, {INTEGER(), REAL(), DOUBLE(), DOUBLE()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt}};

  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 6;

  auto intVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return 1; });
  auto floatVector = makeFlatVector<float>(size, [](vector_size_t row) {
    return -std::numeric_limits<float>::infinity();
  });
  auto doubleVector1 = makeFlatVector<double>(size, [](vector_size_t row) {
    return std::numeric_limits<double>::infinity();
  });
  auto doubleVector2 = makeFlatVector<double>(size, [](vector_size_t row) {
    switch (row) {
      case 0:
        return 1.23;
      case 1:
        return -1.23;
      case 2:
        return std::numeric_limits<double>::infinity();
      case 3:
        return 2.23;
      case 4:
        return -std::numeric_limits<double>::infinity();
      default:
        return -2.23;
    }
  });

  ASSERT_TRUE(
      -std::numeric_limits<float>::infinity() <
          std::numeric_limits<float>::min() &&
      std::numeric_limits<float>::min() <
          std::numeric_limits<float>::infinity());
  auto rowVector = makeRowVector(
      names, {intVector, floatVector, doubleVector1, doubleVector2});
  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t doubleColId = 4;
  EXPECT_EQ(stats->valueCounts.at(doubleColId), size)
      << "Double column value count incorrect";
  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";

  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(doubleColId));
  auto lb = *reinterpret_cast<double*>(lowerBounds.data());
  EXPECT_DOUBLE_EQ(lb, -std::numeric_limits<double>::infinity())
      << "Lower bound should be -infinity";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(doubleColId));
  auto ub = *reinterpret_cast<double*>(upperBounds.data());
  EXPECT_DOUBLE_EQ(ub, std::numeric_limits<double>::infinity())
      << "Upper bound should be infinity";

  lowerBounds = encoding::Base64::decode(stats->lowerBounds.at(2));
  auto flb = *reinterpret_cast<float*>(lowerBounds.data());
  EXPECT_DOUBLE_EQ(flb, -std::numeric_limits<float>::infinity())
      << "Lower bound should be -infinity";

  upperBounds = encoding::Base64::decode(stats->upperBounds.at(2));
  auto fub = *reinterpret_cast<float*>(upperBounds.data());
  EXPECT_DOUBLE_EQ(fub, -std::numeric_limits<float>::infinity())
      << "Upper bound should be -infinity too";
}

TEST_F(IcebergStatsTest, NaNStatsTest) {
  auto rowType = ROW({"double_col"}, {DOUBLE()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 1'000;
  auto expectedNulls = 500;
  auto expectedNaNs = 0;

  auto rowVector = makeRowVector({makeFlatVector<double>(
      size,
      [&](vector_size_t row) {
        expectedNaNs++;
        return std::numeric_limits<double>::quiet_NaN();
      },
      nullEvery(2),
      DOUBLE())});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t doubleColId = 1;
  EXPECT_EQ(stats->valueCounts.at(doubleColId), size)
      << "Double column value count incorrect";
  ASSERT_FALSE(stats->nullValueCounts.empty())
      << "Should have null counts for columns";
  EXPECT_EQ(stats->nullValueCounts.at(doubleColId), expectedNulls)
      << "Double column null count incorrect";
  EXPECT_EQ(stats->nanValueCounts.at(doubleColId), expectedNaNs)
      << "Double column null count incorrect";

  // Do not collect bounds for NULLs and NaNs.
  ASSERT_TRUE(stats->lowerBounds.empty())
      << "Should not have lower bounds for columns";
  ASSERT_TRUE(stats->upperBounds.empty())
      << "Should not have upper bounds for columns";
}

TEST_F(IcebergStatsTest, partitionedTableStatsTest) {
  auto rowType = ROW(
      {"int_col", "date_col", "varchar_col"}, {INTEGER(), DATE(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kBucket, 4},
      {1, TransformType::kDay, std::nullopt},
      {2, TransformType::kTruncate, 2}};

  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;

  auto intVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });

  auto dateVector = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return 18262 + (row % 5); },
      nullptr,
      DATE());

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();

  for (auto i = 0; i < size; ++i) {
    std::string str = fmt::format("str{}", i % 10);
    flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
  }

  auto rowVector =
      makeRowVector(rowType->names(), {intVector, dateVector, varcharVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  // We should have multiple files due to partitioning.
  ASSERT_FALSE(fileStats.empty()) << "Should have statistics for files";
  EXPECT_GT(fileStats.size(), 1)
      << "Expected multiple files due to partitioning";

  for (const auto& stats : fileStats) {
    EXPECT_GE(stats->numRecords, 0)
        << "Each partition file should have records";
    ASSERT_FALSE(stats->valueCounts.empty())
        << "Should have value counts for columns";

    constexpr int32_t intColId = 1;
    constexpr int32_t dateColId = 2;
    constexpr int32_t varcharColId = 3;
    EXPECT_EQ(stats->valueCounts.at(intColId), stats->numRecords)
        << "Integer column value count should match record count";
    EXPECT_EQ(stats->valueCounts.at(dateColId), stats->numRecords)
        << "Date column value count should match record count";
    EXPECT_EQ(stats->valueCounts.at(varcharColId), stats->numRecords)
        << "Varchar column value count should match record count";

    ASSERT_FALSE(stats->lowerBounds.empty())
        << "Should have lower bounds for columns";
    ASSERT_FALSE(stats->upperBounds.empty())
        << "Should have upper bounds for columns";

    EXPECT_FALSE(stats->lowerBounds.at(intColId).empty())
        << "Int column should have non-empty lower bound";
    EXPECT_FALSE(stats->upperBounds.at(intColId).empty())
        << "Int column should have non-empty upper bound";

    EXPECT_FALSE(stats->lowerBounds.at(dateColId).empty())
        << "Date column should have non-empty lower bound";
    EXPECT_FALSE(stats->upperBounds.at(dateColId).empty())
        << "Date column should have non-empty upper bound";

    EXPECT_FALSE(stats->lowerBounds.at(varcharColId).empty())
        << "Varchar column should have non-empty lower bound";
    EXPECT_FALSE(stats->upperBounds.at(varcharColId).empty())
        << "Varchar column should have non-empty upper bound";
  }

  // Verify total record count across all partitions.
  auto totalRecords = 0;
  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
  }
  EXPECT_EQ(totalRecords, size)
      << "Total records across all partitions should match input size";
}

TEST_F(IcebergStatsTest, multiplePartitionTransformsStatsTest) {
  auto rowType =
      ROW({"int_col", "date_col", "varchar_col", "bigint_col"},
          {INTEGER(), DATE(), VARCHAR(), BIGINT()});
  auto outputDir = exec::test::TempDirectoryPath::create();

  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kBucket, 2},
      {1, TransformType::kYear, std::nullopt},
      {2, TransformType::kTruncate, 3},
      {3, TransformType::kIdentity, std::nullopt}};

  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;
  auto intVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row * 10; });
  auto flatIntVector = intVector->asFlatVector<int32_t>();

  auto dateVector = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return 18262 + (row * 100); },
      nullptr,
      DATE());

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (auto i = 0; i < size; ++i) {
    std::string str = fmt::format("prefix{}_value", i % 5);
    flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
  }
  auto bigintVector = makeFlatVector<int64_t>(
      size, [](vector_size_t row) { return (row % 3) * 1'000; });

  auto rowVector = makeRowVector(
      rowType->names(), {intVector, dateVector, varcharVector, bigintVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_FALSE(fileStats.empty()) << "Should have statistics for files";
  EXPECT_GT(fileStats.size(), 1)
      << "Expected multiple files due to partitioning";
  // Check each file's stats
  for (const auto& stats : fileStats) {
    EXPECT_GT(stats->numRecords, 0)
        << "Each partition file should have records";
    constexpr int32_t intColId = 1;
    constexpr int32_t dateColId = 2;
    constexpr int32_t bigintColId = 4;

    if (stats->lowerBounds.find(intColId) != stats->lowerBounds.end()) {
      std::string lowerBounds =
          encoding::Base64::decode(stats->lowerBounds.at(intColId));
      std::string upperBounds =
          encoding::Base64::decode(stats->upperBounds.at(intColId));

      auto lb = *reinterpret_cast<int32_t*>(lowerBounds.data());
      auto ub = *reinterpret_cast<int32_t*>(upperBounds.data());

      EXPECT_LE(lb, ub)
          << "Lower bound should be <= upper bound for int column";
    }

    if (stats->lowerBounds.find(dateColId) != stats->lowerBounds.end()) {
      std::string lowerBounds =
          encoding::Base64::decode(stats->lowerBounds.at(dateColId));
      std::string upperBounds =
          encoding::Base64::decode(stats->upperBounds.at(dateColId));

      auto lb = *reinterpret_cast<int32_t*>(lowerBounds.data());
      auto ub = *reinterpret_cast<int32_t*>(upperBounds.data());

      EXPECT_LE(lb, ub)
          << "Lower bound should be <= upper bound for date column";
    }

    if (stats->lowerBounds.find(bigintColId) != stats->lowerBounds.end()) {
      std::string lowerBounds =
          encoding::Base64::decode(stats->lowerBounds.at(bigintColId));
      std::string upperBounds =
          encoding::Base64::decode(stats->upperBounds.at(bigintColId));

      auto lb = *reinterpret_cast<int64_t*>(lowerBounds.data());
      auto ub = *reinterpret_cast<int64_t*>(upperBounds.data());

      EXPECT_LE(lb, ub)
          << "Lower bound should be <= upper bound for bigint column";
    }
  }
  auto totalRecords = 0;
  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
  }
  EXPECT_EQ(totalRecords, size)
      << "Total records across all partitions should match input size";
}

TEST_F(IcebergStatsTest, partitionedTableWithNullsStatsTest) {
  auto rowType = ROW(
      {"int_col", "date_col", "varchar_col"}, {INTEGER(), DATE(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt},
      {1, TransformType::kMonth, std::nullopt},
      {2, TransformType::kTruncate, 2}};
  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;
  auto expectedNulls = 20;
  auto dateNulls = 15;
  auto intVector = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return row % 10; },
      nullEvery(5),
      INTEGER());
  auto dateVector = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return 18262 + (row % 3) * 30; },
      nullEvery(7),
      DATE());
  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  auto varcharNulls = 0;
  for (auto i = 0; i < size; ++i) {
    if (i % 11 == 0) {
      flatVarcharVector->setNull(i, true);
      varcharNulls++;
    } else {
      std::string str = fmt::format("val{}", i % 5);
      flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
    }
  }

  auto rowVector =
      makeRowVector(rowType->names(), {intVector, dateVector, varcharVector});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_FALSE(fileStats.empty()) << "Should have statistics for files";
  auto totalIntNulls = 0;
  auto totalDateNulls = 0;
  auto totalVarcharNulls = 0;
  auto totalRecords = 0;

  constexpr int32_t intColId = 1;
  constexpr int32_t dateColId = 2;
  constexpr int32_t varcharColId = 3;

  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
    // Add null counts if present.
    if (stats->nullValueCounts.find(intColId) != stats->nullValueCounts.end()) {
      totalIntNulls += stats->nullValueCounts.at(intColId);
    }

    if (stats->nullValueCounts.find(dateColId) !=
        stats->nullValueCounts.end()) {
      totalDateNulls += stats->nullValueCounts.at(dateColId);
    }

    if (stats->nullValueCounts.find(varcharColId) !=
        stats->nullValueCounts.end()) {
      totalVarcharNulls += stats->nullValueCounts.at(varcharColId);
    }

    // Check that null count is less than or equal to value count for each
    // column.
    if (stats->nullValueCounts.find(intColId) != stats->nullValueCounts.end() &&
        stats->valueCounts.find(intColId) != stats->valueCounts.end()) {
      EXPECT_LE(
          stats->nullValueCounts.at(intColId), stats->valueCounts.at(intColId))
          << "Null count should be <= value count for int column";
    }

    if (stats->nullValueCounts.find(dateColId) !=
            stats->nullValueCounts.end() &&
        stats->valueCounts.find(dateColId) != stats->valueCounts.end()) {
      EXPECT_LE(
          stats->nullValueCounts.at(dateColId),
          stats->valueCounts.at(dateColId))
          << "Null count should be <= value count for date column";
    }

    if (stats->nullValueCounts.find(varcharColId) !=
            stats->nullValueCounts.end() &&
        stats->valueCounts.find(varcharColId) != stats->valueCounts.end()) {
      EXPECT_LE(
          stats->nullValueCounts.at(varcharColId),
          stats->valueCounts.at(varcharColId))
          << "Null count should be <= value count for varchar column";
    }
  }

  // Verify total counts match expected.
  EXPECT_EQ(totalRecords, size)
      << "Total records across all partitions should match input size";
  EXPECT_EQ(totalIntNulls, expectedNulls)
      << "Total int nulls should match expected";
  EXPECT_EQ(totalDateNulls, dateNulls)
      << "Total date nulls should match expected";
  EXPECT_EQ(totalVarcharNulls, varcharNulls)
      << "Total varchar nulls should match expected";
}

TEST_F(IcebergStatsTest, mapTypeTest) {
  auto rowType =
      ROW({"int_col", "map_col"}, {INTEGER(), MAP(INTEGER(), VARCHAR())});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  std::vector<
      std::optional<std::vector<std::pair<int32_t, std::optional<StringView>>>>>
      mapData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::pair<int32_t, std::optional<StringView>>> mapRow;
    for (auto j = 0; j < 5; ++j) {
      mapRow.emplace_back(j, StringView("test_value"));
    }
    mapData.push_back(mapRow);
  }

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [&](auto row) { return row * 10; }),
       makeNullableMapVector<int32_t, StringView>(mapData)});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  constexpr int32_t intColId = 1;
  constexpr int32_t mapColId = 3;

  EXPECT_EQ(stats->valueCounts.at(intColId), size)
      << "Int column value count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(intColId), expectedNulls)
      << "Int column null count incorrect";
  EXPECT_EQ(stats->valueCounts.at(mapColId), size * 5)
      << "Map column value count incorrect";
  EXPECT_TRUE(stats->lowerBounds.find(mapColId) == stats->lowerBounds.end())
      << "Map column should not have lower bounds";
  EXPECT_TRUE(stats->upperBounds.find(mapColId) == stats->upperBounds.end())
      << "Map column should not have upper bounds";
}

TEST_F(IcebergStatsTest, arrayTypeTest) {
  auto rowType = ROW({"int_col", "array_col"}, {INTEGER(), ARRAY(VARCHAR())});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  std::vector<std::vector<std::optional<StringView>>> arrayData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::optional<StringView>> arrayRow;
    for (auto j = 0; j < 3; ++j) {
      auto v = fmt::format("item_{}", i * 3 + j);
      arrayRow.emplace_back(StringView(v));
    }
    arrayData.push_back(arrayRow);
  }

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       makeNullableArrayVector<StringView>(arrayData)});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  constexpr int32_t intColId = 1;
  constexpr int32_t arrayColId = 3;

  EXPECT_EQ(stats->valueCounts.at(intColId), size)
      << "Int column value count incorrect";
  EXPECT_EQ(stats->nullValueCounts.at(intColId), expectedNulls)
      << "Int column null count incorrect";
  EXPECT_EQ(stats->valueCounts.at(arrayColId), size * 3)
      << "Array column value count incorrect";
  EXPECT_TRUE(stats->lowerBounds.find(arrayColId) == stats->lowerBounds.end())
      << "Array column should not have lower bounds";
  EXPECT_TRUE(stats->upperBounds.find(arrayColId) == stats->upperBounds.end())
      << "Array column should not have upper bounds";
}

// Test statistics collection for nested struct fields.
// Assume int_col's ID start with 1.
// Struct definition with field IDs:
// struct {
//   int_col: INTEGER (id: 1)
//   struct_col (id: 2) {
//     first_level_id: INTEGER (id: 3)
//     first_level_name: VARCHAR (id: 4)
//     nested_struct (id: 5) {
//       second_level_id: INTEGER (id: 6)
//       second_level_name: VARCHAR (id: 7)
//     }
//   }
// }
// Need to collect statistics for field IDs [1, 3, 4, 6, 7]
TEST_F(IcebergStatsTest, structTypeTest) {
  auto rowType =
      ROW({"int_col", "struct_col"},
          {INTEGER(),
           ROW({"first_level_id", "first_level_name", "nested_struct"},
               {INTEGER(),
                VARCHAR(),
                ROW({"second_level_id", "second_level_name"},
                    {INTEGER(), VARCHAR()})})});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  auto intVector =
      makeFlatVector<int32_t>(size, [](auto row) { return row * 10; });
  auto firstLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row % size; }, nullEvery(5));
  auto firstLevelName = makeFlatVector<StringView>(
      size,
      [](vector_size_t row) {
        auto v = fmt::format("name_{}", row * 10);
        return StringView(v);
      },
      nullEvery(7));

  auto secondLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * size; }, nullEvery(6));
  auto secondLevelName = makeFlatVector<StringView>(
      size,
      [](vector_size_t row) {
        auto v = fmt::format("nested_{}", row * 100);
        return StringView(v);
      },
      nullEvery(8));

  auto nestedStruct = makeRowVector({secondLevelId, secondLevelName});
  auto structVector =
      makeRowVector({firstLevelId, firstLevelName, nestedStruct});

  auto rowVector = makeRowVector({intVector, structVector});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->dataFileStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";

  constexpr int32_t intColId = 1;
  constexpr int32_t tier1ColId = 3;
  constexpr int32_t tier2ColId = 6;
  constexpr int32_t tier2ColId2 = 7;

  EXPECT_EQ(stats->valueCounts.size(), 5);
  EXPECT_EQ(stats->lowerBounds.size(), 5);
  EXPECT_EQ(stats->valueCounts.at(intColId), size);
  EXPECT_EQ(stats->nullValueCounts.at(intColId), expectedNulls);
  EXPECT_EQ(stats->valueCounts.at(tier1ColId), size);
  EXPECT_EQ(stats->valueCounts.at(tier2ColId), size);
  EXPECT_EQ(stats->nullValueCounts.at(tier1ColId), 20);
  EXPECT_EQ(
      encoding::Base64::decode(stats->lowerBounds.at(tier2ColId2)),
      "nested_100");
  EXPECT_EQ(
      encoding::Base64::decode(stats->upperBounds.at(tier2ColId2)),
      "nested_9900");
}

} // namespace facebook::velox::connector::hive::iceberg::test
