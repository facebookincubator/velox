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
  auto expectedIntNulls = 0;

  // Create int column with nulls every 3rd position
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 3 == 0) {
      flatIntVector->setNull(i, true);
      expectedIntNulls++;
    } else {
      flatIntVector->set(i, i * 10);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(), rowType, nullptr, size, std::vector<VectorPtr>{intVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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
  auto lb = *reinterpret_cast<int64_t*>(&lowerBounds);
  EXPECT_EQ(lb, 10);
  EXPECT_FALSE(stats->lowerBounds.at(intColId).empty())
      << "Int column should have non-empty lower bound";
  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(intColId));
  auto ub = *reinterpret_cast<int64_t*>(&upperBounds);
  EXPECT_EQ(ub, 980);
  EXPECT_FALSE(stats->upperBounds.at(intColId).empty())
      << "Int column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, DISABLED_mixedNullTestDwrf) {
  auto rowType = ROW({"int_col"}, {INTEGER()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedIntNulls = 0;

  // Create int column with nulls every 3rd position
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 3 == 0) {
      flatIntVector->setNull(i, true);
      expectedIntNulls++;
    } else {
      flatIntVector->set(i, i * 10);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(), rowType, nullptr, size, std::vector<VectorPtr>{intVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  // Get file statistics
  const auto& fileStats = dataSink->icebergStats();
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
  auto lb = *reinterpret_cast<int64_t*>(&lowerBounds);
  EXPECT_EQ(lb, 10);
  EXPECT_FALSE(stats->lowerBounds.at(intColId).empty())
      << "Int column should have non-empty lower bound";
  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(intColId));
  auto ub = *reinterpret_cast<int64_t*>(&upperBounds);
  EXPECT_EQ(ub, 980);
  EXPECT_FALSE(stats->upperBounds.at(intColId).empty())
      << "Int column should have non-empty upper bound";
}

TEST_F(IcebergStatsTest, bigintStatsTest) {
  auto rowType = ROW({"bigint_col"}, {BIGINT()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  // Create bigint column with nulls every 4th position
  auto bigintVector = BaseVector::create(BIGINT(), size, opPool_.get());
  auto flatBigintVector = bigintVector->asFlatVector<int64_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 4 == 0) {
      flatBigintVector->setNull(i, true);
      expectedNulls++;
    } else {
      flatBigintVector->set(i, i * 1'000'000'000LL);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{bigintVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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
  auto lb = *reinterpret_cast<int64_t*>(&lowerBounds[0]);
  EXPECT_EQ(lb, 1'000'000'000LL);

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(bigintColId));
  auto ub = *reinterpret_cast<int64_t*>(&upperBounds[0]);
  EXPECT_EQ(ub, 99'000'000'000LL);
}

TEST_F(IcebergStatsTest, decimalStatsTest) {
  auto rowType = ROW({"decimal_col"}, {DECIMAL(38, 3)});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;

  // Create decimal column with nulls every 5th position
  auto decimalVector = BaseVector::create(DECIMAL(38, 3), size, opPool_.get());
  auto flatDecimalVector = decimalVector->asFlatVector<int128_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatDecimalVector->setNull(i, true);
      expectedNulls++;
    } else {
      int128_t value = HugeInt::build(i, i * 123);
      flatDecimalVector->set(i, value);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{decimalVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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

  // Create varchar column with nulls every 6th position
  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 6 == 0) {
      flatVarcharVector->setNull(i, true);
      expectedNulls++;
    } else {
      // Create string values with varying lengths
      std::string value =
          "Customer#00000" + std::to_string(i) + "_" + std::string(i % 10, 'a');
      flatVarcharVector->set(i, StringView(value));
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{varcharVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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

  // Decode and verify string bounds
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
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatVarbinaryVector->setNull(i, true);
      expectedNulls++;
    } else {
      // Create binary values with varying content
      std::string value(17, 11);
      value[0] = static_cast<char>(i % 256);
      value[1] = static_cast<char>((i * 3) % 256);
      value[2] = static_cast<char>((i * 7) % 256);
      value[3] = static_cast<char>((i * 11) % 256);
      flatVarbinaryVector->set(i, StringView(value));
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{varbinaryVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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

TEST_F(IcebergStatsTest, multipleDataTypesTest) {
  auto rowType = ROW(
      {"int_col", "bigint_col", "decimal_col", "varchar_col", "varbinary_col"},
      {INTEGER(), BIGINT(), DECIMAL(38, 3), VARCHAR(), VARBINARY()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  // Calculate expected null counts
  int32_t expectedIntNulls = 0;
  int32_t expectedBigintNulls = 0;
  int32_t expectedDecimalNulls = 0;
  int32_t expectedVarcharNulls = 0;
  int32_t expectedVarbinaryNulls = 0;
  // Create columns with different null patterns
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 3 == 0) {
      flatIntVector->setNull(i, true);
      expectedIntNulls++;
    } else {
      flatIntVector->set(i, i * 10);
    }
  }

  auto bigintVector = BaseVector::create(BIGINT(), size, opPool_.get());
  auto flatBigintVector = bigintVector->asFlatVector<int64_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 4 == 0) {
      flatBigintVector->setNull(i, true);
      expectedBigintNulls++;
    } else {
      flatBigintVector->set(i, i * 1000000000LL);
    }
  }

  auto decimalVector = BaseVector::create(DECIMAL(38, 3), size, opPool_.get());
  auto flatDecimalVector = decimalVector->asFlatVector<int128_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatDecimalVector->setNull(i, true);
      expectedDecimalNulls++;
    } else {
      int128_t value = HugeInt::build(i, i * 12345);
      flatDecimalVector->set(i, value);
    }
  }

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < size; ++i) {
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
  for (vector_size_t i = 0; i < size; ++i) {
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

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{
          intVector,
          bigintVector,
          decimalVector,
          varcharVector,
          varbinaryVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";

  // Column IDs are 1-based and match the order in the schema
  constexpr int32_t intColId = 1;
  constexpr int32_t bigintColId = 2;
  constexpr int32_t decimalColId = 3;
  constexpr int32_t varcharColId = 4;
  constexpr int32_t varbinaryColId = 5;

  // Test null value counts
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

  // Test bounds
  ASSERT_FALSE(stats->lowerBounds.empty())
      << "Should have lower bounds for columns";
  ASSERT_FALSE(stats->upperBounds.empty())
      << "Should have upper bounds for columns";

  // Verify all columns have non-empty bounds
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
  auto expectedNulls = 0;

  auto dateVector = BaseVector::create(DATE(), size, opPool_.get());
  auto flatDateVector = dateVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatDateVector->setNull(i, true);
      expectedNulls++;
    } else {
      // Create date values spanning a range
      // Base date: 2020-01-01 (18262) + i days
      flatDateVector->set(i, 18262 + i);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{dateVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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
  auto lb = *reinterpret_cast<int32_t*>(&lowerBounds[0]);
  EXPECT_EQ(lb, 18263) << "Lower bound should be 2020-01-02";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(dateColId));
  auto ub = *reinterpret_cast<int32_t*>(&upperBounds[0]);
  EXPECT_EQ(ub, 18262 + 99) << "Upper bound should be 2020-04-09";
}

TEST_F(IcebergStatsTest, booleanStatsTest) {
  auto rowType = ROW({"boolean_col"}, {BOOLEAN()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;
  auto trueCount = 0;
  auto falseCount = 0;

  // Create boolean column with nulls every 10th position
  auto boolVector = BaseVector::create(BOOLEAN(), size, opPool_.get());
  auto flatBoolVector = boolVector->asFlatVector<bool>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 10 == 0) {
      flatBoolVector->setNull(i, true);
      expectedNulls++;
    } else {
      // Alternate between true and false
      auto value = (i % 2 == 1);
      flatBoolVector->set(i, value);
      if (value) {
        trueCount++;
      } else {
        falseCount++;
      }
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{boolVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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
  // true (1) if both values are present
  if (trueCount > 0 && falseCount > 0) {
    std::string lowerBounds =
        encoding::Base64::decode(stats->lowerBounds.at(boolColId));
    auto lb = *reinterpret_cast<bool*>(&lowerBounds[0]);
    EXPECT_FALSE(lb) << "Lower bound should be false";

    std::string upperBounds =
        encoding::Base64::decode(stats->upperBounds.at(boolColId));
    auto ub = *reinterpret_cast<bool*>(&upperBounds[0]);
    EXPECT_TRUE(ub) << "Upper bound should be true";
  }
}

TEST_F(IcebergStatsTest, emptyStatsTest) {
  auto rowType = ROW({"int_col", "varchar_col"}, {INTEGER(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  // Create an empty row vector (0 rows)
  constexpr vector_size_t size = 0;
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{intVector, varcharVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->icebergStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, size) << "Record count should be 0";
  ASSERT_TRUE(stats->valueCounts.empty())
      << "Should no value counts for columns";
}

TEST_F(IcebergStatsTest, realStatsTest) {
  auto rowType = ROW({"real_col"}, {REAL()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;
  auto expectedNaNs = 0;
  // Create real column with nulls every 5th position and NaN every 10th
  // position
  auto realVector = BaseVector::create(REAL(), size, opPool_.get());
  auto flatRealVector = realVector->asFlatVector<float>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatRealVector->setNull(i, true);
      expectedNulls++;
    } else if (i % 6 == 0) {
      flatRealVector->set(i, std::numeric_limits<float>::quiet_NaN());
      expectedNaNs++;
    } else {
      flatRealVector->set(i, i * 1.5f);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{realVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
  ASSERT_EQ(fileStats.size(), 1) << "Expected exactly one file with stats";
  const auto& stats = fileStats[0];

  EXPECT_EQ(stats->numRecords, size) << "Record count should match input size";
  ASSERT_FALSE(stats->valueCounts.empty())
      << "Should have value counts for columns";
  constexpr int32_t realColId = 1;
  EXPECT_EQ(stats->valueCounts.at(realColId), size)
      << "Real column value count incorrect";

  // Test null value counts
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
  auto lb = *reinterpret_cast<float*>(&lowerBounds[0]);
  EXPECT_FLOAT_EQ(lb, 1.5f) << "Lower bound should be 1.5";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(realColId));
  auto ub = *reinterpret_cast<float*>(&upperBounds[0]);
  EXPECT_FLOAT_EQ(ub, 148.5f) << "Upper bound should be 148.5";
}

TEST_F(IcebergStatsTest, doubleStatsTest) {
  auto rowType = ROW({"double_col"}, {DOUBLE()});

  auto outputDir = exec::test::TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(rowType, outputDir->getPath());
  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;
  auto expectedNaNs = 0;
  auto doubleVector = BaseVector::create(DOUBLE(), size, opPool_.get());
  auto flatDoubleVector = doubleVector->asFlatVector<double>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 6 == 0) {
      flatDoubleVector->setNull(i, true);
      expectedNulls++;
    } else if (i % 11 == 0) {
      flatDoubleVector->set(i, std::numeric_limits<double>::quiet_NaN());
      expectedNaNs++;
    } else if (i % 20 == 0) {
      flatDoubleVector->set(i, std::numeric_limits<double>::infinity());
    } else if (i % 25 == 0) {
      flatDoubleVector->set(i, -std::numeric_limits<double>::infinity());
    } else {
      flatDoubleVector->set(i, i * 2.5);
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{doubleVector});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
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
  // min/max incorrectly
  std::string lowerBounds =
      encoding::Base64::decode(stats->lowerBounds.at(doubleColId));
  auto lb = *reinterpret_cast<double*>(&lowerBounds[0]);
  EXPECT_DOUBLE_EQ(lb, -std::numeric_limits<double>::infinity())
      << "Lower bound should be -infinity";

  std::string upperBounds =
      encoding::Base64::decode(stats->upperBounds.at(doubleColId));
  auto ub = *reinterpret_cast<double*>(&upperBounds[0]);
  EXPECT_DOUBLE_EQ(ub, std::numeric_limits<double>::infinity())
      << "Upper bound should be infinity";
}

TEST_F(IcebergStatsTest, partitionedTableStatsTest) {
  // Create a schema with partition columns
  auto rowType = ROW(
      {"int_col", "date_col", "varchar_col"}, {INTEGER(), DATE(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<std::string> partitionTransforms = {
      "bucket(int_col, 4)", "day(date_col)", "truncate(varchar_col, 2)"};

  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;

  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    flatIntVector->set(i, i);
  }

  auto dateVector = BaseVector::create(DATE(), size, opPool_.get());
  auto flatDateVector = dateVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    flatDateVector->set(i, 18262 + (i % 5));
  }

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();

  for (vector_size_t i = 0; i < size; ++i) {
    std::string str = fmt::format("str{}", i % 10);
    flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{intVector, dateVector, varcharVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());

  const auto& fileStats = dataSink->icebergStats();
  // We should have multiple files due to partitioning
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

  // Verify total record count across all partitions
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

  // Use different partition transforms
  std::vector<std::string> partitionTransforms = {
      "bucket(int_col, 2)",
      "year(date_col)",
      "truncate(varchar_col, 3)",
      "bigint_col"};

  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    flatIntVector->set(i, i * 10);
  }

  auto dateVector = BaseVector::create(DATE(), size, opPool_.get());
  auto flatDateVector = dateVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    flatDateVector->set(i, 18262 + (i * 100));
  }

  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < size; ++i) {
    std::string str = fmt::format("prefix{}_value", i % 5);
    flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
  }
  auto bigintVector = BaseVector::create(BIGINT(), size, opPool_.get());
  auto flatBigintVector = bigintVector->asFlatVector<int64_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    flatBigintVector->set(i, (i % 3) * 1000);
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{
          intVector, dateVector, varcharVector, bigintVector});

  dataSink->appendData(rowVector);

  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->icebergStats();
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

      auto lb = *reinterpret_cast<int32_t*>(&lowerBounds[0]);
      auto ub = *reinterpret_cast<int32_t*>(&upperBounds[0]);

      EXPECT_LE(lb, ub)
          << "Lower bound should be <= upper bound for int column";
    }

    if (stats->lowerBounds.find(dateColId) != stats->lowerBounds.end()) {
      std::string lowerBounds =
          encoding::Base64::decode(stats->lowerBounds.at(dateColId));
      std::string upperBounds =
          encoding::Base64::decode(stats->upperBounds.at(dateColId));

      auto lb = *reinterpret_cast<int32_t*>(&lowerBounds[0]);
      auto ub = *reinterpret_cast<int32_t*>(&upperBounds[0]);

      EXPECT_LE(lb, ub)
          << "Lower bound should be <= upper bound for date column";
    }

    if (stats->lowerBounds.find(bigintColId) != stats->lowerBounds.end()) {
      std::string lowerBounds =
          encoding::Base64::decode(stats->lowerBounds.at(bigintColId));
      std::string upperBounds =
          encoding::Base64::decode(stats->upperBounds.at(bigintColId));

      auto lb = *reinterpret_cast<int64_t*>(&lowerBounds[0]);
      auto ub = *reinterpret_cast<int64_t*>(&upperBounds[0]);

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
  // Create a schema with partition columns
  auto rowType = ROW(
      {"int_col", "date_col", "varchar_col"}, {INTEGER(), DATE(), VARCHAR()});
  auto outputDir = exec::test::TempDirectoryPath::create();
  std::vector<std::string> partitionTransforms = {
      "int_col", "month(date_col)", "truncate(varchar_col, 2)"};
  auto dataSink =
      createIcebergDataSink(rowType, outputDir->getPath(), partitionTransforms);

  constexpr vector_size_t size = 100;
  auto expectedNulls = 0;
  auto intVector = BaseVector::create(INTEGER(), size, opPool_.get());
  auto flatIntVector = intVector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 5 == 0) {
      flatIntVector->setNull(i, true);
      expectedNulls++;
    } else {
      flatIntVector->set(i, i % 10);
    }
  }

  // Create date column with nulls every 7th position
  auto dateVector = BaseVector::create(DATE(), size, opPool_.get());
  auto flatDateVector = dateVector->asFlatVector<int32_t>();
  auto dateNulls = 0;
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 7 == 0) {
      flatDateVector->setNull(i, true);
      dateNulls++;
    } else {
      flatDateVector->set(i, 18262 + (i % 3) * 30);
    }
  }

  // Create varchar column with nulls every 11th position
  auto varcharVector = BaseVector::create(VARCHAR(), size, opPool_.get());
  auto flatVarcharVector = varcharVector->asFlatVector<StringView>();
  auto varcharNulls = 0;
  for (vector_size_t i = 0; i < size; ++i) {
    if (i % 11 == 0) {
      flatVarcharVector->setNull(i, true);
      varcharNulls++;
    } else {
      std::string str = fmt::format("val{}", i % 5);
      flatVarcharVector->set(i, StringView(str.c_str(), str.size()));
    }
  }

  auto rowVector = std::make_shared<RowVector>(
      opPool_.get(),
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{intVector, dateVector, varcharVector});

  dataSink->appendData(rowVector);
  ASSERT_TRUE(dataSink->finish());
  auto commitTasks = dataSink->close();
  ASSERT_FALSE(commitTasks.empty());
  const auto& fileStats = dataSink->icebergStats();
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
    // Add null counts if present
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
    // column
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

  // Verify total counts match expected
  EXPECT_EQ(totalRecords, size)
      << "Total records across all partitions should match input size";
  EXPECT_EQ(totalIntNulls, expectedNulls)
      << "Total int nulls should match expected";
  EXPECT_EQ(totalDateNulls, dateNulls)
      << "Total date nulls should match expected";
  EXPECT_EQ(totalVarcharNulls, varcharNulls)
      << "Total varchar nulls should match expected";
}

} // namespace facebook::velox::connector::hive::iceberg::test
