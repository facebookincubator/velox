/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <string>
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/stats/VectorizedStatistics.h"
#include "velox/type/Type.h"

using namespace facebook;
using namespace facebook::nimble;

class VectorizedFileStatsTests : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::initializeMemoryManager(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

// Test 1: Constructor tests - basic construction with various column stat types

TEST_F(VectorizedFileStatsTests, constructorWithDefaultStats) {
  // Create basic column statistics
  ColumnStatistics stat1(100, 10, 1000, 500);
  ColumnStatistics stat2(200, 20, 2000, 1000);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  // Serialize to verify it was constructed correctly
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());
}

TEST_F(VectorizedFileStatsTests, constructorWithIntegralStats) {
  IntegralStatistics stat1(100, 10, 1000, 500, -50, 150);
  IntegralStatistics stat2(200, 20, 2000, 1000, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());
}

TEST_F(VectorizedFileStatsTests, constructorWithFloatingPointStats) {
  FloatingPointStatistics stat1(100, 10, 1000, 500, -1.5, 2.5);
  FloatingPointStatistics stat2(200, 20, 2000, 1000, 0.0, 100.0);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());
}

TEST_F(VectorizedFileStatsTests, constructorWithStringStats) {
  StringStatistics stat1(100, 10, 1000, 500, "aaa", "zzz");
  StringStatistics stat2(200, 20, 2000, 1000, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());
}

TEST_F(VectorizedFileStatsTests, constructorWithMixedStats) {
  ColumnStatistics defaultStat(100, 10, 1000, 500);
  IntegralStatistics intStat(200, 20, 2000, 1000, -10, 20);
  FloatingPointStatistics floatStat(300, 30, 3000, 1500, 0.5, 99.5);
  StringStatistics stringStat(400, 40, 4000, 2000, "abc", "xyz");

  std::vector<ColumnStatistics*> columnStats{
      &defaultStat, &intStat, &floatStat, &stringStat};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());
}

// Test 2: Round-trip serialize/deserialize stability tests

TEST_F(VectorizedFileStatsTests, roundTripDefaultStats) {
  ColumnStatistics stat1(100, 10, 1000, 500);
  ColumnStatistics stat2(200, 20, 2000, 1000);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  // Serialize again and compare
  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

TEST_F(VectorizedFileStatsTests, roundTripIntegralStats) {
  IntegralStatistics stat1(100, 10, 1000, 500, -50, 150);
  IntegralStatistics stat2(200, 20, 2000, 1000, 0, 1000);
  IntegralStatistics stat3(300, 30, 3000, 1500, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2, &stat3};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

TEST_F(VectorizedFileStatsTests, roundTripFloatingPointStats) {
  FloatingPointStatistics stat1(100, 10, 1000, 500, -1.5, 2.5);
  FloatingPointStatistics stat2(200, 20, 2000, 1000, 0.0, 100.0);
  FloatingPointStatistics stat3(
      300, 30, 3000, 1500, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2, &stat3};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

TEST_F(VectorizedFileStatsTests, roundTripStringStats) {
  StringStatistics stat1(100, 10, 1000, 500, "aaa", "zzz");
  StringStatistics stat2(200, 20, 2000, 1000, "hello", "world");
  StringStatistics stat3(300, 30, 3000, 1500, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> columnStats{&stat1, &stat2, &stat3};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

TEST_F(VectorizedFileStatsTests, roundTripMixedStats) {
  ColumnStatistics defaultStat(100, 10, 1000, 500);
  IntegralStatistics intStat(200, 20, 2000, 1000, -10, 20);
  FloatingPointStatistics floatStat(300, 30, 3000, 1500, 0.5, 99.5);
  StringStatistics stringStat(400, 40, 4000, 2000, "abc", "xyz");

  std::vector<ColumnStatistics*> columnStats{
      &defaultStat, &intStat, &floatStat, &stringStat};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

TEST_F(VectorizedFileStatsTests, roundTripEmptyStats) {
  std::vector<ColumnStatistics*> columnStats{};

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);
}

TEST_F(VectorizedFileStatsTests, roundTripLargeNumberOfStats) {
  constexpr int kNumStats = 100;
  std::vector<std::unique_ptr<ColumnStatistics>> ownedStats;
  std::vector<ColumnStatistics*> columnStats;

  for (int i = 0; i < kNumStats; ++i) {
    auto stat = std::make_unique<IntegralStatistics>(
        i * 10, i, i * 100, i * 50, -i, i * 2);
    columnStats.push_back(stat.get());
    ownedStats.push_back(std::move(stat));
  }

  VectorizedFileStats fileStats(columnStats, leafPool_.get());

  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  nimble::Buffer buffer2(*leafPool_);
  auto serialized2 = deserialized->serialize(buffer2);
  EXPECT_EQ(serialized.size(), serialized2.size());
}

// Test 3: Forward compatibility tests - tests that deserialize can handle
// unrecognized stat stream types gracefully

// Test that VectorizedStatistic::create returns nullptr for unknown types
// This is the key mechanism that enables forward compatibility - when a new
// stat stream type is added, older readers will get nullptr and can skip it

// Test that VectorizedStatistic::create returns nullptr for unknown types
TEST_F(
    VectorizedFileStatsTests,
    vectorizedStatisticCreateReturnsNullForUnknown) {
  // Cast an invalid enum value to StatStreamType
  auto unknownType = static_cast<StatStreamType>(255);
  auto stat = VectorizedStatistic::create(unknownType, leafPool_.get());
  EXPECT_EQ(stat, nullptr);
}

// Test TypedVectorizedStatistic append and valueAt

TEST_F(VectorizedFileStatsTests, typedVectorizedStatisticAppendAndValueAt) {
  auto stat = VectorizedStatistic::create(
      StatStreamType::INTEGRAL_MIN, leafPool_.get());
  ASSERT_NE(stat, nullptr);

  auto* integralStat = stat->as<IntegralVectorizedStatistics>();
  ASSERT_NE(integralStat, nullptr);

  // Append some values
  integralStat->append(10);
  integralStat->append(std::nullopt);
  integralStat->append(-5);
  integralStat->append(100);

  // Verify values
  EXPECT_EQ(integralStat->valueAt(0), 10);
  EXPECT_EQ(integralStat->valueAt(1), std::nullopt);
  EXPECT_EQ(integralStat->valueAt(2), -5);
  EXPECT_EQ(integralStat->valueAt(3), 100);
}

TEST_F(VectorizedFileStatsTests, defaultVectorizedStatisticAppendAndValueAt) {
  auto stat =
      VectorizedStatistic::create(StatStreamType::VALUE_COUNT, leafPool_.get());
  ASSERT_NE(stat, nullptr);

  auto* defaultStat = stat->as<DefaultVectorizedStatistics>();
  ASSERT_NE(defaultStat, nullptr);

  defaultStat->append(100);
  defaultStat->append(200);
  defaultStat->append(std::nullopt);

  EXPECT_EQ(defaultStat->valueAt(0), 100);
  EXPECT_EQ(defaultStat->valueAt(1), 200);
  EXPECT_EQ(defaultStat->valueAt(2), std::nullopt);
}

TEST_F(
    VectorizedFileStatsTests,
    floatingPointVectorizedStatisticAppendAndValueAt) {
  auto stat = VectorizedStatistic::create(
      StatStreamType::FLOATING_POINT_MIN, leafPool_.get());
  ASSERT_NE(stat, nullptr);

  auto* floatStat = stat->as<FloatingPointVectorizedStatistics>();
  ASSERT_NE(floatStat, nullptr);

  floatStat->append(1.5);
  floatStat->append(-2.5);
  floatStat->append(std::nullopt);

  EXPECT_EQ(floatStat->valueAt(0), 1.5);
  EXPECT_EQ(floatStat->valueAt(1), -2.5);
  EXPECT_EQ(floatStat->valueAt(2), std::nullopt);
}

// Test VectorizedStatistic::create for all known types

TEST_F(VectorizedFileStatsTests, vectorizedStatisticCreateForAllKnownTypes) {
  // Default types (uint64_t)
  EXPECT_NE(
      VectorizedStatistic::create(StatStreamType::VALUE_COUNT, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(StatStreamType::NULL_COUNT, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::LOGICAL_SIZE, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::PHYSICAL_SIZE, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::DEDUPLICATED_VALUE_COUNT, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::DEDUPLICATED_LOGICAL_SIZE, leafPool_.get()),
      nullptr);

  // Integral types (int64_t)
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::INTEGRAL_MIN, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::INTEGRAL_MAX, leafPool_.get()),
      nullptr);

  // Floating point types (double)
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::FLOATING_POINT_MIN, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(
          StatStreamType::FLOATING_POINT_MAX, leafPool_.get()),
      nullptr);

  // String types
  EXPECT_NE(
      VectorizedStatistic::create(StatStreamType::STRING_MIN, leafPool_.get()),
      nullptr);
  EXPECT_NE(
      VectorizedStatistic::create(StatStreamType::STRING_MAX, leafPool_.get()),
      nullptr);
}

// Test StatStreamType enum values returned correctly

TEST_F(VectorizedFileStatsTests, vectorizedStatisticTypeReturnsCorrectType) {
  auto valueStat =
      VectorizedStatistic::create(StatStreamType::VALUE_COUNT, leafPool_.get());
  EXPECT_EQ(valueStat->type(), StatStreamType::VALUE_COUNT);

  auto integralMinStat = VectorizedStatistic::create(
      StatStreamType::INTEGRAL_MIN, leafPool_.get());
  EXPECT_EQ(integralMinStat->type(), StatStreamType::INTEGRAL_MIN);

  auto floatMaxStat = VectorizedStatistic::create(
      StatStreamType::FLOATING_POINT_MAX, leafPool_.get());
  EXPECT_EQ(floatMaxStat->type(), StatStreamType::FLOATING_POINT_MAX);

  auto stringMinStat =
      VectorizedStatistic::create(StatStreamType::STRING_MIN, leafPool_.get());
  EXPECT_EQ(stringMinStat->type(), StatStreamType::STRING_MIN);
}

// Helper function to compare column statistics
namespace {

void expectColumnStatisticsEqual(
    const ColumnStatistics& expected,
    const ColumnStatistics& actual) {
  EXPECT_EQ(expected.getValueCount(), actual.getValueCount());
  EXPECT_EQ(expected.getNullCount(), actual.getNullCount());
  EXPECT_EQ(expected.getLogicalSize(), actual.getLogicalSize());
  EXPECT_EQ(expected.getPhysicalSize(), actual.getPhysicalSize());
}

void expectIntegralStatisticsEqual(
    const IntegralStatistics& expected,
    const IntegralStatistics& actual) {
  expectColumnStatisticsEqual(expected, actual);
  EXPECT_EQ(expected.getMin(), actual.getMin());
  EXPECT_EQ(expected.getMax(), actual.getMax());
}

void expectFloatingPointStatisticsEqual(
    const FloatingPointStatistics& expected,
    const FloatingPointStatistics& actual) {
  expectColumnStatisticsEqual(expected, actual);
  EXPECT_EQ(expected.getMin(), actual.getMin());
  EXPECT_EQ(expected.getMax(), actual.getMax());
}

void expectStringStatisticsEqual(
    const StringStatistics& expected,
    const StringStatistics& actual) {
  expectColumnStatisticsEqual(expected, actual);
  EXPECT_EQ(expected.getMin(), actual.getMin());
  EXPECT_EQ(expected.getMax(), actual.getMax());
}

} // namespace

// Test 4: Round-trip tests with schema-based conversion that check statistic
// state equivalence. Workflow: Generate vector<ColumnStatistics*> from schema
// -> Convert to VectorizedFileStats -> Serialize -> Deserialize ->
// toColumnStatistics -> Check equivalence

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripFlatRow) {
  // Create a flat row schema with various primitive types
  auto schema = velox::ROW(
      {{"int_col", velox::BIGINT()},
       {"double_col", velox::DOUBLE()},
       {"string_col", velox::VARCHAR()}});
  auto nimbleType = convertToNimbleType(*schema);

  // Create column statistics matching the schema
  // Schema order: row (default), int (integral), double (float), string
  // (string)
  ColumnStatistics rowStat(1000, 0, 10000, 5000);
  IntegralStatistics intStat(800, 50, 6400, 3200, -100, 200);
  FloatingPointStatistics doubleStat(750, 100, 6000, 3000, -1.5, 99.5);
  StringStatistics stringStat(600, 200, 12000, 6000, "aaa", "zzz");

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &intStat, &doubleStat, &stringStat};

  // Convert to VectorizedFileStats and serialize
  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);
  EXPECT_FALSE(serialized.empty());

  // Deserialize and convert back to column statistics
  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  EXPECT_NE(deserialized, nullptr);

  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 4);

  // Check row stat (index 0)
  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);

  // Check integral stat (index 1)
  auto* actualIntStat = roundTrippedStats[1]->as<IntegralStatistics>();
  ASSERT_NE(actualIntStat, nullptr);
  expectIntegralStatisticsEqual(intStat, *actualIntStat);

  // Check floating point stat (index 2)
  auto* actualDoubleStat = roundTrippedStats[2]->as<FloatingPointStatistics>();
  ASSERT_NE(actualDoubleStat, nullptr);
  expectFloatingPointStatisticsEqual(doubleStat, *actualDoubleStat);

  // Check string stat (index 3)
  auto* actualStringStat = roundTrippedStats[3]->as<StringStatistics>();
  ASSERT_NE(actualStringStat, nullptr);
  expectStringStatisticsEqual(stringStat, *actualStringStat);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripNestedRow) {
  // Create a nested row schema
  auto schema = velox::ROW(
      {{"int_col", velox::INTEGER()},
       {"nested",
        velox::ROW(
            {{"nested_int", velox::BIGINT()},
             {"nested_string", velox::VARCHAR()}})}});
  auto nimbleType = convertToNimbleType(*schema);

  // Schema order: outer_row, int_col, nested_row, nested_int, nested_string
  ColumnStatistics outerRowStat(1000, 0, 20000, 10000);
  IntegralStatistics intStat(900, 50, 3600, 1800, 0, 500);
  ColumnStatistics nestedRowStat(850, 100, 15000, 7500);
  IntegralStatistics nestedIntStat(800, 50, 6400, 3200, -1000, 1000);
  StringStatistics nestedStringStat(750, 100, 9000, 4500, "hello", "world");

  std::vector<ColumnStatistics*> originalStats{
      &outerRowStat,
      &intStat,
      &nestedRowStat,
      &nestedIntStat,
      &nestedStringStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 5);

  // Check each stat
  expectColumnStatisticsEqual(outerRowStat, *roundTrippedStats[0]);

  auto* actualIntStat = roundTrippedStats[1]->as<IntegralStatistics>();
  ASSERT_NE(actualIntStat, nullptr);
  expectIntegralStatisticsEqual(intStat, *actualIntStat);

  expectColumnStatisticsEqual(nestedRowStat, *roundTrippedStats[2]);

  auto* actualNestedIntStat = roundTrippedStats[3]->as<IntegralStatistics>();
  ASSERT_NE(actualNestedIntStat, nullptr);
  expectIntegralStatisticsEqual(nestedIntStat, *actualNestedIntStat);

  auto* actualNestedStringStat = roundTrippedStats[4]->as<StringStatistics>();
  ASSERT_NE(actualNestedStringStat, nullptr);
  expectStringStatisticsEqual(nestedStringStat, *actualNestedStringStat);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripWithArray) {
  // Create a schema with an array type
  auto schema = velox::ROW(
      {{"array_col", velox::ARRAY(velox::BIGINT())},
       {"int_col", velox::INTEGER()}});
  auto nimbleType = convertToNimbleType(*schema);

  // Schema order: row, array, array_element (int), int_col
  ColumnStatistics rowStat(1000, 0, 20000, 10000);
  ColumnStatistics arrayStat(900, 50, 10000, 5000);
  IntegralStatistics elementStat(5000, 0, 40000, 20000, -999, 999);
  IntegralStatistics intColStat(800, 100, 3200, 1600, 0, 100);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &arrayStat, &elementStat, &intColStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 4);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);
  expectColumnStatisticsEqual(arrayStat, *roundTrippedStats[1]);

  auto* actualElementStat = roundTrippedStats[2]->as<IntegralStatistics>();
  ASSERT_NE(actualElementStat, nullptr);
  expectIntegralStatisticsEqual(elementStat, *actualElementStat);

  auto* actualIntColStat = roundTrippedStats[3]->as<IntegralStatistics>();
  ASSERT_NE(actualIntColStat, nullptr);
  expectIntegralStatisticsEqual(intColStat, *actualIntColStat);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripWithMap) {
  // Create a schema with a map type
  auto schema =
      velox::ROW({{"map_col", velox::MAP(velox::VARCHAR(), velox::DOUBLE())}});
  auto nimbleType = convertToNimbleType(*schema);

  // Schema order: row, map, key (string), value (double)
  ColumnStatistics rowStat(1000, 0, 30000, 15000);
  ColumnStatistics mapStat(900, 50, 25000, 12500);
  StringStatistics keyStat(4000, 0, 20000, 10000, "key_a", "key_z");
  FloatingPointStatistics valueStat(4000, 500, 32000, 16000, 0.0, 100.0);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &mapStat, &keyStat, &valueStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 4);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);
  expectColumnStatisticsEqual(mapStat, *roundTrippedStats[1]);

  auto* actualKeyStat = roundTrippedStats[2]->as<StringStatistics>();
  ASSERT_NE(actualKeyStat, nullptr);
  expectStringStatisticsEqual(keyStat, *actualKeyStat);

  auto* actualValueStat = roundTrippedStats[3]->as<FloatingPointStatistics>();
  ASSERT_NE(actualValueStat, nullptr);
  expectFloatingPointStatisticsEqual(valueStat, *actualValueStat);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripWithNulloptMinMax) {
  // Test with nullopt min/max values
  auto schema = velox::ROW(
      {{"int_col", velox::BIGINT()},
       {"double_col", velox::DOUBLE()},
       {"string_col", velox::VARCHAR()}});
  auto nimbleType = convertToNimbleType(*schema);

  // All leaf columns have nullopt for min/max
  ColumnStatistics rowStat(1000, 0, 10000, 5000);
  IntegralStatistics intStat(800, 50, 6400, 3200, std::nullopt, std::nullopt);
  FloatingPointStatistics doubleStat(
      750, 100, 6000, 3000, std::nullopt, std::nullopt);
  StringStatistics stringStat(
      600, 200, 12000, 6000, std::nullopt, std::nullopt);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &intStat, &doubleStat, &stringStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 4);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);

  auto* actualIntStat = roundTrippedStats[1]->as<IntegralStatistics>();
  ASSERT_NE(actualIntStat, nullptr);
  expectIntegralStatisticsEqual(intStat, *actualIntStat);

  auto* actualDoubleStat = roundTrippedStats[2]->as<FloatingPointStatistics>();
  ASSERT_NE(actualDoubleStat, nullptr);
  expectFloatingPointStatisticsEqual(doubleStat, *actualDoubleStat);

  auto* actualStringStat = roundTrippedStats[3]->as<StringStatistics>();
  ASSERT_NE(actualStringStat, nullptr);
  expectStringStatisticsEqual(stringStat, *actualStringStat);
}

// Writes a string min/max longer than the limit as absent and keeps one exactly
// at the limit, leaving counts and sizes unchanged.
TEST_F(VectorizedFileStatsTests, stringStatsLengthLimitDropsLongMinMax) {
  auto schema = velox::ROW(
      {{"at_limit_col", velox::VARCHAR()},
       {"over_limit_col", velox::VARCHAR()}});
  auto nimbleType = convertToNimbleType(*schema);

  ColumnStatistics rowStat(10, 0, 100, 50);
  StringStatistics atLimitStat(10, 1, 40, 20, "abcd", "wxyz");
  StringStatistics overLimitStat(10, 2, 60, 30, "abc", "wxyz12345");
  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &atLimitStat, &overLimitStat};

  VectorizedFileStats fileStats(
      originalStats,
      leafPool_.get(),
      VectorizedFileStats::Options{.stringStatsLengthLimit = 4});
  nimble::Buffer buffer(*leafPool_);
  auto deserialized =
      VectorizedFileStats::deserialize(fileStats.serialize(buffer), *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 3);

  auto* actualAtLimitStat = roundTrippedStats[1]->as<StringStatistics>();
  ASSERT_NE(actualAtLimitStat, nullptr);
  expectStringStatisticsEqual(atLimitStat, *actualAtLimitStat);

  const StringStatistics expectedOverLimitStat(
      10, 2, 60, 30, "abc", std::nullopt);
  auto* actualOverLimitStat = roundTrippedStats[2]->as<StringStatistics>();
  ASSERT_NE(actualOverLimitStat, nullptr);
  expectStringStatisticsEqual(expectedOverLimitStat, *actualOverLimitStat);
}

namespace {

// Splits the output of VectorizedFileStats::serialize() into stat streams.
std::vector<std::string_view> splitStatStreams(std::string_view serialized) {
  const char* pos = serialized.data();
  // Skips the version.
  pos += sizeof(uint16_t);
  const auto numStreams = encoding::read<uint8_t>(pos);
  const auto typeBitWidth = encoding::read<uint8_t>(pos);
  // Skips the stream types and the flag for encoded lengths.
  pos += FixedBitArray::bufferSize(numStreams, typeBitWidth) + sizeof(uint8_t);
  std::vector<uint64_t> lengths;
  lengths.reserve(numStreams);
  for (uint8_t i = 0; i < numStreams; ++i) {
    lengths.push_back(encoding::readUint64(pos));
  }
  std::vector<std::string_view> streams;
  streams.reserve(numStreams);
  for (const auto length : lengths) {
    streams.emplace_back(pos, length);
    pos += length;
  }
  return streams;
}

// Encoding and compression types found in stat streams.
struct StatStreamEncodings {
  std::set<EncodingType> encodingTypes;
  std::set<CompressionType> compressionTypes;
};

// Adds the types used by 'layout' and its nested encodings to 'encodings'.
void collectEncodings(
    const EncodingLayout& layout,
    StatStreamEncodings& encodings) {
  encodings.encodingTypes.insert(layout.encodingType());
  encodings.compressionTypes.insert(layout.compressionType());
  for (uint8_t i = 0; i < layout.childrenCount(); ++i) {
    const auto& child = layout.child(i);
    if (child.has_value()) {
      collectEncodings(*child, encodings);
    }
  }
}

// Adds the types used by Nullable 'stream' to 'encodings'. Captures each child
// separately, since EncodingLayoutCapture keeps only the value child.
void collectStatStreamEncodings(
    std::string_view stream,
    StatStreamEncodings& encodings) {
  encodings.encodingTypes.insert(EncodingPrefix::encodingType(stream));
  const char* pos =
      stream.data() + EncodingPrefix::prefixSize(stream, /*useVarint=*/false);
  const auto valuesSize = encoding::readUint32(pos);
  const std::string_view values{pos, valuesSize};
  const std::string_view nullFlags{
      pos + valuesSize,
      static_cast<size_t>(stream.data() + stream.size() - pos - valuesSize)};
  for (const auto child : {values, nullFlags}) {
    collectEncodings(
        EncodingLayoutCapture::capture(child, Encoding::Options{}), encodings);
  }
}

} // namespace

// Writes wide stats using metadata defaults and reads them back. Column-data
// defaults would compress these streams, encode the sizes, which span many
// orders of magnitude, as Varint, and the mostly-zero null counts as RLE.
TEST_F(VectorizedFileStatsTests, serializeUsesMetadataDefaults) {
  // Cycles integral, floating point and string columns. Every 7th has no
  // bounds.
  constexpr int kNumColumns = 999;
  std::vector<std::string> names;
  std::vector<velox::TypePtr> types;
  std::vector<std::unique_ptr<ColumnStatistics>> ownedStats;
  ownedStats.push_back(
      std::make_unique<ColumnStatistics>(
          10'000, 0, uint64_t{1} << 45, uint64_t{1} << 44));
  for (int i = 0; i < kNumColumns; ++i) {
    names.push_back("c" + std::to_string(i));
    const uint64_t valueCount = 10'000 + i;
    const uint64_t nullCount = i % 50 == 0 ? i : 0;
    const uint64_t logicalSize = (uint64_t{1} << (i % 40)) + i;
    const uint64_t physicalSize = logicalSize / 2;
    const bool hasBounds = i % 7 != 0;
    switch (i % 3) {
      case 0:
        types.push_back(velox::BIGINT());
        ownedStats.push_back(
            std::make_unique<IntegralStatistics>(
                valueCount,
                nullCount,
                logicalSize,
                physicalSize,
                hasBounds ? std::optional<int64_t>{-i} : std::nullopt,
                hasBounds ? std::optional<int64_t>{i} : std::nullopt));
        break;
      case 1:
        types.push_back(velox::DOUBLE());
        ownedStats.push_back(
            std::make_unique<FloatingPointStatistics>(
                valueCount,
                nullCount,
                logicalSize,
                physicalSize,
                hasBounds ? std::optional<double>{-0.5 * i} : std::nullopt,
                hasBounds ? std::optional<double>{0.5 * i} : std::nullopt));
        break;
      default:
        types.push_back(velox::VARCHAR());
        ownedStats.push_back(
            std::make_unique<StringStatistics>(
                valueCount,
                nullCount,
                logicalSize,
                physicalSize,
                hasBounds ? std::optional<
                                std::string>{"min_value_" + std::to_string(i)}
                          : std::nullopt,
                hasBounds ? std::optional<
                                std::string>{"max_value_" + std::to_string(i)}
                          : std::nullopt));
    }
  }
  const auto schema = velox::ROW(std::move(names), std::move(types));
  std::vector<ColumnStatistics*> columnStats;
  columnStats.reserve(ownedStats.size());
  for (const auto& stat : ownedStats) {
    columnStats.push_back(stat.get());
  }

  VectorizedFileStats fileStats(columnStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  const auto serialized = fileStats.serialize(buffer);

  // 4 default streams plus 2 each for integral, floating point and string.
  const auto streams = splitStatStreams(serialized);
  ASSERT_EQ(streams.size(), 10);
  StatStreamEncodings encodings;
  for (const auto stream : streams) {
    collectStatStreamEncodings(stream, encodings);
  }
  EXPECT_THAT(
      encodings.encodingTypes,
      testing::IsSubsetOf(
          std::set<EncodingType>{
              EncodingType::Nullable,
              EncodingType::Constant,
              EncodingType::Trivial,
              EncodingType::FixedBitWidth,
              EncodingType::Dictionary,
          }));
  EXPECT_EQ(
      encodings.compressionTypes,
      std::set<CompressionType>{CompressionType::Uncompressed});

  const auto roundTrippedStats =
      VectorizedFileStats::deserialize(serialized, *leafPool_)
          ->toColumnStatistics(schema, convertToNimbleType(*schema));
  ASSERT_EQ(roundTrippedStats.size(), ownedStats.size());
  expectColumnStatisticsEqual(*ownedStats[0], *roundTrippedStats[0]);
  for (int i = 0; i < kNumColumns; ++i) {
    SCOPED_TRACE("column " + std::to_string(i));
    const auto& expected = *ownedStats[i + 1];
    const auto& actual = *roundTrippedStats[i + 1];
    switch (i % 3) {
      case 0:
        ASSERT_NE(actual.as<IntegralStatistics>(), nullptr);
        expectIntegralStatisticsEqual(
            *expected.as<IntegralStatistics>(),
            *actual.as<IntegralStatistics>());
        break;
      case 1:
        ASSERT_NE(actual.as<FloatingPointStatistics>(), nullptr);
        expectFloatingPointStatisticsEqual(
            *expected.as<FloatingPointStatistics>(),
            *actual.as<FloatingPointStatistics>());
        break;
      default:
        ASSERT_NE(actual.as<StringStatistics>(), nullptr);
        expectStringStatisticsEqual(
            *expected.as<StringStatistics>(), *actual.as<StringStatistics>());
    }
  }
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripWithTimestamp) {
  // Test with timestamp type which uses default stats
  auto schema = velox::ROW(
      {{"timestamp_col", velox::TIMESTAMP()}, {"bool_col", velox::BOOLEAN()}});
  auto nimbleType = convertToNimbleType(*schema);

  // Timestamp and boolean use default stats
  ColumnStatistics rowStat(1000, 0, 20000, 10000);
  ColumnStatistics timestampStat(900, 50, 14400, 7200);
  ColumnStatistics boolStat(900, 50, 900, 450);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &timestampStat, &boolStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 3);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);
  expectColumnStatisticsEqual(timestampStat, *roundTrippedStats[1]);
  expectColumnStatisticsEqual(boolStat, *roundTrippedStats[2]);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripComplexNested) {
  // Create a complex nested schema with multiple levels
  auto schema = velox::ROW(
      {{"id", velox::BIGINT()},
       {"data",
        velox::ROW(
            {{"values", velox::ARRAY(velox::DOUBLE())},
             {"metadata", velox::MAP(velox::VARCHAR(), velox::INTEGER())}})}});
  auto nimbleType = convertToNimbleType(*schema);

  // Schema order:
  // 0: outer_row
  // 1: id (int)
  // 2: data (row)
  // 3: values (array)
  // 4: values element (double)
  // 5: metadata (map)
  // 6: metadata key (string)
  // 7: metadata value (int)

  ColumnStatistics outerRowStat(1000, 0, 50000, 25000);
  IntegralStatistics idStat(1000, 0, 8000, 4000, 1, 1000);
  ColumnStatistics dataRowStat(1000, 0, 42000, 21000);
  ColumnStatistics valuesArrayStat(950, 50, 20000, 10000);
  FloatingPointStatistics valuesElementStat(
      5000, 100, 40000, 20000, -100.0, 100.0);
  ColumnStatistics metadataMapStat(900, 100, 22000, 11000);
  StringStatistics metadataKeyStat(3000, 0, 15000, 7500, "a", "z");
  IntegralStatistics metadataValueStat(3000, 200, 12000, 6000, -50, 50);

  std::vector<ColumnStatistics*> originalStats{
      &outerRowStat,
      &idStat,
      &dataRowStat,
      &valuesArrayStat,
      &valuesElementStat,
      &metadataMapStat,
      &metadataKeyStat,
      &metadataValueStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 8);

  // Verify each stat
  expectColumnStatisticsEqual(outerRowStat, *roundTrippedStats[0]);

  auto* actualIdStat = roundTrippedStats[1]->as<IntegralStatistics>();
  ASSERT_NE(actualIdStat, nullptr);
  expectIntegralStatisticsEqual(idStat, *actualIdStat);

  expectColumnStatisticsEqual(dataRowStat, *roundTrippedStats[2]);
  expectColumnStatisticsEqual(valuesArrayStat, *roundTrippedStats[3]);

  auto* actualValuesElementStat =
      roundTrippedStats[4]->as<FloatingPointStatistics>();
  ASSERT_NE(actualValuesElementStat, nullptr);
  expectFloatingPointStatisticsEqual(
      valuesElementStat, *actualValuesElementStat);

  expectColumnStatisticsEqual(metadataMapStat, *roundTrippedStats[5]);

  auto* actualMetadataKeyStat = roundTrippedStats[6]->as<StringStatistics>();
  ASSERT_NE(actualMetadataKeyStat, nullptr);
  expectStringStatisticsEqual(metadataKeyStat, *actualMetadataKeyStat);

  auto* actualMetadataValueStat =
      roundTrippedStats[7]->as<IntegralStatistics>();
  ASSERT_NE(actualMetadataValueStat, nullptr);
  expectIntegralStatisticsEqual(metadataValueStat, *actualMetadataValueStat);
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripAllIntegralTypes) {
  // Test all integral types to ensure they round-trip correctly
  auto schema = velox::ROW(
      {{"tinyint_col", velox::TINYINT()},
       {"smallint_col", velox::SMALLINT()},
       {"int_col", velox::INTEGER()},
       {"bigint_col", velox::BIGINT()}});
  auto nimbleType = convertToNimbleType(*schema);

  ColumnStatistics rowStat(1000, 0, 20000, 10000);
  IntegralStatistics tinyintStat(950, 50, 950, 475, -128, 127);
  IntegralStatistics smallintStat(900, 100, 1800, 900, -32768, 32767);
  IntegralStatistics intStat(850, 150, 3400, 1700, -2147483648, 2147483647);
  IntegralStatistics bigintStat(
      800, 200, 6400, 3200, -1000000000000, 1000000000000);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &tinyintStat, &smallintStat, &intStat, &bigintStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 5);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);

  for (size_t i = 1; i < 5; ++i) {
    auto* actualStat = roundTrippedStats[i]->as<IntegralStatistics>();
    ASSERT_NE(actualStat, nullptr);
    expectIntegralStatisticsEqual(
        *dynamic_cast<IntegralStatistics*>(originalStats[i]), *actualStat);
  }
}

TEST_F(VectorizedFileStatsTests, schemaBasedRoundTripAllFloatingPointTypes) {
  // Test all floating point types
  auto schema = velox::ROW(
      {{"real_col", velox::REAL()}, {"double_col", velox::DOUBLE()}});
  auto nimbleType = convertToNimbleType(*schema);

  ColumnStatistics rowStat(1000, 0, 12000, 6000);
  FloatingPointStatistics realStat(900, 100, 3600, 1800, -1.5f, 2.5f);
  FloatingPointStatistics doubleStat(800, 200, 6400, 3200, -999.999, 999.999);

  std::vector<ColumnStatistics*> originalStats{
      &rowStat, &realStat, &doubleStat};

  VectorizedFileStats fileStats(originalStats, leafPool_.get());
  nimble::Buffer buffer(*leafPool_);
  auto serialized = fileStats.serialize(buffer);

  auto deserialized = VectorizedFileStats::deserialize(serialized, *leafPool_);
  auto roundTrippedStats = deserialized->toColumnStatistics(schema, nimbleType);
  ASSERT_EQ(roundTrippedStats.size(), 3);

  expectColumnStatisticsEqual(rowStat, *roundTrippedStats[0]);

  auto* actualRealStat = roundTrippedStats[1]->as<FloatingPointStatistics>();
  ASSERT_NE(actualRealStat, nullptr);
  expectFloatingPointStatisticsEqual(realStat, *actualRealStat);

  auto* actualDoubleStat = roundTrippedStats[2]->as<FloatingPointStatistics>();
  ASSERT_NE(actualDoubleStat, nullptr);
  expectFloatingPointStatisticsEqual(doubleStat, *actualDoubleStat);
}
