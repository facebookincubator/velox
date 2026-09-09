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
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

template <typename T>
void testIntegralMinMaxAcrossBatches() {
  IntegralStatisticsCollector collector;

  std::vector<T> firstValues = {
      static_cast<T>(5),
      static_cast<T>(-3),
      static_cast<T>(12),
      static_cast<T>(7),
  };
  collector.addValues(std::span<T>(firstValues));

  auto* stats = collector.getStatsView()->as<IntegralStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin(), -3);
  EXPECT_EQ(stats->getMax(), 12);

  std::vector<T> secondValues = {
      std::numeric_limits<T>::max(),
      static_cast<T>(0),
      std::numeric_limits<T>::min(),
  };
  collector.addValues(std::span<T>(secondValues));

  EXPECT_EQ(
      stats->getMin(), static_cast<int64_t>(std::numeric_limits<T>::min()));
  EXPECT_EQ(
      stats->getMax(), static_cast<int64_t>(std::numeric_limits<T>::max()));
}

template <typename T>
void testFloatingPointMinMaxAcrossBatches() {
  FloatingPointStatisticsCollector collector;

  std::vector<T> firstValues = {
      static_cast<T>(5.5),
      static_cast<T>(-3.25),
      static_cast<T>(12.75),
      static_cast<T>(7.0),
  };
  collector.addValues(std::span<T>(firstValues));

  auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getMin().has_value());
  EXPECT_DOUBLE_EQ(stats->getMin().value(), -3.25);
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_DOUBLE_EQ(stats->getMax().value(), 12.75);

  std::vector<T> secondValues = {
      std::numeric_limits<T>::max(),
      static_cast<T>(0),
      std::numeric_limits<T>::lowest(),
  };
  collector.addValues(std::span<T>(secondValues));

  ASSERT_TRUE(stats->getMin().has_value());
  EXPECT_DOUBLE_EQ(
      stats->getMin().value(),
      static_cast<double>(std::numeric_limits<T>::lowest()));
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_DOUBLE_EQ(
      stats->getMax().value(),
      static_cast<double>(std::numeric_limits<T>::max()));
}

} // namespace

class ColumnStatisticsTests : public ::testing::Test {};

// Base ColumnStatistics Tests

TEST_F(ColumnStatisticsTests, defaultStatistics) {
  // Default constructor.
  {
    ColumnStatistics stat;
    EXPECT_EQ(stat.getValueCount(), 0);
    EXPECT_EQ(stat.getNullCount(), 0);
    EXPECT_EQ(stat.getLogicalSize(), 0);
    EXPECT_EQ(stat.getPhysicalSize(), 0);
    EXPECT_EQ(stat.getType(), StatType::DEFAULT);
  }

  // Some random typical values.
  {
    ColumnStatistics stat(100, 10, 1000, 500);
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getType(), StatType::DEFAULT);
  }

  // Explicit zeroes.
  {
    ColumnStatistics stat(0, 0, 0, 0);
    EXPECT_EQ(stat.getValueCount(), 0);
    EXPECT_EQ(stat.getNullCount(), 0);
    EXPECT_EQ(stat.getLogicalSize(), 0);
    EXPECT_EQ(stat.getPhysicalSize(), 0);
  }
  // Numeric limits
  {
    uint64_t maxVal = std::numeric_limits<uint64_t>::max();
    ColumnStatistics stat(maxVal, maxVal, maxVal, maxVal);
    EXPECT_EQ(stat.getValueCount(), maxVal);
    EXPECT_EQ(stat.getNullCount(), maxVal);
    EXPECT_EQ(stat.getLogicalSize(), maxVal);
    EXPECT_EQ(stat.getPhysicalSize(), maxVal);
  }
}

TEST_F(ColumnStatisticsTests, integralStatistics) {
  {
    IntegralStatistics stat;
    EXPECT_EQ(stat.getValueCount(), 0);
    EXPECT_EQ(stat.getNullCount(), 0);
    EXPECT_EQ(stat.getLogicalSize(), 0);
    EXPECT_EQ(stat.getPhysicalSize(), 0);
    EXPECT_EQ(stat.getType(), StatType::INTEGRAL);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), std::nullopt);
  }

  {
    IntegralStatistics stat(100, 10, 1000, 500, -50, 150);
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getMin(), -50);
    EXPECT_EQ(stat.getMax(), 150);
    EXPECT_EQ(stat.getType(), StatType::INTEGRAL);
  }

  {
    IntegralStatistics stat(100, 10, 1000, 500, std::nullopt, 150);
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), 150);
  }

  {
    int64_t minVal = std::numeric_limits<int64_t>::min();
    int64_t maxVal = std::numeric_limits<int64_t>::max();
    IntegralStatistics stat(100, 10, 1000, 500, minVal, maxVal);
    EXPECT_EQ(stat.getMin(), minVal);
    EXPECT_EQ(stat.getMax(), maxVal);
  }
}

TEST_F(ColumnStatisticsTests, floatingPointStatistics) {
  {
    FloatingPointStatistics stat;
    EXPECT_EQ(stat.getValueCount(), 0);
    EXPECT_EQ(stat.getNullCount(), 0);
    EXPECT_EQ(stat.getLogicalSize(), 0);
    EXPECT_EQ(stat.getPhysicalSize(), 0);
    EXPECT_EQ(stat.getType(), StatType::FLOATING_POINT);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), std::nullopt);
  }

  {
    FloatingPointStatistics stat(100, 10, 1000, 500, -1.5, 2.5);
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getMin(), -1.5);
    EXPECT_EQ(stat.getMax(), 2.5);
    EXPECT_EQ(stat.getType(), StatType::FLOATING_POINT);
  }

  {
    FloatingPointStatistics stat(100, 10, 1000, 500, std::nullopt, 2.5);
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), 2.5);
  }

  {
    double minVal = std::numeric_limits<double>::lowest();
    double maxVal = std::numeric_limits<double>::max();
    FloatingPointStatistics stat(100, 10, 1000, 500, minVal, maxVal);
    EXPECT_EQ(stat.getMin(), minVal);
    EXPECT_EQ(stat.getMax(), maxVal);
  }
}

// StringStatistics Tests

TEST_F(ColumnStatisticsTests, stringStatisticsDefaultConstructor) {
  {
    StringStatistics stat;
    EXPECT_EQ(stat.getValueCount(), 0);
    EXPECT_EQ(stat.getNullCount(), 0);
    EXPECT_EQ(stat.getLogicalSize(), 0);
    EXPECT_EQ(stat.getPhysicalSize(), 0);
    EXPECT_EQ(stat.getType(), StatType::STRING);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), std::nullopt);
  }

  {
    StringStatistics stat(100, 10, 1000, 500, "aaa", "zzz");
    EXPECT_EQ(stat.getValueCount(), 100);
    EXPECT_EQ(stat.getNullCount(), 10);
    EXPECT_EQ(stat.getLogicalSize(), 1000);
    EXPECT_EQ(stat.getPhysicalSize(), 500);
    EXPECT_EQ(stat.getMin(), "aaa");
    EXPECT_EQ(stat.getMax(), "zzz");
    EXPECT_EQ(stat.getType(), StatType::STRING);
  }

  {
    StringStatistics stat(100, 10, 1000, 500, std::nullopt, std::nullopt);
    EXPECT_EQ(stat.getMin(), std::nullopt);
    EXPECT_EQ(stat.getMax(), std::nullopt);
  }

  {
    StringStatistics stat(100, 10, 1000, 500, "", "");
    EXPECT_EQ(stat.getMin(), "");
    EXPECT_EQ(stat.getMax(), "");
  }

  // Long strings
  {
    std::string longStr(1000, 'x');
    StringStatistics stat(100, 10, 1000, 500, longStr, longStr);
    EXPECT_EQ(stat.getMin(), longStr);
    EXPECT_EQ(stat.getMax(), longStr);
  }

  // Unicode strings
  {
    StringStatistics stat(100, 10, 1000, 500, "αβγ", "日本語");
    EXPECT_EQ(stat.getMin(), "αβγ");
    EXPECT_EQ(stat.getMax(), "日本語");
  }
}

TEST_F(ColumnStatisticsTests, deduplicatedColumnStatistics) {
  {
    DeduplicatedColumnStatistics stat;
    EXPECT_EQ(stat.getType(), StatType::DEDUPLICATED);
  }

  {
    IntegralStatistics baseStat(100, 10, 1000, 500, -50, 150);
    DeduplicatedColumnStatistics stat(&baseStat, 80, 800);

    EXPECT_EQ(stat.getDedupedCount(), 80);
    EXPECT_EQ(stat.getDedupedLogicalSize(), 800);
    EXPECT_EQ(stat.getType(), StatType::DEDUPLICATED);

    const auto* base = stat.getBaseStatistics();
    EXPECT_EQ(base->getValueCount(), 100);
    EXPECT_EQ(base->getNullCount(), 10);
  }
}

// toCommonStatistics Tests

TEST_F(ColumnStatisticsTests, toCommonStatisticsIntegral) {
  IntegralStatistics stat(100, 10, 1000, 500, -50, 150);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->getNumberOfValues().has_value());
  EXPECT_EQ(common->getNumberOfValues().value(), 100);
  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_TRUE(common->hasNull().value());
  ASSERT_TRUE(common->getRawSize().has_value());
  EXPECT_EQ(common->getRawSize().value(), 1000);
  ASSERT_TRUE(common->getSize().has_value());
  EXPECT_EQ(common->getSize().value(), 500);

  auto* intStats =
      dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(common.get());
  ASSERT_NE(intStats, nullptr);
  ASSERT_TRUE(intStats->getMinimum().has_value());
  EXPECT_EQ(intStats->getMinimum().value(), -50);
  ASSERT_TRUE(intStats->getMaximum().has_value());
  EXPECT_EQ(intStats->getMaximum().value(), 150);
  EXPECT_EQ(intStats->getSum(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsIntegralNoMinMax) {
  IntegralStatistics stat(50, 0, 400, 200, std::nullopt, std::nullopt);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_FALSE(common->hasNull().value());

  auto* intStats =
      dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(common.get());
  ASSERT_NE(intStats, nullptr);
  EXPECT_EQ(intStats->getMinimum(), std::nullopt);
  EXPECT_EQ(intStats->getMaximum(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsFloatingPoint) {
  FloatingPointStatistics stat(200, 5, 2000, 800, -1.5, 99.9);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->getNumberOfValues().has_value());
  EXPECT_EQ(common->getNumberOfValues().value(), 200);
  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_TRUE(common->hasNull().value());

  auto* fpStats =
      dynamic_cast<velox::dwio::common::DoubleColumnStatistics*>(common.get());
  ASSERT_NE(fpStats, nullptr);
  ASSERT_TRUE(fpStats->getMinimum().has_value());
  EXPECT_DOUBLE_EQ(fpStats->getMinimum().value(), -1.5);
  ASSERT_TRUE(fpStats->getMaximum().has_value());
  EXPECT_DOUBLE_EQ(fpStats->getMaximum().value(), 99.9);
  EXPECT_EQ(fpStats->getSum(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsFloatingPointNoMinMax) {
  FloatingPointStatistics stat(30, 0, 240, 120, std::nullopt, std::nullopt);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_FALSE(common->hasNull().value());

  auto* fpStats =
      dynamic_cast<velox::dwio::common::DoubleColumnStatistics*>(common.get());
  ASSERT_NE(fpStats, nullptr);
  EXPECT_EQ(fpStats->getMinimum(), std::nullopt);
  EXPECT_EQ(fpStats->getMaximum(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsString) {
  StringStatistics stat(300, 20, 3000, 1500, "apple", "zebra");
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->getNumberOfValues().has_value());
  EXPECT_EQ(common->getNumberOfValues().value(), 300);
  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_TRUE(common->hasNull().value());

  auto* strStats =
      dynamic_cast<velox::dwio::common::StringColumnStatistics*>(common.get());
  ASSERT_NE(strStats, nullptr);
  ASSERT_TRUE(strStats->getMinimum().has_value());
  EXPECT_EQ(strStats->getMinimum().value(), "apple");
  ASSERT_TRUE(strStats->getMaximum().has_value());
  EXPECT_EQ(strStats->getMaximum().value(), "zebra");
  EXPECT_EQ(strStats->getTotalLength(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsStringNoMinMax) {
  StringStatistics stat(10, 0, 80, 40, std::nullopt, std::nullopt);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_FALSE(common->hasNull().value());

  auto* strStats =
      dynamic_cast<velox::dwio::common::StringColumnStatistics*>(common.get());
  ASSERT_NE(strStats, nullptr);
  EXPECT_EQ(strStats->getMinimum(), std::nullopt);
  EXPECT_EQ(strStats->getMaximum(), std::nullopt);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsDefault) {
  ColumnStatistics stat(500, 0, 4000, 2000);
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->getNumberOfValues().has_value());
  EXPECT_EQ(common->getNumberOfValues().value(), 500);
  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_FALSE(common->hasNull().value());
  ASSERT_TRUE(common->getRawSize().has_value());
  EXPECT_EQ(common->getRawSize().value(), 4000);
  ASSERT_TRUE(common->getSize().has_value());
  EXPECT_EQ(common->getSize().value(), 2000);

  // Should not cast to any typed subclass.
  EXPECT_EQ(
      dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(common.get()),
      nullptr);
  EXPECT_EQ(
      dynamic_cast<velox::dwio::common::DoubleColumnStatistics*>(common.get()),
      nullptr);
  EXPECT_EQ(
      dynamic_cast<velox::dwio::common::StringColumnStatistics*>(common.get()),
      nullptr);
}

TEST_F(ColumnStatisticsTests, toCommonStatisticsDeduplicated) {
  IntegralStatistics baseStat(100, 10, 1000, 500, -50, 150);
  DeduplicatedColumnStatistics stat(&baseStat, 80, 800);
  // DEDUPLICATED falls through to the base ColumnStatistics path.
  auto common = stat.toCommonStatistics();
  ASSERT_NE(common, nullptr);

  ASSERT_TRUE(common->getNumberOfValues().has_value());
  EXPECT_EQ(common->getNumberOfValues().value(), 100);
  ASSERT_TRUE(common->hasNull().has_value());
  EXPECT_TRUE(common->hasNull().value());

  // Should produce base ColumnStatistics, not IntegerColumnStatistics.
  EXPECT_EQ(
      dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(common.get()),
      nullptr);
}

class StatisticsCollectorTests : public ::testing::Test {};

// Base StatisticsCollector Tests

TEST_F(StatisticsCollectorTests, ctor) {
  StatisticsCollector collector;
  EXPECT_EQ(collector.getValueCount(), 0);
  EXPECT_EQ(collector.getNullCount(), 0);
  EXPECT_EQ(collector.getLogicalSize(), 0);
  EXPECT_EQ(collector.getPhysicalSize(), 0);
  EXPECT_EQ(collector.getType(), StatType::DEFAULT);
}

TEST_F(StatisticsCollectorTests, addCounts) {
  StatisticsCollector collector;
  collector.addCounts(100, 10);
  EXPECT_EQ(collector.getValueCount(), 90); // nonNullCount = 100 - 10
  EXPECT_EQ(collector.getNullCount(), 10);

  collector.addCounts(50, 5);
  EXPECT_EQ(collector.getValueCount(), 135); // 90 + 45 (50 - 5)
  EXPECT_EQ(collector.getNullCount(), 15);
}

TEST_F(StatisticsCollectorTests, addLogicalSize) {
  StatisticsCollector collector;
  collector.addLogicalSize(1000);
  EXPECT_EQ(collector.getLogicalSize(), 1000);

  collector.addLogicalSize(500);
  EXPECT_EQ(collector.getLogicalSize(), 1500);
}

TEST_F(StatisticsCollectorTests, addPhysicalSize) {
  StatisticsCollector collector;
  collector.addPhysicalSize(500);
  EXPECT_EQ(collector.getPhysicalSize(), 500);

  collector.addPhysicalSize(250);
  EXPECT_EQ(collector.getPhysicalSize(), 750);
}

TEST_F(StatisticsCollectorTests, addBoolValues) {
  {
    StatisticsCollector collector;
    bool values[] = {true, true, true, true};
    collector.addValues(std::span<bool>(values, 4));

    EXPECT_EQ(collector.getLogicalSize(), 4);
  }

  {
    StatisticsCollector collector;
    bool values[] = {false, false, false};
    collector.addValues(std::span<bool>(values, 3));

    EXPECT_EQ(collector.getLogicalSize(), 3);
  }

  {
    StatisticsCollector collector;
    bool values[] = {false, true, false};
    collector.addValues(std::span<bool>(values, 3));

    EXPECT_EQ(collector.getLogicalSize(), 3);
  }
}

TEST_F(StatisticsCollectorTests, mergeDefaultStatisticsCollectors) {
  StatisticsCollector collector1;
  collector1.addCounts(100, 10);
  collector1.addLogicalSize(1000);
  collector1.addPhysicalSize(500);

  StatisticsCollector collector2;
  collector2.addCounts(200, 20);
  collector2.addLogicalSize(2000);
  collector2.addPhysicalSize(1000);

  collector1.merge(collector2);

  EXPECT_EQ(collector1.getValueCount(), 270); // (100-10) + (200-20) = 90 + 180
  EXPECT_EQ(collector1.getNullCount(), 30);
  EXPECT_EQ(collector1.getLogicalSize(), 3000);
  EXPECT_EQ(collector1.getPhysicalSize(), 1500);
}

TEST_F(StatisticsCollectorTests, integralStatisticsCollectorCtor) {
  IntegralStatisticsCollector collector;
  auto* stats = collector.getStatsView()->as<IntegralStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin(), std::nullopt);
  EXPECT_EQ(stats->getMax(), std::nullopt);
  // Access getType() via dynamic_cast
  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getType(), StatType::INTEGRAL);
}

TEST_F(StatisticsCollectorTests, integralStatisticsCollectorMerge) {
  IntegralStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);
  dynamic_cast<StatisticsCollector&>(collector1).addLogicalSize(1000);

  IntegralStatisticsCollector collector2;
  dynamic_cast<StatisticsCollector&>(collector2).addCounts(200, 20);
  dynamic_cast<StatisticsCollector&>(collector2).addLogicalSize(2000);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 270); // (100-10) + (200-20) = 90 + 180
  EXPECT_EQ(sbPtr->getNullCount(), 30);
  EXPECT_EQ(sbPtr->getLogicalSize(), 3000);
}

TEST_F(StatisticsCollectorTests, integralStatisticsCollectorMergeWithDefault) {
  IntegralStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);

  StatisticsCollector collector2;
  collector2.addCounts(50, 5);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 135); // (100-10) + (50-5) = 90 + 45
  EXPECT_EQ(sbPtr->getNullCount(), 15);
}

// Regression test: addCounts() is called before addValues() in
// FieldWriter::collectStatistics. The min/max initialization guard must not
// depend on getValueCount() being zero.
TEST_F(StatisticsCollectorTests, integralMinMaxAfterAddCounts) {
  IntegralStatisticsCollector collector;
  // Simulate FieldWriter calling addCounts before addValues.
  dynamic_cast<StatisticsCollector&>(collector).addCounts(10, 0);
  std::vector<int64_t> values = {5, -3, 12, 7};
  collector.addValues(std::span<int64_t>(values));

  auto* stats = collector.getStatsView()->as<IntegralStatistics>();
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getMin().has_value());
  EXPECT_EQ(stats->getMin().value(), -3);
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_EQ(stats->getMax().value(), 12);
}

TEST_F(StatisticsCollectorTests, integralMinMaxAcrossBatches) {
  {
    SCOPED_TRACE("int8_t");
    testIntegralMinMaxAcrossBatches<int8_t>();
  }
  {
    SCOPED_TRACE("int16_t");
    testIntegralMinMaxAcrossBatches<int16_t>();
  }
  {
    SCOPED_TRACE("int32_t");
    testIntegralMinMaxAcrossBatches<int32_t>();
  }
  {
    SCOPED_TRACE("int64_t");
    testIntegralMinMaxAcrossBatches<int64_t>();
  }
}

TEST_F(StatisticsCollectorTests, floatingPointMinMaxAcrossBatches) {
  {
    SCOPED_TRACE("float");
    testFloatingPointMinMaxAcrossBatches<float>();
  }
  {
    SCOPED_TRACE("double");
    testFloatingPointMinMaxAcrossBatches<double>();
  }
}

TEST_F(StatisticsCollectorTests, floatingPointStatisticsCollectorCtor) {
  FloatingPointStatisticsCollector collector;
  auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin(), std::nullopt);
  EXPECT_EQ(stats->getMax(), std::nullopt);
  // Access getType() via dynamic_cast
  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getType(), StatType::FLOATING_POINT);
}

TEST_F(StatisticsCollectorTests, floatingPointStatisticsCollectorMerge) {
  FloatingPointStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);
  dynamic_cast<StatisticsCollector&>(collector1).addLogicalSize(1000);

  FloatingPointStatisticsCollector collector2;
  dynamic_cast<StatisticsCollector&>(collector2).addCounts(200, 20);
  dynamic_cast<StatisticsCollector&>(collector2).addLogicalSize(2000);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 270); // (100-10) + (200-20) = 90 + 180
  EXPECT_EQ(sbPtr->getNullCount(), 30);
  EXPECT_EQ(sbPtr->getLogicalSize(), 3000);
}

TEST_F(
    StatisticsCollectorTests,
    floatingPointStatisticsCollectorMergeWithDefault) {
  FloatingPointStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);

  StatisticsCollector collector2;
  collector2.addCounts(50, 5);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 135); // (100-10) + (50-5) = 90 + 45
  EXPECT_EQ(sbPtr->getNullCount(), 15);
}

// Regression test: same as IntegralMinMaxAfterAddCounts but for floating-point.
TEST_F(StatisticsCollectorTests, floatingPointMinMaxAfterAddCounts) {
  FloatingPointStatisticsCollector collector;
  dynamic_cast<StatisticsCollector&>(collector).addCounts(10, 0);
  std::vector<double> values = {1.5, -2.7, 99.9, 0.0};
  collector.addValues(std::span<double>(values));

  auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getMin().has_value());
  EXPECT_DOUBLE_EQ(stats->getMin().value(), -2.7);
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_DOUBLE_EQ(stats->getMax().value(), 99.9);
}

TEST_F(StatisticsCollectorTests, floatingPointNanBehavior) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  {
    FloatingPointStatisticsCollector collector;
    std::vector<double> values = {1.0, nan, -2.0, 3.0};
    collector.addValues(std::span<double>(values));

    auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
    ASSERT_NE(stats, nullptr);
    ASSERT_TRUE(stats->getMin().has_value());
    EXPECT_DOUBLE_EQ(stats->getMin().value(), -2.0);
    ASSERT_TRUE(stats->getMax().has_value());
    EXPECT_DOUBLE_EQ(stats->getMax().value(), 3.0);
  }
  {
    FloatingPointStatisticsCollector collector;
    std::vector<double> values = {nan, -2.0, 3.0};
    collector.addValues(std::span<double>(values));

    auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
    ASSERT_NE(stats, nullptr);
    ASSERT_TRUE(stats->getMin().has_value());
    EXPECT_TRUE(std::isnan(stats->getMin().value()));
    ASSERT_TRUE(stats->getMax().has_value());
    EXPECT_TRUE(std::isnan(stats->getMax().value()));
  }
  {
    FloatingPointStatisticsCollector collector;
    std::vector<double> firstValues = {0.0, 10.0};
    collector.addValues(std::span<double>(firstValues));
    std::vector<double> secondValues = {nan, -5.0, 15.0};
    collector.addValues(std::span<double>(secondValues));

    auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
    ASSERT_NE(stats, nullptr);
    ASSERT_TRUE(stats->getMin().has_value());
    EXPECT_DOUBLE_EQ(stats->getMin().value(), -5.0);
    ASSERT_TRUE(stats->getMax().has_value());
    EXPECT_DOUBLE_EQ(stats->getMax().value(), 15.0);
  }
}

// StringStatisticsCollector Tests

TEST_F(StatisticsCollectorTests, stringStatisticsCollectorCtor) {
  StringStatisticsCollector collector;
  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin(), std::nullopt);
  EXPECT_EQ(stats->getMax(), std::nullopt);
  // Access getType() via dynamic_cast
  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getType(), StatType::STRING);
}

TEST_F(StatisticsCollectorTests, addStringValues) {
  {
    StringStatisticsCollector collector;
    std::vector<std::string_view> values = {"hello", "world", "test"};
    collector.addValues(std::span<std::string_view>(values));

    auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector);
    ASSERT_NE(sbPtr, nullptr);
    EXPECT_EQ(sbPtr->getLogicalSize(), 5 + 5 + 4);
  }

  {
    StringStatisticsCollector collector;
    std::vector<std::string_view> values = {"", "", ""};
    collector.addValues(std::span<std::string_view>(values));

    auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector);
    ASSERT_NE(sbPtr, nullptr);
    EXPECT_EQ(sbPtr->getLogicalSize(), 0);
  }
}

// Regression test: same as IntegralMinMaxAfterAddCounts but for strings.
TEST_F(StatisticsCollectorTests, stringMinMaxAfterAddCounts) {
  StringStatisticsCollector collector;
  dynamic_cast<StatisticsCollector&>(collector).addCounts(10, 0);
  std::vector<std::string_view> values = {"banana", "apple", "cherry"};
  collector.addValues(std::span<std::string_view>(values));

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getMin().has_value());
  EXPECT_EQ(stats->getMin().value(), "apple");
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_EQ(stats->getMax().value(), "cherry");
}

// The batch scan tracks bounds as views and materializes at most two strings
// per call, so a caller's buffer must not be aliased after addValues returns
// and repeated calls must still accumulate a file-wide min/max. Ascending and
// descending input are the two orders that previously allocated on every value.
TEST_F(StatisticsCollectorTests, stringMinMaxAcrossBatchesDoesNotAliasInput) {
  StringStatisticsCollector collector;
  {
    // Ascending, then let the source buffer die before the next batch.
    std::vector<std::string> owned = {"aaa", "bbb", "ccc", "ddd"};
    std::vector<std::string_view> views(owned.begin(), owned.end());
    collector.addValues(std::span<std::string_view>(views));
  }
  {
    // Descending, disjoint range below the current min.
    std::vector<std::string> owned = {"a99", "a55", "a11"};
    std::vector<std::string_view> views(owned.begin(), owned.end());
    collector.addValues(std::span<std::string_view>(views));
  }
  {
    // Entirely inside the current bounds: neither bound may move.
    std::vector<std::string> owned = {"bzz", "czz"};
    std::vector<std::string_view> views(owned.begin(), owned.end());
    collector.addValues(std::span<std::string_view>(views));
  }
  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getMin().has_value());
  ASSERT_TRUE(stats->getMax().has_value());
  EXPECT_EQ(stats->getMin().value(), "a11");
  EXPECT_EQ(stats->getMax().value(), "ddd");
  EXPECT_EQ(
      dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(),
      12 + 9 + 6);
}

// The scan seeds both bounds from the first value and skips the maximum check
// whenever a value advances the minimum. Cover the orders where that shortcut
// could drop a bound: the extremes trailing the first value, and each extreme
// arriving in a separate batch.
TEST_F(StatisticsCollectorTests, stringMinMaxWhenFirstValueIsNeitherBound) {
  StringStatisticsCollector collector;
  std::vector<std::string_view> middleFirst = {"mmm", "aaa", "zzz"};
  collector.addValues(std::span<std::string_view>(middleFirst));

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin().value(), "aaa");
  EXPECT_EQ(stats->getMax().value(), "zzz");

  // A single-value batch that widens only the maximum.
  std::vector<std::string_view> single = {"zzzz"};
  collector.addValues(std::span<std::string_view>(single));
  EXPECT_EQ(stats->getMin().value(), "aaa");
  EXPECT_EQ(stats->getMax().value(), "zzzz");

  // The empty string sorts below every other value.
  std::vector<std::string_view> withEmpty = {"qqq", ""};
  collector.addValues(std::span<std::string_view>(withEmpty));
  EXPECT_EQ(stats->getMin().value(), "");
  EXPECT_EQ(stats->getMax().value(), "zzzz");

  EXPECT_EQ(
      dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(),
      9 + 4 + 3);
}

// Pins the batched scan against a trivial reference over randomized input:
// arbitrary bytes, mixed lengths, duplicates, empty strings, and batch
// boundaries that fall anywhere relative to the extremes. Any divergence in the
// bound-tracking shortcut shows up here rather than as a wrong file footer.
TEST_F(
    StatisticsCollectorTests,
    stringMinMaxMatchesReferenceOverRandomBatches) {
  std::mt19937 rng{20260830};
  std::uniform_int_distribution<size_t> lengthDist{0, 12};
  std::uniform_int_distribution<int> byteDist{0, 255};
  std::uniform_int_distribution<size_t> batchDist{1, 40};

  for (int trial = 0; trial < 200; ++trial) {
    StringStatisticsCollector collector;
    std::optional<std::string> expectedMin;
    std::optional<std::string> expectedMax;
    uint64_t expectedLogicalSize{0};

    const size_t numBatches = batchDist(rng);
    for (size_t batch = 0; batch < numBatches; ++batch) {
      std::vector<std::string> owned;
      const size_t batchSize = batchDist(rng);
      owned.reserve(batchSize);
      for (size_t i = 0; i < batchSize; ++i) {
        std::string value(lengthDist(rng), '\0');
        for (auto& byte : value) {
          byte = static_cast<char>(byteDist(rng));
        }
        owned.push_back(std::move(value));
      }

      for (const auto& value : owned) {
        expectedLogicalSize += value.size();
        if (!expectedMin.has_value() || value < *expectedMin) {
          expectedMin = value;
        }
        if (!expectedMax.has_value() || value > *expectedMax) {
          expectedMax = value;
        }
      }

      std::vector<std::string_view> views(owned.begin(), owned.end());
      collector.addValues(std::span<std::string_view>(views));
    }

    auto* stats = collector.getStatsView()->as<StringStatistics>();
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->getMin(), expectedMin) << "trial " << trial;
    EXPECT_EQ(stats->getMax(), expectedMax) << "trial " << trial;
    EXPECT_EQ(
        dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(),
        expectedLogicalSize)
        << "trial " << trial;
  }
}

// varchar columns carry arbitrary bytes, so bounds must be ordered and stored
// by length rather than by NUL termination.
TEST_F(StatisticsCollectorTests, stringMinMaxHandlesEmbeddedNulBytes) {
  using namespace std::string_view_literals;
  StringStatisticsCollector collector;

  // "a\0a" sorts above "a\0" purely on length, and both would collapse to "a"
  // under NUL-terminated comparison.
  std::vector<std::string_view> values = {"a\0a"sv, "a\0"sv, "a\0z"sv};
  collector.addValues(std::span<std::string_view>(values));

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin().value(), "a\0"sv);
  EXPECT_EQ(stats->getMax().value(), "a\0z"sv);
  EXPECT_EQ(stats->getMin().value().size(), 2);
  EXPECT_EQ(stats->getMax().value().size(), 3);
  EXPECT_EQ(
      dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(),
      3 + 2 + 3);
}

// An empty batch must leave the accumulator untouched rather than reading
// values.front().
TEST_F(StatisticsCollectorTests, stringMinMaxEmptyBatchIsANoOp) {
  StringStatisticsCollector collector;
  std::vector<std::string_view> empty;
  collector.addValues(std::span<std::string_view>(empty));

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_FALSE(stats->getMin().has_value());
  EXPECT_FALSE(stats->getMax().has_value());
  EXPECT_EQ(dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(), 0);

  std::vector<std::string_view> populated = {"kkk"};
  collector.addValues(std::span<std::string_view>(populated));
  collector.addValues(std::span<std::string_view>(empty));
  EXPECT_EQ(stats->getMin().value(), "kkk");
  EXPECT_EQ(stats->getMax().value(), "kkk");
  EXPECT_EQ(dynamic_cast<StatisticsCollector&>(collector).getLogicalSize(), 3);
}

// A shorter bound assigned over a longer one must not leave the previous
// contents behind, which is the failure mode of reusing the buffer.
TEST_F(
    StatisticsCollectorTests,
    stringMinMaxShrinkingBoundsDoNotRetainOldBytes) {
  StringStatisticsCollector collector;
  std::vector<std::string_view> wide = {"bbbbbbbbbb", "yyyyyyyyyy"};
  collector.addValues(std::span<std::string_view>(wide));

  std::vector<std::string_view> narrow = {"a", "z"};
  collector.addValues(std::span<std::string_view>(narrow));

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getMin().value(), "a");
  EXPECT_EQ(stats->getMax().value(), "z");
}

TEST_F(StatisticsCollectorTests, stringStatisticsCollectorMerge) {
  StringStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);
  dynamic_cast<StatisticsCollector&>(collector1).addLogicalSize(1000);

  StringStatisticsCollector collector2;
  dynamic_cast<StatisticsCollector&>(collector2).addCounts(200, 20);
  dynamic_cast<StatisticsCollector&>(collector2).addLogicalSize(2000);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 270); // (100-10) + (200-20) = 90 + 180
  EXPECT_EQ(sbPtr->getNullCount(), 30);
  EXPECT_EQ(sbPtr->getLogicalSize(), 3000);
}

TEST_F(StatisticsCollectorTests, stringStatisticsCollectorMergeWithDefault) {
  StringStatisticsCollector collector1;
  dynamic_cast<StatisticsCollector&>(collector1).addCounts(100, 10);

  StatisticsCollector collector2;
  collector2.addCounts(50, 5);

  collector1.merge(collector2);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(&collector1);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 135); // (100-10) + (50-5) = 90 + 45
  EXPECT_EQ(sbPtr->getNullCount(), 15);
}

TEST_F(StatisticsCollectorTests, wrapWithDeduplicatedStatisticsCollector) {
  auto baseCollector = std::make_unique<IntegralStatisticsCollector>();
  auto dedupCollector =
      DeduplicatedStatisticsCollector::wrap(std::move(baseCollector));

  EXPECT_NE(dedupCollector, nullptr);
  EXPECT_EQ(dedupCollector->getType(), StatType::DEDUPLICATED);
}

TEST_F(StatisticsCollectorTests, deduplicatedStatisticsCollectorBaseCollector) {
  auto baseCollector = std::make_unique<IntegralStatisticsCollector>();
  auto* basePtr = baseCollector.get();
  DeduplicatedStatisticsCollector dedupCollector(std::move(baseCollector));

  EXPECT_EQ(dedupCollector.getBaseCollector(), basePtr);
  dedupCollector.addCounts(100, 10);
  dedupCollector.addLogicalSize(1000);
  dedupCollector.addPhysicalSize(500);

  auto* sbPtr = dynamic_cast<StatisticsCollector*>(basePtr);
  ASSERT_NE(sbPtr, nullptr);
  EXPECT_EQ(sbPtr->getValueCount(), 90); // 100 - 10 = nonNullCount
  EXPECT_EQ(sbPtr->getNullCount(), 10);
  EXPECT_EQ(sbPtr->getLogicalSize(), 1000);
  EXPECT_EQ(sbPtr->getPhysicalSize(), 500);
}

TEST_F(StatisticsCollectorTests, recordDedupedStats) {
  auto baseCollector = std::make_unique<IntegralStatisticsCollector>();
  DeduplicatedStatisticsCollector dedupCollector(std::move(baseCollector));

  dedupCollector.recordDeduplicatedStats(80, 800);
  auto* dedupStats =
      dedupCollector.getStatsView()->as<DeduplicatedColumnStatistics>();
  ASSERT_NE(dedupStats, nullptr);
  EXPECT_EQ(dedupStats->getDedupedCount(), 80);
  EXPECT_EQ(dedupStats->getDedupedLogicalSize(), 800);

  dedupCollector.recordDeduplicatedStats(20, 200);
  EXPECT_EQ(dedupStats->getDedupedCount(), 100);
  EXPECT_EQ(dedupStats->getDedupedLogicalSize(), 1000);
}

TEST_F(StatisticsCollectorTests, deduplicatedStatisticsCollectorMerge) {
  auto baseCollector1 = std::make_unique<IntegralStatisticsCollector>();
  DeduplicatedStatisticsCollector dedupCollector1(std::move(baseCollector1));
  dedupCollector1.addCounts(100, 10);
  dedupCollector1.recordDeduplicatedStats(80, 800);

  auto baseCollector2 = std::make_unique<IntegralStatisticsCollector>();
  DeduplicatedStatisticsCollector dedupCollector2(std::move(baseCollector2));
  dedupCollector2.addCounts(200, 20);
  dedupCollector2.recordDeduplicatedStats(160, 1600);

  dedupCollector1.merge(dedupCollector2);

  auto* dedupStats =
      dedupCollector1.getStatsView()->as<DeduplicatedColumnStatistics>();
  ASSERT_NE(dedupStats, nullptr);
  EXPECT_EQ(dedupStats->getDedupedCount(), 240);
  EXPECT_EQ(dedupStats->getDedupedLogicalSize(), 2400);
  EXPECT_EQ(
      dedupCollector1.getBaseCollector()->getValueCount(),
      270); // (100-10) + (200-20)
  EXPECT_EQ(dedupCollector1.getBaseCollector()->getNullCount(), 30);
}
