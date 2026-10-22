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
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatsUtils.h"

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

TEST_F(ColumnStatisticsTests, clonePreservesScalarStatistics) {
  IntegralStatistics integralStat(100, 10, 1000, 500, -50, 150);
  auto integralClone = integralStat.clone();
  auto* clonedIntegralStat = integralClone->as<IntegralStatistics>();
  ASSERT_NE(clonedIntegralStat, nullptr);
  EXPECT_EQ(clonedIntegralStat->getValueCount(), 100);
  EXPECT_EQ(clonedIntegralStat->getNullCount(), 10);
  EXPECT_EQ(clonedIntegralStat->getLogicalSize(), 1000);
  EXPECT_EQ(clonedIntegralStat->getPhysicalSize(), 500);
  EXPECT_EQ(clonedIntegralStat->getMin(), -50);
  EXPECT_EQ(clonedIntegralStat->getMax(), 150);

  FloatingPointStatistics floatingPointStat(20, 2, 200, 80, -1.5, 9.5);
  auto floatingPointClone = floatingPointStat.clone();
  auto* clonedFloatingPointStat =
      floatingPointClone->as<FloatingPointStatistics>();
  ASSERT_NE(clonedFloatingPointStat, nullptr);
  EXPECT_EQ(clonedFloatingPointStat->getMin(), -1.5);
  EXPECT_EQ(clonedFloatingPointStat->getMax(), 9.5);

  StringStatistics stringStat(3, 0, 12, 8, "apple", "pear");
  auto stringClone = stringStat.clone();
  auto* clonedStringStat = stringClone->as<StringStatistics>();
  ASSERT_NE(clonedStringStat, nullptr);
  EXPECT_EQ(clonedStringStat->getMin(), "apple");
  EXPECT_EQ(clonedStringStat->getMax(), "pear");
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

TEST_F(StatisticsCollectorTests, resetClearsScalarStatistics) {
  IntegralStatisticsCollector collector;
  collector.addCounts(4, 1);
  std::vector<int64_t> values = {3, 7, -1};
  collector.addValues(std::span<int64_t>(values));
  ASSERT_NE(
      collector.getStatsView()->as<IntegralStatistics>()->getMin(),
      std::nullopt);

  collector.reset();

  auto* stats = collector.getStatsView()->as<IntegralStatistics>();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getValueCount(), 0);
  EXPECT_EQ(stats->getNullCount(), 0);
  EXPECT_EQ(stats->getLogicalSize(), 0);
  EXPECT_EQ(stats->getPhysicalSize(), 0);
  EXPECT_EQ(stats->getMin(), std::nullopt);
  EXPECT_EQ(stats->getMax(), std::nullopt);
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

// NDV / HyperLogLog tests. These require a memory pool to back the HLL
// accumulator, so they live in their own fixture. HLL is approximate; with
// 2048 buckets the standard error is ~2.3%, so cardinality assertions
// use a generous +/-10% band.
class NdvStatisticsCollectorTests : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::initializeMemoryManager(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("ndv_root");
    leafPool_ = rootPool_->addLeafChild("ndv_leaf");
  }

  // Asserts 'actual' is within +/-10% of 'expected' (min band of 1).
  static void expectCardinalityNear(int64_t expected, int64_t actual) {
    const auto tolerance = std::max<int64_t>(1, expected / 10);
    EXPECT_GE(actual, expected - tolerance);
    EXPECT_LE(actual, expected + tolerance);
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(NdvStatisticsCollectorTests, DisabledByDefault) {
  IntegralStatisticsCollector collector;
  EXPECT_FALSE(collector.hllEnabled());
  std::vector<int64_t> values = {1, 2, 3, 4, 5};
  collector.addValues(std::span<int64_t>(values));
  EXPECT_FALSE(collector.getStatsView()->getNdvSketch().has_value());
  EXPECT_FALSE(collector.getStatsView()->getNdvCardinality().has_value());
}

TEST_F(NdvStatisticsCollectorTests, IntegralCardinality) {
  IntegralStatisticsCollector collector;
  collector.enableHll(leafPool_.get());
  ASSERT_TRUE(collector.hllEnabled());

  constexpr int64_t kDistinct = 1000;
  std::vector<int64_t> values;
  values.reserve(kDistinct * 2);
  // Each distinct value appears twice; NDV must count it once.
  for (int64_t i = 0; i < kDistinct; ++i) {
    values.push_back(i);
    values.push_back(i);
  }
  collector.addValues(std::span<int64_t>(values));

  const auto* stats = collector.getStatsView();
  ASSERT_TRUE(stats->getNdvSketch().has_value());
  ASSERT_TRUE(stats->getNdvCardinality().has_value());
  expectCardinalityNear(kDistinct, stats->getNdvCardinality().value());
}

TEST_F(NdvStatisticsCollectorTests, IntegralCardinalityAcrossBatches) {
  IntegralStatisticsCollector collector;
  collector.enableHll(leafPool_.get());

  std::vector<int64_t> firstBatch;
  firstBatch.reserve(400);
  for (int64_t i = 0; i < 400; ++i) {
    firstBatch.push_back(i);
  }
  collector.addValues(std::span<int64_t>(firstBatch));

  // Overlapping batch: only 200 new distinct values (400..599).
  std::vector<int64_t> secondBatch;
  secondBatch.reserve(600 - 200);
  for (int64_t i = 200; i < 600; ++i) {
    secondBatch.push_back(i);
  }
  collector.addValues(std::span<int64_t>(secondBatch));

  const auto card = collector.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(card.has_value());
  expectCardinalityNear(600, card.value());
}

TEST_F(NdvStatisticsCollectorTests, FloatingPointCardinality) {
  FloatingPointStatisticsCollector collector;
  collector.enableHll(leafPool_.get());

  constexpr int64_t kDistinct = 800;
  std::vector<double> values;
  values.reserve(kDistinct);
  for (int64_t i = 0; i < kDistinct; ++i) {
    values.push_back(static_cast<double>(i) + 0.5);
  }
  collector.addValues(std::span<double>(values));

  const auto card = collector.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(card.has_value());
  expectCardinalityNear(kDistinct, card.value());
}

TEST_F(NdvStatisticsCollectorTests, StringCardinality) {
  StringStatisticsCollector collector;
  collector.enableHll(leafPool_.get());

  constexpr int64_t kDistinct = 500;
  std::vector<std::string> owner;
  owner.reserve(kDistinct);
  std::vector<std::string_view> values;
  values.reserve(kDistinct);
  for (int64_t i = 0; i < kDistinct; ++i) {
    owner.push_back("value_" + std::to_string(i));
    values.emplace_back(owner.back());
  }
  collector.addValues(std::span<std::string_view>(values));

  const auto card = collector.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(card.has_value());
  expectCardinalityNear(kDistinct, card.value());
}

TEST_F(NdvStatisticsCollectorTests, SketchSurvivesClone) {
  IntegralStatisticsCollector collector;
  collector.enableHll(leafPool_.get());
  std::vector<int64_t> values;
  values.reserve(300);
  for (int64_t i = 0; i < 300; ++i) {
    values.push_back(i);
  }
  collector.addValues(std::span<int64_t>(values));

  auto cloned = collector.getStatsView()->clone();
  ASSERT_TRUE(cloned->getNdvSketch().has_value());
  EXPECT_EQ(cloned->getNdvSketch(), collector.getStatsView()->getNdvSketch());
  ASSERT_TRUE(cloned->getNdvCardinality().has_value());
  expectCardinalityNear(300, cloned->getNdvCardinality().value());
}

// A deduplicated column reports the sketch of the statistics it wraps, the same
// way it reports their value and null counts.
TEST_F(NdvStatisticsCollectorTests, DeduplicatedStatsExposeBaseSketch) {
  auto baseCollector = std::make_unique<IntegralStatisticsCollector>();
  baseCollector->enableHll(leafPool_.get());
  std::vector<int64_t> values(250);
  std::iota(values.begin(), values.end(), 0);
  baseCollector->addValues(std::span<int64_t>(values));

  auto dedupCollector =
      DeduplicatedStatisticsCollector::wrap(std::move(baseCollector));
  const auto* dedupStats = dedupCollector->getStatsView();
  const auto& baseSketch =
      dedupCollector->getBaseCollector()->getStatsView()->getNdvSketch();

  ASSERT_TRUE(baseSketch.has_value());
  EXPECT_EQ(dedupStats->getNdvSketch(), baseSketch);
  ASSERT_TRUE(dedupStats->getNdvCardinality().has_value());
  expectCardinalityNear(250, dedupStats->getNdvCardinality().value());
  EXPECT_EQ(
      dedupStats->toCommonStatistics()->numDistinct(),
      dedupStats->getNdvCardinality());

  auto cloned = dedupStats->clone();
  EXPECT_EQ(cloned->getNdvSketch(), baseSketch);
}

// Setting a sketch on a deduplicated column stores it on the wrapped
// statistics, which is how the vectorized stats reader attaches it.
TEST_F(NdvStatisticsCollectorTests, DeduplicatedStatsSetSketchOnBase) {
  IntegralStatisticsCollector source;
  source.enableHll(leafPool_.get());
  std::vector<int64_t> values(120);
  std::iota(values.begin(), values.end(), 0);
  source.addValues(std::span<int64_t>(values));
  const auto& sketch = source.getStatsView()->getNdvSketch();
  ASSERT_TRUE(sketch.has_value());

  IntegralStatistics base;
  DeduplicatedColumnStatistics dedupStats(&base, 0, 0);
  dedupStats.setNdvSketch(sketch);

  EXPECT_EQ(base.getNdvSketch(), sketch);
  EXPECT_EQ(dedupStats.getNdvSketch(), sketch);
}

// The serialized sketch reflects every batch added, including batches added
// after an earlier read of the statistics.
TEST_F(NdvStatisticsCollectorTests, SketchReflectsBatchesAfterRead) {
  StringStatisticsCollector collector;
  collector.enableHll(leafPool_.get());
  std::vector<std::string> owner;
  owner.reserve(400);
  for (int64_t i = 0; i < 400; ++i) {
    owner.push_back("v" + std::to_string(i));
  }
  std::vector<std::string_view> first(owner.begin(), owner.begin() + 200);
  std::vector<std::string_view> second(owner.begin() + 200, owner.end());

  collector.addValues(std::span<std::string_view>(first));
  const auto afterFirst = collector.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(afterFirst.has_value());
  expectCardinalityNear(200, afterFirst.value());

  collector.addValues(std::span<std::string_view>(second));
  const auto afterSecond = collector.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(afterSecond.has_value());
  expectCardinalityNear(400, afterSecond.value());
}

TEST_F(NdvStatisticsCollectorTests, ResetClearsSketch) {
  IntegralStatisticsCollector collector;
  collector.enableHll(leafPool_.get());
  std::vector<int64_t> values = {1, 2, 3};
  collector.addValues(std::span<int64_t>(values));
  ASSERT_TRUE(collector.getStatsView()->getNdvSketch().has_value());

  collector.reset();
  EXPECT_FALSE(collector.getStatsView()->getNdvSketch().has_value());
  // Still enabled after reset: accumulating again produces a sketch.
  EXPECT_TRUE(collector.hllEnabled());
  collector.addValues(std::span<int64_t>(values));
  EXPECT_TRUE(collector.getStatsView()->getNdvSketch().has_value());
}

TEST_F(NdvStatisticsCollectorTests, MergeAcrossCollectors) {
  IntegralStatisticsCollector a;
  a.enableHll(leafPool_.get());
  std::vector<int64_t> aValues;
  aValues.reserve(500);
  for (int64_t i = 0; i < 500; ++i) {
    aValues.push_back(i);
  }
  a.addValues(std::span<int64_t>(aValues));

  IntegralStatisticsCollector b;
  b.enableHll(leafPool_.get());
  std::vector<int64_t> bValues;
  // 300 overlap (200..499) + 200 new (500..699) => union 700.
  for (int64_t i = 200; i < 700; ++i) {
    bValues.push_back(i);
  }
  b.addValues(std::span<int64_t>(bValues));

  a.merge(b);
  const auto card = a.getStatsView()->getNdvCardinality();
  ASSERT_TRUE(card.has_value());
  expectCardinalityNear(700, card.value());
}

TEST_F(NdvStatisticsCollectorTests, MergeNdvSketchesHelper) {
  IntegralStatisticsCollector a;
  a.enableHll(leafPool_.get());
  std::vector<int64_t> aValues;
  aValues.reserve(400);
  for (int64_t i = 0; i < 400; ++i) {
    aValues.push_back(i);
  }
  a.addValues(std::span<int64_t>(aValues));

  IntegralStatisticsCollector b;
  b.enableHll(leafPool_.get());
  std::vector<int64_t> bValues;
  bValues.reserve(900 - 400);
  for (int64_t i = 400; i < 900; ++i) {
    bValues.push_back(i);
  }
  b.addValues(std::span<int64_t>(bValues));

  const auto& lhs = a.getStatsView()->getNdvSketch();
  const auto& rhs = b.getStatsView()->getNdvSketch();
  ASSERT_TRUE(lhs.has_value());
  ASSERT_TRUE(rhs.has_value());

  auto merged = ColumnStatistics::mergeNdvSketches(lhs, rhs, leafPool_.get());
  ASSERT_TRUE(merged.has_value());

  ColumnStatistics mergedStats;
  mergedStats.setNdvSketch(std::move(merged));
  ASSERT_TRUE(mergedStats.getNdvCardinality().has_value());
  // Disjoint sets 0..399 and 400..899 => union 900.
  expectCardinalityNear(900, mergedStats.getNdvCardinality().value());

  // One-sided merges return the present operand.
  EXPECT_EQ(
      ColumnStatistics::mergeNdvSketches(lhs, std::nullopt, leafPool_.get()),
      lhs);
  EXPECT_EQ(
      ColumnStatistics::mergeNdvSketches(std::nullopt, rhs, leafPool_.get()),
      rhs);
  EXPECT_FALSE(
      ColumnStatistics::mergeNdvSketches(
          std::nullopt, std::nullopt, leafPool_.get())
          .has_value());
}

// Sketches must use the DWRF writers' 2048 buckets: HLLs with different bucket
// counts cannot be merged.
TEST_F(NdvStatisticsCollectorTests, SketchUses2048Buckets) {
  IntegralStatisticsCollector collector;
  collector.enableHll(leafPool_.get());
  std::vector<int64_t> values = {1, 2, 3};
  collector.addValues(std::span<int64_t>(values));

  const auto& sketch = collector.getStatsView()->getNdvSketch();
  ASSERT_TRUE(sketch.has_value());
  ASSERT_TRUE(velox::common::hll::SparseHlls::canDeserialize(sketch->data()));
  EXPECT_EQ(
      1 << velox::common::hll::SparseHlls::deserializeIndexBitLength(
          sketch->data()),
      2048);
}

// Integers hash as 64-bit longs (like DWRF), so the same values produce the
// same sketch regardless of column width.
TEST_F(NdvStatisticsCollectorTests, IntegralSketchIndependentOfWidth) {
  std::vector<int32_t> narrow;
  std::vector<int64_t> wide;
  for (int32_t i = -200; i < 300; ++i) {
    narrow.push_back(i);
    wide.push_back(i);
  }

  IntegralStatisticsCollector narrowCollector;
  narrowCollector.enableHll(leafPool_.get());
  narrowCollector.addValues(std::span<int32_t>(narrow));
  IntegralStatisticsCollector wideCollector;
  wideCollector.enableHll(leafPool_.get());
  wideCollector.addValues(std::span<int64_t>(wide));

  ASSERT_TRUE(narrowCollector.getStatsView()->getNdvSketch().has_value());
  EXPECT_EQ(
      narrowCollector.getStatsView()->getNdvSketch(),
      wideCollector.getStatsView()->getNdvSketch());
}

// NaNs with different payloads are one distinct value, as with Java's
// floatToIntBits / doubleToLongBits.
TEST_F(NdvStatisticsCollectorTests, NanPayloadsCountOnce) {
  std::vector<double> doubles = {
      std::numeric_limits<double>::quiet_NaN(),
      -std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::signaling_NaN(),
      1.0};
  FloatingPointStatisticsCollector doubleCollector;
  doubleCollector.enableHll(leafPool_.get());
  doubleCollector.addValues(std::span<double>(doubles));
  EXPECT_EQ(doubleCollector.getStatsView()->getNdvCardinality(), 2);

  std::vector<float> floats = {
      std::numeric_limits<float>::quiet_NaN(),
      -std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::signaling_NaN(),
      1.0f};
  FloatingPointStatisticsCollector floatCollector;
  floatCollector.enableHll(leafPool_.get());
  floatCollector.addValues(std::span<float>(floats));
  EXPECT_EQ(floatCollector.getStatsView()->getNdvCardinality(), 2);
}
