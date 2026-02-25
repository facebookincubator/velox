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
#include <thread>
#include <vector>
#include "velox/dwio/common/Statistics.h"

using namespace facebook::velox::dwio::common;
using facebook::velox::RuntimeMetric;

TEST(ColumnStatsTest, BasicOperations) {
  // Default construction
  ColumnStats defaultStats;
  EXPECT_EQ(defaultStats.nodeId, 0);
  EXPECT_TRUE(defaultStats.columnName.empty());
  EXPECT_EQ(defaultStats.decompressCpuTimeNs.load(), 0);
  EXPECT_EQ(defaultStats.numDecompressCalls.load(), 0);

  // Construction with id and name
  ColumnStats stats(1, "test_column");
  EXPECT_EQ(stats.nodeId, 1);
  EXPECT_EQ(stats.columnName, "test_column");

  // Add decompress time accumulates correctly
  stats.addDecompressTime(5000);
  stats.addDecompressTime(3000);
  EXPECT_EQ(stats.decompressCpuTimeNs.load(), 8000);
  EXPECT_EQ(stats.numDecompressCalls.load(), 2);
}

TEST(ColumnStatsTest, ConcurrentAccess) {
  auto stats = std::make_shared<ColumnStats>(1, "test");
  constexpr int kNumThreads = 4;
  constexpr int kIterationsPerThread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&stats]() {
      for (int j = 0; j < kIterationsPerThread; ++j) {
        stats->addDecompressTime(5);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(
      stats->decompressCpuTimeNs.load(),
      kNumThreads * kIterationsPerThread * 5);
  EXPECT_EQ(
      stats->numDecompressCalls.load(), kNumThreads * kIterationsPerThread);
}

TEST(ColumnReaderStatisticsTest, GetOrCreateColumnTimingStats) {
  ColumnReaderStatistics stats;

  // Returns nullptr when disabled
  stats.collectColumnStats = false;
  EXPECT_EQ(stats.getOrCreateColumnStats(1, "col1"), nullptr);

  // Returns valid stats when enabled
  stats.collectColumnStats = true;
  auto result = stats.getOrCreateColumnStats(1, "col1");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->nodeId, 1);
  EXPECT_EQ(result->columnName, "col1");

  // Returns same instance for same nodeId
  EXPECT_EQ(stats.getOrCreateColumnStats(1, "col1").get(), result.get());

  // Returns different instance for different nodeId
  auto result2 = stats.getOrCreateColumnStats(2, "col2");
  EXPECT_NE(result2.get(), result.get());
}

TEST(ColumnReaderStatisticsTest, ExportToRuntimeMetrics) {
  ColumnReaderStatistics stats;
  stats.collectColumnStats = true;

  // Empty stats produces empty result
  std::unordered_map<std::string, RuntimeMetric> result;
  stats.exportToRuntimeMetrics(result);
  EXPECT_TRUE(result.empty());

  // Add timing data
  auto col1 = stats.getOrCreateColumnStats(1, "col1");
  col1->addDecompressTime(5000);

  auto col2 = stats.getOrCreateColumnStats(42); // No name, uses nodeId
  col2->addDecompressTime(3000);

  result.clear();
  stats.exportToRuntimeMetrics(result);

  EXPECT_EQ(result["col1.decompressCpuTimeNs"].sum, 5000);
  EXPECT_EQ(result["col1.numDecompressCalls"].sum, 1);
  EXPECT_EQ(result["column_42.decompressCpuTimeNs"].sum, 3000);
  EXPECT_EQ(result["column_42.numDecompressCalls"].sum, 1);

  // Zero values are not included
  stats.getOrCreateColumnStats(99, "empty_col");
  result.clear();
  stats.exportToRuntimeMetrics(result);
  EXPECT_EQ(result.count("empty_col.decompressCpuTimeNs"), 0);
}

TEST(RuntimeStatisticsTest, ToRuntimeMetricMap) {
  RuntimeStatistics stats;

  // Empty stats produces empty result
  EXPECT_TRUE(stats.toRuntimeMetricMap().empty());

  // Set various stats
  stats.skippedSplits = 5;
  stats.processedSplits = 15;
  stats.skippedStrides = 10;
  stats.processedStrides = 30;
  stats.numStripes = 4;
  stats.columnReaderStatistics.flattenStringDictionaryValues = 1000;

  // Add per-column stats
  stats.columnReaderStatistics.collectColumnStats = true;
  auto colStats =
      stats.columnReaderStatistics.getOrCreateColumnStats(1, "col1");
  colStats->addDecompressTime(5000);

  auto result = stats.toRuntimeMetricMap();

  EXPECT_EQ(result["skippedSplits"].sum, 5);
  EXPECT_EQ(result["processedSplits"].sum, 15);
  EXPECT_EQ(result["skippedStrides"].sum, 10);
  EXPECT_EQ(result["processedStrides"].sum, 30);
  EXPECT_EQ(result["numStripes"].sum, 4);
  EXPECT_EQ(result["flattenStringDictionaryValues"].sum, 1000);
  EXPECT_EQ(result["col1.decompressCpuTimeNs"].sum, 5000);
  EXPECT_EQ(result["col1.numDecompressCalls"].sum, 1);
}

TEST(ColumnReaderStatisticsConcurrencyTest, ConcurrentGetOrCreate) {
  ColumnReaderStatistics stats;
  stats.collectColumnStats = true;
  constexpr int kNumThreads = 4;
  constexpr int kNumColumns = 10;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&stats]() {
      for (uint32_t colId = 0; colId < kNumColumns; ++colId) {
        auto colStats =
            stats.getOrCreateColumnStats(colId, fmt::format("col{}", colId));
        colStats->addDecompressTime(100);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::unordered_map<std::string, RuntimeMetric> result;
  stats.exportToRuntimeMetrics(result);

  for (uint32_t colId = 0; colId < kNumColumns; ++colId) {
    auto key = fmt::format("col{}.decompressCpuTimeNs", colId);
    EXPECT_EQ(result[key].sum, kNumThreads * 100);
  }
}
