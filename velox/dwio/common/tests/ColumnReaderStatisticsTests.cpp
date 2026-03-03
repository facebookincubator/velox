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

TEST(IoCounterTest, BasicOperations) {
  facebook::velox::io::IoCounter counter;

  EXPECT_EQ(counter.sum(), 0);
  EXPECT_EQ(counter.count(), 0);

  counter.increment(5'000);
  counter.increment(3'000);

  EXPECT_EQ(counter.sum(), 8'000);
  EXPECT_EQ(counter.count(), 2);
  EXPECT_EQ(counter.min(), 3'000);
  EXPECT_EQ(counter.max(), 5'000);
}

TEST(IoCounterTest, ConcurrentAccess) {
  facebook::velox::io::IoCounter counter;
  constexpr int kNumThreads = 4;
  constexpr int kIterationsPerThread = 1'000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&counter]() {
      for (int j = 0; j < kIterationsPerThread; ++j) {
        counter.increment(5);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(counter.sum(), kNumThreads * kIterationsPerThread * 5);
  EXPECT_EQ(counter.count(), kNumThreads * kIterationsPerThread);
}

TEST(ColumnMetricsSetTest, GetOrCreate) {
  ColumnMetricsSet metricsSet;

  auto* result = metricsSet.getOrCreate(1);
  ASSERT_NE(result, nullptr);

  // Returns same instance for same nodeId.
  EXPECT_EQ(metricsSet.getOrCreate(1), result);

  // Returns different instance for different nodeId.
  auto* result2 = metricsSet.getOrCreate(2);
  EXPECT_NE(result2, result);
}

TEST(ColumnMetricsSetTest, ToRuntimeMetrics) {
  ColumnMetricsSet metricsSet;

  // Empty stats produces empty result.
  std::unordered_map<std::string, RuntimeMetric> result;
  metricsSet.toRuntimeMetrics(result);
  EXPECT_TRUE(result.empty());

  // Add timing data.
  auto* col1 = metricsSet.getOrCreate(1);
  col1->decompressCPUTimeNanos.increment(5'000);
  col1->decompressCPUTimeNanos.increment(3'000);

  auto* col2 = metricsSet.getOrCreate(42);
  col2->decompressCPUTimeNanos.increment(2'000);

  result.clear();
  metricsSet.toRuntimeMetrics(result);

  // RuntimeMetric has sum/count/min/max, so we only need one metric per column.
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 8'000);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].count, 2);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].min, 3'000);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].max, 5'000);
  EXPECT_EQ(result["column_42.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_42.decompressCPUTimeNanos"].count, 1);

  // Zero values are not included.
  metricsSet.getOrCreate(99);
  result.clear();
  metricsSet.toRuntimeMetrics(result);
  EXPECT_EQ(result.count("column_99.decompressCPUTimeNanos"), 0);
}

TEST(RuntimeStatisticsTest, ToRuntimeMetricMap) {
  RuntimeStatistics stats;

  // Empty stats produces empty result.
  EXPECT_TRUE(stats.toRuntimeMetricMap().empty());

  // Set various stats.
  stats.skippedSplits = 5;
  stats.processedSplits = 15;
  stats.skippedStrides = 10;
  stats.processedStrides = 30;
  stats.numStripes = 4;
  stats.columnReaderStats.flattenStringDictionaryValues = 1'000;

  // Add per-column stats.
  stats.columnReaderStats.columnMetricsSet.emplace();
  auto* colMetrics = stats.columnReaderStats.columnMetricsSet->getOrCreate(1);
  colMetrics->decompressCPUTimeNanos.increment(5'000);

  auto result = stats.toRuntimeMetricMap();

  EXPECT_EQ(result["skippedSplits"].sum, 5);
  EXPECT_EQ(result["processedSplits"].sum, 15);
  EXPECT_EQ(result["skippedStrides"].sum, 10);
  EXPECT_EQ(result["processedStrides"].sum, 30);
  EXPECT_EQ(result["numStripes"].sum, 4);
  EXPECT_EQ(result["flattenStringDictionaryValues"].sum, 1'000);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 5'000);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].count, 1);
}

TEST(ColumnMetricsSetConcurrencyTest, ConcurrentGetOrCreate) {
  ColumnMetricsSet metricsSet;
  constexpr int kNumThreads = 4;
  constexpr int kNumColumns = 10;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&metricsSet]() {
      for (uint32_t colId = 0; colId < kNumColumns; ++colId) {
        auto* colMetrics = metricsSet.getOrCreate(colId);
        colMetrics->decompressCPUTimeNanos.increment(100);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::unordered_map<std::string, RuntimeMetric> result;
  metricsSet.toRuntimeMetrics(result);

  for (uint32_t colId = 0; colId < kNumColumns; ++colId) {
    auto key = fmt::format("column_{}.decompressCPUTimeNanos", colId);
    EXPECT_EQ(result[key].sum, kNumThreads * 100);
    EXPECT_EQ(result[key].count, kNumThreads);
  }
}

TEST(IoCounterTest, MergeStats) {
  facebook::velox::io::IoCounter counter1;
  counter1.increment(5'000);
  counter1.increment(3'000);

  facebook::velox::io::IoCounter counter2;
  counter2.increment(2'000);

  counter1.merge(counter2);

  EXPECT_EQ(counter1.sum(), 10'000);
  EXPECT_EQ(counter1.count(), 3);
}

TEST(ColumnMetricsSetTest, MergeFromWithOverlappingNodeIds) {
  ColumnMetricsSet src;

  auto* srcCol1 = src.getOrCreate(1);
  srcCol1->decompressCPUTimeNanos.increment(5'000);
  srcCol1->decompressCPUTimeNanos.increment(3'000);

  auto* srcCol2 = src.getOrCreate(2);
  srcCol2->decompressCPUTimeNanos.increment(2'000);

  ColumnMetricsSet dst;

  auto* dstCol1 = dst.getOrCreate(1);
  dstCol1->decompressCPUTimeNanos.increment(1'000);

  dst.mergeFrom(src);

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.toRuntimeMetrics(result);

  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 9'000);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].count, 3);
  EXPECT_EQ(result["column_2.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_2.decompressCPUTimeNanos"].count, 1);
}

TEST(ColumnMetricsSetTest, MergeFromWithDisjointNodeIds) {
  ColumnMetricsSet src;

  auto* srcCol3 = src.getOrCreate(3);
  srcCol3->decompressCPUTimeNanos.increment(3'000);

  auto* srcCol4 = src.getOrCreate(4);
  srcCol4->decompressCPUTimeNanos.increment(4'000);

  ColumnMetricsSet dst;

  auto* dstCol1 = dst.getOrCreate(1);
  dstCol1->decompressCPUTimeNanos.increment(1'000);

  auto* dstCol2 = dst.getOrCreate(2);
  dstCol2->decompressCPUTimeNanos.increment(2'000);

  dst.mergeFrom(src);

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.toRuntimeMetrics(result);

  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 1'000);
  EXPECT_EQ(result["column_2.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_3.decompressCPUTimeNanos"].sum, 3'000);
  EXPECT_EQ(result["column_4.decompressCPUTimeNanos"].sum, 4'000);
}

TEST(ColumnMetricsSetTest, MergeFromEmpty) {
  ColumnMetricsSet nonEmpty;

  auto* col = nonEmpty.getOrCreate(1);
  col->decompressCPUTimeNanos.increment(5'000);

  ColumnMetricsSet empty;

  nonEmpty.mergeFrom(empty);

  std::unordered_map<std::string, RuntimeMetric> result;
  nonEmpty.toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 5'000);

  ColumnMetricsSet empty2;
  empty2.mergeFrom(nonEmpty);

  result.clear();
  empty2.toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 5'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromWithColumnMetrics) {
  ColumnReaderStatistics src;
  src.flattenStringDictionaryValues = 100;
  src.columnMetricsSet.emplace();
  src.columnMetricsSet->getOrCreate(1)->decompressCPUTimeNanos.increment(1'000);

  // Merge into stats without columnMetricsSet - creates and populates it.
  ColumnReaderStatistics dst;
  dst.flattenStringDictionaryValues = 50;
  dst.mergeFrom(src);

  EXPECT_EQ(dst.flattenStringDictionaryValues, 150);
  ASSERT_TRUE(dst.columnMetricsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.columnMetricsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 1'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromBothWithColumnMetrics) {
  ColumnReaderStatistics src;
  src.flattenStringDictionaryValues = 100;
  src.columnMetricsSet.emplace();
  src.columnMetricsSet->getOrCreate(1)->decompressCPUTimeNanos.increment(1'000);

  ColumnReaderStatistics dst;
  dst.flattenStringDictionaryValues = 50;
  dst.columnMetricsSet.emplace();
  dst.columnMetricsSet->getOrCreate(1)->decompressCPUTimeNanos.increment(2'000);

  dst.mergeFrom(src);

  EXPECT_EQ(dst.flattenStringDictionaryValues, 150);
  ASSERT_TRUE(dst.columnMetricsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.columnMetricsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 3'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromWithoutColumnMetrics) {
  ColumnReaderStatistics src;
  src.flattenStringDictionaryValues = 100;

  ColumnReaderStatistics dst;
  dst.flattenStringDictionaryValues = 50;
  dst.columnMetricsSet.emplace();
  dst.columnMetricsSet->getOrCreate(1)->decompressCPUTimeNanos.increment(1'000);

  dst.mergeFrom(src);

  EXPECT_EQ(dst.flattenStringDictionaryValues, 150);
  ASSERT_TRUE(dst.columnMetricsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.columnMetricsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.decompressCPUTimeNanos"].sum, 1'000);
}
