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
#include "velox/type/Type.h"

using namespace facebook::velox::dwio::common;
using facebook::velox::RuntimeMetric;
using facebook::velox::TypeKind;

namespace {

constexpr std::string_view kExampleFormatMetricName = "exampleFormatMetric";

constexpr std::pair<std::string_view, facebook::velox::RuntimeCounter::Unit>
    kExampleFormatMetric = {
        kExampleFormatMetricName,
        facebook::velox::RuntimeCounter::Unit::kNone};

constexpr auto kExampleFormat = FileFormat::PARQUET;
constexpr std::string_view kExampleQualifiedFormatMetricName =
    "parquet.exampleFormatMetric";

} // namespace

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

TEST(DecodingStatsSetTest, GetOrCreate) {
  DecodingStatsSet statsSet;

  auto* result = statsSet.getOrCreate(1);
  ASSERT_NE(result, nullptr);

  // Returns same instance for same nodeId.
  EXPECT_EQ(statsSet.getOrCreate(1), result);

  // Returns different instance for different nodeId.
  auto* result2 = statsSet.getOrCreate(2);
  EXPECT_NE(result2, result);
}

TEST(DecodingStatsSetTest, GetOrCreateWithTypeKind) {
  DecodingStatsSet statsSet;

  // Pass type when calling getOrCreate.
  auto* result = statsSet.getOrCreate(1, TypeKind::BIGINT);
  ASSERT_NE(result, nullptr);
  result->decompressCPUTimeNanos.increment(1'000);

  // Returns same instance for same nodeId.
  auto* result2 = statsSet.getOrCreate(1);
  EXPECT_EQ(result2, result);

  // Different nodeId with different type.
  auto* result3 = statsSet.getOrCreate(2, TypeKind::VARCHAR);
  EXPECT_NE(result3, result);
  result3->decompressCPUTimeNanos.increment(2'000);

  // Verify types are used in toRuntimeMetrics.
  std::unordered_map<std::string, RuntimeMetric> metrics;
  statsSet.toRuntimeMetrics(metrics);
  EXPECT_EQ(metrics["column_1.BIGINT.decompressCPUTimeNanos"].sum, 1'000);
  EXPECT_EQ(metrics["column_2.VARCHAR.decompressCPUTimeNanos"].sum, 2'000);
}

TEST(DecodingStatsSetTest, ToRuntimeMetrics) {
  DecodingStatsSet statsSet;

  // Empty stats produces empty result.
  std::unordered_map<std::string, RuntimeMetric> result;
  statsSet.toRuntimeMetrics(result);
  EXPECT_TRUE(result.empty());

  // Add timing data with type information.
  auto* col1 = statsSet.getOrCreate(1, TypeKind::BIGINT);
  col1->decompressCPUTimeNanos.increment(5'000);
  col1->decompressCPUTimeNanos.increment(3'000);

  auto* col2 = statsSet.getOrCreate(42, TypeKind::VARCHAR);
  col2->decompressCPUTimeNanos.increment(2'000);

  // Create a column with type but no data.
  statsSet.getOrCreate(99, TypeKind::DOUBLE);

  result.clear();
  statsSet.toRuntimeMetrics(result);

  // RuntimeMetric has sum/count/min/max, metric name includes type.
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 8'000);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].count, 2);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].min, 3'000);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].max, 5'000);
  EXPECT_EQ(result["column_42.VARCHAR.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_42.VARCHAR.decompressCPUTimeNanos"].count, 1);

  // Zero values are not included.
  statsSet.getOrCreate(99);
  result.clear();
  statsSet.toRuntimeMetrics(result);
  EXPECT_EQ(result.count("column_99.DOUBLE.decompressCPUTimeNanos"), 0);
}

TEST(DecodingStatsSetTest, ToRuntimeMetricsWithInvalidType) {
  DecodingStatsSet statsSet;

  // Add timing data without type information (INVALID type).
  auto* col1 = statsSet.getOrCreate(1);
  col1->decompressCPUTimeNanos.increment(5'000);

  std::unordered_map<std::string, RuntimeMetric> result;
  statsSet.toRuntimeMetrics(result);

  // Should use INVALID as type name.
  EXPECT_EQ(result["column_1.INVALID.decompressCPUTimeNanos"].sum, 5'000);
}

TEST(DecodingStatsSetTest, ToRuntimeMetricsWithDecodeTime) {
  DecodingStatsSet statsSet;

  // Add both decompress and decode timing data.
  auto* col1 = statsSet.getOrCreate(1, TypeKind::BIGINT);
  col1->decompressCPUTimeNanos.increment(5'000);
  col1->decodeCPUTimeNanos.increment(10'000);
  col1->decodeCPUTimeNanos.increment(8'000);

  auto* col2 = statsSet.getOrCreate(2, TypeKind::VARCHAR);
  col2->decodeCPUTimeNanos.increment(3'000);

  std::unordered_map<std::string, RuntimeMetric> result;
  statsSet.toRuntimeMetrics(result);

  // Column 1 has both decompress and decode metrics.
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 5'000);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].sum, 18'000);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].count, 2);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].min, 8'000);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].max, 10'000);

  // Column 2 has only decode metrics.
  EXPECT_EQ(result.count("column_2.VARCHAR.decompressCPUTimeNanos"), 0);
  EXPECT_EQ(result["column_2.VARCHAR.decodeCPUTimeNanos"].sum, 3'000);
  EXPECT_EQ(result["column_2.VARCHAR.decodeCPUTimeNanos"].count, 1);
}

TEST(RuntimeStatisticsTest, ToRuntimeMetricMap) {
  RuntimeStatistics stats;
  stats.columnReaderStats = ColumnReaderStatistics{kExampleFormat};

  // Empty stats produces empty result.
  EXPECT_TRUE(stats.toRuntimeMetricMap().empty());

  // Set various stats.
  stats.skippedSplits = 5;
  stats.processedSplits = 15;
  stats.skippedStrides = 10;
  stats.processedStrides = 30;
  stats.numStripes = 4;
  stats.columnReaderStats.accumulateFormatStat(kExampleFormatMetric, 1'000);

  // Add per-column stats with type.
  stats.columnReaderStats.decodingStatsSet.emplace();
  auto* colStats = stats.columnReaderStats.decodingStatsSet->getOrCreate(
      1, TypeKind::BIGINT);
  colStats->decompressCPUTimeNanos.increment(5'000);
  colStats->decodeCPUTimeNanos.increment(12'000);

  auto result = stats.toRuntimeMetricMap();

  EXPECT_EQ(result["skippedSplits"].sum, 5);
  EXPECT_EQ(result["processedSplits"].sum, 15);
  EXPECT_EQ(result["skippedStrides"].sum, 10);
  EXPECT_EQ(result["processedStrides"].sum, 30);
  EXPECT_EQ(result["numStripes"].sum, 4);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 5'000);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].count, 1);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].sum, 12'000);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].count, 1);
  EXPECT_EQ(result[std::string(kExampleQualifiedFormatMetricName)].sum, 1'000);
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

TEST(DecodingStatsSetTest, MergeFromWithOverlappingNodeIds) {
  DecodingStatsSet src;
  auto* srcCol1 = src.getOrCreate(1, TypeKind::BIGINT);
  srcCol1->decompressCPUTimeNanos.increment(5'000);
  srcCol1->decompressCPUTimeNanos.increment(3'000);
  srcCol1->decodeCPUTimeNanos.increment(10'000);

  auto* srcCol2 = src.getOrCreate(2, TypeKind::VARCHAR);
  srcCol2->decompressCPUTimeNanos.increment(2'000);
  srcCol2->decodeCPUTimeNanos.increment(4'000);

  DecodingStatsSet dst;
  auto* dstCol1 = dst.getOrCreate(1, TypeKind::BIGINT);
  dstCol1->decompressCPUTimeNanos.increment(1'000);
  dstCol1->decodeCPUTimeNanos.increment(6'000);

  dst.mergeFrom(src);

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.toRuntimeMetrics(result);

  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 9'000);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].count, 3);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].sum, 16'000);
  EXPECT_EQ(result["column_1.BIGINT.decodeCPUTimeNanos"].count, 2);
  EXPECT_EQ(result["column_2.VARCHAR.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_2.VARCHAR.decompressCPUTimeNanos"].count, 1);
  EXPECT_EQ(result["column_2.VARCHAR.decodeCPUTimeNanos"].sum, 4'000);
  EXPECT_EQ(result["column_2.VARCHAR.decodeCPUTimeNanos"].count, 1);
}

TEST(DecodingStatsSetTest, MergeFromWithDisjointNodeIds) {
  DecodingStatsSet src;
  auto* srcCol3 = src.getOrCreate(3, TypeKind::DOUBLE);
  srcCol3->decompressCPUTimeNanos.increment(3'000);

  auto* srcCol4 = src.getOrCreate(4, TypeKind::BOOLEAN);
  srcCol4->decompressCPUTimeNanos.increment(4'000);

  DecodingStatsSet dst;
  auto* dstCol1 = dst.getOrCreate(1, TypeKind::BIGINT);
  dstCol1->decompressCPUTimeNanos.increment(1'000);

  auto* dstCol2 = dst.getOrCreate(2, TypeKind::VARCHAR);
  dstCol2->decompressCPUTimeNanos.increment(2'000);

  dst.mergeFrom(src);

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.toRuntimeMetrics(result);

  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 1'000);
  EXPECT_EQ(result["column_2.VARCHAR.decompressCPUTimeNanos"].sum, 2'000);
  EXPECT_EQ(result["column_3.DOUBLE.decompressCPUTimeNanos"].sum, 3'000);
  EXPECT_EQ(result["column_4.BOOLEAN.decompressCPUTimeNanos"].sum, 4'000);
}

TEST(DecodingStatsSetTest, MergeFromEmpty) {
  DecodingStatsSet nonEmpty;
  auto* col = nonEmpty.getOrCreate(1, TypeKind::BIGINT);
  col->decompressCPUTimeNanos.increment(5'000);

  DecodingStatsSet empty;

  nonEmpty.mergeFrom(empty);

  std::unordered_map<std::string, RuntimeMetric> result;
  nonEmpty.toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 5'000);

  DecodingStatsSet empty2;
  empty2.mergeFrom(nonEmpty);

  result.clear();
  empty2.toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 5'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromWithDecodingStats) {
  ColumnReaderStatistics src{kExampleFormat};
  src.accumulateFormatStat(kExampleFormatMetric, 100);
  src.decodingStatsSet.emplace();
  src.decodingStatsSet->getOrCreate(1, TypeKind::BIGINT)
      ->decompressCPUTimeNanos.increment(1'000);

  // Merge into stats without decodingStatsSet - creates and populates it.
  ColumnReaderStatistics dst{kExampleFormat};
  dst.accumulateFormatStat(kExampleFormatMetric, 50);
  dst.mergeFrom(src);

  ASSERT_NE(
      dst.formatStats.find(std::string(kExampleQualifiedFormatMetricName)),
      dst.formatStats.end());
  EXPECT_EQ(
      dst.formatStats.at(std::string(kExampleQualifiedFormatMetricName)).sum,
      150);
  ASSERT_TRUE(dst.decodingStatsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.decodingStatsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 1'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromBothWithDecodingStats) {
  ColumnReaderStatistics src{kExampleFormat};
  src.accumulateFormatStat(kExampleFormatMetric, 100);
  src.decodingStatsSet.emplace();
  src.decodingStatsSet->getOrCreate(1, TypeKind::BIGINT)
      ->decompressCPUTimeNanos.increment(1'000);

  ColumnReaderStatistics dst{kExampleFormat};
  dst.accumulateFormatStat(kExampleFormatMetric, 50);
  dst.decodingStatsSet.emplace();
  dst.decodingStatsSet->getOrCreate(1, TypeKind::BIGINT)
      ->decompressCPUTimeNanos.increment(2'000);

  dst.mergeFrom(src);

  ASSERT_NE(
      dst.formatStats.find(std::string(kExampleQualifiedFormatMetricName)),
      dst.formatStats.end());
  EXPECT_EQ(
      dst.formatStats.at(std::string(kExampleQualifiedFormatMetricName)).sum,
      150);
  ASSERT_TRUE(dst.decodingStatsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.decodingStatsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 3'000);
}

TEST(WithDecompressStatsTest, NonVoidWithCounter) {
  facebook::velox::io::IoCounter counter;
  int result = withDecompressStats(&counter, [] { return 42; });
  EXPECT_EQ(result, 42);
  EXPECT_EQ(counter.count(), 1);
}

TEST(WithDecompressStatsTest, NullCounter) {
  int result = withDecompressStats(nullptr, [] { return 7; });
  EXPECT_EQ(result, 7);

  int sideEffect = 0;
  withDecompressStats(nullptr, [&] { sideEffect = 3; });
  EXPECT_EQ(sideEffect, 3);
}

TEST(ColumnReaderStatisticsTest, MergeFromWithoutDecodingStats) {
  ColumnReaderStatistics src{kExampleFormat};
  src.accumulateFormatStat(kExampleFormatMetric, 100);

  ColumnReaderStatistics dst{kExampleFormat};
  dst.accumulateFormatStat(kExampleFormatMetric, 50);
  dst.decodingStatsSet.emplace();
  dst.decodingStatsSet->getOrCreate(1, TypeKind::BIGINT)
      ->decompressCPUTimeNanos.increment(1'000);

  dst.mergeFrom(src);

  ASSERT_NE(
      dst.formatStats.find(std::string(kExampleQualifiedFormatMetricName)),
      dst.formatStats.end());
  EXPECT_EQ(
      dst.formatStats.at(std::string(kExampleQualifiedFormatMetricName)).sum,
      150);
  ASSERT_TRUE(dst.decodingStatsSet.has_value());

  std::unordered_map<std::string, RuntimeMetric> result;
  dst.decodingStatsSet->toRuntimeMetrics(result);
  EXPECT_EQ(result["column_1.BIGINT.decompressCPUTimeNanos"].sum, 1'000);
}
