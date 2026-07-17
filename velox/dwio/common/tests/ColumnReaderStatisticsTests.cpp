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

TEST(DecodingStatsTest, ToRuntimeMetrics) {
  DecodingStats stats;
  stats.decompressCPUTimeNanos.increment(3'000);
  stats.decompressCPUTimeNanos.increment(5'000);
  stats.decodeCPUTimeNanos.increment(7'000);
  stats.decodeCPUTimeNanos.increment(11'000);

  std::unordered_map<std::string, RuntimeMetric> result;
  stats.toRuntimeMetrics("parquet.column_1.BIGINT", result);

  const auto& decompress =
      result.at("parquet.column_1.BIGINT.decompressCPUTimeNanos");
  EXPECT_EQ(decompress.sum, 8'000);
  EXPECT_EQ(decompress.count, 2);
  EXPECT_EQ(decompress.min, 3'000);
  EXPECT_EQ(decompress.max, 5'000);
  EXPECT_EQ(decompress.unit, facebook::velox::RuntimeCounter::Unit::kNanos);

  const auto& decode = result.at("parquet.column_1.BIGINT.decodeCPUTimeNanos");
  EXPECT_EQ(decode.sum, 18'000);
  EXPECT_EQ(decode.count, 2);
  EXPECT_EQ(decode.min, 7'000);
  EXPECT_EQ(decode.max, 11'000);
  EXPECT_EQ(decode.unit, facebook::velox::RuntimeCounter::Unit::kNanos);
}

TEST(SplitStatisticsTest, ColumnStats) {
  SplitStatistics stats{kExampleFormat};
  auto& column = stats.getOrCreateColumnStats(1, TypeKind::BIGINT);
  EXPECT_EQ(&stats.getOrCreateColumnStats(1, TypeKind::BIGINT), &column);
  EXPECT_NE(&stats.getOrCreateColumnStats(2, TypeKind::VARCHAR), &column);
}

TEST(RuntimeStatisticsTest, ExportWithoutColumnCpuMetrics) {
  const auto schema = TypeWithId::create(
      facebook::velox::ROW(
          {"bigint", "varchar"},
          {facebook::velox::BIGINT(), facebook::velox::VARCHAR()}));
  const RowReaderOptions options;
  ASSERT_FALSE(options.collectColumnCpuMetrics());

  SplitStatistics splitStats{kExampleFormat};
  splitStats.initColumnStatsCollection(*schema, options);
  ASSERT_EQ(splitStats.columnStats.size(), 3);
  for (const auto& [nodeId, stats] : splitStats.columnStats) {
    EXPECT_FALSE(stats.decodingStats.has_value()) << nodeId;
  }

  RuntimeStatistics stats;
  stats.mergeFrom(splitStats);
  EXPECT_TRUE(stats.toRuntimeMetricMap().empty());
}

TEST(RuntimeStatisticsTest, ToRuntimeMetricMap) {
  RuntimeStatistics stats;
  SplitStatistics splitStats{kExampleFormat};

  // Empty stats produces empty result.
  EXPECT_TRUE(stats.toRuntimeMetricMap().empty());

  // Set various stats.
  stats.skippedSplits = 5;
  stats.processedSplits = 15;
  stats.skippedStrides = 10;
  stats.processedStrides = 30;
  stats.numStripes = 4;
  splitStats.getOrCreateColumnStats(1, TypeKind::BIGINT)
      .accumulateStat(kExampleFormatMetric, 1'000);
  splitStats.getOrCreateColumnStats(2, TypeKind::VARCHAR)
      .accumulateStat(kExampleFormatMetric, 2'000);

  // Add per-column stats with type.
  splitStats.getOrCreateColumnStats(1, TypeKind::BIGINT)
      .decodingStats.emplace();
  auto* colStats =
      &*splitStats.getOrCreateColumnStats(1, TypeKind::BIGINT).decodingStats;
  colStats->decompressCPUTimeNanos.increment(5'000);
  colStats->decodeCPUTimeNanos.increment(12'000);
  splitStats.getOrCreateColumnStats(2, TypeKind::VARCHAR)
      .decodingStats.emplace();
  auto* col2Stats =
      &*splitStats.getOrCreateColumnStats(2, TypeKind::VARCHAR).decodingStats;
  col2Stats->decompressCPUTimeNanos.increment(7'000);
  col2Stats->decodeCPUTimeNanos.increment(8'000);
  stats.mergeFrom(splitStats);

  auto result = stats.toRuntimeMetricMap();

  EXPECT_EQ(result["skippedSplits"].sum, 5);
  EXPECT_EQ(result["processedSplits"].sum, 15);
  EXPECT_EQ(result["skippedStrides"].sum, 10);
  EXPECT_EQ(result["processedStrides"].sum, 30);
  EXPECT_EQ(result["numStripes"].sum, 4);
  const auto prefix =
      fmt::format("{}.", FileFormatName::toName(kExampleFormat));
  EXPECT_EQ(
      result[prefix + "column_1.BIGINT.decompressCPUTimeNanos"].sum, 5'000);
  EXPECT_EQ(result[prefix + "column_1.BIGINT.decompressCPUTimeNanos"].count, 1);
  EXPECT_EQ(result[prefix + "column_1.BIGINT.decodeCPUTimeNanos"].sum, 12'000);
  EXPECT_EQ(result[prefix + "column_1.BIGINT.decodeCPUTimeNanos"].count, 1);
  EXPECT_EQ(
      result[prefix + "column_2.VARCHAR.decompressCPUTimeNanos"].sum, 7'000);
  EXPECT_EQ(
      result[prefix + "column_2.VARCHAR.decompressCPUTimeNanos"].count, 1);
  EXPECT_EQ(result[prefix + "column_2.VARCHAR.decodeCPUTimeNanos"].sum, 8'000);
  EXPECT_EQ(result[prefix + "column_2.VARCHAR.decodeCPUTimeNanos"].count, 1);
  EXPECT_EQ(result[prefix + "decompressCPUTimeNanos"].sum, 12'000);
  EXPECT_EQ(result[prefix + "decompressCPUTimeNanos"].count, 2);
  EXPECT_EQ(result[prefix + "decompressCPUTimeNanos"].min, 5'000);
  EXPECT_EQ(result[prefix + "decompressCPUTimeNanos"].max, 7'000);
  EXPECT_EQ(result[prefix + "decodeCPUTimeNanos"].sum, 20'000);
  EXPECT_EQ(result[prefix + "decodeCPUTimeNanos"].count, 2);
  EXPECT_EQ(result[prefix + "decodeCPUTimeNanos"].min, 8'000);
  EXPECT_EQ(result[prefix + "decodeCPUTimeNanos"].max, 12'000);
  const auto& column1Metric = result
      [prefix + "column_1.BIGINT." + std::string(kExampleFormatMetricName)];
  EXPECT_EQ(column1Metric.sum, 1'000);
  EXPECT_EQ(column1Metric.count, 1);
  EXPECT_EQ(column1Metric.min, 1'000);
  EXPECT_EQ(column1Metric.max, 1'000);
  const auto& column2Metric = result
      [prefix + "column_2.VARCHAR." + std::string(kExampleFormatMetricName)];
  EXPECT_EQ(column2Metric.sum, 2'000);
  EXPECT_EQ(column2Metric.count, 1);
  EXPECT_EQ(column2Metric.min, 2'000);
  EXPECT_EQ(column2Metric.max, 2'000);
  const auto& formatMetric =
      result[prefix + std::string(kExampleFormatMetricName)];
  EXPECT_EQ(formatMetric.sum, 3'000);
  EXPECT_EQ(formatMetric.count, 2);
  EXPECT_EQ(formatMetric.min, 1'000);
  EXPECT_EQ(formatMetric.max, 2'000);
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

TEST(ColumnReaderStatisticsTest, MergeFromWithDecodingStats) {
  ColumnReaderStatistics src{TypeKind::BIGINT};
  src.accumulateStat(kExampleFormatMetric, 100);
  src.decodingStats.emplace();
  src.decodingStats->decompressCPUTimeNanos.increment(1'000);

  // Merge into stats without decoding stats creates and populates them.
  ColumnReaderStatistics dst{TypeKind::BIGINT};
  dst.accumulateStat(kExampleFormatMetric, 50);
  dst.mergeFrom(src);

  ASSERT_NE(
      dst.columnMetrics.find(std::string(kExampleFormatMetricName)),
      dst.columnMetrics.end());
  EXPECT_EQ(
      dst.columnMetrics.at(std::string(kExampleFormatMetricName)).sum, 150);
  EXPECT_EQ(dst.typeKind, TypeKind::BIGINT);
  ASSERT_TRUE(dst.decodingStats.has_value());
  EXPECT_EQ(dst.decodingStats->decompressCPUTimeNanos.sum(), 1'000);
}

TEST(ColumnReaderStatisticsTest, MergeFromBothWithDecodingStats) {
  ColumnReaderStatistics src{TypeKind::BIGINT};
  src.accumulateStat(kExampleFormatMetric, 100);
  src.decodingStats.emplace();
  src.decodingStats->decompressCPUTimeNanos.increment(1'000);

  ColumnReaderStatistics dst{TypeKind::BIGINT};
  dst.accumulateStat(kExampleFormatMetric, 50);
  dst.decodingStats.emplace();
  dst.decodingStats->decompressCPUTimeNanos.increment(2'000);

  dst.mergeFrom(src);

  ASSERT_NE(
      dst.columnMetrics.find(std::string(kExampleFormatMetricName)),
      dst.columnMetrics.end());
  EXPECT_EQ(
      dst.columnMetrics.at(std::string(kExampleFormatMetricName)).sum, 150);
  ASSERT_TRUE(dst.decodingStats.has_value());
  EXPECT_EQ(dst.decodingStats->decompressCPUTimeNanos.sum(), 3'000);
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
  ColumnReaderStatistics src{TypeKind::BIGINT};
  src.accumulateStat(kExampleFormatMetric, 100);

  ColumnReaderStatistics dst{TypeKind::BIGINT};
  dst.accumulateStat(kExampleFormatMetric, 50);
  dst.decodingStats.emplace();
  dst.decodingStats->decompressCPUTimeNanos.increment(1'000);

  dst.mergeFrom(src);

  ASSERT_NE(
      dst.columnMetrics.find(std::string(kExampleFormatMetricName)),
      dst.columnMetrics.end());
  EXPECT_EQ(
      dst.columnMetrics.at(std::string(kExampleFormatMetricName)).sum, 150);
  ASSERT_TRUE(dst.decodingStats.has_value());
  EXPECT_EQ(dst.decodingStats->decompressCPUTimeNanos.sum(), 1'000);
}
