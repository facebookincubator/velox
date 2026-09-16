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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <folly/Benchmark.h>
#include <folly/Portability.h>
#include <folly/init/Init.h>

#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"

using namespace facebook::nimble;

namespace {

struct LegacyIntegralStatistics {
  uint64_t logicalSize{0};
  std::optional<int64_t> min;
  std::optional<int64_t> max;
};

struct LegacyFloatingPointStatistics {
  uint64_t logicalSize{0};
  std::optional<double> min;
  std::optional<double> max;
};

struct LegacyStringStatistics {
  uint64_t logicalSize{0};
  std::optional<std::string> min;
  std::optional<std::string> max;
};

template <typename T>
std::vector<T> makeIntegralValues(size_t numValues) {
  std::vector<T> values;
  values.reserve(numValues);

  const auto lower =
      std::max<int64_t>(std::numeric_limits<T>::min(), -1'000'000);
  const auto upper =
      std::min<int64_t>(std::numeric_limits<T>::max(), 1'000'000);
  const auto range = static_cast<uint64_t>(upper - lower) + 1;
  for (size_t i = 0; i < numValues; ++i) {
    const auto mixed = i * 1'103'515'245 + 12'345;
    values.push_back(static_cast<T>(lower + (mixed % range)));
  }

  if (!values.empty()) {
    values[numValues / 3] = std::numeric_limits<T>::min();
    values[(2 * numValues) / 3] = std::numeric_limits<T>::max();
  }
  return values;
}

template <typename T, size_t NumValues>
std::vector<T>& integralValues() {
  static auto* values = new std::vector<T>(makeIntegralValues<T>(NumValues));
  return *values;
}

template <typename T>
std::vector<T> makeFloatingPointValues(size_t numValues) {
  std::vector<T> values;
  values.reserve(numValues);

  for (size_t i = 0; i < numValues; ++i) {
    const auto mixed = i * 1'103'515'245 + 12'345;
    values.push_back(
        static_cast<T>(static_cast<int64_t>(mixed % 2'000'001) - 1'000'000) /
        static_cast<T>(10.0));
  }

  if (!values.empty()) {
    values[numValues / 3] = std::numeric_limits<T>::lowest();
    values[(2 * numValues) / 3] = std::numeric_limits<T>::max();
  }
  return values;
}

template <typename T, size_t NumValues>
std::vector<T>& floatingPointValues() {
  static auto* values =
      new std::vector<T>(makeFloatingPointValues<T>(NumValues));
  return *values;
}

// Ascending input is the shape the batched scan targets: under the legacy
// implementation every value advanced the maximum and allocated a string, while
// random input allocates only on the occasional new extreme.
enum class StringOrder { kRandom, kAscending };

std::vector<std::string> makeStringValues(size_t numValues, StringOrder order) {
  std::vector<std::string> values;
  values.reserve(numValues);

  for (size_t i = 0; i < numValues; ++i) {
    const auto mixed =
        order == StringOrder::kAscending ? i : (i * 1'103'515'245 + 12'345);
    values.push_back(
        fmt::format("{:016x}_padding_to_a_realistic_width", mixed));
  }
  return values;
}

template <StringOrder Order, size_t NumValues>
std::vector<std::string_view>& stringValues() {
  static auto* owned =
      new std::vector<std::string>(makeStringValues(NumValues, Order));
  static auto* views =
      new std::vector<std::string_view>(owned->begin(), owned->end());
  return *views;
}

// Mirrors StringStatisticsCollector::addValues before the batched scan: the
// ternary reassigns the accumulator on every value, so a non-advancing value
// still copy-assigns the optional and an advancing one allocates.
FOLLY_NOINLINE void legacyAddStringValues(
    LegacyStringStatistics& stats,
    std::span<std::string_view> values) {
  if (values.empty()) {
    return;
  }

  if (!stats.min.has_value()) {
    stats.min = std::string(values.front());
  }
  if (!stats.max.has_value()) {
    stats.max = std::string(values.front());
  }

  for (const auto& value : values) {
    stats.logicalSize += value.size();
    stats.min =
        stats.min > value ? std::make_optional(std::string(value)) : stats.min;
    stats.max =
        stats.max < value ? std::make_optional(std::string(value)) : stats.max;
  }
}

template <StringOrder Order, size_t NumValues>
void legacyStringOptionalBenchmark(unsigned int iters) {
  auto& input = stringValues<Order, NumValues>();
  LegacyStringStatistics stats;

  while (iters--) {
    legacyAddStringValues(stats, std::span<std::string_view>(input));
  }

  folly::doNotOptimizeAway(stats.logicalSize);
  folly::doNotOptimizeAway(stats.min);
  folly::doNotOptimizeAway(stats.max);
}

template <StringOrder Order, size_t NumValues>
void currentStringCollectorBenchmark(unsigned int iters) {
  auto& input = stringValues<Order, NumValues>();
  StringStatisticsCollector collector;

  while (iters--) {
    collector.addValues(std::span<std::string_view>(input));
  }

  auto* stats = collector.getStatsView()->as<StringStatistics>();
  folly::doNotOptimizeAway(stats->getLogicalSize());
  folly::doNotOptimizeAway(stats->getMin());
  folly::doNotOptimizeAway(stats->getMax());
}

template <typename T>
FOLLY_NOINLINE void legacyAddIntegralValues(
    LegacyIntegralStatistics& stats,
    std::span<T> values) {
  stats.logicalSize += values.size() * sizeof(T);

  if (values.empty()) {
    return;
  }

  if (!stats.min.has_value()) {
    stats.min = static_cast<int64_t>(values.front());
  }
  if (!stats.max.has_value()) {
    stats.max = static_cast<int64_t>(values.front());
  }

  for (const auto& value : values) {
    auto current = static_cast<int64_t>(value);
    stats.min = stats.min > current ? std::make_optional(current) : stats.min;
    stats.max = stats.max < current ? std::make_optional(current) : stats.max;
  }
}

template <typename T>
FOLLY_NOINLINE void legacyAddFloatingPointValues(
    LegacyFloatingPointStatistics& stats,
    std::span<T> values) {
  stats.logicalSize += values.size() * sizeof(T);

  if (values.empty()) {
    return;
  }

  if (!stats.min.has_value()) {
    stats.min = static_cast<double>(values.front());
  }
  if (!stats.max.has_value()) {
    stats.max = static_cast<double>(values.front());
  }

  for (const auto& value : values) {
    auto current = static_cast<double>(value);
    stats.min = stats.min > current ? std::make_optional(current) : stats.min;
    stats.max = stats.max < current ? std::make_optional(current) : stats.max;
  }
}

template <typename T, size_t NumValues>
void legacyIntegralOptionalBenchmark(unsigned int iters) {
  auto& input = integralValues<T, NumValues>();
  LegacyIntegralStatistics stats;

  while (iters--) {
    legacyAddIntegralValues(stats, std::span<T>(input));
  }

  folly::doNotOptimizeAway(stats.logicalSize);
  folly::doNotOptimizeAway(stats.min);
  folly::doNotOptimizeAway(stats.max);
}

template <typename T, size_t NumValues>
void currentIntegralCollectorBenchmark(unsigned int iters) {
  auto& input = integralValues<T, NumValues>();
  IntegralStatisticsCollector collector;

  while (iters--) {
    collector.addValues(std::span<T>(input));
  }

  auto* stats = collector.getStatsView()->as<IntegralStatistics>();
  folly::doNotOptimizeAway(stats->getLogicalSize());
  folly::doNotOptimizeAway(stats->getMin());
  folly::doNotOptimizeAway(stats->getMax());
}

template <typename T, size_t NumValues>
void legacyFloatingPointOptionalBenchmark(unsigned int iters) {
  auto& input = floatingPointValues<T, NumValues>();
  LegacyFloatingPointStatistics stats;

  while (iters--) {
    legacyAddFloatingPointValues(stats, std::span<T>(input));
  }

  folly::doNotOptimizeAway(stats.logicalSize);
  folly::doNotOptimizeAway(stats.min);
  folly::doNotOptimizeAway(stats.max);
}

template <typename T, size_t NumValues>
void currentFloatingPointCollectorBenchmark(unsigned int iters) {
  auto& input = floatingPointValues<T, NumValues>();
  FloatingPointStatisticsCollector collector;

  while (iters--) {
    collector.addValues(std::span<T>(input));
  }

  auto* stats = collector.getStatsView()->as<FloatingPointStatistics>();
  folly::doNotOptimizeAway(stats->getLogicalSize());
  folly::doNotOptimizeAway(stats->getMin());
  folly::doNotOptimizeAway(stats->getMax());
}

} // namespace

#define INTEGRAL_STATS_BENCHMARKS(type, name, numValues)             \
  BENCHMARK(legacyOptional_##name##_##numValues, iters) {            \
    legacyIntegralOptionalBenchmark<type, numValues>(iters);         \
  }                                                                  \
  BENCHMARK_RELATIVE(currentCollector_##name##_##numValues, iters) { \
    currentIntegralCollectorBenchmark<type, numValues>(iters);       \
  }                                                                  \
  BENCHMARK_DRAW_LINE()

#define FLOATING_POINT_STATS_BENCHMARKS(type, name, numValues)         \
  BENCHMARK(legacyFloatingPointOptional_##name##_##numValues, iters) { \
    legacyFloatingPointOptionalBenchmark<type, numValues>(iters);      \
  }                                                                    \
  BENCHMARK_RELATIVE(                                                  \
      currentFloatingPointCollector_##name##_##numValues, iters) {     \
    currentFloatingPointCollectorBenchmark<type, numValues>(iters);    \
  }                                                                    \
  BENCHMARK_DRAW_LINE()

#define STRING_STATS_BENCHMARKS(order, name, numValues)                    \
  BENCHMARK(legacyStringOptional_##name##_##numValues, iters) {            \
    legacyStringOptionalBenchmark<order, numValues>(iters);                \
  }                                                                        \
  BENCHMARK_RELATIVE(currentStringCollector_##name##_##numValues, iters) { \
    currentStringCollectorBenchmark<order, numValues>(iters);              \
  }                                                                        \
  BENCHMARK_DRAW_LINE()

INTEGRAL_STATS_BENCHMARKS(int8_t, int8, 128);
INTEGRAL_STATS_BENCHMARKS(int8_t, int8, 8192);
INTEGRAL_STATS_BENCHMARKS(int16_t, int16, 128);
INTEGRAL_STATS_BENCHMARKS(int16_t, int16, 8192);
INTEGRAL_STATS_BENCHMARKS(int32_t, int32, 128);
INTEGRAL_STATS_BENCHMARKS(int32_t, int32, 8192);
INTEGRAL_STATS_BENCHMARKS(int64_t, int64, 128);
INTEGRAL_STATS_BENCHMARKS(int64_t, int64, 8192);
FLOATING_POINT_STATS_BENCHMARKS(float, float, 128);
FLOATING_POINT_STATS_BENCHMARKS(float, float, 8192);
FLOATING_POINT_STATS_BENCHMARKS(double, double, 128);
FLOATING_POINT_STATS_BENCHMARKS(double, double, 8192);
STRING_STATS_BENCHMARKS(StringOrder::kRandom, random, 128);
STRING_STATS_BENCHMARKS(StringOrder::kRandom, random, 8192);
STRING_STATS_BENCHMARKS(StringOrder::kAscending, ascending, 128);
STRING_STATS_BENCHMARKS(StringOrder::kAscending, ascending, 8192);

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
