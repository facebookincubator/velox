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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <chrono>
#include <latch>
#include <thread>
#include <vector>

#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::tz {
namespace {

enum class Direction { kToSys, kToLocal };

// Each sample performs the same total work with either one or two workers.
// BENCHMARK_MULTI reports wall-clock time per conversion across all workers.
constexpr unsigned kDays = 365;
constexpr unsigned kRepeats = 4096;
constexpr unsigned kConversions = kDays * kRepeats;

template <Direction direction>
unsigned runConversions(const char* name, int year, unsigned numThreads) {
  folly::BenchmarkSuspender suspender;
  const auto* zone = locateZone(name);
  const auto firstDay =
      std::chrono::sys_days{std::chrono::year{year} / std::chrono::January / 1};
  std::vector<std::chrono::seconds> input;
  input.reserve(kDays);
  for (unsigned day = 0; day < kDays; ++day) {
    // Local noon avoids gaps and overlaps. Using a full year covers both
    // standard and daylight time in New York, including future POSIX rules.
    const auto local = std::chrono::duration_cast<std::chrono::seconds>(
        (firstDay + std::chrono::days{day} + std::chrono::hours{12})
            .time_since_epoch());
    const auto system = zone->to_sys(local);
    if constexpr (direction == Direction::kToSys) {
      input.push_back(local);
    } else {
      input.push_back(system);
    }
    folly::doNotOptimizeAway(zone->to_local(system));
  }

  std::latch ready{numThreads};
  std::latch start{1};
  std::latch done{numThreads};
  std::vector<std::thread> workers;
  workers.reserve(numThreads);
  for (unsigned thread = 0; thread < numThreads; ++thread) {
    workers.emplace_back([&] {
      ready.count_down();
      start.wait();
      uint64_t checksum = 0;
      for (unsigned repeat = 0; repeat < kRepeats / numThreads; ++repeat) {
        for (const auto timestamp : input) {
          if constexpr (direction == Direction::kToSys) {
            checksum += zone->to_sys(timestamp).count();
          } else {
            checksum += zone->to_local(timestamp).count();
          }
        }
      }
      folly::doNotOptimizeAway(checksum);
      done.count_down();
    });
  }

  // Worker creation, time-zone initialization, input generation, and joins
  // are outside the timed section. Workers share the same cached TimeZone.
  ready.wait();
  suspender.dismiss();
  start.count_down();
  done.wait();
  suspender.rehire();
  for (auto& worker : workers) {
    worker.join();
  }
  return kConversions;
}

#define TIME_ZONE_BENCHMARK(name, direction, zone, year, threads)     \
  BENCHMARK_MULTI(name) {                                             \
    return runConversions<Direction::direction>(zone, year, threads); \
  }

TIME_ZONE_BENCHMARK(
    toSys_Shanghai2026_1Thread,
    kToSys,
    "Asia/Shanghai",
    2026,
    1)
TIME_ZONE_BENCHMARK(
    toSys_Shanghai2026_2Threads,
    kToSys,
    "Asia/Shanghai",
    2026,
    2)
TIME_ZONE_BENCHMARK(
    toLocal_Shanghai2026_1Thread,
    kToLocal,
    "Asia/Shanghai",
    2026,
    1)
TIME_ZONE_BENCHMARK(
    toLocal_Shanghai2026_2Threads,
    kToLocal,
    "Asia/Shanghai",
    2026,
    2)
BENCHMARK_DRAW_LINE();
TIME_ZONE_BENCHMARK(
    toSys_NewYork2026_1Thread,
    kToSys,
    "America/New_York",
    2026,
    1)
TIME_ZONE_BENCHMARK(
    toSys_NewYork2026_2Threads,
    kToSys,
    "America/New_York",
    2026,
    2)
TIME_ZONE_BENCHMARK(
    toLocal_NewYork2026_1Thread,
    kToLocal,
    "America/New_York",
    2026,
    1)
TIME_ZONE_BENCHMARK(
    toLocal_NewYork2026_2Threads,
    kToLocal,
    "America/New_York",
    2026,
    2)
BENCHMARK_DRAW_LINE();
TIME_ZONE_BENCHMARK(
    toSys_NewYork2100_1Thread,
    kToSys,
    "America/New_York",
    2100,
    1)
TIME_ZONE_BENCHMARK(
    toSys_NewYork2100_2Threads,
    kToSys,
    "America/New_York",
    2100,
    2)
TIME_ZONE_BENCHMARK(
    toLocal_NewYork2100_1Thread,
    kToLocal,
    "America/New_York",
    2100,
    1)
TIME_ZONE_BENCHMARK(
    toLocal_NewYork2100_2Threads,
    kToLocal,
    "America/New_York",
    2100,
    2)

#undef TIME_ZONE_BENCHMARK

} // namespace
} // namespace facebook::velox::tz

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
}
