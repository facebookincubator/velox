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
#include <random>
#include "velox/external/date/date.h"
#include "velox/type/FastDate.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/WideRangeDateConversion.h"

using namespace facebook::velox;

namespace {

constexpr int32_t kNumValues = 1'000'000;

// Workloads with different ranges of input days. "Tight" represents typical
// query data; "wide" exercises the algorithm closer to its limits.
std::vector<int32_t> tightDays;
std::vector<int32_t> wideDays;

struct Ymd {
  int32_t year;
  uint32_t month;
  uint32_t day;
};
std::vector<Ymd> tightYmd;
std::vector<Ymd> wideYmd;

void fillDays(std::vector<int32_t>& out, int32_t lo, int32_t hi) {
  std::mt19937 rng{0xBEEF};
  std::uniform_int_distribution<int32_t> dist{lo, hi};
  out.resize(kNumValues);
  for (auto& d : out) {
    d = dist(rng);
  }
}

void fillYmd(std::vector<Ymd>& out, const std::vector<int32_t>& days) {
  out.clear();
  out.reserve(days.size());
  for (int32_t d : days) {
    const date::year_month_day ymd{date::sys_days{date::days{d}}};
    out.push_back(
        Ymd{
            static_cast<int32_t>(static_cast<int64_t>(ymd.year())),
            static_cast<unsigned>(ymd.month()),
            static_cast<unsigned>(ymd.day()),
        });
  }
}

// --- Forward direction: epoch-seconds -> std::tm -------------------------
// Two contestants:
//   wide_range  → WideRangeDateConversion::epochToCalendarUtc
//                 (the loop-based fallback that lived inline in
//                 Timestamp::epochToCalendarUtc before this PR).
//   neri_schneider → Timestamp::epochToCalendarUtc (the patched public API,
//                 which uses the Neri-Schneider 2022 fast path for in-range
//                 inputs).
// Both fill the full std::tm including tm_hour/tm_min/tm_sec/tm_wday, so
// the only differing component is the calendar-conversion algorithm.

__attribute__((noinline)) uint64_t
runWideRange(const std::vector<int32_t>& days) {
  uint64_t sum = 0;
  std::tm tm;
  for (int32_t d : days) {
    WideRangeDateConversion::epochToCalendarUtc(int64_t{d} * 86400, tm);
    sum += static_cast<uint64_t>(tm.tm_year) +
        static_cast<uint64_t>(tm.tm_mon) + static_cast<uint64_t>(tm.tm_mday);
  }
  return sum;
}

__attribute__((noinline)) uint64_t
runNeriSchneider(const std::vector<int32_t>& days) {
  uint64_t sum = 0;
  std::tm tm;
  for (int32_t d : days) {
    Timestamp::epochToCalendarUtc(int64_t{d} * 86400, tm);
    sum += static_cast<uint64_t>(tm.tm_year) +
        static_cast<uint64_t>(tm.tm_mon) + static_cast<uint64_t>(tm.tm_mday);
  }
  return sum;
}

BENCHMARK(wide_range_tight) {
  folly::doNotOptimizeAway(runWideRange(tightDays));
}
BENCHMARK_RELATIVE(neri_schneider_tight) {
  folly::doNotOptimizeAway(runNeriSchneider(tightDays));
}

BENCHMARK_DRAW_LINE();

BENCHMARK(wide_range_wide) {
  folly::doNotOptimizeAway(runWideRange(wideDays));
}
BENCHMARK_RELATIVE(neri_schneider_wide) {
  folly::doNotOptimizeAway(runNeriSchneider(wideDays));
}

BENCHMARK_DRAW_LINE();

// --- Inverse direction: (year, month, day) -> epoch-days -----------------
// Same contestant pair, both validating via isValidDate and wrapping the
// result in Expected<int64_t>.

__attribute__((noinline)) uint64_t runWideRangeInv(const std::vector<Ymd>& v) {
  uint64_t s = 0;
  for (const auto& x : v) {
    s += static_cast<uint64_t>(
        WideRangeDateConversion::daysSinceEpochFromDate(x.year, x.month, x.day)
            .value());
  }
  return s;
}

__attribute__((noinline)) uint64_t
runNeriSchneiderInv(const std::vector<Ymd>& v) {
  uint64_t s = 0;
  for (const auto& x : v) {
    s += static_cast<uint64_t>(
        util::daysSinceEpochFromDate(x.year, x.month, x.day).value());
  }
  return s;
}

BENCHMARK(wide_range_inv_tight) {
  folly::doNotOptimizeAway(runWideRangeInv(tightYmd));
}
BENCHMARK_RELATIVE(neri_schneider_inv_tight) {
  folly::doNotOptimizeAway(runNeriSchneiderInv(tightYmd));
}

BENCHMARK_DRAW_LINE();

BENCHMARK(wide_range_inv_wide) {
  folly::doNotOptimizeAway(runWideRangeInv(wideYmd));
}
BENCHMARK_RELATIVE(neri_schneider_inv_wide) {
  folly::doNotOptimizeAway(runNeriSchneiderInv(wideYmd));
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};

  // Tight workload: roughly 1900-2040.
  fillDays(tightDays, -25'567, 25'567);
  // Wide workload: bounded by the algorithm's exact range. Still spans
  // ~3 million years, far wider than any practical Velox DATE.
  fillDays(wideDays, fast_date::kRataDieMin, fast_date::kRataDieMax);

  // Build matching y/m/d inputs for the inverse direction by running the
  // same days through the vendored Hinnant date library (used as the
  // ground-truth reference, not as a bench contestant).
  fillYmd(tightYmd, tightDays);
  fillYmd(wideYmd, wideDays);

  folly::runBenchmarks();
  return 0;
}
