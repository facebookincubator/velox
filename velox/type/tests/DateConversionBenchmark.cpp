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

using namespace facebook::velox;

namespace {

constexpr int32_t kNumValues = 1'000'000;

// Workloads with different ranges of input days. "Tight" represents typical
// query data; "wide" exercises the algorithm closer to its limits.
std::vector<int32_t> tightDays; // ~+/- 70 years from epoch.
std::vector<int32_t> wideDays; // full int32 range.

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
    out.push_back(Ymd{
        static_cast<int32_t>(static_cast<int64_t>(ymd.year())),
        static_cast<unsigned>(ymd.month()),
        static_cast<unsigned>(ymd.day()),
    });
  }
}

// --- Velox's pre-patch implementation, inlined ---------------------------
// Verbatim copy of the loop-based body that lived in
// velox/type/Timestamp.cpp:epochToCalendarUtc before this PR. Kept here so
// the bench can keep reproducing the "before" number after the patch lands;
// otherwise calling Timestamp::epochToCalendarUtc directly would just
// measure the new fast path again.
namespace legacy_loop {
constexpr int kTmYearBase = 1900;
constexpr int64_t kLeapYearOffset = 4'000'000'000LL;
constexpr int64_t kSecondsPerHour = 3600;
constexpr int64_t kSecondsPerDay = 24 * kSecondsPerHour;

inline bool isLeap(int64_t y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}
inline int64_t leapThroughEndOf(int64_t y) {
  y += kLeapYearOffset;
  return y / 4 - y / 100 + y / 400;
}
inline int64_t daysBetweenYears(int64_t y1, int64_t y2) {
  return 365 * (y2 - y1) + leapThroughEndOf(y2 - 1) - leapThroughEndOf(y1 - 1);
}
const int16_t daysBeforeFirstDayOfMonth[][12] = {
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335},
};

// noinline so we pay one function-call layer here too, matching the
// non-inlined Timestamp::epochToCalendarUtc on the patched side (its
// definition lives in a separate TU, so it can't be inlined into the
// bench's runNeriSchneider wrapper).
__attribute__((noinline)) bool epochToCalendarUtc(int64_t epoch, std::tm& tm) {
  constexpr int kDaysPerYear = 365;
  int64_t days = epoch / kSecondsPerDay;
  int64_t rem = epoch % kSecondsPerDay;
  while (rem < 0) {
    rem += kSecondsPerDay;
    --days;
  }
  tm.tm_hour = rem / kSecondsPerHour;
  rem = rem % kSecondsPerHour;
  tm.tm_min = rem / 60;
  tm.tm_sec = rem % 60;
  tm.tm_wday = (4 + days) % 7;
  if (tm.tm_wday < 0) {
    tm.tm_wday += 7;
  }
  int64_t y = 1970;
  if (y + days / kDaysPerYear <= -kLeapYearOffset + 10) {
    return false;
  }
  bool leapYear;
  while (days < 0 || days >= kDaysPerYear + (leapYear = isLeap(y))) {
    auto newy = y + days / kDaysPerYear - (days < 0);
    days -= daysBetweenYears(y, newy);
    y = newy;
  }
  y -= kTmYearBase;
  tm.tm_year = static_cast<int>(y);
  tm.tm_yday = static_cast<int>(days);
  const auto* months = daysBeforeFirstDayOfMonth[leapYear];
  tm.tm_mon = static_cast<int>(
      std::upper_bound(months, months + 12, days) - months - 1);
  tm.tm_mday = static_cast<int>(days - months[tm.tm_mon] + 1);
  tm.tm_isdst = 0;
  return true;
}
} // namespace legacy_loop

__attribute__((noinline)) uint64_t runLegacyLoop(
    const std::vector<int32_t>& days) {
  uint64_t sum = 0;
  std::tm tm;
  for (int32_t d : days) {
    legacy_loop::epochToCalendarUtc(int64_t{d} * 86400, tm);
    sum += static_cast<uint64_t>(tm.tm_year) +
        static_cast<uint64_t>(tm.tm_mon) +
        static_cast<uint64_t>(tm.tm_mday);
  }
  return sum;
}

// --- Chosen path: the patched public API (Timestamp::epochToCalendarUtc) --
// Calling the production API rather than daysToYmd directly so the ratio
// against legacy_loop is apples-to-apples — both fill a full std::tm and
// pay the fast-path range check.
__attribute__((noinline)) uint64_t runNeriSchneider(
    const std::vector<int32_t>& days) {
  uint64_t sum = 0;
  std::tm tm;
  for (int32_t d : days) {
    Timestamp::epochToCalendarUtc(int64_t{d} * 86400, tm);
    sum += static_cast<uint64_t>(tm.tm_year) +
        static_cast<uint64_t>(tm.tm_mon) +
        static_cast<uint64_t>(tm.tm_mday);
  }
  return sum;
}

BENCHMARK(legacy_loop_tight) {
  folly::doNotOptimizeAway(runLegacyLoop(tightDays));
}
BENCHMARK_RELATIVE(neri_schneider_tight) {
  folly::doNotOptimizeAway(runNeriSchneider(tightDays));
}

BENCHMARK_DRAW_LINE();

BENCHMARK(legacy_loop_wide) {
  folly::doNotOptimizeAway(runLegacyLoop(wideDays));
}
BENCHMARK_RELATIVE(neri_schneider_wide) {
  folly::doNotOptimizeAway(runNeriSchneider(wideDays));
}

BENCHMARK_DRAW_LINE();

// --- Inverse direction: (year, month, day) -> epoch-days ----------------

// Verbatim copy of the pre-patch body of util::daysSinceEpochFromDate
// (validation + loop + table). Kept here so the bench can keep reproducing
// the "before" number after the patch lands; calling the live API would
// just measure the new fast path. Same validation and Expected<> wrapping
// as the production code so the comparison against the patched API is
// apples-to-apples.
namespace legacy_loop_inv {
constexpr int32_t kYearInterval = 400;
constexpr int32_t kDaysPerYearInterval = 146097;
constexpr int32_t kCumulativeDays[] =
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
constexpr int32_t kCumulativeLeapDays[] =
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
// For isValidDate: days-in-month tables and proleptic year bounds. These
// values are identical to the production constants in TimestampConversion.h
// and are validated by FastDateTest.
constexpr int32_t kLeapDays[] =
    {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
constexpr int32_t kNormalDays[] =
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
constexpr int32_t kMinYear = -292'275'055;
constexpr int32_t kMaxYear = 292'278'994;
constexpr int32_t kCumulativeYearDays[] = {
    0,      365,    730,    1096,   1461,   1826,   2191,   2557,   2922,
    3287,   3652,   4018,   4383,   4748,   5113,   5479,   5844,   6209,
    6574,   6940,   7305,   7670,   8035,   8401,   8766,   9131,   9496,
    9862,   10227,  10592,  10957,  11323,  11688,  12053,  12418,  12784,
    13149,  13514,  13879,  14245,  14610,  14975,  15340,  15706,  16071,
    16436,  16801,  17167,  17532,  17897,  18262,  18628,  18993,  19358,
    19723,  20089,  20454,  20819,  21184,  21550,  21915,  22280,  22645,
    23011,  23376,  23741,  24106,  24472,  24837,  25202,  25567,  25933,
    26298,  26663,  27028,  27394,  27759,  28124,  28489,  28855,  29220,
    29585,  29950,  30316,  30681,  31046,  31411,  31777,  32142,  32507,
    32872,  33238,  33603,  33968,  34333,  34699,  35064,  35429,  35794,
    36160,  36525,  36890,  37255,  37621,  37986,  38351,  38716,  39082,
    39447,  39812,  40177,  40543,  40908,  41273,  41638,  42004,  42369,
    42734,  43099,  43465,  43830,  44195,  44560,  44926,  45291,  45656,
    46021,  46387,  46752,  47117,  47482,  47847,  48212,  48577,  48942,
    49308,  49673,  50038,  50403,  50769,  51134,  51499,  51864,  52230,
    52595,  52960,  53325,  53691,  54056,  54421,  54786,  55152,  55517,
    55882,  56247,  56613,  56978,  57343,  57708,  58074,  58439,  58804,
    59169,  59535,  59900,  60265,  60630,  60996,  61361,  61726,  62091,
    62457,  62822,  63187,  63552,  63918,  64283,  64648,  65013,  65379,
    65744,  66109,  66474,  66840,  67205,  67570,  67935,  68301,  68666,
    69031,  69396,  69762,  70127,  70492,  70857,  71223,  71588,  71953,
    72318,  72684,  73049,  73414,  73779,  74145,  74510,  74875,  75240,
    75606,  75971,  76336,  76701,  77067,  77432,  77797,  78162,  78528,
    78893,  79258,  79623,  79989,  80354,  80719,  81084,  81450,  81815,
    82180,  82545,  82911,  83276,  83641,  84006,  84371,  84736,  85101,
    85466,  85832,  86197,  86562,  86927,  87293,  87658,  88023,  88388,
    88754,  89119,  89484,  89849,  90215,  90580,  90945,  91310,  91676,
    92041,  92406,  92771,  93137,  93502,  93867,  94232,  94598,  94963,
    95328,  95693,  96059,  96424,  96789,  97154,  97520,  97885,  98250,
    98615,  98981,  99346,  99711,  100076, 100442, 100807, 101172, 101537,
    101903, 102268, 102633, 102998, 103364, 103729, 104094, 104459, 104825,
    105190, 105555, 105920, 106286, 106651, 107016, 107381, 107747, 108112,
    108477, 108842, 109208, 109573, 109938, 110303, 110669, 111034, 111399,
    111764, 112130, 112495, 112860, 113225, 113591, 113956, 114321, 114686,
    115052, 115417, 115782, 116147, 116513, 116878, 117243, 117608, 117974,
    118339, 118704, 119069, 119435, 119800, 120165, 120530, 120895, 121260,
    121625, 121990, 122356, 122721, 123086, 123451, 123817, 124182, 124547,
    124912, 125278, 125643, 126008, 126373, 126739, 127104, 127469, 127834,
    128200, 128565, 128930, 129295, 129661, 130026, 130391, 130756, 131122,
    131487, 131852, 132217, 132583, 132948, 133313, 133678, 134044, 134409,
    134774, 135139, 135505, 135870, 136235, 136600, 136966, 137331, 137696,
    138061, 138427, 138792, 139157, 139522, 139888, 140253, 140618, 140983,
    141349, 141714, 142079, 142444, 142810, 143175, 143540, 143905, 144271,
    144636, 145001, 145366, 145732, 146097,
};

inline bool isLeapYear(int32_t y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

inline bool isValidDate(int32_t year, int32_t month, int32_t day) {
  if (month < 1 || month > 12) {
    return false;
  }
  if (year < kMinYear || year > kMaxYear) {
    return false;
  }
  if (day < 1) {
    return false;
  }
  return isLeapYear(year) ? day <= kLeapDays[month]
                          : day <= kNormalDays[month];
}

// noinline for the same reason as legacy_loop::epochToCalendarUtc above:
// match the non-inlinable production util::daysSinceEpochFromDate on the
// patched side.
__attribute__((noinline)) facebook::velox::Expected<int64_t>
daysSinceEpochFromDate(int32_t year, int32_t month, int32_t day) {
  if (!isValidDate(year, month, day)) {
    return folly::makeUnexpected(facebook::velox::Status::UserError(
        "Date out of range: {}-{}-{}", year, month, day));
  }
  int64_t daysSinceEpoch = 0;
  while (year < 1970) {
    year += kYearInterval;
    daysSinceEpoch -= kDaysPerYearInterval;
  }
  while (year >= 2370) {
    year -= kYearInterval;
    daysSinceEpoch += kDaysPerYearInterval;
  }
  daysSinceEpoch += kCumulativeYearDays[year - 1970];
  daysSinceEpoch += isLeapYear(year) ? kCumulativeLeapDays[month - 1]
                                     : kCumulativeDays[month - 1];
  daysSinceEpoch += day - 1;
  return daysSinceEpoch;
}
} // namespace legacy_loop_inv

__attribute__((noinline)) uint64_t runLegacyLoopInv(
    const std::vector<Ymd>& v) {
  uint64_t s = 0;
  for (const auto& x : v) {
    s += static_cast<uint64_t>(
        legacy_loop_inv::daysSinceEpochFromDate(x.year, x.month, x.day)
            .value());
  }
  return s;
}

// Calling the patched public API so the comparison against legacy_loop_inv
// is apples-to-apples — both contestants run isValidDate, both wrap the
// result in Expected<>, the only difference is the conversion algorithm.
__attribute__((noinline)) uint64_t runNeriSchneiderInv(
    const std::vector<Ymd>& v) {
  uint64_t s = 0;
  for (const auto& x : v) {
    s += static_cast<uint64_t>(
        facebook::velox::util::daysSinceEpochFromDate(
            x.year, x.month, x.day)
            .value());
  }
  return s;
}

BENCHMARK(legacy_loop_inv_tight) {
  folly::doNotOptimizeAway(runLegacyLoopInv(tightYmd));
}
BENCHMARK_RELATIVE(neri_schneider_inv_tight) {
  folly::doNotOptimizeAway(runNeriSchneiderInv(tightYmd));
}

BENCHMARK_DRAW_LINE();

BENCHMARK(legacy_loop_inv_wide) {
  folly::doNotOptimizeAway(runLegacyLoopInv(wideYmd));
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

  // Sanity-check that the legacy and patched paths agree on the tight
  // workload — catches integration bugs where the patched
  // Timestamp::epochToCalendarUtc fails to fill std::tm correctly.
  for (size_t i = 0; i < std::min<size_t>(tightDays.size(), 1000); ++i) {
    const int32_t d = tightDays[i];
    std::tm legacyTm;
    std::tm patchedTm;
    if (!legacy_loop::epochToCalendarUtc(int64_t{d} * 86400, legacyTm)) {
      continue;
    }
    if (!Timestamp::epochToCalendarUtc(int64_t{d} * 86400, patchedTm)) {
      LOG(FATAL) << "patched epochToCalendarUtc returned false at day=" << d;
    }
    if (legacyTm.tm_year != patchedTm.tm_year ||
        legacyTm.tm_mon != patchedTm.tm_mon ||
        legacyTm.tm_mday != patchedTm.tm_mday) {
      LOG(FATAL) << "legacy/patched disagree at day=" << d;
    }
  }

  folly::runBenchmarks();
  return 0;
}
