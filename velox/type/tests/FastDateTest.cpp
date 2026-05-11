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

#include "velox/type/FastDate.h"

#include <gtest/gtest.h>
#include <array>
#include <random>
#include "velox/external/date/date.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/WideRangeDateConversion.h"

namespace facebook::velox::test {
namespace {

// Returns the (year, month, day) computed by Howard Hinnant's date library,
// which we use as the ground truth.
YearMonthDay hinnantYmd(int32_t dayNumber) {
  const date::sys_days sd{date::days{dayNumber}};
  const date::year_month_day ymd{sd};
  return YearMonthDay{
      static_cast<int32_t>(static_cast<int64_t>(ymd.year())),
      static_cast<unsigned>(ymd.month()),
      static_cast<unsigned>(ymd.day()),
  };
}

void expectEq(int32_t dayNumber, YearMonthDay got, YearMonthDay want) {
  EXPECT_EQ(got.year, want.year) << "day=" << dayNumber;
  EXPECT_EQ(got.month, want.month) << "day=" << dayNumber;
  EXPECT_EQ(got.day, want.day) << "day=" << dayNumber;
}

TEST(FastDateTest, knownDates) {
  // (input days since epoch, expected y, m, d). Hand-verifiable cases.
  const std::array<std::tuple<int32_t, int32_t, uint32_t, uint32_t>, 9> cases =
      {{
          {0, 1970, 1, 1}, // Epoch.
          {1, 1970, 1, 2},
          {-1, 1969, 12, 31},
          {31, 1970, 2, 1},
          {59, 1970, 3, 1}, // 1970 not a leap year.
          {365, 1971, 1, 1},
          {-365, 1969, 1, 1},
          {11016, 2000, 2, 29}, // Leap day in a 400-year leap year.
          {-25509, 1900, 2, 28}, // 1900 not a leap year (no Feb 29).
      }};

  for (const auto& [d, y, m, day] : cases) {
    SCOPED_TRACE(testing::Message() << "dayNumber=" << d);
    const auto got = daysToYmd(d);
    EXPECT_EQ(got.year, y);
    EXPECT_EQ(got.month, m);
    EXPECT_EQ(got.day, day);
  }
}

TEST(FastDateTest, matchesHinnantOnTightRange) {
  // Walk every day across roughly +/- 600 years from epoch. This covers all
  // Gregorian rule transitions of practical interest (centuries, 400-year
  // boundaries, leap years).
  for (int32_t d = -219145; d <= 219145; ++d) {
    const auto got = daysToYmd(d);
    const auto want = hinnantYmd(d);
    if (got.year != want.year || got.month != want.month ||
        got.day != want.day) {
      FAIL() << "Mismatch at day=" << d << ": got " << got.year << '-'
             << got.month << '-' << got.day << " want " << want.year << '-'
             << want.month << '-' << want.day;
    }
  }
}

TEST(FastDateTest, matchesHinnantOnRandomFullRange) {
  // 2M random days uniformly across the algorithm's exact range. Spans
  // roughly 3 million years centered on the epoch; behavior outside this
  // range is undefined per FastDate.h.
  std::mt19937 rng{0xC0FFEEu};
  std::uniform_int_distribution<int32_t> dist{
      fast_date::kRataDieMin, fast_date::kRataDieMax};
  for (int i = 0; i < 2'000'000; ++i) {
    const int32_t d = dist(rng);
    const auto got = daysToYmd(d);
    const auto want = hinnantYmd(d);
    ASSERT_EQ(got.year, want.year) << "day=" << d;
    ASSERT_EQ(got.month, want.month) << "day=" << d;
    ASSERT_EQ(got.day, want.day) << "day=" << d;
  }
}

TEST(FastDateTest, boundaries) {
  // Days at every century and 400-year boundary across the algorithm's
  // exact range. Boundaries are where Gregorian rules diverge most.
  for (int32_t centuryYear = fast_date::kYearMin + 100;
       centuryYear <= fast_date::kYearMax - 100;
       centuryYear += 100) {
    const auto sd =
        date::sys_days{date::year{centuryYear} / date::month{3} / date::day{1}};
    const int32_t d = sd.time_since_epoch().count();
    const auto got = daysToYmd(d);
    const auto want = hinnantYmd(d);
    expectEq(d, got, want);
  }
}

// --- Public-API random-input cross-checks (fuzzer-style) -----------------
// These verify the patched Velox functions (Timestamp::epochToCalendarUtc and
// util::daysSinceEpochFromDate), not just the inner FastDate.h primitives.
// Three distributions per direction:
//   tight:     ~+/- 70 years from epoch (the realistic-data case).
//   wide:      across the full fast_date::kRataDie range (algorithm domain).
//   fallback:  way outside the algorithm range, exercising the legacy loop.
//
// Each distribution gets 2M random samples, cross-checked against
// date::year_month_day from velox/external/date/date.h.

namespace {
constexpr int64_t kSecondsPerDay = 86'400;

// Floor-division: seconds -> days, treating negative remainders correctly.
int64_t epochToDays(int64_t epoch) {
  int64_t d = epoch / kSecondsPerDay;
  int64_t r = epoch % kSecondsPerDay;
  if (r < 0) {
    --d;
  }
  return d;
}

// Reference y/m/d for any int64 day count (Hinnant supports the full range).
struct ReferenceYmd {
  int64_t year;
  uint32_t month;
  uint32_t day;
};
ReferenceYmd hinnantRef(int64_t days) {
  const date::sys_days sd{date::days{days}};
  const date::year_month_day ymd{sd};
  return {
      static_cast<int64_t>(ymd.year()),
      static_cast<unsigned>(ymd.month()),
      static_cast<unsigned>(ymd.day()),
  };
}

// Asserts that two std::tm structs agree on every field a caller of
// epochToCalendarUtc could observe. tm_isdst is set to 0 by both paths.
void expectTmEqual(const std::tm& a, const std::tm& b, int64_t epoch) {
  EXPECT_EQ(a.tm_year, b.tm_year) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_mon, b.tm_mon) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_mday, b.tm_mday) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_yday, b.tm_yday) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_wday, b.tm_wday) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_hour, b.tm_hour) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_min, b.tm_min) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_sec, b.tm_sec) << "epoch=" << epoch;
  EXPECT_EQ(a.tm_isdst, b.tm_isdst) << "epoch=" << epoch;
}
} // namespace

TEST(FastDateTest, fuzzEpochToCalendarUtcMatchesHinnant) {
  std::mt19937_64 rng{0xFAB1Eull};
  // Tight: realistic Velox data (~1900-2040).
  std::uniform_int_distribution<int64_t> tight{
      -25'567 * kSecondsPerDay, 25'567 * kSecondsPerDay};
  // Wide: full algorithmic range (~3M years).
  std::uniform_int_distribution<int64_t> wide{
      static_cast<int64_t>(fast_date::kRataDieMin) * kSecondsPerDay,
      static_cast<int64_t>(fast_date::kRataDieMax) * kSecondsPerDay};
  // Fallback: outside the algorithm range on both sides of the epoch,
  // exercising the legacy 400-year-step loop. Bound by the generous
  // validation range (~+/- 290M years) but kept far enough from int64
  // limits that seconds * 86400 doesn't overflow.
  const int64_t fallbackPositiveStart =
      static_cast<int64_t>(fast_date::kRataDieMax + 1) * kSecondsPerDay;
  const int64_t fallbackPositiveEnd =
      (static_cast<int64_t>(fast_date::kYearMax) + 100'000) * 365 *
      kSecondsPerDay;
  const int64_t fallbackNegativeEnd =
      static_cast<int64_t>(fast_date::kRataDieMin - 1) * kSecondsPerDay;
  const int64_t fallbackNegativeStart =
      -(static_cast<int64_t>(fast_date::kYearMax) + 100'000) * 365 *
      kSecondsPerDay;
  std::uniform_int_distribution<int64_t> fallbackPositive{
      fallbackPositiveStart, fallbackPositiveEnd};
  std::uniform_int_distribution<int64_t> fallbackNegative{
      fallbackNegativeStart, fallbackNegativeEnd};
  // Random non-zero seconds-of-day so tm_hour/tm_min/tm_sec are exercised.
  std::uniform_int_distribution<int64_t> secondsOfDay{0, kSecondsPerDay - 1};

  auto check = [](int64_t epoch) {
    std::tm fastTm;
    std::tm wideTm;
    if (!Timestamp::epochToCalendarUtc(epoch, fastTm)) {
      // Out-of-range outputs are an acceptable "false" return on both
      // paths; just verify the wide path agrees and skip.
      ASSERT_FALSE(WideRangeDateConversion::epochToCalendarUtc(epoch, wideTm))
          << "epoch=" << epoch;
      return;
    }
    ASSERT_TRUE(WideRangeDateConversion::epochToCalendarUtc(epoch, wideTm))
        << "epoch=" << epoch;
    // Full std::tm equality: y/m/d, yday, wday, hour, min, sec, isdst.
    expectTmEqual(fastTm, wideTm, epoch);
    // Independent cross-check of the date fields against Hinnant.
    const auto want = hinnantRef(epochToDays(epoch));
    ASSERT_EQ(fastTm.tm_year + 1900, want.year) << "epoch=" << epoch;
    ASSERT_EQ(fastTm.tm_mon + 1, want.month) << "epoch=" << epoch;
    ASSERT_EQ(static_cast<unsigned>(fastTm.tm_mday), want.day)
        << "epoch=" << epoch;
  };

  for (int i = 0; i < 2'000'000; ++i) {
    check(tight(rng) + secondsOfDay(rng));
  }
  for (int i = 0; i < 2'000'000; ++i) {
    check(wide(rng) + secondsOfDay(rng));
  }
  // Fewer fallback samples per side — per-call cost is much higher (the
  // legacy loop runs many iterations for years far from epoch).
  for (int i = 0; i < 500'000; ++i) {
    check(fallbackPositive(rng) + secondsOfDay(rng));
  }
  for (int i = 0; i < 500'000; ++i) {
    check(fallbackNegative(rng) + secondsOfDay(rng));
  }
}

TEST(FastDateTest, fuzzDaysSinceEpochFromDateMatchesHinnant) {
  std::mt19937_64 rng{0xCAFEull};

  auto check = [](int32_t year, uint32_t month, uint32_t day) {
    auto got = util::daysSinceEpochFromDate(year, month, day);
    if (got.hasError()) {
      return;
    }
    const date::sys_days sd{
        date::year{year} / date::month{month} / date::day{day}};
    const int64_t want = sd.time_since_epoch().count();
    ASSERT_EQ(got.value(), want)
        << "y=" << year << " m=" << month << " d=" << day;
  };

  // Tight: realistic Velox years.
  std::uniform_int_distribution<int32_t> tightYear{1900, 2040};
  // Wide: algorithm domain.
  std::uniform_int_distribution<int32_t> wideYear{
      fast_date::kYearMin, fast_date::kYearMax};
  // Fallback: outside the algorithm range on both sides, exercising the
  // legacy 400-year step loops. Year stays within isValidDate's accepted
  // [kMinYear, kMaxYear] range.
  std::uniform_int_distribution<int32_t> fallbackPositiveYear{
      fast_date::kYearMax + 1, fast_date::kYearMax + 1'000'000};
  std::uniform_int_distribution<int32_t> fallbackNegativeYear{
      fast_date::kYearMin - 1'000'000, fast_date::kYearMin - 1};

  // Pick a valid day for the (year, month) pair so we don't trip
  // isValidDate's "31 Feb"-style rejections.
  auto sample = [&rng](int32_t year) {
    const uint32_t month = std::uniform_int_distribution<uint32_t>{1, 12}(rng);
    const date::year_month_day_last lastOfMonth{
        date::year{year} / date::month{month} / date::last};
    const uint32_t maxDay =
        static_cast<unsigned>(static_cast<date::day>(lastOfMonth.day()));
    const uint32_t day =
        std::uniform_int_distribution<uint32_t>{1, maxDay}(rng);
    return std::make_tuple(year, month, day);
  };

  for (int i = 0; i < 2'000'000; ++i) {
    auto [y, m, d] = sample(tightYear(rng));
    check(y, m, d);
  }
  for (int i = 0; i < 2'000'000; ++i) {
    auto [y, m, d] = sample(wideYear(rng));
    check(y, m, d);
  }
  for (int i = 0; i < 500'000; ++i) {
    auto [y, m, d] = sample(fallbackPositiveYear(rng));
    check(y, m, d);
  }
  for (int i = 0; i < 500'000; ++i) {
    auto [y, m, d] = sample(fallbackNegativeYear(rng));
    check(y, m, d);
  }
}

// Verifies that the patched fast path agrees with WideRangeDateConversion
// at every value within ±10 of the four range boundaries. An off-by-one
// in the dispatch range or in the algorithm's stated exact range would
// silently route a few inputs to the wrong branch — this catches that.
TEST(FastDateTest, fastEqualsWideRangeAtBoundaries) {
  // Forward direction: ±10 days around kRataDieMin and kRataDieMax.
  const int32_t forwardBoundaries[] = {
      fast_date::kRataDieMin, fast_date::kRataDieMax};
  for (int32_t boundary : forwardBoundaries) {
    for (int32_t delta = -10; delta <= 10; ++delta) {
      const int32_t days = boundary + delta;
      // Mix in non-trivial seconds-of-day: 12:34:56 UTC.
      const int64_t epoch =
          static_cast<int64_t>(days) * kSecondsPerDay + 45'296;
      std::tm fastTm;
      std::tm wideTm;
      ASSERT_TRUE(Timestamp::epochToCalendarUtc(epoch, fastTm))
          << "days=" << days;
      ASSERT_TRUE(WideRangeDateConversion::epochToCalendarUtc(epoch, wideTm))
          << "days=" << days;
      expectTmEqual(fastTm, wideTm, epoch);
    }
  }

  // Inverse direction: ±10 years around kYearMin and kYearMax. The date
  // is chosen per boundary to lie *one day outside* the algorithm's
  // exact range [Mar 1 kYearMin, Feb 28 kYearMax] at the boundary year:
  //   kYearMin → Feb 28 — one day before Mar 1 kYearMin
  //   kYearMax → Mar  1 — one day after  Feb 28 kYearMax
  // This way, a future relaxation of the dispatch from strict (`<`) to
  // inclusive (`<=`) would route a UB input to the fast path at either
  // boundary, and the fast/wide mismatch (or crash) would fail this
  // test. A single in-range date for both boundaries would only catch
  // the regression at one end.
  const struct {
    int32_t boundary;
    uint32_t month;
    uint32_t day;
  } inverseBoundaries[] = {
      {fast_date::kYearMin, 2u, 28u},
      {fast_date::kYearMax, 3u, 1u},
  };
  for (const auto& bound : inverseBoundaries) {
    for (int32_t delta = -10; delta <= 10; ++delta) {
      const int32_t year = bound.boundary + delta;
      const auto fastResult =
          util::daysSinceEpochFromDate(year, bound.month, bound.day);
      const auto wideResult = WideRangeDateConversion::daysSinceEpochFromDate(
          year, bound.month, bound.day);
      ASSERT_EQ(fastResult.hasError(), wideResult.hasError()) << "y=" << year;
      if (!fastResult.hasError()) {
        EXPECT_EQ(fastResult.value(), wideResult.value()) << "y=" << year;
      }
    }
  }
}

} // namespace
} // namespace facebook::velox::test
