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
#include <climits>
#include <ctime>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/RandomSeed.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

#ifdef _MSC_VER
// Days since 1970-01-01 for a proleptic-Gregorian date (month in [1,12]).
// Howard Hinnant's public-domain days_from_civil; mirrors the day count used
// by Timestamp::calendarUtcToEpoch so the shims below agree with it exactly.
static inline int64_t veloxDaysFromCivil(int64_t y, int64_t m, int64_t d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const int64_t yoe = y - era * 400;
  const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

// MSVC provides neither POSIX gmtime_r nor timegm. The CRT's _gmtime64_s /
// _mkgmtime only cover years 1970-3000 with no negative epochs, which is far
// narrower than Velox's full-range epoch<->calendar conversions. Provide
// full-range UTC replacements (mirroring Timestamp::epochToCalendarUtc /
// calendarUtcToEpoch) so the shared test bodies exercise the same domain on
// Windows as on POSIX glibc, rather than spuriously failing at the CRT limits.
inline time_t timegm(std::tm* tm) {
  int64_t year = static_cast<int64_t>(tm->tm_year) + 1900;
  int64_t month = tm->tm_mon;
  if (month > 11) {
    year += month / 12;
    month %= 12;
  } else if (month < 0) {
    const int64_t yearsDiff = (-month + 11) / 12;
    year -= yearsDiff;
    month += 12 * yearsDiff;
  }
  const int64_t days = veloxDaysFromCivil(year, month + 1, tm->tm_mday);
  return static_cast<time_t>(
      86400LL * days + 3600LL * tm->tm_hour + 60LL * tm->tm_min + tm->tm_sec);
}

inline std::tm* gmtime_r(const time_t* timer, std::tm* result) {
  const int64_t epoch = static_cast<int64_t>(*timer);
  int64_t days = epoch / 86400;
  int64_t rem = epoch % 86400;
  if (rem < 0) {
    rem += 86400;
    --days;
  }
  // civil_from_days: inverse of veloxDaysFromCivil.
  const int64_t z = days + 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const int64_t doe = z - era * 146097;
  const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int64_t y = yoe + era * 400;
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int64_t mp = (5 * doy + 2) / 153;
  const int64_t d = doy - (153 * mp + 2) / 5 + 1;
  const int64_t m = mp < 10 ? mp + 3 : mp - 9;
  const int64_t year = y + (m <= 2);
  const int64_t tmYear = year - 1900;
  // Match epochToCalendarUtc, which fails only when tm_year overflows int.
  if (tmYear > std::numeric_limits<int>::max() ||
      tmYear < std::numeric_limits<int>::min()) {
    return nullptr;
  }
  result->tm_year = static_cast<int>(tmYear);
  result->tm_mon = static_cast<int>(m - 1);
  result->tm_mday = static_cast<int>(d);
  result->tm_hour = static_cast<int>(rem / 3600);
  result->tm_min = static_cast<int>((rem % 3600) / 60);
  result->tm_sec = static_cast<int>(rem % 60);
  int64_t wday = (4 + days) % 7;
  if (wday < 0) {
    wday += 7;
  }
  result->tm_wday = static_cast<int>(wday);
  result->tm_yday = static_cast<int>(days - veloxDaysFromCivil(year, 1, 1));
  result->tm_isdst = 0;
  return result;
}
#endif

namespace facebook::velox {
namespace {

// Portable substitute for std::put_time limited to the %F and %T specifiers
// used by these tests. MSVC's std::put_time/strftime emits "?" for years
// outside a narrow supported range, whereas Velox's Timestamp formatting
// supports the full epoch range; format the fields manually on MSVC so the
// reference oracle matches production. POSIX keeps std::put_time unchanged for
// a genuinely independent reference.
std::string putTimePortable(const std::tm& tm, const char* format) {
#ifdef _MSC_VER
  const auto pad2 = [](int v) {
    auto s = std::to_string(v);
    return s.size() < 2 ? std::string(2 - s.size(), '0') + s : s;
  };
  std::string out;
  for (const char* p = format; *p != '\0'; ++p) {
    if (*p != '%' || *(p + 1) == '\0') {
      out += *p;
      continue;
    }
    switch (*++p) {
      case 'F': // ISO date: year-month-day
        out += std::to_string(tm.tm_year + 1900);
        out += '-';
        out += pad2(tm.tm_mon + 1);
        out += '-';
        out += pad2(tm.tm_mday);
        break;
      case 'T': // ISO time: hour:minute:second
        out += pad2(tm.tm_hour);
        out += ':';
        out += pad2(tm.tm_min);
        out += ':';
        out += pad2(tm.tm_sec);
        break;
      default:
        out += '%';
        out += *p;
    }
  }
  return out;
#else
  std::ostringstream oss;
  oss << std::put_time(&tm, format);
  return oss.str();
#endif
}

std::string timestampToString(
    Timestamp ts,
    const TimestampToStringOptions& options) {
  std::tm tm;
  Timestamp::epochToCalendarUtc(ts.getSeconds(), tm);
  std::string result;
  result.resize(getMaxStringLength(options));
  const auto view = Timestamp::tsToStringView(ts, options, result.data());
  result.resize(view.size());
  return result;
}

TEST(TimestampTest, fromDaysAndNanos) {
  EXPECT_EQ(
      Timestamp(Timestamp::kSecondsInDay + 2, 1),
      Timestamp::fromDaysAndNanos(
          Timestamp::kJulianToUnixEpochDays + 1,
          2 * Timestamp::kNanosInSecond + 1));
  EXPECT_EQ(
      Timestamp(Timestamp::kSecondsInDay + 2, 0),
      Timestamp::fromDaysAndNanos(
          Timestamp::kJulianToUnixEpochDays + 1,
          2 * Timestamp::kNanosInSecond));
  EXPECT_EQ(
      Timestamp(
          Timestamp::kSecondsInDay * 5 - 3, Timestamp::kNanosInSecond - 6),
      Timestamp::fromDaysAndNanos(
          Timestamp::kJulianToUnixEpochDays + 5,
          -2 * Timestamp::kNanosInSecond - 6));
  EXPECT_EQ(
      Timestamp(Timestamp::kSecondsInDay * 5 - 2, 0),
      Timestamp::fromDaysAndNanos(
          Timestamp::kJulianToUnixEpochDays + 5,
          -2 * Timestamp::kNanosInSecond));
}

TEST(TimestampTest, fromMillisAndMicros) {
  int64_t positiveSecond = 10'000;
  int64_t negativeSecond = -10'000;
  uint64_t nano = 123 * 1'000'000;

  Timestamp ts1(positiveSecond, nano);
  int64_t positiveMillis = positiveSecond * 1'000 + nano / 1'000'000;
  int64_t positiveMicros = positiveSecond * 1'000'000 + nano / 1000;
  EXPECT_EQ(ts1, Timestamp::fromMillis(positiveMillis));
  EXPECT_EQ(ts1, Timestamp::fromMicros(positiveMicros));
  EXPECT_EQ(ts1, Timestamp::fromMillis(ts1.toMillis()));
  EXPECT_EQ(ts1, Timestamp::fromMicros(ts1.toMicros()));

  Timestamp ts2(negativeSecond, nano);
  int64_t negativeMillis = negativeSecond * 1'000 + nano / 1'000'000;
  int64_t negativeMicros = negativeSecond * 1'000'000 + nano / 1000;
  EXPECT_EQ(ts2, Timestamp::fromMillis(negativeMillis));
  EXPECT_EQ(ts2, Timestamp::fromMicros(negativeMicros));
  EXPECT_EQ(ts2, Timestamp::fromMillis(ts2.toMillis()));
  EXPECT_EQ(ts2, Timestamp::fromMicros(ts2.toMicros()));

  Timestamp ts3(negativeSecond, 0);
  EXPECT_EQ(ts3, Timestamp::fromMillis(negativeSecond * 1'000));
  EXPECT_EQ(ts3, Timestamp::fromMicros(negativeSecond * 1'000'000));
  EXPECT_EQ(ts3, Timestamp::fromMillis(ts3.toMillis()));
  EXPECT_EQ(ts3, Timestamp::fromMicros(ts3.toMicros()));
}

TEST(TimestampTest, fromNanos) {
  int64_t positiveSecond = 10'000;
  int64_t negativeSecond = -10'000;
  uint64_t nano = 123'456'789;

  Timestamp ts1(positiveSecond, nano);
  int64_t positiveNanos = positiveSecond * 1'000'000'000 + nano;
  EXPECT_EQ(ts1, Timestamp::fromNanos(positiveNanos));
  EXPECT_EQ(ts1, Timestamp::fromNanos(ts1.toNanos()));

  Timestamp ts2(negativeSecond, nano);
  int64_t negativeNanos = negativeSecond * 1'000'000'000 + nano;
  EXPECT_EQ(ts2, Timestamp::fromNanos(negativeNanos));
  EXPECT_EQ(ts2, Timestamp::fromNanos(ts2.toNanos()));

  Timestamp ts3(negativeSecond, 0);
  EXPECT_EQ(ts3, Timestamp::fromNanos(negativeSecond * 1'000'000'000));
  EXPECT_EQ(ts3, Timestamp::fromNanos(ts3.toNanos()));
}

TEST(TimestampTest, arithmeticOverflow) {
  int64_t positiveSecond = Timestamp::kMaxSeconds;
  int64_t negativeSecond = Timestamp::kMinSeconds;
  uint64_t nano = Timestamp::kMaxNanos;

  Timestamp ts1(positiveSecond, nano);
  VELOX_ASSERT_THROW(
      ts1.toMillis(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to milliseconds",
          positiveSecond,
          nano));
  VELOX_ASSERT_THROW(
      ts1.toMicros(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to microseconds",
          positiveSecond,
          nano));
  VELOX_ASSERT_THROW(
      ts1.toNanos(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to nanoseconds",
          positiveSecond,
          nano));

  Timestamp ts2(negativeSecond, 0);
  VELOX_ASSERT_THROW(
      ts2.toMillis(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to milliseconds",
          negativeSecond,
          0));
  VELOX_ASSERT_THROW(
      ts2.toMicros(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to microseconds",
          negativeSecond,
          0));
  VELOX_ASSERT_THROW(
      ts2.toNanos(),
      fmt::format(
          "Could not convert Timestamp({}, {}) to nanoseconds",
          negativeSecond,
          0));
  ASSERT_NO_THROW(Timestamp::minMillis().toMillis());
  ASSERT_NO_THROW(Timestamp::maxMillis().toMillis());
  ASSERT_NO_THROW(Timestamp(-9223372036855, 224'192'000).toMicros());
  ASSERT_NO_THROW(Timestamp(9223372036854, 775'807'000).toMicros());
}

TEST(TimestampTest, toAppend) {
  std::string tsStringZeroValue;
  toAppend(Timestamp(0, 0), &tsStringZeroValue);
  EXPECT_EQ("1970-01-01T00:00:00.000000000", tsStringZeroValue);

  std::string tsStringCommonValue;
  toAppend(Timestamp(946729316, 0), &tsStringCommonValue);
  EXPECT_EQ("2000-01-01T12:21:56.000000000", tsStringCommonValue);

  std::string tsStringFarInFuture;
  toAppend(Timestamp(94668480000, 0), &tsStringFarInFuture);
  EXPECT_EQ("4969-12-04T00:00:00.000000000", tsStringFarInFuture);

  std::string tsStringWithNanos;
  toAppend(Timestamp(946729316, 123), &tsStringWithNanos);
  EXPECT_EQ("2000-01-01T12:21:56.000000123", tsStringWithNanos);

  EXPECT_EQ(
      "2000-01-01T00:00:00.000000000",
      folly::to<std::string>(Timestamp(946684800, 0)));
  EXPECT_EQ(
      "2000-01-01T12:21:56.000000123",
      folly::to<std::string>(Timestamp(946729316, 123)));
  EXPECT_EQ(
      "1970-01-01T02:01:06.000000000",
      folly::to<std::string>(Timestamp(7266, 0)));
  EXPECT_EQ(
      "2000-01-01T12:21:56.129900000",
      folly::to<std::string>(Timestamp(946729316, 129900000)));
}

TEST(TimestampTest, now) {
  using namespace std::chrono;

  auto now = Timestamp::now();

  auto expectedEpochSecs =
      duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  auto expectedEpochMs =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();

  EXPECT_GE(expectedEpochSecs, now.getSeconds());
  EXPECT_GE(expectedEpochMs, now.toMillis());
}

DEBUG_ONLY_TEST(TimestampTest, invalidInput) {
  constexpr uint64_t kUint64Max = std::numeric_limits<uint64_t>::max();
  constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();
  constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
  // Seconds invalid range.
  VELOX_ASSERT_THROW(Timestamp(kInt64Min, 1), "Timestamp seconds out of range");
  VELOX_ASSERT_THROW(Timestamp(kInt64Max, 1), "Timestamp seconds out of range");
  VELOX_ASSERT_THROW(
      Timestamp(Timestamp::kMinSeconds - 1, 1),
      "Timestamp seconds out of range");
  VELOX_ASSERT_THROW(
      Timestamp(Timestamp::kMaxSeconds + 1, 1),
      "Timestamp seconds out of range");

  // Nanos invalid range.
  VELOX_ASSERT_THROW(Timestamp(1, kUint64Max), "Timestamp nanos out of range");
  VELOX_ASSERT_THROW(
      Timestamp(1, Timestamp::kMaxNanos + 1), "Timestamp nanos out of range");
}

TEST(TimestampTest, toString) {
  auto kMin = Timestamp(Timestamp::kMinSeconds, 0);
  auto kMax = Timestamp(Timestamp::kMaxSeconds, Timestamp::kMaxNanos);
  EXPECT_EQ("-292275055-05-16T16:47:04.000000000", kMin.toString());
  EXPECT_EQ("292278994-08-17T07:12:55.999999999", kMax.toString());
  EXPECT_EQ(
      "1-01-01T05:17:32.000000000", Timestamp(-62135577748, 0).toString());
  EXPECT_EQ(
      "-224876953-12-19T16:58:03.000000000",
      Timestamp(-7096493348463717, 0).toString());
  EXPECT_EQ(
      "-1-11-29T19:33:20.000000000", Timestamp(-62170000000, 0).toString());
}

TEST(TimestampTest, toStringPrestoCastBehavior) {
  auto kMin = Timestamp(Timestamp::kMinSeconds, 0);
  auto kMax = Timestamp(Timestamp::kMaxSeconds, Timestamp::kMaxNanos);
  TimestampToStringOptions options = {
      .precision = TimestampToStringOptions::Precision::kMilliseconds,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };
  EXPECT_EQ("-292275055-05-16 16:47:04.000", kMin.toString(options));
  EXPECT_EQ("292278994-08-17 07:12:55.999", kMax.toString(options));
  EXPECT_EQ(
      "0001-01-01 05:17:32.000", Timestamp(-62135577748, 0).toString(options));
  EXPECT_EQ(
      "0000-03-24 13:20:00.000", Timestamp(-62160000000, 0).toString(options));
  EXPECT_EQ(
      "-224876953-12-19 16:58:03.000",
      Timestamp(-7096493348463717, 0).toString(options));
  EXPECT_EQ(
      "-0001-11-29 19:33:20.000", Timestamp(-62170000000, 0).toString(options));
}

namespace {

std::string toStringAlt(
    const Timestamp& t,
    TimestampToStringOptions::Precision precision) {
  auto seconds = t.getSeconds();
  std::tm tmValue;
  VELOX_CHECK_NOT_NULL(gmtime_r((const time_t*)&seconds, &tmValue));
  auto width = static_cast<int>(precision);
  auto value = precision == TimestampToStringOptions::Precision::kMilliseconds
      ? t.getNanos() / 1'000'000
      : t.getNanos();
  std::ostringstream oss;
  oss << putTimePortable(tmValue, "%FT%T");
  oss << '.' << std::setfill('0') << std::setw(width) << value;
  return oss.str();
}

bool checkUtcToEpoch(int year, int mon, int mday, int hour, int min, int sec) {
  SCOPED_TRACE(
      fmt::format(
          "{}-{:02}-{:02} {:02}:{:02}:{:02}", year, mon, mday, hour, min, sec));
  std::tm tm{};
  tm.tm_sec = sec;
  tm.tm_min = min;
  tm.tm_hour = hour;
  tm.tm_mday = mday;
  tm.tm_mon = mon;
  tm.tm_year = year;
  errno = 0;
  auto expected = timegm(&tm);
  bool error = expected == -1 && errno != 0;
  auto actual = Timestamp::calendarUtcToEpoch(tm);
  if (!error) {
    EXPECT_EQ(actual, expected);
  }
  return !error;
}

} // namespace

TEST(TimestampTest, compareWithToStringAlt) {
  std::default_random_engine gen(common::testutil::getRandomSeed(42));
  std::uniform_int_distribution<int64_t> distSec(
      Timestamp::kMinSeconds, Timestamp::kMaxSeconds);
  std::uniform_int_distribution<uint64_t> distNano(0, Timestamp::kMaxNanos);
  for (int i = 0; i < 10'000; ++i) {
    Timestamp t(distSec(gen), distNano(gen));
    for (auto precision :
         {TimestampToStringOptions::Precision::kMilliseconds,
          TimestampToStringOptions::Precision::kNanoseconds}) {
      TimestampToStringOptions options{};
      options.precision = precision;
      ASSERT_EQ(t.toString(options), toStringAlt(t, precision))
          << t.getSeconds() << ' ' << t.getNanos();
    }
  }
}

TEST(TimestampTest, utcToEpoch) {
  ASSERT_TRUE(checkUtcToEpoch(1970, 1, 1, 0, 0, 0));
  ASSERT_TRUE(checkUtcToEpoch(2001, 11, 12, 18, 31, 1));
  ASSERT_TRUE(checkUtcToEpoch(1969, 12, 31, 23, 59, 59));
  ASSERT_TRUE(checkUtcToEpoch(1969, 12, 31, 23, 59, 58));
  ASSERT_TRUE(checkUtcToEpoch(INT32_MAX, 11, 30, 23, 59, 59));
  ASSERT_TRUE(checkUtcToEpoch(INT32_MIN, 1, 1, 0, 0, 0));
  ASSERT_TRUE(checkUtcToEpoch(
      INT32_MAX - INT32_MAX / 11,
      INT32_MAX,
      INT32_MAX,
      INT32_MAX,
      INT32_MAX,
      INT32_MAX));
  ASSERT_TRUE(checkUtcToEpoch(
      INT32_MIN - INT32_MIN / 11,
      INT32_MIN,
      INT32_MIN,
      INT32_MIN,
      INT32_MIN,
      INT32_MIN));
}

TEST(TimestampTest, utcToEpochRandomInputs) {
  std::default_random_engine gen(common::testutil::getRandomSeed(42));
  std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
  for (int i = 0; i < 10'000; ++i) {
    checkUtcToEpoch(
        dist(gen), dist(gen), dist(gen), dist(gen), dist(gen), dist(gen));
  }
}

TEST(TimestampTest, increaseOperator) {
  auto ts = Timestamp(0, 999999998);
  EXPECT_EQ("1970-01-01T00:00:00.999999998", ts.toString());
  ++ts;
  EXPECT_EQ("1970-01-01T00:00:00.999999999", ts.toString());
  ++ts;
  EXPECT_EQ("1970-01-01T00:00:01.000000000", ts.toString());
  ++ts;
  EXPECT_EQ("1970-01-01T00:00:01.000000001", ts.toString());
  ++ts;
  EXPECT_EQ("1970-01-01T00:00:01.000000002", ts.toString());

  auto kMax = Timestamp(Timestamp::kMaxSeconds, Timestamp::kMaxNanos);
  VELOX_ASSERT_THROW(++kMax, "Timestamp nanos out of range");
}

TEST(TimestampTest, decreaseOperator) {
  auto ts = Timestamp(0, 2);
  EXPECT_EQ("1970-01-01T00:00:00.000000002", ts.toString());
  --ts;
  EXPECT_EQ("1970-01-01T00:00:00.000000001", ts.toString());
  --ts;
  EXPECT_EQ("1970-01-01T00:00:00.000000000", ts.toString());
  --ts;
  EXPECT_EQ("1969-12-31T23:59:59.999999999", ts.toString());
  --ts;
  EXPECT_EQ("1969-12-31T23:59:59.999999998", ts.toString());

  auto kMin = Timestamp(Timestamp::kMinSeconds, 0);
  VELOX_ASSERT_THROW(--kMin, "Timestamp nanos out of range");
}

// In debug mode, Timestamp constructor will throw exception if range check
// fails.
#ifdef NDEBUG
TEST(TimestampTest, overflow) {
  Timestamp t(std::numeric_limits<int64_t>::max(), 0);
  VELOX_ASSERT_THROW(
      t.toTimePointMs(false),
      fmt::format(
          "Could not convert Timestamp({}, {}) to milliseconds",
          std::numeric_limits<int64_t>::max(),
          0));
  ASSERT_NO_THROW(t.toTimePointMs(true));
}
#endif

void checkTm(const std::tm& actual, const std::tm& expected) {
  ASSERT_EQ(expected.tm_year, actual.tm_year);
  ASSERT_EQ(expected.tm_yday, actual.tm_yday);
  ASSERT_EQ(expected.tm_mon, actual.tm_mon);
  ASSERT_EQ(expected.tm_mday, actual.tm_mday);
  ASSERT_EQ(expected.tm_wday, actual.tm_wday);
  ASSERT_EQ(expected.tm_hour, actual.tm_hour);
  ASSERT_EQ(expected.tm_min, actual.tm_min);
  ASSERT_EQ(expected.tm_sec, actual.tm_sec);
}

std::string tmToString(
    const std::tm& tmValue,
    uint64_t nanos,
    const std::string& format,
    const TimestampToStringOptions& options) {
  auto width = static_cast<int>(options.precision);
  auto value =
      options.precision == TimestampToStringOptions::Precision::kMilliseconds
      ? nanos / 1'000'000
      : nanos;

  std::ostringstream oss;
  oss << putTimePortable(tmValue, format.c_str());

  if (options.mode != TimestampToStringOptions::Mode::kDateOnly) {
    oss << '.' << std::setfill('0') << std::setw(width) << value;
  }

  return oss.str();
}

TEST(TimestampTest, epochToUtc) {
  std::tm tm{};
  ASSERT_FALSE(Timestamp::epochToCalendarUtc(-(1ll << 60), tm));
  ASSERT_FALSE(Timestamp::epochToCalendarUtc(1ll << 60, tm));
}

TEST(TimestampTest, randomEpochToUtc) {
  uint64_t seed = 42;
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<time_t> dist(
      std::numeric_limits<time_t>::min(), std::numeric_limits<time_t>::max());
  std::tm actual{};
  std::tm expected{};
  for (int i = 0; i < 10'000; ++i) {
    auto epoch = dist(gen);
    SCOPED_TRACE(fmt::format("epoch={}", epoch));
    if (gmtime_r(&epoch, &expected)) {
      ASSERT_TRUE(Timestamp::epochToCalendarUtc(epoch, actual));
      checkTm(actual, expected);
    } else {
      ASSERT_FALSE(Timestamp::epochToCalendarUtc(epoch, actual));
    }
  }
}

void testTmToString(
    const std::string& format,
    const TimestampToStringOptions::Mode mode) {
  uint64_t seed = 42;
  std::default_random_engine gen(seed);

  std::uniform_int_distribution<time_t> dist(
      std::numeric_limits<time_t>::min(), std::numeric_limits<time_t>::max());
  std::uniform_int_distribution<int> nanosDist(0, Timestamp::kMaxNanos);

  std::tm actual{};
  std::tm expected{};

  TimestampToStringOptions options;
  options.mode = mode;

  const std::vector<TimestampToStringOptions::Precision> precisions = {
      TimestampToStringOptions::Precision::kMilliseconds,
      TimestampToStringOptions::Precision::kNanoseconds};

  for (auto precision : precisions) {
    options.precision = precision;
    for (int i = 0; i < 10'000; ++i) {
      auto epoch = dist(gen);
      auto nanos = nanosDist(gen);
      SCOPED_TRACE(
          fmt::format(
              "epoch={}, nanos={}, mode={}, precision={}",
              epoch,
              nanos,
              mode,
              precision));
      if (gmtime_r(&epoch, &expected)) {
        ASSERT_TRUE(Timestamp::epochToCalendarUtc(epoch, actual));
        checkTm(actual, expected);

        std::string actualString;
        actualString.resize(getMaxStringLength(options));
        const auto view = Timestamp::tmToStringView(
            actual, nanos, options, actualString.data());
        actualString.resize(view.size());
        auto expectedString = tmToString(expected, nanos, format, options);
        ASSERT_EQ(expectedString, actualString);
      } else {
        ASSERT_FALSE(Timestamp::epochToCalendarUtc(epoch, actual));
      }
    }
  }
}

TEST(TimestampTest, tmToStringDateOnly) {
  // %F - equivalent to "%Y-%m-%d" (the ISO 8601 date format)
  testTmToString("%F", TimestampToStringOptions::Mode::kDateOnly);
}

TEST(TimestampTest, tmToStringTimeOnly) {
  // %T - equivalent to "%H:%M:%S" (the ISO 8601 time format)
  testTmToString("%T", TimestampToStringOptions::Mode::kTimeOnly);
}

TEST(TimestampTest, tmToStringTimestamp) {
  // %FT%T - equivalent to "%Y-%m-%dT%H:%M:%S" (the ISO 8601 timestamp format)
  testTmToString("%FT%T", TimestampToStringOptions::Mode::kFull);
}

TEST(TimestampTest, leadingPositiveSign) {
  TimestampToStringOptions options = {
      .leadingPositiveSign = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };

  ASSERT_EQ(
      timestampToString(Timestamp(253402231016, 0), options),
      "9999-12-31 04:36:56.000000000");
  ASSERT_EQ(
      timestampToString(Timestamp(253405036800, 0), options),
      "+10000-02-01 16:00:00.000000000");
}

TEST(TimestampTest, skipTrailingZeros) {
  TimestampToStringOptions options = {
      .precision = TimestampToStringOptions::Precision::kMicroseconds,
      .skipTrailingZeros = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };

  ASSERT_EQ(
      timestampToString(Timestamp(-946684800, 0), options),
      "1940-01-02 00:00:00");
  ASSERT_EQ(timestampToString(Timestamp(0, 0), options), "1970-01-01 00:00:00");
  ASSERT_EQ(
      timestampToString(Timestamp(0, 365), options), "1970-01-01 00:00:00");
  ASSERT_EQ(
      timestampToString(Timestamp(0, 65873), options),
      "1970-01-01 00:00:00.000065");
  ASSERT_EQ(
      timestampToString(Timestamp(94668480000, 0), options),
      "4969-12-04 00:00:00");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129999999), options),
      "2000-01-01 12:21:56.129999");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129990000), options),
      "2000-01-01 12:21:56.12999");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129900000), options),
      "2000-01-01 12:21:56.1299");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129000000), options),
      "2000-01-01 12:21:56.129");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129010000), options),
      "2000-01-01 12:21:56.12901");
  ASSERT_EQ(
      timestampToString(Timestamp(946729316, 129001000), options),
      "2000-01-01 12:21:56.129001");
  ASSERT_EQ(
      timestampToString(Timestamp(-50049331200, 726600000), options),
      "0384-01-01 08:00:00.7266");
}

} // namespace
} // namespace facebook::velox
