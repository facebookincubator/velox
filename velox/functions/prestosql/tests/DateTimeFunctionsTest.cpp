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

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/external/tzdb/zoned_time.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class DateTimeFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  std::string daysShort[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};

  std::string daysLong[7] = {
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday"};

  std::string monthsShort[12] = {
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec"};

  std::string monthsLong[12] = {
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December"};

  static std::string padNumber(int number) {
    return number < 10 ? "0" + std::to_string(number) : std::to_string(number);
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  void disableAdjustTimestampToTimezone() {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
    });
  }

  void setQuerySessionStartTime(int64_t sessionStartTime) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionStartTime,
         std::to_string(sessionStartTime)},
    });
  }

 public:
  // Helper class to manipulate timestamp with timezone types in tests. Provided
  // only for convenience.
  struct TimestampWithTimezone {
    TimestampWithTimezone(int64_t milliSeconds, std::string_view timezoneName)
        : milliSeconds_(milliSeconds),
          timezone_(tz::locateZone(timezoneName)) {}

    TimestampWithTimezone(int64_t milliSeconds, const tz::TimeZone* timezone)
        : milliSeconds_(milliSeconds), timezone_(timezone) {}

    int64_t milliSeconds_{0};
    const tz::TimeZone* timezone_;

    static std::optional<int64_t> pack(
        std::optional<TimestampWithTimezone> zone) {
      if (zone.has_value()) {
        return facebook::velox::pack(
            zone->milliSeconds_, zone->timezone_->id());
      }
      return std::nullopt;
    }

    static std::optional<TimestampWithTimezone> unpack(
        std::optional<int64_t> input) {
      if (input.has_value()) {
        return TimestampWithTimezone(
            unpackMillisUtc(input.value()),
            tz::getTimeZoneName(unpackZoneKeyId(input.value())));
      }
      return std::nullopt;
    }

    // Provides a nicer printer for gtest.
    friend std::ostream& operator<<(
        std::ostream& os,
        const TimestampWithTimezone& in) {
      return os << "TimestampWithTimezone(milliSeconds: " << in.milliSeconds_
                << ", timezone: " << *in.timezone_ << ")";
    }
  };

  VectorPtr makeTimestampWithTimeZoneVector(int64_t timestamp, const char* tz) {
    auto tzid = tz::getTimeZoneID(tz);

    return makeNullableFlatVector<int64_t>(
        {pack(timestamp, tzid)}, TIMESTAMP_WITH_TIME_ZONE());
  }

  VectorPtr makeTimestampWithTimeZoneVector(
      vector_size_t size,
      const std::function<int64_t(int32_t row)>& timestampAt,
      const std::function<int16_t(int32_t row)>& timezoneAt) {
    return makeFlatVector<int64_t>(
        size,
        [&](int32_t index) {
          return pack(timestampAt(index), timezoneAt(index));
        },
        nullptr,
        TIMESTAMP_WITH_TIME_ZONE());
  }

  int32_t getCurrentDate(const std::optional<std::string>& timeZone) {
    return parseDate(
        date::format(
            "%Y-%m-%d",
            timeZone.has_value()
                ? tzdb::zoned_time(
                      timeZone.value(), std::chrono::system_clock::now())
                : tzdb::zoned_time(std::chrono::system_clock::now())));
  }
};

bool operator==(
    const DateTimeFunctionsTest::TimestampWithTimezone& a,
    const DateTimeFunctionsTest::TimestampWithTimezone& b) {
  return a.milliSeconds_ == b.milliSeconds_ &&
      a.timezone_->id() == b.timezone_->id();
}

TEST_F(DateTimeFunctionsTest, dateTruncSignatures) {
  auto signatures = getSignatureStrings("date_trunc");
  ASSERT_EQ(4, signatures.size());

  ASSERT_EQ(
      1,
      signatures.count(
          "(varchar,timestamp with time zone) -> timestamp with time zone"));
  ASSERT_EQ(1, signatures.count("(varchar,date) -> date"));
  ASSERT_EQ(1, signatures.count("(varchar,timestamp) -> timestamp"));
  ASSERT_EQ(1, signatures.count("(varchar,time) -> time"));
}

TEST_F(DateTimeFunctionsTest, parseDatetimeSignatures) {
  auto signatures = getSignatureStrings("parse_datetime");
  ASSERT_EQ(1, signatures.size());

  ASSERT_EQ(
      1, signatures.count("(varchar,varchar) -> timestamp with time zone"));
}

TEST_F(DateTimeFunctionsTest, dayOfXxxSignatures) {
  for (const auto& name : {"day", "day_of_month"}) {
    SCOPED_TRACE(name);
    auto signatures = getSignatureStrings(name);
    ASSERT_EQ(4, signatures.size());

    ASSERT_EQ(1, signatures.count("(timestamp with time zone) -> bigint"));
    ASSERT_EQ(1, signatures.count("(date) -> bigint"));
    ASSERT_EQ(1, signatures.count("(timestamp) -> bigint"));
    ASSERT_EQ(1, signatures.count("(interval day to second) -> bigint"));
  }

  for (const auto& name : {"day_of_year", "doy", "day_of_week", "dow"}) {
    SCOPED_TRACE(name);
    auto signatures = getSignatureStrings(name);
    ASSERT_EQ(3, signatures.size());

    ASSERT_EQ(1, signatures.count("(timestamp with time zone) -> bigint"));
    ASSERT_EQ(1, signatures.count("(date) -> bigint"));
    ASSERT_EQ(1, signatures.count("(timestamp) -> bigint"));
  }
}

// Test cases from PrestoDB [1] are covered here as well:
// Timestamp(998474645, 321000000) from "TIMESTAMP '2001-08-22 03:04:05.321'"
// Timestamp(998423705, 321000000) from "TIMESTAMP '2001-08-22 03:04:05.321
// +07:09'"
// [1]https://github.com/prestodb/presto/blob/master/presto-main/src/test/java/com/facebook/presto/operator/scalar/TestDateTimeFunctionsBase.java
TEST_F(DateTimeFunctionsTest, toUnixtime) {
  const auto toUnixtime = [&](std::optional<Timestamp> t) {
    return evaluateOnce<double>("to_unixtime(c0)", t);
  };

  EXPECT_EQ(0, toUnixtime(Timestamp(0, 0)));
  EXPECT_EQ(-0.999991, toUnixtime(Timestamp(-1, 9000)));
  EXPECT_EQ(4000000000, toUnixtime(Timestamp(4000000000, 0)));
  EXPECT_EQ(4000000000.123, toUnixtime(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(-9999999998.9, toUnixtime(Timestamp(-9999999999, 100000000)));
  EXPECT_EQ(998474645.321, toUnixtime(Timestamp(998474645, 321000000)));
  EXPECT_EQ(998423705.321, toUnixtime(Timestamp(998423705, 321000000)));

  const auto toUnixtimeWTZ = [&](int64_t timestamp, const char* tz) {
    auto input = makeTimestampWithTimeZoneVector(timestamp, tz);

    return evaluateOnce<double>("to_unixtime(c0)", makeRowVector({input}));
  };

  // 1639426440000 is milliseconds (from PrestoDb '2021-12-13+20:14+00:00').
  EXPECT_EQ(0, toUnixtimeWTZ(0, "+00:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "+00:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "+03:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "+04:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "-07:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "-00:01"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "+00:01"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "-14:00"));
  EXPECT_EQ(1639426440, toUnixtimeWTZ(1639426440000, "+14:00"));

  // test floating point and negative time
  EXPECT_EQ(16394.26, toUnixtimeWTZ(16394260, "+00:00"));
  EXPECT_EQ(16394.26, toUnixtimeWTZ(16394260, "+03:00"));
  EXPECT_EQ(16394.26, toUnixtimeWTZ(16394260, "-07:00"));
  EXPECT_EQ(-16394.26, toUnixtimeWTZ(-16394260, "+12:00"));
  EXPECT_EQ(-16394.26, toUnixtimeWTZ(-16394260, "-06:00"));
}

TEST_F(DateTimeFunctionsTest, fromUnixtimeRountTrip) {
  const auto testRoundTrip = [&](std::optional<Timestamp> t) {
    auto r = evaluateOnce<Timestamp>("from_unixtime(to_unixtime(c0))", t);
    EXPECT_EQ(r->getSeconds(), t->getSeconds()) << "at " << t->toString();
    EXPECT_NEAR(r->getNanos(), t->getNanos(), 1'000) << "at " << t->toString();
    return r;
  };

  testRoundTrip(Timestamp(0, 0));
  testRoundTrip(Timestamp(-1, 9000000));
  testRoundTrip(Timestamp(4000000000, 0));
  testRoundTrip(Timestamp(4000000000, 123000000));
  testRoundTrip(Timestamp(-9999999999, 100000000));
  testRoundTrip(Timestamp(998474645, 321000000));
  testRoundTrip(Timestamp(998423705, 321000000));
}

TEST_F(DateTimeFunctionsTest, fromUnixtimeWithTimeZone) {
  const auto fromUnixtime = [&](std::optional<double> timestamp,
                                std::optional<std::string> timezoneName) {
    return TimestampWithTimezone::unpack(
        evaluateOnce<int64_t>(
            "from_unixtime(c0, c1)", timestamp, timezoneName));
  };

  // Check null behavior.
  EXPECT_EQ(fromUnixtime(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(fromUnixtime(std::nullopt, "UTC"), std::nullopt);
  EXPECT_EQ(fromUnixtime(0, std::nullopt), std::nullopt);

  EXPECT_EQ(
      fromUnixtime(1631800000.12345, "-01:00"),
      TimestampWithTimezone(1631800000123, "-01:00"));
  EXPECT_EQ(
      fromUnixtime(123.99, "America/Los_Angeles"),
      TimestampWithTimezone(123990, "America/Los_Angeles"));
  EXPECT_EQ(
      fromUnixtime(1667721600.1, "UTC"),
      TimestampWithTimezone(1667721600100, "UTC"));
  EXPECT_EQ(
      fromUnixtime(123, "UTC+1"), TimestampWithTimezone(123000, "+01:00"));
  EXPECT_EQ(
      fromUnixtime(123, "GMT-2"), TimestampWithTimezone(123000, "-02:00"));
  EXPECT_EQ(
      fromUnixtime(123, "UT+14"), TimestampWithTimezone(123000, "+14:00"));
  EXPECT_EQ(
      fromUnixtime(123, "Etc/UTC-8"), TimestampWithTimezone(123000, "-08:00"));
  EXPECT_EQ(
      fromUnixtime(123, "Etc/GMT+5"), TimestampWithTimezone(123000, "-05:00"));
  EXPECT_EQ(
      fromUnixtime(123, "Etc/UT-14"), TimestampWithTimezone(123000, "-14:00"));

  // Nan.
  static const double kNan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(fromUnixtime(kNan, "-04:36"), TimestampWithTimezone(0, "-04:36"));

  // Check rounding behavior.
  EXPECT_EQ(
      fromUnixtime(1.7300479933495E9, "America/Costa_Rica"),
      TimestampWithTimezone(1730047993350, "America/Costa_Rica"));

  // Maximum timestamp.
  EXPECT_EQ(
      fromUnixtime(2251799813685.247, "GMT"),
      TimestampWithTimezone(2251799813685247, "GMT"));
  EXPECT_EQ(
      fromUnixtime(2251799813685.247, "America/Costa_Rica"),
      TimestampWithTimezone(2251799813685247, "America/Costa_Rica"));
  // Minimum timestamp.
  EXPECT_EQ(
      fromUnixtime(-2251799813685.248, "GMT"),
      TimestampWithTimezone(-2251799813685248, "GMT"));
  EXPECT_EQ(
      fromUnixtime(-2251799813685.248, "America/Costa_Rica"),
      TimestampWithTimezone(-2251799813685248, "America/Costa_Rica"));

  // Test overflow in either direction.
  VELOX_ASSERT_THROW(
      fromUnixtime(2251799813685.248, "GMT"), "TimestampWithTimeZone overflow");
  VELOX_ASSERT_THROW(
      fromUnixtime(-2251799813685.249, "GMT"),
      "TimestampWithTimeZone overflow");
}

TEST_F(DateTimeFunctionsTest, fromUnixtimeTzOffset) {
  auto fromOffset = [&](std::optional<double> epoch,
                        std::optional<int64_t> hours,
                        std::optional<int64_t> minutes) {
    auto result = evaluateOnce<int64_t>(
        "from_unixtime(c0, c1, c2)", epoch, hours, minutes);

    auto otherResult = evaluateOnce<int64_t>(
        fmt::format("from_unixtime(c0, {}, {})", *hours, *minutes), epoch);

    VELOX_CHECK_EQ(result.value(), otherResult.value());
    return TimestampWithTimezone::unpack(result.value());
  };

  EXPECT_EQ(TimestampWithTimezone(123'450, "UTC"), fromOffset(123.45, 0, 0));
  EXPECT_EQ(
      TimestampWithTimezone(123'450, "+05:30"), fromOffset(123.45, 5, 30));
  EXPECT_EQ(
      TimestampWithTimezone(123'450, "-08:00"), fromOffset(123.45, -8, 0));

  EXPECT_THROW(
      fromOffset(
          123.45,
          std::numeric_limits<int64_t>::max(),
          std::numeric_limits<int64_t>::max()),
      VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, fromUnixtime) {
  const auto fromUnixtime = [&](std::optional<double> t) {
    return evaluateOnce<Timestamp>("from_unixtime(c0)", t);
  };

  static const double kInf = std::numeric_limits<double>::infinity();
  static const double kNan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_EQ(Timestamp(0, 0), fromUnixtime(0));
  EXPECT_EQ(Timestamp(-1, 9000000), fromUnixtime(-0.991));
  EXPECT_EQ(Timestamp(1, 0), fromUnixtime(1 - 1e-10));
  EXPECT_EQ(Timestamp(4000000000, 0), fromUnixtime(4000000000));
  EXPECT_EQ(
      Timestamp(9'223'372'036'854'775, 807'000'000), fromUnixtime(3.87111e+37));
  EXPECT_EQ(Timestamp(4000000000, 123000000), fromUnixtime(4000000000.123));
  EXPECT_EQ(Timestamp(9'223'372'036'854'775, 807'000'000), fromUnixtime(kInf));
  EXPECT_EQ(
      Timestamp(-9'223'372'036'854'776, 192'000'000), fromUnixtime(-kInf));
  EXPECT_EQ(Timestamp(0, 0), fromUnixtime(kNan));
}

TEST_F(DateTimeFunctionsTest, year) {
  const auto year = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1970, year(Timestamp(0, 0)));
  EXPECT_EQ(1969, year(Timestamp(-1, 9000)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 0)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2001, year(Timestamp(998474645, 321000000)));
  EXPECT_EQ(2001, year(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1969, year(Timestamp(0, 0)));
  EXPECT_EQ(1969, year(Timestamp(-1, Timestamp::kMaxNanos)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 0)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2001, year(Timestamp(998474645, 321000000)));
  EXPECT_EQ(2001, year(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, yearDate) {
  const auto year = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("year(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1970, year(parseDate("1970-01-01")));
  EXPECT_EQ(1969, year(parseDate("1969-12-31")));
  EXPECT_EQ(2020, year(parseDate("2020-01-01")));
  EXPECT_EQ(1920, year(parseDate("1920-12-31")));
}

TEST_F(DateTimeFunctionsTest, yearTimestampWithTimezone) {
  const auto yearTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "year(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      1969, yearTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      1970, yearTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      1973,
      yearTimestampWithTimezone(TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      1966,
      yearTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      2001,
      yearTimestampWithTimezone(TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      1938,
      yearTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, yearTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, weekDate) {
  const auto weekDate = [&](const char* dateString) {
    auto date = std::make_optional(parseDate(dateString));
    auto week = evaluateOnce<int64_t>("week(c0)", DATE(), date).value();
    auto weekOfYear =
        evaluateOnce<int64_t>("week_of_year(c0)", DATE(), date).value();
    VELOX_CHECK_EQ(
        week, weekOfYear, "week and week_of_year must return the same value");
    return week;
  };

  EXPECT_EQ(1, weekDate("1919-12-31"));
  EXPECT_EQ(1, weekDate("1920-01-01"));
  EXPECT_EQ(1, weekDate("1920-01-04"));
  EXPECT_EQ(2, weekDate("1920-01-05"));
  EXPECT_EQ(53, weekDate("1960-01-01"));
  EXPECT_EQ(53, weekDate("1960-01-03"));
  EXPECT_EQ(1, weekDate("1960-01-04"));
  EXPECT_EQ(1, weekDate("1969-12-31"));
  EXPECT_EQ(1, weekDate("1970-01-01"));
  EXPECT_EQ(1, weekDate("0001-01-01"));
  EXPECT_EQ(52, weekDate("9999-12-31"));

  // Test various cases where the last week of the previous year extends into
  // the next year.

  // Leap year that ends on Thursday.
  EXPECT_EQ(53, weekDate("2021-01-01"));
  // Leap year that ends on Friday.
  EXPECT_EQ(53, weekDate("2005-01-01"));
  // Leap year that ends on Saturday.
  EXPECT_EQ(52, weekDate("2017-01-01"));
  // Common year that ends on Thursday.
  EXPECT_EQ(53, weekDate("2016-01-01"));
  // Common year that ends on Friday.
  EXPECT_EQ(52, weekDate("2022-01-01"));
  // Common year that ends on Saturday.
  EXPECT_EQ(52, weekDate("2023-01-01"));
}

TEST_F(DateTimeFunctionsTest, week) {
  const auto weekTimestamp = [&](std::string_view time) {
    auto ts = util::fromTimestampString(
                  time.data(), time.size(), util::TimestampParseMode::kIso8601)
                  .thenOrThrow(folly::identity, [&](const Status& status) {
                    VELOX_USER_FAIL("{}", status.message());
                  });
    auto timestamp =
        std::make_optional(Timestamp(ts.getSeconds() * 100'000, 0));

    auto week = evaluateOnce<int64_t>("week(c0)", timestamp).value();
    auto weekOfYear =
        evaluateOnce<int64_t>("week_of_year(c0)", timestamp).value();
    VELOX_CHECK_EQ(
        week, weekOfYear, "week and week_of_year must return the same value");
    return week;
  };

  EXPECT_EQ(1, weekTimestamp("T00:00:00"));
  EXPECT_EQ(47, weekTimestamp("T11:59:59"));
  EXPECT_EQ(33, weekTimestamp("T06:01:01"));
  EXPECT_EQ(44, weekTimestamp("T06:59:59"));
  EXPECT_EQ(47, weekTimestamp("T12:00:01"));
  EXPECT_EQ(16, weekTimestamp("T12:59:59"));
}

TEST_F(DateTimeFunctionsTest, weekTimestampWithTimezone) {
  const auto weekTimestampTimezone = [&](std::string_view time,
                                         const char* timezone) {
    auto ts = util::fromTimestampString(
                  time.data(), time.size(), util::TimestampParseMode::kIso8601)
                  .thenOrThrow(folly::identity, [&](const Status& status) {
                    VELOX_USER_FAIL("{}", status.message());
                  });

    auto timestamp = ts.getSeconds() * 100'000'000;
    auto week = *evaluateOnce<int64_t>(
        "week(c0)",
        TIMESTAMP_WITH_TIME_ZONE(),
        TimestampWithTimezone::pack(
            TimestampWithTimezone(timestamp, timezone)));
    auto weekOfYear = *evaluateOnce<int64_t>(
        "week_of_year(c0)",
        TIMESTAMP_WITH_TIME_ZONE(),
        TimestampWithTimezone::pack(
            TimestampWithTimezone(timestamp, timezone)));
    VELOX_CHECK_EQ(
        week, weekOfYear, "week and week_of_year must return the same value");
    return week;
  };

  EXPECT_EQ(1, weekTimestampTimezone("T00:00:00", "-12:00"));
  EXPECT_EQ(1, weekTimestampTimezone("T00:00:00", "+12:00"));
  EXPECT_EQ(47, weekTimestampTimezone("T11:59:59", "-12:00"));
  EXPECT_EQ(47, weekTimestampTimezone("T11:59:59", "+12:00"));
  EXPECT_EQ(33, weekTimestampTimezone("T06:01:01", "-12:00"));
  EXPECT_EQ(34, weekTimestampTimezone("T06:01:01", "+12:00"));
  EXPECT_EQ(47, weekTimestampTimezone("T12:00:01", "-12:00"));
  EXPECT_EQ(47, weekTimestampTimezone("T12:00:01", "+12:00"));
}

TEST_F(DateTimeFunctionsTest, quarter) {
  const auto quarter = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("quarter(c0)", date);
  };
  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(1, quarter(Timestamp(0, 0)));
  EXPECT_EQ(4, quarter(Timestamp(-1, 9000)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 0)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2, quarter(Timestamp(990000000, 321000000)));
  EXPECT_EQ(3, quarter(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(4, quarter(Timestamp(0, 0)));
  EXPECT_EQ(4, quarter(Timestamp(-1, Timestamp::kMaxNanos)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 0)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2, quarter(Timestamp(990000000, 321000000)));
  EXPECT_EQ(3, quarter(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, quarterDate) {
  const auto quarter = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("quarter(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(1, quarter(0));
  EXPECT_EQ(4, quarter(-1));
  EXPECT_EQ(4, quarter(-40));
  EXPECT_EQ(2, quarter(110));
  EXPECT_EQ(3, quarter(200));
  EXPECT_EQ(1, quarter(18262));
  EXPECT_EQ(1, quarter(-18262));
}

TEST_F(DateTimeFunctionsTest, quarterTimestampWithTimezone) {
  const auto quarterTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "quarter(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      4, quarterTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      1, quarterTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      4,
      quarterTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      1,
      quarterTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      2,
      quarterTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      3,
      quarterTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, quarterTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, month) {
  const auto month = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(1, month(Timestamp(0, 0)));
  EXPECT_EQ(12, month(Timestamp(-1, 9000)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 0)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(8, month(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, month(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(12, month(Timestamp(0, 0)));
  EXPECT_EQ(12, month(Timestamp(-1, Timestamp::kMaxNanos)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 0)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(8, month(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, month(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, monthDate) {
  const auto month = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("month(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(1, month(0));
  EXPECT_EQ(12, month(-1));
  EXPECT_EQ(11, month(-40));
  EXPECT_EQ(2, month(40));
  EXPECT_EQ(1, month(18262));
  EXPECT_EQ(1, month(-18262));
}

TEST_F(DateTimeFunctionsTest, monthTimestampWithTimezone) {
  const auto monthTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "month(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(12, monthTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(1, monthTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      11,
      monthTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      2,
      monthTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      4,
      monthTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      9,
      monthTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, monthTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, hour) {
  const auto hour = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("hour(c0)", date);
  };
  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(0, hour(Timestamp(0, 0)));
  EXPECT_EQ(23, hour(Timestamp(-1, 9000)));
  EXPECT_EQ(23, hour(Timestamp(-1, Timestamp::kMaxNanos)));
  EXPECT_EQ(7, hour(Timestamp(4000000000, 0)));
  EXPECT_EQ(7, hour(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(10, hour(Timestamp(998474645, 321000000)));
  EXPECT_EQ(19, hour(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(13, hour(Timestamp(0, 0)));
  EXPECT_EQ(12, hour(Timestamp(-1, Timestamp::kMaxNanos)));
  // Disabled for now because the TZ for Pacific/Apia in 2096 varies between
  // systems.
  // EXPECT_EQ(21, hour(Timestamp(4000000000, 0)));
  // EXPECT_EQ(21, hour(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(23, hour(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, hour(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, hourTimestampWithTimezone) {
  const auto hourTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "hour(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      20,
      hourTimestampWithTimezone(TimestampWithTimezone(998423705000, "+01:00")));
  EXPECT_EQ(
      12, hourTimestampWithTimezone(TimestampWithTimezone(41028000, "+01:00")));
  EXPECT_EQ(
      13, hourTimestampWithTimezone(TimestampWithTimezone(41028000, "+02:00")));
  EXPECT_EQ(
      14, hourTimestampWithTimezone(TimestampWithTimezone(41028000, "+03:00")));
  EXPECT_EQ(
      8, hourTimestampWithTimezone(TimestampWithTimezone(41028000, "-03:00")));
  EXPECT_EQ(
      1, hourTimestampWithTimezone(TimestampWithTimezone(41028000, "+14:00")));
  EXPECT_EQ(
      9, hourTimestampWithTimezone(TimestampWithTimezone(-100000, "-14:00")));
  EXPECT_EQ(
      2, hourTimestampWithTimezone(TimestampWithTimezone(-41028000, "+14:00")));
  EXPECT_EQ(std::nullopt, hourTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, hourDate) {
  const auto hour = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("hour(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(0, hour(0));
  EXPECT_EQ(0, hour(-1));
  EXPECT_EQ(0, hour(-40));
  EXPECT_EQ(0, hour(40));
  EXPECT_EQ(0, hour(18262));
  EXPECT_EQ(0, hour(-18262));
}

TEST_F(DateTimeFunctionsTest, hourTime) {
  const auto hour = [&](std::optional<int64_t> time) {
    return evaluateOnce<int64_t>("hour(c0)", TIME(), time);
  };

  // null handling
  EXPECT_EQ(std::nullopt, hour(std::nullopt));

  // boundary tests - core optimization: simple integer division by
  // kMillisInHour (3600000) lower boundary: 0 <= time < 86400000
  EXPECT_EQ(0, hour(0)); // exactly midnight
  EXPECT_EQ(0, hour(3599999)); // 00:59:59.999 - last millisecond of hour 0
  EXPECT_EQ(1, hour(3600000)); // 01:00:00.000 - first millisecond of hour 1
  EXPECT_EQ(23, hour(82800000)); // 23:00:00.000 - first millisecond of hour 23
  EXPECT_EQ(23, hour(86399999)); // 23:59:59.999 - upper boundary exclusive

  // representative samples across all hours
  EXPECT_EQ(12, hour(43200000)); // noon exactly
  EXPECT_EQ(18, hour(64800000)); // 18:00:00.000

  // verify optimization correctness: hour = time / kMillisInHour
  // random test cases to ensure division behaves correctly
  EXPECT_EQ(5, hour(19800000)); // 05:30:00.000
  EXPECT_EQ(10, hour(37800000)); // 10:30:00.000
  EXPECT_EQ(15, hour(54000000)); // 15:00:00.000

  // error conditions - invalid range validation
  EXPECT_THROW(hour(-1), VeloxUserError); // negative time
  EXPECT_THROW(
      hour(86400000),
      VeloxUserError); // exactly 24:00:00.000 (exclusive upper bound)
  EXPECT_THROW(
      hour(std::numeric_limits<int64_t>::max()),
      VeloxUserError); // overflow case
}

TEST_F(DateTimeFunctionsTest, dayOfMonth) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("day_of_month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Timestamp(0, 0)));
  EXPECT_EQ(31, day(Timestamp(-1, 9000)));
  EXPECT_EQ(30, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(1, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(6, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(31, day(Timestamp(1635668100, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(31, day(Timestamp(0, 0)));
  EXPECT_EQ(31, day(Timestamp(-1, 9000)));
  EXPECT_EQ(30, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(1, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(6, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(31, day(Timestamp(1635668100, 0)));
}

TEST_F(DateTimeFunctionsTest, dayOfMonthDate) {
  const auto day = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("day_of_month(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(0));
  EXPECT_EQ(31, day(-1));
  EXPECT_EQ(22, day(-40));
  EXPECT_EQ(10, day(40));
  EXPECT_EQ(1, day(18262));
  EXPECT_EQ(2, day(-18262));
}

TEST_F(DateTimeFunctionsTest, dayOfMonthInterval) {
  const auto day = [&](std::optional<int64_t> millis) {
    auto result =
        evaluateOnce<int64_t>("day_of_month(c0)", INTERVAL_DAY_TIME(), millis);

    auto result2 =
        evaluateOnce<int64_t>("day(c0)", INTERVAL_DAY_TIME(), millis);

    EXPECT_EQ(result, result2);
    return result;
  };

  EXPECT_EQ(1, day(kMillisInDay));
  EXPECT_EQ(1, day(kMillisInDay + kMillisInHour));
  EXPECT_EQ(10, day(10 * kMillisInDay + 7 * kMillisInHour));
  EXPECT_EQ(-10, day(-10 * kMillisInDay - 7 * kMillisInHour));
}

TEST_F(DateTimeFunctionsTest, plusMinusDateIntervalYearMonth) {
  const auto makeInput = [&](const std::string& date, int32_t interval) {
    return makeRowVector({
        makeNullableFlatVector<int32_t>({parseDate(date)}, DATE()),
        makeNullableFlatVector<int32_t>({interval}, INTERVAL_YEAR_MONTH()),
    });
  };

  const auto plus = [&](const std::string& date, int32_t interval) {
    return evaluateOnce<int32_t>("c0 + c1", makeInput(date, interval));
  };

  const auto minus = [&](const std::string& date, int32_t interval) {
    return evaluateOnce<int32_t>("c0 - c1", makeInput(date, interval));
  };

  EXPECT_EQ(parseDate("2021-10-15"), plus("2020-10-15", 12));
  EXPECT_EQ(parseDate("2022-01-15"), plus("2020-10-15", 15));
  EXPECT_EQ(parseDate("2020-11-15"), plus("2020-10-15", 1));
  EXPECT_EQ(parseDate("2020-11-30"), plus("2020-10-30", 1));
  EXPECT_EQ(parseDate("2020-11-30"), plus("2020-10-31", 1));
  EXPECT_EQ(parseDate("2021-02-28"), plus("2020-02-28", 12));
  EXPECT_EQ(parseDate("2021-02-28"), plus("2020-02-29", 12));
  EXPECT_EQ(parseDate("2016-02-29"), plus("2020-02-29", -48));

  EXPECT_EQ(parseDate("2019-10-15"), minus("2020-10-15", 12));
  EXPECT_EQ(parseDate("2019-07-15"), minus("2020-10-15", 15));
  EXPECT_EQ(parseDate("2020-09-15"), minus("2020-10-15", 1));
  EXPECT_EQ(parseDate("2020-09-30"), minus("2020-10-30", 1));
  EXPECT_EQ(parseDate("2020-09-30"), minus("2020-10-31", 1));
  EXPECT_EQ(parseDate("2019-02-28"), minus("2020-02-29", 12));
  EXPECT_EQ(parseDate("2019-02-28"), minus("2020-02-28", 12));
  EXPECT_EQ(parseDate("2024-02-29"), minus("2020-02-29", -48));
}

TEST_F(DateTimeFunctionsTest, plusMinusDateIntervalDayTime) {
  const auto plus = [&](std::optional<int32_t> date,
                        std::optional<int64_t> interval) {
    return evaluateOnce<int32_t>(
        "c0 + c1",
        makeRowVector({
            makeNullableFlatVector<int32_t>({date}, DATE()),
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
        }));
  };
  const auto minus = [&](std::optional<int32_t> date,
                         std::optional<int64_t> interval) {
    return evaluateOnce<int32_t>(
        "c0 - c1",
        makeRowVector({
            makeNullableFlatVector<int32_t>({date}, DATE()),
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
        }));
  };

  const int64_t oneDay(kMillisInDay * 1);
  const int64_t tenDays(kMillisInDay * 10);
  const int64_t partDay(kMillisInHour * 25);
  const int32_t baseDate(20000);
  const int32_t baseDatePlus1(20000 + 1);
  const int32_t baseDatePlus10(20000 + 10);
  const int32_t baseDateMinus1(20000 - 1);
  const int32_t baseDateMinus10(20000 - 10);

  EXPECT_EQ(std::nullopt, plus(std::nullopt, oneDay));
  EXPECT_EQ(std::nullopt, plus(10000, std::nullopt));
  EXPECT_EQ(baseDatePlus1, plus(baseDate, oneDay));
  EXPECT_EQ(baseDatePlus10, plus(baseDate, tenDays));
  EXPECT_EQ(std::nullopt, minus(std::nullopt, oneDay));
  EXPECT_EQ(std::nullopt, minus(10000, std::nullopt));
  EXPECT_EQ(baseDateMinus1, minus(baseDate, oneDay));
  EXPECT_EQ(baseDateMinus10, minus(baseDate, tenDays));

  EXPECT_THROW(plus(baseDate, partDay), VeloxUserError);
  EXPECT_THROW(minus(baseDate, partDay), VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, timestampMinusIntervalYearMonth) {
  const auto minus = [&](std::optional<std::string> timestamp,
                         std::optional<int32_t> interval) {
    return evaluateOnce<std::string>(
        "date_format(date_parse(c0, '%Y-%m-%d %H:%i:%s') - c1, '%Y-%m-%d %H:%i:%s')",
        makeRowVector({
            makeNullableFlatVector<std::string>({timestamp}, VARCHAR()),
            makeNullableFlatVector<int32_t>({interval}, INTERVAL_YEAR_MONTH()),
        }));
  };

  EXPECT_EQ("2001-01-03 04:05:06", minus("2001-02-03 04:05:06", 1));
  EXPECT_EQ("2000-04-03 04:05:06", minus("2001-02-03 04:05:06", 10));
  EXPECT_EQ("1999-06-03 04:05:06", minus("2001-02-03 04:05:06", 20));

  // Some special dates.
  EXPECT_EQ("2001-04-30 04:05:06", minus("2001-05-31 04:05:06", 1));
  EXPECT_EQ("2001-03-30 04:05:06", minus("2001-04-30 04:05:06", 1));
  EXPECT_EQ("2001-02-28 04:05:06", minus("2001-03-30 04:05:06", 1));
  EXPECT_EQ("2000-02-29 04:05:06", minus("2000-03-30 04:05:06", 1));
  EXPECT_EQ("2000-01-29 04:05:06", minus("2000-02-29 04:05:06", 1));

  // Check if it does the right thing if we cross daylight saving boundaries.
  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ("2024-01-01 00:00:00", minus("2024-07-01 00:00:00", 6));
  EXPECT_EQ("2023-07-01 00:00:00", minus("2024-01-01 00:00:00", 6));
}

TEST_F(DateTimeFunctionsTest, timestampPlusIntervalYearMonth) {
  const auto plus = [&](std::optional<std::string> timestamp,
                        std::optional<int32_t> interval) {
    // timestamp + interval.
    auto result1 = evaluateOnce<std::string>(
        "date_format(date_parse(c0, '%Y-%m-%d %H:%i:%s') + c1, '%Y-%m-%d %H:%i:%s')",
        makeRowVector(
            {makeNullableFlatVector<std::string>({timestamp}, VARCHAR()),
             makeNullableFlatVector<int32_t>(
                 {interval}, INTERVAL_YEAR_MONTH())}));

    // interval + timestamp.
    auto result2 = evaluateOnce<std::string>(
        "date_format(c1 + date_parse(c0, '%Y-%m-%d %H:%i:%s'), '%Y-%m-%d %H:%i:%s')",
        makeRowVector(
            {makeNullableFlatVector<std::string>({timestamp}, VARCHAR()),
             makeNullableFlatVector<int32_t>(
                 {interval}, INTERVAL_YEAR_MONTH())}));

    // They should be the same.
    EXPECT_EQ(result1, result2);
    return result1;
  };

  EXPECT_EQ("2001-02-03 04:05:06", plus("2001-01-03 04:05:06", 1));
  EXPECT_EQ("2001-02-03 04:05:06", plus("2000-04-03 04:05:06", 10));
  EXPECT_EQ("2001-02-03 04:05:06", plus("1999-06-03 04:05:06", 20));

  // Some special dates.
  EXPECT_EQ("2001-06-30 04:05:06", plus("2001-05-31 04:05:06", 1));
  EXPECT_EQ("2001-05-30 04:05:06", plus("2001-04-30 04:05:06", 1));
  EXPECT_EQ("2001-02-28 04:05:06", plus("2001-01-31 04:05:06", 1));
  EXPECT_EQ("2000-02-29 04:05:06", plus("2000-01-31 04:05:06", 1));
  EXPECT_EQ("2000-02-29 04:05:06", plus("2000-01-29 04:05:06", 1));

  // Check if it does the right thing if we cross daylight saving boundaries.
  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ("2025-01-01 00:00:00", plus("2024-07-01 00:00:00", 6));
  EXPECT_EQ("2024-07-01 00:00:00", plus("2024-01-01 00:00:00", 6));
}

TEST_F(DateTimeFunctionsTest, plusMinusTimestampIntervalDayTime) {
  constexpr int64_t kLongMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kLongMin = std::numeric_limits<int64_t>::min();

  const auto minus = [&](std::optional<Timestamp> timestamp,
                         std::optional<int64_t> interval) {
    return evaluateOnce<Timestamp>(
        "c0 - c1",
        makeRowVector({
            makeNullableFlatVector<Timestamp>({timestamp}),
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
        }));
  };

  EXPECT_EQ(std::nullopt, minus(std::nullopt, std::nullopt));
  EXPECT_EQ(std::nullopt, minus(std::nullopt, 1));
  EXPECT_EQ(std::nullopt, minus(Timestamp(0, 0), std::nullopt));
  EXPECT_EQ(Timestamp(0, 0), minus(Timestamp(0, 0), 0));
  EXPECT_EQ(Timestamp(0, 0), minus(Timestamp(10, 0), 10'000));
  EXPECT_EQ(Timestamp(-10, 0), minus(Timestamp(10, 0), 20'000));
  EXPECT_EQ(
      Timestamp(-2, 50 * Timestamp::kNanosecondsInMillisecond),
      minus(Timestamp(0, 50 * Timestamp::kNanosecondsInMillisecond), 2'000));
  EXPECT_EQ(
      Timestamp(-3, 995 * Timestamp::kNanosecondsInMillisecond),
      minus(Timestamp(0, 0), 2'005));
  EXPECT_EQ(
      Timestamp(9223372036854774, 809000000),
      minus(Timestamp(-1, 0), kLongMax));
  EXPECT_EQ(
      Timestamp(-9223372036854775, 192000000),
      minus(Timestamp(1, 0), kLongMin));

  const auto plusAndVerify = [&](std::optional<Timestamp> timestamp,
                                 std::optional<int64_t> interval,
                                 std::optional<Timestamp> expected) {
    EXPECT_EQ(
        expected,
        evaluateOnce<Timestamp>(
            "c0 + c1",
            makeRowVector({
                makeNullableFlatVector<Timestamp>({timestamp}),
                makeNullableFlatVector<int64_t>(
                    {interval}, INTERVAL_DAY_TIME()),
            })));
    EXPECT_EQ(
        expected,
        evaluateOnce<Timestamp>(
            "c1 + c0",
            makeRowVector({
                makeNullableFlatVector<Timestamp>({timestamp}),
                makeNullableFlatVector<int64_t>(
                    {interval}, INTERVAL_DAY_TIME()),
            })));
  };

  plusAndVerify(std::nullopt, std::nullopt, std::nullopt);
  plusAndVerify(std::nullopt, 1, std::nullopt);
  plusAndVerify(Timestamp(0, 0), std::nullopt, std::nullopt);
  plusAndVerify(Timestamp(0, 0), 0, Timestamp(0, 0));
  plusAndVerify(Timestamp(0, 0), 10'000, Timestamp(10, 0));
  plusAndVerify(
      Timestamp(0, 0),
      20'005,
      Timestamp(20, 5 * Timestamp::kNanosecondsInMillisecond));
  plusAndVerify(
      Timestamp(0, 0),
      -30'005,
      Timestamp(-31, 995 * Timestamp::kNanosecondsInMillisecond));
  plusAndVerify(
      Timestamp(1, 0), kLongMax, Timestamp(-9223372036854775, 191000000));
  plusAndVerify(
      Timestamp(0, 0), kLongMin, Timestamp(-9223372036854776, 192000000));
  plusAndVerify(
      Timestamp(-1, 0), kLongMin, Timestamp(9223372036854774, 808000000));
}

TEST_F(DateTimeFunctionsTest, timestampWithTimeZonePlusIntervalDayTime) {
  auto test = [&](const std::string& timestamp, int64_t interval) {
    // ts + interval == interval + ts == ts - (-interval) ==
    // date_add('millisecond', interval, ts).
    auto plusResult =
        evaluateOnce<std::string>(
            "cast(plus(cast(c0 as timestamp with time zone), c1) as varchar)",
            {VARCHAR(), INTERVAL_DAY_TIME()},
            std::optional(timestamp),
            std::optional(interval))
            .value();

    auto minusResult =
        evaluateOnce<std::string>(
            "cast(minus(cast(c0 as timestamp with time zone), c1) as varchar)",
            {VARCHAR(), INTERVAL_DAY_TIME()},
            std::optional(timestamp),
            std::optional(-interval))
            .value();

    auto otherPlusResult =
        evaluateOnce<std::string>(
            "cast(plus(c1, cast(c0 as timestamp with time zone)) as varchar)",
            {VARCHAR(), INTERVAL_DAY_TIME()},
            std::optional(timestamp),
            std::optional(interval))
            .value();

    auto dateAddResult =
        evaluateOnce<std::string>(
            "cast(date_add('millisecond', c1, cast(c0 as timestamp with time zone)) as varchar)",
            std::optional(timestamp),
            std::optional(interval))
            .value();

    VELOX_CHECK_EQ(plusResult, minusResult);
    VELOX_CHECK_EQ(plusResult, otherPlusResult);
    VELOX_CHECK_EQ(plusResult, dateAddResult);
    return plusResult;
  };

  EXPECT_EQ(
      "2024-10-04 01:50:00.000 America/Los_Angeles",
      test("2024-10-03 01:50 America/Los_Angeles", 1 * kMillisInDay));
  EXPECT_EQ(
      "2024-10-03 02:50:00.000 America/Los_Angeles",
      test("2024-10-03 01:50 America/Los_Angeles", 1 * kMillisInHour));
  EXPECT_EQ(
      "2024-10-03 01:51:00.000 America/Los_Angeles",
      test("2024-10-03 01:50 America/Los_Angeles", 1 * kMillisInMinute));

  // Testing daylight saving transitions.

  // At the beginning there is a 1 hour gap.
  EXPECT_EQ(
      "2024-03-10 01:30:00.000 America/Los_Angeles",
      test("2024-03-10 03:30 America/Los_Angeles", -1 * kMillisInHour));

  // At the end there is a 1 hour duplication.
  EXPECT_EQ(
      "2024-11-03 01:30:00.000 America/Los_Angeles",
      test("2024-11-03 01:30 America/Los_Angeles", 1 * kMillisInHour));
}

TEST_F(DateTimeFunctionsTest, minusTimestamp) {
  const auto minus = [&](std::optional<int64_t> t1, std::optional<int64_t> t2) {
    const auto timestamp1 = (t1.has_value()) ? Timestamp(t1.value(), 0)
                                             : std::optional<Timestamp>();
    const auto timestamp2 = (t2.has_value()) ? Timestamp(t2.value(), 0)
                                             : std::optional<Timestamp>();
    return evaluateOnce<int64_t>(
        "c0 - c1",
        makeRowVector({
            makeNullableFlatVector<Timestamp>({timestamp1}),
            makeNullableFlatVector<Timestamp>({timestamp2}),
        }));
  };

  EXPECT_EQ(std::nullopt, minus(std::nullopt, std::nullopt));
  EXPECT_EQ(std::nullopt, minus(1, std::nullopt));
  EXPECT_EQ(std::nullopt, minus(std::nullopt, 1));
  EXPECT_EQ(1000, minus(1, 0));
  EXPECT_EQ(-1000, minus(1, 2));
  VELOX_ASSERT_THROW(
      minus(Timestamp::kMinSeconds, Timestamp::kMaxSeconds),
      "Could not convert Timestamp(-9223372036854776, 0) to milliseconds");
}

TEST_F(DateTimeFunctionsTest, timeIntervalDayTime) {
  // Test TIME + IntervalDayTime and IntervalDayTime + TIME arithmetic

  // Helper for Time + IntervalDayTime (also tests commutativity)
  const auto testTimePlusIntervalCommutative =
      [&](int64_t time, int64_t interval) -> std::optional<int64_t> {
    auto result1 = evaluateOnce<int64_t>(
        "plus(c0, c1)",
        makeRowVector({
            makeNullableFlatVector<int64_t>({time}, TIME()),
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
        }));

    auto result2 = evaluateOnce<int64_t>(
        "plus(c0, c1)",
        makeRowVector({
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
            makeNullableFlatVector<int64_t>({time}, TIME()),
        }));

    EXPECT_EQ(result1, result2);
    return result1;
  };

  // Basic hour addition: 03:04:05.321 + 3 hours = 06:04:05.321
  // 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  // 3 hours = 3*60*60*1000 = 10800000 ms
  // 06:04:05.321 = (6*3600 + 4*60 + 5)*1000 + 321 = 21845321 ms
  const int64_t time1 = 11045321; // 03:04:05.321
  const int64_t threeHours = 3 * kMillisInHour; // 3 hours
  EXPECT_EQ(21845321, testTimePlusIntervalCommutative(time1, threeHours));

  // Test 24-hour wraparound: 22:00:00 + 3 hours = 01:00:00
  // 22:00:00 = 22*3600*1000 = 79200000 ms
  // 01:00:00 = 1*3600*1000 = 3600000 ms
  const int64_t time2 = 22 * kMillisInHour; // 22:00:00
  EXPECT_EQ(3600000, testTimePlusIntervalCommutative(time2, threeHours));

  // Test minute addition: 03:04:05.321 + 90 minutes = 04:34:05.321
  // 90 minutes = 90*60*1000 = 5400000 ms
  // 04:34:05.321 = (4*3600 + 34*60 + 5)*1000 + 321 = 16445321 ms
  const int64_t ninetyMinutes = 90 * kMillisInMinute;
  EXPECT_EQ(16445321, testTimePlusIntervalCommutative(time1, ninetyMinutes));

  // Test negative intervals: 03:04:05.321 - 2 hours = 01:04:05.321
  // 01:04:05.321 = (1*3600 + 4*60 + 5)*1000 + 321 = 3845321 ms
  const int64_t twoHours = 2 * kMillisInHour;
  EXPECT_EQ(3845321, testTimePlusIntervalCommutative(time1, -twoHours));

  // Test negative wraparound: 01:00:00 - 3 hours = 22:00:00 (previous day)
  // 01:00:00 = 1*3600*1000 = 3600000 ms
  const int64_t time3 = kMillisInHour; // 01:00:00
  EXPECT_EQ(79200000, testTimePlusIntervalCommutative(time3, -threeHours));

  // Test millisecond precision: 03:04:05.321 + 679 milliseconds = 03:04:06.000
  EXPECT_EQ(11046000, testTimePlusIntervalCommutative(time1, 679));

  // Test day intervals (should not change time of day per Presto behavior)
  // 12:30:45.123 = (12*3600 + 30*60 + 45)*1000 + 123 = 45045123 ms
  const int64_t time4 = 45045123; // 12:30:45.123
  const int64_t oneDay = kMillisInDay;
  EXPECT_EQ(45045123, testTimePlusIntervalCommutative(time4, oneDay));
}

TEST_F(DateTimeFunctionsTest, timeMinusIntervalDayTime) {
  // Test TIME - IntervalDayTime arithmetic (does not support Interval - Time)

  const auto timeMinusInterval =
      [&](int64_t time, int64_t interval) -> std::optional<int64_t> {
    return evaluateOnce<int64_t>(
        "minus(c0, c1)",
        makeRowVector({
            makeNullableFlatVector<int64_t>({time}, TIME()),
            makeNullableFlatVector<int64_t>({interval}, INTERVAL_DAY_TIME()),
        }));
  };

  // Basic hour subtraction: 06:04:05.321 - 3 hours = 03:04:05.321
  // 06:04:05.321 = (6*3600 + 4*60 + 5)*1000 + 321 = 21845321 ms
  // 3 hours = 3*60*60*1000 = 10800000 ms
  // 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  const int64_t time1 = 21845321; // 06:04:05.321
  const int64_t threeHours = 3 * kMillisInHour;
  EXPECT_EQ(11045321, timeMinusInterval(time1, threeHours));

  // Test 24-hour wraparound: 01:00:00 - 3 hours = 22:00:00 (previous day)
  // 01:00:00 = 1*3600*1000 = 3600000 ms
  // 22:00:00 = 22*3600*1000 = 79200000 ms
  const int64_t time2 = kMillisInHour; // 01:00:00
  EXPECT_EQ(79200000, timeMinusInterval(time2, threeHours));

  // Test minute subtraction: 04:34:05.321 - 90 minutes = 03:04:05.321
  // 04:34:05.321 = (4*3600 + 34*60 + 5)*1000 + 321 = 16445321 ms
  // 90 minutes = 90*60*1000 = 5400000 ms
  // 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  const int64_t time3 = 16445321; // 04:34:05.321
  const int64_t ninetyMinutes = 90 * kMillisInMinute;
  EXPECT_EQ(11045321, timeMinusInterval(time3, ninetyMinutes));

  // Test millisecond precision: 03:04:06.000 - 679 milliseconds = 03:04:05.321
  // 03:04:06.000 = (3*3600 + 4*60 + 6)*1000 = 11046000 ms
  // 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  const int64_t time4 = 11046000; // 03:04:06.000
  EXPECT_EQ(11045321, timeMinusInterval(time4, 679));

  // Test day intervals (should not change time of day per Presto behavior)
  // 12:30:45.123 = (12*3600 + 30*60 + 45)*1000 + 123 = 45045123 ms
  const int64_t time5 = 45045123; // 12:30:45.123
  const int64_t oneDay = kMillisInDay;
  EXPECT_EQ(45045123, timeMinusInterval(time5, oneDay));

  // Test subtracting negative intervals (double negative = addition)
  // 03:04:05.321 - (-3 hours) = 03:04:05.321 + 3 hours = 06:04:05.321
  // 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  // 06:04:05.321 = (6*3600 + 4*60 + 5)*1000 + 321 = 21845321 ms
  const int64_t time6 = 11045321; // 03:04:05.321
  EXPECT_EQ(21845321, timeMinusInterval(time6, -threeHours));

  // Test negative interval with wraparound: 22:00:00 - (-3 hours) = 01:00:00
  // 22:00:00 = 22*3600*1000 = 79200000 ms
  // 01:00:00 = 1*3600*1000 = 3600000 ms
  const int64_t time7 = 22 * kMillisInHour; // 22:00:00
  EXPECT_EQ(3600000, timeMinusInterval(time7, -threeHours));
}

TEST_F(DateTimeFunctionsTest, timeMinusTime) {
  // Test TIME - TIME arithmetic returning INTERVAL_DAY_TIME

  const auto timeMinusTime = [&](int64_t time1,
                                 int64_t time2) -> std::optional<int64_t> {
    return evaluateOnce<int64_t>(
        "minus(c0, c1)",
        makeRowVector({
            makeNullableFlatVector<int64_t>({time1}, TIME()),
            makeNullableFlatVector<int64_t>({time2}, TIME()),
        }));
  };

  // Test basic subtraction: 10:00:00 - 08:00:00 = 2 hours = 7200000 ms
  const int64_t tenAM = 10 * kMillisInHour; // 36000000 ms
  const int64_t eightAM = 8 * kMillisInHour; // 28800000 ms
  const int64_t twoHours = 2 * kMillisInHour; // 7200000 ms
  EXPECT_EQ(twoHours, timeMinusTime(tenAM, eightAM));

  // Test reverse (negative result): 08:00:00 - 10:00:00 = -2 hours
  EXPECT_EQ(-twoHours, timeMinusTime(eightAM, tenAM));

  // Test same time: 10:00:00 - 10:00:00 = 0
  EXPECT_EQ(0, timeMinusTime(tenAM, tenAM));

  // Test with millisecond precision
  // 12:30:45.123 - 06:04:05.321 = 23199802 ms
  const int64_t time1 = 12 * kMillisInHour + 30 * kMillisInMinute +
      45 * kMillisInSecond + 123; // 45045123
  const int64_t time2 = 6 * kMillisInHour + 4 * kMillisInMinute +
      5 * kMillisInSecond + 321; // 21845321
  EXPECT_EQ(23199802, timeMinusTime(time1, time2));

  // Test midnight cases
  // 23:59:59.999 - 00:00:00.000 = 86399999 ms (almost full day)
  const int64_t almostMidnight = kMillisInDay - 1; // 86399999
  const int64_t midnight = 0;
  EXPECT_EQ(86399999, timeMinusTime(almostMidnight, midnight));

  // 00:00:00.000 - 23:59:59.999 = -86399999 ms (negative almost full day)
  EXPECT_EQ(-86399999, timeMinusTime(midnight, almostMidnight));

  // Test with NULL values
  const auto timeMinusTimeWithNull =
      [&](std::optional<int64_t> time1,
          std::optional<int64_t> time2) -> std::optional<int64_t> {
    return evaluateOnce<int64_t>(
        "minus(c0, c1)",
        makeRowVector({
            makeNullableFlatVector<int64_t>({time1}, TIME()),
            makeNullableFlatVector<int64_t>({time2}, TIME()),
        }));
  };

  EXPECT_EQ(std::nullopt, timeMinusTimeWithNull(std::nullopt, std::nullopt));
  EXPECT_EQ(std::nullopt, timeMinusTimeWithNull(tenAM, std::nullopt));
  EXPECT_EQ(std::nullopt, timeMinusTimeWithNull(std::nullopt, eightAM));
}

// Comprehensive tests for TimePlusIntervalYearMonthVectorFunction optimizations
// and IntervalYearMonthPlusTimeVectorFunction optimizations
TEST_F(DateTimeFunctionsTest, timeIntervalYearMonthVectorOptimizations) {
  // Helper to create interval year-month values (in months)
  auto months = [](int32_t value) { return value; };

  // Helper to create time values (milliseconds since midnight)
  auto timeMs = [](int32_t hours,
                   int32_t minutes = 0,
                   int32_t seconds = 0,
                   int32_t millis = 0) {
    return static_cast<int64_t>(hours) * 3600000 +
        static_cast<int64_t>(minutes) * 60000 +
        static_cast<int64_t>(seconds) * 1000 + static_cast<int64_t>(millis);
  };

  // Helper to verify time +/- interval results (should be identity function)
  // Tests commutativity for plus: interval + time gives the same result
  // Tests minus: time - interval (only this ordering supported for minus)
  auto testTimePlusMinusIntervalYearMonth =
      [&](const VectorPtr& timeVector,
          const VectorPtr& intervalVector,
          const VectorPtr& expectedResult) {
        // Test: Time + Interval
        auto resultPlus1 = evaluate(
            "plus(c0, c1)", makeRowVector({timeVector, intervalVector}));
        assertEqualVectors(expectedResult, resultPlus1);

        // Test: Interval + Time (commutative for plus)
        auto resultPlus2 = evaluate(
            "plus(c0, c1)", makeRowVector({intervalVector, timeVector}));
        assertEqualVectors(expectedResult, resultPlus2);

        // Test: Time - Interval (identity function, only this ordering)
        auto resultMinus = evaluate(
            "minus(c0, c1)", makeRowVector({timeVector, intervalVector}));
        assertEqualVectors(expectedResult, resultMinus);
      };

  // TEST 1: Constant Vector Optimization - Non-null constant
  {
    const auto timeValue = timeMs(14, 30, 15, 123); // 14:30:15.123
    const auto intervalValue = months(6); // 6 months

    // Create constant vectors
    auto constantTime =
        BaseVector::createConstant(TIME(), timeValue, 1000, pool());
    auto constantInterval = BaseVector::createConstant(
        INTERVAL_YEAR_MONTH(), intervalValue, 1000, pool());

    // Expected result should be same constant time (identity function)
    auto expectedResult =
        BaseVector::createConstant(TIME(), timeValue, 1000, pool());

    // Test: Time + Interval, Interval + Time, and Time - Interval
    testTimePlusMinusIntervalYearMonth(
        constantTime, constantInterval, expectedResult);

    // Verify the result is also constant encoded for efficiency
    auto result = evaluate(
        "plus(c0, c1)", makeRowVector({constantTime, constantInterval}));

    EXPECT_TRUE(result->isConstantEncoding());
    EXPECT_EQ(result->size(), 1000);
    EXPECT_EQ(result->as<ConstantVector<int64_t>>()->valueAt(0), timeValue);
  }

  // TEST 2: Constant Vector Optimization - mixed null and non-null values
  {
    const auto intervalValue = months(12); // 12 months

    // Create time vector with mixed null and non-null values to test null
    // handling
    auto timeVector = makeNullableFlatVector<int64_t>(
        {std::nullopt, timeMs(17, 30, 0), std::nullopt}, TIME());

    auto intervalVector = makeFlatVector<int32_t>(
        {intervalValue, intervalValue, intervalValue}, INTERVAL_YEAR_MONTH());

    // Expected result: null, non-null, null
    auto expectedResult = makeNullableFlatVector<int64_t>(
        {std::nullopt, timeMs(17, 30, 0), std::nullopt}, TIME());

    // Test: Time + Interval, Interval + Time, and Time - Interval
    testTimePlusMinusIntervalYearMonth(
        timeVector, intervalVector, expectedResult);
  }

  // TEST 3: Flat Vector with Nulls
  {
    const std::vector<std::optional<int64_t>> timeValues = {
        timeMs(10, 30, 0), // 10:30:00
        std::nullopt, // null
        timeMs(16, 45, 30), // 16:45:30
        std::nullopt, // null
        timeMs(20, 0, 0) // 20:00:00
    };

    const std::vector<int32_t> intervalValues = {
        months(3), months(6), months(9), months(12), months(18)};

    auto timeVector = makeNullableFlatVector<int64_t>(timeValues, TIME());
    auto intervalVector =
        makeFlatVector<int32_t>(intervalValues, INTERVAL_YEAR_MONTH());

    // Expected result should preserve nulls and non-null values identically
    auto expectedResult = makeNullableFlatVector<int64_t>(timeValues, TIME());

    // Test: Time + Interval, Interval + Time, and Time - Interval
    testTimePlusMinusIntervalYearMonth(
        timeVector, intervalVector, expectedResult);
  }

  // TEST 4: Dictionary Vector (Fallback Path)
  {
    // Create base values for dictionary
    const std::vector<int64_t> baseTimeValues = {
        timeMs(9, 0, 0), // 09:00:00
        timeMs(17, 30, 0), // 17:30:00
        timeMs(12, 0, 0) // 12:00:00
    };

    // Create indices that repeat base values
    const std::vector<vector_size_t> indices = {0, 1, 2, 0, 1, 2, 1, 0};

    auto baseVector = makeFlatVector<int64_t>(baseTimeValues, TIME());

    auto dictionaryTimeVector = wrapInDictionary(
        makeIndices(indices), (vector_size_t)indices.size(), baseVector);

    auto intervalVector = makeFlatVector<int32_t>(
        std::vector<int32_t>(indices.size(), months(6)), INTERVAL_YEAR_MONTH());

    // Expected result should decode dictionary and preserve values
    std::vector<int64_t> expectedValues;
    expectedValues.reserve(indices.size());
    for (auto idx : indices) {
      expectedValues.push_back(baseTimeValues[idx]);
    }
    auto expectedResult = makeFlatVector<int64_t>(expectedValues, TIME());

    // Test: Time + Interval, Interval + Time, and Time - Interval
    testTimePlusMinusIntervalYearMonth(
        dictionaryTimeVector, intervalVector, expectedResult);
  }
}

TEST_F(DateTimeFunctionsTest, minusTimestampWithTimezone) {
  auto minus = [&](const std::string& a, const std::string& b) {
    const auto sql =
        "cast(c0 as timestamp with time zone) - cast(c1 as timestamp with time zone)";

    auto result =
        evaluateOnce<int64_t>(sql, std::optional(a), std::optional(b));
    auto negativeResult =
        evaluateOnce<int64_t>(sql, std::optional(b), std::optional(a));

    // a - b == -(b - a)
    VELOX_CHECK_EQ(result.value(), -negativeResult.value());
    return result.value();
  };

  EXPECT_EQ(
      0,
      minus(
          "2024-04-15 10:20:33 America/New_York",
          "2024-04-15 10:20:33 America/New_York"));

  EXPECT_EQ(
      0,
      minus(
          "2024-04-15 13:20:33 America/New_York",
          "2024-04-15 10:20:33 America/Los_Angeles"));

  EXPECT_EQ(
      -1 * kMillisInHour - 2 * kMillisInMinute - 3 * kMillisInSecond,
      minus(
          "2024-04-15 10:20:33 America/New_York",
          "2024-04-15 11:22:36 America/New_York"));

  EXPECT_EQ(
      -1 * kMillisInHour - 2 * kMillisInMinute - 3 * kMillisInSecond,
      minus(
          "2024-04-15 07:20:33 America/Los_Angeles",
          "2024-04-15 11:22:36 America/New_York"));
}

TEST_F(DateTimeFunctionsTest, dayOfMonthTimestampWithTimezone) {
  const auto dayOfMonthTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "day_of_month(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      31, dayOfMonthTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      1, dayOfMonthTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      30,
      dayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      2,
      dayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      18,
      dayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      14,
      dayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, dayOfMonthTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, dayOfWeek) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("day_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(4, day(Timestamp(0, 0)));
  EXPECT_EQ(3, day(Timestamp(-1, 9000)));
  EXPECT_EQ(1, day(Timestamp(1633940100, 0)));
  EXPECT_EQ(2, day(Timestamp(1634026500, 0)));
  EXPECT_EQ(3, day(Timestamp(1634112900, 0)));
  EXPECT_EQ(4, day(Timestamp(1634199300, 0)));
  EXPECT_EQ(5, day(Timestamp(1634285700, 0)));
  EXPECT_EQ(6, day(Timestamp(1634372100, 0)));
  EXPECT_EQ(7, day(Timestamp(1633853700, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(3, day(Timestamp(0, 0)));
  EXPECT_EQ(3, day(Timestamp(-1, 9000)));
  EXPECT_EQ(1, day(Timestamp(1633940100, 0)));
  EXPECT_EQ(2, day(Timestamp(1634026500, 0)));
  EXPECT_EQ(3, day(Timestamp(1634112900, 0)));
  EXPECT_EQ(4, day(Timestamp(1634199300, 0)));
  EXPECT_EQ(5, day(Timestamp(1634285700, 0)));
  EXPECT_EQ(6, day(Timestamp(1634372100, 0)));
  EXPECT_EQ(7, day(Timestamp(1633853700, 0)));
}

TEST_F(DateTimeFunctionsTest, dayOfWeekDate) {
  const auto day = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("day_of_week(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(4, day(0));
  EXPECT_EQ(3, day(-1));
  EXPECT_EQ(6, day(-40));
  EXPECT_EQ(2, day(40));
  EXPECT_EQ(3, day(18262));
  EXPECT_EQ(5, day(-18262));
}

TEST_F(DateTimeFunctionsTest, dayOfWeekTimestampWithTimezone) {
  const auto dayOfWeekTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "day_of_week(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      3, dayOfWeekTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      4, dayOfWeekTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      5,
      dayOfWeekTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      3,
      dayOfWeekTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      3,
      dayOfWeekTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      3,
      dayOfWeekTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, dayOfWeekTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, dayOfYear) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("day_of_year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Timestamp(0, 0)));
  EXPECT_EQ(365, day(Timestamp(-1, 9000)));
  EXPECT_EQ(273, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(274, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(279, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(304, day(Timestamp(1635668100, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(365, day(Timestamp(0, 0)));
  EXPECT_EQ(365, day(Timestamp(-1, 9000)));
  EXPECT_EQ(273, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(274, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(279, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(304, day(Timestamp(1635668100, 0)));
}

TEST_F(DateTimeFunctionsTest, dayOfYearDate) {
  const auto day = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("day_of_year(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(0));
  EXPECT_EQ(365, day(-1));
  EXPECT_EQ(326, day(-40));
  EXPECT_EQ(41, day(40));
  EXPECT_EQ(1, day(18262));
  EXPECT_EQ(2, day(-18262));
}

TEST_F(DateTimeFunctionsTest, dayOfYearTimestampWithTimezone) {
  const auto dayOfYearTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "day_of_year(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      365, dayOfYearTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      1, dayOfYearTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      334,
      dayOfYearTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      33,
      dayOfYearTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      108,
      dayOfYearTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      257,
      dayOfYearTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, dayOfYearTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, yearOfWeek) {
  const auto yow = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("year_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(Timestamp(0, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-1, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-345600, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-259200, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31536000, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31708800, 0)));
  EXPECT_EQ(1971, yow(Timestamp(31795200, 0)));
  EXPECT_EQ(2021, yow(Timestamp(1632989700, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(Timestamp(0, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-1, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-345600, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-259200, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31536000, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31708800, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31795200, 0)));
  EXPECT_EQ(2021, yow(Timestamp(1632989700, 0)));
}

TEST_F(DateTimeFunctionsTest, yearOfWeekDate) {
  const auto yow = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("year_of_week(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(0));
  EXPECT_EQ(1970, yow(-1));
  EXPECT_EQ(1969, yow(-4));
  EXPECT_EQ(1970, yow(-3));
  EXPECT_EQ(1970, yow(365));
  EXPECT_EQ(1970, yow(367));
  EXPECT_EQ(1971, yow(368));
  EXPECT_EQ(2021, yow(18900));
}

TEST_F(DateTimeFunctionsTest, yearOfWeekTimestampWithTimezone) {
  const auto yearOfWeekTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "year_of_week(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      1970,
      yearOfWeekTimestampWithTimezone(TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      1970,
      yearOfWeekTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      1973,
      yearOfWeekTimestampWithTimezone(
          TimestampWithTimezone(123456789000, "+14:00")));
  EXPECT_EQ(
      1966,
      yearOfWeekTimestampWithTimezone(
          TimestampWithTimezone(-123456789000, "+03:00")));
  EXPECT_EQ(
      2001,
      yearOfWeekTimestampWithTimezone(
          TimestampWithTimezone(987654321000, "-07:00")));
  EXPECT_EQ(
      1938,
      yearOfWeekTimestampWithTimezone(
          TimestampWithTimezone(-987654321000, "-13:00")));
  EXPECT_EQ(std::nullopt, yearOfWeekTimestampWithTimezone(std::nullopt));
}

TEST_F(DateTimeFunctionsTest, minute) {
  const auto minute = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int64_t>("minute(c0)", date);
  };
  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(0, minute(Timestamp(0, 0)));
  EXPECT_EQ(59, minute(Timestamp(-1, 9000)));
  EXPECT_EQ(6, minute(Timestamp(4000000000, 0)));
  EXPECT_EQ(6, minute(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(4, minute(Timestamp(998474645, 321000000)));
  EXPECT_EQ(55, minute(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Asia/Kolkata");

  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(30, minute(Timestamp(0, 0)));
  EXPECT_EQ(29, minute(Timestamp(-1, 9000)));
  EXPECT_EQ(36, minute(Timestamp(4000000000, 0)));
  EXPECT_EQ(36, minute(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(34, minute(Timestamp(998474645, 321000000)));
  EXPECT_EQ(25, minute(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, minuteDate) {
  const auto minute = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("minute(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(0, minute(0));
  EXPECT_EQ(0, minute(-1));
  EXPECT_EQ(0, minute(40));
  EXPECT_EQ(0, minute(40));
  EXPECT_EQ(0, minute(18262));
  EXPECT_EQ(0, minute(-18262));
}

TEST_F(DateTimeFunctionsTest, minuteTimestampWithTimezone) {
  const auto minuteTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "minute(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(std::nullopt, minuteTimestampWithTimezone(std::nullopt));
  EXPECT_EQ(0, minuteTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      30, minuteTimestampWithTimezone(TimestampWithTimezone(0, "+05:30")));
  EXPECT_EQ(
      6,
      minuteTimestampWithTimezone(
          TimestampWithTimezone(4000000000000, "+00:00")));
  EXPECT_EQ(
      36,
      minuteTimestampWithTimezone(
          TimestampWithTimezone(4000000000000, "+05:30")));
  EXPECT_EQ(
      4,
      minuteTimestampWithTimezone(
          TimestampWithTimezone(998474645000, "+00:00")));
  EXPECT_EQ(
      34,
      minuteTimestampWithTimezone(
          TimestampWithTimezone(998474645000, "+05:30")));
  EXPECT_EQ(
      59, minuteTimestampWithTimezone(TimestampWithTimezone(-1000, "+00:00")));
  EXPECT_EQ(
      29, minuteTimestampWithTimezone(TimestampWithTimezone(-1000, "+05:30")));
}

TEST_F(DateTimeFunctionsTest, minuteTime) {
  const auto minute = [&](std::optional<int64_t> time) {
    return evaluateOnce<int64_t>(
        "minute(c0)",
        makeRowVector({makeNullableFlatVector<int64_t>({time}, TIME())}));
  };

  // null handling
  EXPECT_EQ(std::nullopt, minute(std::nullopt));

  // edge cases: beginning and end of valid range
  EXPECT_EQ(0, minute(0)); // 00:00:00.000
  EXPECT_EQ(59, minute(86399999)); // 23:59:59.999

  // boundary values for optimization testing
  EXPECT_EQ(0, minute(59999)); // 00:00:59.999 - last second of first minute
  EXPECT_EQ(1, minute(60000)); // 00:01:00.000 - first second of second minute
  EXPECT_EQ(59, minute(3599999)); // 00:59:59.999 - last second of first hour
  EXPECT_EQ(0, minute(3600000)); // 01:00:00.000 - first second of second hour

  // time values spanning different hours to test optimization
  EXPECT_EQ(
      15,
      minute(
          15 * kMillisInMinute + 30 * kMillisInSecond + 123)); // 00:15:30.123
  EXPECT_EQ(
      30,
      minute(5 * kMillisInHour + 30 * kMillisInMinute + 45000)); // 05:30:45.000
  EXPECT_EQ(
      45, minute(12 * kMillisInHour + 45 * kMillisInMinute)); // 12:45:00.000
  EXPECT_EQ(0, minute(23 * kMillisInHour)); // 23:00:00.000

  // comprehensive minute coverage for edge case testing
  for (int m = 0; m < 60; ++m) {
    EXPECT_EQ(m, minute(m * kMillisInMinute)); // exact minute boundaries
    EXPECT_EQ(m, minute(m * kMillisInMinute + 500)); // mid-second
    EXPECT_EQ(m, minute(m * kMillisInMinute + 59999)); // end of minute
  }

  // hour boundaries with different minute values
  for (int h = 0; h < 24; ++h) {
    EXPECT_EQ(0, minute(h * kMillisInHour)); // start of each hour
    EXPECT_EQ(
        30,
        minute(
            h * kMillisInHour + 30 * kMillisInMinute)); // 30 minutes into hour
    EXPECT_EQ(
        59,
        minute(
            h * kMillisInHour + 59 * kMillisInMinute +
            59999)); // last moment of hour
  }

  // comprehensive data type coverage - test various millisecond precision
  // values
  EXPECT_EQ(
      42,
      minute(13 * kMillisInHour + 42 * kMillisInMinute + 1)); // precision: 1ms
  EXPECT_EQ(
      42,
      minute(
          13 * kMillisInHour + 42 * kMillisInMinute + 10)); // precision: 10ms
  EXPECT_EQ(
      42,
      minute(
          13 * kMillisInHour + 42 * kMillisInMinute + 100)); // precision: 100ms
  EXPECT_EQ(
      42,
      minute(
          13 * kMillisInHour + 42 * kMillisInMinute + 999)); // precision: 999ms

  // performance critical values for optimization validation
  int64_t performanceTestValues[] = {
      0, // midnight
      kMillisInHour / 2, // 30 minutes
      kMillisInHour - 1, // 59:59.999
      kMillisInHour, // 01:00:00.000
      12 * kMillisInHour + 30 * kMillisInMinute, // 12:30:00.000 (noon+)
      86399999 // 23:59:59.999 (end of day)
  };

  int64_t expectedMinutes[] = {0, 30, 59, 0, 30, 59};

  for (size_t i = 0;
       i < sizeof(performanceTestValues) / sizeof(performanceTestValues[0]);
       ++i) {
    EXPECT_EQ(expectedMinutes[i], minute(performanceTestValues[i]));
  }
}

TEST_F(DateTimeFunctionsTest, minuteTimeInvalidRange) {
  const auto minute = [&](int64_t time) {
    return evaluateOnce<int64_t>(
        "minute(c0)", makeRowVector({makeFlatVector<int64_t>({time}, TIME())}));
  };

  // test out-of-range values that should throw errors
  VELOX_ASSERT_THROW(
      minute(-1), "TIME value -1 is out of valid range [0, 86399999]");

  VELOX_ASSERT_THROW(
      minute(86400000),
      "TIME value 86400000 is out of valid range [0, 86399999]");

  VELOX_ASSERT_THROW(
      minute(-86400000),
      "TIME value -86400000 is out of valid range [0, 86399999]");

  VELOX_ASSERT_THROW(
      minute(std::numeric_limits<int64_t>::max()),
      fmt::format(
          "TIME value {} is out of valid range [0, 86399999]",
          std::numeric_limits<int64_t>::max()));

  VELOX_ASSERT_THROW(
      minute(std::numeric_limits<int64_t>::min()),
      fmt::format(
          "TIME value {} is out of valid range [0, 86399999]",
          std::numeric_limits<int64_t>::min()));
}

TEST_F(DateTimeFunctionsTest, second) {
  const auto second = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int64_t>("second(c0)", timestamp);
  };
  EXPECT_EQ(std::nullopt, second(std::nullopt));
  EXPECT_EQ(0, second(Timestamp(0, 0)));
  EXPECT_EQ(40, second(Timestamp(4000000000, 0)));
  EXPECT_EQ(59, second(Timestamp(-1, 123000000)));
  EXPECT_EQ(59, second(Timestamp(-1, Timestamp::kMaxNanos)));
}

TEST_F(DateTimeFunctionsTest, secondDate) {
  const auto second = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("second(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, second(std::nullopt));
  EXPECT_EQ(0, second(0));
  EXPECT_EQ(0, second(-1));
  EXPECT_EQ(0, second(-40));
  EXPECT_EQ(0, second(40));
  EXPECT_EQ(0, second(18262));
  EXPECT_EQ(0, second(-18262));
}

TEST_F(DateTimeFunctionsTest, secondTimestampWithTimezone) {
  const auto secondTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "second(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(std::nullopt, secondTimestampWithTimezone(std::nullopt));
  EXPECT_EQ(0, secondTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(0, secondTimestampWithTimezone(TimestampWithTimezone(0, "+05:30")));
  EXPECT_EQ(
      40,
      secondTimestampWithTimezone(
          TimestampWithTimezone(4000000000000, "+00:00")));
  EXPECT_EQ(
      40,
      secondTimestampWithTimezone(
          TimestampWithTimezone(4000000000000, "+05:30")));
  EXPECT_EQ(
      59, secondTimestampWithTimezone(TimestampWithTimezone(-1000, "+00:00")));
  EXPECT_EQ(
      59, secondTimestampWithTimezone(TimestampWithTimezone(-1000, "+05:30")));
}

TEST_F(DateTimeFunctionsTest, secondTime) {
  const auto second = [&](std::optional<int64_t> time) {
    return evaluateOnce<int64_t>("second(c0)", TIME(), time);
  };

  // null handling
  EXPECT_EQ(std::nullopt, second(std::nullopt));

  // boundary tests - optimization: (time / 1000) % 60
  // lower boundary: 0 <= time < 86400000
  EXPECT_EQ(0, second(0)); // exactly midnight - 00:00:00.000
  EXPECT_EQ(0, second(999)); // 00:00:00.999 - last millisecond of second 0
  EXPECT_EQ(1, second(1000)); // 00:00:01.000 - first millisecond of second 1
  EXPECT_EQ(59, second(59000)); // 00:00:59.000 - first millisecond of second 59
  EXPECT_EQ(0, second(60000)); // 00:01:00.000 - first second of next minute
  EXPECT_EQ(59, second(86399000)); // 23:59:59.000
  EXPECT_EQ(59, second(86399999)); // 23:59:59.999 - upper boundary exclusive

  // representative test cases across different times
  EXPECT_EQ(30, second(30000)); // 00:00:30.000
  EXPECT_EQ(15, second(75000)); // 00:01:15.000
  // 01:01:25 = 3600000 + 60000 + 25000 = 3685000
  EXPECT_EQ(25, second(3685000)); // 01:01:25.000
  EXPECT_EQ(3, second(3723123)); // 01:02:03.123

  // verify optimization correctness with modulo boundary conditions
  // seconds should cycle every 60 seconds regardless of hours/minutes
  EXPECT_EQ(0, second(3600000)); // 01:00:00.000
  EXPECT_EQ(30, second(3630000)); // 01:00:30.000
  EXPECT_EQ(0, second(7200000)); // 02:00:00.000
  EXPECT_EQ(45, second(43245000)); // 12:00:45.000

  // test millisecond precision is correctly truncated
  EXPECT_EQ(15, second(15123)); // 00:00:15.123
  EXPECT_EQ(15, second(15999)); // 00:00:15.999

  // error conditions - invalid range validation
  EXPECT_THROW(second(-1), VeloxUserError); // negative time
  EXPECT_THROW(
      second(86400000),
      VeloxUserError); // exactly 24:00:00.000 (exclusive upper bound)
  EXPECT_THROW(
      second(std::numeric_limits<int64_t>::max()),
      VeloxUserError); // overflow case
}

TEST_F(DateTimeFunctionsTest, millisecond) {
  const auto millisecond = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int64_t>("millisecond(c0)", timestamp);
  };
  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));
  EXPECT_EQ(0, millisecond(Timestamp(0, 0)));
  EXPECT_EQ(0, millisecond(Timestamp(4000000000, 0)));
  EXPECT_EQ(123, millisecond(Timestamp(-1, 123000000)));
  EXPECT_EQ(999, millisecond(Timestamp(-1, Timestamp::kMaxNanos)));
}

TEST_F(DateTimeFunctionsTest, millisecondDate) {
  const auto millisecond = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("millisecond(c0)", DATE(), date);
  };
  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));
  EXPECT_EQ(0, millisecond(0));
  EXPECT_EQ(0, millisecond(-1));
  EXPECT_EQ(0, millisecond(-40));
  EXPECT_EQ(0, millisecond(40));
  EXPECT_EQ(0, millisecond(18262));
  EXPECT_EQ(0, millisecond(-18262));
}

TEST_F(DateTimeFunctionsTest, millisecondTimestampWithTimezone) {
  const auto millisecondTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int64_t>(
            "millisecond(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(std::nullopt, millisecondTimestampWithTimezone(std::nullopt));
  EXPECT_EQ(
      0, millisecondTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      0, millisecondTimestampWithTimezone(TimestampWithTimezone(0, "+05:30")));
  EXPECT_EQ(
      123,
      millisecondTimestampWithTimezone(
          TimestampWithTimezone(4000000000123, "+00:00")));
  EXPECT_EQ(
      123,
      millisecondTimestampWithTimezone(
          TimestampWithTimezone(4000000000123, "+05:30")));
  EXPECT_EQ(
      20,
      millisecondTimestampWithTimezone(TimestampWithTimezone(-980, "+00:00")));
  EXPECT_EQ(
      20,
      millisecondTimestampWithTimezone(TimestampWithTimezone(-980, "+05:30")));
}

TEST_F(DateTimeFunctionsTest, millisecondTime) {
  const auto millisecond = [&](std::optional<int64_t> time) {
    return evaluateOnce<int64_t>("millisecond(c0)", TIME(), time);
  };

  // Null handling
  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));

  // Basic cases
  EXPECT_EQ(0, millisecond(0)); // 00:00:00.000
  EXPECT_EQ(0, millisecond(1000)); // 00:00:01.000
  EXPECT_EQ(123, millisecond(1123)); // 00:00:01.123
  EXPECT_EQ(456, millisecond(2456)); // 00:00:02.456
  EXPECT_EQ(999, millisecond(999)); // 00:00:00.999
  EXPECT_EQ(123, millisecond(3661123)); // 01:01:01.123

  // Boundary values
  EXPECT_EQ(0, millisecond(86399000)); // 23:59:59.000
  EXPECT_EQ(999, millisecond(86399999)); // 23:59:59.999 (max valid TIME)

  // Different time components to verify only millisecond part matters
  EXPECT_EQ(500, millisecond(3600500)); // 01:00:00.500
  EXPECT_EQ(500, millisecond(7200500)); // 02:00:00.500
  EXPECT_EQ(500, millisecond(43200500)); // 12:00:00.500

  // Second boundaries
  EXPECT_EQ(0, millisecond(60000)); // 00:01:00.000
  EXPECT_EQ(999, millisecond(60999)); // 00:01:00.999
  EXPECT_EQ(1, millisecond(60001)); // 00:01:00.001

  // Comprehensive modulo testing - verify millisecond() == time % 1000
  for (int64_t base : {0, 1000, 60000, 3600000, 86399000}) {
    for (int64_t ms = 0; ms < 1000; ms += 100) {
      int64_t timeValue = base + ms;
      if (timeValue <= 86399999) { // Ensure within valid range
        EXPECT_EQ(ms, millisecond(timeValue));
      }
    }
  }

  // Test all possible millisecond values (0-999) in first second
  for (int64_t ms = 0; ms < 1000; ++ms) {
    EXPECT_EQ(ms, millisecond(ms));
  }

  // Test across all hours to ensure hour component doesn't affect result
  for (int hour = 0; hour < 24; ++hour) {
    int64_t baseTime = hour * 3600000; // Convert hour to milliseconds
    EXPECT_EQ(0, millisecond(baseTime)); // .000
    EXPECT_EQ(1, millisecond(baseTime + 1)); // .001
    EXPECT_EQ(500, millisecond(baseTime + 500)); // .500
    EXPECT_EQ(999, millisecond(baseTime + 999)); // .999
  }

  // Test edge cases just before and after boundaries
  EXPECT_EQ(998, millisecond(86399998)); // 23:59:59.998
  EXPECT_EQ(1, millisecond(1001)); // 00:00:01.001
  EXPECT_EQ(999, millisecond(1999)); // 00:00:01.999
  EXPECT_EQ(0, millisecond(2000)); // 00:00:02.000

  // Error cases - values outside valid TIME range [0, 86399999]
  VELOX_ASSERT_THROW(
      millisecond(-1), "TIME value -1 is out of range [0, 86400000)");
  VELOX_ASSERT_THROW(
      millisecond(-1000), "TIME value -1000 is out of range [0, 86400000)");
  VELOX_ASSERT_THROW(
      millisecond(86400000),
      "TIME value 86400000 is out of range [0, 86400000)");
  VELOX_ASSERT_THROW(
      millisecond(100000000),
      "TIME value 100000000 is out of range [0, 86400000)");

  // Test vectorized execution with mixed valid values
  auto timeVector = makeFlatVector<int64_t>(
      {0, // 00:00:00.000
       1123, // 00:00:01.123
       60999, // 00:01:00.999
       3661456, // 01:01:01.456
       43200789, // 12:00:00.789
       86399999}, // 23:59:59.999
      TIME());

  auto result = evaluate<FlatVector<int64_t>>(
      "millisecond(c0)", makeRowVector({timeVector}));

  auto expected = makeFlatVector<int64_t>({0, 123, 999, 456, 789, 999});
  assertEqualVectors(expected, result);

  // Test vectorized execution with nulls
  auto timeVectorWithNulls = makeNullableFlatVector<int64_t>(
      {0, // 00:00:00.000
       std::nullopt, // null
       1123, // 00:00:01.123
       std::nullopt, // null
       86399999}, // 23:59:59.999
      TIME());

  auto resultWithNulls = evaluate<FlatVector<int64_t>>(
      "millisecond(c0)", makeRowVector({timeVectorWithNulls}));

  auto expectedWithNulls = makeNullableFlatVector<int64_t>(
      {0, std::nullopt, 123, std::nullopt, 999});
  assertEqualVectors(expectedWithNulls, resultWithNulls);
}

TEST_F(DateTimeFunctionsTest, millisecondTimeWithTimezone) {
  const auto millisecond = [&](std::optional<int64_t> timeWithTimezone) {
    return evaluateOnce<int64_t>(
        "millisecond(c0)", TIME_WITH_TIME_ZONE(), timeWithTimezone);
  };

  const auto parse = [](std::string_view timeString) -> std::optional<int64_t> {
    auto result =
        util::fromTimeWithTimezoneString(timeString.data(), timeString.size());
    VELOX_CHECK(!result.hasError(), "{}", result.error().message());
    return result.value();
  };

  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));

  EXPECT_EQ(0, millisecond(parse("00:00:00.000+00:00")));
  EXPECT_EQ(123, millisecond(parse("06:11:37.123+00:00")));
  EXPECT_EQ(123, millisecond(parse("06:11:37.123-05:00")));
  EXPECT_EQ(123, millisecond(parse("06:11:37.123+05:30")));
  EXPECT_EQ(999, millisecond(parse("23:59:59.999+14:00")));
  EXPECT_EQ(1, millisecond(parse("00:00:00.001-14:00")));

  auto input = makeNullableFlatVector<int64_t>(
      {
          parse("00:00:00.000+00:00"),
          parse("06:11:37.123+00:00"),
          std::nullopt,
          parse("23:59:59.999+14:00"),
          parse("00:00:00.001-14:00"),
      },
      TIME_WITH_TIME_ZONE());

  auto result =
      evaluate<FlatVector<int64_t>>("millisecond(c0)", makeRowVector({input}));

  auto expected = makeNullableFlatVector<int64_t>(
      {0, 123, std::nullopt, 999, 1});
  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, extractFromIntervalDayTime) {
  const auto millis = 5 * kMillisInDay + 7 * kMillisInHour +
      11 * kMillisInMinute + 13 * kMillisInSecond + 17;

  auto extract = [&](const std::string& unit, int64_t millis) {
    return evaluateOnce<int64_t>(
               fmt::format("{}(c0)", unit),
               INTERVAL_DAY_TIME(),
               std::optional(millis))
        .value();
  };

  EXPECT_EQ(17, extract("millisecond", millis));
  EXPECT_EQ(13, extract("second", millis));
  EXPECT_EQ(11, extract("minute", millis));
  EXPECT_EQ(7, extract("hour", millis));
  EXPECT_EQ(5, extract("day", millis));
}

TEST_F(DateTimeFunctionsTest, extractFromIntervalYearMonth) {
  const auto months = 3 * 12 + 4;

  auto extract = [&](const std::string& unit, int32_t months) {
    return evaluateOnce<int64_t>(
               fmt::format("{}(c0)", unit),
               INTERVAL_YEAR_MONTH(),
               std::optional(months))
        .value();
  };

  EXPECT_EQ(3, extract("year", months));
  EXPECT_EQ(4, extract("month", months));
}

TEST_F(DateTimeFunctionsTest, dateTrunc) {
  const auto dateTrunc = [&](const std::string& unit,
                             std::optional<Timestamp> timestamp) {
    return evaluateOnce<Timestamp>(
        fmt::format("date_trunc('{}', c0)", unit), timestamp);
  };

  disableAdjustTimestampToTimezone();

  EXPECT_EQ(std::nullopt, dateTrunc("second", std::nullopt));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 0)));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 123)));
  EXPECT_EQ(Timestamp(-1, 0), dateTrunc("second", Timestamp(-1, 0)));
  EXPECT_EQ(Timestamp(-1, 0), dateTrunc("second", Timestamp(-1, 123)));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("day", Timestamp(0, 123)));
  EXPECT_EQ(
      Timestamp(998474645, 0),
      dateTrunc("second", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998474640, 0),
      dateTrunc("minute", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998474400, 0),
      dateTrunc("hour", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998438400, 0),
      dateTrunc("day", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998265600, 0),
      dateTrunc("week", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(996624000, 0),
      dateTrunc("month", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(993945600, 0),
      dateTrunc("quarter", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(978307200, 0),
      dateTrunc("year", Timestamp(998'474'645, 321'001'234)));

  setQueryTimeZone("America/Los_Angeles");

  EXPECT_EQ(std::nullopt, dateTrunc("second", std::nullopt));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 0)));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 123)));

  EXPECT_EQ(Timestamp(-57600, 0), dateTrunc("day", Timestamp(0, 0)));
  EXPECT_EQ(
      Timestamp(998474645, 0),
      dateTrunc("second", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998474640, 0),
      dateTrunc("minute", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998474400, 0),
      dateTrunc("hour", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998463600, 0),
      dateTrunc("day", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998290800, 0),
      dateTrunc("week", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(996649200, 0),
      dateTrunc("month", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(993970800, 0),
      dateTrunc("quarter", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(978336000, 0),
      dateTrunc("year", Timestamp(998'474'645, 321'001'234)));

  // Check that truncation during daylight saving transition where conversions
  // may be ambiguous return the right values.
  EXPECT_EQ(
      Timestamp(1667725200, 0), dateTrunc("hour", Timestamp(1667725200, 0)));
  EXPECT_EQ(
      Timestamp(1667725200, 0), dateTrunc("minute", Timestamp(1667725200, 0)));

  setQueryTimeZone("Asia/Kolkata");

  EXPECT_EQ(std::nullopt, dateTrunc("second", std::nullopt));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 0)));
  EXPECT_EQ(Timestamp(0, 0), dateTrunc("second", Timestamp(0, 123)));
  EXPECT_EQ(Timestamp(-19800, 0), dateTrunc("day", Timestamp(0, 0)));
  EXPECT_EQ(
      Timestamp(998474645, 0),
      dateTrunc("second", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998474640, 0),
      dateTrunc("minute", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998472600, 0),
      dateTrunc("hour", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998418600, 0),
      dateTrunc("day", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(998245800, 0),
      dateTrunc("week", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(996604200, 0),
      dateTrunc("month", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(993925800, 0),
      dateTrunc("quarter", Timestamp(998'474'645, 321'001'234)));
  EXPECT_EQ(
      Timestamp(978287400, 0),
      dateTrunc("year", Timestamp(998'474'645, 321'001'234)));
}

TEST_F(DateTimeFunctionsTest, dateTruncDate) {
  const auto dateTrunc = [&](const std::string& unit,
                             std::optional<int32_t> date) {
    return evaluateOnce<int32_t>(
        fmt::format("date_trunc('{}', c0)", unit), DATE(), date);
  };

  EXPECT_EQ(std::nullopt, dateTrunc("year", std::nullopt));

  // Date(0) is 1970-01-01.
  EXPECT_EQ(0, dateTrunc("day", 0));
  EXPECT_EQ(0, dateTrunc("year", 0));
  EXPECT_EQ(0, dateTrunc("quarter", 0));
  EXPECT_EQ(0, dateTrunc("month", 0));
  EXPECT_THROW(dateTrunc("second", 0), VeloxUserError);
  EXPECT_THROW(dateTrunc("minute", 0), VeloxUserError);
  EXPECT_THROW(dateTrunc("hour", 0), VeloxUserError);

  // Date(18297) is 2020-02-05.
  EXPECT_EQ(18297, dateTrunc("day", 18297));
  EXPECT_EQ(18293, dateTrunc("month", 18297));
  EXPECT_EQ(18262, dateTrunc("quarter", 18297));
  EXPECT_EQ(18262, dateTrunc("year", 18297));
  EXPECT_THROW(dateTrunc("second", 18297), VeloxUserError);
  EXPECT_THROW(dateTrunc("minute", 18297), VeloxUserError);
  EXPECT_THROW(dateTrunc("hour", 18297), VeloxUserError);

  // Date(-18297) is 1919-11-28.
  EXPECT_EQ(-18297, dateTrunc("day", -18297));
  EXPECT_EQ(-18324, dateTrunc("month", -18297));
  EXPECT_EQ(-18355, dateTrunc("quarter", -18297));
  EXPECT_EQ(-18628, dateTrunc("year", -18297));
  EXPECT_THROW(dateTrunc("second", -18297), VeloxUserError);
  EXPECT_THROW(dateTrunc("minute", -18297), VeloxUserError);
  EXPECT_THROW(dateTrunc("hour", -18297), VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, dateTruncDateForWeek) {
  const auto dateTrunc = [&](const std::string& unit,
                             std::optional<int32_t> date) {
    return evaluateOnce<int32_t>(
        fmt::format("date_trunc('{}', c0)", unit), DATE(), date);
  };

  // Date(19576) is 2023-08-07, which is Monday, should return Monday
  EXPECT_EQ(19576, dateTrunc("week", 19576));

  // Date(19579) is 2023-08-10, Thur, should return Monday
  EXPECT_EQ(19576, dateTrunc("week", 19579));

  // Date(19570) is 2023-08-01, A non-Monday(Tue) date at the beginning of a
  // month when the preceding Monday falls in the previous month. should return
  // 2023-07-31(19569), which is previous Monday
  EXPECT_EQ(19569, dateTrunc("week", 19570));

  // Date(19358) is 2023-01-01, A non-Monday(Sunday) date at the beginning of
  // January where the preceding Monday falls in the previous year. should
  // return 2022-12-26(19352), which is previous Monday
  EXPECT_EQ(19352, dateTrunc("week", 19358));

  // Date(19783) is 2024-03-01, A non-Monday(Friday) date which will go over to
  // a leap day (February 29th) in a leap year. should return 2024-02-26(19352),
  // which is previous Monday
  EXPECT_EQ(19779, dateTrunc("week", 19783));
}

// Reference dateTruncDateForWeek for test cases explanaitons
TEST_F(DateTimeFunctionsTest, dateTruncTimeStampForWeek) {
  const auto dateTrunc = [&](const std::string& unit,
                             std::optional<Timestamp> timestamp) {
    return evaluateOnce<Timestamp>(
        fmt::format("date_trunc('{}', c0)", unit), timestamp);
  };

  EXPECT_EQ(
      Timestamp(19576 * 24 * 60 * 60, 0),
      dateTrunc("week", Timestamp(19576 * 24 * 60 * 60, 321'001'234)));

  EXPECT_EQ(
      Timestamp(19576 * 24 * 60 * 60, 0),
      dateTrunc("week", Timestamp(19579 * 24 * 60 * 60 + 500, 321'001'234)));

  EXPECT_EQ(
      Timestamp(19569 * 24 * 60 * 60, 0),
      dateTrunc("week", Timestamp(19570 * 24 * 60 * 60 + 500, 321'001'234)));

  EXPECT_EQ(
      Timestamp(19352 * 24 * 60 * 60, 0),
      dateTrunc("week", Timestamp(19358 * 24 * 60 * 60 + 500, 321'001'234)));

  EXPECT_EQ(
      Timestamp(19779 * 24 * 60 * 60, 0),
      dateTrunc("week", Timestamp(19783 * 24 * 60 * 60 + 500, 321'001'234)));
}

// Logical Steps
// 1. Convert Original Millisecond Input to UTC
// 2. Apply Time Zone Offset
// 3. Truncate to the Nearest "Unit"
// 4. Convert Back to UTC (remove Time Zone offset)
// 5. Convert Back to Milliseconds Since the Unix Epoch
TEST_F(DateTimeFunctionsTest, dateTruncTimeStampWithTimezoneForWeek) {
  const auto evaluateDateTrunc = [&](const std::string& truncUnit,
                                     int64_t inputTimestamp,
                                     const std::string& timeZone,
                                     int64_t expectedTimestamp) {
    EXPECT_EQ(
        TimestampWithTimezone::pack(
            TimestampWithTimezone(expectedTimestamp, timeZone.c_str())),
        evaluateOnce<int64_t>(
            fmt::format("date_trunc('{}', c0)", truncUnit),
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(
                TimestampWithTimezone(inputTimestamp, timeZone))));
  };
  // input 2023-08-07 00:00:00 (19576 days) with timeZone +01:00
  // output 2023-08-06 23:00:00" in UTC.(1691362800000)
  auto inputMilli = int64_t(19576) * 24 * 60 * 60 * 1000;
  auto outputMilli = inputMilli - int64_t(1) * 60 * 60 * 1000;
  evaluateDateTrunc("week", inputMilli, "+01:00", outputMilli);

  // Date(19579) is 2023-08-10, Thur, should return Monday UTC (previous Sunday
  // in +03:00 timezone)
  inputMilli = int64_t(19579) * 24 * 60 * 60 * 1000;
  outputMilli = inputMilli - int64_t(3) * 24 * 60 * 60 * 1000 -
      int64_t(3) * 60 * 60 * 1000;
  evaluateDateTrunc("week", inputMilli, "+03:00", outputMilli);

  // Date(19570) is 2023-08-01, A non-Monday(Tue) date at the beginning of a
  // month when the preceding Monday falls in the previous month. should return
  // 2023-07-31(19569), which is previous Monday EXPECT_EQ(19569,
  // dateTrunc("week", 19570));
  inputMilli = int64_t(19570) * 24 * 60 * 60 * 1000;
  outputMilli = inputMilli - int64_t(1) * 24 * 60 * 60 * 1000 -
      int64_t(3) * 60 * 60 * 1000;
  evaluateDateTrunc("week", inputMilli, "+03:00", outputMilli);

  // Date(19570) is 2023-08-01, which is Tuesday; TimeZone is -05:00, so input
  // will become Monday. 2023-07-31 19:00:00, which will truncate to 2023-07-31
  // 00:00:00
  // TODO : Need to double-check with presto logic
  inputMilli = int64_t(19570) * 24 * 60 * 60 * 1000;
  outputMilli =
      int64_t(19569) * 24 * 60 * 60 * 1000 + int64_t(5) * 60 * 60 * 1000;
  evaluateDateTrunc("week", inputMilli, "-05:00", outputMilli);
}

TEST_F(DateTimeFunctionsTest, dateTruncTimeStampWithTimezoneStringForWeek) {
  const auto evaluateDateTruncFromStrings = [&](const std::string& truncUnit,
                                                const std::string&
                                                    inputTimestamp,
                                                const std::string&
                                                    expectedTimestamp) {
    assertEqualVectors(
        evaluate<FlatVector<int64_t>>(
            "parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ')",
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{expectedTimestamp}})})),
        evaluate<FlatVector<int64_t>>(
            fmt::format(
                "date_trunc('{}', parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ'))",
                truncUnit),
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{inputTimestamp}})})));
  };
  // Monday
  evaluateDateTruncFromStrings(
      "week", "2023-08-07+23:01:02+14:00", "2023-08-07+00:00:00+14:00");

  // Thur
  evaluateDateTruncFromStrings(
      "week", "2023-08-10+23:01:02+14:00", "2023-08-07+00:00:00+14:00");

  // 2023-08-01, A non-Monday(Tue) date at the beginning of a
  // month when the preceding Monday falls in the previous month. should return
  // 2023-07-31, which is previous Monday
  evaluateDateTruncFromStrings(
      "week", "2023-08-01+23:01:02+14:00", "2023-07-31+00:00:00+14:00");

  // 2023-01-01, A non-Monday(Sunday) date at the beginning of
  // January where the preceding Monday falls in the previous year. should
  // return 2022-12-26, which is previous Monday
  evaluateDateTruncFromStrings(
      "week", "2023-01-01+23:01:02+14:00", "2022-12-26+00:00:00+14:00");

  // 2024-03-01, A non-Monday(Friday) date which will go over to
  // a leap day (February 29th) in a leap year. should return 2024-02-26,
  // which is previous Monday
  evaluateDateTruncFromStrings(
      "week", "2024-03-01+23:01:02+14:00", "2024-02-26+00:00:00+14:00");
}
TEST_F(DateTimeFunctionsTest, dateTruncTimestampWithTimezone) {
  const auto evaluateDateTrunc = [&](const std::string& truncUnit,
                                     int64_t inputTimestamp,
                                     const std::string& timeZone,
                                     int64_t expectedTimestamp) {
    EXPECT_EQ(
        TimestampWithTimezone::pack(
            TimestampWithTimezone(expectedTimestamp, timeZone.c_str())),
        evaluateOnce<int64_t>(
            fmt::format("date_trunc('{}', c0)", truncUnit),
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(
                TimestampWithTimezone(inputTimestamp, timeZone))));
  };

  evaluateDateTrunc("second", 123, "+01:00", 0);
  evaluateDateTrunc("second", 1123, "-03:00", 1000);
  evaluateDateTrunc("second", -1123, "+03:00", -2000);
  evaluateDateTrunc("second", 1234567000, "+14:00", 1234567000);
  evaluateDateTrunc("second", -1234567000, "-09:00", -1234567000);

  evaluateDateTrunc("minute", 123, "+01:00", 0);
  evaluateDateTrunc("minute", 1123, "-03:00", 0);
  evaluateDateTrunc("minute", -1123, "+03:00", -60000);
  evaluateDateTrunc("minute", 1234567000, "+14:00", 1234560000);
  evaluateDateTrunc("minute", -1234567000, "-09:00", -1234620000);

  evaluateDateTrunc("hour", 123, "+01:00", 0);
  evaluateDateTrunc("hour", 1123, "-03:00", 0);
  evaluateDateTrunc("hour", -1123, "+05:30", -1800000);
  evaluateDateTrunc("hour", 1234567000, "+14:00", 1231200000);
  evaluateDateTrunc("hour", -1234567000, "-09:30", -1236600000);

  evaluateDateTrunc("day", 123, "+01:00", -3600000);
  evaluateDateTrunc("day", 1123, "-03:00", -86400000 + 3600000 * 3);
  evaluateDateTrunc("day", -1123, "+05:30", 0 - 3600000 * 5 - 1800000);
  evaluateDateTrunc("day", 1234567000, "+14:00", 1159200000);
  evaluateDateTrunc("day", -1234567000, "-09:30", -1261800000);

  evaluateDateTrunc("month", 123, "-01:00", -2674800000);
  evaluateDateTrunc("month", 1234567000, "+14:00", -50400000);
  evaluateDateTrunc("month", -1234567000, "-09:30", -2644200000);

  evaluateDateTrunc("quarter", 123, "-01:00", -7945200000);
  evaluateDateTrunc("quarter", 123456789000, "+14:00", 118231200000);
  evaluateDateTrunc("quarter", -123456789000, "-09:30", -126196200000);

  evaluateDateTrunc("year", 123, "-01:00", -31532400000);
  evaluateDateTrunc("year", 123456789000, "+14:00", 94644000000);
  evaluateDateTrunc("year", -123456789000, "-09:30", -126196200000);

  // Test cases that land on an ambiguous time.
  // The first 1 AM
  // 11/3/2024 01:01:01.01 AM GMT-07:00
  evaluateDateTrunc(
      "second", 1730620861100, "America/Los_Angeles", 1730620861000);
  evaluateDateTrunc(
      "minute", 1730620861100, "America/Los_Angeles", 1730620860000);
  evaluateDateTrunc(
      "hour", 1730620861100, "America/Los_Angeles", 1730620800000);

  // The second 1AM
  //  11/3/2024 01:01:01.01 AM GMT-08:00
  evaluateDateTrunc(
      "second", 1730624461100, "America/Los_Angeles", 1730624461000);
  evaluateDateTrunc(
      "minute", 1730624461100, "America/Los_Angeles", 1730624460000);
  evaluateDateTrunc(
      "hour", 1730624461100, "America/Los_Angeles", 1730624400000);

  // Test cases that go back across a "fall back" daylight savings time
  // boundary. (GMT-07:00 -> GMT-08:00)
  //  11/3/2024 01:01:01.01 AM GMT-08:00
  evaluateDateTrunc("day", 1730624461100, "America/Los_Angeles", 1730617200000);
  evaluateDateTrunc(
      "month", 1730624461100, "America/Los_Angeles", 1730444400000);
  evaluateDateTrunc(
      "quarter", 1730624461100, "America/Los_Angeles", 1727766000000);
  // Technically this circles back again to the same daylight savings time zone,
  // but just to make sure we're covered (and it also test leap years).
  evaluateDateTrunc(
      "year", 1730624461100, "America/Los_Angeles", 1704096000000);

  // Test cases that go back across a "spring forward" daylight savings time
  // boundary. (GMT-08:00 -> GMT-07:00)
  //  3/10/2024 03:00:00 AM GMT-08:00
  evaluateDateTrunc("day", 1710064800000, "America/Los_Angeles", 1710057600000);
  evaluateDateTrunc(
      "month", 1710064800000, "America/Los_Angeles", 1709280000000);
  evaluateDateTrunc(
      "quarter", 1710064800000, "America/Los_Angeles", 1704096000000);
  // Technically this circles back again to the same daylight savings time zone,
  // but just to make sure we're covered (and it also test leap years).
  evaluateDateTrunc(
      "year", 1710064800000, "America/Los_Angeles", 1704096000000);

  // Test some cases that are close to hours that don't exist due to DST (it's
  // impossible to truncate to a time in the hour that doesn't exist, so we
  // don't test that case).
  //  3/10/2024 03:01:01.01 AM GMT-08:00
  evaluateDateTrunc(
      "second", 1710064861100, "America/Los_Angeles", 1710064861000);
  evaluateDateTrunc(
      "minute", 1710064861100, "America/Los_Angeles", 1710064860000);
  evaluateDateTrunc(
      "hour", 1710064861100, "America/Los_Angeles", 1710064800000);

  //  3/10/2024 01:59:59.999AM GMT-07:00
  evaluateDateTrunc(
      "second", 1710064799999, "America/Los_Angeles", 1710064799000);
  evaluateDateTrunc(
      "minute", 1710064799999, "America/Los_Angeles", 1710064740000);
  evaluateDateTrunc(
      "hour", 1710064799999, "America/Los_Angeles", 1710061200000);

  const auto evaluateDateTruncFromStrings = [&](const std::string& truncUnit,
                                                const std::string&
                                                    inputTimestamp,
                                                const std::string&
                                                    expectedTimestamp) {
    assertEqualVectors(
        evaluate<FlatVector<int64_t>>(
            "parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ')",
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{expectedTimestamp}})})),
        evaluate<FlatVector<int64_t>>(
            fmt::format(
                "date_trunc('{}', parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ'))",
                truncUnit),
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{inputTimestamp}})})));
  };

  evaluateDateTruncFromStrings(
      "minute", "1972-05-20+23:01:02+14:00", "1972-05-20+23:01:00+14:00");
  evaluateDateTruncFromStrings(
      "minute", "1968-05-20+23:01:02+05:30", "1968-05-20+23:01:00+05:30");
  evaluateDateTruncFromStrings(
      "hour", "1972-05-20+23:01:02+03:00", "1972-05-20+23:00:00+03:00");
  evaluateDateTruncFromStrings(
      "hour", "1968-05-20+23:01:02-09:30", "1968-05-20+23:00:00-09:30");
  evaluateDateTruncFromStrings(
      "day", "1972-05-20+23:01:02-03:00", "1972-05-20+00:00:00-03:00");
  evaluateDateTruncFromStrings(
      "day", "1968-05-20+23:01:02+05:30", "1968-05-20+00:00:00+05:30");
  evaluateDateTruncFromStrings(
      "month", "1972-05-20+23:01:02-03:00", "1972-05-01+00:00:00-03:00");
  evaluateDateTruncFromStrings(
      "month", "1968-05-20+23:01:02+05:30", "1968-05-01+00:00:00+05:30");
  evaluateDateTruncFromStrings(
      "quarter", "1972-05-20+23:01:02-03:00", "1972-04-01+00:00:00-03:00");
  evaluateDateTruncFromStrings(
      "quarter", "1968-05-20+23:01:02+05:30", "1968-04-01+00:00:00+05:30");
  evaluateDateTruncFromStrings(
      "year", "1972-05-20+23:01:02-03:00", "1972-01-01+00:00:00-03:00");
  evaluateDateTruncFromStrings(
      "year", "1968-05-20+23:01:02+05:30", "1968-01-01+00:00:00+05:30");
}

TEST_F(DateTimeFunctionsTest, dateAddDate) {
  const auto dateAdd = [&](const std::string& unit,
                           std::optional<int32_t> value,
                           std::optional<int32_t> date) {
    return evaluateOnce<int32_t>(
        fmt::format("date_add('{}', c0, c1)", unit),
        {INTEGER(), DATE()},
        value,
        date);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateAdd("day", 1, std::nullopt));
  EXPECT_EQ(std::nullopt, dateAdd("month", std::nullopt, 0));

  // Check invalid units
  EXPECT_THROW(dateAdd("millisecond", 1, 0), VeloxUserError);
  EXPECT_THROW(dateAdd("second", 1, 0), VeloxUserError);
  EXPECT_THROW(dateAdd("minute", 1, 0), VeloxUserError);
  EXPECT_THROW(dateAdd("hour", 1, 0), VeloxUserError);
  EXPECT_THROW(dateAdd("invalid_unit", 1, 0), VeloxUserError);

  // Simple tests
  EXPECT_EQ(
      parseDate("2019-03-01"), dateAdd("day", 1, parseDate("2019-02-28")));
  EXPECT_EQ(
      parseDate("2019-03-07"), dateAdd("week", 1, parseDate("2019-02-28")));
  EXPECT_EQ(
      parseDate("2020-03-28"), dateAdd("month", 13, parseDate("2019-02-28")));
  EXPECT_EQ(
      parseDate("2020-02-28"), dateAdd("quarter", 4, parseDate("2019-02-28")));
  EXPECT_EQ(
      parseDate("2020-02-28"), dateAdd("year", 1, parseDate("2019-02-28")));

  // Account for the last day of a year-month
  EXPECT_EQ(
      parseDate("2020-02-29"), dateAdd("day", 395, parseDate("2019-01-30")));
  EXPECT_EQ(
      parseDate("2020-02-29"), dateAdd("month", 13, parseDate("2019-01-30")));
  EXPECT_EQ(
      parseDate("2020-02-29"), dateAdd("quarter", 1, parseDate("2019-11-30")));
  EXPECT_EQ(
      parseDate("2030-02-28"), dateAdd("year", 10, parseDate("2020-02-29")));

  // Check for negative intervals
  EXPECT_EQ(
      parseDate("2019-02-28"), dateAdd("day", -366, parseDate("2020-02-29")));
  EXPECT_EQ(
      parseDate("2020-02-15"), dateAdd("week", -2, parseDate("2020-02-29")));
  EXPECT_EQ(
      parseDate("2019-02-28"), dateAdd("month", -12, parseDate("2020-02-29")));
  EXPECT_EQ(
      parseDate("2019-02-28"), dateAdd("quarter", -4, parseDate("2020-02-29")));
  EXPECT_EQ(
      parseDate("2018-02-28"), dateAdd("year", -2, parseDate("2020-02-29")));
}

TEST_F(DateTimeFunctionsTest, dateAddTimestamp) {
  const auto dateAdd = [&](const std::string& unit,
                           std::optional<int32_t> value,
                           std::optional<Timestamp> timestamp) {
    return evaluateOnce<Timestamp>(
        fmt::format("date_add('{}', c0, c1)", unit), value, timestamp);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateAdd("second", 1, std::nullopt));
  EXPECT_EQ(std::nullopt, dateAdd("month", std::nullopt, Timestamp(0, 0)));

  // Check invalid units
  auto ts = Timestamp(0, 0);
  VELOX_ASSERT_THROW(
      dateAdd("invalid_unit", 1, ts),
      "Unsupported datetime unit: invalid_unit");

  // Simple tests
  EXPECT_EQ(
      Timestamp(1551348061, 999'999) /*2019-02-28 10:01:01.000*/,
      dateAdd(
          "millisecond",
          60 * 1000 + 500,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551434400, 500'999'999) /*2019-03-01 10:00:00.500*/,
      dateAdd(
          "second",
          60 * 60 * 24,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551434400, 500'999'999) /*2019-03-01 10:00:00.500*/,
      dateAdd(
          "minute",
          60 * 24,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551434400, 500'999'999) /*2019-03-01 10:00:00.500*/,
      dateAdd(
          "hour",
          24,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551434400, 500'999'999) /*2019-03-01 10:00:00.500*/,
      dateAdd(
          "day",
          1,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      parseTimestamp("2019-03-07 10:00:00.500"),
      dateAdd("week", 1, parseTimestamp("2019-02-28 10:00:00.500")));
  EXPECT_EQ(
      Timestamp(1585389600, 500'999'999) /*2020-03-28 10:00:00.500*/,
      dateAdd(
          "month",
          12 + 1,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582884000, 500'999'999) /*2020-02-28 10:00:00.500*/,
      dateAdd(
          "quarter",
          4,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582884000, 500'999'999) /*2020-02-28 10:00:00.500*/,
      dateAdd(
          "year",
          1,
          Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/));

  // Test for daylight saving. Daylight saving in US starts at 2021-03-14
  // 02:00:00 PST.
  // When adjust_timestamp_to_timezone is off, no Daylight saving occurs
  EXPECT_EQ(
      Timestamp(
          1615770000, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500' UTC*/,
      dateAdd(
          "millisecond",
          1000 * 60 * 60 * 24,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500 UTC'*/));
  EXPECT_EQ(
      Timestamp(
          1615770000, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500 UTC'*/,
      dateAdd(
          "second",
          60 * 60 * 24,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500 UTC'*/));
  EXPECT_EQ(
      Timestamp(
          1615770000, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500' UTC*/,
      dateAdd(
          "minute",
          60 * 24,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));
  EXPECT_EQ(
      Timestamp(
          1615770000, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500' UTC*/,
      dateAdd(
          "hour",
          24,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));
  EXPECT_EQ(
      Timestamp(
          1615770000, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500' UTC*/,
      dateAdd(
          "day",
          1,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));
  EXPECT_EQ(
      Timestamp(
          1618362000, 500'999'999) /*TIMESTAMP '2021-04-14 01:00:00.500' UTC*/,
      dateAdd(
          "month",
          1,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));
  EXPECT_EQ(
      Timestamp(
          1623632400, 500'999'999) /*TIMESTAMP '2021-06-14 01:00:00.500' UTC*/,
      dateAdd(
          "quarter",
          1,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));
  EXPECT_EQ(
      Timestamp(
          1647219600, 500'999'999) /*TIMESTAMP '2022-03-14 01:00:00.500' UTC*/,
      dateAdd(
          "year",
          1,
          Timestamp(
              1615683600,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' UTC*/));

  // When adjust_timestamp_to_timezone is off, respect Daylight saving in the
  // session time zone
  setQueryTimeZone("America/Los_Angeles");

  EXPECT_EQ(
      Timestamp(1615798800, 0) /*TIMESTAMP '2021-03-15 02:00:00' PST*/,
      dateAdd(
          "millisecond",
          1000 * 60 * 60 * 24,
          Timestamp(1615712400, 0) /*TIMESTAMP '2021-03-14 01:00:00' PST*/));
  EXPECT_EQ(
      Timestamp(1615798800, 0) /*TIMESTAMP '2021-03-15 02:00:00' PST*/,
      dateAdd(
          "second",
          60 * 60 * 24,
          Timestamp(1615712400, 0) /*TIMESTAMP '2021-03-14 01:00:00' PST*/));
  EXPECT_EQ(
      Timestamp(1615798800, 0) /*TIMESTAMP '2021-03-15 02:00:00' PST*/,
      dateAdd(
          "minute",
          60 * 24,
          Timestamp(1615712400, 0) /*TIMESTAMP '2021-03-14 01:00:00' PST*/));
  EXPECT_EQ(
      Timestamp(1615798800, 0) /*TIMESTAMP '2021-03-15 02:00:00' PST*/,
      dateAdd(
          "hour",
          24,
          Timestamp(1615712400, 0) /*TIMESTAMP '2021-03-14 01:00:00' PST*/));
  EXPECT_EQ(
      Timestamp(
          1615795200, 500'999'999) /*TIMESTAMP '2021-03-15 01:00:00.500' PST*/,
      dateAdd(
          "day",
          1,
          Timestamp(
              1615712400,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' PST*/));
  EXPECT_EQ(
      Timestamp(
          1618387200, 500'999'999) /*TIMESTAMP '2021-04-14 01:00:00.500' PST*/,
      dateAdd(
          "month",
          1,
          Timestamp(
              1615712400,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' PST*/));
  EXPECT_EQ(
      Timestamp(
          1623657600, 500'999'999) /*TIMESTAMP '2021-06-14 01:00:00.500' PST*/,
      dateAdd(
          "quarter",
          1,
          Timestamp(
              1615712400,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' PST*/));
  EXPECT_EQ(
      Timestamp(
          1647244800, 500'999'999) /*TIMESTAMP '2022-03-14 01:00:00.500' PST*/,
      dateAdd(
          "year",
          1,
          Timestamp(
              1615712400,
              500'999'999) /*TIMESTAMP '2021-03-14 01:00:00.500' PST*/));

  // Test for coercing to the last day of a year-month
  EXPECT_EQ(
      Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/,
      dateAdd(
          "day",
          365 + 30,
          Timestamp(1548842400, 500'999'999) /*2019-01-30 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/,
      dateAdd(
          "month",
          12 + 1,
          Timestamp(1548842400, 500'999'999) /*2019-01-30 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/,
      dateAdd(
          "quarter",
          1,
          Timestamp(1575108000, 500'999'999) /*2019-11-30 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1898503200, 500'999'999) /*2030-02-28 10:00:00.500*/,
      dateAdd(
          "year",
          10,
          Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/));

  // Test for negative intervals
  EXPECT_EQ(
      Timestamp(1582934400, 999'999) /*2020-02-29 00:00:00.000*/,
      dateAdd(
          "millisecond",
          -60 * 60 * 24 * 1000 - 500,
          Timestamp(1583020800, 500'999'999) /*2020-03-01 00:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582934400, 500'999'999) /*2020-02-29 00:00:00.500*/,
      dateAdd(
          "second",
          -60 * 60 * 24,
          Timestamp(1583020800, 500'999'999) /*2020-03-01 00:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582934400, 500'999'999) /*2020-02-29 00:00:00.500*/,
      dateAdd(
          "minute",
          -60 * 24,
          Timestamp(1583020800, 500'999'999) /*2020-03-01 00:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1582934400, 500'999'999) /*2020-02-29 00:00:00.500*/,
      dateAdd(
          "hour",
          -24,
          Timestamp(1583020800, 500'999'999) /*2020-03-01 00:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/,
      dateAdd(
          "day",
          -366,
          Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/,
      dateAdd(
          "month",
          -12,
          Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1551348000, 500'999'999) /*2019-02-28 10:00:00.500*/,
      dateAdd(
          "quarter",
          -4,
          Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/));
  EXPECT_EQ(
      Timestamp(1519812000, 500'999'999) /*2019-02-28 10:00:00.500*/,
      dateAdd(
          "year",
          -2,
          Timestamp(1582970400, 500'999'999) /*2020-02-29 10:00:00.500*/));

  // Test cases where the result would end up in the nonexistent gap between
  // daylight savings time and standard time. 2023-03-12 02:30:00.000 does not
  // exist in America/Los_Angeles since that hour is skipped.
  EXPECT_EQ(
      Timestamp(1678617000, 0), /*2023-03-12 03:30:00*/
      dateAdd("day", 45, Timestamp(1674729000, 0) /*2023-01-26 02:30:00*/));
  EXPECT_EQ(
      Timestamp(1678617000, 0), /*2023-03-12 03:30:00*/
      dateAdd("day", -45, Timestamp(1682501400, 0) /*2023-04-26 02:30:00*/));
}

TEST_F(DateTimeFunctionsTest, dateAddTimestampWithTimeZone) {
  auto dateAdd =
      [&](std::optional<std::string> unit,
          std::optional<int32_t> value,
          std::optional<TimestampWithTimezone> timestampWithTimezone) {
        auto result = evaluateOnce<int64_t>(
            "date_add(c0, c1, c2)",
            {VARCHAR(), INTEGER(), TIMESTAMP_WITH_TIME_ZONE()},
            unit,
            value,
            TimestampWithTimezone::pack(timestampWithTimezone));
        return TimestampWithTimezone::unpack(result);
      };

  // 1970-01-01 00:00:00.000 UTC-8
  EXPECT_EQ(
      TimestampWithTimezone(432000000, "-08:00"),
      dateAdd("day", 5, TimestampWithTimezone(0, "-08:00")));

  // 2023-01-08 00:00:00.000 UTC-8
  EXPECT_EQ(
      TimestampWithTimezone(1068336000, "-08:00"),
      dateAdd("day", -7, TimestampWithTimezone(1673136000, "-08:00")));

  // 2023-01-08 00:00:00.000 UTC-8
  EXPECT_EQ(
      TimestampWithTimezone(1673135993, "-08:00"),
      dateAdd("millisecond", -7, TimestampWithTimezone(1673136000, "-08:00")));

  // 2023-01-08 00:00:00.000 UTC-8
  EXPECT_EQ(
      TimestampWithTimezone(1673136007, "-08:00"),
      dateAdd("millisecond", +7, TimestampWithTimezone(1673136000, "-08:00")));

  const auto evaluateDateAddFromStrings = [&](const std::string& unit,
                                              int32_t value,
                                              const std::string& inputTimestamp,
                                              const std::string&
                                                  expectedTimestamp) {
    assertEqualVectors(
        evaluate<FlatVector<int64_t>>(
            "parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ')",
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{expectedTimestamp}})})),
        evaluate<FlatVector<int64_t>>(
            fmt::format(
                "date_add('{}', {}, parse_datetime(c0, 'YYYY-MM-dd+HH:mm:ssZZ'))",
                unit,
                value),
            makeRowVector({makeNullableFlatVector<StringView>(
                {StringView{inputTimestamp}})})));
  };

  evaluateDateAddFromStrings(
      "second", 3, "1972-05-20+23:01:02+14:00", "1972-05-20+23:01:05+14:00");
  evaluateDateAddFromStrings(
      "minute", 5, "1972-05-20+23:01:02+14:00", "1972-05-20+23:06:02+14:00");
  evaluateDateAddFromStrings(
      "minute", 10, "1968-02-20+23:01:02+14:00", "1968-02-20+23:11:02+14:00");
  evaluateDateAddFromStrings(
      "hour", 5, "1972-05-20+23:01:02+14:00", "1972-05-21+04:01:02+14:00");
  evaluateDateAddFromStrings(
      "hour", 50, "1968-02-20+23:01:02+14:00", "1968-02-23+01:01:02+14:00");
  evaluateDateAddFromStrings(
      "day", 14, "1972-05-20+23:01:02+14:00", "1972-06-03+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "day", 140, "1968-02-20+23:01:02+14:00", "1968-07-09+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "month", 14, "1972-05-20+23:01:02+14:00", "1973-07-20+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "month", 10, "1968-02-20+23:01:02+14:00", "1968-12-20+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "quarter", 3, "1972-05-20+23:01:02+14:00", "1973-02-20+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "quarter", 30, "1968-02-20+23:01:02+14:00", "1975-08-20+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "year", 3, "1972-05-20+23:01:02+14:00", "1975-05-20+23:01:02+14:00");
  evaluateDateAddFromStrings(
      "year", 3, "1968-02-20+23:01:02+14:00", "1971-02-20+23:01:02+14:00");

  // Tests date_add() on daylight saving transition boundaries.
  //
  // Presto's semantic is to apply the delta to GMT, which means that at times
  // the observed delta may not be linear, in cases when it hits a daylight
  // savings boundary.
  auto dateAddAndCast = [&](std::optional<std::string> unit,
                            std::optional<int32_t> value,
                            std::optional<std::string> timestampString) {
    return evaluateOnce<std::string>(
        "cast(date_add(c0, c1, cast(c2 as timestamp with time zone)) as VARCHAR)",
        unit,
        value,
        timestampString);
  };

  EXPECT_EQ(
      "2024-03-10 03:50:00.000 America/Los_Angeles",
      dateAddAndCast("hour", 1, "2024-03-10 01:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-11-03 01:50:00.000 America/Los_Angeles",
      dateAddAndCast("hour", 1, "2024-11-03 01:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-11-03 00:50:00.000 America/Los_Angeles",
      dateAddAndCast("hour", -1, "2024-11-03 01:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-11-04 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("day", 1, "2024-11-03 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-11-10 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("week", 1, "2024-11-03 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-12-03 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "month", 1, "2024-11-03 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2025-02-03 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "quarter", 1, "2024-11-03 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2025-11-03 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("year", 1, "2024-11-03 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-11-03 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("day", -1, "2024-11-04 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-10-28 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "week", -1, "2024-11-04 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-10-04 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "month", -1, "2024-11-04 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-08-04 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "quarter", -1, "2024-11-04 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2023-11-04 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "year", -1, "2024-11-04 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-03-11 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("day", 1, "2024-03-10 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-03-17 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("week", 1, "2024-03-10 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-04-10 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "month", 1, "2024-03-10 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-06-10 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "quarter", 1, "2024-03-10 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2025-03-10 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("year", 1, "2024-03-10 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-03-10 00:00:00.000 America/Los_Angeles",
      dateAddAndCast("day", -1, "2024-03-11 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-03-04 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "week", -1, "2024-03-11 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2024-02-11 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "month", -1, "2024-03-11 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2023-12-11 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "quarter", -1, "2024-03-11 00:00:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2023-03-11 00:00:00.000 America/Los_Angeles",
      dateAddAndCast(
          "year", -1, "2024-03-11 00:00:00.000 America/Los_Angeles"));

  // Test cases where the result would end up in the nonexistent gap between
  // daylight savings time and standard time. 2023-03-12 02:30:00.000 does not
  // exist in America/Los_Angeles since that hour is skipped.
  EXPECT_EQ(
      "2023-03-12 03:30:00.000 America/Los_Angeles",
      dateAddAndCast("day", 45, "2023-01-26 02:30:00.000 America/Los_Angeles"));
  EXPECT_EQ(
      "2023-03-12 03:30:00.000 America/Los_Angeles",
      dateAddAndCast(
          "day", -45, "2023-04-26 02:30:00.000 America/Los_Angeles"));
}

TEST_F(DateTimeFunctionsTest, dateAddTime) {
  const auto dateAdd = [&](const std::string& unit,
                           std::optional<int32_t> value,
                           std::optional<int64_t> time) {
    return evaluateOnce<int64_t>(
        fmt::format("date_add('{}', c0, c1)", unit),
        {INTEGER(), TIME()},
        value,
        time);
  };

  // basic time additions
  // add milliseconds
  EXPECT_EQ(1000, dateAdd("millisecond", 1000, 0)); // 00:00:00.000 + 1s
  EXPECT_EQ(500, dateAdd("millisecond", -500, 1000)); // 00:00:01.000 - 500ms
  EXPECT_EQ(2000, dateAdd("millisecond", 1000, 1000)); // 00:00:01.000 + 1s

  // add seconds
  EXPECT_EQ(1000, dateAdd("second", 1, 0)); // 00:00:00 + 1s
  EXPECT_EQ(0, dateAdd("second", -1, 1000)); // 00:00:01 - 1s
  EXPECT_EQ(3661000, dateAdd("second", 3661, 0)); // 1 hour 1 minute 1 second

  // add minutes
  EXPECT_EQ(60000, dateAdd("minute", 1, 0)); // 00:00:00 + 1 minute
  EXPECT_EQ(0, dateAdd("minute", -1, 60000)); // 00:01:00 - 1 minute
  EXPECT_EQ(300000, dateAdd("minute", 5, 0)); // 00:00:00 + 5 minutes

  // add hours
  EXPECT_EQ(3600000, dateAdd("hour", 1, 0)); // 00:00:00 + 1 hour
  EXPECT_EQ(0, dateAdd("hour", -1, 3600000)); // 01:00:00 - 1 hour
  EXPECT_EQ(43200000, dateAdd("hour", 12, 0)); // 00:00:00 + 12 hours

  // test wraparound behavior (24-hour modulo)
  EXPECT_EQ(
      3600000,
      dateAdd("hour", 25, 0)); // 00:00:00 + 25 hours = 01:00:00 (wraps)
  EXPECT_EQ(
      82800000, dateAdd("hour", -1, 0)); // 00:00:00 - 1 hour = 23:00:00 (wraps)
  EXPECT_EQ(
      0, dateAdd("hour", 24, 0)); // 00:00:00 + 24 hours = 00:00:00 (wraps)
  EXPECT_EQ(
      0, dateAdd("hour", -24, 0)); // 00:00:00 - 24 hours = 00:00:00 (wraps)

  // test real-world scenario: 09:30:15.500 + various units
  int64_t morning = 9 * 3600000 + 30 * 60000 + 15 * 1000 + 500; // 34215500ms
  EXPECT_EQ(34215750, dateAdd("millisecond", 250, morning)); // +250ms
  EXPECT_EQ(34245500, dateAdd("second", 30, morning)); // +30s
  EXPECT_EQ(35415500, dateAdd("minute", 20, morning)); // +20 minutes
  EXPECT_EQ(48615500, dateAdd("hour", 4, morning)); // +4 hours

  // test boundary values for TIME type (0 to 86399999 ms in a day)
  EXPECT_EQ(0, dateAdd("millisecond", 0, 0)); // 00:00:00.000 + 0ms
  EXPECT_EQ(
      86399999, dateAdd("millisecond", 86399999, 0)); // 00:00:00.000 + max time
  EXPECT_EQ(
      0, dateAdd("millisecond", -86399999, 86399999)); // max time - max time

  // test wraparound with large values
  EXPECT_EQ(
      1000,
      dateAdd(
          "millisecond",
          86401000,
          0)); // 00:00:00 + (24h + 1s) wraps to 00:00:01
  EXPECT_EQ(
      86398000,
      dateAdd(
          "millisecond",
          -2000,
          0)); // 00:00:00 - 2s wraps to 23:59:58

  // negative additions
  EXPECT_EQ(0, dateAdd("second", -60, 60000)); // 00:01:00 - 60s
  EXPECT_EQ(0, dateAdd("minute", -60, 3600000)); // 01:00:00 - 60min
  EXPECT_EQ(75600000, dateAdd("hour", -2, 82800000)); // 23:00:00 - 2h = 21:00

  // test null handling
  EXPECT_EQ(std::nullopt, dateAdd("second", 1, std::nullopt));
  EXPECT_EQ(std::nullopt, dateAdd("second", std::nullopt, 1000));
  EXPECT_EQ(std::nullopt, dateAdd("second", std::nullopt, std::nullopt));

  // test invalid units (TIME only supports time-related units, not
  // date-related, and not microsecond which is not supported in Presto)
  VELOX_ASSERT_THROW(
      dateAdd("microsecond", 1, 0), "microsecond is not a valid TIME field");
  VELOX_ASSERT_THROW(dateAdd("day", 1, 0), "day is not a valid TIME field");
  VELOX_ASSERT_THROW(dateAdd("week", 1, 0), "week is not a valid TIME field");
  VELOX_ASSERT_THROW(dateAdd("month", 1, 0), "month is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateAdd("quarter", 1, 0), "quarter is not a valid TIME field");
  VELOX_ASSERT_THROW(dateAdd("year", 1, 0), "year is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateAdd("invalid", 1, 0), "invalid is not a valid TIME field");

  // edge cases: multiple full day additions
  EXPECT_EQ(
      7200000,
      dateAdd("hour", 50, 0)); // 00:00:00 + 50 hours = 02:00:00 (wraps twice)
  EXPECT_EQ(
      0,
      dateAdd("hour", 72, 0)); // 00:00:00 + 72 hours = 00:00:00 (wraps 3 times)

  // test midnight transitions
  EXPECT_EQ(
      1000, dateAdd("millisecond", 2000, 86399000)); // 23:59:59 + 2s wraps
  EXPECT_EQ(
      86399000, dateAdd("millisecond", -1000, 0)); // 00:00:00 - 1s = 23:59:59
}

TEST_F(DateTimeFunctionsTest, dateDiffDate) {
  const auto dateDiff = [&](const std::string& unit,
                            std::optional<int32_t> date1,
                            std::optional<int32_t> date2) {
    return evaluateOnce<int64_t>(
        fmt::format("date_diff('{}', c0, c1)", unit),
        {DATE(), DATE()},
        date1,
        date2);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateDiff("day", 1, std::nullopt));
  EXPECT_EQ(std::nullopt, dateDiff("month", std::nullopt, 0));

  // Check invalid units
  VELOX_ASSERT_THROW(
      dateDiff("millisecond", 1, 0), "millisecond is not a valid DATE field");
  VELOX_ASSERT_THROW(
      dateDiff("second", 1, 0), "second is not a valid DATE field");
  VELOX_ASSERT_THROW(
      dateDiff("minute", 1, 0), "minute is not a valid DATE field");
  VELOX_ASSERT_THROW(dateDiff("hour", 1, 0), "hour is not a valid DATE field");
  VELOX_ASSERT_THROW(
      dateDiff("invalid_unit", 1, 0),
      "Unsupported datetime unit: invalid_unit");

  // Simple tests
  EXPECT_EQ(
      1, dateDiff("day", parseDate("2019-02-28"), parseDate("2019-03-01")));
  EXPECT_EQ(
      0, dateDiff("week", parseDate("2019-02-28"), parseDate("2019-03-01")));
  EXPECT_EQ(
      2, dateDiff("week", parseDate("2019-02-28"), parseDate("2019-03-15")));
  EXPECT_EQ(
      13, dateDiff("month", parseDate("2019-02-28"), parseDate("2020-03-28")));
  EXPECT_EQ(
      4, dateDiff("quarter", parseDate("2019-02-28"), parseDate("2020-02-28")));
  EXPECT_EQ(
      1, dateDiff("year", parseDate("2019-02-28"), parseDate("2020-02-28")));

  // Verify that units are not case sensitive.
  EXPECT_EQ(
      1, dateDiff("DAY", parseDate("2019-02-28"), parseDate("2019-03-01")));
  EXPECT_EQ(
      1, dateDiff("dAY", parseDate("2019-02-28"), parseDate("2019-03-01")));
  EXPECT_EQ(
      1, dateDiff("Day", parseDate("2019-02-28"), parseDate("2019-03-01")));

  // Account for the last day of a year-month
  EXPECT_EQ(
      395, dateDiff("day", parseDate("2019-01-30"), parseDate("2020-02-29")));
  EXPECT_EQ(
      13, dateDiff("month", parseDate("2019-01-30"), parseDate("2020-02-29")));
  EXPECT_EQ(
      1, dateDiff("quarter", parseDate("2019-11-30"), parseDate("2020-02-29")));
  EXPECT_EQ(
      10, dateDiff("year", parseDate("2020-02-29"), parseDate("2030-02-28")));

  // Check for negative intervals
  EXPECT_EQ(
      -366, dateDiff("day", parseDate("2020-02-29"), parseDate("2019-02-28")));
  EXPECT_EQ(
      0, dateDiff("week", parseDate("2020-02-29"), parseDate("2020-02-25")));
  EXPECT_EQ(
      -64, dateDiff("week", parseDate("2020-02-29"), parseDate("2018-12-02")));
  EXPECT_EQ(
      -12, dateDiff("month", parseDate("2020-02-29"), parseDate("2019-02-28")));
  EXPECT_EQ(
      -4,
      dateDiff("quarter", parseDate("2020-02-29"), parseDate("2019-02-28")));
  EXPECT_EQ(
      -2, dateDiff("year", parseDate("2020-02-29"), parseDate("2018-02-28")));

  // Check Large date
  EXPECT_EQ(
      737790,
      dateDiff("day", parseDate("2020-02-29"), parseDate("4040-02-29")));
  EXPECT_EQ(
      24240,
      dateDiff("month", parseDate("2020-02-29"), parseDate("4040-02-29")));
  EXPECT_EQ(
      8080,
      dateDiff("quarter", parseDate("2020-02-29"), parseDate("4040-02-29")));
  EXPECT_EQ(
      2020, dateDiff("year", parseDate("2020-02-29"), parseDate("4040-02-29")));
}

TEST_F(DateTimeFunctionsTest, dateDiffTimestamp) {
  const auto dateDiff = [&](const std::string& unit,
                            std::optional<Timestamp> timestamp1,
                            std::optional<Timestamp> timestamp2) {
    return evaluateOnce<int64_t>(
        fmt::format("date_diff('{}', c0, c1)", unit), timestamp1, timestamp2);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateDiff("second", Timestamp(1, 0), std::nullopt));
  EXPECT_EQ(std::nullopt, dateDiff("month", std::nullopt, Timestamp(0, 0)));

  // Check invalid units
  VELOX_ASSERT_THROW(
      dateDiff("invalid_unit", Timestamp(1, 0), Timestamp(0, 0)),
      "Unsupported datetime unit: invalid_unit");

  // Check for integer overflow when result unit is month or larger.
  EXPECT_EQ(
      106751991167,
      dateDiff("day", Timestamp(0, 0), Timestamp(9223372036854775, 0)));
  EXPECT_EQ(
      15250284452,
      dateDiff("week", Timestamp(0, 0), Timestamp(9223372036854775, 0)));
  EXPECT_EQ(
      3507324295,
      dateDiff("month", Timestamp(0, 0), Timestamp(9223372036854775, 0)));
  EXPECT_EQ(
      1169108098,
      dateDiff("quarter", Timestamp(0, 0), Timestamp(9223372036854775, 0)));
  EXPECT_EQ(
      292277024,
      dateDiff("year", Timestamp(0, 0), Timestamp(9223372036854775, 0)));

  // Simple tests
  EXPECT_EQ(
      60 * 1000 + 500,
      dateDiff(
          "millisecond",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-02-28 10:01:01.000")));
  EXPECT_EQ(
      60 * 60 * 24,
      dateDiff(
          "second",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-01 10:00:00.500")));
  EXPECT_EQ(
      60 * 24,
      dateDiff(
          "minute",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-01 10:00:00.500")));
  EXPECT_EQ(
      24,
      dateDiff(
          "hour",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-01 10:00:00.500")));
  EXPECT_EQ(
      1,
      dateDiff(
          "day",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-01 10:00:00.500")));
  EXPECT_EQ(
      0,
      dateDiff(
          "week",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-01 10:00:00.500")));
  EXPECT_EQ(
      1,
      dateDiff(
          "week",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2019-03-10 10:00:00.500")));
  EXPECT_EQ(
      12 + 1,
      dateDiff(
          "month",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2020-03-28 10:00:00.500")));
  EXPECT_EQ(
      4,
      dateDiff(
          "quarter",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2020-02-28 10:00:00.500")));
  EXPECT_EQ(
      1,
      dateDiff(
          "year",
          parseTimestamp("2019-02-28 10:00:00.500"),
          parseTimestamp("2020-02-28 10:00:00.500")));

  // Test for daylight saving. Daylight saving in US starts at 2021-03-14
  // 02:00:00 PST.
  // When adjust_timestamp_to_timezone is off, Daylight saving occurs in UTC
  EXPECT_EQ(
      1000 * 60 * 60 * 24,
      dateDiff(
          "millisecond",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-03-15 01:00:00.000")));
  EXPECT_EQ(
      60 * 60 * 24,
      dateDiff(
          "second",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-03-15 01:00:00.000")));
  EXPECT_EQ(
      60 * 24,
      dateDiff(
          "minute",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-03-15 01:00:00.000")));
  EXPECT_EQ(
      24,
      dateDiff(
          "hour",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-03-15 01:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "day",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-03-15 01:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "month",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-04-14 01:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "quarter",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2021-06-14 01:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "year",
          parseTimestamp("2021-03-14 01:00:00.000"),
          parseTimestamp("2022-03-14 01:00:00.000")));

  // When adjust_timestamp_to_timezone is on, respect Daylight saving in the
  // session time zone
  setQueryTimeZone("America/Los_Angeles");

  EXPECT_EQ(
      1000 * 60 * 60 * 24,
      dateDiff(
          "millisecond",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-03-15 09:00:00.000")));
  EXPECT_EQ(
      60 * 60 * 24,
      dateDiff(
          "second",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-03-15 09:00:00.000")));
  EXPECT_EQ(
      60 * 24,
      dateDiff(
          "minute",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-03-15 09:00:00.000")));
  EXPECT_EQ(
      24,
      dateDiff(
          "hour",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-03-15 09:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "day",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-03-15 09:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "month",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-04-14 09:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "quarter",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2021-06-14 09:00:00.000")));
  EXPECT_EQ(
      1,
      dateDiff(
          "year",
          parseTimestamp("2021-03-14 09:00:00.000"),
          parseTimestamp("2022-03-14 09:00:00.000")));

  // Test for respecting the last day of a year-month
  EXPECT_EQ(
      365 + 30,
      dateDiff(
          "day",
          parseTimestamp("2019-01-30 10:00:00.500"),
          parseTimestamp("2020-02-29 10:00:00.500")));
  EXPECT_EQ(
      12 + 1,
      dateDiff(
          "month",
          parseTimestamp("2019-01-30 10:00:00.500"),
          parseTimestamp("2020-02-29 10:00:00.500")));
  EXPECT_EQ(
      1,
      dateDiff(
          "quarter",
          parseTimestamp("2019-11-30 10:00:00.500"),
          parseTimestamp("2020-02-29 10:00:00.500")));
  EXPECT_EQ(
      10,
      dateDiff(
          "year",
          parseTimestamp("2020-02-29 10:00:00.500"),
          parseTimestamp("2030-02-28 10:00:00.500")));

  // Test for negative difference
  EXPECT_EQ(
      -60 * 60 * 24 * 1000 - 500,
      dateDiff(
          "millisecond",
          parseTimestamp("2020-03-01 00:00:00.500"),
          parseTimestamp("2020-02-29 00:00:00.000")));
  EXPECT_EQ(
      -60 * 60 * 24,
      dateDiff(
          "second",
          parseTimestamp("2020-03-01 00:00:00.500"),
          parseTimestamp("2020-02-29 00:00:00.500")));
  EXPECT_EQ(
      -60 * 24,
      dateDiff(
          "minute",
          parseTimestamp("2020-03-01 00:00:00.500"),
          parseTimestamp("2020-02-29 00:00:00.500")));
  EXPECT_EQ(
      -24,
      dateDiff(
          "hour",
          parseTimestamp("2020-03-01 00:00:00.500"),
          parseTimestamp("2020-02-29 00:00:00.500")));
  EXPECT_EQ(
      -366,
      dateDiff(
          "day",
          parseTimestamp("2020-02-29 10:00:00.500"),
          parseTimestamp("2019-02-28 10:00:00.500")));
  EXPECT_EQ(
      -12,
      dateDiff(
          "month",
          parseTimestamp("2020-02-29 10:00:00.500"),
          parseTimestamp("2019-02-28 10:00:00.500")));
  EXPECT_EQ(
      -4,
      dateDiff(
          "quarter",
          parseTimestamp("2020-02-29 10:00:00.500"),
          parseTimestamp("2019-02-28 10:00:00.500")));
  EXPECT_EQ(
      -2,
      dateDiff(
          "year",
          parseTimestamp("2020-02-29 10:00:00.500"),
          parseTimestamp("2018-02-28 10:00:00.500")));
}

TEST_F(DateTimeFunctionsTest, dateDiffTimestampWithTimezone) {
  const auto dateDiff = [&](std::optional<std::string> unit,
                            std::optional<TimestampWithTimezone> input1,
                            std::optional<TimestampWithTimezone> input2) {
    return evaluateOnce<int64_t>(
        "date_diff(c0, c1, c2)",
        {VARCHAR(), TIMESTAMP_WITH_TIME_ZONE(), TIMESTAMP_WITH_TIME_ZONE()},
        unit,
        TimestampWithTimezone::pack(input1),
        TimestampWithTimezone::pack(input2));
  };

  // Null behavior.
  EXPECT_EQ(
      std::nullopt,
      dateDiff(
          std::nullopt,
          TimestampWithTimezone(0, "UTC"),
          TimestampWithTimezone(0, "UTC")));

  EXPECT_EQ(
      std::nullopt,
      dateDiff("asdf", std::nullopt, TimestampWithTimezone(0, "UTC")));

  // timestamp1: 1970-01-01 00:00:00.000 +00:00 (0)
  // timestamp2: 2020-08-25 16:30:10.123 -08:00 (1'598'373'010'123)
  EXPECT_EQ(
      1598373010123,
      dateDiff(
          "millisecond",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      1598373010,
      dateDiff(
          "second",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      26639550,
      dateDiff(
          "minute",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      443992,
      dateDiff(
          "hour",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      18499,
      dateDiff(
          "day",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      2642,
      dateDiff(
          "week",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      607,
      dateDiff(
          "month",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      202,
      dateDiff(
          "quarter",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));
  EXPECT_EQ(
      50,
      dateDiff(
          "year",
          TimestampWithTimezone(0, "+00:00"),
          TimestampWithTimezone(1'598'373'010'123, "America/Los_Angeles")));

  // Test if calculations are being performed in correct zone. Presto behavior
  // is to use the zone of the first parameter. Note that that this UTC interval
  // (a, b) crosses a daylight savings boundary in PST when PST loses one hour.
  // So whenever the calculation is performed in PST, the interval is
  // effectively smaller than 24h and returns zero.
  auto a = parseTimestamp("2024-11-02 17:00:00").toMillis();
  auto b = parseTimestamp("2024-11-03 17:30:00").toMillis();
  EXPECT_EQ(
      1,
      dateDiff(
          "day",
          TimestampWithTimezone(a, "UTC"),
          TimestampWithTimezone(b, "America/Los_Angeles")));
  EXPECT_EQ(
      1,
      dateDiff(
          "day",
          TimestampWithTimezone(a, "UTC"),
          TimestampWithTimezone(b, "UTC")));

  EXPECT_EQ(
      0,
      dateDiff(
          "day",
          TimestampWithTimezone(a, "America/Los_Angeles"),
          TimestampWithTimezone(b, "UTC")));
  EXPECT_EQ(
      0,
      dateDiff(
          "day",
          TimestampWithTimezone(a, "America/Los_Angeles"),
          TimestampWithTimezone(b, "America/Los_Angeles")));

  auto dateDiffAndCast = [&](std::optional<std::string> unit,
                             std::optional<std::string> timestampString1,
                             std::optional<std::string> timestampString2) {
    return evaluateOnce<int64_t>(
        "date_diff(c0, cast(c1 as timestamp with time zone), cast(c2 as timestamp with time zone))",
        unit,
        timestampString1,
        timestampString2);
  };

  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "hour",
          "2024-03-10 01:50:00 America/Los_Angeles",
          "2024-03-10 03:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      0,
      dateDiffAndCast(
          "hour",
          "2024-11-03 01:50:00 America/Los_Angeles",
          "2024-11-03 01:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "hour",
          "2024-11-03 01:50:00 America/Los_Angeles",
          "2024-11-03 00:50:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "day",
          "2024-11-03 00:00:00 America/Los_Angeles",
          "2024-11-04 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "week",
          "2024-11-03 00:00:00 America/Los_Angeles",
          "2024-11-10 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "month",
          "2024-11-03 00:00:00 America/Los_Angeles",
          "2024-12-03 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "quarter",
          "2024-11-03 00:00:00 America/Los_Angeles",
          "2025-02-03 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "year",
          "2024-11-03 00:00:00 America/Los_Angeles",
          "2025-11-03 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "day",
          "2024-11-04 00:00:00 America/Los_Angeles",
          "2024-11-03 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "week",
          "2024-11-04 00:00:00 America/Los_Angeles",
          "2024-10-28 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "month",
          "2024-11-04 00:00:00 America/Los_Angeles",
          "2024-10-04 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "quarter",
          "2024-11-04 00:00:00 America/Los_Angeles",
          "2024-08-04 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "year",
          "2024-11-04 00:00:00 America/Los_Angeles",
          "2023-11-04 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "day",
          "2024-03-10 00:00:00 America/Los_Angeles",
          "2024-03-11 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "week",
          "2024-03-10 00:00:00 America/Los_Angeles",
          "2024-03-17 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "month",
          "2024-03-10 00:00:00 America/Los_Angeles",
          "2024-04-10 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "quarter",
          "2024-03-10 00:00:00 America/Los_Angeles",
          "2024-06-10 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      1,
      dateDiffAndCast(
          "year",
          "2024-03-10 00:00:00 America/Los_Angeles",
          "2025-03-10 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "day",
          "2024-03-11 00:00:00 America/Los_Angeles",
          "2024-03-10 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "week",
          "2024-03-11 00:00:00 America/Los_Angeles",
          "2024-03-04 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "month",
          "2024-03-11 00:00:00 America/Los_Angeles",
          "2024-02-11 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "quarter",
          "2024-03-11 00:00:00 America/Los_Angeles",
          "2023-12-11 00:00:00 America/Los_Angeles"));
  EXPECT_EQ(
      -1,
      dateDiffAndCast(
          "year",
          "2024-03-11 00:00:00 America/Los_Angeles",
          "2023-03-11 00:00:00 America/Los_Angeles"));
}

TEST_F(DateTimeFunctionsTest, parseDatetimeRoundtrip) {
  const auto parseDatetimeRoundTrip =
      [&](const std::optional<std::string>& input,
          const std::optional<std::string>& format) {
        return evaluateOnce<std::string>(
            "cast(parse_datetime(c0, c1) as varchar)", input, format);
      };

  EXPECT_EQ(
      "2024-01-20 01:00:30.127 UTC",
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.12700", "yyyy-MM-dd HH:mm:ss.SSSSS"));
  EXPECT_EQ(
      "2024-01-20 01:00:30.459 UTC",
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.45900000", "yyyy-MM-dd HH:mm:ss.SSSSSSSS"));
  EXPECT_EQ(
      "2024-01-20 01:00:30.617 UTC",
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.6170", "yyyy-MM-dd HH:mm:ss.SSSSSSSS"));

  EXPECT_EQ(
      "2024-01-20 01:00:30.127 UTC",
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.127149", "yyyy-MM-dd HH:mm:ss.SSSSSS"));
  EXPECT_EQ(
      "2024-01-20 01:00:30.127 UTC",
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.127941", "yyyy-MM-dd HH:mm:ss.SSSSSS"));

  VELOX_ASSERT_THROW(
      parseDatetimeRoundTrip(
          "2024-01-20 01:00:30.6170", "yyyy-MM-dd HH:mm:ss.SSS"),
      "Invalid date format");
}

TEST_F(DateTimeFunctionsTest, parseDatetime) {
  const auto parseDatetime = [&](const std::optional<std::string>& input,
                                 const std::optional<std::string>& format) {
    auto result =
        evaluateOnce<int64_t>("parse_datetime(c0, c1)", input, format);
    return TimestampWithTimezone::unpack(result);
  };

  // Check null behavior.
  EXPECT_EQ(std::nullopt, parseDatetime("1970-01-01", std::nullopt));
  EXPECT_EQ(std::nullopt, parseDatetime(std::nullopt, "YYYY-MM-dd"));
  EXPECT_EQ(std::nullopt, parseDatetime(std::nullopt, std::nullopt));

  // Ensure it throws.
  VELOX_ASSERT_THROW(parseDatetime("", ""), "Invalid pattern specification");
  VELOX_ASSERT_THROW(
      parseDatetime("1234", "Y Y"), "Invalid date format: '1234'");

  // Simple tests. More exhaustive tests are provided as part of Joda's
  // implementation.
  EXPECT_EQ(
      TimestampWithTimezone(0, "UTC"),
      parseDatetime("1970-01-01", "YYYY-MM-dd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("1970-01-02", "YYYY-MM-dd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("19700102", "YYYYMMdd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("19700102", "YYYYMdd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("19700102", "YYYYMMd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("19700102", "YYYYMd"));
  EXPECT_EQ(
      TimestampWithTimezone(86400000, "UTC"),
      parseDatetime("19700102", "YYYYMd"));

  // 118860000 is the number of milliseconds since epoch at 1970-01-02
  // 09:01:00.000 UTC.
  EXPECT_EQ(
      TimestampWithTimezone(118860000, "+00:00"),
      parseDatetime("1970-01-02+09:01+00:00", "YYYY-MM-dd+HH:mmZZ"));
  EXPECT_EQ(
      TimestampWithTimezone(118860000, "-09:00"),
      parseDatetime("1970-01-02+00:01-09:00", "YYYY-MM-dd+HH:mmZZ"));
  EXPECT_EQ(
      TimestampWithTimezone(118860000, "-02:00"),
      parseDatetime("1970-01-02+07:01-02:00", "YYYY-MM-dd+HH:mmZZ"));
  EXPECT_EQ(
      TimestampWithTimezone(118860000, "+14:00"),
      parseDatetime("1970-01-02+23:01+14:00", "YYYY-MM-dd+HH:mmZZ"));
  EXPECT_EQ(
      TimestampWithTimezone(198060000, "America/Los_Angeles"),
      parseDatetime("1970-01-02+23:01 PST", "YYYY-MM-dd+HH:mm z"));
  EXPECT_EQ(
      TimestampWithTimezone(169260000, "+00:00"),
      parseDatetime("1970-01-02+23:01 GMT", "YYYY-MM-dd+HH:mm z"));

  setQueryTimeZone("Asia/Kolkata");

  // 66600000 is the number of millisecond since epoch at 1970-01-01
  // 18:30:00.000 UTC.
  EXPECT_EQ(
      TimestampWithTimezone(66600000, "Asia/Kolkata"),
      parseDatetime("1970-01-02+00:00", "YYYY-MM-dd+HH:mm"));
  EXPECT_EQ(
      TimestampWithTimezone(66600000, "-03:00"),
      parseDatetime("1970-01-01+15:30-03:00", "YYYY-MM-dd+HH:mmZZ"));

  // -66600000 is the number of millisecond since epoch at 1969-12-31
  // 05:30:00.000 UTC.
  EXPECT_EQ(
      TimestampWithTimezone(-66600000, "Asia/Kolkata"),
      parseDatetime("1969-12-31+11:00", "YYYY-MM-dd+HH:mm"));
  EXPECT_EQ(
      TimestampWithTimezone(-66600000, "+02:00"),
      parseDatetime("1969-12-31+07:30+02:00", "YYYY-MM-dd+HH:mmZZ"));

  // Joda also lets 'Z' to be UTC|UCT|GMT|GMT0.
  auto ts = TimestampWithTimezone(1708840800000, "GMT");
  EXPECT_EQ(
      ts, parseDatetime("2024-02-25+06:00:99 GMT", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  EXPECT_EQ(
      ts, parseDatetime("2024-02-25+06:00:99 GMT0", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  EXPECT_EQ(
      ts, parseDatetime("2024-02-25+06:00:99 UTC", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  EXPECT_EQ(
      ts, parseDatetime("2024-02-25+06:00:99 UTC", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  // Test a time zone with a prefix.
  EXPECT_EQ(
      TimestampWithTimezone(1708869600000, "America/Los_Angeles"),
      parseDatetime(
          "2024-02-25+06:00:99 America/Los_Angeles",
          "yyyy-MM-dd+HH:mm:99 ZZZ"));
  // Test a time zone with a prefix is greedy. Etc/GMT-1 and Etc/GMT-10 are both
  // valid time zone names.
  EXPECT_EQ(
      TimestampWithTimezone(1708804800000, "Etc/GMT-10"),
      parseDatetime(
          "2024-02-25+06:00:99 Etc/GMT-10", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  // Test a time zone without a prefix is greedy. NZ and NZ-CHAT are both
  // valid time zone names.
  EXPECT_EQ(
      TimestampWithTimezone(1708791300000, "NZ-CHAT"),
      parseDatetime("2024-02-25+06:00:99 NZ-CHAT", "yyyy-MM-dd+HH:mm:99 ZZZ"));
  // Test a time zone with a prefix can handle trailing data.
  EXPECT_EQ(
      TimestampWithTimezone(1708869600000, "America/Los_Angeles"),
      parseDatetime(
          "America/Los_Angeles2024-02-25+06:00:99", "ZZZyyyy-MM-dd+HH:mm:99"));
  // Test a time zone without a prefix can handle trailing data.
  EXPECT_EQ(
      TimestampWithTimezone(1708840800000, "GMT"),
      parseDatetime("GMT2024-02-25+06:00:99", "ZZZyyyy-MM-dd+HH:mm:99"));
  // Test parsing can fall back to checking for time zones without a prefix when
  // a '/' is present but not part of the time zone name.
  EXPECT_EQ(
      TimestampWithTimezone(1708840800000, "GMT"),
      parseDatetime("GMT/2024-02-25+06:00:99", "ZZZ/yyyy-MM-dd+HH:mm:99"));

  // Maximum timestamp.
  EXPECT_EQ(
      TimestampWithTimezone(2251799813685247, "UTC"),
      parseDatetime(
          "73326/09/11 20:14:45.247 UTC", "yyyy/MM/dd HH:mm:ss.SSS ZZZ"));

  // Minimum timestamp.
  EXPECT_EQ(
      TimestampWithTimezone(-2251799813685248, "UTC"),
      parseDatetime(
          "-69387/04/22 03:45:14.752 UCT", "yyyy/MM/dd HH:mm:ss.SSS ZZZ"));

  // Test an invalid time zone without a prefix. (zzz should be used to match
  // abbreviations)
  VELOX_ASSERT_THROW(
      parseDatetime("2024-02-25+06:00:99 PST", "yyyy-MM-dd+HH:mm:99 ZZZ"),
      "Invalid date format: '2024-02-25+06:00:99 PST'");
  // Test an invalid time zone with a prefix that doesn't appear at all.
  VELOX_ASSERT_THROW(
      parseDatetime("2024-02-25+06:00:99 ABC/XYZ", "yyyy-MM-dd+HH:mm:99 ZZZ"),
      "Invalid date format: '2024-02-25+06:00:99 ABC/XYZ'");
  // Test an invalid time zone with a prefix that does appear.
  VELOX_ASSERT_THROW(
      parseDatetime(
          "2024-02-25+06:00:99 America/XYZ", "yyyy-MM-dd+HH:mm:99 ZZZ"),
      "Invalid date format: '2024-02-25+06:00:99 America/XYZ'");

  // Test to ensure we do not support parsing time zone long names (to be
  // consistent with JODA).
  VELOX_ASSERT_THROW(
      parseDatetime(
          "2024-02-25+06:00:99 Pacific Standard Time",
          "yyyy-MM-dd+HH:mm:99 zzzz"),
      "Parsing time zone long names is not supported.");
  VELOX_ASSERT_THROW(
      parseDatetime(
          "2024-02-25+06:00:99 Pacific Standard Time",
          "yyyy-MM-dd+HH:mm:99 zzzzzzzzzz"),
      "Parsing time zone long names is not supported.");

  // Test overflow in either direction.
  VELOX_ASSERT_THROW(
      parseDatetime(
          "73326/09/11 20:14:45.248 UTC", "yyyy/MM/dd HH:mm:ss.SSS ZZZ"),
      "TimestampWithTimeZone overflow");
  VELOX_ASSERT_THROW(
      parseDatetime(
          "-69387/04/22 03:45:14.751 UTC", "yyyy/MM/dd HH:mm:ss.SSS ZZZ"),
      "TimestampWithTimeZone overflow");
}

TEST_F(DateTimeFunctionsTest, formatDateTime) {
  const auto formatDatetime = [&](std::optional<Timestamp> timestamp,
                                  std::optional<std::string> format) {
    return evaluateOnce<std::string>(
        "format_datetime(c0, c1)", timestamp, format);
  };

  // Era test cases - 'G'
  EXPECT_EQ("AD", formatDatetime(parseTimestamp("1970-01-01"), "G"));
  EXPECT_EQ("BC", formatDatetime(parseTimestamp("-100-01-01"), "G"));
  EXPECT_EQ("BC", formatDatetime(parseTimestamp("0-01-01"), "G"));
  EXPECT_EQ("AD", formatDatetime(parseTimestamp("01-01-01"), "G"));
  EXPECT_EQ("AD", formatDatetime(parseTimestamp("01-01-01"), "GGGGGGG"));

  // Century of era test cases - 'C'
  EXPECT_EQ("19", formatDatetime(parseTimestamp("1900-01-01"), "C"));
  EXPECT_EQ("19", formatDatetime(parseTimestamp("1955-01-01"), "C"));
  EXPECT_EQ("20", formatDatetime(parseTimestamp("2000-01-01"), "C"));
  EXPECT_EQ("20", formatDatetime(parseTimestamp("2020-01-01"), "C"));
  EXPECT_EQ("0", formatDatetime(parseTimestamp("0-01-01"), "C"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("-100-01-01"), "C"));
  EXPECT_EQ("19", formatDatetime(parseTimestamp("-1900-01-01"), "C"));
  EXPECT_EQ("000019", formatDatetime(parseTimestamp("1955-01-01"), "CCCCCC"));

  // Year of era test cases - 'Y'
  EXPECT_EQ("1970", formatDatetime(parseTimestamp("1970-01-01"), "Y"));
  EXPECT_EQ("2020", formatDatetime(parseTimestamp("2020-01-01"), "Y"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("0-01-01"), "Y"));
  EXPECT_EQ("101", formatDatetime(parseTimestamp("-100-01-01"), "Y"));
  EXPECT_EQ("70", formatDatetime(parseTimestamp("1970-01-01"), "YY"));
  EXPECT_EQ("70", formatDatetime(parseTimestamp("-1970-01-01"), "YY"));
  EXPECT_EQ("1948", formatDatetime(parseTimestamp("1948-01-01"), "YYY"));
  EXPECT_EQ("1234", formatDatetime(parseTimestamp("1234-01-01"), "YYYY"));
  EXPECT_EQ(
      "0000000001", formatDatetime(parseTimestamp("01-01-01"), "YYYYYYYYYY"));

  // Day of week number - 'e'
  for (int i = 0; i < 31; i++) {
    std::string date("2022-08-" + std::to_string(i + 1));
    EXPECT_EQ(
        std::to_string(i % 7 + 1), formatDatetime(parseTimestamp(date), "e"));
  }
  EXPECT_EQ("000001", formatDatetime(parseTimestamp("2022-08-01"), "eeeeee"));

  // Day of week text - 'E'
  for (int i = 0; i < 31; i++) {
    std::string date("2022-08-" + std::to_string(i + 1));
    EXPECT_EQ(daysShort[i % 7], formatDatetime(parseTimestamp(date), "E"));
    EXPECT_EQ(daysShort[i % 7], formatDatetime(parseTimestamp(date), "EE"));
    EXPECT_EQ(daysShort[i % 7], formatDatetime(parseTimestamp(date), "EEE"));
    EXPECT_EQ(daysLong[i % 7], formatDatetime(parseTimestamp(date), "EEEE"));
    EXPECT_EQ(
        daysLong[i % 7], formatDatetime(parseTimestamp(date), "EEEEEEEE"));
  }

  // Year test cases - 'y'
  EXPECT_EQ("2022", formatDatetime(parseTimestamp("2022-06-20"), "y"));
  EXPECT_EQ("22", formatDatetime(parseTimestamp("2022-06-20"), "yy"));
  EXPECT_EQ("2022", formatDatetime(parseTimestamp("2022-06-20"), "yyy"));
  EXPECT_EQ("2022", formatDatetime(parseTimestamp("2022-06-20"), "yyyy"));

  EXPECT_EQ("10", formatDatetime(parseTimestamp("10-06-20"), "y"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("10-06-20"), "yy"));
  EXPECT_EQ("010", formatDatetime(parseTimestamp("10-06-20"), "yyy"));
  EXPECT_EQ("0010", formatDatetime(parseTimestamp("10-06-20"), "yyyy"));

  EXPECT_EQ("-16", formatDatetime(parseTimestamp("-16-06-20"), "y"));
  EXPECT_EQ("16", formatDatetime(parseTimestamp("-16-06-20"), "yy"));
  EXPECT_EQ("-016", formatDatetime(parseTimestamp("-16-06-20"), "yyy"));
  EXPECT_EQ("-0016", formatDatetime(parseTimestamp("-16-06-20"), "yyyy"));

  EXPECT_EQ("00", formatDatetime(parseTimestamp("-1600-06-20"), "yy"));
  EXPECT_EQ("01", formatDatetime(parseTimestamp("-1601-06-20"), "yy"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("-1610-06-20"), "yy"));

  // Day of year test cases - 'D'
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01"), "D"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("2022-01-10"), "D"));
  EXPECT_EQ("100", formatDatetime(parseTimestamp("2022-04-10"), "D"));
  EXPECT_EQ("365", formatDatetime(parseTimestamp("2022-12-31"), "D"));
  EXPECT_EQ("00100", formatDatetime(parseTimestamp("2022-04-10"), "DDDDD"));

  // Leap year case
  EXPECT_EQ("60", formatDatetime(parseTimestamp("2020-02-29"), "D"));
  EXPECT_EQ("366", formatDatetime(parseTimestamp("2020-12-31"), "D"));

  // Month of year test cases - 'M'
  for (int i = 0; i < 12; i++) {
    auto month = i + 1;
    std::string date("2022-" + std::to_string(month) + "-01");
    EXPECT_EQ(std::to_string(month), formatDatetime(parseTimestamp(date), "M"));
    EXPECT_EQ(padNumber(month), formatDatetime(parseTimestamp(date), "MM"));
    EXPECT_EQ(monthsShort[i], formatDatetime(parseTimestamp(date), "MMM"));
    EXPECT_EQ(monthsLong[i], formatDatetime(parseTimestamp(date), "MMMM"));
    EXPECT_EQ(monthsLong[i], formatDatetime(parseTimestamp(date), "MMMMMMMM"));
  }

  // Day of month test cases - 'd'
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01"), "d"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("2022-01-10"), "d"));
  EXPECT_EQ("28", formatDatetime(parseTimestamp("2022-01-28"), "d"));
  EXPECT_EQ("31", formatDatetime(parseTimestamp("2022-01-31"), "d"));
  EXPECT_EQ(
      "00000031", formatDatetime(parseTimestamp("2022-01-31"), "dddddddd"));

  // Leap year case
  EXPECT_EQ("29", formatDatetime(parseTimestamp("2020-02-29"), "d"));

  // Halfday of day test cases - 'a'
  EXPECT_EQ("AM", formatDatetime(parseTimestamp("2022-01-01 00:00:00"), "a"));
  EXPECT_EQ("AM", formatDatetime(parseTimestamp("2022-01-01 11:59:59"), "a"));
  EXPECT_EQ("PM", formatDatetime(parseTimestamp("2022-01-01 12:00:00"), "a"));
  EXPECT_EQ("PM", formatDatetime(parseTimestamp("2022-01-01 23:59:59"), "a"));
  EXPECT_EQ(
      "AM", formatDatetime(parseTimestamp("2022-01-01 00:00:00"), "aaaaaaaa"));
  EXPECT_EQ(
      "PM", formatDatetime(parseTimestamp("2022-01-01 12:00:00"), "aaaaaaaa"));

  // Hour of halfday test cases - 'K'
  for (int i = 0; i < 24; i++) {
    std::string buildString = "2022-01-01 " + padNumber(i) + ":00:00";
    StringView date(buildString);
    EXPECT_EQ(
        std::to_string(i % 12), formatDatetime(parseTimestamp(date), "K"));
  }
  EXPECT_EQ(
      "00000011",
      formatDatetime(parseTimestamp("2022-01-01 11:00:00"), "KKKKKKKK"));

  // Clockhour of halfday test cases - 'h'
  for (int i = 0; i < 24; i++) {
    std::string buildString = "2022-01-01 " + padNumber(i) + ":00:00";
    StringView date(buildString);
    EXPECT_EQ(
        std::to_string((i + 11) % 12 + 1),
        formatDatetime(parseTimestamp(date), "h"));
  }
  EXPECT_EQ(
      "00000011",
      formatDatetime(parseTimestamp("2022-01-01 11:00:00"), "hhhhhhhh"));

  // Hour of day test cases - 'H'
  for (int i = 0; i < 24; i++) {
    std::string buildString = "2022-01-01 " + padNumber(i) + ":00:00";
    StringView date(buildString);
    EXPECT_EQ(std::to_string(i), formatDatetime(parseTimestamp(date), "H"));
  }
  EXPECT_EQ(
      "00000011",
      formatDatetime(parseTimestamp("2022-01-01 11:00:00"), "HHHHHHHH"));

  // Clockhour of day test cases - 'k'
  for (int i = 0; i < 24; i++) {
    std::string buildString = "2022-01-01 " + padNumber(i) + ":00:00";
    StringView date(buildString);
    EXPECT_EQ(
        std::to_string((i + 23) % 24 + 1),
        formatDatetime(parseTimestamp(date), "k"));
  }
  EXPECT_EQ(
      "00000011",
      formatDatetime(parseTimestamp("2022-01-01 11:00:00"), "kkkkkkkk"));

  // Minute of hour test cases - 'm'
  EXPECT_EQ("0", formatDatetime(parseTimestamp("2022-01-01 00:00:00"), "m"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01 01:01:00"), "m"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("2022-01-01 02:10:00"), "m"));
  EXPECT_EQ("30", formatDatetime(parseTimestamp("2022-01-01 03:30:00"), "m"));
  EXPECT_EQ("59", formatDatetime(parseTimestamp("2022-01-01 04:59:00"), "m"));
  EXPECT_EQ(
      "00000042",
      formatDatetime(parseTimestamp("2022-01-01 00:42:42"), "mmmmmmmm"));

  // Week of the year test cases - 'w'
  EXPECT_EQ("52", formatDatetime(parseTimestamp("2022-01-01 04:59:00"), "w"));
  EXPECT_EQ("52", formatDatetime(parseTimestamp("2022-01-02"), "w"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-03"), "w"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2024-01-01"), "w"));
  EXPECT_EQ("22", formatDatetime(parseTimestamp("1970-05-30"), "w"));

  // Second of minute test cases - 's'
  EXPECT_EQ("0", formatDatetime(parseTimestamp("2022-01-01 00:00:00"), "s"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01 01:01:01"), "s"));
  EXPECT_EQ("10", formatDatetime(parseTimestamp("2022-01-01 02:10:10"), "s"));
  EXPECT_EQ("30", formatDatetime(parseTimestamp("2022-01-01 03:30:30"), "s"));
  EXPECT_EQ("59", formatDatetime(parseTimestamp("2022-01-01 04:59:59"), "s"));
  EXPECT_EQ(
      "00000042",
      formatDatetime(parseTimestamp("2022-01-01 00:42:42"), "ssssssss"));

  // Fraction of second test cases - 'S'
  EXPECT_EQ("0", formatDatetime(parseTimestamp("2022-01-01 00:00:00.0"), "S"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01 00:00:00.1"), "S"));
  EXPECT_EQ("1", formatDatetime(parseTimestamp("2022-01-01 01:01:01.11"), "S"));
  EXPECT_EQ(
      "11", formatDatetime(parseTimestamp("2022-01-01 02:10:10.11"), "SS"));
  EXPECT_EQ(
      "9", formatDatetime(parseTimestamp("2022-01-01 03:30:30.999"), "S"));
  EXPECT_EQ(
      "99", formatDatetime(parseTimestamp("2022-01-01 03:30:30.999"), "SS"));
  EXPECT_EQ(
      "999", formatDatetime(parseTimestamp("2022-01-01 03:30:30.999"), "SSS"));
  EXPECT_EQ(
      "12300000",
      formatDatetime(parseTimestamp("2022-01-01 03:30:30.123"), "SSSSSSSS"));
  EXPECT_EQ(
      "0990",
      formatDatetime(parseTimestamp("2022-01-01 03:30:30.099"), "SSSS"));
  EXPECT_EQ(
      "0010",
      formatDatetime(parseTimestamp("2022-01-01 03:30:30.001"), "SSSS"));

  // Test the max and min years are formatted correctly.
  EXPECT_EQ(
      "2922789",
      formatDatetime(parseTimestamp("292278993-12-31 23:59:59.999"), "C"));
  EXPECT_EQ(
      "292278993",
      formatDatetime(parseTimestamp("292278993-12-31 23:59:59.999"), "YYYY"));
  EXPECT_EQ(
      "292278993",
      formatDatetime(parseTimestamp("292278993-12-31 23:59:59.999"), "xxxx"));
  EXPECT_EQ(
      "292278993",
      formatDatetime(parseTimestamp("292278993-12-31 23:59:59.999"), "yyyy"));
  EXPECT_EQ(
      "2922750",
      formatDatetime(parseTimestamp("-292275054-01-01 00:00:00.000"), "C"));
  EXPECT_EQ(
      "292275055",
      formatDatetime(parseTimestamp("-292275054-01-01 00:00:00.000"), "YYYY"));
  EXPECT_EQ(
      "-292275054",
      formatDatetime(parseTimestamp("-292275054-01-01 00:00:00.000"), "xxxx"));
  EXPECT_EQ(
      "-292275054",
      formatDatetime(parseTimestamp("-292275054-01-01 00:00:00.000"), "yyyy"));

  // Time zone test cases - 'Z'
  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ("-08:00", formatDatetime(parseTimestamp("1970-01-01"), "ZZ"));
  EXPECT_EQ("-0800", formatDatetime(parseTimestamp("1970-01-01"), "Z"));

  setQueryTimeZone("Asia/Kolkata");
  EXPECT_EQ(
      "Asia/Kolkata",
      formatDatetime(
          parseTimestamp("1970-01-01"), "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"));
  EXPECT_EQ(
      "Asia/Kolkata", formatDatetime(parseTimestamp("1970-01-01"), "ZZZZ"));
  EXPECT_EQ(
      "Asia/Kolkata", formatDatetime(parseTimestamp("1970-01-01"), "ZZZ"));
  EXPECT_EQ("+05:30", formatDatetime(parseTimestamp("1970-01-01"), "ZZ"));
  EXPECT_EQ("+0530", formatDatetime(parseTimestamp("1970-01-01"), "Z"));

  // Time zone test cases - 'z'
  // Timezone that has an abbreviation
  EXPECT_EQ("IST", formatDatetime(parseTimestamp("1970-01-01"), "zzz"));
  EXPECT_EQ("IST", formatDatetime(parseTimestamp("1970-01-01"), "zz"));
  EXPECT_EQ("IST", formatDatetime(parseTimestamp("1970-01-01"), "z"));
  EXPECT_EQ(
      "India Standard Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzz"));
  EXPECT_EQ(
      "India Standard Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzzzzzzzzzzzzzzzzzzzz"));

  // Timezone that has no abbreviations so uses GMT offset
  setQueryTimeZone("Asia/Atyrau");
  EXPECT_EQ("GMT+05:00", formatDatetime(parseTimestamp("1970-01-01"), "zzz"));
  EXPECT_EQ("GMT+05:00", formatDatetime(parseTimestamp("1970-01-01"), "zz"));
  EXPECT_EQ("GMT+05:00", formatDatetime(parseTimestamp("1970-01-01"), "z"));
  EXPECT_EQ(
      "West Kazakhstan Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzz"));
  EXPECT_EQ(
      "West Kazakhstan Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzzzzzzzzzzzzzzzzzzzz"));

  // Test daylight savings.
  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ("PST", formatDatetime(parseTimestamp("1970-01-01"), "z"));
  EXPECT_EQ("PDT", formatDatetime(parseTimestamp("1970-10-01"), "z"));
  EXPECT_EQ("PST", formatDatetime(parseTimestamp("2024-03-10 01:00"), "z"));
  EXPECT_EQ("PDT", formatDatetime(parseTimestamp("2024-03-10 03:00"), "z"));
  EXPECT_EQ("PDT", formatDatetime(parseTimestamp("2024-11-03 01:00"), "z"));
  EXPECT_EQ("PST", formatDatetime(parseTimestamp("2024-11-03 02:00"), "z"));
  EXPECT_EQ(
      "Pacific Standard Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzz"));
  EXPECT_EQ(
      "Pacific Daylight Time",
      formatDatetime(parseTimestamp("1970-10-01"), "zzzz"));
  EXPECT_EQ(
      "Pacific Standard Time",
      formatDatetime(parseTimestamp("2024-03-10 01:00"), "zzzz"));
  EXPECT_EQ(
      "Pacific Daylight Time",
      formatDatetime(parseTimestamp("2024-03-10 03:00"), "zzzz"));
  EXPECT_EQ(
      "Pacific Daylight Time",
      formatDatetime(parseTimestamp("2024-11-03 01:00"), "zzzz"));
  EXPECT_EQ(
      "Pacific Standard Time",
      formatDatetime(parseTimestamp("2024-11-03 02:00"), "zzzz"));

  // Test ambiguous time.
  EXPECT_EQ(
      "PDT", formatDatetime(parseTimestamp("2024-11-03 01:30:00"), "zzz"));
  EXPECT_EQ(
      "Pacific Daylight Time",
      formatDatetime(parseTimestamp("2024-11-03 01:30:00"), "zzzz"));

  // Test a long abbreviation.
  setQueryTimeZone("Asia/Colombo");
  EXPECT_EQ("IST", formatDatetime(parseTimestamp("1970-10-01"), "z"));
  EXPECT_EQ(
      "India Standard Time",
      formatDatetime(parseTimestamp("1970-10-01"), "zzzz"));

  // Test a long long name.
  setQueryTimeZone("Australia/Eucla");
  EXPECT_EQ("ACWST", formatDatetime(parseTimestamp("1970-10-01"), "z"));
  EXPECT_EQ(
      "Australian Central Western Standard Time",
      formatDatetime(parseTimestamp("1970-10-01"), "zzzz"));

  // Test a time zone that doesn't follow the standard abbrevation in the IANA
  // Time Zone Database, i.e. it relies on our map in TimeZoneNames.cpp.
  setQueryTimeZone("Asia/Dubai");
  // According to the IANA time zone database the abbreviation should be +04.
  EXPECT_EQ("GST", formatDatetime(parseTimestamp("1970-10-01"), "z"));
  EXPECT_EQ(
      "Gulf Standard Time",
      formatDatetime(parseTimestamp("1970-10-01"), "zzzz"));

  // Test UTC specifically (because it's so common).
  setQueryTimeZone("UTC");
  EXPECT_EQ("UTC", formatDatetime(parseTimestamp("1970-10-01"), "z"));
  EXPECT_EQ(
      "Coordinated Universal Time",
      formatDatetime(parseTimestamp("1970-10-01"), "zzzz"));

  // Test a time zone name that is linked to another (that gets replaced when
  // converted to a string).
  setQueryTimeZone("US/Pacific");
  EXPECT_EQ("PST", formatDatetime(parseTimestamp("1970-01-01"), "zzz"));
  EXPECT_EQ(
      "Pacific Standard Time",
      formatDatetime(parseTimestamp("1970-01-01"), "zzzz"));
  EXPECT_EQ(
      "America/Los_Angeles",
      formatDatetime(parseTimestamp("1970-01-01"), "ZZZ"));

  // Test the Etc/... time zones.
  auto testFormatTimeZoneID =
      [&](const std::string& inputTimeZoneID,
          const std::string& expectedFormattedTimeZoneID) {
        setQueryTimeZone(inputTimeZoneID);
        EXPECT_EQ(
            expectedFormattedTimeZoneID,
            formatDatetime(parseTimestamp("1970-01-01"), "ZZZ"));
      };
  testFormatTimeZoneID("Etc/GMT", "UTC");
  testFormatTimeZoneID("Etc/GMT+0", "UTC");
  testFormatTimeZoneID("Etc/GMT+1", "-01:00");
  testFormatTimeZoneID("Etc/GMT+10", "-10:00");
  testFormatTimeZoneID("Etc/GMT+12", "-12:00");
  testFormatTimeZoneID("Etc/GMT-0", "UTC");
  testFormatTimeZoneID("Etc/GMT-2", "+02:00");
  testFormatTimeZoneID("Etc/GMT-11", "+11:00");
  testFormatTimeZoneID("Etc/GMT-14", "+14:00");
  testFormatTimeZoneID("Etc/GMT0", "UTC");
  testFormatTimeZoneID("Etc/Greenwich", "UTC");
  testFormatTimeZoneID("Etc/UCT", "UTC");
  testFormatTimeZoneID("Etc/Universal", "UTC");
  testFormatTimeZoneID("Etc/UTC", "UTC");
  testFormatTimeZoneID("Etc/Zulu", "UTC");
  // These do not explicitly start with "Etc/" but they link to time zone IDs
  // that do.
  testFormatTimeZoneID("GMT0", "UTC");
  testFormatTimeZoneID("Greenwich", "UTC");
  testFormatTimeZoneID("UCT", "UTC");
  testFormatTimeZoneID("UTC", "UTC");
  testFormatTimeZoneID("Zulu", "UTC");

  setQueryTimeZone("Asia/Kolkata");
  // Literal test cases.
  EXPECT_EQ("hello", formatDatetime(parseTimestamp("1970-01-01"), "'hello'"));
  EXPECT_EQ("'", formatDatetime(parseTimestamp("1970-01-01"), "''"));
  EXPECT_EQ(
      "1970 ' 1970", formatDatetime(parseTimestamp("1970-01-01"), "y '' y"));
  EXPECT_EQ(
      "he'llo", formatDatetime(parseTimestamp("1970-01-01"), "'he''llo'"));
  EXPECT_EQ(
      "'he'llo'",
      formatDatetime(parseTimestamp("1970-01-01"), "'''he''llo'''"));
  EXPECT_EQ(
      "1234567890", formatDatetime(parseTimestamp("1970-01-01"), "1234567890"));
  EXPECT_EQ(
      "\\\"!@#$%^&*()-+[]{}||`~<>.,?/;:1234567890",
      formatDatetime(
          parseTimestamp("1970-01-01"),
          "\\\"!@#$%^&*()-+[]{}||`~<>.,?/;:1234567890"));

  // Multi-specifier and literal formats.
  EXPECT_EQ(
      "AD 19 1970 4 Thu 1970 1 1 1 AM 8 8 8 8 3 11 5 Asia/Kolkata",
      formatDatetime(
          parseTimestamp("1970-01-01 02:33:11.5"),
          "G C Y e E y D M d a K h H k m s S ZZZ"));
  EXPECT_EQ(
      "AD 19 1970 4 asdfghjklzxcvbnmqwertyuiop Thu ' 1970 1 1 1 AM 8 8 8 8 3 11 5 1234567890\\\"!@#$%^&*()-+`~{}[];:,./ Asia/Kolkata",
      formatDatetime(
          parseTimestamp("1970-01-01 02:33:11.5"),
          "G C Y e 'asdfghjklzxcvbnmqwertyuiop' E '' y D M d a K h H k m s S 1234567890\\\"!@#$%^&*()-+`~{}[];:,./ ZZZ"));

  disableAdjustTimestampToTimezone();
  EXPECT_EQ(
      "1970-01-01 00:00:00",
      formatDatetime(
          parseTimestamp("1970-01-01 00:00:00"), "YYYY-MM-dd HH:mm:ss"));

  // User format errors or unsupported errors.
  EXPECT_THROW(
      formatDatetime(parseTimestamp("1970-01-01"), "q"), VeloxUserError);
  EXPECT_THROW(
      formatDatetime(parseTimestamp("1970-01-01"), "'abcd"), VeloxUserError);

  // Time zone name patterns aren't supported when there isn't a time zone
  // available.
  EXPECT_THROW(
      formatDatetime(parseTimestamp("1970-01-01"), "z"), VeloxUserError);
  EXPECT_THROW(
      formatDatetime(parseTimestamp("1970-01-01"), "zz"), VeloxUserError);
  EXPECT_THROW(
      formatDatetime(parseTimestamp("1970-01-01"), "zzz"), VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, formatDateTimeTimezone) {
  const auto formatDatetimeWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone,
          std::optional<std::string> format) {
        return evaluateOnce<std::string>(
            "format_datetime(c0, c1)",
            {TIMESTAMP_WITH_TIME_ZONE(), VARCHAR()},
            TimestampWithTimezone::pack(timestampWithTimezone),
            format);
      };

  // UTC explicitly set.
  EXPECT_EQ(
      "1970-01-01 00:00:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "UTC"), "YYYY-MM-dd HH:mm:ss"));

  // Check that string is adjusted to the timezone set.
  EXPECT_EQ(
      "1970-01-01 05:30:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "Asia/Kolkata"), "YYYY-MM-dd HH:mm:ss"));

  EXPECT_EQ(
      "1969-12-31 16:00:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "America/Los_Angeles"),
          "YYYY-MM-dd HH:mm:ss"));

  // Make sure format_datetime() works with timezone offsets.
  EXPECT_EQ(
      "1969-12-31 16:00:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "-08:00"), "YYYY-MM-dd HH:mm:ss"));
  EXPECT_EQ(
      "1969-12-31 23:45:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "-00:15"), "YYYY-MM-dd HH:mm:ss"));
  EXPECT_EQ(
      "1970-01-01 00:07:00",
      formatDatetimeWithTimezone(
          TimestampWithTimezone(0, "+00:07"), "YYYY-MM-dd HH:mm:ss"));
}

TEST_F(DateTimeFunctionsTest, dateFormat) {
  const auto dateFormat = [&](std::optional<Timestamp> timestamp,
                              std::optional<std::string> format) {
    return evaluateOnce<std::string>("date_format(c0, c1)", timestamp, format);
  };

  // Check null behaviors.
  EXPECT_EQ(std::nullopt, dateFormat(std::nullopt, "%Y"));
  EXPECT_EQ(std::nullopt, dateFormat(Timestamp(0, 0), std::nullopt));

  // Normal cases.
  EXPECT_EQ("1970-01-01", dateFormat(parseTimestamp("1970-01-01"), "%Y-%m-%d"));
  EXPECT_EQ(
      "2000-02-29 12:00:00 AM",
      dateFormat(parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %r"));
  EXPECT_EQ(
      "2000-02-29 00:00:00.987000",
      dateFormat(
          parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));
  EXPECT_EQ(
      "-2000-02-29 00:00:00.987000",
      dateFormat(
          parseTimestamp("-2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));

  // Varying digit year cases.
  EXPECT_EQ("06", dateFormat(parseTimestamp("-6-06-20"), "%y"));
  EXPECT_EQ("-0006", dateFormat(parseTimestamp("-6-06-20"), "%Y"));
  EXPECT_EQ("16", dateFormat(parseTimestamp("-16-06-20"), "%y"));
  EXPECT_EQ("-0016", dateFormat(parseTimestamp("-16-06-20"), "%Y"));
  EXPECT_EQ("66", dateFormat(parseTimestamp("-166-06-20"), "%y"));
  EXPECT_EQ("-0166", dateFormat(parseTimestamp("-166-06-20"), "%Y"));
  EXPECT_EQ("66", dateFormat(parseTimestamp("-1666-06-20"), "%y"));
  EXPECT_EQ("00", dateFormat(parseTimestamp("-1900-06-20"), "%y"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("-1901-06-20"), "%y"));
  EXPECT_EQ("10", dateFormat(parseTimestamp("-1910-06-20"), "%y"));
  EXPECT_EQ("12", dateFormat(parseTimestamp("-12-06-20"), "%y"));
  EXPECT_EQ("00", dateFormat(parseTimestamp("1900-06-20"), "%y"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("1901-06-20"), "%y"));
  EXPECT_EQ("10", dateFormat(parseTimestamp("1910-06-20"), "%y"));

  // Day of week cases.
  for (int i = 0; i < 8; i++) {
    std::string date("1996-01-0" + std::to_string(i + 1));
    // Full length name.
    EXPECT_EQ(daysLong[i % 7], dateFormat(parseTimestamp(date), "%W"));
    // Abbreviated name.
    EXPECT_EQ(daysShort[i % 7], dateFormat(parseTimestamp(date), "%a"));
  }

  // Month cases.
  for (int i = 0; i < 12; i++) {
    std::string date("1996-" + std::to_string(i + 1) + "-01");
    std::string monthNum = std::to_string(i + 1);

    // Full length name.
    EXPECT_EQ(monthsLong[i % 12], dateFormat(parseTimestamp(date), "%M"));

    // Abbreviated name.
    EXPECT_EQ(monthsShort[i % 12], dateFormat(parseTimestamp(date), "%b"));

    // Numeric.
    EXPECT_EQ(monthNum, dateFormat(parseTimestamp(date), "%c"));

    // Numeric 0-padded.
    if (i + 1 < 10) {
      EXPECT_EQ("0" + monthNum, dateFormat(parseTimestamp(date), "%m"));
    } else {
      EXPECT_EQ(monthNum, dateFormat(parseTimestamp(date), "%m"));
    }
  }

  // Day of month cases.
  for (int i = 1; i <= 31; i++) {
    std::string dayOfMonth = std::to_string(i);
    std::string date("1970-01-" + dayOfMonth);
    EXPECT_EQ(dayOfMonth, dateFormat(parseTimestamp(date), "%e"));
    if (i < 10) {
      EXPECT_EQ("0" + dayOfMonth, dateFormat(parseTimestamp(date), "%d"));
    } else {
      EXPECT_EQ(dayOfMonth, dateFormat(parseTimestamp(date), "%d"));
    }
  }

  // Week of the year cases. Follows ISO week date format.
  //   https://en.wikipedia.org/wiki/ISO_week_date
  EXPECT_EQ("01", dateFormat(parseTimestamp("2024-01-01"), "%v"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("2024-01-07"), "%v"));
  EXPECT_EQ("02", dateFormat(parseTimestamp("2024-01-08"), "%v"));
  EXPECT_EQ("52", dateFormat(parseTimestamp("2024-12-29"), "%v"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("2024-12-30"), "%v"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("2024-12-31"), "%v"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("2025-01-01"), "%v"));
  EXPECT_EQ("53", dateFormat(parseTimestamp("2021-01-01"), "%v"));
  EXPECT_EQ("53", dateFormat(parseTimestamp("2021-01-03"), "%v"));
  EXPECT_EQ("01", dateFormat(parseTimestamp("2021-01-04"), "%v"));

  // Fraction of second cases.
  EXPECT_EQ(
      "000000", dateFormat(parseTimestamp("2022-01-01 00:00:00.0"), "%f"));
  EXPECT_EQ(
      "100000", dateFormat(parseTimestamp("2022-01-01 00:00:00.1"), "%f"));
  EXPECT_EQ(
      "110000", dateFormat(parseTimestamp("2022-01-01 01:01:01.11"), "%f"));
  EXPECT_EQ(
      "110000", dateFormat(parseTimestamp("2022-01-01 02:10:10.11"), "%f"));
  EXPECT_EQ(
      "999000", dateFormat(parseTimestamp("2022-01-01 03:30:30.999"), "%f"));
  EXPECT_EQ(
      "999000", dateFormat(parseTimestamp("2022-01-01 03:30:30.999"), "%f"));
  EXPECT_EQ(
      "999000", dateFormat(parseTimestamp("2022-01-01 03:30:30.999"), "%f"));
  EXPECT_EQ(
      "123000", dateFormat(parseTimestamp("2022-01-01 03:30:30.123"), "%f"));
  EXPECT_EQ(
      "099000", dateFormat(parseTimestamp("2022-01-01 03:30:30.099"), "%f"));
  EXPECT_EQ(
      "001000", dateFormat(parseTimestamp("2022-01-01 03:30:30.001234"), "%f"));

  // Hour cases.
  for (int i = 0; i < 24; i++) {
    std::string hour = std::to_string(i);
    int clockHour = (i + 11) % 12 + 1;
    std::string clockHourString = std::to_string(clockHour);
    std::string toBuild = "1996-01-01 " + hour + ":00:00";
    StringView date(toBuild);
    EXPECT_EQ(hour, dateFormat(parseTimestamp(date), "%k"));
    if (i < 10) {
      EXPECT_EQ("0" + hour, dateFormat(parseTimestamp(date), "%H"));
    } else {
      EXPECT_EQ(hour, dateFormat(parseTimestamp(date), "%H"));
    }

    EXPECT_EQ(clockHourString, dateFormat(parseTimestamp(date), "%l"));
    if (clockHour < 10) {
      EXPECT_EQ("0" + clockHourString, dateFormat(parseTimestamp(date), "%h"));
      EXPECT_EQ("0" + clockHourString, dateFormat(parseTimestamp(date), "%I"));
    } else {
      EXPECT_EQ(clockHourString, dateFormat(parseTimestamp(date), "%h"));
      EXPECT_EQ(clockHourString, dateFormat(parseTimestamp(date), "%I"));
    }
  }

  // Minute cases.
  for (int i = 0; i < 60; i++) {
    std::string minute = std::to_string(i);
    std::string toBuild = "1996-01-01 00:" + minute + ":00";
    StringView date(toBuild);
    if (i < 10) {
      EXPECT_EQ("0" + minute, dateFormat(parseTimestamp(date), "%i"));
    } else {
      EXPECT_EQ(minute, dateFormat(parseTimestamp(date), "%i"));
    }
  }

  // Second cases.
  for (int i = 0; i < 60; i++) {
    std::string second = std::to_string(i);
    std::string toBuild = "1996-01-01 00:00:" + second;
    StringView date(toBuild);
    if (i < 10) {
      EXPECT_EQ("0" + second, dateFormat(parseTimestamp(date), "%S"));
      EXPECT_EQ("0" + second, dateFormat(parseTimestamp(date), "%s"));
    } else {
      EXPECT_EQ(second, dateFormat(parseTimestamp(date), "%S"));
      EXPECT_EQ(second, dateFormat(parseTimestamp(date), "%s"));
    }
  }

  // Day of year cases.
  EXPECT_EQ("001", dateFormat(parseTimestamp("2022-01-01"), "%j"));
  EXPECT_EQ("010", dateFormat(parseTimestamp("2022-01-10"), "%j"));
  EXPECT_EQ("100", dateFormat(parseTimestamp("2022-04-10"), "%j"));
  EXPECT_EQ("365", dateFormat(parseTimestamp("2022-12-31"), "%j"));

  // Halfday of day cases.
  EXPECT_EQ("AM", dateFormat(parseTimestamp("2022-01-01 00:00:00"), "%p"));
  EXPECT_EQ("AM", dateFormat(parseTimestamp("2022-01-01 11:59:59"), "%p"));
  EXPECT_EQ("PM", dateFormat(parseTimestamp("2022-01-01 12:00:00"), "%p"));
  EXPECT_EQ("PM", dateFormat(parseTimestamp("2022-01-01 23:59:59"), "%p"));

  // 12-hour time cases.
  EXPECT_EQ(
      "12:00:00 AM", dateFormat(parseTimestamp("2022-01-01 00:00:00"), "%r"));
  EXPECT_EQ(
      "11:59:59 AM", dateFormat(parseTimestamp("2022-01-01 11:59:59"), "%r"));
  EXPECT_EQ(
      "12:00:00 PM", dateFormat(parseTimestamp("2022-01-01 12:00:00"), "%r"));
  EXPECT_EQ(
      "11:59:59 PM", dateFormat(parseTimestamp("2022-01-01 23:59:59"), "%r"));

  // 24-hour time cases.
  EXPECT_EQ(
      "00:00:00", dateFormat(parseTimestamp("2022-01-01 00:00:00"), "%T"));
  EXPECT_EQ(
      "11:59:59", dateFormat(parseTimestamp("2022-01-01 11:59:59"), "%T"));
  EXPECT_EQ(
      "12:00:00", dateFormat(parseTimestamp("2022-01-01 12:00:00"), "%T"));
  EXPECT_EQ(
      "23:59:59", dateFormat(parseTimestamp("2022-01-01 23:59:59"), "%T"));

  // Percent followed by non-existent specifier case.
  EXPECT_EQ("q", dateFormat(parseTimestamp("1970-01-01"), "%q"));
  EXPECT_EQ("z", dateFormat(parseTimestamp("1970-01-01"), "%z"));
  EXPECT_EQ("g", dateFormat(parseTimestamp("1970-01-01"), "%g"));

  // With timezone. Indian Standard Time (IST) UTC+5:30.
  setQueryTimeZone("Asia/Kolkata");

  EXPECT_EQ("1970-01-01", dateFormat(parseTimestamp("1970-01-01"), "%Y-%m-%d"));
  EXPECT_EQ(
      "2000-02-29 05:30:00 AM",
      dateFormat(parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %r"));
  EXPECT_EQ(
      "2000-02-29 05:30:00.987000",
      dateFormat(
          parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));
  EXPECT_EQ(
      "-2000-02-29 05:53:28.987000",
      dateFormat(
          parseTimestamp("-2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));

  // Same timestamps with a different timezone. Pacific Daylight Time (North
  // America) PDT UTC-8:00.
  setQueryTimeZone("America/Los_Angeles");

  EXPECT_EQ("1969-12-31", dateFormat(parseTimestamp("1970-01-01"), "%Y-%m-%d"));
  EXPECT_EQ(
      "2000-02-28 04:00:00 PM",
      dateFormat(parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %r"));
  EXPECT_EQ(
      "2000-02-28 16:00:00.987000",
      dateFormat(
          parseTimestamp("2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));
  EXPECT_EQ(
      "-2000-02-28 16:07:02.987000",
      dateFormat(
          parseTimestamp("-2000-02-29 00:00:00.987"), "%Y-%m-%d %H:%i:%s.%f"));

  // User format errors or unsupported errors.
  const auto timestamp = parseTimestamp("-2000-02-29 00:00:00.987");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%D"),
      "Date format specifier is not supported: %D");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%U"),
      "Date format specifier is not supported: %U");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%u"),
      "Date format specifier is not supported: %u");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%V"),
      "Date format specifier is not supported: %V");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%w"),
      "Date format specifier is not supported: %w");
  VELOX_ASSERT_THROW(
      dateFormat(timestamp, "%X"),
      "Date format specifier is not supported: %X");
}

TEST_F(DateTimeFunctionsTest, dateFormatTimestampWithTimezone) {
  const auto dateFormatTimestampWithTimezone =
      [&](const std::string& formatString,
          std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<std::string>(
            fmt::format("date_format(c0, '{}')", formatString),
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };

  EXPECT_EQ(
      "1969-12-31 11:00:00 PM",
      dateFormatTimestampWithTimezone(
          "%Y-%m-%d %r", TimestampWithTimezone(0, "-01:00")));
  EXPECT_EQ(
      "1973-11-30 12:33:09 AM",
      dateFormatTimestampWithTimezone(
          "%Y-%m-%d %r", TimestampWithTimezone(123456789000, "+03:00")));
  EXPECT_EQ(
      "1966-02-01 12:26:51 PM",
      dateFormatTimestampWithTimezone(
          "%Y-%m-%d %r", TimestampWithTimezone(-123456789000, "-14:00")));
  EXPECT_EQ(
      "2001-04-19 18:25:21.000000",
      dateFormatTimestampWithTimezone(
          "%Y-%m-%d %H:%i:%s.%f",
          TimestampWithTimezone(987654321000, "+14:00")));
  EXPECT_EQ(
      "1938-09-14 23:34:39.000000",
      dateFormatTimestampWithTimezone(
          "%Y-%m-%d %H:%i:%s.%f",
          TimestampWithTimezone(-987654321000, "+04:00")));
  EXPECT_EQ(
      "70-August-22 17:55:15 PM",
      dateFormatTimestampWithTimezone(
          "%y-%M-%e %T %p", TimestampWithTimezone(20220915000, "-07:00")));
  EXPECT_EQ(
      "69-May-11 20:04:45 PM",
      dateFormatTimestampWithTimezone(
          "%y-%M-%e %T %p", TimestampWithTimezone(-20220915000, "-03:00")));
}

TEST_F(DateTimeFunctionsTest, test_week_year) {
  const auto dateFormat = [&](std::optional<Timestamp> timestamp,
                              std::optional<std::string> format) {
    return evaluateOnce<std::string>("date_format(c0, c1)", timestamp, format);
  };
  auto rst_wy = dateFormat(Timestamp(1609545600, 0), "%x");
  EXPECT_EQ("2020", rst_wy);

  EXPECT_EQ(
      "1999-52",
      dateFormat(parseTimestamp("1999-12-31 23:59:59.999"), "%x-%v"));
  // 2023-01-01 is a Sunday, so it's part of the last week of 2022 (week 52)
  // according to ISO week date system.
  EXPECT_EQ(
      "2022-52",
      dateFormat(parseTimestamp("2023-01-01 00:00:00.000"), "%x-%v"));
}

TEST_F(DateTimeFunctionsTest, fromIso8601Date) {
  const auto fromIso = [&](const std::string& input) {
    return evaluateOnce<int32_t, std::string>("from_iso8601_date(c0)", input);
  };

  EXPECT_EQ(0, fromIso("1970-01-01"));
  EXPECT_EQ(9, fromIso("1970-01-10"));
  EXPECT_EQ(-1, fromIso("1969-12-31"));
  EXPECT_EQ(0, fromIso("1970"));
  EXPECT_EQ(0, fromIso("1970-01"));
  EXPECT_EQ(0, fromIso("1970-1"));
  EXPECT_EQ(8, fromIso("1970-1-9"));
  EXPECT_EQ(-31, fromIso("1969-12"));
  EXPECT_EQ(-31, fromIso("1969-12-1"));
  EXPECT_EQ(-31, fromIso("1969-12-01"));
  EXPECT_EQ(-719862, fromIso("-1-2-1"));

  VELOX_ASSERT_THROW(fromIso(" 2024-01-12"), "Unable to parse date value");
  VELOX_ASSERT_THROW(fromIso("2024-01-12  "), "Unable to parse date value");
  VELOX_ASSERT_THROW(fromIso("2024 "), "Unable to parse date value");
  VELOX_ASSERT_THROW(fromIso("2024-01-xx"), "Unable to parse date value");
  VELOX_ASSERT_THROW(
      fromIso("2024-01-02T12:31:00"), "Unable to parse date value");
  VELOX_ASSERT_THROW(
      fromIso("2024-01-02 12:31:00"), "Unable to parse date value");
}

TEST_F(DateTimeFunctionsTest, fromIso8601Timestamp) {
  const auto fromIso = [&](const std::string& input) {
    auto result =
        evaluateOnce<int64_t, std::string>("from_iso8601_timestamp(c0)", input);
    return TimestampWithTimezone::unpack(result);
  };

  // Full strings with different time zones.
  const auto millis = kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute +
      56 * kMillisInSecond + 123;
  const std::string ts = "1970-01-02T11:38:56.123";

  EXPECT_EQ(
      TimestampWithTimezone(millis + 5 * kMillisInHour, "-05:00"),
      fromIso(ts + "-05:00"));

  EXPECT_EQ(
      TimestampWithTimezone(millis - 8 * kMillisInHour, "+08:00"),
      fromIso(ts + "+08:00"));

  EXPECT_EQ(TimestampWithTimezone(millis, "UTC"), fromIso(ts + "Z"));

  EXPECT_EQ(TimestampWithTimezone(millis, "UTC"), fromIso(ts));

  // Maximum timestamp.
  EXPECT_EQ(
      TimestampWithTimezone(2251799813685247, "UTC"),
      fromIso("73326-09-11T20:14:45.247"));

  // Minimum timestamp.
  EXPECT_EQ(
      TimestampWithTimezone(-2251799813685248, "UTC"),
      fromIso("-69387-04-22T03:45:14.752"));

  // Partial strings with different session time zones.
  struct {
    const tz::TimeZone* timezone;
    int32_t offset;
  } timezones[] = {
      {tz::locateZone("America/New_York"), -5 * kMillisInHour},
      {tz::locateZone("Asia/Kolkata"),
       5 * kMillisInHour + 30 * kMillisInMinute},
  };

  for (const auto& timezone : timezones) {
    setQueryTimeZone(timezone.timezone->name());

    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute +
                56 * kMillisInSecond + 123 - 3 * kMillisInHour,
            tz::locateZone("+03:00")),
        fromIso("1970-01-02T11:38:56.123+03:00"));

    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute +
                56 * kMillisInSecond + 123 - timezone.offset,
            timezone.timezone),
        fromIso("1970-01-02T11:38:56.123"));

    // Comma separator between seconds and microseconds.
    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute +
                56 * kMillisInSecond + 123 - timezone.offset,
            timezone.timezone),
        fromIso("1970-01-02T11:38:56,123"));

    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute +
                56 * kMillisInSecond - timezone.offset,
            timezone.timezone),
        fromIso("1970-01-02T11:38:56"));

    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour + 38 * kMillisInMinute -
                timezone.offset,
            timezone.timezone),
        fromIso("1970-01-02T11:38"));

    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay + 11 * kMillisInHour - timezone.offset,
            timezone.timezone),
        fromIso("1970-01-02T11"));

    // No time.
    EXPECT_EQ(
        TimestampWithTimezone(
            kMillisInDay - timezone.offset, timezone.timezone),
        fromIso("1970-01-02"));

    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970-01-01"));
    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970-01"));
    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970"));

    // Trailing time separator.
    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970-01-01T"));
    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970-01T"));
    EXPECT_EQ(
        TimestampWithTimezone(-timezone.offset, timezone.timezone),
        fromIso("1970T"));

    // No time but with a time zone.
    EXPECT_EQ(
        TimestampWithTimezone(-1 * kMillisInHour, tz::locateZone("+01:00")),
        fromIso("1970-01-01T+01:00"));
    EXPECT_EQ(
        TimestampWithTimezone(2 * kMillisInHour, tz::locateZone("-02:00")),
        fromIso("1970-01T-02:00"));
    EXPECT_EQ(
        TimestampWithTimezone(-14 * kMillisInHour, tz::locateZone("+14:00")),
        fromIso("1970T+14:00"));

    // No date.
    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond +
                123 - timezone.offset,
            timezone.timezone),
        fromIso("T11:38:56.123"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond +
                123 - timezone.offset,
            timezone.timezone),
        fromIso("T11:38:56,123"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond -
                timezone.offset,
            timezone.timezone),
        fromIso("T11:38:56"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute - timezone.offset,
            timezone.timezone),
        fromIso("T11:38"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour - timezone.offset, timezone.timezone),
        fromIso("T11"));

    // No date but with a time zone.
    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond +
                123 + 14 * kMillisInHour,
            tz::locateZone("-14:00")),
        fromIso("T11:38:56.123-14:00"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond +
                123 - 11 * kMillisInHour,
            tz::locateZone("+11:00")),
        fromIso("T11:38:56,123+11:00"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute + 56 * kMillisInSecond +
                12 * kMillisInHour,
            tz::locateZone("-12:00")),
        fromIso("T11:38:56-12:00"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 38 * kMillisInMinute - 7 * kMillisInHour,
            tz::locateZone("+07:00")),
        fromIso("T11:38+07:00"));

    EXPECT_EQ(
        TimestampWithTimezone(
            11 * kMillisInHour + 8 * kMillisInHour, tz::locateZone("-08:00")),
        fromIso("T11-08:00"));
  }

  VELOX_ASSERT_THROW(
      fromIso("1970-01-02 11:38"),
      R"(Unable to parse timestamp value: "1970-01-02 11:38")");

  VELOX_ASSERT_THROW(
      fromIso("1970-01-02T11:38:56.123 America/New_York"),
      R"(Unable to parse timestamp value: "1970-01-02T11:38:56.123 America/New_York")");
  VELOX_ASSERT_THROW(
      fromIso("1970-01-02T11:38:56+16:00:01"),
      "Unknown timezone value: \"+16:00:01\"");

  VELOX_ASSERT_THROW(fromIso("T"), R"(Unable to parse timestamp value: "T")");

  // Leading and trailing spaces are not allowed.
  VELOX_ASSERT_THROW(
      fromIso(" 1970-01-02"),
      R"(Unable to parse timestamp value: " 1970-01-02")");

  VELOX_ASSERT_THROW(
      fromIso("1970-01-02 "),
      R"(Unable to parse timestamp value: "1970-01-02 ")");

  // Test overflow in either direction.
  setQueryTimeZone("UTC");
  VELOX_ASSERT_THROW(
      fromIso("73326-09-11T20:14:45.248"), "TimestampWithTimeZone overflow");
  VELOX_ASSERT_THROW(
      fromIso("-69387-04-22T03:45:14.751"), "TimestampWithTimeZone overflow");
}

TEST_F(DateTimeFunctionsTest, dateParseMonthOfYearText) {
  auto parseAndFormat = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>(
        "date_format(date_parse(c0, '%M_%Y'), '%Y-%m')", input);
  };

  EXPECT_EQ(parseAndFormat(std::nullopt), std::nullopt);
  EXPECT_EQ(parseAndFormat("jan_2024"), "2024-01");
  EXPECT_EQ(parseAndFormat("JAN_2024"), "2024-01");
  EXPECT_EQ(parseAndFormat("january_2024"), "2024-01");
  EXPECT_EQ(parseAndFormat("JANUARY_2024"), "2024-01");

  EXPECT_EQ(parseAndFormat("feb_2024"), "2024-02");
  EXPECT_EQ(parseAndFormat("FEB_2024"), "2024-02");
  EXPECT_EQ(parseAndFormat("february_2024"), "2024-02");
  EXPECT_EQ(parseAndFormat("FEBRUARY_2024"), "2024-02");

  EXPECT_EQ(parseAndFormat("mar_2024"), "2024-03");
  EXPECT_EQ(parseAndFormat("MAR_2024"), "2024-03");
  EXPECT_EQ(parseAndFormat("march_2024"), "2024-03");
  EXPECT_EQ(parseAndFormat("MARCH_2024"), "2024-03");

  EXPECT_EQ(parseAndFormat("apr_2024"), "2024-04");
  EXPECT_EQ(parseAndFormat("APR_2024"), "2024-04");
  EXPECT_EQ(parseAndFormat("april_2024"), "2024-04");
  EXPECT_EQ(parseAndFormat("APRIL_2024"), "2024-04");

  EXPECT_EQ(parseAndFormat("may_2024"), "2024-05");
  EXPECT_EQ(parseAndFormat("MAY_2024"), "2024-05");

  EXPECT_EQ(parseAndFormat("jun_2024"), "2024-06");
  EXPECT_EQ(parseAndFormat("JUN_2024"), "2024-06");
  EXPECT_EQ(parseAndFormat("june_2024"), "2024-06");
  EXPECT_EQ(parseAndFormat("JUNE_2024"), "2024-06");

  EXPECT_EQ(parseAndFormat("jul_2024"), "2024-07");
  EXPECT_EQ(parseAndFormat("JUL_2024"), "2024-07");
  EXPECT_EQ(parseAndFormat("july_2024"), "2024-07");
  EXPECT_EQ(parseAndFormat("JULY_2024"), "2024-07");

  EXPECT_EQ(parseAndFormat("aug_2024"), "2024-08");
  EXPECT_EQ(parseAndFormat("AUG_2024"), "2024-08");
  EXPECT_EQ(parseAndFormat("august_2024"), "2024-08");
  EXPECT_EQ(parseAndFormat("AUGUST_2024"), "2024-08");

  EXPECT_EQ(parseAndFormat("sep_2024"), "2024-09");
  EXPECT_EQ(parseAndFormat("SEP_2024"), "2024-09");
  EXPECT_EQ(parseAndFormat("september_2024"), "2024-09");
  EXPECT_EQ(parseAndFormat("SEPTEMBER_2024"), "2024-09");

  EXPECT_EQ(parseAndFormat("oct_2024"), "2024-10");
  EXPECT_EQ(parseAndFormat("OCT_2024"), "2024-10");
  EXPECT_EQ(parseAndFormat("october_2024"), "2024-10");
  EXPECT_EQ(parseAndFormat("OCTOBER_2024"), "2024-10");

  EXPECT_EQ(parseAndFormat("nov_2024"), "2024-11");
  EXPECT_EQ(parseAndFormat("NOV_2024"), "2024-11");
  EXPECT_EQ(parseAndFormat("november_2024"), "2024-11");
  EXPECT_EQ(parseAndFormat("NOVEMBER_2024"), "2024-11");

  EXPECT_EQ(parseAndFormat("dec_2024"), "2024-12");
  EXPECT_EQ(parseAndFormat("DEC_2024"), "2024-12");
  EXPECT_EQ(parseAndFormat("december_2024"), "2024-12");
  EXPECT_EQ(parseAndFormat("DECEMBER_2024"), "2024-12");
}

TEST_F(DateTimeFunctionsTest, dateParse) {
  const auto dateParse = [&](std::optional<std::string> input,
                             std::optional<std::string> format) {
    return evaluateOnce<Timestamp>("date_parse(c0, c1)", input, format);
  };

  // Check null behavior.
  EXPECT_EQ(std::nullopt, dateParse("1970-01-01", std::nullopt));
  EXPECT_EQ(std::nullopt, dateParse(std::nullopt, "YYYY-MM-dd"));
  EXPECT_EQ(std::nullopt, dateParse(std::nullopt, std::nullopt));

  // Simple tests. More exhaustive tests are provided in DateTimeFormatterTest.
  EXPECT_EQ(Timestamp(86400, 0), dateParse("1970-01-02", "%Y-%m-%d"));
  EXPECT_EQ(Timestamp(0, 0), dateParse("1970-01-01", "%Y-%m-%d"));
  EXPECT_EQ(Timestamp(86400, 0), dateParse("19700102", "%Y%m%d"));

  // Tests for differing query timezones
  // 118860000 is the number of milliseconds since epoch at 1970-01-02
  // 09:01:00.000 UTC.
  EXPECT_EQ(
      Timestamp(118860, 0), dateParse("1970-01-02+09:01", "%Y-%m-%d+%H:%i"));

  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ(
      Timestamp(118860, 0), dateParse("1970-01-02+01:01", "%Y-%m-%d+%H:%i"));

  setQueryTimeZone("America/Noronha");
  EXPECT_EQ(
      Timestamp(118860, 0), dateParse("1970-01-02+07:01", "%Y-%m-%d+%H:%i"));

  setQueryTimeZone("+04:00");
  EXPECT_EQ(
      Timestamp(118860, 0), dateParse("1970-01-02+13:01", "%Y-%m-%d+%H:%i"));

  setQueryTimeZone("Asia/Kolkata");
  // 66600000 is the number of millisecond since epoch at 1970-01-01
  // 18:30:00.000 UTC.
  EXPECT_EQ(
      Timestamp(66600, 0), dateParse("1970-01-02+00:00", "%Y-%m-%d+%H:%i"));

  // -66600000 is the number of millisecond since epoch at 1969-12-31
  // 05:30:00.000 UTC.
  EXPECT_EQ(
      Timestamp(-66600, 0), dateParse("1969-12-31+11:00", "%Y-%m-%d+%H:%i"));

  setQueryTimeZone("America/Los_Angeles");
  // Tests if it uses weekdateformat if %v not present but %a is present.
  EXPECT_EQ(
      Timestamp(1730707200, 0),
      dateParse("04-Nov-2024 (Mon)", "%d-%b-%Y (%a)"));

  VELOX_ASSERT_THROW(dateParse("", "%y+"), "Invalid date format: ''");
  VELOX_ASSERT_THROW(dateParse("1", "%y+"), "Invalid date format: '1'");
  VELOX_ASSERT_THROW(dateParse("116", "%y+"), "Invalid date format: '116'");
}

TEST_F(DateTimeFunctionsTest, dateFunctionVarchar) {
  const auto dateFunction = [&](const std::optional<std::string>& dateString) {
    return evaluateOnce<int32_t>("date(c0)", dateString);
  };

  // Date(0) is 1970-01-01.
  EXPECT_EQ(0, dateFunction("1970-01-01"));
  // Date(18297) is 2020-02-05.
  EXPECT_EQ(18297, dateFunction("2020-02-05"));
  // Date(-18297) is 1919-11-28.
  EXPECT_EQ(-18297, dateFunction("1919-11-28"));

  // Allow leading and trailing spaces.
  EXPECT_EQ(18297, dateFunction("   2020-02-05"));
  EXPECT_EQ(18297, dateFunction("  2020-02-05   "));
  EXPECT_EQ(18297, dateFunction("2020-02-05 "));

  // Illegal date format.
  VELOX_ASSERT_THROW(
      dateFunction("2020-02-05 11:00"),
      "Unable to parse date value: \"2020-02-05 11:00\"");
}

TEST_F(DateTimeFunctionsTest, dateFunctionTimestamp) {
  static const int64_t kSecondsInDay = 86'400;
  static const uint64_t kNanosInSecond = 1'000'000'000;

  const auto dateFunction = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int32_t>("date(c0)", timestamp);
  };

  EXPECT_EQ(0, dateFunction(Timestamp()));
  EXPECT_EQ(1, dateFunction(Timestamp(kSecondsInDay, 0)));
  EXPECT_EQ(-1, dateFunction(Timestamp(-kSecondsInDay, 0)));
  EXPECT_EQ(18297, dateFunction(Timestamp(18297 * kSecondsInDay, 0)));
  EXPECT_EQ(18297, dateFunction(Timestamp(18297 * kSecondsInDay, 123)));
  EXPECT_EQ(-18297, dateFunction(Timestamp(-18297 * kSecondsInDay, 0)));
  EXPECT_EQ(-18297, dateFunction(Timestamp(-18297 * kSecondsInDay, 123)));

  // Last second of day 0
  EXPECT_EQ(0, dateFunction(Timestamp(kSecondsInDay - 1, 0)));
  // Last nanosecond of day 0
  EXPECT_EQ(0, dateFunction(Timestamp(kSecondsInDay - 1, kNanosInSecond - 1)));

  // Last second of day -1
  EXPECT_EQ(-1, dateFunction(Timestamp(-1, 0)));
  // Last nanosecond of day -1
  EXPECT_EQ(-1, dateFunction(Timestamp(-1, kNanosInSecond - 1)));

  // Last second of day 18297
  EXPECT_EQ(
      18297,
      dateFunction(Timestamp(18297 * kSecondsInDay + kSecondsInDay - 1, 0)));
  // Last nanosecond of day 18297
  EXPECT_EQ(
      18297,
      dateFunction(Timestamp(
          18297 * kSecondsInDay + kSecondsInDay - 1, kNanosInSecond - 1)));

  // Last second of day -18297
  EXPECT_EQ(
      -18297,
      dateFunction(Timestamp(-18297 * kSecondsInDay + kSecondsInDay - 1, 0)));
  // Last nanosecond of day -18297
  EXPECT_EQ(
      -18297,
      dateFunction(Timestamp(
          -18297 * kSecondsInDay + kSecondsInDay - 1, kNanosInSecond - 1)));
}

TEST_F(DateTimeFunctionsTest, dateFunctionTimestampWithTimezone) {
  static const int64_t kSecondsInDay = 86'400;

  const auto dateFunction =
      [&](std::optional<int64_t> timestamp,
          const std::optional<std::string>& timeZoneName) {
        auto r1 = evaluateOnce<int32_t>(
            "date(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(
                TimestampWithTimezone(*timestamp, *timeZoneName)));
        auto r2 = evaluateOnce<int32_t>(
            "cast(c0 as date)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(
                TimestampWithTimezone(*timestamp, *timeZoneName)));
        EXPECT_EQ(r1, r2);
        return r1;
      };

  // 1970-01-01 00:00:00.000 +00:00
  EXPECT_EQ(0, dateFunction(0, "+00:00"));
  EXPECT_EQ(0, dateFunction(0, "Europe/London"));
  // 1970-01-01 00:00:00.000 -08:00
  EXPECT_EQ(-1, dateFunction(0, "-08:00"));
  EXPECT_EQ(-1, dateFunction(0, "America/Los_Angeles"));
  // 1970-01-01 00:00:00.000 +08:00
  EXPECT_EQ(0, dateFunction(0, "+08:00"));
  EXPECT_EQ(0, dateFunction(0, "Asia/Chongqing"));
  // 1970-01-01 18:00:00.000 +08:00
  EXPECT_EQ(1, dateFunction(18 * 3'600 * 1'000, "+08:00"));
  EXPECT_EQ(1, dateFunction(18 * 3'600 * 1'000, "Asia/Chongqing"));
  // 1970-01-01 06:00:00.000 -08:00
  EXPECT_EQ(-1, dateFunction(6 * 3'600 * 1'000, "-08:00"));
  EXPECT_EQ(-1, dateFunction(6 * 3'600 * 1'000, "America/Los_Angeles"));

  // 2020-02-05 10:00:00.000 +08:00
  EXPECT_EQ(
      18297,
      dateFunction((18297 * kSecondsInDay + 10 * 3'600) * 1'000, "+08:00"));
  EXPECT_EQ(
      18297,
      dateFunction(
          (18297 * kSecondsInDay + 10 * 3'600) * 1'000, "Asia/Chongqing"));
  // 2020-02-05 20:00:00.000 +08:00
  EXPECT_EQ(
      18298,
      dateFunction((18297 * kSecondsInDay + 20 * 3'600) * 1'000, "+08:00"));
  EXPECT_EQ(
      18298,
      dateFunction(
          (18297 * kSecondsInDay + 20 * 3'600) * 1'000, "Asia/Chongqing"));
  // 2020-02-05 16:00:00.000 -08:00
  EXPECT_EQ(
      18297,
      dateFunction((18297 * kSecondsInDay + 16 * 3'600) * 1'000, "-08:00"));
  EXPECT_EQ(
      18297,
      dateFunction(
          (18297 * kSecondsInDay + 16 * 3'600) * 1'000, "America/Los_Angeles"));
  // 2020-02-05 06:00:00.000 -08:00
  EXPECT_EQ(
      18296,
      dateFunction((18297 * kSecondsInDay + 6 * 3'600) * 1'000, "-08:00"));
  EXPECT_EQ(
      18296,
      dateFunction(
          (18297 * kSecondsInDay + 6 * 3'600) * 1'000, "America/Los_Angeles"));

  // 1919-11-28 10:00:00.000 +08:00
  EXPECT_EQ(
      -18297,
      dateFunction((-18297 * kSecondsInDay + 10 * 3'600) * 1'000, "+08:00"));
  EXPECT_EQ(
      -18297,
      dateFunction(
          (-18297 * kSecondsInDay + 10 * 3'600) * 1'000, "Asia/Chongqing"));
  // 1919-11-28 20:00:00.000 +08:00
  EXPECT_EQ(
      -18296,
      dateFunction((-18297 * kSecondsInDay + 20 * 3'600) * 1'000, "+08:00"));
  EXPECT_EQ(
      -18296,
      dateFunction(
          (-18297 * kSecondsInDay + 20 * 3'600) * 1'000, "Asia/Chongqing"));
  // 1919-11-28 16:00:00.000 -08:00
  EXPECT_EQ(
      -18297,
      dateFunction((-18297 * kSecondsInDay + 16 * 3'600) * 1'000, "-08:00"));
  EXPECT_EQ(
      -18297,
      dateFunction(
          (-18297 * kSecondsInDay + 16 * 3'600) * 1'000,
          "America/Los_Angeles"));
  // 1919-11-28 06:00:00.000 -08:00
  EXPECT_EQ(
      -18298,
      dateFunction((-18297 * kSecondsInDay + 6 * 3'600) * 1'000, "-08:00"));
  EXPECT_EQ(
      -18298,
      dateFunction(
          (-18297 * kSecondsInDay + 6 * 3'600) * 1'000, "America/Los_Angeles"));
}

TEST_F(DateTimeFunctionsTest, castDateForDateFunction) {
  setQueryTimeZone("America/Los_Angeles");

  static const int64_t kSecondsInDay = 86'400;
  static const uint64_t kNanosInSecond = 1'000'000'000;
  const auto castDateTest = [&](std::optional<Timestamp> timestamp) {
    auto r1 = evaluateOnce<int32_t>("cast(c0 as date)", timestamp);
    auto r2 = evaluateOnce<int32_t>("date(c0)", timestamp);
    EXPECT_EQ(r1, r2);
    return r1;
  };

  // Note adjustments for PST timezone.
  EXPECT_EQ(-1, castDateTest(Timestamp()));
  EXPECT_EQ(0, castDateTest(Timestamp(kSecondsInDay, 0)));
  EXPECT_EQ(-2, castDateTest(Timestamp(-kSecondsInDay, 0)));
  EXPECT_EQ(18296, castDateTest(Timestamp(18297 * kSecondsInDay, 0)));
  EXPECT_EQ(18296, castDateTest(Timestamp(18297 * kSecondsInDay, 123)));
  EXPECT_EQ(-18298, castDateTest(Timestamp(-18297 * kSecondsInDay, 0)));
  EXPECT_EQ(-18298, castDateTest(Timestamp(-18297 * kSecondsInDay, 123)));

  // Last second of day 0.
  EXPECT_EQ(0, castDateTest(Timestamp(kSecondsInDay - 1, 0)));
  // Last nanosecond of day 0.
  EXPECT_EQ(0, castDateTest(Timestamp(kSecondsInDay - 1, kNanosInSecond - 1)));

  // Last second of day -1.
  EXPECT_EQ(-1, castDateTest(Timestamp(-1, 0)));
  // Last nanosecond of day -1.
  EXPECT_EQ(-1, castDateTest(Timestamp(-1, kNanosInSecond - 1)));

  // Last second of day 18297.
  EXPECT_EQ(
      18297,
      castDateTest(Timestamp(18297 * kSecondsInDay + kSecondsInDay - 1, 0)));
  // Last nanosecond of day 18297.
  EXPECT_EQ(
      18297,
      castDateTest(Timestamp(
          18297 * kSecondsInDay + kSecondsInDay - 1, kNanosInSecond - 1)));

  // Last second of day -18297.
  EXPECT_EQ(
      -18297,
      castDateTest(Timestamp(-18297 * kSecondsInDay + kSecondsInDay - 1, 0)));
  // Last nanosecond of day -18297.
  EXPECT_EQ(
      -18297,
      castDateTest(Timestamp(
          -18297 * kSecondsInDay + kSecondsInDay - 1, kNanosInSecond - 1)));

  // Timestamps in the distant future in different DST time zones.
  EXPECT_EQ(376358, castDateTest(Timestamp(32517359891, 0)));
  EXPECT_EQ(376231, castDateTest(Timestamp(32506387200, 0)));
}

TEST_F(DateTimeFunctionsTest, currentDateWithTimezone) {
  // Since the execution of the code is slightly delayed, it is difficult for us
  // to get the correct value of current_date. If you compare directly based on
  // the current time, you may get wrong result at the last second of the day,
  // and current_date may be the next day of the comparison value. In order to
  // avoid this situation, we compute a new comparison value after the execution
  // of current_date, so that the result of current_date is either consistent
  // with the first comparison value or the second comparison value, and the
  // difference between the two comparison values is at most one day.
  auto emptyRowVector = makeRowVector(ROW({}), 1);
  auto tz = "America/Los_Angeles";
  setQueryTimeZone(tz);
  auto dateBefore = getCurrentDate(tz);
  auto result = evaluateOnce<int32_t>("current_date()", emptyRowVector);
  auto dateAfter = getCurrentDate(tz);

  EXPECT_TRUE(result.has_value());
  EXPECT_LE(dateBefore, result);
  EXPECT_LE(result, dateAfter);
  EXPECT_LE(dateAfter - dateBefore, 1);
}

TEST_F(DateTimeFunctionsTest, currentDateWithoutTimezone) {
  auto emptyRowVector = makeRowVector(ROW({}), 1);

  // Do not set the timezone, so the timezone obtained from QueryConfig
  // will be nullptr.
  auto dateBefore = getCurrentDate(std::nullopt);
  auto result = evaluateOnce<int32_t>("current_date()", emptyRowVector);
  auto dateAfter = getCurrentDate(std::nullopt);

  EXPECT_TRUE(result.has_value());
  EXPECT_LE(dateBefore, result);
  EXPECT_LE(result, dateAfter);
  EXPECT_LE(dateAfter - dateBefore, 1);
}

TEST_F(DateTimeFunctionsTest, timeZoneHour) {
  const auto timezone_hour = [&](const char* time, const char* timezone) {
    Timestamp ts = parseTimestamp(time);
    auto timestamp = ts.toMillis();
    auto hour = evaluateOnce<int64_t>(
                    "timezone_hour(c0)",
                    TIMESTAMP_WITH_TIME_ZONE(),
                    TimestampWithTimezone::pack(
                        TimestampWithTimezone(timestamp, timezone)))
                    .value();
    return hour;
  };

  // Asia/Kolkata - should return 5 throughout the year
  EXPECT_EQ(5, timezone_hour("2023-01-01 03:20:00", "Asia/Kolkata"));
  EXPECT_EQ(5, timezone_hour("2023-06-01 03:20:00", "Asia/Kolkata"));

  // America/Los_Angeles - Day light savings is from March 12 to Nov 5
  EXPECT_EQ(-8, timezone_hour("2023-03-11 12:00:00", "America/Los_Angeles"));
  EXPECT_EQ(-8, timezone_hour("2023-03-12 02:30:00", "America/Los_Angeles"));
  EXPECT_EQ(-7, timezone_hour("2023-03-13 12:00:00", "America/Los_Angeles"));
  EXPECT_EQ(-7, timezone_hour("2023-11-05 01:30:00", "America/Los_Angeles"));
  EXPECT_EQ(-8, timezone_hour("2023-12-05 01:30:00", "America/Los_Angeles"));

  // Different time with same date
  EXPECT_EQ(-4, timezone_hour("2023-01-01 03:20:00", "Canada/Atlantic"));
  EXPECT_EQ(-4, timezone_hour("2023-01-01 10:00:00", "Canada/Atlantic"));

  // By definition (+/-) 00:00 offsets should always return the hour part of the
  // offset itself.
  EXPECT_EQ(0, timezone_hour("2023-12-05 01:30:00", "+00:00"));
  EXPECT_EQ(8, timezone_hour("2023-12-05 01:30:00", "+08:00"));
  EXPECT_EQ(-10, timezone_hour("2023-12-05 01:30:00", "-10:00"));

  // Invalid inputs
  VELOX_ASSERT_THROW(
      timezone_hour("invalid_date", "Canada/Atlantic"),
      "Unable to parse timestamp value: \"invalid_date\", expected format is (YYYY-MM-DD HH:MM:SS[.MS])");
  VELOX_ASSERT_THROW(
      timezone_hour("123456", "Canada/Atlantic"),
      "Unable to parse timestamp value: \"123456\", expected format is (YYYY-MM-DD HH:MM:SS[.MS])");
}

TEST_F(DateTimeFunctionsTest, timeZoneMinute) {
  const auto timezone_minute = [&](const char* time, const char* timezone) {
    Timestamp ts = parseTimestamp(time);
    auto timestamp = ts.toMillis();
    auto minute = evaluateOnce<int64_t>(
                      "timezone_minute(c0)",
                      TIMESTAMP_WITH_TIME_ZONE(),
                      TimestampWithTimezone::pack(
                          TimestampWithTimezone(timestamp, timezone)))
                      .value();
    return minute;
  };

  EXPECT_EQ(30, timezone_minute("1970-01-01 03:20:00", "Asia/Kolkata"));
  EXPECT_EQ(0, timezone_minute("1970-01-01 03:20:00", "America/Los_Angeles"));
  EXPECT_EQ(0, timezone_minute("1970-05-01 04:20:00", "America/Los_Angeles"));
  EXPECT_EQ(0, timezone_minute("1970-01-01 03:20:00", "Canada/Atlantic"));
  EXPECT_EQ(30, timezone_minute("1970-01-01 03:20:00", "Asia/Katmandu"));
  EXPECT_EQ(45, timezone_minute("1970-01-01 03:20:00", "Pacific/Chatham"));

  // By definition (+/-) 00:00 offsets should always return the minute part of
  // the offset itself.
  EXPECT_EQ(0, timezone_minute("2023-12-05 01:30:00", "+00:00"));
  EXPECT_EQ(17, timezone_minute("2023-12-05 01:30:00", "+08:17"));
  EXPECT_EQ(-59, timezone_minute("2023-12-05 01:30:00", "-10:59"));

  VELOX_ASSERT_THROW(
      timezone_minute("abc", "Pacific/Chatham"),
      "Unable to parse timestamp value: \"abc\", expected format is (YYYY-MM-DD HH:MM:SS[.MS])");
  VELOX_ASSERT_THROW(
      timezone_minute("2023-", "Pacific/Chatham"),
      "Unable to parse timestamp value: \"2023-\", expected format is (YYYY-MM-DD HH:MM:SS[.MS])");
}

TEST_F(DateTimeFunctionsTest, timestampWithTimezoneComparisons) {
  auto runAndCompare = [&](const std::string& expr,
                           const RowVectorPtr& inputs,
                           const VectorPtr& expectedResult) {
    auto actual = evaluate(expr, inputs);
    test::assertEqualVectors(expectedResult, actual);
  };

  /// Timestamp with timezone is internally represented with the milliseconds
  /// already converted to UTC and thus normalized. The timezone does not play
  /// a role in the comparison.
  /// For example, 1970-01-01-06:00:00+02:00 is stored as
  /// TIMESTAMP WITH TIMEZONE value (14400000, 960). 960 being the tzid
  /// representing +02:00. And 1970-01-01-04:00:00+00:00 is stored as (14400000,
  /// 0). These timestamps are equivalent.
  auto timestampsLhs = std::vector<int64_t>{0, 0, 1000};
  auto timezonesLhs = std::vector<TimeZoneKey>{900, 900, 800};
  VectorPtr timestampWithTimezoneLhs = makeTimestampWithTimeZoneVector(
      timestampsLhs.size(),
      [&](auto row) { return timestampsLhs[row]; },
      [&](auto row) { return timezonesLhs[row]; });

  auto timestampsRhs = std::vector<int64_t>{0, 1000, 0};
  auto timezonesRhs = std::vector<TimeZoneKey>{900, 900, 800};
  VectorPtr timestampWithTimezoneRhs = makeTimestampWithTimeZoneVector(
      timestampsRhs.size(),
      [&](auto row) { return timestampsRhs[row]; },
      [&](auto row) { return timezonesRhs[row]; });
  auto inputs =
      makeRowVector({timestampWithTimezoneLhs, timestampWithTimezoneRhs});

  auto expectedEq = makeFlatVector<bool>({true, false, false});
  runAndCompare("c0 = c1", inputs, expectedEq);

  auto expectedNeq = makeFlatVector<bool>({false, true, true});
  runAndCompare("c0 != c1", inputs, expectedNeq);

  auto expectedLt = makeFlatVector<bool>({false, true, false});
  runAndCompare("c0 < c1", inputs, expectedLt);

  auto expectedGt = makeFlatVector<bool>({false, false, true});
  runAndCompare("c0 > c1", inputs, expectedGt);

  auto expectedLte = makeFlatVector<bool>({true, true, false});
  runAndCompare("c0 <= c1", inputs, expectedLte);

  auto expectedGte = makeFlatVector<bool>({true, false, true});
  runAndCompare("c0 >= c1", inputs, expectedGte);

  auto expectedBetween = makeNullableFlatVector<bool>({true, true, false});
  runAndCompare("c0 between c0 and c1", inputs, expectedBetween);
}

TEST_F(DateTimeFunctionsTest, timeComparisons) {
  const auto eq = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 = c1", {TIME(), TIME()}, a, b);
  };
  const auto neq = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 != c1", {TIME(), TIME()}, a, b);
  };
  const auto lt = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 < c1", {TIME(), TIME()}, a, b);
  };
  const auto lte = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 <= c1", {TIME(), TIME()}, a, b);
  };
  const auto gt = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 > c1", {TIME(), TIME()}, a, b);
  };
  const auto gte = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>("c0 >= c1", {TIME(), TIME()}, a, b);
  };
  const auto between = [&](std::optional<int64_t> value,
                           std::optional<int64_t> lower,
                           std::optional<int64_t> upper) {
    return evaluateOnce<bool>(
        "c0 between c1 and c2", {TIME(), TIME(), TIME()}, value, lower, upper);
  };

  // test equality
  EXPECT_EQ(true, eq(0, 0));
  EXPECT_EQ(false, eq(0, 1000));
  EXPECT_EQ(true, eq(3600000, 3600000)); // 01:00:00
  EXPECT_EQ(std::nullopt, eq(std::nullopt, 0));
  EXPECT_EQ(std::nullopt, eq(0, std::nullopt));

  // test inequality
  EXPECT_EQ(false, neq(0, 0));
  EXPECT_EQ(true, neq(0, 1000));
  EXPECT_EQ(false, neq(3600000, 3600000));

  // test less than
  EXPECT_EQ(false, lt(0, 0));
  EXPECT_EQ(true, lt(0, 1000));
  EXPECT_EQ(false, lt(1000, 0));
  EXPECT_EQ(true, lt(3600000, 7200000)); // 01:00:00 < 02:00:00

  // test less than or equal
  EXPECT_EQ(true, lte(0, 0));
  EXPECT_EQ(true, lte(0, 1000));
  EXPECT_EQ(false, lte(1000, 0));

  // test greater than
  EXPECT_EQ(false, gt(0, 0));
  EXPECT_EQ(false, gt(0, 1000));
  EXPECT_EQ(true, gt(1000, 0));
  EXPECT_EQ(true, gt(7200000, 3600000)); // 02:00:00 > 01:00:00

  // test greater than or equal
  EXPECT_EQ(true, gte(0, 0));
  EXPECT_EQ(false, gte(0, 1000));
  EXPECT_EQ(true, gte(1000, 0));

  // test between
  EXPECT_EQ(true, between(1000, 0, 2000));
  EXPECT_EQ(true, between(0, 0, 2000)); // inclusive lower
  EXPECT_EQ(true, between(2000, 0, 2000)); // inclusive upper
  EXPECT_EQ(false, between(3000, 0, 2000));
  EXPECT_EQ(false, between(0, 1000, 2000));
  EXPECT_EQ(
      true,
      between(
          3600000, 3600000, 7200000)); // 01:00:00 between 01:00:00 and 02:00:00
  EXPECT_EQ(std::nullopt, between(std::nullopt, 0, 1000));
  EXPECT_EQ(std::nullopt, between(500, std::nullopt, 1000));
  EXPECT_EQ(std::nullopt, between(500, 0, std::nullopt));
}

TEST_F(DateTimeFunctionsTest, timeWithTimezoneComparisons) {
  const auto eq = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 = c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto neq = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 != c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto lt = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 < c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto lte = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 <= c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto gt = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 > c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto gte = [&](std::optional<int64_t> a, std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 >= c1", {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()}, a, b);
  };
  const auto between = [&](std::optional<int64_t> value,
                           std::optional<int64_t> lower,
                           std::optional<int64_t> upper) {
    return evaluateOnce<bool>(
        "c0 between c1 and c2",
        {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()},
        value,
        lower,
        upper);
  };

  // test distinct_from - unlike regular comparison operators,
  // distinct_from treats NULLs as values that can be compared
  const auto distinctFrom = [&](std::optional<int64_t> a,
                                std::optional<int64_t> b) {
    return evaluateOnce<bool>(
        "c0 is distinct from c1",
        {TIME_WITH_TIME_ZONE(), TIME_WITH_TIME_ZONE()},
        a,
        b);
  };

  // Helper to parse TIME WITH TIME ZONE from string
  const auto parse = [](const std::string& timeStr) -> std::optional<int64_t> {
    auto result =
        util::fromTimeWithTimezoneString(timeStr.c_str(), timeStr.size());
    if (result.hasError()) {
      throw std::runtime_error("Parse error: " + result.error().message());
    }
    return result.value();
  };

  // test equality - same UTC time in different timezones should be equal
  EXPECT_EQ(true, eq(parse("00:00:00.000+00:00"), parse("01:00:00.000+01:00")));
  EXPECT_EQ(true, eq(parse("12:30:45.123+00:00"), parse("20:30:45.123+08:00")));
  EXPECT_EQ(
      false, eq(parse("00:00:00.000+00:00"), parse("00:00:01.000+00:00")));
  EXPECT_EQ(std::nullopt, eq(std::nullopt, parse("00:00:00.000+00:00")));

  // test inequality
  EXPECT_EQ(
      true, neq(parse("00:00:00.000+00:00"), parse("00:00:00.000+01:00")));
  EXPECT_EQ(
      true, neq(parse("00:00:00.000+00:00"), parse("00:00:01.000+00:00")));
  EXPECT_EQ(
      true, neq(parse("01:00:00.000+01:00"), parse("01:00:00.000+03:00")));

  // test less than
  EXPECT_EQ(
      false, lt(parse("00:00:00.000+00:00"), parse("00:00:00.000+01:00")));
  EXPECT_EQ(true, lt(parse("00:00:01.000+01:00"), parse("00:00:00.000+00:00")));
  EXPECT_EQ(
      false, lt(parse("00:00:01.000+00:00"), parse("00:00:00.000+01:00")));
  EXPECT_EQ(
      false, lt(parse("01:00:00.000+01:00"), parse("02:00:00.000+03:00")));

  // test less than or equal
  EXPECT_EQ(
      false, lte(parse("00:00:00.000+00:00"), parse("00:00:00.000+01:00")));
  EXPECT_EQ(
      false, lte(parse("00:00:01.000+00:00"), parse("00:00:00.000+01:00")));

  // test greater than
  EXPECT_EQ(true, gt(parse("00:00:00.000+00:00"), parse("00:00:00.000+01:00")));
  EXPECT_EQ(
      false, gt(parse("01:00:00.000+03:00"), parse("02:00:00.000+01:00")));

  // test greater than or equal
  EXPECT_EQ(
      true, gte(parse("00:00:00.000+00:00"), parse("00:00:00.000+00:00")));
  EXPECT_EQ(
      true, gte(parse("00:00:00.000+00:00"), parse("00:00:01.000+01:00")));
  EXPECT_EQ(
      true, gte(parse("00:00:01.000+00:00"), parse("00:00:00.000+01:00")));

  // test between
  EXPECT_EQ(
      true,
      between(
          parse("02:00:00.000+02:00"),
          parse("00:00:00.000+01:00"),
          parse("00:00:01.000+00:00")));
  EXPECT_EQ(
      std::nullopt,
      between(
          parse("00:00:00.500+00:00"),
          parse("00:00:00.000+01:00"),
          std::nullopt));

  // test distinct from
  // Same UTC time in different timezones should not be distinct
  EXPECT_EQ(
      false,
      distinctFrom(parse("01:00:00.000+01:00"), parse("02:00:00.000+02:00")));

  // Different times should be distinct
  EXPECT_EQ(
      true,
      distinctFrom(parse("00:00:00.000+00:00"), parse("00:00:01.000+00:00")));
  EXPECT_EQ(
      true,
      distinctFrom(parse("01:00:00.000+01:00"), parse("02:00:00.000+03:00")));

  // NULL handling: NULL is distinct from non-NULL
  EXPECT_EQ(true, distinctFrom(std::nullopt, parse("00:00:00.000+00:00")));

  // NULL is NOT distinct from NULL
  EXPECT_EQ(false, distinctFrom(std::nullopt, std::nullopt));

  // Test edge cases with day wrap-around (normalization logic)
  EXPECT_EQ(
      false, eq(parse("01:00:00.000+08:00"), parse("02:00:00.000+00:00")));
  EXPECT_EQ(
      false, gt(parse("01:00:00.000+08:00"), parse("02:00:00.000+00:00")));

  EXPECT_EQ(
      false, eq(parse("23:00:00.000-08:00"), parse("16:00:00.000+08:00")));
  EXPECT_EQ(true, lt(parse("23:00:00.000-08:00"), parse("18:00:00.000-14:00")));

  EXPECT_EQ(
      false, eq(parse("00:30:00.000-02:00"), parse("00:30:00.000+02:00")));
  EXPECT_EQ(true, gt(parse("00:30:00.000-02:00"), parse("00:30:00.000+02:00")));
}

TEST_F(DateTimeFunctionsTest, castDateToTimestamp) {
  const int64_t kSecondsInDay = kMillisInDay / 1'000;
  const auto castDateToTimestamp = [&](const std::optional<int32_t> date) {
    return evaluateOnce<Timestamp>("cast(c0 AS timestamp)", DATE(), date);
  };

  EXPECT_EQ(Timestamp(0, 0), castDateToTimestamp(parseDate("1970-01-01")));
  EXPECT_EQ(
      Timestamp(kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1970-01-02")));
  EXPECT_EQ(
      Timestamp(2 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1970-01-03")));
  EXPECT_EQ(
      Timestamp(18297 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("2020-02-05")));
  EXPECT_EQ(
      Timestamp(-1 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1969-12-31")));
  EXPECT_EQ(
      Timestamp(-18297 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1919-11-28")));

  const auto tz = "America/Los_Angeles";
  const auto kTimezoneOffset = 8 * kMillisInHour / 1'000;
  setQueryTimeZone(tz);
  EXPECT_EQ(
      Timestamp(kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("1970-01-01")));
  EXPECT_EQ(
      Timestamp(kSecondsInDay + kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("1970-01-02")));
  EXPECT_EQ(
      Timestamp(2 * kSecondsInDay + kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("1970-01-03")));
  EXPECT_EQ(
      Timestamp(18297 * kSecondsInDay + kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("2020-02-05")));
  EXPECT_EQ(
      Timestamp(-1 * kSecondsInDay + kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("1969-12-31")));
  EXPECT_EQ(
      Timestamp(-18297 * kSecondsInDay + kTimezoneOffset, 0),
      castDateToTimestamp(parseDate("1919-11-28")));

  disableAdjustTimestampToTimezone();
  EXPECT_EQ(Timestamp(0, 0), castDateToTimestamp(parseDate("1970-01-01")));
  EXPECT_EQ(
      Timestamp(kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1970-01-02")));
  EXPECT_EQ(
      Timestamp(2 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1970-01-03")));
  EXPECT_EQ(
      Timestamp(18297 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("2020-02-05")));
  EXPECT_EQ(
      Timestamp(-1 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1969-12-31")));
  EXPECT_EQ(
      Timestamp(-18297 * kSecondsInDay, 0),
      castDateToTimestamp(parseDate("1919-11-28")));
}

TEST_F(DateTimeFunctionsTest, lastDayOfMonthDate) {
  const auto lastDayFunc = [&](const std::optional<int32_t> date) {
    return evaluateOnce<int32_t>("last_day_of_month(c0)", DATE(), date);
  };

  const auto lastDay = [&](const StringView& dateStr) {
    return lastDayFunc(parseDate(dateStr));
  };

  EXPECT_EQ(std::nullopt, lastDayFunc(std::nullopt));
  EXPECT_EQ(parseDate("1970-01-31"), lastDay("1970-01-01"));
  EXPECT_EQ(parseDate("2008-02-29"), lastDay("2008-02-01"));
  EXPECT_EQ(parseDate("2023-02-28"), lastDay("2023-02-01"));
  EXPECT_EQ(parseDate("2023-02-28"), lastDay("2023-02-01"));
  EXPECT_EQ(parseDate("2023-03-31"), lastDay("2023-03-11"));
  EXPECT_EQ(parseDate("2023-04-30"), lastDay("2023-04-21"));
  EXPECT_EQ(parseDate("2023-05-31"), lastDay("2023-05-09"));
  EXPECT_EQ(parseDate("2023-06-30"), lastDay("2023-06-01"));
  EXPECT_EQ(parseDate("2023-07-31"), lastDay("2023-07-31"));
  EXPECT_EQ(parseDate("2023-07-31"), lastDay("2023-07-31"));
  EXPECT_EQ(parseDate("2023-07-31"), lastDay("2023-07-11"));
  EXPECT_EQ(parseDate("2023-08-31"), lastDay("2023-08-01"));
  EXPECT_EQ(parseDate("2023-09-30"), lastDay("2023-09-09"));
  EXPECT_EQ(parseDate("2023-10-31"), lastDay("2023-10-01"));
  EXPECT_EQ(parseDate("2023-11-30"), lastDay("2023-11-11"));
  EXPECT_EQ(parseDate("2023-12-31"), lastDay("2023-12-12"));
}

TEST_F(DateTimeFunctionsTest, lastDayOfMonthTimestamp) {
  const auto lastDayFunc = [&](const std::optional<Timestamp>& date) {
    return evaluateOnce<int32_t>("last_day_of_month(c0)", date);
  };

  const auto lastDay = [&](const StringView& dateStr) {
    return lastDayFunc(parseTimestamp(dateStr));
  };

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, lastDayFunc(std::nullopt));
  EXPECT_EQ(parseDate("1970-01-31"), lastDay("1970-01-01 20:23:00.007"));
  EXPECT_EQ(parseDate("1970-01-31"), lastDay("1970-01-01 12:00:00.001"));
  EXPECT_EQ(parseDate("2008-02-29"), lastDay("2008-02-01 12:00:00"));
  EXPECT_EQ(parseDate("2023-02-28"), lastDay("2023-02-01 23:59:59.999"));
  EXPECT_EQ(parseDate("2023-02-28"), lastDay("2023-02-01 12:00:00"));
  EXPECT_EQ(parseDate("2023-03-31"), lastDay("2023-03-11 12:00:00"));
}

TEST_F(DateTimeFunctionsTest, lastDayOfMonthTimestampWithTimezone) {
  const auto lastDayOfMonthTimestampWithTimezone =
      [&](std::optional<TimestampWithTimezone> timestampWithTimezone) {
        return evaluateOnce<int32_t>(
            "last_day_of_month(c0)",
            TIMESTAMP_WITH_TIME_ZONE(),
            TimestampWithTimezone::pack(timestampWithTimezone));
      };
  EXPECT_EQ(
      parseDate("1970-01-31"),
      lastDayOfMonthTimestampWithTimezone(TimestampWithTimezone(0, "+00:00")));
  EXPECT_EQ(
      parseDate("1969-12-31"),
      lastDayOfMonthTimestampWithTimezone(TimestampWithTimezone(0, "-02:00")));
  EXPECT_EQ(
      parseDate("2008-02-29"),
      lastDayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(1201881600000, "+02:00")));
  EXPECT_EQ(
      parseDate("2008-01-31"),
      lastDayOfMonthTimestampWithTimezone(
          TimestampWithTimezone(1201795200000, "-02:00")));
}

TEST_F(DateTimeFunctionsTest, fromUnixtimeDouble) {
  auto input = makeFlatVector<double>(
      {1623748302.,
       1623748302.0,
       1623748302.02,
       1623748302.023,
       1623748303.123,
       1623748304.009,
       1623748304.001,
       1623748304.999,
       1623748304.001290,
       1623748304.001890,
       1623748304.999390,
       1623748304.999590});
  auto actual =
      evaluate("cast(from_unixtime(c0) as varchar)", makeRowVector({input}));
  auto expected = makeFlatVector<StringView>({
      "2021-06-15 09:11:42.000",
      "2021-06-15 09:11:42.000",
      "2021-06-15 09:11:42.020",
      "2021-06-15 09:11:42.023",
      "2021-06-15 09:11:43.123",
      "2021-06-15 09:11:44.009",
      "2021-06-15 09:11:44.001",
      "2021-06-15 09:11:44.999",
      "2021-06-15 09:11:44.001",
      "2021-06-15 09:11:44.002",
      "2021-06-15 09:11:44.999",
      "2021-06-15 09:11:45.000",
  });
  assertEqualVectors(expected, actual);
}

TEST_F(DateTimeFunctionsTest, toISO8601Date) {
  const auto toISO8601 = [&](const char* dateString) {
    return evaluateOnce<std::string>(
        "to_iso8601(c0)", DATE(), std::make_optional(parseDate(dateString)));
  };

  EXPECT_EQ("1970-01-01", toISO8601("1970-01-01"));
  EXPECT_EQ("2020-02-05", toISO8601("2020-02-05"));
  EXPECT_EQ("1919-11-28", toISO8601("1919-11-28"));
  EXPECT_EQ("4653-07-01", toISO8601("4653-07-01"));
  EXPECT_EQ("1844-10-14", toISO8601("1844-10-14"));
  EXPECT_EQ("0001-01-01", toISO8601("1-01-01"));
  EXPECT_EQ("9999-12-31", toISO8601("9999-12-31"));
  EXPECT_EQ("872343-04-19", toISO8601("872343-04-19"));
  EXPECT_EQ("-3492-10-05", toISO8601("-3492-10-05"));
  EXPECT_EQ("-0653-07-12", toISO8601("-653-07-12"));
}

TEST_F(DateTimeFunctionsTest, toISO8601Timestamp) {
  const auto toIso = [&](const char* timestamp) {
    return evaluateOnce<std::string>(
        "to_iso8601(c0)", std::make_optional(parseTimestamp(timestamp)));
  };
  disableAdjustTimestampToTimezone();
  EXPECT_EQ("2024-11-01T10:00:00.000Z", toIso("2024-11-01 10:00"));
  EXPECT_EQ("2024-11-04T10:00:00.000Z", toIso("2024-11-04 10:00"));
  EXPECT_EQ("2024-11-04T15:05:34.100Z", toIso("2024-11-04 15:05:34.1"));
  EXPECT_EQ("2024-11-04T15:05:34.123Z", toIso("2024-11-04 15:05:34.123"));
  EXPECT_EQ("0022-11-01T10:00:00.000Z", toIso("22-11-01 10:00"));

  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ("2024-11-01T03:00:00.000-07:00", toIso("2024-11-01 10:00"));

  setQueryTimeZone("America/New_York");
  EXPECT_EQ("2024-11-01T06:00:00.000-04:00", toIso("2024-11-01 10:00"));
  EXPECT_EQ("2024-11-04T05:00:00.000-05:00", toIso("2024-11-04 10:00"));
  EXPECT_EQ("2024-11-04T10:05:34.100-05:00", toIso("2024-11-04 15:05:34.1"));
  EXPECT_EQ("2024-11-04T10:05:34.123-05:00", toIso("2024-11-04 15:05:34.123"));
  EXPECT_EQ("0022-11-01T05:03:58.000-04:56:02", toIso("22-11-01 10:00"));

  setQueryTimeZone("Asia/Kathmandu");
  EXPECT_EQ("2024-11-01T15:45:00.000+05:45", toIso("2024-11-01 10:00"));
  EXPECT_EQ("0022-11-01T15:41:16.000+05:41:16", toIso("22-11-01 10:00"));
  EXPECT_EQ("0022-11-01T15:41:16.000+05:41:16", toIso("22-11-01 10:00"));
}

TEST_F(DateTimeFunctionsTest, toISO8601TimestampWithTimezone) {
  const auto toIso = [&](const char* timestamp, const char* timezone) {
    const auto* timeZone = tz::locateZone(timezone);
    auto ts = parseTimestamp(timestamp);
    ts.toGMT(*timeZone);

    return evaluateOnce<std::string>(
        "to_iso8601(c0)",
        TIMESTAMP_WITH_TIME_ZONE(),
        std::make_optional(pack(ts.toMillis(), timeZone->id())));
  };

  EXPECT_EQ(
      "2024-11-01T10:00:00.000-04:00",
      toIso("2024-11-01 10:00", "America/New_York"));
  EXPECT_EQ(
      "2024-11-04T10:00:45.120-05:00",
      toIso("2024-11-04 10:00:45.12", "America/New_York"));
  EXPECT_EQ(
      "0022-11-01T10:00:00.000-04:56:02",
      toIso("22-11-01 10:00", "America/New_York"));

  EXPECT_EQ(
      "2024-11-01T10:00:00.000+05:45",
      toIso("2024-11-01 10:00", "Asia/Kathmandu"));
  EXPECT_EQ(
      "0022-11-01T10:00:00.000+05:41:16",
      toIso("22-11-01 10:00", "Asia/Kathmandu"));

  EXPECT_EQ("2024-11-01T10:00:00.000Z", toIso("2024-11-01 10:00", "UTC"));
  EXPECT_EQ("2024-11-04T10:00:45.120Z", toIso("2024-11-04 10:00:45.12", "UTC"));
  EXPECT_EQ("0022-11-01T10:00:00.000Z", toIso("22-11-01 10:00", "UTC"));
}

TEST_F(DateTimeFunctionsTest, atTimezoneTest) {
  const auto at_timezone = [&](std::optional<int64_t> timestampWithTimezone,
                               std::optional<std::string> targetTimezone) {
    return evaluateOnce<int64_t>(
        "at_timezone(c0, c1)",
        {TIMESTAMP_WITH_TIME_ZONE(), VARCHAR()},
        timestampWithTimezone,
        targetTimezone);
  };

  EXPECT_EQ(
      at_timezone(
          pack(1500101514, tz::getTimeZoneID("Asia/Kathmandu")),
          "America/Boise"),
      pack(1500101514, tz::getTimeZoneID("America/Boise")));

  EXPECT_EQ(
      at_timezone(
          pack(1500101514, tz::getTimeZoneID("America/Boise")),
          "Europe/London"),
      pack(1500101514, tz::getTimeZoneID("Europe/London")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("Canada/Yukon")),
          "Australia/Melbourne"),
      pack(1500321297, tz::getTimeZoneID("Australia/Melbourne")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("Atlantic/Bermuda")),
          "Pacific/Fiji"),
      pack(1500321297, tz::getTimeZoneID("Pacific/Fiji")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")), "UTC+8"),
      pack(1500321297, tz::getTimeZoneID("+08:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")), "GMT-7"),
      pack(1500321297, tz::getTimeZoneID("-07:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")), "UT+6"),
      pack(1500321297, tz::getTimeZoneID("+06:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")),
          "Etc/UTC-13"),
      pack(1500321297, tz::getTimeZoneID("-13:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")),
          "Etc/GMT+12"),
      pack(1500321297, tz::getTimeZoneID("-12:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("America/Los_Angeles")),
          "Etc/UT-11"),
      pack(1500321297, tz::getTimeZoneID("-11:00")));

  EXPECT_EQ(
      at_timezone(
          pack(1500321297, tz::getTimeZoneID("Atlantic/Bermuda")),
          std::nullopt),
      std::nullopt);

  EXPECT_EQ(at_timezone(std::nullopt, "Pacific/Fiji"), std::nullopt);
}

TEST_F(DateTimeFunctionsTest, atTimezoneTimeWithTimezoneTest) {
  using namespace facebook::velox::util;

  const auto at_timezone = [&](std::optional<int64_t> timeWithTimezone,
                               std::optional<std::string> targetTimezone) {
    return evaluateOnce<int64_t>(
        "at_timezone(c0, c1)",
        {TIME_WITH_TIME_ZONE(), VARCHAR()},
        timeWithTimezone,
        targetTimezone);
  };

  // Helper to create TIME WITH TIME ZONE values
  const auto makeTimeWithTz = [](const std::string& timeStr) -> int64_t {
    auto result = fromTimeWithTimezoneString(timeStr.c_str(), timeStr.size());
    if (result.hasError()) {
      throw std::runtime_error("Parse error: " + result.error().message());
    }
    return result.value();
  };

  // Test 1: Change from +05:30 to +08:00
  // Input: 10:30:00+05:30 (which is 05:00:00 UTC)
  // Output: Same UTC time (05:00:00 UTC) with +08:00 offset
  auto input1 = makeTimeWithTz("10:30:00+05:30");
  auto expected1 = makeTimeWithTz("13:00:00+08:00"); // Same UTC moment
  EXPECT_EQ(at_timezone(input1, "+08:00"), expected1);

  // Test 2: Change from -08:00 to +00:00 (UTC)
  // Input: 14:00:00-08:00 (which is 22:00:00 UTC)
  // Output: Same UTC time with +00:00 offset
  auto input2 = makeTimeWithTz("14:00:00-08:00");
  auto expected2 = makeTimeWithTz("22:00:00+00:00");
  EXPECT_EQ(at_timezone(input2, "+00:00"), expected2);

  // Test 3: Change from +00:00 to -05:00
  // Input: 12:00:00+00:00 (which is 12:00:00 UTC)
  // Output: Same UTC time with -05:00 offset
  auto input3 = makeTimeWithTz("12:00:00+00:00");
  auto expected3 = makeTimeWithTz("07:00:00-05:00");
  EXPECT_EQ(at_timezone(input3, "-05:00"), expected3);

  // Test 4: Change from +01:00 to -11:00
  // Input: 23:30:00+01:00 (which is 22:30:00 UTC)
  // Output: Same UTC time with -11:00 offset
  auto input4 = makeTimeWithTz("23:30:00+01:00");
  auto expected4 = makeTimeWithTz("11:30:00-11:00");
  EXPECT_EQ(at_timezone(input4, "-11:00"), expected4);

  // Test 5: With milliseconds - +05:30 to -08:00
  // Input: 10:30:45.123+05:30 (which is 05:00:45.123 UTC)
  // Output: Same UTC time with -08:00 offset
  auto input5 = makeTimeWithTz("10:30:45.123+05:30");
  auto expected5 = makeTimeWithTz("21:00:45.123-08:00");
  EXPECT_EQ(at_timezone(input5, "-08:00"), expected5);

  // Test 6: Different offset format - using +HH format
  auto input6 = makeTimeWithTz("15:00:00+02:00");
  auto expected6 = makeTimeWithTz("08:00:00-05:00");
  EXPECT_EQ(at_timezone(input6, "-05"), expected6);

  // Test 7: Different offset format - using +HH:mm format (Presto-compatible)
  // Note: at_timezone uses allowCompactFormat=false to match Presto behavior,
  // so we must use +HH:mm format, not +HHmm
  auto input7 = makeTimeWithTz("08:15:30+00:00");
  auto expected7 = makeTimeWithTz("13:45:30+05:30");
  EXPECT_EQ(at_timezone(input7, "+05:30"), expected7);

  // Test 8: Null input time
  EXPECT_EQ(at_timezone(std::nullopt, "+05:00"), std::nullopt);

  // Test 9: Null target timezone
  EXPECT_EQ(
      at_timezone(makeTimeWithTz("12:00:00+00:00"), std::nullopt),
      std::nullopt);

  // Test 10: Invalid timezone offset format should throw
  EXPECT_THROW(
      at_timezone(makeTimeWithTz("12:00:00+00:00"), "invalid"), VeloxUserError);

  // Test 11: Timezone offset out of valid range should throw
  EXPECT_THROW(
      at_timezone(makeTimeWithTz("12:00:00+00:00"), "+15:00"), VeloxUserError);

  EXPECT_THROW(
      at_timezone(makeTimeWithTz("12:00:00+00:00"), "-15:00"), VeloxUserError);

  // Test 12: timezone IANA Names should throw
  EXPECT_THROW(
      at_timezone(makeTimeWithTz("12:00:00+00:00"), "America/Los_Angeles"),
      VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, toMilliseconds) {
  EXPECT_EQ(
      123,
      evaluateOnce<int64_t>(
          "to_milliseconds(c0)",
          INTERVAL_DAY_TIME(),
          std::optional<int64_t>(123)));
}

TEST_F(DateTimeFunctionsTest, parseDuration) {
  const auto parseDuration = [&](std::optional<std::string> amountUnit) {
    return evaluateOnce<int64_t>(
        "parse_duration(c0)", VARCHAR(), std::move(amountUnit));
  };
  // All units
  int64_t expectedValue = 0;
  EXPECT_EQ(expectedValue, parseDuration("5.8ns"));
  expectedValue = 6;
  EXPECT_EQ(expectedValue, parseDuration("5800000.3ns"));
  expectedValue = 0;
  EXPECT_EQ(expectedValue, parseDuration("5.8us"));
  expectedValue = 5;
  EXPECT_EQ(expectedValue, parseDuration("5400.3us"));
  expectedValue = 43;
  EXPECT_EQ(expectedValue, parseDuration("42.8ms"));
  expectedValue = 5300;
  EXPECT_EQ(expectedValue, parseDuration("5.3s"));
  const int64_t hoursPerDay = 24;
  const int64_t minutesPerHour = 60;
  const int64_t secondsPerMinute = 60;
  expectedValue = 5800 * secondsPerMinute;
  EXPECT_EQ(expectedValue, parseDuration("5.8m"));
  expectedValue = 5800 * minutesPerHour * secondsPerMinute;
  EXPECT_EQ(expectedValue, parseDuration("5.8h"));
  expectedValue = 3810 * hoursPerDay * minutesPerHour * secondsPerMinute;
  EXPECT_EQ(expectedValue, parseDuration("3.81d"));
  // Blank spaces
  EXPECT_EQ(expectedValue, parseDuration(" 3.81d  "));
  EXPECT_EQ(expectedValue, parseDuration("3.81  d"));
  EXPECT_EQ(expectedValue, parseDuration(" 3.81  d  "));
  // No point
  expectedValue = 5000 * secondsPerMinute;
  EXPECT_EQ(expectedValue, parseDuration("5m"));
  // Too large
  std::string maxDoubleValue =
      std::to_string(std::numeric_limits<double>::max());
  std::string tooLargeDuration = maxDoubleValue + "s";
  VELOX_ASSERT_THROW(
      parseDuration(tooLargeDuration),
      "Value in s unit is too large to be represented in ms unit as an int64_t");
  tooLargeDuration = maxDoubleValue + "ms";
  VELOX_ASSERT_THROW(
      parseDuration(tooLargeDuration),
      "Value in ms unit is too large to be represented in ms unit as an int64_t");
  std::string outOfRangeValue = "1" + maxDoubleValue;
  std::string outOfRangeDuration = outOfRangeValue + "ms";
  VELOX_ASSERT_THROW(
      parseDuration(outOfRangeDuration),
      "Input duration value is out of range for double: " + outOfRangeValue);
  // Input format
  VELOX_ASSERT_THROW(
      parseDuration("ab.81d"),
      "Input duration is not a valid data duration string: ab.81d");
  VELOX_ASSERT_THROW(
      parseDuration(".81d"),
      "Input duration is not a valid data duration string: .81d");
  VELOX_ASSERT_THROW(
      parseDuration("3.abd"),
      "Input duration is not a valid data duration string: 3.abd");
  VELOX_ASSERT_THROW(
      parseDuration("3.d"),
      "Input duration is not a valid data duration string: 3.d");
  VELOX_ASSERT_THROW(
      parseDuration("3. d"),
      "Input duration is not a valid data duration string: 3. d");
  VELOX_ASSERT_THROW(
      parseDuration("1.23e5ms"),
      "Input duration is not a valid data duration string: 1.23e5ms");
  VELOX_ASSERT_THROW(
      parseDuration("1.23E5ms"),
      "Input duration is not a valid data duration string: 1.23E5ms");
  // Unit format
  VELOX_ASSERT_THROW(parseDuration("3.81a"), "Unknown time unit: a");
  VELOX_ASSERT_THROW(parseDuration("3.81as"), "Unknown time unit: as");
}

TEST_F(DateTimeFunctionsTest, xxHash64FunctionDate) {
  const auto xxhash64 = [&](std::optional<int32_t> date) {
    return evaluateOnce<int64_t>("xxhash64_internal(c0)", DATE(), date);
  };

  EXPECT_EQ(std::nullopt, xxhash64(std::nullopt));

  // Epoch
  EXPECT_EQ(3803688792395291579, xxhash64(parseDate("1970-01-01")));
  EXPECT_EQ(3734916545851684445, xxhash64(parseDate("2024-10-07")));
  EXPECT_EQ(1385444150471264300, xxhash64(parseDate("2025-01-10")));
  EXPECT_EQ(-6977822845260490347, xxhash64(parseDate("1970-01-02")));
  // Leap date
  EXPECT_EQ(-5306598937769828126, xxhash64(parseDate("2020-02-29")));
  // Max supported date
  EXPECT_EQ(3856043376106280085, xxhash64(parseDate("9999-12-31")));
  // Y2K
  EXPECT_EQ(-7612541860844473816, xxhash64(parseDate("2000-01-01")));
}

TEST_F(DateTimeFunctionsTest, xxHash64FunctionTimestamp) {
  const auto xxhash64 = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int64_t>(
        "xxhash64_internal(c0)", TIMESTAMP(), timestamp);
  };

  EXPECT_EQ(std::nullopt, xxhash64(std::nullopt));

  // Epoch
  EXPECT_EQ(
      3803688792395291579, xxhash64(parseTimestamp("1970-01-01 00:00:00.000")));
  EXPECT_EQ(
      -6977822845260490347,
      xxhash64(parseTimestamp("1970-01-01 00:00:00.001")));
  EXPECT_EQ(
      1480120953668144074, xxhash64(parseTimestamp("2023-05-15 12:30:45.123")));
  EXPECT_EQ(
      -204179099607410674, xxhash64(parseTimestamp("2023-05-15 12:30:45.456")));
  // Current time
  EXPECT_NE(std::nullopt, xxhash64(Timestamp::now()));
  // Future time
  EXPECT_EQ(
      2784427479311108994, xxhash64(parseTimestamp("2050-12-31 23:59:59.999")));
  // Past time
  EXPECT_EQ(
      7585368295023641328, xxhash64(parseTimestamp("1900-01-01 00:00:00.000")));
}

TEST_F(DateTimeFunctionsTest, xxHash64FunctionTime) {
  const auto xxhash64 = [&](std::optional<int64_t> time) {
    return evaluateOnce<int64_t>("xxhash64_internal(c0)", TIME(), time);
  };

  // Test NULL handling
  EXPECT_EQ(std::nullopt, xxhash64(std::nullopt));

  // Test determinism - same input should give same output
  auto result1 = xxhash64(43200000);
  auto result2 = xxhash64(43200000);
  EXPECT_EQ(result1, result2);

  // Test that different inputs give different outputs
  auto hash0 = xxhash64(0);
  auto hash1 = xxhash64(1);
  auto hashNoon = xxhash64(43200000);
  auto hashEnd = xxhash64(86399999);

  EXPECT_NE(hash0, hash1);
  EXPECT_NE(hash0, hashNoon);
  EXPECT_NE(hash0, hashEnd);
  EXPECT_NE(hashNoon, hashEnd);

  // Test boundary values don't crash
  EXPECT_TRUE(xxhash64(0).has_value());
  EXPECT_TRUE(xxhash64(86399999).has_value());

  // Test known hash values validated against Presto xxhash64
  // Query: SELECT from_big_endian_64(xxhash64(to_big_endian_64(value)))
  EXPECT_EQ(3803688792395291579, xxhash64(0)); // Midnight
  EXPECT_EQ(-6980583299780818982, xxhash64(1)); // 1 millisecond after midnight
  EXPECT_EQ(7848233046982034517, xxhash64(43200000)); // Noon (12:00:00.000)
  EXPECT_EQ(
      5892092673475229733, xxhash64(86399999)); // End of day (23:59:59.999)
  EXPECT_EQ(-3599997350390034763, xxhash64(1234)); // Arbitrary value
}

TEST_F(DateTimeFunctionsTest, currentTimestamp) {
  const auto callCurrentTimestamp =
      [&](int64_t sessionStartTime,
          const std::optional<std::string>& timeZone) {
        if (timeZone.has_value()) {
          setSessionStartTimeAndTimeZone(sessionStartTime, timeZone.value());
        } else {
          setQuerySessionStartTime(sessionStartTime);
        }

        auto rowVector = makeRowVector({});
        rowVector->resize(1);

        auto result = evaluate("current_timestamp()", rowVector);
        DecodedVector decoded(*result);
        return decoded.valueAt<int64_t>(0);
      };

  // Test without timezone
  EXPECT_THROW(
      {
        try {
          callCurrentTimestamp(0, std::nullopt);
        } catch (const VeloxException& e) {
          EXPECT_EQ(e.exceptionType(), VeloxException::Type::kUser);
          throw;
        }
      },
      VeloxException);

  // Test with timezone America/Los_Angeles
  auto laPacked = callCurrentTimestamp(1758499200000, "America/Los_Angeles");
  auto la = TimestampWithTimezone::unpack(laPacked);
  ASSERT_TRUE(la.has_value());

  EXPECT_EQ(la->timezone_->name(), "America/Los_Angeles");
  EXPECT_EQ(la->milliSeconds_, 1758499200000);
}

TEST_F(DateTimeFunctionsTest, localtime) {
  const auto localtime = [&](int64_t sessionStartTime,
                             const std::optional<std::string>& timeZone) {
    if (timeZone.has_value()) {
      setSessionStartTimeAndTimeZone(sessionStartTime, timeZone.value());
    } else {
      setQuerySessionStartTime(sessionStartTime);
    }

    auto rowVector = makeRowVector({});
    rowVector->resize(1);
    auto result = evaluate("localtime()", rowVector);
    DecodedVector decoded(*result);
    return decoded.valueAt<int64_t>(0);
  };

  auto utcVal = localtime(0, std::nullopt);
  EXPECT_EQ(utcVal, 0);

  auto localVal =
      localtime(1758499200000, "America/Los_Angeles"); // Midnight UTC
  EXPECT_EQ(localVal, 0);

  // Test during daylight saving time
  localVal = localtime(1710061200000, "America/Los_Angeles");
  EXPECT_EQ(localVal, 32400000); // 9 AM UTC
}

TEST_F(DateTimeFunctionsTest, dateDiffTime) {
  const auto dateDiff = [&](const std::string& unit,
                            std::optional<int64_t> time1,
                            std::optional<int64_t> time2) {
    return evaluateOnce<int64_t>(
        fmt::format("date_diff('{}', c0, c1)", unit),
        {TIME(), TIME()},
        time1,
        time2);
  };

  // Basic time differences - following Presto's date_diff(unit, x1, x2) = x2 -
  // x1 pattern

  // Test millisecond differences
  EXPECT_EQ(
      1000, dateDiff("millisecond", 0, 1000)); // 00:00:00.000 to 00:00:01.000
  EXPECT_EQ(
      -1000,
      dateDiff(
          "millisecond", 1000, 0)); // 00:00:01.000 to 00:00:00.000 (negative)
  EXPECT_EQ(0, dateDiff("millisecond", 1000, 1000)); // Same time
  EXPECT_EQ(
      500, dateDiff("millisecond", 500, 1000)); // 00:00:00.500 to 00:00:01.000

  // Test second differences
  EXPECT_EQ(1, dateDiff("second", 0, 1000)); // 1 second difference
  EXPECT_EQ(-1, dateDiff("second", 1000, 0)); // Negative 1 second
  EXPECT_EQ(0, dateDiff("second", 1000, 1000)); // Same time
  EXPECT_EQ(30, dateDiff("second", 15000, 45000)); // 30 seconds (15s to 45s)
  EXPECT_EQ(3661, dateDiff("second", 0, 3661000)); // 1 hour 1 minute 1 second

  // Test minute differences
  EXPECT_EQ(1, dateDiff("minute", 0, 60000)); // 1 minute (00:00 to 00:01)
  EXPECT_EQ(-1, dateDiff("minute", 60000, 0)); // Negative 1 minute
  EXPECT_EQ(0, dateDiff("minute", 60000, 60000)); // Same time
  EXPECT_EQ(5, dateDiff("minute", 0, 300000)); // 5 minutes (00:00 to 00:05)
  EXPECT_EQ(61, dateDiff("minute", 0, 3660000)); // 1 hour 1 minute

  // Test hour differences
  EXPECT_EQ(1, dateDiff("hour", 0, 3600000)); // 1 hour (00:00 to 01:00)
  EXPECT_EQ(-1, dateDiff("hour", 3600000, 0)); // Negative 1 hour
  EXPECT_EQ(0, dateDiff("hour", 3600000, 3600000)); // Same time
  EXPECT_EQ(12, dateDiff("hour", 0, 43200000)); // 12 hours (00:00 to 12:00)
  EXPECT_EQ(23, dateDiff("hour", 0, 82800000)); // 23 hours (00:00 to 23:00)

  // Test boundary values for TIME type (0 to 86399999 ms in a day)
  EXPECT_EQ(0, dateDiff("millisecond", 0, 0)); // 00:00:00.000 to 00:00:00.000
  EXPECT_EQ(
      86399999,
      dateDiff("millisecond", 0, 86399999)); // 00:00:00.000 to 23:59:59.999
  EXPECT_EQ(
      -86399999,
      dateDiff("millisecond", 86399999, 0)); // 23:59:59.999 to 00:00:00.000
  EXPECT_EQ(
      86399, dateDiff("second", 0, 86399999)); // Full day minus 1ms in seconds
  EXPECT_EQ(
      1439, dateDiff("minute", 0, 86399999)); // Full day minus 1ms in minutes
  EXPECT_EQ(23, dateDiff("hour", 0, 86399999)); // Full day minus 1ms in hours

  // Test fractional truncation behavior (consistent with Presto integer
  // division)
  EXPECT_EQ(0, dateDiff("second", 0, 999)); // 999ms < 1000ms, truncates to 0
  EXPECT_EQ(
      59, dateDiff("minute", 0, 3599999)); // 59.99999 minutes truncates to 59
  EXPECT_EQ(0, dateDiff("hour", 0, 3599999)); // 0.99999 hours truncates to 0

  // Test real-world scenario: 09:30:15.500 to 14:45:30.750
  int64_t morning = 9 * 3600000 + 30 * 60000 + 15 * 1000 + 500; // 34215500ms
  int64_t afternoon = 14 * 3600000 + 45 * 60000 + 30 * 1000 + 750; // 53130750ms

  EXPECT_EQ(18915250, dateDiff("millisecond", morning, afternoon));
  EXPECT_EQ(
      18915,
      dateDiff("second", morning, afternoon)); // 18915.25s truncated to 18915
  EXPECT_EQ(
      315,
      dateDiff(
          "minute", morning, afternoon)); // 315.254... minutes truncated to 315
  EXPECT_EQ(
      5, dateDiff("hour", morning, afternoon)); // 5.254... hours truncated to 5
  EXPECT_EQ(
      315,
      dateDiff(
          "minute", morning, afternoon)); // 315.254... minutes truncated to 315
  EXPECT_EQ(
      5, dateDiff("hour", morning, afternoon)); // 5.254... hours truncated to 5

  // Test null handling (consistent with Presto null propagation)
  EXPECT_EQ(std::nullopt, dateDiff("second", 1000, std::nullopt));
  EXPECT_EQ(std::nullopt, dateDiff("second", std::nullopt, 1000));
  EXPECT_EQ(std::nullopt, dateDiff("second", std::nullopt, std::nullopt));

  // Test invalid units (TIME only supports millisecond, second, minute, and
  // hour - microsecond is not supported in Presto)
  VELOX_ASSERT_THROW(
      dateDiff("microsecond", 0, 1000),
      "microsecond is not a valid TIME field");
  VELOX_ASSERT_THROW(dateDiff("day", 0, 1000), "day is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateDiff("week", 0, 1000), "week is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateDiff("month", 0, 1000), "month is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateDiff("quarter", 0, 1000), "quarter is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateDiff("year", 0, 1000), "year is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateDiff("invalid", 0, 1000), "invalid is not a valid TIME field");

  // Additional edge cases
  // Note: This represents going from 23:59:59.999 to 00:00:00.000 of next day,
  // but since TIME type represents time within a single day, this results in
  // negative difference
  EXPECT_EQ(
      -86399999,
      dateDiff(
          "millisecond",
          86399999,
          0)); // 23:59:59.999 to 00:00:00.000 (negative)
}

TEST_F(DateTimeFunctionsTest, dateDiffTimeVariableUnit) {
  // This test validates that when the unit parameter is variable (passed as a
  // column with different values per row), the function correctly processes
  // each row with its own unit value rather than incorrectly caching a single
  // unit. This tests batch processing with varying units in a single
  // evaluation.
  auto data = makeRowVector({
      makeFlatVector<std::string>(
          {"millisecond", "second", "minute", "hour", "millisecond"}),
      makeFlatVector<int64_t>({0, 0, 0, 0, 1000}, TIME()),
      makeFlatVector<int64_t>({1000, 1000, 60000, 3600000, 2000}, TIME()),
  });

  auto result = evaluate("date_diff(c0, c1, c2)", data);
  auto expected = makeFlatVector<int64_t>({
      1000, // millisecond: 0 to 1000
      1, // second: 0 to 1000
      1, // minute: 0 to 60000
      1, // hour: 0 to 3600000
      1000, // millisecond: 1000 to 2000
  });

  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, dateTruncTimestampVariableUnit) {
  // This test validates that when the unit parameter is variable (passed as a
  // column with different values per row), date_trunc correctly processes each
  // row with its own unit value rather than incorrectly caching a single unit.
  auto timestamps = makeFlatVector<Timestamp>({
      Timestamp(998474645, 321001234), // 2001-08-22 03:04:05.321001234
      Timestamp(998474645, 321001234), // 2001-08-22 03:04:05.321001234
      Timestamp(998474645, 321001234), // 2001-08-22 03:04:05.321001234
      Timestamp(998474645, 321001234), // 2001-08-22 03:04:05.321001234
      Timestamp(998474645, 321001234), // 2001-08-22 03:04:05.321001234
  });

  auto data = makeRowVector({
      makeFlatVector<std::string>({"second", "minute", "hour", "day", "month"}),
      timestamps,
  });

  auto result = evaluate("date_trunc(c0, c1)", data);

  auto expected = makeFlatVector<Timestamp>({
      Timestamp(998474645, 0), // second: 2001-08-22 03:04:05.000
      Timestamp(998474640, 0), // minute: 2001-08-22 03:04:00.000
      Timestamp(998474400, 0), // hour: 2001-08-22 03:00:00.000
      Timestamp(998438400, 0), // day: 2001-08-22 00:00:00.000
      Timestamp(996624000, 0), // month: 2001-08-01 00:00:00.000
  });

  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, dateTruncDateVariableUnit) {
  // This test validates that when the unit parameter is variable (passed as a
  // column with different values per row), date_trunc correctly processes each
  // row with its own unit value for DATE type.
  auto dates = makeFlatVector<int32_t>(
      {parseDate("2020-02-29"),
       parseDate("2020-02-29"),
       parseDate("2020-02-29"),
       parseDate("2020-02-29")},
      DATE());

  auto data = makeRowVector({
      makeFlatVector<std::string>({"day", "week", "month", "quarter"}),
      dates,
  });

  auto result = evaluate("date_trunc(c0, c1)", data);

  auto expected = makeFlatVector<int32_t>(
      {parseDate("2020-02-29"), // day: same
       parseDate("2020-02-24"), // week: Monday of that week
       parseDate("2020-02-01"), // month: first day of month
       parseDate("2020-01-01")}, // quarter: first day of quarter
      DATE());

  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, dateTruncTime) {
  const auto dateTrunc = [&](const std::string& unit,
                             std::optional<int64_t> time) {
    return evaluateOnce<int64_t>(
        fmt::format("date_trunc('{}', c0)", unit), TIME(), time);
  };

  EXPECT_EQ(std::nullopt, dateTrunc("second", std::nullopt));

  // TIME value: 03:04:05.321 = (3*3600 + 4*60 + 5)*1000 + 321 = 11045321 ms
  const int64_t time1 = 11045321;

  EXPECT_EQ(11045321, dateTrunc("millisecond", time1)); // no change
  EXPECT_EQ(11045000, dateTrunc("second", time1)); // 03:04:05.000
  EXPECT_EQ(11040000, dateTrunc("minute", time1)); // 03:04:00.000
  EXPECT_EQ(10800000, dateTrunc("hour", time1)); // 03:00:00.000

  // TIME value: 00:00:00.000 = 0 ms (midnight)
  EXPECT_EQ(0, dateTrunc("millisecond", 0));
  EXPECT_EQ(0, dateTrunc("second", 0));
  EXPECT_EQ(0, dateTrunc("minute", 0));
  EXPECT_EQ(0, dateTrunc("hour", 0));

  // TIME value: 23:59:59.999 = (23*3600 + 59*60 + 59)*1000 + 999 = 86399999 ms
  const int64_t time2 = 86399999;
  EXPECT_EQ(86399999, dateTrunc("millisecond", time2)); // no change
  EXPECT_EQ(86399000, dateTrunc("second", time2)); // 23:59:59.000
  EXPECT_EQ(86340000, dateTrunc("minute", time2)); // 23:59:00.000
  EXPECT_EQ(82800000, dateTrunc("hour", time2)); // 23:00:00.000

  // TIME value: 12:30:45.123 = (12*3600 + 30*60 + 45)*1000 + 123 = 45045123 ms
  const int64_t time3 = 45045123;
  EXPECT_EQ(45045123, dateTrunc("millisecond", time3)); // no change
  EXPECT_EQ(45045000, dateTrunc("second", time3)); // 12:30:45.000
  EXPECT_EQ(45000000, dateTrunc("minute", time3)); // 12:30:00.000
  EXPECT_EQ(43200000, dateTrunc("hour", time3)); // 12:00:00.000

  // Invalid unit tests - TIME only supports millisecond, second, minute, and
  // hour (microsecond is not supported in Presto)
  VELOX_ASSERT_THROW(
      dateTrunc("microsecond", time1), "microsecond is not a valid TIME field");
  VELOX_ASSERT_THROW(dateTrunc("day", time1), "day is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateTrunc("week", time1), "week is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateTrunc("month", time1), "month is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateTrunc("quarter", time1), "quarter is not a valid TIME field");
  VELOX_ASSERT_THROW(
      dateTrunc("year", time1), "year is not a valid TIME field");
}

TEST_F(DateTimeFunctionsTest, dateAddTimestampVariableUnit) {
  // This test validates that when the unit parameter is variable (passed as a
  // column with different values per row), date_add correctly processes each
  // row with its own unit value.
  auto timestamps = makeFlatVector<Timestamp>({
      Timestamp(0, 0), // 1970-01-01 00:00:00.000
      Timestamp(0, 0), // 1970-01-01 00:00:00.000
      Timestamp(0, 0), // 1970-01-01 00:00:00.000
      Timestamp(0, 0), // 1970-01-01 00:00:00.000
      Timestamp(0, 0), // 1970-01-01 00:00:00.000
  });

  auto data = makeRowVector({
      makeFlatVector<std::string>({"second", "minute", "hour", "day", "month"}),
      makeFlatVector<int64_t>({1, 1, 1, 1, 1}),
      timestamps,
  });

  auto result = evaluate("date_add(c0, c1, c2)", data);

  auto expected = makeFlatVector<Timestamp>({
      Timestamp(1, 0), // second: 1970-01-01 00:00:01.000
      Timestamp(60, 0), // minute: 1970-01-01 00:01:00.000
      Timestamp(3600, 0), // hour: 1970-01-01 01:00:00.000
      Timestamp(86400, 0), // day: 1970-01-02 00:00:00.000
      Timestamp(2678400, 0), // month: 1970-02-01 00:00:00.000
  });

  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, dateAddDateVariableUnit) {
  // This test validates that when the unit parameter is variable (passed as a
  // column with different values per row), date_add correctly processes each
  // row with its own unit value for DATE type.
  auto dates = makeFlatVector<int32_t>(
      {parseDate("2020-01-31"),
       parseDate("2020-01-31"),
       parseDate("2020-01-31"),
       parseDate("2020-01-31")},
      DATE());

  auto data = makeRowVector({
      makeFlatVector<std::string>({"day", "week", "month", "year"}),
      makeFlatVector<int64_t>({1, 1, 1, 1}),
      dates,
  });

  auto result = evaluate("date_add(c0, c1, c2)", data);

  auto expected = makeFlatVector<int32_t>(
      {parseDate("2020-02-01"), // day: 2020-01-31 + 1 day
       parseDate("2020-02-07"), // week: 2020-01-31 + 7 days
       parseDate("2020-02-29"), // month: 2020-01-31 + 1 month (leap year)
       parseDate("2021-01-31")}, // year: 2020-01-31 + 1 year
      DATE());

  assertEqualVectors(expected, result);
}

TEST_F(DateTimeFunctionsTest, currentTimezone) {
  {
    setQueryTimeZone("Asia/Kolkata");
    auto tz = evaluateOnce<std::string>(
        "current_timezone()", makeRowVector(ROW({}), 1));
    ASSERT_TRUE(tz.has_value());
    EXPECT_EQ(tz.value(), "Asia/Kolkata");
  }

  {
    setQueryTimeZone("America/New_York");
    auto tz = evaluateOnce<std::string>(
        "current_timezone()", makeRowVector(ROW({}), 1));
    ASSERT_TRUE(tz.has_value());
    EXPECT_EQ(tz.value(), "America/New_York");
  }
}
