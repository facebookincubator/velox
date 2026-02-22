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

#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Time.h"

using facebook::velox::util::biasEncode;
using facebook::velox::util::pack;

namespace facebook::velox {
namespace {

inline int64_t getUTCMillis(
    int64_t hours,
    int64_t minutes = 0,
    int64_t seconds = 0,
    int64_t millis = 0) {
  return hours * util::kMillisInHour + minutes * util::kMillisInMinute +
      seconds * util::kMillisInSecond + millis;
}

class TimeWithTimezoneCastTest : public functions::test::CastBaseTest {
 protected:
  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  void setQueryTimeZoneAndStartTime(
      const std::string& timeZone,
      int64_t startTimeMs) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
        {core::QueryConfig::kSessionStartTime, std::to_string(startTimeMs)},
    });
  }

  void disableAdjustTimestampToTimezone() {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
    });
  }

  // Helper to pack TIME WITH TIME ZONE value
  int64_t packTimeWithTZ(int64_t timeMillis, int16_t offsetMinutes) {
    auto encodedOffset = biasEncode(offsetMinutes);
    return pack(timeMillis, encodedOffset);
  }
};

TEST_F(TimeWithTimezoneCastTest, toVarchar) {
  // Test comprehensive scenarios: various times, timezones, boundary values,
  // and nulls. Note: Time is stored in UTC, so when displaying, we convert
  // UTC time to local time by adding the timezone offset.
  auto input = makeNullableFlatVector<int64_t>(
      {
          // Constant values to test any regressions
          // in pack and biasEncode functions.
          58755883008,
          // 11:31:37.123 UTC with timezone +05:30 -> displays as 17:01:37.123
          169972216978,
          // 11:31:37.123 UTC with timezone -05:00 -> displays as 06:31:37.123
          169972216349,
          // UTC time 06:11:37.123 with UTC timezone -> displays as
          // 06:11:37.123+00:00
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(0)),
          // UTC time 06:11:37.123 with -05:00 timezone -> displays as
          // 01:11:37.123-05:00
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(-5 * 60)),
          // UTC time 06:11:37.123 with +05:30 timezone -> displays as
          // 11:41:37.123+05:30
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(5 * 60 + 30)),
          // Boundary: Midnight UTC with UTC timezone
          pack(0, biasEncode(0)),
          // Boundary: Almost end of day UTC with UTC timezone
          pack(getUTCMillis(23, 59, 59, 999), biasEncode(0)),
          // Null value.
          std::nullopt,
          // UTC time 12:00:00 with +14:00 timezone -> displays as
          // 02:00:00.000+14:00 (next day wrap)
          pack(getUTCMillis(12), biasEncode(14 * 60)),
          // UTC time 12:00:00 with -14:00 timezone -> displays as
          // 22:00:00.000-14:00 (prev day wrap)
          pack(getUTCMillis(12), biasEncode(-14 * 60)),
          // Edge case: 1 millisecond after midnight UTC with UTC timezone
          pack(1, biasEncode(0)),
          // UTC time 12:34:56.789 with -04:30 timezone -> displays as
          // 08:04:56.789-04:30
          pack(getUTCMillis(12, 34, 56, 789), biasEncode(-4 * 60 - 30)),
          // UTC time 09:08:07 with +05:45 timezone -> displays as
          // 14:53:07.000+05:45
          pack(getUTCMillis(9, 8, 7), biasEncode(5 * 60 + 45)),
          // Another null value.
          std::nullopt,
      },
      TIME_WITH_TIME_ZONE());

  auto expected = makeNullableFlatVector<std::string>({
      "03:59:04.698+00:00",
      "17:01:37.123+05:30",
      "06:31:37.123-05:00",
      "06:11:37.123+00:00",
      "01:11:37.123-05:00",
      "11:41:37.123+05:30",
      "00:00:00.000+00:00",
      "23:59:59.999+00:00",
      std::nullopt,
      "02:00:00.000+14:00",
      "22:00:00.000-14:00",
      "00:00:00.001+00:00",
      "08:04:56.789-04:30",
      "14:53:07.000+05:45",
      std::nullopt,
  });

  testCast(input, expected);
}

TEST_F(TimeWithTimezoneCastTest, fromVarchar) {
  // Test casting VARCHAR to TIME WITH TIME ZONE with various formats.
  // Note: Input strings represent local time, which gets converted to UTC for
  // storage.
  auto input = makeNullableFlatVector<std::string>({
      // Basic format: H:m:s.SSS+HH:mm
      "6:11:37.123+00:00",
      "6:11:37.123-05:00", // Local 06:11:37.123 in UTC-5 = 11:11:37.123 UTC
      "6:11:37.123+05:30", // Local 06:11:37.123 in UTC+5:30 = 00:41:37.123 UTC
      // Format without milliseconds: H:m:s+HH:mm
      "12:34:56+01:00", // Local 12:34:56 in UTC+1 = 11:34:56 UTC
      "12:34:56-08:00", // Local 12:34:56 in UTC-8 = 20:34:56 UTC
      // Compact format: H:m+HH:mm
      "1:30+05:30", // Local 01:30 in UTC+5:30 = 20:00 UTC (prev day)
      "23:45-07:00", // Local 23:45 in UTC-7 = 06:45 UTC (next day)
      // Format with space before timezone: H:m:s.SSS +HH:mm
      "14:22:11.456 +02:00", // Local 14:22:11.456 in UTC+2 = 12:22:11.456 UTC
      "9:15:30.789 -04:30", // Local 09:15:30.789 in UTC-4:30 = 13:45:30.789 UTC
      // Compact timezone format (no colon): H:m:s+HHmm
      "3:4:5+0709", // Local 03:04:05 in UTC+7:09 = 19:55:05 UTC (prev day)
      "15:45:30-0800", // Local 15:45:30 in UTC-8 = 23:45:30 UTC
      // Short timezone format: H:m:s+HH
      "8:30:00+05", // Local 08:30:00 in UTC+5 = 03:30:00 UTC
      "16:45:30-08", // Local 16:45:30 in UTC-8 = 00:45:30 UTC (next day)
      // Boundary: Midnight with UTC
      "0:0:0.000+00:00",
      // Boundary: Almost end of day
      "23:59:59.999+00:00",
      // Boundary: Maximum positive timezone offset (+14:00)
      "12:00:00.000+14:00", // Local 12:00 in UTC+14 = 22:00 UTC (prev day)
      // Boundary: Maximum negative timezone offset (-14:00)
      "12:00:00.000-14:00", // Local 12:00 in UTC-14 = 02:00 UTC (next day)
      // Edge case: 1 millisecond after midnight
      "0:0:0.001+00:00",
      // Various valid formats with different hour/minute combinations
      "9:8:7.000+05:45", // Local 09:08:07 in UTC+5:45 = 03:23:07 UTC
      "1:2:3+01:30", // Local 01:02:03 in UTC+1:30 = 23:32:03 UTC (prev day)
      // Null value
      std::nullopt,
      // Compact format with space: H:m +HH
      "7:30 +03", // Local 07:30 in UTC+3 = 04:30 UTC
      "18:45 -06", // Local 18:45 in UTC-6 = 00:45 UTC (next day)
      // Double-digit hours and minutes: HH:mm:ss.SSS+HH:mm
      "06:11:37.123+00:00",
      "12:34:56.789-04:30", // Local 12:34:56.789 in UTC-4:30 = 17:04:56.789 UTC
      "03:59:04.698+00:00",
      "17:01:37.123+05:30",
      "06:31:37.123-05:00",
  });

  auto expected = makeNullableFlatVector<int64_t>(
      {
          // "6:11:37.123+00:00" -> UTC 06:11:37.123
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(0)),
          // "6:11:37.123-05:00" -> UTC 11:11:37.123
          pack(getUTCMillis(11, 11, 37, 123), biasEncode(-5 * 60)),
          // "6:11:37.123+05:30" -> UTC 00:41:37.123
          pack(getUTCMillis(0, 41, 37, 123), biasEncode(5 * 60 + 30)),
          // "12:34:56+01:00" -> UTC 11:34:56
          pack(getUTCMillis(11, 34, 56), biasEncode(1 * 60)),
          // "12:34:56-08:00" -> UTC 20:34:56
          pack(getUTCMillis(20, 34, 56), biasEncode(-8 * 60)),
          // "1:30+05:30" -> UTC 20:00:00 (prev day wraps to 20:00)
          pack(getUTCMillis(20, 0), biasEncode(5 * 60 + 30)),
          // "23:45-07:00" -> UTC 06:45:00 (next day wraps to 06:45)
          pack(getUTCMillis(6, 45), biasEncode(-7 * 60)),
          // "14:22:11.456 +02:00" -> UTC 12:22:11.456
          pack(getUTCMillis(12, 22, 11, 456), biasEncode(2 * 60)),
          // "9:15:30.789 -04:30" -> UTC 13:45:30.789
          pack(getUTCMillis(13, 45, 30, 789), biasEncode(-4 * 60 - 30)),
          // "3:4:5+0709" -> UTC 19:55:05 (prev day wraps to 19:55:05)
          pack(getUTCMillis(19, 55, 5), biasEncode(7 * 60 + 9)),
          // "15:45:30-0800" -> UTC 23:45:30
          pack(getUTCMillis(23, 45, 30), biasEncode(-8 * 60)),
          // "8:30:00+05" -> UTC 03:30:00
          pack(getUTCMillis(3, 30), biasEncode(5 * 60)),
          // "16:45:30-08" -> UTC 00:45:30 (next day wraps to 00:45:30)
          pack(getUTCMillis(0, 45, 30), biasEncode(-8 * 60)),
          // "0:0:0.000+00:00" -> UTC 00:00:00.000
          pack(0, biasEncode(0)),
          // "23:59:59.999+00:00" -> UTC 23:59:59.999
          pack(getUTCMillis(23, 59, 59, 999), biasEncode(0)),
          // "12:00:00.000+14:00" -> UTC 22:00:00 (prev day wraps to 22:00)
          pack(getUTCMillis(22, 0, 0), biasEncode(14 * 60)),
          // "12:00:00.000-14:00" -> UTC 02:00:00 (next day wraps to 02:00)
          pack(getUTCMillis(2, 0, 0), biasEncode(-14 * 60)),
          // "0:0:0.001+00:00" -> UTC 00:00:00.001
          pack(1, biasEncode(0)),
          // "9:8:7.000+05:45" -> UTC 03:23:07
          pack(getUTCMillis(3, 23, 7), biasEncode(5 * 60 + 45)),
          // "1:2:3+01:30" -> UTC 23:32:03 (prev day wraps to 23:32:03)
          pack(getUTCMillis(23, 32, 3), biasEncode(1 * 60 + 30)),
          // Null value
          std::nullopt,
          // "7:30 +03" -> UTC 04:30
          pack(getUTCMillis(4, 30), biasEncode(3 * 60)),
          // "18:45 -06" -> UTC 00:45 (next day wraps to 00:45)
          pack(getUTCMillis(0, 45), biasEncode(-6 * 60)),
          // "06:11:37.123+00:00" -> UTC 06:11:37.123
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(0)),
          // "12:34:56.789-04:30" -> UTC 17:04:56.789
          pack(getUTCMillis(17, 4, 56, 789), biasEncode(-4 * 60 - 30)),
          // Constant values to test any regressions
          // in pack and biasEncode functions.
          58755883008,
          // 11:31:37.123 UTC with timezone +05:30 -> displays as 17:01:37.123
          169972216978,
          // 11:31:37.123 UTC with timezone -05:00 -> displays as 06:31:37.123
          169972216349,
      },
      TIME_WITH_TIME_ZONE());

  testCast(input, expected);
}

TEST_F(TimeWithTimezoneCastTest, fromVarcharInvalid) {
  // Test invalid VARCHAR inputs that should throw errors.
  auto testInvalidCast = [&](const std::string& input,
                             const std::string& expectedError) {
    VELOX_ASSERT_THROW(
        evaluateCast(
            VARCHAR(),
            TIME_WITH_TIME_ZONE(),
            makeRowVector({makeFlatVector<std::string>({input})})),
        expectedError);
  };

  // Invalid: Missing timezone offset
  testInvalidCast("12:34:56", "missing timezone offset");

  // Invalid: Hour out of range (25 > 23)
  testInvalidCast("25:00:00+00:00", "Invalid hour value");

  // Invalid: Minute out of range (60 >= 60)
  testInvalidCast("12:60:00+00:00", "Invalid minute value");

  // Invalid: Second out of range (60 >= 60)
  testInvalidCast("12:34:60+00:00", "Invalid second value");

  // Invalid: Microsecond precision not supported (4 digits after decimal)
  testInvalidCast("12:34:56.1234+00:00", "Microsecond precision not supported");

  // Invalid: Timezone offset out of range (+15:00 > +14:00)
  testInvalidCast("12:00:00+15:00", "out of range");

  // Invalid: Timezone offset out of range (-15:00 < -14:00)
  testInvalidCast("12:00:00-15:00", "out of range");

  // Invalid: Missing time component
  testInvalidCast("+05:00", "missing time component");

  // Invalid: Empty string
  testInvalidCast("", "empty string");

  // Invalid: Malformed format (missing colon in time)
  testInvalidCast("1234+00:00", "expected ':' after hour");

  // Invalid: Trailing characters after timezone
  testInvalidCast("12:34:56+00:00extra", "unexpected trailing characters");

  // Invalid: Number of digits in timezone offset hours is not 2
  testInvalidCast("12:34:56+1:00", "digits before ':' not equal to 2");

  // Invalid: Number of digits in timezone offset hours is not 2
  testInvalidCast("12:34:56+01:", "failed to parse minutes after ':'");

  // Invalid: IANA/Named timezones not supported (e.g., America/Los_Angeles)
  testInvalidCast("12:34:56 America/Los_Angeles", "missing timezone offset");

  // Invalid: IANA/Named timezones not supported (e.g., UTC)
  testInvalidCast("14:30:45 UTC", "missing timezone offset");

  // Invalid: IANA/Named timezones not supported (e.g., Asia/Tokyo)
  testInvalidCast("09:15:30.123 Asia/Tokyo", "missing timezone offset");
}

TEST_F(TimeWithTimezoneCastTest, toTime) {
  // Test casting TIME WITH TIME ZONE to TIME extracts the local time component
  // (UTC time converted to local time by adding the timezone offset),
  // discarding the timezone information.
  auto input = makeNullableFlatVector<int64_t>(
      {
          // UTC 06:11:37.123 with +00:00 -> Local 06:11:37.123
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(0)),
          // UTC 06:11:37.123 with -05:00 -> Local 01:11:37.123
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(-5 * 60)),
          // UTC 06:11:37.123 with +05:30 -> Local 11:41:37.123
          pack(getUTCMillis(6, 11, 37, 123), biasEncode(5 * 60 + 30)),
          // Boundary: Midnight UTC with +00:00 -> Local 00:00:00.000
          pack(0, biasEncode(0)),
          // Boundary: Almost end of day UTC with +00:00 -> Local 23:59:59.999
          pack(getUTCMillis(23, 59, 59, 999), biasEncode(0)),
          // Null value.
          std::nullopt,
          // UTC 12:30:00 with +14:00 -> Local 02:30:00 (next day wrap)
          pack(getUTCMillis(12, 30), biasEncode(14 * 60)),
          // UTC 12:30:00 with -14:00 -> Local 22:30:00 (prev day wrap)
          pack(getUTCMillis(12, 30), biasEncode(-14 * 60)),
          // Edge case: 1 millisecond after midnight UTC with +00:00 -> Local
          // 00:00:00.001
          pack(1, biasEncode(0)),
          // UTC 15:45:30.500 with -08:00 -> Local 07:45:30.500
          pack(getUTCMillis(15, 45, 30, 500), biasEncode(-8 * 60)),
          // Another null value.
          std::nullopt,
      },
      TIME_WITH_TIME_ZONE());

  auto expected = makeNullableFlatVector<int64_t>(
      {
          // UTC 06:11:37.123 + 0 = 06:11:37.123
          getUTCMillis(6, 11, 37, 123),
          // UTC 06:11:37.123 - 5 hours = 01:11:37.123
          getUTCMillis(1, 11, 37, 123),
          // UTC 06:11:37.123 + 5:30 = 11:41:37.123
          getUTCMillis(11, 41, 37, 123),
          // Midnight
          0,
          // Almost end of day
          getUTCMillis(23, 59, 59, 999),
          // Null
          std::nullopt,
          // UTC 12:30 + 14 hours = 26:30 -> wraps to 02:30
          getUTCMillis(2, 30),
          // UTC 12:30 - 14 hours = -1:30 -> wraps to 22:30
          getUTCMillis(22, 30),
          // 1 millisecond after midnight
          1,
          // UTC 15:45:30.500 - 8 hours = 07:45:30.500
          getUTCMillis(7, 45, 30, 500),
          // Null
          std::nullopt,
      },
      TIME());

  testCast(input, expected);

  // UTC 14:22:11.456 with +05:30 -> Local 19:52:11.456
  const int64_t timeWithTz =
      pack(getUTCMillis(14, 22, 11, 456), biasEncode(5 * 60 + 30));

  auto inputConstantVector =
      makeConstant(timeWithTz, 10, TIME_WITH_TIME_ZONE());

  const int64_t expectedTime = getUTCMillis(19, 52, 11, 456);
  auto expectedConstantVector = makeConstant(expectedTime, 10, TIME());

  testCast(inputConstantVector, expectedConstantVector);
}

TEST_F(TimeWithTimezoneCastTest, fromTime) {
  {
    // Test casting TIME to TIME WITH TIME ZONE with various times
    // TIME values are in local time, and get converted to UTC for storage
    auto input = makeFlatVector<int64_t>(
        {
            0, // 00:00:00.000 (midnight) local
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                321, // 03:04:05.321 local
            12 * kMillisInHour, // 12:00:00.000 (noon) local
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999, // 23:59:59.999 (end of day) local
        },
        TIME());

    setQueryTimeZone("America/Los_Angeles");
    // LA is UTC-8, so offset is -8 * 60 = -480 minutes
    // Local times are converted to UTC: local - (-8 hours) = local + 8 hours
    auto expected = makeFlatVector<int64_t>(
        {
            // 00:00:00 local -> 08:00:00 UTC
            packTimeWithTZ(8 * kMillisInHour, -480),
            // 03:04:05.321 local -> 11:04:05.321 UTC
            packTimeWithTZ(
                11 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321,
                -480),
            // 12:00:00 local -> 20:00:00 UTC
            packTimeWithTZ(20 * kMillisInHour, -480),
            // 23:59:59.999 local -> 07:59:59.999 UTC (next day, wraps to
            // 07:59:59.999)
            packTimeWithTZ(
                7 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with nulls
    auto input = makeNullableFlatVector<int64_t>(
        {0,
         std::nullopt,
         12 * kMillisInHour,
         std::nullopt,
         23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
             999},
        TIME());

    setQueryTimeZone("America/Los_Angeles");
    // LA is UTC-8, local times are converted to UTC: local + 8 hours

    auto expected = makeNullableFlatVector<int64_t>(
        {// 00:00:00 local -> 08:00:00 UTC
         packTimeWithTZ(8 * kMillisInHour, -480),
         std::nullopt,
         // 12:00:00 local -> 20:00:00 UTC
         packTimeWithTZ(20 * kMillisInHour, -480),
         std::nullopt,
         // 23:59:59.999 local -> 07:59:59.999 UTC (wraps to next day)
         packTimeWithTZ(
             7 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                 999,
             -480)},
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with different timezone (Asia/Shanghai, UTC+8)
    auto input = makeFlatVector<int64_t>(
        {
            0,
            12 * kMillisInHour,
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");

    // Shanghai is UTC+8, so offset is 8 * 60 = 480 minutes
    // Local times are converted to UTC: local - 8 hours
    auto expected = makeFlatVector<int64_t>(
        {
            // 00:00:00 local -> 16:00:00 UTC (previous day, wraps to 16:00)
            packTimeWithTZ(16 * kMillisInHour, 480),
            // 12:00:00 local -> 04:00:00 UTC
            packTimeWithTZ(4 * kMillisInHour, 480),
            // 23:59:59.999 local -> 15:59:59.999 UTC
            packTimeWithTZ(
                15 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with adjustTimestampToTimezone disabled
    // When disabled, offset should be 0 (UTC)
    auto input = makeFlatVector<int64_t>(
        {
            0,
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond + 321,
            12 * kMillisInHour,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");
    disableAdjustTimestampToTimezone();

    // When adjustTimestampToTimezone is false, offset is 0 (UTC)
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(0, 0),
            packTimeWithTZ(
                3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321,
                0),
            packTimeWithTZ(12 * kMillisInHour, 0),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test constant TIME vector cast to TIME WITH TIME ZONE (non-null)
    auto constantTimeVector = BaseVector::wrapInConstant(
        1000, 0, makeFlatVector<int64_t>({12 * kMillisInHour}, TIME()));

    setQueryTimeZone("America/New_York");

    // NY is UTC-5, so offset is -5 * 60 = -300 minutes
    // 12:00:00 local -> 17:00:00 UTC
    auto expected = BaseVector::wrapInConstant(
        1000,
        0,
        makeFlatVector<int64_t>(
            {packTimeWithTZ(17 * kMillisInHour, -300)}, TIME_WITH_TIME_ZONE()));

    testCast(constantTimeVector, expected);
  }

  {
    // Test constant TIME vector cast to TIME WITH TIME ZONE (null)
    auto nullTimeVector = BaseVector::createNullConstant(TIME(), 500, pool());

    setQueryTimeZone("America/Los_Angeles");

    auto expected =
        BaseVector::createNullConstant(TIME_WITH_TIME_ZONE(), 500, pool());

    testCast(nullTimeVector, expected);
  }

  {
    // Test constant TIME vector with different sizes
    setQueryTimeZone("America/Los_Angeles");

    for (auto size : {1, 10, 100, 1000}) {
      auto constantTimeVector = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
               321},
              TIME()));

      // LA is UTC-8, so offset is -8 * 60 = -480 minutes
      // 03:04:05.321 local -> 11:04:05.321 UTC
      auto expected = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {packTimeWithTZ(
                  11 * kMillisInHour + 4 * kMillisInMinute +
                      5 * kMillisInSecond + 321,
                  -480)},
              TIME_WITH_TIME_ZONE()));

      testCast(constantTimeVector, expected);
    }
  }

  {
    // Test basic cast for completeness
    auto input =
        makeFlatVector<int64_t>({0, 12 * kMillisInHour, 86399999}, TIME());

    setQueryTimeZone("America/Los_Angeles");

    // LA is UTC-8, local times are converted to UTC: local + 8 hours
    auto expected = makeFlatVector<int64_t>(
        {
            // 00:00:00 local -> 08:00:00 UTC
            packTimeWithTZ(8 * kMillisInHour, -480),
            // 12:00:00 local -> 20:00:00 UTC
            packTimeWithTZ(20 * kMillisInHour, -480),
            // 23:59:59.999 local -> 07:59:59.999 UTC (wraps to next day)
            packTimeWithTZ(
                7 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with session start time during DST (summer)
    // 2023-07-15 12:00:00 UTC (during DST in America/Los_Angeles)
    // LA is UTC-7 during DST, so offset is -7 * 60 = -420 minutes
    int64_t sessionStartMs = 1689426000000LL; // 2023-07-15 12:00:00 UTC
    auto input = makeFlatVector<int64_t>(
        {
            0,
            12 * kMillisInHour,
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999,
        },
        TIME());

    setQueryTimeZoneAndStartTime("America/Los_Angeles", sessionStartMs);

    // During DST, LA is UTC-7 (PDT), so offset is -420 minutes
    // Local times are converted to UTC: local - (-7 hours) = local + 7 hours
    auto expected = makeFlatVector<int64_t>(
        {
            // 00:00:00 local -> 07:00:00 UTC
            packTimeWithTZ(7 * kMillisInHour, -420),
            // 12:00:00 local -> 19:00:00 UTC
            packTimeWithTZ(19 * kMillisInHour, -420),
            // 23:59:59.999 local -> 06:59:59.999 UTC (next day wraps to
            // 06:59:59.999)
            packTimeWithTZ(
                6 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                -420),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with session start time during standard time (winter)
    // 2023-01-15 12:00:00 UTC (during standard time in America/Los_Angeles)
    // LA is UTC-8 during standard time, so offset is -8 * 60 = -480 minutes
    int64_t sessionStartMs = 1673779200000LL; // 2023-01-15 12:00:00 UTC
    auto input = makeFlatVector<int64_t>(
        {
            0,
            12 * kMillisInHour,
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999,
        },
        TIME());

    setQueryTimeZoneAndStartTime("America/Los_Angeles", sessionStartMs);

    // During standard time, LA is UTC-8 (PST), so offset is -480 minutes
    // Local times are converted to UTC: local - (-8 hours) = local + 8 hours
    auto expected = makeFlatVector<int64_t>(
        {
            // 00:00:00 local -> 08:00:00 UTC
            packTimeWithTZ(8 * kMillisInHour, -480),
            // 12:00:00 local -> 20:00:00 UTC
            packTimeWithTZ(20 * kMillisInHour, -480),
            // 23:59:59.999 local -> 07:59:59.999 UTC (next day wraps to
            // 07:59:59.999)
            packTimeWithTZ(
                7 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }
}

TEST_F(TimeWithTimezoneCastTest, fromTimeVerifyUtcStorage) {
  // This test verifies that when casting TIME to TIME WITH TIME ZONE,
  // the time component is stored as UTC milliseconds, not local milliseconds.
  // TIME WITH TIME ZONE stores:
  // - Upper 52 bits: UTC milliseconds since midnight
  // - Lower 12 bits: timezone offset

  {
    // TIME '14:00:00' in America/Los_Angeles (UTC-8)
    // Local time: 14:00:00 -> UTC time: 22:00:00
    auto input = makeFlatVector<int64_t>(
        {
            14 * kMillisInHour, // 14:00:00 local
        },
        TIME());

    setQueryTimeZone("America/Los_Angeles");
    // LA is UTC-8, so offset is -8 * 60 = -480 minutes

    // Cast TIME to TIME WITH TIME ZONE using testCast
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(22 * kMillisInHour, -480), // Expected: UTC 22:00
        },
        TIME_WITH_TIME_ZONE());

    // This will fail with the current buggy implementation
    testCast(input, expected);
  }

  {
    // TIME '06:00:00' in Asia/Shanghai (UTC+8)
    // Local time: 06:00:00 -> UTC time: 22:00:00 (previous day)
    auto input = makeFlatVector<int64_t>(
        {
            6 * kMillisInHour, // 06:00:00 local
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");
    // Shanghai is UTC+8, so offset is 8 * 60 = 480 minutes

    auto expected = makeFlatVector<int64_t>(
        {
            // BUG: This will fail because current code stores local time
            // (06:00), not UTC time (22:00)
            packTimeWithTZ(22 * kMillisInHour, 480), // Expected: UTC 22:00
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test case 3: Round-trip test
    // TIME -> TIME WITH TIME ZONE -> TIME should preserve the original value
    auto input = makeFlatVector<int64_t>(
        {
            14 * kMillisInHour + 30 * kMillisInMinute, // 14:30:00
        },
        TIME());

    setQueryTimeZone("America/Los_Angeles");

    // First cast TIME to TIME WITH TIME ZONE
    auto timeWithTzExpected = makeFlatVector<int64_t>(
        {
            // UTC time should be 22:30:00 (14:30 - (-8:00) = 22:30)
            packTimeWithTZ(22 * kMillisInHour + 30 * kMillisInMinute, -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, timeWithTzExpected);

    // Then cast back to TIME - should preserve original local time
    auto timeExpected = makeFlatVector<int64_t>(
        {
            14 * kMillisInHour + 30 * kMillisInMinute,
        },
        TIME());

    testCast(timeWithTzExpected, timeExpected);
  }

  {
    // Verify consistency with fromVarchar
    // Casting "14:00:00-08:00" from VARCHAR should produce the same result
    // as casting TIME '14:00:00' with America/Los_Angeles timezone

    setQueryTimeZone("America/Los_Angeles");

    // Cast TIME to TIME WITH TIME ZONE
    auto timeInput = makeFlatVector<int64_t>({14 * kMillisInHour}, TIME());

    // Cast VARCHAR to TIME WITH TIME ZONE
    auto varcharInput = makeFlatVector<std::string>({"14:00:00.000-08:00"});

    // Both should produce the same result: UTC 22:00:00 with offset -480
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(22 * kMillisInHour, -480),
        },
        TIME_WITH_TIME_ZONE());

    // Test TIME cast
    testCast(timeInput, expected);

    // Test VARCHAR cast
    testCast(varcharInput, expected);
  }
}

} // namespace
} // namespace facebook::velox
