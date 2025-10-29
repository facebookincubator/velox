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

#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Time.h"

using facebook::velox::util::biasEncode;
using facebook::velox::util::pack;

namespace facebook::velox {
namespace {

class TimeWithTimezoneCastTest : public functions::test::CastBaseTest {};

TEST_F(TimeWithTimezoneCastTest, toVarchar) {
  // Test comprehensive scenarios: various times, timezones, boundary values,
  // and nulls.
  auto input = makeNullableFlatVector<int64_t>(
      {
          // Basic time with UTC.
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(0)),
          // Same time with negative offset (EST).
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(-5 * 60)),
          // Same time with positive offset including half-hour (IST).
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(5 * 60 + 30)),
          // Boundary: Midnight.
          pack(0, biasEncode(0)),
          // Boundary: Almost end of day.
          pack(
              23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                  999,
              biasEncode(0)),
          // Null value.
          std::nullopt,
          // Boundary: Maximum positive timezone offset (+14:00).
          pack(12 * kMillisInHour, biasEncode(14 * 60)),
          // Boundary: Maximum negative timezone offset (-14:00).
          pack(12 * kMillisInHour, biasEncode(-14 * 60)),
          // Edge case: 1 millisecond after midnight.
          pack(1, biasEncode(0)),
          // Various time components with negative offset including minutes.
          pack(
              12 * kMillisInHour + 34 * kMillisInMinute + 56 * kMillisInSecond +
                  789,
              biasEncode(-4 * 60 - 30)),
          // Time with no milliseconds component.
          pack(
              9 * kMillisInHour + 8 * kMillisInMinute + 7 * kMillisInSecond,
              biasEncode(5 * 60 + 45)),
          // Another null value.
          std::nullopt,
      },
      TIME_WITH_TIME_ZONE());

  auto expected = makeNullableFlatVector<std::string>({
      "06:11:37.123+00:00",
      "06:11:37.123-05:00",
      "06:11:37.123+05:30",
      "00:00:00.000+00:00",
      "23:59:59.999+00:00",
      std::nullopt,
      "12:00:00.000+14:00",
      "12:00:00.000-14:00",
      "00:00:00.001+00:00",
      "12:34:56.789-04:30",
      "09:08:07.000+05:45",
      std::nullopt,
  });

  testCast(input, expected);
}

TEST_F(TimeWithTimezoneCastTest, fromVarchar) {
  // Test casting VARCHAR to TIME WITH TIME ZONE with various formats.
  auto input = makeNullableFlatVector<std::string>({
      // Basic format: H:m:s.SSS+HH:mm
      "6:11:37.123+00:00",
      "6:11:37.123-05:00",
      "6:11:37.123+05:30",
      // Format without milliseconds: H:m:s+HH:mm
      "12:34:56+01:00",
      "12:34:56-08:00",
      // Compact format: H:m+HH:mm
      "1:30+05:30",
      "23:45-07:00",
      // Format with space before timezone: H:m:s.SSS +HH:mm
      "14:22:11.456 +02:00",
      "9:15:30.789 -04:30",
      // Compact timezone format (no colon): H:m:s+HHmm
      "3:4:5+0709",
      "15:45:30-0800",
      // Short timezone format: H:m:s+HH
      "8:30:00+05",
      "16:45:30-08",
      // Boundary: Midnight with UTC
      "0:0:0.000+00:00",
      // Boundary: Almost end of day
      "23:59:59.999+00:00",
      // Boundary: Maximum positive timezone offset (+14:00)
      "12:00:00.000+14:00",
      // Boundary: Maximum negative timezone offset (-14:00)
      "12:00:00.000-14:00",
      // Edge case: 1 millisecond after midnight
      "0:0:0.001+00:00",
      // Various valid formats with different hour/minute combinations
      "9:8:7.000+05:45",
      "1:2:3+01:30",
      // Null value
      std::nullopt,
      // Compact format with space: H:m +HH
      "7:30 +03",
      "18:45 -06",
      // Double-digit hours and minutes: HH:mm:ss.SSS+HH:mm
      "06:11:37.123+00:00",
      "12:34:56.789-04:30",
  });

  auto expected = makeNullableFlatVector<int64_t>(
      {
          // 6:11:37.123+00:00
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(0)),
          // 6:11:37.123-05:00
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(-5 * 60)),
          // 6:11:37.123+05:30
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(5 * 60 + 30)),
          // 12:34:56+01:00
          pack(
              12 * kMillisInHour + 34 * kMillisInMinute + 56 * kMillisInSecond,
              biasEncode(1 * 60)),
          // 12:34:56-08:00
          pack(
              12 * kMillisInHour + 34 * kMillisInMinute + 56 * kMillisInSecond,
              biasEncode(-8 * 60)),
          // 1:30+05:30
          pack(
              1 * kMillisInHour + 30 * kMillisInMinute,
              biasEncode(5 * 60 + 30)),
          // 23:45-07:00
          pack(23 * kMillisInHour + 45 * kMillisInMinute, biasEncode(-7 * 60)),
          // 14:22:11.456 +02:00
          pack(
              14 * kMillisInHour + 22 * kMillisInMinute + 11 * kMillisInSecond +
                  456,
              biasEncode(2 * 60)),
          // 9:15:30.789 -04:30
          pack(
              9 * kMillisInHour + 15 * kMillisInMinute + 30 * kMillisInSecond +
                  789,
              biasEncode(-4 * 60 - 30)),
          // 3:4:5+0709
          pack(
              3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond,
              biasEncode(7 * 60 + 9)),
          // 15:45:30-0800
          pack(
              15 * kMillisInHour + 45 * kMillisInMinute + 30 * kMillisInSecond,
              biasEncode(-8 * 60)),
          // 8:30:00+05
          pack(8 * kMillisInHour + 30 * kMillisInMinute, biasEncode(5 * 60)),
          // 16:45:30-08
          pack(
              16 * kMillisInHour + 45 * kMillisInMinute + 30 * kMillisInSecond,
              biasEncode(-8 * 60)),
          // 0:0:0.000+00:00
          pack(0, biasEncode(0)),
          // 23:59:59.999+00:00
          pack(
              23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                  999,
              biasEncode(0)),
          // 12:00:00.000+14:00
          pack(12 * kMillisInHour, biasEncode(14 * 60)),
          // 12:00:00.000-14:00
          pack(12 * kMillisInHour, biasEncode(-14 * 60)),
          // 0:0:0.001+00:00
          pack(1, biasEncode(0)),
          // 9:8:7.000+05:45
          pack(
              9 * kMillisInHour + 8 * kMillisInMinute + 7 * kMillisInSecond,
              biasEncode(5 * 60 + 45)),
          // 1:2:3+01:30
          pack(
              1 * kMillisInHour + 2 * kMillisInMinute + 3 * kMillisInSecond,
              biasEncode(1 * 60 + 30)),
          // Null value
          std::nullopt,
          // 7:30 +03
          pack(7 * kMillisInHour + 30 * kMillisInMinute, biasEncode(3 * 60)),
          // 18:45 -06
          pack(18 * kMillisInHour + 45 * kMillisInMinute, biasEncode(-6 * 60)),
          // 06:11:37.123+00:00
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              biasEncode(0)),
          // 12:34:56.789-04:30
          pack(
              12 * kMillisInHour + 34 * kMillisInMinute + 56 * kMillisInSecond +
                  789,
              biasEncode(-4 * 60 - 30)),
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
} // namespace
} // namespace facebook::velox
