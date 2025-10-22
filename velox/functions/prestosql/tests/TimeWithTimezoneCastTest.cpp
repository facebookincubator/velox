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
              TimeWithTimezoneType::biasEncode(0)),
          // Same time with negative offset (EST).
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              TimeWithTimezoneType::biasEncode(-5 * 60)),
          // Same time with positive offset including half-hour (IST).
          pack(
              6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond +
                  123,
              TimeWithTimezoneType::biasEncode(5 * 60 + 30)),
          // Boundary: Midnight.
          pack(0, TimeWithTimezoneType::biasEncode(0)),
          // Boundary: Almost end of day.
          pack(
              23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                  999,
              TimeWithTimezoneType::biasEncode(0)),
          // Null value.
          std::nullopt,
          // Boundary: Maximum positive timezone offset (+14:00).
          pack(12 * kMillisInHour, TimeWithTimezoneType::biasEncode(14 * 60)),
          // Boundary: Maximum negative timezone offset (-14:00).
          pack(12 * kMillisInHour, TimeWithTimezoneType::biasEncode(-14 * 60)),
          // Edge case: 1 millisecond after midnight.
          pack(1, TimeWithTimezoneType::biasEncode(0)),
          // Various time components with negative offset including minutes.
          pack(
              12 * kMillisInHour + 34 * kMillisInMinute + 56 * kMillisInSecond +
                  789,
              TimeWithTimezoneType::biasEncode(-4 * 60 - 30)),
          // Time with no milliseconds component.
          pack(
              9 * kMillisInHour + 8 * kMillisInMinute + 7 * kMillisInSecond,
              TimeWithTimezoneType::biasEncode(5 * 60 + 45)),
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

} // namespace
} // namespace facebook::velox
