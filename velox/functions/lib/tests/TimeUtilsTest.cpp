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

#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/type/FastDate.h"

namespace facebook::velox::functions {
namespace {

TEST(TimeUtilsTest, fromDateTimeUnitString) {
  // Return null when unit string is invalid and throwIfInvalid is false.
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("microsecond", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("dd", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("mon", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("mm", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("yyyy", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("yy", false));

  // Throw when unit string is invalid and throwIfInvalid is true.
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("", true), "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("microsecond", true),
      "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("dd", true), "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("mon", true), "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("mm", true), "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("yyyy", true), "Unsupported datetime unit:");
  VELOX_ASSERT_THROW(
      fromDateTimeUnitString("yy", true), "Unsupported datetime unit:");

  ASSERT_EQ(
      DateTimeUnit::kMillisecond, fromDateTimeUnitString("millisecond", false));
  ASSERT_EQ(DateTimeUnit::kSecond, fromDateTimeUnitString("second", false));
  ASSERT_EQ(DateTimeUnit::kMinute, fromDateTimeUnitString("minute", false));
  ASSERT_EQ(DateTimeUnit::kHour, fromDateTimeUnitString("hour", false));
  ASSERT_EQ(DateTimeUnit::kDay, fromDateTimeUnitString("day", false));
  ASSERT_EQ(DateTimeUnit::kWeek, fromDateTimeUnitString("week", false));
  ASSERT_EQ(DateTimeUnit::kMonth, fromDateTimeUnitString("month", false));
  ASSERT_EQ(DateTimeUnit::kQuarter, fromDateTimeUnitString("quarter", false));
  ASSERT_EQ(DateTimeUnit::kYear, fromDateTimeUnitString("year", false));

  ASSERT_EQ(
      DateTimeUnit::kMicrosecond,
      fromDateTimeUnitString("microsecond", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kMillisecond,
      fromDateTimeUnitString("millisecond", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kSecond,
      fromDateTimeUnitString("second", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kMinute,
      fromDateTimeUnitString("minute", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kHour, fromDateTimeUnitString("hour", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kDay, fromDateTimeUnitString("day", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kDay, fromDateTimeUnitString("dd", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kWeek, fromDateTimeUnitString("week", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kMonth, fromDateTimeUnitString("month", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kMonth, fromDateTimeUnitString("mon", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kMonth, fromDateTimeUnitString("mm", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kQuarter,
      fromDateTimeUnitString("quarter", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kYear, fromDateTimeUnitString("year", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kYear, fromDateTimeUnitString("yyyy", false, true, true));
  ASSERT_EQ(
      DateTimeUnit::kYear, fromDateTimeUnitString("yy", false, true, true));
}

TEST(TimeUtilsTest, adjustEpoch) {
  EXPECT_EQ(Timestamp(998474640, 0), adjustEpoch(998474645, 60));
  EXPECT_EQ(Timestamp(998474400, 0), adjustEpoch(998474645, 60 * 60));
  EXPECT_EQ(Timestamp(998438400, 0), adjustEpoch(998474645, 24 * 60 * 60));
  EXPECT_EQ(Timestamp(-120, 0), adjustEpoch(-61, 60));
}

TEST(TimeUtilsTest, truncateDate) {
  // Epoch day 0 is 1970-01-01, a Thursday.
  EXPECT_EQ(0, truncateDate(0, DateTimeUnit::kDay));
  EXPECT_EQ(-3, truncateDate(0, DateTimeUnit::kWeek)); // Monday 1969-12-29.
  EXPECT_EQ(0, truncateDate(0, DateTimeUnit::kMonth));
  EXPECT_EQ(0, truncateDate(0, DateTimeUnit::kQuarter));
  EXPECT_EQ(0, truncateDate(0, DateTimeUnit::kYear));

  // Day 4 is Monday 1970-01-05; day 3 is Sunday 1970-01-04.
  EXPECT_EQ(4, truncateDate(4, DateTimeUnit::kWeek));
  EXPECT_EQ(-3, truncateDate(3, DateTimeUnit::kWeek)); // Back to 1969-12-29.

  // Negative / pre-epoch days.
  EXPECT_EQ(-1, truncateDate(-1, DateTimeUnit::kDay)); // 1969-12-31.
  EXPECT_EQ(-3, truncateDate(-1, DateTimeUnit::kWeek)); // Monday 1969-12-29.
  EXPECT_EQ(-31, truncateDate(-1, DateTimeUnit::kMonth)); // 1969-12-01.
  EXPECT_EQ(-365, truncateDate(-1, DateTimeUnit::kYear)); // 1969-01-01.
  EXPECT_EQ(-10, truncateDate(-7, DateTimeUnit::kWeek)); // 1969-12-25 Thu.

  // INT32 boundaries for kWeek exercise the signed-overflow paths that must
  // stay defined under sanitizers.
  EXPECT_EQ(
      2147483643,
      truncateDate(std::numeric_limits<int32_t>::max(), DateTimeUnit::kWeek));
  // Near INT32_MIN the truncated unit start is unrepresentable as an int32 day,
  // so every unit rejects it rather than returning a wrapped value.
  for (auto unit :
       {DateTimeUnit::kWeek,
        DateTimeUnit::kMonth,
        DateTimeUnit::kQuarter,
        DateTimeUnit::kYear}) {
    VELOX_ASSERT_THROW(
        truncateDate(std::numeric_limits<int32_t>::min(), unit),
        "Date is out of range for truncation");
  }

  // Fast-path boundaries near the FastDate year limits.
  EXPECT_EQ(
      ymdToDays(fast_date::kYearMax - 1, 1, 1),
      truncateDate(
          ymdToDays(fast_date::kYearMax - 1, 6, 15), DateTimeUnit::kYear));
  EXPECT_EQ(
      ymdToDays(fast_date::kYearMin + 1, 6, 1),
      truncateDate(
          ymdToDays(fast_date::kYearMin + 1, 6, 15), DateTimeUnit::kMonth));

  // Inputs just outside the FastDate day range fall through to the std::tm
  // path; results must stay consistent with the fast path. Expected day
  // numbers were computed independently via proleptic Gregorian.
  EXPECT_EQ(
      1061042246,
      truncateDate(fast_date::kRataDieMax + 1, DateTimeUnit::kYear));
  EXPECT_EQ(
      1061042397,
      truncateDate(fast_date::kRataDieMax + 1, DateTimeUnit::kMonth));
  EXPECT_EQ(
      -12699482, truncateDate(fast_date::kRataDieMin - 1, DateTimeUnit::kYear));
  EXPECT_EQ(
      -12699451,
      truncateDate(fast_date::kRataDieMin - 1, DateTimeUnit::kMonth));
  EXPECT_EQ(
      2147483455,
      truncateDate(std::numeric_limits<int32_t>::max(), DateTimeUnit::kYear));
}

TEST(TimeUtilsTest, truncateTimestamp) {
  auto* timezone = tz::locateZone("GMT");

  EXPECT_EQ(
      Timestamp(0, 0),
      truncateTimestamp(Timestamp(0, 0), DateTimeUnit::kSecond, timezone));
  EXPECT_EQ(
      Timestamp(0, 0),
      truncateTimestamp(Timestamp(0, 123), DateTimeUnit::kSecond, timezone));
  EXPECT_EQ(
      Timestamp(-1, 0),
      truncateTimestamp(Timestamp(-1, 0), DateTimeUnit::kSecond, timezone));

  EXPECT_EQ(
      Timestamp(998474645, 321'001'000),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMicrosecond,
          timezone));
  EXPECT_EQ(
      Timestamp(998474645, 321'000'000),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMillisecond,
          timezone));
  EXPECT_EQ(
      Timestamp(998474645, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kSecond,
          timezone));
  EXPECT_EQ(
      Timestamp(998474640, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMinute,
          timezone));
  EXPECT_EQ(
      Timestamp(998474400, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kHour, timezone));
  EXPECT_EQ(
      Timestamp(998438400, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kDay, timezone));
  EXPECT_EQ(
      Timestamp(998265600, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kWeek, timezone));
  EXPECT_EQ(
      Timestamp(996624000, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kMonth, timezone));
  EXPECT_EQ(
      Timestamp(993945600, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kQuarter,
          timezone));
  EXPECT_EQ(
      Timestamp(978307200, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kYear, timezone));

  auto* timezone1 = tz::locateZone("America/Los_Angeles");
  EXPECT_EQ(
      Timestamp(0, 0),
      truncateTimestamp(Timestamp(0, 0), DateTimeUnit::kSecond, timezone1));
  EXPECT_EQ(
      Timestamp(0, 0),
      truncateTimestamp(Timestamp(0, 123), DateTimeUnit::kSecond, timezone1));
  EXPECT_EQ(
      Timestamp(-57600, 0),
      truncateTimestamp(Timestamp(0, 0), DateTimeUnit::kDay, timezone1));

  EXPECT_EQ(
      Timestamp(998474645, 321'001'000),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMicrosecond,
          timezone1));
  EXPECT_EQ(
      Timestamp(998474645, 321'000'000),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMillisecond,
          timezone1));
  EXPECT_EQ(
      Timestamp(998474645, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kSecond,
          timezone1));
  EXPECT_EQ(
      Timestamp(998474640, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMinute,
          timezone1));
  EXPECT_EQ(
      Timestamp(998474400, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kHour, timezone1));
  EXPECT_EQ(
      Timestamp(998463600, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kDay, timezone1));
  EXPECT_EQ(
      Timestamp(998290800, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kWeek, timezone1));
  EXPECT_EQ(
      Timestamp(996649200, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kMonth,
          timezone1));
  EXPECT_EQ(
      Timestamp(993970800, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234),
          DateTimeUnit::kQuarter,
          timezone1));
  EXPECT_EQ(
      Timestamp(978336000, 0),
      truncateTimestamp(
          Timestamp(998'474'645, 321'001'234), DateTimeUnit::kYear, timezone1));
}

TEST(TimeUtilsTest, truncateTimestampTimeZone) {
  auto* la = tz::locateZone("America/Los_Angeles");

  // 2026-03-15 17:00 UTC = 2026-03-15 10:00 PDT (Sunday, post DST switch).
  // Week truncation -> Monday 2026-03-09 00:00 PDT = 2026-03-09 07:00 UTC.
  EXPECT_EQ(
      Timestamp(1773039600, 0),
      truncateTimestamp(Timestamp(1773594000, 0), DateTimeUnit::kWeek, la));

  // 2026-02-15 18:00 UTC = 2026-02-15 10:00 PST.
  // Month truncation -> 2026-02-01 00:00 PST = 2026-02-01 08:00 UTC.
  EXPECT_EQ(
      Timestamp(1769932800, 0),
      truncateTimestamp(Timestamp(1771178400, 0), DateTimeUnit::kMonth, la));

  // Pre-epoch with a negative UTC offset exercises floor division end-to-end.
  // 1969-12-31 23:30 UTC = 1969-12-31 15:30 PST (Wednesday).
  // Week truncation -> Monday 1969-12-29 00:00 PST = 1969-12-29 08:00 UTC.
  EXPECT_EQ(
      Timestamp(-230400, 0),
      truncateTimestamp(Timestamp(-1800, 0), DateTimeUnit::kWeek, la));

  // Non-integer-hour offsets. 2026-06-15 12:00 UTC is a Monday in both zones.
  auto* kolkata = tz::locateZone("Asia/Kolkata"); // +05:30.
  EXPECT_EQ(
      Timestamp(1781461800, 0),
      truncateTimestamp(
          Timestamp(1781524800, 0), DateTimeUnit::kWeek, kolkata));
  EXPECT_EQ(
      Timestamp(1780252200, 0),
      truncateTimestamp(
          Timestamp(1781524800, 0), DateTimeUnit::kMonth, kolkata));

  auto* kathmandu = tz::locateZone("Asia/Kathmandu"); // +05:45.
  EXPECT_EQ(
      Timestamp(1781460900, 0),
      truncateTimestamp(
          Timestamp(1781524800, 0), DateTimeUnit::kWeek, kathmandu));
  EXPECT_EQ(
      Timestamp(1780251300, 0),
      truncateTimestamp(
          Timestamp(1781524800, 0), DateTimeUnit::kMonth, kathmandu));
}
} // namespace
} // namespace facebook::velox::functions
