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
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox {
namespace {

class TimestampWithTimeZoneCastTest : public functions::test::CastBaseTest {
 public:
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

  // Helper to parse a timestamp with timezone string and return packed value
  // e.g., "2023-08-05 10:15:00.000 Europe/London"
  int64_t parseTimestampWithTz(const std::string& str) {
    auto parsed =
        util::fromTimestampWithTimezoneString(
            str.c_str(), str.size(), util::TimestampParseMode::kPrestoCast)
            .thenOrThrow(folly::identity, [&](const Status& status) {
              VELOX_USER_FAIL("{}", status.message());
            });

    // The parsed timestamp is in local time, need to convert to UTC
    auto ts = parsed.timestamp;
    ts.toGMT(*parsed.timeZone);
    return pack(ts.toMillis(), parsed.timeZone->id());
  }

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
};

TEST_F(TimestampWithTimeZoneCastTest, fromTimestamp) {
  const auto tsVector = makeNullableFlatVector<Timestamp>(
      {Timestamp(1996, 0), std::nullopt, Timestamp(19920, 0)});
  auto timestamps =
      std::vector<int64_t>{1996 * kMillisInSecond, 0, 19920 * kMillisInSecond};
  auto timezones = std::vector<TimeZoneKey>{0, 0, 0};
  const auto expected = makeTimestampWithTimeZoneVector(
      timestamps.size(),
      [&](int32_t index) { return timestamps[index]; },
      [&](int32_t index) { return timezones[index]; });
  expected->setNull(1, true);

  testCast(tsVector, expected);
}

TEST_F(TimestampWithTimeZoneCastTest, fromVarchar) {
  const auto stringVector = makeNullableFlatVector<StringView>(
      {std::nullopt,
       "2012-10-31 01:00:47 America/Denver",
       "2012-10-31 01:00:47 -06:00",
       "1994-05-06 15:49 Europe/Vienna",
       "1979-02-24 08:33:31 Pacific/Chatham",
       "1979-02-24 08:33:31 +13:45",
       "1979-02-24 Atlantic/Bermuda"});

  // Varchar representations above hold local time (relative to specified time
  // zone). For instance, the first string represents a wall-clock displaying
  // '2012-10-31 01:00:47' in Denver. Below, we use UTC representations of the
  // above local wall-clocks, to match the UTC timepoints held in the
  // TimestampWithTimezone type.
  auto denverUTC = parseTimestamp("2012-10-31 07:00:47").toMillis();
  auto viennaUTC = parseTimestamp("1994-05-06 13:49:00").toMillis();
  auto chathamUTC = parseTimestamp("1979-02-23 18:48:31").toMillis();
  auto bermudaUTC = parseTimestamp("1979-02-24 04:00:00").toMillis();

  auto timestamps = std::vector<int64_t>{
      0, denverUTC, denverUTC, viennaUTC, chathamUTC, chathamUTC, bermudaUTC};

  auto timezones = std::vector<TimeZoneKey>{
      {0,
       (int16_t)tz::getTimeZoneID("America/Denver"),
       (int16_t)tz::getTimeZoneID("-06:00"),
       (int16_t)tz::getTimeZoneID("Europe/Vienna"),
       (int16_t)tz::getTimeZoneID("Pacific/Chatham"),
       (int16_t)tz::getTimeZoneID("+13:45"),
       (int16_t)tz::getTimeZoneID("Atlantic/Bermuda")}};

  const auto expected = makeTimestampWithTimeZoneVector(
      timestamps.size(),
      [&](int32_t index) { return timestamps[index]; },
      [&](int32_t index) { return timezones[index]; });
  expected->setNull(0, true);

  testCast(stringVector, expected);
}

TEST_F(TimestampWithTimeZoneCastTest, toVarchar) {
  // 1970-01-01 06:11:37.123 UTC in 4 different time zones.
  const int64_t utcMillis =
      6 * kMillisInHour + 11 * kMillisInMinute + 37 * kMillisInSecond + 123;
  auto input = makeFlatVector<int64_t>(
      {
          // -5 hours.
          pack(utcMillis, tz::getTimeZoneID("America/New_York")),
          // -8 hours.
          pack(utcMillis, tz::getTimeZoneID("America/Los_Angeles")),
          // +8 hours.
          pack(utcMillis, tz::getTimeZoneID("Asia/Shanghai")),
          // +5:30 hours.
          pack(utcMillis, tz::getTimeZoneID("Asia/Calcutta")),
      },
      TIMESTAMP_WITH_TIME_ZONE());

  auto expected = makeFlatVector<std::string>({
      "1970-01-01 01:11:37.123 America/New_York",
      "1969-12-31 22:11:37.123 America/Los_Angeles",
      "1970-01-01 14:11:37.123 Asia/Shanghai",
      "1970-01-01 11:41:37.123 Asia/Kolkata", // Asia/Calcutta is linked to
                                              // Asia/Kolkata.
  });

  auto result = evaluate("cast(c0 as varchar)", makeRowVector({input}));
  test::assertEqualVectors(expected, result);
}

TEST_F(TimestampWithTimeZoneCastTest, fromVarcharWithoutTimezone) {
  setQueryTimeZone("America/Denver");

  const auto stringVector =
      makeNullableFlatVector<StringView>({"2012-10-31 01:00:47"});
  auto denverUTC = parseTimestamp("2012-10-31 07:00:47").toMillis();

  auto timestamps = std::vector<int64_t>{denverUTC};

  auto timezones =
      std::vector<TimeZoneKey>{(int16_t)tz::getTimeZoneID("America/Denver")};

  const auto expected = makeTimestampWithTimeZoneVector(
      timestamps.size(),
      [&](int32_t index) { return timestamps[index]; },
      [&](int32_t index) { return timezones[index]; });

  testCast(stringVector, expected);
}

TEST_F(TimestampWithTimeZoneCastTest, fromVarcharInvalidInput) {
  const auto invalidStringVector1 = makeNullableFlatVector<StringView>(
      {"2012-10-31 01:00:47fooAmerica/Los_Angeles"});

  const auto invalidStringVector2 = makeNullableFlatVector<StringView>(
      {"2012-10-31 01:00:47 America/California"});

  const auto invalidStringVector3 = makeNullableFlatVector<StringView>(
      {"2012-10-31foo01:00:47 America/Los_Angeles"});

  const auto invalidStringVector4 = makeNullableFlatVector<StringView>(
      {"2012-10-31 35:00:47 America/Los_Angeles"});

  const auto invalidStringVector5 =
      makeNullableFlatVector<StringView>({"2012 America/Los_Angeles"});

  const auto invalidStringVector6 =
      makeNullableFlatVector<StringView>({"2012-10 America/Los_Angeles"});

  const auto invalidStringVector7 =
      makeNullableFlatVector<StringView>({"2012-10-01 +16:00"});

  auto millis = parseTimestamp("2012-10-31 07:00:47").toMillis();
  auto timestamps = std::vector<int64_t>{millis};

  auto timezones =
      std::vector<TimeZoneKey>{(int16_t)tz::getTimeZoneID("America/Denver")};

  const auto expected = makeTimestampWithTimeZoneVector(
      timestamps.size(),
      [&](int32_t index) { return timestamps[index]; },
      [&](int32_t index) { return timezones[index]; });

  VELOX_ASSERT_THROW(
      testCast(invalidStringVector1, expected),
      "Unknown timezone value: \"fooAmerica/Los_Angeles\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector2, expected),
      "Unknown timezone value: \"America/California\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector3, expected),
      "Unknown timezone value: \"foo01:00:47\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector4, expected),
      "Unknown timezone value: \"35:00:47\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector5, expected),
      "Unable to parse timestamp value: \"2012 America/Los_Angeles\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector6, expected),
      "Unable to parse timestamp value: \"2012-10 America/Los_Angeles\"");
  VELOX_ASSERT_THROW(
      testCast(invalidStringVector7, expected),
      "Unknown timezone value in: \"2012-10-01 +16:00\"");
}

TEST_F(TimestampWithTimeZoneCastTest, toTimestamp) {
  auto timestamps =
      std::vector<int64_t>{1996 * kMillisInSecond, 0, 19920 * kMillisInSecond};
  auto timezones = std::vector<TimeZoneKey>{0, 0, 1825 /*America/Los_Angeles*/};

  const auto tsWithTZVector = makeTimestampWithTimeZoneVector(
      timestamps.size(),
      [&](int32_t index) { return timestamps[index]; },
      [&](int32_t index) { return timezones[index]; });
  tsWithTZVector->setNull(1, true);

  for (const char* timezone : {"America/Los_Angeles", "America/New_York"}) {
    setQueryTimeZone(timezone);
    auto expected = makeNullableFlatVector<Timestamp>(
        {Timestamp(1996, 0), std::nullopt, Timestamp(19920, 0)});
    testCast(tsWithTZVector, expected);

    // Cast('1969-12-31 16:00:00 -08:00' as timestamp).
    auto result = evaluateOnce<Timestamp>(
        "cast(from_unixtime(c0, 'America/Los_Angeles') as timestamp)",
        makeRowVector({makeFlatVector<double>(std::vector<double>(1, 0))}));
    // 1970-01-01 00:00:00.
    EXPECT_EQ(Timestamp(0, 0), result.value());
  }

  disableAdjustTimestampToTimezone();
  auto expected = makeNullableFlatVector<Timestamp>(
      {Timestamp(1996, 0), std::nullopt, Timestamp(19920 - 8 * 3600, 0)});
  testCast(tsWithTZVector, expected);

  // Cast('1969-12-31 16:00:00 -08:00' as timestamp).
  auto result = evaluateOnce<Timestamp>(
      "cast(from_unixtime(c0, 'America/Los_Angeles') as timestamp)",
      makeRowVector({makeFlatVector<double>(std::vector<double>(1, 0))}));
  // 1969-12-31 16:00:00.
  EXPECT_EQ(Timestamp(-28800, 0), result.value());
}

TEST_F(TimestampWithTimeZoneCastTest, toDate) {
  auto input = makeFlatVector<int64_t>(
      {
          // 6AM UTC is 1AM EST (same day), 10PM PST (previous day), 2PM CST
          // (same day).
          pack(6 * kMillisInHour, tz::getTimeZoneID("America/New_York")),
          pack(6 * kMillisInHour, tz::getTimeZoneID("America/Los_Angeles")),
          pack(6 * kMillisInHour, tz::getTimeZoneID("Asia/Shanghai")),
          // 6PM UTC is 1PM EST (same day), 10AM PST (same day), 2AM CST (next
          // day).
          pack(18 * kMillisInHour, tz::getTimeZoneID("America/New_York")),
          pack(18 * kMillisInHour, tz::getTimeZoneID("America/Los_Angeles")),
          pack(18 * kMillisInHour, tz::getTimeZoneID("Asia/Shanghai")),
      },
      TIMESTAMP_WITH_TIME_ZONE());
  auto expected = makeFlatVector<int32_t>({0, -1, 0, 0, 0, 1}, DATE());

  auto result = evaluate("cast(c0 as date)", makeRowVector({input}));
  test::assertEqualVectors(expected, result);

  // Verify that session time zone doesn't affect the result.

  for (auto tz : {"America/New_York", "America/Los_Angeles", "Asia/Shanghai"}) {
    setQueryTimeZone(tz);
    result = evaluate("cast(c0 as date)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }
}

TEST_F(TimestampWithTimeZoneCastTest, fromDate) {
  auto input = makeFlatVector<int32_t>({-1, 0, 1}, DATE());

  setQueryTimeZone("America/New_York");

  auto tzId = tz::getTimeZoneID("America/New_York");
  auto tzOffset = -5 * kMillisInHour;
  auto expected = makeFlatVector<int64_t>(
      {
          pack(-kMillisInDay - tzOffset, tzId),
          pack(-tzOffset, tzId),
          pack(kMillisInDay - tzOffset, tzId),
      },
      TIMESTAMP_WITH_TIME_ZONE());
  auto result =
      evaluate("cast(c0 as timestamp with time zone)", makeRowVector({input}));
  test::assertEqualVectors(expected, result);

  setQueryTimeZone("Asia/Shanghai");

  tzId = tz::getTimeZoneID("Asia/Shanghai");
  tzOffset = 8 * kMillisInHour;
  expected = makeFlatVector<int64_t>(
      {
          pack(-kMillisInDay - tzOffset, tzId),
          pack(-tzOffset, tzId),
          pack(kMillisInDay - tzOffset, tzId),
      },
      TIMESTAMP_WITH_TIME_ZONE());

  result =
      evaluate("cast(c0 as timestamp with time zone)", makeRowVector({input}));
  test::assertEqualVectors(expected, result);
}

TEST_F(TimestampWithTimeZoneCastTest, fromTime) {
  {
    // Test casting TIME to TIMESTAMP WITH TIME ZONE with default session start
    // time (epoch). TIME values represent time since midnight (in milliseconds)
    auto input = makeFlatVector<int64_t>(
        {
            0, // 00:00:00.000 (midnight)
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                321, // 03:04:05.321
            12 * kMillisInHour, // 12:00:00.000 (noon)
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999, // 23:59:59.999 (end of day)
        },
        TIME());

    setQueryTimeZone("America/Los_Angeles");

    auto tzId = tz::getTimeZoneID("America/Los_Angeles");
    // When session start time is not set (default = 0), uses epoch (1970-01-01)
    // LA is -8 hours from UTC (PST), so 00:00:00 PST = 08:00:00 UTC
    auto expected = makeFlatVector<int64_t>(
        {
            pack(8 * kMillisInHour, tzId), // 00:00:00.000 PST -> 08:00:00 UTC
            pack(
                3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321 + 8 * kMillisInHour,
                tzId), // 03:04:05.321 PST -> 11:04:05.321 UTC
            pack(
                12 * kMillisInHour + 8 * kMillisInHour,
                tzId), // 12:00:00.000 PST -> 20:00:00 UTC
            pack(
                23 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999 + 8 * kMillisInHour,
                tzId), // 23:59:59.999 PST -> 07:59:59.999 UTC (next day)
        },
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }

  {
    // Test casting TIME to TIMESTAMP WITH TIME ZONE with session start time set
    // to 2023-01-15 10:30:00 UTC (1673780400000 ms)
    // This should use the date from session start time (2023-01-15), not epoch
    auto input = makeFlatVector<int64_t>(
        {
            0, // 00:00:00.000 (midnight)
            12 * kMillisInHour, // 12:00:00.000 (noon)
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999, // 23:59:59.999
        },
        TIME());

    // Session start time: 2023-01-15 10:30:00 UTC = 1673779800000 ms
    // In LA timezone (PST, -8 hours), this is 2023-01-15 02:30:00 PST
    // The date portion is 2023-01-15, which is 19372 days from epoch
    // 2023-01-15 00:00:00 UTC = 19372 * 86400 * 1000 = 1673740800000 ms
    int64_t sessionStartMs = 1673779800000LL; // 2023-01-15 10:30:00 UTC
    setQueryTimeZoneAndStartTime("America/Los_Angeles", sessionStartMs);

    auto tzId = tz::getTimeZoneID("America/Los_Angeles");
    // Date portion: 2023-01-15 in LA timezone = 2023-01-15 00:00:00 PST
    // = 2023-01-15 08:00:00 UTC = 1673740800000 + 8*3600*1000 = 1673769600000
    int64_t dateBaseMs = 1673769600000LL; // 2023-01-15 00:00:00 UTC
    auto expected = makeFlatVector<int64_t>(
        {
            pack(
                dateBaseMs,
                tzId), // 2023-01-15 00:00:00.000 PST in UTC
            pack(
                dateBaseMs + 12 * kMillisInHour,
                tzId), // 2023-01-15 12:00:00.000 PST in UTC
            pack(
                dateBaseMs + 23 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                tzId), // 2023-01-15 23:59:59.999 PST in UTC
        },
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
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

    auto tzId = tz::getTimeZoneID("America/Los_Angeles");
    auto expected = makeNullableFlatVector<int64_t>(
        {pack(8 * kMillisInHour, tzId),
         std::nullopt,
         pack(12 * kMillisInHour + 8 * kMillisInHour, tzId),
         std::nullopt,
         pack(
             23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                 999 + 8 * kMillisInHour,
             tzId)},
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }

  {
    // Test with different timezone (Asia/Shanghai, +8 hours from UTC)
    auto input = makeFlatVector<int64_t>(
        {
            0,
            12 * kMillisInHour,
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");

    auto tzId = tz::getTimeZoneID("Asia/Shanghai");
    // Shanghai is +8 hours from UTC, so 00:00:00 Shanghai = 16:00:00 UTC (prev
    // day)
    auto expected = makeFlatVector<int64_t>(
        {
            pack(
                -8 * kMillisInHour,
                tzId), // 00:00:00 CST -> 16:00:00 UTC (prev day)
            pack(
                12 * kMillisInHour - 8 * kMillisInHour,
                tzId), // 12:00:00 CST -> 04:00:00 UTC
            pack(
                23 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999 - 8 * kMillisInHour,
                tzId), // 23:59:59.999 CST -> 15:59:59.999 UTC
        },
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }

  {
    // Test with adjustTimestampToTimezone disabled
    // When disabled, TIME values are treated as already in UTC
    auto input = makeFlatVector<int64_t>(
        {
            0,
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond + 321,
            12 * kMillisInHour,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");
    disableAdjustTimestampToTimezone();

    auto tzId = tz::getTimeZoneID("Asia/Shanghai");
    auto expected = makeFlatVector<int64_t>(
        {
            pack(0, tzId),
            pack(
                3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321,
                tzId),
            pack(12 * kMillisInHour, tzId),
        },
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }

  {
    // Test constant TIME vector cast to TIMESTAMP WITH TIME ZONE (non-null)
    // This tests the optimization path for constant vectors
    auto constantTimeVector = BaseVector::wrapInConstant(
        1000, 0, makeFlatVector<int64_t>({12 * kMillisInHour}, TIME()));

    setQueryTimeZone("America/New_York");

    auto tzId = tz::getTimeZoneID("America/New_York");
    // NY is -5 hours from UTC (EST), so 12:00:00 EST = 17:00:00 UTC

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)",
        makeRowVector({constantTimeVector}));

    // Should return a constant vector with the adjusted value
    auto expected = BaseVector::wrapInConstant(
        1000,
        0,
        makeFlatVector<int64_t>(
            {pack(12 * kMillisInHour + 5 * kMillisInHour, tzId)},
            TIMESTAMP_WITH_TIME_ZONE()));

    test::assertEqualVectors(expected, result);
    // Verify the result is actually constant-encoded (optimization worked)
    ASSERT_TRUE(result->isConstantEncoding());
  }

  {
    // Test constant TIME vector cast to TIMESTAMP WITH TIME ZONE (null)
    // This tests the optimization path for null constant vectors
    auto nullTimeVector = BaseVector::createNullConstant(TIME(), 500, pool());

    setQueryTimeZone("America/Los_Angeles");

    auto result = evaluate(
        "cast(c0 as timestamp with time zone)",
        makeRowVector({nullTimeVector}));

    // Should return a null constant vector
    auto expected =
        BaseVector::createNullConstant(TIMESTAMP_WITH_TIME_ZONE(), 500, pool());

    test::assertEqualVectors(expected, result);
    // Verify the result is actually constant-encoded (optimization worked)
    ASSERT_TRUE(result->isConstantEncoding());
    ASSERT_TRUE(result->isNullAt(0));
  }

  {
    // Test constant TIME vector with different sizes
    // This ensures the optimization correctly handles different vector sizes
    setQueryTimeZone("America/Los_Angeles");
    auto tzId = tz::getTimeZoneID("America/Los_Angeles");

    for (auto size : {1, 10, 100, 1000}) {
      auto constantTimeVector = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
               321},
              TIME()));

      auto result = evaluate(
          "cast(c0 as timestamp with time zone)",
          makeRowVector({constantTimeVector}));

      auto expected = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {pack(
                  3 * kMillisInHour + 4 * kMillisInMinute +
                      5 * kMillisInSecond + 321 + 8 * kMillisInHour,
                  tzId)},
              TIMESTAMP_WITH_TIME_ZONE()));

      test::assertEqualVectors(expected, result);
      ASSERT_TRUE(result->isConstantEncoding());
      ASSERT_EQ(result->size(), size);
    }
  }

  {
    // Test try_cast for TIME to TIMESTAMP WITH TIME ZONE
    auto input =
        makeFlatVector<int64_t>({0, 12 * kMillisInHour, 86399999}, TIME());

    setQueryTimeZone("America/Los_Angeles");

    auto tzId = tz::getTimeZoneID("America/Los_Angeles");
    auto expected = makeFlatVector<int64_t>(
        {
            pack(8 * kMillisInHour, tzId), // 00:00:00 PST -> 08:00:00 UTC
            pack(
                12 * kMillisInHour + 8 * kMillisInHour,
                tzId), // 12:00:00 PST -> 20:00:00 UTC
            pack(
                86399999 + 8 * kMillisInHour,
                tzId), // 23:59:59.999 PST -> 07:59:59.999 UTC (next day)
        },
        TIMESTAMP_WITH_TIME_ZONE());

    auto result = evaluate(
        "try_cast(c0 as timestamp with time zone)", makeRowVector({input}));
    test::assertEqualVectors(expected, result);
  }
}

TEST_F(TimestampWithTimeZoneCastTest, toTime) {
  // Test casting TIMESTAMP WITH TIME ZONE to TIME
  // The TIME value should represent the local time in the given timezone
  // This matches Presto's behavior (see TestTimestampWithTimeZone.java:34-59)

  // Test data with named timezones to verify DST handling
  // DST rules are applied based on the UTC timestamp's date, not current date
  auto input = makeNullableFlatVector<int64_t>(
      {// May 15, 2023 10:30:45.123 EDT (America/New_York in DST)
       parseTimestampWithTz("2023-05-15 10:30:45.123 America/New_York"),
       // January 10, 2023 08:15:30.500 GMT (Europe/London in standard time)
       parseTimestampWithTz("2023-01-10 08:15:30.500 Europe/London"),
       // January 15, 2023 09:30:00.000 EST (America/New_York in standard time)
       parseTimestampWithTz("2023-01-15 09:30:00.000 America/New_York"),
       // August 5, 2023 10:15:00.000 BST (Europe/London in DST)
       parseTimestampWithTz("2023-08-05 10:15:00.000 Europe/London"),
       // July 20, 2023 08:59:59.999 JST (Asia/Tokyo, no DST)
       parseTimestampWithTz("2023-07-21 08:59:59.999 Asia/Tokyo"),
       // Epoch in UTC
       parseTimestampWithTz("1970-01-01 00:00:00.000 UTC"),
       // December 24, 2023 22:00:00.000 PST (America/Los_Angeles in standard
       // time)
       parseTimestampWithTz("2023-12-24 22:00:00.000 America/Los_Angeles"),
       // Null value
       std::nullopt,
       // Presto test case: 2001-01-22 03:04:05.321 +07:09
       parseTimestampWithTz("2001-01-22 03:04:05.321 +07:09")},
      TIMESTAMP_WITH_TIME_ZONE());

  auto expected = makeNullableFlatVector<int64_t>(
      {// 10:30:45.123
       10 * kMillisInHour + 30 * kMillisInMinute + 45 * kMillisInSecond + 123,
       // 08:15:30.500
       8 * kMillisInHour + 15 * kMillisInMinute + 30 * kMillisInSecond + 500,
       // 09:30:00.000 (EST, demonstrates standard time for America/New_York)
       9 * kMillisInHour + 30 * kMillisInMinute,
       // 10:15:00.000 (BST, demonstrates DST for Europe/London)
       10 * kMillisInHour + 15 * kMillisInMinute,
       // 08:59:59.999
       8 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond + 999,
       // 00:00:00.000
       0,
       // 22:00:00.000
       22 * kMillisInHour,
       // Null value
       std::nullopt,
       // 03:04:05.321
       3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond + 321},
      TIME());

  testCast(input, expected);
}

} // namespace
} // namespace facebook::velox
