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

#include <stdint.h>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class DateTimeFunctionsTest : public SparkFunctionBaseTest {
 protected:
  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  Date parseDate(const std::string& dateStr) {
    Date returnDate;
    parseTo(dateStr, returnDate);
    return returnDate;
  }
};

TEST_F(DateTimeFunctionsTest, year) {
  const auto year = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("year(c0)", date);
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
  EXPECT_EQ(1969, year(Timestamp(-1, 12300000000)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 0)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2001, year(Timestamp(998474645, 321000000)));
  EXPECT_EQ(2001, year(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeFunctionsTest, yearDate) {
  const auto year = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1970, year(Date(0)));
  EXPECT_EQ(1969, year(Date(-1)));
  EXPECT_EQ(2020, year(Date(18262)));
  EXPECT_EQ(1920, year(Date(-18262)));
}

TEST_F(DateTimeFunctionsTest, unixTimestamp) {
  const auto unixTimestamp = [&](std::optional<StringView> dateStr) {
    return evaluateOnce<int64_t>("unix_timestamp(c0)", dateStr);
  };

  EXPECT_EQ(0, unixTimestamp("1970-01-01 00:00:00"));
  EXPECT_EQ(1, unixTimestamp("1970-01-01 00:00:01"));
  EXPECT_EQ(61, unixTimestamp("1970-01-01 00:01:01"));

  setQueryTimeZone("America/Los_Angeles");

  EXPECT_EQ(28800, unixTimestamp("1970-01-01 00:00:00"));
  EXPECT_EQ(1670859931, unixTimestamp("2022-12-12 07:45:31"));

  // Empty or malformed input returns null.
  EXPECT_EQ(std::nullopt, unixTimestamp(std::nullopt));
  EXPECT_EQ(std::nullopt, unixTimestamp("1970-01-01"));
  EXPECT_EQ(std::nullopt, unixTimestamp("00:00:00"));
  EXPECT_EQ(std::nullopt, unixTimestamp(""));
  EXPECT_EQ(std::nullopt, unixTimestamp("malformed input"));
}

TEST_F(DateTimeFunctionsTest, unixTimestampCurrent) {
  // Need a mock row vector so we can pump exactly one record out.
  auto mockRowVector =
      makeRowVector({BaseVector::createNullConstant(UNKNOWN(), 1, pool())});

  // Safe bet that unix epoch (in seconds) should be between 500M and 5B.
  auto epoch = evaluateOnce<int64_t>("unix_timestamp()", mockRowVector);
  EXPECT_GE(epoch, 500'000'000);
  EXPECT_LT(epoch, 5'000'000'000);

  // Spark doesn't seem to adjust based on timezones.
  auto gmtEpoch = evaluateOnce<int64_t>("unix_timestamp()", mockRowVector);
  setQueryTimeZone("America/Los_Angeles");
  auto laEpoch = evaluateOnce<int64_t>("unix_timestamp()", mockRowVector);
  EXPECT_EQ(gmtEpoch, laEpoch);
}

TEST_F(DateTimeFunctionsTest, unixTimestampCustomFormat) {
  const auto unixTimestamp = [&](std::optional<StringView> dateStr,
                                 std::optional<StringView> formatStr) {
    return evaluateOnce<int64_t>("unix_timestamp(c0, c1)", dateStr, formatStr);
  };

  EXPECT_EQ(0, unixTimestamp("1970-01-01", "yyyy-MM-dd"));
  EXPECT_EQ(-31536000, unixTimestamp("1969", "YYYY"));
  EXPECT_EQ(86400, unixTimestamp("1970-01-02", "yyyy-MM-dd"));
  EXPECT_EQ(86410, unixTimestamp("1970-01-02 00:00:10", "yyyy-MM-dd HH:mm:ss"));

  // Literal.
  EXPECT_EQ(
      1670831131,
      unixTimestamp("2022-12-12 asd 07:45:31", "yyyy-MM-dd 'asd' HH:mm:ss"));

  // Invalid format returns null (unclosed quoted literal).
  EXPECT_EQ(
      std::nullopt,
      unixTimestamp("2022-12-12 asd 07:45:31", "yyyy-MM-dd 'asd HH:mm:ss"));
}

// unix_timestamp and to_unix_timestamp are aliases.
TEST_F(DateTimeFunctionsTest, toUnixTimestamp) {
  std::optional<StringView> dateStr = "1970-01-01 08:32:11"_sv;
  std::optional<StringView> formatStr = "YYYY-MM-dd HH:mm:ss"_sv;

  EXPECT_EQ(
      evaluateOnce<int64_t>("unix_timestamp(c0)", dateStr),
      evaluateOnce<int64_t>("to_unix_timestamp(c0)", dateStr));
  EXPECT_EQ(
      evaluateOnce<int64_t>("unix_timestamp(c0, c1)", dateStr, formatStr),
      evaluateOnce<int64_t>("to_unix_timestamp(c0, c1)", dateStr, formatStr));

  // to_unix_timestamp does not provide an overoaded without any parameters.
  EXPECT_THROW(evaluateOnce<int64_t>("to_unix_timestamp()"), VeloxUserError);
}

TEST_F(DateTimeFunctionsTest, dateAdd) {
  const auto dateAddInt32 = [&](std::optional<Date> date,
                                std::optional<int32_t> value) {
    return evaluateOnce<Date>("date_add(c0, c1)", date, value);
  };
  const auto dateAddInt16 = [&](std::optional<Date> date,
                                std::optional<int16_t> value) {
    return evaluateOnce<Date>("date_add(c0, c1)", date, value);
  };
  const auto dateAddInt8 = [&](std::optional<Date> date,
                               std::optional<int8_t> value) {
    return evaluateOnce<Date>("date_add(c0, c1)", date, value);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateAddInt32(std::nullopt, 1));
  EXPECT_EQ(std::nullopt, dateAddInt16(std::nullopt, 1));
  EXPECT_EQ(std::nullopt, dateAddInt8(std::nullopt, 1));

  // Simple tests
  EXPECT_EQ(parseDate("2019-03-01"), dateAddInt32(parseDate("2019-02-28"), 1));
  EXPECT_EQ(parseDate("2019-03-01"), dateAddInt16(parseDate("2019-02-28"), 1));
  EXPECT_EQ(parseDate("2019-03-01"), dateAddInt8(parseDate("2019-02-28"), 1));

  // Account for the last day of a year-month
  EXPECT_EQ(
      parseDate("2020-02-29"), dateAddInt32(parseDate("2019-01-30"), 395));
  EXPECT_EQ(
      parseDate("2020-02-29"), dateAddInt16(parseDate("2019-01-30"), 395));

  // Check for negative intervals
  EXPECT_EQ(
      parseDate("2019-02-28"), dateAddInt32(parseDate("2020-02-29"), -366));
  EXPECT_EQ(
      parseDate("2019-02-28"), dateAddInt16(parseDate("2020-02-29"), -366));
}

TEST_F(DateTimeFunctionsTest, dateDiff) {
  const auto dateDiff = [&](std::optional<Date> date1,
                            std::optional<Date> date2) {
    return evaluateOnce<int32_t>("date_diff(c0, c1)", date1, date2);
  };

  // Check null behaviors
  EXPECT_EQ(std::nullopt, dateDiff(Date(1), std::nullopt));
  EXPECT_EQ(std::nullopt, dateDiff(std::nullopt, Date(0)));

  // Simple tests
  EXPECT_EQ(1, dateDiff(parseDate("2019-02-28"), parseDate("2019-03-01")));

  // Account for the last day of a year-month
  EXPECT_EQ(395, dateDiff(parseDate("2019-01-30"), parseDate("2020-02-29")));

  // Check for negative intervals
  EXPECT_EQ(-366, dateDiff(parseDate("2020-02-29"), parseDate("2019-02-28")));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
