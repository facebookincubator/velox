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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class ToTimestampNtzTest : public SparkFunctionBaseTest {
 protected:
  void setAnsiEnabled(bool value) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          value ? "true" : "false"}});
  }

  void enableLegacyFormatter() {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kLegacyDateFormatter),
          "true"}});
  }

  std::optional<Timestamp> evalToTimestampNtzWithFormat(
      std::optional<std::string> input,
      std::optional<std::string> format) {
    return evaluateOnce<Timestamp>(
        "to_timestamp_ntz(c0, c1)",
        {VARCHAR(), VARCHAR()},
        std::move(input),
        std::move(format));
  }
};

TEST_F(ToTimestampNtzTest, sessionTimezoneIgnored) {
  setTimezone("Asia/Shanghai");

  EXPECT_EQ(
      parseTimestamp("1970-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("1970-01-01", "yyyy-MM-dd"));
  EXPECT_EQ(
      parseTimestamp("1970-01-01 08:00:00"),
      evalToTimestampNtzWithFormat(
          "1970-01-01 08:00:00", "yyyy-MM-dd HH:mm:ss"));

  // The offset parsed via the format's ZZ specifier is discarded too, same
  // as a timezone suffix in the input string with no format given.
  EXPECT_EQ(
      parseTimestamp("2021-11-22 10:54:27"),
      evalToTimestampNtzWithFormat(
          "2021-11-22 10:54:27+08:00", "yyyy-MM-dd HH:mm:ssZZ"));
}

TEST_F(ToTimestampNtzTest, withFormat) {
  EXPECT_EQ(
      parseTimestamp("1970-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("1970-01-01", "yyyy-MM-dd"));
  EXPECT_EQ(
      std::nullopt, evalToTimestampNtzWithFormat("1970-01-01", "yyyy-MM"));
  EXPECT_EQ(
      std::nullopt, evalToTimestampNtzWithFormat(std::nullopt, "yyyy-MM-dd"));
}

TEST_F(ToTimestampNtzTest, withFormatAnsiInvalidInput) {
  setAnsiEnabled(false);
  EXPECT_EQ(
      std::nullopt, evalToTimestampNtzWithFormat("1970-01-01", "yyyy-MM"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("1970-01-01", "yyyy-MM"),
      "Invalid date format");
}

TEST_F(ToTimestampNtzTest, withFormatVector) {
  auto input = makeFlatVector<std::string>(
      {"1970-01-01", "2016-12-31 00:12:00", "1970-01-01"});
  auto format = makeFlatVector<std::string>(
      {"yyyy-MM-dd", "yyyy-MM-dd HH:mm:ss", "yyyy-MM"});
  auto data = makeRowVector({input, format});

  setAnsiEnabled(false);
  auto result = evaluate("to_timestamp_ntz(c0, c1)", data);
  velox::test::assertEqualVectors(
      makeNullableFlatVector<Timestamp>(
          {parseTimestamp("1970-01-01 00:00:00"),
           parseTimestamp("2016-12-31 00:12:00"),
           std::nullopt},
          TIMESTAMP_UTC()),
      result);

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evaluate("to_timestamp_ntz(c0, c1)", data), "Invalid date format");
}

TEST_F(ToTimestampNtzTest, outputType) {
  auto dataWithFormat = makeRowVector(
      {makeFlatVector<std::string>({"1970-01-01"}),
       makeFlatVector<std::string>({"yyyy-MM-dd"})});
  auto resultWithFormat = evaluate("to_timestamp_ntz(c0, c1)", dataWithFormat);
  ASSERT_TRUE(resultWithFormat->type()->equivalent(*TIMESTAMP_UTC()));
}

// Unlike get_timestamp, this never honors spark.legacy_date_formatter.
TEST_F(ToTimestampNtzTest, legacyFormatterIgnored) {
  enableLegacyFormatter();

  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("2020/01/24", "AA/MM/dd"),
      "Specifier A is not supported");
}

// Fraction-of-second digits truncate (not round) past the sixth digit.
TEST_F(ToTimestampNtzTest, fractionalSecondPrecision) {
  EXPECT_EQ(
      Timestamp(0, 100'000'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1", "yyyy-MM-dd HH:mm:ss.S"));
  EXPECT_EQ(
      Timestamp(0, 123'456'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.123456", "yyyy-MM-dd HH:mm:ss.SSSSSS"));
  // 7-9 S's accept matching-width input but still truncate.
  EXPECT_EQ(
      Timestamp(0, 123'456'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1234567", "yyyy-MM-dd HH:mm:ss.SSSSSSS"));
  EXPECT_EQ(
      Timestamp(0, 123'456'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.12345678", "yyyy-MM-dd HH:mm:ss.SSSSSSSS"));
  EXPECT_EQ(
      Timestamp(0, 123'456'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.123456789", "yyyy-MM-dd HH:mm:ss.SSSSSSSSS"));
  // An S-count is a maximum; fewer digits are zero-padded.
  EXPECT_EQ(
      Timestamp(0, 100'000'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1", "yyyy-MM-dd HH:mm:ss.SSSSSS"));
  EXPECT_EQ(
      Timestamp(0, 123'000'000),
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.123", "yyyy-MM-dd HH:mm:ss.SSSSSS"));
}

// Vectorized: 6-9 matching S/digits truncate; a mismatched count rejects.
TEST_F(ToTimestampNtzTest, fractionalSecondVector) {
  auto input = makeFlatVector<std::string>(
      {"1970-01-01 00:00:00.123456",
       "1970-01-01 00:00:00.1234567",
       "1970-01-01 00:00:00.12345678",
       "1970-01-01 00:00:00.123456789",
       "1970-01-01 00:00:00.1234567"});
  auto format = makeFlatVector<std::string>(
      {"yyyy-MM-dd HH:mm:ss.SSSSSS",
       "yyyy-MM-dd HH:mm:ss.SSSSSSS",
       "yyyy-MM-dd HH:mm:ss.SSSSSSSS",
       "yyyy-MM-dd HH:mm:ss.SSSSSSSSS",
       "yyyy-MM-dd HH:mm:ss.SSSSSS"});
  auto data = makeRowVector({input, format});

  setAnsiEnabled(false);
  auto result = evaluate("to_timestamp_ntz(c0, c1)", data);
  velox::test::assertEqualVectors(
      makeNullableFlatVector<Timestamp>(
          {Timestamp(0, 123'456'000),
           Timestamp(0, 123'456'000),
           Timestamp(0, 123'456'000),
           Timestamp(0, 123'456'000),
           std::nullopt},
          TIMESTAMP_UTC()),
      result);
}

// A pattern's S-count bounds how many fractional digits are consumed.
TEST_F(ToTimestampNtzTest, fractionalSecondDigitCountMismatch) {
  setAnsiEnabled(false);
  EXPECT_EQ(
      std::nullopt,
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1234567", "yyyy-MM-dd HH:mm:ss.SSSSSS"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1234567", "yyyy-MM-dd HH:mm:ss.SSSSSS"),
      "Invalid date format");
}

// A two-digit year always resolves to 2000 + value, unlike get_timestamp.
TEST_F(ToTimestampNtzTest, twoDigitYear) {
  EXPECT_EQ(
      parseTimestamp("2070-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("70-01-01", "yy-MM-dd"));
  EXPECT_EQ(
      parseTimestamp("2099-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("99-01-01", "yy-MM-dd"));
  EXPECT_EQ(
      parseTimestamp("2000-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("00-01-01", "yy-MM-dd"));

  enableLegacyFormatter();
  EXPECT_EQ(
      parseTimestamp("2070-01-01 00:00:00"),
      evalToTimestampNtzWithFormat("70-01-01", "yy-MM-dd"));
}

// The 2-arg (formatted) form never catches overflow, even with ANSI off.
TEST_F(ToTimestampNtzTest, formattedOverflowUnconditional) {
  setAnsiEnabled(false);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("+300000-01-01", "yyyy-MM-dd"),
      "Timestamp out of range");

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("+300000-01-01", "yyyy-MM-dd"),
      "Timestamp out of range");
}

// Exact microsecond-range boundaries for the 2-arg (formatted) form.
TEST_F(ToTimestampNtzTest, formattedBoundary) {
  const auto* pattern = "yyyy-MM-dd HH:mm:ss.SSSSSS";
  EXPECT_EQ(
      Timestamp(9223372036854, 775807000),
      evalToTimestampNtzWithFormat("+294247-01-10 04:00:54.775807", pattern));
  EXPECT_EQ(
      Timestamp(-9223372036855, 224192000),
      evalToTimestampNtzWithFormat("-290308-12-21 19:59:05.224192", pattern));

  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("+294247-01-10 04:00:54.775808", pattern),
      "Timestamp out of range");
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("-290308-12-21 19:59:05.224191", pattern),
      "Timestamp out of range");
}

// Vectorized two-digit year, unaffected by the legacy formatter flag.
TEST_F(ToTimestampNtzTest, twoDigitYearVector) {
  auto input = makeFlatVector<std::string>({"70-01-01", "99-01-01"});
  auto format = makeFlatVector<std::string>({"yy-MM-dd", "yy-MM-dd"});
  auto data = makeRowVector({input, format});
  auto expected = makeNullableFlatVector<Timestamp>(
      {parseTimestamp("2070-01-01 00:00:00"),
       parseTimestamp("2099-01-01 00:00:00")},
      TIMESTAMP_UTC());

  velox::test::assertEqualVectors(
      expected, evaluate("to_timestamp_ntz(c0, c1)", data));

  enableLegacyFormatter();
  velox::test::assertEqualVectors(
      expected, evaluate("to_timestamp_ntz(c0, c1)", data));
}

// Unlike Joda, a missing year defaults to 1970, not 2000.
TEST_F(ToTimestampNtzTest, missingYearDefaultsTo1970) {
  for (bool ansiEnabled : {false, true}) {
    setAnsiEnabled(ansiEnabled);
    EXPECT_EQ(
        parseTimestamp("1970-12-25 00:00:00"),
        evalToTimestampNtzWithFormat("12-25", "MM-dd"));
    EXPECT_EQ(
        parseTimestamp("1970-04-09 00:00:00"),
        evalToTimestampNtzWithFormat("04-09", "MM-dd"));
    EXPECT_EQ(
        parseTimestamp("1970-04-09 00:00:00"),
        evalToTimestampNtzWithFormat("Apr 09", "MMM dd"));
    EXPECT_EQ(
        parseTimestamp("1970-01-09 00:00:00"),
        evalToTimestampNtzWithFormat("09", "dd"));
    // 1970 is not a leap year, so day-of-year 100 is April 10.
    EXPECT_EQ(
        parseTimestamp("1970-04-10 00:00:00"),
        evalToTimestampNtzWithFormat("100", "D"));
  }
}

// 1970 is not a leap year, so the resolved date rejects Feb 29.
TEST_F(ToTimestampNtzTest, missingYearRejectsFeb29) {
  setAnsiEnabled(false);
  EXPECT_EQ(std::nullopt, evalToTimestampNtzWithFormat("02-29", "MM-dd"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat("02-29", "MM-dd"),
      "Value 29 for dayOfMonth must be in the range [1,28] for year 1970 "
      "and month 2.");
}

// Y, w, e, E and C are rejected for SPARK_ISO, matching Spark's own
// rejection. x is also rejected, though Spark accepts it as a zone offset
// (see the SPARK_ISO comment in DateTimeFormatter.h). Invalid-pattern
// throws are unconditional, independent of ANSI mode.
TEST_F(ToTimestampNtzTest, rejectedPatternLetters) {
  const std::vector<std::pair<std::string, std::string>> patterns = {
      {"YYYY-MM-dd", "Specifier Y is not supported."},
      {"yyyy-MM-dd w", "Specifier w is not supported."},
      {"yyyy-MM-dd e", "Specifier e is not supported."},
      {"yyyy-MM-dd E", "Specifier E is not supported."},
      {"CC-MM-dd", "Specifier C is not supported."},
      {"yyyy-MM-ddx", "Specifier x is not supported."},
  };
  for (const auto& [pattern, message] : patterns) {
    setAnsiEnabled(false);
    VELOX_ASSERT_THROW(
        evalToTimestampNtzWithFormat("2020-01-01", pattern), message);

    setAnsiEnabled(true);
    VELOX_ASSERT_THROW(
        evalToTimestampNtzWithFormat("2020-01-01", pattern), message);
  }
}

// S is capped at 9 pattern letters, matching Spark's own rejection of
// patterns with more than 9.
TEST_F(ToTimestampNtzTest, fractionalSecondTooManyDigits) {
  setAnsiEnabled(false);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1234567890", "yyyy-MM-dd HH:mm:ss.SSSSSSSSSS"),
      "Specifier S supports at most 9 pattern letters, got: 10");

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtzWithFormat(
          "1970-01-01 00:00:00.1234567890123456789",
          "yyyy-MM-dd HH:mm:ss.SSSSSSSSSSSSSSSSSSS"),
      "Specifier S supports at most 9 pattern letters, got: 19");
}

// Vectorized: missing year defaults to 1970 (rows 0-2); the resulting
// invalid date (Feb 29 in 1970, row 3) is null with ANSI off, throws with
// ANSI on.
TEST_F(ToTimestampNtzTest, missingYearVector) {
  auto input = makeFlatVector<std::string>({"12-25", "100", "04-09", "02-29"});
  auto format = makeFlatVector<std::string>({"MM-dd", "D", "MM-dd", "MM-dd"});
  auto data = makeRowVector({input, format});

  setAnsiEnabled(false);
  auto result = evaluate("to_timestamp_ntz(c0, c1)", data);
  velox::test::assertEqualVectors(
      makeNullableFlatVector<Timestamp>(
          {parseTimestamp("1970-12-25 00:00:00"),
           parseTimestamp("1970-04-10 00:00:00"),
           parseTimestamp("1970-04-09 00:00:00"),
           std::nullopt},
          TIMESTAMP_UTC()),
      result);

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evaluate("to_timestamp_ntz(c0, c1)", data),
      "Value 29 for dayOfMonth must be in the range [1,28] for year 1970 "
      "and month 2.");
}

// Vectorized: a rejected letter throws, unconditionally (per
// rejectedPatternLetters above).
TEST_F(ToTimestampNtzTest, rejectedPatternLetterVector) {
  auto input = makeFlatVector<std::string>({"2020-01-01", "2020-01-01"});
  auto format = makeFlatVector<std::string>({"yyyy-MM-dd", "yyyy-MM-dd w"});
  auto data = makeRowVector({input, format});

  setAnsiEnabled(false);
  VELOX_ASSERT_THROW(
      evaluate("to_timestamp_ntz(c0, c1)", data),
      "Specifier w is not supported.");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
