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

  std::optional<Timestamp> evalToTimestampNtz(
      std::optional<std::string> input) {
    return evaluateOnce<Timestamp>(
        "to_timestamp_ntz(c0)", {VARCHAR()}, std::move(input));
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

TEST_F(ToTimestampNtzTest, basic) {
  EXPECT_EQ(
      parseTimestamp("2016-12-31 00:12:00"),
      evalToTimestampNtz("2016-12-31 00:12:00"));
  EXPECT_EQ(
      parseTimestamp("1970-01-01 00:00:00"), evalToTimestampNtz("1970-01-01"));
  EXPECT_EQ(std::nullopt, evalToTimestampNtz(std::nullopt));
}

// Timezone suffix is ignored, see SPARK-37326.
TEST_F(ToTimestampNtzTest, timezoneSuffixDiscarded) {
  EXPECT_EQ(
      parseTimestamp("2021-11-22 10:54:27"),
      evalToTimestampNtz("2021-11-22 10:54:27 +08:00"));
  EXPECT_EQ(
      parseTimestamp("2021-11-22 10:54:27"),
      evalToTimestampNtz("2021-11-22 10:54:27Z"));
}

TEST_F(ToTimestampNtzTest, sessionTimezoneIgnored) {
  setTimezone("Asia/Shanghai");

  EXPECT_EQ(
      parseTimestamp("1970-01-01 00:00:00"), evalToTimestampNtz("1970-01-01"));
  EXPECT_EQ(
      parseTimestamp("1970-01-01 08:00:00"),
      evalToTimestampNtz("1970-01-01 08:00:00"));

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

TEST_F(ToTimestampNtzTest, ansiInvalidInput) {
  setAnsiEnabled(false);
  EXPECT_EQ(std::nullopt, evalToTimestampNtz("not a timestamp"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtz("not a timestamp"), "Unable to parse timestamp");
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
  auto data = makeRowVector({makeFlatVector<std::string>({"1970-01-01"})});
  auto result = evaluate("to_timestamp_ntz(c0)", data);
  ASSERT_TRUE(result->type()->equivalent(*TIMESTAMP_UTC()));

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

// The 1-arg (cast) form rejects overflow under the normal ANSI-gated policy.
TEST_F(ToTimestampNtzTest, oneArgOverflow) {
  setAnsiEnabled(false);
  EXPECT_EQ(std::nullopt, evalToTimestampNtz("300000-01-01"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtz("300000-01-01"), "Timestamp out of range");
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

// Exact microsecond-range boundaries for the 1-arg (cast) form.
TEST_F(ToTimestampNtzTest, oneArgBoundary) {
  EXPECT_EQ(
      Timestamp(9223372036854, 775807000),
      evalToTimestampNtz("294247-01-10 04:00:54.775807"));
  EXPECT_EQ(
      Timestamp(-9223372036855, 224192000),
      evalToTimestampNtz("-290308-12-21 19:59:05.224192"));

  setAnsiEnabled(false);
  EXPECT_EQ(std::nullopt, evalToTimestampNtz("294247-01-10 04:00:54.775808"));
  EXPECT_EQ(std::nullopt, evalToTimestampNtz("-290308-12-21 19:59:05.224191"));

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      evalToTimestampNtz("294247-01-10 04:00:54.775808"),
      "Timestamp out of range");
  VELOX_ASSERT_THROW(
      evalToTimestampNtz("-290308-12-21 19:59:05.224191"),
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

} // namespace
} // namespace facebook::velox::functions::sparksql::test
