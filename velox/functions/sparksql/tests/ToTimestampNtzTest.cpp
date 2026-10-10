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

} // namespace
} // namespace facebook::velox::functions::sparksql::test
