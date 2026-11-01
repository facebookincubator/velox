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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>

#include "velox/common/text/TextFieldParser.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::text {
namespace {

TEST(TextFieldParserTest, integer) {
  EXPECT_EQ(
      TextFieldParser::parseInteger<int32_t>("123"),
      std::optional<int32_t>{123});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int32_t>("-123.45"),
      std::optional<int32_t>{-123});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int32_t>("123."),
      std::optional<int32_t>{123});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int64_t>("-9223372036854775808"),
      std::optional<int64_t>{std::numeric_limits<int64_t>::min()});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int16_t>("32767"),
      std::optional<int16_t>{std::numeric_limits<int16_t>::max()});

  EXPECT_FALSE(TextFieldParser::parseInteger<int8_t>("128").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int16_t>("32768").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>("+123").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>(" 123").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>("123.4.5").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseInteger<int64_t>("9223372036854775808")
          .has_value());

  EXPECT_EQ(
      TextFieldParser::parseInteger<int8_t>("-128"),
      std::optional<int8_t>{std::numeric_limits<int8_t>::min()});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int32_t>("00012"),
      std::optional<int32_t>{12});
  EXPECT_EQ(
      TextFieldParser::parseInteger<int32_t>("-0"), std::optional<int32_t>{0});
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>("").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>("-").has_value());
  EXPECT_FALSE(TextFieldParser::parseInteger<int32_t>("12a").has_value());
  // An embedded null byte is not a valid trailing character.
  const std::string_view embeddedNull{"1\0002", 3};
  EXPECT_FALSE(
      TextFieldParser::parseInteger<int32_t>(embeddedNull).has_value());
}

TEST(TextFieldParserTest, boolean) {
  EXPECT_EQ(TextFieldParser::parseBoolean("true"), std::optional<bool>{true});
  EXPECT_EQ(TextFieldParser::parseBoolean("TRUE"), std::optional<bool>{true});
  EXPECT_EQ(TextFieldParser::parseBoolean("FALSE"), std::optional<bool>{false});
  EXPECT_EQ(TextFieldParser::parseBoolean("fAlSe"), std::optional<bool>{false});
  EXPECT_FALSE(TextFieldParser::parseBoolean("").has_value());
  EXPECT_FALSE(TextFieldParser::parseBoolean("1").has_value());
  EXPECT_FALSE(TextFieldParser::parseBoolean(" true").has_value());
}

TEST(TextFieldParserTest, floatingPoint) {
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>(".5"),
      std::optional<double>{0.5});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>(" 1.25 "),
      std::optional<double>{1.25});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<float>("1.25"),
      std::optional<float>{1.25f});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("0x1.8p1"),
      std::optional<double>{3.0});
  EXPECT_TRUE(std::isnan(*TextFieldParser::parseFloatingPoint<double>("nan")));
  EXPECT_EQ(
      *TextFieldParser::parseFloatingPoint<double>("-Inf"),
      -std::numeric_limits<double>::infinity());
  EXPECT_FALSE(
      TextFieldParser::parseFloatingPoint<double>("value").has_value());
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("\t2.5"),
      std::optional<double>{2.5});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("2.5\t"),
      std::optional<double>{2.5});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>(" .5 "),
      std::optional<double>{0.5});
  EXPECT_FALSE(TextFieldParser::parseFloatingPoint<double>("   ").has_value());
  EXPECT_FALSE(TextFieldParser::parseFloatingPoint<double>("1.5x").has_value());
  // An embedded null byte stops the scan before the end of the field.
  const std::string_view embeddedNull{"1.5\0001", 5};
  EXPECT_FALSE(
      TextFieldParser::parseFloatingPoint<double>(embeddedNull).has_value());
  // Out-of-range values are kept: overflow yields infinity and underflow a
  // denormal.
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("1e400"),
      std::optional<double>{std::numeric_limits<double>::infinity()});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<float>("1e39"),
      std::optional<float>{std::numeric_limits<float>::infinity()});
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("4.9e-324"),
      std::optional<double>{std::numeric_limits<double>::denorm_min()});
  // Only the listed special values are accepted.
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("INFINITY"),
      std::optional<double>{std::numeric_limits<double>::infinity()});
  EXPECT_FALSE(TextFieldParser::parseFloatingPoint<double>("-NaN").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseFloatingPoint<double>("Infinityx").has_value());
  // Unlike integers, a leading '+' is accepted.
  EXPECT_EQ(
      TextFieldParser::parseFloatingPoint<double>("+1.5"),
      std::optional<double>{1.5});

  // Inputs shorter than 128 bytes are parsed from a stack buffer, longer ones
  // from a heap buffer.
  for (const auto size : {127, 128, 133}) {
    SCOPED_TRACE(size);
    const std::string longFraction = "0." + std::string(size - 3, '0') + "1";
    ASSERT_EQ(longFraction.size(), static_cast<size_t>(size));
    const auto longValue =
        TextFieldParser::parseFloatingPoint<double>(longFraction);
    ASSERT_TRUE(longValue.has_value());
    EXPECT_DOUBLE_EQ(*longValue, std::pow(10.0, -(size - 2)));
  }
}

TEST(TextFieldParserTest, decimal) {
  auto decimal = TextFieldParser::parseDecimal<int64_t>("123.45", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), 12'345);

  decimal = TextFieldParser::parseDecimal<int64_t>("-123.45", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), -12'345);

  // Extra fractional digits round half away from zero.
  decimal = TextFieldParser::parseDecimal<int64_t>("1.235", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), 124);
  decimal = TextFieldParser::parseDecimal<int64_t>("-1.235", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), -124);
  decimal = TextFieldParser::parseDecimal<int64_t>("1.234", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), 123);

  auto longDecimal =
      TextFieldParser::parseDecimal<int128_t>("12345678901234567890.12", 22, 2);
  ASSERT_TRUE(longDecimal.hasValue());
  EXPECT_EQ(longDecimal.value(), HugeInt::parse("1234567890123456789012"));

  EXPECT_FALSE(
      TextFieldParser::parseDecimal<int128_t>(
          "999999999999999999999999999999999999999", 38, 0)
          .hasValue());
  EXPECT_FALSE(TextFieldParser::parseDecimal<int64_t>("abc", 10, 2).hasValue());
}

TEST(TextFieldParserTest, date) {
  auto date = TextFieldParser::parseDate("1970-01-01");
  ASSERT_TRUE(date.hasValue());
  EXPECT_EQ(date.value(), 0);

  date = TextFieldParser::parseDate("1969-12-31");
  ASSERT_TRUE(date.hasValue());
  EXPECT_EQ(date.value(), -1);

  EXPECT_FALSE(TextFieldParser::parseDate("not-a-date").hasValue());
  date = TextFieldParser::parseDate("2500000000-01-01");
  ASSERT_TRUE(date.hasError());
  EXPECT_TRUE(date.error().isUserError());
  EXPECT_THAT(date.error().message(), testing::HasSubstr("integer overflow"));
}

TEST(TextFieldParserTest, timestamp) {
  const auto& losAngeles = *tz::locateZone("America/Los_Angeles");
  const auto& utc = *tz::locateZone("UTC");

  auto timestamp =
      TextFieldParser::parseTimestamp("1970-01-01 00:00:00", losAngeles);
  ASSERT_TRUE(timestamp.hasValue());
  EXPECT_EQ(timestamp.value(), Timestamp(28'800, 0));

  timestamp = TextFieldParser::parseTimestamp("1970-01-01 00:00:00", utc);
  ASSERT_TRUE(timestamp.hasValue());
  EXPECT_EQ(timestamp.value(), Timestamp(0, 0));

  // Ambiguous local time resolves to the earlier instant (PDT).
  timestamp =
      TextFieldParser::parseTimestamp("2024-11-03 01:30:00", losAngeles);
  ASSERT_TRUE(timestamp.hasValue());
  EXPECT_EQ(timestamp.value(), Timestamp(1'730'622'600, 0));

  // Nonexistent local time (daylight saving time gap).
  timestamp =
      TextFieldParser::parseTimestamp("2024-03-10 02:30:00", losAngeles);
  ASSERT_TRUE(timestamp.hasError());
  EXPECT_TRUE(timestamp.error().isUserError());
  EXPECT_THAT(timestamp.error().message(), testing::HasSubstr("is in a gap"));
  EXPECT_TRUE(
      TextFieldParser::parseTimestamp("2024-03-10 02:30:00", utc).hasValue());

  EXPECT_FALSE(
      TextFieldParser::parseTimestamp("not-a-timestamp", utc).hasValue());
  timestamp = TextFieldParser::parseTimestamp("2500000000-01-01 00:00:00", utc);
  ASSERT_TRUE(timestamp.hasError());
  EXPECT_TRUE(timestamp.error().isUserError());
  EXPECT_THAT(
      timestamp.error().message(), testing::HasSubstr("integer overflow"));
}

TEST(TextFieldParserTest, varbinary) {
  std::array<char, 16> decoded;
  EXPECT_EQ(TextFieldParser::parseVarbinary("aGVsbG8=", decoded), "hello");
  // Invalid base64 is returned as-is.
  EXPECT_EQ(
      TextFieldParser::parseVarbinary("not base64", decoded), "not base64");
  EXPECT_EQ(TextFieldParser::parseVarbinary("=", decoded), "=");
  // Padding-only input decodes to an empty value.
  EXPECT_EQ(TextFieldParser::parseVarbinary("====", decoded), "");
  EXPECT_EQ(TextFieldParser::parseVarbinary("", decoded), "");
}

} // namespace
} // namespace facebook::velox::text
