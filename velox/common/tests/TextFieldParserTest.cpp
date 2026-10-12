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

#include <locale.h>
#include <cmath>
#include <limits>

#include "velox/common/text/TextFieldParser.h"

namespace facebook::velox::text {
namespace {

TEST(TextFieldParserTest, integer) {
  EXPECT_EQ(
      TextFieldParser::parseNarrowInteger<int32_t>("123"),
      std::optional<int32_t>{123});
  EXPECT_EQ(
      TextFieldParser::parseNarrowInteger<int32_t>("-123.45"),
      std::optional<int32_t>{-123});
  EXPECT_EQ(
      TextFieldParser::parseNarrowInteger<int32_t>("123."),
      std::optional<int32_t>{123});
  EXPECT_EQ(
      TextFieldParser::parseNarrowInteger<int64_t>("-9223372036854775808"),
      std::optional<int64_t>{std::numeric_limits<int64_t>::min()});
  EXPECT_EQ(
      TextFieldParser::parseNarrowInteger<int16_t>("32767"),
      std::optional<int16_t>{std::numeric_limits<int16_t>::max()});

  EXPECT_FALSE(TextFieldParser::parseNarrowInteger<int8_t>("128").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseNarrowInteger<int16_t>("32768").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseNarrowInteger<int32_t>("+123").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseNarrowInteger<int32_t>(" 123").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseNarrowInteger<int32_t>("123.4.5").has_value());
  EXPECT_FALSE(
      TextFieldParser::parseNarrowInteger<int64_t>("9223372036854775808")
          .has_value());
}

TEST(TextFieldParserTest, boolean) {
  EXPECT_EQ(TextFieldParser::parseBoolean("true"), std::optional<bool>{true});
  EXPECT_EQ(TextFieldParser::parseBoolean("FALSE"), std::optional<bool>{false});
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
  EXPECT_TRUE(std::isnan(*TextFieldParser::parseFloatingPoint<double>("nan")));
  EXPECT_EQ(
      *TextFieldParser::parseFloatingPoint<double>("-Inf"),
      -std::numeric_limits<double>::infinity());
  EXPECT_FALSE(
      TextFieldParser::parseFloatingPoint<double>("value").has_value());

  const std::string longFraction = "0." + std::string(130, '0') + "1";
  const auto longValue =
      TextFieldParser::parseFloatingPoint<double>(longFraction);
  ASSERT_TRUE(longValue.has_value());
  EXPECT_DOUBLE_EQ(*longValue, 1e-131);
}

TEST(TextFieldParserTest, floatingPointUsesCNumericLocale) {
  locale_t commaLocale{nullptr};
  for (const char* localeName :
       {"de_DE.UTF-8", "fr_FR.UTF-8", "es_ES.UTF-8", "it_IT.UTF-8"}) {
    commaLocale = newlocale(LC_NUMERIC_MASK, localeName, nullptr);
    if (commaLocale != nullptr) {
      break;
    }
  }
  if (commaLocale == nullptr) {
    GTEST_SKIP() << "No comma-decimal locale is installed.";
  }

  const locale_t previousLocale = uselocale(commaLocale);
  const auto dotDecimal = TextFieldParser::parseFloatingPoint<double>("1.5");
  const auto commaDecimal = TextFieldParser::parseFloatingPoint<double>("1,5");
  uselocale(previousLocale);
  freelocale(commaLocale);

  EXPECT_EQ(dotDecimal, std::optional<double>{1.5});
  EXPECT_FALSE(commaDecimal.has_value());
}

TEST(TextFieldParserTest, typedConversions) {
  auto decimal = TextFieldParser::parseDecimal<int64_t>("123.45", 10, 2);
  ASSERT_TRUE(decimal.hasValue());
  EXPECT_EQ(decimal.value(), 12345);

  auto scientific = TextFieldParser::parseDecimal<int64_t>("4.5e-2", 10, 1);
  ASSERT_TRUE(scientific.hasValue());
  EXPECT_EQ(scientific.value(), 0);

  auto expandedScientific = TextFieldParser::parseDecimal<int64_t>(
      "0.0000000000000000000000000000000000000001e40", 10, 2);
  ASSERT_TRUE(expandedScientific.hasValue());
  EXPECT_EQ(expandedScientific.value(), 100);
  EXPECT_FALSE(
      TextFieldParser::parseDecimal<int64_t>("1e-2147483648", 10, 2)
          .hasValue());

  auto longDecimal =
      TextFieldParser::parseDecimal<int128_t>("12345678901234567890.12", 22, 2);
  ASSERT_TRUE(longDecimal.hasValue());
  EXPECT_EQ(longDecimal.value(), HugeInt::parse("1234567890123456789012"));
  EXPECT_FALSE(
      TextFieldParser::parseDecimal<int128_t>(
          "999999999999999999999999999999999999999", 38, 0)
          .hasValue());

  auto date = TextFieldParser::parseDate("1970-01-01");
  ASSERT_TRUE(date.hasValue());
  EXPECT_EQ(date.value(), 0);
  EXPECT_FALSE(TextFieldParser::parseDate("not-a-date").hasValue());
  EXPECT_FALSE(TextFieldParser::parseDate("2500000000-01-01").hasValue());

  auto timestamp = TextFieldParser::parseTimestamp("1970-01-01 00:00:00");
  ASSERT_TRUE(timestamp.hasValue());
  EXPECT_EQ(timestamp.value(), Timestamp(28'800, 0));
  EXPECT_FALSE(
      TextFieldParser::parseTimestamp("2024-03-10 02:30:00").hasValue());
  EXPECT_FALSE(TextFieldParser::parseTimestamp("not-a-timestamp").hasValue());
  EXPECT_FALSE(
      TextFieldParser::parseTimestamp("2500000000-01-01 00:00:00").hasValue());

  std::string decoded;
  TextFieldParser::parseVarbinary("aGVsbG8=", decoded);
  EXPECT_EQ(decoded, "hello");
  TextFieldParser::parseVarbinary("not base64", decoded);
  EXPECT_EQ(decoded, "not base64");
}

} // namespace
} // namespace facebook::velox::text
