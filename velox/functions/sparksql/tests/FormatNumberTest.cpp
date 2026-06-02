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
#include <cmath>
#include <limits>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {

class FormatNumberTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<std::string> formatNumber(T value, int32_t decimalPlaces) {
    return evaluateOnce<std::string>(
        "format_number(c0, c1)",
        std::optional<T>(value),
        std::optional<int32_t>(decimalPlaces));
  }

  std::optional<std::string> formatInt32(int32_t value, int32_t d) {
    return formatNumber<int32_t>(value, d);
  }

  std::optional<std::string> formatInt64(int64_t value, int32_t d) {
    return formatNumber<int64_t>(value, d);
  }

  std::optional<std::string> formatDouble(double value, int32_t d) {
    return formatNumber<double>(value, d);
  }

  std::optional<std::string> formatFloat(float value, int32_t d) {
    return formatNumber<float>(value, d);
  }
};

TEST_F(FormatNumberTest, integers) {
  EXPECT_EQ(formatInt32(12345, 0), "12,345");
  EXPECT_EQ(formatInt32(12345, 2), "12,345.00");
  EXPECT_EQ(formatInt32(0, 0), "0");
  EXPECT_EQ(formatInt32(0, 3), "0.000");
  EXPECT_EQ(formatInt32(-12345, 0), "-12,345");
  EXPECT_EQ(formatInt32(-12345, 2), "-12,345.00");
  EXPECT_EQ(formatInt32(5, 2), "5.00");

  EXPECT_EQ(formatInt64(1234567890, 0), "1,234,567,890");
  EXPECT_EQ(formatInt64(1234567890, 4), "1,234,567,890.0000");
  EXPECT_EQ(formatInt64(-9876543210LL, 0), "-9,876,543,210");

  EXPECT_EQ(formatNumber<int8_t>(127, 0), "127");
  EXPECT_EQ(formatNumber<int8_t>(-128, 0), "-128");
  EXPECT_EQ(formatNumber<int8_t>(42, 1), "42.0");

  EXPECT_EQ(formatNumber<int16_t>(1234, 0), "1,234");
  EXPECT_EQ(formatNumber<int16_t>(-1234, 2), "-1,234.00");
}

TEST_F(FormatNumberTest, bigintPrecision) {
  // Values above 2^53 are formatted exactly without double conversion.
  EXPECT_EQ(formatInt64(9007199254740993LL, 0), "9,007,199,254,740,993");
  EXPECT_EQ(
      formatInt64(std::numeric_limits<int64_t>::max(), 0),
      "9,223,372,036,854,775,807");
  EXPECT_EQ(
      formatInt64(std::numeric_limits<int64_t>::min(), 0),
      "-9,223,372,036,854,775,808");
  EXPECT_EQ(formatInt64(9007199254740993LL, 2), "9,007,199,254,740,993.00");
}

TEST_F(FormatNumberTest, floatingPoint) {
  EXPECT_EQ(formatDouble(12345.678, 2), "12,345.68");
  EXPECT_EQ(formatDouble(12345.678, 0), "12,346");
  EXPECT_EQ(formatDouble(0.123, 3), "0.123");
  EXPECT_EQ(formatDouble(-1234.5, 1), "-1,234.5");
  EXPECT_EQ(formatDouble(12831273.234, 3), "12,831,273.234");

  EXPECT_EQ(formatFloat(1234.5f, 1), "1,234.5");
  EXPECT_EQ(formatFloat(0.0f, 2), "0.00");
  EXPECT_EQ(formatFloat(-99.99f, 1), "-100.0");
}

TEST_F(FormatNumberTest, halfEvenRounding) {
  // HALF_EVEN (banker's rounding) matches Java DecimalFormat default.
  EXPECT_EQ(formatDouble(0.5, 0), "0");
  EXPECT_EQ(formatDouble(1.5, 0), "2");
  EXPECT_EQ(formatDouble(2.5, 0), "2");
  EXPECT_EQ(formatDouble(3.5, 0), "4");
  EXPECT_EQ(formatDouble(5.5, 0), "6");
  EXPECT_EQ(formatDouble(5.4, 0), "5");
  EXPECT_EQ(formatDouble(5.6, 0), "6");
  EXPECT_EQ(formatDouble(-0.5, 0), "-0");
  EXPECT_EQ(formatDouble(-1.5, 0), "-2");
  EXPECT_EQ(formatDouble(-2.5, 0), "-2");
}

TEST_F(FormatNumberTest, binaryTieRounding) {
  // These lock fmt's HALF_EVEN contract for double-representable tie cases.
  // Verified against vanilla Spark (sha 12b25952).
  EXPECT_EQ(formatDouble(1.225, 2), "1.23");
  EXPECT_EQ(formatDouble(2.675, 2), "2.67");
  EXPECT_EQ(formatDouble(1.15, 1), "1.1");
  EXPECT_EQ(formatDouble(1.25, 1), "1.2");
  EXPECT_EQ(formatDouble(-1.225, 2), "-1.23");
  EXPECT_EQ(formatDouble(-2.675, 2), "-2.67");
}

TEST_F(FormatNumberTest, negativeDecimalPlaces) {
  // d < 0 returns null per Spark semantics.
  EXPECT_EQ(
      (evaluateOnce<std::string, int32_t, int32_t>(
          "format_number(c0, c1)", 12345, -1)),
      std::nullopt);
  EXPECT_EQ(
      (evaluateOnce<std::string, double, int32_t>(
          "format_number(c0, c1)", 12345.6, -2)),
      std::nullopt);
}

TEST_F(FormatNumberTest, boundaryValues) {
  EXPECT_EQ(formatInt32(0, 0), "0");
  EXPECT_EQ(formatInt32(1, 0), "1");
  EXPECT_EQ(formatInt32(-1, 0), "-1");
  EXPECT_EQ(formatInt32(999, 0), "999");
  EXPECT_EQ(formatInt32(1000, 0), "1,000");
  EXPECT_EQ(formatInt32(999999, 0), "999,999");
  EXPECT_EQ(formatInt32(1000000, 0), "1,000,000");
}

TEST_F(FormatNumberTest, nanAndInfinity) {
  // Verified against Spark's FormatNumber which passes directly to
  // DecimalFormat.format(). DecimalFormat uses DecimalFormatSymbols for
  // NaN ("NaN") and infinity ("\u221E") in US locale.
  EXPECT_EQ(formatDouble(std::numeric_limits<double>::quiet_NaN(), 2), "NaN");
  EXPECT_EQ(formatDouble(std::numeric_limits<double>::infinity(), 2), "\u221E");
  EXPECT_EQ(
      formatDouble(-std::numeric_limits<double>::infinity(), 2), "-\u221E");
  EXPECT_EQ(formatFloat(std::numeric_limits<float>::quiet_NaN(), 0), "NaN");
  EXPECT_EQ(formatFloat(std::numeric_limits<float>::infinity(), 0), "\u221E");
}

TEST_F(FormatNumberTest, largeValues) {
  EXPECT_EQ(
      formatInt32(std::numeric_limits<int32_t>::max(), 0), "2,147,483,647");
  EXPECT_EQ(
      formatInt32(std::numeric_limits<int32_t>::min(), 0), "-2,147,483,648");
}

TEST_F(FormatNumberTest, negativeZero) {
  // Java DecimalFormat preserves the sign for -0.0.
  // Verified against Spark: SELECT format_number(-0.0d, 0) → "-0".
  EXPECT_EQ(formatDouble(-0.0, 0), "-0");
  EXPECT_EQ(formatDouble(-0.0, 2), "-0.00");
  EXPECT_EQ(formatDouble(-0.0, 5), "-0.00000");
  // Negative values that round to zero also preserve the sign.
  EXPECT_EQ(formatDouble(-0.4, 0), "-0");
  EXPECT_EQ(formatDouble(-0.0000001, 0), "-0");
  EXPECT_EQ(formatDouble(-0.0000001, 3), "-0.000");
}

TEST_F(FormatNumberTest, roundingCarry) {
  EXPECT_EQ(formatDouble(999.9, 0), "1,000");
  EXPECT_EQ(formatDouble(999.95, 1), "1,000.0");
  EXPECT_EQ(formatDouble(999999.9, 0), "1,000,000");
  EXPECT_EQ(formatDouble(-999.9, 0), "-1,000");
}

TEST_F(FormatNumberTest, largeFractionDigits) {
  // Java DecimalFormat caps fraction digits at 340.
  auto result = formatDouble(1.0, 340);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 342u);

  // d > 340 is capped at 340.
  auto resultCapped = formatDouble(1.0, 400);
  ASSERT_TRUE(resultCapped.has_value());
  EXPECT_EQ(resultCapped->size(), 342u);

  auto intResult = formatInt64(1, 400);
  ASSERT_TRUE(intResult.has_value());
  EXPECT_EQ(intResult->size(), 342u);
}

} // namespace facebook::velox::functions::sparksql::test
