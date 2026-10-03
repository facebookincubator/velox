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

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <folly/ScopeGuard.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/BRound.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class BRoundTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  void test(T value, int32_t scale, T expected) {
    const auto actual =
        evaluateOnce<T, T, int32_t>("bround(c0, c1)", value, scale);
    ASSERT_TRUE(actual.has_value());
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual.value()));
    } else {
      EXPECT_EQ(actual.value(), expected)
          << "value: " << value << ", scale: " << scale;
    }
  }

  template <typename T>
  void testUnary(T value, T expected) {
    const auto actual = evaluateOnce<T, T>("bround(c0)", value);
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual.value(), expected) << "value: " << value;
  }

  template <typename T>
  void testPositiveZero(T value, int32_t scale) {
    const auto actual =
        evaluateOnce<T, T, int32_t>("bround(c0, c1)", value, scale);
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual.value(), static_cast<T>(0));
    EXPECT_FALSE(std::signbit(actual.value()));
  }
};

TEST_F(BRoundTest, floatingPointMidpoints) {
  for (const auto& [value, expected] : std::vector<std::pair<double, double>>{
           {0.5, 0.0},
           {1.5, 2.0},
           {2.5, 2.0},
           {3.5, 4.0},
           {4.5, 4.0},
           {-0.5, 0.0},
           {-1.5, -2.0},
           {-2.5, -2.0},
           {-3.5, -4.0},
       }) {
    test(value, 0, expected);
  }

  test(std::nextafter(0.5, 0.0), 0, 0.0);
  test(std::nextafter(0.5, 1.0), 0, 1.0);
  test(std::nextafter(-0.5, -1.0), 0, -1.0);
  test(std::nextafter(-0.5, 0.0), 0, 0.0);

  test(1.49, 0, 1.0);
  test(1.51, 0, 2.0);
  test(-1.49, 0, -1.0);
  test(-1.51, 0, -2.0);
  test(1.234, 2, 1.23);
  test(1.235, 2, 1.24);
  test(1.245, 2, 1.24);
  test(-1.235, 2, -1.24);
  test(0.575, 2, 0.58);
  test(-0.575, 2, -0.58);
  test(150.0, -2, 200.0);
  test(250.0, -2, 200.0);
  test(350.0, -2, 400.0);
  test(-150.0, -2, -200.0);
  test(-250.0, -2, -200.0);

  for (const auto& [value, expected] : std::vector<std::pair<double, double>>{
           {0.05, 0.0},
           {0.15, 0.2},
           {0.25, 0.2},
           {0.35, 0.4},
           {0.45, 0.4},
           {0.55, 0.6},
           {0.65, 0.6},
           {0.75, 0.8},
           {0.85, 0.8},
           {0.95, 1.0},
       }) {
    test(value, 1, expected);
    test(-value, 1, -expected);
  }
}

TEST_F(BRoundTest, java17DecimalConversion) {
  const double value = std::bit_cast<double>(uint64_t{0x43b657dddce43c03});
  const double expected = std::bit_cast<double>(uint64_t{0x43b657dddce43c01});
  test(value, -3, expected);

  const double overflowingLongPath =
      std::bit_cast<double>(uint64_t{0x4530000000000041});
  const double overflowingLongPathExpected =
      std::bit_cast<double>(uint64_t{0x4530000000000040});
  test(overflowingLongPath, -10, overflowingLongPathExpected);
  test(-overflowingLongPath, -10, -overflowingLongPathExpected);

  const double overflowingMargin =
      std::bit_cast<double>(uint64_t{0x4540000000000006});
  const double overflowingMarginExpected =
      std::bit_cast<double>(uint64_t{0x4540000000000005});
  test(overflowingMargin, -10, overflowingMarginExpected);
  test(-overflowingMargin, -10, -overflowingMarginExpected);

  testPositiveZero(std::bit_cast<double>(uint64_t{0x3f677e2dadf20e5f}), 0);
  testPositiveZero(std::bit_cast<float>(uint32_t{0x3b265345}), 0);

  const float largeFloat = std::bit_cast<float>(uint32_t{0x6a256fa6});
  const float largeFloatExpected = std::bit_cast<float>(uint32_t{0x6aa56fa6});
  test(largeFloat, -26, largeFloatExpected);
  test(-largeFloat, -26, -largeFloatExpected);

  VELOX_ASSERT_THROW(
      (evaluateOnce<double, double, int32_t>(
          "bround(c0, c1)", overflowingMargin, 536'870'911)),
      "BigInteger would overflow supported range while rounding to scale "
      "536870911");
  VELOX_ASSERT_THROW(
      (evaluateOnce<float, float, int32_t>(
          "bround(c0, c1)", largeFloat, 536'870'911)),
      "BigInteger would overflow supported range while rounding to scale "
      "536870911");
}

TEST_F(BRoundTest, floatUsesExactBinaryValue) {
  test(1.25f, 1, 1.2f);
  test(1.35f, 1, 1.4f);
  test(0.575f, 2, 0.57f);
  test(-0.575f, 2, -0.57f);
  test(15.0f, -1, 20.0f);
  test(25.0f, -1, 20.0f);
  test(35.0f, -1, 40.0f);
}

TEST_F(BRoundTest, specialValuesAndZero) {
  test(
      std::numeric_limits<double>::quiet_NaN(),
      0,
      std::numeric_limits<double>::quiet_NaN());
  test(
      std::numeric_limits<double>::infinity(),
      2,
      std::numeric_limits<double>::infinity());
  test(
      -std::numeric_limits<double>::infinity(),
      -2,
      -std::numeric_limits<double>::infinity());
  test(
      std::numeric_limits<float>::quiet_NaN(),
      0,
      std::numeric_limits<float>::quiet_NaN());
  test(
      std::numeric_limits<float>::infinity(),
      2,
      std::numeric_limits<float>::infinity());

  testPositiveZero(-0.0, 5);
  testPositiveZero(-0.4, 0);
  testPositiveZero(-0.5, 0);
  testPositiveZero(-1.0, -3);
  testPositiveZero(-0.0f, 5);
  testPositiveZero(-0.4f, 0);
  testPositiveZero(-0.5f, 0);
}

TEST_F(BRoundTest, subnormalResults) {
  const auto doubleMin = std::numeric_limits<double>::denorm_min();
  test(doubleMin, 324, doubleMin);
  test(-doubleMin, 324, -doubleMin);
  test(
      std::bit_cast<double>(uint64_t{0x000fffffffffffff}),
      309,
      std::bit_cast<double>(uint64_t{0x000fd1d7d505cd02}));
  test(
      std::bit_cast<double>(uint64_t{0x0000000000000100}),
      322,
      std::bit_cast<double>(uint64_t{0x0000000000000107}));

  const auto floatMin = std::numeric_limits<float>::denorm_min();
  test(floatMin, 45, floatMin);
  test(-floatMin, 45, -floatMin);
  test(
      std::bit_cast<float>(uint32_t{0x00400000}),
      40,
      std::bit_cast<float>(uint32_t{0x00403ecd}));
}

TEST_F(BRoundTest, integralScales) {
  test<int64_t>(15, -1, 20);
  test<int64_t>(25, -1, 20);
  test<int64_t>(35, -1, 40);
  test<int64_t>(45, -1, 40);
  test<int64_t>(-15, -1, -20);
  test<int64_t>(-25, -1, -20);

  test<int32_t>(150, -2, 200);
  test<int32_t>(250, -2, 200);
  test<int32_t>(350, -2, 400);
  test<int32_t>(-150, -2, -200);

  test<int64_t>(42, 0, 42);
  test<int64_t>(42, 5, 42);
  test<int32_t>(-7, 3, -7);
  test<int16_t>(100, 2, 100);
  test<int8_t>(5, 1, 5);
}

TEST_F(BRoundTest, integralOverflowWrapping) {
  test<int64_t>(
      std::numeric_limits<int64_t>::max(), -1, -9223372036854775806LL);
  test<int64_t>(std::numeric_limits<int64_t>::min(), -1, 9223372036854775806LL);
  test<int64_t>(
      std::numeric_limits<int64_t>::max(), -19, -8446744073709551616LL);
  test<int64_t>(
      std::numeric_limits<int64_t>::min(), -19, 8446744073709551616LL);
  test<int64_t>(5'000'000'000'000'000'000LL, -19, 0);
  test<int64_t>(5'000'000'000'000'000'001LL, -19, -8446744073709551616LL);
  test<int64_t>(-5'000'000'000'000'000'000LL, -19, 0);
  test<int64_t>(-5'000'000'000'000'000'001LL, -19, 8446744073709551616LL);

  test<int32_t>(std::numeric_limits<int32_t>::max(), -1, -2147483646);
  test<int32_t>(std::numeric_limits<int32_t>::min(), -1, 2147483646);
  test<int16_t>(std::numeric_limits<int16_t>::max(), -1, -32766);
  test<int16_t>(std::numeric_limits<int16_t>::min(), -1, 32766);
  test<int8_t>(std::numeric_limits<int8_t>::max(), -1, -126);
  test<int8_t>(std::numeric_limits<int8_t>::min(), -1, 126);
}

TEST_F(BRoundTest, unaryAndNulls) {
  testUnary(2.5, 2.0);
  testUnary(3.5, 4.0);
  testUnary(1.4, 1.0);
  testUnary(2.5f, 2.0f);
  testUnary(3.5f, 4.0f);
  testUnary<int64_t>(42, 42);

  EXPECT_FALSE(
      (evaluateOnce<double, double, int32_t>("bround(c0, c1)", std::nullopt, 0))
          .has_value());
  EXPECT_FALSE((evaluateOnce<double, double, int32_t>(
                    "bround(c0, c1)", 2.5, std::nullopt))
                   .has_value());
  EXPECT_FALSE((evaluateOnce<int64_t, int64_t, int32_t>(
                    "bround(c0, c1)", std::nullopt, -1))
                   .has_value());
  EXPECT_FALSE((evaluateOnce<float, float, int32_t>(
                    "bround(c0, c1)", 2.5f, std::nullopt))
                   .has_value());
  EXPECT_FALSE((evaluateOnce<int64_t, int64_t, int32_t>(
                    "bround(c0, c1)", int64_t{25}, std::nullopt))
                   .has_value());
}

TEST_F(BRoundTest, largeScalesAndExtremes) {
  for (const auto value : {1.0, -1.0, 0.5, 2.5}) {
    test(value, 42, value);
    test(value, -42, 0.0);
  }

  VELOX_ASSERT_THROW(
      (evaluateOnce<double, double, int32_t>(
          "bround(c0, c1)", 1.0, std::numeric_limits<int32_t>::max())),
      "BigInteger would overflow supported range while rounding to scale "
      "2147483647");
  VELOX_ASSERT_THROW(
      (evaluateOnce<float, float, int32_t>(
          "bround(c0, c1)", 1.25f, std::numeric_limits<int32_t>::max())),
      "BigInteger would overflow supported range while rounding to scale "
      "2147483647");
  EXPECT_FALSE(
      (evaluateOnce<double, double, int32_t>(
           "try(bround(c0, c1))", 1.0, std::numeric_limits<int32_t>::max()))
          .has_value());
  testPositiveZero(0.0, std::numeric_limits<int32_t>::max());
  testPositiveZero(-0.0, std::numeric_limits<int32_t>::max());
  test(std::ldexp(1.0, 53), 0, std::ldexp(1.0, 53));
  test(std::ldexp(1.0f, 24), 0, std::ldexp(1.0f, 24));

  test(
      std::numeric_limits<double>::max(),
      -308,
      std::numeric_limits<double>::infinity());
  test(
      -std::numeric_limits<double>::max(),
      -308,
      -std::numeric_limits<double>::infinity());
  testPositiveZero(std::numeric_limits<double>::denorm_min(), 323);
}

TEST_F(BRoundTest, scaleUnderflow) {
  const auto exactDoubleLimit =
      static_cast<int32_t>(2 - kMaxJavaBigIntegerPowerOfTenExponent);
  test(1.25, exactDoubleLimit, 0.0);

  const std::vector<int32_t> scales = {
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1,
      static_cast<int32_t>(2 - (kMaxJavaBigIntegerPowerOfTenExponent + 1)),
      -1'000'000'000,
  };
  for (const auto scale : scales) {
    test(0.0, scale, 0.0);
    VELOX_ASSERT_THROW(
        (evaluateOnce<double, double, int32_t>("bround(c0, c1)", 1.25, scale)),
        "Underflow while rounding to scale " + std::to_string(scale));
    EXPECT_FALSE((evaluateOnce<double, double, int32_t>(
                      "try(bround(c0, c1))", 1.25, scale))
                     .has_value());
  }

  const auto minScale = std::numeric_limits<int32_t>::min();
  test<int64_t>(0, minScale, 0);
  test<int32_t>(0, minScale, 0);
  test<int16_t>(0, minScale, 0);
  test<int8_t>(0, minScale, 0);
  test<int64_t>(
      1, -static_cast<int32_t>(kMaxJavaBigIntegerPowerOfTenExponent), 0);
  VELOX_ASSERT_THROW(
      (evaluateOnce<int64_t, int64_t, int32_t>(
          "bround(c0, c1)", int64_t{1}, minScale)),
      "Underflow while rounding to scale -2147483648");
  EXPECT_FALSE(
      (evaluateOnce<int64_t, int64_t, int32_t>(
           "try(bround(c0, c1))",
           int64_t{1},
           -static_cast<int32_t>(kMaxJavaBigIntegerPowerOfTenExponent + 1)))
          .has_value());
}

TEST_F(BRoundTest, ignoresFloatingPointRoundingMode) {
  const auto originalMode = std::fegetround();
  auto restoreMode =
      folly::makeGuard([originalMode]() { std::fesetround(originalMode); });
  for (const auto mode :
       {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    test(1.25, 1, 1.2);
    EXPECT_EQ(std::fegetround(), mode);
    ASSERT_EQ(std::fesetround(mode), 0);
    test(1.25f, 1, 1.2f);
    EXPECT_EQ(std::fegetround(), mode);
  }
  ASSERT_EQ(std::fesetround(originalMode), 0);
  restoreMode.dismiss();
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
