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
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/BRound.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class BRoundTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> bround(
      std::optional<T> value,
      std::optional<int32_t> scale) {
    return evaluateOnce<T>(
        fmt::format(
            "bround(c0, cast({} as integer))",
            scale ? std::to_string(*scale) : "null"),
        value);
  }

  template <typename T>
  std::optional<T> bround(std::optional<T> value) {
    return evaluateOnce<T, T>("bround(c0)", value);
  }

  void setAnsiEnabled(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }

  template <typename T>
  void testIntegralEncodings() {
    const auto input = makeNullableFlatVector<T>(
        {14,  15,  16,  24,  25,  26,  34,  35,  36, -14,
         -15, -16, -24, -25, -26, -34, -35, -36, 0,  std::nullopt});
    const auto expected = makeNullableFlatVector<T>(
        {10,  20,  20,  20,  20,  30,  30,  40,  40, -10,
         -20, -20, -20, -20, -30, -30, -40, -40, 0,  std::nullopt});
    const auto row = makeRowVector({input});
    testEncodings(
        makeTypedExpr("bround(c0, cast(-1 as integer))", row->rowType()),
        {input},
        expected);
  }

  template <typename T>
  void testIntegralOverflow() {
    const auto maximum = std::numeric_limits<T>::max();
    const auto minimum = std::numeric_limits<T>::min();
    const auto wrappedPositive = static_cast<T>(minimum + 2);
    const auto wrappedNegative = static_cast<T>(maximum - 1);

    setAnsiEnabled(false);
    EXPECT_EQ(bround<T>(maximum, -1), wrappedPositive);
    EXPECT_EQ(bround<T>(minimum, -1), wrappedNegative);

    setAnsiEnabled(true);
    VELOX_ASSERT_THROW(bround<T>(maximum, -1), "Arithmetic overflow");
    VELOX_ASSERT_THROW(bround<T>(minimum, -1), "Arithmetic overflow");
    EXPECT_EQ(bround<T>(T{25}, -1), T{20});
    EXPECT_EQ(bround<T>(T{-35}, -1), T{-40});
  }

  template <typename T>
  void testIntegralExtremeScaleUnderflow() {
    constexpr int32_t kMaximumSupportedScale =
        -detail::kMaxJavaBigIntegerPowerOfTenExponent;
    constexpr int32_t kFirstUnderflowScale = kMaximumSupportedScale - 1;
    constexpr auto kMinimumScale = std::numeric_limits<int32_t>::min();
    for (const bool ansiEnabled : {false, true}) {
      setAnsiEnabled(ansiEnabled);
      EXPECT_EQ(bround<T>(T{1}, kMaximumSupportedScale), T{0});
      EXPECT_EQ(bround<T>(T{0}, kFirstUnderflowScale), T{0});
      VELOX_ASSERT_THROW(
          bround<T>(T{1}, kFirstUnderflowScale),
          "Underflow while rounding to scale -536870920");
      EXPECT_EQ(bround<T>(T{0}, kMinimumScale), T{0});
      VELOX_ASSERT_THROW(
          bround<T>(T{1}, kMinimumScale),
          "Underflow while rounding to scale -2147483648");
      EXPECT_EQ(
          (evaluateOnce<T, T>(
              "try(bround(c0, cast(-536870920 as integer)))", T{1})),
          std::nullopt);
    }
  }
};

TEST_F(BRoundTest, floatingPointHalfEven) {
  EXPECT_EQ(bround<double>(0.5, 0), 0.0);
  EXPECT_EQ(bround<double>(1.5, 0), 2.0);
  EXPECT_EQ(bround<double>(2.5, 0), 2.0);
  EXPECT_EQ(bround<double>(3.5, 0), 4.0);
  EXPECT_EQ(bround<double>(-2.5, 0), -2.0);
  EXPECT_EQ(bround<double>(-3.5, 0), -4.0);

  EXPECT_EQ(bround<double>(1.25, 1), 1.2);
  EXPECT_EQ(bround<double>(1.75, 1), 1.8);
  EXPECT_EQ(bround<double>(1.245, 2), 1.25);
  // Spark converts through a decimal string and rounds 0.575 to 0.58. Velox
  // intentionally rounds the binary value directly.
  EXPECT_EQ(bround<double>(0.575, 2), 0.57);
  EXPECT_EQ(bround<double>(-0.575, 2), -0.57);
  EXPECT_EQ(bround<double>(150.0, -2), 200.0);
  EXPECT_EQ(bround<double>(250.0, -2), 200.0);
  EXPECT_EQ(bround<double>(350.0, -2), 400.0);

  EXPECT_EQ(bround<float>(1.25f, 1), 1.2f);
  EXPECT_EQ(bround<float>(1.75f, 1), 1.8f);
  EXPECT_EQ(bround<float>(15.0f, -1), 20.0f);
  EXPECT_EQ(bround<float>(25.0f, -1), 20.0f);
}

TEST_F(BRoundTest, floatingPointNeighbors) {
  EXPECT_EQ(bround<double>(std::nextafter(0.5, 0.0), 0), 0.0);
  EXPECT_EQ(bround<double>(std::nextafter(0.5, 1.0), 0), 1.0);
  EXPECT_EQ(bround<double>(std::nextafter(-0.5, -1.0), 0), -1.0);
  EXPECT_EQ(bround<double>(std::nextafter(-0.5, 0.0), 0), 0.0);
}

TEST_F(BRoundTest, specialValuesAndScales) {
  EXPECT_TRUE(
      std::isnan(
          bround<double>(std::numeric_limits<double>::quiet_NaN(), 2).value()));
  EXPECT_EQ(
      bround<double>(std::numeric_limits<double>::infinity(), 2),
      std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      bround<double>(-std::numeric_limits<double>::infinity(), -2),
      -std::numeric_limits<double>::infinity());

  EXPECT_EQ(
      bround<double>(
          std::numeric_limits<double>::max(),
          std::numeric_limits<int32_t>::max()),
      std::numeric_limits<double>::max());
  EXPECT_EQ(
      bround<double>(
          std::numeric_limits<double>::max(),
          std::numeric_limits<int32_t>::min()),
      0.0);

  for (const auto scale : {-10, 0, 10}) {
    const auto rounded = bround<double>(-0.0, scale).value();
    EXPECT_EQ(rounded, 0.0);
    EXPECT_FALSE(std::signbit(rounded));
  }
}

TEST_F(BRoundTest, constantScaleRequired) {
  const auto input = makeRowVector(
      {makeFlatVector<double>({2.5, 3.5}), makeFlatVector<int32_t>({0, 1})});
  VELOX_ASSERT_THROW(evaluate("bround(c0, c1)", input), "constant");
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<double>({2.0, 4.0}),
      evaluate("bround(c0, cast(subtract(1, 1) as integer))", input));
}

TEST_F(BRoundTest, integralTypesAndEncodings) {
  setAnsiEnabled(false);
  testIntegralEncodings<int8_t>();
  testIntegralEncodings<int16_t>();
  testIntegralEncodings<int32_t>();
  testIntegralEncodings<int64_t>();

  EXPECT_EQ(bround<int64_t>(42, 0), 42);
  EXPECT_EQ(bround<int64_t>(42, 5), 42);
}

TEST_F(BRoundTest, integralExtremeScaleUnderflow) {
  testIntegralExtremeScaleUnderflow<int8_t>();
  testIntegralExtremeScaleUnderflow<int16_t>();
  testIntegralExtremeScaleUnderflow<int32_t>();
  testIntegralExtremeScaleUnderflow<int64_t>();
}

TEST_F(BRoundTest, integralOverflowMode) {
  testIntegralOverflow<int8_t>();
  testIntegralOverflow<int16_t>();
  testIntegralOverflow<int32_t>();
  testIntegralOverflow<int64_t>();

  setAnsiEnabled(false);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::max(), -19),
      -8'446'744'073'709'551'616LL);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::min(), -19),
      8'446'744'073'709'551'616LL);

  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      bround<int64_t>(std::numeric_limits<int64_t>::max(), -19),
      "Arithmetic overflow");
}

TEST_F(BRoundTest, capturesAnsiModeAtInitialization) {
  const auto input =
      makeRowVector({makeFlatVector<int8_t>({127, 25, -35, -128})});
  const auto rowType = input->rowType();

  setAnsiEnabled(false);
  auto legacy = compileExpression("bround(c0, cast(-1 as integer))", rowType);
  setAnsiEnabled(true);
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int8_t>({-126, 20, -40, 126}), evaluate(*legacy, input));

  auto ansi =
      compileExpression("try(bround(c0, cast(-1 as integer)))", rowType);
  setAnsiEnabled(false);
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>({std::nullopt, 20, -40, std::nullopt}),
      evaluate(*ansi, input));
}

TEST_F(BRoundTest, partialSelectionAndTry) {
  setAnsiEnabled(true);
  const auto input = makeRowVector(
      {makeNullableFlatVector<int8_t>({25, 127, -35, -128, std::nullopt, 45})});
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>(
          {20, std::nullopt, -40, std::nullopt, std::nullopt, 40}),
      evaluate("try(bround(c0, cast(-1 as integer)))", input));

  SelectivityVector selected(input->size(), false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.setValid(5, true);
  selected.updateBounds();
  const auto result = evaluate<SimpleVector<int8_t>>(
      "bround(c0, cast(-1 as integer))", input, selected);
  EXPECT_EQ(result->valueAt(0), 20);
  EXPECT_EQ(result->valueAt(2), -40);
  EXPECT_EQ(result->valueAt(5), 40);
}

TEST_F(BRoundTest, unaryAndNulls) {
  EXPECT_EQ(bround<double>(2.5), 2.0);
  EXPECT_EQ(bround<double>(3.5), 4.0);
  EXPECT_EQ(bround<int64_t>(42), 42);

  EXPECT_EQ(bround<double>(std::nullopt, 0), std::nullopt);
  EXPECT_EQ(bround<double>(2.5, std::nullopt), std::nullopt);
  EXPECT_EQ(bround<int64_t>(std::nullopt, -1), std::nullopt);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
