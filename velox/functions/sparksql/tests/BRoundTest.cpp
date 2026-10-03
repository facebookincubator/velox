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

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <folly/ScopeGuard.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/BRound.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

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
    return evaluateOnce<T>("bround(c0)", value);
  }

  template <typename T>
  std::optional<T> broundWithMode(
      std::optional<T> value,
      std::optional<int32_t> scale,
      bool ansiEnabled) {
    return evaluateOnce<T>(
        fmt::format(
            "bround(c0, cast({} as integer), {})",
            scale ? std::to_string(*scale) : "null",
            ansiEnabled ? "true" : "false"),
        value);
  }

  void setAnsiEnabled(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }

  template <typename T>
  void testIntegral() {
    const auto input = makeNullableFlatVector<T>(
        {14,  15,  16,  24,  25,  26,  34,  35,  36, -14,
         -15, -16, -24, -25, -26, -34, -35, -36, 0,  std::nullopt});
    const auto expected = makeNullableFlatVector<T>(
        {10,  20,  20,  20,  20,  30,  30,  40,  40, -10,
         -20, -20, -20, -20, -30, -30, -40, -40, 0,  std::nullopt});
    auto row = makeRowVector({input});
    testEncodings(
        makeTypedExpr("bround(c0, cast(-1 as integer))", row->rowType()),
        {input},
        expected);
    for (const auto& expression :
         {"bround(c0)",
          "bround(c0, cast(0 as integer))",
          "bround(c0, cast(100 as integer))"}) {
      testEncodings(makeTypedExpr(expression, row->rowType()), {input}, input);
    }
    for (int32_t scale : {-20, -39, -400}) {
      EXPECT_EQ(bround<T>(std::numeric_limits<T>::max(), scale), 0);
      EXPECT_EQ(bround<T>(std::numeric_limits<T>::min(), scale), 0);
    }
  }

  template <typename T>
  void testScaleRange() {
    for (int32_t scale : {-400, -399}) {
      EXPECT_EQ(bround<T>(std::numeric_limits<T>::max(), scale), 0);
      EXPECT_EQ(bround<T>(std::numeric_limits<T>::lowest(), scale), 0);
    }
    for (int32_t scale : {399, 400}) {
      EXPECT_EQ(
          bround<T>(std::numeric_limits<T>::max(), scale),
          std::numeric_limits<T>::max());
      EXPECT_EQ(
          bround<T>(std::numeric_limits<T>::lowest(), scale),
          std::numeric_limits<T>::lowest());
    }
    EXPECT_EQ(bround<T>(T{15}, std::nullopt), std::nullopt);
    for (int32_t scale :
         {std::numeric_limits<int32_t>::min(),
          std::numeric_limits<int32_t>::min() + 1,
          -1000,
          -401,
          401,
          1000,
          std::numeric_limits<int32_t>::max() - 1,
          std::numeric_limits<int32_t>::max()}) {
      VELOX_ASSERT_THROW(bround<T>(T{0}, scale), "between -400 and 400");
      VELOX_ASSERT_THROW(bround<T>(T{15}, scale), "between -400 and 400");
      if constexpr (std::is_floating_point_v<T>) {
        VELOX_ASSERT_THROW(
            bround<T>(std::numeric_limits<T>::infinity(), scale),
            "between -400 and 400");
        VELOX_ASSERT_THROW(
            bround<T>(std::numeric_limits<T>::quiet_NaN(), scale),
            "between -400 and 400");
      }
    }
  }

  template <typename T>
  void testCapturedMode() {
    const auto maximum = std::numeric_limits<T>::max();
    const auto minimum = std::numeric_limits<T>::min();
    const auto wrappedPositive = static_cast<T>(minimum + 2);
    const auto wrappedNegative = static_cast<T>(maximum - 1);
    const auto input = makeNullableFlatVector<T>(
        {maximum, minimum, 15, -15, 25, -25, std::nullopt});
    const auto row = makeRowVector({input});
    const auto legacyExpected = makeNullableFlatVector<T>(
        {wrappedPositive, wrappedNegative, 20, -20, 20, -20, std::nullopt});
    const auto ansiExpected = makeNullableFlatVector<T>(
        {std::nullopt, std::nullopt, 20, -20, 20, -20, std::nullopt});
    const std::string legacy = "bround(c0, cast(-1 as integer), false)";
    const std::string ansi = "try(bround(c0, cast(-1 as integer), true))";

    setAnsiEnabled(false);
    VELOX_ASSERT_THROW(
        broundWithMode<T>(maximum, -1, true), "Arithmetic overflow");
    VELOX_ASSERT_THROW(
        broundWithMode<T>(minimum, -1, true), "Arithmetic overflow");
    testEncodings(makeTypedExpr(ansi, row->rowType()), {input}, ansiExpected);
    auto compiledAnsi = compileExpression(ansi, row->rowType());
    facebook::velox::test::assertEqualVectors(
        ansiExpected, evaluate(*compiledAnsi, row));
    setAnsiEnabled(true);
    facebook::velox::test::assertEqualVectors(
        ansiExpected, evaluate(*compiledAnsi, row));

    EXPECT_EQ(broundWithMode<T>(maximum, -1, false), wrappedPositive);
    EXPECT_EQ(broundWithMode<T>(minimum, -1, false), wrappedNegative);
    testEncodings(
        makeTypedExpr(legacy, row->rowType()), {input}, legacyExpected);
    auto compiledLegacy = compileExpression(legacy, row->rowType());
    facebook::velox::test::assertEqualVectors(
        legacyExpected, evaluate(*compiledLegacy, row));
    setAnsiEnabled(false);
    facebook::velox::test::assertEqualVectors(
        legacyExpected, evaluate(*compiledLegacy, row));

    for (bool explicitMode : {false, true}) {
      setAnsiEnabled(!explicitMode);
      EXPECT_EQ(
          broundWithMode<T>(std::nullopt, -1, explicitMode), std::nullopt);
      EXPECT_EQ(
          broundWithMode<T>(maximum, std::nullopt, explicitMode), std::nullopt);
      EXPECT_EQ(
          broundWithMode<T>(std::nullopt, std::nullopt, explicitMode),
          std::nullopt);
      EXPECT_EQ(broundWithMode<T>(T{15}, -400, explicitMode), 0);
      EXPECT_EQ(broundWithMode<T>(T{15}, 400, explicitMode), 15);
      for (int32_t scale : {-401, 401}) {
        VELOX_ASSERT_THROW(
            broundWithMode<T>(T{0}, scale, explicitMode),
            "between -400 and 400");
      }
    }
  }
};

TEST_F(BRoundTest, capturedIntegralMode) {
  testCapturedMode<int8_t>();
  testCapturedMode<int16_t>();
  testCapturedMode<int32_t>();
  testCapturedMode<int64_t>();
}

TEST_F(BRoundTest, capturedModeRequiresConstants) {
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>({15, 25}),
       makeFlatVector<int32_t>({-1, 0}),
       makeFlatVector<bool>({false, true})});
  VELOX_ASSERT_THROW(
      evaluate("bround(c0, c1, true)", input), "second argument");
  VELOX_ASSERT_THROW(
      evaluate("bround(c0, cast(-1 as integer), c2)", input), "third argument");
}

TEST_F(BRoundTest, capturedModeIntegralOnly) {
  for (const auto& input :
       {VectorPtr(makeFlatVector<float>({1.5})),
        VectorPtr(makeFlatVector<double>({1.5})),
        VectorPtr(makeFlatVector<int64_t>({15}, DECIMAL(3, 1)))}) {
    VELOX_ASSERT_THROW(
        evaluate(
            "bround(c0, cast(-1 as integer), true)", makeRowVector({input})),
        "signature is not supported");
  }
}

TEST_F(BRoundTest, capturedModePrefix) {
  registerBRoundFunctions("captured_");
  setAnsiEnabled(true);
  EXPECT_EQ(
      evaluateOnce<int8_t>(
          "captured_bround(c0, cast(-1 as integer), false)",
          std::optional<int8_t>{127}),
      -126);
}

TEST_F(BRoundTest, supportedScaleInterval) {
  for (bool ansi : {false, true}) {
    setAnsiEnabled(ansi);
    testScaleRange<int8_t>();
    testScaleRange<int16_t>();
    testScaleRange<int32_t>();
    testScaleRange<int64_t>();
    testScaleRange<float>();
    testScaleRange<double>();
  }
}

TEST_F(BRoundTest, invalidScaleInitialization) {
  const auto input =
      makeRowVector({makeNullableFlatVector<double>({0.0, 1.5, std::nullopt})});
  // Simple-function initialization errors are deferred by the expression
  // engine.
  const auto expression =
      compileExpression("bround(c0, cast(401 as integer))", input->rowType());
  VELOX_ASSERT_THROW(evaluate(*expression, input), "between -400 and 400");
  facebook::velox::test::assertEqualVectors(
      BaseVector::createNullConstant(DOUBLE(), 3, pool()),
      evaluate("try(bround(c0, cast(401 as integer)))", input));
}

TEST_F(BRoundTest, integralTypesAndEncodings) {
  testIntegral<int8_t>();
  testIntegral<int16_t>();
  testIntegral<int32_t>();
  testIntegral<int64_t>();
}

TEST_F(BRoundTest, constantScaleRequired) {
  auto input = makeRowVector(
      {makeFlatVector<double>({2.5, 3.5}), makeFlatVector<int32_t>({0, 1})});
  VELOX_ASSERT_THROW(evaluate("bround(c0, c1)", input), "constant");
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<double>({2.0, 4.0}),
      evaluate("bround(c0, cast(subtract(1, 1) as integer))", input));
}

TEST_F(BRoundTest, floatingPointHalfEven) {
  EXPECT_DOUBLE_EQ(bround<double>(2.5, 0).value(), 2.0);
  EXPECT_DOUBLE_EQ(bround<double>(3.5, 0).value(), 4.0);
  EXPECT_DOUBLE_EQ(bround<double>(-2.5, 0).value(), -2.0);
  EXPECT_DOUBLE_EQ(bround<double>(-3.5, 0).value(), -4.0);
  EXPECT_DOUBLE_EQ(bround<double>(2.45, 1).value(), 2.4);
  EXPECT_DOUBLE_EQ(bround<double>(2.55, 1).value(), 2.6);
  EXPECT_DOUBLE_EQ(bround<double>(1e17, 2).value(), 1e17);
  EXPECT_DOUBLE_EQ(
      bround<double>(-2.890717809193079e19, -10).value(), -2.890717809e19);
  EXPECT_DOUBLE_EQ(
      bround<double>(std::numeric_limits<double>::max(), 2).value(),
      std::numeric_limits<double>::max());
  EXPECT_EQ(
      bround<double>(std::numeric_limits<double>::denorm_min(), 0).value(),
      0.0);

  EXPECT_FLOAT_EQ(bround<float>(2.5f, 0).value(), 2.0f);
  EXPECT_FLOAT_EQ(bround<float>(3.5f, 0).value(), 4.0f);
  EXPECT_FLOAT_EQ(bround<float>(1e17f, 2).value(), 1e17f);
}

TEST_F(BRoundTest, integralNegativeScale) {
  EXPECT_EQ(bround<int8_t>(25, -1), 20);
  EXPECT_EQ(bround<int8_t>(35, -1), 40);
  EXPECT_EQ(bround<int16_t>(-25, -1), -20);
  EXPECT_EQ(bround<int16_t>(-35, -1), -40);
  EXPECT_EQ(bround<int32_t>(2'450, -2), 2'400);
  EXPECT_EQ(bround<int32_t>(2'550, -2), 2'600);
  EXPECT_EQ(bround<int64_t>(525, -2), 500);
}

TEST_F(BRoundTest, bigintScaleNineteenWrapsInLegacyMode) {
  setAnsiEnabled(false);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::max(), -19),
      -8'446'744'073'709'551'616LL);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::min(), -19),
      8'446'744'073'709'551'616LL);
  EXPECT_EQ(bround<int64_t>(5'000'000'000'000'000'000LL, -19), 0);
}

TEST_F(BRoundTest, integralOverflowWrapsInLegacyMode) {
  setAnsiEnabled(false);
  EXPECT_EQ(bround<int8_t>(127, -1), -126);
  EXPECT_EQ(bround<int8_t>(-128, -1), 126);
  EXPECT_EQ(bround<int16_t>(32'767, -1), -32'766);
  EXPECT_EQ(bround<int16_t>(-32'768, -1), 32'766);
  EXPECT_EQ(bround<int32_t>(2'147'483'647, -1), -2'147'483'646);
  EXPECT_EQ(bround<int32_t>(-2'147'483'648, -1), 2'147'483'646);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::max(), -1),
      -9'223'372'036'854'775'806LL);
  EXPECT_EQ(
      bround<int64_t>(std::numeric_limits<int64_t>::min(), -1),
      9'223'372'036'854'775'806LL);
}

TEST_F(BRoundTest, integralOverflowThrowsInAnsiMode) {
  setAnsiEnabled(true);
  VELOX_ASSERT_THROW(
      bround<int16_t>(std::numeric_limits<int16_t>::max(), -1),
      "Arithmetic overflow");
  VELOX_ASSERT_THROW(
      bround<int8_t>(std::numeric_limits<int8_t>::max(), -1),
      "Arithmetic overflow");
  VELOX_ASSERT_THROW(
      bround<int32_t>(std::numeric_limits<int32_t>::max(), -1),
      "Arithmetic overflow");
  VELOX_ASSERT_THROW(
      bround<int64_t>(std::numeric_limits<int64_t>::max(), -19),
      "Arithmetic overflow");
}

TEST_F(BRoundTest, floatingPointTiesAndNeighbors) {
  for (double tie : {0.5, 1.5, 2.5, 3.5, 4.5, 15.5, 16.5}) {
    const double even =
        static_cast<int64_t>(tie) % 2 == 0 ? std::floor(tie) : std::ceil(tie);
    for (double sign : {-1.0, 1.0}) {
      const double value = sign * tie;
      EXPECT_EQ(bround<double>(value, 0), sign * even);
      EXPECT_EQ(
          bround<double>(std::nextafter(value, -INFINITY), 0),
          std::floor(value));
      EXPECT_EQ(
          bround<double>(std::nextafter(value, INFINITY), 0), std::ceil(value));
    }
  }
  EXPECT_EQ(bround<double>(std::nextafter(2.45, 0.0), 1), 2.4);
  EXPECT_EQ(bround<double>(std::nextafter(2.45, INFINITY), 1), 2.5);
  // Spark widens REAL before obtaining its shortest decimal representation.
  EXPECT_EQ(bround<float>(2.45f, 1), 2.5f);
  EXPECT_EQ(bround<float>(2.55f, 1), 2.5f);
  EXPECT_EQ(bround<float>(-2.45f, 1), -2.5f);
  EXPECT_EQ(bround<float>(-2.55f, 1), -2.5f);
}

TEST_F(BRoundTest, floatingPointJdk21DecimalSelection) {
  // JDK17 can select a longer decimal string and round these inputs
  // differently. These expected bits come from registered Spark BROUND running
  // on JDK21.
  const std::array<std::pair<uint64_t, uint64_t>, 3> examples{{
      {0x438ba9e9e9365ded, 0x438ba9e9e9365ddd},
      {0x43abc16d674ec7fc, 0x43abc16d674ec800},
      {0x43abc16d674ec804, 0x43abc16d674ec800},
  }};
  for (bool ansi : {false, true}) {
    setAnsiEnabled(ansi);
    for (const auto& [input, expected] : examples) {
      for (uint64_t sign : {uint64_t{0}, uint64_t{1} << 63}) {
        EXPECT_EQ(
            std::bit_cast<uint64_t>(
                bround<double>(std::bit_cast<double>(input | sign), -3)
                    .value()),
            expected | sign);
      }
    }
  }
}

TEST_F(BRoundTest, floatingPointRangeAndEncoding) {
  const auto maximum = std::numeric_limits<double>::max();
  const auto smallest = std::numeric_limits<double>::denorm_min();
  const auto input = makeNullableFlatVector<double>(
      {maximum,
       -maximum,
       smallest,
       -smallest,
       std::ldexp(1.0, 52),
       std::ldexp(1.0, 53),
       std::ldexp(1.0, 63),
       -std::ldexp(1.0, 63),
       std::nullopt});
  const auto expected = makeNullableFlatVector<double>(
      {maximum,
       -maximum,
       0,
       0,
       std::ldexp(1.0, 52),
       std::ldexp(1.0, 53),
       std::ldexp(1.0, 63),
       -std::ldexp(1.0, 63),
       std::nullopt});
  auto row = makeRowVector({input});
  testEncodings(makeTypedExpr("bround(c0)", row->rowType()), {input}, expected);
  EXPECT_EQ(bround<double>(maximum, -308), INFINITY);
  EXPECT_EQ(bround<double>(-maximum, -308), -INFINITY);
  EXPECT_EQ(bround<double>(maximum, -309), 0);
  EXPECT_EQ(bround<float>(std::numeric_limits<float>::max(), -35), INFINITY);
  EXPECT_EQ(bround<double>(smallest, 324), smallest);
  EXPECT_EQ(bround<double>(smallest, 323), 0);
  EXPECT_EQ(
      bround<float>(std::numeric_limits<float>::denorm_min(), 45),
      std::numeric_limits<float>::denorm_min());
}

TEST_F(BRoundTest, partialSelectionAndTry) {
  setAnsiEnabled(true);
  const auto input = makeRowVector(
      {makeNullableFlatVector<int8_t>({25, 127, -35, -128, std::nullopt, 45}),
       makeFlatVector<bool>({true, false, true, false, false, true})});
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>(
          {20, std::nullopt, -40, std::nullopt, std::nullopt, 40}),
      evaluate("try(bround(c0, cast(-1 as integer)))", input));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>({20, 127, -40, -128, std::nullopt, 40}),
      evaluate("if(c1, bround(c0, cast(-1 as integer)), c0)", input));
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
  const auto lazy = makeRowVector({wrapInLazyDictionary(input->childAt(0))});
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>(
          {20, std::nullopt, -40, std::nullopt, std::nullopt, 40}),
      evaluate("try(bround(c0, cast(-1 as integer)))", lazy));
}

TEST_F(BRoundTest, floatingPointEnvironment) {
  const auto original = std::fegetround();
  // Restore the process environment even if an assertion or evaluation fails.
  const auto restore = folly::makeGuard([&]() { std::fesetround(original); });
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_EQ(bround<double>(2.55, 1), 2.6);
    EXPECT_EQ(bround<double>(-2.55, 1), -2.6);
    EXPECT_EQ(bround<double>(2.5, 0), 2.0);
    EXPECT_EQ(bround<double>(3.5, 0), 4.0);
    EXPECT_EQ(bround<float>(2.45f, 1), 2.5f);
  }
}

TEST_F(BRoundTest, specialValuesAndNulls) {
  EXPECT_TRUE(
      std::isnan(
          bround<double>(std::numeric_limits<double>::quiet_NaN(), 2).value()));
  EXPECT_EQ(
      bround<double>(std::numeric_limits<double>::infinity(), 2),
      std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      bround<double>(-std::numeric_limits<double>::infinity(), 2),
      -std::numeric_limits<double>::infinity());

  const auto negativeZero = bround<double>(-0.0, 2).value();
  EXPECT_EQ(negativeZero, 0.0);
  EXPECT_FALSE(std::signbit(negativeZero));

  EXPECT_EQ(bround<double>(std::nullopt, 2), std::nullopt);
  EXPECT_EQ(bround<double>(2.5, std::nullopt), std::nullopt);
  EXPECT_EQ(bround<int64_t>(std::nullopt), std::nullopt);
  for (int32_t scale : {-309, -1, 0, 1, 325}) {
    EXPECT_FALSE(std::signbit(bround<double>(-0.0, scale).value()));
    EXPECT_FALSE(std::signbit(bround<float>(-0.0f, scale).value()));
  }
  const auto nan = std::bit_cast<double>(uint64_t{0xfff8000000000042});
  EXPECT_EQ(
      std::bit_cast<uint64_t>(bround<double>(nan, 1).value()),
      std::bit_cast<uint64_t>(nan));
  setAnsiEnabled(true);
  const auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 0})});
  facebook::velox::test::assertEqualVectors(
      BaseVector::createNullConstant(BIGINT(), 3, pool()),
      evaluate("bround(checked_div(c0, c0), cast(null as integer))", input));
}

TEST_F(BRoundTest, floatingPointScaleBoundaries) {
  for (const int32_t scale : {-400, 400}) {
    EXPECT_FALSE(std::signbit(bround<double>(-0.0, scale).value()));
    EXPECT_FALSE(std::signbit(bround<float>(-0.0f, scale).value()));
    EXPECT_EQ(
        bround<double>(std::numeric_limits<double>::infinity(), scale),
        std::numeric_limits<double>::infinity());
    EXPECT_TRUE(
        std::isnan(
            bround<double>(std::numeric_limits<double>::quiet_NaN(), scale)
                .value()));
  }
}

TEST_F(BRoundTest, unaryUsesScaleZero) {
  EXPECT_DOUBLE_EQ(bround<double>(2.5).value(), 2.0);
  EXPECT_DOUBLE_EQ(bround<double>(3.5).value(), 4.0);
  EXPECT_EQ(bround<float>(3.5f), 4.0f);
  EXPECT_EQ(bround<int64_t>(25), 25);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
