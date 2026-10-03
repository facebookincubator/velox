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

#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>

#include <folly/ScopeGuard.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/sparksql/Rounding.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class RoundTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> round(std::optional<T> value, std::optional<int32_t> scale) {
    return evaluateOnce<T>(
        fmt::format(
            "round(c0, cast({} as integer))",
            scale ? std::to_string(*scale) : "null"),
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
        {14,
         15,
         16,
         24,
         25,
         26,
         -14,
         -15,
         -16,
         -24,
         -25,
         -26,
         0,
         std::nullopt});
    const auto expected = makeNullableFlatVector<T>(
        {10,
         20,
         20,
         20,
         30,
         30,
         -10,
         -20,
         -20,
         -20,
         -30,
         -30,
         0,
         std::nullopt});
    const auto rowType = makeRowVector({input})->rowType();
    testEncodings(
        makeTypedExpr("round(c0, cast(-1 as integer))", rowType),
        {input},
        expected);
    for (const auto& expression :
         {"round(c0)",
          "round(c0, cast(0 as integer))",
          "round(c0, cast(400 as integer))"}) {
      testEncodings(makeTypedExpr(expression, rowType), {input}, input);
    }
    for (int32_t scale : {-20, -39, -400}) {
      EXPECT_EQ(round<T>(std::numeric_limits<T>::max(), scale), 0);
      EXPECT_EQ(round<T>(std::numeric_limits<T>::min(), scale), 0);
    }
    EXPECT_EQ(round<T>(T{25}, std::nullopt), std::nullopt);
    for (int32_t scale :
         {std::numeric_limits<int32_t>::min(),
          -401,
          401,
          std::numeric_limits<int32_t>::max()}) {
      VELOX_ASSERT_THROW(round<T>(T{0}, scale), "between -400 and 400");
    }
  }

  template <typename T>
  void testCapturedMode(const std::string& name) {
    const auto maximum = std::numeric_limits<T>::max();
    const auto minimum = std::numeric_limits<T>::min();
    const auto input = makeNullableFlatVector<T>(
        {maximum, minimum, 25, -25, std::nullopt, 15});
    const auto legacyExpected = makeNullableFlatVector<T>(
        {static_cast<T>(minimum + 2),
         static_cast<T>(maximum - 1),
         30,
         -30,
         std::nullopt,
         20});
    const auto ansiExpected = makeNullableFlatVector<T>(
        {std::nullopt, std::nullopt, 30, -30, std::nullopt, 20});
    const auto row = makeRowVector({input});
    for (bool explicitMode : {false, true}) {
      const auto expression = explicitMode
          ? fmt::format("try({}(c0, cast(-1 as integer), true))", name)
          : fmt::format("{}(c0, cast(-1 as integer), false)", name);
      const auto expected = explicitMode ? ansiExpected : legacyExpected;
      setAnsiEnabled(!explicitMode);
      auto compiled = compileExpression(expression, row->rowType());
      testEncodings(
          makeTypedExpr(expression, row->rowType()), {input}, expected);
      facebook::velox::test::assertEqualVectors(
          expected, evaluate(*compiled, row));
      setAnsiEnabled(explicitMode);
      facebook::velox::test::assertEqualVectors(
          expected, evaluate(*compiled, row));
      facebook::velox::test::assertEqualVectors(
          expected,
          evaluate(expression, makeRowVector({wrapInLazyDictionary(input)})));
      const auto nullScale = fmt::format(
          "{}(c0, cast(null as integer), {})",
          name,
          explicitMode ? "true" : "false");
      testEncodings(
          makeTypedExpr(nullScale, row->rowType()),
          {input},
          BaseVector::createNullConstant(input->type(), input->size(), pool()));
    }
  }

  template <typename T>
  void testFloating() {
    for (T tie : {T{0.5}, T{1.5}, T{2.5}, T{15.5}, T{16.5}}) {
      for (T sign : {T{-1}, T{1}}) {
        const auto value = sign * tie;
        EXPECT_EQ(round<T>(value, 0), sign * std::ceil(tie));
        EXPECT_EQ(
            round<T>(std::nextafter(value, -INFINITY), 0), std::floor(value));
        EXPECT_EQ(
            round<T>(std::nextafter(value, INFINITY), 0), std::ceil(value));
      }
    }
    const auto maximum = std::numeric_limits<T>::max();
    const auto smallest = std::numeric_limits<T>::denorm_min();
    const auto input = makeNullableFlatVector<T>(
        {maximum,
         -maximum,
         smallest,
         -smallest,
         T{2.5},
         T{-2.5},
         std::ldexp(T{1}, std::numeric_limits<T>::digits),
         std::ldexp(T{1}, 63),
         std::nullopt});
    const auto expected = makeNullableFlatVector<T>(
        {maximum,
         -maximum,
         0,
         0,
         3,
         -3,
         std::ldexp(T{1}, std::numeric_limits<T>::digits),
         std::ldexp(T{1}, 63),
         std::nullopt});
    testEncodings(
        makeTypedExpr("round(c0)", makeRowVector({input})->rowType()),
        {input},
        expected);
    for (int32_t scale : {-400, -1, 0, 1, 400}) {
      EXPECT_EQ(round<T>(INFINITY, scale), INFINITY);
      EXPECT_EQ(round<T>(-INFINITY, scale), -INFINITY);
      EXPECT_TRUE(
          std::isnan(
              round<T>(std::numeric_limits<T>::quiet_NaN(), scale).value()));
      EXPECT_FALSE(std::signbit(round<T>(T{-0.0}, scale).value()));
      EXPECT_EQ(round<T>(std::nullopt, scale), std::nullopt);
    }
    EXPECT_EQ(round<T>(T{2.5}, std::nullopt), std::nullopt);
    for (int32_t scale :
         {std::numeric_limits<int32_t>::min(),
          -401,
          401,
          std::numeric_limits<int32_t>::max()}) {
      VELOX_ASSERT_THROW(round<T>(T{0}, scale), "between -400 and 400");
      VELOX_ASSERT_THROW(round<T>(INFINITY, scale), "between -400 and 400");
      VELOX_ASSERT_THROW(
          round<T>(std::numeric_limits<T>::quiet_NaN(), scale),
          "between -400 and 400");
    }
  }
};

TEST_F(RoundTest, issue10929) {
  EXPECT_EQ(round<double>(0.575, 2), 0.58);
  EXPECT_EQ(round<double>(0.5549999999999999, 2), 0.55);
  EXPECT_EQ(round<double>(0.499999999999994, 0), 0.0);
  EXPECT_EQ(round<double>(-0.575, 2), -0.58);
}

TEST_F(RoundTest, integralNegativeScale) {
  EXPECT_EQ(round<int8_t>(25, -1), 30);
  EXPECT_EQ(round<int16_t>(-25, -1), -30);
  EXPECT_EQ(round<int32_t>(2'450, -2), 2'500);
  EXPECT_EQ(round<int64_t>(525, -2), 500);
}

TEST_F(RoundTest, integralTypesAndEncodings) {
  testIntegral<int8_t>();
  testIntegral<int16_t>();
  testIntegral<int32_t>();
  testIntegral<int64_t>();
}

TEST_F(RoundTest, integralOverflow) {
  for (bool ansi : {false, true}) {
    setAnsiEnabled(ansi);
    if (ansi) {
      VELOX_ASSERT_THROW(round<int8_t>(127, -1), "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          round<int64_t>(5'000'000'000'000'000'000LL, -19),
          "Arithmetic overflow");
    } else {
      EXPECT_EQ(round<int8_t>(127, -1), -126);
      EXPECT_EQ(round<int8_t>(-128, -1), 126);
      EXPECT_EQ(
          round<int64_t>(5'000'000'000'000'000'000LL, -19),
          -8'446'744'073'709'551'616LL);
      EXPECT_EQ(
          round<int64_t>(-5'000'000'000'000'000'000LL, -19),
          8'446'744'073'709'551'616LL);
    }
  }
}

TEST_F(RoundTest, canonicalDecimalAndWidenedReal) {
  EXPECT_EQ(round<double>(2.675, 2), 2.68);
  EXPECT_EQ(round<double>(1.005, 2), 1.01);
  EXPECT_EQ(round<float>(2.55f, 1), 2.5f);
  EXPECT_EQ(round<float>(-2.55f, 1), -2.5f);
  EXPECT_EQ(round<double>(std::numeric_limits<double>::max(), -309), 0);
  const auto input = std::bit_cast<double>(uint64_t{0x43abc16d674ec7fc});
  EXPECT_EQ(
      std::bit_cast<uint64_t>(round<double>(input, -3).value()),
      uint64_t{0x43abc16d674ec800});
}

TEST_F(RoundTest, positiveZero) {
  for (int32_t scale : {-1, 0, 1, 400}) {
    EXPECT_FALSE(std::signbit(round<double>(-0.0, scale).value()));
    EXPECT_FALSE(std::signbit(round<float>(-0.0f, scale).value()));
  }
  EXPECT_FALSE(std::signbit(round<double>(-0.1, 0).value()));
  EXPECT_FALSE(
      std::signbit(
          evaluateOnce<double>("bround(c0)", std::optional<double>{-0.5})
              .value()));
}

TEST_F(RoundTest, capturedModeOverridesQuery) {
  setAnsiEnabled(true);
  EXPECT_EQ(
      evaluateOnce<int8_t>(
          "round(c0, cast(-1 as integer), false)", std::optional<int8_t>{127}),
      -126);
  setAnsiEnabled(false);
  VELOX_ASSERT_THROW(
      evaluateOnce<int8_t>(
          "round(c0, cast(-1 as integer), true)", std::optional<int8_t>{127}),
      "Arithmetic overflow");
}

TEST_F(RoundTest, prefixedRegistration) {
  registerRoundFunctions("round_prefix_");
  setAnsiEnabled(true);
  EXPECT_EQ(
      evaluateOnce<int8_t>(
          "round_prefix_round(c0, cast(-1 as integer), false)",
          std::optional<int8_t>{127}),
      -126);
  EXPECT_EQ(
      evaluateOnce<double>(
          "round_prefix_round(c0)", std::optional<double>{2.5}),
      3.0);
  EXPECT_EQ(
      evaluateOnce<double>(
          "round_prefix_spark_round(c0, cast(2 as integer))",
          std::optional<double>{0.575}),
      0.58);
}

TEST_F(RoundTest, capturedIntegralModesAndEncodings) {
  testCapturedMode<int8_t>("round");
  testCapturedMode<int16_t>("round");
  testCapturedMode<int32_t>("round");
  testCapturedMode<int64_t>("round");
}

TEST_F(RoundTest, integrationCapturedModes) {
  testCapturedMode<int8_t>("spark_round");
  testCapturedMode<int16_t>("spark_round");
  testCapturedMode<int32_t>("spark_round");
  testCapturedMode<int64_t>("spark_round");
}

TEST_F(RoundTest, integrationFloatingAndUnary) {
  EXPECT_EQ(
      evaluateOnce<double>(
          "spark_round(c0, cast(2 as integer))", std::optional<double>{0.575}),
      0.58);
  EXPECT_EQ(
      evaluateOnce<float>(
          "spark_round(c0, cast(1 as integer))", std::optional<float>{2.55f}),
      2.5f);
  EXPECT_EQ(
      evaluateOnce<double>("spark_round(c0)", std::optional<double>{-2.5}),
      -3.0);
  VELOX_ASSERT_THROW(
      evaluateOnce<double>(
          "spark_round(c0, cast(401 as integer))", std::optional<double>{0}),
      "between -400 and 400");
  VELOX_ASSERT_THROW(
      evaluateOnce<double>(
          "spark_round(c0, cast(0 as integer), true)",
          std::optional<double>{1}),
      "signature is not supported");
}

TEST_F(RoundTest, partialSelectionAndTry) {
  setAnsiEnabled(false);
  const auto input = makeRowVector(
      {makeNullableFlatVector<int8_t>({25, 127, -25, -128, std::nullopt, 45}),
       makeFlatVector<bool>({true, false, true, false, false, true})});
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>(
          {30, std::nullopt, -30, std::nullopt, std::nullopt, 50}),
      evaluate("try(round(c0, cast(-1 as integer), true))", input));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int8_t>({30, 127, -30, -128, std::nullopt, 50}),
      evaluate("if(c1, round(c0, cast(-1 as integer), true), c0)", input));
  SelectivityVector selected(input->size(), false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.setValid(5, true);
  selected.updateBounds();
  const auto result = evaluate<SimpleVector<int8_t>>(
      "round(c0, cast(-1 as integer), true)", input, selected);
  EXPECT_EQ(result->valueAt(0), 30);
  EXPECT_EQ(result->valueAt(2), -30);
  EXPECT_EQ(result->valueAt(5), 50);
}

TEST_F(RoundTest, floatingTypesAndEncodings) {
  testFloating<float>();
  testFloating<double>();
  EXPECT_EQ(round<double>(std::numeric_limits<double>::max(), -308), INFINITY);
  EXPECT_EQ(
      round<double>(-std::numeric_limits<double>::max(), -308), -INFINITY);
  EXPECT_EQ(round<float>(std::numeric_limits<float>::max(), -35), INFINITY);
  const auto smallest = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(round<double>(smallest, 324), smallest);
  // Java selects at least two significant decimal digits: 4.9e-324, not
  // the one-digit shortest representation 5e-324.
  EXPECT_EQ(round<double>(smallest, 323), 0);
  EXPECT_EQ(round<double>(5e-323, 322), 0);
  EXPECT_EQ(
      round<float>(std::numeric_limits<float>::denorm_min(), 45),
      std::numeric_limits<float>::denorm_min());
  const auto nan = std::bit_cast<double>(uint64_t{0xfff8000000000042});
  EXPECT_EQ(
      std::bit_cast<uint64_t>(round<double>(nan, 1).value()),
      std::bit_cast<uint64_t>(nan));
  const auto floatNan = std::bit_cast<float>(uint32_t{0xffc00042});
  EXPECT_EQ(
      std::bit_cast<uint32_t>(round<float>(floatNan, 1).value()),
      std::bit_cast<uint32_t>(floatNan));
}

TEST_F(RoundTest, floatingEnvironment) {
  const auto original = std::fegetround();
  const auto restore = folly::makeGuard([&]() { std::fesetround(original); });
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_EQ(round<double>(2.55, 1), 2.6);
    EXPECT_EQ(round<double>(-2.55, 1), -2.6);
    EXPECT_EQ(round<double>(2.5, 0), 3.0);
    EXPECT_EQ(round<double>(-0.5, 0), -1.0);
    EXPECT_EQ(round<float>(2.45f, 1), 2.5f);
  }
}

TEST_F(RoundTest, nullScaleSkipsChild) {
  setAnsiEnabled(true);
  const auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 0})});
  for (
      const auto& expression :
      {"round(checked_div(c0, c0), cast(null as integer))",
       "round(checked_div(c0, c0), cast(null as integer), true)",
       "round(checked_div(cast(1 as bigint), cast(0 as bigint)), cast(null as integer))"}) {
    facebook::velox::test::assertEqualVectors(
        BaseVector::createNullConstant(BIGINT(), 3, pool()),
        evaluate(expression, input));
  }
}

TEST_F(RoundTest, constantAndFoldableArguments) {
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>({25, -25}),
       makeFlatVector<int32_t>({-1, 0}),
       makeFlatVector<bool>({false, true})});
  VELOX_ASSERT_THROW(evaluate("round(c0, c1, true)", input), "second argument");
  VELOX_ASSERT_THROW(
      evaluate("round(c0, cast(-1 as integer), c2)", input), "third argument");
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int64_t>({30, -30}),
      evaluate("round(c0, cast(subtract(0, 1) as integer), false)", input));
  for (const auto& value :
       {VectorPtr(makeFlatVector<float>({1.5})),
        VectorPtr(makeFlatVector<double>({1.5})),
        VectorPtr(makeFlatVector<int64_t>({15}, DECIMAL(3, 1)))}) {
    VELOX_ASSERT_THROW(
        evaluate(
            "round(c0, cast(-1 as integer), true)", makeRowVector({value})),
        "signature is not supported");
  }
}

TEST_F(RoundTest, unsupportedScaleAndVariableScale) {
  for (int32_t scale : {-401, 401}) {
    VELOX_ASSERT_THROW(round<double>(0.0, scale), "between -400 and 400");
    VELOX_ASSERT_THROW(round<int64_t>(0, scale), "between -400 and 400");
  }
  auto input = makeRowVector(
      {makeFlatVector<double>({2.5, 3.5}), makeFlatVector<int32_t>({0, 1})});
  VELOX_ASSERT_THROW(evaluate("round(c0, c1)", input), "constant");
}

TEST_F(RoundTest, prestoRegistrationUnchanged) {
  registerFunction<functions::RoundFunction, double, double, int32_t>(
      {"presto_round_regression"});
  registerFunction<functions::RoundFunction, int64_t, int64_t, int32_t>(
      {"presto_round_regression"});
  EXPECT_EQ(
      evaluateOnce<int64_t>(
          "presto_round_regression(c0, cast(-1 as integer))",
          std::optional<int64_t>{25}),
      25);
  EXPECT_EQ(
      evaluateOnce<double>(
          "presto_round_regression(c0, cast(2 as integer))",
          std::optional<double>{0.575}),
      0.57);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
