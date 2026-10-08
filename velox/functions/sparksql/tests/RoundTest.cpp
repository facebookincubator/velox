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
#include <cmath>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
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
    EXPECT_EQ(round<T>(T{25}, std::numeric_limits<int32_t>::min()), 0);
    EXPECT_EQ(round<T>(T{25}, std::numeric_limits<int32_t>::max()), 25);
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

};

TEST_F(RoundTest, publicBinaryFloatingBaseline) {
  EXPECT_EQ(round<double>(0.575, 2), 0.57);
  EXPECT_EQ(round<double>(-0.575, 2), -0.57);
  EXPECT_EQ(round<double>(1.005, 2), 1.0);
  auto input = makeRowVector(
      {makeFlatVector<double>({0.575, 1.005}),
       makeFlatVector<int32_t>({2, 2})});
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<double>({0.57, 1.0}), evaluate("round(c0, c1)", input));
  for (const auto* expression :
       {"spark_round(c0)", "spark_round(c0, cast(2 as integer))"}) {
    VELOX_ASSERT_THROW(
        evaluate(expression, input), "signature is not supported");
  }
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

TEST_F(RoundTest, roundRegistration) {
  EXPECT_EQ(round<double>(2.5, 0), 3.0);
  EXPECT_EQ(
      evaluateOnce<int64_t>(
          "spark_round(c0, cast(-1 as integer), false)",
          std::optional<int64_t>{25}),
      30);
  EXPECT_TRUE(exec::isFunctionCallToSpecialFormRegistered("decimal_round"));
  EXPECT_TRUE(
      exec::isFunctionCallToSpecialFormRegistered("decimal_spark_round"));
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
  VELOX_ASSERT_THROW(
      evaluateOnce<double>(
          "round_prefix_spark_round(c0, cast(2 as integer))",
          std::optional<double>{0.575}),
      "signature is not supported");
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

TEST_F(RoundTest, integrationSignatures) {
  VELOX_ASSERT_THROW(
      evaluateOnce<float>(
          "spark_round(c0, cast(1 as integer))", std::optional<float>{2.55f}),
      "signature is not supported");
  VELOX_ASSERT_THROW(
      evaluateOnce<int64_t>("spark_round(c0)", std::optional<int64_t>{25}),
      "signature is not supported");
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

TEST_F(RoundTest, floatingRegistrationsUseUnchangedBinaryHelper) {
  const auto check = [&]<typename T>() {
    using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    for (T value :
         {T{0.575},
          T{-0.575},
          T{2.55},
          T{-0.0},
          T{0.0},
          T{2.5},
          std::numeric_limits<T>::max(),
          std::numeric_limits<T>::denorm_min(),
          std::numeric_limits<T>::infinity(),
          -std::numeric_limits<T>::infinity()}) {
      for (int32_t scale :
           {std::numeric_limits<int32_t>::min(),
            -400,
            -38,
            -1,
            0,
            1,
            2,
            38,
            400,
            std::numeric_limits<int32_t>::max()}) {
        const auto expected = functions::round<T, int32_t>(value, scale);
        const auto actual = round<T>(value, scale).value();
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_EQ(std::bit_cast<Bits>(expected), std::bit_cast<Bits>(actual));
        }
      }
      EXPECT_EQ(
          std::bit_cast<Bits>(
              evaluateOnce<T>("round(c0)", std::optional<T>{value}).value()),
          std::bit_cast<Bits>(functions::round<T, int32_t>(value, 0)));
    }
    EXPECT_EQ(round<T>(T{2.5}, std::nullopt), std::nullopt);
  };
  check.template operator()<float>();
  check.template operator()<double>();
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

TEST_F(RoundTest, integralVariableScaleRejected) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({25, 35}), makeFlatVector<int32_t>({0, 1})});
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
