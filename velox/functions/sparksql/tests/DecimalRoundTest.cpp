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

#include "velox/functions/sparksql/specialforms/DecimalRound.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/BRound.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalRoundTest : public SparkFunctionBaseTest {
 protected:
  /// Computes result precision and scale for decimal rounding. Matches Spark's
  /// logic from version 3.3+.
  static std::pair<uint8_t, uint8_t> getResultPrecisionScale(
      uint8_t precision,
      uint8_t scale,
      int32_t roundScale) {
    const int32_t integralLeastNumDigits = precision - scale + 1;
    if (roundScale < 0) {
      const auto newPrecision = std::max(
          integralLeastNumDigits,
          -std::max(
              roundScale,
              -static_cast<int32_t>(LongDecimalType::kMaxPrecision)) +
              1);
      return {
          std::min(
              newPrecision,
              static_cast<int32_t>(LongDecimalType::kMaxPrecision)),
          0};
    }
    const uint8_t newScale = std::min(static_cast<int32_t>(scale), roundScale);
    return {
        std::min(
            integralLeastNumDigits + newScale,
            static_cast<int32_t>(LongDecimalType::kMaxPrecision)),
        newScale};
  }

  core::CallTypedExprPtr createDecimalRounding(
      const TypePtr& inputType,
      const std::optional<int32_t>& scaleOpt,
      bool castScale,
      const char* functionName) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0")};
    int32_t scale = 0;
    if (scaleOpt.has_value()) {
      scale = scaleOpt.value();
      if (castScale) {
        // It is a common case in Spark for the second argument to be cast from
        // bigint to integer.
        inputs.emplace_back(
            std::make_shared<core::CastTypedExpr>(
                INTEGER(),
                std::make_shared<core::ConstantTypedExpr>(
                    BIGINT(), variant((int64_t)scale)),
                true /*nullOnFailure*/));
      } else {
        inputs.emplace_back(
            std::make_shared<core::ConstantTypedExpr>(
                INTEGER(), variant(scale)));
      }
    }

    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        getResultPrecisionScale(inputPrecision, inputScale, scale);
    return std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale), std::move(inputs), functionName);
  }

  core::CallTypedExprPtr createDecimalRound(
      const TypePtr& inputType,
      const std::optional<int32_t>& scaleOpt,
      bool castScale) {
    return createDecimalRounding(inputType, scaleOpt, castScale, kRoundDecimal);
  }

  void testDecimalRound(
      const VectorPtr& input,
      const std::optional<int32_t>& scaleOpt,
      const VectorPtr& expected) {
    for (auto castScale : {true, false}) {
      auto expr = createDecimalRound(input->type(), scaleOpt, castScale);
      testEncodings(expr, {input}, expected);
    }
  }

  void testDecimalBRound(
      const VectorPtr& input,
      const std::optional<int32_t>& scaleOpt,
      const VectorPtr& expected) {
    for (auto castScale : {true, false}) {
      auto expr = createDecimalRounding(
          input->type(), scaleOpt, castScale, kBRoundDecimal);
      testEncodings(expr, {input}, expected);
    }
  }

  void testDecimalRoundingWithNullScale(
      const VectorPtr& input,
      const VectorPtr& expected,
      const char* functionName) {
    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*input->type());
    const auto [resultPrecision, resultScale] =
        getResultPrecisionScale(inputPrecision, inputScale, 0);
    auto expr = std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(input->type(), "c0"),
            std::make_shared<core::ConstantTypedExpr>(
                INTEGER(), variant::null(TypeKind::INTEGER))},
        functionName);
    testEncodings(expr, {input}, expected);
  }
};

TEST_F(DecimalRoundTest, round) {
  // Round to 'scale'.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      3,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 3)));

  // Round to 'scale - 1'.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      2,
      makeFlatVector<int64_t>({12, 55, -100, 0}, DECIMAL(3, 2)));

  // Round to 0 decimal scale.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      0,
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      std::nullopt,
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      0,
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      std::nullopt,
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));

  // Round to negative decimal scale.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      -1,
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      -1,
      makeFlatVector<int64_t>({10, 60, -100, 0}, DECIMAL(3, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      -3,
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(4, 0)));

  // Round long decimals to short decimals.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5000000000000000000, -999999999999999999, 0},
          DECIMAL(19, 19)),
      14,
      makeNullableFlatVector<int64_t>(
          {12345678901235, 50000000000000, -10'000'000'000'000, 0},
          DECIMAL(15, 14)));
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      -9,
      makeFlatVector<int64_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(15, 0)));

  // Round long decimals to long decimals.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      14,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(20, 5)));
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 5)),
      -9,
      makeFlatVector<int128_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(28, 0)));

  // Result precision is 38.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 0)),
      -38,
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));

  // Round to a scale exceeding the max precision of long decimal.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      std::numeric_limits<int32_t>::max(),
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 1)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      std::numeric_limits<int32_t>::min(),
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));

  // Round to INT_MAX and INT_MIN.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      std::numeric_limits<int32_t>::max(),
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 1)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      std::numeric_limits<int32_t>::min(),
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, bround) {
  testDecimalBRound(
      makeNullableFlatVector<int64_t>({25, 35, std::nullopt}, DECIMAL(3, 1)),
      std::nullopt,
      makeNullableFlatVector<int64_t>({2, 4, std::nullopt}, DECIMAL(3, 0)));

  testDecimalBRound(
      makeNullableFlatVector<int128_t>(
          {DecimalUtil::kPowersOfTen[38] - 1, 0, std::nullopt},
          DECIMAL(38, 38)),
      std::nullopt,
      makeNullableFlatVector<int64_t>({1, 0, std::nullopt}, DECIMAL(1, 0)));

  testDecimalBRound(
      makeFlatVector<int64_t>({25, 35, -25, -35}, DECIMAL(3, 1)),
      0,
      makeFlatVector<int64_t>({2, 4, -2, -4}, DECIMAL(3, 0)));

  testDecimalBRound(
      makeFlatVector<int64_t>({245, 255, -245, -255}, DECIMAL(4, 2)),
      1,
      makeFlatVector<int64_t>({24, 26, -24, -26}, DECIMAL(4, 1)));

  const int128_t sixTenths =
      DecimalUtil::kPowersOfTen[37] * static_cast<int128_t>(6);
  testDecimalBRound(
      makeFlatVector<int128_t>({sixTenths, -sixTenths, 0}, DECIMAL(38, 38)),
      -1,
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(2, 0)));

  testDecimalRoundingWithNullScale(
      makeFlatVector<int64_t>({25, 35, 45}, DECIMAL(3, 1)),
      BaseVector::createNullConstant(DECIMAL(3, 0), 3, pool()),
      kBRoundDecimal);
}

TEST_F(DecimalRoundTest, negativeScaleOverflow) {
  const int128_t maxDecimal = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input = makeFlatVector<int128_t>({maxDecimal}, DECIMAL(38, 0));
  const auto row = makeRowVector({input});

  const auto expr =
      createDecimalRounding(input->type(), -1, false, kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(expr, row), "Decimal overflow");
}

TEST_F(DecimalRoundTest, broundMixedErrors) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input = makeNullableFlatVector<int128_t>(
      {25, maximum, -35, -maximum, std::nullopt, 45}, type);
  const auto call = createDecimalRounding(type, -1, false, kBRoundDecimal);
  const auto tryCall = std::make_shared<core::CallTypedExpr>(
      call->type(), std::vector<core::TypedExprPtr>{call}, "try");
  testEncodings(
      tryCall,
      {input},
      makeNullableFlatVector<int128_t>(
          {20, std::nullopt, -40, std::nullopt, std::nullopt, 40}, type));

  SelectivityVector selected(input->size(), false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.setValid(5, true);
  selected.updateBounds();
  const auto result =
      evaluate<SimpleVector<int128_t>>(call, makeRowVector({input}), selected);
  EXPECT_EQ(result->valueAt(0), 20);
  EXPECT_EQ(result->valueAt(2), -40);
  EXPECT_EQ(result->valueAt(5), 40);
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int128_t>(
          {20, std::nullopt, -40, std::nullopt, std::nullopt, 40}, type),
      evaluate(tryCall, makeRowVector({wrapInLazyDictionary(input)})));
  const auto condition =
      makeFlatVector<bool>({true, false, true, false, false, true});
  const auto branch = std::make_shared<core::CallTypedExpr>(
      type,
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c1"),
          call,
          std::make_shared<core::FieldAccessTypedExpr>(type, "c0")},
      "if");
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int128_t>(
          {20, maximum, -40, -maximum, std::nullopt, 40}, type),
      evaluate(branch, makeRowVector({input, condition})));
}

TEST_F(DecimalRoundTest, broundPrecisionAndScaleMatrix) {
  // Exercise every scale at the storage-width and maximum-precision boundaries.
  for (uint8_t precision : {1, 18, 19, 20, 37, 38}) {
    for (uint8_t scale = 0; scale <= precision; ++scale) {
      SCOPED_TRACE(fmt::format("decimal({}, {})", precision, scale));
      const auto type = DECIMAL(precision, scale);
      const auto input = type->isShortDecimal()
          ? VectorPtr(
                makeNullableFlatVector<int64_t>(
                    {2, 5, 6, -5, -6, 0, std::nullopt}, type))
          : VectorPtr(
                makeNullableFlatVector<int128_t>(
                    {2, 5, 6, -5, -6, 0, std::nullopt}, type));
      for (int32_t requested :
           {static_cast<int32_t>(scale), scale + 1, scale - 1, -39, -100}) {
        const auto [resultPrecision, resultScale] =
            getResultPrecisionScale(precision, scale, requested);
        const auto resultType = DECIMAL(resultPrecision, resultScale);
        // These small unscaled inputs are either unchanged, rounded across a
        // one-digit midpoint, or eliminated entirely by a coarse scale.
        const int128_t unit = requested < 0 && requested == scale - 1 ? 10 : 1;
        const std::vector<std::optional<int128_t>> values = requested >= scale
            ? std::vector<
                  std::optional<int128_t>>{2, 5, 6, -5, -6, 0, std::nullopt}
            : requested == scale - 1
            ? std::vector<std::optional<
                  int128_t>>{0, 0, unit, 0, -unit, 0, std::nullopt}
            : std::vector<std::optional<int128_t>>{
                  0, 0, 0, 0, 0, 0, std::nullopt};
        VectorPtr expected;
        if (resultType->isShortDecimal()) {
          std::vector<std::optional<int64_t>> shortValues;
          for (const auto& value : values) {
            shortValues.emplace_back(
                value ? std::optional<int64_t>(*value) : std::nullopt);
          }
          expected = makeNullableFlatVector<int64_t>(shortValues, resultType);
        } else {
          expected = makeNullableFlatVector<int128_t>(values, resultType);
        }
        testDecimalBRound(input, requested, expected);
      }
    }
  }
}

TEST_F(DecimalRoundTest, broundInvalidOutputType) {
  const auto input = makeRowVector({makeFlatVector<int128_t>(
      {DecimalUtil::kPowersOfTen[37]}, DECIMAL(38, 0))});
  const auto badCall = std::make_shared<core::CallTypedExpr>(
      DECIMAL(1, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DECIMAL(38, 0), "c0")},
      kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(badCall, input), "result type");
  const auto wrongScale = std::make_shared<core::CallTypedExpr>(
      DECIMAL(38, 1),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DECIMAL(38, 0), "c0")},
      kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(wrongScale, input), "result type");
}

TEST_F(DecimalRoundTest, broundPrecisionCarry) {
  for (uint8_t precision : {1, 18, 19, 20, 37, 38}) {
    for (uint8_t scale : {uint8_t{0}, uint8_t{1}, precision}) {
      const auto type = DECIMAL(precision, scale);
      const auto maximum = DecimalUtil::kPowersOfTen[precision] - 1;
      const auto input = type->isShortDecimal()
          ? VectorPtr(
                makeNullableFlatVector<int64_t>(
                    {static_cast<int64_t>(maximum),
                     -static_cast<int64_t>(maximum),
                     std::nullopt},
                    type))
          : VectorPtr(
                makeNullableFlatVector<int128_t>(
                    {maximum, -maximum, std::nullopt}, type));
      for (int32_t requested : {static_cast<int32_t>(scale), 0}) {
        const auto [resultPrecision, resultScale] =
            getResultPrecisionScale(precision, scale, requested);
        const auto resultType = DECIMAL(resultPrecision, resultScale);
        const int128_t expectedValue = scale == requested
            ? maximum
            : DecimalUtil::kPowersOfTen[precision - scale];
        const auto expected = resultType->isShortDecimal()
            ? VectorPtr(
                  makeNullableFlatVector<int64_t>(
                      {static_cast<int64_t>(expectedValue),
                       -static_cast<int64_t>(expectedValue),
                       std::nullopt},
                      resultType))
            : VectorPtr(
                  makeNullableFlatVector<int128_t>(
                      {expectedValue, -expectedValue, std::nullopt},
                      resultType));
        testDecimalBRound(input, requested, expected);
      }
    }
  }
}

TEST_F(DecimalRoundTest, broundCoarseScaleBoundary) {
  const auto half = 5 * DecimalUtil::kPowersOfTen[37];
  const auto type = DECIMAL(38, 0);
  const auto input = makeFlatVector<int128_t>(
      {half - 1, half, half + 1, -half + 1, -half, -half - 1}, type);
  const auto call = createDecimalRounding(type, -38, false, kBRoundDecimal);
  const auto tryCall = std::make_shared<core::CallTypedExpr>(
      type, std::vector<core::TypedExprPtr>{call}, "try");
  testEncodings(
      tryCall,
      {input},
      makeNullableFlatVector<int128_t>(
          {0, 0, std::nullopt, 0, 0, std::nullopt}, type));
  testDecimalBRound(
      input, -39, makeFlatVector<int128_t>({0, 0, 0, 0, 0, 0}, type));
}

TEST_F(DecimalRoundTest, broundSupportedScaleInterval) {
  const auto shortInput =
      makeNullableFlatVector<int64_t>({25, -25, std::nullopt}, DECIMAL(3, 1));
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto longInput = makeNullableFlatVector<int128_t>(
      {maximum, -maximum, std::nullopt}, DECIMAL(38, 38));
  for (int32_t scale : {-400, -399}) {
    const auto zero =
        makeNullableFlatVector<int128_t>({0, 0, std::nullopt}, DECIMAL(38, 0));
    testDecimalBRound(shortInput, scale, zero);
    testDecimalBRound(longInput, scale, zero);
  }
  for (int32_t scale : {399, 400}) {
    testDecimalBRound(
        shortInput,
        scale,
        makeNullableFlatVector<int64_t>(
            {25, -25, std::nullopt}, DECIMAL(4, 1)));
    testDecimalBRound(longInput, scale, longInput);
  }
  for (const auto& input : {VectorPtr(shortInput), VectorPtr(longInput)}) {
    for (int32_t scale :
         {std::numeric_limits<int32_t>::min(),
          std::numeric_limits<int32_t>::min() + 1,
          -1000,
          -401,
          401,
          1000,
          std::numeric_limits<int32_t>::max()}) {
      const auto call =
          createDecimalRounding(input->type(), scale, false, kBRoundDecimal);
      VELOX_ASSERT_THROW(
          std::make_unique<exec::ExprSet>(
              std::vector<core::TypedExprPtr>{call}, &execCtx_),
          "between -400 and 400");
    }
  }
}

TEST_F(DecimalRoundTest, broundNullScaleSkipsChild) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input =
      makeNullableFlatVector<int128_t>({maximum, -maximum, std::nullopt}, type);
  const auto child = createDecimalRounding(type, -1, false, kBRoundDecimal);
  const auto call = std::make_shared<core::CallTypedExpr>(
      type,
      std::vector<core::TypedExprPtr>{
          child,
          std::make_shared<core::ConstantTypedExpr>(
              INTEGER(), variant::null(TypeKind::INTEGER))},
      kBRoundDecimal);
  testEncodings(
      call,
      {input},
      BaseVector::createNullConstant(type, input->size(), pool()));
}

TEST_F(DecimalRoundTest, broundScaleValidation) {
  const auto type = DECIMAL(3, 1);
  const auto input = makeRowVector(
      {makeFlatVector<int64_t>({25, 35}, type),
       makeFlatVector<int32_t>({0, 1})});
  const auto call = std::make_shared<core::CallTypedExpr>(
      DECIMAL(3, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(type, "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(call, input), "constant expression");
  const auto wrongType = std::make_shared<core::CallTypedExpr>(
      DECIMAL(3, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(type, "c0"),
          std::make_shared<core::ConstantTypedExpr>(
              BIGINT(), variant(int64_t{0}))},
      kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(wrongType, input), "INTEGER");
}

TEST_F(DecimalRoundTest, broundPrefixedRegistration) {
  registerBRoundFunctions("test_prefix_");
  const auto input =
      makeNullableFlatVector<int64_t>({25, 35, std::nullopt}, DECIMAL(3, 1));
  const auto call = createDecimalRounding(
      input->type(), 0, false, "test_prefix_decimal_bround");
  testEncodings(
      call,
      {input},
      makeNullableFlatVector<int64_t>({2, 4, std::nullopt}, DECIMAL(3, 0)));
  EXPECT_EQ(
      evaluateOnce<double>(
          "test_prefix_bround(c0)", std::optional<double>{3.5}),
      4.0);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
