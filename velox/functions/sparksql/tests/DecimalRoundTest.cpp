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

#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
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

  core::CallTypedExprPtr createDecimalRound(
      const TypePtr& inputType,
      const std::optional<int32_t>& scaleOpt,
      bool castScale) {
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
        DECIMAL(resultPrecision, resultScale),
        std::move(inputs),
        kRoundDecimal);
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

  // Preserve a caller-resolved result precision.
  const auto inputType = DECIMAL(3, 2);
  auto expression = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(3, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
          std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(0)),
      },
      kRoundDecimal);
  testEncodings(
      expression,
      {makeFlatVector<int64_t>({123, 552, -999, 0}, inputType)},
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(3, 0)));

  // Zero fits a narrow caller-resolved result precision even when the
  // negative-scale multiplier exceeds the result precision.
  const auto narrowInputType = DECIMAL(38, 0);
  expression = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(1, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(narrowInputType, "c0"),
          std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(-38)),
      },
      kRoundDecimal);
  testEncodings(
      expression,
      {makeFlatVector<int128_t>({1, -1, 0}, narrowInputType)},
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(1, 0)));

  // Enforce a narrow caller-resolved precision for nonzero results.
  const auto shortInputType = DECIMAL(3, 0);
  expression = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(1, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(shortInputType, "c0"),
          std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(-1)),
      },
      kRoundDecimal);
  const auto shortInput =
      makeFlatVector<int64_t>({14, 4, -14, -4, 0}, shortInputType);
  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({shortInput})),
      "Decimal overflow in round.");
  const auto tryExpression = std::make_shared<const core::CallTypedExpr>(
      expression->type(), std::vector<core::TypedExprPtr>{expression}, "try");
  testEncodings(
      tryExpression,
      {shortInput},
      makeNullableFlatVector<int64_t>(
          {std::nullopt, 0, std::nullopt, 0, 0}, DECIMAL(1, 0)));

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

TEST_F(DecimalRoundTest, extremeNegativeScales) {
  const auto type = DECIMAL(3, 2);
  const auto input = makeFlatVector<int64_t>({0, 1, -1}, type);
  testDecimalRound(
      input,
      std::numeric_limits<int32_t>::min(),
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));

  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto boundaryInput =
      makeFlatVector<int128_t>({maximum, -maximum, 0}, DECIMAL(38, 0));
  testDecimalRound(
      boundaryInput,
      std::numeric_limits<int32_t>::min(),
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));
  testDecimalRound(
      boundaryInput, -39, makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, flatNoNullsFastPath) {
  auto supportsFastPath = [&](const TypePtr& inputType, int32_t scale) {
    std::vector<core::TypedExprPtr> expressions = {
        createDecimalRound(inputType, scale, false)};
    exec::ExprSet exprSet(std::move(expressions), &execCtx_);
    return exprSet.exprs().front()->supportsFlatNoNullsFastPath();
  };

  EXPECT_TRUE(supportsFastPath(DECIMAL(3, 2), 0));
  EXPECT_FALSE(supportsFastPath(DECIMAL(38, 0), -1));
  EXPECT_TRUE(
      supportsFastPath(DECIMAL(3, 2), std::numeric_limits<int32_t>::min()));
}

TEST_F(DecimalRoundTest, nullScale) {
  const auto inputType = DECIMAL(3, 2);
  const auto input = makeFlatVector<int64_t>({125, 135, 0}, inputType);
  auto expression = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(2, 0),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
          core::ConstantTypedExpr::makeNull(INTEGER()),
      },
      kRoundDecimal);

  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({input})),
      "The second argument of decimal_round must not be NULL.");
}

TEST_F(DecimalRoundTest, mismatchedResultScale) {
  const auto inputType = DECIMAL(3, 2);
  auto expression = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(3, 1),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
          std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(0)),
      },
      kRoundDecimal);

  VELOX_ASSERT_THROW(
      evaluate(
          expression,
          makeRowVector({makeFlatVector<int64_t>({123, -123, 0}, inputType)})),
      "Result scale must match the resolved scale for function: decimal_round.");
}

TEST_F(DecimalRoundTest, negativeScaleOverflow) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto values =
      makeFlatVector<int128_t>({maximum, 14, -maximum, 25}, type);
  const auto expression = createDecimalRound(type, -1, false);

  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({values})),
      "Decimal overflow in round.");

  const auto tryExpression = std::make_shared<const core::CallTypedExpr>(
      expression->type(), std::vector<core::TypedExprPtr>{expression}, "try");
  testEncodings(
      tryExpression,
      {values},
      makeNullableFlatVector<int128_t>(
          {std::nullopt, 10, std::nullopt, 30}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, scaleBoundaryMinus38) {
  const auto type = DECIMAL(38, 0);
  const auto divisor = DecimalUtil::kPowersOfTen[38];
  const auto half = divisor / 2;
  const auto values = makeFlatVector<int128_t>(
      {
          half - 1,
          half,
          half + 1,
          -half + 1,
          -half,
          -half - 1,
          0,
      },
      type);
  const auto expression = createDecimalRound(type, -38, false);
  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({values})),
      "Decimal overflow in round.");

  const auto tryExpression = std::make_shared<const core::CallTypedExpr>(
      expression->type(), std::vector<core::TypedExprPtr>{expression}, "try");
  testEncodings(
      tryExpression,
      {values},
      makeNullableFlatVector<int128_t>(
          {
              0,
              std::nullopt,
              std::nullopt,
              0,
              std::nullopt,
              std::nullopt,
              0,
          },
          DECIMAL(38, 0)));
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
