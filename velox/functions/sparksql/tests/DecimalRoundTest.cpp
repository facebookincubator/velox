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

#include <bit>
#include <limits>
#include <string>

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

  static std::pair<uint8_t, uint8_t> getBRoundResultPrecisionScale(
      uint8_t precision,
      uint8_t scale,
      int32_t roundScale) {
    const int32_t integralLeastNumDigits = precision - scale + 1;
    if (roundScale < 0) {
      const int32_t requiredPrecision = std::bit_cast<int32_t>(
          uint32_t{0} - static_cast<uint32_t>(roundScale) + uint32_t{1});
      return {
          std::min(
              std::max(integralLeastNumDigits, requiredPrecision),
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
      bool castScale,
      const std::string& functionName = kRoundDecimal) {
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
    const auto [resultPrecision, resultScale] = functionName == kBRoundDecimal
        ? getBRoundResultPrecisionScale(inputPrecision, inputScale, scale)
        : getResultPrecisionScale(inputPrecision, inputScale, scale);
    return std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale), std::move(inputs), functionName);
  }

  core::CallTypedExprPtr createDecimalBRoundWithNullScale(
      const TypePtr& inputType) {
    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        getBRoundResultPrecisionScale(inputPrecision, inputScale, 0);
    return std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
            core::ConstantTypedExpr::makeNull(INTEGER()),
        },
        kBRoundDecimal);
  }

  void testDecimalRound(
      const VectorPtr& input,
      const std::optional<int32_t>& scaleOpt,
      const VectorPtr& expected,
      const std::string& functionName = kRoundDecimal) {
    for (auto castScale : {true, false}) {
      auto expr =
          createDecimalRound(input->type(), scaleOpt, castScale, functionName);
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
  testDecimalRound(
      makeFlatVector<int64_t>({125, 135, 145, -125, -135, -145}, DECIMAL(3, 2)),
      1,
      makeFlatVector<int64_t>({12, 14, 14, -12, -14, -14}, DECIMAL(3, 1)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>({150, 250, 350, 450, -150, -250}, DECIMAL(3, 1)),
      -1,
      makeFlatVector<int64_t>({20, 20, 40, 40, -20, -20}, DECIMAL(3, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int128_t>(
          {
              DecimalUtil::kPowersOfTen[19] - 150,
              -DecimalUtil::kPowersOfTen[19] + 150,
              0,
          },
          DECIMAL(19, 2)),
      0,
      makeFlatVector<int64_t>(
          {
              DecimalUtil::kPowersOfTen[17] - 2,
              -DecimalUtil::kPowersOfTen[17] + 2,
              0,
          },
          DECIMAL(18, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>(
          {
              DecimalUtil::kPowersOfTen[18] - 5,
              -DecimalUtil::kPowersOfTen[18] + 5,
              0,
          },
          DECIMAL(18, 0)),
      -1,
      makeFlatVector<int128_t>(
          {
              DecimalUtil::kPowersOfTen[18],
              -DecimalUtil::kPowersOfTen[18],
              0,
          },
          DECIMAL(19, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(3, 2)),
      std::numeric_limits<int32_t>::min(),
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(2, 0)),
      kBRoundDecimal);
  testDecimalRound(
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(3, 2)),
      std::numeric_limits<int32_t>::min() + 1,
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(2, 0)),
      kBRoundDecimal);
  testDecimalRound(
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(3, 2)),
      std::numeric_limits<int32_t>::min() + 2,
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)),
      kBRoundDecimal);
}

TEST_F(DecimalRoundTest, broundNulls) {
  testDecimalRound(
      makeNullableFlatVector<int64_t>(
          {125, std::nullopt, 135, 0}, DECIMAL(3, 2)),
      1,
      makeNullableFlatVector<int64_t>({12, std::nullopt, 14, 0}, DECIMAL(3, 1)),
      kBRoundDecimal);
  testDecimalRound(
      makeNullableFlatVector<int128_t>(
          {125, std::nullopt, 135, 0}, DECIMAL(19, 2)),
      1,
      makeNullableFlatVector<int128_t>(
          {12, std::nullopt, 14, 0}, DECIMAL(19, 1)),
      kBRoundDecimal);

  const auto shortInput = makeFlatVector<int64_t>({125, 135, 0}, DECIMAL(3, 2));
  testEncodings(
      createDecimalBRoundWithNullScale(shortInput->type()),
      {shortInput},
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(2, 0)));

  const auto longInput =
      makeFlatVector<int128_t>({125, 135, 0}, DECIMAL(38, 2));
  testEncodings(
      createDecimalBRoundWithNullScale(longInput->type()),
      {longInput},
      makeNullableFlatVector<int128_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(37, 0)));
}

TEST_F(DecimalRoundTest, broundExtremeScaleUnderflow) {
  const auto type = DECIMAL(3, 2);
  const auto input = makeFlatVector<int64_t>({0, 1, 0, -1, 0}, type);
  const auto expression = createDecimalRound(
      type, std::numeric_limits<int32_t>::min(), false, kBRoundDecimal);
  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({input})),
      "Underflow while rounding to scale -2147483648");

  const auto tryExpression = std::make_shared<const core::CallTypedExpr>(
      expression->type(), std::vector<core::TypedExprPtr>{expression}, "try");
  testEncodings(
      tryExpression,
      {input},
      makeNullableFlatVector<int64_t>(
          {0, std::nullopt, 0, std::nullopt, 0}, DECIMAL(2, 0)));
}

TEST_F(DecimalRoundTest, broundUnderflowBoundary) {
  constexpr int32_t kMaximumSupportedScale =
      2 - detail::kMaxJavaBigIntegerPowerOfTenExponent;
  constexpr int32_t kFirstUnderflowScale = kMaximumSupportedScale - 1;
  const auto type = DECIMAL(3, 2);

  testDecimalRound(
      makeFlatVector<int64_t>({1, -1, 0}, type),
      kMaximumSupportedScale,
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)),
      kBRoundDecimal);

  const auto input = makeFlatVector<int64_t>({0, 1, 0, -1, 0}, type);
  const auto expression =
      createDecimalRound(type, kFirstUnderflowScale, false, kBRoundDecimal);
  VELOX_ASSERT_THROW(
      evaluate(expression, makeRowVector({input})),
      "Underflow while rounding to scale -536870918");

  const auto tryExpression = std::make_shared<const core::CallTypedExpr>(
      expression->type(), std::vector<core::TypedExprPtr>{expression}, "try");
  testEncodings(
      tryExpression,
      {input},
      makeNullableFlatVector<int128_t>(
          {0, std::nullopt, 0, std::nullopt, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, broundOverflow) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input =
      makeRowVector({makeFlatVector<int128_t>({maximum, -maximum}, type)});
  const auto expression = createDecimalRound(type, -1, false, kBRoundDecimal);
  VELOX_ASSERT_THROW(evaluate(expression, input), "Decimal overflow");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
