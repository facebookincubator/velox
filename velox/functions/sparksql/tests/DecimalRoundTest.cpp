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
#include <string>
#include <vector>

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
      const int32_t requiredPrecision =
          roundScale <= std::numeric_limits<int32_t>::min() + 1
          ? std::numeric_limits<int32_t>::min()
          : -roundScale + 1;
      const auto newPrecision =
          std::max(integralLeastNumDigits, requiredPrecision);
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
    const auto [resultPrecision, resultScale] =
        getResultPrecisionScale(inputPrecision, inputScale, scale);
    return std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale), std::move(inputs), functionName);
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

  void testDecimalRoundThrows(
      const VectorPtr& input,
      int32_t scale,
      const std::string& functionName,
      const std::string& message) {
    for (auto castScale : {true, false}) {
      auto expr =
          createDecimalRound(input->type(), scale, castScale, functionName);
      VELOX_ASSERT_THROW(testEncodings(expr, {input}, nullptr), message);
    }
  }

  void testDecimalRoundTry(
      const VectorPtr& input,
      int32_t scale,
      const VectorPtr& expected,
      const std::string& functionName) {
    for (auto castScale : {true, false}) {
      auto roundExpr =
          createDecimalRound(input->type(), scale, castScale, functionName);
      auto tryExpr = std::make_shared<const core::CallTypedExpr>(
          roundExpr->type(), std::vector<core::TypedExprPtr>{roundExpr}, "try");
      testEncodings(tryExpr, {input}, expected);
    }
  }

  template <typename TInput, typename TResult = TInput>
  void testExtremeScale(
      const TypePtr& inputType,
      const std::string& functionName,
      int32_t scale = std::numeric_limits<int32_t>::min()) {
    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        getResultPrecisionScale(inputPrecision, inputScale, scale);
    const auto resultType = DECIMAL(resultPrecision, resultScale);

    testDecimalRound(
        makeFlatVector<TInput>({0, 0, 0}, inputType),
        scale,
        makeFlatVector<TResult>({0, 0, 0}, resultType),
        functionName);
    testDecimalRoundThrows(
        makeFlatVector<TInput>({1, 1, 1}, inputType),
        scale,
        functionName,
        "Underflow while rounding to scale " + std::to_string(scale));
    testDecimalRoundTry(
        makeFlatVector<TInput>({0, 1, 0, -1}, inputType),
        scale,
        makeNullableFlatVector<TResult>(
            {0, std::nullopt, 0, std::nullopt}, resultType),
        functionName);
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

  // Round to INT_MAX and INT_MIN.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      std::numeric_limits<int32_t>::max(),
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 1)));
  testExtremeScale<int64_t>(DECIMAL(3, 1), kRoundDecimal);
  testExtremeScale<int128_t>(DECIMAL(30, 1), kRoundDecimal);
  testExtremeScale<int64_t>(
      DECIMAL(3, 2), kRoundDecimal, std::numeric_limits<int32_t>::min() + 1);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kRoundDecimal, std::numeric_limits<int32_t>::min() + 2);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kRoundDecimal, std::numeric_limits<int32_t>::min() + 3);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2),
      kRoundDecimal,
      static_cast<int32_t>(2 - (kMaxJavaBigIntegerPowerOfTenExponent + 1)));
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kRoundDecimal, -1'000'000'000);
  testDecimalRound(
      makeFlatVector<int64_t>({1, -1, 0}, DECIMAL(3, 2)),
      static_cast<int32_t>(2 - kMaxJavaBigIntegerPowerOfTenExponent),
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, bround) {
  testDecimalRound(
      makeFlatVector<int64_t>({125, 135, 145, -125, -135, -145}, DECIMAL(3, 2)),
      1,
      makeFlatVector<int64_t>({12, 14, 14, -12, -14, -14}, DECIMAL(3, 1)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>({250, 350, 450, -250, -350, -450}, DECIMAL(4, 2)),
      0,
      makeFlatVector<int64_t>({2, 4, 4, -2, -4, -4}, DECIMAL(3, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>({150, 250, 350, 450, -150, -250}, DECIMAL(3, 1)),
      -1,
      makeFlatVector<int64_t>({20, 20, 40, 40, -20, -20}, DECIMAL(3, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int128_t>(
          {
              DecimalUtil::kPowersOfTen[37] + 5,
              DecimalUtil::kPowersOfTen[37] + 15,
              -DecimalUtil::kPowersOfTen[37] - 5,
              -DecimalUtil::kPowersOfTen[37] - 15,
          },
          DECIMAL(38, 1)),
      0,
      makeFlatVector<int128_t>(
          {
              DecimalUtil::kPowersOfTen[36],
              DecimalUtil::kPowersOfTen[36] + 2,
              -DecimalUtil::kPowersOfTen[36],
              -DecimalUtil::kPowersOfTen[36] - 2,
          },
          DECIMAL(38, 0)),
      kBRoundDecimal);

  // Long decimal to short decimal.
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

  // Short decimal to long decimal.
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
      makeFlatVector<int128_t>(
          {
              int128_t{6} * DecimalUtil::kPowersOfTen[37],
              -int128_t{6} * DecimalUtil::kPowersOfTen[37],
              0,
          },
          DECIMAL(38, 38)),
      -1,
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(2, 0)),
      kBRoundDecimal);

  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      std::nullopt,
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)),
      kBRoundDecimal);

  testExtremeScale<int64_t>(DECIMAL(3, 1), kBRoundDecimal);
  testExtremeScale<int128_t>(DECIMAL(30, 1), kBRoundDecimal);
  testExtremeScale<int64_t>(
      DECIMAL(3, 2), kBRoundDecimal, std::numeric_limits<int32_t>::min() + 1);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kBRoundDecimal, std::numeric_limits<int32_t>::min() + 2);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kBRoundDecimal, std::numeric_limits<int32_t>::min() + 3);
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2),
      kBRoundDecimal,
      static_cast<int32_t>(2 - (kMaxJavaBigIntegerPowerOfTenExponent + 1)));
  testExtremeScale<int64_t, int128_t>(
      DECIMAL(3, 2), kBRoundDecimal, -1'000'000'000);
  testDecimalRound(
      makeFlatVector<int64_t>({1, -1, 0}, DECIMAL(3, 2)),
      static_cast<int32_t>(2 - kMaxJavaBigIntegerPowerOfTenExponent),
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)),
      kBRoundDecimal);
}

TEST_F(DecimalRoundTest, overflow) {
  const auto type = DECIMAL(38, 0);
  const auto max = DecimalUtil::kPowersOfTen[38] - 1;
  for (const auto functionName : {kRoundDecimal, kBRoundDecimal}) {
    testDecimalRoundThrows(
        makeFlatVector<int128_t>({max, max, max}, type),
        -1,
        functionName,
        "Overflow while rounding decimal to precision 38 and scale 0");
    testDecimalRoundTry(
        makeFlatVector<int128_t>({0, max, 0, -max}, type),
        -1,
        makeNullableFlatVector<int128_t>(
            {0, std::nullopt, 0, std::nullopt}, type),
        functionName);
  }
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
