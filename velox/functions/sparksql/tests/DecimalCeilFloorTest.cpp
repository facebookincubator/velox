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

#include "velox/functions/sparksql/specialforms/DecimalCeilFloor.h"
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalCeilFloorTest : public SparkFunctionBaseTest {
 protected:
  enum class Mode { kCeil, kFloor };

  core::CallTypedExprPtr
  createCall(const TypePtr& inputType, int32_t scale, Mode mode) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
        std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(scale))};
    const auto [precision, inputScale] = getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        DecimalCeilFloorCallToSpecialFormBase::getResultPrecisionScale(
            precision, inputScale, scale);
    const std::string& name = mode == Mode::kCeil
        ? std::string(DecimalCeilCallToSpecialForm::kCeilDecimal)
        : std::string(DecimalFloorCallToSpecialForm::kFloorDecimal);
    return std::make_shared<const core::CallTypedExpr>(
        DECIMAL(resultPrecision, resultScale), std::move(inputs), name);
  }

  void testCall(
      const VectorPtr& input,
      int32_t scale,
      Mode mode,
      const VectorPtr& expected) {
    auto expr = createCall(input->type(), scale, mode);
    testEncodings(expr, {input}, expected);
  }
};

TEST_F(DecimalCeilFloorTest, ceilPositiveScale) {
  // scale >= input scale: identity (value and scale unchanged; only precision
  // widens). Per Spark RoundCeil.dataType, newScale = min(s, _scale), so the
  // result scale stays at the input scale (2), not the requested scale (3).
  testCall(
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(3, 2)),
      3,
      Mode::kCeil,
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(4, 2)));

  // scale < input scale: round toward +∞.
  testCall(
      makeFlatVector<int64_t>({1234, -1234, 1000, -1000, 0}, DECIMAL(5, 3)),
      1,
      Mode::kCeil,
      makeFlatVector<int64_t>({13, -12, 10, -10, 0}, DECIMAL(4, 1)));

  // scale = 0.
  testCall(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      0,
      Mode::kCeil,
      makeFlatVector<int64_t>({1, 1, 0, 0}, DECIMAL(1, 0)));
}

TEST_F(DecimalCeilFloorTest, floorPositiveScale) {
  testCall(
      makeFlatVector<int64_t>({1234, -1234, 1000, -1000, 0}, DECIMAL(5, 3)),
      1,
      Mode::kFloor,
      makeFlatVector<int64_t>({12, -13, 10, -10, 0}, DECIMAL(4, 1)));

  testCall(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      0,
      Mode::kFloor,
      makeFlatVector<int64_t>({0, 0, -1, 0}, DECIMAL(1, 0)));
}

TEST_F(DecimalCeilFloorTest, negativeScale) {
  // ceil(99.0, -1) -> 100. Spark widens precision to max(p-s+1, |scale|+1).
  testCall(
      makeFlatVector<int64_t>({990, 110, -110, 0}, DECIMAL(4, 1)),
      -1,
      Mode::kCeil,
      makeFlatVector<int64_t>({100, 20, -10, 0}, DECIMAL(4, 0)));

  testCall(
      makeFlatVector<int64_t>({990, 110, -110, 0}, DECIMAL(4, 1)),
      -1,
      Mode::kFloor,
      makeFlatVector<int64_t>({90, 10, -20, 0}, DECIMAL(4, 0)));

  // Scale lower than the integral magnitude becomes 0.
  testCall(
      makeFlatVector<int64_t>({990, -990, 500}, DECIMAL(4, 1)),
      -3,
      Mode::kCeil,
      makeFlatVector<int64_t>({1000, 0, 1000}, DECIMAL(4, 0)));
}

TEST_F(DecimalCeilFloorTest, scaleAboveInputScale) {
  // For scale >= input scale, value is unchanged but precision widens.
  testCall(
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(5, 2)),
      5,
      Mode::kCeil,
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(6, 2)));

  // Same identity behavior for floor.
  testCall(
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(5, 2)),
      5,
      Mode::kFloor,
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(6, 2)));

  // scale == input scale is also a no-op (newScale = min(s, _scale) = s,
  // result precision widens to integralLeastNumDigits + newScale).
  testCall(
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(5, 2)),
      2,
      Mode::kCeil,
      makeFlatVector<int64_t>({1234, -1234, 0}, DECIMAL(6, 2)));
}

TEST_F(DecimalCeilFloorTest, highInputScaleNegativeRoundScale) {
  // DECIMAL(38, 38) with negative round scale would compute divDigits = 39,
  // exceeding kPowersOfTen bounds. Verify the cap to 38 still produces
  // Spark-correct results (any input <= 10^38 - 1 is < 10^38, so dividing
  // by either 10^38 or 10^39 yields quotient 0 with full remainder).
  // Spark result type for (p=38, s=38, _scale=-1): DECIMAL(2, 0).
  // Values: 10^37 represents 0.1, -10^37 represents -0.1.
  constexpr int64_t kTenToTheNine = 1000000000LL;
  const int128_t kTenToTheEighteen =
      static_cast<int128_t>(kTenToTheNine) * kTenToTheNine;
  const int128_t pointOne =
      kTenToTheEighteen * kTenToTheEighteen * static_cast<int128_t>(10);
  testCall(
      makeFlatVector<int128_t>({pointOne, -pointOne, 0}, DECIMAL(38, 38)),
      -1,
      Mode::kCeil,
      makeFlatVector<int64_t>({10, 0, 0}, DECIMAL(2, 0)));

  testCall(
      makeFlatVector<int128_t>({pointOne, -pointOne, 0}, DECIMAL(38, 38)),
      -1,
      Mode::kFloor,
      makeFlatVector<int64_t>({0, -10, 0}, DECIMAL(2, 0)));
}

TEST_F(DecimalCeilFloorTest, longDecimal) {
  // Long-decimal (precision > 18) input. Output type is DECIMAL(18, 2) which
  // is short decimal (int64).
  testCall(
      makeFlatVector<int128_t>(
          {1234567890123456789LL, -1234567890123456789LL, 0}, DECIMAL(20, 5)),
      2,
      Mode::kCeil,
      makeFlatVector<int64_t>(
          {1234567890123457LL, -1234567890123456LL, 0}, DECIMAL(18, 2)));

  // Long-decimal floor path.
  testCall(
      makeFlatVector<int128_t>(
          {1234567890123456789LL, -1234567890123456789LL, 0}, DECIMAL(20, 5)),
      2,
      Mode::kFloor,
      makeFlatVector<int64_t>(
          {1234567890123456LL, -1234567890123457LL, 0}, DECIMAL(18, 2)));
}

TEST_F(DecimalCeilFloorTest, longToLongDecimal) {
  // Long-decimal input → long-decimal result (divide-only path, no multiply).
  // DECIMAL(38, 2) with scale=1 → DECIMAL(38, 1).
  const int128_t val = static_cast<int128_t>(99999999999999999LL) * 100 + 55;
  testCall(
      makeFlatVector<int128_t>({val, -val, 0}, DECIMAL(38, 2)),
      1,
      Mode::kCeil,
      makeFlatVector<int128_t>(
          {static_cast<int128_t>(99999999999999999LL) * 10 + 6,
           -static_cast<int128_t>(99999999999999999LL) * 10 - 5,
           0},
          DECIMAL(38, 1)));

  testCall(
      makeFlatVector<int128_t>({val, -val, 0}, DECIMAL(38, 2)),
      1,
      Mode::kFloor,
      makeFlatVector<int128_t>(
          {static_cast<int128_t>(99999999999999999LL) * 10 + 5,
           -static_cast<int128_t>(99999999999999999LL) * 10 - 6,
           0},
          DECIMAL(38, 1)));
}

TEST_F(DecimalCeilFloorTest, shortToLongDecimal) {
  // Short-decimal input → long-decimal result.
  // DECIMAL(18, 0) with scale=-1 → result precision 19 → long decimal.
  testCall(
      makeFlatVector<int64_t>(
          {999999999999999999LL, -999999999999999999LL, 55LL}, DECIMAL(18, 0)),
      -1,
      Mode::kCeil,
      makeFlatVector<int128_t>(
          {static_cast<int128_t>(1000000000000000000LL),
           static_cast<int128_t>(-999999999999999990LL),
           static_cast<int128_t>(60)},
          DECIMAL(19, 0)));

  testCall(
      makeFlatVector<int64_t>(
          {999999999999999999LL, -999999999999999999LL, 55LL}, DECIMAL(18, 0)),
      -1,
      Mode::kFloor,
      makeFlatVector<int128_t>(
          {static_cast<int128_t>(999999999999999990LL),
           static_cast<int128_t>(-1000000000000000000LL),
           static_cast<int128_t>(50)},
          DECIMAL(19, 0)));
}

TEST_F(DecimalCeilFloorTest, nullPropagation) {
  testCall(
      makeNullableFlatVector<int64_t>({123, std::nullopt, -456}, DECIMAL(5, 2)),
      0,
      Mode::kCeil,
      makeNullableFlatVector<int64_t>({2, std::nullopt, -4}, DECIMAL(4, 0)));
}

TEST_F(DecimalCeilFloorTest, precisionOverflow) {
  // ceil(9.9999, -1) should produce 10 -> Decimal(2, 0); large input may
  // overflow when widened. Build a 38-precision input and request a tiny
  // scale to provoke the overflow-to-NULL path.
  // Construct value just below 10^38 with scale 0; multiplying by 10 to round
  // away to a wider precision overflows the 38-digit cap and must yield NULL.
  const int128_t huge = DecimalUtil::kLongDecimalMax; // 10^38 - 1
  testCall(
      makeFlatVector<int128_t>({huge, huge, huge}, DECIMAL(38, 0)),
      -1,
      Mode::kCeil,
      makeNullableFlatVector<int128_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(38, 0)));

  // floor with negative values should also overflow to NULL.
  testCall(
      makeFlatVector<int128_t>({-huge, -huge, -huge}, DECIMAL(38, 0)),
      -1,
      Mode::kFloor,
      makeNullableFlatVector<int128_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(38, 0)));

  // Mixed: some overflow, some don't.
  // ceil(huge, -1) overflows (quotient+1 = 10^37 which == maxAbs).
  // ceil(0, -1) = 0 (no overflow).
  // ceil(-huge, -1) does NOT overflow: quotient = -(10^37-1), remainder < 0,
  // so no +1 adjustment → result = -(10^37-1)*10 = valid 38-digit number.
  const int128_t ceilNegHuge =
      -(DecimalUtil::kPowersOfTen[37] - 1) * static_cast<int128_t>(10);
  testCall(
      makeFlatVector<int128_t>({huge, 0, -huge}, DECIMAL(38, 0)),
      -1,
      Mode::kCeil,
      makeNullableFlatVector<int128_t>(
          {std::nullopt, static_cast<int128_t>(0), ceilNegHuge},
          DECIMAL(38, 0)));
}

TEST_F(DecimalCeilFloorTest, scaleClampBoundaries) {
  // scale = -100 → clamped to -38. For DECIMAL(5, 2) → result DECIMAL(38, 0).
  // Any small value divided by 10^(2+38)=10^40 (capped to 10^38) yields
  // quotient=0. Ceil of 0.01 should be 10^38 which overflows → NULL.
  // But 0 stays 0.
  testCall(
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(5, 2)),
      -100,
      Mode::kCeil,
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));

  // scale = 100 → clamped to 38 → treated as >= input scale → identity.
  testCall(
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(5, 2)),
      100,
      Mode::kCeil,
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(6, 2)));

  testCall(
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(5, 2)),
      100,
      Mode::kFloor,
      makeFlatVector<int64_t>({123, -456, 0}, DECIMAL(6, 2)));
}

TEST_F(DecimalCeilFloorTest, negativeScaleRemainderZero) {
  // When remainder is 0, both ceil and floor return the same value.
  // 100.0 (unscaled 1000, DECIMAL(4,1)) with scale=-1 → already divisible
  // by 10, so both ceil and floor should be 100.
  testCall(
      makeFlatVector<int64_t>({1000, -1000, 0}, DECIMAL(4, 1)),
      -1,
      Mode::kCeil,
      makeFlatVector<int64_t>({100, -100, 0}, DECIMAL(4, 0)));

  testCall(
      makeFlatVector<int64_t>({1000, -1000, 0}, DECIMAL(4, 1)),
      -1,
      Mode::kFloor,
      makeFlatVector<int64_t>({100, -100, 0}, DECIMAL(4, 0)));
}

TEST_F(DecimalCeilFloorTest, scaleBoundaryMinus38) {
  // DECIMAL(38, 0), scale=-38. Result DECIMAL(38, 0).
  // Any value not exactly a multiple of 10^38 should:
  //   ceil → 10^38 for positive (overflow → NULL), 0 for negative
  //   floor → 0 for positive, -10^38 for negative (overflow → NULL)
  testCall(
      makeFlatVector<int128_t>({1, -1, 0}, DECIMAL(38, 0)),
      -38,
      Mode::kCeil,
      makeNullableFlatVector<int128_t>(
          {std::nullopt, static_cast<int128_t>(0), static_cast<int128_t>(0)},
          DECIMAL(38, 0)));

  testCall(
      makeFlatVector<int128_t>({1, -1, 0}, DECIMAL(38, 0)),
      -38,
      Mode::kFloor,
      makeNullableFlatVector<int128_t>(
          {static_cast<int128_t>(0), std::nullopt, static_cast<int128_t>(0)},
          DECIMAL(38, 0)));
}

TEST_F(DecimalCeilFloorTest, floorIdentityLongDecimal) {
  // Identity path (scale >= inputScale) for long decimal floor.
  constexpr int64_t kTenToNine = 1000000000LL;
  const int128_t bigVal =
      static_cast<int128_t>(kTenToNine) * kTenToNine * kTenToNine * 123;
  testCall(
      makeFlatVector<int128_t>({bigVal, -bigVal, 0}, DECIMAL(38, 38)),
      38,
      Mode::kFloor,
      makeFlatVector<int128_t>({bigVal, -bigVal, 0}, DECIMAL(38, 38)));
}

TEST_F(DecimalCeilFloorTest, nonConstantScaleError) {
  // The scale argument must be a constant. Passing a non-constant (field
  // reference) should fail during special form construction.
  auto inputType = DECIMAL(10, 5);
  std::vector<core::TypedExprPtr> inputs = {
      std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
      std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")};
  // Result type is irrelevant — error fires before it's used.
  auto callExpr = std::make_shared<const core::CallTypedExpr>(
      DECIMAL(8, 2), std::move(inputs), "decimal_ceil");
  auto input = makeFlatVector<int64_t>({12345}, inputType);
  auto scaleCol = makeFlatVector<int32_t>({2});
  VELOX_ASSERT_THROW(
      evaluate(callExpr, makeRowVector({input, scaleCol})),
      "The second argument of decimal_ceil must be a constant expression.");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
