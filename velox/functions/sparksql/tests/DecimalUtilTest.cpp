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

#include "velox/functions/sparksql/DecimalUtil.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalUtilTest : public testing::Test {
 protected:
  template <typename R, typename A, typename B>
  void testDivideWithRoundUp(
      A a,
      B b,
      int32_t aRescale,
      R expectedResult,
      bool expectedOverflow) {
    R r;
    bool overflow = false;
    DecimalUtil::divideWithRoundUp<R, A, B>(r, a, b, aRescale, overflow);
    ASSERT_EQ(overflow, expectedOverflow);
    ASSERT_EQ(r, expectedResult);
  }

  template <bool allowPrecisionLoss>
  void testComputeDivideResultPrecisionScale(
      const uint8_t aPrecision,
      const uint8_t aScale,
      const uint8_t bPrecision,
      const uint8_t bScale,
      std::pair<uint8_t, uint8_t> expected) {
    ASSERT_EQ(
        DecimalUtil::computeDivideResultPrecisionScale<allowPrecisionLoss>(
            aPrecision, aScale, bPrecision, bScale),
        expected);
  }
};
} // namespace

TEST_F(DecimalUtilTest, divideWithRoundUp) {
  testDivideWithRoundUp<int64_t, int64_t, int64_t>(60, 30, 3, 2000, false);
  testDivideWithRoundUp<int64_t, int64_t, int64_t>(
      6, velox::DecimalUtil::kPowersOfTen[17], 20, 6000, false);
}

TEST_F(DecimalUtilTest, minLeadingZeros) {
  auto result =
      DecimalUtil::minLeadingZeros<int64_t, int64_t>(10000, 6000000, 10, 12);
  ASSERT_EQ(result, 1);

  result = DecimalUtil::minLeadingZeros<int64_t, int128_t>(
      10000, 6'000'000'000'000'000'000, 10, 12);
  ASSERT_EQ(result, 16);

  result = DecimalUtil::minLeadingZeros<int128_t, int128_t>(
      velox::DecimalUtil::kLongDecimalMax,
      velox::DecimalUtil::kLongDecimalMin,
      10,
      12);
  ASSERT_EQ(result, 0);
}

TEST_F(DecimalUtilTest, bounded) {
  auto testBounded = [](uint8_t rPrecision,
                        uint8_t rScale,
                        std::pair<uint8_t, uint8_t> expected) {
    ASSERT_EQ(DecimalUtil::bounded(rPrecision, rScale), expected);
  };

  testBounded(10, 3, {10, 3});
  testBounded(40, 3, {38, 3});
  testBounded(44, 42, {38, 38});
}

TEST_F(DecimalUtilTest, computeDivideResultPrecisionScale) {
  // Test with allowPrecisionLoss = true.
  testComputeDivideResultPrecisionScale<true>(10, 2, 5, 1, {17, 8});
  testComputeDivideResultPrecisionScale<true>(38, 10, 10, 5, {38, 6});
  testComputeDivideResultPrecisionScale<true>(1, 0, 1, 0, {7, 6});
  testComputeDivideResultPrecisionScale<true>(20, 2, 20, 2, {38, 18});

  // Test with allowPrecisionLoss = false.
  testComputeDivideResultPrecisionScale<false>(10, 2, 5, 1, {17, 8});
  testComputeDivideResultPrecisionScale<false>(38, 10, 5, 3, {38, 11});
  testComputeDivideResultPrecisionScale<false>(1, 0, 1, 0, {7, 6});
  testComputeDivideResultPrecisionScale<false>(30, 5, 10, 5, {38, 11});
}

} // namespace facebook::velox::functions::sparksql::test
