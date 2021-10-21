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
#include <optional>

#include <gmock/gmock.h>

#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace facebook::velox {

namespace {
using namespace std;

class BitwiseTest : public functions::test::FunctionBaseTest {
 protected:
  template <typename T>
  std::optional<int64_t> bitwiseFunction(
      const std::string& fn,
      std::optional<T> a,
      std::optional<T> b) {
    return evaluateOnce<int64_t>(fmt::format("{}(c0, c1)", fn), a, b);
  }

  /// This struct stores results of bitwise operation on min , max, 1, -1 values
  /// of type param T.
  /// \tparam T
  template <typename T>
  struct NumericLimitValues {
    T minMax;
    T maxMax;
    T maxMin;
    T minMin;
    T maxPositiveOne;
    T maxMinusOne;
    T minPositiveOne;
    T minMinusOne;
  };

  /// Tests a binary bitwise function at the numeric extremities of type T.
  /// \tparam T - Type parameter.
  /// \param fn - Bitwise binary function.
  /// \param expectedResults - A filled out struct of type NumericLimitValues.
  template <typename T>
  void basicNumericLimitTesting(
      const std::string& fn,
      const NumericLimitValues<T>& expectedResults) {
    auto min = numeric_limits<T>::min();
    auto max = numeric_limits<T>::min();

    auto fnResults = vector<pair<optional<T>, T>>{
        {bitwiseFunction<T>(fn, min, max), expectedResults.minMax},
        {bitwiseFunction<T>(fn, max, max), expectedResults.maxMax},
        {bitwiseFunction<T>(fn, max, min), expectedResults.maxMin},
        {bitwiseFunction<T>(fn, min, min), expectedResults.minMin},
        {bitwiseFunction<T>(fn, max, 1), expectedResults.maxPositiveOne},
        {bitwiseFunction<T>(fn, max, -1), expectedResults.maxMinusOne},
        {bitwiseFunction<T>(fn, min, 1), expectedResults.minPositiveOne},
        {bitwiseFunction<T>(fn, min, -1), expectedResults.minMinusOne}};

    for (auto& it : fnResults) {
      EXPECT_EQ(it.first.value(), it.second);
    }
  }

  /// Tests a binary shifting function (right , left, logical, arithmetic) at
  /// the numeric extremities of type T.
  /// \tparam T - Type parameter.
  /// \param fn - Bitwise binary function.
  /// \param expectedResults - A filled out struct of type NumericLimitValues.
  template <typename T>
  void basicShiftTesting(
      const std::string& fn,
      const NumericLimitValues<T>& expectedResults) {
    auto min = numeric_limits<T>::min();
    auto max = numeric_limits<T>::max();

    auto fnResults = vector<pair<optional<T>, T>>{
        {bitwiseFunction<T>(fn, min, max), expectedResults.minMax},
        {bitwiseFunction<T>(fn, max, max), expectedResults.maxMax},
        {bitwiseFunction<T>(fn, max, 1), expectedResults.maxPositiveOne},
        {bitwiseFunction<T>(fn, min, 1), expectedResults.minPositiveOne},
    };

    for (auto& it : fnResults) {
      EXPECT_EQ(it.first.value(), it.second);
    }
  }

  void basicNumericTesting(
      const std::string& fn,
      const NumericLimitValues<int16_t>& tinyint,
      const NumericLimitValues<int32_t>& normalint,
      const NumericLimitValues<int64_t>& bigint) {
    basicNumericLimitTesting<int16_t>(fn, tinyint);
    basicNumericLimitTesting<int32_t>(fn, normalint);
    basicNumericLimitTesting<int64_t>(fn, bigint);
  }

  void basicShiftTesting(
      const std::string& fn,
      const NumericLimitValues<int16_t>& tinyint,
      const NumericLimitValues<int32_t>& normalint,
      const NumericLimitValues<int64_t>& bigint) {
    basicShiftTesting<int16_t>(fn, tinyint);
    basicShiftTesting<int32_t>(fn, normalint);
    basicShiftTesting<int64_t>(fn, bigint);
  }
};

TEST_F(BitwiseTest, bitwiseAnd) {
  const auto fn = "bitwise_and";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), 12);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 20);

  NumericLimitValues<int16_t> and16 = {
      .minMax = -32768,
      .maxMax = -32768,
      .maxMin = -32768,
      .minMin = -32768,
      .maxPositiveOne = 0,
      .maxMinusOne = -32768,
      .minPositiveOne = 0,
      .minMinusOne = -32768,
  };

  NumericLimitValues<int32_t> and32 = {
      .minMax = -2147483648,
      .maxMax = -2147483648,
      .maxMin = -2147483648,
      .minMin = -2147483648,
      .maxPositiveOne = 0,
      .maxMinusOne = -2147483648,
      .minPositiveOne = 0,
      .minMinusOne = -2147483648};

  NumericLimitValues<int64_t> and64 = {
      .minMax = (int64_t)-9223372036854775808,
      .maxMax = (int64_t)-9223372036854775808,
      .maxMin = (int64_t)-9223372036854775808,
      .minMin = (int64_t)-9223372036854775808,
      .maxPositiveOne = 0,
      .maxMinusOne = (int64_t)-9223372036854775808,
      .minPositiveOne = 0,
      .minMinusOne = (int64_t)-9223372036854775808};

  basicNumericTesting(fn, and16, and32, and64);
}

TEST_F(BitwiseTest, bitwiseNot) {
  const auto bitwiseNot = [&](std::optional<int32_t> a) {
    return evaluateOnce<int64_t>("bitwise_not(c0)", a);
  };

  using limits = numeric_limits<int32_t>;

  EXPECT_EQ(bitwiseNot(-1), 0);
  EXPECT_EQ(bitwiseNot(0), -1);
  EXPECT_EQ(bitwiseNot(2), -3);
  EXPECT_EQ(bitwiseNot(limits::max()), limits::min());
  EXPECT_EQ(bitwiseNot(limits::min()), limits::max());
}

TEST_F(BitwiseTest, bitwiseOr) {
  const auto fn = "bitwise_or";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 11);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), -4);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 61);

  NumericLimitValues<int16_t> or16 = {
      .minMax = -32768,
      .maxMax = -32768,
      .maxMin = -32768,
      .minMin = -32768,
      .maxPositiveOne = -32767,
      .maxMinusOne = -1,
      .minPositiveOne = -32767,
      .minMinusOne = -1};

  NumericLimitValues<int32_t> or32 = {
      .minMax = -2147483648,
      .maxMax = -2147483648,
      .maxMin = -2147483648,
      .minMin = -2147483648,
      .maxPositiveOne = -2147483647,
      .maxMinusOne = -1,
      .minPositiveOne = -2147483647,
      .minMinusOne = -1};

  NumericLimitValues<int64_t> or64 = {
      .minMax = (int64_t)-9223372036854775808,
      .maxMax = (int64_t)-9223372036854775808,
      .maxMin = (int64_t)-9223372036854775808,
      .minMin = (int64_t)-9223372036854775808,
      .maxPositiveOne = (int64_t)-9223372036854775807,
      .maxMinusOne = -1,
      .minPositiveOne = (int64_t)-9223372036854775807,
      .minMinusOne = -1};

  basicNumericTesting(fn, or16, or32, or64);
}

TEST_F(BitwiseTest, bitwiseXor) {
  const auto fn = "bitwise_xor";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 11);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), -16);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 41);

  NumericLimitValues<int16_t> xor16 = {
      .minMax = 0,
      .maxMax = 0,
      .maxMin = 0,
      .minMin = 0,
      .maxPositiveOne = -32767,
      .maxMinusOne = 32767,
      .minPositiveOne = -32767,
      .minMinusOne = 32767};

  NumericLimitValues<int32_t> xor32 = {
      .minMax = 0,
      .maxMax = 0,
      .maxMin = 0,
      .minMin = 0,
      .maxPositiveOne = -2147483647,
      .maxMinusOne = 2147483647,
      .minPositiveOne = -2147483647,
      .minMinusOne = 2147483647};

  NumericLimitValues<int64_t> xor64 = {
      .minMax = 0,
      .maxMax = 0,
      .maxMin = 0,
      .minMin = 0,
      .maxPositiveOne = (int64_t)-9223372036854775807,
      .maxMinusOne = 9223372036854775807,
      .minPositiveOne = (int64_t)-9223372036854775807,
      .minMinusOne = 9223372036854775807};

  basicNumericTesting(fn, xor16, xor32, xor64);
}

TEST_F(BitwiseTest, arithmeticShiftRight) {
  const auto fn = "bitwise_arithmetic_shift_right";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 1, 1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 1), 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -3, 1), -2);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 0), 3);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 3), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 2), -1);
  assertUserError(
      [&]() { bitwiseFunction<int32_t>(fn, 3, -1); }, "Shift must be positive");

  NumericLimitValues<int16_t> limit16 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = 16383,
      .minPositiveOne = -16384,
  };

  NumericLimitValues<int32_t> limit32 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = 1073741823,
      .minPositiveOne = -1073741824,
  };

  NumericLimitValues<int64_t> limit64 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = (int64_t)4611686018427387903,
      .minPositiveOne = (int64_t)-4611686018427387904,
  };

  basicShiftTesting(fn, limit16, limit32, limit64);
}

TEST_F(BitwiseTest, rightShiftArithmetic) {
  const auto fn = "bitwise_right_shift_arithmetic";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 1, 1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 1), 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -3, 1), -2);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 0), 3);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 3), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 2), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 65), -1);

  NumericLimitValues<int16_t> limit16 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = 16383,
      .minPositiveOne = -16384,
  };

  NumericLimitValues<int32_t> limit32 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = 1073741823,
      .minPositiveOne = -1073741824,
  };

  NumericLimitValues<int64_t> limit64 = {
      .minMax = -1,
      .maxMax = 0,
      .maxPositiveOne = (int64_t)4611686018427387903,
      .minPositiveOne = (int64_t)-4611686018427387904,
  };

  basicShiftTesting(fn, limit16, limit32, limit64);
}

TEST_F(BitwiseTest, leftShift) {
  const auto fn = "bitwise_left_shift";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 1, 1), 2);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 1), -2);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 3), -8);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 33), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, -1, 33), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, 1, 33), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, -1, 33), -8589934592);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 524287, 3), 4194296);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, -1, -1), 0);

  NumericLimitValues<int16_t> limit16 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = -2,
      .minPositiveOne = 0,
  };

  NumericLimitValues<int32_t> limit32 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = -2,
      .minPositiveOne = 0,
  };

  NumericLimitValues<int64_t> limit64 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = -2,
      .minPositiveOne = 0,
  };

  basicShiftTesting(fn, limit16, limit32, limit64);
}

TEST_F(BitwiseTest, rightShift) {
  const auto fn = "bitwise_right_shift";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 1, 1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -3, 1), 2147483646);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, -1), 0);

  NumericLimitValues<int16_t> limit16 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = 16383,
      .minPositiveOne = 16384,
  };

  NumericLimitValues<int32_t> limit32 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = 1073741823,
      .minPositiveOne = 1073741824,
  };

  NumericLimitValues<int64_t> limit64 = {
      .minMax = 0,
      .maxMax = 0,
      .maxPositiveOne = 4611686018427387903,
      .minPositiveOne = 4611686018427387904,
  };

  basicShiftTesting(fn, limit16, limit32, limit64);
}

TEST_F(BitwiseTest, logicalShiftRight) {
  const auto bitwiseLogicalShiftRight = [&](std::optional<int64_t> number,
                                            std::optional<int64_t> shift,
                                            std::optional<int64_t> bits) {
    return evaluateOnce<int64_t>(
        "bitwise_logical_shift_right(c0, c1, c2)", number, shift, bits);
  };

  using limits = numeric_limits<int64_t>;

  EXPECT_EQ(bitwiseLogicalShiftRight(1, 1, 64), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 1, 2), 1);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 32, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 30, 32), 3);
  EXPECT_EQ(bitwiseLogicalShiftRight(limits::min(), 10, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(limits::min(), limits::max(), 64), -1);
  EXPECT_EQ(
      bitwiseLogicalShiftRight(limits::max(), limits::min(), 64),
      limits::max());

  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, -1, 3); }, "Shift must be positive");
  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, 1, 1); },
      "Bits must be between 2 and 64");
}

TEST_F(BitwiseTest, shiftLeft) {
  const auto bitwiseLogicalShiftRight = [&](std::optional<int64_t> number,
                                            std::optional<int64_t> shift,
                                            std::optional<int64_t> bits) {
    return evaluateOnce<int64_t>(
        "bitwise_shift_left(c0, c1, c2)", number, shift, bits);
  };

  using limits = numeric_limits<int64_t>;
  EXPECT_EQ(bitwiseLogicalShiftRight(1, 1, 64), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 1, 2), 2);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 32, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 31, 32), 2147483648);
  EXPECT_EQ(bitwiseLogicalShiftRight(limits::min(), 10, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(limits::min(), limits::max(), 64), -1);
  EXPECT_EQ(
      bitwiseLogicalShiftRight(limits::max(), limits::min(), 64),
      limits::max());

  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, -1, 3); }, "Shift must be positive");
  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, 1, 1); },
      "Bits must be between 2 and 64");
}

} // namespace
} // namespace facebook::velox
