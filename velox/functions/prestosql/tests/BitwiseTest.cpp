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

static constexpr auto kMin16 = std::numeric_limits<int16_t>::min();
static constexpr auto kMax16 = std::numeric_limits<int16_t>::max();
static constexpr auto kMin32 = std::numeric_limits<int32_t>::min();
static constexpr auto kMax32 = std::numeric_limits<int32_t>::max();
static constexpr auto kMin64 = std::numeric_limits<int64_t>::min();
static constexpr auto kMax64 = std::numeric_limits<int64_t>::max();

class BitwiseTest : public functions::test::FunctionBaseTest {
 protected:
  template <typename T>
  std::optional<int64_t> bitwiseFunction(
      const std::string& fn,
      std::optional<T> a,
      std::optional<T> b) {
    return evaluateOnce<int64_t>(fmt::format("{}(c0, c1)", fn), a, b);
  }
};

TEST_F(BitwiseTest, bitwiseAnd) {
  const auto fn = "bitwise_and";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), 12);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 20);

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), kMax16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMin16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMin16), kMin16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, -1), kMax16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, -1), kMin16);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), kMax32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMin32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMin32), kMin32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, -1), kMax32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, -1), kMin32);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), kMax64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMin64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMin64), kMin64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, -1), kMax64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, -1), kMin64);
}

TEST_F(BitwiseTest, bitwiseNot) {
  const auto bitwiseNot = [&](std::optional<int32_t> a) {
    return evaluateOnce<int64_t>("bitwise_not(c0)", a);
  };

  EXPECT_EQ(bitwiseNot(-1), 0);
  EXPECT_EQ(bitwiseNot(0), -1);
  EXPECT_EQ(bitwiseNot(2), -3);
  EXPECT_EQ(bitwiseNot(kMax32), kMin32);
  EXPECT_EQ(bitwiseNot(kMin32), kMax32);
  EXPECT_EQ(bitwiseNot(kMax16), kMin16);
  EXPECT_EQ(bitwiseNot(kMin16), kMax16);
}

TEST_F(BitwiseTest, bitwiseOr) {
  const auto fn = "bitwise_or";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 11);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), -4);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 61);

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), kMax16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMin16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMin16), kMin16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, -1), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), kMin16 + 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, -1), -1);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), kMax32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMin32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMin32), kMin32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, -1), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), kMin32 + 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, -1), -1);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), kMax64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMin64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMin64), kMin64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, -1), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), kMin64 + 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, -1), -1);
}

TEST_F(BitwiseTest, bitwiseXor) {
  const auto fn = "bitwise_xor";

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 0, -1), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 3, 8), 11);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -4, 12), -16);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, 60, 21), 41);

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMin16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMin16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16 - 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, -1), kMin16);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), kMin16 + 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, -1), kMax16);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMin32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMin32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32 - 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, -1), kMin32);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), kMin32 + 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, -1), kMax32);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMin64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMin64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64 - 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, -1), kMin64);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), kMin64 + 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, -1), kMax64);
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

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16 >> 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), kMin16 >> 1);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32 >> 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), kMin32 >> 1);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64 >> 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), kMin64 >> 1);
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

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), -1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16 >> 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), kMin16 >> 1);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), -1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32 >> 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), kMin32 >> 1);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), -1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64 >> 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), kMin64 >> 1);
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

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16 << 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), kMin16 << 1);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32 << 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1), kMin32 << 1);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64 << 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1), kMin64 << 1);
}

TEST_F(BitwiseTest, rightShift) {
  const auto fn = "bitwise_right_shift";

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, 1, 1), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -3, 1), 2147483646);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, 32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, -1, -1), 0);

  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, kMax16), 0);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMax16, 1), kMax16 >> 1);
  EXPECT_EQ(bitwiseFunction<int16_t>(fn, kMin16, 1), (kMax16 >> 1) + 1);

  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, kMax32), 0);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMax32, 1), kMax32 >> 1);
  EXPECT_EQ(bitwiseFunction<int32_t>(fn, kMin32, 1).value(), (kMax32 >> 1) + 1);

  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, kMax64), 0);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMax64, 1), kMax64 >> 1);
  EXPECT_EQ(bitwiseFunction<int64_t>(fn, kMin64, 1).value(), (kMax64 >> 1) + 1);
}

TEST_F(BitwiseTest, logicalShiftRight) {
  const auto bitwiseLogicalShiftRight = [&](std::optional<int64_t> number,
                                            std::optional<int64_t> shift,
                                            std::optional<int64_t> bits) {
    return evaluateOnce<int64_t>(
        "bitwise_logical_shift_right(c0, c1, c2)", number, shift, bits);
  };

  EXPECT_EQ(bitwiseLogicalShiftRight(1, 1, 64), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 1, 2), 1);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 32, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 30, 32), 3);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMin64, 10, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMin64, kMax64, 64), -1);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMax64, kMin64, 64), kMax64);

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

  EXPECT_EQ(bitwiseLogicalShiftRight(1, 1, 64), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 1, 2), 2);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 32, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(-1, 31, 32), 2147483648);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMin64, 10, 32), 0);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMin64, kMax64, 64), -1);
  EXPECT_EQ(bitwiseLogicalShiftRight(kMax64, kMin64, 64), kMax64);

  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, -1, 3); }, "Shift must be positive");
  assertUserError(
      [&]() { bitwiseLogicalShiftRight(3, 1, 1); },
      "Bits must be between 2 and 64");
}

} // namespace
} // namespace facebook::velox
