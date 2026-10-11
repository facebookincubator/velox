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

#include "velox/common/base/CheckedArithmetic.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

namespace facebook::velox {
namespace {

using boost::multiprecision::cpp_int;

static_assert(noexcept(
    addOverflow(std::declval<int128_t>(), 1, static_cast<int128_t*>(nullptr))));
static_assert(noexcept(subtractOverflow(
    std::declval<int64_t>(),
    1,
    static_cast<int64_t*>(nullptr))));
static_assert(noexcept(multiplyOverflow(
    std::declval<uint128_t>(),
    1,
    static_cast<uint128_t*>(nullptr))));

template <typename T>
cpp_int asBigInteger(T value) {
  const auto bits = static_cast<uint128_t>(value);
  cpp_int result = static_cast<uint64_t>(bits >> 64);
  result <<= 64;
  result += static_cast<uint64_t>(bits);
  if (value < 0) {
    result -= cpp_int{1} << 128;
  }
  return result;
}

template <typename T>
std::vector<T> boundaries() {
  const T min = std::numeric_limits<T>::min();
  const T max = std::numeric_limits<T>::max();
  std::vector<T> values = {T{0}, T{1}, T{2}, max, T(max - 1), T(max / 2)};
  if constexpr (std::numeric_limits<T>::is_signed) {
    values.insert(values.end(), {T{-1}, T{-2}, min, T(min + 1), T(min / 2)});
  }
  return values;
}

template <typename R>
void checkResult(const cpp_int& expected, bool overflow, R result) {
  const auto min = asBigInteger(std::numeric_limits<R>::min());
  const auto max = asBigInteger(std::numeric_limits<R>::max());
  ASSERT_EQ(overflow, expected < min || expected > max);
  const cpp_int modulus = cpp_int{1} << (sizeof(R) * 8);
  cpp_int wrapped = expected % modulus;
  if (wrapped < 0) {
    wrapped += modulus;
  }
  if (std::numeric_limits<R>::is_signed && wrapped > max) {
    wrapped -= modulus;
  }
  ASSERT_EQ(asBigInteger(result), wrapped);
}

template <typename A, typename B, typename R>
void testBoundaries() {
  for (const auto a : boundaries<A>()) {
    for (const auto b : boundaries<B>()) {
      const auto left = asBigInteger(a);
      const auto right = asBigInteger(b);
      R result;
      bool overflow = addOverflow(a, b, &result);
      checkResult(left + right, overflow, result);
      overflow = subtractOverflow(a, b, &result);
      checkResult(left - right, overflow, result);
      overflow = multiplyOverflow(a, b, &result);
      checkResult(left * right, overflow, result);
    }
  }
}

template <typename T>
void testAliasing() {
  for (const auto a : boundaries<T>()) {
    for (const auto b : boundaries<T>()) {
      T result = a;
      auto overflow = addOverflow(result, b, &result);
      checkResult(asBigInteger(a) + asBigInteger(b), overflow, result);
      result = b;
      overflow = subtractOverflow(a, result, &result);
      checkResult(asBigInteger(a) - asBigInteger(b), overflow, result);
      result = a;
      overflow = multiplyOverflow(result, b, &result);
      checkResult(asBigInteger(a) * asBigInteger(b), overflow, result);
    }
  }
}

TEST(OverflowTest, signedBoundaries) {
  testBoundaries<int8_t, int8_t, int8_t>();
  testBoundaries<int16_t, int16_t, int16_t>();
  testBoundaries<int32_t, int32_t, int32_t>();
  testBoundaries<int64_t, int64_t, int64_t>();
  testBoundaries<int128_t, int128_t, int128_t>();
}

TEST(OverflowTest, unsignedBoundaries) {
  testBoundaries<uint8_t, uint8_t, uint8_t>();
  testBoundaries<uint16_t, uint16_t, uint16_t>();
  testBoundaries<uint32_t, uint32_t, uint32_t>();
  testBoundaries<uint64_t, uint64_t, uint64_t>();
  testBoundaries<uint128_t, uint128_t, uint128_t>();
}

TEST(OverflowTest, mixedWidthsAndSigns) {
  testBoundaries<int64_t, int128_t, int64_t>();
  testBoundaries<int128_t, int, int128_t>();
  testBoundaries<int128_t, uint128_t, int128_t>();
  testBoundaries<uint128_t, uint128_t, int64_t>();
  testBoundaries<int128_t, int128_t, uint128_t>();
  testBoundaries<int64_t, uint64_t, int128_t>();
  testBoundaries<int128_t, uint128_t, int8_t>();
  testBoundaries<int64_t, uint64_t, uint8_t>();
}

TEST(OverflowTest, aliasesEitherOperand) {
  testAliasing<int64_t>();
  testAliasing<uint64_t>();
  testAliasing<int128_t>();
  testAliasing<uint128_t>();
}

TEST(OverflowTest, booleanOperands) {
  int128_t result;
  EXPECT_FALSE(addOverflow(true, int128_t{1}, &result));
  EXPECT_EQ(result, int128_t{2});
  EXPECT_FALSE(
      multiplyOverflow(std::numeric_limits<uint128_t>::max(), false, &result));
  EXPECT_EQ(result, int128_t{0});
}

} // namespace
} // namespace facebook::velox
