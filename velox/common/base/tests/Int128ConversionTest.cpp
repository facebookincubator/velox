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

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

namespace facebook::velox {
namespace {

using Code = folly::ConversionCode;

TEST(Int128ConversionTest, fullInputAndBoundaries) {
  const auto min = std::numeric_limits<int128_t>::min();
  const auto max = std::numeric_limits<int128_t>::max();
  const auto umax = std::numeric_limits<uint128_t>::max();
  EXPECT_EQ(
      tryToWithInt128<int128_t>("-170141183460469231731687303715884105728")
          .value(),
      min);
  EXPECT_EQ(
      tryToWithInt128<int128_t>("170141183460469231731687303715884105727")
          .value(),
      max);
  EXPECT_EQ(
      tryToWithInt128<uint128_t>("340282366920938463463374607431768211455")
          .value(),
      umax);
  EXPECT_EQ(
      tryToWithInt128<int128_t>("170141183460469231731687303715884105728")
          .error(),
      Code::POSITIVE_OVERFLOW);
  EXPECT_EQ(
      tryToWithInt128<int128_t>("-170141183460469231731687303715884105729")
          .error(),
      Code::NEGATIVE_OVERFLOW);
  EXPECT_EQ(
      tryToWithInt128<uint128_t>("340282366920938463463374607431768211456")
          .error(),
      Code::POSITIVE_OVERFLOW);
  EXPECT_EQ(tryToWithInt128<int128_t>(" \t+0042\r\n").value(), int128_t{42});
  EXPECT_EQ(tryToWithInt128<int128_t>("-0").value(), int128_t{0});
  EXPECT_EQ(
      tryToWithInt128<uint128_t>(std::string(100, '0') + "42").value(),
      uint128_t{42});
  EXPECT_EQ(
      tryToWithInt128<int128_t>(std::string_view("42\0", 3)).error(),
      Code::NON_WHITESPACE_AFTER_END);
  EXPECT_EQ(
      tryToWithInt128<int128_t>(std::string_view("42\xff", 3)).error(),
      Code::NON_WHITESPACE_AFTER_END);

  for (const auto* text :
       {"", " ", "+", "-", "--1", "+ 1", "x", "1x", "1 2", "0x10", "1.5"}) {
    EXPECT_EQ(
        tryToWithInt128<int128_t>(text).error(),
        folly::tryTo<int64_t>(text).error())
        << text;
  }
  for (const auto* text : {"", " ", "+1", "-1", "x", "1x", "1 2"}) {
    EXPECT_EQ(
        tryToWithInt128<uint128_t>(text).error(),
        folly::tryTo<uint64_t>(text).error())
        << text;
  }
}

TEST(Int128ConversionTest, integerRangeAndBoolean) {
  const auto min64 = std::numeric_limits<int64_t>::min();
  const auto max64 = std::numeric_limits<int64_t>::max();
  const auto umax64 = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(tryToWithInt128<int128_t>(min64).value(), int128_t{min64});
  EXPECT_EQ(tryToWithInt128<int128_t>(umax64).value(), int128_t{umax64});
  EXPECT_EQ(tryToWithInt128<int64_t>(int128_t{min64}).value(), min64);
  EXPECT_EQ(tryToWithInt128<int64_t>(int128_t{max64}).value(), max64);
  EXPECT_EQ(
      tryToWithInt128<int64_t>(int128_t{min64} - 1).error(),
      Code::ARITH_NEGATIVE_OVERFLOW);
  EXPECT_EQ(
      tryToWithInt128<int64_t>(int128_t{max64} + 1).error(),
      Code::ARITH_POSITIVE_OVERFLOW);
  EXPECT_EQ(
      tryToWithInt128<uint128_t>(int128_t{-1}).error(),
      Code::ARITH_NEGATIVE_OVERFLOW);
  EXPECT_EQ(
      tryToWithInt128<int128_t>(uint128_t{1} << 127).error(),
      Code::ARITH_POSITIVE_OVERFLOW);
  EXPECT_EQ(tryToWithInt128<uint64_t>(uint128_t{umax64}).value(), umax64);
  EXPECT_EQ(
      tryToWithInt128<uint64_t>(uint128_t{umax64} + 1).error(),
      Code::ARITH_POSITIVE_OVERFLOW);
  EXPECT_EQ(tryToWithInt128<int128_t>(true).value(), int128_t{1});
  EXPECT_EQ(tryToWithInt128<uint128_t>(false).value(), uint128_t{0});
  EXPECT_TRUE(tryToWithInt128<bool>(int128_t{-1}).value());
  EXPECT_FALSE(tryToWithInt128<bool>(uint128_t{0}).value());
}

template <typename F, typename T>
void testFloating() {
  const F upper = std::ldexp(F{1}, std::numeric_limits<T>::digits);
  const F belowUpper = std::nextafter(upper, F{0});
  EXPECT_TRUE(tryToWithInt128<T>(belowUpper).hasValue());
  EXPECT_EQ(tryToWithInt128<T>(upper).error(), Code::ARITH_LOSS_OF_PRECISION);
  EXPECT_EQ(
      tryToWithInt128<T>(std::numeric_limits<F>::infinity()).error(),
      Code::ARITH_LOSS_OF_PRECISION);
  EXPECT_EQ(
      tryToWithInt128<T>(std::numeric_limits<F>::quiet_NaN()).error(),
      Code::ARITH_LOSS_OF_PRECISION);
  EXPECT_EQ(tryToWithInt128<T>(F{1.5}).error(), Code::ARITH_LOSS_OF_PRECISION);
  EXPECT_EQ(tryToWithInt128<T>(F{-0.0}).value(), T{0});
  EXPECT_EQ(tryToWithInt128<T>(F{42}).value(), T{42});
  if constexpr (std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(
        tryToWithInt128<T>(-upper).value(), std::numeric_limits<T>::min());
    EXPECT_EQ(
        tryToWithInt128<T>(std::nextafter(-upper, -INFINITY)).error(),
        Code::ARITH_LOSS_OF_PRECISION);
    EXPECT_EQ(
        tryToWithInt128<F>(std::numeric_limits<T>::min()).value(), -upper);
  } else {
    EXPECT_EQ(tryToWithInt128<T>(F{-1}).error(), Code::ARITH_LOSS_OF_PRECISION);
  }
  EXPECT_EQ(
      tryToWithInt128<F>(std::numeric_limits<T>::max()).error(),
      Code::ARITH_LOSS_OF_PRECISION);
  EXPECT_EQ(tryToWithInt128<F>(T{42}).value(), F{42});
}

TEST(Int128ConversionTest, checkedFloatingConversions) {
  testFloating<float, int128_t>();
  testFloating<double, int128_t>();
  testFloating<long double, int128_t>();
  testFloating<double, uint128_t>();
  testFloating<long double, uint128_t>();
}

} // namespace
} // namespace facebook::velox
