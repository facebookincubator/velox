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
#include <limits>
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox {
namespace {

template <typename T>
void testSigned() {
  const T min = std::numeric_limits<T>::min();
  const T max = std::numeric_limits<T>::max();
  EXPECT_EQ(checkedPlus<T>(min, max), T{-1});
  EXPECT_EQ(checkedMinus<T>(min, -1), min + 1);
  EXPECT_EQ(checkedMultiply<T>(min, 1), min);
  EXPECT_EQ(checkedMultiply<T>(min, 0), T{0});
  EXPECT_EQ(checkedDivide<T>(min, 1), min);
  EXPECT_EQ(checkedDivide<T>(max, -1), -max);
  VELOX_ASSERT_USER_THROW(checkedPlus<T>(max, 1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedMinus<T>(min, 1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedMultiply<T>(min, -1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedDivide<T>(min, -1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedDivide<T>(1, 0), "division by zero");
}

template <typename T>
void testUnsigned() {
  const T max = std::numeric_limits<T>::max();
  EXPECT_EQ(checkedPlus<T>(max, 0), max);
  EXPECT_EQ(checkedMultiply<T>(max, 0), T{0});
  EXPECT_EQ(checkedDivide<T>(0, max), T{0});
  EXPECT_EQ(checkedDivide<T>(max, max), T{1});
  VELOX_ASSERT_USER_THROW(checkedPlus<T>(max, 1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedMinus<T>(0, 1), "overflow");
  VELOX_ASSERT_USER_THROW(checkedMultiply<T>(max, 2), "overflow");
  VELOX_ASSERT_USER_THROW(checkedDivide<T>(1, 0), "division by zero");
}

TEST(CheckedArithmeticTest, signed64) {
  testSigned<int64_t>();
}

TEST(CheckedArithmeticTest, signed128) {
  testSigned<int128_t>();
}

TEST(CheckedArithmeticTest, unsigned64) {
  testUnsigned<uint64_t>();
}

TEST(CheckedArithmeticTest, unsigned128) {
  testUnsigned<uint128_t>();
}

} // namespace
} // namespace facebook::velox
