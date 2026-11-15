/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "velox/dwio/nimble/common/OrderedInteger.h"

using namespace ::facebook::nimble;

namespace {

template <typename T>
class OrderedIntegerTest : public ::testing::Test {
 protected:
  using PhysicalType = typename TypeTraits<T>::physicalType;

  // Returns the extremes and their neighbors in ascending logical order.
  static std::vector<T> ascendingValues() {
    constexpr auto kMin = std::numeric_limits<T>::min();
    constexpr auto kMax = std::numeric_limits<T>::max();
    if constexpr (std::is_signed_v<T>) {
      return {
          kMin,
          static_cast<T>(kMin + 1),
          -1,
          0,
          1,
          static_cast<T>(kMax - 1),
          kMax};
    } else {
      return {kMin, static_cast<T>(kMin + 1), static_cast<T>(kMax - 1), kMax};
    }
  }

  static uint64_t ordered(T value) {
    return toOrdered<T>(static_cast<PhysicalType>(value));
  }
};

using IntegerTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t>;
TYPED_TEST_SUITE(OrderedIntegerTest, IntegerTypes);

TYPED_TEST(OrderedIntegerTest, preservesOrder) {
  const auto values = TestFixture::ascendingValues();
  for (size_t i = 1; i < values.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_LT(
        TestFixture::ordered(values[i - 1]), TestFixture::ordered(values[i]));
  }
}

TYPED_TEST(OrderedIntegerTest, roundTrips) {
  using T = TypeParam;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (const auto value : TestFixture::ascendingValues()) {
    SCOPED_TRACE(static_cast<int64_t>(value));
    const auto physical = static_cast<PhysicalType>(value);
    EXPECT_EQ(fromOrdered<T>(toOrdered<T>(physical)), physical);
  }
}

TYPED_TEST(OrderedIntegerTest, mapsMinimumToZero) {
  using T = TypeParam;
  EXPECT_EQ(TestFixture::ordered(std::numeric_limits<T>::min()), 0);
  // Unsigned values are only widened.
  if constexpr (std::is_unsigned_v<T>) {
    EXPECT_EQ(
        TestFixture::ordered(std::numeric_limits<T>::max()),
        uint64_t{std::numeric_limits<T>::max()});
  }
}

} // namespace

TEST(OrderedIntegerTest, documentedExamples) {
  EXPECT_EQ(toOrdered<int8_t>(static_cast<uint8_t>(int8_t{-128})), 0x00);
  EXPECT_EQ(toOrdered<int8_t>(static_cast<uint8_t>(int8_t{-1})), 0x7F);
  EXPECT_EQ(toOrdered<int8_t>(static_cast<uint8_t>(int8_t{0})), 0x80);
  EXPECT_EQ(toOrdered<int8_t>(static_cast<uint8_t>(int8_t{127})), 0xFF);
  EXPECT_EQ(fromOrdered<int8_t>(0x7F), static_cast<uint8_t>(int8_t{-1}));
}
