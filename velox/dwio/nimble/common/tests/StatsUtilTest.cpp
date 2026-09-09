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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/StatsUtil.h"

using namespace facebook::nimble;

namespace {

template <typename T>
void expectIntegralMinMax(T expectedMin, T expectedMax, std::vector<T> values) {
  const auto minMax = findMinMax(std::span<const T>{values});
  EXPECT_EQ(minMax.min, expectedMin);
  EXPECT_EQ(minMax.max, expectedMax);
}

template <typename T>
void expectFloatingPointMinMax(
    T expectedMin,
    T expectedMax,
    std::vector<T> values) {
  const auto minMax = findMinMax(std::span<const T>{values});
  ASSERT_TRUE(minMax.has_value());
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_FLOAT_EQ(minMax->min, expectedMin);
    EXPECT_FLOAT_EQ(minMax->max, expectedMax);
  } else {
    EXPECT_DOUBLE_EQ(minMax->min, expectedMin);
    EXPECT_DOUBLE_EQ(minMax->max, expectedMax);
  }
}

} // namespace

TEST(StatsUtilTest, findMinMaxHandlesIntegralTypes) {
  std::vector<int32_t> signedValues(257);
  for (size_t i = 0; i < signedValues.size(); ++i) {
    signedValues[i] = static_cast<int32_t>(i) - 128;
  }
  signedValues[13] = std::numeric_limits<int32_t>::min();
  signedValues[199] = std::numeric_limits<int32_t>::max();
  expectIntegralMinMax<int32_t>(
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max(),
      signedValues);

  std::vector<uint64_t> unsignedValues(257);
  for (size_t i = 0; i < unsignedValues.size(); ++i) {
    unsignedValues[i] = 1000 + i;
  }
  unsignedValues[17] = 0;
  unsignedValues[211] = std::numeric_limits<uint64_t>::max();
  expectIntegralMinMax<uint64_t>(
      0, std::numeric_limits<uint64_t>::max(), unsignedValues);
}

TEST(StatsUtilTest, findMinMaxHandlesStringViews) {
  using namespace std::string_view_literals;

  // Bounds trail the first value, so seeding from values.front() and skipping
  // the max check when a value lowers the min must still find both.
  std::vector<std::string_view> values = {"mmm"sv, "aaa"sv, "zzz"sv, "qqq"sv};
  auto minMax = findMinMax(std::span<std::string_view>(values));
  EXPECT_EQ(minMax.min, "aaa"sv);
  EXPECT_EQ(minMax.max, "zzz"sv);

  // Ordering is by bytes and length, not NUL termination.
  std::vector<std::string_view> withNuls = {"a\0a"sv, "a\0"sv, "a\0z"sv};
  minMax = findMinMax(std::span<std::string_view>(withNuls));
  EXPECT_EQ(minMax.min, "a\0"sv);
  EXPECT_EQ(minMax.max, "a\0z"sv);

  std::vector<std::string_view> single = {"only"sv};
  minMax = findMinMax(std::span<std::string_view>(single));
  EXPECT_EQ(minMax.min, "only"sv);
  EXPECT_EQ(minMax.max, "only"sv);

  // The returned bounds are views into the input, not copies.
  std::vector<std::string_view> aliased = {"beta"sv, "alpha"sv};
  minMax = findMinMax(std::span<std::string_view>(aliased));
  EXPECT_EQ(minMax.min.data(), aliased[1].data());
  EXPECT_EQ(minMax.max.data(), aliased[0].data());
}

TEST(StatsUtilTest, findMinMaxHandlesFloatingPointTypes) {
  std::vector<float> floatValues(257, 1.25F);
  floatValues[11] = -123.5F;
  floatValues[211] = 456.75F;
  expectFloatingPointMinMax<float>(-123.5F, 456.75F, floatValues);

  std::vector<double> doubleValues(257, 2.5);
  doubleValues[31] = -9876.5;
  doubleValues[199] = 54321.25;
  expectFloatingPointMinMax<double>(-9876.5, 54321.25, doubleValues);
}

TEST(StatsUtilTest, findMinMaxReturnsEmptyForFloatingPointNaN) {
  std::vector<float> floatValues(257, 1.25F);
  floatValues[211] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(findMinMax(std::span<const float>{floatValues}), std::nullopt);

  std::vector<double> doubleValues(257, 2.5);
  doubleValues[31] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(findMinMax(std::span<const double>{doubleValues}), std::nullopt);
}
