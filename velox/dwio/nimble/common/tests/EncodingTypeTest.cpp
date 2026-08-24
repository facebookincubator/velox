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

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble::test {

// --- EncodingPhysicalType<int32_t>: identity mapping ---

TEST(EncodingTypeTest, int32IdentityMapping) {
  int32_t value = 42;
  auto physical = EncodingPhysicalType<int32_t>::asEncodingPhysicalType(value);
  static_assert(std::is_same_v<decltype(physical), uint32_t>);
  auto logical = EncodingPhysicalType<int32_t>::asEncodingLogicalType(physical);
  EXPECT_EQ(logical, 42);
}

TEST(EncodingTypeTest, int32NegativeRoundTrip) {
  int32_t value = -12345;
  auto physical = EncodingPhysicalType<int32_t>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<int32_t>::asEncodingLogicalType(physical);
  EXPECT_EQ(logical, -12345);
}

TEST(EncodingTypeTest, int32Limits) {
  {
    int32_t value = std::numeric_limits<int32_t>::min();
    auto physical =
        EncodingPhysicalType<int32_t>::asEncodingPhysicalType(value);
    auto logical =
        EncodingPhysicalType<int32_t>::asEncodingLogicalType(physical);
    EXPECT_EQ(logical, std::numeric_limits<int32_t>::min());
  }
  {
    int32_t value = std::numeric_limits<int32_t>::max();
    auto physical =
        EncodingPhysicalType<int32_t>::asEncodingPhysicalType(value);
    auto logical =
        EncodingPhysicalType<int32_t>::asEncodingLogicalType(physical);
    EXPECT_EQ(logical, std::numeric_limits<int32_t>::max());
  }
}

// --- EncodingPhysicalType<float>: float <-> uint32_t bitwise conversion ---

TEST(EncodingTypeTest, floatRoundTrip) {
  float value = 3.14f;
  auto physical = EncodingPhysicalType<float>::asEncodingPhysicalType(value);
  static_assert(std::is_same_v<decltype(physical), uint32_t>);
  auto logical = EncodingPhysicalType<float>::asEncodingLogicalType(physical);
  EXPECT_FLOAT_EQ(logical, 3.14f);
}

TEST(EncodingTypeTest, floatNegativeZero) {
  float value = -0.0f;
  auto physical = EncodingPhysicalType<float>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<float>::asEncodingLogicalType(physical);
  // -0.0 and +0.0 compare equal but have different bit patterns
  EXPECT_EQ(logical, 0.0f);
  uint32_t bits;
  std::memcpy(&bits, &logical, sizeof(float));
  EXPECT_NE(bits, 0u); // -0.0f has sign bit set
}

TEST(EncodingTypeTest, floatNaN) {
  float value = std::numeric_limits<float>::quiet_NaN();
  auto physical = EncodingPhysicalType<float>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<float>::asEncodingLogicalType(physical);
  EXPECT_TRUE(std::isnan(logical));
}

TEST(EncodingTypeTest, floatInfinity) {
  float value = std::numeric_limits<float>::infinity();
  auto physical = EncodingPhysicalType<float>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<float>::asEncodingLogicalType(physical);
  EXPECT_EQ(logical, std::numeric_limits<float>::infinity());
}

// --- EncodingPhysicalType<double>: double <-> uint64_t bitwise conversion ---

TEST(EncodingTypeTest, doubleRoundTrip) {
  double value = 2.718281828;
  auto physical = EncodingPhysicalType<double>::asEncodingPhysicalType(value);
  static_assert(std::is_same_v<decltype(physical), uint64_t>);
  auto logical = EncodingPhysicalType<double>::asEncodingLogicalType(physical);
  EXPECT_DOUBLE_EQ(logical, 2.718281828);
}

TEST(EncodingTypeTest, doubleNaN) {
  double value = std::numeric_limits<double>::quiet_NaN();
  auto physical = EncodingPhysicalType<double>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<double>::asEncodingLogicalType(physical);
  EXPECT_TRUE(std::isnan(logical));
}

TEST(EncodingTypeTest, doubleNegativeZero) {
  double value = -0.0;
  auto physical = EncodingPhysicalType<double>::asEncodingPhysicalType(value);
  auto logical = EncodingPhysicalType<double>::asEncodingLogicalType(physical);
  EXPECT_EQ(logical, 0.0);
  uint64_t bits;
  std::memcpy(&bits, &logical, sizeof(double));
  EXPECT_NE(bits, 0u); // -0.0 has sign bit set
}

// --- asEncodingPhysicalTypeSpan ---

TEST(EncodingTypeTest, asEncodingPhysicalTypeSpanFloat) {
  std::vector<float> values = {1.0f, 2.0f, 3.0f, -0.0f};
  auto span = std::span<const float>(values);
  auto physicalSpan =
      EncodingPhysicalType<float>::asEncodingPhysicalTypeSpan(span);

  EXPECT_EQ(physicalSpan.size(), values.size());
  // Verify the bitwise reinterpretation is consistent
  for (size_t i = 0; i < values.size(); ++i) {
    auto expected =
        EncodingPhysicalType<float>::asEncodingPhysicalType(values[i]);
    EXPECT_EQ(physicalSpan[i], expected);
  }
}

TEST(EncodingTypeTest, asEncodingPhysicalTypeSpanInt32) {
  std::vector<int32_t> values = {-1, 0, 1, 100};
  auto span = std::span<const int32_t>(values);
  auto physicalSpan =
      EncodingPhysicalType<int32_t>::asEncodingPhysicalTypeSpan(span);

  EXPECT_EQ(physicalSpan.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    auto expected =
        EncodingPhysicalType<int32_t>::asEncodingPhysicalType(values[i]);
    EXPECT_EQ(physicalSpan[i], expected);
  }
}

TEST(EncodingTypeTest, asEncodingPhysicalType) {
  using PhysicalType = EncodingPhysicalType<float>::type;

  std::vector<float> values = {1.0f, 2.0f, 3.0f};
  auto physicalSpan = EncodingPhysicalType<float>::asEncodingPhysicalTypeSpan(
      std::span<float>{values});
  auto physicalValues = [&] {
    return std::vector<PhysicalType>{physicalSpan.begin(), physicalSpan.end()};
  };

  const std::vector<PhysicalType> expectedBefore{
      EncodingPhysicalType<float>::asEncodingPhysicalType(1.0f),
      EncodingPhysicalType<float>::asEncodingPhysicalType(2.0f),
      EncodingPhysicalType<float>::asEncodingPhysicalType(3.0f)};
  EXPECT_EQ(expectedBefore, physicalValues());

  physicalSpan[1] = EncodingPhysicalType<float>::asEncodingPhysicalType(-3.5f);
  const std::vector<PhysicalType> expectedAfter{
      EncodingPhysicalType<float>::asEncodingPhysicalType(1.0f),
      EncodingPhysicalType<float>::asEncodingPhysicalType(-3.5f),
      EncodingPhysicalType<float>::asEncodingPhysicalType(3.0f)};
  EXPECT_EQ(expectedAfter, physicalValues());
  EXPECT_FLOAT_EQ(values[1], -3.5f);
}

TEST(EncodingTypeTest, asEncodingPhysicalTypeSpanEmpty) {
  std::vector<float> values;
  auto span = std::span<const float>(values);
  auto physicalSpan =
      EncodingPhysicalType<float>::asEncodingPhysicalTypeSpan(span);
  EXPECT_EQ(physicalSpan.size(), 0);
}

} // namespace facebook::nimble::test
