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

#include "velox/dwio/parquet/reader/BigEndianDecimal.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/HugeInt.h"

using namespace std::string_view_literals;

namespace facebook::velox::parquet {
namespace {

// Returns the 'numBytes' least significant bytes of 'value' in big-endian
// order.
std::string toBigEndian(int128_t value, size_t numBytes) {
  std::string bytes(numBytes, '\0');
  for (size_t i = 0; i < numBytes; ++i) {
    bytes[numBytes - 1 - i] = static_cast<char>(value >> (8 * i));
  }
  return bytes;
}

TEST(BigEndianDecimalTest, shortDecimal) {
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\x00"sv), 0);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\x7F"sv), 127);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\x80"sv), -128);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\xFF"sv), -1);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\x00\x80"sv), 128);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\xFF\x7F"sv), -129);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\x00\x98\x96\x81"sv), 10'000'001);
  EXPECT_EQ(decimalFromBigEndian<int64_t>("\xFF\x67\x69\x7F"sv), -10'000'001);
  EXPECT_EQ(
      decimalFromBigEndian<int64_t>("\x7F\xFF\xFF\xFF\xFF\xFF\xFF\xFF"sv),
      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(
      decimalFromBigEndian<int64_t>("\x80\x00\x00\x00\x00\x00\x00\x00"sv),
      std::numeric_limits<int64_t>::min());
}

TEST(BigEndianDecimalTest, longDecimal) {
  const auto large = HugeInt::build(1'546'093'991, 4'054'979'645);
  for (const int128_t value : {int128_t{-479}, large, -large}) {
    EXPECT_EQ(decimalFromBigEndian<int128_t>(toBigEndian(value, 16)), value);
  }
  EXPECT_EQ(decimalFromBigEndian<int128_t>("\x80"sv), -128);
  EXPECT_EQ(
      decimalFromBigEndian<int128_t>("\x00\x80\x00\x00\x00\x00\x00\x00\x00"sv),
      HugeInt::build(0, 0x8000'0000'0000'0000));
  EXPECT_EQ(
      decimalFromBigEndian<int128_t>("\xFF\x7F\xFF\xFF\xFF\xFF\xFF\xFF\xFF"sv),
      -HugeInt::build(0, 0x8000'0000'0000'0001));
}

TEST(BigEndianDecimalTest, everyWidth) {
  // Checks the largest and smallest value that fits in each width.
  for (size_t numBytes = 1; numBytes <= sizeof(int128_t); ++numBytes) {
    SCOPED_TRACE(fmt::format("numBytes: {}", numBytes));
    const auto max =
        static_cast<int128_t>((uint128_t{1} << (8 * numBytes - 1)) - 1);
    const auto min = -max - 1;
    for (const auto value : {max, min, int128_t{-1}, int128_t{0}}) {
      if (numBytes <= sizeof(int64_t)) {
        EXPECT_EQ(
            decimalFromBigEndian<int64_t>(toBigEndian(value, numBytes)), value);
      }
      EXPECT_EQ(
          decimalFromBigEndian<int128_t>(toBigEndian(value, numBytes)), value);
    }
  }
}

TEST(BigEndianDecimalTest, invalidLength) {
  VELOX_ASSERT_THROW(
      decimalFromBigEndian<int64_t>(""sv),
      "Decimal value must have at least one byte");
  VELOX_ASSERT_THROW(
      decimalFromBigEndian<int128_t>(""sv),
      "Decimal value must have at least one byte");
  VELOX_ASSERT_THROW(
      decimalFromBigEndian<int64_t>(std::string(9, '\0')),
      "Decimal value has more bytes than the target integer type");
  VELOX_ASSERT_THROW(
      decimalFromBigEndian<int128_t>(std::string(17, '\0')),
      "Decimal value has more bytes than the target integer type");
}

} // namespace
} // namespace facebook::velox::parquet
