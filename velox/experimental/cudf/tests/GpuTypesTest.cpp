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

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/types/GpuStringView.cuh"
#include "velox/experimental/cudf/types/GpuTimestamp.cuh"

#include <gtest/gtest.h>

// Direct include of the shadow header: this .cpp target intentionally does
// NOT add the gpu_shadows path to its include search order, so we must
// reach the shadow by its full repo path. None of the other includes
// above pull in the real `velox/common/base/BitUtil.h`, so there's no ODR
// risk.
#include "velox/experimental/cudf/functions/gpu_shadows/velox/common/base/BitUtil.h"

using namespace facebook::velox::gpu;

TEST(GpuTypesTest, resolverPrimitives) {
  static_assert(std::is_same_v<GpuExec::resolver<double>::in_type, double>);
  static_assert(std::is_same_v<GpuExec::resolver<int64_t>::in_type, int64_t>);
  static_assert(std::is_same_v<GpuExec::resolver<int32_t>::in_type, int32_t>);
  static_assert(std::is_same_v<GpuExec::resolver<float>::in_type, float>);
  static_assert(std::is_same_v<GpuExec::resolver<bool>::in_type, bool>);
  static_assert(
      std::is_same_v<GpuExec::resolver<double>::null_free_in_type, double>);
}

TEST(GpuTypesTest, resolverVarchar) {
  using R = GpuExec::resolver<facebook::velox::Varchar>;
  static_assert(std::is_same_v<R::in_type, GpuStringView>);
  static_assert(std::is_same_v<R::out_type, GpuStringView>);
  static_assert(std::is_same_v<R::null_free_in_type, GpuStringView>);
}

TEST(GpuTypesTest, resolverVarbinary) {
  using R = GpuExec::resolver<facebook::velox::Varbinary>;
  static_assert(std::is_same_v<R::in_type, GpuStringView>);
}

TEST(GpuTypesTest, resolverDate) {
  using R = GpuExec::resolver<facebook::velox::Date>;
  static_assert(std::is_same_v<R::in_type, int32_t>);
}

TEST(GpuTypesTest, resolverIntervalDayTime) {
  using R = GpuExec::resolver<facebook::velox::IntervalDayTime>;
  static_assert(std::is_same_v<R::in_type, int64_t>);
}

TEST(GpuTypesTest, resolverIntervalYearMonth) {
  using R = GpuExec::resolver<facebook::velox::IntervalYearMonth>;
  static_assert(std::is_same_v<R::in_type, int32_t>);
}

TEST(GpuTypesTest, resolverTime) {
  using R = GpuExec::resolver<facebook::velox::Time>;
  static_assert(std::is_same_v<R::in_type, int64_t>);
}

TEST(GpuTypesTest, resolverTimestamp) {
  using R = GpuExec::resolver<facebook::velox::Timestamp>;
  static_assert(std::is_same_v<R::in_type, GpuTimestamp>);
  static_assert(std::is_same_v<R::out_type, GpuTimestamp>);
  static_assert(std::is_same_v<R::null_free_in_type, GpuTimestamp>);
}

TEST(GpuTypesTest, gpuStringViewBasic) {
  const char* s = "hello";
  GpuStringView sv(s, 5);
  EXPECT_EQ(sv.size(), 5);
  EXPECT_FALSE(sv.empty());
  EXPECT_EQ(sv.data(), s);
  EXPECT_EQ(sv.begin(), s);
  EXPECT_EQ(sv.end(), s + 5);
}

TEST(GpuTypesTest, gpuStringViewEquality) {
  const char* s1 = "hello";
  const char* s2 = "hello";
  const char* s3 = "world";
  GpuStringView sv1(s1, 5);
  GpuStringView sv2(s2, 5);
  GpuStringView sv3(s3, 5);
  GpuStringView empty;

  EXPECT_EQ(sv1, sv2);
  EXPECT_NE(sv1, sv3);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);
}

TEST(GpuTypesTest, gpuTimestampComparison) {
  GpuTimestamp a(100, 500);
  GpuTimestamp b(100, 600);
  GpuTimestamp c(101, 0);
  GpuTimestamp d(100, 500);

  EXPECT_EQ(a, d);
  EXPECT_NE(a, b);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b < c);
  EXPECT_TRUE(a <= d);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(c > b);
  EXPECT_TRUE(c >= b);
  EXPECT_TRUE(a >= d);
}

TEST(GpuTypesTest, gpuTimestampDefault) {
  GpuTimestamp t;
  EXPECT_EQ(t.seconds, 0);
  EXPECT_EQ(t.nanos, 0u);
}

TEST(GpuTypesTest, bitsCountBitsSingleWord) {
  // 0b1011_0101 -> 5 bits set.
  uint64_t word = 0xB5;
  EXPECT_EQ(5, facebook::velox::bits::countBits(&word, 0, 8));
  // Range [2, 8) over the same word: 0b1011_01 -> 4 bits set above bit 2.
  EXPECT_EQ(4, facebook::velox::bits::countBits(&word, 2, 8));
  // Empty range.
  EXPECT_EQ(0, facebook::velox::bits::countBits(&word, 4, 4));
  // Whole 64 bits all set.
  uint64_t full = ~uint64_t{0};
  EXPECT_EQ(64, facebook::velox::bits::countBits(&full, 0, 64));
  // Single bit at position 35.
  uint64_t one = uint64_t{1} << 35;
  EXPECT_EQ(1, facebook::velox::bits::countBits(&one, 0, 64));
  EXPECT_EQ(0, facebook::velox::bits::countBits(&one, 36, 64));
}

TEST(GpuTypesTest, bitsCountBitsPreconditions) {
  // Defensive precondition checks: negative `begin` or `end <= begin`
  // returns 0 rather than triggering implementation-defined behavior
  // from signed right-shift.
  uint64_t word = ~uint64_t{0};
  EXPECT_EQ(0, facebook::velox::bits::countBits(&word, -1, 64));
  EXPECT_EQ(0, facebook::velox::bits::countBits(&word, -10, -5));
  EXPECT_EQ(0, facebook::velox::bits::countBits(&word, 10, 5)); // end < begin
  EXPECT_EQ(0, facebook::velox::bits::countBits(&word, 10, 10)); // empty
}

TEST(GpuTypesTest, bitsCountBitsMultiWord) {
  uint64_t words[3] = {~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};
  // Full 3 words: 192 bits set.
  EXPECT_EQ(192, facebook::velox::bits::countBits(words, 0, 192));
  // Range straddling word boundary: [60, 70).
  EXPECT_EQ(10, facebook::velox::bits::countBits(words, 60, 70));
  // Only the middle word.
  EXPECT_EQ(64, facebook::velox::bits::countBits(words, 64, 128));

  uint64_t mixed[2] = {0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL};
  // First word: 32 set, second word: 32 set.
  EXPECT_EQ(64, facebook::velox::bits::countBits(mixed, 0, 128));
}
