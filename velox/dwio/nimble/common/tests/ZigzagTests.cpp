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

#include <limits>

#include "velox/dwio/nimble/common/Zigzag.h"

using namespace ::facebook::nimble::zigzag;

TEST(ZigzagTest, encode32) {
  EXPECT_EQ(zigzagEncode32(0), 0);
  EXPECT_EQ(zigzagEncode32(-1), 1);
  EXPECT_EQ(zigzagEncode32(1), 2);
  EXPECT_EQ(zigzagEncode32(-2), 3);
  EXPECT_EQ(zigzagEncode32(2), 4);
  EXPECT_EQ(zigzagEncode32(-3), 5);
  EXPECT_EQ(zigzagEncode32(std::numeric_limits<int32_t>::max()), 4294967294U);
  EXPECT_EQ(zigzagEncode32(std::numeric_limits<int32_t>::min()), 4294967295U);
}

TEST(ZigzagTest, decode32) {
  EXPECT_EQ(zigzagDecode32(0), 0);
  EXPECT_EQ(zigzagDecode32(1), -1);
  EXPECT_EQ(zigzagDecode32(2), 1);
  EXPECT_EQ(zigzagDecode32(3), -2);
  EXPECT_EQ(zigzagDecode32(4), 2);
  EXPECT_EQ(zigzagDecode32(5), -3);
  EXPECT_EQ(zigzagDecode32(4294967294U), std::numeric_limits<int32_t>::max());
  EXPECT_EQ(zigzagDecode32(4294967295U), std::numeric_limits<int32_t>::min());
}

TEST(ZigzagTest, roundTrip32) {
  for (int32_t val :
       {0,
        1,
        -1,
        100,
        -100,
        1'000'000,
        -1'000'000,
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min()}) {
    EXPECT_EQ(zigzagDecode32(zigzagEncode32(val)), val);
  }
}

TEST(ZigzagTest, encode64) {
  EXPECT_EQ(zigzagEncode64(0), 0);
  EXPECT_EQ(zigzagEncode64(-1), 1);
  EXPECT_EQ(zigzagEncode64(1), 2);
  EXPECT_EQ(zigzagEncode64(-2), 3);
  EXPECT_EQ(
      zigzagEncode64(std::numeric_limits<int64_t>::max()),
      std::numeric_limits<uint64_t>::max() - 1);
  EXPECT_EQ(
      zigzagEncode64(std::numeric_limits<int64_t>::min()),
      std::numeric_limits<uint64_t>::max());
}

TEST(ZigzagTest, roundTrip64) {
  for (int64_t val :
       {int64_t{0},
        int64_t{1},
        int64_t{-1},
        int64_t{100},
        int64_t{-100},
        int64_t{1'000'000'000'000},
        int64_t{-1'000'000'000'000},
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min()}) {
    EXPECT_EQ(zigzagDecode64(zigzagEncode64(val)), val);
  }
}
