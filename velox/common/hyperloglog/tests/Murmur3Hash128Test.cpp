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

#include "velox/common/hyperloglog/Murmur3Hash128.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

namespace facebook::velox::common::hll {
namespace {

uint64_t javaCompatOf(const std::string& s) {
  return static_cast<uint64_t>(Murmur3Hash128::hash64JavaCompat(
      s.data(), static_cast<int32_t>(s.size()), 0));
}

uint64_t legacyOf(const std::string& s) {
  return static_cast<uint64_t>(
      Murmur3Hash128::hash64(s.data(), static_cast<int32_t>(s.size()), 0));
}

std::string bytes(std::initializer_list<int> vals) {
  std::string s;
  for (int v : vals) {
    s.push_back(static_cast<char>(v));
  }
  return s;
}

// hash64ForLong is an independent port of the same airlift method that
// hash64JavaCompat implements for an 8-byte input, so the two must agree for
// every int64.
TEST(Murmur3Hash128Test, javaCompatMatchesHash64ForLong) {
  std::vector<int64_t> values = {
      0,
      1,
      127, // largest value whose low byte stays below 0x80
      128, // first value with a tail byte >= 0x80
      200,
      255,
      256,
      -1,
      -128,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(),
      1LL << 40,
  };
  for (int64_t i = 0; i < 2000; ++i) {
    values.push_back(i);
    values.push_back(-i);
  }

  for (int64_t value : values) {
    SCOPED_TRACE(fmt::format("value: {}", value));
    EXPECT_EQ(
        Murmur3Hash128::hash64JavaCompat(&value, sizeof(value), 0),
        Murmur3Hash128::hash64ForLong(value, 0));
  }
}

// hash() reads its tail bytes through const uint8_t*, so it is unaffected by
// char signedness. Its first output word is the hash64 return value by
// construction, which makes it an in-tree oracle for arbitrary lengths.
TEST(Murmur3Hash128Test, javaCompatMatchesHash128FirstWord) {
  std::string data;
  for (int i = 0; i < 300; ++i) {
    // Cycles through every byte value, so tails cover 0x00..0xFF.
    data.push_back(static_cast<char>(i * 7 % 256));

    SCOPED_TRACE(fmt::format("length: {}", data.size()));
    uint64_t out[2];
    Murmur3Hash128::hash(
        data.data(), static_cast<int32_t>(data.size()), 0, out);
    EXPECT_EQ(javaCompatOf(data), out[0]);
  }
}

// Values produced by io.airlift.slice.Murmur3Hash128.hash64.
TEST(Murmur3Hash128Test, javaCompatibilityVectors) {
  EXPECT_EQ(javaCompatOf(""), 0x0000000000000000ULL);
  EXPECT_EQ(javaCompatOf("hello"), 0xcbd8a7b341bd9b02ULL);
  EXPECT_EQ(javaCompatOf(bytes({0x80})), 0x61619a676395018aULL);
  // UTF-8 encoding of U+00E9.
  EXPECT_EQ(javaCompatOf(bytes({0xc3, 0xa9})), 0xc9187aa411d463e8ULL);
  EXPECT_EQ(
      javaCompatOf(std::string(15, static_cast<char>(0xff))),
      0x2c9d1a48cb13ee54ULL);
}

// hash64() is load-bearing for already-persisted approx_set, make_set_digest
// and khyperloglog_agg sketches, so its output must not move. These values pin
// the pre-existing signed-tail behaviour. They intentionally differ from
// javaCompatibilityVectors wherever a tail byte is >= 0x80.
TEST(Murmur3Hash128Test, legacyHash64IsUnchanged) {
  EXPECT_EQ(legacyOf(""), 0x0000000000000000ULL);
  EXPECT_EQ(legacyOf("hello"), 0xcbd8a7b341bd9b02ULL);
  EXPECT_EQ(legacyOf(bytes({0x80})), 0xb6aa75aff6f3b434ULL);
  EXPECT_EQ(legacyOf(bytes({0xc3, 0xa9})), 0x4bcacd1ed0f55280ULL);
  EXPECT_EQ(
      legacyOf(std::string(15, static_cast<char>(0xff))),
      0xe187e84efa24f091ULL);
}

// The two entry points may only diverge when a tail byte is >= 0x80: inputs
// that are a whole number of 16-byte blocks, or whose tail is pure ASCII, must
// hash identically under both.
TEST(Murmur3Hash128Test, entryPointsAgreeWhenNoHighTailByte) {
  // No tail at all.
  const auto block = bytes(
      {0xf0,
       0xf1,
       0xf2,
       0xf3,
       0xf4,
       0xf5,
       0xf6,
       0xf7,
       0xf8,
       0xf9,
       0xfa,
       0xfb,
       0xfc,
       0xfd,
       0xfe,
       0xff});
  ASSERT_EQ(block.size(), 16);
  EXPECT_EQ(javaCompatOf(block), legacyOf(block));

  // Pure-ASCII tails of every length.
  std::string ascii;
  for (int i = 0; i < 40; ++i) {
    ascii.push_back(static_cast<char>('a' + (i % 26)));
    SCOPED_TRACE(fmt::format("length: {}", ascii.size()));
    EXPECT_EQ(javaCompatOf(ascii), legacyOf(ascii));
  }

  // A high tail byte must change the result.
  EXPECT_NE(javaCompatOf(bytes({0x80})), legacyOf(bytes({0x80})));
}

} // namespace
} // namespace facebook::velox::common::hll
