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

#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace facebook::velox::common::hll;

namespace {

int64_t hash64OfBytes(int64_t value, int64_t seed) {
  char bytes[sizeof(int64_t)];
  std::memcpy(bytes, &value, sizeof(bytes));
  return Murmur3Hash128::hash64(bytes, sizeof(bytes), seed);
}

} // namespace

// hash64ForLong is the closed form of hash64 over an 8 byte input, so the two
// must agree for every value and seed. They diverge whenever a byte is >= 0x80
// if the tail is read through a signed char.
TEST(Murmur3Hash128Test, hash64AgreesWithHash64ForLong) {
  std::vector<int64_t> values = {
      0,
      1,
      -1,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(),
      0x80, // single high byte
      0xff,
      0x7f7f7f7f7f7f7f7fLL, // no high bytes
      static_cast<int64_t>(0x8080808080808080ULL), // all high bytes
      100012345678901LL, // realistic user id
  };

  std::mt19937_64 rng(12345);
  for (int i = 0; i < 10000; ++i) {
    values.push_back(static_cast<int64_t>(rng()));
  }

  for (auto value : values) {
    for (int64_t seed : {int64_t{0}, int64_t{1}, int64_t{0x5bd1e995}}) {
      EXPECT_EQ(
          hash64OfBytes(value, seed),
          Murmur3Hash128::hash64ForLong(value, seed))
          << "value=" << value << " seed=" << seed;
    }
  }
}

// Bytes below 0x80 are unaffected by the signedness of the tail read, so an
// all-ASCII input hashes the same either way. This pins the boundary: the same
// eight bytes must hash identically whether they arrive as a buffer or a long.
TEST(Murmur3Hash128Test, hash64ZeroExtendsHighTailBytes) {
  // "aaaaaa\xc3\xa9" is the UTF-8 encoding of u8"aaaaaaé" and is exactly the
  // little endian representation of the BIGINT below.
  const std::string highByteInput = "aaaaaa\xc3\xa9";
  ASSERT_EQ(highByteInput.size(), sizeof(int64_t));

  int64_t asLong;
  std::memcpy(&asLong, highByteInput.data(), sizeof(asLong));
  ASSERT_EQ(asLong, -6214015989967658655LL);

  EXPECT_EQ(
      Murmur3Hash128::hash64(
          highByteInput.data(), highByteInput.size(), /*seed=*/0),
      Murmur3Hash128::hash64ForLong(asLong, /*seed=*/0));

  const std::string asciiInput = "aaaaaaaa";
  int64_t asciiAsLong;
  std::memcpy(&asciiAsLong, asciiInput.data(), sizeof(asciiAsLong));
  EXPECT_EQ(
      Murmur3Hash128::hash64(asciiInput.data(), asciiInput.size(), /*seed=*/0),
      Murmur3Hash128::hash64ForLong(asciiAsLong, /*seed=*/0));
}

// Lengths 9 to 15 populate the k2 lane of the tail switch, which hash64ForLong
// cannot reach. Expected values were computed from the Murmur3-128 reference
// definition, which masks each tail byte with & 0xFF.
TEST(Murmur3Hash128Test, hash64MatchesReferenceForK2TailLengths) {
  const std::vector<std::pair<size_t, int64_t>> expected = {
      {9, -3243918217540306855LL},
      {10, -7998242445834668767LL},
      {11, -4701723541745919412LL},
      {12, 1137560024168287992LL},
      {13, -2349531979471542545LL},
      {14, -2514671622296473736LL},
      {15, 5186136466562847178LL},
  };

  for (const auto& [length, hash] : expected) {
    // 'a' repeated, terminated by the two byte UTF-8 encoding of u8"é" so that
    // the highest tail offsets carry bytes >= 0x80.
    std::string input(length - 2, 'a');
    input.append("\xc3\xa9");
    ASSERT_EQ(input.size(), length);
    EXPECT_EQ(
        Murmur3Hash128::hash64(input.data(), input.size(), /*seed=*/0), hash)
        << "length=" << length;
  }
}
