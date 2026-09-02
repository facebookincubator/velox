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

// hash64ForLong is hash64 specialized for an 8 byte input, so the two must
// return the same value for every input and seed.
TEST(Murmur3Hash128Test, hash64AgreesWithHash64ForLong) {
  std::vector<int64_t> values = {
      0,
      1,
      -1,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(),
      0x80,
      0xff,
      0x7f7f7f7f7f7f7f7fLL,
      static_cast<int64_t>(0x8080808080808080ULL),
      100012345678901LL,
  };

  std::mt19937_64 rng(12345);
  for (int i = 0; i < 10000; ++i) {
    values.push_back(static_cast<int64_t>(rng()));
  }

  for (auto value : values) {
    for (int64_t seed : {int64_t{0}, int64_t{1}, int64_t{0x5bd1e995}}) {
      ASSERT_EQ(
          hash64OfBytes(value, seed),
          Murmur3Hash128::hash64ForLong(value, seed))
          << "value=" << value << " seed=" << seed;
    }
  }
}

TEST(Murmur3Hash128Test, hash64ZeroExtendsHighTailBytes) {
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

// Inputs of 9 to 15 bytes reach tail positions an 8 byte input cannot, so
// hash64ForLong is unusable as the oracle. Expected values were computed from
// airlift/slice Murmur3Hash128.java#L152, the reference cited on hash64 in
// Murmur3Hash128.h, which masks each tail byte with & 0xFF.
TEST(Murmur3Hash128Test, hash64MatchesReferenceForNineToFifteenByteInputs) {
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
    std::string input(length - 2, 'a');
    input.append("\xc3\xa9");
    ASSERT_EQ(input.size(), length);
    EXPECT_EQ(
        Murmur3Hash128::hash64(input.data(), input.size(), /*seed=*/0), hash)
        << "length=" << length;
  }
}
