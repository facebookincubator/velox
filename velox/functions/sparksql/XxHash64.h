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

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::sparksql {

class XxHash64 final {
 public:
  using SeedType = int64_t;
  using ReturnType = int64_t;

  static uint64_t hashInt32(int32_t input, uint64_t seed) {
    uint64_t hash = static_cast<uint64_t>(seed) + PRIME64_5 + 4ULL;
    uint64_t data = static_cast<uint64_t>(static_cast<uint32_t>(input));
    hash ^= data * PRIME64_1;
    hash = bits::rotateLeft64(hash, 23) * PRIME64_2 + PRIME64_3;
    return fmix(hash);
  }

  static uint64_t hashInt64(int64_t input, uint64_t seed) {
    uint64_t hash = static_cast<uint64_t>(seed) + PRIME64_5 + 8ULL;
    uint64_t data = static_cast<uint64_t>(input);
    hash ^= bits::rotateLeft64(data * PRIME64_2, 31) * PRIME64_1;
    hash = bits::rotateLeft64(hash, 27) * PRIME64_1 + PRIME64_4;
    return fmix(hash);
  }

  // Floating point numbers are hashed as if they are integers, with
  // -0f defined to have the same output as +0f.
  static uint64_t hashFloat(float input, uint64_t seed) {
    uint32_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));
    if (input == -0.f) {
      bits = 0;
    }
    return hashInt32(static_cast<int32_t>(bits), seed);
  }

  static uint64_t hashDouble(double input, uint64_t seed) {
    uint64_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));
    if (input == -0.) {
      bits = 0;
    }
    return hashInt64(static_cast<int64_t>(bits), seed);
  }

  static uint64_t hashBytes(const StringView& input, uint64_t seed) {
    const char* i = input.data();
    const char* const end = input.data() + input.size();

    uint64_t hash = hashBytesByWords(input, seed);
    uint32_t length = input.size();
    auto offset = i + (length & -8);
    if (offset + 4L <= end) {
      uint32_t value = 0;
      std::memcpy(&value, offset, sizeof(value));
      hash ^= static_cast<uint64_t>(value) * PRIME64_1;
      hash = bits::rotateLeft64(hash, 23) * PRIME64_2 + PRIME64_3;
      offset += 4L;
    }

    while (offset < end) {
      hash ^= static_cast<uint64_t>(static_cast<uint8_t>(*offset)) * PRIME64_5;
      hash = bits::rotateLeft64(hash, 11) * PRIME64_1;
      offset++;
    }
    return fmix(hash);
  }

  static uint64_t hashLongDecimal(int128_t input, uint64_t seed) {
    char out[sizeof(int128_t)];
    int32_t length = DecimalUtil::toByteArray(input, out);
    return hashBytes(StringView(out, length), seed);
  }

  static uint64_t hashTimestamp(Timestamp input, uint64_t seed) {
    return hashInt64(input.toMicros(), seed);
  }

 private:
  static constexpr uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
  static constexpr uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
  static constexpr uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
  static constexpr uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
  static constexpr uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

  static uint64_t fmix(uint64_t hash) {
    hash ^= hash >> 33;
    hash *= PRIME64_2;
    hash ^= hash >> 29;
    hash *= PRIME64_3;
    hash ^= hash >> 32;
    return hash;
  }

  static uint64_t hashBytesByWords(const StringView& input, uint64_t seed) {
    const char* i = input.data();
    const char* const end = input.data() + input.size();
    uint32_t length = input.size();
    uint64_t hash;
    if (length >= 32) {
      uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
      uint64_t v2 = seed + PRIME64_2;
      uint64_t v3 = seed;
      uint64_t v4 = seed - PRIME64_1;
      const char* const limit = end - 32;
      while (i <= limit) {
        uint64_t word = 0;
        std::memcpy(&word, i, sizeof(word));
        v1 = bits::rotateLeft64(v1 + word * PRIME64_2, 31) * PRIME64_1;
        i += 8;

        std::memcpy(&word, i, sizeof(word));
        v2 = bits::rotateLeft64(v2 + word * PRIME64_2, 31) * PRIME64_1;
        i += 8;

        std::memcpy(&word, i, sizeof(word));
        v3 = bits::rotateLeft64(v3 + word * PRIME64_2, 31) * PRIME64_1;
        i += 8;

        std::memcpy(&word, i, sizeof(word));
        v4 = bits::rotateLeft64(v4 + word * PRIME64_2, 31) * PRIME64_1;
        i += 8;
      }
      hash = bits::rotateLeft64(v1, 1) + bits::rotateLeft64(v2, 7) +
          bits::rotateLeft64(v3, 12) + bits::rotateLeft64(v4, 18);
      v1 *= PRIME64_2;
      v1 = bits::rotateLeft64(v1, 31);
      v1 *= PRIME64_1;
      hash ^= v1;
      hash = hash * PRIME64_1 + PRIME64_4;

      v2 *= PRIME64_2;
      v2 = bits::rotateLeft64(v2, 31);
      v2 *= PRIME64_1;
      hash ^= v2;
      hash = hash * PRIME64_1 + PRIME64_4;

      v3 *= PRIME64_2;
      v3 = bits::rotateLeft64(v3, 31);
      v3 *= PRIME64_1;
      hash ^= v3;
      hash = hash * PRIME64_1 + PRIME64_4;

      v4 *= PRIME64_2;
      v4 = bits::rotateLeft64(v4, 31);
      v4 *= PRIME64_1;
      hash ^= v4;
      hash = hash * PRIME64_1 + PRIME64_4;
    } else {
      hash = seed + PRIME64_5;
    }

    hash += length;

    for (; i <= end - 8; i += 8) {
      uint64_t word = 0;
      std::memcpy(&word, i, sizeof(word));
      hash ^= bits::rotateLeft64(word * PRIME64_2, 31) * PRIME64_1;
      hash = bits::rotateLeft64(hash, 27) * PRIME64_1 + PRIME64_4;
    }
    return hash;
  }
};

} // namespace facebook::velox::functions::sparksql
