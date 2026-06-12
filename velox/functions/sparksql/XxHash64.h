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

#include <cstdint>

#include "velox/common/base/BitUtil.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

/// Spark-compatible XXH64 helper.
class XxHash64 final {
 public:
  using SeedType = int64_t;
  using ReturnType = int64_t;

  /// Hashes a 32-bit integer value.
  static uint64_t hashInt32(const int32_t input, uint64_t seed) {
    int64_t hash = seed + PRIME64_5 + 4L;
    hash ^= static_cast<int64_t>((input & 0xFFFFFFFFL) * PRIME64_1);
    hash = bits::rotateLeft64(hash, 23) * PRIME64_2 + PRIME64_3;
    return fmix(hash);
  }

  /// Hashes a 64-bit integer value.
  static uint64_t hashInt64(int64_t input, uint64_t seed) {
    int64_t hash = seed + PRIME64_5 + 8L;
    hash ^= bits::rotateLeft64(input * PRIME64_2, 31) * PRIME64_1;
    hash = bits::rotateLeft64(hash, 27) * PRIME64_1 + PRIME64_4;
    return fmix(hash);
  }

  /// Hashes a float value, treating -0.0f as +0.0f.
  static uint64_t hashFloat(float input, uint64_t seed) {
    return hashInt32(
        input == -0.f ? 0 : *reinterpret_cast<uint32_t*>(&input), seed);
  }

  /// Hashes a double value, treating -0.0 as +0.0.
  static uint64_t hashDouble(double input, uint64_t seed) {
    return hashInt64(
        input == -0. ? 0 : *reinterpret_cast<uint64_t*>(&input), seed);
  }

  /// Hashes a byte sequence.
  static uint64_t hashBytes(const StringView& input, uint64_t seed) {
    const char* i = input.data();
    const char* const end = input.data() + input.size();

    uint64_t hash = hashBytesByWords(input, seed);
    uint32_t length = input.size();
    auto offset = i + (length & -8);
    if (offset + 4L <= end) {
      hash ^= (*reinterpret_cast<const uint64_t*>(offset) & 0xFFFFFFFFL) *
          PRIME64_1;
      hash = bits::rotateLeft64(hash, 23) * PRIME64_2 + PRIME64_3;
      offset += 4L;
    }

    while (offset < end) {
      hash ^= (*reinterpret_cast<const uint64_t*>(offset) & 0xFFL) * PRIME64_5;
      hash = bits::rotateLeft64(hash, 11) * PRIME64_1;
      offset++;
    }
    return fmix(hash);
  }

  /// Hashes a long decimal value using Spark's byte representation.
  static uint64_t hashLongDecimal(int128_t input, uint64_t seed) {
    char out[sizeof(int128_t)];
    int32_t length = DecimalUtil::toByteArray(input, out);
    return hashBytes(StringView(out, length), seed);
  }

  /// Hashes a timestamp by its microsecond value.
  static uint64_t hashTimestamp(Timestamp input, uint64_t seed) {
    return hashInt64(input.toMicros(), seed);
  }

 private:
  static const uint64_t PRIME64_1 = 0x9E3779B185EBCA87L;
  static const uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FL;
  static const uint64_t PRIME64_3 = 0x165667B19E3779F9L;
  static const uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63L;
  static const uint64_t PRIME64_5 = 0x27D4EB2F165667C5L;

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
      for (; i <= end - 32; i += 32) {
        v1 = bits::rotateLeft64(
                 v1 + (*reinterpret_cast<const uint64_t*>(i) * PRIME64_2), 31) *
            PRIME64_1;
        v2 = bits::rotateLeft64(
                 v2 + (*reinterpret_cast<const uint64_t*>(i + 8) * PRIME64_2),
                 31) *
            PRIME64_1;
        v3 = bits::rotateLeft64(
                 v3 + (*reinterpret_cast<const uint64_t*>(i + 16) * PRIME64_2),
                 31) *
            PRIME64_1;
        v4 = bits::rotateLeft64(
                 v4 + (*reinterpret_cast<const uint64_t*>(i + 24) * PRIME64_2),
                 31) *
            PRIME64_1;
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
      hash ^= bits::rotateLeft64(
                  *reinterpret_cast<const uint64_t*>(i) * PRIME64_2, 31) *
          PRIME64_1;
      hash = bits::rotateLeft64(hash, 27) * PRIME64_1 + PRIME64_4;
    }
    return hash;
  }
};

} // namespace facebook::velox::functions::sparksql
