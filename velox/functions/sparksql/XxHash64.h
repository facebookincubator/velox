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
  static uint64_t hashBytes(const StringView& input, uint64_t seed);

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

  static uint64_t hashBytesByWords(const StringView& input, uint64_t seed);
};

} // namespace facebook::velox::functions::sparksql
