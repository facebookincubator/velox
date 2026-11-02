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

#include <folly/CPortability.h>

#include "velox/common/base/BitUtil.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

/// Spark-compatible XXH64 helper.
///
/// hashInt32, hashInt64, hashFloat, hashDouble, hashTimestamp and fmix are
/// force-inlined so that per-row loops over fixed-width inputs can vectorize.
/// Defining them in the header is not enough: callers such as Hash.cpp exhaust
/// GCC's inline-unit-growth budget, which can leave a call per row.
class XxHash64 final {
 public:
  using SeedType = int64_t;
  using ReturnType = int64_t;

  /// Hashes a 32-bit integer value.
  FOLLY_ALWAYS_INLINE static uint64_t hashInt32(
      const int32_t input,
      uint64_t seed) {
    int64_t hash = seed + kPrime64_5 + 4L;
    hash ^= static_cast<int64_t>((input & 0xFFFFFFFFL) * kPrime64_1);
    hash = bits::rotateLeft64(hash, 23) * kPrime64_2 + kPrime64_3;
    return fmix(hash);
  }

  /// Hashes a 64-bit integer value.
  FOLLY_ALWAYS_INLINE static uint64_t hashInt64(int64_t input, uint64_t seed) {
    int64_t hash = seed + kPrime64_5 + 8L;
    hash ^= bits::rotateLeft64(input * kPrime64_2, 31) * kPrime64_1;
    hash = bits::rotateLeft64(hash, 27) * kPrime64_1 + kPrime64_4;
    return fmix(hash);
  }

  /// Hashes a float value, treating -0.0f as +0.0f.
  FOLLY_ALWAYS_INLINE static uint64_t hashFloat(float input, uint64_t seed) {
    return hashInt32(
        input == -0.f ? 0 : *reinterpret_cast<uint32_t*>(&input), seed);
  }

  /// Hashes a double value, treating -0.0 as +0.0.
  FOLLY_ALWAYS_INLINE static uint64_t hashDouble(double input, uint64_t seed) {
    return hashInt64(
        input == -0. ? 0 : *reinterpret_cast<uint64_t*>(&input), seed);
  }

  /// Hashes a byte sequence.
  static uint64_t hashBytes(const StringView& input, uint64_t seed);

  /// Hashes a long decimal value using Spark's byte representation.
  static uint64_t hashLongDecimal(int128_t input, uint64_t seed);

  /// Hashes a timestamp by its microsecond value.
  FOLLY_ALWAYS_INLINE static uint64_t hashTimestamp(
      Timestamp input,
      uint64_t seed) {
    return hashInt64(input.toMicros(), seed);
  }

 private:
  static constexpr uint64_t kPrime64_1 = 0x9E3779B185EBCA87L;
  static constexpr uint64_t kPrime64_2 = 0xC2B2AE3D27D4EB4FL;
  static constexpr uint64_t kPrime64_3 = 0x165667B19E3779F9L;
  static constexpr uint64_t kPrime64_4 = 0x85EBCA77C2B2AE63L;
  static constexpr uint64_t kPrime64_5 = 0x27D4EB2F165667C5L;

  FOLLY_ALWAYS_INLINE static uint64_t fmix(uint64_t hash) {
    hash ^= hash >> 33;
    hash *= kPrime64_2;
    hash ^= hash >> 29;
    hash *= kPrime64_3;
    hash ^= hash >> 32;
    return hash;
  }

  static uint64_t hashBytesByWords(const StringView& input, uint64_t seed);
};

} // namespace facebook::velox::functions::sparksql
