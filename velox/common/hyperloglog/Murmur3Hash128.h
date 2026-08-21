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

namespace facebook::velox::common::hll {

/// This method implements the Murmur3Hash128::hash64 methods to match Presto
/// Java.
class Murmur3Hash128 {
 public:
  /// This method implements
  /// https://github.com/airlift/slice/blob/44896889dcef7a16a2c14800a4a392934909c2cc/src/main/java/io/airlift/slice/Murmur3Hash128.java#L152.
  ///
  /// NOTE: this reads tail bytes as signed, so for any input whose length is
  /// not a multiple of 16 and whose tail contains a byte >= 0x80 it does NOT
  /// match Presto Java. Existing sketches were built with this behaviour, so it
  /// is preserved. Use hash64JavaCompat() when cross-engine byte compatibility
  /// with Presto Java is required.
  static int64_t hash64(const void* data, int32_t length, int64_t seed);

  /// Same as hash64(), but masks each tail byte with 0xFF the way Presto Java
  /// does, so the result matches airlift's Murmur3Hash128.hash64 for every
  /// input. Used by the Java-compatible KHyperLogLog aggregate.
  static int64_t
  hash64JavaCompat(const void* data, int32_t length, int64_t seed);

  /// This method implements
  /// https://github.com/airlift/slice/blob/44896889dcef7a16a2c14800a4a392934909c2cc/src/main/java/io/airlift/slice/Murmur3Hash128.java#L248.
  static int64_t hash64ForLong(int64_t data, int64_t seed) {
    uint64_t h2 = seed ^ sizeof(int64_t);
    uint64_t h1 = h2 + (h2 ^ (bits::rotateLeft64(data * C1, 31) * C2));
    return static_cast<int64_t>(mix64(h1) + mix64(h1 + h2));
  }

  static void
  hash(const void* key, const int32_t len, const uint32_t seed, void* out);

 private:
  // Shared body of hash64() and hash64JavaCompat(). When JavaCompat is true the
  // trailing bytes are read as unsigned, matching Presto Java; otherwise they
  // are read through a signed char, preserving the original Velox behaviour
  // that already-persisted sketches depend on.
  template <bool JavaCompat>
  static int64_t hash64Impl(const void* data, int32_t length, int64_t seed);

  static constexpr uint64_t C1 = 0x87c37b91114253d5L;
  static constexpr uint64_t C2 = 0x4cf5ad432745937fL;

  static uint64_t mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53L;
    k ^= k >> 33;

    return k;
  }
};

} // namespace facebook::velox::common::hll
