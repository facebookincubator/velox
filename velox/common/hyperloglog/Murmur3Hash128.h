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
  static int64_t hash64(const void* data, int32_t length, int64_t seed);

  /// This method implements
  /// https://github.com/airlift/slice/blob/44896889dcef7a16a2c14800a4a392934909c2cc/src/main/java/io/airlift/slice/Murmur3Hash128.java#L248.
  static int64_t hash64ForLong(int64_t data, int64_t seed) {
    int64_t h2 = seed ^ sizeof(int64_t);
    int64_t h1 = h2 + (h2 ^ (bits::rotateLeft64(data * C1, 31) * C2));
    return static_cast<uint64_t>(mix64(h1)) + mix64(h1 + h2);
  }

 private:
  static constexpr uint64_t C1 = 0x87c37b91114253d5L;
  static constexpr uint64_t C2 = 0x4cf5ad432745937fL;

  static int64_t mix64(int64_t k) {
    uint64_t unsignedK = static_cast<uint64_t>(k);
    unsignedK ^= unsignedK >> 33;
    unsignedK *= 0xff51afd7ed558ccdL;
    unsignedK ^= unsignedK >> 33;
    unsignedK *= 0xc4ceb9fe1a85ec53L;
    unsignedK ^= unsignedK >> 33;

    k = static_cast<int64_t>(unsignedK);
    return k;
  }
};

} // namespace facebook::velox::common::hll
