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

#include "velox/type/HugeInt.h"
#include "velox/type/StringView.h"

namespace facebook::velox::connector::hive::iceberg {
constexpr uint32_t kDefaultSeed = 0;
constexpr uint32_t kC1 = 0xCC9E2D51;
constexpr uint32_t kC2 = 0x1B873593;

class Murmur3Hash32 final {
 public:
  static int32_t hash(uint64_t value);

  static int32_t hash(const StringView& value);

  static int32_t hash(const char* const data, size_t length);

  static int32_t hashDecimal(int128_t value);

 private:
  FOLLY_ALWAYS_INLINE static uint32_t mixK1(uint32_t k1) {
    k1 *= kC1;
    k1 = ((k1) << (15)) | ((k1) >> (32 - (15)));
    k1 *= kC2;
    return k1;
  }

  FOLLY_ALWAYS_INLINE static uint32_t mixH1(uint32_t h1, uint32_t k1) {
    h1 ^= k1;
    h1 = ((h1) << (13)) | ((h1) >> (32 - (13)));
    h1 = h1 * 5 + 0xE6546B64;
    return h1;
  }

  FOLLY_ALWAYS_INLINE static uint32_t fmix32(uint32_t h1, size_t length) {
    h1 ^= length;
    h1 ^= h1 >> 16;
    h1 *= 0x85EBCA6B;
    h1 ^= h1 >> 13;
    h1 *= 0xC2B2AE35;
    h1 ^= h1 >> 16;
    return h1;
  }
};

} // namespace facebook::velox::connector::hive::iceberg
