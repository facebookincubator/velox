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
#include "velox/type/StringView.h"

namespace facebook::velox::connectors::hive::iceberg {
constexpr uint32_t DEFAULT_SEED = 0;
constexpr uint32_t C1 = 0xCC9E2D51;
constexpr uint32_t C2 = 0x1B873593;

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))

#define FMIX32(h)    \
  (h) ^= (h) >> 16;  \
  (h) *= 0x85EBCA6B; \
  (h) ^= (h) >> 13;  \
  (h) *= 0xC2B2AE35; \
  (h) ^= (h) >> 16;

class Murmur3_32 final {
 public:
  static uint32_t hash(int32_t value, uint32_t seed = DEFAULT_SEED);

  static uint32_t hash(uint32_t value, uint32_t seed = DEFAULT_SEED);

  static uint32_t hash(uint64_t value, uint32_t seed = DEFAULT_SEED);

  static uint32_t hash(int64_t value, uint32_t seed = DEFAULT_SEED);

  static uint32_t hash(folly::int128_t value, uint32_t seed = DEFAULT_SEED);

  static uint32_t hash(
      const facebook::velox::StringView& value,
      uint32_t seed = DEFAULT_SEED);

  static uint32_t
  hash(const uint8_t* data, size_t length, uint32_t seed = DEFAULT_SEED);

  static uint32_t hashDecimal(int64_t value, uint32_t seed = DEFAULT_SEED);

 private:
  static uint32_t mixK1(uint32_t k1) {
    k1 *= C1;
    k1 = ROTL32(k1, 15);
    k1 *= C2;
    return k1;
  }

  static uint32_t mixH1(uint32_t h1, uint32_t k1) {
    h1 ^= k1;
    h1 = ROTL32(h1, 13);
    h1 = h1 * 5 + 0xE6546B64;
    return h1;
  }

  static std::shared_ptr<uint8_t> intToMinimalBytes(
      int64_t value,
      int32_t* length);
};
} // namespace facebook::velox::connectors::hive::iceberg
