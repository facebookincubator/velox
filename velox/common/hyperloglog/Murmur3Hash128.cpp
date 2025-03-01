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

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::hll {

namespace {
const uint64_t C1 = 0x87c37b91114253d5L;
const uint64_t C2 = 0x4cf5ad432745937fL;
} // namespace

int64_t getLong(const void* data, int32_t offset) {
  return *(int64_t*)((char*)data + offset);
}

char getByte(const void* data, int32_t offset) {
  return *((char*)data + offset);
}

int64_t rotateLeft(int64_t n, int distance) {
  distance %= 64;
  return (n << distance) |
      (n >> (64 - distance) & (((uint64_t)1 << distance) - 1));
}

int64_t mix64(int64_t k) {
  uint64_t unsignedK = (uint64_t)k;
  unsignedK ^= unsignedK >> 33;
  unsignedK *= 0xff51afd7ed558ccdL;
  unsignedK ^= unsignedK >> 33;
  unsignedK *= 0xc4ceb9fe1a85ec53L;
  unsignedK ^= unsignedK >> 33;

  k = (int64_t)unsignedK;
  return k;
}

int64_t murmur3Hash64ForLong(int64_t data, int64_t seed) {
  int64_t h2 = seed ^ sizeof(int64_t);
  int64_t h1 = h2 + (h2 ^ (rotateLeft(data * C1, 31) * C2));
  return (uint64_t)mix64(h1) + mix64(h1 + h2);
}

int64_t murmur3Hash64(const void* data, int32_t length, int64_t seed) {
  VELOX_CHECK_NOT_NULL(data);
  const int32_t fastLimit = length - (2 * sizeof(int64_t)) + 1;

  int64_t h1 = seed;
  int64_t h2 = seed;

  int32_t current = 0;
  while (current < fastLimit) {
    VELOX_DCHECK_LE(current + 2 * sizeof(int64_t), length);
    int64_t k1 = getLong(data, current);
    current += sizeof(int64_t);

    int64_t k2 = getLong(data, current);
    current += sizeof(int64_t);

    k1 = k1 * C1;
    k1 = rotateLeft(k1, 31);
    k1 = k1 * C2;
    h1 ^= k1;

    h1 = rotateLeft(h1, 27);
    h1 = (uint64_t)h1 + h2;
    h1 = (uint64_t)h1 * 5 + 0x52dce729L;

    k2 = k2 * C2;
    k2 = rotateLeft(k2, 33);
    k2 = k2 * C1;
    h2 ^= k2;

    h2 = rotateLeft(h2, 31);
    h2 = (uint64_t)h2 + h1;
    h2 = (uint64_t)h2 * 5 + 0x38495ab5L;
  }

  int64_t k1 = 0;
  int64_t k2 = 0;

  VELOX_DCHECK_LE(current + (length & 15), length);
  switch (length & 15) {
    case 15:
      k2 ^= ((int64_t)getByte(data, current + 14)) << 48;
      [[fallthrough]];
    case 14:
      k2 ^= ((int64_t)getByte(data, current + 13)) << 40;
      [[fallthrough]];
    case 13:
      k2 ^= ((int64_t)getByte(data, current + 12)) << 32;
      [[fallthrough]];
    case 12:
      k2 ^= ((int64_t)getByte(data, current + 11)) << 24;
      [[fallthrough]];
    case 11:
      k2 ^= ((int64_t)getByte(data, current + 10)) << 16;
      [[fallthrough]];
    case 10:
      k2 ^= ((int64_t)getByte(data, current + 9)) << 8;
      [[fallthrough]];
    case 9:
      k2 ^= ((int64_t)getByte(data, current + 8)) << 0;

      k2 = k2 * C2;
      k2 = rotateLeft(k2, 33);
      k2 = k2 * C1;
      h2 ^= k2;
      [[fallthrough]];

    case 8:
      k1 ^= ((int64_t)getByte(data, current + 7)) << 56;
      [[fallthrough]];
    case 7:
      k1 ^= ((int64_t)getByte(data, current + 6)) << 48;
      [[fallthrough]];
    case 6:
      k1 ^= ((int64_t)getByte(data, current + 5)) << 40;
      [[fallthrough]];
    case 5:
      k1 ^= ((int64_t)getByte(data, current + 4)) << 32;
      [[fallthrough]];
    case 4:
      k1 ^= ((int64_t)getByte(data, current + 3)) << 24;
      [[fallthrough]];
    case 3:
      k1 ^= ((int64_t)getByte(data, current + 2)) << 16;
      [[fallthrough]];
    case 2:
      k1 ^= ((int64_t)getByte(data, current + 1)) << 8;
      [[fallthrough]];
    case 1:
      k1 ^= ((int64_t)getByte(data, current + 0)) << 0;

      k1 = k1 * C1;
      k1 = rotateLeft(k1, 31);
      k1 = k1 * C2;
      h1 ^= k1;
      break;
    default:
      break;
  }

  h1 ^= length;
  h2 ^= length;

  h1 = (uint64_t)h1 + h2;
  h2 = (uint64_t)h2 + h1;

  h1 = mix64(h1);
  h2 = mix64(h2);

  return (uint64_t)h1 + h2;
}

} // namespace facebook::velox::common::hll
