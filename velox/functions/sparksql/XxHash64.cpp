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

#include "velox/functions/sparksql/XxHash64.h"

#include "velox/common/base/BitUtil.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

uint64_t XxHash64::hashInt32(const int32_t input, uint64_t seed) {
  int64_t hash = seed + kPrime64_5 + 4L;
  hash ^= static_cast<int64_t>((input & 0xFFFFFFFFL) * kPrime64_1);
  hash = bits::rotateLeft64(hash, 23) * kPrime64_2 + kPrime64_3;
  return fmix(hash);
}

uint64_t XxHash64::hashInt64(int64_t input, uint64_t seed) {
  int64_t hash = seed + kPrime64_5 + 8L;
  hash ^= bits::rotateLeft64(input * kPrime64_2, 31) * kPrime64_1;
  hash = bits::rotateLeft64(hash, 27) * kPrime64_1 + kPrime64_4;
  return fmix(hash);
}

uint64_t XxHash64::hashLongDecimal(int128_t input, uint64_t seed) {
  char out[sizeof(int128_t)];
  int32_t length = DecimalUtil::toByteArray(input, out);
  return hashBytes(StringView(out, length), seed);
}

uint64_t XxHash64::fmix(uint64_t hash) {
  hash ^= hash >> 33;
  hash *= kPrime64_2;
  hash ^= hash >> 29;
  hash *= kPrime64_3;
  hash ^= hash >> 32;
  return hash;
}

uint64_t XxHash64::hashBytesByWords(const StringView& input, uint64_t seed) {
  const char* i = input.data();
  const char* const end = input.data() + input.size();
  uint32_t length = input.size();
  uint64_t hash;
  if (length >= 32) {
    uint64_t v1 = seed + kPrime64_1 + kPrime64_2;
    uint64_t v2 = seed + kPrime64_2;
    uint64_t v3 = seed;
    uint64_t v4 = seed - kPrime64_1;
    for (; i <= end - 32; i += 32) {
      v1 = bits::rotateLeft64(
               v1 + (*reinterpret_cast<const uint64_t*>(i) * kPrime64_2), 31) *
          kPrime64_1;
      v2 = bits::rotateLeft64(
               v2 + (*reinterpret_cast<const uint64_t*>(i + 8) * kPrime64_2),
               31) *
          kPrime64_1;
      v3 = bits::rotateLeft64(
               v3 + (*reinterpret_cast<const uint64_t*>(i + 16) * kPrime64_2),
               31) *
          kPrime64_1;
      v4 = bits::rotateLeft64(
               v4 + (*reinterpret_cast<const uint64_t*>(i + 24) * kPrime64_2),
               31) *
          kPrime64_1;
    }
    hash = bits::rotateLeft64(v1, 1) + bits::rotateLeft64(v2, 7) +
        bits::rotateLeft64(v3, 12) + bits::rotateLeft64(v4, 18);
    v1 *= kPrime64_2;
    v1 = bits::rotateLeft64(v1, 31);
    v1 *= kPrime64_1;
    hash ^= v1;
    hash = hash * kPrime64_1 + kPrime64_4;

    v2 *= kPrime64_2;
    v2 = bits::rotateLeft64(v2, 31);
    v2 *= kPrime64_1;
    hash ^= v2;
    hash = hash * kPrime64_1 + kPrime64_4;

    v3 *= kPrime64_2;
    v3 = bits::rotateLeft64(v3, 31);
    v3 *= kPrime64_1;
    hash ^= v3;
    hash = hash * kPrime64_1 + kPrime64_4;

    v4 *= kPrime64_2;
    v4 = bits::rotateLeft64(v4, 31);
    v4 *= kPrime64_1;
    hash ^= v4;
    hash = hash * kPrime64_1 + kPrime64_4;
  } else {
    hash = seed + kPrime64_5;
  }

  hash += length;

  for (; i <= end - 8; i += 8) {
    hash ^= bits::rotateLeft64(
                *reinterpret_cast<const uint64_t*>(i) * kPrime64_2, 31) *
        kPrime64_1;
    hash = bits::rotateLeft64(hash, 27) * kPrime64_1 + kPrime64_4;
  }
  return hash;
}

uint64_t XxHash64::hashBytes(const StringView& input, uint64_t seed) {
  const char* i = input.data();
  const char* const end = input.data() + input.size();

  uint64_t hash = hashBytesByWords(input, seed);
  uint32_t length = input.size();
  auto offset = i + (length & -8);
  if (offset + 4L <= end) {
    hash ^=
        (*reinterpret_cast<const uint64_t*>(offset) & 0xFFFFFFFFL) * kPrime64_1;
    hash = bits::rotateLeft64(hash, 23) * kPrime64_2 + kPrime64_3;
    offset += 4L;
  }

  while (offset < end) {
    hash ^= (*reinterpret_cast<const uint64_t*>(offset) & 0xFFL) * kPrime64_5;
    hash = bits::rotateLeft64(hash, 11) * kPrime64_1;
    offset++;
  }
  return fmix(hash);
}

} // namespace facebook::velox::functions::sparksql
