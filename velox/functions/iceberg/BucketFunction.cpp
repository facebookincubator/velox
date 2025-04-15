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

#include "velox/functions/iceberg/BucketFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::iceberg {
namespace {

class Murmur3_32HashFunction {
 public:
  static int hashBigint(uint64_t input) {
    uint32_t low = input;
    uint32_t high = input >> 32;
    auto k1 = mixK1(low);
    auto h1 = mixH1(kSeed, k1);
    k1 = mixK1(high);
    h1 = mixH1(h1, k1);
    return fmix(h1, 8);
  }

  static int32_t hashString(const char* const input, uint32_t len) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(input);
    const int32_t nblocks = len / 4;

    uint32_t h1 = kSeed;

    // Process 4-byte chunks.
    for (int32_t i = 0; i < nblocks; i++) {
      uint32_t k1 = *reinterpret_cast<const uint32_t*>(data + i * 4);
      k1 = mixK1(k1);
      h1 = mixH1(h1, k1);
    }

    // Process remaining bytes.
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
      case 3:
        k1 ^= static_cast<uint32_t>(tail[2]) << 16;
        [[fallthrough]];
      case 2:
        k1 ^= static_cast<uint32_t>(tail[1]) << 8;
        [[fallthrough]];
      case 1:
        k1 ^= static_cast<uint32_t>(tail[0]);
    }

    h1 ^= mixK1(k1);
    return fmix(h1, len);
  }

 private:
  FOLLY_ALWAYS_INLINE static uint32_t mixK1(uint32_t k1) {
    k1 *= 0xcc9e2d51;
    k1 = bits::rotateLeft(k1, 15);
    k1 *= 0x1b873593;
    return k1;
  }

  FOLLY_ALWAYS_INLINE static uint32_t mixH1(uint32_t h1, uint32_t k1) {
    h1 ^= k1;
    h1 = bits::rotateLeft(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    return h1;
  }

  FOLLY_ALWAYS_INLINE static uint32_t fmix(uint32_t h1, uint32_t length) {
    h1 ^= length;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
  }

  static constexpr int kSeed = 0;
};

FOLLY_ALWAYS_INLINE int apply(int numBuckets, int hashedValue) {
    return (hashedValue & INT_MAX) % numBuckets;
}

template <typename T>
struct BucketDecimalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(int32_t& out, const int32_t& numBuckets, const TInput& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto length = DecimalUtil::getByteArrayLength(input);
    char bytes[length];
    DecimalUtil::toByteArray(input, bytes);
    const auto hash = Murmur3_32HashFunction::hashString(bytes, length);
    out = apply(numBuckets, hash);
  }
};

template <typename T>
struct BucketFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(int32_t& out, const int32_t& numBuckets, const TInput& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto hash = Murmur3_32HashFunction::hashBigint(input);
    out = apply(numBuckets, hash);
  }

  FOLLY_ALWAYS_INLINE void call(
      int32_t& out,
      const int32_t& numBuckets,
      const arg_type<Varchar>& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto hash =
        Murmur3_32HashFunction::hashString(input.data(), input.size());
    out = apply(numBuckets, hash);
  }

  FOLLY_ALWAYS_INLINE void call(
      int32_t& out,
      const int32_t& numBuckets,
      const arg_type<Timestamp>& input) {
    const auto hash = Murmur3_32HashFunction::hashBigint(input.toMicros());
    out = apply(numBuckets, hash);
  }
};
} // namespace

void registerBucketFunctions(const std::string& prefix) {
  registerFunction<BucketFunction, int32_t, int32_t, int32_t>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, int64_t>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Varchar>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Date>({prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Timestamp>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Varbinary>(
    {prefix + "bucket"});

  registerFunction<
    BucketDecimalFunction,
    int32_t, int32_t,
    LongDecimal<P1, S1>>({prefix + "bucket"});

  registerFunction<
    BucketDecimalFunction,
    int32_t, int32_t,
    ShortDecimal<P1, S1>>({prefix + "bucket"});
}

} // namespace facebook::velox::functions::iceberg
