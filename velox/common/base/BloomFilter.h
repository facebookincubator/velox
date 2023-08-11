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

#include <xsimd/xsimd.hpp>
#include <cstdint>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"

namespace facebook::velox {
// BloomFilter filter with groups of 64 bits, of which 4 are set. The hash
// number has 4 6 bit fields that selct the bits in the word and the
// remaining bits select the word in the filter. With 8 bits per
// expected entry, we get ~2% false positives. 'hashInput' determines
// if the value added or checked needs to be hashed. If this is false,
// we assume that the input is already a 64 bit hash number.
template <typename Allocator = std::allocator<uint64_t>>
class BloomFilter {
 public:
  explicit BloomFilter() : bits_{Allocator()} {}
  explicit BloomFilter(const Allocator& allocator) : bits_{allocator} {}

  // Prepares 'this' for use with an expected 'capacity'
  // entries. Drops any prior content.
  void reset(int32_t capacity) {
    bits_.clear();
    // 2 bytes per value.
    bits_.resize(std::max<int32_t>(4, bits::nextPowerOfTwo(capacity) / 4));
  }

  bool isSet() const {
    return bits_.size() > 0;
  }

  // Adds 'value'.
  // Input is hashed uint64_t value, optional hash function is
  // folly::hasher<InputType>()(value).
  void insert(uint64_t value) {
    set(bits_.data(), bits_.size(), value);
  }

  // Input is hashed uint64_t value, optional hash function is
  // folly::hasher<InputType>()(value).
  bool mayContain(uint64_t value) const {
    return test(bits_.data(), bits_.size(), value);
  }

  void mayContainForBatch(const uint64_t* value, bool* result, int32_t size)
      const {
    testForBatch(bits_.data(), bits_.size(), value, result, size);
  }

  void merge(const char* serialized) {
    common::InputByteStream stream(serialized);
    auto version = stream.read<int8_t>();
    VELOX_USER_CHECK_EQ(kBloomFilterV1, version);
    auto size = stream.read<int32_t>();
    bits_.resize(size);
    auto bitsdata =
        reinterpret_cast<const uint64_t*>(serialized + stream.offset());
    if (bits_.size() == 0) {
      for (auto i = 0; i < size; i++) {
        bits_[i] = bitsdata[i];
      }
      return;
    } else if (size == 0) {
      return;
    }
    VELOX_DCHECK_EQ(bits_.size(), size);
    bits::orBits(bits_.data(), bitsdata, 0, 64 * size);
  }

  uint32_t serializedSize() const {
    return 1 /* version */
        + 4 /* number of bits */
        + bits_.size() * 8;
  }

  void serialize(char* output) const {
    common::OutputByteStream stream(output);
    stream.appendOne(kBloomFilterV1);
    stream.appendOne((int32_t)bits_.size());
    for (auto bit : bits_) {
      stream.appendOne(bit);
    }
  }

 private:
  // We use 4 independent hash functions by taking 24 bits of
  // the hash code and breaking these up into 4 groups of 6 bits. Each group
  // represents a number between 0 and 63 (2^6-1) and maps to one bit in a
  // 64-bit number. We combine these to get a 64-bit number with up to 4 bits
  // set.
  inline static uint64_t bloomMask(uint64_t hashCode) {
    return (1L << (hashCode & 63)) | (1L << ((hashCode >> 6) & 63)) |
        (1L << ((hashCode >> 12) & 63)) | (1L << ((hashCode >> 18) & 63));
  }

  // Skip 24 bits used for bloomMask and use the next N bits of the hash code
  // as index. N = log2(bloomSize). bloomSize must be a power of 2.
  inline static uint32_t bloomIndex(uint32_t bloomSize, uint64_t hashCode) {
    return ((hashCode >> 24) & (bloomSize - 1));
  }

  inline static void
  set(uint64_t* FOLLY_NONNULL bloom, int32_t bloomSize, uint64_t hashCode) {
    auto mask = bloomMask(hashCode);
    auto index = bloomIndex(bloomSize, hashCode);
    bloom[index] |= mask;
  }

  inline static bool test(
      const uint64_t* FOLLY_NONNULL bloom,
      int32_t bloomSize,
      uint64_t hashCode) {
    auto mask = bloomMask(hashCode);
    auto index = bloomIndex(bloomSize, hashCode);
    return mask == (bloom[index] & mask);
  }

  inline static void testForBatch(
      const uint64_t* FOLLY_NONNULL bloom,
      int32_t bloomSize,
      const uint64_t* hashCode,
      bool* result,
      int32_t size) {
    VELOX_DCHECK_EQ(size, 64);
#if XSIMD_WITH_AVX2
    int64_t buffer[4];
    int j = 0;
    __m256i const1 = _mm256_set1_epi64x(1);
    __m256i const63 = _mm256_set1_epi64x(63);
    __m256i maskSize = _mm256_set1_epi64x(bloomSize - 1);
    for (int i = 0; i < 16; ++i) {
      __m256i hashVec = _mm256_loadu_si256((__m256i*)hashCode);
      __m256i mask1 =
          _mm256_sllv_epi64(const1, _mm256_and_si256(hashVec, const63));
      __m256i mask2 = _mm256_or_si256(
          mask1,
          _mm256_sllv_epi64(
              const1,
              _mm256_and_si256(_mm256_srli_epi64(hashVec, 6), const63)));
      __m256i mask3 = _mm256_or_si256(
          mask2,
          _mm256_sllv_epi64(
              const1,
              _mm256_and_si256(_mm256_srli_epi64(hashVec, 12), const63)));
      __m256i mask = _mm256_or_si256(
          mask3,
          _mm256_sllv_epi64(
              const1,
              _mm256_and_si256(_mm256_srli_epi64(hashVec, 18), const63)));
      __m256i index =
          _mm256_and_si256(_mm256_srli_epi64(hashVec, 24), maskSize);
      __m256i content = _mm256_i64gather_epi64(
          reinterpret_cast<const long long*>(bloom), index, 8);
      __m256i res = _mm256_cmpeq_epi64(_mm256_and_si256(content, mask), mask);
      _mm256_storeu_si256((__m256i*)buffer, res);
      result[j] = buffer[0] != 0;
      result[j + 1] = buffer[1] != 0;
      result[j + 2] = buffer[2] != 0;
      result[j + 3] = buffer[3] != 0;
      hashCode += 4;
      j += 4;
    }
#else
    for (int i = 0; i < 64; ++i) {
      result[i] = test(bloom, bloomSize, hashCode[i]);
    }
#endif // XSIMD_WITH_AVX2
  }

  const int8_t kBloomFilterV1 = 1;
  std::vector<uint64_t, Allocator> bits_;
};

} // namespace facebook::velox
