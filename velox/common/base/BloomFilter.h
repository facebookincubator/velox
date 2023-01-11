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
#include <vector>

#include <folly/Hash.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/type/StringView.h"

namespace facebook::velox {

// BloomFilter filter with groups of 64 bits, of which 4 are set. The hash
// number has 4 6 bit fields that selct the bits in the word and the
// remaining bits select the word in the filter. With 8 bits per
// expected entry, we get ~2% false positives. 'hashInput' determines
// if the value added or checked needs to be hashed. If this is false,
// we assume that the input is already a 64 bit hash number.
// case:
// InputType can be one of folly hasher support type when hashInput = false
// InputType can only be uint64_t when hashInput = true
template <class InputType = uint64_t, bool hashInput = true>
class BloomFilter {
 public:
  BloomFilter(){};
  BloomFilter(std::vector<uint64_t> bits) : bits_(bits){};

  // Prepares 'this' for use with an expected 'capacity'
  // entries. Drops any prior content.
  void reset(int32_t capacity) {
    bits_.clear();
    // 2 bytes per value.
    bits_.resize(std::max<int32_t>(4, bits::nextPowerOfTwo(capacity) / 4));
  }

  bool isSet() {
    return bits_.size() > 0;
  }

  // Adds 'value'.
  void insert(InputType value) {
    set(bits_.data(),
        bits_.size(),
        hashInput ? folly::hasher<InputType>()(value) : value);
  }

  bool mayContain(InputType value) const {
    return test(
        bits_.data(),
        bits_.size(),
        hashInput ? folly::hasher<InputType>()(value) : value);
  }

  // Combines the two bloomFilter bits_ using bitwise OR.
  void merge(BloomFilter& bloomFilter) {
    if (bits_.size() == 0) {
      bits_ = bloomFilter.bits_;
      return;
    } else if (bloomFilter.bits_.size() == 0) {
      return;
    }
    VELOX_CHECK_EQ(bits_.size(), bloomFilter.bits_.size());
    for (auto i = 0; i < bloomFilter.bits_.size(); i++) {
      bits_[i] |= bloomFilter.bits_[i];
    }
  }

  uint32_t serializedSize() {
    return 4 /* number of bits */
        + bits_.size() * 8;
  }

  void serialize(StringView& output) {
    char* outputBuffer = const_cast<char*>(output.data());
    common::OutputByteStream stream(outputBuffer);
    stream.appendOne((int32_t)bits_.size());
    for (auto bit : bits_) {
      stream.appendOne(bit);
    }
  }

  static void deserialize(const char* serialized, BloomFilter& output) {
    common::InputByteStream stream(serialized);
    auto size = stream.read<int32_t>();
    output.bits_.resize(size);
    auto bitsdata =
        reinterpret_cast<const uint64_t*>(serialized + stream.offset());
    for (auto i = 0; i < size; i++) {
      output.bits_[i] = bitsdata[i];
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

  std::vector<uint64_t> bits_;
};

} // namespace facebook::velox
