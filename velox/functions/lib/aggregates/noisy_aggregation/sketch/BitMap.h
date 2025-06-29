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
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"

namespace facebook::velox::functions::aggregate {

/// A C++ implementation of the Bitmap class used in SfmSketch.
/// This is a wrapper around std::vector<int64_t> which provides functionality
/// similar to Java's BitSet.
class BitMap {
 public:
  explicit BitMap(uint64_t length) : length_(length) {
    bits_.resize(velox::bits::nbytes(length_));
  }

  BitMap(const BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't copy from null bitmap.");
    length_ = other.length_;
    bits_.resize(velox::bits::nbytes(length_));
    std::memcpy(bits_.data(), other.bits_.data(), bits_.size());
  }

  BitMap& operator=(const BitMap& other) {
    if (this == &other) {
      return *this; // Self-assignment check
    }
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't copy from null bitmap.");
    length_ = other.length_;
    bits_.resize(velox::bits::nbytes(length_));
    std::memcpy(bits_.data(), other.bits_.data(), bits_.size());
    return *this;
  }

  // Explicitly default the remaining special member functions
  ~BitMap() = default;
  BitMap(BitMap&&) = default;
  BitMap& operator=(BitMap&&) = default;

  BitMap(uint64_t length, const std::vector<int8_t>& bits) {
    VELOX_CHECK_EQ(
        length, bits.size() * 8, "Can't create bitmap of different length.");
    length_ = length;
    bits_ = bits;
  }

  BitMap(uint64_t length, BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't copy from null bitmap.");
    VELOX_CHECK_EQ(
        length, other.length_, "Can't copy bitmaps of different lengths.");
    length_ = length;
    bits_.resize(velox::bits::nbytes(length_));
    std::memcpy(bits_.data(), other.bits_.data(), bits_.size());
  }

  bool getBit(uint64_t position) const {
    return bits::isBitSet(bits_.data(), position);
  }

  void setBit(uint64_t position, bool value) {
    bits::setBit(bits_.data(), position, value);
  }

  void flipBit(uint64_t position) {
    auto bitsAs8Bit = reinterpret_cast<uint8_t*>(bits_.data());
    bitsAs8Bit[position / 8] ^= 1 << (position % 8);
  }

  uint64_t length() const {
    return length_;
  }

  // Count the number of bits set to 1.
  uint64_t bitCount() const {
    uint64_t count = 0;
    const uint64_t* wordBits = reinterpret_cast<const uint64_t*>(bits_.data());

    // Calculate number of full 64-bit words we can safely access
    uint64_t numBytesAvailable = bits_.size();
    uint64_t numFullWords = numBytesAvailable / 8; // 8 bytes per uint64_t

    // Count full 64-bit words
    for (uint64_t i = 0; i < numFullWords; ++i) {
      count += static_cast<uint64_t>(__builtin_popcountll(wordBits[i]));
    }

    // Handle remaining bytes that don't form a complete 64-bit word
    uint64_t remainingBytes = numBytesAvailable % 8;
    if (remainingBytes > 0) {
      uint64_t partialWord = 0;
      const uint8_t* bytePtr =
          reinterpret_cast<const uint8_t*>(bits_.data()) + (numFullWords * 8);
      for (uint64_t i = 0; i < remainingBytes; ++i) {
        partialWord |= (static_cast<uint64_t>(bytePtr[i]) << (i * 8));
      }

      // Only count bits up to the actual length
      uint64_t bitsInPartialWord =
          std::min(remainingBytes * 8, length_ - (numFullWords * 64));
      if (bitsInPartialWord < 64) {
        uint64_t mask = (1ULL << bitsInPartialWord) - 1;
        partialWord &= mask;
      }
      count += static_cast<uint64_t>(__builtin_popcountll(partialWord));
    }

    return count;
  }

  // This is for serializing and store the bitmap in the database.
  // Strip any trailing zeros and return the vector of integers.
  std::vector<int8_t> toCompactIntVector() const {
    if (bits_.empty()) {
      return {};
    }

    size_t lastNonZeroIndex = bits_.size() - 1;
    while (lastNonZeroIndex >= 0 && bits_[lastNonZeroIndex] == 0) {
      --lastNonZeroIndex;
    }

    if (lastNonZeroIndex < 0) {
      return {};
    }
    return std::vector<int8_t>(
        bits_.begin(),
        bits_.begin() + static_cast<int64_t>(lastNonZeroIndex) + 1);
  }

  size_t serializedSize() const {
    return sizeof(length_) + sizeof(int8_t) * bits_.size();
  }

  void serialize(char* output) const {
    common::OutputByteStream stream(output);
    stream.appendOne(length_);
    stream.append(
        reinterpret_cast<const char*>(bits_.data()),
        static_cast<int32_t>(bits_.size()));
  }

  static BitMap deserialize(const char* serialized) {
    common::InputByteStream stream(serialized);
    uint64_t length = stream.read<uint64_t>();
    std::vector<int8_t> bits;
    bits.resize(velox::bits::nbytes(length));
    stream.copyTo(bits.data(), static_cast<int>(bits.size()));
    return BitMap(length, bits);
  }

  void bitwiseOr(const BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't combine with null bitmap.");
    VELOX_CHECK_EQ(
        length_, other.length_, "Can't OR two bitmaps of different lengths.");
    for (size_t i = 0; i < bits_.size(); ++i) {
      bits_[i] = static_cast<int8_t>(bits_[i] | other.bits_[i]);
    }
  }

  template <typename RandomizationStrategy>
  void flipBit(
      uint64_t position,
      double probability,
      RandomizationStrategy randomizationStrategy) {
    if (randomizationStrategy.nextBoolean(probability)) {
      flipBit(position);
    }
  }

  template <typename RandomizationStrategy>
  void flipAll(
      double probability,
      RandomizationStrategy randomizationStrategy) {
    for (uint64_t i = 0; i < length_; ++i) {
      if (randomizationStrategy.nextBoolean(probability)) {
        flipBit(i);
      }
    }
  }

 private:
  uint64_t length_;
  std::vector<int8_t> bits_;
};

} // namespace facebook::velox::functions::aggregate
