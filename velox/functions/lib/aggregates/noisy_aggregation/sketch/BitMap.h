// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <vector>
#include "velox/common/base/BitUtil.h"

namespace facebook::velox::function::aggregate {

/// A C++ implementation of the Bitmap class used in SfmSketch.
/// This is a wrapper around std::vector<int64_t> which provides functionality
/// similar to Java's BitSet.
class BitMap {
 public:
  explicit BitMap(int64_t length) : length_(length) {
    bits_.resize(velox::bits::nwords(length_));
  }

  BitMap(int64_t length, const std::vector<int64_t>& bits) {
    VELOX_CHECK_EQ(
        length, bits.size() * 64, "Can't create bitmap of different length.");
    length_ = length;
    bits_ = bits;
  }

  BitMap(int64_t length, BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't copy from null bitmap.");
    VELOX_CHECK_EQ(
        length, other.length_, "Can't copy bitmaps of different lengths.");
    length_ = length;
    bits_.resize(velox::bits::nwords(length_));
    bits::copyBits(
        reinterpret_cast<const uint64_t*>(other.bits_.data()),
        0,
        reinterpret_cast<uint64_t*>(bits_.data()),
        0,
        length_);
  }

  bool getBit(int64_t position) const {
    return bits::isBitSet(bits_.data(), position);
  }

  void setBit(int64_t position, bool value) {
    bits::setBit(bits_.data(), position, value);
  }

  void flipBit(int64_t position) {
    auto bitsAs8Bit = reinterpret_cast<uint8_t*>(bits_.data());
    bitsAs8Bit[position / 8] ^= 1 << (position % 8);
  }

  int64_t length() const {
    return length_;
  }

  // Count the number of bits set to 1.
  int64_t cardinality() const {
    return bits::countBits(
        reinterpret_cast<const uint64_t*>(bits_.data()), 0, length_);
  }

  // This is for serializing and store the bitmap in the database.
  // Strip any trailing zeros and return the vector of integers.
  std::vector<int64_t> toIntVector() const {
    if (bits_.empty()) {
      return {};
    }

    int64_t lastNonZeroIndex = bits_.size() - 1;
    while (lastNonZeroIndex >= 0 && bits_[lastNonZeroIndex] == 0) {
      --lastNonZeroIndex;
    }

    if (lastNonZeroIndex < 0) {
      return {};
    }
    return std::vector<int64_t>(
        bits_.begin(), bits_.begin() + lastNonZeroIndex + 1);
  }

  void bitwiseOr(const BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't combine with null bitmap.");
    VELOX_CHECK_EQ(
        length_, other.length_, "Can't OR two bitmaps of different lengths.");
    bits::orBits(
        reinterpret_cast<uint64_t*>(bits_.data()),
        reinterpret_cast<const uint64_t*>(other.bits_.data()),
        0,
        length_);
  }

  void bitwiseXor(const BitMap& other) {
    VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't combine with null bitmap.");
    VELOX_CHECK_EQ(
        length_, other.length_, "Can't XOR two bitmaps of different lengths.");
    uint64_t* target = reinterpret_cast<uint64_t*>(bits_.data());
    const uint64_t* left = reinterpret_cast<const uint64_t*>(bits_.data());
    const uint64_t* right =
        reinterpret_cast<const uint64_t*>(other.bits_.data());
    for (int32_t idx = 0; idx < length_ / 64; ++idx) {
      target[idx] = left[idx] ^ right[idx];
    }
  }

  template <typename RandomizationStrategy>
  void flipBit(
      int position,
      double probability,
      RandomizationStrategy randomizationStrategy) {
    if (randomizationStrategy.nextBoolean(probability)) {
      flipBit(position);
    }
  }

  template <typename RandomizationStrategy>
  void flipAll(
      double probability,
      RandomizationStrategy& randomizationStrategy) {
    for (int64_t i = 0; i < length_; ++i) {
      if (randomizationStrategy.nextBoolean(probability)) {
        flipBit(i);
      }
    }
  }

 private:
  int64_t length_;
  std::vector<int64_t> bits_;
};

} // namespace facebook::velox::function::aggregate
