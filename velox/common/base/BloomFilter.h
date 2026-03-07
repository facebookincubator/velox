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

#include <memory>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"

namespace facebook::velox {

/// BloomFilterBase provides static methods to test and set bits in a Bloom
/// filter.
class BloomFilterBase {
 protected:
  static bool test(const uint64_t* bloom, int32_t bloomSize, uint64_t hashCode);
  static void set(uint64_t* bloom, int32_t bloomSize, uint64_t hashCode);
  static constexpr int8_t kBloomFilterV1 = 1;
};

/// BloomFilterView is a non-owning view of a serialized Bloom filter.
/// It reads and validates the version and size from the serialized data.
class BloomFilterView : BloomFilterBase {
 public:
  /// Constructs a view from serialized Bloom filter data.
  /// Does not take ownership of the data.
  explicit BloomFilterView(const char* serializedBloom);

  /// Tests if a value may be contained in the Bloom filter.
  /// Input is hashed uint64_t value, optional hash function is
  /// folly::hasher<InputType>()(value).
  bool mayContain(uint64_t value) const {
    return test(bitsData_, size_, value);
  }

  int32_t size() const {
    return size_;
  }

  const uint64_t* bitsData() const {
    return bitsData_;
  }

 private:
  int8_t version_;
  int32_t size_;
  const uint64_t* bitsData_;
};

/// BloomFilter filter with groups of 64 bits, of which 4 are set. The hash
/// number has 4 6 bit fields that select the bits in the word and the
/// remaining bits select the word in the filter. With 8 bits per
/// expected entry, we get ~2% false positives. 'hashInput' determines
/// if the value added or checked needs to be hashed. If this is false,
/// we assume that the input is already a 64-bit hash number.
template <typename Allocator = std::allocator<uint64_t>>
class BloomFilter : BloomFilterBase {
 public:
  explicit BloomFilter() : bits_{Allocator()} {}
  explicit BloomFilter(const Allocator& allocator) : bits_{allocator} {}

  /// Prepares 'this' for use with an expected 'capacity'
  /// entries. Drops any prior content.
  void reset(int32_t capacity) {
    bits_.clear();
    // 2 bytes per value.
    bits_.resize(std::max<int32_t>(4, bits::nextPowerOfTwo(capacity) / 4));
  }

  /// Returns true if 'this' has been initialized with merge() or insert().
  bool isSet() const {
    return bits_.size() > 0;
  }

  /// Adds 'value'.
  /// Input is hashed uint64_t value, optional hash function is
  /// folly::hasher<InputType>()(value).
  void insert(uint64_t value) {
    set(bits_.data(), bits_.size(), value);
  }

  /// Input is hashed uint64_t value, optional hash function is
  /// folly::hasher<InputType>()(value).
  bool mayContain(uint64_t value) const {
    return test(bits_.data(), bits_.size(), value);
  }

  /// Tests an input value directly on a serialized Bloom filter.
  /// For the implementation V1, this API involves no copy of the bloom
  /// filter data.
  static bool mayContain(const char* serializedBloom, uint64_t value) {
    const BloomFilterView view(serializedBloom);
    return view.mayContain(value);
  }

  /// Merges another serialized Bloom filter into 'this'. The other Bloom
  /// filter must have the same size as 'this' or 'this' must be empty.
  void merge(const char* otherSerialized) {
    const BloomFilterView otherView(otherSerialized);
    auto otherSize = otherView.size();
    if (otherSize == 0) {
      VELOX_DCHECK_EQ(bits_.size(), 0);
      return;
    }
    auto otherBitsData = otherView.bitsData();
    if (bits_.empty()) {
      bits_.assign(otherBitsData, otherBitsData + otherSize);
    } else {
      VELOX_DCHECK_EQ(bits_.size(), otherSize);
      bits::orBits(bits_.data(), otherBitsData, 0, 64 * otherSize);
    }
  }

  uint32_t serializedSize() const {
    return 1 /* version */
        + 4 /* number of bits */
        + bits_.size() * 8;
  }

  void serialize(char* output) const {
    common::OutputByteStream stream(output);
    stream.appendOne(kBloomFilterV1);
    stream.appendOne(static_cast<int32_t>(bits_.size()));
    for (auto bit : bits_) {
      stream.appendOne(bit);
    }
  }

  /// Computes m (total bits of Bloom filter) which is expected to achieve,
  /// for the specified expected insertions, the required false positive
  /// probability.
  ///
  /// See
  /// http://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives
  /// for the formula.
  ///
  /// @param n expected insertions (must be positive).
  /// @param p false positive rate (must be 0 < p < 1).
  static int64_t optimalNumOfBits(int64_t n, double p) {
    return static_cast<int64_t>(
        -n * std::log(p) / (std::log(2.0) * std::log(2.0)));
  }

  /// Computes m (total bits of Bloom filter) which is expected to achieve.
  /// The smaller the expectedNumItems, the smaller the fpp.
  ///
  /// @param expectedNumItems expected number of items to insert.
  /// @param maxNumItems maximum number of items.
  static int64_t optimalNumOfBits(
      int64_t expectedNumItems,
      int64_t maxNumItems) {
    double ratio = static_cast<double>(expectedNumItems) / maxNumItems;
    double fpp = kDefaultFpp * std::min(ratio, 1.0);
    return optimalNumOfBits(expectedNumItems, fpp);
  }

 private:
  static constexpr double kDefaultFpp = 0.03;

  std::vector<uint64_t, Allocator> bits_;
};

} // namespace facebook::velox
