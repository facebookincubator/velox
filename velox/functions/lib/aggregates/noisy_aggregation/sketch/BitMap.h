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
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::functions::aggregate {

/// A C++ implementation of the Bitmap class used in SfmSketch.
/// This is a wrapper around std::vector<int8_t> which provides functionality
/// similar to Java's BitSet. We can't use existing C++ implementations of
/// bit_map using std::vector<int64_> because they are not compatible with
/// Java's BitSet.toByteArray() which truncates trailing 0s in byte size.
class BitMap {
 public:
  explicit BitMap(uint32_t length, HashStringAllocator* allocator);

  // This is for deserializing the bitmap from the intermediate varbinary (C++
  // format).
  BitMap(const char* serialized, HashStringAllocator* allocator);

  bool bitAt(uint32_t position) const;

  void setBit(uint32_t position, bool value);

  void flipBit(uint32_t position);

  uint32_t length() const {
    return length_;
  }

  // This method returns reference to the underlying vector of bytes.
  // used for deserialization.
  std::vector<int8_t, facebook::velox::StlAllocator<int8_t>>& getMutableBits() {
    return bits_;
  }

  // Count the number of bits set to 1.
  uint32_t countBits() const;

  // This is for serializing and store the bitmap in the database.
  // Strip trailing zeros in byte size and return the vector of integers.
  std::vector<int8_t, facebook::velox::StlAllocator<int8_t>>
  toCompactIntVector() const;

  size_t serializedSize() const;

  void serialize(char* output) const;

  // Deserialize BitMap from truncated serialization,
  // Simulate Java byte array (BitSet.toByteArray() format)
  static BitMap deserialize(
      const char* bytes,
      uint32_t actualLength, // this is the serialized length after truncation
      uint32_t expectedLength, // this length equal to buckets * precison
      HashStringAllocator* allocator);

  void bitwiseOr(const BitMap& other);

  template <typename RandomizationStrategy>
  void flipBit(
      uint32_t position,
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
    for (uint32_t i = 0; i < length_; ++i) {
      if (randomizationStrategy.nextBoolean(probability)) {
        flipBit(i);
      }
    }
  }

 private:
  uint32_t length_;
  std::vector<int8_t, facebook::velox::StlAllocator<int8_t>> bits_;
};

} // namespace facebook::velox::functions::aggregate
