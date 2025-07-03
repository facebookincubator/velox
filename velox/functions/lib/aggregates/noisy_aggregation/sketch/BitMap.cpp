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

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/BitMap.h"
#include <cstdint>
#include <vector>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::functions::aggregate {

using Allocator = facebook::velox::StlAllocator<int8_t>;

BitMap::BitMap(uint32_t length, HashStringAllocator* allocator)
    : length_(length), bits_(Allocator(allocator)) {
  bits_.resize(velox::bits::nbytes(length_));
}

BitMap::BitMap(const char* serialized, HashStringAllocator* allocator)
    : bits_(Allocator(allocator)) {
  common::InputByteStream stream(serialized);
  length_ = stream.read<uint32_t>();

  bits_.resize(velox::bits::nbytes(static_cast<uint64_t>(length_)));
  stream.copyTo(bits_.data(), static_cast<int>(bits_.size()));
}

bool BitMap::bitAt(uint32_t position) const {
  return bits::isBitSet(bits_.data(), static_cast<uint64_t>(position));
}

void BitMap::setBit(uint32_t position, bool value) {
  bits::setBit(bits_.data(), static_cast<uint64_t>(position), value);
}

void BitMap::flipBit(uint32_t position) {
  bits::setBit(
      bits_.data(),
      static_cast<uint64_t>(position),
      !bits::isBitSet(bits_.data(), position));
}

uint32_t BitMap::countBits() const {
  const uint64_t* wordBits = reinterpret_cast<const uint64_t*>(bits_.data());
  return static_cast<uint32_t>(bits::countBits(wordBits, 0, length_));
}

std::vector<int8_t, Allocator> BitMap::toCompactIntVector() const {
  if (bits_.empty()) {
    return std::vector<int8_t, Allocator>(Allocator(bits_.get_allocator()));
  }

  // Find the last non-zero byte.
  const uint64_t* wordBits = reinterpret_cast<const uint64_t*>(bits_.data());
  int32_t lastSetBit = bits::findLastBit(wordBits, 0, length_);

  if (lastSetBit < 0) {
    // No bits are set
    return std::vector<int8_t, Allocator>(Allocator(bits_.get_allocator()));
  }

  // Calculate the byte index that contains the last set bit
  int64_t lastNonZeroIndex = lastSetBit / 8;

  return std::vector<int8_t, Allocator>(
      bits_.begin(),
      bits_.begin() + lastNonZeroIndex + 1,
      Allocator(bits_.get_allocator()));
}

size_t BitMap::serializedSize() const {
  return sizeof(uint32_t) + sizeof(int8_t) * toCompactIntVector().size();
}

void BitMap::serialize(char* output) const {
  common::OutputByteStream stream(output);
  auto compactVector = toCompactIntVector();
  uint32_t serializedLength =
      static_cast<uint32_t>(compactVector.size() * sizeof(int8_t));
  stream.appendOne(serializedLength);
  stream.append(
      reinterpret_cast<const char*>(compactVector.data()),
      static_cast<int32_t>(compactVector.size()));
}

BitMap BitMap::deserialize(
    const char* bytes,
    uint32_t actualLength, // this is the serialized length after truncation
    uint32_t expectedLength, // this length equal to buckets * precison
    HashStringAllocator* allocator) {
  BitMap bitmap(expectedLength, allocator);

  if (actualLength > 0) {
    // Java's BitSet.toByteArray() returns bytes in little-endian format
    // and truncates trailing zeros. We need to copy these bytes and
    // ensure the rest of the bitmap is zero-initialized.
    uint32_t bytesToCopy = std::min(
        actualLength,
        static_cast<uint32_t>(velox::bits::nbytes(expectedLength)));

    std::memcpy(bitmap.getMutableBits().data(), bytes, bytesToCopy);
  }

  return bitmap;
}

void BitMap::bitwiseOr(const BitMap& other) {
  VELOX_CHECK_NOT_NULL(other.bits_.data(), "Can't combine with null bitmap.");
  VELOX_CHECK_EQ(
      length_, other.length_, "Can't OR two bitmaps of different lengths.");

  uint64_t* target = reinterpret_cast<uint64_t*>(bits_.data());
  const uint64_t* right = reinterpret_cast<const uint64_t*>(other.bits_.data());
  bits::orBits(target, right, 0, length_);
}
} // namespace facebook::velox::functions::aggregate
