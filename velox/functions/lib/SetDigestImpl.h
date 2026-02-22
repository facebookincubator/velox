/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/hyperloglog/Murmur3Hash128.h"

#include <limits>
#include <vector>

namespace facebook::velox::functions {

namespace detail {
constexpr int8_t kUncompressedFormat = 1;

inline uint64_t computeHash(int64_t value) {
  return common::hll::Murmur3Hash128::hash64ForLong(value, 0);
}

inline uint64_t computeHash(StringView value) {
  return common::hll::Murmur3Hash128::hash64(value.data(), value.size(), 0);
}
} // namespace detail

template <typename T>
SetDigest<T>::SetDigest(
    HashStringAllocator* allocator,
    int8_t indexBitLength,
    int32_t maxHashes)
    : minhash_{MinHashAllocator(allocator)},
      hllAccumulator_{indexBitLength, allocator},
      maxHashes_{maxHashes},
      allocator_{allocator},
      hllTotalEstimatedSerializedSize_{0} {
  // Validate indexBitLength matches Java validation
  VELOX_CHECK_GT(indexBitLength, 0, "indexBitLength must be > 0");
  VELOX_CHECK_LE(indexBitLength, 16, "indexBitLength must be <= 16");

  // Validate maxHashes
  VELOX_CHECK_GT(maxHashes, 0, "maxHashes must be > 0");

  // Verify default indexBitLength matches kNumberOfBuckets
  static_assert(
      (1 << SetDigest<T>::kDefaultIndexBitLength) ==
          SetDigest<T>::kNumberOfBuckets,
      "Default index bit length must match NUMBER_OF_BUCKETS");

  // Initialize HLL size after HLL is constructed
  increaseTotalHllSize();
}

template <typename T>
bool SetDigest<T>::isExact() const {
  return static_cast<int32_t>(minhash_.size()) < maxHashes_;
}

template <typename T>
void SetDigest<T>::add(T value) {
  uint64_t hash = detail::computeHash(value);
  addHash(hash);
  decreaseTotalHllSize();
  hllAccumulator_.insertHash(hash);
  increaseTotalHllSize();
}

template <typename T>
void SetDigest<T>::addHash(uint64_t hash) {
  int64_t signedHash = static_cast<int64_t>(hash);
  auto it = minhash_.find(signedHash);
  int16_t currentCount = (it != minhash_.end()) ? it->second : 0;

  if (currentCount < std::numeric_limits<int16_t>::max()) {
    minhash_[signedHash] = static_cast<int16_t>(currentCount + 1);
  }

  while (static_cast<int32_t>(minhash_.size()) > maxHashes_) {
    minhash_.erase(std::prev(minhash_.end()));
  }
}

template <typename T>
inline void SetDigest<T>::increaseTotalHllSize() {
  hllTotalEstimatedSerializedSize_ += hllAccumulator_.serializedSize();
}

template <typename T>
inline void SetDigest<T>::decreaseTotalHllSize() {
  hllTotalEstimatedSerializedSize_ -= hllAccumulator_.serializedSize();
}

template <typename T>
int32_t SetDigest<T>::estimatedSerializedSize() const {
  return static_cast<int32_t>(
      sizeof(int8_t) + // format byte
      sizeof(int32_t) + // HLL length
      hllTotalEstimatedSerializedSize_ + // HLL data
      sizeof(int32_t) + // maxHashes
      sizeof(int32_t) + // minhash size
      minhash_.size() * (sizeof(int64_t) + sizeof(int16_t))); // keys + values
}

template <typename T>
void SetDigest<T>::serialize(char* out) const {
  common::OutputByteStream stream(out);

  stream.appendOne(detail::kUncompressedFormat);
  stream.appendOne(hllTotalEstimatedSerializedSize_);

  // Use const_cast because HllAccumulator::serialize() is not const,
  // but serialization does not logically modify the object.
  const_cast<common::hll::HllAccumulator<int64_t, true, HashStringAllocator>&>(
      hllAccumulator_)
      .serialize(out + stream.offset());
  stream = common::OutputByteStream(
      out, stream.offset() + hllTotalEstimatedSerializedSize_);

  stream.appendOne(maxHashes_);
  stream.appendOne(static_cast<int32_t>(minhash_.size()));

  for (const auto& entry : minhash_) {
    stream.appendOne(entry.first);
  }

  for (const auto& entry : minhash_) {
    stream.appendOne(entry.second);
  }

  VELOX_CHECK_EQ(stream.offset(), estimatedSerializedSize());
}

template <typename T>
Status SetDigest<T>::deserialize(const char* data, int32_t size) {
  if (size < static_cast<int32_t>(sizeof(int8_t) + sizeof(int32_t))) {
    return Status::UserError("Input too small to be a valid SetDigest");
  }

  common::InputByteStream stream(data);

  int8_t format = stream.read<int8_t>();
  if (format != detail::kUncompressedFormat) {
    return Status::UserError(
        "Unexpected SetDigest format: expected {}, got {}",
        detail::kUncompressedFormat,
        format);
  }

  // Read HLL length
  int32_t hllLength = stream.read<int32_t>();
  if (hllLength < 0) {
    return Status::UserError("Invalid HLL length in SetDigest serialization");
  }
  if (hllLength >
      size - static_cast<int32_t>(sizeof(int8_t) + sizeof(int32_t))) {
    return Status::UserError("HLL length exceeds input size");
  }

  // Deserialize HLL using HllAccumulator
  auto deserializedHll =
      common::hll::HllAccumulator<int64_t, true, HashStringAllocator>::
          deserialize(data + stream.offset(), allocator_);
  decreaseTotalHllSize();
  hllAccumulator_ = std::move(*deserializedHll);
  increaseTotalHllSize();

  stream = common::InputByteStream(data + stream.offset() + hllLength);

  // Calculate remaining bytes after HLL data
  int32_t bytesConsumed = sizeof(int8_t) + sizeof(int32_t) + hllLength;
  int32_t remainingBytes = size - bytesConsumed;

  if (remainingBytes < static_cast<int32_t>(2 * sizeof(int32_t))) {
    return Status::UserError(
        "Input too small to contain maxHashes and minhash size");
  }

  maxHashes_ = stream.read<int32_t>();
  if (maxHashes_ <= 0) {
    return Status::UserError("maxHashes must be > 0");
  }

  int32_t minhashSize = stream.read<int32_t>();
  if (minhashSize < 0) {
    return Status::UserError("minhash size cannot be negative");
  }

  // Validate that there's enough data for the minhash entries
  int32_t expectedMinhashBytes =
      minhashSize * (sizeof(int64_t) + sizeof(int16_t));
  if (remainingBytes - 2 * static_cast<int32_t>(sizeof(int32_t)) <
      expectedMinhashBytes) {
    return Status::UserError("Input too small to contain minhash entries");
  }

  std::vector<int64_t> keys;
  keys.reserve(minhashSize);
  for (int32_t i = 0; i < minhashSize; i++) {
    keys.push_back(stream.read<int64_t>());
  }

  for (int32_t i = 0; i < minhashSize; i++) {
    int16_t count = stream.read<int16_t>();
    minhash_[keys[i]] = count;
  }

  return Status::OK();
}

template <typename T>
void SetDigest<T>::mergeWith(const SetDigest& other) {
  // Merge HyperLogLog using HllAccumulator
  decreaseTotalHllSize();
  hllAccumulator_.mergeWith(other.hllAccumulator_);
  increaseTotalHllSize();

  // Merge minhash maps
  for (const auto& entry : other.minhash_) {
    int64_t key = entry.first;
    int16_t otherCount = entry.second;

    auto it = minhash_.find(key);
    int16_t currentCount = (it != minhash_.end()) ? it->second : 0;

    // Add counts, saturating at int16_t max
    int32_t newCount =
        static_cast<int32_t>(currentCount) + static_cast<int32_t>(otherCount);
    if (newCount > std::numeric_limits<int16_t>::max()) {
      minhash_[key] = std::numeric_limits<int16_t>::max();
    } else {
      minhash_[key] = static_cast<int16_t>(newCount);
    }
  }

  // Remove largest hash values when we exceed maxHashes
  while (static_cast<int32_t>(minhash_.size()) > maxHashes_) {
    minhash_.erase(std::prev(minhash_.end()));
  }
}

template <typename T>
int64_t SetDigest<T>::cardinality() const {
  if (isExact()) {
    return static_cast<int64_t>(minhash_.size());
  }
  return hllAccumulator_.cardinality();
}

template <typename T>
int64_t SetDigest<T>::exactIntersectionCardinality(
    const SetDigest& left,
    const SetDigest& right) {
  VELOX_USER_CHECK(
      left.isExact(), "exact intersection cannot operate on approximate sets");
  VELOX_USER_CHECK(
      right.isExact(), "exact intersection cannot operate on approximate sets");

  const auto& [smaller, larger] = left.minhash_.size() <= right.minhash_.size()
      ? std::tie(left, right)
      : std::tie(right, left);

  int64_t count = 0;
  for (const auto& entry : smaller.minhash_) {
    if (larger.minhash_.count(entry.first)) {
      count++;
    }
  }
  return count;
}

template <typename T>
double SetDigest<T>::jaccardIndex(
    const SetDigest& left,
    const SetDigest& right) {
  if (left.minhash_.empty() && right.minhash_.empty()) {
    return 1.0;
  }

  int32_t sizeOfSmallerSet = std::min(
      static_cast<int32_t>(left.minhash_.size()),
      static_cast<int32_t>(right.minhash_.size()));

  if (sizeOfSmallerSet == 0) {
    return 0.0;
  }

  auto itLeft = left.minhash_.begin();
  auto itRight = right.minhash_.begin();

  int32_t intersectionCnt = 0;
  int32_t unionCnt = 0;

  while (itLeft != left.minhash_.end() && itRight != right.minhash_.end() &&
         unionCnt < sizeOfSmallerSet) {
    if (itLeft->first < itRight->first) {
      ++itLeft;
    } else if (itRight->first < itLeft->first) {
      ++itRight;
    } else {
      intersectionCnt++;
      ++itLeft;
      ++itRight;
    }
    unionCnt++;
  }

  return static_cast<double>(intersectionCnt) /
      static_cast<double>(sizeOfSmallerSet);
}

template <typename T>
std::map<int64_t, int16_t> SetDigest<T>::getHashCounts() const {
  std::map<int64_t, int16_t> result;
  for (const auto& entry : minhash_) {
    result[entry.first] = entry.second;
  }
  return result;
}

} // namespace facebook::velox::functions
