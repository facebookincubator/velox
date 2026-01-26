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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/hyperloglog/Murmur3Hash128.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox::common::hll {

namespace detail {
constexpr uint8_t kVersionByte = 1;
constexpr int64_t kHashOutputHalfRange = INT64_MAX;
constexpr size_t kHeaderSize = sizeof(uint8_t) // version
    + 4 * sizeof(int32_t); // maxSize, hllBuckets, minhashSize, hllsTotalSize;

template <typename TJoinKey>
static int64_t hashKey(TJoinKey joinKey) {
  int64_t result;
  if constexpr (std::is_same_v<TJoinKey, Timestamp>) {
    result = joinKey.toMillis();
  } else if constexpr (
      std::is_same_v<TJoinKey, int128_t> ||
      std::is_same_v<TJoinKey, uint128_t>) {
    return Murmur3Hash128::hash64(&joinKey, sizeof(joinKey), 0);
  } else if constexpr (std::is_integral_v<TJoinKey>) {
    result = static_cast<int64_t>(joinKey);
  } else if constexpr (std::is_same_v<TJoinKey, float>) {
    // Cast to double first, then extract bits, based on implicit coercion
    double dbl = static_cast<double>(joinKey);
    std::memcpy(&result, &dbl, sizeof(result));
  } else if constexpr (std::is_same_v<TJoinKey, double>) {
    std::memcpy(&result, &joinKey, sizeof(result));
  } else if constexpr (std::is_same_v<TJoinKey, StringView>) {
    result =
        common::hll::Murmur3Hash128::hash64(joinKey.data(), joinKey.size(), 0);
  } else {
    VELOX_UNREACHABLE("Unsupported input type: {}", typeid(TJoinKey).name());
  }

  return Murmur3Hash128::hash64(&result, sizeof(result), 0);
}
} // namespace detail

template <typename TUii, typename TAllocator>
Expected<std::unique_ptr<KHyperLogLog<TUii, TAllocator>>>
KHyperLogLog<TUii, TAllocator>::deserialize(
    const char* data,
    size_t size,
    TAllocator* allocator) {
  VELOX_RETURN_UNEXPECTED_IF(
      size < sizeof(uint8_t),
      Status::UserError("Invalid KHyperLogLog data: too small"));

  InputByteStream stream(data);

  uint8_t version = stream.read<uint8_t>();
  VELOX_RETURN_UNEXPECTED_IF(
      version != detail::kVersionByte,
      Status::UserError("Unsupported KHyperLogLog version"));

  VELOX_RETURN_UNEXPECTED_IF(
      size < detail::kHeaderSize,
      Status::UserError("Invalid KHyperLogLog data: insufficient header size"));

  // Header values
  int32_t maxSize = stream.read<int32_t>();
  int32_t hllBuckets = stream.read<int32_t>();
  int32_t minhashSize = stream.read<int32_t>();
  int32_t totalHllSize = stream.read<int32_t>();

  if (minhashSize == 0) {
    VELOX_RETURN_UNEXPECTED_IF(
        totalHllSize != 0,
        Status::UserError(
            "Invalid KHyperLogLog data: minhashSize is 0 but totalHllSize is not 0"));
    size_t remainingSize = size - stream.offset();
    VELOX_RETURN_UNEXPECTED_IF(
        remainingSize != 0,
        Status::UserError(
            "Invalid KHyperLogLog data: minhashSize is 0 but extra data remains"));
    return std::make_unique<KHyperLogLog<TUii, TAllocator>>(
        maxSize, hllBuckets, allocator);
  }

  // Validate remaining size.
  size_t expectedRemainingSize =
      minhashSize * (sizeof(int32_t) + sizeof(int64_t)) + totalHllSize;
  size_t remainingSize = size - stream.offset();
  VELOX_RETURN_UNEXPECTED_IF(
      remainingSize < expectedRemainingSize,
      Status::UserError(
          "Invalid KHyperLogLog data: insufficient data for minhash and HLLs"));

  // Read HLL sizes.
  std::vector<int32_t> hllSizes(minhashSize);
  stream.copyTo(hllSizes.data(), minhashSize);

  // Read keys.
  std::vector<int64_t> keys(minhashSize);
  stream.copyTo(keys.data(), minhashSize);

  // Create KHyperLogLog instance.
  auto result = std::make_unique<KHyperLogLog<TUii, TAllocator>>(
      maxSize, hllBuckets, allocator);

  // Deserialize HLLs.
  const char* hllData = data + stream.offset();
  remainingSize = size - stream.offset();

  size_t sumOfHllSizes = 0;
  for (int32_t hllSize : hllSizes) {
    sumOfHllSizes += hllSize;
  }

  VELOX_RETURN_UNEXPECTED_IF(
      remainingSize < sumOfHllSizes,
      Status::UserError(
          "Invalid KHyperLogLog data: insufficient data for HLLs"));

  for (int32_t i = 0; i < minhashSize; ++i) {
    int32_t hllSize = hllSizes[i];
    int64_t key = keys[i];

    auto hll =
        HllAccumulator<TUii, true, TAllocator>::deserialize(hllData, allocator);
    result->increaseTotalHllSize(*hll);
    result->minhash_.emplace(key, std::move(*hll));

    hllData += hllSize;
    remainingSize -= hllSize;
  }

  return result;
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::serialize(char* outputBuffer) {
  OutputByteStream stream(outputBuffer);

  // Write version.
  stream.appendOne<uint8_t>(detail::kVersionByte);

  // Write header.
  stream.appendOne<int32_t>(maxSize_);
  stream.appendOne<int32_t>(hllBuckets_);
  stream.appendOne<int32_t>(static_cast<int32_t>(minhash_.size()));

  // Write the sum of all HLL sizes.
  int32_t totalHllSize = 0;
  for (auto& [key, hll] : minhash_) {
    totalHllSize += hll.serializedSize();
  }
  stream.appendOne<int32_t>(totalHllSize);

  // Write HLL sizes in sorted key order. minHash_ is a sorted map, so no
  // additional sorting is necessary for deterministic serialization.
  int32_t maxSerializedSize = 0;
  for (auto& [key, hll] : minhash_) {
    int32_t hllSerializedSize = hll.serializedSize();
    stream.appendOne<int32_t>(hllSerializedSize);
    maxSerializedSize = std::max(maxSerializedSize, hllSerializedSize);
  }

  // Write keys in sorted order.
  for (const auto& [key, hll] : minhash_) {
    stream.appendOne<int64_t>(key);
  }

  // Write serialized HLLs in sorted key order.
  std::string hllBuffer(maxSerializedSize, '\0');
  for (auto& [key, hll] : minhash_) {
    const_cast<HllAccumulator<TUii, true, TAllocator>&>(hll).serialize(
        hllBuffer.data());
    stream.append(hllBuffer.data(), hll.serializedSize());
  }
}

template <typename TUii, typename TAllocator>
size_t KHyperLogLog<TUii, TAllocator>::estimatedSerializedSize() const {
  return detail::kHeaderSize + // header: version, maxSize, hllBuckets,
                               // minhashSize, totalHllSize
      minhash_.size() * sizeof(int32_t) + // individual HLL sizes +
      minhash_.size() * sizeof(int64_t) + // minhash keys +
      hllsTotalEstimatedSerializedSize_; // sum of all HLL serialized sizes
}

template <typename TUii, typename TAllocator>
template <typename TJoinKey>
void KHyperLogLog<TUii, TAllocator>::add(TJoinKey joinKey, TUii uii) {
  update(detail::hashKey(joinKey), uii);
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::update(int64_t hash, TUii uii) {
  auto it = minhash_.end();
  if (!(isExact() || (minhash_.size() > 0 && hash < maxKey()) ||
        ((it = minhash_.find(hash)) != minhash_.end()))) {
    return;
  }

  // Get or create HLL for this hash.
  HllAccumulator<TUii, true, TAllocator>* hll;
  if (it == minhash_.end()) {
    auto [iterator, inserted] = minhash_.emplace(
        hash,
        HllAccumulator<TUii, true, TAllocator>(indexBitLength_, allocator_));
    hll = &iterator->second;
  } else {
    hll = &it->second;
    decreaseTotalHllSize(*hll);
  }

  hll->append(uii);

  increaseTotalHllSize(*hll);
  removeOverflowEntries();
}

template <typename TUii, typename TAllocator>
int64_t KHyperLogLog<TUii, TAllocator>::cardinality() const {
  if (isExact()) {
    return static_cast<int64_t>(minhash_.size());
  }

  // Estimate cardinality by calculating the average spacing (density) between
  // stored hash values, then extrapolating to the full hash space.
  // The hash range is the full 64-bit hash range (2^64) which does not fit in a
  // double, so both the hash range and the density are halved to keep
  // proportions mathematically correct. The "-1" is a statistical bias
  // correction from "On Synopses for Distinct-Value Estimation Under Multiset
  // Operations" by Beyer et. al.
  // Use unsigned arithmetic to avoid overflow for large maxKey values.
  uint64_t hashesRange =
      static_cast<uint64_t>(maxKey()) - static_cast<uint64_t>(INT64_MIN);
  double halfDensity =
      static_cast<double>(hashesRange) / (minhash_.size() - 1) / 2.0;
  return static_cast<int64_t>(
      static_cast<double>(detail::kHashOutputHalfRange) / halfDensity);
}

template <typename TUii, typename TAllocator>
bool KHyperLogLog<TUii, TAllocator>::isExact() const {
  return static_cast<int32_t>(minhash_.size()) < maxSize_;
}

template <typename TUii, typename TAllocator>
size_t KHyperLogLog<TUii, TAllocator>::minhashSize() const {
  return minhash_.size();
}

template <typename TUii, typename TAllocator>
int64_t KHyperLogLog<TUii, TAllocator>::exactIntersectionCardinality(
    const KHyperLogLog<TUii, TAllocator>& left,
    const KHyperLogLog<TUii, TAllocator>& right) {
  VELOX_CHECK(
      left.isExact(),
      "exactIntersectionCardinality cannot operate on approximate sets");
  VELOX_CHECK(
      right.isExact(),
      "exactIntersectionCardinality cannot operate on approximate sets");

  // Optimize by iterating through the smaller map and checking the larger one.
  const auto& smaller = left.minhash_.size() <= right.minhash_.size()
      ? left.minhash_
      : right.minhash_;
  const auto& larger = left.minhash_.size() <= right.minhash_.size()
      ? right.minhash_
      : left.minhash_;

  // Count intersection of keys.
  int64_t intersectionCnt = 0;
  for (const auto& [key, hll] : smaller) {
    if (larger.contains(key)) {
      intersectionCnt++;
    }
  }

  return intersectionCnt;
}

template <typename TUii, typename TAllocator>
double KHyperLogLog<TUii, TAllocator>::jaccardIndex(
    const KHyperLogLog<TUii, TAllocator>& left,
    const KHyperLogLog<TUii, TAllocator>& right) {
  if (left.minhash_.empty() && right.minhash_.empty()) {
    return 1.0;
  }
  auto smallerSize = std::min(left.minhash_.size(), right.minhash_.size());

  if (smallerSize == 0) {
    return 0.0;
  }

  auto itA = left.minhash_.begin();
  auto itB = right.minhash_.begin();

  auto intersectionCnt = 0;
  auto unionCnt = 0;

  // Merge the two sorted sequences, counting intersection along the way.
  while (itA != left.minhash_.end() && itB != right.minhash_.end() &&
         unionCnt < smallerSize) {
    if (itA->first < itB->first) {
      ++itA;
    } else if (itB->first < itA->first) {
      ++itB;
    } else {
      intersectionCnt++;
      ++itA;
      ++itB;
    }
    unionCnt++;
  }

  return static_cast<double>(intersectionCnt) / smallerSize;
}

template <typename TUii, typename TAllocator>
std::unique_ptr<KHyperLogLog<TUii, TAllocator>>
KHyperLogLog<TUii, TAllocator>::merge(
    std::unique_ptr<KHyperLogLog<TUii, TAllocator>> left,
    std::unique_ptr<KHyperLogLog<TUii, TAllocator>> right) {
  // The KHLL with the smaller K will be used as base. This is because if a KHLL
  // with a smaller K is merged into a KHLL with a larger K, the smaller KHLL's
  // minhash will not cover all of the larger minhash's range. Instead, we want
  // to keep only the smallest K number of HLLs in the new KHLL.
  if (left->maxSize_ <= right->maxSize_) {
    left->mergeWith(*right);
    return std::move(left);
  } else {
    right->mergeWith(*left);
    return std::move(right);
  }
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::mergeWith(
    const KHyperLogLog<TUii, TAllocator>& other) {
  for (const auto& [key, otherHll] : other.minhash_) {
    auto it = minhash_.find(key);
    if (it != minhash_.end()) {
      decreaseTotalHllSize(it->second);
      it->second.mergeWith(otherHll);
      increaseTotalHllSize(it->second);
    } else {
      HllAccumulator<TUii, true, TAllocator> newHll(
          indexBitLength_, allocator_);
      newHll.mergeWith(otherHll);
      increaseTotalHllSize(newHll);
      minhash_.emplace(key, std::move(newHll));
    }
  }

  removeOverflowEntries();
}

template <typename TUii, typename TAllocator>
double KHyperLogLog<TUii, TAllocator>::reidentificationPotential(
    int64_t threshold) const {
  if (minhash_.empty()) {
    return 0.0;
  }
  int64_t highlyUniqueValues = 0;

  for (const auto& [key, hll] : minhash_) {
    if (hll.cardinality() <= threshold) {
      highlyUniqueValues++;
    }
  }

  return static_cast<double>(highlyUniqueValues) / minhash_.size();
}

template <typename TUii, typename TAllocator>
folly::F14FastMap<int64_t, double>
KHyperLogLog<TUii, TAllocator>::uniquenessDistribution(
    int64_t histogramSize) const {
  folly::F14FastMap<int64_t, double> out;

  for (int64_t i = 1; i <= histogramSize; ++i) {
    out[i] = 0.0;
  }

  int64_t size = minhash_.size();
  if (size == 0) {
    return out;
  }

  double bucketScale = 1.0 / static_cast<double>(size);
  for (const auto& [key, hll] : minhash_) {
    int64_t cardinality = hll.cardinality();
    int64_t bucket = std::min(cardinality, histogramSize);
    out[bucket] += bucketScale;
  }

  return out;
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::removeOverflowEntries() {
  while (static_cast<int32_t>(minhash_.size()) > maxSize_) {
    auto maxIt = std::prev(minhash_.end());
    if (maxIt != minhash_.end()) {
      decreaseTotalHllSize(maxIt->second);
      minhash_.erase(maxIt);
    }
  }
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::increaseTotalHllSize(
    HllAccumulator<TUii, true, TAllocator>& hll) {
  hllsTotalEstimatedSerializedSize_ += hll.serializedSize();
}

template <typename TUii, typename TAllocator>
void KHyperLogLog<TUii, TAllocator>::decreaseTotalHllSize(
    HllAccumulator<TUii, true, TAllocator>& hll) {
  hllsTotalEstimatedSerializedSize_ -= hll.serializedSize();
}

} // namespace facebook::velox::common::hll
