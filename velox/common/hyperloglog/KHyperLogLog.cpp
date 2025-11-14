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

#include "velox/common/hyperloglog/KHyperLogLog.h"

#include <algorithm>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/hyperloglog/Murmur3Hash128.h"

namespace {
constexpr uint8_t kVersionByte = 1;
constexpr int64_t kHashOutputHalfRange = INT64_MAX;

// numBuckets = 2^bits
int8_t toIndexBitLength(int32_t numHLLBuckets) {
  return static_cast<int8_t>(std::log2(numHLLBuckets));
}
} // namespace

namespace facebook::velox::common::hll {

template <typename TAllocator>
std::unique_ptr<KHyperLogLog<TAllocator>> KHyperLogLog<TAllocator>::deserialize(
    const char* data,
    size_t size,
    TAllocator* allocator) {
  VELOX_CHECK_GE(size, sizeof(uint8_t), "Invalid KHyperLogLog data: too small");

  InputByteStream stream(data);

  uint8_t version = stream.read<uint8_t>();
  VELOX_CHECK_EQ(
      version, kVersionByte, "Unsupported KHyperLogLog version: {}", version);

  VELOX_CHECK_GE(
      size,
      sizeof(uint8_t) + 4 * sizeof(int32_t),
      "Invalid KHyperLogLog data: insufficient header size");

  // Header values
  int32_t maxSize = stream.read<int32_t>();
  int32_t hllBuckets = stream.read<int32_t>();
  int32_t minhashSize = stream.read<int32_t>();
  int32_t totalHllSize = stream.read<int32_t>();

  // Validate remaining size.
  size_t expectedRemainingSize =
      minhashSize * (sizeof(int32_t) + sizeof(int64_t)) + totalHllSize;
  size_t currentOffset = stream.offset();
  VELOX_CHECK_GE(
      size - currentOffset,
      expectedRemainingSize,
      "Invalid KHyperLogLog data: insufficient data for minhash and HLLs");

  // Read HLL sizes.
  std::vector<int32_t> hllSizes(minhashSize);
  for (int32_t i = 0; i < minhashSize; ++i) {
    hllSizes[i] = stream.read<int32_t>();
  }

  // Read keys.
  std::vector<int64_t> keys(minhashSize);
  for (int32_t i = 0; i < minhashSize; ++i) {
    keys[i] = stream.read<int64_t>();
  }

  // Create KHyperLogLog instance.
  auto result = std::make_unique<KHyperLogLog<TAllocator>>(
      maxSize, hllBuckets, allocator);

  // Deserialize HLLs.
  const char* hllData = data + stream.offset();
  size_t remainingSize = size - stream.offset();

  for (int32_t i = 0; i < minhashSize; ++i) {
    int32_t hllSize = hllSizes[i];
    int64_t key = keys[i];

    VELOX_CHECK_GE(
        remainingSize,
        static_cast<size_t>(hllSize),
        "Invalid KHyperLogLog data: insufficient data for HLL {}",
        i);

    auto hll = HllWrapper<TAllocator>::deserialize(hllData, allocator);
    result->increaseTotalHllSize(*hll);
    result->minhash_.emplace(key, std::move(*hll));

    hllData += hllSize;
    remainingSize -= hllSize;
  }

  return result;
}

template <typename TAllocator>
std::string KHyperLogLog<TAllocator>::serialize() {
  size_t totalSize = estimatedSerializedSize();
  std::string result(totalSize, '\0');
  OutputByteStream stream(result.data());

  // Write version.
  stream.appendOne<uint8_t>(kVersionByte);

  // Write header.
  stream.appendOne<int32_t>(maxSize_);
  stream.appendOne<int32_t>(hllBuckets_);
  stream.appendOne<int32_t>(static_cast<int32_t>(minhash_.size()));

  // Calculate total HLL size and serialize HLLs in sorted key order.
  int32_t totalHllSize = 0;
  std::vector<int32_t> hllSizes;
  std::vector<std::string> serializedHlls;

  for (const auto& [key, hll] : minhash_) {
    int32_t hllSerializedSize = hll.serializedSize();
    std::string serializedHll(hllSerializedSize, '\0');
    const_cast<HllWrapper<TAllocator>&>(hll).serialize(serializedHll.data());
    serializedHlls.push_back(serializedHll);
    hllSizes.push_back(hllSerializedSize);
    totalHllSize += hllSerializedSize;
  }

  stream.appendOne<int32_t>(totalHllSize);

  // Write HLL sizes in sorted key order.
  for (int32_t hllSize : hllSizes) {
    stream.appendOne<int32_t>(hllSize);
  }

  // Write keys in sorted order.
  for (const auto& [key, hll] : minhash_) {
    stream.appendOne<int64_t>(key);
  }

  // Write serialized HLLs in sorted key order.
  for (const auto& serializedHll : serializedHlls) {
    stream.append(serializedHll.data(), serializedHll.size());
  }

  return result;
}

template <typename TAllocator>
size_t KHyperLogLog<TAllocator>::estimatedSerializedSize() const {
  return sizeof(uint8_t) + // version +
      (4 * sizeof(int32_t)) + // header (maxSize, hllBuckets, minhashSize,
                              // totalHllSize) +
      minhash_.size() * sizeof(int64_t) + // minhash keys +
      minhash_.size() * sizeof(int32_t) + // individual HLL sizes +
      hllsTotalEstimatedSerializedSize_; // sum of all HLL serialized sizes
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::add(int64_t joinKey, int64_t uii) {
  int64_t hash = Murmur3Hash128::hash64(&joinKey, sizeof(joinKey), 0);
  update(hash, uii);
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::update(int64_t hash, int64_t uii) {
  if (!(minhash_.find(hash) != minhash_.end() || isExact() ||
        (minhash_.size() > 0 && hash < maxKey()))) {
    return;
  }

  // Get or create HLL for this hash.
  auto it = minhash_.find(hash);
  HllWrapper<TAllocator>* hll;
  if (it == minhash_.end()) {
    auto [iterator, inserted] = minhash_.emplace(
        hash,
        HllWrapper<TAllocator>(toIndexBitLength(hllBuckets_), allocator_));
    hll = &iterator->second;
  } else {
    hll = &it->second;
    decreaseTotalHllSize(*hll);
  }

  int64_t uiiHash = Murmur3Hash128::hash64ForLong(uii, 0);
  hll->insertHash(static_cast<uint64_t>(uiiHash));
  increaseTotalHllSize(*hll);

  removeOverflowEntries();
}

template <typename TAllocator>
int64_t KHyperLogLog<TAllocator>::cardinality() const {
  if (isExact()) {
    return static_cast<int64_t>(minhash_.size());
  }
  if (minhash_.empty()) {
    return 0;
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
  return static_cast<int64_t>(kHashOutputHalfRange / halfDensity);
}

template <typename TAllocator>
bool KHyperLogLog<TAllocator>::isExact() const {
  return static_cast<int32_t>(minhash_.size()) < maxSize_;
}

template <typename TAllocator>
size_t KHyperLogLog<TAllocator>::minhashSize() const {
  return minhash_.size();
}

template <typename TAllocator>
int64_t KHyperLogLog<TAllocator>::exactIntersectionCardinality(
    const KHyperLogLog<TAllocator>& khll1,
    const KHyperLogLog<TAllocator>& khll2) {
  VELOX_CHECK(
      khll1.isExact() && khll2.isExact(),
      "exactIntersectionCardinality cannot operate on approximate sets");

  // Count intersection of keys.
  int64_t intersection = 0;
  for (const auto& [key, hll] : khll1.minhash_) {
    if (khll2.minhash_.find(key) != khll2.minhash_.end()) {
      intersection++;
    }
  }

  return intersection;
}

template <typename TAllocator>
double KHyperLogLog<TAllocator>::jaccardIndex(
    const KHyperLogLog<TAllocator>& khll1,
    const KHyperLogLog<TAllocator>& khll2) {
  auto sizeOfSmallerSet =
      std::min(khll1.minhash_.size(), khll2.minhash_.size());

  if (sizeOfSmallerSet == 0) {
    return 0.0;
  }

  auto itA = khll1.minhash_.begin();
  auto itB = khll2.minhash_.begin();

  auto intersection = 0;
  auto unionCount = 0;

  // Merge the two sorted sequences, counting intersection along the way.
  while (itA != khll1.minhash_.end() && itB != khll2.minhash_.end() &&
         unionCount < sizeOfSmallerSet) {
    if (itA->first < itB->first) {
      ++itA;
    } else if (itB->first < itA->first) {
      ++itB;
    } else {
      intersection++;
      ++itA;
      ++itB;
    }
    unionCount++;
  }

  return static_cast<double>(intersection) /
      static_cast<double>(sizeOfSmallerSet);
}

template <typename TAllocator>
std::unique_ptr<KHyperLogLog<TAllocator>> KHyperLogLog<TAllocator>::merge(
    KHyperLogLog<TAllocator>& khll1,
    KHyperLogLog<TAllocator>& khll2) {
  // The khll with the smaller K will be used as base. This is because if a khll
  // with a smaller K is merged into a khll with a larger K, the smaller khll's
  // minhash will not cover all of the larger minhash's range. Instead, we want
  // to keep only the smallest K number of HLLs in the new khll.
  std::unique_ptr<KHyperLogLog<TAllocator>> result;
  if (khll1.maxSize_ <= khll2.maxSize_) {
    khll1.mergeWith(khll2);
    result = std::make_unique<KHyperLogLog<TAllocator>>(khll1);
  } else {
    khll2.mergeWith(khll1);
    result = std::make_unique<KHyperLogLog<TAllocator>>(khll2);
  }
  return result;
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::mergeWith(
    const KHyperLogLog<TAllocator>& other) {
  for (const auto& [key, otherHll] : other.minhash_) {
    auto it = minhash_.find(key);
    if (it != minhash_.end()) {
      decreaseTotalHllSize(it->second);
      it->second.mergeWith(otherHll);
      increaseTotalHllSize(it->second);
    } else {
      HllWrapper<TAllocator> newHll(toIndexBitLength(hllBuckets_), allocator_);
      newHll.mergeWith(otherHll);
      increaseTotalHllSize(newHll);
      minhash_.emplace(key, std::move(newHll));
    }
  }

  removeOverflowEntries();
}

template <typename TAllocator>
double KHyperLogLog<TAllocator>::reidentificationPotential(
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

template <typename TAllocator>
std::map<int64_t, double> KHyperLogLog<TAllocator>::uniquenessDistribution(
    int64_t histogramSize) const {
  std::map<int64_t, double> out;

  for (int64_t i = 1; i <= histogramSize; ++i) {
    out[i] = 0.0;
  }

  int64_t size = minhash_.size();
  if (size == 0) {
    return out;
  }

  for (const auto& [key, hll] : minhash_) {
    int64_t cardinality = hll.cardinality();
    int64_t bucket = std::min(cardinality, histogramSize);
    out[bucket] += 1.0 / static_cast<double>(size);
  }

  return out;
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::removeOverflowEntries() {
  while (static_cast<int32_t>(minhash_.size()) > maxSize_) {
    auto maxIt = std::prev(minhash_.end());
    if (maxIt != minhash_.end()) {
      decreaseTotalHllSize(maxIt->second);
      minhash_.erase(maxIt);
    }
  }
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::increaseTotalHllSize(
    const HllWrapper<TAllocator>& hll) {
  hllsTotalEstimatedSerializedSize_ += hll.serializedSize();
}

template <typename TAllocator>
void KHyperLogLog<TAllocator>::decreaseTotalHllSize(
    const HllWrapper<TAllocator>& hll) {
  hllsTotalEstimatedSerializedSize_ -= hll.serializedSize();
}

// Explicit instantiation for common allocator types
template class KHyperLogLog<HashStringAllocator>;
template class KHyperLogLog<memory::MemoryPool>;

} // namespace facebook::velox::common::hll
