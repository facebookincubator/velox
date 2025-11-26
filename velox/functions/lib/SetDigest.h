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
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/Murmur3Hash128.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/type/StringView.h"

#include <folly/Bits.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace facebook::velox::functions {

namespace {
constexpr int8_t kUncompressedFormat = 1;
} // namespace

/// SetDigest is a probabilistic data structure for estimating set cardinality
/// and performing set operations. It combines HyperLogLog for cardinality
/// estimation with MinHash for exact counting and intersection operations.
class SetDigest {
 public:
  static constexpr int32_t kDefaultMaxHashes = 8192;
  static constexpr int32_t kNumberOfBuckets = 2048;
  static constexpr int8_t kDefaultIndexBitLength = 11;

  /// Construct a SetDigest with specified parameters.
  /// @param allocator Memory allocator for internal data structures.
  /// @param indexBitLength Number of bits for HLL indexing (default 11 = 2048
  /// buckets). Must be in range (0, 16].
  /// @param maxHashes Maximum number of hashes to store in MinHash (default
  /// 8192). When exceeded, digest becomes approximate.
  explicit SetDigest(
      HashStringAllocator* allocator,
      int8_t indexBitLength = kDefaultIndexBitLength,
      int32_t maxHashes = kDefaultMaxHashes);

  /// Add a new 64-bit integer value to the digest.
  /// @param value The value to add.
  void add(int64_t value);

  /// Add a new string value to the digest.
  /// @param value The string value to add.
  void add(StringView value);

  /// Merge this digest with another digest.
  /// @param other The other digest to merge into this one.
  void mergeWith(const SetDigest& other);

  /// Returns true if the digest is exact (cardinality < maxHashes).
  /// When exact, cardinality() returns the exact count.
  bool isExact() const;

  /// Returns the estimated cardinality of the set.
  /// If isExact() is true, returns the exact count.
  /// Otherwise, returns HyperLogLog estimate.
  int64_t cardinality() const;

  /// Calculate the exact intersection cardinality between two digests.
  /// Both digests must be exact (isExact() == true).
  /// @param a First digest (must be exact).
  /// @param b Second digest (must be exact).
  /// @return The exact number of elements in the intersection.
  static int64_t exactIntersectionCardinality(
      const SetDigest& a,
      const SetDigest& b);

  /// Compute the Jaccard index (similarity coefficient) between two digests.
  /// Uses MinHash estimation. Returns a value in [0, 1] where 1 means
  /// identical sets and 0 means disjoint sets.
  /// @param a First digest.
  /// @param b Second digest.
  /// @return Estimated Jaccard index.
  static double jaccardIndex(const SetDigest& a, const SetDigest& b);

  /// Calculate the size needed for serialization.
  /// @return The number of bytes needed to serialize this digest.
  int32_t estimatedSerializedSize() const;

  /// Serialize the digest into bytes. The serialization is versioned and
  /// compatible with Presto Java.
  /// @param out Pre-allocated memory at least estimatedSerializedSize() in
  /// size.
  void serialize(char* out);

  /// Deserialize a SetDigest from serialized input.
  /// Serialization produced by Presto Java can be used as input.
  /// @param data The input serialization.
  /// @param size The size of the serialization in bytes.
  void deserialize(const char* data, int32_t size);

  /// Get a copy of the MinHash hash counts.
  /// Returns a map from hash value to count.
  /// @return A map of hash values to their counts.
  std::map<int64_t, int16_t> getHashCounts() const;

 private:
  void addHash(uint64_t hash);
  void convertToDense();

  using MinHashAllocator = StlAllocator<std::pair<const int64_t, int16_t>>;
  std::map<int64_t, int16_t, std::less<int64_t>, MinHashAllocator> minhash_;

  // HyperLogLog: starts as Sparse, converts to Dense when needed
  common::hll::SparseHll<> sparseHll_;
  common::hll::DenseHll<> denseHll_;
  bool isSparse_{true};
  int8_t indexBitLength_{11}; // Default to 2^11 = 2048 buckets
  int32_t maxHashes_{kDefaultMaxHashes};
  HashStringAllocator* allocator_;
};

inline SetDigest::SetDigest(
    HashStringAllocator* allocator,
    int8_t indexBitLength,
    int32_t maxHashes)
    : minhash_{MinHashAllocator(allocator)},
      sparseHll_{allocator},
      denseHll_{allocator},
      indexBitLength_{indexBitLength},
      maxHashes_{maxHashes},
      allocator_{allocator} {
  // Validate indexBitLength matches Java validation
  VELOX_CHECK_GT(indexBitLength, 0, "indexBitLength must be > 0");
  VELOX_CHECK_LE(indexBitLength, 16, "indexBitLength must be <= 16");

  // Validate maxHashes
  VELOX_CHECK_GT(maxHashes, 0, "maxHashes must be > 0");

  // Verify default indexBitLength matches kNumberOfBuckets
  static_assert(
      (1 << kDefaultIndexBitLength) == kNumberOfBuckets,
      "Default index bit length must match NUMBER_OF_BUCKETS");

  sparseHll_.setSoftMemoryLimit(
      common::hll::DenseHlls::estimateInMemorySize(indexBitLength_));
}

inline bool SetDigest::isExact() const {
  return static_cast<int32_t>(minhash_.size()) < maxHashes_;
}

inline void SetDigest::add(int64_t value) {
  uint64_t hash = common::hll::Murmur3Hash128::hash64ForLong(value, 0);

  addHash(hash);

  if (isSparse_) {
    if (sparseHll_.insertHash(hash)) {
      // insertHash returns true when over limit, convert to dense
      convertToDense();
    }
  } else {
    denseHll_.insertHash(hash);
  }
}

inline void SetDigest::add(StringView value) {
  uint64_t hash =
      common::hll::Murmur3Hash128::hash64(value.data(), value.size(), 0);

  addHash(hash);

  // Add to HyperLogLog
  if (isSparse_) {
    if (sparseHll_.insertHash(hash)) {
      convertToDense();
    }
  } else {
    denseHll_.insertHash(hash);
  }
}

inline void SetDigest::addHash(uint64_t hash) {
  auto it = minhash_.find(hash);
  int16_t currentCount = (it != minhash_.end()) ? it->second : 0;

  if (currentCount < std::numeric_limits<int16_t>::max()) {
    minhash_[hash] = static_cast<int16_t>(currentCount + 1);
  }

  while (static_cast<int32_t>(minhash_.size()) > maxHashes_) {
    auto lastIt = minhash_.end();
    --lastIt;
    minhash_.erase(lastIt);
  }
}

inline void SetDigest::convertToDense() {
  isSparse_ = false;
  denseHll_.initialize(indexBitLength_);
  sparseHll_.toDense(denseHll_);
  sparseHll_.reset();
}

namespace setdigest::detail {

static_assert(folly::kIsLittleEndian);

template <typename T>
void write(T value, char*& out) {
  folly::storeUnaligned(out, value);
  out += sizeof(T);
}

template <typename T>
void write(const T* values, int count, char*& out) {
  auto size = sizeof(T) * count;
  std::memcpy(out, values, size);
  out += size;
}

template <typename T>
T read(const char*& in) {
  T value = folly::loadUnaligned<T>(in);
  in += sizeof(T);
  return value;
}

} // namespace setdigest::detail

inline int32_t SetDigest::estimatedSerializedSize() const {
  int32_t hllSize =
      isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();

  return sizeof(int8_t) + // format byte
      sizeof(int32_t) + // HLL length
      hllSize + // HLL data
      sizeof(int32_t) + // maxHashes
      sizeof(int32_t) + // minhash size
      minhash_.size() * (sizeof(int64_t) + sizeof(int16_t)); // keys + values
}

inline void SetDigest::serialize(char* out) {
  const char* oldOut = out;

  setdigest::detail::write(kUncompressedFormat, out);

  int32_t hllSize =
      isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();
  setdigest::detail::write(hllSize, out);

  if (isSparse_) {
    sparseHll_.serialize(indexBitLength_, out);
  } else {
    denseHll_.serialize(out);
  }
  out += hllSize;

  setdigest::detail::write(maxHashes_, out);

  int32_t minhashSize = static_cast<int32_t>(minhash_.size());
  setdigest::detail::write(minhashSize, out);

  for (const auto& entry : minhash_) {
    setdigest::detail::write(entry.first, out);
  }

  for (const auto& entry : minhash_) {
    setdigest::detail::write(entry.second, out);
  }

  VELOX_CHECK_EQ(out - oldOut, estimatedSerializedSize());
}

inline void SetDigest::deserialize(const char* data, int32_t size) {
  VELOX_USER_CHECK_GE(
      size,
      sizeof(int8_t) + sizeof(int32_t),
      "Input too small to be a valid SetDigest");

  const char* in = data;

  int8_t format = setdigest::detail::read<int8_t>(in);
  VELOX_USER_CHECK_EQ(
      format, kUncompressedFormat, "Unexpected SetDigest format");

  // Read HLL length
  int32_t hllLength = setdigest::detail::read<int32_t>(in);
  VELOX_USER_CHECK_GE(
      hllLength, 0, "Invalid HLL length in SetDigest serialization");
  VELOX_USER_CHECK_LE(
      hllLength,
      size - (sizeof(int8_t) + sizeof(int32_t)),
      "HLL length exceeds input size");

  // Deserialize HLL - validate using canDeserialize before constructing.
  const char* hllData = in;

  if (common::hll::SparseHlls::canDeserialize(hllData)) {
    indexBitLength_ =
        common::hll::SparseHlls::deserializeIndexBitLength(hllData);
    sparseHll_.setSoftMemoryLimit(
        common::hll::DenseHlls::estimateInMemorySize(indexBitLength_));
    isSparse_ = true;
    sparseHll_ = common::hll::SparseHll<>(hllData, allocator_);
  } else if (common::hll::DenseHlls::canDeserialize(hllData)) {
    indexBitLength_ =
        common::hll::DenseHlls::deserializeIndexBitLength(hllData);
    isSparse_ = false;
    denseHll_ = common::hll::DenseHll<>(hllData, allocator_);
  } else {
    VELOX_USER_FAIL("Cannot deserialize SetDigest: invalid HLL data");
  }
  in += hllLength;

  maxHashes_ = setdigest::detail::read<int32_t>(in);

  int32_t minhashSize = setdigest::detail::read<int32_t>(in);

  std::vector<int64_t> keys;
  keys.reserve(minhashSize);
  for (int32_t i = 0; i < minhashSize; i++) {
    keys.push_back(setdigest::detail::read<int64_t>(in));
  }

  for (int32_t i = 0; i < minhashSize; i++) {
    int16_t count = setdigest::detail::read<int16_t>(in);
    minhash_[keys[i]] = count;
  }
}

inline void SetDigest::mergeWith(const SetDigest& other) {
  // Merge HyperLogLog
  if (other.isSparse_) {
    if (isSparse_) {
      sparseHll_.mergeWith(other.sparseHll_);
      if (sparseHll_.overLimit()) {
        convertToDense();
      }
    } else {
      other.sparseHll_.toDense(denseHll_);
    }
  } else {
    if (isSparse_) {
      convertToDense();
    }
    denseHll_.mergeWith(other.denseHll_);
  }

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
    auto lastIt = minhash_.end();
    --lastIt;
    minhash_.erase(lastIt);
  }
}

inline int64_t SetDigest::cardinality() const {
  if (isExact()) {
    return static_cast<int64_t>(minhash_.size());
  }
  return isSparse_ ? sparseHll_.cardinality() : denseHll_.cardinality();
}

inline int64_t SetDigest::exactIntersectionCardinality(
    const SetDigest& a,
    const SetDigest& b) {
  VELOX_USER_CHECK(
      a.isExact(), "exact intersection cannot operate on approximate sets");
  VELOX_USER_CHECK(
      b.isExact(), "exact intersection cannot operate on approximate sets");

  int64_t count = 0;
  for (const auto& entry : a.minhash_) {
    if (b.minhash_.count(entry.first)) {
      count++;
    }
  }
  return count;
}

inline double SetDigest::jaccardIndex(const SetDigest& a, const SetDigest& b) {
  int32_t sizeOfSmallerSet = std::min(
      static_cast<int32_t>(a.minhash_.size()),
      static_cast<int32_t>(b.minhash_.size()));

  if (sizeOfSmallerSet == 0) {
    return (a.minhash_.size() == 0 && b.minhash_.size() == 0) ? 1.0 : 0.0;
  }

  std::set<int64_t> minUnion;
  for (const auto& entry : a.minhash_) {
    minUnion.insert(entry.first);
  }
  for (const auto& entry : b.minhash_) {
    minUnion.insert(entry.first);
  }

  int32_t intersection = 0;
  int32_t i = 0;
  for (int64_t key : minUnion) {
    if (a.minhash_.count(key) && b.minhash_.count(key)) {
      intersection++;
    }
    i++;
    if (i >= sizeOfSmallerSet) {
      break;
    }
  }

  return static_cast<double>(intersection) /
      static_cast<double>(sizeOfSmallerSet);
}

inline std::map<int64_t, int16_t> SetDigest::getHashCounts() const {
  std::map<int64_t, int16_t> result;
  for (const auto& entry : minhash_) {
    result[entry.first] = entry.second;
  }
  return result;
}

} // namespace facebook::velox::functions
