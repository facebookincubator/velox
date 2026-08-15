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

#include "velox/common/base/Status.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/type/StringView.h"

#include <cstdint>
#include <map>
#include <type_traits>

namespace facebook::velox::functions {

/// SetDigest is a probabilistic data structure for estimating set cardinality
/// and performing set operations. It combines HyperLogLog for cardinality
/// estimation with MinHash for exact counting and intersection operations.
///
/// Template parameters:
/// - T: The element type, which must be either int64_t or StringView.
template <typename T>
class SetDigest {
  static_assert(
      std::is_same_v<T, int64_t> || std::is_same_v<T, StringView>,
      "SetDigest only supports int64_t or StringView");

 public:
  template <typename U>
  using TStlAllocator = HashStringAllocator::TStlAllocator<U>;

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

  /// Add a new value to the digest.
  /// @param value The value to add.
  void add(T value);

  /// Merge this digest with another digest.
  /// @param other The other digest to merge into this one.
  void mergeWith(const SetDigest& other);

  /// Returns true if the digest is exact (cardinality < maxHashes).
  /// When exact, cardinality() returns the exact count.
  /// Used in exactIntersectionCardinality().
  bool isExact() const;

  /// Returns the estimated cardinality of the set.
  /// If isExact() is true, returns the exact count.
  /// Otherwise, returns HyperLogLog estimate.
  int64_t cardinality() const;

  /// Calculate the exact intersection cardinality between two digests.
  /// Both digests must be exact (isExact() == true).
  /// @param left First digest (must be exact).
  /// @param right Second digest (must be exact).
  /// @return The exact number of elements in the intersection.
  static int64_t exactIntersectionCardinality(
      const SetDigest& left,
      const SetDigest& right);

  /// Compute the Jaccard index (similarity coefficient) between two digests.
  /// Uses MinHash estimation. Returns a value in [0, 1] where 1 means
  /// identical sets and 0 means disjoint sets.
  /// @param left First digest.
  /// @param right Second digest.
  /// @return Estimated Jaccard index.
  static double jaccardIndex(const SetDigest& left, const SetDigest& right);

  /// Calculate the size needed for serialization.
  /// @return The number of bytes needed to serialize this digest.
  int32_t estimatedSerializedSize() const;

  /// Serialize the digest into bytes. The serialization is versioned and
  /// compatible with Presto Java.
  /// @param out Pre-allocated memory at least estimatedSerializedSize() in
  /// size.
  void serialize(char* out) const;

  /// Deserialize a SetDigest from serialized input.
  /// Serialization produced by Presto Java can be used as input.
  /// @param data The input serialization.
  /// @param size The size of the serialization in bytes.
  /// @return Status::OK() on success, or an error Status on failure.
  Status deserialize(const char* data, int32_t size);

  /// Get a copy of the MinHash hash counts.
  /// Returns a map from hash value to count.
  /// @return A map of hash values to their counts.
  std::map<int64_t, int16_t> getHashCounts() const;

 private:
  void addHash(uint64_t hash);

  void increaseTotalHllSize();
  void decreaseTotalHllSize();

  using MinHashAllocator = TStlAllocator<std::pair<const int64_t, int16_t>>;
  std::map<int64_t, int16_t, std::less<int64_t>, MinHashAllocator> minhash_;

  common::hll::HllAccumulator<int64_t, true, HashStringAllocator>
      hllAccumulator_;
  int32_t maxHashes_{kDefaultMaxHashes};
  HashStringAllocator* allocator_;

  /// Tracks the HLL serialized size, updated when HLL changes.
  int32_t hllTotalEstimatedSerializedSize_{0};
};

} // namespace facebook::velox::functions

#include "velox/functions/lib/SetDigestImpl.h"
