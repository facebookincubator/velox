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

#include <map>
#include <memory>

#include <folly/container/F14Map.h>
#include "velox/common/base/Status.h"
#include "velox/functions/lib/HllAccumulator.h"

namespace facebook::velox::common::hll {

/// See "KHyperLogLog: EstimatingReidentifiability and Joinability of Large Data
/// at Scale" by Chia et al., 2019.
/// https://research.google/pubs/khyperloglog-estimating-reidentifiability-and-joinability-of-large-data-at-scale/
template <typename TUii, typename TAllocator>
class KHyperLogLog {
 public:
  template <typename U>
  using TStlAllocator = typename TAllocator::template TStlAllocator<U>;

  static constexpr int32_t kDefaultHllBuckets = 256;
  static constexpr int32_t kDefaultMaxSize = 4096;

  explicit KHyperLogLog(TAllocator* allocator)
      : KHyperLogLog(kDefaultMaxSize, kDefaultHllBuckets, allocator) {}

  KHyperLogLog(int maxSize, int hllBuckets, TAllocator* allocator)
      : maxSize_(maxSize),
        hllBuckets_(hllBuckets),
        allocator_(allocator),
        minhash_(
            TStlAllocator<std::pair<
                const int64_t,
                HllAccumulator<TUii, true, TAllocator>>>(allocator)) {
    VELOX_CHECK(
        hllBuckets > 0 && (hllBuckets & (hllBuckets - 1)) == 0,
        "hllBuckets must be a power of 2");
    // numBuckets = 2^bits
    indexBitLength_ = static_cast<int8_t>(std::log2(hllBuckets));
  }

  /// Creates a KHyperLogLog instance from serialized data.
  /// Returns an error status if the data is invalid.
  static Expected<std::unique_ptr<KHyperLogLog<TUii, TAllocator>>>
  deserialize(const char* data, size_t size, TAllocator* allocator);

  /// Serializes the KHyperLogLog state to a varbinary.
  /// The caller must ensure there is enough space at the buffer location, using
  /// estimatedSerializedSize(). Note: Not const because DenseHll::serialize()
  /// may reorder internal overflow arrays for deterministic serialization.
  void serialize(char* outputBuffer);

  /// Returns an estimate of the size in bytes of the serialized
  /// representation.
  size_t estimatedSerializedSize() const;

  /// Convert JoinKey to int64_t, hash, then add with its associated User
  /// Identifying Information (UII) to the existing KHLL.
  template <typename TJoinKey>
  void add(TJoinKey joinKey, TUii uii);

  /// Returns the estimated cardinality (number of distinct join keys).
  /// When isExact() is true, returns exact count. Otherwise, uses min-hash
  /// density estimation to extrapolate to the full hash space.
  int64_t cardinality() const;

  /// Returns whether the KHyperLogLog is in exact mode.
  /// Exact mode means the number of distinct values is less than maxSize,
  /// so all values are being tracked exactly without approximation.
  bool isExact() const;

  size_t minhashSize() const;

  /// Counts the exact number of common join keys between two KHyperLogLogs.
  /// Both instances must be in exact mode (isExact() == true), otherwise
  /// throws.
  static int64_t exactIntersectionCardinality(
      const KHyperLogLog<TUii, TAllocator>& left,
      const KHyperLogLog<TUii, TAllocator>& right);

  /// Computes the Jaccard index (similarity coefficient) between two
  /// KHyperLogLogs. Uses the min-hash approach: looks at the first
  /// min(|left.minhash|, |right.minhash|) entries in the sorted union of keys
  /// and calculates the proportion that appear in both sets.
  static double jaccardIndex(
      const KHyperLogLog<TUii, TAllocator>& left,
      const KHyperLogLog<TUii, TAllocator>& right);

  /// Merges two KHyperLogLog instances into a new instance.
  /// Uses the one with the smaller maxSize as the base to avoid losing
  /// resolution. Takes ownership of the input unique_ptrs and returns one
  /// of them to avoid copying.
  static std::unique_ptr<KHyperLogLog<TUii, TAllocator>> merge(
      std::unique_ptr<KHyperLogLog<TUii, TAllocator>> left,
      std::unique_ptr<KHyperLogLog<TUii, TAllocator>> right);

  /// Calculates the reidentification potential, which is the proportion of
  /// values with cardinality (number of distinct UIIs) at or below the given
  /// threshold. A higher value indicates more values are highly unique and
  /// could potentially be used to reidentify individuals.
  double reidentificationPotential(int64_t threshold) const;

  /// Returns a histogram of the uniqueness distribution with specified number
  /// of buckets. For each value, determines its cardinality (number of distinct
  /// UIIs) and increments the corresponding bucket. For hash values larger than
  /// the histogram size, the cardinality will be added to the last bucket at
  /// histogramSize.
  folly::F14FastMap<int64_t, double> uniquenessDistribution(
      int64_t histogramSize) const;

  /// Merges another KHyperLogLog instance into this one (modifies this
  /// instance).
  void mergeWith(const KHyperLogLog<TUii, TAllocator>& other);

  /// Merges another serialized KHyperLogLog into this one.
  /// The serialized data is assumed to be valid.
  void mergeWith(StringView serialized, TAllocator* allocator) {
    auto other = common::hll::KHyperLogLog<TUii, TAllocator>::deserialize(
        serialized.data(), serialized.size(), allocator);
    VELOX_CHECK(other.hasValue(), "Failed to deserialize KHyperLogLog");
    mergeWith(*other.value());
  }

 private:
  void update(int64_t hash, TUii uii);

  void removeOverflowEntries();

  void increaseTotalHllSize(HllAccumulator<TUii, true, TAllocator>& hll);

  void decreaseTotalHllSize(HllAccumulator<TUii, true, TAllocator>& hll);

  int64_t maxKey() const {
    if (minhash_.empty()) {
      return INT64_MIN;
    }
    return minhash_.rbegin()->first;
  }

  int32_t maxSize_;
  int32_t hllBuckets_;
  int8_t indexBitLength_;
  TAllocator* allocator_;

  std::map<
      int64_t,
      HllAccumulator<TUii, true, TAllocator>,
      std::less<int64_t>,
      TStlAllocator<
          std::pair<const int64_t, HllAccumulator<TUii, true, TAllocator>>>>
      minhash_;

  size_t hllsTotalEstimatedSerializedSize_{0};
};

} // namespace facebook::velox::common::hll

#include "velox/functions/lib/KHyperLogLogImpl.h"
