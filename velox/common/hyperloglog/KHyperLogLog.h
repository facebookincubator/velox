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
#include "velox/common/hyperloglog/HllAccumulator.h"

namespace facebook::velox::common::hll {

/// See "KHyperLogLog: EstimatingReidentifiability and Joinability of Large Data
/// at Scale" by Chia et al., 2019.
/// https://research.google/pubs/khyperloglog-estimating-reidentifiability-and-joinability-of-large-data-at-scale/
template <typename TAllocator>
class KHyperLogLog {
 public:
  template <typename U>
  using TStlAllocator = typename TAllocator::template TStlAllocator<U>;

  static constexpr int32_t kDefaultHllBuckets = 256;
  static constexpr int32_t kDefaultMaxSize = 4096;
  static constexpr int64_t kDefaultHistogramSize = 256;

  explicit KHyperLogLog(TAllocator* allocator)
      : KHyperLogLog(kDefaultMaxSize, kDefaultHllBuckets, allocator) {}

  KHyperLogLog(int maxSize, int hllBuckets, TAllocator* allocator)
      : maxSize_(maxSize),
        hllBuckets_(hllBuckets),
        allocator_(allocator),
        minhash_(
            TStlAllocator<std::pair<
                const int64_t,
                HllAccumulator<int64_t, true, TAllocator>>>(allocator)) {
    VELOX_CHECK(
        hllBuckets > 0 && (hllBuckets & (hllBuckets - 1)) == 0,
        "hllBuckets must be a power of 2");
    // numBuckets = 2^bits
    indexBitLength_ = static_cast<int8_t>(std::log2(hllBuckets));
  }

  /// Creates a KHyperLogLog instance from serialized data.
  static std::unique_ptr<KHyperLogLog<TAllocator>>
  deserialize(const char* data, size_t size, TAllocator* allocator);

  /// Serializes the KHyperLogLog state to a varbinary.
  /// The caller must ensure there is enough space at the buffer location, using
  /// estimatedSerializedSize(). Note: Not const because DenseHll::serialize()
  /// may reorder internal overflow arrays for deterministic serialization.
  void serialize(char* outputBuffer);

  /// Returns an estimate of the size in bytes of the serialized
  /// representation.
  size_t estimatedSerializedSize() const;

  /// Adds an integer join_key with its associated User Identifying Information
  /// (UII) to the existing KHLL.
  void add(int64_t joinKey, int64_t uii);

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
      const KHyperLogLog<TAllocator>& left,
      const KHyperLogLog<TAllocator>& right);

  /// Computes the Jaccard index (similarity coefficient) between two
  /// KHyperLogLogs. Uses the min-hash approach: looks at the first
  /// min(|left.minhash|, |right.minhash|) entries in the sorted union of keys
  /// and calculates the proportion that appear in both sets.
  static double jaccardIndex(
      const KHyperLogLog<TAllocator>& left,
      const KHyperLogLog<TAllocator>& right);

  /// Merges two KHyperLogLog instances into a new instance.
  /// Uses the one with the smaller maxSize as the base to avoid losing
  /// resolution.
  static std::unique_ptr<KHyperLogLog<TAllocator>> merge(
      KHyperLogLog<TAllocator>& left,
      KHyperLogLog<TAllocator>& right);

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
  void mergeWith(const KHyperLogLog<TAllocator>& other);

 private:
  void update(int64_t hash, int64_t uii);

  void removeOverflowEntries();

  void increaseTotalHllSize(HllAccumulator<int64_t, true, TAllocator>& hll);

  void decreaseTotalHllSize(HllAccumulator<int64_t, true, TAllocator>& hll);

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
      HllAccumulator<int64_t, true, TAllocator>,
      std::less<int64_t>,
      TStlAllocator<
          std::pair<const int64_t, HllAccumulator<int64_t, true, TAllocator>>>>
      minhash_;

  size_t hllsTotalEstimatedSerializedSize_{0};
};

} // namespace facebook::velox::common::hll
