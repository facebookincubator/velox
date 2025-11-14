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
#include <string>

#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"

namespace facebook::velox::common::hll {

/// Wrapper for HyperLogLog that can hold either SparseHll or DenseHll.
/// Converts from sparse to dense representation when memory
/// limits are exceeded. Similar to HllAccumulator pattern.
template <typename TAllocator>
class HllWrapper {
 public:
  explicit HllWrapper(int8_t indexBitLength, TAllocator* allocator)
      : isSparse_(true),
        indexBitLength_(indexBitLength),
        allocator_(allocator),
        sparseHll_(allocator),
        denseHll_(allocator) {
    // Set soft memory limit for sparse HLL to convert to dense when exceeded.
    sparseHll_.setSoftMemoryLimit(
        DenseHlls::estimateInMemorySize(indexBitLength_));
  }

  /// Creates an HllWrapper instance from serialized data.
  static std::unique_ptr<HllWrapper> deserialize(
      const char* data,
      TAllocator* allocator) {
    if (SparseHlls::canDeserialize(data)) {
      int8_t indexBitLength = SparseHlls::deserializeIndexBitLength(data);
      auto wrapper = std::make_unique<HllWrapper>(indexBitLength, allocator);
      wrapper->sparseHll_ = SparseHll<TAllocator>(data, allocator);
      wrapper->sparseHll_.setSoftMemoryLimit(
          DenseHlls::estimateInMemorySize(indexBitLength));
      return wrapper;
    } else if (DenseHlls::canDeserialize(data)) {
      int8_t indexBitLength = DenseHlls::deserializeIndexBitLength(data);
      auto wrapper = std::make_unique<HllWrapper>(indexBitLength, allocator);
      wrapper->denseHll_ = DenseHll<TAllocator>(data, allocator);
      wrapper->isSparse_ = false;
      return wrapper;
    } else {
      VELOX_FAIL("Cannot deserialize HyperLogLog");
    }
  }

  /// Inserts a hash value into the HyperLogLog.
  /// Converts from sparse to dense representation if memory
  /// limit is exceeded.
  void insertHash(uint64_t hash) {
    if (isSparse_) {
      // insertHash returns true if soft memory limit exceeded
      if (sparseHll_.insertHash(hash)) {
        toDense();
      }
    } else {
      denseHll_.insertHash(hash);
    }
  }

  /// Returns the estimated cardinality from the underlying HLL.
  int64_t cardinality() const {
    return isSparse_ ? sparseHll_.cardinality() : denseHll_.cardinality();
  }

  /// Returns the size of the serialized representation.
  int32_t serializedSize() const {
    return isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();
  }

  /// Serializes the HLL state to the provided buffer.
  /// The buffer must have at least serializedSize() bytes available.
  void serialize(char* output) {
    if (isSparse_) {
      sparseHll_.serialize(indexBitLength_, output);
    } else {
      denseHll_.serialize(output);
    }
  }

  /// Merges another HllWrapper into this one.
  /// Automatically converts to dense representation if necessary.
  void mergeWith(const HllWrapper& other) {
    if (isSparse_ && other.isSparse_) {
      sparseHll_.mergeWith(other.sparseHll_);
      if (sparseHll_.overLimit()) {
        toDense();
      }
    } else {
      // At least one is dense - convert both to dense.
      if (isSparse_) {
        toDense();
      }
      if (other.isSparse_) {
        other.sparseHll_.toDense(denseHll_);
      } else {
        denseHll_.mergeWith(other.denseHll_);
      }
    }
  }

 private:
  void toDense() {
    if (!isSparse_) {
      return;
    }
    isSparse_ = false;
    denseHll_.initialize(indexBitLength_);
    sparseHll_.toDense(denseHll_);
    sparseHll_.reset();
  }

  bool isSparse_;
  int8_t indexBitLength_;
  TAllocator* allocator_;
  SparseHll<TAllocator> sparseHll_;
  DenseHll<TAllocator> denseHll_;
};

/// See "KHyperLogLog: EstimatingReidentifiability and Joinability of Large Data
/// at Scale" by Chia et al., 2019.
/// https://research.google/pubs/khyperloglog-estimating-reidentifiability-and-joinability-of-large-data-at-scale/
template <typename TAllocator>
class KHyperLogLog {
 public:
  template <typename U>
  using TStlAllocator = typename TAllocator::template TStlAllocator<U>;

  static constexpr int kDefaultHllBuckets = 256;
  static constexpr int kDefaultMaxSize = 4096;
  static constexpr int64_t kDefaultHistogramSize = 256;

  explicit KHyperLogLog(TAllocator* allocator)
      : KHyperLogLog(kDefaultMaxSize, kDefaultHllBuckets, allocator) {}

  KHyperLogLog(int maxSize, int hllBuckets, TAllocator* allocator)
      : maxSize_(maxSize),
        hllBuckets_(hllBuckets),
        allocator_(allocator),
        minhash_(
            TStlAllocator<std::pair<const int64_t, HllWrapper<TAllocator>>>(
                allocator)) {}

  /// Creates a KHyperLogLog instance from serialized data.
  static std::unique_ptr<KHyperLogLog<TAllocator>>
  deserialize(const char* data, size_t size, TAllocator* allocator);

  /// Serializes the KHyperLogLog state to a string.
  /// Note: Not const because DenseHll::serialize() may reorder internal
  /// overflow arrays for deterministic serialization.
  std::string serialize();

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
  /// Both instances must be in exact mode (isExact() == true).
  static int64_t exactIntersectionCardinality(
      const KHyperLogLog<TAllocator>& khll1,
      const KHyperLogLog<TAllocator>& khll2);

  /// Computes the Jaccard index (similarity coefficient) between two
  /// KHyperLogLogs. Uses the min-hash approach: looks at the first
  /// min(|khll1.minhash|, |khll2.minhash|) entries in the sorted union of keys
  /// and calculates the proportion that appear in both sets.
  static double jaccardIndex(
      const KHyperLogLog<TAllocator>& khll1,
      const KHyperLogLog<TAllocator>& khll2);

  /// Merges two KHyperLogLog instances into a new instance.
  /// Uses the one with the smaller maxSize as the base to avoid losing
  /// resolution.
  static std::unique_ptr<KHyperLogLog<TAllocator>> merge(
      KHyperLogLog<TAllocator>& khll1,
      KHyperLogLog<TAllocator>& khll2);

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
  std::map<int64_t, double> uniquenessDistribution(int64_t histogramSize) const;

  /// Merges another KHyperLogLog instance into this one (modifies this
  /// instance).
  void mergeWith(const KHyperLogLog<TAllocator>& other);

 private:
  void update(int64_t hash, int64_t uii);

  void removeOverflowEntries();

  void increaseTotalHllSize(const HllWrapper<TAllocator>& hll);

  void decreaseTotalHllSize(const HllWrapper<TAllocator>& hll);

  int64_t maxKey() const {
    if (minhash_.empty()) {
      return INT64_MIN;
    }
    return minhash_.rbegin()->first;
  }

  int32_t maxSize_;
  int32_t hllBuckets_;
  TAllocator* allocator_;

  std::map<
      int64_t,
      HllWrapper<TAllocator>,
      std::less<int64_t>,
      TStlAllocator<std::pair<const int64_t, HllWrapper<TAllocator>>>>
      minhash_;

  size_t hllsTotalEstimatedSerializedSize_{0};
};

} // namespace facebook::velox::common::hll
