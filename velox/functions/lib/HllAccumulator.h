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

#define XXH_INLINE_ALL

#include <xxhash.h>
#include <cmath>

#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/Murmur3Hash128.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::common::hll {

namespace detail {
template <typename T, bool HllAsFinalResult>
inline uint64_t hashOne(const T& value) {
  if constexpr (std::is_same_v<T, Timestamp>) {
    return hashOne<int64_t, HllAsFinalResult>(value.toMillis());
  }
  if constexpr (HllAsFinalResult) {
    if constexpr (std::is_same_v<T, int64_t>) {
      return common::hll::Murmur3Hash128::hash64ForLong(value, 0);
    } else if constexpr (std::is_same_v<T, double>) {
      return common::hll::Murmur3Hash128::hash64ForLong(
          *reinterpret_cast<const int64_t*>(&value), 0);
    }
    return common::hll::Murmur3Hash128::hash64(&value, sizeof(T), 0);
  } else {
    return XXH64(&value, sizeof(T), 0);
  }
}

template <>
inline uint64_t hashOne<StringView, false>(const StringView& value) {
  return XXH64(value.data(), value.size(), 0);
}

template <>
inline uint64_t hashOne<StringView, true>(const StringView& value) {
  return common::hll::Murmur3Hash128::hash64(value.data(), value.size(), 0);
}

} // namespace detail

template <
    typename T,
    bool HllAsFinalResult,
    typename TAllocator = HashStringAllocator>
struct HllAccumulator {
  explicit HllAccumulator(TAllocator* allocator)
      : sparseHll_{allocator}, denseHll_{allocator} {}

  explicit HllAccumulator(int8_t indexBitLength, TAllocator* allocator)
      : isSparse_(true),
        indexBitLength_(indexBitLength),
        sparseHll_(allocator),
        denseHll_(allocator) {
    // Set soft memory limit for sparse HLL to convert to dense when exceeded.
    sparseHll_.setSoftMemoryLimit(
        DenseHlls::estimateInMemorySize(indexBitLength_));
  }

  void setIndexBitLength(int8_t indexBitLength) {
    indexBitLength_ = indexBitLength;
    sparseHll_.setSoftMemoryLimit(
        DenseHlls::estimateInMemorySize(indexBitLength_));
  }

  /// Creates an HllAccumulator instance from serialized data.
  static std::unique_ptr<HllAccumulator> deserialize(
      const char* data,
      TAllocator* allocator) {
    if (SparseHlls::canDeserialize(data)) {
      int8_t indexBitLength = SparseHlls::deserializeIndexBitLength(data);
      auto wrapper =
          std::make_unique<HllAccumulator>(indexBitLength, allocator);
      wrapper->sparseHll_ = SparseHll<TAllocator>(data, allocator);
      wrapper->sparseHll_.setSoftMemoryLimit(
          DenseHlls::estimateInMemorySize(indexBitLength));
      return wrapper;
    } else if (DenseHlls::canDeserialize(data)) {
      int8_t indexBitLength = DenseHlls::deserializeIndexBitLength(data);
      auto wrapper =
          std::make_unique<HllAccumulator>(indexBitLength, allocator);
      wrapper->denseHll_ = DenseHll<TAllocator>(data, allocator);
      wrapper->isSparse_ = false;
      return wrapper;
    } else {
      VELOX_FAIL("Cannot deserialize HyperLogLog");
    }
  }

  void append(T value) {
    const auto hash = detail::hashOne<T, HllAsFinalResult>(value);
    insertHash(hash);
  }

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

  int64_t cardinality() const {
    return isSparse_ ? sparseHll_.cardinality() : denseHll_.cardinality();
  }

  void mergeWith(StringView serialized, TAllocator* allocator) {
    auto input = serialized.data();
    if (indexBitLength_ < 0) {
      // deserializeIndexBitLength is the same between Dense and Sparse HLL
      setIndexBitLength(DenseHlls::deserializeIndexBitLength(input));
    }

    if (SparseHlls::canDeserialize(input)) {
      SparseHll<TAllocator> other{input, allocator};
      mergeWithSparse(other);
    } else if (DenseHlls::canDeserialize(input)) {
      DenseHll<TAllocator> other{input, allocator};
      mergeWithDense(other);
    } else {
      VELOX_USER_FAIL("Unexpected type of HLL");
    }
  }

  void mergeWith(const HllAccumulator& other) {
    if (indexBitLength_ < 0) {
      setIndexBitLength(other.indexBitLength_);
    }
    if (other.isSparse_) {
      mergeWithSparse(other.sparseHll_);
    } else {
      mergeWithDense(other.denseHll_);
    }
  }

  int32_t serializedSize() {
    return isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();
  }

  void serialize(char* outputBuffer) {
    return isSparse_ ? sparseHll_.serialize(indexBitLength_, outputBuffer)
                     : denseHll_.serialize(outputBuffer);
  }

  bool isSparse() const {
    return isSparse_;
  }

 private:
  void toDense() {
    isSparse_ = false;
    denseHll_.initialize(indexBitLength_);
    sparseHll_.toDense(denseHll_);
    sparseHll_.reset();
  }

  void mergeWithSparse(const SparseHll<TAllocator>& other) {
    if (isSparse_) {
      sparseHll_.mergeWith(other);
      if (sparseHll_.overLimit()) {
        toDense();
      }
    } else {
      other.toDense(denseHll_);
    }
  }

  void mergeWithDense(const DenseHll<TAllocator>& other) {
    if (isSparse_) {
      toDense();
    }
    VELOX_USER_CHECK_EQ(
        indexBitLength_,
        other.indexBitLength(),
        "Cannot merge HLLs with different number of buckets");
    denseHll_.mergeWith(other);
  }

  bool isSparse_{true};
  int8_t indexBitLength_{-1};
  SparseHll<TAllocator> sparseHll_;
  DenseHll<TAllocator> denseHll_;
};

template <>
struct HllAccumulator<bool, false> {
  explicit HllAccumulator(HashStringAllocator* /*allocator*/) {}

  void append(bool value) {
    approxDistinctState_ |= (1 << value);
  }

  int64_t cardinality() const {
    return (approxDistinctState_ & 1) + ((approxDistinctState_ & 2) >> 1);
  }

  void mergeWith(
      StringView /*serialized*/,
      HashStringAllocator* /*allocator*/) {
    VELOX_UNREACHABLE(
        "APPROX_DISTINCT<BOOLEAN> unsupported mergeWith(StringView, HashStringAllocator*)");
  }

  void mergeWith(int8_t data) {
    approxDistinctState_ |= data;
  }

  int32_t serializedSize() const {
    return sizeof(int8_t);
  }

  void serialize(char* /*outputBuffer*/) {
    VELOX_UNREACHABLE("APPROX_DISTINCT<BOOLEAN> unsupported serialize(char*)");
  }

  void setIndexBitLength(int8_t /*indexBitLength*/) {}

  int8_t getState() const {
    return approxDistinctState_;
  }

 private:
  int8_t approxDistinctState_{0};
};
} // namespace facebook::velox::common::hll
