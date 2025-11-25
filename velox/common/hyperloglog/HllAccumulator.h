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

// Use timestamp.toMillis() to compute hash value.
template <>
inline uint64_t hashOne<Timestamp, false>(const Timestamp& value) {
  return hashOne<int64_t, false>(value.toMillis());
}

template <>
inline uint64_t hashOne<Timestamp, true>(const Timestamp& /*value*/) {
  VELOX_UNREACHABLE("approx_set(timestamp) is not supported.");
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

template <typename T, bool HllAsFinalResult>
struct HllAccumulator {
  explicit HllAccumulator(HashStringAllocator* allocator)
      : sparseHll_{allocator}, denseHll_{allocator} {}

  void setIndexBitLength(int8_t indexBitLength) {
    indexBitLength_ = indexBitLength;
    sparseHll_.setSoftMemoryLimit(
        common::hll::DenseHlls::estimateInMemorySize(indexBitLength_));
  }

  void append(T value) {
    const auto hash = detail::hashOne<T, HllAsFinalResult>(value);

    if (isSparse_) {
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

  void mergeWith(StringView serialized, HashStringAllocator* allocator) {
    auto input = serialized.data();
    if (common::hll::SparseHlls::canDeserialize(input)) {
      if (isSparse_) {
        sparseHll_.mergeWith(input);
        if (indexBitLength_ < 0) {
          setIndexBitLength(
              common::hll::DenseHlls::deserializeIndexBitLength(input));
        }
        if (sparseHll_.overLimit()) {
          toDense();
        }
      } else {
        common::hll::SparseHll<> other{input, allocator};
        other.toDense(denseHll_);
      }
    } else if (common::hll::DenseHlls::canDeserialize(input)) {
      if (isSparse_) {
        if (indexBitLength_ < 0) {
          setIndexBitLength(
              common::hll::DenseHlls::deserializeIndexBitLength(input));
        }
        toDense();
      }
      denseHll_.mergeWith(input);
    } else {
      VELOX_USER_FAIL("Unexpected type of HLL");
    }
  }

  int32_t serializedSize() {
    return isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();
  }

  void serialize(char* outputBuffer) {
    return isSparse_ ? sparseHll_.serialize(indexBitLength_, outputBuffer)
                     : denseHll_.serialize(outputBuffer);
  }

 private:
  void toDense() {
    isSparse_ = false;
    denseHll_.initialize(indexBitLength_);
    sparseHll_.toDense(denseHll_);
    sparseHll_.reset();
  }

  bool isSparse_{true};
  int8_t indexBitLength_{-1};
  common::hll::SparseHll<> sparseHll_;
  common::hll::DenseHll<> denseHll_;
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
