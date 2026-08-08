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

#include <cstdint>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::functions {

/// Hive-compatible hash primitives, mirroring the hashCode implementation in
/// Hive's ObjectInspectorUtils (HIVE-12025). Shared between the Hive
/// connector's partition function and Spark's `hive_hash` scalar function so
/// both produce bit-identical bucket assignments.
class HiveHash {
 public:
  /// Folds 'oneHash' into 'aggregateHash'. When 'mix' is false the aggregate is
  /// replaced (used for the first column); otherwise it is combined as
  /// 31 * aggregateHash + oneHash (Hive's field-folding rule).
  static void mergeHash(bool mix, uint32_t oneHash, uint32_t& aggregateHash) {
    aggregateHash = mix ? aggregateHash * 31 + oneHash : oneHash;
  }

  /// Hashes a 64-bit value by xoring its high and low 32-bit words, matching
  /// Hive's LongWritable hashCode.
  static int32_t hashInt64(int64_t value) {
    return ((*reinterpret_cast<uint64_t*>(&value)) >> 32) ^ value;
  }

#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
  __attribute__((no_sanitize("integer")))
#endif
#endif
  static uint32_t hashBytes(StringView bytes, int32_t initialValue) {
    uint32_t hash = initialValue;
    auto* data = bytes.data();
    for (auto i = 0; i < bytes.size(); ++i) {
      hash = hash * 31 + *reinterpret_cast<const int8_t*>(data + i);
    }
    return hash;
  }

  /// Hashes a timestamp using Hive's (seconds << 30 | nanos) packing.
  static int32_t hashTimestamp(const Timestamp& ts) {
    return hashInt64((ts.getSeconds() << 30) | ts.getNanos());
  }

  /// Computes the Hive hash of a single non-null value of the given 'kind'.
  /// Throws VELOX_UNSUPPORTED for kinds outside the supported primitive set.
  template <TypeKind kind>
  static uint32_t hashOne(typename TypeTraits<kind>::NativeType value);

  /// Folds the Hive hash of every selected row of 'values' into 'hashes',
  /// treating nulls as hash 0. When 'mix' is false the per-row hash replaces
  /// the accumulator (first column); otherwise it is combined via mergeHash.
  template <TypeKind kind>
  static void hashPrimitive(
      const DecodedVector& values,
      const SelectivityVector& rows,
      bool mix,
      std::vector<uint32_t>& hashes);
};

template <TypeKind kind>
inline uint32_t HiveHash::hashOne(
    typename TypeTraits<kind>::NativeType /* value */) {
  VELOX_UNSUPPORTED(
      "Hive hash doesn't support {} type", TypeTraits<kind>::name);
  return 0; // Make compiler happy.
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::BOOLEAN>(bool value) {
  return value ? 1 : 0;
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::TINYINT>(int8_t value) {
  return static_cast<uint32_t>(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::SMALLINT>(int16_t value) {
  return static_cast<uint32_t>(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::INTEGER>(int32_t value) {
  return static_cast<uint32_t>(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::REAL>(float value) {
  return static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(&value));
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::BIGINT>(int64_t value) {
  return hashInt64(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::DOUBLE>(double value) {
  return hashInt64(*reinterpret_cast<const int64_t*>(&value));
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::VARCHAR>(StringView value) {
  return hashBytes(value, 0);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::VARBINARY>(StringView value) {
  return hashBytes(value, 0);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::TIMESTAMP>(Timestamp value) {
  return hashTimestamp(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::UNKNOWN>(UnknownValue /*value*/) {
  VELOX_FAIL("Unknown values cannot be non-NULL");
}

template <TypeKind kind>
inline void HiveHash::hashPrimitive(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes) {
  if (rows.isAllSelected()) {
    // The compiler seems to be a little fickle with optimizations.
    // Although rows.applyToSelected should do roughly the same thing, doing
    // this here along with assigning rows.size() to a variable seems to help
    // the compiler to inline hashOne showing a 50% performance improvement in
    // benchmarks.
    vector_size_t numRows = rows.size();
    for (auto i = 0; i < numRows; ++i) {
      const uint32_t hash = values.isNullAt(i)
          ? 0
          : hashOne<kind>(
                values.valueAt<typename TypeTraits<kind>::NativeType>(i));
      mergeHash(mix, hash, hashes[i]);
    }
  } else {
    rows.applyToSelected([&](auto row) INLINE_LAMBDA {
      const uint32_t hash = values.isNullAt(row)
          ? 0
          : hashOne<kind>(
                values.valueAt<typename TypeTraits<kind>::NativeType>(row));
      mergeHash(mix, hash, hashes[row]);
    });
  }
}

} // namespace facebook::velox::functions
