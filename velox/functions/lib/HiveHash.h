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

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::functions {

/// Computes Hive-compatible hashes, mirroring the hashCode implementation in
/// Hive's ObjectInspectorUtils (HIVE-12025), with optional zero normalization.
class HiveHash {
 public:
  /// Controls whether both floating-point signed zeros hash to zero.
  enum class ZeroMode {
    /// Preserves signed zero, matching Hive.
    kPreserve,
    /// Hashes both signs of zero as zero, matching Spark.
    kNormalize,
  };

  /// Pools must outlive this instance.
  HiveHash(
      std::vector<std::unique_ptr<std::vector<uint32_t>>>& hashesPool,
      std::vector<std::unique_ptr<SelectivityVector>>& rowsPool,
      std::vector<std::unique_ptr<DecodedVector>>& decodedVectorsPool,
      ZeroMode zeroMode = ZeroMode::kPreserve)
      : decodedVectorsPool_(decodedVectorsPool),
        rowsPool_(rowsPool),
        hashesPool_(hashesPool),
        zeroMode_(zeroMode) {}

  /// Folds selected values into 'hashes' using Hive's recursive hashing rules.
  /// Applies the instance's zero mode at every nesting depth.
  /// Supports existing Hive hash primitives, arrays, maps, and rows. Nulls
  /// contribute zero. Replaces the accumulator when 'mix' is false; otherwise
  /// combines it as 31 * accumulator + valueHash. Requires 'hashes' to cover
  /// 'rows.end()'. Scratch buffers below 'poolIndex' are reserved for the
  /// caller. Concurrent evaluations must use separate pools.
  void hash(
      const DecodedVector& values,
      const SelectivityVector& rows,
      bool mix,
      std::vector<uint32_t>& hashes,
      size_t poolIndex = 0);

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

  /// Hashes the bytes of 'bytes' as 31 * hash + byte starting from
  /// 'initialValue', matching Hive's Text/BytesWritable hashCode. Bytes are
  /// interpreted as signed, as in Java.
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
  /// Preserves signed zero and canonicalizes NaNs as Java does.
  /// Throws VELOX_UNSUPPORTED for kinds outside the supported primitive set.
  template <TypeKind kind>
  static uint32_t hashOne(typename TypeTraits<kind>::NativeType value);

  /// Folds the Hive hash of every selected row of 'values' into 'hashes',
  /// treating nulls as hash 0 and applying 'kZeroMode' to floating-point
  /// values. When 'mix' is false the per-row hash replaces the accumulator
  /// (first column); otherwise it is combined via mergeHash.
  template <TypeKind kind, ZeroMode kZeroMode = ZeroMode::kPreserve>
  static void hashPrimitive(
      const DecodedVector& values,
      const SelectivityVector& rows,
      bool mix,
      std::vector<uint32_t>& hashes);

 private:
  // Hashes a type using scratch buffers starting at 'poolIndex'.
  template <TypeKind kind>
  void hashTyped(
      const DecodedVector& values,
      const SelectivityVector& rows,
      bool mix,
      std::vector<uint32_t>& hashes,
      size_t poolIndex);

  // Retrieves a reusable decoder whose address remains stable during recursion.
  DecodedVector& getDecodedVector(size_t poolIndex);

  // Retrieves a reusable selection for referenced container elements.
  SelectivityVector& getRows(size_t poolIndex);

  // Retrieves reusable per-element hash storage.
  std::vector<uint32_t>& getHashes(size_t poolIndex);

  // Keeps scratch objects at stable addresses as recursion grows the pools.
  std::vector<std::unique_ptr<DecodedVector>>& decodedVectorsPool_;

  // Tracks selected child positions at each recursion level.
  std::vector<std::unique_ptr<SelectivityVector>>& rowsPool_;

  // Stores intermediate child hashes at each recursion level.
  std::vector<std::unique_ptr<std::vector<uint32_t>>>& hashesPool_;

  // Controls signed-zero hashing at every nesting depth.
  ZeroMode zeroMode_;
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
  // Match Hive's Float.floatToIntBits call in ObjectInspectorUtils.hashCode.
  // https://github.com/apache/hive/blob/ba83fd7bff/serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java
  return std::isnan(value) ? 0x7fc00000U : std::bit_cast<uint32_t>(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::BIGINT>(int64_t value) {
  return hashInt64(value);
}

template <>
inline uint32_t HiveHash::hashOne<TypeKind::DOUBLE>(double value) {
  // Match Hive's Double.doubleToLongBits call in ObjectInspectorUtils.hashCode.
  // https://github.com/apache/hive/blob/ba83fd7bff/serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java
  const auto bits = std::isnan(value) ? int64_t{0x7ff8000000000000}
                                      : std::bit_cast<int64_t>(value);
  return hashInt64(bits);
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

template <TypeKind kind, HiveHash::ZeroMode kZeroMode>
inline void HiveHash::hashPrimitive(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes) {
  auto hashRow = [&](auto row) INLINE_LAMBDA -> uint32_t {
    if (values.isNullAt(row)) {
      return 0;
    }
    const auto value =
        values.valueAt<typename TypeTraits<kind>::NativeType>(row);
    if constexpr (
        (kind == TypeKind::REAL || kind == TypeKind::DOUBLE) &&
        kZeroMode == ZeroMode::kNormalize) {
      if (value == 0) {
        return 0;
      }
    }
    return hashOne<kind>(value);
  };
  if (rows.isAllSelected()) {
    // The compiler seems to be a little fickle with optimizations.
    // Although rows.applyToSelected should do roughly the same thing, doing
    // this here along with assigning rows.size() to a variable seems to help
    // the compiler to inline hashOne showing a 50% performance improvement in
    // benchmarks.
    vector_size_t numRows = rows.size();
    for (auto i = 0; i < numRows; ++i) {
      mergeHash(mix, hashRow(i), hashes[i]);
    }
  } else {
    rows.applyToSelected([&](auto row) INLINE_LAMBDA {
      mergeHash(mix, hashRow(row), hashes[row]);
    });
  }
}

} // namespace facebook::velox::functions
