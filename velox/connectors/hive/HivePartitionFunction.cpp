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
#include "velox/connectors/hive/HivePartitionFunction.h"
#include <bit>

namespace facebook::velox::connector::hive {

namespace {
template <TypeKind kind>
void hashTyped(
    const DecodedVector& /* values */,
    vector_size_t /* size */,
    bool /* mix */,
    std::vector<uint32_t>& /* hashes */) {
  VELOX_UNSUPPORTED(
      "Hive partitioning function doesn't support {} type",
      TypeTraits<kind>::name);
}

template <>
void hashTyped<TypeKind::BOOLEAN>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    const int32_t hash =
        (values.isNullAt(i)) ? 0 : (values.valueAt<bool>(i) ? 1 : 0);
    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::TINYINT>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    uint32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = static_cast<uint32_t>(values.valueAt<int8_t>(i));
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::SMALLINT>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    uint32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = static_cast<uint32_t>(values.valueAt<int16_t>(i));
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::INTEGER>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    uint32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = static_cast<uint32_t>(values.valueAt<int32_t>(i));
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::REAL>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
    static_assert(sizeof(float) == sizeof(uint32_t));
  for (auto i = 0; i < size; ++i) {
    uint32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
        auto val = values.valueAt<float>(i);
        memcpy(&hash, &val, sizeof hash);
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

int32_t hashInt64(int64_t value) {
  return ((*reinterpret_cast<uint64_t*>(&value)) >> 32) ^ value;
}

template <>
void hashTyped<TypeKind::DOUBLE>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
    static_assert(sizeof(float) == sizeof(uint32_t));
  for (auto i = 0; i < size; ++i) {
    int32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
        auto val_double = values.valueAt<double>(i);
        int64_t val_int64;
        memcpy(&val_int64, &val_double, sizeof val_int64);
        hash = hashInt64(val_int64);
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::BIGINT>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    const int32_t hash =
        (values.isNullAt(i)) ? 0 : hashInt64(values.valueAt<int64_t>(i));
    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
__attribute__((no_sanitize("integer")))
#endif
#endif
uint32_t
hashBytes(StringView bytes, int32_t initialValue) {
  uint32_t hash = initialValue;
  auto* data = bytes.data();
  for (auto i = 0; i < bytes.size(); ++i) {
    hash = hash * 31 + *reinterpret_cast<const int8_t*>(data + i);
  }
  return hash;
}

void hashTypedStringView(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    const uint32_t hash =
        (values.isNullAt(i)) ? 0 : hashBytes(values.valueAt<StringView>(i), 0);
    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::VARCHAR>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
    hashTypedStringView(values, size, mix, hashes);
}

template <>
void hashTyped<TypeKind::VARBINARY>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
    hashTypedStringView(values, size, mix, hashes);
}

int32_t hashTimestamp(Timestamp ts) {
    return hashInt64((ts.getSeconds() << 30) | ts.getNanos());
}

template <>
void hashTyped<TypeKind::TIMESTAMP>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    int32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = hashTimestamp(values.valueAt<Timestamp>(i));
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

template <>
void hashTyped<TypeKind::DATE>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    int32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = values.valueAt<Date>(i).days();
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

void hash(
    const DecodedVector& values,
    TypeKind typeKind,
    vector_size_t size,
    bool mix,
    std::vector<uint32_t>& hashes) {
  // This function mirrors the behavior of function hashCode in
  // HIVE-12025 ba83fd7bff
  // serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java
  // https://github.com/apache/hive/blob/ba83fd7bff/serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java

  // HIVE-7148 proposed change to bucketing hash algorithms. If that gets
  // implemented, this function will need to change significantly.

  VELOX_DYNAMIC_TYPE_DISPATCH(hashTyped, typeKind, values, size, mix, hashes);
}

void hashPrecomputed(
    uint32_t precomputedHash,
    vector_size_t numRows,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < numRows; ++i) {
    hashes[i] = mix ? hashes[i] * 31 + precomputedHash : precomputedHash;
  }
}
} // namespace

HivePartitionFunction::HivePartitionFunction(
    int numBuckets,
    std::vector<int> bucketToPartition,
    std::vector<ChannelIndex> keyChannels,
    const std::vector<std::shared_ptr<BaseVector>>& constValues)
    : numBuckets_{numBuckets},
      bucketToPartition_{bucketToPartition},
      keyChannels_{std::move(keyChannels)} {
  decodedVectors_.resize(keyChannels_.size());
  precomputedHashes_.resize(keyChannels_.size());
  size_t constChannel{0};
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    if (keyChannels_[i] == kConstantChannel) {
      precompute(*(constValues[constChannel++]), i);
    }
  }
}

void HivePartitionFunction::partition(
    const RowVector& input,
    std::vector<uint32_t>& partitions) {
  const auto numRows = input.size();

  if (numRows > hashes_.size()) {
    rows_.resize(numRows);
    rows_.setAll();
    hashes_.resize(numRows);
  }

  partitions.resize(numRows);
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    if (keyChannels_[i] != kConstantChannel) {
      const auto& keyVector = input.childAt(keyChannels_[i]);
      decodedVectors_[i].decode(*keyVector, rows_);
      hash(
          decodedVectors_[i],
          keyVector->typeKind(),
          keyVector->size(),
          i > 0,
          hashes_);
    } else {
      hashPrecomputed(precomputedHashes_[i], numRows, i > 0, hashes_);
    }
  }

  static const int32_t kInt32Max = std::numeric_limits<int32_t>::max();

  for (auto i = 0; i < numRows; ++i) {
    partitions[i] =
        bucketToPartition_[((hashes_[i] & kInt32Max) % numBuckets_)];
  }
}

void HivePartitionFunction::precompute(
    const BaseVector& value,
    size_t channelIndex) {
  if (value.isNullAt(0)) {
    precomputedHashes_[channelIndex] = 0;
    return;
  }

  const SelectivityVector rows(1, true);
  decodedVectors_[channelIndex].decode(value, rows);

  std::vector<uint32_t> hashes{1};
  hash(decodedVectors_[channelIndex], value.typeKind(), 1, false, hashes);
  precomputedHashes_[channelIndex] = hashes[0];
}

} // namespace facebook::velox::connector::hive
