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

namespace facebook::velox::connector::hive {

namespace {
template <TypeKind kind>
void hashTyped(
    const DecodedVector& /* values */,
    vector_size_t /* size */,
    bool /* mix */,
    std::vector<int32_t>& /* hashes */) {
  VELOX_UNSUPPORTED(
      "Hive partitioning function doesn't support {} type",
      TypeTraits<kind>::name);
}

template <>
void hashTyped<TypeKind::BOOLEAN>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<int32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    uint32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = values.valueAt<bool>(i) ? 1 : 0;
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

int32_t hashInt64(int64_t value) {
  return ((*reinterpret_cast<uint64_t*>(&value)) >> 32) ^ value;
}

template <>
void hashTyped<TypeKind::BIGINT>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<int32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    int32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = hashInt64(values.valueAt<int64_t>(i));
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
__attribute__((no_sanitize("integer")))
#endif
#endif
int32_t
hashBytes(StringView bytes, int32_t initialValue) {
  int32_t hash = initialValue;
  auto* data = bytes.data();
  for (auto i = 0; i < bytes.size(); ++i) {
    hash = hash * 31 + *reinterpret_cast<const int8_t*>(data + i);
  }
  return hash;
}

template <>
void hashTyped<TypeKind::VARCHAR>(
    const DecodedVector& values,
    vector_size_t size,
    bool mix,
    std::vector<int32_t>& hashes) {
  for (auto i = 0; i < size; ++i) {
    int32_t hash;
    if (values.isNullAt(i)) {
      hash = 0;
    } else {
      hash = hashBytes(values.valueAt<StringView>(i), 0);
    }

    hashes[i] = mix ? hashes[i] * 31 + hash : hash;
  }
}

void hash(
    const DecodedVector& values,
    TypeKind typeKind,
    vector_size_t size,
    bool mix,
    std::vector<int32_t>& hashes) {
  // This function mirrors the behavior of function hashCode in
  // HIVE-12025 ba83fd7bff
  // serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java
  // https://github.com/apache/hive/blob/ba83fd7bff/serde/src/java/org/apache/hadoop/hive/serde2/objectinspector/ObjectInspectorUtils.java

  // HIVE-7148 proposed change to bucketing hash algorithms. If that gets
  // implemented, this function will need to change significantly.

  VELOX_DYNAMIC_TYPE_DISPATCH(hashTyped, typeKind, values, size, mix, hashes);
}
} // namespace

HivePartitionFunction::HivePartitionFunction(
    int numBuckets,
    std::vector<int> bucketToPartition,
    std::vector<ChannelIndex> keyChannels)
    : numBuckets_{numBuckets},
      bucketToPartition_{bucketToPartition},
      keyChannels_{std::move(keyChannels)} {
  decodedVectors_.resize(keyChannels_.size());
}

void HivePartitionFunction::partition(
    const RowVector& input,
    std::vector<uint32_t>& partitions) {
  auto size = input.size();

  if (size > hashes_.size()) {
    rows_.resize(size);
    rows_.setAll();
    hashes_.resize(size);
  }

  partitions.resize(size);
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    auto keyVector = input.childAt(keyChannels_[i]);
    decodedVectors_[i].decode(*keyVector, rows_);
    hash(
        decodedVectors_[i],
        keyVector->typeKind(),
        keyVector->size(),
        i > 0,
        hashes_);
  }

  static const int32_t kInt32Max = std::numeric_limits<int32_t>::max();

  for (auto i = 0; i < size; ++i) {
    partitions[i] =
        bucketToPartition_[((hashes_[i] & kInt32Max) % numBuckets_)];
  }
}
} // namespace facebook::velox::connector::hive
