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

#include "velox/vector/FlatMapKeyIndex.h"

#include "velox/vector/DecodedVector.h"

namespace facebook::velox::detail {

FlatMapHashedKeyIndex::FlatMapHashedKeyIndex(const BaseVector& distinctKeys) {
  for (vector_size_t i = 0; i < distinctKeys.size(); ++i) {
    add(distinctKeys.hashValueAt(i));
  }
}

void FlatMapHashedKeyIndex::add(uint64_t hash) {
  const auto channel = static_cast<column_index_t>(keyToChannel_.size());
  keyToChannel_.insert({hash, channel});
}

namespace {

// Reads `vector` as a SimpleVector<T>, loading it first if it is lazy.
template <typename T>
const SimpleVector<T>& asSimple(const BaseVector& vector) {
  return *vector.loadedVector()->asUnchecked<SimpleVector<T>>();
}

template <TypeKind kKind>
FlatMapKeyIndexVariant makeIndexForKind(const BaseVector& distinctKeys) {
  using T = typename TypeTraits<kKind>::NativeType;
  if constexpr (is_flat_map_primitive_key_v<T>) {
    DecodedVector decoded(distinctKeys);
    FlatMapPrimitiveKeyIndex<T> index;
    index.reserve(decoded.size());
    for (vector_size_t i = 0; i < decoded.size(); ++i) {
      // A value index cannot hold a null key.
      if (decoded.isNullAt(i)) {
        return FlatMapHashedKeyIndex(distinctKeys);
      }
      index.insert_or_assign(
          decoded.valueAt<T>(i), static_cast<column_index_t>(i));
    }
    return index;
  } else {
    return FlatMapHashedKeyIndex(distinctKeys);
  }
}

FlatMapKeyIndexVariant makeIndex(const BaseVector& distinctKeys) {
  if (distinctKeys.type()->providesCustomComparison()) {
    return FlatMapHashedKeyIndex(distinctKeys);
  }
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      makeIndexForKind, distinctKeys.typeKind(), distinctKeys);
}

} // namespace

FlatMapKeyIndex::FlatMapKeyIndex(const BaseVector& distinctKeys)
    : index_(makeIndex(distinctKeys)) {}

void FlatMapKeyIndex::appendLast(const BaseVector& distinctKeys) {
  const vector_size_t channel = distinctKeys.size() - 1;
  // A value index cannot hold a null key, so switch to hashing every key.
  if (distinctKeys.isNullAt(channel) &&
      !std::holds_alternative<FlatMapHashedKeyIndex>(index_)) {
    index_ = FlatMapHashedKeyIndex(distinctKeys);
    return;
  }
  std::visit(
      [&](auto& index) {
        using TIndex = std::decay_t<decltype(index)>;
        if constexpr (std::is_same_v<TIndex, FlatMapHashedKeyIndex>) {
          index.add(distinctKeys.hashValueAt(channel));
        } else {
          using T = typename TIndex::key_type;
          index.insert_or_assign(
              asSimple<T>(distinctKeys).valueAt(channel),
              static_cast<column_index_t>(channel));
        }
      },
      index_);
}

std::optional<column_index_t> FlatMapKeyIndex::find(
    const BaseVector& distinctKeys,
    const BaseVector& keys,
    vector_size_t index) const {
  VELOX_CHECK_EQ(
      keys.typeKind(),
      distinctKeys.typeKind(),
      "Incompatible vector type for flat map vector keys: {}",
      keys.type()->toString());
  return std::visit(
      [&](const auto& keyIndex) -> std::optional<column_index_t> {
        using TIndex = std::decay_t<decltype(keyIndex)>;
        if constexpr (std::is_same_v<TIndex, FlatMapHashedKeyIndex>) {
          return keyIndex.find(
              keys.hashValueAt(index), [&](column_index_t channel) {
                return keys.equalValueAt(
                    &distinctKeys, index, static_cast<vector_size_t>(channel));
              });
        } else {
          if (keys.isNullAt(index)) {
            return std::nullopt;
          }
          using T = typename TIndex::key_type;
          return findValue(keyIndex, asSimple<T>(keys).valueAt(index));
        }
      },
      index_);
}

} // namespace facebook::velox::detail
