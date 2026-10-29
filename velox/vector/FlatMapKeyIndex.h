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

#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

#include <optional>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::detail {

/// General index over a FlatMapVector's distinct keys, for keys FlatMapKeyIndex
/// cannot index by value: complex types such as ROW, ARRAY and MAP, types with
/// custom comparison, and distinct keys that contain a null. It maps each key's
/// hash, as BaseVector::hashValueAt() computes it, to the key's channel, so it
/// works for any key type without knowing its C++ type. Distinct keys can share
/// a hash, so a lookup confirms each candidate by comparing the actual keys.
class FlatMapHashedKeyIndex {
 public:
  explicit FlatMapHashedKeyIndex(const BaseVector& distinctKeys);

  /// Indexes the next channel, whose key hashes to `hash`. Builds the index
  /// one key at a time, and lets tests add colliding hashes directly.
  void add(uint64_t hash);

  /// Returns the first channel under `hash` for which `isMatch(channel)` holds.
  template <typename TIsMatch>
  std::optional<column_index_t> find(uint64_t hash, TIsMatch&& isMatch) const {
    auto range = keyToChannel_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
      if (isMatch(it->second)) {
        return it->second;
      }
    }
    return std::nullopt;
  }

 private:
  std::unordered_multimap<uint64_t, column_index_t> keyToChannel_;
};

/// Fast-path index for FlatMapVector keys of C++ type `T`: each key by value,
/// mapped to its channel.
template <typename T>
using FlatMapPrimitiveKeyIndex = folly::F14FastMap<T, column_index_t>;

/// The index a FlatMapKeyIndex holds: a primitive key index for one of the
/// fast-path key types listed here, or the general hashed index.
using FlatMapKeyIndexVariant = std::variant<
    FlatMapHashedKeyIndex,
    FlatMapPrimitiveKeyIndex<bool>,
    FlatMapPrimitiveKeyIndex<int8_t>,
    FlatMapPrimitiveKeyIndex<int16_t>,
    FlatMapPrimitiveKeyIndex<int32_t>,
    FlatMapPrimitiveKeyIndex<int64_t>>;

/// True if `TIndex` is one of the alternatives of the std::variant `TVariant`;
/// the standard library has no such trait. The partial specialization unpacks
/// the variant's alternatives and compares each with `TIndex` through
/// std::disjunction.
template <typename TIndex, typename TVariant>
struct is_variant_alternative;

template <typename TIndex, typename... TIndexes>
struct is_variant_alternative<TIndex, std::variant<TIndexes...>>
    : std::disjunction<std::is_same<TIndex, TIndexes>...> {};

/// Whether keys of C++ type `T` take the fast path. It is derived from
/// FlatMapKeyIndexVariant so the fast-path types are listed only there. Code
/// that names FlatMapPrimitiveKeyIndex<T> tests it with `if constexpr`, because
/// that code does not compile for a type without that alternative.
template <typename T>
inline constexpr bool is_flat_map_primitive_key_v = is_variant_alternative<
    FlatMapPrimitiveKeyIndex<T>,
    FlatMapKeyIndexVariant>::value;

/// Returns the channel of `key` in `index`, if present.
template <typename T>
std::optional<column_index_t> findValue(
    const FlatMapPrimitiveKeyIndex<T>& index,
    T key) {
  auto it = index.find(key);
  if (it == index.end()) {
    return std::nullopt;
  }
  return it->second;
}

/// Maps each distinct key of a FlatMapVector to its channel, the position of
/// that key's values.
///
/// Keys of these types take a fast path, one per alternative of
/// FlatMapKeyIndexVariant:
///
///   - BOOLEAN
///   - TINYINT
///   - SMALLINT
///   - INTEGER
///   - BIGINT
///
/// Such keys are stored by value in an F14 map, which hashes and compares them
/// itself. Since distinct keys are distinct values, there are no collisions to
/// handle, and a lookup is a single probe with no virtual calls into the keys
/// vector. Typed lookups such as FlatMapVector::getKeyChannel(int64_t) probe
/// the map directly. In the fast path, a repeated key resolves to its last
/// channel.
///
/// Other key types, types with custom comparison, and distinct keys that
/// contain a null use FlatMapHashedKeyIndex, where a null key matches a null
/// key. Both paths treat keys as BaseVector::hashValueAt() and equalValueAt()
/// do, so the choice only affects speed.
class FlatMapKeyIndex {
 public:
  /// Indexes every key in `distinctKeys`, in order.
  explicit FlatMapKeyIndex(const BaseVector& distinctKeys);

  /// Indexes the last key in `distinctKeys`, which was just appended to them.
  /// FlatMapVector::appendDistinctKey() calls it when copyRanges() brings in a
  /// key the vector did not have.
  void appendLast(const BaseVector& distinctKeys);

  /// Returns the channel of the key at `index` in `keys`, a vector of the same
  /// type as `distinctKeys`.
  std::optional<column_index_t> find(
      const BaseVector& distinctKeys,
      const BaseVector& keys,
      vector_size_t index) const;

  /// Returns the channel of `key`. Throws if `T` is not the C++ type of
  /// `distinctKeys`.
  template <typename T>
  std::optional<column_index_t> find(const BaseVector& distinctKeys, T key)
      const {
    if constexpr (is_flat_map_primitive_key_v<T>) {
      if (const auto* index =
              std::get_if<FlatMapPrimitiveKeyIndex<T>>(&index_)) {
        return findValue(*index, key);
      }
    }
    const auto* hashed = std::get_if<FlatMapHashedKeyIndex>(&index_);
    const auto* simpleKeys = distinctKeys.loadedVector()->as<SimpleVector<T>>();
    VELOX_CHECK(
        hashed != nullptr && simpleKeys != nullptr,
        "Incompatible vector type for flat map vector keys: {}",
        distinctKeys.toString());
    return hashed->find(folly::hasher<T>{}(key), [&](column_index_t channel) {
      return simpleKeys->valueAt(static_cast<vector_size_t>(channel)) == key;
    });
  }

 private:
  FlatMapKeyIndexVariant index_;
};

} // namespace facebook::velox::detail
