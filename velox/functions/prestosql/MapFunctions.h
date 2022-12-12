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

#include <fmt/format.h>
#include "folly/container/F14Set.h"

#include "velox/functions/Udf.h"

namespace facebook::velox::functions {

template <typename V>
bool constexpr provide_std_interface =
    CppToType<V>::isPrimitiveType && !std::is_same_v<Varchar, V> &&
    !std::is_same_v<Varbinary, V> && !std::is_same_v<Any, V>;

/// Function Signature: map_from_entries(array(row(K, V))) -> map(K, V)
/// Returns a map created from the given array of entries.
template <typename TExecCtx, typename K, typename V>
struct MapFromEntriesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExecCtx);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Map<K, V>>& out,
      const arg_type<Array<Row<K, V>>>& inputArray) {
    folly::F14FastSet<arg_type<K>> uniqueSet;
    for (const auto& tuple : inputArray) {
      // 1. Do not accept null for any map entry
      // NOTE: Do not use inputArray.mayHaveNulls() here because it will
      // return false in simplyEval path (all initial nulls will be removed)
      VELOX_USER_CHECK(tuple.has_value(), "map entry cannot be null");
      const auto& key = tuple.value().template at<0>();
      // 2. Do not accept null for any map key, but accept a null map value
      VELOX_USER_CHECK(key.has_value(), "map key cannot be null");
      // 3. Do not accept duplicate keys
      VELOX_USER_CHECK(
          uniqueSet.insert(*key).second,
          fmt::format(
              "Duplicate keys ({}) are not allowed", toString(key.value())));

      const auto& value = tuple.value().template at<1>();
      emplace(out, key, value);
    }
    return;
  }

  // TODO: move this function to `MapWriter` class
  void emplace(
      out_type<Map<K, V>>& out,
      const exec::OptionalAccessor<K>& key,
      const exec::OptionalAccessor<V>& value) {
    if (value.has_value()) {
      auto [keyWriter, valueWriter] = out.add_item();
      // set key
      if constexpr (provide_std_interface<K>) {
        keyWriter = *key;
      } else {
        keyWriter.setNoCopy(*key);
      }
      // set value
      if constexpr (provide_std_interface<V>) {
        valueWriter = *value;
      } else {
        valueWriter.setNoCopy(*value);
      }
    } else {
      // Value is null
      auto& keyWriter = out.add_null();
      // copy key
      if constexpr (provide_std_interface<K>) {
        keyWriter = *key;
      } else {
        keyWriter.setNoCopy(*key);
      }
    }
  }

  template <typename T>
  std::string toString(const T& val) {
    static_assert(
        CppToType<T>::isPrimitiveType, "Only support primitive types");
    if constexpr (std::is_same_v<T, StringView>) {
      return val.str();
    } else {
      return std::to_string(val);
    }
  }
};
} // namespace facebook::velox::functions
