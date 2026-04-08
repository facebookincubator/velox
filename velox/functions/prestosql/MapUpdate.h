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

#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

/// Fast path for primitive type keys: map_update(m, array[1, 2], array[10,
/// 20]).
/// Updates values in a map for specified keys. If a key exists, its value is
/// updated in place (preserving order). If a key doesn't exist, it is added
/// to the end of the map.
template <typename TExec, typename Key>
struct MapUpdatePrimitiveFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Key, Generic<T1>>>& out,
      const arg_type<Map<Key, Generic<T1>>>& inputMap,
      const arg_type<Array<Key>>& keys,
      const arg_type<Array<Generic<T1>>>& values) {
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build map from key -> index in the update arrays.
    typename util::floating_point::
        HashMapNaNAwareTypeTraits<arg_type<Key>, size_t>::Type updateKeyIndex;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i].has_value()) {
        VELOX_USER_CHECK(
            updateKeyIndex.emplace(keys[i].value(), i).second,
            "Duplicate key in keys array");
      }
    }

    // Track which update keys have been used (for keys already in inputMap).
    util::floating_point::HashSetNaNAware<arg_type<Key>> usedUpdateKeys;

    // Process entries from input map, updating values where needed.
    for (const auto& entry : inputMap) {
      auto it = updateKeyIndex.find(entry.first);
      if (it != updateKeyIndex.end()) {
        // Key is being updated - use new value.
        usedUpdateKeys.insert(entry.first);
        size_t idx = it->second;
        if (!values[idx].has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter = entry.first;
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter = entry.first;
          valueWriter.copy_from(values[idx].value());
        }
      } else {
        // Key is not being updated - keep original value.
        if (!entry.second.has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter = entry.first;
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter = entry.first;
          valueWriter.copy_from(entry.second.value());
        }
      }
    }

    // Add new keys from update arrays (keys not in original map).
    auto keysIt = keys.begin();
    auto valuesIt = values.begin();
    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value() && !usedUpdateKeys.contains(keysIt->value())) {
        if (!valuesIt->has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter = keysIt->value();
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter = keysIt->value();
          valueWriter.copy_from(valuesIt->value());
        }
      }
    }
  }
};

/// String version for varchar keys.
template <typename TExec>
struct MapUpdateVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Varchar, Generic<T1>>>& out,
      const arg_type<Map<Varchar, Generic<T1>>>& inputMap,
      const arg_type<Array<Varchar>>& keys,
      const arg_type<Array<Generic<T1>>>& values) {
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build map from key -> index in the update arrays.
    folly::F14FastMap<StringView, size_t> updateKeyIndex;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i].has_value()) {
        VELOX_USER_CHECK(
            updateKeyIndex.emplace(keys[i].value(), i).second,
            "Duplicate key in keys array");
      }
    }

    // Track which update keys have been used (for keys already in inputMap).
    folly::F14FastSet<StringView> usedUpdateKeys;

    // Process entries from input map, updating values where needed.
    for (const auto& entry : inputMap) {
      auto it = updateKeyIndex.find(entry.first);
      if (it != updateKeyIndex.end()) {
        // Key is being updated - use new value.
        usedUpdateKeys.insert(entry.first);
        size_t idx = it->second;
        if (!values[idx].has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(entry.first);
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(entry.first);
          valueWriter.copy_from(values[idx].value());
        }
      } else {
        // Key is not being updated - keep original value.
        if (!entry.second.has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(entry.first);
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(entry.first);
          valueWriter.copy_from(entry.second.value());
        }
      }
    }

    // Add new keys from update arrays (keys not in original map).
    auto keysIt = keys.begin();
    auto valuesIt = values.begin();
    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value() && !usedUpdateKeys.contains(keysIt->value())) {
        if (!valuesIt->has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(keysIt->value());
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(keysIt->value());
          valueWriter.copy_from(valuesIt->value());
        }
      }
    }
  }
};

struct MapUpdateFunctionEqualComparator {
  bool operator()(const exec::GenericView& lhs, const exec::GenericView& rhs)
      const {
    static constexpr auto kEqualValueAtFlags = CompareFlags::equality(
        CompareFlags::NullHandlingMode::kNullAsIndeterminate);

    auto result = lhs.compare(rhs, kEqualValueAtFlags);

    VELOX_USER_CHECK(
        result.has_value(), "Comparison on null elements is not supported");

    return result.value() == 0;
  }
};

/// Generic implementation for complex types.
template <typename TExec>
struct MapUpdateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, Generic<T2>>>& out,
      const arg_type<Map<Generic<T1>, Generic<T2>>>& inputMap,
      const arg_type<Array<Generic<T1>>>& keys,
      const arg_type<Array<Generic<T2>>>& values) {
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build map from key -> index in the update arrays.
    folly::F14FastMap<
        exec::GenericView,
        size_t,
        std::hash<exec::GenericView>,
        MapUpdateFunctionEqualComparator>
        updateKeyIndex;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i].has_value()) {
        VELOX_USER_CHECK(
            updateKeyIndex.emplace(keys[i].value(), i).second,
            "Duplicate key in keys array");
      }
    }

    // Track which update keys have been used (for keys already in inputMap).
    folly::F14FastSet<
        exec::GenericView,
        std::hash<exec::GenericView>,
        MapUpdateFunctionEqualComparator>
        usedUpdateKeys;

    // Process entries from input map, updating values where needed.
    for (const auto& entry : inputMap) {
      auto it = updateKeyIndex.find(entry.first);
      if (it != updateKeyIndex.end()) {
        // Key is being updated - use new value.
        usedUpdateKeys.insert(entry.first);
        size_t idx = it->second;
        if (!values[idx].has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(entry.first);
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(entry.first);
          valueWriter.copy_from(values[idx].value());
        }
      } else {
        // Key is not being updated - keep original value.
        if (!entry.second.has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(entry.first);
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(entry.first);
          valueWriter.copy_from(entry.second.value());
        }
      }
    }

    // Add new keys from update arrays (keys not in original map).
    auto keysIt = keys.begin();
    auto valuesIt = values.begin();
    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value() && !usedUpdateKeys.contains(keysIt->value())) {
        if (!valuesIt->has_value()) {
          auto& keyWriter = out.add_null();
          keyWriter.copy_from(keysIt->value());
        } else {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter.copy_from(keysIt->value());
          valueWriter.copy_from(valuesIt->value());
        }
      }
    }
  }
};

} // namespace facebook::velox::functions
