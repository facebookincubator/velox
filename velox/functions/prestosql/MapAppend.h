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

#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

/// Fast path for primitive type keys: map_append(m, array[1, 2], array[10,
/// 20]).
template <typename TExec, typename Key>
struct MapAppendPrimitiveFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Key, Generic<T1>>>& out,
      const arg_type<Map<Key, Generic<T1>>>& inputMap,
      const arg_type<Array<Key>>& keys,
      const arg_type<Array<Generic<T1>>>& values) {
    // Check that keys and values arrays have the same size
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build a set of existing keys for fast lookup
    util::floating_point::HashSetNaNAware<arg_type<Key>> existingKeys;
    for (const auto& entry : inputMap) {
      existingKeys.emplace(entry.first);
    }

    // Build a set of new keys to check for duplicates and verify they don't
    // exist
    util::floating_point::HashSetNaNAware<arg_type<Key>> newKeys;

    auto keysIt = keys.begin();
    auto valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
        const auto& key = keysIt->value();

        // Check if key already exists in the input map
        VELOX_USER_CHECK(
            !existingKeys.contains(key), "Key already exists in the map");

        // Check for duplicate keys in the new keys array
        VELOX_USER_CHECK(
            newKeys.emplace(key).second, "Duplicate key in keys array");
      }
    }

    // First, copy all existing entries from the input map
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter = entry.first;
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter = entry.first;
        valueWriter.copy_from(entry.second.value());
      }
    }

    // Then, add new entries from keys and values arrays
    keysIt = keys.begin();
    valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
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
struct MapAppendVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Varchar, Generic<T1>>>& out,
      const arg_type<Map<Varchar, Generic<T1>>>& inputMap,
      const arg_type<Array<Varchar>>& keys,
      const arg_type<Array<Generic<T1>>>& values) {
    // Check that keys and values arrays have the same size
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build a set of existing keys for fast lookup
    folly::F14FastSet<StringView> existingKeys;
    for (const auto& entry : inputMap) {
      existingKeys.emplace(entry.first);
    }

    // Build a set of new keys to check for duplicates and verify they don't
    // exist
    folly::F14FastSet<StringView> newKeys;

    auto keysIt = keys.begin();
    auto valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
        const auto& key = keysIt->value();

        // Check if key already exists in the input map
        VELOX_USER_CHECK(
            !existingKeys.contains(key), "Key already exists in the map");

        // Check for duplicate keys in the new keys array
        VELOX_USER_CHECK(
            newKeys.emplace(key).second, "Duplicate key in keys array");
      }
    }

    // First, copy all existing entries from the input map
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);
        valueWriter.copy_from(entry.second.value());
      }
    }

    // Then, add new entries from keys and values arrays
    keysIt = keys.begin();
    valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
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

struct MapAppendFunctionEqualComparator {
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
struct MapAppendFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, Generic<T2>>>& out,
      const arg_type<Map<Generic<T1>, Generic<T2>>>& inputMap,
      const arg_type<Array<Generic<T1>>>& keys,
      const arg_type<Array<Generic<T2>>>& values) {
    // Check that keys and values arrays have the same size
    VELOX_USER_CHECK_EQ(
        keys.size(),
        values.size(),
        "Keys and values arrays must have the same length");

    // Build a set of existing keys for fast lookup
    folly::F14FastSet<
        exec::GenericView,
        std::hash<exec::GenericView>,
        MapAppendFunctionEqualComparator>
        existingKeys;

    for (const auto& entry : inputMap) {
      existingKeys.emplace(entry.first);
    }

    // Build a set of new keys to check for duplicates and verify they don't
    // exist
    folly::F14FastSet<
        exec::GenericView,
        std::hash<exec::GenericView>,
        MapAppendFunctionEqualComparator>
        newKeys;

    auto keysIt = keys.begin();
    auto valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
        const auto& key = keysIt->value();

        // Check if key already exists in the input map
        VELOX_USER_CHECK(
            !existingKeys.contains(key), "Key already exists in the map");

        // Check for duplicate keys in the new keys array
        VELOX_USER_CHECK(
            newKeys.emplace(key).second, "Duplicate key in keys array");
      }
    }

    // First, copy all existing entries from the input map
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);
        valueWriter.copy_from(entry.second.value());
      }
    }

    // Then, add new entries from keys and values arrays
    keysIt = keys.begin();
    valuesIt = values.begin();

    for (; keysIt != keys.end(); ++keysIt, ++valuesIt) {
      if (keysIt->has_value()) {
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
