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
#include "velox/functions/Udf.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

/// Fast path for constant primitive type keys.
template <typename TExec, typename Key>
struct RemapKeysPrimitiveFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Key, Generic<T1>>>& out,
      const arg_type<Map<Key, Generic<T1>>>& inputMap,
      const arg_type<Array<Key>>& oldKeys,
      const arg_type<Array<Key>>& newKeys) {
    if (inputMap.empty()) {
      return;
    }

    // Build mapping from old keys to new keys
    folly::F14FastMap<arg_type<Key>, arg_type<Key>> keyMapping;

    // The old and new key arrays don't need to be the same size.
    // If they differ, we only process up to the shorter array's length.
    auto oldKeysSize = oldKeys.size();
    auto newKeysSize = newKeys.size();
    auto minSize = std::min(oldKeysSize, newKeysSize);

    auto oldIt = oldKeys.begin();
    auto newIt = newKeys.begin();

    for (size_t i = 0; i < minSize; ++i, ++oldIt, ++newIt) {
      if (oldIt->has_value() && newIt->has_value()) {
        keyMapping[oldIt->value()] = newIt->value();
      }
    }

    // Iterate through input map and remap keys
    for (const auto& entry : inputMap) {
      auto it = keyMapping.find(entry.first);

      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        if (it != keyMapping.end()) {
          keyWriter = it->second;
        } else {
          keyWriter = entry.first;
        }
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        if (it != keyMapping.end()) {
          keyWriter = it->second;
        } else {
          keyWriter = entry.first;
        }
        valueWriter.copy_from(entry.second.value());
      }
    }
  }
};

/// String version that avoids copy of strings.
template <typename TExec>
struct RemapKeysVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  static constexpr int32_t reuse_strings_from_arg = 0;

  void call(
      out_type<Map<Varchar, Generic<T1>>>& out,
      const arg_type<Map<Varchar, Generic<T1>>>& inputMap,
      const arg_type<Array<Varchar>>& oldKeys,
      const arg_type<Array<Varchar>>& newKeys) {
    if (inputMap.empty()) {
      return;
    }

    // Build mapping from old keys to new keys
    folly::F14FastMap<StringView, StringView> keyMapping;

    auto oldKeysSize = oldKeys.size();
    auto newKeysSize = newKeys.size();
    auto minSize = std::min(oldKeysSize, newKeysSize);

    auto oldIt = oldKeys.begin();
    auto newIt = newKeys.begin();

    for (size_t i = 0; i < minSize; ++i, ++oldIt, ++newIt) {
      if (oldIt->has_value() && newIt->has_value()) {
        keyMapping[oldIt->value()] = newIt->value();
      }
    }

    // Iterate through input map and remap keys
    for (const auto& entry : inputMap) {
      auto it = keyMapping.find(entry.first);

      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        if (it != keyMapping.end()) {
          // Key is remapped - must copy from newKeys array
          keyWriter.copy_from(it->second);
        } else {
          // Key is not remapped - can reuse from inputMap
          keyWriter.setNoCopy(entry.first);
        }
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        if (it != keyMapping.end()) {
          // Key is remapped - must copy from newKeys array
          keyWriter.copy_from(it->second);
        } else {
          // Key is not remapped - can reuse from inputMap
          keyWriter.setNoCopy(entry.first);
        }
        valueWriter.copy_from(entry.second.value());
      }
    }
  }
};

struct RemapKeysFunctionEqualComparator {
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

struct RemapKeysFunctionHasher {
  size_t operator()(const exec::GenericView& value) const {
    return value.hash();
  }
};

/// Generic implementation for complex types.
template <typename TExec>
struct RemapKeysFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, Generic<T2>>>& out,
      const arg_type<Map<Generic<T1>, Generic<T2>>>& inputMap,
      const arg_type<Array<Generic<T1>>>& oldKeys,
      const arg_type<Array<Generic<T1>>>& newKeys) {
    if (inputMap.empty()) {
      return;
    }

    auto oldKeysSize = oldKeys.size();
    auto newKeysSize = newKeys.size();
    auto minSize = std::min(oldKeysSize, newKeysSize);

    // Iterate through input map and remap keys
    for (const auto& entry : inputMap) {
      std::optional<size_t> matchIndex;

      // Search for matching old key
      auto oldIt = oldKeys.begin();
      for (size_t i = 0; i < minSize; ++i, ++oldIt) {
        if (oldIt->has_value()) {
          RemapKeysFunctionEqualComparator comparator;
          if (comparator(entry.first, oldIt->value())) {
            matchIndex = i;
            break;
          }
        }
      }

      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        if (matchIndex.has_value()) {
          auto newIt = newKeys.begin();
          std::advance(newIt, matchIndex.value());
          if (newIt->has_value()) {
            keyWriter.copy_from(newIt->value());
          } else {
            keyWriter.copy_from(entry.first);
          }
        } else {
          keyWriter.copy_from(entry.first);
        }
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        if (matchIndex.has_value()) {
          auto newIt = newKeys.begin();
          std::advance(newIt, matchIndex.value());
          if (newIt->has_value()) {
            keyWriter.copy_from(newIt->value());
          } else {
            keyWriter.copy_from(entry.first);
          }
        } else {
          keyWriter.copy_from(entry.first);
        }
        valueWriter.copy_from(entry.second.value());
      }
    }
  }
};

} // namespace facebook::velox::functions
