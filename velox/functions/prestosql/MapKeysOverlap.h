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
#include "velox/functions/Udf.h"

namespace facebook::velox::functions {

template <typename TExec, typename Key>
struct MapKeysOverlapPrimitiveFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& out,
      const arg_type<Map<Key, Generic<T1>>>& inputMap,
      const arg_type<Array<Key>>& keys) {
    if (inputMap.empty() || keys.empty()) {
      out = false;
      return;
    }

    folly::F14FastSet<arg_type<Key>> keySet;

    for (const auto& key : keys) {
      if (key.has_value()) {
        keySet.insert(key.value());
      }
    }

    for (const auto& entry : inputMap) {
      if (keySet.find(entry.first) != keySet.end()) {
        out = true;
        return;
      }
    }

    out = false;
  }
};

template <typename TExec>
struct MapKeysOverlapVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& out,
      const arg_type<Map<Varchar, Generic<T1>>>& inputMap,
      const arg_type<Array<Varchar>>& keys) {
    if (inputMap.empty() || keys.empty()) {
      out = false;
      return;
    }

    folly::F14FastSet<StringView> keySet;

    for (const auto& key : keys) {
      if (key.has_value()) {
        keySet.insert(key.value());
      }
    }

    for (const auto& entry : inputMap) {
      if (keySet.find(entry.first) != keySet.end()) {
        out = true;
        return;
      }
    }

    out = false;
  }
};

struct MapKeysOverlapFunctionEqualComparator {
  bool operator()(const exec::GenericView& lhs, const exec::GenericView& rhs)
      const {
    static constexpr auto kEqualValueAtFlags = CompareFlags::equality(
        CompareFlags::NullHandlingMode::kNullAsIndeterminate);

    auto result = lhs.compare(rhs, kEqualValueAtFlags);

    // If comparison returns indeterminate (null), treat as not equal
    if (!result.has_value()) {
      return false;
    }

    return result.value() == 0;
  }
};

template <typename TExec>
struct MapKeysOverlapFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& out,
      const arg_type<Map<Generic<T1>, Generic<T2>>>& inputMap,
      const arg_type<Array<Generic<T1>>>& keys) {
    if (inputMap.empty() || keys.empty()) {
      out = false;
      return;
    }

    searchKeys_.clear();
    for (const auto& key : keys) {
      if (key.has_value()) {
        searchKeys_.emplace(key.value());
      }
    }

    if (searchKeys_.empty()) {
      out = false;
      return;
    }

    for (const auto& entry : inputMap) {
      if (searchKeys_.contains(entry.first)) {
        out = true;
        return;
      }
    }

    out = false;
  }

 private:
  folly::F14FastSet<
      exec::GenericView,
      std::hash<exec::GenericView>,
      MapKeysOverlapFunctionEqualComparator>
      searchKeys_;
};

} // namespace facebook::velox::functions
