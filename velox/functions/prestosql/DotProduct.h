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

#include <cstdint>

#include <folly/CPortability.h>
#include <folly/container/F14Map.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/type/SimpleFunctionApi.h"

namespace facebook::velox::functions {

/// Computes the dot product of two arrays.
/// The dot product is the sum of element-wise products of corresponding
/// elements. Both arrays must have the same length. If either array is null,
/// returns null. If arrays have different lengths, throws an error.
/// Null elements in arrays are treated as zero.
/// NaN values propagate through the computation (IEEE 754 behavior).
template <typename TExec, typename T>
struct DotProductFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename TOutput>
  FOLLY_ALWAYS_INLINE void call(
      TOutput& out,
      const arg_type<Array<T>>& array1,
      const arg_type<Array<T>>& array2) {
    const auto size1 = array1.size();
    const auto size2 = array2.size();

    VELOX_USER_CHECK_EQ(
        size1,
        size2,
        "dot_product requires arrays of equal length, but got {} and {}",
        size1,
        size2);

    TOutput sum = 0;
    for (vector_size_t i = 0; i < size1; ++i) {
      const auto& val1 = array1[i];
      const auto& val2 = array2[i];

      if (val1.has_value() && val2.has_value()) {
        if constexpr (std::is_same_v<TOutput, int64_t>) {
          auto product = checkedMultiply<TOutput>(
              static_cast<TOutput>(val1.value()),
              static_cast<TOutput>(val2.value()));
          sum = checkedPlus<TOutput>(sum, product);
        } else {
          sum += static_cast<TOutput>(val1.value()) *
              static_cast<TOutput>(val2.value());
        }
      }
    }
    out = sum;
  }
};

/// Computes the dot product of two maps.
/// For maps, the dot product is computed by multiplying values with matching
/// keys and summing the results. Keys present in only one map contribute zero
/// to the result. If either map is null, returns null.
/// Null values in maps are treated as zero.
template <typename TExec, typename K, typename V>
struct MapDotProductFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename TOutput>
  FOLLY_ALWAYS_INLINE void call(
      TOutput& out,
      const arg_type<Map<K, V>>& map1,
      const arg_type<Map<K, V>>& map2) {
    TOutput sum = 0;

    // Build a lookup map from the smaller map for O(1) key lookup.
    // This reduces complexity from O(n*m) to O(n+m).
    const auto& smallMap = map1.size() <= map2.size() ? map1 : map2;
    const auto& largeMap = map1.size() <= map2.size() ? map2 : map1;

    folly::F14FastMap<arg_type<K>, arg_type<V>> lookup;
    lookup.reserve(smallMap.size());
    for (const auto& [key, val] : smallMap) {
      if (val.has_value()) {
        lookup.emplace(key, val.value());
      }
    }

    // Iterate through the larger map and look up matching keys.
    for (const auto& [key, val] : largeMap) {
      if (!val.has_value()) {
        continue;
      }

      auto it = lookup.find(key);
      if (it != lookup.end()) {
        if constexpr (std::is_same_v<TOutput, int64_t>) {
          auto product = checkedMultiply<TOutput>(
              static_cast<TOutput>(val.value()),
              static_cast<TOutput>(it->second));
          sum = checkedPlus<TOutput>(sum, product);
        } else {
          sum += static_cast<TOutput>(val.value()) *
              static_cast<TOutput>(it->second);
        }
      }
    }

    out = sum;
  }
};

} // namespace facebook::velox::functions
