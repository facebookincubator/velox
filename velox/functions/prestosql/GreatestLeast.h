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

#include <cmath>
#include <optional>
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {
namespace details {
/**
 * This class implements two functions:
 *
 * greatest(value1, value2, ..., valueN) → [same as input]
 * Returns the largest of the provided values.
 *
 * least(value1, value2, ..., valueN) → [same as input]
 * Returns the smallest of the provided values.
 *
 * For DOUBLE and REAL type, NaN is considered as the biggest according to
 * https://github.com/prestodb/presto/issues/22391
 **/
template <typename TExec, typename T, bool isLeast>
struct ExtremeValueFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<T>& result,
      const arg_type<T>& firstElement,
      const arg_type<Variadic<T>>& remainingElement) {
    // Track the winner: nullopt means firstElement, a value means index in
    // remainingElement
    std::optional<vector_size_t> winnerIdx;

    for (size_t i = 0; i < remainingElement.size(); ++i) {
      const auto& cur = remainingElement[i].value();
      const auto& best = winnerIdx.has_value()
          ? remainingElement[*winnerIdx].value()
          : firstElement;

      if constexpr (isLeast) {
        if (smallerThan(cur, best)) {
          winnerIdx = i;
        }
      } else {
        if (greaterThan(cur, best)) {
          winnerIdx = i;
        }
      }
    }

    if (!winnerIdx.has_value()) {
      result = extractValue(firstElement);
    } else {
      result = extractValue(remainingElement[*winnerIdx].value());
    }
  }

 private:
  template <typename U>
  auto extractValue(const U& wrapper) const {
    return wrapper;
  }

  template <typename U>
  U extractValue(
      const exec::CustomTypeWithCustomComparisonView<U>& wrapper) const {
    return *wrapper;
  }

  template <typename K>
  bool greaterThan(const K& lhs, const K& rhs) const {
    if constexpr (std::is_same_v<K, double> || std::is_same_v<K, float>) {
      if (std::isnan(lhs)) {
        return true;
      }

      if (std::isnan(rhs)) {
        return false;
      }
    }

    return lhs > rhs;
  }

  template <typename K>
  bool smallerThan(const K& lhs, const K& rhs) const {
    if constexpr (std::is_same_v<K, double> || std::is_same_v<K, float>) {
      if (std::isnan(lhs)) {
        return false;
      }

      if (std::isnan(rhs)) {
        return true;
      }
    }

    return lhs < rhs;
  }
};
} // namespace details

template <typename TExec, typename T>
using LeastFunction = details::ExtremeValueFunction<TExec, T, true>;

template <typename TExec, typename T>
using GreatestFunction = details::ExtremeValueFunction<TExec, T, false>;

} // namespace facebook::velox::functions
