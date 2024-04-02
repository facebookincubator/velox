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

#include <functions/Macros.h>
#include <cmath>

namespace facebook::velox::functions {

template <typename TExec, typename T, bool isLeast>
struct ExtremeValueFunction;

template <typename TExec, typename T>
using LeastFunction = ExtremeValueFunction<TExec, T, true>;

template <typename TExec, typename T>
using GreatestFunction = ExtremeValueFunction<TExec, T, false>;

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
 * https://github.com/prestodb/presto/wiki/Presto-NaN-behavior
 **/
template <typename TExec, typename T, bool isLeast>
struct ExtremeValueFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<T>& result,
      const arg_type<T>& firstElement,
      const arg_type<Variadic<T>>& remainingElement) {
    auto currentValue = firstElement;

    for (auto i = 0; i < remainingElement.size(); ++i) {
      VELOX_USER_CHECK(remainingElement[i].has_value());
      auto candidateValue = remainingElement[i].value();

      if constexpr (isLeast) {
        if (smallerThan(candidateValue, currentValue)) {
          currentValue = candidateValue;
        }
      } else {
        if (greaterThan(candidateValue, currentValue)) {
          currentValue = candidateValue;
        }
      }
    }

    result = currentValue;
  }

 private:
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

} // namespace facebook::velox::functions
