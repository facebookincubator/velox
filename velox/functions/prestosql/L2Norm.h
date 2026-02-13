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

#include <cmath>

#include <folly/CPortability.h>
#include "velox/functions/Macros.h"
#include "velox/type/SimpleFunctionApi.h"

namespace facebook::velox::functions {

/// L2_NORM function for arrays.
/// Calculates the Euclidean norm (L2 norm) of an array: sqrt(sum(x^2))
/// Returns the L2 norm as a double.
///
/// Function Signature:
///   L2_NORM(ARRAY<T>) -> DOUBLE where T is numeric
///
/// Behavior:
///   - Returns 0.0 for empty arrays
///   - Returns null if the input array is null
///   - Null elements in the array are skipped (not included in calculation)
///   - Supports all numeric types (int, float, double)
template <typename TExec>
struct ArrayL2NormFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(double& out, const TInput& inputArray) {
    if (inputArray.size() == 0) {
      out = 0.0;
      return true;
    }

    double maxAbs = 0.0;
    for (const auto& item : inputArray) {
      if (item.has_value()) {
        maxAbs = std::max(maxAbs, std::abs(static_cast<double>(item.value())));
      }
    }

    if (maxAbs == 0.0) {
      out = 0.0;
      return true;
    }

    double sumOfSquares = 0.0;
    for (const auto& item : inputArray) {
      if (item.has_value()) {
        auto scaled = static_cast<double>(item.value()) / maxAbs;
        sumOfSquares += scaled * scaled;
      }
    }

    out = maxAbs * std::sqrt(sumOfSquares);
    return true;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool callNullFree(double& out, const TInput& inputArray) {
    if (inputArray.size() == 0) {
      out = 0.0;
      return true;
    }

    double maxAbs = 0.0;
    for (const auto& item : inputArray) {
      maxAbs = std::max(maxAbs, std::abs(static_cast<double>(item)));
    }

    if (maxAbs == 0.0) {
      out = 0.0;
      return true;
    }

    double sumOfSquares = 0.0;
    for (const auto& item : inputArray) {
      auto scaled = static_cast<double>(item) / maxAbs;
      sumOfSquares += scaled * scaled;
    }

    out = maxAbs * std::sqrt(sumOfSquares);
    return true;
  }
};

/// L2_NORM function for maps.
/// Calculates the Euclidean norm (L2 norm) of map values: sqrt(sum(v^2))
/// Returns the L2 norm as a double.
///
/// Function Signature:
///   L2_NORM(MAP<K, V>) -> DOUBLE where V is numeric
///
/// Behavior:
///   - Returns 0.0 for empty maps
///   - Returns null if the input map is null
///   - Null values in the map are skipped (not included in calculation)
///   - Keys are ignored, only values are used
template <typename TExec, typename K, typename V>
struct MapL2NormFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE bool call(
      double& out,
      const arg_type<Map<K, V>>& inputMap) {
    if (inputMap.size() == 0) {
      out = 0.0;
      return true;
    }

    double maxAbs = 0.0;
    for (const auto& entry : inputMap) {
      if (entry.second.has_value()) {
        maxAbs = std::max(
            maxAbs, std::abs(static_cast<double>(entry.second.value())));
      }
    }

    if (maxAbs == 0.0) {
      out = 0.0;
      return true;
    }

    double sumOfSquares = 0.0;
    for (const auto& entry : inputMap) {
      if (entry.second.has_value()) {
        auto scaled = static_cast<double>(entry.second.value()) / maxAbs;
        sumOfSquares += scaled * scaled;
      }
    }

    out = maxAbs * std::sqrt(sumOfSquares);
    return true;
  }

  FOLLY_ALWAYS_INLINE bool callNullFree(
      double& out,
      const null_free_arg_type<Map<K, V>>& inputMap) {
    if (inputMap.size() == 0) {
      out = 0.0;
      return true;
    }

    double maxAbs = 0.0;
    for (const auto& entry : inputMap) {
      maxAbs = std::max(maxAbs, std::abs(static_cast<double>(entry.second)));
    }

    if (maxAbs == 0.0) {
      out = 0.0;
      return true;
    }

    double sumOfSquares = 0.0;
    for (const auto& entry : inputMap) {
      auto scaled = static_cast<double>(entry.second) / maxAbs;
      sumOfSquares += scaled * scaled;
    }

    out = maxAbs * std::sqrt(sumOfSquares);
    return true;
  }
};

} // namespace facebook::velox::functions
