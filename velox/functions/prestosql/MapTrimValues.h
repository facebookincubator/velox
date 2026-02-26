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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

/// map_trim_values(map(K, array(V)), n) -> map(K, array(V))
///
/// Trims the value arrays in a map to a specified maximum size.
/// This function is useful for optimizing memory usage and performance
/// for large feature maps where the value arrays may grow unbounded.
///
/// Returns a map where each value array is trimmed to at most n elements.
/// If n is negative, returns the original map unchanged.
/// If n is 0, returns a map where all values are empty arrays.
/// If a value array has fewer than n elements, it is left unchanged.
/// Null map values are preserved in the output.
/// Null elements within arrays are also preserved.
template <typename TExec>
struct MapTrimValuesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, Array<Generic<T2>>>>& out,
      const arg_type<Map<Generic<T1>, Array<Generic<T2>>>>& inputMap,
      int64_t n) {
    // If n is negative, preserve the original map as-is
    if (n < 0) {
      out.copy_from(inputMap);
      return;
    }

    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
      } else {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);

        const auto& valueArray = entry.second.value();
        const auto arraySize = static_cast<size_t>(valueArray.size());
        const auto trimSize = std::min(static_cast<size_t>(n), arraySize);

        size_t count = 0;
        for (const auto& element : valueArray) {
          if (count >= trimSize) {
            break;
          }
          if (element.has_value()) {
            auto& elementWriter = valueWriter.add_item();
            elementWriter.copy_from(element.value());
          } else {
            valueWriter.add_null();
          }
          ++count;
        }
      }
    }
  }
};

} // namespace facebook::velox::functions
