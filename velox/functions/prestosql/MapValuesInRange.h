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

#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

/// map_values_in_range(map(K, V), lower_bound, upper_bound) -> map(K, V)
///
/// Returns a map containing only the entries from the input map whose values
/// fall within the specified range [lower_bound, upper_bound] (inclusive).
/// Entries with values less than lower_bound or greater than upper_bound are
/// removed. Entries with null values are preserved in the output.
/// If lower_bound or upper_bound is null, that bound is not applied.
template <typename TExec, typename K, typename V>
struct MapValuesInRangeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<K, V>>& out,
      const arg_type<Map<K, V>>& inputMap,
      const arg_type<V>& lowerBound,
      const arg_type<V>& upperBound) {
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        if constexpr (std::is_same_v<K, Varchar>) {
          keyWriter.copy_from(entry.first);
        } else {
          keyWriter = entry.first;
        }
        continue;
      }

      const auto& value = entry.second.value();

      if (value >= lowerBound && value <= upperBound) {
        auto [keyWriter, valueWriter] = out.add_item();
        if constexpr (std::is_same_v<K, Varchar>) {
          keyWriter.copy_from(entry.first);
        } else {
          keyWriter = entry.first;
        }
        if constexpr (std::is_same_v<V, Varchar>) {
          valueWriter.copy_from(value);
        } else {
          valueWriter = value;
        }
      }
    }
  }
};

/// Varchar key version for zero-copy semantics.
template <typename TExec, typename V>
struct MapValuesInRangeVarcharKeyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  // NOLINT: Framework-required variable name for zero-copy string semantics.
  static constexpr int32_t reuse_strings_from_arg = 0;

  void call(
      out_type<Map<Varchar, V>>& out,
      const arg_type<Map<Varchar, V>>& inputMap,
      const arg_type<V>& lowerBound,
      const arg_type<V>& upperBound) {
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
        continue;
      }

      const auto& value = entry.second.value();

      if (value >= lowerBound && value <= upperBound) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);
        if constexpr (std::is_same_v<V, Varchar>) {
          valueWriter.copy_from(value);
        } else {
          valueWriter = value;
        }
      }
    }
  }
};

/// Generic implementation for complex types.
template <typename TExec>
struct MapValuesInRangeGenericFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Generic<T1>, double>>& out,
      const arg_type<Map<Generic<T1>, double>>& inputMap,
      const double& lowerBound,
      const double& upperBound) {
    for (const auto& entry : inputMap) {
      if (!entry.second.has_value()) {
        auto& keyWriter = out.add_null();
        keyWriter.copy_from(entry.first);
        continue;
      }

      const auto& value = entry.second.value();

      if (value >= lowerBound && value <= upperBound) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter.copy_from(entry.first);
        valueWriter = value;
      }
    }
  }
};

} // namespace facebook::velox::functions
