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

#include <type_traits>

#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Macros.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

namespace detail {

// Returns true if 'lhs' is strictly less than 'rhs'. For floating-point
// types this uses the NaN-aware ordering used elsewhere in Velox/Presto:
// NaN compares greater than all non-NaN values, NaN == NaN.
template <typename T>
FOLLY_ALWAYS_INLINE bool nanAwareLess(const T& lhs, const T& rhs) {
  if constexpr (std::is_floating_point_v<T>) {
    return util::floating_point::NaNAwareLessThan<T>{}(lhs, rhs);
  } else {
    return lhs < rhs;
  }
}

// Returns true if 'lhs' is strictly greater than 'rhs', using NaN-aware
// ordering for floating-point types.
template <typename T>
FOLLY_ALWAYS_INLINE bool nanAwareGreater(const T& lhs, const T& rhs) {
  if constexpr (std::is_floating_point_v<T>) {
    return util::floating_point::NaNAwareGreaterThan<T>{}(lhs, rhs);
  } else {
    return lhs > rhs;
  }
}

} // namespace detail

/// map_subset_key_in_range(map(K, V), low_key, high_key) -> map(K, V)
///
/// Returns a sub-map containing only the entries from the input map whose
/// keys fall within the inclusive range [low_key, high_key]. Both bounds
/// are inclusive. If low_key > high_key, returns an empty map. Entries with
/// null values are preserved (Velox map keys cannot be null).
///
/// Null-input semantics: the function uses Velox's default null behavior
/// (`void call`, not `callNullable`), so if the input map, low_key, or
/// high_key is null, the result is null.
///
/// Floating-point keys use NaN-aware comparison: NaN is treated as greater
/// than all non-NaN values, and equal to itself.
///
/// Fast path for primitive key types.
template <typename TExec, typename K>
struct MapSubsetKeyInRangeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<K, Generic<T1>>>& out,
      const arg_type<Map<K, Generic<T1>>>& inputMap,
      const arg_type<K>& lowKey,
      const arg_type<K>& highKey) {
    if (detail::nanAwareGreater(lowKey, highKey)) {
      return;
    }

    for (const auto& entry : inputMap) {
      if (detail::nanAwareLess(entry.first, lowKey) ||
          detail::nanAwareGreater(entry.first, highKey)) {
        continue;
      }

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
};

/// Varchar key version with zero-copy semantics for keys.
template <typename TExec>
struct MapSubsetKeyInRangeVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  // NOLINT: Framework-required variable name for zero-copy string semantics.
  static constexpr int32_t reuse_strings_from_arg = 0;

  void call(
      out_type<Map<Varchar, Generic<T1>>>& out,
      const arg_type<Map<Varchar, Generic<T1>>>& inputMap,
      const arg_type<Varchar>& lowKey,
      const arg_type<Varchar>& highKey) {
    if (lowKey > highKey) {
      return;
    }

    for (const auto& entry : inputMap) {
      if (entry.first < lowKey || entry.first > highKey) {
        continue;
      }

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
};

/// Generic implementation for orderable key types not covered by the
/// primitive or varchar fast paths. Uses GenericView::compare with
/// kNullAsIndeterminate, which delegates to the type-aware comparator and
/// therefore already handles NaN according to Velox/Presto semantics for
/// any floating-point components nested inside the key.
template <typename TExec>
struct MapSubsetKeyInRangeGenericFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(
      out_type<Map<Orderable<T1>, Generic<T2>>>& out,
      const arg_type<Map<Orderable<T1>, Generic<T2>>>& inputMap,
      const arg_type<Orderable<T1>>& lowKey,
      const arg_type<Orderable<T1>>& highKey) {
    static constexpr CompareFlags kFlags{
        false /*nullsFirst*/,
        true /*ascending*/,
        false /*equalsOnly*/,
        CompareFlags::NullHandlingMode::kNullAsIndeterminate};

    auto lowVsHigh = lowKey.compare(highKey, kFlags);
    VELOX_USER_CHECK(
        lowVsHigh.has_value(), "Comparison on null elements is not supported");
    if (lowVsHigh.value() > 0) {
      return;
    }

    for (const auto& entry : inputMap) {
      auto lowCmp = entry.first.compare(lowKey, kFlags);
      VELOX_USER_CHECK(
          lowCmp.has_value(), "Comparison on null elements is not supported");
      if (lowCmp.value() < 0) {
        continue;
      }

      auto highCmp = entry.first.compare(highKey, kFlags);
      VELOX_USER_CHECK(
          highCmp.has_value(), "Comparison on null elements is not supported");
      if (highCmp.value() > 0) {
        continue;
      }

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
};

} // namespace facebook::velox::functions
