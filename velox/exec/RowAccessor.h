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

#include "velox/common/base/CompareFlags.h"
#include "velox/type/Type.h"
#include "velox/vector/TypeAliases.h"

#include <concepts>
#include <optional>

namespace facebook::velox::exec {

/// Defines storage access required by window partition algorithms.
///
/// All row indexes are absolute positions in the window partition.
template <typename T>
concept RowAccessor = requires(
    const T& rows,
    vector_size_t row,
    vector_size_t lhs,
    vector_size_t rhs,
    column_index_t frameColumn,
    const CompareFlags& flags) {
  /// Returns the first row retained by the accessor.
  { rows.startRow() } -> std::same_as<vector_size_t>;

  /// Returns one past the last row retained by the accessor.
  { rows.partitionEnd() } -> std::same_as<vector_size_t>;

  /// Returns true if the accessor has retained the row immediately preceding
  /// startRow() for cross-batch peer comparison.
  { rows.hasPreviousRow() } -> std::same_as<bool>;

  /// Compares the retained previous row with 'row' using ORDER BY keys.
  { rows.previousRowEquals(row) } -> std::same_as<bool>;

  /// Compares two partition rows using ORDER BY keys.
  { rows.rowsEqual(lhs, rhs) } -> std::same_as<bool>;

  /// Compares the ORDER BY value at the first row argument with the k RANGE
  /// frame-bound value from 'frameColumn' at the second row argument. Returns 0
  /// for equality, < 0 or > 0 for ordered non-equal values according to
  /// 'flags', and nullopt if the comparison cannot produce an ordered result
  /// for this pair.
  {
    rows.compareFrameValue(row, row, frameColumn, flags)
  } -> std::same_as<std::optional<int32_t>>;

  /// Returns true if the frame-bound value in 'frameColumn' is null at 'row'.
  /// Nullness does not depend on the value type, so this is not templated.
  { rows.frameValueIsNull(row, frameColumn) } -> std::same_as<bool>;

  /// Returns true if the ORDER BY value is null at 'row'.
  { rows.orderByValueIsNull(row) } -> std::same_as<bool>;

  /// Returns true if the frame-bound value in 'frameColumn' is NaN at 'row'.
  /// NaN checks are templated on the column storage type so the typed load
  /// matches the column width. Window k RANGE frame bounds instantiate these
  /// methods only for REAL(float) and DOUBLE(double).
  {
    rows.template frameValueIsNan<float>(row, frameColumn)
  } -> std::same_as<bool>;
  {
    rows.template frameValueIsNan<double>(row, frameColumn)
  } -> std::same_as<bool>;

  /// Returns true if the ORDER BY value is NaN at 'row'. Checked for both
  /// floating-point types for the same reason as frameValueIsNan<T>().
  { rows.template orderByValueIsNan<float>(row) } -> std::same_as<bool>;
  { rows.template orderByValueIsNan<double>(row) } -> std::same_as<bool>;
};

} // namespace facebook::velox::exec
