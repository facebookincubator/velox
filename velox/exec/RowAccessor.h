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
/// The row indexes are absolute positions in the window partition. The previous
/// row methods expose a retained row immediately before the current batch for
/// cross-batch peer comparison; callers own the retained row lifecycle.
/// Frame-value methods compare or inspect the frame-bound column for k RANGE
/// frames, while order-by methods inspect the first ORDER BY column.
template <typename T>
concept RowAccessor = requires(
    const T& rows,
    vector_size_t row,
    vector_size_t lhs,
    vector_size_t rhs,
    column_index_t frameColumn,
    const CompareFlags& flags) {
  { rows.startRow() } -> std::same_as<vector_size_t>;
  { rows.partitionEnd() } -> std::same_as<vector_size_t>;

  { rows.hasPreviousRow() } -> std::same_as<bool>;
  { rows.previousRowEquals(row) } -> std::same_as<bool>;
  { rows.rowsEqual(lhs, rhs) } -> std::same_as<bool>;

  {
    rows.compareFrameValue(row, row, frameColumn, flags)
  } -> std::same_as<std::optional<int32_t>>;
  { rows.frameValueIsNull(row, frameColumn) } -> std::same_as<bool>;
  { rows.orderByValueIsNull(row) } -> std::same_as<bool>;

  {
    rows.template frameValueIsNan<float>(row, frameColumn)
  } -> std::same_as<bool>;
  { rows.template orderByValueIsNan<float>(row) } -> std::same_as<bool>;
  {
    rows.template frameValueIsNan<double>(row, frameColumn)
  } -> std::same_as<bool>;
  { rows.template orderByValueIsNan<double>(row) } -> std::same_as<bool>;
};

} // namespace facebook::velox::exec
