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
#include "velox/common/base/Exceptions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/window/WindowPartitionAccessor.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/TypeAliases.h"

#include <algorithm>
#include <type_traits>

namespace facebook::velox::exec::window {

/// Computes k RANGE frame bounds over storage-specific window partition rows.
class KRangeFrameBound {
 public:
  /// Computes k RANGE frame bounds for rows in [startRow, startRow + numRows).
  template <WindowPartitionAccessor Rows>
  static void compute(
      const Rows& rows,
      bool isStartBound,
      column_index_t frameColumn,
      const core::SortOrder& sortOrder,
      const TypePtr& frameType,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerBounds,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) {
    if (numRows == 0) {
      return;
    }

    CompareFlags flags;
    flags.ascending = sortOrder.isAscending();
    flags.nullsFirst = sortOrder.isNullsFirst();

    if (frameType->isReal()) {
      computeTyped<float>(
          rows,
          isStartBound,
          flags,
          frameColumn,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else if (frameType->isDouble()) {
      computeTyped<double>(
          rows,
          isStartBound,
          flags,
          frameColumn,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else {
      computeTyped<void>(
          rows,
          isStartBound,
          flags,
          frameColumn,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    }
  }

 private:
  // Returns the first row in [begin, end) for which 'isAfter' is true, or
  // 'end' if there is none. 'isAfter' must be false before the result and true
  // from it on. Checks rows 1, 2, 4, ... away from 'hint', then binary searches
  // the last gap. Positions are int64_t so that the steps cannot overflow
  // vector_size_t.
  template <typename IsAfter>
  static vector_size_t
  gallop(int64_t begin, int64_t end, int64_t hint, const IsAfter& isAfter) {
    hint = std::clamp(hint, begin, end);

    // Brackets the result in [low, high], where 'high' is 'end' or a row for
    // which 'isAfter' is true.
    int64_t low = begin;
    int64_t high = end;
    if (hint < end && !isAfter(hint)) {
      low = hint + 1;
      for (int64_t step = 1; hint + step < end; step *= 2) {
        if (isAfter(hint + step)) {
          high = hint + step;
          break;
        }
        low = hint + step + 1;
      }
    } else {
      high = hint;
      for (int64_t step = 1; hint - step >= begin; step *= 2) {
        if (!isAfter(hint - step)) {
          low = hint - step + 1;
          break;
        }
        high = hint - step;
      }
    }

    while (low < high) {
      const auto mid = low + (high - low) / 2;
      if (isAfter(mid)) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return static_cast<vector_size_t>(low);
  }

  template <typename T, WindowPartitionAccessor Rows>
  static void computeTyped(
      const Rows& rows,
      bool isStartBound,
      const CompareFlags& flags,
      column_index_t frameColumn,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerBounds,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) {
    // Successive rows have nearby frame bounds, so each search starts at the
    // previous one.
    vector_size_t hint = startRow;
    for (auto i = 0; i < numRows; ++i) {
      const auto currentRow = startRow + i;

      if constexpr (std::is_floating_point_v<T>) {
        if (rows.template frameValueIsNan<T>(currentRow, frameColumn) &&
            !rows.template orderByValueIsNan<T>(currentRow)) {
          validFrames.setValid(i, false);
          continue;
        }
      }

      VELOX_USER_CHECK_EQ(
          rows.frameValueIsNull(currentRow, frameColumn),
          rows.orderByValueIsNull(currentRow),
          "k RANGE frame bound must be null exactly when the ORDER BY key is null");

      const auto compareResult =
          rows.compareFrameValue(currentRow, currentRow, frameColumn, flags);
      if (compareResult.has_value() && compareResult.value() == 0) {
        rawFrameBounds[i] = rawPeerBounds[i];
        continue;
      }

      // The first row at or after the bound value for a start bound, or
      // strictly after it for an end bound.
      const auto position =
          gallop(rows.startRow(), rows.partitionEnd(), hint, [&](int64_t row) {
            const auto result = rows.compareFrameValue(
                static_cast<vector_size_t>(row),
                currentRow,
                frameColumn,
                flags);
            // 'flags' compare nulls as values, so the result is always
            // ordered.
            VELOX_CHECK(result.has_value());
            return isStartBound ? result.value() >= 0 : result.value() > 0;
          });
      hint = position;
      // A bound value after every row is reported as partitionEnd() + 1.
      if (position == rows.partitionEnd()) {
        rawFrameBounds[i] = rows.partitionEnd() + 1;
      } else {
        rawFrameBounds[i] = isStartBound ? position : position - 1;
      }
    }
  }
};

} // namespace facebook::velox::exec::window
