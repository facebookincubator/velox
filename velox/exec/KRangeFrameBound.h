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
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/TypeAliases.h"

#include <optional>
#include <type_traits>

namespace facebook::velox::exec {

/// Computes k RANGE frame bounds over storage-specific window partition rows.
class KRangeFrameBound {
 public:
  /// Computes k RANGE frame bounds for rows in [startRow, startRow + numRows).
  ///
  /// RowAccessor contract:
  /// - startRow() returns the absolute first row retained by the accessor.
  /// - partitionEnd() returns the absolute end row of retained rows.
  /// - compareFrameValue(orderByRow, frameValueRow, flags) compares the
  ///   order-by value at 'orderByRow' with the frame-bound value at
  ///   'frameValueRow'.
  /// - frameValueIsNull(row) and orderByValueIsNull(row) expose null states
  ///   for the debug invariant.
  /// - frameValueIsNan<T>(row) and orderByValueIsNan<T>(row) expose typed
  ///   NaN checks for floating-point k RANGE frame bounds.
  ///
  /// The accessor is read-only. Schema metadata is passed as arguments rather
  /// than exposed through the accessor.
  template <typename RowAccessor>
  static void compute(
      const RowAccessor& rows,
      bool isStartBound,
      bool isPreceding,
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
          isPreceding,
          flags,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else if (frameType->isDouble()) {
      computeTyped<double>(
          rows,
          isStartBound,
          isPreceding,
          flags,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else {
      computeTyped<void>(
          rows,
          isStartBound,
          isPreceding,
          flags,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    }
  }

 private:
  template <typename RowAccessor>
  static vector_size_t linearSearch(
      const RowAccessor& rows,
      bool firstMatch,
      vector_size_t start,
      vector_size_t end,
      vector_size_t frameValueRow,
      const CompareFlags& flags) {
    if (start >= end) {
      return end == rows.partitionEnd() ? rows.partitionEnd() + 1 : -1;
    }

    for (auto row = start; row < end; ++row) {
      const auto compareResult =
          rows.compareFrameValue(row, frameValueRow, flags);
      if (compareResult.has_value() && compareResult.value() == 0) {
        if (firstMatch) {
          return row;
        }
      }

      if (compareResult.has_value() && compareResult.value() > 0) {
        return firstMatch ? row : row - 1;
      }
    }

    return end == rows.partitionEnd() ? rows.partitionEnd() + 1 : -1;
  }

  template <typename RowAccessor>
  static vector_size_t search(
      const RowAccessor& rows,
      bool firstMatch,
      vector_size_t start,
      vector_size_t end,
      vector_size_t frameValueRow,
      const CompareFlags& flags) {
    auto begin = start;
    auto finish = end;

    while (finish - begin >= 2) {
      const auto mid = begin + (finish - begin) / 2;
      const auto compareResult =
          rows.compareFrameValue(mid, frameValueRow, flags);
      if (!compareResult.has_value() || compareResult.value() >= 0) {
        finish = mid;
      } else {
        begin = mid;
      }
    }

    return linearSearch(rows, firstMatch, begin, end, frameValueRow, flags);
  }

  template <typename T, typename RowAccessor>
  static void computeTyped(
      const RowAccessor& rows,
      bool isStartBound,
      bool isPreceding,
      const CompareFlags& flags,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerBounds,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) {
    for (auto i = 0; i < numRows; ++i) {
      const auto currentRow = startRow + i;

      if constexpr (std::is_floating_point_v<T>) {
        if (rows.template frameValueIsNan<T>(currentRow) &&
            !rows.template orderByValueIsNan<T>(currentRow)) {
          validFrames.setValid(i, false);
          continue;
        }
      }

      VELOX_DCHECK_EQ(
          rows.frameValueIsNull(currentRow),
          rows.orderByValueIsNull(currentRow));

      const auto compareResult =
          rows.compareFrameValue(currentRow, currentRow, flags);
      if (compareResult.has_value() && compareResult.value() == 0) {
        rawFrameBounds[i] = rawPeerBounds[i];
      } else {
        const auto searchStart = isPreceding ? rows.startRow() : currentRow;
        const auto searchEnd =
            isPreceding ? currentRow + 1 : rows.partitionEnd();
        rawFrameBounds[i] = search(
            rows, isStartBound, searchStart, searchEnd, currentRow, flags);
      }
    }
  }
};

} // namespace facebook::velox::exec
