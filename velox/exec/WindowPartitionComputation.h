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

#include "velox/exec/WindowPartition.h"

#include <algorithm>
#include <type_traits>

namespace facebook::velox::exec {

/// Computes peer group bounds over storage-specific window partition rows.
class PeerGroupComputation {
 public:
  /// Computes peer starts and ends for a sequential range of rows. If 'rows'
  /// contains a previous row, clears it after using it for peer comparison.
  template <typename RowAccessor>
  static std::pair<vector_size_t, vector_size_t> computeBuffers(
      RowAccessor& rows,
      vector_size_t start,
      vector_size_t end,
      vector_size_t prevPeerStart,
      vector_size_t prevPeerEnd,
      vector_size_t* rawPeerStarts,
      vector_size_t* rawPeerEnds) {
    VELOX_CHECK_LE(start, rows.partitionEnd());
    VELOX_CHECK_LE(end, rows.partitionEnd());

    auto peerStart = prevPeerStart;
    auto peerEnd = prevPeerEnd;
    vector_size_t next = start;
    vector_size_t index = 0;

    if (rows.hasPreviousRow() && start < end) {
      const auto samePeer = rows.previousRowEquals(start);
      rows.clearPreviousRow();
      if (samePeer) {
        peerEnd = findEnd(rows, start, rows.partitionEnd());
        for (; next < std::min(end, peerEnd); ++next, ++index) {
          rawPeerStarts[index] = peerStart;
          rawPeerEnds[index] = peerEnd - 1;
        }
      }
    }

    for (; next < end; ++next, ++index) {
      if (next == 0 || next >= peerEnd) {
        peerStart = next;
        peerEnd = findEnd(rows, peerStart, rows.partitionEnd());
      }

      rawPeerStarts[index] = peerStart;
      rawPeerEnds[index] = peerEnd - 1;
    }

    VELOX_CHECK_EQ(index, end - start);
    return {peerStart, peerEnd};
  }

 private:
  template <typename RowAccessor>
  static vector_size_t findEnd(
      const RowAccessor& rows,
      vector_size_t peerStart,
      vector_size_t partitionEnd) {
    auto peerEnd = peerStart + 1;
    while (peerEnd < partitionEnd && rows.rowsEqual(peerStart, peerEnd)) {
      ++peerEnd;
    }
    return peerEnd;
  }
};

/// Searches RANGE frame bounds over storage-specific window partition rows.
class FrameBoundSearch {
 public:
  /// Finds the smallest frame start and largest frame end for selected rows.
  static std::pair<vector_size_t, vector_size_t> findMinMaxFrameBounds(
      const SelectivityVector& validRows,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds) {
    auto rawFrameStarts = frameStarts->as<vector_size_t>();
    auto rawFrameEnds = frameEnds->as<vector_size_t>();

    auto firstValidRow = validRows.begin();
    vector_size_t minFrame = rawFrameStarts[firstValidRow];
    vector_size_t maxFrame = rawFrameEnds[firstValidRow];
    validRows.applyToSelected([&](auto i) {
      minFrame = std::min(minFrame, rawFrameStarts[i]);
      maxFrame = std::max(maxFrame, rawFrameEnds[i]);
    });
    return {minFrame, maxFrame};
  }

  /// Computes k RANGE frame bounds for a storage-specific row accessor.
  template <typename RowAccessor>
  static void computeKRangeFrameBounds(
      const RowAccessor& rows,
      bool isStartBound,
      bool isPreceding,
      column_index_t frameColumn,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerBounds,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) {
    if (numRows == 0) {
      return;
    }

    CompareFlags flags;
    const auto sortOrder = rows.sortOrder();
    flags.ascending = sortOrder.isAscending();
    flags.nullsFirst = sortOrder.isNullsFirst();

    const auto frameType = rows.frameColumnType(frameColumn);
    if (frameType->isReal()) {
      updateKRangeFrameBounds<float>(
          rows,
          isStartBound,
          isPreceding,
          flags,
          frameColumn,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else if (frameType->isDouble()) {
      updateKRangeFrameBounds<double>(
          rows,
          isStartBound,
          isPreceding,
          flags,
          frameColumn,
          startRow,
          numRows,
          rawPeerBounds,
          rawFrameBounds,
          validFrames);
    } else {
      updateKRangeFrameBounds<void>(
          rows,
          isStartBound,
          isPreceding,
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
  template <typename RowAccessor>
  static vector_size_t linearSearch(
      const RowAccessor& rows,
      bool firstMatch,
      vector_size_t start,
      vector_size_t end,
      vector_size_t frameValueRow,
      column_index_t orderByColumn,
      column_index_t frameColumn,
      const CompareFlags& flags) {
    if (start >= end) {
      return end == rows.partitionEnd() ? rows.partitionEnd() + 1 : -1;
    }

    for (auto row = start; row < end; ++row) {
      const auto compareResult = rows.compareFrameValue(
          row, frameValueRow, orderByColumn, frameColumn, flags);
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
      column_index_t orderByColumn,
      column_index_t frameColumn,
      const CompareFlags& flags) {
    auto begin = start;
    auto finish = end;

    while (finish - begin >= 2) {
      const auto mid = begin + (finish - begin) / 2;
      const auto compareResult = rows.compareFrameValue(
          mid, frameValueRow, orderByColumn, frameColumn, flags);
      if (!compareResult.has_value() || compareResult.value() >= 0) {
        finish = mid;
      } else {
        begin = mid;
      }
    }

    return linearSearch(
        rows,
        firstMatch,
        begin,
        end,
        frameValueRow,
        orderByColumn,
        frameColumn,
        flags);
  }

  template <typename T, typename RowAccessor>
  static void updateKRangeFrameBounds(
      const RowAccessor& rows,
      bool isStartBound,
      bool isPreceding,
      const CompareFlags& flags,
      column_index_t frameColumn,
      vector_size_t startRow,
      vector_size_t numRows,
      const vector_size_t* rawPeerBounds,
      vector_size_t* rawFrameBounds,
      SelectivityVector& validFrames) {
    const auto orderByColumn = rows.orderByColumn();
    for (auto i = 0; i < numRows; ++i) {
      const auto currentRow = startRow + i;

      if constexpr (std::is_floating_point_v<T>) {
        if (rows.template isInvalidNanFrameBound<T>(
                currentRow, orderByColumn, frameColumn)) {
          validFrames.setValid(i, false);
          continue;
        }
      }

      rows.checkFrameAndOrderNulls(currentRow, orderByColumn, frameColumn);

      const auto compareResult = rows.compareFrameValue(
          currentRow, currentRow, orderByColumn, frameColumn, flags);
      if (compareResult.has_value() && compareResult.value() == 0) {
        rawFrameBounds[i] = rawPeerBounds[i];
      } else {
        const auto searchStart = isPreceding ? rows.startRow() : currentRow;
        const auto searchEnd =
            isPreceding ? currentRow + 1 : rows.partitionEnd();
        rawFrameBounds[i] = search(
            rows,
            isStartBound,
            searchStart,
            searchEnd,
            currentRow,
            orderByColumn,
            frameColumn,
            flags);
      }
    }
  }
};

} // namespace facebook::velox::exec
