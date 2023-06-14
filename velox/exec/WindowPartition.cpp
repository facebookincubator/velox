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
#include "velox/exec/WindowPartition.h"

namespace facebook::velox::exec {

WindowPartition::WindowPartition(
    RowContainer* data,
    const std::vector<exec::RowColumn>& columns,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo)
    : data_(data), columns_(columns), sortKeyInfo_(sortKeyInfo) {}

void WindowPartition::resetPartition(const folly::Range<char**>& rows) {
  partition_ = rows;
}

void WindowPartition::extractColumn(
    int32_t columnIndex,
    folly::Range<const vector_size_t*> rowNumbers,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  RowContainer::extractColumn(
      partition_.data(),
      rowNumbers,
      columns_[columnIndex],
      resultOffset,
      result);
}

void WindowPartition::extractColumn(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  RowContainer::extractColumn(
      partition_.data() + partitionOffset,
      numRows,
      columns_[columnIndex],
      resultOffset,
      result);
}

void WindowPartition::extractNulls(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    const BufferPtr& nullsBuffer) const {
  RowContainer::extractNulls(
      partition_.data() + partitionOffset,
      numRows,
      columns_[columnIndex],
      nullsBuffer);
}

namespace {

std::pair<vector_size_t, vector_size_t> findMinMaxFrameBounds(
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

}; // namespace

std::optional<std::pair<vector_size_t, vector_size_t>>
WindowPartition::extractNulls(
    column_index_t col,
    const SelectivityVector& validRows,
    const BufferPtr& frameStarts,
    const BufferPtr& frameEnds,
    BufferPtr* nulls) const {
  VELOX_CHECK(validRows.hasSelections(), "Buffer has no active rows");
  auto [minFrame, maxFrame] =
      findMinMaxFrameBounds(validRows, frameStarts, frameEnds);

  // Add 1 since maxFrame is the index of the frame end row.
  auto framesSize = maxFrame - minFrame + 1;
  AlignedBuffer::reallocate<bool>(nulls, framesSize);

  extractNulls(col, minFrame, framesSize, *nulls);
  auto foundNull =
      bits::findFirstBit((*nulls)->as<uint64_t>(), 0, framesSize) >= 0;
  return foundNull ? std::make_optional(std::make_pair(minFrame, framesSize))
                   : std::nullopt;
}

bool WindowPartition::compareRowsWithSortKeys(const char* lhs, const char* rhs)
    const {
  if (lhs == rhs) {
    return false;
  }
  for (auto& key : sortKeyInfo_) {
    if (auto result = data_->compare(
            lhs,
            rhs,
            key.first,
            {key.second.isNullsFirst(), key.second.isAscending(), false})) {
      return result < 0;
    }
  }
  return false;
}

std::pair<vector_size_t, vector_size_t> WindowPartition::computePeerBuffers(
    vector_size_t start,
    vector_size_t end,
    vector_size_t prevPeerStart,
    vector_size_t prevPeerEnd,
    vector_size_t* rawPeerStarts,
    vector_size_t* rawPeerEnds) const {
  auto peerCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithSortKeys(lhs, rhs);
  };

  VELOX_CHECK_LE(end, numRows());

  auto lastPartitionRow = numRows() - 1;
  auto peerStart = prevPeerStart;
  auto peerEnd = prevPeerEnd;
  for (auto i = start, j = 0; i < end; i++, j++) {
    // When traversing input partition rows, the peers are the rows
    // with the same values for the ORDER BY clause. These rows
    // are equal in some ways and affect the results of ranking functions.
    // This logic exploits the fact that all rows between the peerStart
    // and peerEnd have the same values for rawPeerStarts and rawPeerEnds.
    // So we can compute them just once and reuse across the rows in that peer
    // interval. Note: peerStart and peerEnd can be maintained across
    // getOutput calls. Hence, they are returned to the caller.

    if (i == 0 || i >= peerEnd) {
      // Compute peerStart and peerEnd rows for the first row of the partition
      // or when past the previous peerGroup.
      peerStart = i;
      peerEnd = i;
      while (peerEnd <= lastPartitionRow) {
        if (peerCompare(partition_[peerStart], partition_[peerEnd])) {
          break;
        }
        peerEnd++;
      }
    }

    rawPeerStarts[j] = peerStart;
    rawPeerEnds[j] = peerEnd - 1;
  }
  return {peerStart, peerEnd};
}

// Searches for start[frameColumn] in orderByColumn. Depending on
// preceding or following, this function traverses from start
// to the respective end of partition looking for the frame value.
// The current implementation is a very naive sequential search.
// There are few ideas for future optimizations:
// i)  The current code traverses from start to param.limit
//  a single row at a time. This can be improved to skipping
//  multiple rows.
// ii) Binary search style.
// iii) Use cached value of previous row result to start searching
// from instead of the current start row. Since row values
// show good locality this could give good results.
template <typename BoundTest>
vector_size_t WindowPartition::searchFrameValue(
    const RangeSearchParams<BoundTest>& params,
    vector_size_t start,
    column_index_t orderByColumn,
    column_index_t frameColumn) const {
  auto startRow = partition_[start];
  for (vector_size_t i = start; i >= 0 && i < numRows(); i += params.step) {
    auto compareResult =
        data_->compare(partition_[i], startRow, orderByColumn, frameColumn);

    // The bound value was found. Return if firstMatch required.
    // If the last match is required, then we need to find the first row that
    // crosses the bound and return the previous (or following, based on skip)
    // row.
    if (compareResult == 0) {
      if (params.firstMatch) {
        return i;
      }
    }

    // Bound is crossed. Last match needs the previous row.
    // But for first row matches, this is the first
    // row that has crossed, but not equals boundary (The equal boundary case
    // is covered by the condition above). So the bound matches this row itself.
    if (params.boundTest(compareResult)) {
      if (params.firstMatch) {
        return i;
      } else {
        return i - params.step;
      }
    }
  }

  // Return a row beyond the partition boundary. The logic to determine valid
  // frames handles the out of bound and empty frames from this value.
  return params.step == 1 ? numRows() + 1 : -1;
}

template <typename BoundTest>
void WindowPartition::updateKRangeFrameBounds(
    const RangeSearchParams<BoundTest>& params,
    column_index_t frameColumn,
    vector_size_t startRow,
    vector_size_t numRows,
    const vector_size_t* rawPeerBounds,
    vector_size_t* rawFrameBounds) const {
  column_index_t orderByColumn = sortKeyInfo_[0].first;
  RowColumn frameRowColumn = data_->columnAt(frameColumn);

  for (auto i = 0; i < numRows; i++) {
    auto currentRow = startRow + i;
    bool frameIsNull = RowContainer::isNullAt(
        partition_[currentRow],
        frameRowColumn.nullByte(),
        frameRowColumn.nullMask());
    // For NULL values, CURRENT ROW semantics apply. So get frame bound from
    // peer buffer.
    if (frameIsNull) {
      rawFrameBounds[i] = rawPeerBounds[i];
    } else {
      // This does a naive search that looks for the frame value from the
      // current row to the partition boundary in a sequential manner. This
      // search can be optimized to start from a previously cached row value
      // instead.
      rawFrameBounds[i] =
          searchFrameValue(params, currentRow, orderByColumn, frameColumn);
    }
  }
}

void WindowPartition::computeKRangeFrameBounds(
    bool isStartBound,
    bool isPreceding,
    column_index_t frameColumn,
    vector_size_t startRow,
    vector_size_t numRows,
    const vector_size_t* rawPeerBuffer,
    vector_size_t* rawFrameBounds) const {
  typedef bool (*boundTest)(int);

  if (isPreceding) {
    updateKRangeFrameBounds(
        RangeSearchParams<boundTest>(
            {!isStartBound,
             -1,
             sortKeyInfo_[0].second.isAscending()
             ? [](int compareResult) -> bool { return compareResult < 0; }
             : [](int compareResult) -> bool { return compareResult > 0; }}),
        frameColumn,
        startRow,
        numRows,
        rawPeerBuffer,
        rawFrameBounds);
  } else {
    updateKRangeFrameBounds(
        RangeSearchParams<boundTest>(
            {isStartBound,
             1,
             sortKeyInfo_[0].second.isAscending()
             ? [](int compareResult) -> bool { return compareResult > 0; }
             : [](int compareResult) -> bool { return compareResult < 0; }}),
        frameColumn,
        startRow,
        numRows,
        rawPeerBuffer,
        rawFrameBounds);
  }
}

} // namespace facebook::velox::exec
