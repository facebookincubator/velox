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
#include "velox/exec/KRangeFrameBound.h"
#include "velox/exec/PeerGroupComputation.h"

#include <algorithm>

namespace facebook::velox::exec {

WindowPartition::WindowPartition(
    RowContainer* data,
    const folly::Range<char**>& rows,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo,
    bool partial,
    bool complete)
    : partial_(partial),
      data_(data),
      partition_(rows),
      complete_(complete),
      inputMapping_(inputMapping),
      sortKeyInfo_(sortKeyInfo) {
  VELOX_CHECK_NE(partial_, complete_);
  VELOX_CHECK_NE(complete_, partition_.empty());

  for (auto index : inputMapping_) {
    columns_.emplace_back(data_->columnAt(index));
  }
}

WindowPartition::WindowPartition(
    RowContainer* data,
    const folly::Range<char**>& rows,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo)
    : WindowPartition(data, rows, inputMapping, sortKeyInfo, false, true) {}

WindowPartition::WindowPartition(
    RowContainer* data,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo)
    : WindowPartition(data, {}, inputMapping, sortKeyInfo, true, false) {}

void WindowPartition::addRows(const std::vector<char*>& rows) {
  checkPartial();
  rows_.insert(rows_.end(), rows.begin(), rows.end());
  partition_ = folly::Range(rows_.data(), rows_.size());
}

void WindowPartition::eraseRows(vector_size_t numRows) {
  checkPartial();
  VELOX_CHECK_GE(data_->numRows(), numRows);
  data_->eraseRows(folly::Range<char**>(rows_.data(), numRows));
}

void WindowPartition::removeProcessedRows(vector_size_t numRows) {
  checkPartial();

  VELOX_CHECK_NULL(previousRow_);
  if (complete_ && rows_.size() == numRows) {
    eraseRows(numRows);
  } else {
    eraseRows(numRows - 1);
    previousRow_ = rows_[numRows - 1];
  }

  rows_.erase(rows_.cbegin(), rows_.cbegin() + numRows);
  partition_ = folly::Range(rows_.data(), rows_.size());
  startRow_ += numRows;
}

vector_size_t WindowPartition::numRowsForProcessing(
    vector_size_t partitionOffset) const {
  if (partial_) {
    return partition_.size();
  } else {
    return partition_.size() - partitionOffset;
  }
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
      data_->columnHasNulls(inputMapping_[columnIndex]),
      resultOffset,
      result);
}

void WindowPartition::extractColumn(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  VELOX_CHECK_GE(partitionOffset, startRow_);
  RowContainer::extractColumn(
      partition_.data() + partitionOffset - startRow_,
      numRows,
      columns_[columnIndex],
      data_->columnHasNulls(inputMapping_[columnIndex]),
      resultOffset,
      result);
}

void WindowPartition::extractNulls(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    const BufferPtr& nullsBuffer) const {
  VELOX_CHECK_GE(partitionOffset, startRow_);
  RowContainer::extractNulls(
      partition_.data() + partitionOffset - startRow_,
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

} // namespace

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

void WindowPartition::removePreviousRow() {
  VELOX_CHECK_NOT_NULL(previousRow_);
  data_->eraseRows(folly::Range<char**>(&previousRow_, 1));
  previousRow_ = nullptr;
}

class WindowPartition::RowContainerPeerAccessor {
 public:
  explicit RowContainerPeerAccessor(WindowPartition& partition)
      : partition_(partition) {}

  vector_size_t partitionEnd() const {
    return partition_.startRow_ + partition_.partition_.size();
  }

  bool hasPreviousRow() const {
    return partition_.previousRow_ != nullptr;
  }

  bool previousRowEquals(vector_size_t row) const {
    return !partition_.compareRowsWithSortKeys(
        partition_.previousRow_, rowAt(row));
  }

  bool rowsEqual(vector_size_t lhs, vector_size_t rhs) const {
    return !partition_.compareRowsWithSortKeys(rowAt(lhs), rowAt(rhs));
  }

 private:
  char* rowAt(vector_size_t row) const {
    return partition_.partition_[row - partition_.startRow_];
  }

  WindowPartition& partition_;
};

std::pair<vector_size_t, vector_size_t> WindowPartition::computePeerBuffers(
    vector_size_t start,
    vector_size_t end,
    vector_size_t prevPeerStart,
    vector_size_t prevPeerEnd,
    vector_size_t* rawPeerStarts,
    vector_size_t* rawPeerEnds) {
  RowContainerPeerAccessor rows{*this};
  auto result = PeerGroupComputation::compute(
      rows, start, end, prevPeerStart, prevPeerEnd, rawPeerStarts, rawPeerEnds);
  if (result.previousRowConsumed) {
    removePreviousRow();
  }
  return {result.peerStart, result.peerEnd};
}

class WindowPartition::RowContainerKRangeFrameAccessor {
 public:
  RowContainerKRangeFrameAccessor(
      const WindowPartition& partition,
      column_index_t orderByColumn,
      column_index_t frameColumn)
      : partition_(partition),
        orderByColumn_(orderByColumn),
        mappedFrameColumn_(partition.inputMapping_[frameColumn]),
        orderByRowColumn_(partition.data_->columnAt(orderByColumn)),
        frameRowColumn_(partition.columns_[frameColumn]) {}

  vector_size_t startRow() const {
    return partition_.startRow_;
  }

  vector_size_t partitionEnd() const {
    return partition_.startRow_ + partition_.partition_.size();
  }

  std::optional<int32_t> compareFrameValue(
      vector_size_t orderByRow,
      vector_size_t frameValueRow,
      const CompareFlags& flags) const {
    return partition_.data_->compare(
        rowAt(orderByRow),
        rowAt(frameValueRow),
        orderByColumn_,
        mappedFrameColumn_,
        flags);
  }

  bool frameValueIsNull(vector_size_t row) const {
    return isNullAt(rowAt(row), frameRowColumn_);
  }

  bool orderByValueIsNull(vector_size_t row) const {
    return isNullAt(rowAt(row), orderByRowColumn_);
  }

  template <typename T>
  bool frameValueIsNan(vector_size_t row) const {
    return partition_.data_->isNanAt<T>(rowAt(row), frameRowColumn_);
  }

  template <typename T>
  bool orderByValueIsNan(vector_size_t row) const {
    return partition_.data_->isNanAt<T>(rowAt(row), orderByRowColumn_);
  }

 private:
  char* rowAt(vector_size_t row) const {
    return partition_.partition_[row - partition_.startRow_];
  }

  static bool isNullAt(const char* row, RowColumn column) {
    return RowContainer::isNullAt(row, column.nullByte(), column.nullMask());
  }

  const WindowPartition& partition_;
  const column_index_t orderByColumn_;
  const column_index_t mappedFrameColumn_;
  const RowColumn orderByRowColumn_;
  const RowColumn frameRowColumn_;
};

void WindowPartition::computeKRangeFrameBounds(
    bool isStartBound,
    bool isPreceding,
    column_index_t frameColumn,
    vector_size_t startRow,
    vector_size_t numRows,
    const vector_size_t* rawPeerBuffer,
    vector_size_t* rawFrameBounds,
    SelectivityVector& validFrames) const {
  const auto orderByColumn = sortKeyInfo_[0].first;
  const auto sortOrder = sortKeyInfo_[0].second;
  const auto frameType = data_->columnTypes()[inputMapping_[frameColumn]];

  RowContainerKRangeFrameAccessor rows{*this, orderByColumn, frameColumn};
  KRangeFrameBound::compute(
      rows,
      isStartBound,
      isPreceding,
      sortOrder,
      frameType,
      startRow,
      numRows,
      rawPeerBuffer,
      rawFrameBounds,
      validFrames);
}

} // namespace facebook::velox::exec
