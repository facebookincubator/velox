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
#include "velox/exec/window/VectorWindowPartition.h"
#include "velox/vector/SimpleVector.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace facebook::velox::exec::window {

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

VectorWindowPartition::VectorWindowPartition(
    const std::vector<column_index_t>& inputChannels,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo)
    : WindowPartition(inputMapping, sortKeyInfo, true),
      inputChannels_(inputChannels) {
  blockPrefixSums_.push_back(0);
}

vector_size_t VectorWindowPartition::numRows() const {
  return totalRows_;
}

vector_size_t VectorWindowPartition::numRowsForProcessing(
    vector_size_t /*partitionOffset*/) const {
  return totalRows_;
}

void VectorWindowPartition::addRows(const std::vector<char*>& /*rows*/) {
  VELOX_FAIL("VectorWindowPartition does not support RowContainer rows");
}

void VectorWindowPartition::addBlock(
    const RowVectorPtr& input,
    vector_size_t startRow,
    vector_size_t endRow) {
  VELOX_CHECK_NOT_NULL(input, "Input vector must not be null");
  VELOX_CHECK_LE(
      startRow, endRow, "startRow must be less than or equal to endRow");
  VELOX_CHECK_LE(
      endRow, input->size(), "endRow must be less than or equal to input size");

  RowBlock block{input, startRow, endRow};
  if (block.size() == 0) {
    return;
  }

  blocks_.push_back(block);
  blockPrefixSums_.push_back(blockPrefixSums_.back() + block.size());
  totalRows_ += block.size();
}

void VectorWindowPartition::removeProcessedRows(vector_size_t numRows) {
  VELOX_CHECK_LE(numRows, totalRows_);
  if (numRows == 0) {
    return;
  }

  if (complete() && numRows == totalRows_) {
    previousRef_ = {};
  } else {
    previousRef_ = rowAt(startRow_ + numRows - 1);
  }

  auto remaining = numRows;
  while (remaining > 0) {
    auto& block = blocks_.front();
    const auto blockSize = block.size();
    if (remaining >= blockSize) {
      blocks_.pop_front();
      remaining -= blockSize;
    } else {
      block.startRow += remaining;
      remaining = 0;
    }
  }

  startRow_ += numRows;
  rebuildPrefixSums();
}

void VectorWindowPartition::extractColumn(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  VELOX_CHECK_GE(partitionOffset, startRow_);
  if (numRows == 0) {
    return;
  }

  result->resize(resultOffset + numRows);

  auto [blockIndex, localRow] = findBlock(partitionOffset - startRow_);
  auto remaining = numRows;
  auto outputOffset = resultOffset;
  while (remaining > 0) {
    const auto& block = blocks_[blockIndex];
    const auto numRowsToCopy = std::min(block.endRow - localRow, remaining);
    result->copy(
        block.input->childAt(columnIndex).get(),
        outputOffset,
        localRow,
        numRowsToCopy);

    outputOffset += numRowsToCopy;
    remaining -= numRowsToCopy;
    if (remaining > 0) {
      ++blockIndex;
      localRow = blocks_[blockIndex].startRow;
    }
  }
}

void VectorWindowPartition::extractColumn(
    int32_t columnIndex,
    folly::Range<const vector_size_t*> rowNumbers,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  if (rowNumbers.empty()) {
    return;
  }

  result->resize(resultOffset + rowNumbers.size());

  for (auto i = 0; i < rowNumbers.size(); ++i) {
    const auto rowNumber = rowNumbers[i];
    if (rowNumber < 0) {
      result->setNull(resultOffset + i, true);
      continue;
    }

    VELOX_CHECK_GE(rowNumber, startRow_);
    const auto [blockIndex, localRow] = findBlock(rowNumber - startRow_);
    result->copy(
        blocks_[blockIndex].input->childAt(columnIndex).get(),
        resultOffset + i,
        localRow,
        1);
  }
}

void VectorWindowPartition::extractNulls(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    const BufferPtr& nullsBuffer) const {
  VELOX_CHECK_GE(partitionOffset, startRow_);
  if (numRows == 0) {
    return;
  }

  auto* rawNulls = nullsBuffer->asMutable<uint64_t>();
  bits::fillBits(rawNulls, 0, numRows, false);

  auto [blockIndex, localRow] = findBlock(partitionOffset - startRow_);
  vector_size_t processedRows = 0;
  while (processedRows < numRows) {
    const auto& block = blocks_[blockIndex];
    const auto input = block.input->childAt(columnIndex);
    const auto numRowsToProcess =
        std::min(block.endRow - localRow, numRows - processedRows);

    for (auto i = 0; i < numRowsToProcess; ++i) {
      if (input->isNullAt(localRow + i)) {
        bits::setBit(rawNulls, processedRows + i, true);
      }
    }

    processedRows += numRowsToProcess;
    if (processedRows < numRows) {
      ++blockIndex;
      localRow = blocks_[blockIndex].startRow;
    }
  }
}

std::optional<std::pair<vector_size_t, vector_size_t>>
VectorWindowPartition::extractNulls(
    column_index_t col,
    const SelectivityVector& validRows,
    const BufferPtr& frameStarts,
    const BufferPtr& frameEnds,
    BufferPtr* nulls) const {
  VELOX_CHECK(validRows.hasSelections(), "Buffer has no active rows");
  auto [minFrame, maxFrame] =
      findMinMaxFrameBounds(validRows, frameStarts, frameEnds);

  const auto framesSize = maxFrame - minFrame + 1;
  AlignedBuffer::reallocate<bool>(nulls, framesSize);

  extractNulls(col, minFrame, framesSize, *nulls);
  const auto foundNull =
      bits::findFirstBit((*nulls)->as<uint64_t>(), 0, framesSize) >= 0;
  return foundNull ? std::make_optional(std::make_pair(minFrame, framesSize))
                   : std::nullopt;
}

std::pair<vector_size_t, vector_size_t>
VectorWindowPartition::computePeerBuffers(
    vector_size_t start,
    vector_size_t end,
    vector_size_t prevPeerStart,
    vector_size_t prevPeerEnd,
    vector_size_t* rawPeerStarts,
    vector_size_t* rawPeerEnds) {
  VELOX_CHECK_LE(start, startRow_ + totalRows_);
  VELOX_CHECK_LE(end, startRow_ + totalRows_);

  auto peerStart = prevPeerStart;
  auto peerEnd = prevPeerEnd;
  vector_size_t next = start;
  vector_size_t index = 0;

  if (previousRef_.isValid() && start < end) {
    const auto samePeer = rowsEqual(previousRef_, rowAt(start), sortKeyInfo());
    previousRef_ = {};
    if (samePeer) {
      peerEnd = findPeerRowEndIndex(start, startRow_ + totalRows_);
      for (; next < std::min(end, peerEnd); ++next, ++index) {
        rawPeerStarts[index] = peerStart;
        rawPeerEnds[index] = peerEnd - 1;
      }
    }
  }

  for (; next < end; ++next, ++index) {
    if (next == 0 || next >= peerEnd) {
      peerStart = next;
      peerEnd = findPeerRowEndIndex(peerStart, startRow_ + totalRows_);
    }

    rawPeerStarts[index] = peerStart;
    rawPeerEnds[index] = peerEnd - 1;
  }

  VELOX_CHECK_EQ(index, end - start);
  return {peerStart, peerEnd};
}

template <typename T>
void VectorWindowPartition::updateKRangeFrameBounds(
    bool isStartBound,
    bool isPreceding,
    column_index_t frameColumn,
    vector_size_t startRow,
    vector_size_t numRows,
    const vector_size_t* rawPeerBounds,
    vector_size_t* rawFrameBounds,
    SelectivityVector& validFrames) const {
  const auto orderByColumn = inputChannels_[sortKeyInfo()[0].first];
  const auto sortOrder = sortKeyInfo()[0].second;
  CompareFlags flags;
  flags.ascending = sortOrder.isAscending();
  flags.nullsFirst = sortOrder.isNullsFirst();

  auto [blockIndex, localRow] = findBlock(startRow - startRow_);
  auto blockEndRow = blockPrefixSums_[blockIndex + 1] + startRow_;
  for (auto i = 0; i < numRows; ++i) {
    const auto currentRow = startRow + i;
    while (currentRow >= blockEndRow) {
      ++blockIndex;
      localRow = blocks_[blockIndex].startRow;
      blockEndRow = blockPrefixSums_[blockIndex + 1] + startRow_;
    }

    const auto& block = blocks_[blockIndex];
    const auto frameValue = block.input->childAt(frameColumn);
    const auto orderByValue = block.input->childAt(orderByColumn);
    if (isInvalidNanFrameBound<T>(frameValue, orderByValue, localRow)) {
      validFrames.setValid(i, false);
    } else if (
        orderByValue->compare(frameValue.get(), localRow, localRow, flags) ==
        0) {
      rawFrameBounds[i] = rawPeerBounds[i];
    } else {
      const auto searchStart = isPreceding ? startRow_ : currentRow;
      const auto searchEnd =
          isPreceding ? currentRow + 1 : startRow_ + totalRows_;
      rawFrameBounds[i] = searchFrameValue(
          isStartBound,
          searchStart,
          searchEnd,
          orderByColumn,
          frameValue,
          localRow,
          flags);
    }

    ++localRow;
  }
}

template <typename T>
bool VectorWindowPartition::isInvalidNanFrameBound(
    const VectorPtr& frameValue,
    const VectorPtr& orderByValue,
    vector_size_t row) const {
  if constexpr (std::is_floating_point_v<T>) {
    return isNanAt<T>(frameValue, row) && !isNanAt<T>(orderByValue, row);
  }
  return false;
}

template <typename T>
bool VectorWindowPartition::isNanAt(const VectorPtr& vector, vector_size_t row)
    const {
  return !vector->isNullAt(row) &&
      std::isnan(vector->loadedVector()->as<SimpleVector<T>>()->valueAt(row));
}

void VectorWindowPartition::computeKRangeFrameBounds(
    bool isStartBound,
    bool isPreceding,
    column_index_t frameColumn,
    vector_size_t startRow,
    vector_size_t numRows,
    const vector_size_t* rawPeerBounds,
    vector_size_t* rawFrameBounds,
    SelectivityVector& validFrames) const {
  if (numRows == 0) {
    return;
  }

  const auto frameType = blocks_.front().input->childAt(frameColumn)->type();
  if (frameType->isReal()) {
    updateKRangeFrameBounds<float>(
        isStartBound,
        isPreceding,
        frameColumn,
        startRow,
        numRows,
        rawPeerBounds,
        rawFrameBounds,
        validFrames);
  } else if (frameType->isDouble()) {
    updateKRangeFrameBounds<double>(
        isStartBound,
        isPreceding,
        frameColumn,
        startRow,
        numRows,
        rawPeerBounds,
        rawFrameBounds,
        validFrames);
  } else {
    updateKRangeFrameBounds<void>(
        isStartBound,
        isPreceding,
        frameColumn,
        startRow,
        numRows,
        rawPeerBounds,
        rawFrameBounds,
        validFrames);
  }
}

std::pair<size_t, vector_size_t> VectorWindowPartition::findBlock(
    vector_size_t row) const {
  VELOX_CHECK_LT(row, totalRows_);

  const auto it =
      std::upper_bound(blockPrefixSums_.begin(), blockPrefixSums_.end(), row);
  const size_t blockIndex = std::distance(blockPrefixSums_.begin(), it) - 1;
  const auto offsetInBlock = row - blockPrefixSums_[blockIndex];
  return {blockIndex, blocks_[blockIndex].startRow + offsetInBlock};
}

VectorWindowPartition::RowReference VectorWindowPartition::rowAt(
    vector_size_t row) const {
  VELOX_CHECK_GE(row, startRow_);
  const auto [blockIndex, localRow] = findBlock(row - startRow_);
  return {blocks_[blockIndex].input, localRow};
}

bool VectorWindowPartition::rowsEqual(
    const RowReference& lhs,
    const RowReference& rhs,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo)
    const {
  for (const auto& key : keyInfo) {
    const auto inputColumn = inputChannels_[key.first];
    if (!lhs.input->childAt(inputColumn)
             ->equalValueAt(
                 rhs.input->childAt(inputColumn).get(), lhs.row, rhs.row)) {
      return false;
    }
  }
  return true;
}

vector_size_t VectorWindowPartition::findPeerRowEndIndex(
    vector_size_t peerStart,
    vector_size_t lastRow) const {
  auto peerEnd = peerStart + 1;
  const auto startRef = rowAt(peerStart);
  while (peerEnd < lastRow &&
         rowsEqual(startRef, rowAt(peerEnd), sortKeyInfo())) {
    ++peerEnd;
  }
  return peerEnd;
}

vector_size_t VectorWindowPartition::searchFrameValue(
    bool firstMatch,
    vector_size_t start,
    vector_size_t end,
    column_index_t orderByColumn,
    const VectorPtr& frameValue,
    vector_size_t frameValueIndex,
    const CompareFlags& flags) const {
  auto begin = start;
  auto finish = end;

  while (finish - begin >= 2) {
    const auto mid = begin + (finish - begin) / 2;
    const auto [blockIndex, localRow] = findBlock(mid - startRow_);
    const auto& block = blocks_[blockIndex];
    const auto compareResult =
        block.input->childAt(orderByColumn)
            ->compare(frameValue.get(), localRow, frameValueIndex, flags);
    if (!compareResult.has_value() || compareResult.value() >= 0) {
      finish = mid;
    } else {
      begin = mid;
    }
  }

  return linearSearchFrameValue(
      firstMatch,
      begin,
      end,
      orderByColumn,
      frameValue,
      frameValueIndex,
      flags);
}

vector_size_t VectorWindowPartition::linearSearchFrameValue(
    bool firstMatch,
    vector_size_t start,
    vector_size_t end,
    column_index_t orderByColumn,
    const VectorPtr& frameValue,
    vector_size_t frameValueIndex,
    const CompareFlags& flags) const {
  const auto partitionEnd = startRow_ + totalRows_;
  if (start >= end) {
    return end == partitionEnd ? partitionEnd + 1 : -1;
  }

  auto [blockIndex, localRow] = findBlock(start - startRow_);
  auto blockEndRow = blockPrefixSums_[blockIndex + 1] + startRow_;
  for (auto row = start; row < end; ++row) {
    while (row >= blockEndRow) {
      ++blockIndex;
      localRow = blocks_[blockIndex].startRow;
      blockEndRow = blockPrefixSums_[blockIndex + 1] + startRow_;
    }

    const auto& block = blocks_[blockIndex];
    const auto compareResult =
        block.input->childAt(orderByColumn)
            ->compare(frameValue.get(), localRow, frameValueIndex, flags);
    if (compareResult.has_value() && compareResult.value() == 0) {
      if (firstMatch) {
        return row;
      }
    }

    if (compareResult.has_value() && compareResult.value() > 0) {
      return firstMatch ? row : row - 1;
    }

    ++localRow;
  }

  return end == partitionEnd ? partitionEnd + 1 : -1;
}

void VectorWindowPartition::rebuildPrefixSums() {
  blockPrefixSums_.clear();
  blockPrefixSums_.push_back(0);
  for (const auto& block : blocks_) {
    blockPrefixSums_.push_back(blockPrefixSums_.back() + block.size());
  }
  totalRows_ = blockPrefixSums_.back();
}

} // namespace facebook::velox::exec::window
