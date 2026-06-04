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
#include "velox/exec/window/KRangeFrameBound.h"
#include "velox/exec/window/PeerGroupComputation.h"
#include "velox/vector/SimpleVector.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace facebook::velox::exec::window {

namespace {

// Points to a row in a retained input vector.
struct RowReference {
  // Input vector that owns the referenced row.
  RowVectorPtr input;

  // Row number in 'input'.
  vector_size_t row;
};

void appendUnique(
    std::vector<column_index_t>& channels,
    column_index_t channel) {
  if (std::find(channels.begin(), channels.end(), channel) == channels.end()) {
    channels.push_back(channel);
  }
}

// Returns the deduplicated input channels referenced by the sort keys, in
// first-seen order.
std::vector<column_index_t> keyChannels(
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(keyInfo.size());
  for (const auto& key : keyInfo) {
    appendUnique(channels, inputChannels[key.first]);
  }
  return channels;
}

} // namespace

VectorWindowPartition::VectorWindowPartition(
    const std::vector<column_index_t>& inputChannels,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo,
    memory::MemoryPool* pool)
    : WindowPartition(inputMapping, sortKeyInfo, true),
      previousRowKeyChannels_(keyChannels(sortKeyInfo, inputChannels)),
      inputChannels_(inputChannels),
      pool_(pool) {
  VELOX_CHECK_NOT_NULL(pool_);
}

void VectorWindowPartition::addRows(const std::vector<char*>& /*rows*/) {
  VELOX_FAIL("VectorWindowPartition does not support RowContainer rows");
}

void VectorWindowPartition::addRows(
    const RowVectorPtr& input,
    vector_size_t startRow,
    vector_size_t endRow) {
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
    previousRow_.clear();
  } else {
    const auto [blockIndex, localRow] = findBlock(numRows - 1);
    previousRow_.capture(
        blocks_[blockIndex].input, localRow, previousRowKeyChannels_, pool_);
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

class VectorWindowPartition::VectorAccessor {
 public:
  explicit VectorAccessor(const VectorWindowPartition& partition)
      : partition_(partition) {}

  vector_size_t startRow() const {
    return partition_.startRow_;
  }

  vector_size_t partitionEnd() const {
    return partition_.startRow_ + partition_.totalRows_;
  }

  bool hasPreviousRow() const {
    return partition_.previousRow_.isValid();
  }

  bool previousRowEquals(vector_size_t row) const {
    const auto rowRef = rowAt(row);
    return partition_.previousRow_.rowsEqual(
        rowRef.input,
        rowRef.row,
        partition_.sortKeyInfo(),
        partition_.inputChannels_);
  }

  bool rowsEqual(vector_size_t lhs, vector_size_t rhs) const {
    return rowsEqual(rowAt(lhs), rowAt(rhs), partition_.sortKeyInfo());
  }

  std::optional<int32_t> compareFrameValue(
      vector_size_t orderByRow,
      vector_size_t frameValueRow,
      column_index_t frameColumn,
      const CompareFlags& flags) const {
    const auto orderByRef = rowAt(orderByRow);
    const auto frameValueRef = rowAt(frameValueRow);
    return orderByRef.input->childAt(orderByColumn())
        ->compare(
            frameValueRef.input->childAt(frameColumn).get(),
            orderByRef.row,
            frameValueRef.row,
            flags);
  }

  bool frameValueIsNull(vector_size_t row, column_index_t frameColumn) const {
    const auto rowRef = rowAt(row);
    return rowRef.input->childAt(frameColumn)->isNullAt(rowRef.row);
  }

  bool orderByValueIsNull(vector_size_t row) const {
    const auto rowRef = rowAt(row);
    return rowRef.input->childAt(orderByColumn())->isNullAt(rowRef.row);
  }

  template <typename T>
  bool frameValueIsNan(vector_size_t row, column_index_t frameColumn) const {
    const auto rowRef = rowAt(row);
    return partition_.isNanAt<T>(
        rowRef.input->childAt(frameColumn), rowRef.row);
  }

  template <typename T>
  bool orderByValueIsNan(vector_size_t row) const {
    const auto rowRef = rowAt(row);
    return partition_.isNanAt<T>(
        rowRef.input->childAt(orderByColumn()), rowRef.row);
  }

 private:
  column_index_t orderByColumn() const {
    return partition_.inputChannels_[partition_.sortKeyInfo()[0].first];
  }

  // Returns a reference to the absolute partition row.
  RowReference rowAt(vector_size_t row) const {
    VELOX_CHECK_GE(row, partition_.startRow_);
    const auto [blockIndex, localRow] =
        partition_.findBlock(row - partition_.startRow_);
    return {partition_.blocks_[blockIndex].input, localRow};
  }

  // Returns true if two retained rows are equal over the specified keys.
  bool rowsEqual(
      const RowReference& lhs,
      const RowReference& rhs,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo)
      const {
    for (const auto& key : keyInfo) {
      const auto inputColumn = partition_.inputChannels_[key.first];
      if (!lhs.input->childAt(inputColumn)
               ->equalValueAt(
                   rhs.input->childAt(inputColumn).get(), lhs.row, rhs.row)) {
        return false;
      }
    }
    return true;
  }

  const VectorWindowPartition& partition_;
};

std::pair<vector_size_t, vector_size_t>
VectorWindowPartition::computePeerBuffers(
    vector_size_t start,
    vector_size_t end,
    vector_size_t prevPeerStart,
    vector_size_t prevPeerEnd,
    vector_size_t* rawPeerStarts,
    vector_size_t* rawPeerEnds) {
  VectorAccessor rows{*this};
  auto result = PeerGroupComputation::compute(
      rows, start, end, prevPeerStart, prevPeerEnd, rawPeerStarts, rawPeerEnds);
  if (result.previousRowConsumed) {
    previousRow_.clear();
  }
  return {result.peerStart, result.peerEnd};
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

  const auto sortOrder = sortKeyInfo()[0].second;
  const auto frameType = blocks_.front().input->childAt(frameColumn)->type();

  VectorAccessor rows{*this};
  KRangeFrameBound::compute(
      rows,
      isStartBound,
      isPreceding,
      frameColumn,
      sortOrder,
      frameType,
      startRow,
      numRows,
      rawPeerBounds,
      rawFrameBounds,
      validFrames);
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

void VectorWindowPartition::rebuildPrefixSums() {
  blockPrefixSums_.clear();
  blockPrefixSums_.push_back(0);
  for (const auto& block : blocks_) {
    blockPrefixSums_.push_back(blockPrefixSums_.back() + block.size());
  }
  totalRows_ = blockPrefixSums_.back();
}

} // namespace facebook::velox::exec::window
