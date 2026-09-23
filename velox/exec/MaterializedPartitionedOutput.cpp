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

#include "velox/exec/MaterializedPartitionedOutput.h"

#include <algorithm>
#include <cstring>

#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/serializers/RowSerializer.h"

namespace facebook::velox::exec {

MaterializedPartitionedOutput::MaterializedPartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    const std::shared_ptr<MaterializedOutputBufferManager>& manager)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "MaterializedPartitionedOutput"),
      numDestinations_(planNode->numPartitions()),
      broadcast_(planNode->isBroadcast()),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      keyChannels_(toChannels(planNode->inputType(), planNode->keys())),
      partitionFunction_(
          broadcast_ || numDestinations_ == 1
              ? nullptr
              : planNode->partitionFunctionSpec().create(
                    numDestinations_,
                    false)),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      buffer_(manager->buffer(ctx->task->taskId())),
      targetSizeInBytes_(manager->outputBatchSizeBytes(numDestinations_)),
      rowGroupMaxBytes_(buffer_->partitionDrainThreshold()),
      fixedRowSize_(
          row::CompactRow::fixedRowSize(
              std::dynamic_pointer_cast<const RowType>(
                  planNode->outputType()))) {
  VELOX_CHECK_GT(numDestinations_, 0);
  VELOX_CHECK_NOT_NULL(buffer_);
}

void MaterializedPartitionedOutput::initializeInput(RowVectorPtr input) {
  if (outputType_->size() == 0) {
    output_ = std::make_shared<RowVector>(
        input->pool(),
        outputType_,
        nullptr,
        input->size(),
        std::vector<VectorPtr>{});
  } else if (outputChannels_.empty()) {
    output_ = std::move(input);
  } else {
    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputChannels_.size());
    for (const auto channel : outputChannels_) {
      outputColumns.push_back(input->childAt(channel));
    }
    output_ = std::make_shared<RowVector>(
        input->pool(),
        outputType_,
        nullptr,
        input->size(),
        std::move(outputColumns));
  }

  for (column_index_t channel = 0; channel < output_->childrenSize();
       ++channel) {
    output_->childAt(channel)->loadedVector();
  }
}

void MaterializedPartitionedOutput::computePartitions(
    const RowVector& input,
    int32_t numRows) {
  partitions_.resize(numRows);
  if (broadcast_ || numDestinations_ == 1) {
    std::fill(partitions_.begin(), partitions_.end(), 0);
    return;
  }
  const auto singlePartition =
      partitionFunction_->partition(input, partitions_);
  if (singlePartition.has_value()) {
    std::fill(partitions_.begin(), partitions_.end(), singlePartition.value());
  }
}

void MaterializedPartitionedOutput::ensureFlatBufferCapacity(
    int64_t additionalBytes) {
  const auto requiredSize = flatBufferSize_ + additionalBytes;
  const auto currentCapacity = flatBuffer_ ? flatBuffer_->capacity() : 0;
  if (flatBuffer_ != nullptr &&
      requiredSize <= static_cast<int64_t>(currentCapacity)) {
    if (requiredSize > static_cast<int64_t>(flatBuffer_->size())) {
      flatBuffer_->setSize(requiredSize);
    }
    return;
  }
  const auto newSize = std::max<int64_t>(
      {requiredSize, static_cast<int64_t>(currentCapacity) * 2, 1});
  if (flatBuffer_ == nullptr) {
    flatBuffer_ = AlignedBuffer::allocate<char>(newSize, pool());
  } else {
    AlignedBuffer::reallocate<char>(&flatBuffer_, newSize);
  }
}

void MaterializedPartitionedOutput::serializeFixedWidthRows(
    row::CompactRow& compactRow,
    int32_t numRows) {
  const auto startRow = rowCount_;
  const auto fixedRowSize = fixedRowSize_.value();
  const auto batchBytes = static_cast<int64_t>(numRows) * fixedRowSize;

  rowSizes_.resize(startRow + numRows);
  std::fill(rowSizes_.begin() + startRow, rowSizes_.end(), fixedRowSize);
  ensureFlatBufferCapacity(batchBytes);
  rowOffsets_.resize(startRow + numRows);
  rowPartitions_.resize(startRow + numRows);

  std::vector<size_t> bufferOffsets(numRows);
  for (vector_size_t i = 0; i < numRows; ++i) {
    rowOffsets_[startRow + i] = flatBufferSize_;
    rowPartitions_[startRow + i] = partitions_[i];
    bufferOffsets[i] = static_cast<size_t>(flatBufferSize_);
    flatBufferSize_ += fixedRowSize;
  }
  std::memset(
      flatBuffer_->asMutable<char>() + rowOffsets_[startRow], 0, batchBytes);
  compactRow.serialize(
      0, numRows, bufferOffsets.data(), flatBuffer_->asMutable<char>());
  rowCount_ = startRow + numRows;
}

void MaterializedPartitionedOutput::serializeVariableWidthRows(
    row::CompactRow& compactRow,
    int32_t numRows) {
  const auto startRow = rowCount_;
  rowSizes_.resize(startRow + numRows);

  int64_t batchBytes = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    const auto rowSize = compactRow.rowSize(i);
    rowSizes_[startRow + i] = rowSize;
    batchBytes += rowSize;
  }
  ensureFlatBufferCapacity(batchBytes);
  rowOffsets_.resize(startRow + numRows);
  rowPartitions_.resize(startRow + numRows);

  for (vector_size_t i = 0; i < numRows; ++i) {
    const auto rowSize = rowSizes_[startRow + i];
    rowOffsets_[startRow + i] = flatBufferSize_;
    rowPartitions_[startRow + i] = partitions_[i];
    std::memset(flatBuffer_->asMutable<char>() + flatBufferSize_, 0, rowSize);
    compactRow.serialize(i, flatBuffer_->asMutable<char>() + flatBufferSize_);
    flatBufferSize_ += rowSize;
  }
  rowCount_ = startRow + numRows;
}

void MaterializedPartitionedOutput::serializeRows(
    row::CompactRow& compactRow,
    int32_t numRows) {
  if (fixedRowSize_.has_value()) {
    serializeFixedWidthRows(compactRow, numRows);
  } else {
    serializeVariableWidthRows(compactRow, numRows);
  }
}

void MaterializedPartitionedOutput::collectNullRows(
    const RowVector& input,
    int32_t numRows) {
  rows_.resize(numRows);
  rows_.setAll();
  nullRows_.resize(numRows);
  nullRows_.clearAll();
  decodedVectors_.resize(keyChannels_.size());

  for (size_t keyIndex = 0; keyIndex < keyChannels_.size(); ++keyIndex) {
    const auto keyChannel = keyChannels_[keyIndex];
    if (keyChannel == kConstantChannel) {
      continue;
    }
    const auto& keyVector = input.childAt(keyChannel);
    if (!keyVector->mayHaveNulls()) {
      continue;
    }
    auto& decoded = decodedVectors_[keyIndex];
    decoded.decode(*keyVector, rows_);
    if (const auto* rawNulls = decoded.nulls(&rows_)) {
      bits::orWithNegatedBits(
          nullRows_.asMutableRange().bits(), rawNulls, 0, numRows);
    }
  }
  nullRows_.updateBounds();
}

std::vector<int32_t> MaterializedPartitionedOutput::selectRowsToReplicate(
    int32_t numRows) {
  std::vector<int32_t> rowsToReplicate;
  int32_t firstNullRow = 0;
  if (!replicatedAny_) {
    rowsToReplicate.push_back(0);
    replicatedAny_ = true;
    firstNullRow = 1;
  }
  for (int32_t row = firstNullRow; row < numRows; ++row) {
    if (nullRows_.isValid(row)) {
      rowsToReplicate.push_back(row);
    }
  }
  return rowsToReplicate;
}

void MaterializedPartitionedOutput::appendReplicaEntries(
    int32_t serializeStartRow,
    const std::vector<int32_t>& rowsToReplicate) {
  const auto extraEntries =
      static_cast<size_t>(numDestinations_ - 1) * rowsToReplicate.size();
  rowOffsets_.reserve(rowOffsets_.size() + extraEntries);
  rowSizes_.reserve(rowSizes_.size() + extraEntries);
  rowPartitions_.reserve(rowPartitions_.size() + extraEntries);

  for (const auto inputRow : rowsToReplicate) {
    const auto serializedRow = serializeStartRow + inputRow;
    const auto offset = rowOffsets_[serializedRow];
    const auto size = rowSizes_[serializedRow];
    rowPartitions_[serializedRow] = 0;
    for (uint32_t partition = 1;
         partition < static_cast<uint32_t>(numDestinations_);
         ++partition) {
      rowOffsets_.push_back(offset);
      rowSizes_.push_back(size);
      rowPartitions_.push_back(partition);
      ++rowCount_;
    }
  }
}

void MaterializedPartitionedOutput::expandReplicateRows(
    int32_t serializeStartRow,
    int32_t numRows) {
  const auto rowsToReplicate = selectRowsToReplicate(numRows);
  if (!rowsToReplicate.empty()) {
    appendReplicaEntries(serializeStartRow, rowsToReplicate);
  }
}

void MaterializedPartitionedOutput::addInput(RowVectorPtr input) {
  auto rawInput = input;
  initializeInput(std::move(input));
  VELOX_CHECK_NOT_NULL(output_);
  const auto numRows = output_->size();
  if (numRows == 0) {
    output_.reset();
    return;
  }

  computePartitions(*rawInput, numRows);
  if (shouldReplicate()) {
    collectNullRows(*rawInput, numRows);
  }

  const auto serializeStartRow = rowCount_;
  row::CompactRow compactRow(output_);
  serializeRows(compactRow, numRows);
  if (shouldReplicate()) {
    expandReplicateRows(serializeStartRow, numRows);
  }
  output_.reset();

  if (flatBufferSize_ >= targetSizeInBytes_) {
    flushBatch();
  }

  blockingReason_ = buffer_->isBlocked(&future_);
}

void MaterializedPartitionedOutput::flushRowGroup(
    int32_t partition,
    std::vector<int32_t>& rowIndices) {
  using TRowSize = serializer::TRowSize;
  const auto headerSize = serializer::detail::RowGroupHeader::size();

  int64_t rowDataBytes = 0;
  for (const auto row : rowIndices) {
    rowDataBytes += sizeof(TRowSize) + rowSizes_[row];
  }
  const auto totalBytes = headerSize + rowDataBytes;
  auto rowGroup = buffer_->allocateTrackedIOBuf(totalBytes);
  auto* destination = rowGroup->writableData();

  serializer::detail::RowGroupHeader header;
  header.uncompressedSize = static_cast<int32_t>(rowDataBytes);
  header.compressedSize = static_cast<int32_t>(rowDataBytes);
  header.compressed = false;
  header.write(reinterpret_cast<char*>(destination));
  destination += headerSize;

  for (const auto row : rowIndices) {
    const TRowSize rowSize =
        folly::Endian::big(static_cast<TRowSize>(rowSizes_[row]));
    std::memcpy(destination, &rowSize, sizeof(TRowSize));
    destination += sizeof(TRowSize);
    std::memcpy(
        destination,
        flatBuffer_->asMutable<char>() + rowOffsets_[row],
        rowSizes_[row]);
    destination += rowSizes_[row];
  }
  rowGroup->append(totalBytes);

  buffer_->enqueue(partition, std::move(rowGroup));
  rowIndices.clear();
}

void MaterializedPartitionedOutput::flushBatch() {
  if (rowCount_ == 0) {
    return;
  }

  std::vector<std::vector<int32_t>> partitionRows(numDestinations_);
  for (int32_t row = 0; row < rowCount_; ++row) {
    if (broadcast_) {
      for (auto& rows : partitionRows) {
        rows.push_back(row);
      }
    } else {
      partitionRows[rowPartitions_[row]].push_back(row);
    }
  }

  for (int32_t partition = 0; partition < numDestinations_; ++partition) {
    const auto& rows = partitionRows[partition];
    if (rows.empty()) {
      continue;
    }

    std::vector<int32_t> rowGroupRows;
    rowGroupRows.reserve(rows.size());
    int64_t rowGroupBytes = serializer::detail::RowGroupHeader::size();
    for (const auto row : rows) {
      const auto serializedRowBytes =
          static_cast<int64_t>(sizeof(serializer::TRowSize)) + rowSizes_[row];
      if (!rowGroupRows.empty() &&
          rowGroupBytes + serializedRowBytes > rowGroupMaxBytes_) {
        flushRowGroup(partition, rowGroupRows);
        rowGroupBytes = serializer::detail::RowGroupHeader::size();
      }
      rowGroupRows.push_back(row);
      rowGroupBytes += serializedRowBytes;
    }
    if (!rowGroupRows.empty()) {
      flushRowGroup(partition, rowGroupRows);
    }
  }

  rowOffsets_.clear();
  rowSizes_.clear();
  rowPartitions_.clear();
  rowCount_ = 0;
  flatBufferSize_ = 0;
}

RowVectorPtr MaterializedPartitionedOutput::getOutput() {
  return nullptr;
}

void MaterializedPartitionedOutput::noMoreInput() {
  Operator::noMoreInput();
  finish();
}

BlockingReason MaterializedPartitionedOutput::isBlocked(
    ContinueFuture* future) {
  if (blockingReason_ == BlockingReason::kNotBlocked) {
    return BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  const auto reason = blockingReason_;
  blockingReason_ = BlockingReason::kNotBlocked;
  return reason;
}

bool MaterializedPartitionedOutput::isFinished() {
  return finished_;
}

void MaterializedPartitionedOutput::recordBufferStats() {
  for (const auto& [name, value] : buffer_->stats()) {
    addRuntimeStat(name, RuntimeCounter(value));
  }
}

void MaterializedPartitionedOutput::close() {
  if (!finished_) {
    buffer_->abort();
  }
  if (isLastDriver_) {
    recordBufferStats();
  }
  Operator::close();
}

void MaterializedPartitionedOutput::finish() {
  if (finished_) {
    return;
  }
  flushBatch();

  std::vector<ContinuePromise> peerPromises;
  std::vector<std::shared_ptr<Driver>> peers;
  ContinueFuture peerFuture;
  auto* driverContext = operatorCtx()->driverCtx();
  const auto isLast = driverContext->task->allPeersFinished(
      planNodeId(), driverContext->driver, &peerFuture, peerPromises, peers);
  if (isLast) {
    // allPeersFinished returns true only for the last driver. That driver owns
    // the shared buffer's terminal snapshot and publishes it exactly once.
    isLastDriver_ = true;
    buffer_->noMoreData();
    driverContext->task->setAllOutputConsumed();
    for (auto& promise : peerPromises) {
      promise.setValue();
    }
  }
  finished_ = true;
}

} // namespace facebook::velox::exec
