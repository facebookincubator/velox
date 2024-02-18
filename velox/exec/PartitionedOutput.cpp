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

#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec {

namespace detail {
BlockingReason Destination::advance(
    uint64_t maxBytes,
    const std::vector<vector_size_t>& sizes,
    const RowVectorPtr& output,
    OutputBufferManager& bufferManager,
    const std::function<void()>& bufferReleaseFn,
    bool* atEnd,
    ContinueFuture* future,
    Scratch& scratch) {
  if (!type_) {
    type_ = output->type();
  }
  if (rowIdx_ >= rows_.size()) {
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  }

  const auto firstRow = rowIdx_;
  const uint32_t adjustedMaxBytes = (maxBytes * targetSizePct_) / 100;
  if (bytesInCurrent_ >= adjustedMaxBytes) {
    return flush(bufferManager, bufferReleaseFn, future);
  }

  // Collect rows to serialize.
  bool shouldFlush = false;
  while (rowIdx_ < rows_.size() && !shouldFlush) {
    bytesInCurrent_ += sizes[rowIdx_];
    ++rowIdx_;
    ++rowsInCurrent_;
    shouldFlush =
        bytesInCurrent_ >= adjustedMaxBytes || rowsInCurrent_ >= targetNumRows_;
  }

  // Serialize
  if (!current_) {
    current_ = std::make_unique<VectorStreamGroup>(pool_);
    auto rowType = asRowType(output->type());
    serializer::presto::PrestoVectorSerde::PrestoOptions options;
    options.compressionKind = common::CompressionKind::CompressionKind_LZ4;
    current_->createStreamTree(rowType, rowsInCurrent_, &options);
  }
  current_->append(
      output, folly::Range(&rows_[firstRow], rowIdx_ - firstRow), scratch);
  // record(output, firstRow, rowIdx_);
  // Update output state variable.
  if (rowIdx_ == rows_.size()) {
    *atEnd = true;
  }
  if (shouldFlush || (eagerFlush_ && rowsInCurrent_ > 0)) {
    return flush(bufferManager, bufferReleaseFn, future);
  }
  return BlockingReason::kNotBlocked;
}

BlockingReason Destination::flush(
    OutputBufferManager& bufferManager,
    const std::function<void()>& bufferReleaseFn,
    ContinueFuture* future) {
  if (!current_ || rowsInCurrent_ == 0) {
    return BlockingReason::kNotBlocked;
  }

  // Upper limit of message size with no columns.
  constexpr int32_t kMinMessageSize = 128;
  auto listener = bufferManager.newListener();
  IOBufOutputStream stream(
      *current_->pool(),
      listener.get(),
      std::max<int64_t>(kMinMessageSize, current_->size()));
  const int64_t flushedRows = rowsInCurrent_;

  // raw_vector<vector_size_t> empty;
  // history_.push_back(History{nullptr, std::move(empty)});

  current_->flush(&stream);

  const int64_t flushedBytes = stream.tellp();

  bytesInCurrent_ = 0;
  rowsInCurrent_ = 0;
  setTargetSizePct();
  auto iobuf = stream.getIOBuf(bufferReleaseFn);
  // check(iobuf);
  // current_ = nullptr;
  current_->clear();
  bool blocked = bufferManager.enqueue(
      taskId_,
      destination_,
      std::make_unique<SerializedPage>(std::move(iobuf), nullptr, flushedRows),
      future);

  recordEnqueued_(flushedBytes, flushedRows);

  return blocked ? BlockingReason::kWaitForConsumer
                 : BlockingReason::kNotBlocked;
}

void Destination::updateStats(Operator* op) {
  if (current_ && current_->serializer()) {
    auto serializerStats = current_->serializer()->runtimeStats();
    auto lockedStats = op->stats().wlock();
    for (auto& pair : serializerStats) {
      lockedStats->addRuntimeStat(pair.first, pair.second);
    }
  }
}

void Destination::record(
    const RowVectorPtr& input,
    int32_t begin,
    int32_t end) {
  raw_vector<vector_size_t> temp;
  for (auto i = begin; i < end; ++i) {
    temp.push_back(rows_[i]);
  }

  auto children = input->children();
  RowVectorPtr copy = std::make_shared<RowVector>(
      input->pool(), input->type(), nullptr /*nulls*/, input->size(), children);
  history_.push_back(History{std::move(copy), std::move(temp)});
}

void Destination::check(std::unique_ptr<folly::IOBuf>& iobuf) {
  std::vector<ByteRange> ranges;
  for (auto& range : *iobuf) {
    ranges.push_back(ByteRange{
        reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(range.data())),
        static_cast<int32_t>(range.size()),
        0});
  }
  ByteInputStream in(std::move(ranges));
  RowVectorPtr result;
  getVectorSerde()->deserialize(
      &in, pool_, std::static_pointer_cast<const RowType>(type_), &result, 0);
  VELOX_CHECK_LT(0, result->size());
}

void Destination::replay() {
  Scratch scratch;
  auto group = std::make_unique<VectorStreamGroup>(pool_);
  auto rowType = asRowType(type_);
  serializer::presto::PrestoVectorSerde::PrestoOptions options;
  options.compressionKind = common::CompressionKind::CompressionKind_LZ4;
  group->createStreamTree(rowType, 100, &options);
  for (auto i = 0; i < history_.size(); ++i) {
    auto& record = history_[i];
    if (record.rows == nullptr) {
      IOBufOutputStream stream(
          *group->pool(), nullptr, std::max<int64_t>(128, group->size()));

      group->flush(&stream);
      auto iobuf = stream.getIOBuf(nullptr);
      check(iobuf);
    } else {
      group->append(
          record.rows,
          folly::Range(record.indices.data(), record.indices.size()),
          scratch);
    }
  }
}

bool Destination::chooseAdvance(
    const RowVectorPtr& output,
    uint64_t maxBytes,
    const std::vector<vector_size_t>& sizes,
    raw_vector<detail::Destination*>& needMoreAdvance,
    raw_vector<detail::Destination*>& toFlush) {
  if (rowIdx_ >= rows_.size()) {
    return false;
  }

  if (!type_) {
    type_ = output->type();
  }

  firstRow_ = rowIdx_;
  const uint32_t adjustedMaxBytes = (maxBytes * targetSizePct_) / 100;
  if (bytesInCurrent_ >= adjustedMaxBytes) {
    toFlush.push_back(this);
    return true;
  }

  // Collect rows to serialize.
  bool shouldFlush = false;
  while (rowIdx_ < rows_.size() && !shouldFlush) {
    bytesInCurrent_ += sizes[rowIdx_];
    ++rowIdx_;
    ++rowsInCurrent_;
    shouldFlush =
        bytesInCurrent_ >= adjustedMaxBytes || rowsInCurrent_ >= targetNumRows_;
  }

  if (!current_) {
    current_ = std::make_unique<VectorStreamGroup>(pool_);
    auto rowType = asRowType(output->type());
    serializer::presto::PrestoVectorSerde::PrestoOptions options;
    options.compressionKind = common::CompressionKind::CompressionKind_LZ4;

    current_->createStreamTree(rowType, rowsInCurrent_, &options);
  }

  if (rowIdx_ < rows_.size()) {
    needMoreAdvance.push_back(this);
  }

  if (shouldFlush || (eagerFlush_ && rowsInCurrent_ > 0)) {
    toFlush.push_back(this);
  }
  return true;
}

} // namespace detail

PartitionedOutput::PartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "PartitionedOutput"),
      keyChannels_(toChannels(planNode->inputType(), planNode->keys())),
      numDestinations_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      partitionFunction_(
          numDestinations_ == 1
              ? nullptr
              : planNode->partitionFunctionSpec().create(numDestinations_)),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      bufferManager_(OutputBufferManager::getInstance()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      maxBufferedBytes_(ctx->task->queryCtx()
                            ->queryConfig()
                            .maxPartitionedOutputBufferSize()),
      eagerFlush_(eagerFlush) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
    VELOX_USER_CHECK_NULL(partitionFunction_);
  }
}

void PartitionedOutput::initializeInput(RowVectorPtr input) {
  input_ = std::move(input);
  if (outputType_->size() == 0) {
    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr /*nulls*/,
        input_->size(),
        std::vector<VectorPtr>{});
  } else if (outputChannels_.empty()) {
    output_ = input_;
  } else {
    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputChannels_.size());
    for (auto i : outputChannels_) {
      outputColumns.push_back(input_->childAt(i));
    }

    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr /*nulls*/,
        input_->size(),
        outputColumns);
  }
  if (output_->size() > 1 /* && !encodingCandidates_.empty()*/) {
    rows_.resize(output_->size());
    rows_.setAll();
    for (auto i = 0; i < output_->childrenSize(); ++i) {
      maybeEncode(i);
    }
  }
}

template <TypeKind Kind>
bool isAllSameFlat(const BaseVector& vector, vector_size_t size) {
  using T = typename KindToFlatVector<Kind>::WrapperType;
  auto flat = vector.asUnchecked<FlatVector<T>>();
  auto rawValues = flat->rawValues();
  T first = rawValues[0];
  for (auto i = 1; i < size; ++i) {
    if (first != rawValues[i]) {
      return false;
    }
  }
  return true;
}

void PartitionedOutput::maybeEncode(column_index_t i) {
  auto& column = BaseVector::loadedVectorShared(output_->childAt(i));
  if (column->typeKind() == TypeKind::BOOLEAN) {
    return;
  }
  if (column->encoding() == VectorEncoding::Simple::CONSTANT) {
    return;
  }
  // If there is a null, values will either not all be the same or all be null,
  // which can just as well be serialized as flat.
  if (column->isNullAt(0)) {
    return;
  }
  // Quick return if first and last are different.
  if (!column->equalValueAt(column.get(), 0, rows_.end() - 1)) {
    return;
  }

  tempDecoded_.decode(*column, rows_);
  if (!tempDecoded_.isIdentityMapping()) {
    auto indices = tempDecoded_.indices();
    auto first = indices[0];
    for (auto i = 1; i < rows_.end(); ++i) {
      if (indices[i] != first) {
        if (tempDecoded_.isNullAt(i)) {
          return;
        }
        if (!tempDecoded_.base()->equalValueAt(
                tempDecoded_.base(), first, indices[i])) {
          return;
        }
      }
    }
    replaceOutputColumn(i, BaseVector::wrapInConstant(rows_.end(), 0, column));
    return;
  }
  if (column->mayHaveNulls()) {
    return;
  }
  if (column->encoding() == VectorEncoding::Simple::FLAT) {
    if (!VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
            isAllSameFlat, column->typeKind(), *column, rows_.end() - 1)) {
      return;
    }
  } else {
    for (auto i = 1; i < rows_.end() - 1; ++i) {
      if (!column->equalValueAt(column.get(), 0, i)) {
        return;
      }
    }
  }

  replaceOutputColumn(i, BaseVector::wrapInConstant(rows_.end(), 0, column));
}

void PartitionedOutput::replaceOutputColumn(int32_t i, VectorPtr column) {
  if (input_ == output_) {
    auto children = input_->children();
    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr /*nulls*/,
        input_->size(),
        children);
  }
  output_->childAt(i) = column;
}

void PartitionedOutput::initializeDestinations() {
  if (destinations_.empty()) {
    auto taskId = operatorCtx_->taskId();
    for (int i = 0; i < numDestinations_; ++i) {
      destinations_.push_back(std::make_unique<detail::Destination>(
          taskId, i, pool(), eagerFlush_, [&](uint64_t bytes, uint64_t rows) {
            auto lockedStats = stats_.wlock();
            lockedStats->addOutputVector(bytes, rows);
          }));
    }
  }
}

void PartitionedOutput::initializeSizeBuffers() {
  auto numInput = input_->size();
  if (numInput > rowSize_.size()) {
    rowSize_.resize(numInput);
    sizePointers_.resize(numInput);
    // Set all the size pointers since 'rowSize_' may have been reallocated.
    for (vector_size_t i = 0; i < numInput; ++i) {
      sizePointers_[i] = &rowSize_[i];
    }
  }
}

void PartitionedOutput::estimateRowSizes() {
  auto numInput = input_->size();
  std::fill(rowSize_.begin(), rowSize_.end(), 0);
  raw_vector<vector_size_t> storage;
  auto numbers = iota(numInput, storage);
  for (int i = 0; i < output_->childrenSize(); ++i) {
    VectorStreamGroup::estimateSerializedSize(
        output_->childAt(i),
        folly::Range(numbers, numInput),
        sizePointers_.data(),
        scratch_);
  }
}

void PartitionedOutput::addInput(RowVectorPtr input) {
  initializeInput(std::move(input));

  initializeDestinations();

  initializeSizeBuffers();

  estimateRowSizes();

  for (auto& destination : destinations_) {
    destination->beginBatch();
  }

  auto numInput = input_->size();
  if (numDestinations_ == 1) {
    destinations_[0]->addRows(IndexRange{0, numInput});
  } else {
    auto singlePartition = partitionFunction_->partition(*input_, partitions_);
    if (replicateNullsAndAny_) {
      collectNullRows();

      vector_size_t start = 0;
      if (!replicatedAny_) {
        for (auto& destination : destinations_) {
          destination->addRow(0);
        }
        replicatedAny_ = true;
        // Make sure not to replicate first row twice.
        start = 1;
      }
      for (auto i = start; i < numInput; ++i) {
        if (nullRows_.isValid(i)) {
          for (auto& destination : destinations_) {
            destination->addRow(i);
          }
        } else {
          if (singlePartition.has_value()) {
            destinations_[singlePartition.value()]->addRow(i);
          } else {
            destinations_[partitions_[i]]->addRow(i);
          }
        }
      }
    } else {
      if (singlePartition.has_value()) {
        destinations_[singlePartition.value()]->addRows(
            IndexRange{0, numInput});
      } else {
        for (vector_size_t i = 0; i < numInput; ++i) {
          destinations_[partitions_[i]]->addRow(i);
        }
      }
    }
  }
}

void PartitionedOutput::collectNullRows() {
  auto size = input_->size();
  rows_.resize(size);
  rows_.setAll();

  nullRows_.resize(size);
  nullRows_.clearAll();

  decodedVectors_.resize(keyChannels_.size());

  for (auto i : keyChannels_) {
    if (i == kConstantChannel) {
      continue;
    }
    auto& keyVector = input_->childAt(i);
    if (keyVector->mayHaveNulls()) {
      decodedVectors_[i].decode(*keyVector, rows_);
      if (auto* rawNulls = decodedVectors_[i].nulls(&rows_)) {
        bits::orWithNegatedBits(
            nullRows_.asMutableRange().bits(), rawNulls, 0, size);
      }
    }
  }
  nullRows_.updateBounds();
}

RowVectorPtr PartitionedOutput::getOutput() {
  getOutputColumnwise();
  return nullptr;
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;
  detail::Destination* blockedDestination = nullptr;
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  // Limit serialized pages to 1MB.
  static const uint64_t kMaxPageSize = 1 << 20;
  const uint64_t maxPageSize = std::max<uint64_t>(
      kMinDestinationSize,
      std::min<uint64_t>(kMaxPageSize, maxBufferedBytes_ / numDestinations_));

  bool workLeft;
  do {
    workLeft = false;
    for (auto& destination : destinations_) {
      bool atEnd = false;
      blockingReason_ = destination->advance(
          maxPageSize,
          rowSize_,
          output_,
          *bufferManager,
          bufferReleaseFn_,
          &atEnd,
          &future_,
          scratch_);
      if (blockingReason_ != BlockingReason::kNotBlocked) {
        blockedDestination = destination.get();
        workLeft = false;
        // We stop on first blocked. Adding data to unflushed targets
        // would be possible but could allocate memory. We wait for
        // free space in the outgoing queue.
        break;
      }
      if (!atEnd) {
        workLeft = true;
      }
    }
  } while (workLeft);

  if (blockedDestination) {
    // If we are going off-thread, we may as well make the output in
    // progress for other destinations available, unless it is too
    // small to be worth transfer.
    for (auto& destination : destinations_) {
      if (destination.get() == blockedDestination ||
          destination->serializedBytes() < kMinDestinationSize) {
        continue;
      }
      destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
    }
    return nullptr;
  }
  // All of 'output_' is written into the destinations. We are finishing, hence
  // move all the destinations to the output queue. This will not grow memory
  // and hence does not need blocking.
  if (noMoreInput_) {
    for (auto& destination : destinations_) {
      if (destination->isFinished()) {
        continue;
      }
      destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
      destination->setFinished();
      destination->updateStats(this);
    }

    bufferManager->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }
  // The input is fully processed, drop the reference to allow reuse.
  input_ = nullptr;
  output_ = nullptr;
  return nullptr;
}

bool PartitionedOutput::isFinished() {
  return finished_;
}

void PartitionedOutput::getOutputColumnwise() {
  if (finished_) {
    return;
  }

  blockingReason_ = BlockingReason::kNotBlocked;
  detail::Destination* blockedDestination = nullptr;
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  // Limit serialized pages to 1MB.
  static const uint64_t kMaxPageSize = 1 << 20;
  const uint64_t maxPageSize = std::max<uint64_t>(
      kMinDestinationSize,
      std::min<uint64_t>(kMaxPageSize, maxBufferedBytes_ / numDestinations_));

  if (!destinations_.empty()) {
    for (;;) {
      toFlush_.clear();
      toAdvance_.clear();
      nextToAdvance_.clear();
      for (auto& destination : destinations_) {
        if (destination->chooseAdvance(
                output_, maxPageSize, rowSize_, nextToAdvance_, toFlush_)) {
          toAdvance_.push_back(destination.get());
        }
      }
      for (auto column = 0;
           !toAdvance_.empty() && column < output_->childrenSize();
           ++column) {
        for (auto i = 0; i < toAdvance_.size(); ++i) {
          auto destination = toAdvance_[i];
          destination->streamGroup()->serializer()->appendColumn(
              output_,
              column,
              folly::Range(
                  destination->rows().data() + destination->firstRow(),
                  destination->rowIdx() - destination->firstRow()),
              scratch_);
        }
      }
      for (auto i = 0; i < toAdvance_.size(); ++i) {
        auto destination = toAdvance_[i];
        destination->streamGroup()->serializer()->incrementRows(
            destination->rowIdx() - destination->firstRow());
      }
      for (auto* destination : toFlush_) {
        auto reason = destination->flush(
            *bufferManager,
            bufferReleaseFn_,
            blockedDestination ? nullptr : &future_);
        if (reason != BlockingReason::kNotBlocked &&
            blockedDestination == nullptr) {
          blockingReason_ = reason;
          blockedDestination = destination;
        }
      }
      if (blockedDestination != nullptr || nextToAdvance_.empty()) {
        break;
      }
    }

    if (blockedDestination) {
      // If we are going off-thread, we may as well make the output in
      // progress for other destinations available, unless it is too
      // small to be worth transfer.
      for (auto& destination : destinations_) {
        if (destination->serializedBytes() < kMinDestinationSize) {
          continue;
        }
        destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
      }
      return;
    }
  }
  // All of 'output_' is written into the destinations. We are finishing, hence
  // move all the destinations to the output queue. This will not grow memory
  // and hence does not need blocking.
  if (noMoreInput_) {
    for (auto& destination : destinations_) {
      if (destination->isFinished()) {
        continue;
      }
      destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
      destination->setFinished();
      destination->updateStats(this);
    }

    bufferManager->noMoreData(operatorCtx_->task()->taskId());
    destinations_.clear();
    finished_ = true;
  }
  // The input is fully processed, drop the reference to allow reuse.
  input_ = nullptr;
  output_ = nullptr;
  return;
}

} // namespace facebook::velox::exec
