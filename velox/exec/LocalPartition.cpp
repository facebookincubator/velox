/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/LocalPartition.h"
#include "velox/common/Casts.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"
#include "velox/vector/EncodedVectorCopy.h"

namespace facebook::velox::exec {
namespace {
void notify(std::vector<ContinuePromise>& promises) {
  for (auto& promise : promises) {
    promise.setValue();
  }
}
} // namespace

bool LocalExchangeMemoryManager::increaseMemoryUsage(
    ContinueFuture* future,
    int64_t added) {
  std::lock_guard<std::mutex> l(mutex_);
  bufferedBytes_ += added;

  if (bufferedBytes_ >= maxBufferSize_) {
    promises_.emplace_back("LocalExchangeMemoryManager::updateMemoryUsage");
    *future = promises_.back().getSemiFuture();
    return true;
  }

  return false;
}

std::vector<ContinuePromise> LocalExchangeMemoryManager::decreaseMemoryUsage(
    int64_t removed) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    bufferedBytes_ -= removed;

    if (bufferedBytes_ < maxBufferSize_) {
      promises = std::move(promises_);
    }
  }
  return promises;
}

void LocalExchangeVectorPool::push(const RowVectorPtr& vector, int64_t size) {
  pool_.withWLock([&](auto& pool) {
    if (totalSize_ + size <= capacity_) {
      pool.emplace(vector, size);
      totalSize_ += size;
    }
  });
}

RowVectorPtr LocalExchangeVectorPool::pop() {
  return pool_.withWLock([&](auto& pool) -> RowVectorPtr {
    while (!pool.empty()) {
      auto [vector, size] = std::move(pool.front());
      pool.pop();
      totalSize_ -= size;
      VELOX_CHECK_GE(totalSize_, 0);
      if (vector.use_count() == 1) {
        return vector;
      }
    }
    VELOX_CHECK_EQ(totalSize_, 0);
    return nullptr;
  });
}

void LocalExchangeQueue::addProducer() {
  queue_.withWLock([&](auto& /*queue*/) {
    VELOX_CHECK(!noMoreProducers_, "addProducer called after noMoreProducers");
    ++pendingProducers_;
  });
}

void LocalExchangeQueue::noMoreProducers() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK(!noMoreProducers_, "noMoreProducers can be called only once");
    noMoreProducers_ = true;
    if (pendingProducers_ == 0) {
      // No more data will be produced.
      consumerPromises = std::move(consumerPromises_);
    }
  });
  notify(consumerPromises);
}

void LocalExchangeQueue::drain() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK(!closed_, "Queue is closed");
    ++drainedProducers_;
    VELOX_CHECK_LE(drainedProducers_, pendingProducers_);
    if (drainedProducers_ != pendingProducers_) {
      return;
    }
    consumerPromises = std::move(consumerPromises_);
  });
  notify(consumerPromises);
}

BlockingReason LocalExchangeQueue::enqueue(
    RowVectorPtr input,
    int64_t inputBytes,
    ContinueFuture* future) {
  std::vector<ContinuePromise> consumerPromises;
  bool blockedOnConsumer = false;
  const bool isClosed = queue_.withWLock([&](auto& queue) {
    if (closed_) {
      return true;
    }
    queue.emplace(std::move(input), inputBytes);
    consumerPromises = std::move(consumerPromises_);

    if (memoryManager_->increaseMemoryUsage(future, inputBytes)) {
      blockedOnConsumer = true;
    }

    return false;
  });

  if (isClosed) {
    return BlockingReason::kNotBlocked;
  }

  notify(consumerPromises);

  if (blockedOnConsumer) {
    return BlockingReason::kWaitForConsumer;
  }

  return BlockingReason::kNotBlocked;
}

void LocalExchangeQueue::noMoreData() {
  std::vector<ContinuePromise> consumerPromises;
  queue_.withWLock([&](auto& queue) {
    VELOX_CHECK_EQ(drainedProducers_, 0);
    VELOX_CHECK_GT(pendingProducers_, 0);
    --pendingProducers_;
    if (noMoreProducers_ && pendingProducers_ == 0) {
      consumerPromises = std::move(consumerPromises_);
    }
  });
  notify(consumerPromises);
}

BlockingReason LocalExchangeQueue::next(
    ContinueFuture* future,
    memory::MemoryPool* pool,
    RowVectorPtr* data,
    bool& drained) {
  drained = false;
  int64_t size{0};
  std::vector<ContinuePromise> memoryPromises;
  const auto blockingReason = queue_.withWLock([&](auto& queue) {
    *data = nullptr;
    if (queue.empty()) {
      if (isFinishedLocked(queue)) {
        return BlockingReason::kNotBlocked;
      }
      if (testAndClearDrainedLocked()) {
        drained = true;
        return BlockingReason::kNotBlocked;
      }

      consumerPromises_.emplace_back("LocalExchangeQueue::next");
      *future = consumerPromises_.back().getSemiFuture();

      return BlockingReason::kWaitForProducer;
    }

    std::tie(*data, size) = std::move(queue.front());
    queue.pop();

    memoryPromises = memoryManager_->decreaseMemoryUsage(size);
    return BlockingReason::kNotBlocked;
  });

  notify(memoryPromises);
  if (*data != nullptr) {
    vectorPool_->push(*data, size);
  }
  return blockingReason;
}

bool LocalExchangeQueue::isFinishedLocked(const Queue& queue) const {
  if (closed_) {
    return true;
  }

  if (noMoreProducers_ && pendingProducers_ == 0 && queue.empty()) {
    return true;
  }

  return false;
}

bool LocalExchangeQueue::testAndClearDrainedLocked() {
  VELOX_CHECK(!closed_);
  VELOX_CHECK_GT(pendingProducers_, 0);
  if (pendingProducers_ != drainedProducers_) {
    return false;
  }
  drainedProducers_ = 0;
  return true;
}

bool LocalExchangeQueue::isFinished() {
  return queue_.withWLock([&](auto& queue) { return isFinishedLocked(queue); });
}

bool LocalExchangeQueue::testingProducersDone() const {
  return queue_.withRLock(
      [&](auto& queue) { return noMoreProducers_ && pendingProducers_ == 0; });
}

void LocalExchangeQueue::close() {
  std::vector<ContinuePromise> consumerPromises;
  std::vector<ContinuePromise> memoryPromises;
  queue_.withWLock([&](auto& queue) {
    uint64_t freedBytes = 0;
    while (!queue.empty()) {
      freedBytes += queue.front().second;
      queue.pop();
    }

    if (freedBytes) {
      memoryPromises = memoryManager_->decreaseMemoryUsage(freedBytes);
    }

    consumerPromises = std::move(consumerPromises_);
    closed_ = true;
  });
  notify(consumerPromises);
  notify(memoryPromises);
}

LocalExchange::LocalExchange(
    int32_t operatorId,
    DriverCtx* ctx,
    RowTypePtr outputType,
    const std::string& planNodeId,
    int partition)
    : SourceOperator(
          ctx,
          std::move(outputType),
          operatorId,
          planNodeId,
          OperatorType::kLocalExchange),
      partition_{partition},
      queue_{operatorCtx_->task()->getLocalExchangeQueue(
          ctx->splitGroupId,
          planNodeId,
          partition)} {}

BlockingReason LocalExchange::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    auto reason = blockingReason_;
    blockingReason_ = BlockingReason::kNotBlocked;
    return reason;
  }

  return BlockingReason::kNotBlocked;
}

RowVectorPtr LocalExchange::getOutput() {
  if (hasDrained()) {
    return nullptr;
  }

  RowVectorPtr data;
  bool drained{false};
  blockingReason_ = queue_->next(&future_, pool(), &data, drained);
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    VELOX_CHECK(future_.valid());
    VELOX_CHECK(!drained);
    return nullptr;
  }

  if (data != nullptr) {
    VELOX_CHECK(!drained);
    auto lockedStats = stats_.wlock();
    lockedStats->addInputVector(data->estimateFlatSize(), data->size());
    return data;
  }

  if (drained) {
    VELOX_CHECK(!isDraining());
    operatorCtx_->driver()->drainOutput();
  } else {
    VELOX_CHECK(queue_->isFinished());
  }
  return nullptr;
}

bool LocalExchange::isFinished() {
  return queue_->isFinished();
}

void LocalExchange::close() {
  Operator::close();
  if (queue_) {
    queue_->close();
  }
}

LocalPartition::LocalPartition(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::LocalPartitionNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          OperatorType::kLocalPartition),
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()},
      partitionFunction_(
          numPartitions_ == 1 ? nullptr
                              : planNode->partitionFunctionSpec().create(
                                    numPartitions_,
                                    /*localExchange=*/true)),
      singlePartitionBufferSize_{
          (numPartitions_ <
               ctx->queryConfig()
                   .minLocalExchangePartitionCountToUsePartitionBuffer() ||
           eagerFlush)
              ? 0
              : ctx->queryConfig().maxLocalExchangePartitionBufferSize()},
      partitionBufferPreserveEncoding_{
          ctx->queryConfig().localExchangePartitionBufferPreserveEncoding()} {
  VELOX_CHECK(numPartitions_ == 1 || partitionFunction_ != nullptr);
  for (auto& queue : queues_) {
    queue->addProducer();
  }
  if (numPartitions_ > 0) {
    indexBuffers_.resize(numPartitions_);
    rawIndices_.resize(numPartitions_);
  }
}

void LocalPartition::allocateIndexBuffers(
    const std::vector<vector_size_t>& sizes) {
  VELOX_CHECK_EQ(indexBuffers_.size(), sizes.size());
  VELOX_CHECK_EQ(rawIndices_.size(), sizes.size());

  for (auto i = 0; i < sizes.size(); ++i) {
    const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
    if ((indexBuffers_[i] == nullptr) ||
        (indexBuffers_[i]->capacity() < indicesBufferBytes) ||
        !indexBuffers_[i]->unique()) {
      indexBuffers_[i] = allocateIndices(sizes[i], pool());
    } else {
      const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
      indexBuffers_[i]->setSize(indicesBufferBytes);
    }
    rawIndices_[i] = indexBuffers_[i]->asMutable<vector_size_t>();
  }
}

RowVectorPtr LocalPartition::wrapChildren(
    const RowVectorPtr& input,
    vector_size_t size,
    const BufferPtr& indices,
    RowVectorPtr reusable) {
  RowVectorPtr result;
  if (!reusable) {
    result = std::make_shared<RowVector>(
        pool(),
        input->type(),
        nullptr,
        size,
        std::vector<VectorPtr>(input->childrenSize()));
  } else {
    VELOX_CHECK(!reusable->mayHaveNulls());
    VELOX_CHECK_EQ(reusable.use_count(), 1);
    reusable->unsafeResize(size);
    result = std::move(reusable);
  }
  VELOX_CHECK_NOT_NULL(result);

  for (auto i = 0; i < input->childrenSize(); ++i) {
    auto& child = result->childAt(i);
    if (child && child->encoding() == VectorEncoding::Simple::DICTIONARY &&
        child.use_count() == 1) {
      child->BaseVector::resize(size);
      child->setWrapInfo(indices);
      child->setValueVector(input->childAt(i));
    } else {
      child = BaseVector::wrapInDictionary(
          nullptr, indices, size, input->childAt(i));
    }
  }

  result->updateContainsLazyNotLoaded();
  return result;
}

void LocalPartition::copy(
    const RowVectorPtr& input,
    const folly::Range<const BaseVector::CopyRange*>& ranges,
    const size_t partition,
    VectorPtr& target) {
  if (ranges.empty()) {
    return;
  }

  if (partitionBufferPreserveEncoding_) {
    encodedVectorCopy(
        EncodedVectorCopyOptions{pool(), false, 0.5}, input, ranges, target);
    return;
  }

  if (!target) {
    target = getOrCreateVector(partition);
  }
  target->resize(target->size() + ranges.size());
  target->copyRanges(input.get(), ranges);
}

VectorPtr LocalPartition::getOrCreateVector(const size_t partition) {
  auto reusable = queues_[partition]->getVector();
  if (reusable) {
    VELOX_CHECK_EQ(reusable->type(), outputType_);
    reusable->unsafeResize(0);
    for (auto i = 0; i < reusable->childrenSize(); ++i) {
      reusable->childAt(i) = nullptr;
    }
    return reusable;
  } else {
    return BaseVector::create<RowVector>(outputType_, 0, pool());
  }
}

void LocalPartition::populatePartitionBuffer(
    const RowVectorPtr& input,
    const vector_size_t numPartitionRows,
    const size_t partition,
    const vector_size_t* rawIndices,
    uint64_t& totalPartitionBufferSizeExcludingString,
    uint64_t& totalPartitionStringBufferSize) {
  VELOX_CHECK_GT(singlePartitionBufferSize_, 0);
  copyRanges_.resize(numPartitionRows);

  auto& partitionBuffer = partitionBuffers_[partition];
  auto targetIndex = 0;
  if (partitionBuffer) {
    targetIndex = partitionBuffer->size();
  }
  for (int i = 0; i < numPartitionRows; i++) {
    copyRanges_[i] = {rawIndices[i], targetIndex, 1};
    targetIndex++;
  }

  copy(input, copyRanges_, partition, partitionBuffer);

  if (partitionBuffer) {
    uint64_t stringBufferSize{0};
    auto totalSize = partitionBuffer->retainedSize(stringBufferSize);
    totalPartitionBufferSizeExcludingString += totalSize - stringBufferSize;
    totalPartitionStringBufferSize += stringBufferSize;
  }
}

RowVectorPtr LocalPartition::createPartition(
    const RowVectorPtr& input,
    const vector_size_t numPartitionRows,
    const size_t partition,
    const BufferPtr& indices) {
  RowVectorPtr partitionData{nullptr};
  if (singlePartitionBufferSize_ > 0) {
    auto& partitionBuffer = partitionBuffers_[partition];
    if (partitionBuffer) {
      partitionData =
          checkedPointerCast<RowVector, BaseVector>(partitionBuffer);
      partitionBuffers_[partition] = nullptr;
    }
  } else if (numPartitionRows > 0) {
    partitionData = wrapChildren(
        input, numPartitionRows, indices, queues_[partition]->getVector());
  }
  return partitionData;
}

void LocalPartition::populateAndEnqueuePartitions(
    RowVectorPtr input,
    const std::vector<vector_size_t>& numRowsPerPartition,
    const std::vector<BufferPtr>& indexBuffers,
    const std::vector<vector_size_t*>& rawIndicesBuffers) {
  uint64_t totalPartitionBufferSizeExcludingString = 0;
  uint64_t totalPartitionStringBufferSize = 0;
  uint16_t nonEmptyPartitionCount = 0;

  // Populate partition buffers if in buffer mode.
  if (singlePartitionBufferSize_ > 0) {
    if (partitionBuffers_.empty()) {
      partitionBuffers_.resize(numPartitions_);
    }
    for (auto partition = 0; partition < numPartitions_; partition++) {
      populatePartitionBuffer(
          input,
          numRowsPerPartition[partition],
          partition,
          rawIndicesBuffers[partition],
          totalPartitionBufferSizeExcludingString,
          totalPartitionStringBufferSize);
      if (partitionBuffers_[partition]) {
        nonEmptyPartitionCount++;
      }
    }
  } else {
    nonEmptyPartitionCount = numPartitions_ -
        std::count(numRowsPerPartition.begin(), numRowsPerPartition.end(), 0);
  }
  VELOX_CHECK_GT(
      nonEmptyPartitionCount,
      0,
      "Input rows should be assigned to at least one partition");

  // Calculate the partition buffer size across all partitions with amortized
  // string buffer sizes.
  auto balancedTotalPartitionBufferSize =
      totalPartitionBufferSizeExcludingString +
      (totalPartitionStringBufferSize / nonEmptyPartitionCount);
  auto inputRetainedSize = input->retainedSize();

  // Enqueue all partitions if one of the following conditions is met:
  // 1. This operator is not in buffer mode.
  // 2. This operator is in buffer mode and the total buffer size across all
  // partitions exceeds 'singlePartitionBufferSize_ * numPartitions_'.
  if (singlePartitionBufferSize_ == 0 ||
      balancedTotalPartitionBufferSize >=
          singlePartitionBufferSize_ * numPartitions_) {
    auto perPartitionAmortizedSize =
        (singlePartitionBufferSize_ > 0 ? balancedTotalPartitionBufferSize
                                        : inputRetainedSize) /
        nonEmptyPartitionCount;
    for (auto partition = 0; partition < numPartitions_; partition++) {
      auto partitionSize = numRowsPerPartition[partition];
      auto partitionData = createPartition(
          input, partitionSize, partition, indexBuffers[partition]);
      if (!partitionData) {
        continue;
      }

      ContinueFuture future;
      auto reason = queues_[partition]->enqueue(
          std::move(partitionData), perPartitionAmortizedSize, &future);
      if (reason != BlockingReason::kNotBlocked) {
        blockingReasons_.push_back(reason);
        futures_.push_back(std::move(future));
      }
    }
  }
}

void LocalPartition::addInput(RowVectorPtr input) {
  prepareForInput(input);
  if (input->size() == 0) {
    return;
  }

  const auto singlePartition = numPartitions_ == 1
      ? 0
      : partitionFunction_->partition(*input, partitions_);
  if (singlePartition.has_value()) {
    ContinueFuture future;
    auto blockingReason = queues_[singlePartition.value()]->enqueue(
        input, input->retainedSize(), &future);
    if (blockingReason != BlockingReason::kNotBlocked) {
      blockingReasons_.push_back(blockingReason);
      futures_.push_back(std::move(future));
    }
    return;
  }

  const auto numInput = input->size();
  std::vector<vector_size_t> maxIndex(numPartitions_, 0);
  for (auto i = 0; i < numInput; ++i) {
    ++maxIndex[partitions_[i]];
  }
  allocateIndexBuffers(maxIndex);

  std::fill(maxIndex.begin(), maxIndex.end(), 0);
  for (auto i = 0; i < numInput; ++i) {
    auto partition = partitions_[i];
    rawIndices_[partition][maxIndex[partition]] = i;
    ++maxIndex[partition];
  }

  populateAndEnqueuePartitions(input, maxIndex, indexBuffers_, rawIndices_);
}

void LocalPartition::prepareForInput(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  // Lazy vectors must be loaded or processed to ensure the late materialized in
  // order.
  for (auto& child : input->children()) {
    child->loadedVector();
  }
}

BlockingReason LocalPartition::isBlocked(ContinueFuture* future) {
  if (!futures_.empty()) {
    auto blockingReason = blockingReasons_.front();
    *future = folly::collectAll(futures_.begin(), futures_.end()).unit();
    futures_.clear();
    blockingReasons_.clear();
    return blockingReason;
  }
  return BlockingReason::kNotBlocked;
}

void LocalPartition::noMoreInput() {
  Operator::noMoreInput();
  if (!partitionBuffers_.empty()) {
    uint64_t totalPartitionBufferSizeExcludingString = 0;
    uint64_t totalPartitionStringBufferSize = 0;
    uint16_t nonEmptyPartitionCount = 0;
    for (auto partition = 0; partition < numPartitions_; partition++) {
      if (partitionBuffers_[partition]) {
        uint64_t stringBufferSize{0};
        auto totalSize =
            partitionBuffers_[partition]->retainedSize(stringBufferSize);
        totalPartitionBufferSizeExcludingString += totalSize - stringBufferSize;
        totalPartitionStringBufferSize += stringBufferSize;
        nonEmptyPartitionCount++;
      }
    }
    if (nonEmptyPartitionCount > 0) {
      auto balancedPartitionBufferSize =
          totalPartitionBufferSizeExcludingString +
          (totalPartitionStringBufferSize / nonEmptyPartitionCount);
      for (auto partition = 0; partition < numPartitions_; partition++) {
        if (partitionBuffers_[partition]) {
          auto partitionData = checkedPointerCast<RowVector, BaseVector>(
              partitionBuffers_[partition]);
          ContinueFuture future;

          queues_[partition]->enqueue(
              partitionData,
              balancedPartitionBufferSize / nonEmptyPartitionCount,
              &future);
        }
        partitionBuffers_[partition] = nullptr;
      }
    }
    partitionBuffers_.resize(0);
    copyRanges_.resize(0);
  }
  for (const auto& queue : queues_) {
    queue->noMoreData();
  }
}

bool LocalPartition::isFinished() {
  if (!futures_.empty() || !noMoreInput_) {
    return false;
  }

  return true;
}

RowVectorPtr LocalPartition::getOutput() {
  if (!isDraining()) {
    return nullptr;
  }
  for (auto& queue : queues_) {
    queue->drain();
  }
  finishDrain();
  return nullptr;
}
} // namespace facebook::velox::exec
