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

#include "velox/exec/MaterializedOutputBuffer.h"

#include <algorithm>
#include <numeric>
#include <thread>

#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {
namespace {

constexpr double kReclaimDrainThresholdRatio = 0.67;
constexpr int64_t kDrainChunkMultiplier = 2;
constexpr int64_t kHighWatermarkNumerator = 9;
constexpr int64_t kLowWatermarkNumerator = 7;
constexpr int64_t kWatermarkDenominator = 10;

int64_t computePartitionDrainThreshold(
    int32_t numPartitions,
    int64_t maxBufferedBytes,
    int64_t requestedThreshold) {
  VELOX_CHECK_GT(numPartitions, 0);
  VELOX_CHECK_GE(maxBufferedBytes, numPartitions);
  VELOX_CHECK_GT(requestedThreshold, 0);
  return std::min(requestedThreshold, maxBufferedBytes / numPartitions);
}

struct TrackedBufferInfo {
  memory::MemoryPool* pool;
  size_t size;
};

} // namespace

std::string MaterializedOutputBuffer::stateName(State state) {
  switch (state) {
    case State::kActive:
      return "kActive";
    case State::kDraining:
      return "kDraining";
    case State::kClosed:
      return "kClosed";
    case State::kAborted:
      return "kAborted";
  }
  return fmt::format("Unknown({})", static_cast<int>(state));
}

MaterializedOutputBuffer::PartitionBuffer::PartitionBuffer(
    int32_t partition,
    int64_t drainThreshold,
    MaterializedOutputBuffer* buffer)
    : partition_(partition), drainThreshold_(drainThreshold), buffer_(buffer) {}

bool MaterializedOutputBuffer::PartitionBuffer::tryAcquireFlushing() {
  bool expected = false;
  return flushing_.compare_exchange_strong(expected, true);
}

void MaterializedOutputBuffer::PartitionBuffer::releaseFlushing() {
  flushing_ = false;
}

int64_t MaterializedOutputBuffer::PartitionBuffer::drainAndFlush() {
  int64_t totalDrainedBytes = 0;
  std::deque<std::unique_ptr<folly::IOBuf>> chunk;
  int64_t chunkBytes = 0;

  const auto flushChunk = [&]() {
    buffer_->flushDrained(partition_, chunk);
    bufferedBytes_.fetch_sub(chunkBytes);
    totalDrainedBytes += chunkBytes;
    chunk.clear();
    chunkBytes = 0;
  };

  std::unique_ptr<folly::IOBuf> rowGroup;
  while (rowGroupsQueue_.try_dequeue(rowGroup)) {
    const auto rowGroupBytes =
        static_cast<int64_t>(rowGroup->computeChainDataLength());
    if (chunkBytes > 0 &&
        chunkBytes + rowGroupBytes > buffer_->drainChunkThresholdBytes_) {
      flushChunk();
    }
    chunk.push_back(std::move(rowGroup));
    chunkBytes += rowGroupBytes;
  }
  if (!chunk.empty()) {
    flushChunk();
  }
  return totalDrainedBytes;
}

int64_t MaterializedOutputBuffer::PartitionBuffer::tryDrainPartition(
    int64_t targetBytes) {
  if (!tryAcquireFlushing()) {
    ++buffer_->concurrentAppendCount_;
    return 0;
  }
  ++buffer_->flushAcquireCount_;

  int64_t totalDrainedBytes = 0;
  do {
    {
      SCOPE_EXIT {
        releaseFlushing();
      };
      totalDrainedBytes += drainAndFlush();
    }
  } while (bufferedBytes_ >= targetBytes && tryAcquireFlushing());
  return totalDrainedBytes;
}

int64_t MaterializedOutputBuffer::PartitionBuffer::finish(bool success) {
  while (!tryAcquireFlushing()) {
    std::this_thread::yield();
  }
  SCOPE_EXIT {
    releaseFlushing();
  };
  if (closed_.exchange(true)) {
    return 0;
  }
  if (success) {
    return drainAndFlush();
  }

  int64_t discardedBytes = 0;
  std::unique_ptr<folly::IOBuf> rowGroup;
  while (rowGroupsQueue_.try_dequeue(rowGroup)) {
    discardedBytes += static_cast<int64_t>(rowGroup->computeChainDataLength());
  }
  bufferedBytes_.fetch_sub(discardedBytes);
  return discardedBytes;
}

int64_t MaterializedOutputBuffer::PartitionBuffer::enqueue(
    std::unique_ptr<folly::IOBuf> rowGroup) {
  VELOX_CHECK(!closed_, "enqueue called on closed partition");
  const auto rowGroupBytes =
      static_cast<int64_t>(rowGroup->computeChainDataLength());
  rowGroupsQueue_.enqueue(std::move(rowGroup));
  const auto newBytes = (bufferedBytes_ += rowGroupBytes);
  return newBytes >= drainThreshold_ ? tryDrainPartition(drainThreshold_) : 0;
}

MaterializedOutputBuffer::MaterializedOutputBuffer(
    int32_t numPartitions,
    std::shared_ptr<ExchangeSink> sink,
    int64_t maxBufferedBytes,
    int64_t requestedPartitionDrainThreshold,
    std::shared_ptr<memory::MemoryPool> pool)
    : numPartitions_(numPartitions),
      maxBufferedBytes_(maxBufferedBytes),
      partitionDrainThreshold_(computePartitionDrainThreshold(
          numPartitions,
          maxBufferedBytes,
          requestedPartitionDrainThreshold)),
      reclaimDrainThresholdBytes_(
          std::max<int64_t>(
              1,
              partitionDrainThreshold_ * kReclaimDrainThresholdRatio)),
      drainChunkThresholdBytes_(
          partitionDrainThreshold_ * kDrainChunkMultiplier),
      highWatermarkBytes_(
          maxBufferedBytes_ * kHighWatermarkNumerator / kWatermarkDenominator),
      lowWatermarkBytes_(
          maxBufferedBytes_ * kLowWatermarkNumerator / kWatermarkDenominator),
      pool_(std::move(pool)),
      sink_(std::move(sink)) {
  VELOX_CHECK_NOT_NULL(sink_);
  initializePartitions();
}

MaterializedOutputBuffer::~MaterializedOutputBuffer() {
  if (state_ != State::kClosed && state_ != State::kAborted) {
    try {
      abort();
    } catch (...) {
      LOG(ERROR) << "MaterializedOutputBuffer abort failed in destructor";
    }
  }
}

void MaterializedOutputBuffer::initializePartitions() {
  VELOX_CHECK_GT(lowWatermarkBytes_, 0);
  VELOX_CHECK_LT(lowWatermarkBytes_, highWatermarkBytes_);
  VELOX_CHECK_LE(highWatermarkBytes_, maxBufferedBytes_);
  VELOX_CHECK_LT(
      numPartitions_ * reclaimDrainThresholdBytes_,
      lowWatermarkBytes_,
      "reclaim drain threshold is too high for the low watermark");

  partitionBuffers_.reserve(numPartitions_);
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    partitionBuffers_.push_back(
        std::make_unique<PartitionBuffer>(
            partition, partitionDrainThreshold_, this));
  }
}

std::unique_ptr<folly::IOBuf> MaterializedOutputBuffer::allocateTrackedIOBuf(
    size_t size) {
  if (pool_ == nullptr) {
    return folly::IOBuf::create(size);
  }
  void* buffer = pool_->allocate(size);
  auto* info = new TrackedBufferInfo{pool_.get(), size};
  auto iobuf =
      folly::IOBuf::takeOwnership(buffer, size, freeTrackedIOBuf, info);
  iobuf->trimEnd(size);
  return iobuf;
}

void MaterializedOutputBuffer::freeTrackedIOBuf(void* buffer, void* userData) {
  auto* info = static_cast<TrackedBufferInfo*>(userData);
  info->pool->free(buffer, info->size);
  delete info;
}

void MaterializedOutputBuffer::enqueue(
    int32_t partition,
    std::unique_ptr<folly::IOBuf> rowGroup) {
  VELOX_CHECK_GE(partition, 0);
  VELOX_CHECK_LT(partition, numPartitions_);
  if (state_ == State::kAborted) {
    return;
  }
  VELOX_CHECK_EQ(
      state_.load(), State::kActive, "enqueue called after noMoreData()");

  const auto rowGroupBytes =
      static_cast<int64_t>(rowGroup->computeChainDataLength());
  const auto currentBytes = (bufferedBytes_ += rowGroupBytes);
  auto peakBytes = peakBufferedBytes_.load(std::memory_order_relaxed);
  while (currentBytes > peakBytes &&
         !peakBufferedBytes_.compare_exchange_weak(
             peakBytes, currentBytes, std::memory_order_relaxed)) {
  }

  const auto drainedBytes =
      partitionBuffers_[partition]->enqueue(std::move(rowGroup));
  if (drainedBytes > 0) {
    updateDrainStats(drainedBytes);
    const auto hasBlockedDrivers = blockedPromises_.withRLock(
        [](const auto& promises) { return !promises.empty(); });
    if (hasBlockedDrivers && bufferedBytes_ >= lowWatermarkBytes_) {
      tryDrainForBackpressure();
    }
    maybeWakeBlockedDrivers();
  }
}

bool MaterializedOutputBuffer::isBufferFull() const {
  return bufferedBytes_ >= highWatermarkBytes_;
}

void MaterializedOutputBuffer::addBlockedPromise(ContinueFuture* future) {
  ContinuePromise promise{"MaterializedOutputBuffer::addBlockedPromise"};
  *future = promise.getSemiFuture();
  ++backpressureBlockCount_;
  blockedPromises_.withWLock(
      [&](auto& promises) { promises.push_back(std::move(promise)); });
  maybeWakeBlockedDrivers();
}

void MaterializedOutputBuffer::tryDrainForBackpressure() {
  while (bufferedBytes_ >= lowWatermarkBytes_) {
    const auto drainedBytes = tryDrainPartitionsInternal();
    backpressureDrainedBytes_ += static_cast<int64_t>(drainedBytes);
    if (drainedBytes == 0) {
      break;
    }
  }
}

BlockingReason MaterializedOutputBuffer::isBlocked(ContinueFuture* future) {
  if (!isBufferFull()) {
    return BlockingReason::kNotBlocked;
  }

  tryDrainForBackpressure();
  if (isBufferFull() && reclaimableBufferedBytes() > 0) {
    addBlockedPromise(future);
    return BlockingReason::kWaitForConsumer;
  }
  if (isBufferFull()) {
    LOG_EVERY_N(WARNING, 256)
        << "MaterializedOutputBuffer is full with no reclaimable data; "
           "leaving the producer runnable";
  }
  return BlockingReason::kNotBlocked;
}

void MaterializedOutputBuffer::maybeWakeBlockedDrivers() {
  if (bufferedBytes_ >= lowWatermarkBytes_) {
    return;
  }
  std::vector<ContinuePromise> promisesToFulfill;
  blockedPromises_.withWLock(
      [&](auto& promises) { promisesToFulfill.swap(promises); });
  backpressureWakeCount_ += static_cast<int64_t>(promisesToFulfill.size());
  for (auto& promise : promisesToFulfill) {
    promise.setValue();
  }
}

void MaterializedOutputBuffer::flushDrained(
    int32_t partition,
    std::deque<std::unique_ptr<folly::IOBuf>>& rowGroups) {
  if (rowGroups.empty()) {
    return;
  }
  sink_->appendBatch(partition, std::move(rowGroups));
  ++appendCount_;
}

void MaterializedOutputBuffer::updateDrainStats(int64_t drainedBytes) {
  ++drainCount_;
  drainedBytes_ += drainedBytes;
  bufferedBytes_.fetch_sub(drainedBytes);
}

int64_t MaterializedOutputBuffer::drainPartitionInternal(
    int32_t partition,
    int64_t targetBytes) {
  VELOX_CHECK_GE(partition, 0);
  VELOX_CHECK_LT(partition, numPartitions_);
  const auto drainedBytes =
      partitionBuffers_[partition]->tryDrainPartition(targetBytes);
  if (drainedBytes > 0) {
    updateDrainStats(drainedBytes);
    maybeWakeBlockedDrivers();
  }
  return drainedBytes;
}

int64_t MaterializedOutputBuffer::drainPartition(int32_t partition) {
  return drainPartitionInternal(partition, 1);
}

uint64_t MaterializedOutputBuffer::drainAll() {
  uint64_t drainedBytes = 0;
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    drainedBytes += drainPartition(partition);
  }
  return drainedBytes;
}

uint64_t MaterializedOutputBuffer::reclaimableBufferedBytes() const {
  uint64_t reclaimableBytes = 0;
  for (const auto& partition : partitionBuffers_) {
    const auto partitionBytes = partition->bufferedBytes_.load();
    if (partitionBytes > reclaimDrainThresholdBytes_) {
      reclaimableBytes += partitionBytes - reclaimDrainThresholdBytes_;
    }
  }
  return reclaimableBytes;
}

uint64_t MaterializedOutputBuffer::tryDrainPartitionsInternal() {
  std::vector<int32_t> orderedPartitions(numPartitions_);
  std::iota(orderedPartitions.begin(), orderedPartitions.end(), 0);

  std::vector<int64_t> partitionSizes(numPartitions_);
  for (int32_t partition = 0; partition < numPartitions_; ++partition) {
    partitionSizes[partition] = partitionBuffers_[partition]->bufferedBytes_;
  }
  std::sort(
      orderedPartitions.begin(),
      orderedPartitions.end(),
      [&](int32_t lhs, int32_t rhs) {
        return partitionSizes[lhs] > partitionSizes[rhs];
      });

  uint64_t drainedBytes = 0;
  for (const auto partition : orderedPartitions) {
    if (partitionSizes[partition] < reclaimDrainThresholdBytes_) {
      break;
    }
    drainedBytes +=
        drainPartitionInternal(partition, reclaimDrainThresholdBytes_);
  }
  return drainedBytes;
}

void MaterializedOutputBuffer::noMoreData() {
  std::lock_guard<std::mutex> lock(lifecycleMutex_);
  auto expectedState = State::kActive;
  if (!state_.compare_exchange_strong(expectedState, State::kDraining)) {
    return;
  }

  try {
    for (auto& partition : partitionBuffers_) {
      const auto drainedBytes = partition->finish(true);
      if (drainedBytes > 0) {
        updateDrainStats(drainedBytes);
      }
    }
    output_ = sink_->finish();
    state_ = State::kClosed;
    maybeWakeBlockedDrivers();
  } catch (...) {
    state_ = State::kAborted;
    for (auto& partition : partitionBuffers_) {
      partition->finish(false);
    }
    bufferedBytes_ = 0;
    try {
      sink_->abort();
    } catch (...) {
    }
    maybeWakeBlockedDrivers();
    throw;
  }
}

void MaterializedOutputBuffer::abort() {
  std::lock_guard<std::mutex> lock(lifecycleMutex_);
  const auto previousState = state_.exchange(State::kAborted);
  if (previousState == State::kClosed || previousState == State::kAborted) {
    return;
  }

  for (auto& partition : partitionBuffers_) {
    partition->finish(false);
  }
  bufferedBytes_ = 0;
  sink_->abort();
  output_.reset();
  maybeWakeBlockedDrivers();
}

void MaterializedOutputBuffer::setNumDrivers(uint32_t numDrivers) {
  std::lock_guard<std::mutex> lock(driverMutex_);
  if (numDrivers_ == 0) {
    numDrivers_ = numDrivers;
  }
}

bool MaterializedOutputBuffer::noMoreDrivers() {
  bool isLastDriver = false;
  {
    std::lock_guard<std::mutex> lock(driverMutex_);
    ++numFinishedDrivers_;
    isLastDriver = numDrivers_ > 0 && numFinishedDrivers_ >= numDrivers_;
  }
  if (isLastDriver) {
    noMoreData();
  }
  return isLastDriver;
}

folly::F14FastMap<std::string, int64_t> MaterializedOutputBuffer::stats()
    const {
  auto result = sink_->stats();
  result["materializedOutputBuffer.drainedBytes"] = drainedBytes_;
  result["materializedOutputBuffer.drainCount"] = drainCount_;
  result["materializedOutputBuffer.currentDrainThreshold"] =
      partitionDrainThreshold_;
  result["materializedOutputBuffer.totalAppendCalls"] = appendCount_;
  result["materializedOutputBuffer.peakBufferedBytes"] = peakBufferedBytes_;
  result["materializedOutputBuffer.concurrentAppendCount"] =
      concurrentAppendCount_;
  result["materializedOutputBuffer.flushAcquireCount"] = flushAcquireCount_;
  result["materializedOutputBuffer.backpressureBlockCount"] =
      backpressureBlockCount_;
  result["materializedOutputBuffer.backpressureWakeCount"] =
      backpressureWakeCount_;
  result["materializedOutputBuffer.backpressureDrainedBytes"] =
      backpressureDrainedBytes_;
  return result;
}

folly::F14FastMap<int32_t, std::string>
MaterializedOutputBuffer::outputLocations() const {
  std::lock_guard<std::mutex> lock(lifecycleMutex_);
  return output_.has_value() ? output_->locations
                             : folly::F14FastMap<int32_t, std::string>{};
}

} // namespace facebook::velox::exec
