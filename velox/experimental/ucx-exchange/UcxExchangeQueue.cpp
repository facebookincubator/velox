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
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

namespace facebook::velox::ucx_exchange {

void UcxExchangeQueue::noMoreSources() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    noMoreSources_ = true;
    promises = checkCompleteLocked();
  }
  clearPromises(promises);
}

void UcxExchangeQueue::close() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    promises = closeLocked();
  }
  clearPromises(promises);
}

void UcxExchangeQueue::enqueueLocked(
    PackedTableWithStreamPtr&& data,
    std::vector<ContinuePromise>& promises,
    uint64_t reservedReceiveBytes,
    bool* receiveReservationActive) {
  if (receiveAccountingClosed_) {
    if (receiveReservationActive != nullptr) {
      *receiveReservationActive = false;
    }
    return;
  }

  if (data == nullptr) {
    VELOX_CHECK_EQ(
        reservedReceiveBytes,
        0,
        "An end marker cannot consume a receive reservation");
    VELOX_CHECK_NULL(receiveReservationActive);
    ++numCompleted_;
    VLOG(2) << "[EX-QUEUE] source completed (null enqueued)"
            << " numCompleted=" << numCompleted_
            << " numSources=" << numSources_
            << " noMoreSources=" << noMoreSources_;
    auto completedPromises = checkCompleteLocked();
    promises.reserve(promises.size() + completedPromises.size());
    for (auto& promise : completedPromises) {
      promises.push_back(std::move(promise));
    }
    return;
  }

  auto dataSize = data->gpuDataSize();
  if (reservedReceiveBytes > 0) {
    VELOX_CHECK_EQ(
        reservedReceiveBytes,
        dataSize,
        "UCX receive reservation does not match received data");
    VELOX_CHECK_GE(
        reservedBytes_,
        reservedReceiveBytes,
        "UCX receive reservation released twice");
  }
  VELOX_CHECK(
      receiveReservationActive == nullptr || *receiveReservationActive,
      "UCX receive reservation is not active");
  VELOX_CHECK_LE(
      dataSize,
      std::numeric_limits<uint64_t>::max() - totalBytes_,
      "UCX receive queue byte accounting overflow");

  VELOX_CHECK_LE(
      promises_.size(),
      static_cast<size_t>(numberOfConsumers_),
      "Too many UCX exchange consumers are waiting");
  const auto unblockedConsumers =
      static_cast<size_t>(numberOfConsumers_) - promises_.size();
  VELOX_CHECK_LT(
      queue_.size(), queue_.max_size(), "UCX receive queue size overflow");
  const auto projectedQueueSize = queue_.size() + 1;
  const auto unassignedTables = projectedQueueSize > unblockedConsumers
      ? projectedQueueSize - unblockedConsumers
      : 0;
  const auto consumersToWake = std::min(promises_.size(), unassignedTables);
  VELOX_CHECK_LE(
      consumersToWake,
      promises.max_size() - promises.size(),
      "UCX exchange wake-up count overflow");
  // Reserve before committing the table and byte accounting. Moving a
  // ContinuePromise is noexcept, so fulfilling this precomputed set cannot
  // allocate after the commit.
  promises.reserve(promises.size() + consumersToWake);

  // Allocate/link the host queue node before changing byte counters. If this
  // throws, the caller still owns both the table and its receive reservation.
  queue_.push_back(std::move(data));
  publishSizeLocked();

  // Moving ownership from a posted receive to this queue is not memory
  // recovery and must not grow the adaptive prefetch window.
  if (reservedReceiveBytes > 0) {
    reservedBytes_ -= reservedReceiveBytes;
  }
  if (receiveReservationActive != nullptr) {
    // Publish the ownership transfer before any diagnostics or promise
    // handoff. If either unexpectedly throws, source cleanup must not release
    // this reservation a second time.
    *receiveReservationActive = false;
  }
  totalBytes_ += dataSize;
  if (peakBytes_ < totalBytes_) {
    peakBytes_ = totalBytes_;
  }

  if (receivedTables_ < std::numeric_limits<uint64_t>::max()) {
    ++receivedTables_;
  }
  receivedBytes_ = saturatingAdd(receivedBytes_, dataSize);

  // High-water-mark alerts: log when queue size crosses thresholds.
  auto newSize = static_cast<int64_t>(queue_.size());
  if (newSize > peakSize_) {
    if ((peakSize_ < 100 && newSize >= 100) ||
        (peakSize_ < 1000 && newSize >= 1000) ||
        (peakSize_ < 10000 && newSize >= 10000)) {
      VLOG(1) << "[EX-QUEUE] high water mark: queueSize=" << newSize
              << " peakBytes=" << peakBytes_
              << " receivedTables=" << receivedTables_;
    }
    peakSize_ = newSize;
  }

  for (size_t i = 0; i < consumersToWake; ++i) {
    // Resume one of the waiting drivers.
    auto it = promises_.begin();
    promises.push_back(std::move(it->second));
    promises_.erase(it);
  }
  if (consumersToWake > 0) {
    VLOG(2) << "[EX-QUEUE] waking " << consumersToWake << " consumers"
            << " queueSize=" << queue_.size();
  }
}

bool UcxExchangeQueue::tryReserveReceiveBytesLocked(uint64_t bytes) {
  if (receiveAccountingClosed_) {
    return false;
  }

  if (receiveHighWaterBytes_ > 0) {
    maybeGrowReceivePrefetchLimitLocked();
    const auto queuedBytes = queuedReceiveBytesLocked();
    const auto receiveLimit = receivePrefetchByteLimitLocked();
    if (queuedBytes > 0 &&
        (bytes > receiveLimit || queuedBytes > receiveLimit - bytes)) {
      return false;
    }
  }

  VELOX_CHECK_LE(
      bytes,
      std::numeric_limits<uint64_t>::max() - reservedBytes_,
      "UCX receive reservation accounting overflow");
  reservedBytes_ += bytes;
  return true;
}

void UcxExchangeQueue::releaseReceiveBytesLocked(uint64_t bytes) {
  if (receiveAccountingClosed_) {
    return;
  }
  consumeReceiveReservationLocked(bytes);
  maybeGrowReceivePrefetchLimitLocked();
}

void UcxExchangeQueue::consumeReceiveReservationLocked(uint64_t bytes) {
  if (receiveAccountingClosed_) {
    return;
  }
  VELOX_CHECK_GE(
      reservedBytes_, bytes, "UCX receive reservation released twice");
  reservedBytes_ -= bytes;
}

void UcxExchangeQueue::releaseInFlightReceiveBytesLocked(uint64_t bytes) {
  if (receiveAccountingClosed_) {
    return;
  }
  VELOX_CHECK_GE(
      inFlightBytes_, bytes, "UCX in-flight receive bytes released twice");
  inFlightBytes_ -= bytes;
  maybeGrowReceivePrefetchLimitLocked();
}

bool UcxExchangeQueue::recordReceiveAllocationPressureLocked(
    uint64_t attemptedBytes) {
  if (receiveAccountingClosed_ || receiveHighWaterBytes_ == 0) {
    return false;
  }

  const auto drainableBytes = queuedReceiveBytesLocked();
  if (drainableBytes == 0) {
    // Downstream operators may retain dequeued input until EOS. Only queued or
    // reserved receive memory is independently drainable and guaranteed to
    // produce a wake-up; retrying on in-flight-only ownership can deadlock EOS.
    return false;
  }

  const auto defaultLimit = defaultReceivePrefetchByteLimitLocked();
  const auto pressureBytes = saturatingAdd(drainableBytes, attemptedBytes);
  const auto reducedLimit = pressureBytes - pressureBytes / 8;
  const auto minimumLimit =
      std::max<uint64_t>(receiveHighWaterBytes_, attemptedBytes);
  receivePrefetchByteLimit_ =
      std::min(defaultLimit, std::max(minimumLimit, reducedLimit));
  receivePrefetchLimitActive_ = true;
  return true;
}

void UcxExchangeQueue::addPromiseLocked(
    int consumerId,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  ContinuePromise promise{"UcxExchangeQueue::dequeue"};
  *future = promise.getSemiFuture();
  auto it = promises_.find(consumerId);
  if (it != promises_.end()) {
    // resolve stale promises outside the lock to avoid broken promises
    *stalePromise = std::move(it->second);
    it->second = std::move(promise);
  } else {
    promises_[consumerId] = std::move(promise);
  }
  VELOX_CHECK_LE(promises_.size(), numberOfConsumers_);
}

PackedTableWithStreamPtr UcxExchangeQueue::dequeueLocked(
    int consumerId,
    bool* atEnd,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  VELOX_CHECK_NOT_NULL(future);
  if (!error_.empty()) {
    *atEnd = true;
    VELOX_FAIL(error_);
  }

  *atEnd = false;

  // check whether the queue is empty.
  PackedTableWithStreamPtr data = nullptr;
  if (queue_.empty()) {
    if (atEnd_) {
      *atEnd = true;
    } else {
      VLOG(2) << "[EX-QUEUE] consumer=" << consumerId
              << " blocked (empty queue, waiting for data)"
              << " numSources=" << numSources_
              << " numCompleted=" << numCompleted_
              << " waitingConsumers=" << (promises_.size() + 1);
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return data;
  }

  data = std::move(queue_.front());
  queue_.pop_front();
  publishSizeLocked();
  const auto dataSize = static_cast<uint64_t>(data->gpuDataSize());
  VELOX_CHECK_GE(totalBytes_, dataSize);
  totalBytes_ -= dataSize;
  if (tracksInFlightReceiveBytes()) {
    VELOX_CHECK_LE(
        dataSize,
        std::numeric_limits<uint64_t>::max() - inFlightBytes_,
        "UCX in-flight receive byte accounting overflow");
    inFlightBytes_ += dataSize;
  }

  return data;
}

void UcxExchangeQueue::setError(std::string_view error) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!error_.empty()) {
      return;
    }
    error_ = error;
    atEnd_ = true;
    // NOTE: clear the serialized page queue as we won't consume from an
    // errored queue.
    queue_.clear();
    publishSizeLocked();
    totalBytes_ = 0;
    reservedBytes_ = 0;
    inFlightBytes_ = 0;
    receiveAccountingClosed_ = true;
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
}

} // namespace facebook::velox::ucx_exchange
