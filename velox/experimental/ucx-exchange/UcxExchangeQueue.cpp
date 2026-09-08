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
    std::vector<ContinuePromise>& promises) {
  if (data == nullptr) {
    ++numCompleted_;
    auto completedPromises = checkCompleteLocked();
    promises.reserve(promises.size() + completedPromises.size());
    for (auto& promise : completedPromises) {
      promises.push_back(std::move(promise));
    }
    return;
  }

  auto dataSize = data->gpuDataSize();
  totalBytes_ += dataSize;

  ++receivedTables_;
  receivedBytes_ += dataSize;

  queue_.push_back(std::move(data));

  while (!promises_.empty()) {
    VELOX_CHECK_LE(promises_.size(), numberOfConsumers_);
    const int32_t unblockedConsumers = numberOfConsumers_ - promises_.size();
    const int64_t unassignedTables = queue_.size() - unblockedConsumers;
    if (unassignedTables <= 0) {
      break;
    }
    // Resume one of the waiting drivers.
    auto it = promises_.begin();
    promises.push_back(std::move(it->second));
    promises_.erase(it);
  }
}

bool UcxExchangeQueue::tryReserveReceiveBytesLocked(uint64_t bytes) {
  if (receiveHighWaterBytes_ > 0) {
    maybeGrowReceivePrefetchLimitLocked();
    const auto queuedBytes = queuedReceiveBytesLocked();
    const auto receiveLimit = receivePrefetchByteLimitLocked();
    if (queuedBytes > 0 &&
        (bytes > receiveLimit || queuedBytes > receiveLimit - bytes)) {
      return false;
    }
  }

  reservedBytes_ += bytes;
  return true;
}

void UcxExchangeQueue::releaseReceiveBytesLocked(uint64_t bytes) {
  if (reservedBytes_ < bytes) {
    reservedBytes_ = 0;
  } else {
    reservedBytes_ -= bytes;
  }
  maybeGrowReceivePrefetchLimitLocked();
}

void UcxExchangeQueue::releaseInFlightReceiveBytesLocked(uint64_t bytes) {
  if (inFlightBytes_ < bytes) {
    inFlightBytes_ = 0;
  } else {
    inFlightBytes_ -= bytes;
  }

  maybeGrowReceivePrefetchLimitLocked();
}

bool UcxExchangeQueue::recordReceiveAllocationPressureLocked(
    uint64_t attemptedBytes) {
  if (receiveHighWaterBytes_ == 0) {
    return false;
  }

  const auto retainedBytes = retainedReceiveBytesLocked();
  if (retainedBytes == 0) {
    return false;
  }

  const auto defaultLimit = defaultReceivePrefetchByteLimitLocked();
  const auto pressureBytes = retainedBytes + attemptedBytes;
  const auto reducedLimit = pressureBytes - pressureBytes / 8;
  const auto minimumLimit =
      std::max<uint64_t>(receiveHighWaterBytes_, attemptedBytes);
  receivePrefetchByteLimit_ =
      std::min<uint64_t>(defaultLimit, std::max(minimumLimit, reducedLimit));
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
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return data;
  }

  data = std::move(queue_.front());
  queue_.pop_front();
  const auto dataSize = static_cast<int64_t>(data->gpuDataSize());
  VELOX_CHECK_GE(totalBytes_, dataSize);
  totalBytes_ -= dataSize;
  if (tracksInFlightReceiveBytes()) {
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
    totalBytes_ = 0;
    reservedBytes_ = 0;
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
}

} // namespace facebook::velox::ucx_exchange
