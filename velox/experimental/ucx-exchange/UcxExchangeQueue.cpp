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
  totalBytes_ += dataSize;
  if (peakBytes_ < totalBytes_) {
    peakBytes_ = totalBytes_;
  }

  ++receivedTables_;
  receivedBytes_ += dataSize;

  queue_.push_back(std::move(data));

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

  size_t wokenConsumers = 0;
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
    ++wokenConsumers;
  }
  if (wokenConsumers > 0) {
    VLOG(2) << "[EX-QUEUE] waking " << wokenConsumers << " consumers"
            << " queueSize=" << queue_.size();
  }
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
  totalBytes_ -= data->gpuDataSize();

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
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
}

} // namespace facebook::velox::ucx_exchange
