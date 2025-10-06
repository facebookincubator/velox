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
#include "velox/experimental/cudf-exchange/CudfExchangeQueue.h"

namespace facebook::velox::cudf_exchange {

void CudfExchangeQueue::noMoreSources() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    noMoreSources_ = true;
    promises = checkCompleteLocked();
  }
  clearPromises(promises);
}

void CudfExchangeQueue::close() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    promises = closeLocked();
  }
  clearPromises(promises);
}

void CudfExchangeQueue::enqueueLocked(
    std::unique_ptr<cudf::packed_columns>&& columns,
    std::vector<ContinuePromise>& promises) {
  if (columns == nullptr) {
    ++numCompleted_;
    auto completedPromises = checkCompleteLocked();
    promises.reserve(promises.size() + completedPromises.size());
    for (auto& promise : completedPromises) {
      promises.push_back(std::move(promise));
    }
    return;
  }

  totalBytes_ += columns->gpu_data->size();
  if (peakBytes_ < totalBytes_) {
    peakBytes_ = totalBytes_;
  }

  ++receivedColumns_;
  receivedBytes_ += columns->gpu_data->size();

  queue_.push_back(std::move(columns));
  while (!promises_.empty()) {
    VELOX_CHECK_LE(promises_.size(), numberOfConsumers_);
    const int32_t unblockedConsumers = numberOfConsumers_ - promises_.size();
    const int64_t unasignedColumns = queue_.size() - unblockedConsumers;
    if (unasignedColumns <= 0) {
      break;
    }
    // Resume one of the waiting drivers.
    auto it = promises_.begin();
    promises.push_back(std::move(it->second));
    promises_.erase(it);
  }
}

void CudfExchangeQueue::addPromiseLocked(
    int consumerId,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  ContinuePromise promise{"CudfExchangeQueue::dequeue"};
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

std::unique_ptr<cudf::packed_columns> CudfExchangeQueue::dequeueLocked(
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
  std::unique_ptr<cudf::packed_columns> column = nullptr;
  if (queue_.empty()) {
    if (atEnd_) {
      *atEnd = true;
    } else {
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return column;
  }

  column = std::move(queue_.front());
  queue_.pop_front();
  receivedColumns_++;
  totalBytes_ -= column->gpu_data->size();

  return column;
}

void CudfExchangeQueue::setError(const std::string& error) {
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

} // namespace facebook::velox::cudf_exchange
