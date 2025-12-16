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
#pragma once

#include "velox/exec/SerializedPage.h"

#include <mutex>

namespace facebook::velox::exec {

/// Queue of results retrieved from source. Owned by shared_ptr by
/// Exchange and client threads and registered callbacks waiting
/// for input.
class ExchangeQueue {
 public:
  explicit ExchangeQueue(
      int32_t numberOfConsumers,
      uint64_t minOutputBatchBytes)
      : numberOfConsumers_{numberOfConsumers},
        minOutputBatchBytes_{minOutputBatchBytes} {
    VELOX_CHECK_GE(numberOfConsumers, 1);
  }

  ~ExchangeQueue() {
    clearAllPromises();
  }

  std::mutex& mutex() {
    return mutex_;
  }

  bool empty() const {
    return queue_.empty();
  }

  /// Enqueues 'page' to the queue. One random promise(top of promise queue)
  /// associated with the future that is waiting for the data from the queue is
  /// returned in 'promises' if 'page' is not nullptr. When 'page' is nullptr
  /// and the queue is completed serving data, all left over promises will be
  /// returned in 'promises'. When 'page' is nullptr and the queue is not
  /// completed serving data, no 'promises' will be added and returned.
  void enqueueLocked(
      std::unique_ptr<SerializedPageBase>&& page,
      std::vector<ContinuePromise>& promises);

  /// If data is permanently not available, e.g. the source cannot be
  /// contacted, this registers an error message and causes the reading
  /// Exchanges to throw with the message.
  void setError(const std::string& error);

  /// Returns pages of data.
  ///
  /// Returns empty list if no data is available. If data is still expected,
  /// sets 'atEnd' to false and 'future' to a Future that will complete when
  /// data arrives. If no more data is expected, sets 'atEnd' to true. Returns
  /// at least one page if data is available. If multiple pages are available,
  /// returns as many pages as fit within 'maxBytes', but no fewer than one.
  /// Calling this method with 'maxBytes' of 1 returns at most one page.
  ///
  /// The data may be compressed, in which case 'maxBytes' applies to compressed
  /// size.
  std::vector<std::unique_ptr<SerializedPageBase>> dequeueLocked(
      int consumerId,
      uint32_t maxBytes,
      bool* atEnd,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  /// Returns the total bytes held by SerializedPages in 'this'.
  int64_t totalBytes() const {
    return totalBytes_;
  }

  /// Returns the maximum value of total bytes.
  uint64_t peakBytes() const {
    return peakBytes_;
  }

  /// Returns total number of pages received from all sources.
  uint64_t receivedPages() const {
    return receivedPages_;
  }

  /// Returns an average size of received pages. Returns 0 if hasn't received
  /// any pages yet.
  uint64_t averageReceivedPageBytes() const {
    return receivedPages_ > 0 ? receivedBytes_ / receivedPages_ : 0;
  }

  void addSourceLocked() {
    VELOX_CHECK(!noMoreSources_, "addSource called after noMoreSources");
    numSources_++;
  }

  void noMoreSources();

  bool hasNoMoreSources() const {
    return noMoreSources_;
  }

  void close();

 private:
  std::vector<ContinuePromise> closeLocked() {
    queue_.clear();
    return clearAllPromisesLocked();
  }

  std::vector<ContinuePromise> checkNoMoreInput() {
    if (noMoreSources_ && numCompleted_ == numSources_) {
      noMoreInput_ = true;
      return clearAllPromisesLocked();
    }
    return {};
  }

  void addPromiseLocked(
      int consumerId,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  void clearAllPromises() {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      promises = clearAllPromisesLocked();
    }
    clearPromises(promises);
  }

  std::vector<ContinuePromise> clearAllPromisesLocked() {
    std::vector<ContinuePromise> promises;
    promises.reserve(promises_.size());

    auto it = promises_.begin();
    while (it != promises_.end()) {
      promises.push_back(std::move(it->second));
      it = promises_.erase(it);
    }
    VELOX_CHECK(promises_.empty());
    return promises;
  }

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  int64_t minOutputBatchBytesLocked() const;

  const int32_t numberOfConsumers_;
  const uint64_t minOutputBatchBytes_;

  int numCompleted_{0};
  int numSources_{0};
  tsan_atomic<bool> noMoreSources_{false};
  // True if no more pages will be enqueued. This can be due to all sources
  // completing normally or an error. Note that the queue itself may still
  // contain data to be consumed.
  bool noMoreInput_{false};

  std::mutex mutex_;
  std::deque<std::unique_ptr<SerializedPageBase>> queue_;
  // The map from consumer id to the waiting promise
  folly::F14FastMap<int, ContinuePromise> promises_;

  // When set, all promises will be realized and the next dequeue will
  // throw an exception with this message.
  std::string error_;
  // Total size of SerializedPages in queue.
  int64_t totalBytes_{0};
  // Number of SerializedPages received.
  int64_t receivedPages_{0};
  // Total size of SerializedPages received. Used to calculate an average
  // expected size.
  int64_t receivedBytes_{0};
  // Maximum value of totalBytes_.
  int64_t peakBytes_{0};
};
} // namespace facebook::velox::exec
