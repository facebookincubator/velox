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

#include <algorithm>

namespace facebook::velox::ucx_exchange {

UcxExchangeQueue::~UcxExchangeQueue() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
}

UcxExchangeQueue::SourceId UcxExchangeQueue::addSourceLocked() {
  VELOX_CHECK(!noMoreSources_, "addSource called after noMoreSources");
  sources_.emplace_back();
  return static_cast<SourceId>(sources_.size() - 1);
}

UcxExchangeQueue::SourceId UcxExchangeQueue::addReleasedSourceLocked() {
  VELOX_CHECK(closed_, "A released source must be added after close");
  sources_.push_back({SourceState::kReleased, std::nullopt});
  return static_cast<SourceId>(sources_.size() - 1);
}

void UcxExchangeQueue::sourceOpened(SourceId id) {
  std::lock_guard<std::mutex> l(mutex_);
  // Reached from UCX callbacks, which must not throw.
  VELOX_DCHECK_LT(static_cast<size_t>(id), sources_.size());
  auto& entry = sources_[id];
  entry.state =
      decideSource(
          entry.state, SourceEvent::kResponseAccepted, ChannelOutcome::clean())
          .next;
}

bool UcxExchangeQueue::isFinishedLocked() const {
  return noMoreSources_ &&
      std::all_of(sources_.begin(), sources_.end(), [](const auto& entry) {
           return entry.state == SourceState::kEnded ||
               entry.state == SourceState::kReleased;
         });
}

std::deque<PackedTableWithStreamPtr> UcxExchangeQueue::takeTablesLocked() {
  std::deque<PackedTableWithStreamPtr> tables;
  tables.swap(queue_);
  publishSizeLocked();
  totalBytes_ = 0;
  return tables;
}

void UcxExchangeQueue::noMoreSources() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    noMoreSources_ = true;
    if (isFinishedLocked()) {
      promises = clearAllPromisesLocked();
    }
  }
  clearPromises(promises);
}

void UcxExchangeQueue::end(SourceId id, ChannelOutcome outcome) {
  std::vector<ContinuePromise> promises;
  std::deque<PackedTableWithStreamPtr> tables;
  // The error this call decided, logged after the lock is released. An error
  // that comes after the source ended or was released is not logged.
  std::optional<std::string> decidedError;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // Reached from UCX callbacks, which must not throw.
    VELOX_DCHECK_LT(static_cast<size_t>(id), sources_.size());
    auto& entry = sources_[id];
    auto decision = decideSource(entry.state, SourceEvent::kEnded, outcome);
    entry.state = decision.next;
    if (!decision.outcome.has_value()) {
      return;
    }
    entry.outcome = std::move(decision.outcome);
    VLOG(2) << "[EX-QUEUE] source " << id << " ended: " << entry.outcome->kind;
    if (!entry.outcome->isClean()) {
      decidedError = entry.outcome->error;
    }
    if (!entry.outcome->isClean() && !failedSource_.has_value() && !closed_) {
      failedSource_ = id;
      // A failed consumer reads nothing more.
      tables = takeTablesLocked();
      promises = clearAllPromisesLocked();
    } else if (isFinishedLocked()) {
      promises = clearAllPromisesLocked();
    }
  }
  if (decidedError.has_value()) {
    VLOG(0) << "[EX-QUEUE] source " << id << " failed: " << *decidedError;
  }
  clearPromises(promises);
}

std::vector<UcxExchangeQueue::SourceId> UcxExchangeQueue::close() {
  std::vector<SourceId> toTell;
  std::vector<ContinuePromise> promises;
  std::deque<PackedTableWithStreamPtr> tables;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (closed_) {
      return toTell;
    }
    closed_ = true;
    for (SourceId id = 0; id < static_cast<SourceId>(sources_.size()); ++id) {
      auto& entry = sources_[id];
      const auto decision = decideSource(
          entry.state, SourceEvent::kClientClosed, ChannelOutcome::clean());
      entry.state = decision.next;
      if (decision.tellServer) {
        toTell.push_back(id);
      }
    }
    tables = takeTablesLocked();
    promises = clearAllPromisesLocked();
  }
  clearPromises(promises);
  return toTell;
}

bool UcxExchangeQueue::isClosed() {
  std::lock_guard<std::mutex> l(mutex_);
  return closed_;
}

SourceState UcxExchangeQueue::sourceState(SourceId id) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_LT(static_cast<size_t>(id), sources_.size());
  return sources_[id].state;
}

bool UcxExchangeQueue::enqueueTable(
    SourceId id,
    PackedTableWithStreamPtr& table,
    std::vector<ContinuePromise>& promises) {
  std::lock_guard<std::mutex> l(mutex_);
  // Reached from UCX callbacks, which must not throw.
  VELOX_DCHECK_LT(static_cast<size_t>(id), sources_.size());
  if (sources_[id].state != SourceState::kOpen || failedSource_.has_value() ||
      closed_) {
    return false;
  }

  auto dataSize = table->gpuDataSize();
  totalBytes_ += dataSize;
  if (peakBytes_ < totalBytes_) {
    peakBytes_ = totalBytes_;
  }

  ++receivedTables_;
  receivedBytes_ += dataSize;

  queue_.push_back(std::move(table));
  publishSizeLocked();

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

std::vector<ContinuePromise> UcxExchangeQueue::clearAllPromisesLocked() {
  std::vector<ContinuePromise> promises;
  promises.reserve(promises_.size());
  for (auto& [consumerId, promise] : promises_) {
    promises.push_back(std::move(promise));
  }
  promises_.clear();
  return promises;
}

PackedTableWithStreamPtr UcxExchangeQueue::dequeueLocked(
    int consumerId,
    bool* atEnd,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  VELOX_CHECK_NOT_NULL(future);
  // An error decided before close() still fails the reads after it.
  if (failedSource_.has_value()) {
    *atEnd = true;
    VELOX_FAIL(sources_[*failedSource_].outcome->error);
  }
  if (closed_) {
    *atEnd = true;
    return nullptr;
  }

  *atEnd = false;
  if (queue_.empty()) {
    if (isFinishedLocked()) {
      *atEnd = true;
    } else {
      VLOG(2) << "[EX-QUEUE] consumer=" << consumerId
              << " blocked (empty queue, waiting for data)"
              << " numSources=" << sources_.size()
              << " waitingConsumers=" << (promises_.size() + 1);
      addPromiseLocked(consumerId, future, stalePromise);
    }
    return nullptr;
  }

  auto data = std::move(queue_.front());
  queue_.pop_front();
  publishSizeLocked();
  totalBytes_ -= data->gpuDataSize();
  return data;
}

} // namespace facebook::velox::ucx_exchange
