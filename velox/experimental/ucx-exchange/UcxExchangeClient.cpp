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
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"

namespace facebook::velox::ucx_exchange {

UcxExchangeClient::UcxExchangeClient(
    std::string taskId,
    int destination,
    int32_t numberOfConsumers,
    uint64_t receiveHighWaterBytes,
    int32_t requestDataSizesMaxWaitSec)
    : taskId_{std::move(taskId)},
      maxQueuedColumns_(kDefaultMaxQueuedColumns),
      requestDataSizesMaxWaitSec_(requestDataSizesMaxWaitSec),
      queue_(
          std::make_shared<UcxExchangeQueue>(
              numberOfConsumers,
              receiveHighWaterBytes)) {
  VELOX_CHECK_GE(
      destination, 0, "Exchange client destination must not be negative");
}

void UcxExchangeClient::addRemoteTaskId(std::string_view remoteTaskId) {
  std::shared_ptr<UcxExchangeSource> source;
  std::shared_ptr<UcxExchangeSource> toStart;
  std::shared_ptr<UcxExchangeSource> toClose;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    const auto [remoteTaskIdIt, inserted] =
        remoteTaskIds_.emplace(remoteTaskId);
    if (!inserted) {
      // Do not add sources twice. Presto protocol may add duplicate sources
      // and the task updates have no guarantees of arriving in order.
      return;
    }

    try {
      // create() constructs a dormant source; start() below is the publication
      // point and is intentionally outside this queue critical section.
      source = UcxExchangeSource::create(taskId_, *remoteTaskIdIt, queue_);
    } catch (...) {
      remoteTaskIds_.erase(remoteTaskIdIt);
      throw;
    }

    if (closed_) {
      toClose = std::move(source);
    } else {
      try {
        // Retain and account for the source before publishing it. A fast
        // Communicator thread may complete the handshake immediately after
        // start(), so setRegistered() must already be visible at that point.
        sources_.push_back(source);
        queue_->addSourceLocked();
        source->setRegistered();
        toStart = source;
      } catch (...) {
        if (!sources_.empty() && sources_.back() == source) {
          sources_.pop_back();
        }
        remoteTaskIds_.erase(remoteTaskIdIt);
        throw;
      }
      VLOG(3) << "@" << taskId_
              << " Added remote split for task: " << remoteTaskId;
    }
  }

  // Outside of lock.
  if (toClose) {
    toClose->close();
    return;
  }

  try {
    toStart->start();
  } catch (...) {
    // The source was already counted in the queue. close() delivers its end
    // marker even if Communicator registration failed partway through.
    try {
      toStart->close();
    } catch (...) {
      try {
        queue_->setError("Failed to start or close UCX exchange source");
      } catch (...) {
        LOG(ERROR) << "Failed to start or close UCX exchange source and to "
                      "record the queue error";
      }
    }
    throw;
  }
}

void UcxExchangeClient::noMoreRemoteTasks() {
  VLOG(3) << "@" << taskId_ << " UcxExchangeClient::noMoreRemoteTasks called.";
  queue_->noMoreSources();
}

void UcxExchangeClient::close() {
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    closed_ = true;
  }

  // Outside of mutex. Keep the stable source list until client destruction so
  // a repeated close (including the destructor after Task shutdown) retries a
  // cleanup handoff that may have failed under transient host allocation
  // pressure. addRemoteTaskId() cannot mutate sources_ after closed_ is set.
  for (auto& source : sources_) {
    source->close();
  }
  queue_->close();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxExchangeClient::stats() const {
  // TODO: Implement stats collection.
  folly::F14FastMap<std::string, RuntimeMetric> stats;
  return stats;
}

PackedTableWithStreamPtr
UcxExchangeClient::next(int consumerId, bool* atEnd, ContinueFuture* future) {
  VLOG(3) << "@" << taskId_ << " UcxExchangeClient::next called for consumerId "
          << consumerId;
  PackedTableWithStreamPtr data;
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  std::vector<std::shared_ptr<UcxExchangeSource>> sourcesToResume;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (queue_->isInError()) {
      // A terminal receive error must not be masked as clean EOS merely
      // because failReceive() also closed this client.
      return queue_->dequeueLocked(consumerId, atEnd, future, &stalePromise);
    }
    if (closed_) {
      *atEnd = true;
      return data;
    }

    // not closed, get packed table from the queue.
    *atEnd = false;
    data = queue_->dequeueLocked(consumerId, atEnd, future, &stalePromise);
    if (*atEnd) {
      // queue is closed!
      return data;
    }

    // TODO: Review this primitive form of flow control.
    // Maybe need to inspect the #bytes rather than the #tables?
    // Don't request more data when queue size exceeds the configured limit.
    // NOTE: This check is currently a no-op because the UCX exchange is
    // push-based — there is no mechanism to "request" or "not request" more
    // data. The server pushes unconditionally. Real backpressure is
    // implemented in UcxExchangeSource::process() (ReadyToReceive state).
    if (data != nullptr && queue_->size() > maxQueuedColumns_) {
      if (!inFlowControl_) {
        inFlowControl_ = true;
        VLOG(1) << "[FLOW-CTRL] @" << taskId_ << " consumer=" << consumerId
                << " entering flow control"
                << " queueSize=" << queue_->size()
                << " maxQueued=" << maxQueuedColumns_;
      }
      return data;
    } else if (inFlowControl_ && data != nullptr) {
      inFlowControl_ = false;
      VLOG(1) << "[FLOW-CTRL] @" << taskId_ << " consumer=" << consumerId
              << " leaving flow control"
              << " queueSize=" << queue_->size()
              << " maxQueued=" << maxQueuedColumns_;
    }

    // Per-stage progress counters.
    if (data != nullptr) {
      ++totalDequeued_;
      if (totalDequeued_ % 1000 == 0) {
        VLOG(1) << "[PROGRESS] @" << taskId_ << " consumer=" << consumerId
                << " dequeued=" << totalDequeued_
                << " queueSize=" << queue_->size()
                << " queueBytes=" << queue_->totalBytes();
      }
    }

    // Collect sources that need resuming while holding the lock. Sources
    // publish their dormant state under this same mutex, making the decision
    // atomic with respect to dequeue and preventing a lost wake-up.
    // We call resumeFromBackpressure() outside the lock to avoid a
    // lock-ordering hazard: it acquires WorkQueue::mutex_ via
    // addToWorkQueue(), and holding queue_->mutex_ here would impose
    // queue_->mutex_ → WorkQueue::mutex_ ordering.
    const bool shouldResume =
        queue_->size() <= UcxExchangeSource::kBackpressureLowWaterMark &&
        queue_->receiveCanPrefetchLocked();
    if (shouldResume) {
      sourcesToResume.assign(sources_.begin(), sources_.end());
    }
  }

  // Outside of lock: resume backpressured sources and fulfill stale promise.
  for (auto& source : sourcesToResume) {
    source->resumeFromBackpressure();
  }
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  return data;
}

void UcxExchangeClient::releaseInFlightReceiveBytes(uint64_t bytes) {
  std::vector<std::shared_ptr<UcxExchangeSource>> sourcesToResume;
  {
    std::lock_guard<std::mutex> lock(queue_->mutex());
    queue_->releaseInFlightReceiveBytesLocked(bytes);
    const bool shouldResume = !closed_ &&
        queue_->size() <= UcxExchangeSource::kBackpressureLowWaterMark &&
        queue_->receiveCanPrefetchLocked();
    if (shouldResume) {
      sourcesToResume.assign(sources_.begin(), sources_.end());
    }
  }

  // addToWorkQueue() takes its own mutex, so never call it while holding the
  // exchange queue mutex.
  for (auto& source : sourcesToResume) {
    source->resumeFromBackpressure();
  }
}

void UcxExchangeClient::failReceive(std::string_view error) noexcept {
  try {
    queue_->setError(error);
  } catch (...) {
    LOG(ERROR) << "Failed to record terminal UCX receive error";
  }

  try {
    close();
  } catch (...) {
    LOG(ERROR) << "Failed to close UCX exchange client after terminal receive "
                  "error";
  }
}

UcxExchangeClient::~UcxExchangeClient() {
  close();
}

std::string UcxExchangeClient::toString() const {
  std::stringstream out;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    for (auto& source : sources_) {
      out << source->toString() << std::endl;
    }
  }
  return out.str();
}

folly::dynamic UcxExchangeClient::toJson() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["taskId"] = taskId_;
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    obj["closed"] = closed_;
    for (auto& source : sources_) {
      clientsObj[std::to_string(index++)] = source->toJson();
    }
  }
  obj["clients"] = clientsObj;
  return obj;
}

} // namespace facebook::velox::ucx_exchange
