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

void UcxExchangeClient::addRemoteTaskId(std::string_view remoteTaskId) {
  std::shared_ptr<UcxExchangeSource> toClose;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    bool duplicate = !remoteTaskIds_.insert(std::string{remoteTaskId}).second;
    if (duplicate) {
      // Do not add sources twice. Presto protocol may add duplicate sources
      // and the task updates have no guarantees of arriving in order.
      return;
    }

    std::shared_ptr<UcxExchangeSource> source;
    source = UcxExchangeSource::create(taskId_, remoteTaskId, queue_);

    if (closed_) {
      toClose = std::move(source);
    } else {
      sources_.push_back(source);
      queue_->addSourceLocked();
      source->setRegistered();
      VLOG(3) << "@" << taskId_
              << " Added remote split for task: " << remoteTaskId;
    }
  }

  // Outside of lock.
  if (toClose) {
    toClose->close();
  }
}

void UcxExchangeClient::noMoreRemoteTasks() {
  VLOG(3) << "@" << taskId_ << " UcxExchangeClient::noMoreRemoteTasks called.";
  queue_->noMoreSources();
}

void UcxExchangeClient::close() {
  std::vector<std::shared_ptr<UcxExchangeSource>> sources;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return;
    }
    closed_ = true;
    sources = std::move(sources_);
  }

  // Outside of mutex.
  for (auto& source : sources) {
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

    // Collect sources that need resuming while holding the lock.
    // We call resumeFromBackpressure() outside the lock to avoid a
    // lock-ordering hazard: it acquires WorkQueue::mutex_ via
    // addToWorkQueue(), and holding queue_->mutex_ here would impose
    // queue_->mutex_ → WorkQueue::mutex_ ordering.
    if (data != nullptr &&
        queue_->size() <= UcxExchangeSource::kBackpressureLowWaterMark) {
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
  obj["closed"] = closed_;
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    for (auto& source : sources_) {
      clientsObj[std::to_string(index++)] = source->toJson();
    }
  }
  obj["clients"] = clientsObj;
  return obj;
}

} // namespace facebook::velox::ucx_exchange
