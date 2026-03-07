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
#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"

namespace facebook::velox::cudf_exchange {

void CudfExchangeClient::addRemoteTaskId(const std::string& remoteTaskId) {
  std::shared_ptr<CudfExchangeSource> toClose;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    bool duplicate = !remoteTaskIds_.insert(remoteTaskId).second;
    if (duplicate) {
      // Do not add sources twice. Presto protocol may add duplicate sources
      // and the task updates have no guarantees of arriving in order.
      return;
    }

    std::shared_ptr<CudfExchangeSource> source;
    source = CudfExchangeSource::create(taskId_, remoteTaskId, queue_);

    if (closed_) {
      toClose = std::move(source);
    } else {
      sources_.push_back(source);
      queue_->addSourceLocked();
      VLOG(3) << "@" << taskId_
              << " Added remote split for task: " << remoteTaskId;
    }
  }

  // Outside of lock.
  if (toClose) {
    toClose->close();
  }
}

void CudfExchangeClient::noMoreRemoteTasks() {
  VLOG(3) << "@" << taskId_ << " CudfExchangeClient::noMoreRemoteTasks called.";
  queue_->noMoreSources();
}

void CudfExchangeClient::close() {
  std::vector<std::shared_ptr<CudfExchangeSource>> sources;
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

folly::F14FastMap<std::string, RuntimeMetric> CudfExchangeClient::stats()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> stats;
#if 0
  std::lock_guard<std::mutex> l(queue_->mutex());

  for (const auto& source : sources_) {
    if (source->supportsMetrics()) {
      for (const auto& [name, value] : source->metrics()) {
        if (UNLIKELY(stats.count(name) == 0)) {
          stats.insert(std::pair(name, RuntimeMetric(value.unit)));
        }
        stats[name].merge(value);
      }
    } else {
      for (const auto& [name, value] : source->stats()) {
        stats[name].addValue(value);
      }
    }
  }

  stats["peakBytes"] =
      RuntimeMetric(queue_->peakBytes(), RuntimeCounter::Unit::kBytes);
  stats["numReceivedPages"] = RuntimeMetric(queue_->receivedPages());
  stats["averageReceivedPageBytes"] = RuntimeMetric(
      queue_->averageReceivedPageBytes(), RuntimeCounter::Unit::kBytes);
#endif
  return stats;
}

PackedTableWithStreamPtr
CudfExchangeClient::next(int consumerId, bool* atEnd, ContinueFuture* future) {
  VLOG(3) << "@" << taskId_
          << " CudfExchangeClient::next called for consumerId " << consumerId;
  PackedTableWithStreamPtr data;
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
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
    if (data != nullptr && queue_->size() > maxQueuedColumns_) {
      return data;
    }
  }

  // Outside of lock
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  return data;
}

CudfExchangeClient::~CudfExchangeClient() {
  close();
}

std::string CudfExchangeClient::toString() const {
  std::stringstream out;
  for (auto& source : sources_) {
    out << source->toString() << std::endl;
  }
  return out.str();
}

folly::dynamic CudfExchangeClient::toJson() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["taskId"] = taskId_;
  obj["closed"] = closed_;
  folly::dynamic clientsObj = folly::dynamic::object;
  int index = 0;
  for (auto& source : sources_) {
    clientsObj[std::to_string(index++)] = source->toJson();
  }
  obj["clients"] = clientsObj;
  return obj;
}

} // namespace facebook::velox::cudf_exchange
