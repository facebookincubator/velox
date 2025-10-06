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
  std::vector<std::shared_ptr<CudfExchangeSource>> requestSpecs;
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
    source = CudfExchangeSource::create(remoteTaskId, queue_);

    if (closed_) {
      toClose = std::move(source);
    } else {
      sources_.push_back(source);
      queue_->addSourceLocked();
      readySources_.push(source);
      VLOG(3) << "Added remote split for task: " << remoteTaskId;
      requestSpecs = pickSourcesToRequestLocked();
    }
  }

  // Outside of lock.
  if (toClose) {
    toClose->close();
  } else {
    request(std::move(requestSpecs));
  }
}

void CudfExchangeClient::noMoreRemoteTasks() {
  VLOG(3) << "CudfExchangeClient::noMoreRemoteTasks called.";
  queue_->noMoreSources();
}

void CudfExchangeClient::close() {
  std::vector<std::shared_ptr<CudfExchangeSource>> sources;
  std::queue<std::shared_ptr<CudfExchangeSource>> readySources;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return;
    }
    closed_ = true;
    sources = std::move(sources_);
    readySources = std::move(readySources_);
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

std::unique_ptr<cudf::packed_columns>
CudfExchangeClient::next(int consumerId, bool* atEnd, ContinueFuture* future) {
  VLOG(3) << "CudfExchangeClient::next called.";
  std::vector<std::shared_ptr<CudfExchangeSource>> requestSpecs;
  std::unique_ptr<cudf::packed_columns> columns;
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      *atEnd = true;
      return columns;
    }

    // not closed, get packed columns from the queue.
    *atEnd = false;
    columns = queue_->dequeueLocked(consumerId, atEnd, future, &stalePromise);
    if (*atEnd) {
      // queue is closed!
      return columns;
    }

    // TODO: Review this primitive form of flow control.
    // Maybe need to inspect the #bytes rather than the #columns?
    // Don't request more data when queue size exceeds the configured limit.
    if (columns != nullptr && queue_->size() > maxQueuedColumns_) {
      return columns;
    }

    requestSpecs = pickSourcesToRequestLocked();
  }

  // Outside of lock
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  // not at end and still room in the queue: request more data from all
  // eligible sources.
  request(std::move(requestSpecs));
  return columns;
}

void CudfExchangeClient::request(
    std::vector<std::shared_ptr<CudfExchangeSource>>&& sources) {
  auto self = shared_from_this();
  for (auto& source : sources) {
    VLOG(3) << "Requesting data from source: " << source->toString();
    auto future = folly::SemiFuture<ExchangeSource::Response>::makeEmpty();
    future = source->request(kRequestDataMaxWait);
    VELOX_CHECK(future.valid());
    std::move(future)
        .via(executor_)
        .thenValue(
            [self, source = std::move(source), sendTimeMs = getCurrentTimeMs()](
                ExchangeSource::Response&& response) {
              const auto requestTimeMs = getCurrentTimeMs() - sendTimeMs;
              RECORD_HISTOGRAM_METRIC_VALUE(
                  kMetricExchangeDataTimeMs, requestTimeMs);
              RECORD_METRIC_VALUE(kMetricExchangeDataBytes, response.bytes);
              RECORD_HISTOGRAM_METRIC_VALUE(
                  kMetricExchangeDataSize, response.bytes);
              RECORD_METRIC_VALUE(kMetricExchangeDataCount);

              VLOG(3) << "Received data from source: " << source->toString();

              std::vector<std::shared_ptr<CudfExchangeSource>> requestSpecs;
              {
                std::lock_guard<std::mutex> l(self->queue_->mutex());
                if (self->closed_) {
                  return;
                }
                // add source back into the ready queue when not at end.
                if (!response.atEnd) {
                  self->readySources_.push(std::move(source));
                  requestSpecs = self->pickSourcesToRequestLocked();
                }
              }
              self->request(std::move(requestSpecs));
            })
        .thenError(
            folly::tag_t<std::exception>{}, [self](const std::exception& e) {
              self->queue_->setError(e.what());
            });
  }
}

std::vector<std::shared_ptr<CudfExchangeSource>>
CudfExchangeClient::pickSourcesToRequestLocked() {
  if (closed_) {
    return {};
  }
  std::vector<std::shared_ptr<CudfExchangeSource>> requestSpecs;
  while (!readySources_.empty()) {
    auto& source = readySources_.front();
    VELOX_CHECK(source->shouldRequestLocked());
    requestSpecs.push_back(std::move(source));
    readySources_.pop();
  }

  return requestSpecs;
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
