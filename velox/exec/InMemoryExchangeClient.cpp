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
#include "velox/exec/InMemoryExchangeClient.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/Merge.h"

namespace facebook::velox::exec {

InMemoryExchangeClient::InMemoryExchangeClient(
    std::string taskId,
    int destination,
    int64_t maxQueuedBytes,
    int32_t numberOfConsumers,
    uint64_t minOutputBatchBytes,
    memory::MemoryPool* pool,
    folly::Executor* executor,
    int32_t requestDataSizesMaxWaitSec,
    bool skipRequestDataSizeWithSingleSource,
    bool lazyFetching)
    : taskId_{std::move(taskId)},
      destination_(destination),
      maxQueuedBytes_{maxQueuedBytes},
      requestDataSizesMaxWaitSec_{requestDataSizesMaxWaitSec},
      pool_(pool),
      executor_(executor),
      queue_(
          std::make_shared<ExchangeQueue>(
              numberOfConsumers,
              minOutputBatchBytes)),
      // See comment in 'pickSourcesToRequestLocked' for why this is needed
      // for 'minOutputBatchBytes_'. Note: ExchangeQueue does not need max(1,
      // minOutputBatchBytes) because for 'MergeExchangeSource', we want
      // ExchangeQueue 'minOutputBatchBytes' to be 0 so that it always
      // unblocks. In short, 0 has a special meaning for ExchangeQueue
      minOutputBatchBytes_(
          std::max(static_cast<uint64_t>(1), minOutputBatchBytes)),
      skipRequestDataSizeWithSingleSource_(skipRequestDataSizeWithSingleSource),
      lazyFetching_(lazyFetching) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_NOT_NULL(executor_);
  // NOTE: the executor is used to run async response callback from the
  // exchange source. The provided executor must not be
  // folly::InlineLikeExecutor, otherwise it might cause potential deadlock as
  // the response callback in exchange client might call back into the
  // exchange source under uncertain execution context. For instance, the
  // exchange client might inline close the exchange source from a background
  // thread of the exchange source, and the close needs to wait for this
  // background thread to complete first.
  VELOX_CHECK_NULL(dynamic_cast<const folly::InlineLikeExecutor*>(executor_));
  VELOX_CHECK_GE(
      destination, 0, "Exchange client destination must not be negative");
}

// static
std::shared_ptr<ExchangeTransportEntry>
InMemoryExchangeClient::makeDefaultTransportEntry() {
  return ExchangeTransportEntry::make<InMemoryExchangeClient>(
      [](const ExchangeClientContext& context) {
        // The two byte limits come from the context, not from 'queryConfig':
        // the caller may be sizing a per-source merge budget rather than a
        // whole-node one.
        const auto& queryConfig = context.queryConfig;
        return std::make_shared<InMemoryExchangeClient>(
            context.taskId,
            context.destination,
            context.maxExchangeBufferSize,
            context.numberOfConsumers,
            context.minExchangeOutputBatchBytes,
            context.pool,
            context.executor,
            queryConfig.requestDataSizesMaxWaitSec(),
            queryConfig.singleSourceExchangeOptimizationEnabled(),
            queryConfig.exchangeLazyFetchingEnabled());
      },
      [](int32_t operatorId,
         DriverCtx* ctx,
         const std::shared_ptr<const core::ExchangeNode>& node,
         const std::shared_ptr<InMemoryExchangeClient>& client)
          -> std::unique_ptr<Operator> {
        return std::make_unique<Exchange>(operatorId, ctx, node, client);
      },
      [](int32_t operatorId,
         DriverCtx* ctx,
         const std::shared_ptr<const core::ExchangeNode>& node,
         const std::shared_ptr<InMemoryExchangeClient>& /*client*/)
          -> std::unique_ptr<Operator> {
        const auto mergeExchangeNode =
            std::dynamic_pointer_cast<const core::MergeExchangeNode>(node);
        VELOX_CHECK_NOT_NULL(
            mergeExchangeNode,
            "MergeExchange requires a MergeExchangeNode, plan node: {}",
            node->id());
        return std::make_unique<MergeExchange>(
            operatorId, ctx, mergeExchangeNode);
      });
}

void InMemoryExchangeClient::addRemoteTaskId(const std::string& remoteTaskId) {
  std::vector<RequestSpec> requestSpecs;
  std::shared_ptr<ExchangeSource> toClose;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    bool duplicate = !remoteTaskIds_.insert(remoteTaskId).second;
    if (duplicate) {
      // Do not add sources twice. Presto protocol may add duplicate sources
      // and the task updates have no guarantees of arriving in order.
      return;
    }

    std::shared_ptr<ExchangeSource> source;
    try {
      source =
          ExchangeSource::create(remoteTaskId, destination_, queue_, pool_);
    } catch (const VeloxException&) {
      throw;
    } catch (const std::exception& e) {
      // 'remoteTaskId' can be very long. Truncate to 128 characters.
      VELOX_FAIL(
          "Failed to create ExchangeSource: {}. Task ID: {}.",
          e.what(),
          remoteTaskId.substr(0, 128));
    }

    if (closed_) {
      toClose = std::move(source);
    } else {
      sources_.push_back(source);
      queue_->addSourceLocked();
      emptySources_.push(source);
      // When lazyFetching_ is true, I/O will be triggered lazily when next() is
      // called from Exchange::isBlocked(). This allows waiter tasks using
      // cached hash tables to skip I/O entirely when the table is already
      // cached - the HashBuild operator will finish before
      // Exchange::isBlocked() is ever called, so no unnecessary data fetching
      // occurs.
      if (!lazyFetching_) {
        // Start fetching data immediately.
        requestSpecs = pickSourcesToRequestLocked();
      }
    }
  }

  // Outside of lock.
  if (toClose) {
    toClose->close();
  } else {
    request(std::move(requestSpecs));
  }
}

void InMemoryExchangeClient::noMoreRemoteTasks() {
  queue_->noMoreSources();
}

void InMemoryExchangeClient::close() {
  std::vector<std::shared_ptr<ExchangeSource>> sources;
  std::queue<ProducingSource> producingSources;
  std::queue<std::shared_ptr<ExchangeSource>> emptySources;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      return;
    }

    // Capture stats BEFORE clearing sources_.
    // This allows stats() to return meaningful data even after close().
    stats_ = collectStatsLocked();

    closed_ = true;
    sources = std::move(sources_);
    producingSources = std::move(producingSources_);
    emptySources = std::move(emptySources_);
  }

  // Outside of mutex.
  for (auto& source : sources) {
    source->close();
  }
  queue_->close();
}

folly::F14FastMap<std::string, RuntimeMetric> InMemoryExchangeClient::stats() {
  std::lock_guard<std::mutex> l(queue_->mutex());
  if (stats_.empty()) {
    stats_ = collectStatsLocked();
  }
  return stats_;
}

folly::F14FastMap<std::string, RuntimeMetric>
InMemoryExchangeClient::collectStatsLocked() const {
  folly::F14FastMap<std::string, RuntimeMetric> stats;

  for (const auto& source : sources_) {
    if (source->supportsMetrics()) {
      for (const auto& [name, value] : source->metrics()) {
        auto [iter, inserted] = stats.try_emplace(name, value.unit);
        iter->second.merge(value);
      }
    } else {
      for (const auto& [name, value] : source->stats()) {
        auto [iter, inserted] = stats.try_emplace(name);
        iter->second.addValue(value);
      }
    }
  }

  stats.insert_or_assign(
      "peakBytes",
      RuntimeMetric(queue_->peakBytes(), RuntimeCounter::Unit::kBytes));
  stats.insert_or_assign(
      "numReceivedPages", RuntimeMetric(queue_->receivedPages()));
  stats.insert_or_assign(
      "averageReceivedPageBytes",
      RuntimeMetric(
          queue_->averageReceivedPageBytes(), RuntimeCounter::Unit::kBytes));

  return stats;
}

std::vector<std::unique_ptr<SerializedPageBase>> InMemoryExchangeClient::next(
    int consumerId,
    uint32_t maxBytes,
    bool* atEnd,
    ContinueFuture* future) {
  std::vector<RequestSpec> requestSpecs;
  std::vector<std::unique_ptr<SerializedPageBase>> pages;
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  {
    std::lock_guard<std::mutex> l(queue_->mutex());
    if (closed_) {
      *atEnd = true;
      return pages;
    }

    *atEnd = false;
    pages = queue_->dequeueLocked(
        consumerId, maxBytes, atEnd, future, &stalePromise);
    if (*atEnd) {
      return pages;
    }

    if (!pages.empty() && queue_->totalBytes() > maxQueuedBytes_) {
      return pages;
    }

    requestSpecs = pickSourcesToRequestLocked();
  }

  // Outside of lock
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  request(std::move(requestSpecs));
  return pages;
}

void InMemoryExchangeClient::request(std::vector<RequestSpec>&& requestSpecs) {
  auto self = shared_from_this();
  for (auto& spec : requestSpecs) {
    auto future = folly::SemiFuture<ExchangeSource::Response>::makeEmpty();
    if (spec.maxBytes == 0) {
      future = spec.source->requestDataSizes(requestDataSizesMaxWaitSec_);
    } else {
      future = spec.source->request(spec.maxBytes, kRequestDataMaxWait);
    }
    VELOX_CHECK(future.valid());
    std::move(future)
        .via(executor_)
        .thenValue(
            [self, spec = std::move(spec), sendTimeMs = getCurrentTimeMs()](
                ExchangeSource::Response&& response) {
              const auto requestTimeMs = getCurrentTimeMs() - sendTimeMs;
              if (spec.maxBytes == 0) {
                RECORD_HISTOGRAM_METRIC_VALUE(
                    kMetricExchangeDataSizeTimeMs, requestTimeMs);
                RECORD_METRIC_VALUE(kMetricExchangeDataSizeCount);
              } else {
                RECORD_HISTOGRAM_METRIC_VALUE(
                    kMetricExchangeDataTimeMs, requestTimeMs);
                RECORD_METRIC_VALUE(kMetricExchangeDataBytes, response.bytes);
                RECORD_HISTOGRAM_METRIC_VALUE(
                    kMetricExchangeDataSize, response.bytes);
                RECORD_METRIC_VALUE(kMetricExchangeDataCount);
              }

              bool pauseCurrentSource{false};
              std::vector<RequestSpec> requestSpecs;
              std::shared_ptr<ExchangeSource> currentSource = spec.source;
              {
                std::lock_guard<std::mutex> l(self->queue_->mutex());
                if (self->closed_) {
                  return;
                }
                if (!response.atEnd) {
                  if (!response.remainingBytes.empty()) {
                    for (auto bytes : response.remainingBytes) {
                      VELOX_CHECK_GT(bytes, 0);
                    }
                    self->producingSources_.push(
                        {std::move(spec.source),
                         std::move(response.remainingBytes)});
                  } else {
                    self->emptySources_.push(std::move(spec.source));
                  }
                }
                self->totalPendingBytes_ -= spec.maxBytes;
                requestSpecs = self->pickSourcesToRequestLocked();
                pauseCurrentSource =
                    std::find_if(
                        requestSpecs.begin(),
                        requestSpecs.end(),
                        [&currentSource](const RequestSpec& spec) -> bool {
                          return spec.source.get() == currentSource.get();
                        }) == requestSpecs.end();
              }
              if (pauseCurrentSource) {
                currentSource->pause();
              }
              self->request(std::move(requestSpecs));
            })
        .thenError(
            folly::tag_t<std::exception>{}, [self](const std::exception& e) {
              self->queue_->setError(e.what());
            });
  }
}

std::vector<InMemoryExchangeClient::RequestSpec>
InMemoryExchangeClient::pickSourcesToRequestLocked() {
  if (closed_) {
    return {};
  }
  if (skipRequestDataSizeWithSingleSource()) {
    return pickupSingleSourceToRequestLocked();
  }
  std::vector<RequestSpec> requestSpecs;
  while (!emptySources_.empty()) {
    auto& source = emptySources_.front();
    VELOX_CHECK(source->shouldRequestLocked());
    requestSpecs.push_back({std::move(source), 0});
    emptySources_.pop();
  }
  int64_t availableSpace =
      maxQueuedBytes_ - queue_->totalBytes() - totalPendingBytes_;
  while (availableSpace > 0 && !producingSources_.empty()) {
    auto& source = producingSources_.front().source;
    int64_t requestBytes = 0;
    for (auto bytes : producingSources_.front().remainingBytes) {
      availableSpace -= bytes;
      if (availableSpace < 0) {
        break;
      }
      requestBytes += bytes;
    }
    if (requestBytes == 0) {
      VELOX_CHECK_LT(availableSpace, 0);
      break;
    }
    VELOX_CHECK(source->shouldRequestLocked());
    requestSpecs.push_back({std::move(source), requestBytes});
    producingSources_.pop();
    totalPendingBytes_ += requestBytes;
  }

  if ((queue_->totalBytes() + totalPendingBytes_ < minOutputBatchBytes_) &&
      !producingSources_.empty()) {
    // Two cases which we request an out-of-band data transfer:
    // 1. We have full capacity but still cannot initiate one single data
    //    transfer. Let the transfer happen in this case to avoid getting stuck.
    //
    // 2. We have some data in the queue that is not big enough for consumers,
    //    and it is big enough to not allow ExchangeClient to initiate request
    //    for more data. Let transfer happen in this case to avoid this deadlock
    //    situation.
    auto& source = producingSources_.front().source;
    auto requestBytes = producingSources_.front().remainingBytes.at(0);
    LOG(INFO) << "Requesting large single page " << requestBytes
              << " bytes, exceeding capacity " << maxQueuedBytes_;
    VELOX_CHECK(source->shouldRequestLocked());
    requestSpecs.push_back({std::move(source), requestBytes});
    producingSources_.pop();
    totalPendingBytes_ += requestBytes;
  }
  return requestSpecs;
}

std::vector<InMemoryExchangeClient::RequestSpec>
InMemoryExchangeClient::pickupSingleSourceToRequestLocked() {
  VELOX_CHECK_EQ(sources_.size(), 1);
  VELOX_CHECK(!closed_);
  if (emptySources_.empty() && producingSources_.empty()) {
    return {};
  }

  VELOX_CHECK_EQ(totalPendingBytes_, 0);
  VELOX_CHECK_LE(!!emptySources_.empty() + !!producingSources_.empty(), 1);
  const auto requestBytes = maxQueuedBytes_ - queue_->totalBytes();

  if (requestBytes <= 0) {
    return {};
  }
  std::vector<RequestSpec> requestSpecs;
  SCOPE_EXIT {
    totalPendingBytes_ += requestBytes;
  };
  if (!emptySources_.empty()) {
    VELOX_CHECK_EQ(emptySources_.size(), 1);
    auto& source = emptySources_.front();
    VELOX_CHECK(source->shouldRequestLocked());
    requestSpecs.push_back({std::move(source), requestBytes});
    emptySources_.pop();
    return requestSpecs;
  }

  VELOX_CHECK_EQ(producingSources_.size(), 1);
  auto& source = producingSources_.front().source;
  VELOX_CHECK(source->shouldRequestLocked());
  VELOX_CHECK(!producingSources_.front().remainingBytes.empty());
  requestSpecs.push_back({std::move(source), requestBytes});
  producingSources_.pop();
  return requestSpecs;
}

InMemoryExchangeClient::~InMemoryExchangeClient() {
  close();
}

std::string InMemoryExchangeClient::toString() const {
  std::stringstream out;
  for (auto& source : sources_) {
    out << source->toString() << std::endl;
  }
  return out.str();
}

folly::dynamic InMemoryExchangeClient::toJson() const {
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

} // namespace facebook::velox::exec
