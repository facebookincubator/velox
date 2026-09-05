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

#include "velox/exec/ExchangeClient.h"
#include "velox/exec/ExchangeQueue.h"
#include "velox/exec/ExchangeSource.h"

namespace facebook::velox::exec {

struct ExchangeTransportEntry;

/// Handle for a set of producers reached through ExchangeSource, buffering
/// their pages in an in-memory ExchangeQueue. This may be shared by multiple
/// Exchange operators, one per consumer thread.
///
/// This is the client of the built-in in-memory transport
/// (core::TransportKind::kInMemory). Its data plane -- next() and queue() -- is
/// not part of the ExchangeClient interface: the stock Exchange operator is
/// registered together with this client and therefore reaches it directly.
class InMemoryExchangeClient
    : public ExchangeClient,
      public std::enable_shared_from_this<InMemoryExchangeClient> {
 public:
  static constexpr int32_t kDefaultMaxQueuedBytes = 32 << 20; // 32 MB.
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};

  /// @param taskId Id of the consuming task, for logging.
  /// @param destination Index of the output buffer to fetch from producers.
  /// @param maxQueuedBytes Soft limit on bytes buffered in the queue.
  /// @param numberOfConsumers Number of Exchange operators sharing this client.
  /// @param minOutputBatchBytes Minimum bytes to accumulate in the queue before
  /// unblocking a consumer. 0 unblocks as soon as any page arrives.
  /// @param pool Memory pool the received pages are allocated from.
  /// @param executor Executor running the exchange sources' response callbacks.
  /// Must not be a folly::InlineLikeExecutor.
  /// @param requestDataSizesMaxWaitSec Max wait for a data-size request.
  /// @param skipRequestDataSizeWithSingleSource If true, skips the data-size
  /// round trip when there is exactly one source.
  /// @param lazyFetching If true, defers fetching until next() is called
  /// instead of starting when a remote task is added.
  InMemoryExchangeClient(
      std::string taskId,
      int destination,
      int64_t maxQueuedBytes,
      int32_t numberOfConsumers,
      uint64_t minOutputBatchBytes,
      memory::MemoryPool* pool,
      folly::Executor* executor,
      int32_t requestDataSizesMaxWaitSec = 10,
      bool skipRequestDataSizeWithSingleSource = false,
      bool lazyFetching = false);

  ~InMemoryExchangeClient() override;

  /// Builds the registry entry for the built-in in-memory transport, pairing
  /// this client with the stock Exchange and MergeExchange operators.
  /// ExchangeTransportRegistry::global() registers it under kInMemory.
  static std::shared_ptr<ExchangeTransportEntry> makeDefaultTransportEntry();

  /// Memory pool the received pages are allocated from.
  memory::MemoryPool* pool() const {
    return pool_;
  }

  void addRemoteTaskId(const std::string& remoteTaskId) override;

  void noMoreRemoteTasks() override;

  void close() override;

  folly::F14FastMap<std::string, RuntimeMetric> stats() override;

  std::string toString() const override;

  folly::dynamic toJson() const override;

  /// Queue the received pages are buffered in. Part of the in-memory data
  /// plane, used by the Exchange operator and by MergeExchangeSource.
  const std::shared_ptr<ExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns up to 'maxBytes' pages of data, but no less than one.
  ///
  /// If no data is available returns empty list and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  ///
  /// The data may be compressed, in which case 'maxBytes' applies to compressed
  /// size.
  std::vector<std::unique_ptr<SerializedPageBase>>
  next(int consumerId, uint32_t maxBytes, bool* atEnd, ContinueFuture* future);

  /// Max wait for a data-size request to a producer.
  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return requestDataSizesMaxWaitSec_;
  }

  /// Ids of the upstream tasks added so far.
  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  struct RequestSpec {
    std::shared_ptr<ExchangeSource> source;

    // How much bytes to request from this source.  0 bytes means request data
    // sizes only.
    int64_t maxBytes;
  };

  struct ProducingSource {
    std::shared_ptr<ExchangeSource> source;
    std::vector<int64_t> remainingBytes;
  };

  // Selects exchange sources to request data from based on available queue
  // capacity. Handles multiple sources by first requesting data sizes from all
  // empty sources, then requesting actual data from producing sources based on
  // their remaining bytes and available capacity. May initiate out-of-band
  // transfers for large pages that exceed capacity to avoid deadlock
  // situations. For single source case, delegates to
  // pickupSingleSourceToRequestLocked which sets max request bytes based on
  // available queue space instead of reported remaining bytes from exchange
  // sources.
  std::vector<RequestSpec> pickSourcesToRequestLocked();

  // Specialized single-source request picker for single-source exchange
  // clients. Sets the max request bytes based on available space in the queue
  // rather than the reported remaining bytes from exchange sources. The reason
  // is that single source has no other alternative so just fetch as much as
  // possible from that source. Returns a request spec for the single source
  // when there is available capacity in the queue and no pending requests. If
  // capacity is unavailable or requests are already pending, returns empty
  // vector.
  std::vector<RequestSpec> pickupSingleSourceToRequestLocked();
  void request(std::vector<RequestSpec>&& requestSpecs);

  // Returns true if skip request data size optimization is enabled for single
  // source exchanges.
  bool skipRequestDataSizeWithSingleSource() const {
    return skipRequestDataSizeWithSingleSource_ && queue_->hasNoMoreSources() &&
        sources_.size() == 1;
  }

  folly::F14FastMap<std::string, RuntimeMetric> collectStatsLocked() const;

  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int64_t maxQueuedBytes_;
  const std::chrono::seconds requestDataSizesMaxWaitSec_;

  memory::MemoryPool* const pool_;
  folly::Executor* const executor_;
  const std::shared_ptr<ExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<ExchangeSource>> sources_;
  bool closed_{false};

  folly::F14FastMap<std::string, RuntimeMetric> stats_;

  // The minimum byte size the consumer is expected to consume from
  // the exchange queue.
  const uint64_t minOutputBatchBytes_;

  // Enable single source exchange optimization query config flag
  // when there is only one exchange source.
  const bool skipRequestDataSizeWithSingleSource_;

  // If true, defer fetching until next() is called.
  // If false (default), start fetching data immediately when remote tasks are
  // added.
  const bool lazyFetching_;

  // Total number of bytes in flight.
  int64_t totalPendingBytes_{0};

  // A queue of sources that have returned non-empty response from the latest
  // request.
  std::queue<ProducingSource> producingSources_;
  // A queue of sources that returned empty response from the latest request.
  std::queue<std::shared_ptr<ExchangeSource>> emptySources_;
};

} // namespace facebook::velox::exec
