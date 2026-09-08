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
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

namespace facebook::velox::ucx_exchange {

/// Handle for a set of producers reached over UCX, buffering the
/// cudf::packed_columns they push in a UcxExchangeQueue. This may be shared by
/// multiple UcxExchange operators, one per consumer thread.
///
/// This is the client of the UCX transport (core::TransportKind::kUcx). Its
/// data plane -- next() and queue() -- is deliberately not part of the
/// exec::ExchangeClient interface: the payloads are GPU buffers rather than
/// serialized pages. The UcxExchange operator is registered together with this
/// client in exec::ExchangeTransportRegistry, so it always knows the concrete
/// client type and reaches the data plane directly.
class UcxExchangeClient
    : public exec::ExchangeClient,
      public std::enable_shared_from_this<UcxExchangeClient> {
 public:
  /// Flow control limit on the number of elements buffered in the queue.
  static constexpr int32_t kDefaultMaxQueuedColumns = 32;

  /// Max wait for a batch of data to accumulate in the queue.
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};

  /// @param taskId Id of the consuming task, for logging.
  /// @param destination Index of the partition to fetch from the producers.
  /// @param numberOfConsumers Number of UcxExchange operators sharing this
  /// client.
  /// @param requestDataSizesMaxWaitSec Max wait for a data-size request.
  UcxExchangeClient(
      std::string taskId,
      int destination,
      int32_t numberOfConsumers,
      int32_t requestDataSizesMaxWaitSec = 10);

  ~UcxExchangeClient() override;

  /// Creates a UCX exchange source and starts fetching data from the upstream
  /// task identified by 'remoteTaskId'. If close() has been called already,
  /// creates an exchange source and immediately closes it to notify the
  /// upstream task that its data is no longer needed. Repeated calls with the
  /// same 'remoteTaskId' are ignored.
  void addRemoteTaskId(const std::string& remoteTaskId) override;

  void noMoreRemoteTasks() override;

  void close() override;

  folly::F14FastMap<std::string, RuntimeMetric> stats() override;

  std::string toString() const override;

  folly::dynamic toJson() const override;

  /// Queue the received packed tables are buffered in. Part of the UCX data
  /// plane, used by the UcxExchange operator and by UcxExchangeSource.
  const std::shared_ptr<UcxExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns a PackedTableWithStream object from the queue or null.
  ///
  /// If no data is available returns a nullptr and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  PackedTableWithStreamPtr
  next(int consumerId, bool* atEnd, ContinueFuture* future);

  /// Max wait for a data-size request to a producer.
  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return requestDataSizesMaxWaitSec_;
  }

  /// Ids of the upstream tasks added so far.
  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int32_t maxQueuedColumns_;
  const std::chrono::seconds requestDataSizesMaxWaitSec_;

  const std::shared_ptr<UcxExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<UcxExchangeSource>> sources_;
  bool closed_{false};

  // Total number of packed columns in flight.
  int64_t totalPendingColumns_{0};

  // Diagnostic counters for progress and flow control.
  int64_t totalDequeued_{0};
  bool inFlowControl_{false};
};

} // namespace facebook::velox::ucx_exchange
