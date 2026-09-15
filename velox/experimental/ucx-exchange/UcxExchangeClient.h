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

#include <string_view>

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
class UcxExchangeClient : public exec::ExchangeClient {
 public:
  /// Flow control limit on the number of elements buffered in the queue.
  static constexpr int32_t kDefaultMaxQueuedColumns = 32;

  /// @param taskId Id of the consuming task, for logging.
  /// @param destination Partition the consuming task reads. Only validated:
  /// each split URL names the partition its source fetches.
  /// @param numberOfConsumers Number of UcxExchange operators sharing this
  /// client.
  /// @param receiveHighWaterBytes Query-level target for receive buffering.
  /// Zero disables byte accounting and retains the legacy table-count limit.
  /// @param requestDataSizesMaxWaitSec Max wait for a data-size request.
  UcxExchangeClient(
      std::string taskId,
      int destination,
      int32_t numberOfConsumers,
      uint64_t receiveHighWaterBytes = 0,
      int32_t requestDataSizesMaxWaitSec = 10);

  ~UcxExchangeClient() override;

  /// Creates a UCX exchange source and starts fetching data from the upstream
  /// task identified by 'remoteTaskId'. If close() has been called already,
  /// creates an exchange source and immediately closes it to notify the
  /// upstream task that its data is no longer needed. Repeated calls with the
  /// same 'remoteTaskId' are ignored.
  void addRemoteTaskId(std::string_view remoteTaskId) override;

  void noMoreRemoteTasks() override;

  void close() override;

  folly::F14FastMap<std::string, RuntimeMetric> stats() const override;

  std::string toString() const override;

  folly::dynamic toJson() const override;

  /// Returns the queue the received packed tables are buffered in. Part of the
  /// UCX data plane, used by the UcxExchange operator and by UcxExchangeSource.
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

  /// Called exactly once when a dequeued packed table's CudfVector releases
  /// its GPU storage.
  void releaseInFlightReceiveBytes(uint64_t bytes);

  /// Terminates receive processing after downstream discovers that retained
  /// GPU data is not safe to release (for example, stream synchronization
  /// failed). Safe to call from a CudfVector release callback.
  void failReceive(std::string_view error) noexcept;

  bool tracksInFlightReceiveBytes() const {
    return queue_->tracksInFlightReceiveBytes();
  }

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
  const int32_t maxQueuedColumns_;
  const std::chrono::seconds requestDataSizesMaxWaitSec_;

  const std::shared_ptr<UcxExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<UcxExchangeSource>> sources_;
  // Guarded by queue_->mutex().
  bool closed_{false};

  // Diagnostic counters for progress and flow control.
  int64_t totalDequeued_{0};
  bool inFlowControl_{false};
};

} // namespace facebook::velox::ucx_exchange
