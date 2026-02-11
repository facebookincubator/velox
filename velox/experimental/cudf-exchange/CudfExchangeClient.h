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

#include "velox/experimental/cudf-exchange/CudfExchangeQueue.h"
#include "velox/experimental/cudf-exchange/CudfExchangeSource.h"

namespace facebook::velox::cudf_exchange {

// Handle for a set of producers. This may be shared by multiple CudfExchanges,
// one per consumer thread.
class CudfExchangeClient
    : public std::enable_shared_from_this<CudfExchangeClient> {
 public:
  // used for some primitive type of flow control, limits the size of elements
  // in the CudfExchangeQueue
  static constexpr int32_t kDefaultMaxQueuedColumns = 32;
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};

  CudfExchangeClient(
      std::string taskId,
      int destination,
      int32_t numberOfConsumers,
      int32_t requestDataSizesMaxWaitSec = 10)
      : taskId_{std::move(taskId)},
        destination_(destination),
        maxQueuedColumns_(kDefaultMaxQueuedColumns),
        kRequestDataSizesMaxWaitSec_(requestDataSizesMaxWaitSec),
        queue_(std::make_shared<CudfExchangeQueue>(numberOfConsumers)) {
    VELOX_CHECK_GE(
        destination, 0, "Exchange client destination must not be negative");
  }

  ~CudfExchangeClient();

  // Creates a cudf exchange source and starts fetching data from the specified
  // upstream task. If 'close' has been called already, creates an exchange
  // source and immediately closes it to notify the upstream task that data is
  // no longer needed. Repeated calls with the same 'taskId' are ignored.
  void addRemoteTaskId(const std::string& remoteTaskId);

  void noMoreRemoteTasks();

  // Closes all exchange sources.
  void close();

  // Returns runtime statistics aggregated across all of the exchange sources.
  // ExchangeClient is expected to report background CPU time by including a
  // runtime metric named Operator::kBackgroundCpuTimeNanos.
  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

  const std::shared_ptr<CudfExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns a PackedTableWithStream object from the queue or null.
  ///
  /// If no data is available returns a nullptr and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  ///
  PackedTableWithStreamPtr
  next(int consumerId, bool* atEnd, ContinueFuture* future);

  std::string toString() const;

  folly::dynamic toJson() const;

  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return kRequestDataSizesMaxWaitSec_;
  }

  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int32_t maxQueuedColumns_;
  const std::chrono::seconds kRequestDataSizesMaxWaitSec_;

  const std::shared_ptr<CudfExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<CudfExchangeSource>> sources_;
  bool closed_{false};

  // Total number of packed_clumns in flight.
  int64_t totalPendingColumns_{0};
};

} // namespace facebook::velox::cudf_exchange
