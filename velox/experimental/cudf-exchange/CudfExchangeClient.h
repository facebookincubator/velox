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
      folly::Executor* executor,
      int32_t requestDataSizesMaxWaitSec = 10)
      : taskId_{std::move(taskId)},
        destination_(destination),
        maxQueuedColumns_(kDefaultMaxQueuedColumns),
        kRequestDataSizesMaxWaitSec_(requestDataSizesMaxWaitSec),
        executor_(executor),
        queue_(std::make_shared<CudfExchangeQueue>(numberOfConsumers)) {
    VELOX_CHECK_NOT_NULL(executor_);
    // NOTE: the executor is used to run async response callback from the
    // cudf exchange source. The provided executor must not be
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
  // runtime metric named ExchangeClient::kBackgroundCpuTimeMs.
  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

  const std::shared_ptr<CudfExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns a packed_columns object from the queue or null.
  ///
  /// If no data is available returns a nullptr and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  ///
  std::unique_ptr<cudf::packed_columns>
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
  std::vector<std::shared_ptr<CudfExchangeSource>> pickSourcesToRequestLocked();

  void request(std::vector<std::shared_ptr<CudfExchangeSource>>&& requestSpecs);

  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int32_t maxQueuedColumns_;
  const std::chrono::seconds kRequestDataSizesMaxWaitSec_;

  folly::Executor* const executor_;
  const std::shared_ptr<CudfExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<CudfExchangeSource>> sources_;
  bool closed_{false};

  // Total number of packed_clumns in flight.
  int64_t totalPendingColumns_{0};

  // A queue of sources for which no reply is pending and that are not
  // atEnd. These are the sources to which requests are going to be sent.
  std::queue<std::shared_ptr<CudfExchangeSource>> readySources_;
};

} // namespace facebook::velox::cudf_exchange
