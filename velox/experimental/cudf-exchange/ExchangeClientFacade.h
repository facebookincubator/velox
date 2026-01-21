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

#include <folly/Uri.h>
#include "velox/exec/ExchangeClient.h"
#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"

using namespace facebook::velox::exec;

namespace facebook::velox::cudf_exchange {

// Define the return types from the two types of exchange clients as a variant
using SerPageVector =
    std::vector<std::unique_ptr<facebook::velox::exec::SerializedPageBase>>;
// PackedTableWithStreamPtr is defined in CudfExchangeQueue.h
using ResultVariant = std::variant<SerPageVector, PackedTableWithStreamPtr>;

// The exchange client facade encapsulates both the cudf exchange client and the
// http exchange client.
class ExchangeClientFacade {
 public:
  ExchangeClientFacade(
      const std::string& taskId,
      int pipelineId,
      std::shared_ptr<CudfExchangeClient> cudfExchangeClient,
      std::shared_ptr<ExchangeClient> httpExchangeClient);

  // one of the below activate methods must be called first before any of the
  // facade methods can be called.

  // activate cudf exchange and wire up the function pointers.
  void activateCudfExchangeClient();

  // activate http exchange and wire up the function pointers.
  void activateHttpExchangeClient();

  // The facaded methods.
  void addRemoteTaskId(const std::string& remoteTaskId);
  void noMoreRemoteTasks();

  // Depending on the underlying client, a different return type is used.
  ResultVariant next(
      int consumerId,
      uint32_t maxBytes,
      bool* atEnd,
      facebook::velox::ContinueFuture* future);

  void close();

  void addPromiseLocked(
      int consumerId,
      ContinueFuture* future,
      ContinuePromise* stalePromise);

  std::vector<ContinuePromise> clearAllPromisesLocked() {
    std::vector<ContinuePromise> promises;
    promises.reserve(promises_.size());
    auto it = promises_.begin();
    while (it != promises_.end()) {
      promises.push_back(std::move(it->second));
      it = promises_.erase(it);
    }
    VELOX_CHECK(promises_.empty());
    return promises;
  }

  static void clearPromises(std::vector<ContinuePromise>& promises) {
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  std::mutex mutex_;

  // When multiple exchange operators are present, the none-primary ones
  // need to wait until the first split arrives, using below map.
  folly::F14FastMap<int, ContinuePromise> promises_;

  folly::F14FastMap<std::string, facebook::velox::RuntimeMetric> stats();

  std::shared_ptr<CudfExchangeClient> cudfExchangeClient_;
  std::shared_ptr<ExchangeClient> httpExchangeClient_;
  const folly::Uri kCoordinatorUri_;
  const std::string taskId_;
  const int pipelineId_;

  bool usesHttp_{false};
  bool usesCudf_{false};

  std::function<void(const std::string&)> addRemoteTaskId_;
  std::function<void()> noMoreRemoteTasks_;
  std::function<
      ResultVariant(int, uint32_t, bool*, facebook::velox::ContinueFuture*)>
      next_;
  std::function<
      folly::F14FastMap<std::string, facebook::velox::RuntimeMetric>()>
      stats_;
  std::function<void()> close_;
};

} // namespace facebook::velox::cudf_exchange
