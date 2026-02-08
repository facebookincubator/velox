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
#include "velox/experimental/cudf-exchange/ExchangeClientFacade.h"
#include "velox/experimental/cudf-exchange/NetUtil.h"

namespace facebook::velox::cudf_exchange {

ExchangeClientFacade::ExchangeClientFacade(
    const std::string& taskId,
    int pipelineId,
    std::shared_ptr<CudfExchangeClient> cudfExchangeClient,
    std::shared_ptr<ExchangeClient> httpExchangeClient)
    : cudfExchangeClient_{cudfExchangeClient},
      httpExchangeClient_{httpExchangeClient},
      kCoordinatorUri_{Communicator::getInstance()->getCoordinatorUrl()},
      taskId_{taskId},
      pipelineId_{pipelineId} {
  VLOG(3) << "@" << taskId_ << "#" << pipelineId_ << " [" << this << "]"
          << " ExchangeClientFacade created";
}

void ExchangeClientFacade::activateCudfExchangeClient() {
  VELOX_CHECK(
      !usesHttp_, "Can't switch from Cudf to Http while operator is active.");
  if (usesCudf_) {
    // already activated.
    return;
  }
  addRemoteTaskId_ =
      [c = cudfExchangeClient_](const std::string& remoteTaskId) {
        c->addRemoteTaskId(remoteTaskId);
      };
  noMoreRemoteTasks_ = [c = cudfExchangeClient_]() { c->noMoreRemoteTasks(); };
  next_ = [c = cudfExchangeClient_](
              int consumerId,
              uint32_t maxBytes,
              bool* atEnd,
              facebook::velox::ContinueFuture* future) -> ResultVariant {
    return c->next(consumerId, atEnd, future);
  };
  stats_ = [c = cudfExchangeClient_]() { return c->stats(); };
  close_ = [c = cudfExchangeClient_]() { c->close(); };
  usesCudf_ = true;
}

void ExchangeClientFacade::activateHttpExchangeClient() {
  VELOX_CHECK(
      !usesCudf_, "Can't switch from Http to Cudf while operator is active.");
  if (usesHttp_) {
    // already activated.
    return;
  }
  addRemoteTaskId_ =
      [c = httpExchangeClient_](const std::string& remoteTaskId) {
        c->addRemoteTaskId(remoteTaskId);
      };
  noMoreRemoteTasks_ = [c = httpExchangeClient_]() { c->noMoreRemoteTasks(); };
  next_ = [c = httpExchangeClient_](
              int consumerId,
              uint32_t maxBytes,
              bool* atEnd,
              facebook::velox::ContinueFuture* future) -> ResultVariant {
    return c->next(consumerId, maxBytes, atEnd, future);
  };
  stats_ = [c = httpExchangeClient_]() { return c->stats(); };
  close_ = [c = httpExchangeClient_]() { c->close(); };
  usesHttp_ = true;
}

void ExchangeClientFacade::addRemoteTaskId(const std::string& remoteTaskId) {
  // dissect the remote task id.
  folly::Uri uri(remoteTaskId);
  const std::string host = uri.host();
  int port = uri.port();
  // DNS name resolution outside lock.
  bool remoteIsCoordinator =
      (isSameHost(host, kCoordinatorUri_.host()) &&
       (port == kCoordinatorUri_.port()));
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (remoteIsCoordinator) {
      VLOG(3) << "@" << taskId_ << "#" << pipelineId_ << " [" << this << "]"
              << " Activating HTTP exchange client for remote task id: "
              << remoteTaskId;
      activateHttpExchangeClient();
    } else {
      VLOG(3) << "@" << taskId_ << "#" << pipelineId_ << " [" << this << "]"
              << " Activating Cudf exchange client for remote task id: "
              << remoteTaskId;
      activateCudfExchangeClient();
    }
    addRemoteTaskId_(remoteTaskId);
    promises.reserve(promises_.size());
    auto it = promises_.begin();
    while (it != promises_.end()) {
      promises.push_back(std::move(it->second));
      it = promises_.erase(it);
    }
    VELOX_CHECK(promises_.empty());
  }
  // outside of lock: wake up exchange client facades that have been waiting
  // for the initial split.
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void ExchangeClientFacade::noMoreRemoteTasks() {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(
      noMoreRemoteTasks_, "noMoreRemoteTasks called but no client set!");
  noMoreRemoteTasks_();
}

// Depending on the underlying client, a different return type is used.
ResultVariant ExchangeClientFacade::next(
    int consumerId,
    uint32_t maxBytes,
    bool* atEnd,
    facebook::velox::ContinueFuture* future) {
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  ResultVariant result; // initializes to an empty result.
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!usesCudf_ && !usesHttp_) {
      // not using any client, need to wait for the first split
      // to resolve the client. Initialize the future.
      addPromiseLocked(consumerId, future, &stalePromise);
    } else {
      result = next_(consumerId, maxBytes, atEnd, future);
    }
  }
  // Outside of lock: complete the stale promise (if any)
  if (stalePromise.valid()) {
    stalePromise.setValue();
  }
  return result;
}

void ExchangeClientFacade::close() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    promises = clearAllPromisesLocked();
    // It's possible to reach close() before any splits were received,
    // in which case no client is activated. This can happen during error
    // handling or task cancellation. Safe to skip if no client is set.
    if (close_) {
      close_();
    }
  }
  clearPromises(promises);
}

folly::F14FastMap<std::string, facebook::velox::RuntimeMetric>
ExchangeClientFacade::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  // It's possible to reach stats() before any splits were received,
  // in which case no client is activated. This can happen during error
  // handling or task cancellation. Return empty stats if no client is set.
  if (!stats_) {
    return {};
  }
  return stats_();
}

void ExchangeClientFacade::addPromiseLocked(
    int consumerId,
    ContinueFuture* future,
    ContinuePromise* stalePromise) {
  ContinuePromise promise{"ExchangeClientFacade::waitForSplit"};
  *future = promise.getSemiFuture();
  auto it = promises_.find(consumerId);
  if (it != promises_.end()) {
    // resolve stale promises outside the lock to avoid broken promises
    *stalePromise = std::move(it->second);
    it->second = std::move(promise);
  } else {
    promises_[consumerId] = std::move(promise);
  }
}

} // namespace facebook::velox::cudf_exchange
