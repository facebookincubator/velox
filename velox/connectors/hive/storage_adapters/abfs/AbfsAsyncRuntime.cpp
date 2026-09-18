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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsAsyncRuntime.h"

#include "velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h"

#include <folly/ExceptionWrapper.h>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/fibers/FiberManagerMap.h>
#include <folly/futures/Promise.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace facebook::velox::filesystems {
namespace {

constexpr char kOverloadedMessage[] = "ABFS async runtime queue is full";
constexpr char kShutdownMessage[] = "ABFS async runtime is shutting down";
constexpr char kResolverOverloadedMessage[] = "ABFS DNS resolver queue is full";
constexpr char kEndpointCacheFullMessage[] =
    "ABFS endpoint cache is full of active resolutions";
constexpr char kAuthOverloadedMessage[] =
    "ABFS authentication refresh queue is full";

struct FiberRuntimeBinding {
  const void* runtimeState{nullptr};
  const AbfsAsyncAuthService* authService{nullptr};
  const folly::CancellationToken* cancellationToken{nullptr};
};

class ScopedFiberRuntimeBinding final {
 public:
  ScopedFiberRuntimeBinding(
      const void* runtimeState,
      const AbfsAsyncAuthService* authService,
      const folly::CancellationToken& cancellationToken)
      : binding_(folly::fibers::local<FiberRuntimeBinding>()),
        previous_(binding_) {
    binding_.runtimeState = runtimeState;
    binding_.authService = authService;
    binding_.cancellationToken = &cancellationToken;
  }

  ~ScopedFiberRuntimeBinding() {
    binding_ = previous_;
  }

 private:
  FiberRuntimeBinding& binding_;
  FiberRuntimeBinding previous_;
};

class SystemEndpointResolver final : public AbfsEndpointResolver {
 public:
  folly::SocketAddress resolve(std::string_view host, uint16_t port) override {
    return folly::SocketAddress(std::string(host), port, true);
  }
};

class BoundedResolverExecutor final {
 public:
  struct Metrics {
    size_t active{0};
    size_t queued{0};
    size_t peakActive{0};
  };

  BoundedResolverExecutor(size_t numThreads, size_t maxQueued)
      : maxQueued_(maxQueued) {
    threads_.reserve(numThreads);
    for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
      threads_.emplace_back([this] { run(); });
    }
  }

  ~BoundedResolverExecutor() {
    shutdown();
  }

  bool submit(folly::Function<void()> task, folly::Function<void()> cancel) {
    std::lock_guard lock(mutex_);
    if (shuttingDown_ || queue_.size() >= maxQueued_) {
      return false;
    }
    queue_.push_back({std::move(task), std::move(cancel)});
    condition_.notify_one();
    return true;
  }

  void shutdown() noexcept {
    std::call_once(shutdownOnce_, [this] {
      std::deque<Job> cancelled;
      {
        std::lock_guard lock(mutex_);
        shuttingDown_ = true;
        cancelled.swap(queue_);
      }
      for (auto& job : cancelled) {
        try {
          job.cancel();
        } catch (...) {
        }
      }
      condition_.notify_all();
      for (auto& thread : threads_) {
        if (thread.joinable()) {
          thread.join();
        }
      }
    });
  }

  Metrics metrics() const {
    std::lock_guard lock(mutex_);
    return {active_, queue_.size(), peakActive_};
  }

 private:
  struct Job {
    folly::Function<void()> task;
    folly::Function<void()> cancel;
  };

  void run() noexcept {
    while (true) {
      std::optional<Job> job;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(
            lock, [this] { return shuttingDown_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;
        }
        job.emplace(std::move(queue_.front()));
        queue_.pop_front();
        ++active_;
        peakActive_ = std::max(peakActive_, active_);
      }
      try {
        job->task();
      } catch (...) {
      }
      {
        std::lock_guard lock(mutex_);
        --active_;
      }
    }
  }

  const size_t maxQueued_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Job> queue_;
  std::vector<std::thread> threads_;
  std::once_flag shutdownOnce_;
  bool shuttingDown_{false};
  size_t active_{0};
  size_t peakActive_{0};
};

void validateOptions(const AbfsAsyncRuntimeOptions& options) {
  if (options.numEventThreads == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires at least one event thread");
  }
  if (options.maxActiveRequests == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires at least one active request");
  }
  if (options.fiberStackBytes == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires a positive fiber stack size");
  }
  if (options.maxActiveRequests >
      std::numeric_limits<size_t>::max() / options.fiberStackBytes) {
    throw std::invalid_argument(
        "ABFS async runtime fiber stack capacity exceeds size_t");
  }
  if (options.numResolverThreads == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires at least one resolver thread");
  }
  if (options.maxQueuedResolutions == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires a positive resolver queue bound");
  }
  if (options.numAuthThreads == 0 || options.numAuthThreads > 2) {
    throw std::invalid_argument(
        "ABFS async runtime requires one or two authentication threads");
  }
  if (options.maxQueuedAuthRefreshes == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires a positive authentication queue bound");
  }
  if (options.maxEndpointCacheEntries == 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires a positive endpoint cache bound");
  }
  if (options.dnsCacheTtl.count() <= 0 || options.dnsFailureTtl.count() <= 0) {
    throw std::invalid_argument(
        "ABFS async runtime requires positive DNS cache lifetimes");
  }
}

AbfsAsyncRuntimeOptions validatedOptions(AbfsAsyncRuntimeOptions options) {
  validateOptions(options);
  return options;
}

} // namespace

class AbfsAsyncAuthService::State final
    : public std::enable_shared_from_this<AbfsAsyncAuthService::State> {
 public:
  State(size_t numThreads, size_t maxQueuedRefreshes)
      : numThreads_(numThreads), maxQueuedRefreshes_(maxQueuedRefreshes) {
    if (numThreads_ == 0 || numThreads_ > 2) {
      throw std::invalid_argument(
          "ABFS authentication service requires one or two workers");
    }
    if (maxQueuedRefreshes_ == 0) {
      throw std::invalid_argument(
          "ABFS authentication service requires a positive queue bound");
    }
    threads_.reserve(numThreads);
    for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
      threads_.emplace_back([this] { run(); });
    }
  }

  ~State() {
    shutdown();
  }

  std::string refresh(
      const AbfsAsyncAuthKey& key,
      RefreshCallback callback,
      const folly::CancellationToken& cancellationToken) {
    auto result = std::make_shared<Result>();
    bool notifyWorker{false};
    {
      std::lock_guard lock(mutex_);
      if (shuttingDown_) {
        throw std::runtime_error(kShutdownMessage);
      }
      const auto existing = inFlight_.find(key);
      if (existing != inFlight_.end()) {
        existing->second->waiters.push_back(result);
        ++sharedRefreshes_;
      } else {
        if (queue_.size() >= maxQueuedRefreshes_) {
          ++overloadedRefreshes_;
          throw std::runtime_error(kAuthOverloadedMessage);
        }
        auto refresh =
            std::make_shared<Refresh>(key, std::move(callback), result);
        ++waitingRefreshes_;
        inFlight_.emplace(key, refresh);
        queue_.push_back(std::move(refresh));
        notifyWorker = true;
      }
      if (!notifyWorker) {
        ++waitingRefreshes_;
      }
    }
    if (notifyWorker) {
      condition_.notify_one();
    }

    {
      const auto weakState = weak_from_this();
      folly::CancellationCallback cancellationCallback(
          cancellationToken, [weakState, key, result] {
            if (const auto state = weakState.lock()) {
              state->cancelWaiter(key, result);
            } else {
              result->notify();
            }
          });
      result->wait.wait();
    }
    if (cancellationToken.isCancellationRequested()) {
      throw std::runtime_error(kShutdownMessage);
    }

    std::lock_guard lock(result->mutex);
    if (result->failure != nullptr) {
      std::rethrow_exception(result->failure);
    }
    return result->token;
  }

  Metrics metrics() const {
    std::lock_guard lock(mutex_);
    return {
        numThreads_,
        maxQueuedRefreshes_,
        activeRefreshes_,
        queue_.size(),
        inFlight_.size(),
        waitingRefreshes_,
        refreshCallbacks_,
        sharedRefreshes_,
        completedRefreshes_,
        overloadedRefreshes_,
        cancelledRefreshes_,
    };
  }

  void shutdown() noexcept {
    std::call_once(shutdownOnce_, [this] {
      std::vector<std::shared_ptr<Result>> cancelled;
      {
        std::lock_guard lock(mutex_);
        shuttingDown_ = true;
        while (!queue_.empty()) {
          auto refresh = std::move(queue_.front());
          queue_.pop_front();
          inFlight_.erase(refresh->key);
          waitingRefreshes_ -= refresh->waiters.size();
          cancelledRefreshes_ += refresh->waiters.size();
          for (auto& waiter : refresh->waiters) {
            cancelled.push_back(std::move(waiter));
          }
        }
      }
      for (auto& result : cancelled) {
        result->complete(
            {}, std::make_exception_ptr(std::runtime_error(kShutdownMessage)));
      }
      condition_.notify_all();
      for (auto& thread : threads_) {
        if (thread.joinable()) {
          thread.join();
        }
      }
    });
  }

 private:
  struct Result {
    void complete(std::string value, std::exception_ptr exception) {
      {
        std::lock_guard lock(mutex);
        token = std::move(value);
        failure = std::move(exception);
      }
      notify();
    }

    void notify() {
      bool expected{false};
      if (notified.compare_exchange_strong(expected, true)) {
        wait.post();
      }
    }

    folly::fibers::Baton wait;
    std::mutex mutex;
    std::atomic<bool> notified{false};
    std::string token;
    std::exception_ptr failure;
  };

  struct Refresh {
    Refresh(
        AbfsAsyncAuthKey refreshKey,
        RefreshCallback refreshCallback,
        std::shared_ptr<Result> firstWaiter)
        : key(std::move(refreshKey)), callback(std::move(refreshCallback)) {
      waiters.push_back(std::move(firstWaiter));
    }

    AbfsAsyncAuthKey key;
    RefreshCallback callback;
    std::vector<std::shared_ptr<Result>> waiters;
  };

  struct KeyHash {
    size_t operator()(const AbfsAsyncAuthKey& key) const noexcept {
      size_t result = std::hash<std::string>{}(key.account);
      result = result * 31 + std::hash<std::string>{}(key.fileSystem);
      result = result * 31 + std::hash<std::string>{}(key.path);
      return result * 31 + std::hash<std::string>{}(key.operation);
    }
  };

  void cancelWaiter(
      const AbfsAsyncAuthKey& key,
      const std::shared_ptr<Result>& result) noexcept {
    {
      std::lock_guard lock(mutex_);
      const auto existing = inFlight_.find(key);
      if (existing != inFlight_.end()) {
        auto& waiters = existing->second->waiters;
        const auto waiter = std::find(waiters.begin(), waiters.end(), result);
        if (waiter != waiters.end()) {
          waiters.erase(waiter);
          --waitingRefreshes_;
          ++cancelledRefreshes_;
          if (waiters.empty()) {
            const auto queued =
                std::find(queue_.begin(), queue_.end(), existing->second);
            if (queued != queue_.end()) {
              queue_.erase(queued);
              inFlight_.erase(existing);
            }
          }
        }
      }
    }
    result->notify();
  }

  void run() noexcept {
    while (true) {
      std::shared_ptr<Refresh> refresh;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(
            lock, [this] { return shuttingDown_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;
        }
        refresh = std::move(queue_.front());
        queue_.pop_front();
        ++activeRefreshes_;
        ++refreshCallbacks_;
      }

      std::string token;
      std::exception_ptr failure;
      try {
        token = refresh->callback();
      } catch (...) {
        failure = std::current_exception();
      }

      std::vector<std::shared_ptr<Result>> waiters;
      {
        std::lock_guard lock(mutex_);
        --activeRefreshes_;
        const auto existing = inFlight_.find(refresh->key);
        if (existing != inFlight_.end() && existing->second == refresh) {
          inFlight_.erase(existing);
        }
        waitingRefreshes_ -= refresh->waiters.size();
        completedRefreshes_ += refresh->waiters.size();
        waiters.swap(refresh->waiters);
      }
      for (auto& waiter : waiters) {
        waiter->complete(token, failure);
      }
    }
  }

  const size_t numThreads_;
  const size_t maxQueuedRefreshes_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::shared_ptr<Refresh>> queue_;
  std::unordered_map<AbfsAsyncAuthKey, std::shared_ptr<Refresh>, KeyHash>
      inFlight_;
  std::vector<std::thread> threads_;
  std::once_flag shutdownOnce_;
  bool shuttingDown_{false};
  size_t activeRefreshes_{0};
  size_t waitingRefreshes_{0};
  size_t refreshCallbacks_{0};
  size_t sharedRefreshes_{0};
  size_t completedRefreshes_{0};
  size_t overloadedRefreshes_{0};
  size_t cancelledRefreshes_{0};
};

AbfsAsyncAuthService::AbfsAsyncAuthService(
    size_t numThreads,
    size_t maxQueuedRefreshes)
    : state_(std::make_shared<State>(numThreads, maxQueuedRefreshes)) {}

AbfsAsyncAuthService::~AbfsAsyncAuthService() {
  state_->shutdown();
}

std::string AbfsAsyncAuthService::refresh(
    const AbfsAsyncAuthKey& key,
    RefreshCallback callback) {
  const auto& binding = folly::fibers::local<FiberRuntimeBinding>();
  if (binding.authService != this || binding.cancellationToken == nullptr) {
    throw std::logic_error(
        "ABFS authentication refresh used outside an active runtime request");
  }
  if (!callback) {
    throw std::invalid_argument(
        "ABFS authentication refresh callback must not be empty");
  }
  return state_->refresh(key, std::move(callback), *binding.cancellationToken);
}

AbfsAsyncAuthService::Metrics AbfsAsyncAuthService::metrics() const {
  return state_->metrics();
}

void AbfsAsyncAuthService::shutdown() {
  state_->shutdown();
}

class AbfsAsyncRuntime::RuntimeState final
    : public std::enable_shared_from_this<RuntimeState> {
 private:
  class EndpointState;
  struct Request;
  using RequestTask = folly::Function<
      void(FollyHttpTransport*, const folly::CancellationToken&)>;

 public:
  explicit RuntimeState(AbfsAsyncRuntimeOptions options)
      : options_(validatedOptions(std::move(options))),
        endpointResolver_(
            options_.endpointResolver != nullptr
                ? options_.endpointResolver
                : std::make_shared<SystemEndpointResolver>()),
        resolverExecutor_(
            std::make_unique<BoundedResolverExecutor>(
                options_.numResolverThreads,
                options_.maxQueuedResolutions)),
        authService_(
            std::make_shared<AbfsAsyncAuthService>(
                options_.numAuthThreads,
                options_.maxQueuedAuthRefreshes)) {
    fiberOptions_.stackSize = options_.fiberStackBytes;
    fiberOptions_.stackSizeMultiplier = 1;
    fiberOptions_.recordStackEvery = options_.recordFiberStackEvery;
    shards_.reserve(options_.numEventThreads);
    for (size_t shardIndex = 0; shardIndex < options_.numEventThreads;
         ++shardIndex) {
      shards_.push_back(std::make_unique<folly::ScopedEventBaseThread>());
    }
  }

  folly::SemiFuture<folly::Unit> submit(std::string endpointKey, Task task) {
    if (endpointKey.empty()) {
      throw std::invalid_argument(
          "ABFS async runtime endpoint key must not be empty");
    }
    if (!task) {
      throw std::invalid_argument("ABFS async runtime task must not be empty");
    }

    return submitRequest(
        shardFor(endpointKey),
        nullptr,
        [task = std::move(task)](
            FollyHttpTransport*,
            const folly::CancellationToken& cancellationToken) mutable {
          task(cancellationToken);
        });
  }

  folly::SemiFuture<folly::Unit> submit(
      AbfsAsyncEndpointOptions endpointOptions,
      EndpointTask task) {
    if (endpointOptions.endpointKey.empty()) {
      throw std::invalid_argument(
          "ABFS async runtime endpoint key must not be empty");
    }
    if (!task) {
      throw std::invalid_argument("ABFS async runtime task must not be empty");
    }
    if (endpointOptions.maxConnections == 0) {
      throw std::invalid_argument(
          "ABFS async runtime endpoint connection limit must be positive");
    }
    if (!endpointOptions.hostname.empty() && endpointOptions.port == 0) {
      throw std::invalid_argument(
          "ABFS async runtime resolved endpoint port must be positive");
    }

    std::shared_ptr<EndpointState> endpoint;
    std::shared_ptr<EndpointState> retiredEndpoint;
    const char* rejection{nullptr};
    const auto now = dnsNow();
    {
      std::lock_guard lock(mutex_);
      if (!shuttingDown_) {
        auto iterator = endpoints_.find(endpointOptions.endpointKey);
        if (iterator != endpoints_.end()) {
          endpoint = iterator->second;
          if (!endpoint->matches(endpointOptions)) {
            throw std::invalid_argument(
                "ABFS async runtime endpoint options changed for an existing key");
          }
          const auto expiration =
              endpointExpirations_.find(endpointOptions.endpointKey);
          if (endpoint->resolutionComplete() &&
              expiration != endpointExpirations_.end() &&
              now >= expiration->second) {
            retiredEndpoint = std::move(iterator->second);
            endpointAccess_.erase(endpointOptions.endpointKey);
            endpointExpirations_.erase(expiration);
            endpoints_.erase(iterator);
            endpoint.reset();
            ++dnsCacheExpirations_;
          } else {
            endpointAccess_[endpointOptions.endpointKey] =
                ++endpointAccessSequence_;
            if (endpoint->requiresResolution()) {
              ++dnsCacheHits_;
            }
          }
        }
        if (endpoint == nullptr) {
          if (endpoints_.size() >= options_.maxEndpointCacheEntries) {
            auto eviction = endpoints_.end();
            size_t oldestAccess = std::numeric_limits<size_t>::max();
            for (auto candidate = endpoints_.begin();
                 candidate != endpoints_.end();
                 ++candidate) {
              const bool resolutionComplete =
                  candidate->second->resolutionComplete();
              const auto access = endpointAccess_.at(candidate->first);
              if (resolutionComplete && access < oldestAccess) {
                eviction = candidate;
                oldestAccess = access;
              }
            }
            if (eviction == endpoints_.end()) {
              ++endpointCacheRejections_;
              rejection = kEndpointCacheFullMessage;
            } else {
              retiredEndpoint = std::move(eviction->second);
              endpointAccess_.erase(eviction->first);
              endpointExpirations_.erase(eviction->first);
              endpoints_.erase(eviction);
              ++endpointCacheEvictions_;
            }
          }
        }
        if (endpoint == nullptr && rejection == nullptr) {
          endpoint = std::make_shared<EndpointState>(
              shardFor(endpointOptions.endpointKey),
              std::move(endpointOptions));
          endpoints_.emplace(endpoint->key(), endpoint);
          endpointAccess_[endpoint->key()] = ++endpointAccessSequence_;
          if (endpoint->requiresResolution()) {
            ++dnsCacheMisses_;
          }
        }
      }
    }
    retireEndpoint(std::move(retiredEndpoint));
    if (rejection != nullptr) {
      auto contract = folly::makePromiseContract<folly::Unit>();
      contract.promise.setException(
          folly::make_exception_wrapper<std::runtime_error>(rejection));
      return std::move(contract.future);
    }
    if (endpoint == nullptr) {
      auto contract = folly::makePromiseContract<folly::Unit>();
      contract.promise.setException(
          folly::make_exception_wrapper<std::runtime_error>(kShutdownMessage));
      return std::move(contract.future);
    }
    return submitRequest(
        endpoint->shard(),
        endpoint,
        [task = std::move(task)](
            FollyHttpTransport* transport,
            const folly::CancellationToken& cancellationToken) mutable {
          task(*transport, cancellationToken);
        });
  }

  folly::SemiFuture<folly::Unit> submitRequest(
      size_t shard,
      std::shared_ptr<EndpointState> endpoint,
      RequestTask task) {
    auto contract = folly::makePromiseContract<folly::Unit>();
    auto future = std::move(contract.future);
    auto request = std::make_shared<Request>(
        shard,
        std::move(endpoint),
        std::move(task),
        std::move(contract.promise));
    request->promise.setInterruptHandler(
        [state = std::weak_ptr<RuntimeState>(shared_from_this()),
         request = std::weak_ptr<Request>(request)](
            const folly::exception_wrapper& interruption) {
          auto lockedState = state.lock();
          auto lockedRequest = request.lock();
          if (lockedState != nullptr && lockedRequest != nullptr) {
            lockedState->cancelRequest(lockedRequest, interruption);
          }
        });
    bool dispatchNow{false};
    const char* rejection{nullptr};
    {
      std::lock_guard lock(mutex_);
      if (shuttingDown_) {
        rejection = kShutdownMessage;
      } else if (
          active_.size() >= options_.maxActiveRequests &&
          queue_.size() >= options_.maxQueuedRequests) {
        ++overloadedRequests_;
        rejection = kOverloadedMessage;
      } else {
        ++acceptedRequests_;
        if (active_.size() < options_.maxActiveRequests) {
          active_.insert(request);
          peakActiveRequests_ = std::max(peakActiveRequests_, active_.size());
          dispatchNow = true;
        } else {
          queue_.push_back(request);
        }
      }
    }
    if (rejection != nullptr) {
      request->fail(rejection);
      retireRejectedRequest(std::move(request));
      return future;
    }
    if (dispatchNow) {
      dispatch(std::move(request));
    }
    return future;
  }

  void cancelRequest(
      const std::shared_ptr<Request>& request,
      const folly::exception_wrapper& interruption) noexcept {
    bool active{false};
    bool queued{false};
    {
      std::lock_guard lock(mutex_);
      if (active_.find(request) != active_.end()) {
        request->interruption = interruption;
        active = true;
      } else {
        const auto iterator = std::find(queue_.begin(), queue_.end(), request);
        if (iterator != queue_.end()) {
          request->interruption = interruption;
          queue_.erase(iterator);
          ++cancelledRequests_;
          queued = true;
        }
      }
    }
    if (active) {
      request->cancellationSource.requestCancellation();
    } else if (queued) {
      try {
        request->promise.setException(interruption);
      } catch (...) {
      }
      retireRejectedRequest(request);
    }
  }

  void shutdown() {
    if (isRuntimeThread()) {
      throw std::logic_error(
          "ABFS async runtime shutdown cannot run on a runtime thread");
    }

    std::deque<std::shared_ptr<Request>> queued;
    std::vector<std::shared_ptr<Request>> active;
    {
      std::lock_guard lock(mutex_);
      if (!shuttingDown_) {
        shuttingDown_ = true;
        queued.swap(queue_);
        cancelledRequests_ += queued.size();
        for (const auto& request : active_) {
          request->shutdownRequested.store(true);
          active.push_back(request);
        }
      }
    }
    for (const auto& request : active) {
      request->cancellationSource.requestCancellation();
    }
    active.clear();
    destroyQueued(std::move(queued));
    resolverExecutor_->shutdown();

    {
      std::unique_lock lock(mutex_);
      shutdownCondition_.wait(lock, [this] { return active_.empty(); });
    }
    authService_->shutdown();
    destroyEndpoints();
    for (const auto& shard : shards_) {
      shard->getEventBase()->runImmediatelyOrRunInEventBaseThreadAndWait([] {});
    }
  }

  Metrics metrics() const {
    const auto resolverMetrics = resolverExecutor_->metrics();
    std::lock_guard lock(mutex_);
    return {
        shards_.size(),
        options_.maxActiveRequests,
        options_.maxQueuedRequests,
        options_.fiberStackBytes,
        endpoints_.size(),
        resolverMetrics.active,
        resolverMetrics.queued,
        resolverMetrics.peakActive,
        dnsCacheHits_,
        dnsCacheMisses_,
        endpointCacheEvictions_,
        endpointCacheRejections_,
        dnsCacheExpirations_,
        dnsResolutions_,
        dnsResolutionFailures_,
        active_.size(),
        queue_.size(),
        peakActiveRequests_,
        peakActiveRequests_ * options_.fiberStackBytes,
        measuredFiberStackHighWatermarkBytes_,
        acceptedRequests_,
        overloadedRequests_,
        completedRequests_,
        cancelledRequests_,
    };
  }

  bool isRuntimeThread() const {
    const auto threadId = std::this_thread::get_id();
    return std::any_of(shards_.begin(), shards_.end(), [&](const auto& shard) {
      return shard->getThreadId() == threadId;
    });
  }

  void waitForRetryDelay(std::chrono::milliseconds delay) const {
    if (delay.count() <= 0) {
      return;
    }
    const auto& binding = folly::fibers::local<FiberRuntimeBinding>();
    if (binding.runtimeState != this || binding.cancellationToken == nullptr) {
      throw std::logic_error(
          "ABFS retry delay used outside an active runtime request");
    }

    folly::fibers::Baton delayWait;
    bool interrupted{false};
    {
      folly::CancellationCallback cancellationCallback(
          *binding.cancellationToken, [&delayWait] { delayWait.post(); });
      interrupted = delayWait.timed_wait(delay);
    }
    if (interrupted || binding.cancellationToken->isCancellationRequested()) {
      throw std::runtime_error(kShutdownMessage);
    }
  }

  std::shared_ptr<AbfsAsyncAuthService> authService() const {
    return authService_;
  }

 private:
  class EndpointState final {
   public:
    EndpointState(size_t shard, AbfsAsyncEndpointOptions options)
        : shard_(shard), options_(std::move(options)) {}

    size_t shard() const noexcept {
      return shard_;
    }

    const std::string& key() const noexcept {
      return options_.endpointKey;
    }

    bool requiresResolution() const noexcept {
      return !options_.hostname.empty();
    }

    bool resolutionComplete() const noexcept {
      return !requiresResolution() || resolutionComplete_.load();
    }

    const std::string& hostname() const noexcept {
      return options_.hostname;
    }

    uint16_t port() const noexcept {
      return options_.port;
    }

    bool matches(const AbfsAsyncEndpointOptions& options) const {
      return options_.endpointKey == options.endpointKey &&
          options_.hostname == options.hostname &&
          options_.port == options.port &&
          options_.channelEndpoint.connectAddress ==
          options.channelEndpoint.connectAddress &&
          options_.channelEndpoint.serverName ==
          options.channelEndpoint.serverName &&
          options_.channelEndpoint.security ==
          options.channelEndpoint.security &&
          options_.channelEndpoint.connectTimeout ==
          options.channelEndpoint.connectTimeout &&
          options_.channelEndpoint.tlsHandshakeTimeout ==
          options.channelEndpoint.tlsHandshakeTimeout &&
          options_.channelEndpoint.additionalTrustedCaPath ==
          options.channelEndpoint.additionalTrustedCaPath &&
          options_.httpLimits.maxStatusLineBytes ==
          options.httpLimits.maxStatusLineBytes &&
          options_.httpLimits.maxHeaderBytes ==
          options.httpLimits.maxHeaderBytes &&
          options_.httpLimits.maxRequestBodyBytes ==
          options.httpLimits.maxRequestBodyBytes &&
          options_.httpLimits.maxBufferedResponseBodyBytes ==
          options.httpLimits.maxBufferedResponseBodyBytes &&
          options_.httpLimits.maxIngressBytes ==
          options.httpLimits.maxIngressBytes &&
          options_.httpLimits.maxInformationalResponses ==
          options.httpLimits.maxInformationalResponses &&
          options_.httpTimeouts.write == options.httpTimeouts.write &&
          options_.httpTimeouts.firstByteAndHeaders ==
          options.httpTimeouts.firstByteAndHeaders &&
          options_.httpTimeouts.bodyIdle == options.httpTimeouts.bodyIdle &&
          options_.httpTimeouts.total == options.httpTimeouts.total &&
          options_.httpTimeouts.connectionAcquire ==
          options.httpTimeouts.connectionAcquire &&
          options_.httpTimeouts.connectionIdle ==
          options.httpTimeouts.connectionIdle &&
          options_.maxConnections == options.maxConnections;
    }

    bool beginResolution() {
      if (resolutionStarted_) {
        return false;
      }
      resolutionStarted_ = true;
      return true;
    }

    void waitForResolution() {
      if (!requiresResolution()) {
        return;
      }
      if (!resolutionFinished_) {
        auto waiter = std::make_shared<folly::fibers::Baton>();
        resolutionWaiters_.push_back(waiter);
        waiter->wait();
      }
      if (resolutionFailure_ != nullptr) {
        std::rethrow_exception(resolutionFailure_);
      }
    }

    void finishResolution(
        std::optional<folly::SocketAddress> address,
        std::exception_ptr failure) {
      if (resolutionFinished_) {
        return;
      }
      resolvedAddress_ = std::move(address);
      resolutionFailure_ = std::move(failure);
      resolutionFinished_ = true;
      resolutionComplete_.store(true);
      for (const auto& waiter : resolutionWaiters_) {
        waiter->post();
      }
      resolutionWaiters_.clear();
    }

    FollyHttpTransport& transport(folly::EventBase* eventBase) {
      if (!eventBase->isInEventBaseThread()) {
        throw std::logic_error(
            "ABFS async endpoint transport used from the wrong EventBase");
      }
      if (transport_ == nullptr) {
        auto channelEndpoint = options_.channelEndpoint;
        if (requiresResolution()) {
          if (!resolvedAddress_.has_value()) {
            throw std::logic_error(
                "ABFS async endpoint transport used before DNS resolution");
          }
          channelEndpoint.connectAddress = *resolvedAddress_;
        }
        auto factory = std::make_shared<EventSocketChannelFactory>(eventBase);
        transport_ = std::make_shared<FollyHttpTransport>(
            std::move(factory),
            std::move(channelEndpoint),
            options_.httpLimits,
            options_.httpTimeouts,
            options_.maxConnections);
      }
      return *transport_;
    }

   private:
    size_t shard_;
    AbfsAsyncEndpointOptions options_;
    bool resolutionStarted_{false};
    bool resolutionFinished_{false};
    std::atomic<bool> resolutionComplete_{false};
    std::optional<folly::SocketAddress> resolvedAddress_;
    std::exception_ptr resolutionFailure_;
    std::vector<std::shared_ptr<folly::fibers::Baton>> resolutionWaiters_;
    std::shared_ptr<FollyHttpTransport> transport_;
  };

  struct Request {
    Request(
        size_t requestShard,
        std::shared_ptr<EndpointState> requestEndpoint,
        RequestTask requestTask,
        folly::Promise<folly::Unit> requestPromise)
        : shard(requestShard),
          endpoint(std::move(requestEndpoint)),
          task(std::move(requestTask)),
          promise(std::move(requestPromise)) {}

    void fail(const char* message) noexcept {
      try {
        promise.setException(
            folly::make_exception_wrapper<std::runtime_error>(message));
      } catch (...) {
      }
    }

    size_t shard;
    std::shared_ptr<EndpointState> endpoint;
    RequestTask task;
    folly::Promise<folly::Unit> promise;
    folly::CancellationSource cancellationSource;
    std::atomic<bool> shutdownRequested{false};
    std::optional<folly::exception_wrapper> interruption;
  };

  size_t shardFor(const std::string& endpointKey) const {
    return std::hash<std::string>{}(endpointKey) % shards_.size();
  }

  std::chrono::steady_clock::time_point dnsNow() const {
    return options_.dnsClock ? options_.dnsClock()
                             : std::chrono::steady_clock::now();
  }

  void recordResolution(
      const std::shared_ptr<EndpointState>& endpoint,
      bool succeeded) {
    const auto expiration =
        dnsNow() + (succeeded ? options_.dnsCacheTtl : options_.dnsFailureTtl);
    std::lock_guard lock(mutex_);
    if (!succeeded) {
      ++dnsResolutionFailures_;
    }
    const auto iterator = endpoints_.find(endpoint->key());
    if (iterator != endpoints_.end() && iterator->second == endpoint) {
      endpointExpirations_[endpoint->key()] = expiration;
    }
  }

  void resolveEndpoint(const std::shared_ptr<EndpointState>& endpoint) {
    const auto shard = endpoint->shard();
    auto* eventBase = shards_[shard]->getEventBase();
    auto complete = [this, endpoint, eventBase](
                        std::optional<folly::SocketAddress> address,
                        std::exception_ptr failure) mutable {
      recordResolution(endpoint, failure == nullptr);
      eventBase->runInEventBaseThread([endpoint,
                                       address = std::move(address),
                                       failure = std::move(failure)]() mutable {
        endpoint->finishResolution(std::move(address), std::move(failure));
      });
    };
    const auto hostname = endpoint->hostname();
    const auto port = endpoint->port();
    const bool accepted = resolverExecutor_->submit(
        [resolver = endpointResolver_,
         hostname,
         port,
         complete = complete]() mutable {
          try {
            complete(resolver->resolve(hostname, port), nullptr);
          } catch (...) {
            complete(std::nullopt, std::current_exception());
          }
        },
        [complete = std::move(complete)]() mutable {
          complete(
              std::nullopt,
              std::make_exception_ptr(std::runtime_error(kShutdownMessage)));
        });
    if (accepted) {
      std::lock_guard lock(mutex_);
      ++dnsResolutions_;
      return;
    }
    complete(
        std::nullopt,
        std::make_exception_ptr(
            std::runtime_error(kResolverOverloadedMessage)));
  }

  void prepareEndpoint(const std::shared_ptr<EndpointState>& endpoint) {
    if (endpoint == nullptr || !endpoint->requiresResolution()) {
      return;
    }
    if (endpoint->beginResolution()) {
      resolveEndpoint(endpoint);
    }
    endpoint->waitForResolution();
  }

  void dispatch(std::shared_ptr<Request> request) {
    auto* eventBase = shards_[request->shard]->getEventBase();
    eventBase->runInEventBaseThread([state = shared_from_this(),
                                     request = std::move(request)]() mutable {
      auto* eventBase = state->shards_[request->shard]->getEventBase();
      auto& fiberManager =
          folly::fibers::getFiberManager(*eventBase, state->fiberOptions_);
      fiberManager.add([state = std::move(state),
                        request = std::move(request),
                        &fiberManager]() mutable {
        std::exception_ptr failure;
        try {
          FollyHttpTransport* transport{nullptr};
          if (request->endpoint != nullptr) {
            state->prepareEndpoint(request->endpoint);
            transport = &request->endpoint->transport(
                state->shards_[request->shard]->getEventBase());
          }
          const auto cancellationToken = request->cancellationSource.getToken();
          ScopedFiberRuntimeBinding binding(
              state.get(), state->authService_.get(), cancellationToken);
          request->task(transport, cancellationToken);
        } catch (...) {
          failure = std::current_exception();
        }
        state->recordFiberStackHighWatermark(fiberManager);
        state->complete(std::move(request), std::move(failure));
      });
    });
  }

  void recordFiberStackHighWatermark(
      folly::fibers::FiberManager& fiberManager) {
    if (fiberOptions_.recordStackEvery == 0) {
      return;
    }
    const auto stackHighWatermark = fiberManager.runInMainContext(
        [&fiberManager] { return fiberManager.stackHighWatermark(); });
    std::lock_guard lock(mutex_);
    measuredFiberStackHighWatermarkBytes_ =
        std::max(measuredFiberStackHighWatermarkBytes_, stackHighWatermark);
  }

  void complete(
      std::shared_ptr<Request> request,
      std::exception_ptr failure) noexcept {
    bool shutdownRequested{false};
    std::optional<folly::exception_wrapper> interruption;
    std::shared_ptr<Request> next;
    {
      std::lock_guard lock(mutex_);
      shutdownRequested = request->shutdownRequested.load();
      interruption = request->interruption;
      active_.erase(request);
      ++completedRequests_;
      if (shutdownRequested || interruption.has_value()) {
        ++cancelledRequests_;
      }
      if (!shuttingDown_ && !queue_.empty()) {
        next = std::move(queue_.front());
        queue_.pop_front();
        active_.insert(next);
      }
    }

    try {
      if (shutdownRequested) {
        request->promise.setException(
            folly::make_exception_wrapper<std::runtime_error>(
                kShutdownMessage));
      } else if (interruption.has_value()) {
        request->promise.setException(std::move(*interruption));
      } else if (failure != nullptr) {
        request->promise.setException(folly::exception_wrapper(failure));
      } else {
        request->promise.setValue();
      }
    } catch (...) {
    }
    shutdownCondition_.notify_all();
    if (next != nullptr) {
      dispatch(std::move(next));
    }
  }

  void destroyQueued(std::deque<std::shared_ptr<Request>> queued) {
    std::vector<std::vector<std::shared_ptr<Request>>> requestsByShard(
        shards_.size());
    while (!queued.empty()) {
      auto request = std::move(queued.front());
      queued.pop_front();
      requestsByShard[request->shard].push_back(std::move(request));
    }
    for (size_t shardIndex = 0; shardIndex < shards_.size(); ++shardIndex) {
      if (requestsByShard[shardIndex].empty()) {
        continue;
      }
      shards_[shardIndex]
          ->getEventBase()
          ->runImmediatelyOrRunInEventBaseThreadAndWait(
              [requests = std::move(requestsByShard[shardIndex])]() mutable {
                for (const auto& request : requests) {
                  request->fail(kShutdownMessage);
                }
                requests.clear();
              });
    }
  }

  void retireEndpoint(std::shared_ptr<EndpointState> endpoint) {
    if (endpoint == nullptr) {
      return;
    }
    shards_[endpoint->shard()]->getEventBase()->runInEventBaseThread(
        [endpoint = std::move(endpoint)]() mutable { endpoint.reset(); });
  }

  void retireRejectedRequest(std::shared_ptr<Request> request) {
    if (request->endpoint == nullptr) {
      return;
    }
    shards_[request->shard]->getEventBase()->runInEventBaseThread(
        [request = std::move(request)]() mutable { request.reset(); });
  }

  void destroyEndpoints() {
    std::vector<std::vector<std::shared_ptr<EndpointState>>> endpointsByShard(
        shards_.size());
    {
      std::lock_guard lock(mutex_);
      for (auto& [key, endpoint] : endpoints_) {
        endpointsByShard[endpoint->shard()].push_back(std::move(endpoint));
      }
      endpoints_.clear();
      endpointExpirations_.clear();
      endpointAccess_.clear();
    }
    for (size_t shardIndex = 0; shardIndex < shards_.size(); ++shardIndex) {
      if (endpointsByShard[shardIndex].empty()) {
        continue;
      }
      shards_[shardIndex]
          ->getEventBase()
          ->runImmediatelyOrRunInEventBaseThreadAndWait(
              [endpoints = std::move(endpointsByShard[shardIndex])]() mutable {
                endpoints.clear();
              });
    }
  }

  AbfsAsyncRuntimeOptions options_;
  std::shared_ptr<AbfsEndpointResolver> endpointResolver_;
  std::unique_ptr<BoundedResolverExecutor> resolverExecutor_;
  std::shared_ptr<AbfsAsyncAuthService> authService_;
  folly::fibers::FiberManager::Options fiberOptions_;
  std::vector<std::unique_ptr<folly::ScopedEventBaseThread>> shards_;
  mutable std::mutex mutex_;
  std::condition_variable shutdownCondition_;
  std::unordered_map<std::string, std::shared_ptr<EndpointState>> endpoints_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      endpointExpirations_;
  std::unordered_map<std::string, size_t> endpointAccess_;
  std::deque<std::shared_ptr<Request>> queue_;
  std::unordered_set<std::shared_ptr<Request>> active_;
  bool shuttingDown_{false};
  size_t endpointAccessSequence_{0};
  size_t peakActiveRequests_{0};
  size_t measuredFiberStackHighWatermarkBytes_{0};
  size_t dnsCacheHits_{0};
  size_t dnsCacheMisses_{0};
  size_t endpointCacheEvictions_{0};
  size_t endpointCacheRejections_{0};
  size_t dnsCacheExpirations_{0};
  size_t dnsResolutions_{0};
  size_t dnsResolutionFailures_{0};
  size_t acceptedRequests_{0};
  size_t overloadedRequests_{0};
  size_t completedRequests_{0};
  size_t cancelledRequests_{0};
};

AbfsAsyncRuntime::AbfsAsyncRuntime(AbfsAsyncRuntimeOptions options)
    : state_(std::make_shared<RuntimeState>(options)) {}

AbfsAsyncRuntime::~AbfsAsyncRuntime() {
  if (state_->isRuntimeThread()) {
    std::thread([state = std::move(state_)]() mutable {
      try {
        state->shutdown();
      } catch (...) {
      }
    }).detach();
    return;
  }
  try {
    state_->shutdown();
  } catch (...) {
  }
}

folly::SemiFuture<folly::Unit> AbfsAsyncRuntime::submit(
    std::string endpointKey,
    Task task) {
  return state_->submit(std::move(endpointKey), std::move(task));
}

folly::SemiFuture<folly::Unit> AbfsAsyncRuntime::submit(
    AbfsAsyncEndpointOptions endpoint,
    EndpointTask task) {
  return state_->submit(std::move(endpoint), std::move(task));
}

void AbfsAsyncRuntime::shutdown() {
  state_->shutdown();
}

void AbfsAsyncRuntime::waitForRetryDelay(
    std::chrono::milliseconds delay) const {
  state_->waitForRetryDelay(delay);
}

std::shared_ptr<AbfsAsyncAuthService> AbfsAsyncRuntime::authService() const {
  return state_->authService();
}

AbfsAsyncRuntime::Metrics AbfsAsyncRuntime::metrics() const {
  return state_->metrics();
}

bool AbfsAsyncRuntime::isRuntimeThread() const {
  return state_->isRuntimeThread();
}

} // namespace facebook::velox::filesystems
