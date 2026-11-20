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
#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h"

#include <azure/core/http/policies/policy.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/url.hpp>
#include <folly/ScopeGuard.h>
#include <folly/SocketAddress.h>
#include <folly/fibers/Baton.h>
#include <folly/fibers/FiberManager.h>
#include <folly/futures/Promise.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace facebook::velox::filesystems {
namespace {

constexpr auto kWaitTimeout = std::chrono::seconds(5);

class RetrySequencePolicy final
    : public Azure::Core::Http::Policies::HttpPolicy {
 public:
  explicit RetrySequencePolicy(std::shared_ptr<std::atomic<size_t>> attempts)
      : attempts_(std::move(attempts)) {}

  std::unique_ptr<Azure::Core::Http::RawResponse> Send(
      Azure::Core::Http::Request&,
      Azure::Core::Http::Policies::NextHttpPolicy,
      const Azure::Core::Context&) const override {
    const auto attempt = ++*attempts_;
    return std::make_unique<Azure::Core::Http::RawResponse>(
        1,
        1,
        attempt == 1 ? Azure::Core::Http::HttpStatusCode::RequestTimeout
                     : Azure::Core::Http::HttpStatusCode::Ok,
        "Test");
  }

  std::unique_ptr<HttpPolicy> Clone() const override {
    return std::make_unique<RetrySequencePolicy>(*this);
  }

 private:
  std::shared_ptr<std::atomic<size_t>> attempts_;
};

class RuntimeEndpointServer {
 public:
  RuntimeEndpointServer() {
    listenSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSocket_ < 0) {
      throw std::runtime_error("failed to create runtime endpoint listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0);
    if (bind(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0 ||
        listen(listenSocket_, 2) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to bind runtime endpoint listener");
    }
    socklen_t addressLength = sizeof(address);
    if (getsockname(
            listenSocket_,
            reinterpret_cast<sockaddr*>(&address),
            &addressLength) < 0) {
      close(listenSocket_);
      listenSocket_ = -1;
      throw std::runtime_error("failed to inspect runtime endpoint listener");
    }
    port_ = ntohs(address.sin_port);
  }

  ~RuntimeEndpointServer() {
    stop();
  }

  RuntimeEndpointServer(const RuntimeEndpointServer&) = delete;
  RuntimeEndpointServer& operator=(const RuntimeEndpointServer&) = delete;

  folly::SocketAddress address() const {
    return folly::SocketAddress("127.0.0.1", port_);
  }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  void stop() noexcept {
    if (listenSocket_ >= 0) {
      shutdown(listenSocket_, SHUT_RDWR);
    }
    const auto clientSocket = activeClientSocket_.load();
    if (clientSocket >= 0) {
      shutdown(clientSocket, SHUT_RDWR);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    if (listenSocket_ >= 0) {
      close(listenSocket_);
      listenSocket_ = -1;
    }
  }

  bool failed() const noexcept {
    return failed_;
  }

  bool firstClientClosed() const noexcept {
    return firstClientClosed_;
  }

  size_t requests() const noexcept {
    return requests_;
  }

 private:
  static bool readRequest(int clientSocket) noexcept {
    std::array<char, 4 * 1'024> buffer{};
    size_t bytesRead{0};
    while (bytesRead < buffer.size()) {
      pollfd clientPoll{clientSocket, POLLIN, 0};
      if (poll(&clientPoll, 1, 2'000) <= 0 || !(clientPoll.revents & POLLIN)) {
        return false;
      }
      const auto bytes = recv(
          clientSocket,
          buffer.data() + bytesRead,
          buffer.size() - bytesRead,
          0);
      if (bytes <= 0) {
        return false;
      }
      bytesRead += static_cast<size_t>(bytes);
      if (std::strstr(buffer.data(), "\r\n\r\n") != nullptr) {
        return true;
      }
    }
    return false;
  }

  static bool
  sendAll(int clientSocket, const char* data, size_t size) noexcept {
    size_t bytesSent{0};
    while (bytesSent < size) {
      const auto bytes =
          send(clientSocket, data + bytesSent, size - bytesSent, MSG_NOSIGNAL);
      if (bytes < 0 && errno == EINTR) {
        continue;
      }
      if (bytes <= 0) {
        return false;
      }
      bytesSent += static_cast<size_t>(bytes);
    }
    return true;
  }

  void run() noexcept {
    try {
      for (size_t requestIndex = 0; requestIndex < 2; ++requestIndex) {
        pollfd listenerPoll{listenSocket_, POLLIN, 0};
        if (poll(&listenerPoll, 1, 2'000) <= 0 ||
            !(listenerPoll.revents & POLLIN)) {
          failed_ = true;
          return;
        }
        const auto clientSocket = accept(listenSocket_, nullptr, nullptr);
        if (clientSocket < 0) {
          failed_ = true;
          return;
        }
        activeClientSocket_ = clientSocket;
        auto closeClient = folly::makeGuard([&] {
          close(clientSocket);
          activeClientSocket_ = -1;
        });
        if (!readRequest(clientSocket)) {
          failed_ = true;
          return;
        }
        ++requests_;
        if (requestIndex == 0) {
          static constexpr char kHeldResponse[] =
              "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n"
              "Connection: keep-alive\r\n\r\n";
          if (!sendAll(
                  clientSocket, kHeldResponse, sizeof(kHeldResponse) - 1)) {
            failed_ = true;
            return;
          }
          pollfd clientPoll{clientSocket, POLLIN, 0};
          if (poll(&clientPoll, 1, 2'000) <= 0 ||
              !(clientPoll.revents & (POLLIN | POLLHUP | POLLERR))) {
            failed_ = true;
            return;
          }
          char byte{};
          firstClientClosed_ = recv(clientSocket, &byte, 1, 0) == 0;
          if (!firstClientClosed_) {
            failed_ = true;
            return;
          }
          continue;
        }
        static constexpr char kCompleteResponse[] =
            "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n"
            "Connection: close\r\n\r\nbody";
        if (!sendAll(
                clientSocket,
                kCompleteResponse,
                sizeof(kCompleteResponse) - 1)) {
          failed_ = true;
          return;
        }
      }
    } catch (...) {
      failed_ = true;
    }
  }

  int listenSocket_{-1};
  uint16_t port_{0};
  std::atomic<int> activeClientSocket_{-1};
  std::atomic<bool> failed_{false};
  std::atomic<bool> firstClientClosed_{false};
  std::atomic<size_t> requests_{0};
  std::thread thread_;
};

struct RequestLifetimeTracker {
  void record(std::string requestName) {
    {
      std::lock_guard lock(mutex);
      destroyedRequests.emplace_back(
          std::move(requestName), std::this_thread::get_id());
    }
    condition.notify_all();
  }

  bool waitForCount(size_t count) {
    std::unique_lock lock(mutex);
    return condition.wait_for(
        lock, kWaitTimeout, [&] { return destroyedRequests.size() >= count; });
  }

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<std::pair<std::string, std::thread::id>> destroyedRequests;
};

struct PrivateRequestState {
  PrivateRequestState(
      std::shared_ptr<RequestLifetimeTracker> requestTracker,
      std::string requestName)
      : tracker(std::move(requestTracker)), name(std::move(requestName)) {}

  ~PrivateRequestState() {
    tracker->record(std::move(name));
  }

  std::shared_ptr<RequestLifetimeTracker> tracker;
  std::string name;
};

class TestingResolver final : public AbfsEndpointResolver {
 public:
  folly::SocketAddress resolve(std::string_view host, uint16_t port) override {
    std::lock_guard lock(mutex_);
    ++calls_[std::string(host)];
    resolverThreads_.insert(std::this_thread::get_id());
    if (failures_.contains(host)) {
      throw std::runtime_error("injected DNS failure");
    }
    return folly::SocketAddress("127.0.0.1", port);
  }

  void setFailure(const std::string& host, bool fail) {
    std::lock_guard lock(mutex_);
    if (fail) {
      failures_.insert(host);
    } else {
      failures_.erase(host);
    }
  }

  size_t calls(const std::string& host) const {
    std::lock_guard lock(mutex_);
    const auto iterator = calls_.find(host);
    return iterator == calls_.end() ? 0 : iterator->second;
  }

  std::set<std::thread::id> resolverThreads() const {
    std::lock_guard lock(mutex_);
    return resolverThreads_;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, size_t> calls_;
  std::set<std::string, std::less<>> failures_;
  std::set<std::thread::id> resolverThreads_;
};

class BlockingResolver final : public AbfsEndpointResolver {
 public:
  folly::SocketAddress resolve(std::string_view, uint16_t port) override {
    std::unique_lock lock(mutex_);
    ++calls_;
    started_.notify_all();
    release_.wait(lock, [this] { return released_; });
    return folly::SocketAddress("127.0.0.1", port);
  }

  bool waitForCalls(size_t calls) {
    std::unique_lock lock(mutex_);
    return started_.wait_for(
        lock, kWaitTimeout, [&] { return calls_ >= calls; });
  }

  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    release_.notify_all();
  }

  size_t calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable started_;
  std::condition_variable release_;
  bool released_{false};
  size_t calls_{0};
};

AbfsAsyncEndpointOptions unresolvedEndpoint(
    std::string endpointKey,
    std::string hostname) {
  AbfsAsyncEndpointOptions endpoint;
  endpoint.endpointKey = std::move(endpointKey);
  endpoint.hostname = std::move(hostname);
  endpoint.port = 443;
  endpoint.channelEndpoint.serverName = endpoint.hostname;
  return endpoint;
}

bool waitForQueuedDnsResolutions(
    const AbfsAsyncRuntime& runtime,
    size_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (runtime.metrics().queuedDnsResolutions == expected) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

bool waitForAuthWaiters(
    const std::shared_ptr<AbfsAsyncAuthService>& authService,
    size_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (authService->metrics().waitingRefreshes == expected) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

std::string endpointKeyForShard(size_t shard, size_t numShards) {
  for (size_t suffix = 0;; ++suffix) {
    auto endpointKey = "account-" + std::to_string(suffix);
    if (std::hash<std::string>{}(endpointKey) % numShards == shard) {
      return endpointKey;
    }
  }
}

TEST(AbfsAsyncRuntimeTest, BoundsActiveAndQueuedRequestsAcrossShards) {
  AbfsAsyncRuntimeOptions options;
  options.numEventThreads = 3;
  options.maxActiveRequests = 3;
  options.maxQueuedRequests = 2;
  options.fiberStackBytes = 128 * 1'024;
  options.recordFiberStackEvery = 1;
  AbfsAsyncRuntime runtime(options);
  std::array<folly::fibers::Baton, 3> releaseActive;
  std::vector<folly::SemiFuture<folly::Unit>> active;
  std::vector<folly::SemiFuture<folly::Unit>> started;
  std::mutex runtimeThreadsMutex;
  std::set<std::thread::id> runtimeThreads;
  for (size_t shard = 0; shard < options.numEventThreads; ++shard) {
    auto startedContract = folly::makePromiseContract<folly::Unit>();
    active.push_back(runtime.submit(
        endpointKeyForShard(shard, options.numEventThreads),
        [promise = std::move(startedContract.promise),
         &release = releaseActive[shard],
         &runtimeThreadsMutex,
         &runtimeThreads](const folly::CancellationToken&) mutable {
          {
            std::lock_guard lock(runtimeThreadsMutex);
            runtimeThreads.insert(std::this_thread::get_id());
          }
          promise.setValue();
          release.wait();
        }));
    started.push_back(std::move(startedContract.future));
  }
  for (auto& future : started) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }

  std::atomic<size_t> queuedRuns{0};
  std::vector<folly::SemiFuture<folly::Unit>> queued;
  for (size_t requestIndex = 0; requestIndex < options.maxQueuedRequests;
       ++requestIndex) {
    queued.push_back(runtime.submit(
        "queued-" + std::to_string(requestIndex),
        [&queuedRuns](const folly::CancellationToken&) { ++queuedRuns; }));
  }
  auto overloaded =
      runtime.submit("overloaded", [](const folly::CancellationToken&) {});

  EXPECT_TRUE(overloaded.isReady());
  EXPECT_THROW(std::move(overloaded).get(kWaitTimeout), std::runtime_error);
  auto saturated = runtime.metrics();
  EXPECT_EQ(runtimeThreads.size(), options.numEventThreads);
  EXPECT_EQ(saturated.activeRequests, options.maxActiveRequests);
  EXPECT_EQ(saturated.queuedRequests, options.maxQueuedRequests);
  EXPECT_EQ(saturated.peakActiveRequests, options.maxActiveRequests);
  EXPECT_EQ(saturated.fiberStackBytes, options.fiberStackBytes);
  EXPECT_EQ(
      saturated.estimatedPeakFiberStackBytes,
      options.maxActiveRequests * options.fiberStackBytes);
  EXPECT_EQ(
      saturated.acceptedRequests,
      options.maxActiveRequests + options.maxQueuedRequests);
  EXPECT_EQ(saturated.overloadedRequests, 1);

  for (auto& release : releaseActive) {
    release.post();
  }
  for (auto& future : active) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }
  for (auto& future : queued) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }
  EXPECT_EQ(queuedRuns, options.maxQueuedRequests);
  auto completed = runtime.metrics();
  EXPECT_EQ(completed.activeRequests, 0);
  EXPECT_EQ(completed.queuedRequests, 0);
  EXPECT_EQ(
      completed.completedRequests,
      options.maxActiveRequests + options.maxQueuedRequests);
#ifndef FOLLY_SANITIZE_ADDRESS
  EXPECT_GT(completed.measuredFiberStackHighWatermarkBytes, 0);
  EXPECT_LT(
      completed.measuredFiberStackHighWatermarkBytes,
      completed.fiberStackBytes);
#else
  EXPECT_EQ(completed.measuredFiberStackHighWatermarkBytes, 0);
#endif
  std::cout << "STAGE2_BOUND_METRICS shards=" << completed.numShards
            << " peak_active=" << completed.peakActiveRequests
            << " max_queued=" << completed.maxQueuedRequests
            << " fiber_stack_bytes=" << completed.fiberStackBytes
            << " estimated_peak_fiber_stack_bytes="
            << completed.estimatedPeakFiberStackBytes
            << " measured_fiber_stack_high_watermark_bytes="
            << completed.measuredFiberStackHighWatermarkBytes << '\n';
}

TEST(AbfsAsyncRuntimeTest, RejectsOverflowingFiberStackCapacity) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = std::numeric_limits<size_t>::max();
  options.fiberStackBytes = 2;

  EXPECT_THROW(AbfsAsyncRuntime runtime(options), std::invalid_argument);
}

TEST(AbfsAsyncRuntimeTest, ShutdownCancelsActiveAndQueuedRequests) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 1;
  options.maxQueuedRequests = 1;
  AbfsAsyncRuntime runtime(options);
  auto startedContract = folly::makePromiseContract<folly::Unit>();

  auto active = runtime.submit(
      "account.dfs.core.windows.net",
      [promise = std::move(startedContract.promise)](
          const folly::CancellationToken& cancellationToken) mutable {
        promise.setValue();
        folly::fibers::Baton cancellationWait;
        while (!cancellationToken.isCancellationRequested()) {
          cancellationWait.timed_wait(std::chrono::milliseconds(10));
        }
      });
  std::move(startedContract.future).get(kWaitTimeout);
  std::atomic<size_t> queuedRuns{0};
  auto queued = runtime.submit(
      "account.dfs.core.windows.net",
      [&queuedRuns](const folly::CancellationToken&) { ++queuedRuns; });

  runtime.shutdown();

  EXPECT_THROW(std::move(active).get(kWaitTimeout), std::runtime_error);
  EXPECT_THROW(std::move(queued).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(queuedRuns, 0);
  auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.activeRequests, 0);
  EXPECT_EQ(metrics.queuedRequests, 0);
  EXPECT_EQ(metrics.completedRequests, 1);
  EXPECT_EQ(metrics.cancelledRequests, 2);
  EXPECT_NO_THROW(runtime.shutdown());
}

TEST(AbfsAsyncRuntimeTest, RetryDelayYieldsAndShutdownCancels) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 2;
  AbfsAsyncRuntime runtime(options);
  auto startedContract = folly::makePromiseContract<folly::Unit>();

  auto delayed = runtime.submit(
      "account.dfs.core.windows.net",
      [&runtime, promise = std::move(startedContract.promise)](
          const folly::CancellationToken&) mutable {
        promise.setValue();
        runtime.waitForRetryDelay(std::chrono::seconds(30));
      });
  std::move(startedContract.future).get(kWaitTimeout);

  std::atomic<size_t> unrelatedRuns{0};
  auto unrelated = runtime.submit(
      "account.dfs.core.windows.net",
      [&unrelatedRuns](const folly::CancellationToken&) { ++unrelatedRuns; });
  EXPECT_NO_THROW(std::move(unrelated).get(std::chrono::seconds(1)));
  EXPECT_EQ(unrelatedRuns, 1);

  const auto shutdownStart = std::chrono::steady_clock::now();
  runtime.shutdown();
  const auto shutdownDuration =
      std::chrono::steady_clock::now() - shutdownStart;

  EXPECT_LT(shutdownDuration, std::chrono::seconds(1));
  EXPECT_THROW(std::move(delayed).get(kWaitTimeout), std::runtime_error);
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.completedRequests, 2);
  EXPECT_EQ(metrics.cancelledRequests, 1);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshYieldsToSiblingRequest) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 2;
  AbfsAsyncRuntime runtime(options);
  auto authService = runtime.authService();
  const auto submittingThread = std::this_thread::get_id();
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::thread::id eventBaseThread;
  std::thread::id callbackThread;
  std::string returnedToken;

  auto refreshing = runtime.submit(
      "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
        eventBaseThread = std::this_thread::get_id();
        returnedToken = authService->refresh(
            AbfsAsyncAuthKey{"account", "filesystem", "path", "read"}, [&] {
              std::unique_lock lock(mutex);
              callbackThread = std::this_thread::get_id();
              callbackIsRunning = true;
              callbackStarted.notify_all();
              releaseCallback.wait(lock, [&] { return callbackIsReleased; });
              return std::string{"synthetic-token"};
            });
      });

  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  });

  std::atomic<size_t> siblingRuns{0};
  auto sibling = runtime.submit(
      "account.dfs.core.windows.net",
      [&siblingRuns](const folly::CancellationToken&) { ++siblingRuns; });
  EXPECT_NO_THROW(std::move(sibling).get(std::chrono::seconds(1)));
  EXPECT_EQ(siblingRuns, 1);

  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  releaseGuard.dismiss();
  EXPECT_NO_THROW(std::move(refreshing).get(kWaitTimeout));
  EXPECT_EQ(returnedToken, "synthetic-token");
  EXPECT_NE(callbackThread, submittingThread);
  EXPECT_NE(callbackThread, eventBaseThread);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshSharesOneCallbackForSameKey) {
  constexpr size_t kWaiters = 64;
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = kWaiters;
  AbfsAsyncRuntime runtime(options);
  auto authService = runtime.authService();
  const AbfsAsyncAuthKey key{"account", "filesystem", "path", "read"};
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::atomic<size_t> callbackRuns{0};
  std::array<std::string, kWaiters> returnedTokens;
  std::vector<folly::SemiFuture<folly::Unit>> refreshes;

  for (size_t waiter = 0; waiter < kWaiters; ++waiter) {
    refreshes.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&, waiter](const folly::CancellationToken&) {
          returnedTokens[waiter] = authService->refresh(key, [&] {
            ++callbackRuns;
            std::unique_lock lock(mutex);
            callbackIsRunning = true;
            callbackStarted.notify_all();
            releaseCallback.wait(lock, [&] { return callbackIsReleased; });
            return std::string{"shared-token"};
          });
        }));
  }
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }
  ASSERT_TRUE(waitForAuthWaiters(authService, kWaiters));
  EXPECT_EQ(callbackRuns, 1);

  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  releaseGuard.dismiss();
  for (auto& refresh : refreshes) {
    EXPECT_NO_THROW(std::move(refresh).get(kWaitTimeout));
  }
  EXPECT_EQ(callbackRuns, 1);
  for (const auto& token : returnedTokens) {
    EXPECT_EQ(token, "shared-token");
  }
  const auto metrics = authService->metrics();
  EXPECT_EQ(metrics.refreshCallbacks, 1);
  EXPECT_EQ(metrics.sharedRefreshes, kWaiters - 1);
  EXPECT_EQ(metrics.completedRefreshes, kWaiters);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshBoundsDistinctKeys) {
  AbfsAsyncRuntimeOptions options;
  options.numAuthThreads = 2;
  options.maxQueuedAuthRefreshes = 2;
  AbfsAsyncRuntime runtime(options);
  auto authService = runtime.authService();
  std::mutex mutex;
  std::condition_variable callbacksStarted;
  std::condition_variable releaseCallbacks;
  bool callbacksAreReleased{false};
  size_t runningCallbacks{0};
  std::atomic<size_t> callbackRuns{0};
  std::vector<folly::SemiFuture<folly::Unit>> refreshes;

  auto submitRefresh = [&](size_t keyIndex) {
    const AbfsAsyncAuthKey key{
        "account", "filesystem", "path-" + std::to_string(keyIndex), "read"};
    refreshes.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&, key](const folly::CancellationToken&) {
          authService->refresh(key, [&] {
            ++callbackRuns;
            std::unique_lock lock(mutex);
            ++runningCallbacks;
            callbacksStarted.notify_all();
            releaseCallbacks.wait(lock, [&] { return callbacksAreReleased; });
            return std::string{"synthetic-token"};
          });
        }));
  };

  submitRefresh(0);
  submitRefresh(1);
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbacksAreReleased = true;
    releaseCallbacks.notify_all();
  });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbacksStarted.wait_for(
        lock, kWaitTimeout, [&] { return runningCallbacks == 2; }));
  }
  submitRefresh(2);
  submitRefresh(3);
  ASSERT_TRUE(waitForAuthWaiters(authService, 4));
  const auto saturated = authService->metrics();
  EXPECT_EQ(saturated.numWorkers, 2);
  EXPECT_EQ(saturated.activeRefreshes, 2);
  EXPECT_EQ(saturated.queuedRefreshes, 2);
  EXPECT_EQ(saturated.inFlightRefreshes, 4);

  const AbfsAsyncAuthKey overloadedKey{
      "account", "filesystem", "overloaded-path", "read"};
  auto overloaded = runtime.submit(
      "account.dfs.core.windows.net",
      [authService, overloadedKey](const folly::CancellationToken&) {
        authService->refresh(
            overloadedKey, [] { return std::string{"unexpected-token"}; });
      });
  EXPECT_THROW(
      std::move(overloaded).get(std::chrono::seconds(1)), std::runtime_error);

  std::atomic<size_t> siblingRuns{0};
  auto sibling = runtime.submit(
      "account.dfs.core.windows.net",
      [&siblingRuns](const folly::CancellationToken&) { ++siblingRuns; });
  EXPECT_NO_THROW(std::move(sibling).get(std::chrono::seconds(1)));
  EXPECT_EQ(siblingRuns, 1);

  {
    std::lock_guard lock(mutex);
    callbacksAreReleased = true;
    releaseCallbacks.notify_all();
  }
  releaseGuard.dismiss();
  for (auto& refresh : refreshes) {
    EXPECT_NO_THROW(std::move(refresh).get(kWaitTimeout));
  }
  EXPECT_EQ(callbackRuns, 4);
  const auto completed = authService->metrics();
  EXPECT_EQ(completed.activeRefreshes, 0);
  EXPECT_EQ(completed.queuedRefreshes, 0);
  EXPECT_EQ(completed.inFlightRefreshes, 0);
  EXPECT_EQ(completed.waitingRefreshes, 0);
  EXPECT_EQ(completed.overloadedRefreshes, 1);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshFansOutCallbackException) {
  constexpr size_t kWaiters = 8;
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = kWaiters;
  AbfsAsyncRuntime runtime(options);
  auto authService = runtime.authService();
  const AbfsAsyncAuthKey key{"account", "filesystem", "path", "read"};
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::atomic<size_t> callbackRuns{0};
  std::vector<folly::SemiFuture<folly::Unit>> refreshes;

  for (size_t waiter = 0; waiter < kWaiters; ++waiter) {
    refreshes.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&, key](const folly::CancellationToken&) {
          authService->refresh(key, [&]() -> std::string {
            ++callbackRuns;
            std::unique_lock lock(mutex);
            callbackIsRunning = true;
            callbackStarted.notify_all();
            releaseCallback.wait(lock, [&] { return callbackIsReleased; });
            throw std::runtime_error("synthetic refresh failure");
          });
        }));
  }
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }
  ASSERT_TRUE(waitForAuthWaiters(authService, kWaiters));
  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  releaseGuard.dismiss();

  for (auto& refresh : refreshes) {
    EXPECT_THROW(std::move(refresh).get(kWaitTimeout), std::runtime_error);
  }
  EXPECT_EQ(callbackRuns, 1);
  EXPECT_EQ(authService->metrics().completedRefreshes, kWaiters);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshCancellationPreservesOtherWaiter) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto authService = runtime.authService();
  const AbfsAsyncAuthKey key{"account", "filesystem", "path", "read"};
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::atomic<size_t> callbackRuns{0};

  auto submitRefresh = [&] {
    return runtime.submit(
        "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
          authService->refresh(key, [&] {
            ++callbackRuns;
            std::unique_lock lock(mutex);
            callbackIsRunning = true;
            callbackStarted.notify_all();
            releaseCallback.wait(lock, [&] { return callbackIsReleased; });
            return std::string{"shared-token"};
          });
        });
  };
  auto cancelled = submitRefresh();
  auto successful = submitRefresh();
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }
  ASSERT_TRUE(waitForAuthWaiters(authService, 2));

  cancelled.cancel();
  EXPECT_THROW(
      std::move(cancelled).get(std::chrono::seconds(1)),
      folly::FutureCancellation);
  ASSERT_TRUE(waitForAuthWaiters(authService, 1));
  EXPECT_EQ(authService->metrics().cancelledRefreshes, 1);

  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  releaseGuard.dismiss();
  EXPECT_NO_THROW(std::move(successful).get(kWaitTimeout));
  EXPECT_EQ(callbackRuns, 1);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshDropsUnstartedCancelledKey) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto authService = runtime.authService();
  const AbfsAsyncAuthKey activeKey{
      "account", "filesystem", "active-path", "read"};
  const AbfsAsyncAuthKey queuedKey{
      "account", "filesystem", "queued-path", "read"};
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::atomic<size_t> queuedCallbackRuns{0};

  auto active = runtime.submit(
      "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
        authService->refresh(activeKey, [&] {
          std::unique_lock lock(mutex);
          callbackIsRunning = true;
          callbackStarted.notify_all();
          releaseCallback.wait(lock, [&] { return callbackIsReleased; });
          return std::string{"active-token"};
        });
      });
  auto releaseGuard = folly::makeGuard([&] {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }

  auto submitQueued = [&] {
    return runtime.submit(
        "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
          authService->refresh(queuedKey, [&] {
            ++queuedCallbackRuns;
            return std::string{"queued-token"};
          });
        });
  };
  auto firstQueued = submitQueued();
  auto secondQueued = submitQueued();
  ASSERT_TRUE(waitForAuthWaiters(authService, 3));
  EXPECT_EQ(authService->metrics().queuedRefreshes, 1);

  firstQueued.cancel();
  secondQueued.cancel();
  EXPECT_THROW(
      std::move(firstQueued).get(std::chrono::seconds(1)),
      folly::FutureCancellation);
  EXPECT_THROW(
      std::move(secondQueued).get(std::chrono::seconds(1)),
      folly::FutureCancellation);
  ASSERT_TRUE(waitForAuthWaiters(authService, 1));
  const auto cancelled = authService->metrics();
  EXPECT_EQ(cancelled.queuedRefreshes, 0);
  EXPECT_EQ(cancelled.inFlightRefreshes, 1);
  EXPECT_EQ(cancelled.cancelledRefreshes, 2);
  EXPECT_EQ(queuedCallbackRuns, 0);

  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  releaseGuard.dismiss();
  EXPECT_NO_THROW(std::move(active).get(kWaitTimeout));
  EXPECT_EQ(queuedCallbackRuns, 0);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshShutdownWaitsForExecutingCallback) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto authService = runtime.authService();
  const AbfsAsyncAuthKey activeKey{
      "account", "filesystem", "active-path", "read"};
  const AbfsAsyncAuthKey queuedKey{
      "account", "filesystem", "queued-path", "read"};
  std::mutex mutex;
  std::condition_variable callbackStarted;
  std::condition_variable releaseCallback;
  bool callbackIsRunning{false};
  bool callbackIsReleased{false};
  std::atomic<size_t> queuedCallbackRuns{0};
  std::atomic<bool> shutdownReturned{false};

  auto active = runtime.submit(
      "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
        authService->refresh(activeKey, [&] {
          std::unique_lock lock(mutex);
          callbackIsRunning = true;
          callbackStarted.notify_all();
          releaseCallback.wait(lock, [&] { return callbackIsReleased; });
          return std::string{"active-token"};
        });
      });
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(callbackStarted.wait_for(
        lock, kWaitTimeout, [&] { return callbackIsRunning; }));
  }
  auto queued = runtime.submit(
      "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
        authService->refresh(queuedKey, [&] {
          ++queuedCallbackRuns;
          return std::string{"queued-token"};
        });
      });
  ASSERT_TRUE(waitForAuthWaiters(authService, 2));

  std::thread shutdownThread([&] {
    runtime.shutdown();
    shutdownReturned = true;
  });
  auto shutdownGuard = folly::makeGuard([&] {
    {
      std::lock_guard lock(mutex);
      callbackIsReleased = true;
      releaseCallback.notify_all();
    }
    if (shutdownThread.joinable()) {
      shutdownThread.join();
    }
  });
  EXPECT_THROW(
      std::move(active).get(std::chrono::seconds(1)), std::runtime_error);
  EXPECT_THROW(
      std::move(queued).get(std::chrono::seconds(1)), std::runtime_error);
  EXPECT_FALSE(shutdownReturned);
  EXPECT_EQ(queuedCallbackRuns, 0);

  {
    std::lock_guard lock(mutex);
    callbackIsReleased = true;
    releaseCallback.notify_all();
  }
  shutdownThread.join();
  shutdownGuard.dismiss();
  EXPECT_TRUE(shutdownReturned);
  EXPECT_EQ(queuedCallbackRuns, 0);
  EXPECT_EQ(authService->metrics().cancelledRefreshes, 2);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshRejectsCallsOutsideRequestFiber) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto authService = runtime.authService();

  EXPECT_THROW(
      authService->refresh(
          AbfsAsyncAuthKey{"account", "filesystem", "path", "read"},
          [] { return std::string{"unexpected-token"}; }),
      std::logic_error);
}

TEST(AbfsAsyncRuntimeTest, AuthRefreshCompletionRacesCancellationAndShutdown) {
  constexpr size_t kIterations = 100;
  size_t successfulFutures{0};
  size_t cancelledFutures{0};

  for (size_t iteration = 0; iteration < kIterations; ++iteration) {
    AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
    auto authService = runtime.authService();
    std::mutex mutex;
    std::condition_variable callbackStarted;
    std::condition_variable releaseCallback;
    bool callbackIsRunning{false};
    bool callbackIsReleased{false};
    auto refreshing = runtime.submit(
        "account.dfs.core.windows.net", [&](const folly::CancellationToken&) {
          authService->refresh(
              AbfsAsyncAuthKey{"account", "filesystem", "path", "read"}, [&] {
                std::unique_lock lock(mutex);
                callbackIsRunning = true;
                callbackStarted.notify_all();
                releaseCallback.wait(lock, [&] { return callbackIsReleased; });
                return std::string{"synthetic-token"};
              });
        });
    {
      std::unique_lock lock(mutex);
      ASSERT_TRUE(callbackStarted.wait_for(
          lock, kWaitTimeout, [&] { return callbackIsRunning; }));
    }

    std::thread cancellationThread([&refreshing] { refreshing.cancel(); });
    std::thread shutdownThread([&runtime] { runtime.shutdown(); });
    {
      std::lock_guard lock(mutex);
      callbackIsReleased = true;
      releaseCallback.notify_all();
    }
    cancellationThread.join();
    shutdownThread.join();

    try {
      std::move(refreshing).get(kWaitTimeout);
      ++successfulFutures;
    } catch (const folly::FutureCancellation&) {
      ++cancelledFutures;
    } catch (const std::runtime_error&) {
      ++cancelledFutures;
    }
    const auto runtimeMetrics = runtime.metrics();
    EXPECT_EQ(runtimeMetrics.completedRequests, 1);
    EXPECT_LE(runtimeMetrics.cancelledRequests, 1);
    const auto authMetrics = authService->metrics();
    EXPECT_EQ(authMetrics.activeRefreshes, 0);
    EXPECT_EQ(authMetrics.queuedRefreshes, 0);
    EXPECT_EQ(authMetrics.inFlightRefreshes, 0);
    EXPECT_EQ(authMetrics.waitingRefreshes, 0);
  }

  EXPECT_EQ(successfulFutures + cancelledFutures, kIterations);
}

TEST(AbfsAsyncRuntimeTest, FutureCancellationInterruptsRetryDelay) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto startedContract = folly::makePromiseContract<folly::Unit>();

  auto delayed = runtime.submit(
      "account.dfs.core.windows.net",
      [&runtime, promise = std::move(startedContract.promise)](
          const folly::CancellationToken&) mutable {
        promise.setValue();
        runtime.waitForRetryDelay(std::chrono::seconds(30));
      });
  std::move(startedContract.future).get(kWaitTimeout);

  delayed.cancel();
  EXPECT_THROW(
      std::move(delayed).get(std::chrono::seconds(1)),
      folly::FutureCancellation);

  auto afterCancellation = runtime.submit(
      "account.dfs.core.windows.net", [](const folly::CancellationToken&) {});
  EXPECT_NO_THROW(std::move(afterCancellation).get(kWaitTimeout));
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.activeRequests, 0);
  EXPECT_EQ(metrics.completedRequests, 2);
  EXPECT_EQ(metrics.cancelledRequests, 1);
}

TEST(AbfsAsyncRuntimeTest, FutureCancellationRemovesQueuedRequest) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 1;
  options.maxQueuedRequests = 1;
  AbfsAsyncRuntime runtime(options);
  auto startedContract = folly::makePromiseContract<folly::Unit>();
  auto release = std::make_shared<folly::fibers::Baton>();

  auto active = runtime.submit(
      "account.dfs.core.windows.net",
      [release, promise = std::move(startedContract.promise)](
          const folly::CancellationToken&) mutable {
        promise.setValue();
        release->wait();
      });
  std::move(startedContract.future).get(kWaitTimeout);

  std::atomic<size_t> queuedRuns{0};
  auto queued = runtime.submit(
      "account.dfs.core.windows.net",
      [&queuedRuns](const folly::CancellationToken&) { ++queuedRuns; });
  queued.cancel();
  EXPECT_THROW(std::move(queued).get(kWaitTimeout), folly::FutureCancellation);

  release->post();
  EXPECT_NO_THROW(std::move(active).get(kWaitTimeout));
  EXPECT_EQ(queuedRuns, 0);
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.activeRequests, 0);
  EXPECT_EQ(metrics.queuedRequests, 0);
  EXPECT_EQ(metrics.completedRequests, 1);
  EXPECT_EQ(metrics.cancelledRequests, 1);
}

TEST(AbfsAsyncRuntimeTest, RetryDelayTimerExpires) {
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  std::atomic<int64_t> elapsedMilliseconds{0};

  auto delayed = runtime.submit(
      "account.dfs.core.windows.net",
      [&runtime, &elapsedMilliseconds](const folly::CancellationToken&) {
        const auto start = std::chrono::steady_clock::now();
        runtime.waitForRetryDelay(std::chrono::milliseconds(20));
        elapsedMilliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
      });

  EXPECT_NO_THROW(std::move(delayed).get(kWaitTimeout));
  EXPECT_GE(elapsedMilliseconds, 10);
  EXPECT_LT(elapsedMilliseconds, 1'000);
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.completedRequests, 1);
  EXPECT_EQ(metrics.cancelledRequests, 0);
}

TEST(AbfsAsyncRuntimeTest, RetryDelayExpiryRacesShutdown) {
  constexpr size_t kIterations = 100;
  size_t successfulRequests{0};
  size_t cancelledRequests{0};

  for (size_t iteration = 0; iteration < kIterations; ++iteration) {
    AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
    auto startedContract = folly::makePromiseContract<folly::Unit>();
    auto delayed = runtime.submit(
        "account.dfs.core.windows.net",
        [&runtime, promise = std::move(startedContract.promise)](
            const folly::CancellationToken&) mutable {
          promise.setValue();
          runtime.waitForRetryDelay(std::chrono::milliseconds(1));
        });
    std::move(startedContract.future).get(kWaitTimeout);

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    runtime.shutdown();
    try {
      std::move(delayed).get(kWaitTimeout);
      ++successfulRequests;
    } catch (const std::runtime_error&) {
      ++cancelledRequests;
    }

    const auto metrics = runtime.metrics();
    EXPECT_EQ(metrics.activeRequests, 0);
    EXPECT_EQ(metrics.completedRequests, 1);
    EXPECT_LE(metrics.cancelledRequests, 1);
  }

  EXPECT_EQ(successfulRequests + cancelledRequests, kIterations);
}

TEST(AbfsAsyncRuntimeTest, RetryDelayCancellationRacesExpiryAndShutdown) {
  constexpr size_t kIterations = 100;
  size_t completedFutures{0};
  size_t cancelledFutures{0};

  for (size_t iteration = 0; iteration < kIterations; ++iteration) {
    AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
    auto startedContract = folly::makePromiseContract<folly::Unit>();
    auto delayed = runtime.submit(
        "account.dfs.core.windows.net",
        [&runtime, promise = std::move(startedContract.promise)](
            const folly::CancellationToken&) mutable {
          promise.setValue();
          runtime.waitForRetryDelay(std::chrono::milliseconds(1));
        });
    std::move(startedContract.future).get(kWaitTimeout);

    std::thread cancelThread([&delayed] { delayed.cancel(); });
    runtime.shutdown();
    cancelThread.join();

    try {
      std::move(delayed).get(kWaitTimeout);
      ++completedFutures;
    } catch (const folly::FutureCancellation&) {
      ++cancelledFutures;
    } catch (const std::runtime_error&) {
      ++cancelledFutures;
    }

    const auto metrics = runtime.metrics();
    EXPECT_EQ(metrics.activeRequests, 0);
    EXPECT_EQ(metrics.completedRequests, 1);
    EXPECT_EQ(metrics.cancelledRequests, 1);
  }

  EXPECT_EQ(completedFutures + cancelledFutures, kIterations);
}

TEST(AbfsAsyncRuntimeTest, AzureRetryPolicyYieldsToUnrelatedRequest) {
  AbfsAsyncRuntimeOptions runtimeOptions;
  runtimeOptions.maxActiveRequests = 2;
  AbfsAsyncRuntime runtime(runtimeOptions);
  auto delayStartedContract = folly::makePromiseContract<folly::Unit>();
  auto attempts = std::make_shared<std::atomic<size_t>>(0);

  auto retried = runtime.submit(
      "account.dfs.core.windows.net",
      [&runtime, attempts, promise = std::move(delayStartedContract.promise)](
          const folly::CancellationToken&) mutable {
        Azure::Core::Http::Policies::RetryOptions retryOptions;
        retryOptions.MaxRetries = 1;
        retryOptions.RetryDelay = std::chrono::milliseconds(100);
        retryOptions.MaxRetryDelay = retryOptions.RetryDelay;
        retryOptions.RetryDelayCallback = [&runtime, &promise](
                                              std::chrono::milliseconds delay,
                                              const Azure::Core::Context&) {
          promise.setValue();
          runtime.waitForRetryDelay(delay);
        };

        std::vector<std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy>>
            policies;
        policies.emplace_back(
            std::make_unique<
                Azure::Core::Http::Policies::_internal::RetryPolicy>(
                retryOptions));
        policies.emplace_back(std::make_unique<RetrySequencePolicy>(attempts));
        Azure::Core::Http::_internal::HttpPipeline pipeline(policies);
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url("https://www.microsoft.com"));
        pipeline.Send(request, Azure::Core::Context{});
      });
  std::move(delayStartedContract.future).get(kWaitTimeout);

  std::atomic<size_t> unrelatedRuns{0};
  auto unrelated = runtime.submit(
      "account.dfs.core.windows.net",
      [&unrelatedRuns](const folly::CancellationToken&) { ++unrelatedRuns; });

  EXPECT_NO_THROW(std::move(unrelated).get(std::chrono::seconds(1)));
  EXPECT_EQ(unrelatedRuns, 1);
  EXPECT_NO_THROW(std::move(retried).get(kWaitTimeout));
  EXPECT_EQ(*attempts, 2);
}

TEST(AbfsAsyncRuntimeTest, DestroysLastOwnerFromRuntimeFiber) {
  auto runtime = std::make_shared<AbfsAsyncRuntime>(AbfsAsyncRuntimeOptions{});
  auto runtimeHolder =
      std::make_shared<std::shared_ptr<AbfsAsyncRuntime>>(runtime);
  auto future = runtime->submit(
      "account.dfs.core.windows.net",
      [runtimeHolder](const folly::CancellationToken& cancellationToken) {
        runtimeHolder->reset();
        folly::fibers::Baton cancellationWait;
        while (!cancellationToken.isCancellationRequested()) {
          cancellationWait.timed_wait(std::chrono::milliseconds(10));
        }
      });
  runtime.reset();

  EXPECT_THROW(std::move(future).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(*runtimeHolder, nullptr);
}

TEST(AbfsAsyncRuntimeTest, AssignsEndpointToOneRuntimeShard) {
  AbfsAsyncRuntimeOptions options;
  options.numEventThreads = 2;
  options.maxActiveRequests = 8;
  AbfsAsyncRuntime runtime(options);
  std::mutex mutex;
  std::set<std::thread::id> runtimeThreads;
  std::vector<folly::SemiFuture<folly::Unit>> futures;

  for (size_t requestIndex = 0; requestIndex < 8; ++requestIndex) {
    futures.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&mutex, &runtimeThreads](const folly::CancellationToken&) {
          if (!folly::fibers::onFiber()) {
            throw std::logic_error("ABFS runtime task did not run in a fiber");
          }
          std::lock_guard lock(mutex);
          runtimeThreads.insert(std::this_thread::get_id());
        }));
  }
  for (auto& future : futures) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }

  EXPECT_EQ(runtimeThreads.size(), 1);
  auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.numShards, 2);
  EXPECT_LE(metrics.peakActiveRequests, options.maxActiveRequests);
  EXPECT_EQ(metrics.completedRequests, futures.size());
}

TEST(AbfsAsyncRuntimeTest, OwnsEndpointTransportOnAssignedShard) {
  AbfsAsyncRuntimeOptions options;
  options.numEventThreads = 2;
  options.maxActiveRequests = 2;
  AbfsAsyncRuntime runtime(options);
  AbfsAsyncEndpointOptions endpoint;
  endpoint.endpointKey = "account.dfs.core.windows.net";
  endpoint.channelEndpoint.connectAddress =
      folly::SocketAddress("127.0.0.1", 80);
  endpoint.channelEndpoint.serverName = endpoint.endpointKey;
  std::mutex mutex;
  std::set<std::thread::id> runtimeThreads;
  std::set<FollyHttpTransport*> transports;

  std::vector<folly::SemiFuture<folly::Unit>> futures;
  for (size_t requestIndex = 0; requestIndex < 2; ++requestIndex) {
    futures.push_back(runtime.submit(
        endpoint,
        [&](FollyHttpTransport& transport, const folly::CancellationToken&) {
          std::lock_guard lock(mutex);
          runtimeThreads.insert(std::this_thread::get_id());
          transports.insert(&transport);
        }));
  }
  for (auto& future : futures) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }

  EXPECT_EQ(runtimeThreads.size(), 1);
  EXPECT_EQ(transports.size(), 1);
}

TEST(AbfsAsyncRuntimeTest, DestroysRejectedEndpointRequestOnAssignedShard) {
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 1;
  options.maxQueuedRequests = 0;
  AbfsAsyncRuntime runtime(options);
  AbfsAsyncEndpointOptions endpoint;
  endpoint.endpointKey = "rejected-endpoint-request";
  endpoint.channelEndpoint.connectAddress =
      folly::SocketAddress("127.0.0.1", 80);
  endpoint.channelEndpoint.serverName = endpoint.endpointKey;
  auto tracker = std::make_shared<RequestLifetimeTracker>();
  folly::fibers::Baton releaseActive;
  auto releaseGuard = folly::makeGuard([&] { releaseActive.post(); });
  auto started = folly::makePromiseContract<folly::Unit>();
  std::thread::id runtimeThread;

  auto active = runtime.submit(
      endpoint,
      [&runtimeThread, promise = std::move(started.promise), &releaseActive](
          FollyHttpTransport&, const folly::CancellationToken&) mutable {
        runtimeThread = std::this_thread::get_id();
        promise.setValue();
        releaseActive.wait();
      });
  std::move(started.future).get(kWaitTimeout);
  auto rejected = runtime.submit(
      endpoint,
      [requestState =
           std::make_shared<PrivateRequestState>(tracker, "rejected")](
          FollyHttpTransport&, const folly::CancellationToken&) {});

  EXPECT_TRUE(rejected.isReady());
  EXPECT_THROW(std::move(rejected).get(kWaitTimeout), std::runtime_error);
  ASSERT_TRUE(tracker->waitForCount(1));
  {
    std::lock_guard lock(tracker->mutex);
    ASSERT_EQ(tracker->destroyedRequests.size(), 1);
    EXPECT_EQ(tracker->destroyedRequests.front().first, "rejected");
    EXPECT_EQ(tracker->destroyedRequests.front().second, runtimeThread);
    EXPECT_NE(
        tracker->destroyedRequests.front().second, std::this_thread::get_id());
  }

  releaseActive.post();
  releaseGuard.dismiss();
  EXPECT_NO_THROW(std::move(active).get(kWaitTimeout));
}

TEST(AbfsAsyncRuntimeTest, SharesDnsResolutionOffRuntimeThreads) {
  auto resolver = std::make_shared<TestingResolver>();
  AbfsAsyncRuntimeOptions options;
  options.numEventThreads = 2;
  options.maxActiveRequests = 2;
  options.endpointResolver = resolver;
  AbfsAsyncRuntime runtime(options);
  AbfsAsyncEndpointOptions endpoint;
  endpoint.endpointKey = "shared-resolution";
  endpoint.hostname = "account.dfs.core.windows.net";
  endpoint.port = 443;
  endpoint.channelEndpoint.serverName = endpoint.hostname;
  std::mutex mutex;
  std::set<std::thread::id> runtimeThreads;
  const auto submittingThread = std::this_thread::get_id();

  std::vector<folly::SemiFuture<folly::Unit>> futures;
  for (size_t requestIndex = 0; requestIndex < 2; ++requestIndex) {
    futures.push_back(runtime.submit(
        endpoint, [&](FollyHttpTransport&, const folly::CancellationToken&) {
          std::lock_guard lock(mutex);
          runtimeThreads.insert(std::this_thread::get_id());
        }));
  }
  for (auto& future : futures) {
    EXPECT_NO_THROW(std::move(future).get(kWaitTimeout));
  }

  EXPECT_EQ(resolver->calls(endpoint.hostname), 1);
  ASSERT_EQ(runtimeThreads.size(), 1);
  const auto resolverThreads = resolver->resolverThreads();
  ASSERT_EQ(resolverThreads.size(), 1);
  EXPECT_NE(*resolverThreads.begin(), submittingThread);
  EXPECT_NE(*resolverThreads.begin(), *runtimeThreads.begin());
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.dnsCacheMisses, 1);
  EXPECT_EQ(metrics.dnsCacheHits, 1);
  EXPECT_EQ(metrics.dnsResolutions, 1);
}

TEST(AbfsAsyncRuntimeTest, ExpiresPositiveAndNegativeDnsEntries) {
  auto resolver = std::make_shared<TestingResolver>();
  auto now = std::chrono::steady_clock::time_point{};
  AbfsAsyncRuntimeOptions options;
  options.endpointResolver = resolver;
  options.dnsClock = [&now] { return now; };
  options.dnsCacheTtl = std::chrono::milliseconds(10);
  options.dnsFailureTtl = std::chrono::milliseconds(5);
  AbfsAsyncRuntime runtime(options);
  const auto success = unresolvedEndpoint("expiring-success", "success.test");
  const auto failure = unresolvedEndpoint("expiring-failure", "failure.test");
  resolver->setFailure(failure.hostname, true);
  auto task = [](FollyHttpTransport&, const folly::CancellationToken&) {};

  EXPECT_NO_THROW(runtime.submit(success, task).get(kWaitTimeout));
  now += options.dnsCacheTtl;
  EXPECT_NO_THROW(runtime.submit(success, task).get(kWaitTimeout));
  EXPECT_EQ(resolver->calls(success.hostname), 2);

  EXPECT_THROW(
      runtime.submit(failure, task).get(kWaitTimeout), std::runtime_error);
  EXPECT_THROW(
      runtime.submit(failure, task).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(resolver->calls(failure.hostname), 1);
  now += options.dnsFailureTtl;
  EXPECT_THROW(
      runtime.submit(failure, task).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(resolver->calls(failure.hostname), 2);

  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.dnsCacheMisses, 4);
  EXPECT_EQ(metrics.dnsCacheHits, 1);
  EXPECT_EQ(metrics.dnsCacheExpirations, 2);
  EXPECT_EQ(metrics.dnsResolutions, 4);
  EXPECT_EQ(metrics.dnsResolutionFailures, 2);
}

TEST(AbfsAsyncRuntimeTest, BoundsEndpointCacheAndResolverQueue) {
  auto resolver = std::make_shared<BlockingResolver>();
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 3;
  options.numResolverThreads = 1;
  options.maxQueuedResolutions = 1;
  options.maxEndpointCacheEntries = 2;
  options.endpointResolver = resolver;
  AbfsAsyncRuntime runtime(options);
  auto task = [](FollyHttpTransport&, const folly::CancellationToken&) {};

  auto active = runtime.submit(
      unresolvedEndpoint("active-resolution", "active.test"), task);
  ASSERT_TRUE(resolver->waitForCalls(1));
  auto queued = runtime.submit(
      unresolvedEndpoint("queued-resolution", "queued.test"), task);
  ASSERT_TRUE(waitForQueuedDnsResolutions(runtime, 1));
  auto rejected = runtime.submit(
      unresolvedEndpoint("rejected-resolution", "rejected.test"), task);

  EXPECT_TRUE(rejected.isReady());
  EXPECT_THROW(std::move(rejected).get(kWaitTimeout), std::runtime_error);
  const auto saturated = runtime.metrics();
  EXPECT_EQ(saturated.numEndpoints, 2);
  EXPECT_EQ(saturated.activeDnsResolutions, 1);
  EXPECT_EQ(saturated.queuedDnsResolutions, 1);
  EXPECT_EQ(saturated.endpointCacheRejections, 1);

  resolver->release();
  EXPECT_NO_THROW(std::move(active).get(kWaitTimeout));
  EXPECT_NO_THROW(std::move(queued).get(kWaitTimeout));

  auto third = runtime.submit(
      unresolvedEndpoint("third-resolution", "third.test"), task);
  EXPECT_NO_THROW(std::move(third).get(kWaitTimeout));
  const auto completed = runtime.metrics();
  EXPECT_EQ(completed.numEndpoints, 2);
  EXPECT_EQ(completed.endpointCacheEvictions, 1);
  EXPECT_EQ(completed.peakDnsResolutions, 1);
}

TEST(AbfsAsyncRuntimeTest, ShutdownCancelsQueuedDnsResolution) {
  auto resolver = std::make_shared<BlockingResolver>();
  AbfsAsyncRuntimeOptions options;
  options.maxActiveRequests = 2;
  options.numResolverThreads = 1;
  options.maxQueuedResolutions = 1;
  options.maxEndpointCacheEntries = 2;
  options.endpointResolver = resolver;
  AbfsAsyncRuntime runtime(options);
  auto task = [](FollyHttpTransport&, const folly::CancellationToken&) {};

  auto active = runtime.submit(
      unresolvedEndpoint("active-shutdown", "active.test"), task);
  ASSERT_TRUE(resolver->waitForCalls(1));
  auto queued = runtime.submit(
      unresolvedEndpoint("queued-shutdown", "queued.test"), task);
  ASSERT_TRUE(waitForQueuedDnsResolutions(runtime, 1));
  std::thread shutdownThread([&runtime] { runtime.shutdown(); });
  auto joinShutdown = folly::makeGuard([&] {
    resolver->release();
    shutdownThread.join();
  });

  EXPECT_THROW(std::move(queued).get(kWaitTimeout), std::runtime_error);
  resolver->release();
  shutdownThread.join();
  joinShutdown.dismiss();
  EXPECT_THROW(std::move(active).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(resolver->calls(), 1);
  const auto metrics = runtime.metrics();
  EXPECT_EQ(metrics.activeDnsResolutions, 0);
  EXPECT_EQ(metrics.queuedDnsResolutions, 0);
  EXPECT_EQ(metrics.dnsResolutions, 2);
  EXPECT_EQ(metrics.dnsResolutionFailures, 1);
  EXPECT_EQ(metrics.cancelledRequests, 2);
}

TEST(AbfsAsyncRuntimeTest, DestroysEndpointRequestsOnAssignedShard) {
  RuntimeEndpointServer server;
  server.start();
  AbfsAsyncRuntimeOptions options;
  options.numEventThreads = 2;
  options.maxActiveRequests = 2;
  options.maxQueuedRequests = 1;
  auto runtime = std::make_unique<AbfsAsyncRuntime>(options);
  AbfsAsyncEndpointOptions endpoint;
  endpoint.endpointKey = "runtime-owned-endpoint";
  endpoint.channelEndpoint.connectAddress = server.address();
  endpoint.channelEndpoint.serverName = "127.0.0.1";
  endpoint.httpTimeouts.connectionAcquire = std::chrono::seconds(2);
  endpoint.httpTimeouts.total = std::chrono::seconds(2);
  endpoint.maxConnections = 1;
  const auto requestUrl =
      "http://127.0.0.1:" + std::to_string(server.address().getPort()) +
      "/lifetime";
  auto tracker = std::make_shared<RequestLifetimeTracker>();
  auto bodyHeld = folly::makePromiseContract<folly::Unit>();
  auto waiterReady = folly::makePromiseContract<folly::Unit>();

  auto first = runtime->submit(
      endpoint,
      [requestState = std::make_shared<PrivateRequestState>(tracker, "body"),
       requestUrl,
       bodyPromise = std::move(bodyHeld.promise),
       waiterPromise = std::move(waiterReady.promise)](
          FollyHttpTransport& transport,
          const folly::CancellationToken& cancellationToken) mutable {
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get,
            Azure::Core::Url(requestUrl),
            false);
        auto response = transport.Send(request, Azure::Core::Context{});
        auto body = response->ExtractBodyStream();
        bodyPromise.setValue();
        folly::fibers::Baton delay;
        while (transport.poolMetrics().waitingFibers != 1) {
          delay.timed_wait(std::chrono::milliseconds(1));
        }
        waiterPromise.setValue();
        while (!cancellationToken.isCancellationRequested()) {
          delay.timed_wait(std::chrono::milliseconds(1));
        }
      });
  std::move(bodyHeld.future).get(kWaitTimeout);

  auto second = runtime->submit(
      endpoint,
      [requestState = std::make_shared<PrivateRequestState>(tracker, "waiter"),
       requestUrl](
          FollyHttpTransport& transport, const folly::CancellationToken&) {
        Azure::Core::Http::Request request(
            Azure::Core::Http::HttpMethod::Get, Azure::Core::Url(requestUrl));
        auto response = transport.Send(request, Azure::Core::Context{});
        if (response->GetBody() != (std::vector<uint8_t>{'b', 'o', 'd', 'y'})) {
          throw std::runtime_error("runtime endpoint response was invalid");
        }
      });
  std::move(waiterReady.future).get(kWaitTimeout);
  std::atomic<size_t> queuedRuns{0};
  auto queued = runtime->submit(
      endpoint,
      [requestState = std::make_shared<PrivateRequestState>(tracker, "queued"),
       &queuedRuns](FollyHttpTransport&, const folly::CancellationToken&) {
        ++queuedRuns;
      });
  EXPECT_EQ(runtime->metrics().queuedRequests, 1);

  runtime.reset();

  EXPECT_THROW(std::move(first).get(kWaitTimeout), std::runtime_error);
  EXPECT_THROW(std::move(second).get(kWaitTimeout), std::runtime_error);
  EXPECT_THROW(std::move(queued).get(kWaitTimeout), std::runtime_error);
  server.stop();
  EXPECT_FALSE(server.failed());
  EXPECT_TRUE(server.firstClientClosed());
  EXPECT_EQ(server.requests(), 2);
  EXPECT_EQ(queuedRuns, 0);
  std::lock_guard lock(tracker->mutex);
  ASSERT_EQ(tracker->destroyedRequests.size(), 3);
  std::set<std::string> destroyedNames;
  std::set<std::thread::id> destructionThreads;
  for (const auto& [name, threadId] : tracker->destroyedRequests) {
    destroyedNames.insert(name);
    destructionThreads.insert(threadId);
  }
  EXPECT_EQ(
      destroyedNames, (std::set<std::string>{"body", "queued", "waiter"}));
  EXPECT_EQ(destructionThreads.size(), 1);
  EXPECT_NE(*destructionThreads.begin(), std::this_thread::get_id());
}

} // namespace
} // namespace facebook::velox::filesystems
