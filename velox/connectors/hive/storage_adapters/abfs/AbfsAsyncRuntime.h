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

#include "velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h"
#include "velox/connectors/hive/storage_adapters/abfs/HttpConnection.h"

#include <folly/CancellationToken.h>
#include <folly/Function.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace facebook::velox::filesystems {

class FollyHttpTransport;

/// Identifies one dynamic authentication refresh without ambiguous encoding.
struct AbfsAsyncAuthKey {
  /// Identifies the storage account.
  std::string account;
  /// Identifies the filesystem or container.
  std::string fileSystem;
  /// Identifies the path within the filesystem.
  std::string path;
  /// Identifies the operation requiring authorization.
  std::string operation;

  /// Compares the complete semantic refresh identity.
  bool operator==(const AbfsAsyncAuthKey&) const = default;
};

/// Runs blocking authentication refresh callbacks outside EventBase threads.
class AbfsAsyncAuthService final {
 public:
  /// Produces one refreshed authorization token.
  using RefreshCallback = folly::Function<std::string()>;

  /// Reports an immutable snapshot of authentication refresh state.
  struct Metrics {
    /// Holds the configured authentication worker count.
    size_t numWorkers{0};
    /// Holds the configured distinct-refresh queue bound.
    size_t maxQueuedRefreshes{0};
    /// Counts callbacks currently executing on authentication workers.
    size_t activeRefreshes{0};
    /// Counts distinct refreshes waiting for an authentication worker.
    size_t queuedRefreshes{0};
    /// Counts distinct queued or executing refresh keys.
    size_t inFlightRefreshes{0};
    /// Counts fibers waiting for refresh results.
    size_t waitingRefreshes{0};
    /// Counts refresh callbacks started by workers.
    size_t refreshCallbacks{0};
    /// Counts fibers joined to an existing keyed refresh.
    size_t sharedRefreshes{0};
    /// Counts fibers delivered a callback result or exception.
    size_t completedRefreshes{0};
    /// Counts distinct refreshes rejected at the queue bound.
    size_t overloadedRefreshes{0};
    /// Counts refresh waiters cancelled before callback completion.
    size_t cancelledRefreshes{0};
  };

  /// Starts a bounded set of blocking authentication workers.
  AbfsAsyncAuthService(size_t numThreads, size_t maxQueuedRefreshes);

  /// Cancels queued refreshes and joins all authentication workers.
  ~AbfsAsyncAuthService();

  AbfsAsyncAuthService(const AbfsAsyncAuthService&) = delete;
  AbfsAsyncAuthService& operator=(const AbfsAsyncAuthService&) = delete;

  /// Runs or joins a keyed refresh while suspending only the current fiber.
  std::string refresh(const AbfsAsyncAuthKey& key, RefreshCallback callback);

  /// Returns a thread-safe snapshot of refresh and queue counters.
  Metrics metrics() const;

  /// Cancels queued refreshes and waits for executing callbacks to return.
  void shutdown();

 private:
  class State;

  /// Keeps worker and refresh state alive through callback completion.
  std::shared_ptr<State> state_;
};

/// Resolves endpoint hostnames outside runtime EventBase threads.
class AbfsEndpointResolver {
 public:
  /// Destroys the resolver interface.
  virtual ~AbfsEndpointResolver() = default;

  /// Resolves one hostname and port synchronously on a resolver worker.
  virtual folly::SocketAddress resolve(
      std::string_view host,
      uint16_t port) = 0;
};

/// Configures bounded execution resources for one ABFS filesystem instance.
struct AbfsAsyncRuntimeOptions {
  /// Sets the number of EventBase and FiberManager shards.
  size_t numEventThreads{1};
  /// Bounds requests executing in fibers across all shards.
  size_t maxActiveRequests{64};
  /// Bounds accepted requests waiting for an active slot.
  size_t maxQueuedRequests{1'024};
  /// Sets the stack size of each active request fiber.
  size_t fiberStackBytes{256 * 1'024};
  /// Samples exact stack use for every Nth fiber, or disables sampling at zero.
  size_t recordFiberStackEvery{0};
  /// Sets the number of threads allowed to block in DNS resolution.
  size_t numResolverThreads{1};
  /// Bounds DNS resolutions waiting for a resolver thread.
  size_t maxQueuedResolutions{64};
  /// Sets the number of threads allowed to block in authentication refresh.
  size_t numAuthThreads{1};
  /// Bounds authentication refreshes waiting for a worker.
  size_t maxQueuedAuthRefreshes{64};
  /// Bounds endpoint states and DNS entries retained by the runtime.
  size_t maxEndpointCacheEntries{256};
  /// Sets the lifetime of a successful DNS result.
  std::chrono::milliseconds dnsCacheTtl{std::chrono::minutes(5)};
  /// Sets the lifetime of a failed DNS result.
  std::chrono::milliseconds dnsFailureTtl{std::chrono::seconds(5)};
  /// Overrides synchronous hostname resolution for tests.
  std::shared_ptr<AbfsEndpointResolver> endpointResolver;
  /// Overrides the thread-safe monotonic DNS cache clock for tests.
  std::function<std::chrono::steady_clock::time_point()> dnsClock;
};

/// Configures one runtime-owned HTTP endpoint.
struct AbfsAsyncEndpointOptions {
  /// Identifies an endpoint within one runtime.
  std::string endpointKey;
  /// Supplies a hostname to resolve, or stays empty for a numeric endpoint.
  std::string hostname;
  /// Supplies the service port when hostname resolution is requested.
  uint16_t port{0};
  /// Supplies the numeric address or HTTP identity used for connections.
  AsyncChannelEndpoint channelEndpoint;
  /// Bounds HTTP parser and buffering resources.
  HttpLimits httpLimits;
  /// Bounds HTTP transaction and connection-pool lifetimes.
  HttpTimeouts httpTimeouts;
  /// Bounds physical connections owned by this endpoint.
  size_t maxConnections{16};
};

/// Runs synchronous Azure operations in a bounded set of EventBase fibers.
class AbfsAsyncRuntime final {
 public:
  /// Runs one admitted request and observes cooperative runtime cancellation.
  using Task = folly::Function<void(const folly::CancellationToken&)>;

  /// Runs one request using its runtime-owned endpoint transport.
  using EndpointTask = folly::Function<
      void(FollyHttpTransport&, const folly::CancellationToken&)>;

  /// Reports an immutable snapshot of runtime admission state.
  struct Metrics {
    /// Holds the number of EventBase and FiberManager shards.
    size_t numShards{0};
    /// Holds the configured active-request limit.
    size_t maxActiveRequests{0};
    /// Holds the configured inactive-queue limit.
    size_t maxQueuedRequests{0};
    /// Holds the configured stack size of each active request fiber.
    size_t fiberStackBytes{0};
    /// Counts endpoint states currently owned by the runtime.
    size_t numEndpoints{0};
    /// Counts resolver tasks currently executing.
    size_t activeDnsResolutions{0};
    /// Counts resolver tasks waiting for a worker.
    size_t queuedDnsResolutions{0};
    /// Records the largest simultaneous resolver-task count.
    size_t peakDnsResolutions{0};
    /// Counts endpoint cache lookups served by an existing entry.
    size_t dnsCacheHits{0};
    /// Counts endpoint cache lookups that created an entry.
    size_t dnsCacheMisses{0};
    /// Counts endpoint entries removed to enforce the cache bound.
    size_t endpointCacheEvictions{0};
    /// Counts endpoints rejected while all cache entries are resolving.
    size_t endpointCacheRejections{0};
    /// Counts expired endpoint entries replaced by a later lookup.
    size_t dnsCacheExpirations{0};
    /// Counts DNS jobs accepted by the resolver executor.
    size_t dnsResolutions{0};
    /// Counts failed, rejected, or shutdown-cancelled DNS jobs.
    size_t dnsResolutionFailures{0};
    /// Counts requests currently executing in fibers.
    size_t activeRequests{0};
    /// Counts accepted requests waiting for an active slot.
    size_t queuedRequests{0};
    /// Records the largest simultaneous active-request count.
    size_t peakActiveRequests{0};
    /// Estimates peak stack capacity from configured size and active fibers.
    size_t estimatedPeakFiberStackBytes{0};
    /// Records the largest sampled fiber stack use in bytes.
    size_t measuredFiberStackHighWatermarkBytes{0};
    /// Counts requests accepted for execution or queueing.
    size_t acceptedRequests{0};
    /// Counts submissions rejected because the runtime was full.
    size_t overloadedRequests{0};
    /// Counts active requests that reached terminal completion.
    size_t completedRequests{0};
    /// Counts queued or active requests cancelled during shutdown.
    size_t cancelledRequests{0};
  };

  /// Starts a config-scoped runtime with validated resource bounds.
  explicit AbfsAsyncRuntime(AbfsAsyncRuntimeOptions options);

  /// Cancels admitted work and joins all runtime threads.
  ~AbfsAsyncRuntime();

  AbfsAsyncRuntime(const AbfsAsyncRuntime&) = delete;
  AbfsAsyncRuntime& operator=(const AbfsAsyncRuntime&) = delete;

  /// Enqueues work without blocking. Cancelling the future cancels the request.
  folly::SemiFuture<folly::Unit> submit(std::string endpointKey, Task task);

  /// Enqueues endpoint work. Cancelling the future cancels the request.
  folly::SemiFuture<folly::Unit> submit(
      AbfsAsyncEndpointOptions endpoint,
      EndpointTask task);

  /// Cancels active work, fails queued work, and waits for terminal completion.
  void shutdown();

  /// Waits cooperatively for a retry delay from an active runtime request.
  void waitForRetryDelay(std::chrono::milliseconds delay) const;

  /// Returns the runtime-owned bounded authentication service.
  std::shared_ptr<AbfsAsyncAuthService> authService() const;

  /// Returns a thread-safe snapshot of admission and completion counters.
  Metrics metrics() const;

  /// Returns true when called from one of this runtime's EventBase threads.
  bool isRuntimeThread() const;

 private:
  class RuntimeState;

  /// Keeps request state alive until shutdown and all callbacks complete.
  std::shared_ptr<RuntimeState> state_;
};

} // namespace facebook::velox::filesystems
