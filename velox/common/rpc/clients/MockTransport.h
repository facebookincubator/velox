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

#include <atomic>
#include <chrono>
#include <random>

#include "velox/common/rpc/RPCTypes.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

namespace facebook::velox::rpc {

/// Transport double with configurable latency and error rate, for exercising
/// the async dispatch path without a real backend.
/// Thread-safe for concurrent use. Uses a thread pool executor for async
/// execution — either a shared executor passed in, or a local one created
/// per-client.
class MockTransport {
 public:
  /// Creates a mock client with configurable latency and error rate.
  /// @param latency Simulated RPC latency (default 200ms).
  /// @param errorRate Probability of error per request (0.0-1.0, default 0).
  /// @param executor Shared executor for async work. If nullptr, creates a
  ///   local thread pool. Pass a shared executor for global throttling across
  ///   query instances.
  explicit MockTransport(
      std::chrono::milliseconds latency = std::chrono::milliseconds(200),
      double errorRate = 0.0,
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor = nullptr);

  ~MockTransport();

  folly::SemiFuture<RPCResponse> call(const RPCRequest& request);

  folly::SemiFuture<std::vector<RPCResponse>> callBatch(
      const std::vector<RPCRequest>& requests);

  /// Returns the total number of RPC calls made.
  int64_t callCount() const {
    return callCount_.load();
  }

  /// Resets the call counter. An installed error burst stays installed and
  /// cannot be replaced, so ordinals restart at 0 and the same burst window
  /// applies again. Use a fresh client to test a different burst.
  void resetCallCount() {
    callCount_.store(0);
  }

  /// Configures a deterministic error burst for congestion / AIMD tests.
  /// Requests whose 0-based call ordinal falls in [firstCall, lastCall) are
  /// failed with a response tagged `errorKind`, simulating a timed
  /// backend-overload window (rate-limit / timeout). Ordinals are reserved on
  /// the caller thread, one per call() request and a contiguous run per
  /// callBatch() request, so the burst covers a fixed, timing-independent set
  /// of requests. Disabled when firstCall >= lastCall (the default).
  struct ErrorBurst {
    int64_t firstCall{0};
    int64_t lastCall{0};
    RPCErrorKind errorKind{RPCErrorKind::kRateLimited};
  };

  /// Installs the error burst. May be called at most once, and only before the
  /// first request is dispatched; thereafter the burst is only read, never
  /// mutated. Throws if called twice or after a request has been dispatched.
  void setErrorBurst(const ErrorBurst& burst);

 private:
  RPCResponse generateResponse(const RPCRequest& request, bool isError);

  // Returns the burst error kind for a request at 0-based `ordinal`, or kNone
  // when the ordinal is outside the configured burst window.
  RPCErrorKind burstErrorKind(int64_t ordinal) const;

  const std::chrono::milliseconds latency_;
  const double errorRate_;
  std::atomic<int64_t> callCount_{0};
  // Guards publication of errorBurst_ to the dispatch threads: written with
  // release in setErrorBurst(), read with acquire in burstErrorKind().
  std::atomic<bool> burstInstalled_{false};
  // Configured error-burst window; read only once burstInstalled_ is true.
  ErrorBurst errorBurst_{};

  /// Shared executor (may be shared across clients for global throttling).
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  /// Locally-owned executor (created when no shared executor is provided).
  std::shared_ptr<folly::CPUThreadPoolExecutor> ownedExecutor_;
};

} // namespace facebook::velox::rpc
