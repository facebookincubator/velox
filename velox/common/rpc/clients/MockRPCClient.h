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

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/rpc/IRPCClient.h"

namespace facebook::velox::rpc {

/// Mock RPC client that simulates backend latency for testing.
/// Thread-safe for concurrent use. Uses a thread pool executor for async
/// execution — either a shared executor passed in, or a local one created
/// per-client.
class MockRPCClient : public IRPCClient {
 public:
  /// Creates a mock client with configurable latency and error rate.
  /// @param latency Simulated RPC latency (default 200ms).
  /// @param errorRate Probability of error per request (0.0-1.0, default 0).
  /// @param executor Shared executor for async work. If nullptr, creates a
  ///   local thread pool. Pass a shared executor for global throttling across
  ///   query instances.
  explicit MockRPCClient(
      std::chrono::milliseconds latency = std::chrono::milliseconds(200),
      double errorRate = 0.0,
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor = nullptr);

  ~MockRPCClient() override;

  folly::SemiFuture<RPCResponse> call(const RPCRequest& request) override;

  folly::SemiFuture<std::vector<RPCResponse>> callBatch(
      const std::vector<RPCRequest>& requests) override;

  /// Returns the total number of RPC calls made.
  int64_t callCount() const {
    return callCount_.load();
  }

  /// Resets the call counter.
  void resetCallCount() {
    callCount_.store(0);
  }

 private:
  RPCResponse generateResponse(const RPCRequest& request, bool isError);

  const std::chrono::milliseconds latency_;
  const double errorRate_;
  std::atomic<int64_t> callCount_{0};

  /// Shared executor (may be shared across clients for global throttling).
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  /// Locally-owned executor (created when no shared executor is provided).
  std::shared_ptr<folly::CPUThreadPoolExecutor> ownedExecutor_;
};

} // namespace facebook::velox::rpc
