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
#include <memory>
#include <random>
#include <vector>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/rpc/RPCTypes.h"

namespace facebook::velox::exec::rpc::test {

/// Simulates backend timing and failure for functions under test.
///
/// It decides *when* a call completes and *whether* that call fails. It never
/// decides what a response contains — each function builds its own responses,
/// so this is a policy the tests share rather than a backend they share.
///
/// Thread-safe. Uses a shared executor when given one, otherwise a local
/// thread pool.
class ResponseSimulator {
 public:
  /// A deterministic failure window, for congestion / AIMD tests. Calls whose
  /// 0-based ordinal falls in [firstCall, lastCall) fail with `errorKind`.
  /// Ordinals are reserved on the caller thread — one per call, a contiguous
  /// run per batch — so the window covers a fixed, timing-independent set of
  /// calls. Disabled when firstCall >= lastCall.
  struct ErrorBurst {
    int64_t firstCall{0};
    int64_t lastCall{0};
    velox::rpc::RPCErrorKind errorKind{velox::rpc::RPCErrorKind::kRateLimited};
  };

  explicit ResponseSimulator(
      std::chrono::milliseconds latency = std::chrono::milliseconds(200),
      double errorRate = 0.0,
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor = nullptr)
      : latency_(latency),
        errorRate_(errorRate),
        ownedExecutor_(
            executor == nullptr
                ? std::make_shared<folly::CPUThreadPoolExecutor>(4)
                : nullptr),
        executor_(executor != nullptr ? std::move(executor) : ownedExecutor_) {}

  /// Reserves one call ordinal and returns a future that fires after the
  /// simulated latency, carrying the failure kind for this call — kNone when
  /// it should succeed.
  folly::SemiFuture<velox::rpc::RPCErrorKind> nextCall() {
    const auto kind = outcomeFor(callCount_.fetch_add(1));
    return delayed(kind);
  }

  /// Reserves `count` contiguous ordinals for a single batch, and returns the
  /// per-call outcomes after the simulated latency.
  folly::SemiFuture<std::vector<velox::rpc::RPCErrorKind>> nextBatch(
      size_t count) {
    const auto first = callCount_.fetch_add(static_cast<int64_t>(count));
    std::vector<velox::rpc::RPCErrorKind> kinds;
    kinds.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      kinds.push_back(outcomeFor(first + static_cast<int64_t>(i)));
    }
    if (latency_ == std::chrono::milliseconds::zero()) {
      return folly::via(
                 executor_.get(),
                 [kinds = std::move(kinds)]() mutable { return kinds; })
          .semi();
    }
    return folly::futures::sleep(latency_)
        .via(executor_.get())
        .thenValue([kinds = std::move(kinds)](auto&&) { return kinds; })
        .semi();
  }

  int64_t callCount() const {
    return callCount_.load();
  }

  /// Resets the call counter. An installed burst stays installed, so ordinals
  /// restart at 0 and the same window applies again. Use a fresh simulator to
  /// test a different burst.
  void resetCallCount() {
    callCount_.store(0);
  }

  /// Installs the failure window. May be called at most once, and only before
  /// the first call is reserved.
  void setErrorBurst(const ErrorBurst& burst) {
    VELOX_CHECK(
        !burstInstalled_.load(std::memory_order_acquire) &&
            callCount_.load() == 0,
        "ResponseSimulator: error burst must be installed once, before the "
        "first call");
    errorBurst_ = burst;
    burstInstalled_.store(true, std::memory_order_release);
  }

 private:
  folly::SemiFuture<velox::rpc::RPCErrorKind> delayed(
      velox::rpc::RPCErrorKind kind) {
    if (latency_ == std::chrono::milliseconds::zero()) {
      return folly::via(executor_.get(), [kind] { return kind; }).semi();
    }
    return folly::futures::sleep(latency_)
        .via(executor_.get())
        .thenValue([kind](auto&&) { return kind; })
        .semi();
  }

  velox::rpc::RPCErrorKind outcomeFor(int64_t ordinal) {
    if (burstInstalled_.load(std::memory_order_acquire) &&
        ordinal >= errorBurst_.firstCall && ordinal < errorBurst_.lastCall) {
      return errorBurst_.errorKind;
    }
    if (errorRate_ > 0.0) {
      thread_local std::mt19937 rng{std::random_device{}()};
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      if (dist(rng) < errorRate_) {
        return velox::rpc::RPCErrorKind::kBackendError;
      }
    }
    return velox::rpc::RPCErrorKind::kNone;
  }

  const std::chrono::milliseconds latency_;
  const double errorRate_;
  const std::shared_ptr<folly::CPUThreadPoolExecutor> ownedExecutor_;
  const std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::atomic<int64_t> callCount_{0};
  std::atomic<bool> burstInstalled_{false};
  ErrorBurst errorBurst_{};
};

} // namespace facebook::velox::exec::rpc::test
