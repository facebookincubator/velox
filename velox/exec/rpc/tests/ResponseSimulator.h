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
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor = nullptr)
      : latency_(latency),
        ownedExecutor_(
            executor == nullptr
                ? std::make_shared<folly::CPUThreadPoolExecutor>(4)
                : nullptr),
        executor_(executor != nullptr ? executor : ownedExecutor_) {}

  /// Reserves one call ordinal and returns a future that fires after the
  /// simulated latency, carrying the failure kind for this call — kNone when
  /// it should succeed.
  folly::SemiFuture<velox::rpc::RPCErrorKind> nextCall() {
    const auto kind = outcomeFor(callCount_.fetch_add(1));
    return delayed(kind);
  }

  /// Reserves one call ordinal and returns `make(kind)` after the simulated
  /// latency, in one continuation. Callers that build a value from the outcome
  /// should prefer this over nextCall().deferValue(...), which costs a second
  /// continuation per call -- visible when the caller is a benchmark.
  template <typename F>
  auto nextCallMapped(F&& make)
      -> folly::SemiFuture<decltype(make(velox::rpc::RPCErrorKind::kNone))> {
    const auto kind = outcomeFor(callCount_.fetch_add(1));
    return delayedBy([kind = kind, make = std::forward<F>(make)]() mutable {
      return make(kind);
    });
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
    return delayed(std::move(kinds));
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
  // Runs 'produce' on the simulator's executor once the simulated latency has
  // elapsed, in a single continuation. Everything that waits goes through
  // here, so the per-call and per-batch paths cannot drift on how zero latency
  // is handled.
  template <typename F>
  auto delayedBy(F&& produce) -> folly::SemiFuture<decltype(produce())> {
    if (latency_ == std::chrono::milliseconds::zero()) {
      return folly::via(executor_.get(), std::forward<F>(produce)).semi();
    }
    return folly::futures::sleep(latency_)
        .via(executor_.get())
        .thenValue([produce = std::forward<F>(produce)](auto&&) mutable {
          return produce();
        })
        .semi();
  }

  template <typename T>
  folly::SemiFuture<T> delayed(T value) {
    return delayedBy(
        [value = std::move(value)]() mutable { return std::move(value); });
  }

  velox::rpc::RPCErrorKind outcomeFor(int64_t ordinal) const {
    if (burstInstalled_.load(std::memory_order_acquire) &&
        ordinal >= errorBurst_.firstCall && ordinal < errorBurst_.lastCall) {
      return errorBurst_.errorKind;
    }
    return velox::rpc::RPCErrorKind::kNone;
  }

  const std::chrono::milliseconds latency_;
  const std::shared_ptr<folly::CPUThreadPoolExecutor> ownedExecutor_;
  const std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::atomic<int64_t> callCount_{0};
  std::atomic<bool> burstInstalled_{false};
  ErrorBurst errorBurst_{};
};

} // namespace facebook::velox::exec::rpc::test
