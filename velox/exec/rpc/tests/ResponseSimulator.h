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
  /// Defines a deterministic failure window for congestion and AIMD tests.
  /// Calls whose
  /// 0-based ordinal falls in [firstCall, lastCall) fail with `errorKind`.
  /// Ordinals are reserved on the caller thread — one per call, a contiguous
  /// run per batch — so the window covers a fixed, timing-independent set of
  /// calls. Disabled when firstCall >= lastCall.
  struct ErrorBurst {
    /// Marks the first failing call ordinal.
    int64_t firstCall{0};
    /// Marks the first succeeding call ordinal after the burst.
    int64_t lastCall{0};
    /// Classifies calls inside the failure window.
    velox::rpc::RPCErrorKind errorKind{velox::rpc::RPCErrorKind::kRateLimited};
  };

  /// Constructs a simulator with the requested completion latency.
  explicit ResponseSimulator(
      std::chrono::milliseconds latency = std::chrono::milliseconds(200),
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor = nullptr);

  /// Reserves one call ordinal and returns a future that fires after the
  /// simulated latency, carrying the failure kind for this call — kNone when
  /// it should succeed.
  folly::SemiFuture<velox::rpc::RPCErrorKind> nextCall();

  /// Reserves one call ordinal and returns `make(kind)` after the simulated
  /// latency, in one continuation. Callers that build a value from the outcome
  /// should prefer this over nextCall().deferValue(...), which costs a second
  /// continuation per call -- visible when the caller is a benchmark.
  template <typename Factory>
  auto nextCallMapped(Factory&& make)
      -> folly::SemiFuture<decltype(make(velox::rpc::RPCErrorKind::kNone))> {
    const auto kind = outcomeFor(callCount_.fetch_add(1));
    return delayedBy(
        [kind = kind, make = std::forward<Factory>(make)]() mutable {
          return make(kind);
        });
  }

  /// Reserves `count` contiguous ordinals for a single batch, and returns the
  /// per-call outcomes after the simulated latency.
  folly::SemiFuture<std::vector<velox::rpc::RPCErrorKind>> nextBatch(
      size_t numCalls);

  /// Returns the number of call ordinals reserved so far.
  int64_t callCount() const;

  /// Resets the call counter. An installed burst stays installed, so ordinals
  /// restart at 0 and the same window applies again. Use a fresh simulator to
  /// test a different burst.
  void resetCallCount();

  /// Installs the failure window. May be called at most once, and only before
  /// the first call is reserved.
  void setErrorBurst(const ErrorBurst& burst);

 private:
  // Runs 'produce' on the simulator's executor once the simulated latency has
  // elapsed, in a single continuation. Everything that waits goes through
  // here, so the per-call and per-batch paths cannot drift on how zero latency
  // is handled.
  template <typename Producer>
  auto delayedBy(Producer&& produce) -> folly::SemiFuture<decltype(produce())> {
    if (latency_ == std::chrono::milliseconds::zero()) {
      return folly::via(executor_.get(), std::forward<Producer>(produce))
          .semi();
    }
    return folly::futures::sleep(latency_)
        .via(executor_.get())
        .thenValue([produce = std::forward<Producer>(produce)](auto&&) mutable {
          return produce();
        })
        .semi();
  }

  template <typename T>
  folly::SemiFuture<T> delayed(T value) {
    return delayedBy(
        [value = std::move(value)]() mutable { return std::move(value); });
  }

  // Returns the configured outcome for one reserved call ordinal.
  velox::rpc::RPCErrorKind outcomeFor(int64_t ordinal) const;

  // Delays each simulated completion by this duration.
  const std::chrono::milliseconds latency_;
  // Owns the executor when the caller does not provide one.
  const std::shared_ptr<folly::CPUThreadPoolExecutor> ownedExecutor_;
  // Schedules simulated completions.
  const std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  // Assigns deterministic ordinals across individual and batch calls.
  std::atomic<int64_t> callCount_{0};
  // Prevents replacing a burst after concurrent calls can observe it.
  std::atomic<bool> burstInstalled_{false};
  // Defines the optional deterministic failure window.
  ErrorBurst errorBurst_{};
};

} // namespace facebook::velox::exec::rpc::test
