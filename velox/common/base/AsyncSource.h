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

#include <fmt/core.h>
#include <folly/Unit.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Future.h>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>
#include "velox/common/time/CpuWallTimer.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/common/process/ThreadDebugInfo.h"
#include "velox/common/testutil/TestValue.h"

namespace facebook::velox {

/// A future-like object that prefabricates Items on an executor and
/// allows consumer threads to pick items as they are ready. If the
/// consumer needs the item before the executor started making it, the
/// consumer will make it instead. If multiple consumers request the
/// same item, exactly one gets it. Propagates exceptions to the
/// consumer.
template <typename Item>
class AsyncSource {
 public:
  explicit AsyncSource(std::function<std::unique_ptr<Item>()> itemMaker)
      : itemMaker_(std::move(itemMaker)) {
    VELOX_CHECK_NOT_NULL(itemMaker_);
    if (process::GetThreadDebugInfo() != nullptr) {
      auto* currentThreadDebugInfo = process::GetThreadDebugInfo();
      // We explicitly leave out the callback when copying the ThreadDebugInfo
      // as that may have captured state that goes out of scope by the time
      // itemMaker_ is called.
      threadDebugInfo_ = std::make_optional<process::ThreadDebugInfo>(
          {currentThreadDebugInfo->queryId_,
           currentThreadDebugInfo->taskId_,
           nullptr});
    }
  }

  ~AsyncSource() {
    const auto currentState = state();
    VELOX_CHECK(
        currentState == State::kFinished || currentState == State::kCancelled ||
            currentState == State::kFailed,
        "AsyncSource should be properly finished, cancelled, or failed, unexpected state: {}",
        stateName(currentState));
  }

  /// Makes an item if it is not already made. To be called on a background
  /// executor.
  void prepare() {
    common::testutil::TestValue::adjust(
        "facebook::velox::AsyncSource::prepare", this);
    std::function<std::unique_ptr<Item>()> itemMaker{nullptr};
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (state() != State::kInit) {
        VELOX_CHECK_NULL(itemMaker_);
        return;
      }
      setState(State::kMaking);
      std::swap(itemMaker, itemMaker_);
    }
    makeItem(std::move(itemMaker));
  }

  /// Returns the item to the first caller and nullptr to subsequent callers.
  /// If the item is preparing on the executor, waits for the item and
  /// otherwise makes it on the caller thread.
  std::unique_ptr<Item> move() {
    common::testutil::TestValue::adjust(
        "facebook::velox::AsyncSource::move", this);
    auto currentState = state();
    std::function<std::unique_ptr<Item>()> itemMaker{nullptr};
    ContinueFuture wait;
    {
      std::lock_guard<std::mutex> l(mutex_);
      const auto currentState = state();
      switch (currentState) {
        case State::kFinished:
        case State::kCancelled:
          return nullptr;
        case State::kFailed:
          VELOX_CHECK_NOT_NULL(exception_);
          std::rethrow_exception(exception_);
        case State::kPrepared:
          VELOX_CHECK(promises_.empty());
          setState(State::kFinished);
          return std::move(item_);
        case State::kMaking:
          VELOX_CHECK_NULL(itemMaker_);
          VELOX_CHECK_NULL(item_);
          if (!promises_.empty()) {
            // Somebody else is already waiting for the item to be made.
            return nullptr;
          }
          promises_.emplace_back("AsyncSource::move");
          wait = promises_.back().getSemiFuture();
          break;
        case State::kInit:
          VELOX_CHECK_NOT_NULL(itemMaker_);
          VELOX_CHECK_NULL(item_);
          setState(State::kMaking);
          std::swap(itemMaker, itemMaker_);
          break;
      }
    }
    // Outside of mutex_.
    if (itemMaker != nullptr) {
      makeItem(std::move(itemMaker));
    }

    makeWait(std::move(wait));

    std::lock_guard<std::mutex> l(mutex_);
    currentState = state();
    if (exception_ != nullptr) {
      checkState(currentState, State::kFailed);
      std::rethrow_exception(exception_);
    }
    if (currentState == State::kFinished) {
      // Another move() or close() might have grabbed the item first.
      return nullptr;
    }
    checkState(currentState, State::kPrepared);
    setState(State::kFinished);
    return std::move(item_);
  }

  /// If true, move() will not block. But there is no guarantee that somebody
  /// else will not get the item first.
  bool hasValue() const {
    tsan_lock_guard<std::mutex> l(mutex_);
    return item_ != nullptr || exception_ != nullptr;
  }

  /// Returns the timing of making the item. If the item was made on the
  /// calling thread, the timing is 0 since only off-thread activity needs to
  /// be added to the caller's timing.
  const CpuWallTiming& prepareTiming() const {
    return makeTiming_;
  }

  /// Cancels the task if it hasn't started yet or if item is already prepared.
  /// If the task is making, the task will continue but AsyncSource
  /// is marked as cancelled to allow proper cleanup in destructor.
  void cancel() {
    std::lock_guard<std::mutex> l(mutex_);
    const auto currentState = state();
    switch (currentState) {
      case State::kInit:
        VELOX_CHECK_NOT_NULL(itemMaker_);
        VELOX_CHECK_NULL(item_);
        itemMaker_ = nullptr;
        setState(State::kCancelled);
        return;
      case State::kPrepared:
        VELOX_CHECK_NULL(itemMaker_);
        item_ = nullptr;
        setState(State::kCancelled);
        return;
      default:
        return;
    }
  }

  /// This function assists the caller in ensuring that resources allocated in
  /// AsyncSource are promptly released:
  /// 1. Waits for the completion of the 'itemMaker_' function if it is
  /// executing in the thread pool.
  /// 2. Resets the 'itemMaker_' function if it has not started yet.
  /// 3. Cleans up the 'item_' if 'itemMaker_' has completed, but the result
  /// 'item_' has not been returned to the caller.
  void close() {
    auto currentState = state();
    if (currentState == State::kFinished || currentState == State::kFailed ||
        currentState == State::kCancelled) {
      return;
    }
    ContinueFuture wait;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (tryCloseLocked()) {
        return;
      }
      checkState(state(), State::kMaking);
      promises_.emplace_back("AsyncSource::close");
      wait = promises_.back().getSemiFuture();
    }

    makeWait(std::move(wait));

    {
      std::lock_guard<std::mutex> l(mutex_);
      const auto closed = tryCloseLocked();
      VELOX_CHECK(closed, "Unexpected close failure");
    }
  }

 private:
  // State transition diagram:
  //
  //     ┌───────┐   prepare()   ┌─────────┐   success   ┌──────────┐
  //     │ kInit │ ────────────► │ kMaking │ ──────────► │kPrepared │
  //     └───────┘    move()     └─────────┘             └──────────┘
  //       │   │                      │                     │    │
  //       │   │ close()              │ exception           │    │ cancel()
  //       │   │                      ▼                     │    │
  //       │   │                ┌──────────┐   move()       │    │
  //       │   │                │ kFailed  │   close()      │    │
  //       │   │                └──────────┘                │    │
  //       │   │                                            │    │
  //       │   └──────────────────────┬─────────────────────┘    │
  //       │                          ▼                          │
  //       │ cancel()           ┌───────────┐                    │
  //       │                    │ kFinished │                    │
  //       │                    └───────────┘                    │
  //       │                                                     │
  //       └──────────────────────────┬──────────────────────────┘
  //                                  ▼
  //                            ┌───────────┐
  //                            │kCancelled │
  //                            └───────────┘
  //
  enum class State : uint8_t {
    // Initial state before prepare() or move() is called.
    kInit = 0,
    // prepare() is executing.
    kMaking = 1,
    // prepare() has completed and item is ready.
    kPrepared = 2,
    // prepare() has failed with an exception.
    kFailed = 3,
    // move() or close() has been called and the AsyncSource is finished.
    kFinished = 4,
    // cancel() has been called.
    kCancelled = 5,
  };

  State state() const {
    return state_.load(std::memory_order_acquire);
  }

  static std::string stateName(State state) {
    switch (state) {
      case State::kInit:
        return "INIT";
      case State::kMaking:
        return "MAKING";
      case State::kPrepared:
        return "PREPARED";
      case State::kFailed:
        return "FAILED";
      case State::kFinished:
        return "FINISHED";
      case State::kCancelled:
        return "CANCELLED";
      default:
        VELOX_UNREACHABLE("Unknown state: {}", static_cast<int>(state));
    }
  }

  inline void checkState(State actualState, State expectedState) const {
    VELOX_CHECK(
        actualState == expectedState,
        "Unexpected state: {}, expected: {}",
        stateName(actualState),
        stateName(expectedState));
  }

  inline void setState(State newState) {
    const auto oldState = state();
    VELOX_CHECK(
        isValidStateTransition(oldState, newState),
        "Invalid state transition from {} to {}",
        stateName(oldState),
        stateName(newState));
    state_.store(newState, std::memory_order_release);
  }

  static bool isValidStateTransition(State oldState, State newState) {
    switch (oldState) {
      case State::kInit:
        return newState == State::kMaking || newState == State::kFinished ||
            newState == State::kCancelled;
      case State::kMaking:
        return newState == State::kPrepared || newState == State::kFailed;
      case State::kPrepared:
        return newState == State::kFinished || newState == State::kCancelled;
      case State::kFailed:
      case State::kFinished:
      case State::kCancelled:
        return false;
      default:
        VELOX_UNREACHABLE("Unknown state: {}", stateName(oldState));
    }
  }

  // Makes item with timing, handles exceptions, state transitions, and promise
  // signaling.
  void makeItem(std::function<std::unique_ptr<Item>()>&& itemMaker) {
    VELOX_CHECK_NOT_NULL(itemMaker);
    std::unique_ptr<Item> item;
    std::exception_ptr exceptionPtr;
    try {
      CpuWallTimer timer(makeTiming_);
      process::ScopedThreadDebugInfo threadDebugInfo(
          threadDebugInfo_.has_value() ? &threadDebugInfo_.value() : nullptr);
      item = itemMaker();
    } catch (std::exception&) {
      exceptionPtr = std::current_exception();
    }

    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      VELOX_CHECK_NULL(item_);
      VELOX_CHECK_NULL(itemMaker_);
      checkState(state(), State::kMaking);
      if (FOLLY_LIKELY(exceptionPtr == nullptr)) {
        item_ = std::move(item);
        setState(State::kPrepared);
      } else {
        setExceptionLocked(std::move(exceptionPtr));
      }
      promises.swap(promises_);
    }
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  inline void setExceptionLocked(std::exception_ptr exception) {
    VELOX_CHECK_NULL(exception_);
    exception_ = std::move(exception);
    setState(State::kFailed);
  }

  // Waits for the promise to be fulfilled and injects a TestValue for testing
  // race conditions.
  inline void makeWait(ContinueFuture&& wait) {
    common::testutil::TestValue::adjust(
        "facebook::velox::AsyncSource::makeWait", this);
    auto& exec = folly::QueuedImmediateExecutor::instance();
    std::move(wait).via(&exec).wait();
  }

  // Attempts to close immediately while holding the lock. Returns true if
  // close completed, false if caller needs to wait for making to complete.
  inline bool tryCloseLocked() {
    const auto currentState = state();
    switch (currentState) {
      case State::kInit:
        VELOX_CHECK_NOT_NULL(itemMaker_);
        VELOX_CHECK_NULL(item_);
        itemMaker_ = nullptr;
        setState(State::kFinished);
        return true;
      case State::kPrepared:
        VELOX_CHECK_NULL(itemMaker_);
        item_ = nullptr;
        setState(State::kFinished);
        return true;
      case State::kMaking:
        return false;
      default:
        VELOX_CHECK_NULL(itemMaker_);
        VELOX_CHECK_NULL(item_);
        return true;
    }
  }

  // Stored context (if present upon construction) so they can be restored
  // when itemMaker_ is invoked.
  std::optional<process::ThreadDebugInfo> threadDebugInfo_;

  std::atomic<State> state_{State::kInit};
  CpuWallTiming makeTiming_;
  mutable std::mutex mutex_;
  std::vector<ContinuePromise> promises_;
  std::unique_ptr<Item> item_;
  std::function<std::unique_ptr<Item>()> itemMaker_;
  std::exception_ptr exception_;
};

} // namespace facebook::velox
