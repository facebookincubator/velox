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
#include <mutex>

#include <folly/container/F14Set.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <folly/system/ThreadId.h>

namespace facebook::velox::io {

/// Tracks the OS thread IDs of the registered IO executor pool. Used by
/// DCHECK in block-on-future sites in the Velox read path: blocking on a
/// future settled through the IO executor from inside the executor pool
/// would self-deadlock at executor_size=1.
///
/// Architectural invariant:
/// No executor thread blocks on a future settled through the same executor.
/// The five enumerated block-on-future sites in velox/dwio must only be
/// invoked from the consumer thread (Velox query worker). This registry
/// lets each site enforce the invariant via a DCHECK that compiles out in
/// release builds.
///
/// Lifetime: process-global, populated once at executor construction
/// (typically by the host runner's connector setup). After registration
/// the thread-ID set is read-only; checks are lock-free atomic loads
/// against a frozen set.
class IoExecutorThreadRegistry {
 public:
  static IoExecutorThreadRegistry& instance() {
    static IoExecutorThreadRegistry kInstance;
    return kInstance;
  }

  /// Register all worker threads of the IO executor pool. Posts
  /// `tasksPerThread * numThreads` no-op tasks; each task captures its
  /// own `folly::getCurrentThreadID()`. With enough tasks (default 8
  /// per thread) the probability that every worker runs at least one
  /// task is overwhelming; we wait for all tasks to complete before
  /// publishing the set.
  ///
  /// Robust across folly versions that don't expose getThreadIds() on
  /// IOThreadPoolExecutor (which is the standard hostile case — the
  /// public API for enumerating worker threads is inconsistent).
  ///
  /// Call once at startup, after the executor is constructed.
  void registerExecutor(
      folly::IOThreadPoolExecutor* exec,
      int tasksPerThread = 8) {
    const size_t numThreads = exec->numThreads();
    const size_t numTasks = numThreads * static_cast<size_t>(tasksPerThread);

    struct RegistrationState {
      explicit RegistrationState(size_t taskCount) : remaining(taskCount) {}
      folly::F14FastSet<uint64_t> seen;
      std::mutex seenMutex;
      std::atomic<size_t> remaining;
      folly::Baton<> done;
    };

    auto state = std::make_shared<RegistrationState>(numTasks);

    for (size_t i = 0; i < numTasks; ++i) {
      exec->add([state]() {
        const uint64_t tid = folly::getCurrentThreadID();
        {
          std::lock_guard<std::mutex> g(state->seenMutex);
          state->seen.insert(tid);
        }
        if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          state->done.post();
        }
      });
    }

    // Bounded wait so a misconfigured executor can't hang startup.
    const bool registrationCompleted =
        state->done.try_wait_for(std::chrono::seconds(30));
    if (!registrationCompleted) {
      // Timed out collecting worker-thread IDs. Do not publish a partial
      // registration; callers will see "not registered" until a successful
      // attempt. `state` stays alive for any already-queued tasks via captures.
      return;
    }

    {
      std::lock_guard<std::mutex> g(state->seenMutex);
      tids_ = std::move(state->seen);
    }
    // Release-store the registered flag last so the check below reads a
    // fully-populated set under acquire semantics.
    registered_.store(true, std::memory_order_release);
  }

  /// Returns true if the current thread is one of the registered IO
  /// executor threads. Returns false if no executor has been registered
  /// (so checks at sites that run before initialization are no-ops).
  bool isOnExecutor() const {
    if (!registered_.load(std::memory_order_acquire)) {
      return false;
    }
    return tids_.count(folly::getCurrentThreadID()) > 0;
  }

 private:
  // folly::getCurrentThreadID() returns uint64_t (stable OS-level TID);
  // match the type. std::thread::id is implementation-defined and won't
  // compare equal to folly::getCurrentThreadID() values.
  folly::F14FastSet<uint64_t> tids_;
  std::atomic<bool> registered_{false};
};

/// Convenience accessor for DCHECK sites. Use as:
///   DCHECK(!facebook::velox::io::isOnIoExecutorThread())
///       << "block-on-future called on IO executor thread; "
///          "self-deadlock risk at executor_size=1";
inline bool isOnIoExecutorThread() {
  return IoExecutorThreadRegistry::instance().isOnExecutor();
}

} // namespace facebook::velox::io
