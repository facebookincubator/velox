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

#include <folly/futures/Future.h>
#include <mutex>
#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::core {

enum class StopReason {
  // Keep running.
  kNone,
  // Go off thread and do not schedule more activity.
  kPause,
  // Stop and free all. This is returned once and the thread that gets
  // this value is responsible for freeing the state associated with
  // the thread. Other threads will get kAlreadyTerminated after the
  // first thread has received kTerminate.
  kTerminate,
  kAlreadyTerminated,
  // Go off thread and then enqueue to the back of the runnable queue.
  kYield,
  // Must wait for external events.
  kBlock,
  // No more data to produce.
  kAtEnd,
  kAlreadyOnThread
};

// Represents a Driver's state. This is used for cancellation, forcing
// release of and for waiting for memory. The fields are serialized on
// the mutex of the Driver's CancelPool.
struct ThreadState {
  // The thread currently running this.
  std::thread::id thread;
  // The tid of 'thread'. Allows finding the thread in a debugger.
  int32_t tid;
  // True if queued on an executor but not on thread.
  bool enqueued{false};
  // True if being terminated or already terminated.
  bool isTerminated{false};
  // True if there is a future outstanding that will schedule this on an
  // executor thread when some promise is realized.
  bool hasBlockingFuture{false};
  // True if on thread but in a section waiting for RPC or memory
  // strategy decision. The thread is not supposed to access its
  // memory, which a third party can revoke while the thread is in
  // this state.
  bool isCancelFree{false};
  const char* continueFile = nullptr;
  int32_t continueLine{0};

  bool isOnThread() const {
    return thread != std::thread::id();
  }

  void setThread() {
    thread = std::this_thread::get_id();
    tid = gettid();
  }

  void clearThread() {
    thread = std::thread::id(); // no thread.
    tid = 0;
  }
};

#define SETCONT(state)             \
  {                                \
    state.continueLine = __LINE__; \
    state.continueFile = __FILE__; \
  }

class CancelPool {
 public:
  // Returns kNone if no pause or terminate is requested. The thread count is
  // incremented if kNone is returned. If something else is returned the
  // calling tread should unwind and return itself to its pool.
  StopReason enter(ThreadState& state) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(state.enqueued);
    state.enqueued = false;
    if (state.isTerminated) {
      return StopReason::kAlreadyTerminated;
    }
    if (state.isOnThread()) {
      return StopReason::kAlreadyOnThread;
    }
    auto reason = shouldStopLocked();
    if (reason == StopReason::kTerminate) {
      state.isTerminated = true;
    }
    if (reason == StopReason::kNone) {
      ++numThreads_;
      state.setThread();
      state.hasBlockingFuture = false;
    }
    return reason;
  }

  StopReason enterForTerminate(ThreadState& state) {
    std::lock_guard<std::mutex> l(mutex_);
    if (state.isOnThread() || state.isTerminated) {
      state.isTerminated = true;
      return StopReason::kAlreadyOnThread;
    }
    state.isTerminated = true;
    state.setThread();
    return StopReason::kTerminate;
  }

  StopReason leave(ThreadState& state) {
    std::lock_guard<std::mutex> l(mutex_);
    if (--numThreads_ == 0) {
      finished();
    }
    state.clearThread();
    if (state.isTerminated) {
      return StopReason::kTerminate;
    }
    auto reason = shouldStopLocked();
    if (reason == StopReason::kTerminate) {
      state.isTerminated = true;
    }
    return reason;
  }

  // Enters a cancel-free section where the caller stays on thread but
  // is not accounted as being on the thread.  Returns kNone if no
  // terminate is requested. The thread count is decremented if kNone
  // is returned. If thread count goes to zero, waiting promises are
  // realized. If kNone is not returned the calling tread should
  // unwind and return itself to its pool.
  StopReason enterCancelFree(ThreadState& state) {
    VELOX_CHECK(!state.hasBlockingFuture);
    VELOX_CHECK(state.isOnThread());
    std::lock_guard<std::mutex> l(mutex_);
    if (state.isTerminated) {
      return StopReason::kAlreadyTerminated;
    }
    if (!state.isOnThread()) {
      return StopReason::kAlreadyTerminated;
    }
    auto reason = shouldStopLocked();
    if (reason == StopReason::kTerminate) {
      state.isTerminated = true;
    }
    // A pause will not stop entering the cancel free section. It will
    // just ack that the thread is no longer in inside the
    // CancelPool. The pause can wait at the exit of the cancel free
    // section.
    if (reason == StopReason::kNone || reason == StopReason::kPause) {
      state.isCancelFree = true;
      if (--numThreads_ == 0) {
        finished();
      }
    }
    return StopReason::kNone;
  }

  StopReason leaveCancelFree(ThreadState& state) {
    for (;;) {
      {
        std::lock_guard<std::mutex> l(mutex_);
        ++numThreads_;
        state.isCancelFree = false;
        if (state.isTerminated) {
          return StopReason::kAlreadyTerminated;
        }
        if (terminateRequested_) {
          state.isTerminated = true;
          return StopReason::kTerminate;
        }
        if (!pauseRequested_) {
          // For yield or anything but pause  we return here.
          return StopReason::kNone;
        }
        --numThreads_;
        state.isCancelFree = true;
      }
      // If the pause flag is on when trying to reenter, sleep a while
      // outside of the mutex and recheck. This is rare and not time
      // critical. Can happen if memory interrupt sets pause while
      // already inside a cancel free section for other reason, like
      // IO.
      std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
    }
  }

  // Returns a stop reason without synchronization. If the stop reason
  // is yield, then atomically decrements the count of threads that
  // are to yield.
  StopReason shouldStop() {
    if (terminateRequested_) {
      return StopReason::kTerminate;
    }
    if (pauseRequested_) {
      return StopReason::kPause;
    }
    if (toYield_) {
      std::lock_guard<std::mutex> l(mutex_);
      return shouldStopLocked();
    }
    return StopReason::kNone;
  }

  void requestPause(bool pause) {
    std::lock_guard<std::mutex> l(mutex_);
    pauseRequested_ = pause;
  }

  void requestTerminate() {
    terminateRequested_ = true;
  }

  void requestYield() {
    std::lock_guard<std::mutex> l(mutex_);
    toYield_ = numThreads_;
  }

  bool terminateRequested() const {
    return terminateRequested_;
  }

  bool pauseRequested() const {
    return pauseRequested_;
  }

  // Returns a future that is completed when all threads have acknowledged
  // terminate or pause. If the future is realized there is no running activity
  // on behalf of threads that have entered 'this'.
  folly::SemiFuture<bool> finishFuture() {
    auto [promise, future] =
        makeVeloxPromiseContract<bool>("CancelPool::finishFuture");
    std::lock_guard<std::mutex> l(mutex_);
    if (numThreads_ == 0) {
      promise.setValue(true);
      return std::move(future);
    }
    finishPromises_.push_back(std::move(promise));
    return std::move(future);
  }

  std::mutex* mutex() {
    return &mutex_;
  }

 private:
  void finished() {
    for (auto& promise : finishPromises_) {
      promise.setValue(true);
    }
    finishPromises_.clear();
  }

  StopReason shouldStopLocked() {
    if (terminateRequested_) {
      return StopReason::kTerminate;
    }
    if (pauseRequested_) {
      return StopReason::kPause;
    }
    if (toYield_) {
      --toYield_;
      return StopReason::kYield;
    }
    return StopReason::kNone;
  }

  std::mutex mutex_;
  bool pauseRequested_ = false;
  bool terminateRequested_ = false;
  int32_t toYield_ = 0;
  int32_t numThreads_ = 0;
  std::vector<VeloxPromise<bool>> finishPromises_;
};

using CancelPoolPtr = std::shared_ptr<CancelPool>;

} // namespace facebook::velox::core
