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

#include <sys/uio.h>
#include <memory>
#include <string>

#include <folly/coro/Task.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/io/async/EventBaseManager.h>

namespace facebook::velox::io {

/// Configuration for the io_uring-backed async IO context.
/// Must be set via AsyncIoContext::setConfig() before the first get() call.
struct AsyncIoConfig {
  /// Number of IO threads. Each thread gets its own io_uring instance — no
  /// sharing or contention across threads. IO requests are distributed
  /// round-robin. Total system-wide in-flight IO capacity is
  /// numThreads * maxCapacity.
  uint32_t numThreads{4};

  /// Maximum SQEs (submission queue entries) pushed to the kernel per
  /// io_uring_enter syscall. Also determines the SQ ring size (2 * maxSubmit).
  /// Larger values batch more IOs per syscall (lower overhead) but increase
  /// per-call latency.
  uint32_t maxSubmit{512};

  /// Maximum in-flight IO operations per io_uring instance. Sets the CQ
  /// (completion queue) ring size — a fixed kernel allocation created at
  /// io_uring_setup, never resized after. When all slots are occupied, new
  /// submissions suspend the calling coroutine until a completion frees a slot.
  uint32_t maxCapacity{1'024};

  /// Minimum CQ ring size. If the kernel cannot allocate maxCapacity entries
  /// (e.g., due to memory limits), folly halves the capacity and retries down
  /// to this floor. Provides graceful degradation at the cost of reduced IO
  /// concurrency.
  uint32_t minCapacity{512};

  /// Number of pre-registered fd slots per io_uring instance (0 disables).
  /// Pre-registering file descriptors skips the per-IO fd lookup (fget/fput)
  /// in the kernel — operations reference fds by index into a kernel-side
  /// array instead of the process fd table. Should be >= the number of files
  /// opened concurrently. Managed internally by folly, transparent to callers.
  uint32_t registeredFds{4'096};
  /// Enable kernel-side SQ polling thread (eliminates io_uring_enter syscalls
  /// under sustained load).
  bool sqpoll{false};
  /// SQ poll thread idle timeout before sleeping (sqpoll mode only). After this
  /// duration with no IO, the kernel poll thread sleeps to save CPU. The next
  /// IO submission pays one io_uring_enter syscall to wake it. Larger values
  /// keep the thread spinning longer between bursts (lower wake-up latency,
  /// more CPU waste); smaller values save CPU but add a syscall on the first IO
  /// after idle. Ignored when sqpoll is false.
  uint32_t sqpollIdleMs{2'000};

  std::string toString() const;
};

/// Provides async file IO via io_uring, integrated with folly coroutines.
///
/// Wraps a pool of io_uring-backed EventBase threads. Each thread owns its own
/// io_uring instance — no contention across threads. IO requests are
/// distributed round-robin across threads.
///
/// Modeled after rexdb/tiny_file/AsyncIO. The coroutine methods use a
/// promise/future bridge: the calling coroutine suspends, the IO is submitted
/// on an EventBase thread via backend->queueRead(), and the coroutine resumes
/// when the CQE completion fulfills the promise.
///
/// Usage:
///   AsyncIoContext::setConfig({.numThreads = 8});
///   auto& ctx = AsyncIoContext::get();
///   int bytesRead = co_await ctx.coPread(fd, offset, buf, size);
class AsyncIoContext {
 public:
  explicit AsyncIoContext(const AsyncIoConfig& config);
  ~AsyncIoContext() = default;

  /// Sets the global config. Must be called before the first get().
  static void setConfig(AsyncIoConfig config);

  /// Returns the global config.
  static const AsyncIoConfig& config();

  /// Returns the shared singleton. Lazily constructed on first call using the
  /// global config.
  static AsyncIoContext& get();

  /// Returns true if the kernel supports io_uring.
  static bool available();

  /// Async positioned read via io_uring.
  folly::coro::Task<int> coPread(int fd, off_t offset, void* buf, size_t size);

  /// Async positioned vectored read via io_uring.
  folly::coro::Task<int>
  coPreadv(int fd, off_t offset, const iovec* iov, int iovcnt);

 private:
  // Returns an EventBase chosen round-robin across IO threads. Each call
  // returns the next thread's EventBase in rotation, distributing IO requests
  // evenly across io_uring instances. May return a different EventBase on each
  // call.
  folly::EventBase* getEventBase();

  // Controls how EventBase instances are created for each IO thread. We inject
  // a custom backendFactory so each thread gets an EventBase backed by
  // IoUringBackend instead of the default epoll. Must outlive executor_.
  std::unique_ptr<folly::EventBaseManager> ebm_;
  // Pool of IO threads, each running an EventBase::loopForever() that drives
  // its own io_uring ring (submit SQEs, reap CQEs, invoke callbacks).
  // getEventBase() round-robins across these threads.
  std::unique_ptr<folly::IOThreadPoolExecutor> executor_;
};

} // namespace facebook::velox::io
