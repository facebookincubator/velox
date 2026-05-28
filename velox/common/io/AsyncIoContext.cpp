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

#include "velox/common/io/AsyncIoContext.h"

#include <fmt/format.h>

#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <glog/logging.h>

#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Allocation.h"

#include <folly/io/async/Liburing.h>

#if FOLLY_HAS_LIBURING
#include <folly/io/async/IoUringBackend.h>
#include <folly/io/async/IoUringOptions.h>
#define VELOX_HAS_IO_URING 1
#else
#define VELOX_HAS_IO_URING 0
#endif

namespace facebook::velox::io {

using memory::AllocationTraits;

namespace {

inline void checkPageAligned(const void* ptr, const char* label) {
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(ptr) % AllocationTraits::kPageSize,
      0,
      "{} must be page-aligned ({}-byte)",
      label,
      AllocationTraits::kPageSize);
}

inline void checkPageAligned(off_t offset, const char* label) {
  VELOX_CHECK_EQ(
      offset % AllocationTraits::kPageSize,
      0,
      "{} must be page-aligned ({}-byte)",
      label,
      AllocationTraits::kPageSize);
}

AsyncIoConfig& globalConfig() {
  static AsyncIoConfig config;
  return config;
}

#if VELOX_HAS_IO_URING
std::unique_ptr<folly::IoUringBackend> createIoUringBackend(
    const AsyncIoConfig& config) {
  folly::IoUringOptions options;
  options.setMaxSubmit(config.maxSubmit);
  options.setCapacity(config.maxCapacity);
  options.setMinCapacity(config.minCapacity);
  options.setRegisterRingFd(true);

  if (config.sqpoll) {
    // SQ poll: kernel spawns a dedicated thread that polls the submission and
    // completion queues, eliminating syscall overhead at the cost of burning a
    // CPU core. Best for sustained high-throughput IO.
    options.setFlags(
        folly::IoUringOptions::POLL_SQ | folly::IoUringOptions::POLL_CQ);
    options.setSQIdle(std::chrono::milliseconds(config.sqpollIdleMs));
  } else {
    // Coop + DeferTaskRun: no dedicated kernel thread. Completions are deferred
    // until we explicitly enter the kernel, avoiding interrupt storms and
    // context switches. Lower CPU cost than SQ poll, slightly higher latency.
    options.setTaskRunCoop(true);
    options.setDeferTaskRun(true);
  }

  // Pre-register file descriptors with io_uring to skip per-IO fd lookup
  // (fget/fput) in the kernel. Operations reference fds by index into the
  // kernel's internal array instead of the process fd table.
  if (config.registeredFds > 0) {
    options.setUseRegisteredFds(config.registeredFds);
  }

  return std::make_unique<folly::IoUringBackend>(std::move(options));
}

folly::IoUringBackend* ioUringBackend(folly::EventBase* eventBase) {
  return checkedPointerCast<folly::IoUringBackend>(eventBase->getBackend());
}
#endif // VELOX_HAS_IO_URING

} // namespace

std::string AsyncIoConfig::toString() const {
  return fmt::format(
      "AsyncIoConfig(numThreads={}, maxSubmit={}, maxCapacity={}, minCapacity={}, "
      "registeredFds={}, sqpoll={}, sqpollIdleMs={})",
      numThreads,
      maxSubmit,
      maxCapacity,
      minCapacity,
      registeredFds,
      sqpoll,
      sqpollIdleMs);
}

AsyncIoContext::AsyncIoContext(const AsyncIoConfig& config) {
#if VELOX_HAS_IO_URING
  LOG(INFO) << "AsyncIoContext: " << config.toString();

  ebm_ = std::make_unique<folly::EventBaseManager>(
      folly::EventBase::Options().setBackendFactory(
          [config]() { return createIoUringBackend(config); }));

  executor_ = std::make_unique<folly::IOThreadPoolExecutor>(
      config.numThreads,
      std::make_shared<folly::NamedThreadFactory>("VeloxAsyncIO"),
      ebm_.get());
#else
  VELOX_NYI("AsyncIoContext requires io_uring support (folly::IoUringBackend)");
#endif
}

/*static*/ void AsyncIoContext::setConfig(AsyncIoConfig config) {
  globalConfig() = std::move(config);
}

/*static*/ const AsyncIoConfig& AsyncIoContext::config() {
  return globalConfig();
}

/*static*/ AsyncIoContext& AsyncIoContext::get() {
  static AsyncIoContext instance(globalConfig());
  return instance;
}

/*static*/ bool AsyncIoContext::available() {
#if VELOX_HAS_IO_URING
  return folly::IoUringBackend::isAvailable();
#else
  return false;
#endif
}

#if VELOX_HAS_IO_URING
folly::EventBase* AsyncIoContext::getEventBase() {
  return executor_->getEventBase();
}

folly::coro::Task<int>
AsyncIoContext::coPread(int fd, off_t offset, void* buf, size_t size) {
  checkPageAligned(buf, "coPread buffer");
  checkPageAligned(offset, "coPread offset");
  auto* eventBase = getEventBase();
  auto* backend = ioUringBackend(eventBase);

  auto [promise, future] = folly::makePromiseContract<int>();
  eventBase->runInEventBaseThread([backend,
                                   fd,
                                   buf,
                                   size,
                                   offset,
                                   _promise = std::move(promise)]() mutable {
    backend->queueRead(
        fd,
        buf,
        size,
        offset,
        [_promise = std::move(_promise)](int result) mutable {
          _promise.setValue(result);
        });
  });
  co_return co_await std::move(future);
}

folly::coro::Task<int>
AsyncIoContext::coPreadv(int fd, off_t offset, const iovec* iov, int iovcnt) {
  checkPageAligned(offset, "coPreadv offset");
  for (int i = 0; i < iovcnt; ++i) {
    checkPageAligned(iov[i].iov_base, "coPreadv iovec buffer");
  }
  auto* eventBase = getEventBase();
  auto* backend = ioUringBackend(eventBase);

  auto [promise, future] = folly::makePromiseContract<int>();
  eventBase->runInEventBaseThread([backend,
                                   fd,
                                   iov,
                                   iovcnt,
                                   offset,
                                   _promise = std::move(promise)]() mutable {
    backend->queueReadv(
        fd,
        folly::Range<const iovec*>(iov, iovcnt),
        offset,
        [_promise = std::move(_promise)](int result) mutable {
          _promise.setValue(result);
        });
  });
  co_return co_await std::move(future);
}
#endif // VELOX_HAS_IO_URING

} // namespace facebook::velox::io
