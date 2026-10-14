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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CachingDataSource.h"

#include "velox/common/caching/FileIds.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <folly/Conv.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// How many times a read looks up one piece while other readers fill it.
constexpr int kMaxLookupAttempts = 32;

// AsyncDataCache stores entry lengths in an int32_t. Large individual reads
// remain legal datasource requests, but must bypass that cache.
constexpr size_t kMaxCacheEntryBytes = std::numeric_limits<int32_t>::max();

// The most async device reads that run at once: KvikIO's
// KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS, with KvikIO's default of 256.
// KvikIO's 0 means unlimited. Use 64 then so cache allocations stay bounded.
size_t maxConcurrentReads() {
  static const size_t value = [] {
    size_t requests = 256;
    if (const auto* env =
            std::getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) {
      requests = folly::to<size_t>(env);
    }
    return requests == 0 ? size_t{64} : requests;
  }();
  return value;
}

// Keep remote cache fills out of the connector I/O executor. Sized by
// cudf.kvikio_cache_threads.
folly::Executor* remoteReadExecutor() {
  static auto* executor = [] {
    const auto threads = CudfConfig::getInstance().kvikioCacheThreads;
    VELOX_USER_CHECK_GT(
        threads, 0, "{} must be positive", CudfConfig::kCudfKvikioCacheThreads);
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfRemoteIO"));
  }();
  return executor;
}

// cuDF exposes std::future, not a completion callback. A read that waits for
// its KvikIO fill, or for another reader's fill of the same entry, blocks a
// thread of this executor rather than a remoteReadExecutor thread, so copies of
// cache hits never queue behind waits. At most maxConcurrentReads() reads run
// at once, so that many threads let all of them wait at the same time.
folly::Executor* fillWaitExecutor() {
  static auto* executor = new folly::CPUThreadPoolExecutor(
      maxConcurrentReads(),
      std::make_shared<folly::NamedThreadFactory>("CudfCacheWait"));
  return executor;
}

int streamDevice(cuda::stream_ref stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.get(), &device));
#else
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

void finishDeviceRead(cuda::stream_ref stream, int device) {
  const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
  try {
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  } catch (...) {
    // Do not release a destination while a failed stream may still reference
    // it. A device-wide fence is the last resort before propagating.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    throw;
  }
}

// One reusable pinned staging buffer per thread and device, plus the event
// marking its last H2D copy.
struct PinnedStagingSlot {
  static constexpr size_t kAlignment = 64;

  // The memory resource that allocated 'data', which must also free it.
  std::optional<rmm::host_device_async_resource_ref> mr;
  uint8_t* data{nullptr};
  size_t capacity{0};
  cudaEvent_t event{nullptr};

  uint8_t* reserve(size_t size) {
    // Wait for this slot's previous copy to land so the memory is safe to
    // overwrite.
    if (event != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(event));
    }
    if (capacity < size) {
      if (data != nullptr) {
        mr->deallocate_sync(std::exchange(data, nullptr), capacity, kAlignment);
        capacity = 0;
      }
      // Leave an empty, reusable slot if allocation throws.
      mr = cudf::get_pinned_memory_resource();
      data = static_cast<uint8_t*>(mr->allocate_sync(size, kAlignment));
      capacity = size;
    }
    return data;
  }

  void recordCopy(cudaStream_t stream) {
    if (event == nullptr) {
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  }
};

// Retain the high-water staging capacity until process exit. Do not invoke
// CUDA during thread-pool/static teardown: the runtime may already be gone.
PinnedStagingSlot& stagingSlot(int device) {
  static thread_local std::unordered_map<int, PinnedStagingSlot> slots;
  return slots[device];
}

// A caller may destroy the destination once its future is consumed or
// discarded. The executor's future alone only marks H2D submission, so the
// outer completion also fences the copy. Wait outside the remote executor
// rather than occupying an I/O thread during H2D completion.
struct DeviceReadCompletion {
  // The number of bytes whose H2D copies were submitted.
  std::future<size_t> submitted;
  cuda::stream_ref stream;
  int device;
  std::shared_ptr<std::atomic<bool>> published;

  DeviceReadCompletion(
      std::future<size_t> f,
      cuda::stream_ref s,
      int d,
      std::shared_ptr<std::atomic<bool>> ioPublished)
      : submitted(std::move(f)),
        stream(s),
        device(d),
        published(std::move(ioPublished)) {}
  DeviceReadCompletion(DeviceReadCompletion&&) = default;
  DeviceReadCompletion(const DeviceReadCompletion&) = delete;
  ~DeviceReadCompletion() {
    if (submitted.valid() && published->load()) {
      try {
        get();
      } catch (...) {
      }
    }
  }
  static size_t finish(size_t bytes, cuda::stream_ref stream, int device) {
    finishDeviceRead(stream, device);
    return bytes;
  }
  size_t get() {
    return finish(submitted.get(), stream, device);
  }
};

} // namespace

struct CachingDataSource::State {
  struct AsyncRead;
  std::unique_ptr<cudf::io::datasource> delegate;
  cache::AsyncDataCache* cache;
  StringIdLease fileNum;
  std::shared_ptr<IoStats> ioStats;

  State(
      std::unique_ptr<cudf::io::datasource> source,
      std::string_view path,
      cache::AsyncDataCache* dataCache,
      std::shared_ptr<IoStats> stats)
      : delegate(std::move(source)),
        cache(dataCache),
        fileNum(fileIds(), std::string(path)),
        ioStats(std::move(stats)) {
    VELOX_CHECK_NOT_NULL(delegate);
    VELOX_CHECK_NOT_NULL(cache);
  }

  void addBytes(const std::string& name, size_t bytes) {
    if (ioStats) {
      ioStats->addCounter(
          name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
    }
  }

  void addCount(const std::string& name, int64_t value = 1) {
    if (ioStats) {
      ioStats->addCounter(name, RuntimeCounter(value));
    }
  }

  size_t clamp(size_t offset, size_t size) const {
    const auto fileSize = delegate->size();
    return offset < fileSize ? std::min(size, fileSize - offset) : 0;
  }

  static size_t fragmentSize(const cache::CachePin& pin, size_t requested) {
    const auto size = pin.checkedEntry()->size();
    VELOX_CHECK_GT(size, 0, "Cache fragments must make progress");
    return std::min<size_t>(requested, size);
  }

  void recordHit(size_t bytes, size_t requested) {
    addBytes("cudfKvikioCacheHitBytes", bytes);
    if (bytes < requested) {
      addCount("cudfKvikioCacheReusedPrefixes");
      addBytes("cudfKvikioCacheReusedPrefixBytes", bytes);
    }
  }

  // Looks up the cache entry at 'offset'. Returns a shared pin on a hit, or an
  // exclusive pin with contiguous memory for the caller to fill on a miss. The
  // entry may be shorter than 'size'. Returns an empty pin and sets 'wait'
  // while another reader fills the entry, and an empty pin without 'wait' when
  // the range cannot be cached.
  cache::CachePin
  lookup(size_t offset, size_t size, folly::SemiFuture<bool>& wait) {
    if (size > kMaxCacheEntryBytes) {
      return {};
    }
    cache::CachePin pin;
    try {
      pin = cache->findOrCreate(
          {fileNum.id(), offset},
          size,
          /*contiguous=*/true,
          &wait,
          cache::CacheEntrySizePolicy::kAllowSmaller);
    } catch (const VeloxRuntimeError& error) {
      if (error.errorCode() != error_code::kNoCacheSpace) {
        throw;
      }
      return {};
    }
    if (pin.empty()) {
      return pin;
    }
    auto* entry = pin.checkedEntry();
    entry->getAndClearFirstUseFlag();
    if (entry->isExclusive() && !entry->hasContiguousData()) {
      return {};
    }
    return pin;
  }

  // Returns a shared pin on the entry at 'offset', filling it synchronously on
  // a miss. The entry may be shorter than 'size'. Returns an empty pin when
  // the cache cannot admit the range.
  cache::CachePin pinFragment(size_t offset, size_t size) {
    for (int attempt = 0; attempt < kMaxLookupAttempts; ++attempt) {
      auto wait = folly::SemiFuture<bool>::makeEmpty();
      auto pin = lookup(offset, size, wait);
      if (pin.empty()) {
        if (!wait.valid()) {
          return {};
        }
        std::move(wait).via(&folly::QueuedImmediateExecutor::instance()).wait();
        continue;
      }
      auto* entry = pin.checkedEntry();
      if (!entry->isExclusive()) {
        recordHit(fragmentSize(pin, size), size);
        return pin;
      }
      const auto actual = delegate->host_read(
          offset, size, reinterpret_cast<uint8_t*>(entry->contiguousData()));
      // Do not publish uninitialized tail bytes after a short read. An
      // exception releases the exclusive pin and abandons the incomplete fill.
      VELOX_CHECK_EQ(actual, size, "Short KvikIO read while filling cache");
      entry->setExclusiveToShared();
      addBytes("cudfKvikioCacheMissBytes", size);
      return pin;
    }
    return {};
  }

  size_t readHost(size_t offset, size_t size, uint8_t* dst) {
    const auto bytes = clamp(offset, size);
    if (bytes == 0) {
      return 0;
    }
    VELOX_CHECK_NOT_NULL(dst);
    size_t copied = 0;
    while (copied < bytes) {
      const auto remaining = bytes - copied;
      auto pin = pinFragment(offset + copied, remaining);
      if (pin.empty()) {
        // Already consumed prefixes remain useful, even when the remaining
        // suffix cannot be cached. Never bypass/refetch the whole request.
        const auto actual =
            delegate->host_read(offset + copied, remaining, dst + copied);
        addBytes("cudfKvikioCacheBypassBytes", actual);
        return copied + actual;
      }
      const auto fragmentBytes = fragmentSize(pin, remaining);
      copyCached(pin, fragmentBytes, dst + copied);
      copied += fragmentBytes;
    }
    return copied;
  }

  static void
  copyCached(const cache::CachePin& pin, size_t bytes, uint8_t* dst) {
    auto* entry = pin.checkedEntry();
    if (entry->hasContiguousData()) {
      std::memcpy(dst, entry->contiguousData(), bytes);
      return;
    }
    // Another reader may have populated this key using page-run allocation.
    // contiguous=true controls new allocations, not the layout of cache hits.
    for (const auto& range : entry->dataRanges(bytes)) {
      std::memcpy(dst, range.data(), range.size());
      dst += range.size();
    }
  }

  std::unique_ptr<cudf::io::datasource::buffer> readHost(
      size_t offset,
      size_t size) {
    std::vector<uint8_t> bytes(clamp(offset, size));
    bytes.resize(readHost(offset, bytes.size(), bytes.data()));
    return cudf::io::datasource::buffer::create(std::move(bytes));
  }

  // Synchronous device read: consume cached fragments and fill misses inline.
  // Returns the number of bytes whose H2D copies were submitted.
  size_t submitDeviceRead(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      cuda::stream_ref stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    size_t submitted = 0;
    try {
      while (submitted < bytes) {
        const auto remaining = bytes - submitted;
        auto pin = pinFragment(offset + submitted, remaining);
        const auto fragmentBytes =
            pin.empty() ? remaining : fragmentSize(pin, remaining);
        const auto actual = submitReadyDeviceRead(
            std::move(pin),
            offset + submitted,
            fragmentBytes,
            dst + submitted,
            stream,
            device);
        submitted += actual;
        if (actual < fragmentBytes) {
          break;
        }
      }
      return submitted;
    } catch (...) {
      // A later lookup/read can fail while an earlier prefix is still in
      // flight. Fence before the destination can be released.
      finishDeviceRead(stream, device);
      throw;
    }
  }

  // Stages one fragment through the thread's pinned buffer and submits its
  // H2D copy. 'pin' is shared and fully filled, or empty for an uncached
  // range that is read synchronously from the delegate. Returns the number of
  // bytes submitted.
  size_t submitReadyDeviceRead(
      cache::CachePin pin,
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      cuda::stream_ref stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    auto& slot = stagingSlot(device);
    auto* staging = slot.reserve(bytes);
    size_t actual;
    if (!pin.empty()) {
      copyCached(pin, bytes, staging);
      actual = bytes;
      pin.clear();
    } else {
      actual = delegate->host_read(offset, bytes, staging);
      addBytes("cudfKvikioCacheBypassBytes", actual);
    }
    try {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          dst, staging, actual, cudaMemcpyHostToDevice, stream.get()));
      slot.recordCopy(stream.get());
    } catch (...) {
      finishDeviceRead(stream, device);
      throw;
    }
    return actual;
  }
};

// One asynchronous device read. run() makes as much progress as it can on a
// remoteReadExecutor() thread without blocking. While its KvikIO fill, or
// another reader's fill of the same entry, is pending, the read waits on a
// fillWaitExecutor() thread and then continues in run().
struct CachingDataSource::State::AsyncRead final
    : std::enable_shared_from_this<AsyncRead> {
  std::shared_ptr<State> state;
  // Set to the number of bytes whose H2D copies were submitted.
  std::shared_ptr<std::promise<size_t>> promise;
  size_t offset;
  size_t bytes;
  uint8_t* dst;
  cuda::stream_ref stream;
  int device;
  cache::CachePin pin;
  size_t submittedBytes{0};
  size_t fragmentBytes{0};
  folly::SemiFuture<bool> cacheWait = folly::SemiFuture<bool>::makeEmpty();
  std::future<size_t> hostRead;
  int attempts{0};

  AsyncRead(
      std::shared_ptr<State> s,
      std::shared_ptr<std::promise<size_t>> p,
      size_t o,
      size_t b,
      uint8_t* d,
      cuda::stream_ref st,
      int dev)
      : state(std::move(s)),
        promise(std::move(p)),
        offset(o),
        bytes(b),
        dst(d),
        stream(st),
        device(dev) {}

  // Starts 'read', or queues it while maxConcurrentReads() reads are running.
  // Queued reads have not allocated cache entries yet, so a large pass cannot
  // allocate host buffers for every queued request at once.
  static void start(std::shared_ptr<AsyncRead> read) {
    auto& queue = startQueue();
    {
      std::lock_guard<std::mutex> lock(queue.mutex);
      if (queue.running == maxConcurrentReads()) {
        queue.waiting.push_back(std::move(read));
        return;
      }
      ++queue.running;
    }
    read->resume();
  }

  struct StartQueue {
    std::mutex mutex;
    size_t running{0};
    std::deque<std::shared_ptr<AsyncRead>> waiting;
  };

  static StartQueue& startQueue() {
    // Never destroyed, like the executors.
    static auto* queue = new StartQueue;
    return *queue;
  }

  // Ends this read's turn and starts the next queued read, if any.
  static void finishOne() noexcept {
    std::shared_ptr<AsyncRead> next;
    {
      auto& queue = startQueue();
      std::lock_guard<std::mutex> lock(queue.mutex);
      if (queue.waiting.empty()) {
        --queue.running;
      } else {
        next = std::move(queue.waiting.front());
        queue.waiting.pop_front();
      }
    }
    if (next) {
      next->resume();
    }
  }

  // Continues this read in run() on remoteReadExecutor().
  void resume() noexcept {
    try {
      remoteReadExecutor()->add([self = shared_from_this()] { self->run(); });
    } catch (...) {
      fail(std::current_exception());
    }
  }

  // Waits on a fillWaitExecutor() thread for the pending fill, then resumes.
  void waitThenResume() noexcept {
    try {
      fillWaitExecutor()->add([self = shared_from_this()] {
        if (self->cacheWait.valid()) {
          self->cacheWait.wait();
        } else {
          self->hostRead.wait();
        }
        self->resume();
      });
    } catch (...) {
      fail(std::current_exception());
    }
  }

  void fail(std::exception_ptr error) noexcept {
    // A std::future destructor need not wait for external writes. Even if
    // scheduling/allocation fails after dispatch, drain before abandoning the
    // exclusive cache allocation or allowing destination destruction.
    if (hostRead.valid()) {
      try {
        hostRead.get();
      } catch (...) {
      }
    }
    if (submittedBytes != 0) {
      try {
        finishDeviceRead(stream, device);
      } catch (...) {
      }
    }
    pin.clear();
    promise->set_exception(error);
    finishOne();
  }

  void run() noexcept {
    try {
      const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
      if (hostRead.valid()) {
        const auto actual = hostRead.get();
        VELOX_CHECK_EQ(
            actual, fragmentBytes, "Short KvikIO read while filling cache");
        pin.checkedEntry()->setExclusiveToShared();
        state->addBytes("cudfKvikioCacheMissBytes", fragmentBytes);
        copyFragment();
      }
      if (cacheWait.valid()) {
        std::move(cacheWait).get();
        cacheWait = folly::SemiFuture<bool>::makeEmpty();
      }
      while (submittedBytes < bytes) {
        const auto remaining = bytes - submittedBytes;
        if (++attempts > kMaxLookupAttempts) {
          bypass();
          return;
        }
        pin = state->lookup(offset + submittedBytes, remaining, cacheWait);
        if (pin.empty()) {
          if (cacheWait.valid()) {
            waitThenResume();
          } else {
            bypass();
          }
          return;
        }
        auto* entry = pin.checkedEntry();
        fragmentBytes = State::fragmentSize(pin, remaining);
        if (!entry->isExclusive()) {
          state->recordHit(fragmentBytes, remaining);
          copyFragment();
          continue;
        }
        hostRead = state->delegate->host_read_async(
            offset + submittedBytes,
            fragmentBytes,
            reinterpret_cast<uint8_t*>(entry->contiguousData()));
        VELOX_CHECK(hostRead.valid(), "KvikIO returned an invalid read future");
        waitThenResume();
        return;
      }
      complete();
    } catch (...) {
      fail(std::current_exception());
    }
  }

  void copyFragment() {
    submittedBytes += state->submitReadyDeviceRead(
        std::move(pin),
        offset + submittedBytes,
        fragmentBytes,
        dst + submittedBytes,
        stream,
        device);
    attempts = 0;
  }

  // Reads the rest of the range synchronously, bypassing the cache.
  void bypass() {
    fragmentBytes = bytes - submittedBytes;
    copyFragment();
    complete();
  }

  void complete() {
    promise->set_value(submittedBytes);
    finishOne();
  }
};

CachingDataSource::CachingDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    std::shared_ptr<IoStats> ioStats)
    : state_(
          std::make_shared<State>(
              std::move(delegate),
              path,
              cache,
              std::move(ioStats))) {}

CachingDataSource::~CachingDataSource() = default;

size_t CachingDataSource::size() const {
  return state_->delegate->size();
}

bool CachingDataSource::supports_device_read() const {
  return true;
}

bool CachingDataSource::is_device_read_preferred(size_t bytes) const {
  return state_->delegate->is_device_read_preferred(bytes);
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::host_read(
    size_t offset,
    size_t bytes) {
  return state_->readHost(offset, bytes);
}

size_t CachingDataSource::host_read(size_t offset, size_t bytes, uint8_t* dst) {
  return state_->readHost(offset, bytes, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CachingDataSource::host_read_async(size_t offset, size_t bytes) {
  return std::async(std::launch::deferred, [state = state_, offset, bytes] {
    return state->readHost(offset, bytes);
  });
}

std::future<size_t>
CachingDataSource::host_read_async(size_t offset, size_t bytes, uint8_t* dst) {
  return std::async(
      std::launch::deferred, [state = state_, offset, bytes, dst] {
        return state->readHost(offset, bytes, dst);
      });
}

std::future<size_t> CachingDataSource::device_read_async(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    cuda::stream_ref stream) {
  bytes = state_->clamp(offset, bytes);
  if (bytes == 0) {
    std::promise<size_t> ready;
    ready.set_value(0);
    return ready.get_future();
  }
  VELOX_CHECK_NOT_NULL(dst);
  const auto device = streamDevice(stream);
  auto promise = std::make_shared<std::promise<size_t>>();
  auto submitted = promise->get_future();
  auto read = std::make_shared<State::AsyncRead>(
      state_, promise, offset, bytes, dst, stream, device);
  auto published = std::make_shared<std::atomic<bool>>(false);
  // Construct the destination-lifetime fence before publishing any I/O.
  auto completion = std::async(
      std::launch::deferred,
      [completion = DeviceReadCompletion(
           std::move(submitted), stream, device, published)]() mutable {
        return completion.get();
      });
  published->store(true);
  try {
    State::AsyncRead::start(std::move(read));
  } catch (...) {
    promise->set_exception(std::current_exception());
    throw;
  }
  return completion;
}

size_t CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    cuda::stream_ref stream) {
  bytes = state_->clamp(offset, bytes);
  if (bytes == 0) {
    return 0;
  }
  VELOX_CHECK_NOT_NULL(dst);
  const auto device = streamDevice(stream);
  // A synchronous caller may already occupy the connector executor. Queuing
  // the read back to that executor and waiting would deadlock when all its
  // threads are occupied by such callers. Keep scheduling in the async API;
  // perform synchronous reads inline, including the destination-lifetime fence.
  return DeviceReadCompletion::finish(
      state_->submitDeviceRead(offset, bytes, dst, stream, device),
      stream,
      device);
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    cuda::stream_ref stream) {
  const auto readSize = state_->clamp(offset, bytes);
  rmm::device_buffer result(
      readSize, stream, cudf::get_current_device_resource_ref());
  const auto actual = device_read(
      offset, readSize, static_cast<uint8_t*>(result.data()), stream);
  result.resize(actual, stream);
  return datasource::buffer::create(std::move(result));
}

std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats) {
  if (cache == nullptr || !cacheable) {
    return delegate;
  }
  return std::make_unique<CachingDataSource>(
      std::move(delegate), path, cache, std::move(ioStats));
}

} // namespace facebook::velox::cudf_velox::connector::hive
