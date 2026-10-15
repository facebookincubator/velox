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
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// AsyncDataCache stores entry lengths in an int32_t. Large individual reads
// remain legal datasource requests, but must bypass that cache.
constexpr size_t kMaxCacheEntryBytes = std::numeric_limits<int32_t>::max();

// Async device reads run in two steps on two pools. A fill thread pins the
// range in the cache and reads misses from the delegate, blocking until the
// bytes arrive. A copy thread then copies the pinned bytes to the GPU. So reads
// waiting for S3 never hold a copy thread, and only the few copy threads own
// pinned staging buffers.

// The number of fill threads: KvikIO's
// KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS, with KvikIO's default of 256.
// KvikIO's 0 means unlimited. Use 64 then so cache allocations stay bounded.
folly::Executor* cacheFillExecutor() {
  static auto* executor = [] {
    size_t threads = 256;
    if (const auto* env =
            std::getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) {
      threads = folly::to<size_t>(env);
    }
    return new folly::CPUThreadPoolExecutor(
        threads == 0 ? size_t{64} : threads,
        std::make_shared<folly::NamedThreadFactory>("CudfCacheFill"));
  }();
  return executor;
}

// The number of copy threads is cudf.kvikio_cache_threads.
folly::Executor* cacheCopyExecutor() {
  static auto* executor = [] {
    const auto threads = CudfConfig::getInstance().kvikioCacheThreads;
    VELOX_USER_CHECK_GT(
        threads, 0, "{} must be positive", CudfConfig::kCudfKvikioCacheThreads);
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfCacheCopy"));
  }();
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
// outer completion also waits for the copy. It waits on the caller's thread,
// not on a pool thread.
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

  // Looks up the cache entry at 'offset'. On a miss, creates an exclusive
  // entry for the caller to fill. Returns an empty pin while another reader
  // fills the entry, setting 'wait' if given, or when the cache has no room.
  cache::CachePin
  lookup(size_t offset, size_t size, folly::SemiFuture<bool>* wait) {
    cache::CachePin pin;
    try {
      pin = cache->findOrCreate(
          {fileNum.id(), offset},
          size,
          /*contiguous=*/true,
          wait,
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

  // Returns a shared pin on the entry at 'offset', reading it from the
  // delegate on a miss. The entry may be shorter than 'size'. Returns an empty
  // pin when the range cannot be cached.
  cache::CachePin pinFragment(size_t offset, size_t size) {
    if (size > kMaxCacheEntryBytes) {
      return {};
    }
    auto wait = folly::SemiFuture<bool>::makeEmpty();
    auto pin = lookup(offset, size, &wait);
    if (pin.empty() && wait.valid()) {
      // Another reader is filling this entry. Wait for it, then look it up once
      // more. If it is still being filled, the caller reads it uncached.
      std::move(wait).via(&folly::QueuedImmediateExecutor::instance()).wait();
      pin = lookup(offset, size, nullptr);
    }
    if (pin.empty()) {
      return pin;
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

  // Pins the cache entries that cover the range from 'offset', filling misses.
  // Stops at the first piece that cannot be cached.
  std::vector<cache::CachePin> pinRange(size_t offset, size_t bytes) {
    std::vector<cache::CachePin> pins;
    size_t pinned = 0;
    while (pinned < bytes) {
      auto pin = pinFragment(offset + pinned, bytes - pinned);
      if (pin.empty()) {
        break;
      }
      pinned += fragmentSize(pin, bytes - pinned);
      pins.push_back(std::move(pin));
    }
    return pins;
  }

  // Submits the H2D copies of 'pins' from pinRange(), then reads the rest of
  // the range uncached. Returns the number of bytes submitted.
  size_t submitDeviceRead(
      std::vector<cache::CachePin> pins,
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      cuda::stream_ref stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    size_t submitted = 0;
    try {
      for (auto& pin : pins) {
        const auto fragmentBytes = fragmentSize(pin, bytes - submitted);
        submitted += submitReadyDeviceRead(
            std::move(pin),
            offset + submitted,
            fragmentBytes,
            dst + submitted,
            stream,
            device);
      }
      if (submitted < bytes) {
        submitted += submitReadyDeviceRead(
            {},
            offset + submitted,
            bytes - submitted,
            dst + submitted,
            stream,
            device);
      }
      return submitted;
    } catch (...) {
      // A later read can fail while an earlier prefix is still in flight.
      // Fence before the destination can be released.
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
  auto published = std::make_shared<std::atomic<bool>>(false);
  // Construct the destination-lifetime fence before starting any I/O.
  auto completion = std::async(
      std::launch::deferred,
      [completion = DeviceReadCompletion(
           promise->get_future(), stream, device, published)]() mutable {
        return completion.get();
      });
  published->store(true);
  try {
    cacheFillExecutor()->add(
        [state = state_, promise, offset, bytes, dst, stream, device] {
          try {
            auto pins = state->pinRange(offset, bytes);
            cacheCopyExecutor()->add([state,
                                      promise,
                                      pins = std::move(pins),
                                      offset,
                                      bytes,
                                      dst,
                                      stream,
                                      device]() mutable {
              try {
                promise->set_value(state->submitDeviceRead(
                    std::move(pins), offset, bytes, dst, stream, device));
              } catch (...) {
                promise->set_exception(std::current_exception());
              }
            });
          } catch (...) {
            promise->set_exception(std::current_exception());
          }
        });
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
      state_->submitDeviceRead(
          state_->pinRange(offset, bytes), offset, bytes, dst, stream, device),
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
