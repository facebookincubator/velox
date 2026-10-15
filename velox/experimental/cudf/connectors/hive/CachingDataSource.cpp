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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// AsyncDataCache stores entry sizes in an int32_t. Larger pieces are read
// without the cache.
constexpr size_t kMaxCacheEntryBytes = std::numeric_limits<int32_t>::max();

// device_read_async() works in two steps, on two thread pools:
//
//   1. A fill thread puts the range in the cache. On a miss it reads the bytes
//      from KvikIO and blocks until they arrive.
//   2. A copy thread copies the cached bytes to the GPU, through its own
//      pinned buffer.
//
// Reads waiting for S3 therefore never hold a copy thread, and only the few
// copy threads own pinned buffers.

// One fill thread per S3 request KvikIO runs at once
// (KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS, default 256). KvikIO's 0 means
// unlimited. Use 64 then, so cache allocations stay bounded.
folly::Executor* fillThreads() {
  static auto* pool = [] {
    size_t threads = 256;
    if (const char* value =
            std::getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) {
      threads = folly::to<size_t>(value);
    }
    if (threads == 0) {
      threads = 64;
    }
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfCacheFill"));
  }();
  return pool;
}

// cudf.kvikio_cache_threads copy threads.
folly::Executor* copyThreads() {
  static auto* pool = [] {
    const auto threads = CudfConfig::getInstance().kvikioCacheThreads;
    VELOX_USER_CHECK_GT(
        threads, 0, "{} must be positive", CudfConfig::kCudfKvikioCacheThreads);
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfCacheCopy"));
  }();
  return pool;
}

int deviceOf(cuda::stream_ref stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.get(), &device));
#else
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

// Blocks until every copy queued on 'stream' has finished, so the caller may
// use or free the destination.
void waitForStream(cuda::stream_ref stream, int device) {
  const rmm::cuda_set_device_raii onDevice{rmm::cuda_device_id{device}};
  try {
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  } catch (...) {
    // The stream failed, but a copy may still be writing to the destination.
    // Wait for the whole GPU before letting the caller free it.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    throw;
  }
}

// Pinned host memory that bytes are copied through on their way to the GPU.
// The GPU copies from pinned memory in the background, which it cannot do from
// the cache's ordinary memory.
struct PinnedBuffer {
  static constexpr size_t kAlignment = 64;

  rmm::host_device_async_resource_ref allocator =
      cudf::get_pinned_memory_resource();
  uint8_t* data{nullptr};
  size_t size{0};
  // Recorded after the last copy out of 'data'.
  cudaEvent_t lastCopy{nullptr};

  // Waits until the last copy out of the buffer has finished, grows the buffer
  // to at least 'bytes' if needed, and returns it.
  uint8_t* prepare(size_t bytes) {
    if (lastCopy != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(lastCopy));
    }
    if (size < bytes) {
      if (data != nullptr) {
        allocator.deallocate_sync(data, size, kAlignment);
        data = nullptr;
        size = 0;
      }
      data = static_cast<uint8_t*>(allocator.allocate_sync(bytes, kAlignment));
      size = bytes;
    }
    return data;
  }

  // Queues a copy of the first 'bytes' bytes of the buffer to 'dst' on the GPU.
  void copyToGpu(size_t bytes, uint8_t* dst, cuda::stream_ref stream) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst, data, bytes, cudaMemcpyHostToDevice, stream.get()));
    if (lastCopy == nullptr) {
      CUDF_CUDA_TRY(
          cudaEventCreateWithFlags(&lastCopy, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(lastCopy, stream.get()));
  }
};

// One pinned buffer per thread and GPU. It is never freed: at process exit the
// CUDA runtime may already be gone.
PinnedBuffer& pinnedBuffer(int device) {
  static thread_local std::unordered_map<int, PinnedBuffer> buffers;
  return buffers[device];
}

// One device read: 'bytes' bytes at file 'offset' go to 'dst' on 'device',
// copied on 'stream'.
struct DeviceRead {
  size_t offset;
  size_t bytes;
  uint8_t* dst;
  cuda::stream_ref stream;
  int device;
};

// Lives inside the future that device_read_async() returns. 'copiesQueued'
// becomes ready once every GPU copy of the read is queued. Both get() and the
// destructor then also wait for those copies to finish, so cuDF can never free
// the destination while a copy still writes to it, even if it drops the future
// without calling get().
struct WaitForGpuCopies {
  std::future<size_t> copiesQueued;
  cuda::stream_ref stream;
  int device;

  WaitForGpuCopies(
      std::future<size_t> queued,
      cuda::stream_ref copyStream,
      int copyDevice)
      : copiesQueued(std::move(queued)),
        stream(copyStream),
        device(copyDevice) {}
  WaitForGpuCopies(WaitForGpuCopies&&) = default;
  WaitForGpuCopies(const WaitForGpuCopies&) = delete;

  ~WaitForGpuCopies() {
    // A moved-from object has nothing to wait for.
    if (copiesQueued.valid()) {
      try {
        get();
      } catch (...) {
      }
    }
  }

  size_t get() {
    const auto bytes = copiesQueued.get();
    waitForStream(stream, device);
    return bytes;
  }
};

} // namespace

// Does the reading. Async work holds a shared_ptr to it, so it outlives the
// CachingDataSource that cuDF may destroy first.
struct CachingDataSource::Reader {
  std::unique_ptr<cudf::io::datasource> kvikioSource;
  cache::AsyncDataCache* cache;
  StringIdLease fileId;
  std::shared_ptr<IoStats> ioStats;

  Reader(
      std::unique_ptr<cudf::io::datasource> source,
      std::string_view path,
      cache::AsyncDataCache* dataCache,
      std::shared_ptr<IoStats> stats)
      : kvikioSource(std::move(source)),
        cache(dataCache),
        fileId(fileIds(), std::string(path)),
        ioStats(std::move(stats)) {
    VELOX_CHECK_NOT_NULL(kvikioSource);
    VELOX_CHECK_NOT_NULL(cache);
  }

  void countBytes(const std::string& name, size_t bytes) {
    if (ioStats) {
      ioStats->addCounter(
          name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
    }
  }

  void count(const std::string& name) {
    if (ioStats) {
      ioStats->addCounter(name, RuntimeCounter(1));
    }
  }

  // Returns how many of the 'bytes' bytes at 'offset' are inside the file.
  size_t trimToFile(size_t offset, size_t bytes) const {
    const auto fileSize = kvikioSource->size();
    return offset < fileSize ? std::min(bytes, fileSize - offset) : 0;
  }

  // How many of the 'wanted' bytes the cache entry in 'pin' holds. An entry
  // may be shorter than the request (CacheEntrySizePolicy::kAllowSmaller).
  static size_t pieceSize(const cache::CachePin& pin, size_t wanted) {
    const auto entrySize = pin.checkedEntry()->size();
    VELOX_CHECK_GT(entrySize, 0, "A cache entry must hold at least one byte");
    return std::min<size_t>(wanted, entrySize);
  }

  // Finds the cache entry at 'offset', or creates an empty one. Returns:
  //   - a shared pin: the entry is filled and ready to read;
  //   - an exclusive pin: a new, empty entry that the caller must fill;
  //   - an empty pin: the cache cannot be used for this piece now.
  // When the empty pin is because another reader is filling the entry, and
  // 'otherReaderFill' is not null, it is set to wait for that reader.
  cache::CachePin findOrCreateEntry(
      size_t offset,
      size_t size,
      folly::SemiFuture<bool>* otherReaderFill) {
    cache::CachePin pin;
    try {
      pin = cache->findOrCreate(
          {fileId.id(), offset},
          size,
          /*contiguous=*/true,
          otherReaderFill,
          cache::CacheEntrySizePolicy::kAllowSmaller);
    } catch (const VeloxRuntimeError& error) {
      if (error.errorCode() != error_code::kNoCacheSpace) {
        throw;
      }
      // The cache has no room.
      return {};
    }
    if (pin.empty()) {
      // Another reader is filling the entry.
      return pin;
    }
    auto* entry = pin.checkedEntry();
    entry->getAndClearFirstUseFlag();
    if (entry->isExclusive() && !entry->hasContiguousData()) {
      // KvikIO can only read into one block of memory, and the new entry
      // is not one block.
      return {};
    }
    return pin;
  }

  // Returns a pin on a filled cache entry that starts at 'offset', reading the
  // bytes from KvikIO on a miss. The entry may hold fewer than 'size' bytes.
  // Returns an empty pin when the piece must be read without the cache.
  cache::CachePin cachePiece(size_t offset, size_t size) {
    if (size > kMaxCacheEntryBytes) {
      return {};
    }
    auto otherReaderFill = folly::SemiFuture<bool>::makeEmpty();
    auto pin = findOrCreateEntry(offset, size, &otherReaderFill);
    if (pin.empty() && otherReaderFill.valid()) {
      // Another reader is filling this entry. Wait for it and look once more.
      // If it is still being filled, the caller reads it without the cache.
      std::move(otherReaderFill)
          .via(&folly::QueuedImmediateExecutor::instance())
          .wait();
      pin = findOrCreateEntry(offset, size, nullptr);
    }
    if (pin.empty()) {
      return pin;
    }
    auto* entry = pin.checkedEntry();
    if (!entry->isExclusive()) {
      const auto bytes = pieceSize(pin, size);
      countBytes("cudfKvikioCacheHitBytes", bytes);
      if (bytes < size) {
        count("cudfKvikioCacheReusedPrefixes");
        countBytes("cudfKvikioCacheReusedPrefixBytes", bytes);
      }
      return pin;
    }
    const auto bytesRead = kvikioSource->host_read(
        offset, size, reinterpret_cast<uint8_t*>(entry->contiguousData()));
    // Never make a partly filled entry readable. Throwing releases the
    // exclusive pin, which drops the entry.
    VELOX_CHECK_EQ(bytesRead, size, "Short KvikIO read while filling cache");
    entry->setExclusiveToShared();
    countBytes("cudfKvikioCacheMissBytes", size);
    return pin;
  }

  // Puts the range from 'offset' in the cache, piece by piece, and returns a
  // pin per piece. Stops at the first piece that cannot be cached. The rest of
  // the range is then read without the cache.
  std::vector<cache::CachePin> cacheRange(size_t offset, size_t bytes) {
    std::vector<cache::CachePin> pins;
    size_t cached = 0;
    while (cached < bytes) {
      auto pin = cachePiece(offset + cached, bytes - cached);
      if (pin.empty()) {
        break;
      }
      cached += pieceSize(pin, bytes - cached);
      pins.push_back(std::move(pin));
    }
    return pins;
  }

  static void
  copyFromEntry(const cache::CachePin& pin, size_t bytes, uint8_t* dst) {
    auto* entry = pin.checkedEntry();
    if (entry->hasContiguousData()) {
      std::memcpy(dst, entry->contiguousData(), bytes);
      return;
    }
    // Entries made by other readers may be split into several blocks.
    for (const auto& block : entry->dataRanges(bytes)) {
      std::memcpy(dst, block.data(), block.size());
      dst += block.size();
    }
  }

  size_t readToHost(size_t offset, size_t size, uint8_t* dst) {
    const auto bytes = trimToFile(offset, size);
    if (bytes == 0) {
      return 0;
    }
    VELOX_CHECK_NOT_NULL(dst);
    size_t copied = 0;
    while (copied < bytes) {
      const auto wanted = bytes - copied;
      auto pin = cachePiece(offset + copied, wanted);
      if (pin.empty()) {
        // Keep the pieces already copied. Read only the rest.
        const auto bytesRead =
            kvikioSource->host_read(offset + copied, wanted, dst + copied);
        countBytes("cudfKvikioCacheBypassBytes", bytesRead);
        return copied + bytesRead;
      }
      const auto pieceBytes = pieceSize(pin, wanted);
      copyFromEntry(pin, pieceBytes, dst + copied);
      copied += pieceBytes;
    }
    return copied;
  }

  std::unique_ptr<cudf::io::datasource::buffer> readToHost(
      size_t offset,
      size_t size) {
    std::vector<uint8_t> bytes(trimToFile(offset, size));
    bytes.resize(readToHost(offset, bytes.size(), bytes.data()));
    return cudf::io::datasource::buffer::create(std::move(bytes));
  }

  // Queues the GPU copy of 'bytes' bytes of a cached piece, through this
  // thread's pinned buffer.
  static void copyCachedPieceToGpu(
      cache::CachePin pin,
      size_t bytes,
      uint8_t* dst,
      const DeviceRead& read) {
    auto& buffer = pinnedBuffer(read.device);
    copyFromEntry(pin, bytes, buffer.prepare(bytes));
    // The bytes are in the pinned buffer. The cache may evict the entry.
    pin.clear();
    buffer.copyToGpu(bytes, dst, read.stream);
  }

  // Reads 'bytes' bytes at 'offset' from KvikIO, without the cache, and queues
  // their GPU copy through this thread's pinned buffer. Returns the number of
  // bytes read.
  size_t copyUncachedToGpu(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      const DeviceRead& read) {
    auto& buffer = pinnedBuffer(read.device);
    const auto bytesRead =
        kvikioSource->host_read(offset, bytes, buffer.prepare(bytes));
    countBytes("cudfKvikioCacheBypassBytes", bytesRead);
    buffer.copyToGpu(bytesRead, dst, read.stream);
    return bytesRead;
  }

  // Queues the GPU copies of the cached pieces in 'pins', from cacheRange(),
  // then reads the rest of the range without the cache. Returns the number of
  // bytes queued.
  size_t copyRangeToGpu(
      std::vector<cache::CachePin> pins,
      const DeviceRead& read) {
    const rmm::cuda_set_device_raii onDevice{rmm::cuda_device_id{read.device}};
    size_t queued = 0;
    try {
      for (auto& pin : pins) {
        const auto pieceBytes = pieceSize(pin, read.bytes - queued);
        copyCachedPieceToGpu(
            std::move(pin), pieceBytes, read.dst + queued, read);
        queued += pieceBytes;
      }
      if (queued < read.bytes) {
        queued += copyUncachedToGpu(
            read.offset + queued, read.bytes - queued, read.dst + queued, read);
      }
      return queued;
    } catch (...) {
      // Copies queued before the failure may still write to 'dst'. Wait for
      // them before the caller can free it.
      waitForStream(read.stream, read.device);
      throw;
    }
  }
};

CachingDataSource::CachingDataSource(
    std::unique_ptr<cudf::io::datasource> kvikioSource,
    std::string_view path,
    cache::AsyncDataCache* cache,
    std::shared_ptr<IoStats> ioStats)
    : reader_(
          std::make_shared<Reader>(
              std::move(kvikioSource),
              path,
              cache,
              std::move(ioStats))) {}

CachingDataSource::~CachingDataSource() = default;

size_t CachingDataSource::size() const {
  return reader_->kvikioSource->size();
}

bool CachingDataSource::supports_device_read() const {
  return true;
}

bool CachingDataSource::is_device_read_preferred(size_t bytes) const {
  return reader_->kvikioSource->is_device_read_preferred(bytes);
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::host_read(
    size_t offset,
    size_t bytes) {
  return reader_->readToHost(offset, bytes);
}

size_t CachingDataSource::host_read(size_t offset, size_t bytes, uint8_t* dst) {
  return reader_->readToHost(offset, bytes, dst);
}

// The host reads run when cuDF calls get(), on cuDF's thread.
std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CachingDataSource::host_read_async(size_t offset, size_t bytes) {
  return std::async(std::launch::deferred, [reader = reader_, offset, bytes] {
    return reader->readToHost(offset, bytes);
  });
}

std::future<size_t>
CachingDataSource::host_read_async(size_t offset, size_t bytes, uint8_t* dst) {
  return std::async(
      std::launch::deferred, [reader = reader_, offset, bytes, dst] {
        return reader->readToHost(offset, bytes, dst);
      });
}

std::future<size_t> CachingDataSource::device_read_async(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    cuda::stream_ref stream) {
  bytes = reader_->trimToFile(offset, bytes);
  if (bytes == 0) {
    std::promise<size_t> nothingToRead;
    nothingToRead.set_value(0);
    return nothingToRead.get_future();
  }
  VELOX_CHECK_NOT_NULL(dst);
  const DeviceRead read{offset, bytes, dst, stream, deviceOf(stream)};
  auto copiesQueuedPromise = std::make_shared<std::promise<size_t>>();
  auto copiesQueued = copiesQueuedPromise->get_future();

  // Step 1 on a fill thread, then step 2 on a copy thread.
  fillThreads()->add([reader = reader_, read, copiesQueuedPromise] {
    try {
      auto pins = reader->cacheRange(read.offset, read.bytes);
      copyThreads()->add([reader,
                          read,
                          copiesQueuedPromise,
                          pins = std::move(pins)]() mutable {
        try {
          copiesQueuedPromise->set_value(
              reader->copyRangeToGpu(std::move(pins), read));
        } catch (...) {
          copiesQueuedPromise->set_exception(std::current_exception());
        }
      });
    } catch (...) {
      copiesQueuedPromise->set_exception(std::current_exception());
    }
  });

  // The read is running now. If creating the future below throws, the
  // WaitForGpuCopies inside it is destroyed first, which waits for the read.
  return std::async(
      std::launch::deferred,
      [wait = WaitForGpuCopies(
           std::move(copiesQueued), read.stream, read.device)]() mutable {
        return wait.get();
      });
}

size_t CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    cuda::stream_ref stream) {
  bytes = reader_->trimToFile(offset, bytes);
  if (bytes == 0) {
    return 0;
  }
  VELOX_CHECK_NOT_NULL(dst);
  const DeviceRead read{offset, bytes, dst, stream, deviceOf(stream)};
  // Run on the caller's thread. Handing the read to a pool and waiting could
  // deadlock if the caller is itself a pool thread that the read needs.
  const auto queued =
      reader_->copyRangeToGpu(reader_->cacheRange(offset, bytes), read);
  waitForStream(read.stream, read.device);
  return queued;
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    cuda::stream_ref stream) {
  const auto readSize = reader_->trimToFile(offset, bytes);
  rmm::device_buffer result(
      readSize, stream, cudf::get_current_device_resource_ref());
  const auto bytesRead = device_read(
      offset, readSize, static_cast<uint8_t*>(result.data()), stream);
  result.resize(bytesRead, stream);
  return datasource::buffer::create(std::move(result));
}

std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> kvikioSource,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats) {
  if (cache == nullptr || !cacheable) {
    return kvikioSource;
  }
  return std::make_unique<CachingDataSource>(
      std::move(kvikioSource), path, cache, std::move(ioStats));
}

} // namespace facebook::velox::cudf_velox::connector::hive
