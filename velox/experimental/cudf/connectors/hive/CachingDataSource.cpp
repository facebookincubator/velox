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
#include <future>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

// A cuDF datasource for a file on S3 that serves every read (host_read,
// host_read_async, device_read and device_read_async) from Velox's async data
// cache, and does remote reads through KvikIO for the bytes it lacks.
//
// The requested byte range is served piece by piece, starting at its offset.
// Each piece comes from one async data cache entry, looked up by (file, offset
// of the piece):
//   - Entry found: copy its bytes. The entry may be shorter than the rest of
//     the range (CacheEntrySizePolicy::kAllowSmaller). The next piece then
//     starts where this entry ends.
//   - No entry: do a remote read of the rest of the range into a new entry,
//     then copy its bytes.
//   - Entry being filled by another read: wait for it, then look it up again.
//   - The async data cache cannot hold the piece (not enough free memory for
//     it, or 2 GiB or more, the most an entry can hold): do a remote read of
//     the rest of the range without the async data cache.
//
// Device reads copy each piece from host memory into a pinned buffer, then to
// the GPU. device_read_async() uses two thread pools: a fill thread brings the
// range into the async data cache, waiting for its remote reads, then a copy
// thread copies it to the GPU. The returned future is ready once the bytes are
// in GPU memory.

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// AsyncDataCache stores entry sizes in an int32_t. Larger pieces are read
// without the cache.
constexpr size_t kMaxCacheEntryBytes = std::numeric_limits<int32_t>::max();

// Threads that put ranges in the cache. A fill thread waits while KvikIO reads
// a miss. The thread count equals KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS
// (default 256), or 64 when that is 0 (concurrency unlimited).
folly::Executor* fillExecutor() {
  static auto* executor = [] {
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
  return executor;
}

// Threads that copy a range to the GPU, each through its own pinned buffer.
// The thread count is cudf.kvikio_cache_threads, kept low to limit pinned
// memory. A copy thread waits for a remote read only for the part of a range
// that the async data cache cannot hold.
folly::Executor* copyExecutor() {
  static auto* executor = [] {
    const auto threads = CudfConfig::getInstance().kvikioCacheThreads;
    VELOX_USER_CHECK_GT(
        threads, 0, "{} must be positive", CudfConfig::kCudfKvikioCacheThreads);
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfCacheCopy"));
  }();
  return executor;
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

// Blocks until every copy queued on 'stream' has finished. The caller may then
// use or free the destination.
void waitForStream(cuda::stream_ref stream, int device) {
  const rmm::cuda_set_device_raii onDevice{rmm::cuda_device_id{device}};
  try {
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  } catch (...) {
    // Make sure no copy still writes to the destination.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    throw;
  }
}

// Pinned host memory used as a bounce buffer for copies from the async data
// cache to the GPU. Bytes read without the async data cache use it too.
class PinnedBuffer {
 public:
  // Waits until the last copy out of the buffer has finished, grows the buffer
  // to at least 'bytes' if needed, and returns it.
  uint8_t* reserve(size_t bytes) {
    if (lastCopy_ != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(lastCopy_));
    }
    if (size_ < bytes) {
      if (data_ != nullptr) {
        allocator_.deallocate_sync(data_, size_);
        data_ = nullptr;
        size_ = 0;
      }
      data_ = static_cast<uint8_t*>(allocator_.allocate_sync(bytes));
      size_ = bytes;
    }
    return data_;
  }

  // Queues a copy of the first 'bytes' bytes of the buffer to 'dst' on the GPU.
  void copyToGpu(size_t bytes, uint8_t* dst, cuda::stream_ref stream) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst, data_, bytes, cudaMemcpyHostToDevice, stream.get()));
    if (lastCopy_ == nullptr) {
      CUDF_CUDA_TRY(
          cudaEventCreateWithFlags(&lastCopy_, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(lastCopy_, stream.get()));
  }

 private:
  rmm::host_device_async_resource_ref allocator_ =
      cudf::get_pinned_memory_resource();
  uint8_t* data_{nullptr};
  size_t size_{0};
  // Recorded after the last copy out of 'data_'.
  cudaEvent_t lastCopy_{nullptr};
};

// One pinned buffer per thread and GPU, kept until the process exits.
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

// Holds the KvikIO source and does the reads. Async work keeps it alive, since
// cuDF may destroy the CachingDataSource before that work finishes.
class Reader {
 public:
  Reader(
      std::unique_ptr<cudf::io::datasource> source,
      std::string_view path,
      cache::AsyncDataCache* dataCache,
      std::shared_ptr<IoStats> stats)
      : kvikioSource_(std::move(source)),
        cache_(dataCache),
        fileId_(fileIds(), std::string(path)),
        ioStats_(std::move(stats)) {
    VELOX_CHECK_NOT_NULL(kvikioSource_);
    VELOX_CHECK_NOT_NULL(cache_);
  }

  size_t fileSize() const {
    return kvikioSource_->size();
  }

  bool isDeviceReadPreferred(size_t bytes) const {
    return kvikioSource_->is_device_read_preferred(bytes);
  }

  // Returns how many of the 'bytes' bytes at 'offset' are inside the file.
  size_t trimToFile(size_t offset, size_t bytes) const {
    const auto fileSize = kvikioSource_->size();
    return offset < fileSize ? std::min(bytes, fileSize - offset) : 0;
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
            kvikioSource_->host_read(offset + copied, wanted, dst + copied);
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
      // Wait for the copies already queued before rethrowing.
      waitForStream(read.stream, read.device);
      throw;
    }
  }

 private:
  void countBytes(const std::string& name, size_t bytes) {
    if (ioStats_) {
      ioStats_->addCounter(
          name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
    }
  }

  void count(const std::string& name) {
    if (ioStats_) {
      ioStats_->addCounter(name, RuntimeCounter(1));
    }
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
      pin = cache_->findOrCreate(
          {fileId_.id(), offset},
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
    // Too big for one async data cache entry.
    if (size > kMaxCacheEntryBytes) {
      return {};
    }
    auto otherReaderFill = folly::SemiFuture<bool>::makeEmpty();
    auto pin = findOrCreateEntry(offset, size, &otherReaderFill);
    // Another read is filling this entry. Wait for it, then look it up again.
    if (pin.empty() && otherReaderFill.valid()) {
      std::move(otherReaderFill)
          .via(&folly::QueuedImmediateExecutor::instance())
          .wait();
      pin = findOrCreateEntry(offset, size, nullptr);
    }
    // The async data cache cannot be used for this piece: no room, or the
    // entry is still being filled after the wait.
    if (pin.empty()) {
      return pin;
    }
    auto* entry = pin.checkedEntry();
    // Hit: the entry is already filled. It may hold fewer than 'size' bytes.
    if (!entry->isExclusive()) {
      const auto bytes = pieceSize(pin, size);
      countBytes("cudfKvikioCacheHitBytes", bytes);
      if (bytes < size) {
        count("cudfKvikioCacheReusedPrefixes");
        countBytes("cudfKvikioCacheReusedPrefixBytes", bytes);
      }
      return pin;
    }
    // Miss: this read created the entry (exclusive) and fills it from S3.
    const auto bytesRead = kvikioSource_->host_read(
        offset, size, reinterpret_cast<uint8_t*>(entry->contiguousData()));
    // Never make a partly filled entry readable.
    VELOX_CHECK_EQ(bytesRead, size, "Short KvikIO read while filling cache");
    // Filled: make the entry readable by other reads (shared).
    entry->setExclusiveToShared();
    countBytes("cudfKvikioCacheMissBytes", size);
    return pin;
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

  // Queues the GPU copy of 'bytes' bytes of a cached piece, through this
  // thread's pinned buffer.
  static void copyCachedPieceToGpu(
      cache::CachePin pin,
      size_t bytes,
      uint8_t* dst,
      const DeviceRead& read) {
    auto& buffer = pinnedBuffer(read.device);
    copyFromEntry(pin, bytes, buffer.reserve(bytes));
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
        kvikioSource_->host_read(offset, bytes, buffer.reserve(bytes));
    countBytes("cudfKvikioCacheBypassBytes", bytesRead);
    buffer.copyToGpu(bytesRead, dst, read.stream);
    return bytesRead;
  }

  std::unique_ptr<cudf::io::datasource> kvikioSource_;
  cache::AsyncDataCache* cache_;
  StringIdLease fileId_;
  std::shared_ptr<IoStats> ioStats_;
};

// The datasource cuDF reads from. The reads themselves are done by Reader.
class CachingDataSource final : public cudf::io::datasource {
 public:
  CachingDataSource(
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

  size_t size() const override {
    return reader_->fileSize();
  }

  bool supports_device_read() const override {
    return true;
  }

  bool is_device_read_preferred(size_t bytes) const override {
    return reader_->isDeviceReadPreferred(bytes);
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t bytes) override {
    return reader_->readToHost(offset, bytes);
  }

  size_t host_read(size_t offset, size_t bytes, uint8_t* dst) override {
    return reader_->readToHost(offset, bytes, dst);
  }

  // The host reads run when cuDF calls get(), on cuDF's thread.
  std::future<std::unique_ptr<buffer>> host_read_async(
      size_t offset,
      size_t bytes) override {
    return std::async(std::launch::deferred, [reader = reader_, offset, bytes] {
      return reader->readToHost(offset, bytes);
    });
  }

  std::future<size_t> host_read_async(size_t offset, size_t bytes, uint8_t* dst)
      override {
    return std::async(
        std::launch::deferred, [reader = reader_, offset, bytes, dst] {
          return reader->readToHost(offset, bytes, dst);
        });
  }

  std::future<size_t> device_read_async(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      cuda::stream_ref stream) override {
    bytes = reader_->trimToFile(offset, bytes);
    if (bytes == 0) {
      std::promise<size_t> nothingToRead;
      nothingToRead.set_value(0);
      return nothingToRead.get_future();
    }
    VELOX_CHECK_NOT_NULL(dst);
    const DeviceRead read{offset, bytes, dst, stream, deviceOf(stream)};
    // Ready once the copy thread has queued every GPU copy of the read.
    auto copiesQueuedPromise = std::make_shared<std::promise<size_t>>();
    auto copiesQueued = copiesQueuedPromise->get_future();

    // Put the range in the cache on a fill thread, then copy it to the GPU on
    // a copy thread.
    fillExecutor()->add([reader = reader_, read, copiesQueuedPromise] {
      try {
        auto pins = reader->cacheRange(read.offset, read.bytes);
        copyExecutor()->add([reader,
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

    // Return a future whose get() waits until the bytes are in GPU memory.
    return std::async(
        std::launch::deferred,
        [copiesQueued = std::move(copiesQueued), read]() mutable {
          const auto bytes = copiesQueued.get();
          waitForStream(read.stream, read.device);
          return bytes;
        });
  }

  size_t device_read(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      cuda::stream_ref stream) override {
    bytes = reader_->trimToFile(offset, bytes);
    if (bytes == 0) {
      return 0;
    }
    VELOX_CHECK_NOT_NULL(dst);
    const DeviceRead read{offset, bytes, dst, stream, deviceOf(stream)};
    // Runs on the caller's thread, not on the two pools.
    const auto queued =
        reader_->copyRangeToGpu(reader_->cacheRange(offset, bytes), read);
    waitForStream(read.stream, read.device);
    return queued;
  }

  std::unique_ptr<buffer>
  device_read(size_t offset, size_t bytes, cuda::stream_ref stream) override {
    const auto readSize = reader_->trimToFile(offset, bytes);
    rmm::device_buffer result(
        readSize, stream, cudf::get_current_device_resource_ref());
    const auto bytesRead = device_read(
        offset, readSize, static_cast<uint8_t*>(result.data()), stream);
    result.resize(bytesRead, stream);
    return buffer::create(std::move(result));
  }

 private:
  std::shared_ptr<Reader> reader_;
};

} // namespace

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
