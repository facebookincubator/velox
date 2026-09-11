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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/Casts.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <folly/futures/Future.h>
#include <folly/system/HardwareConcurrency.h>

#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <vector>

namespace {

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

// Executor for remote read with a number of threads equal to:
// - Environment variable KVIKIO_NTHREADS if specified.
// - Otherwise, 5 * CPUs.
folly::Executor* remoteReadExecutor() {
  static auto* executor = [] {
    constexpr size_t kThreadsPerCore = 5;
    size_t numThreads = folly::available_concurrency() * kThreadsPerCore;
    if (const char* value = std::getenv("KVIKIO_NTHREADS")) {
      if (const auto parsed = std::strtoull(value, nullptr, 10); parsed > 0) {
        numThreads = parsed;
      }
    }
    return new folly::CPUThreadPoolExecutor(
        numThreads,
        std::make_shared<folly::NamedThreadFactory>("CudfRemoteIO"));
  }();
  return executor;
}

// A host buffer drawn from cuDF's pinned memory pool.
class PinnedStagingBuffer {
 public:
  explicit PinnedStagingBuffer(size_t size)
      : mr_(cudf::get_pinned_memory_resource()),
        size_(size),
        data_(static_cast<uint8_t*>(mr_.allocate_sync(size))) {}

  ~PinnedStagingBuffer() {
    mr_.deallocate_sync(data_, size_);
  }

  PinnedStagingBuffer(const PinnedStagingBuffer&) = delete;
  PinnedStagingBuffer& operator=(const PinnedStagingBuffer&) = delete;

  uint8_t* data() const {
    return data_;
  }

 private:
  rmm::host_device_async_resource_ref mr_;
  size_t size_;
  uint8_t* data_;
};

// One reusable pinned staging buffer per thread, plus the event marking its
// last H2D copy.
struct PinnedStagingSlot {
  PinnedStagingBuffer* buffer{nullptr};
  size_t capacity{0};
  cudaEvent_t event{nullptr};

  uint8_t* reserve(size_t size) {
    // First wait for this slot's previous copy to land so the memory is safe to
    // overwrite
    if (event != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(event));
    }
    if (capacity < size) {
      delete buffer;
      buffer = new PinnedStagingBuffer(size);
      capacity = size;
    }
    return buffer->data();
  }

  void recordCopy(cudaStream_t stream) {
    if (event == nullptr) {
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  }
};

CachingDataSource::CachingDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    const std::string& path,
    folly::Executor* executor)
    : delegate_(std::move(delegate)),
      fileSize_(delegate_->size()),
      executor_(executor),
      cache_(velox::cache::AsyncDataCache::getInstance()),
      fileNum_(fileIds(), path) {}

CachingDataSource::~CachingDataSource() = default;

size_t CachingDataSource::size() const {
  return fileSize_;
}

bool CachingDataSource::supports_device_read() const {
  return true;
}

bool CachingDataSource::is_device_read_preferred(size_t size) const {
  return delegate_->is_device_read_preferred(size);
}

velox::cache::CachePin CachingDataSource::pinRange(
    uint64_t offset,
    uint64_t size) {
  const velox::cache::RawFileCacheKey key{fileNum_.id(), offset};
  constexpr int kMaxAttempts = 32;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    folly::SemiFuture<bool> wait(false);
    velox::cache::CachePin pin;
    try {
      pin = cache_->findOrCreate(key, size, /*contiguous=*/true, &wait);
    } catch (const VeloxRuntimeError&) {
      // Fall back to an uncached read.
      return {};
    }
    if (pin.empty()) {
      if (!wait.valid()) {
        return {};
      }
      std::move(wait).via(&folly::QueuedImmediateExecutor::instance()).wait();
      continue;
    }
    auto* entry = pin.checkedEntry();
    entry->getAndClearFirstUseFlag();
    // Hit
    if (!entry->isExclusive()) {
      return pin;
    }
    // Miss
    if (!entry->hasContiguousData()) {
      return {};
    }
    delegate_->host_read(
        offset, size, reinterpret_cast<uint8_t*>(entry->contiguousData()));
    entry->setExclusiveToShared();
    return pin;
  }
  return {};
}

void CachingDataSource::readThroughCache(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  auto pin = pinRange(offset, size);
  if (pin.empty()) {
    delegate_->host_read(offset, size, dst);
    return;
  }
  std::memcpy(dst, pin.checkedEntry()->contiguousData(), size);
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::host_read(
    size_t offset,
    size_t size) {
  if (cache_ == nullptr) {
    return delegate_->host_read(offset, size);
  }
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readThroughCache(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t CachingDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (cache_ == nullptr) {
    return delegate_->host_read(offset, size, dst);
  }
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readThroughCache(offset, readSize, dst);
  return readSize;
}

std::future<size_t> CachingDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  if (cache_ == nullptr || executor_ == nullptr) {
    return delegate_->device_read_async(offset, size, dst, stream);
  }
  const velox::cache::RawFileCacheKey key{fileNum_.id(), offset};
  auto* readExecutor = cache_->exists(key) ? executor_ : remoteReadExecutor();
  auto future =
      folly::via(readExecutor)
          .thenValue([this, offset, size, dst, stream](auto&&) -> size_t {
            return this->device_read(offset, size, dst, stream);
          });
  return toStdFuture(std::move(future));
}

size_t CachingDataSource::device_read(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  if (cache_ == nullptr) {
    return delegate_->device_read(offset, size, dst, stream);
  }
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  if (readSize == 0) {
    return 0;
  }
  static thread_local PinnedStagingSlot slot;
  uint8_t* staging = slot.reserve(readSize);
  readThroughCache(offset, readSize, staging);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, staging, readSize, cudaMemcpyDefault, stream.value()));
  slot.recordCopy(stream.value());
  return readSize;
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::device_read(
    size_t offset,
    size_t size,
    rmm::cuda_stream_view stream) {
  return delegate_->device_read(offset, size, stream);
}

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back(
      [dst, size, sharedStream](rmm::cuda_stream_view stream) {
        std::vector<uint8_t> buffer(size);
        sharedStream->readFully(reinterpret_cast<char*>(buffer.data()), size);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst, buffer.data(), size, cudaMemcpyDefault, stream.value()));
      });
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);
  std::lock_guard<std::mutex> lock(ioBatchMutex());
  for (auto& deviceLoad : pendingDeviceLoads_) {
    deviceLoad(stream);
  }
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyDefault,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) { return acc + byteRange.size(); });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData + acc, static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    auto iter =
        cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
    std::for_each(
        iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
          const auto& byteRange = cuda::std::get<0>(tuple);
          const auto& destination = cuda::std::get<1>(tuple);
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              const_cast<uint8_t*>(destination.data()));
        });

    // load buffered input data source
    auto syncFunction = [](std::shared_ptr<cudf::io::datasource> dataSource,
                           rmm::cuda_stream_view stream) {
      auto buffer =
          checkedPointerCast<BufferedInputDataSource>(dataSource.get());
      buffer->load(stream);
    };

    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, syncFunction, dataSource, stream)};
  }

  // KvikIO dataSource: Impl borrowed from `fetch_byte_ranges_to_device_async()`
  // in `parquet_io_utils.cpp` in cuDF.
  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<uint8_t*> destinations;

  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      if (nextOffset != ioOffset + ioSize) {
        break;
      }
      ioSize += byteRanges[nextChunk].size();
      nextChunk++;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinations.push_back(
          const_cast<uint8_t*>(columnChunkData[chunk].data()));
    }
    chunk = nextChunk;
  }
  VELOX_CHECK_EQ(
      ioOffsets.size(),
      ioSizes.size(),
      "Number of IO offsets and sizes must be equal");
  VELOX_CHECK_EQ(
      ioSizes.size(),
      destinations.size(),
      "Number of IO sizes and destinations must be equal");

  auto iter = cuda::make_zip_iterator(
      ioOffsets.begin(), ioSizes.begin(), destinations.begin());

  std::vector<std::future<size_t>> deviceReadTasks;
  std::vector<std::future<size_t>> hostReadTasks;
  deviceReadTasks.reserve(ioOffsets.size());
  hostReadTasks.reserve(ioOffsets.size());

  // device_read_async is not guaranteed to follow stream-ordering (see
  // datasource API docs)
  stream.synchronize();

  {
    std::lock_guard<std::mutex> lock(ioBatchMutex());

    std::for_each(iter, iter + ioOffsets.size(), [&](const auto& tuple) {
      const auto ioOffset = cuda::std::get<0>(tuple);
      const auto ioSize = cuda::std::get<1>(tuple);
      const auto dest = cuda::std::get<2>(tuple);

      if (dataSource->supports_device_read() and
          dataSource->is_device_read_preferred(ioSize)) {
        deviceReadTasks.emplace_back(
            dataSource->device_read_async(ioOffset, ioSize, dest, stream));
      } else {
        // TODO(mh): We can't yet guarantee (without a safe thread pool) that
        // all `cudaMemcpyAsync`s will be launched by the time we release the
        // mutex. That said, this is a rare usecase as host-buffer data should
        // prefer using a `BufferedInputDataSource` datasource.
        hostReadTasks.emplace_back(
            std::async(
                std::launch::async,
                [dataSource, ioOffset, ioSize, dest, stream]() {
                  auto hostBuffer = dataSource->host_read(ioOffset, ioSize);
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                      dest,
                      hostBuffer->data(),
                      hostBuffer->size(),
                      cudaMemcpyDefault,
                      stream.value()));
                  return ioSize;
                }));
      }
    });
  }

  auto syncFunction = [](decltype(hostReadTasks)&& hostReadTasks,
                         decltype(deviceReadTasks)&& deviceReadTasks) {
    for (auto& task : hostReadTasks) {
      task.get();
    }
    for (auto& task : deviceReadTasks) {
      task.get();
    }
  };

  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(
          std::launch::deferred,
          std::move(syncFunction),
          std::move(hostReadTasks),
          std::move(deviceReadTasks))};
}

} // namespace facebook::velox::cudf_velox::connector::hive
