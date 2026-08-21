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
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderIOHelpers.h"

#include "velox/common/Casts.h"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/system/HardwareConcurrency.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Runs fetch reads off-thread with a bounded number of in-flight batches.
folly::CPUThreadPoolExecutor& asyncReadExecutor() {
  constexpr size_t kMaxThreads{8};
  static folly::CPUThreadPoolExecutor executor{std::clamp(
      static_cast<size_t>(folly::available_concurrency()),
      size_t{1},
      kMaxThreads)};
  return executor;
}

// Submits fetch reads. Returned future carries any failure.
std::future<void> submitAsyncRead(std::function<void()> reads) {
  auto task = std::make_shared<std::packaged_task<void()>>(std::move(reads));
  auto pending = task->get_future();
  asyncReadExecutor().add([task]() { (*task)(); });
  return pending;
}

// Reads adjacent byte ranges into device memory.
struct CoalescedRead {
  size_t offset;
  size_t size;
  uint8_t* destination;
};

// Merges adjacent ranges to minimize data-source reads.
std::vector<CoalescedRead> coalesceByteRanges(
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    const std::vector<cudf::device_span<const uint8_t>>& byteRangeData) {
  std::vector<CoalescedRead> reads;
  reads.reserve(byteRanges.size());

  for (size_t range = 0; range < byteRanges.size();) {
    const auto offset = static_cast<size_t>(byteRanges[range].offset());
    auto size = static_cast<size_t>(byteRanges[range].size());
    size_t nextRange = range + 1;
    while (nextRange < byteRanges.size() and
           static_cast<size_t>(byteRanges[nextRange].offset()) ==
               offset + size) {
      size += static_cast<size_t>(byteRanges[nextRange].size());
      ++nextRange;
    }
    if (size != 0) {
      reads.push_back(
          {.offset = offset,
           .size = size,
           .destination = const_cast<uint8_t*>(byteRangeData[range].data())});
    }
    range = nextRange;
  }

  return reads;
}

// Helper for the KvikIO path to read coalesced byte ranges into corresponding
// destinations either directly or via host memory (requires a stream sync).
void readByteRangesToDevice(
    cudf::io::datasource& dataSource,
    const std::vector<CoalescedRead>& reads,
    rmm::cuda_stream_view stream) {
  // Schedule host reads for this caller threads with no interleaving
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> hostBuffers;
  std::vector<const void*> copySources;
  std::vector<void*> copyDestinations;
  std::vector<size_t> copySizes;
  {
    std::lock_guard<std::mutex> lock(hostReadMutex());
    for (const auto& read : reads) {
      if (dataSource.is_device_read_preferred(read.size)) {
        continue;
      }
      hostBuffers.push_back(dataSource.host_read(read.offset, read.size));
      copySources.push_back(hostBuffers.back()->data());
      copyDestinations.push_back(read.destination);
      copySizes.push_back(read.size);
    }
  }

  std::vector<std::future<size_t>> deviceReadTasks;
  deviceReadTasks.reserve(reads.size());

  // `device_read_async` for KvikIO data source is not guaranteed to follow
  // stream-ordering (see datasource API docs).
  stream.synchronize();

  // Submit all device reads for this caller together to prevent interleaving
  // across fetches.
  {
    std::lock_guard<std::mutex> lock(deviceReadMutex());
    for (const auto& read : reads) {
      if (dataSource.is_device_read_preferred(read.size)) {
        deviceReadTasks.emplace_back(dataSource.device_read_async(
            read.offset, read.size, read.destination, stream));
      }
    }

    if (not hostBuffers.empty()) {
      CUDF_CUDA_TRY(
          cudf::detail::memcpy_batch_async(
              copyDestinations.data(),
              copySources.data(),
              copySizes.data(),
              copyDestinations.size(),
              stream));
    }
  }

  // Wait for all device reads to complete.
  for (auto& task : deviceReadTasks) {
    task.get();
  }

  // Keep host buffers alive (if any) until H2D batched copy finishes.
  if (not hostBuffers.empty()) {
    stream.synchronize();
  }
}

} // namespace

bool ByteRangeReadGuard::tryStart() {
  auto pending = State::kPending;
  return state_.compare_exchange_strong(pending, State::kReading);
}

bool ByteRangeReadGuard::tryCancel() {
  auto pending = State::kPending;
  return state_.compare_exchange_strong(pending, State::kCancelled);
}

void ByteRangeFetch::wait() {
  if (pending.valid()) {
    pending.get();
  }
}

void ByteRangeFetch::abandon() {
  if (not pending.valid()) {
    return;
  }

  // Cancel pending reads rather than read and discard them.
  if (readGuard != nullptr and readGuard->tryCancel()) {
    if (cancelCleanup) {
      cancelCleanup();
    }
    pending = {};
    return;
  }

  pending.get();
}

std::mutex& deviceReadMutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& hostReadMutex() {
  static std::mutex mutex;
  return mutex;
}

std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info>
        pageIndexByteRanges) {
  std::vector<std::reference_wrapper<cudf::io::datasource>> dataSources{
      std::ref(*dataSource)};
  auto buffers = cudf::io::parquet::fetch_page_indexes_to_host(
      dataSources, pageIndexByteRanges);

  std::vector<cudf::host_span<const uint8_t>> spans;
  spans.reserve(buffers.size());
  std::transform(
      buffers.begin(),
      buffers.end(),
      std::back_inserter(spans),
      [](const auto& buffer) {
        return cudf::host_span<const uint8_t>{*buffer};
      });

  return {std::move(buffers), std::move(spans)};
}

ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      size_t{0},
      [](auto accumulated, const auto& byteRange) {
        return accumulated + byteRange.size();
      });

  // Allocate single device buffer for all byte ranges.
  std::vector<rmm::device_buffer> byteRangeBuffers{};
  byteRangeBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute the device span of each byte range within that buffer.
  std::vector<cudf::device_span<const uint8_t>> byteRangeData{};
  byteRangeData.reserve(byteRanges.size());
  auto* bufferData = static_cast<uint8_t*>(byteRangeBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      size_t{0},
      [&](auto accumulated, const auto& byteRange) {
        byteRangeData.emplace_back(
            bufferData + accumulated, static_cast<size_t>(byteRange.size()));
        return accumulated + byteRange.size();
      });

  auto readGuard = std::make_shared<ByteRangeReadGuard>();

  // For BufferedInputDataSource, enqueue regions and plan here so that its
  // prefetchable reads are handed to the Velox IO executor before
  // actual load asynchronously.
  if (auto* bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    {
      std::scoped_lock<std::mutex> lock(hostReadMutex());
      for (size_t range = 0; range < byteRanges.size(); ++range) {
        bufferedInput->enqueueForDevice(
            static_cast<uint64_t>(byteRanges[range].offset()),
            static_cast<uint64_t>(byteRanges[range].size()),
            const_cast<uint8_t*>(byteRangeData[range].data()));
      }
      bufferedInput->startLoad();
    }

    auto pending = submitAsyncRead([dataSource, stream, readGuard]() {
      if (not readGuard->tryStart()) {
        return;
      }
      checkedPointerCast<BufferedInputDataSource>(dataSource.get())
          ->finishLoad(stream);
    });

    return {
        .buffers = std::move(byteRangeBuffers),
        .data = std::move(byteRangeData),
        .pending = std::move(pending),
        .readGuard = std::move(readGuard),
        .cancelCleanup = [dataSource]() {
          checkedPointerCast<BufferedInputDataSource>(dataSource.get())
              ->discardPendingLoads();
        }};
  }

  // For KvikIO data source path, coalesce byte ranges and submit async host
  // and/or device reads.
  auto pending =
      submitAsyncRead([dataSource,
                       reads = coalesceByteRanges(byteRanges, byteRangeData),
                       stream,
                       readGuard]() {
        if (not readGuard->tryStart()) {
          return;
        }
        readByteRangesToDevice(*dataSource, reads, stream);
      });

  return {
      .buffers = std::move(byteRangeBuffers),
      .data = std::move(byteRangeData),
      .pending = std::move(pending),
      .readGuard = std::move(readGuard),
      .cancelCleanup = nullptr};
}

} // namespace facebook::velox::cudf_velox::connector::hive
