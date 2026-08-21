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

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

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
  pendingDeviceLoads_.push_back(
      {.stream = input_->enqueue({offset, size}), .dst = dst, .size = size});
}

void BufferedInputDataSource::startLoad() {
  // Plans coalesced reads and starts prefetchable ones without waiting.
  input_->load(velox::dwio::common::LogType::FILE);
}

void BufferedInputDataSource::discardPendingLoads() {
  pendingDeviceLoads_.clear();
}

void BufferedInputDataSource::finishLoad(rmm::cuda_stream_view stream) {
  // Consumes each enqueued stream once to avoid rereading exhausted streams.
  const auto pendingLoads = std::exchange(pendingDeviceLoads_, {});
  if (pendingLoads.empty()) {
    return;
  }

  // Prepare copy destinations and sizes while computing batch size.
  std::vector<void*> copyDestinations;
  std::vector<size_t> copySizes;
  copyDestinations.reserve(pendingLoads.size());
  copySizes.reserve(pendingLoads.size());
  const auto totalSize = std::accumulate(
      pendingLoads.begin(),
      pendingLoads.end(),
      size_t{0},
      [&](auto accumulated, const auto& pendingLoad) {
        copyDestinations.push_back(pendingLoad.dst);
        copySizes.push_back(pendingLoad.size);
        return accumulated + pendingLoad.size;
      });

  // Allocate single pinned buffer for the whole batch.
  auto hostBuffer =
      cudf::detail::make_pinned_vector_async<uint8_t>(totalSize, stream);

  // Read each region into its buffer slice.
  std::vector<const void*> copySources;
  copySources.reserve(pendingLoads.size());

  size_t bufferOffset = 0;
  for (const auto& pendingLoad : pendingLoads) {
    auto* copySource = hostBuffer.data() + bufferOffset;
    pendingLoad.stream->readFully(
        reinterpret_cast<char*>(copySource), pendingLoad.size);
    copySources.push_back(copySource);
    bufferOffset += pendingLoad.size;
  }

  {
    // Submit the copies of the batch without interleaving them with the
    // batches of other fetches.
    std::scoped_lock<std::mutex> lock(deviceReadMutex());
    CUDF_CUDA_TRY(
        cudf::detail::memcpy_batch_async(
            copyDestinations.data(),
            copySources.data(),
            copySizes.data(),
            copyDestinations.size(),
            stream));
  }

  // The staging buffer must remain valid until the copies finish.
  stream.synchronize();
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

} // namespace facebook::velox::cudf_velox::connector::hive
