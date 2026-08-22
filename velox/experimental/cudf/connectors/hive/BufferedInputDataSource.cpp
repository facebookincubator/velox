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

#include "velox/common/base/Exceptions.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/system/HardwareConcurrency.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Runs blocking fetch region reads outside Velox IO executor with higher
// concurrency; a few in flight do not saturate storage.
folly::Executor* regionReadExecutor() {
  constexpr auto kMaxThreads = size_t{64};
  static folly::CPUThreadPoolExecutor executor{std::clamp(
      static_cast<size_t>(folly::available_concurrency()),
      size_t{1},
      kMaxThreads)};
  return &executor;
}

// Reads pending-load region and records each contiguous span as a copy from
// BufferedInput-owned buffers, without host staging.
HostToDeviceCopies collectRegionCopies(
    const BufferedInputDataSource::PendingDeviceLoad& pendingLoad) {
  HostToDeviceCopies copies;
  uint64_t collected = 0;
  const void* data = nullptr;
  int32_t size = 0;
  while (collected < pendingLoad.size and
         pendingLoad.stream->Next(&data, &size)) {
    const auto spanSize =
        std::min<uint64_t>(size, pendingLoad.size - collected);
    copies.sources.push_back(data);
    copies.destinations.push_back(pendingLoad.dst + collected);
    copies.sizes.push_back(spanSize);
    collected += spanSize;
  }
  VELOX_CHECK_EQ(
      collected,
      pendingLoad.size,
      "BufferedInput served fewer bytes than the enqueued region");
  return copies;
}

} // namespace

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

  // One task per region: a single thread does not saturate storage bandwidth.
  std::vector<folly::Future<HostToDeviceCopies>> reads;
  reads.reserve(pendingLoads.size());
  for (const auto& pendingLoad : pendingLoads) {
    reads.push_back(folly::via(regionReadExecutor(), [&pendingLoad]() {
      return collectRegionCopies(pendingLoad);
    }));
  }

  // Waits for every read: the running ones reference this frame.
  auto regionCopies = folly::collectAll(std::move(reads)).get();
  for (auto& regionCopy : regionCopies) {
    regionCopy.throwUnlessValue();
  }

  const auto spans = std::accumulate(
      regionCopies.begin(),
      regionCopies.end(),
      size_t{0},
      [](auto accumulated, const auto& regionCopy) {
        return accumulated + regionCopy->sizes.size();
      });
  HostToDeviceCopies copies{spans};

  for (auto& regionCopy : regionCopies) {
    copies.append(std::move(*regionCopy));
  }

  {
    // Submit the copies of the batch without interleaving them with the
    // batches of other fetches.
    std::scoped_lock<std::mutex> lock(deviceReadMutex());
    copies.submitAsync(stream);
  }

  // The source buffers must remain valid until the copies finish.
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
