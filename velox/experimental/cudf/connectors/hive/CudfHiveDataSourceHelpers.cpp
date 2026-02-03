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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.hpp"

#include "velox/dwio/common/BufferedInput.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <folly/futures/Future.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace {
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

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
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
                          cudaMemcpyHostToDevice,
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

std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource) {
  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len = sizeof(file_ender_s);
  const size_t len = dataSource->size();

  const auto header_buffer = dataSource->host_read(0, header_len);
  const auto ender_buffer = dataSource->host_read(len - ender_len, ender_len);
  const auto header =
      reinterpret_cast<const file_header_s*>(header_buffer->data());
  const auto ender =
      reinterpret_cast<const file_ender_s*>(ender_buffer->data());
  VELOX_CHECK(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic =
      (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  VELOX_CHECK(
      header->magic == parquet_magic && ender->magic == parquet_magic,
      "Corrupted header or footer");
  VELOX_CHECK(
      ender->footer_len != 0 &&
          ender->footer_len <= (len - header_len - ender_len),
      "Incorrect footer length");

  return dataSource->host_read(
      len - ender->footer_len - ender_len, ender->footer_len);
}

std::vector<rmm::device_buffer> fetchByteRanges(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<cudf::io::text::byte_range_info const> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  static std::mutex mutex;
  std::vector<rmm::device_buffer> columnChunkBuffers(byteRanges.size());
  {
    std::lock_guard<std::mutex> lock(mutex);
    std::transform(
        byteRanges.begin(),
        byteRanges.end(),
        columnChunkBuffers.begin(),
        [&](const auto& byteRange) {
          // Pad the buffer size to be a multiple of 8 bytes
          constexpr size_t bufferPaddingMultiple = 8;
          auto buffer = rmm::device_buffer(
              cudf::util::round_up_safe<size_t>(
                  byteRange.size(), bufferPaddingMultiple),
              stream,
              mr);
          // Directly read the column chunk data to the device buffer if
          // supported
          if (dataSource->supports_device_read() and
              dataSource->is_device_read_preferred(byteRange.size())) {
            std::ignore = dataSource->device_read_async(
                byteRange.offset(),
                byteRange.size(),
                static_cast<uint8_t*>(buffer.data()),
                stream);
          } else {
            // Read the column chunk data to the host buffer and copy it to
            // the device buffer
            auto hostBuffer =
                dataSource->host_read(byteRange.offset(), byteRange.size());
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                buffer.data(),
                hostBuffer->data(),
                byteRange.size(),
                cudaMemcpyHostToDevice,
                stream.value()));
          }
          return buffer;
        });
  }
  return columnChunkBuffers;
}

} // namespace facebook::velox::cudf_velox::connector::hive
