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

#pragma once

#include "velox/dwio/common/BufferedInput.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// ---------------- Internal helper ----------------
// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  explicit BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
};

// Fetch a host buffer containing parquet source footer from a data source.
std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource);

/**
 * @brief Converts a span of device buffers into a vector of corresponding
 * device spans
 *
 * @tparam T Type of output device spans
 * @param buffers Host span of device buffers
 * @return Device spans corresponding to the input device buffers
 */
template <typename T>
std::vector<cudf::device_span<T const>> makeDeviceSpans(
    cudf::host_span<rmm::device_buffer const> buffers)
  requires(sizeof(T) == 1)
{
  std::vector<cudf::device_span<T const>> deviceSpans(buffers.size());
  std::transform(
      buffers.begin(),
      buffers.end(),
      deviceSpans.begin(),
      [](auto const& buffer) {
        return cudf::device_span<T const>{
            static_cast<T const*>(buffer.data()), buffer.size()};
      });
  return deviceSpans;
}

/**
 * @brief Fetches a list of byte ranges from the data source into device buffers
 *
 * @param dataSource Data source
 * @param byteRanges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Device buffers containing the fetched byte ranges
 */
std::vector<rmm::device_buffer> fetchByteRanges(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<cudf::io::text::byte_range_info const> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::connector::hive
