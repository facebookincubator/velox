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

  // Use the enqueue API from dwio::common::BufferedInput.
  // Pass a device buffer to copy to after load.
  void enqueueForDevice(uint64_t offset, uint64_t size, uint8_t* dst);

  // loads and copies to device.
  void load(rmm::cuda_stream_view stream);

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  std::vector<std::function<void(rmm::cuda_stream_view stream)>>
      pendingDeviceLoads_;
};

/**
 * @brief Hybrid scan reader state
 *
 * This struct is used to store the column chunk data for the hybrid scan reader
 * and a once flag to ensure the setup is only done once.
 */
struct HybridScanState {
  HybridScanState() : isHybridScanSetup_(std::make_unique<std::once_flag>()) {}

  std::vector<rmm::device_buffer> columnChunkBuffers_;
  std::vector<cudf::device_span<uint8_t const>> columnChunkData_;
  std::unique_ptr<std::once_flag> isHybridScanSetup_;
};

/**
 * @brief Fetches a host buffer of Parquet footer bytes from the input data
 * source
 *
 * @param dataSource Input data source
 * @return Host buffer containing footer bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource);

/**
 * @brief Fetches a host buffer of Parquet page index from the input data source
 *
 * @param dataSource Input datasource
 * @param pageIndexBytes Byte range of page index
 * @return Host buffer containing page index bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetchPageIndexBytes(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::io::text::byte_range_info const pageIndexBytes);

/**
 * @brief Fetches a list of byte ranges from a host buffer into device buffers
 *
 * @param dataSource Input datasource
 * @param byteRanges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple containing the device buffers, the device spans of the
 * fetched data, and a future to wait on the read tasks
 */
std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<uint8_t const>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<cudf::io::text::byte_range_info const> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::connector::hive
