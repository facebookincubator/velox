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

#include <future>
#include <memory>
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

  // Use the enqueue API from dwio::common::BufferedInput.
  // Pass a device buffer to copy to after load.
  void enqueueForDevice(uint64_t offset, uint64_t size, uint8_t* dst);

  // Plans the reads of the regions enqueued since the previous load and
  // submits the prefetchable ones to the IO executor. Returns without waiting
  // for them, so it is safe to call from a task running on that executor.
  void startLoad();

  // Drains the regions enqueued since the previous load and copies them to
  // the device buffers they were enqueued with. Blocks until every read of
  // the batch has completed.
  void finishLoad(rmm::cuda_stream_view stream);

 private:
  // A region enqueued for reading into device memory. `startLoad()` plans the
  // read of `stream` and `finishLoad()` drains it into `dst`.
  struct PendingDeviceLoad {
    std::shared_ptr<facebook::velox::dwio::common::SeekableInputStream> stream;
    uint8_t* dst;
    uint64_t size;
  };

  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  std::vector<PendingDeviceLoad> pendingDeviceLoads_;
};

// Tracks progress of byte ranges being fetched into device memory.
struct ByteRangeFetch {
  // Stores physical device data.
  std::vector<rmm::device_buffer> buffers;

  // Device span into `buffers` for each requested byte range.
  std::vector<cudf::device_span<const uint8_t>> data;

  // Waits for all reads to complete. Must be waited on before using `data`.
  std::future<void> pending;

  // Indicates whether reads must complete before releasing `buffers`.
  bool writesInFlight{false};
};

// Tracks row group pass materialization progress for a split.
struct RowGroupPassState {
  // Stores row group pass information in read order.
  std::vector<std::vector<std::vector<cudf::size_type>>> passes;

  // Tracks the current pass being materialized.
  size_t currentPass{0};

  // Indicates whether chunking is set up for the current pass.
  bool isChunkingSetup{false};

  // Owns the device data for the current pass.
  ByteRangeFetch fetch;
};

// Asynchronously fetches byte ranges from a data source into device buffers.
ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Asyncronously fetches page-index buffers to host and returns views into them.
std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> pageIndexByteRanges);

} // namespace facebook::velox::cudf_velox::connector::hive
