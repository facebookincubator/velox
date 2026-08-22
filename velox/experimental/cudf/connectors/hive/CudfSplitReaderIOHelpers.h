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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <future>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// Owns device buffers and tracks reads filling them.
struct ByteRangeFetch {
  // Device buffers for requested byte ranges.
  std::vector<rmm::device_buffer> buffers;

  // Device span per requested range. Valid after 'wait()'.
  std::vector<cudf::device_span<const uint8_t>> data;

  // Completes when reads finish. Invalid without an in-flight fetch.
  std::future<void> pending;

  // Waits for reads and rethrows any failure. No-op when complete.
  void wait();

  // Waits for active reads before releasing their destination buffers.
  void abandon();
};

// Starts cuDF's asynchronous byte-range fetch. Wait for returned fetch before
// using its data.
ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Reads page-index ranges into host buffers and returns them with their views.
std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> pageIndexByteRanges);

} // namespace facebook::velox::cudf_velox::connector::hive
