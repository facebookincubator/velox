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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderIOHelpers.h"

#include <cudf/io/parquet_io_utils.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

void ByteRangeFetch::wait() {
  if (pending.valid()) {
    pending.get();
  }
}

void ByteRangeFetch::abandon() {
  wait();
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
  auto [buffers, data, pending] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          *dataSource, {byteRanges.data(), byteRanges.size()}, stream, mr);
  return {
      .buffers = std::move(buffers),
      .data = std::move(data),
      .pending = std::move(pending)};
}

} // namespace facebook::velox::cudf_velox::connector::hive
