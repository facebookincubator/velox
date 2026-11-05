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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/io/parquet_io_utils.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

std::vector<std::reference_wrapper<cudf::io::datasource>> toReferences(
    std::span<const std::shared_ptr<cudf::io::datasource>> dataSources) {
  std::vector<std::reference_wrapper<cudf::io::datasource>> references;
  references.reserve(dataSources.size());
  for (const auto& dataSource : dataSources) {
    references.push_back(std::ref(*dataSource));
  }
  return references;
}

cudf::io::parquet::io_submission_policy submissionPolicy(
    bool serializeSubmissions) {
  return serializeSubmissions
      ? cudf::io::parquet::io_submission_policy::SERIALIZE
      : cudf::io::parquet::io_submission_policy::INTERLEAVE;
}

} // namespace

void ByteRangeFetch::wait() {
  if (pending.valid()) {
    pending.get();
  }
}

std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    std::span<const std::shared_ptr<cudf::io::datasource>> dataSources,
    cudf::host_span<const cudf::io::text::byte_range_info>
        pageIndexByteRanges) {
  auto buffers = cudf::io::parquet::fetch_page_indexes_to_host(
      toReferences(dataSources), pageIndexByteRanges);

  std::vector<cudf::host_span<const uint8_t>> spans;
  spans.reserve(buffers.size());
  std::transform(
      buffers.begin(),
      buffers.end(),
      std::back_inserter(spans),
      [](const auto& buffer) {
        return buffer->size() == 0 ? cudf::host_span<const uint8_t>{}
                                   : cudf::host_span<const uint8_t>{*buffer};
      });

  return {std::move(buffers), std::move(spans)};
}

ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr) {
  auto [buffers, data, pending] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          *dataSource,
          {byteRanges.data(), byteRanges.size()},
          submissionPolicy(serializeSubmissions),
          stream,
          mr);
  return {
      .buffers = std::move(buffers),
      .data = std::move(data),
      .pending = std::move(pending)};
}

ByteRangeFetch fetchByteRangesAsync(
    std::span<const std::shared_ptr<cudf::io::datasource>> dataSources,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    cudf::host_span<const cudf::size_type> sourceIndices,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr) {
  if (dataSources.size() == 1) {
    return fetchByteRangesAsync(
        dataSources.front(), byteRanges, serializeSubmissions, stream, mr);
  }
  VELOX_CHECK_EQ(
      byteRanges.size(),
      sourceIndices.size(),
      "Expected a data source index per byte range");

  // Group the ranges by data source, remembering where each one goes back.
  std::vector<std::vector<cudf::io::text::byte_range_info>> sourceByteRanges(
      dataSources.size());
  std::vector<std::pair<cudf::size_type, size_t>> positions;
  positions.reserve(byteRanges.size());
  for (size_t i = 0; i < byteRanges.size(); ++i) {
    const auto source = sourceIndices[i];
    VELOX_CHECK(
        source >= 0 and std::cmp_less(source, dataSources.size()),
        "Byte range data source index {} out of range",
        source);
    positions.emplace_back(source, sourceByteRanges[source].size());
    sourceByteRanges[source].push_back(byteRanges[i]);
  }

  auto [buffers, sourceData, pending] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          toReferences(dataSources),
          sourceByteRanges,
          submissionPolicy(serializeSubmissions),
          stream,
          mr);

  std::vector<cudf::device_span<const uint8_t>> data;
  data.reserve(positions.size());
  for (const auto& [source, index] : positions) {
    data.push_back(sourceData[source][index]);
  }
  return {
      .buffers = std::move(buffers),
      .data = std::move(data),
      .pending = std::move(pending)};
}

} // namespace facebook::velox::cudf_velox::connector::hive
