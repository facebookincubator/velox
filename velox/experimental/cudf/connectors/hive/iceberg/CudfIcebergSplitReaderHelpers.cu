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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReaderHelpers.hpp"

#include <cudf/column/column.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

#include <cstring>
#include <stdexcept>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

using roaring_bitmap_type =
    cuco::experimental::roaring_bitmap<
        cuda::std::uint64_t,
        rmm::mr::polymorphic_allocator<char>>;

struct NegateBool {
  __device__ bool operator()(bool b) const {
    return !b;
  }
};

static constexpr uint8_t kDvMagic[] = {0xD1, 0xD3, 0x39, 0x64};

uint32_t readU32BE(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
      (static_cast<uint32_t>(p[1]) << 16) |
      (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

} // namespace

std::unique_ptr<cudf::table> applyDeletionVector(
    cudf::table_view const& table,
    const void* dvPayload,
    std::size_t dvPayloadSize,
    std::size_t startRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numRows = table.num_rows();
  if (numRows == 0 || dvPayloadSize == 0) {
    return std::make_unique<cudf::table>(table, stream, mr);
  }

  // Construct the cuco 64-bit roaring bitmap from the Roaring64 payload.
  auto const* payloadBytes =
      static_cast<cuda::std::byte const*>(dvPayload);
  roaring_bitmap_type bitmap(
      payloadBytes, rmm::mr::polymorphic_allocator<char>{}, stream);

  // Build a row index column: sequential UINT64 values [startRow, startRow + numRows).
  auto rowIndices =
      rmm::device_buffer(numRows * sizeof(std::size_t), stream, mr);
  auto* rowIndicesPtr = static_cast<std::size_t*>(rowIndices.data());
  thrust::sequence(
      rmm::exec_policy_nosync(stream),
      rowIndicesPtr,
      rowIndicesPtr + numRows,
      startRow);

  // Query the bitmap: for each row index, check if it's deleted.
  // The output is negated: true means "keep this row" (not deleted).
  auto rowMask = rmm::device_buffer(numRows * sizeof(bool), stream, mr);
  auto rowMaskIter = thrust::make_transform_output_iterator(
      static_cast<bool*>(rowMask.data()), NegateBool{});

  bitmap.contains(
      rowIndicesPtr, rowIndicesPtr + numRows, rowMaskIter, stream);

  // Build a BOOL8 column_view over the mask buffer.
  auto maskColumn = cudf::column_view(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      rowMask.data(),
      nullptr,
      0);

  return cudf::apply_boolean_mask(table, maskColumn, stream, mr);
}

void parseDvBlobEnvelope(
    const std::string& blob,
    std::size_t& payloadOffset,
    std::size_t& payloadSize) {
  // Check for the deletion-vector-v1 wrapper:
  //   [4B BE combined_length] [4B magic] [vector payload ...] [4B BE CRC]
  // Minimum wrapper size: 4 + 4 + 0 + 4 = 12 bytes.
  if (blob.size() >= 12) {
    const auto* raw = reinterpret_cast<const uint8_t*>(blob.data());
    if (std::memcmp(raw + 4, kDvMagic, 4) == 0) {
      uint32_t combinedLength = readU32BE(raw);
      // combinedLength = len(magic) + len(vector_payload)
      // vector_payload starts at offset 8 (4 for length + 4 for magic)
      // vector_payload length = combinedLength - 4 (magic size)
      if (combinedLength >= 4 &&
          blob.size() >= static_cast<std::size_t>(4 + combinedLength + 4)) {
        payloadOffset = 8;
        payloadSize = combinedLength - 4;
        return;
      }
    }
  }

  // No wrapper detected; treat the entire blob as raw Roaring64 payload.
  payloadOffset = 0;
  payloadSize = blob.size();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
