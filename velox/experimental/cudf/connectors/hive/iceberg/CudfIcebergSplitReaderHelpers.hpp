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

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Applies a deletion vector to a cudf table chunk on the GPU.
///
/// The DV payload must be in 64-bit Roaring bitmap "portable" format:
///   [uint64 LE: num_buckets]
///   For each bucket:
///     [uint32 LE: bucket_key (upper 32 bits)]
///     [standard 32-bit Roaring bitmap in portable format]
///
/// This is the payload inside an Iceberg deletion-vector-v1 blob after
/// stripping the 4-byte big-endian combined length, 4-byte magic, and
/// 4-byte CRC wrapper.
///
/// @param table The cudf table chunk to filter.
/// @param dvPayload The Roaring64 DV payload bytes (host memory).
/// @param dvPayloadSize Size of the DV payload in bytes.
/// @param startRow The absolute row index of the first row in this chunk
///   within the data file. Used to build the row index for DV querying.
/// @param stream CUDA stream for kernel launches and data transfers.
/// @param mr Device memory resource for allocating the output table.
/// @return A new cudf table with deleted rows removed.
std::unique_ptr<cudf::table> applyDeletionVector(
    cudf::table_view const& table,
    const void* dvPayload,
    std::size_t dvPayloadSize,
    std::size_t startRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Strips the Iceberg deletion-vector-v1 blob wrapper and returns the offset
/// and length of the Roaring64 payload within the blob.
///
/// The DV-v1 blob format:
///   [4B big-endian: combined length of (magic + vector)]
///   [4B magic: 0xD1 0xD3 0x39 0x64]
///   [vector payload: Roaring64 portable format]
///   [4B big-endian: CRC-32 of (magic + vector)]
///
/// If the blob does not start with the expected wrapper (magic not found at
/// offset 4), the entire blob is assumed to be a raw Roaring64 payload.
///
/// @param blob The raw DV blob bytes.
/// @param payloadOffset [out] Byte offset of the Roaring64 payload within blob.
/// @param payloadSize [out] Size of the Roaring64 payload in bytes.
void parseDvBlobEnvelope(
    const std::string& blob,
    std::size_t& payloadOffset,
    std::size_t& payloadSize);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
