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

// Device primitives with which CudfPrestoOutput lays out Presto SerializedPage
// bytes on the GPU. Needs only cuDF, RMM and CUDA headers.

#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::detail {

/// The offsets of one strings column, addressed the way cuDF stores them: an
/// int32 or int64 offsets child that the column's slice offset indexes into.
struct StringOffsets {
  const void* offsets;
  bool offsetsInt64;
  /// Offsets index of the column's first row.
  int64_t firstIndex;
};

/// Writes the Presto null bitmap of a row range: bit (7 - k) of byte j is set
/// when row 8 * j + k of the range is null. The pad bits of the last byte are
/// zero.
struct NullBytesTask {
  /// cuDF validity mask (1 = valid).
  const cudf::bitmask_type* mask;
  /// Mask bit of the first row of the range.
  int64_t firstBit;
  int32_t numRows;
  uint8_t* dst;
};

/// Writes the Presto VARIABLE_WIDTH offsets of a row range: the end offset of
/// each row relative to the start of the range, as little-endian int32.
struct StringOffsetsTask {
  StringOffsets offsets;
  /// First row of the range, counted from the column's first row.
  int64_t firstRow;
  int32_t numRows;
  uint8_t* dst;
};

/// One device-to-device byte copy.
struct CopyTask {
  const uint8_t* src;
  uint8_t* dst;
  size_t size;
};

/// Splits 'numRows' rows into pages and returns the row boundaries, numPages +
/// 1 values from 0 to numRows, or {0} without rows. A row costs
/// 'fixedBitsPerRow' bits plus the bytes of its strings. A page holds at least
/// one row and at most 'maxRows' rows, and exceeds 'pageBytes' by less than
/// one row.
std::vector<int64_t> prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    std::span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Runs every NullBytesTask.
void fillPrestoNullBytes(
    std::span<const NullBytesTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Runs every StringOffsetsTask.
void fillPrestoStringOffsets(
    std::span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Runs all copies with one batched device memcpy.
void batchedDeviceCopy(
    std::span<const CopyTask> tasks,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Returns the offset at which each of 'rows' starts, rows counted from the
/// column's first row. A row equal to the row count yields the end offset.
std::vector<int64_t> gatherStringOffsets(
    const StringOffsets& offsets,
    std::span<const int64_t> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// dst[i] = floor(src[i] * mul / div). Exactly one of mul, div is 1, and
/// 'dst' may be 'src'. Returns false when a product does not fit in int64, as
/// Timestamp::toMillis() fails then; 'dst' is unspecified in that case.
bool timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    int64_t mul,
    int64_t div,
    int64_t* dst,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Rewrites two's complement decimals as the sign-magnitude values Presto Java
/// reads from an INT128_ARRAY block: a negative value becomes its magnitude
/// with the top bit set. 'dst' may be 'src'.
void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream);

/// Copies the values of the valid rows into 'dst' back to back. 'width' is
/// the element size in bytes, one of 1, 2, 4, 8, 16. Returns the number of
/// values copied.
int32_t compactValid(
    const uint8_t* src,
    int32_t width,
    const cudf::bitmask_type* mask,
    int64_t firstBit,
    int32_t numRows,
    uint8_t* dst,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::detail
