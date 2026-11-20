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

// Device primitives for laying out Presto SerializedPage bytes on the GPU. They
// return no device memory and take their scratch from get_temp_mr(). A host
// span is read before the function returns, unless it is device accessible
// (pinned memory, as cudf::detail::make_pinned_vector_async gives): that one
// is read in stream order and must stay valid until then.

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace facebook::velox::cudf_velox::detail {

/// The offsets of one strings column, addressed the way cuDF stores them: an
/// int32 or int64 offsets child, never decreasing, that the column's slice
/// offset indexes into.
struct StringOffsets {
  const void* offsets;
  /// INT32 or INT64.
  cudf::data_type offsetsType;
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
  /// Receives ceil(numRows / 8) bytes.
  uint8_t* dst;
};

/// Writes the Presto VARIABLE_WIDTH offsets of a row range: the end offset of
/// each row relative to the start of the range, as little-endian int32.
struct StringOffsetsTask {
  StringOffsets offsets;
  /// First row of the range, counted from the column's first row.
  int64_t firstRow;
  int32_t numRows;
  /// Receives numRows int32 offsets.
  uint8_t* dst;
};

/// One device-to-device byte copy.
struct CopyTask {
  const uint8_t* src;
  uint8_t* dst;
  size_t size;
};

/// What prestoPageStats() and hasNonEmptyNullStrings() read of one column.
struct PageStatsColumn {
  /// cuDF validity mask (1 = valid), or null for a column without nulls.
  const cudf::bitmask_type* mask;
  /// Mask bit of the column's first row.
  int64_t firstBit;
  /// The offsets of a strings column; 'offsets.offsets' is null for a column
  /// of another type.
  StringOffsets offsets;
};

/// What prestoPageStats() tells of one column at one page boundary.
struct PageStats {
  /// Offset at which the strings of the boundary row start, the end offset of
  /// the column at the last boundary; 0 for a column without offsets.
  int64_t charsOffset;
  /// Null rows of the page that starts at the boundary; 0 for a column
  /// without a mask and at the last boundary, where no page starts.
  int32_t numNulls;
};

/// Splits 'numRows' rows into pages and returns the row boundaries, numPages +
/// 1 values from 0 to numRows, or {0} without rows. A row costs
/// 'fixedBitsPerRow' bits, which is not negative, plus the bytes of its
/// strings. A page holds at least one row and at most 'maxRows' rows, and
/// exceeds 'pageBytes' by less than one row. 'pageBytes', at most INT64_MAX /
/// 8, and 'maxRows' are positive.
std::vector<int64_t> prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    cudf::host_span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream);

/// Runs every NullBytesTask.
void fillPrestoNullBytes(
    cudf::host_span<const NullBytesTask> tasks,
    cuda::stream_ref stream);

/// Runs every StringOffsetsTask. Returns false when the strings of a range
/// take more bytes than an int32 offset holds, so that no Presto page can
/// carry the range; 'dst' is unspecified in that case.
bool fillPrestoStringOffsets(
    cudf::host_span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream);

/// Runs all copies with one batched device memcpy.
void batchedDeviceCopy(
    cudf::host_span<const CopyTask> tasks,
    cuda::stream_ref stream);

/// Returns the PageStats of every column at every page boundary:
/// columns.size() * bounds.size() values, those of a column next to each
/// other. 'bounds' are the row boundaries of the pages as
/// prestoPageBoundaries() returns them: from 0 to the row count, never
/// decreasing. Waits for the device once, however many columns there are.
std::vector<PageStats> prestoPageStats(
    cudf::host_span<const PageStatsColumn> columns,
    cudf::host_span<const int64_t> bounds,
    cuda::stream_ref stream);

/// Returns for each column whether one of its 'numRows' rows is null and has
/// string bytes, which cuDF allows and a Presto page cannot hold; false for a
/// column without a mask or without offsets. Waits for the device once,
/// however many columns there are.
std::vector<bool> hasNonEmptyNullStrings(
    cudf::host_span<const PageStatsColumn> columns,
    int32_t numRows,
    cuda::stream_ref stream);

/// Copies the values of the non-null rows of a row range to 'dst', next to
/// each other and in row order, as a Presto block holds them. A value has
/// 'width' bytes: 1, 2, 4, 8 or 16. 'src' holds the value of the first row of
/// the range, and bit 'firstBit' of the cuDF validity 'mask' (1 = valid) is
/// that row's. 'dst' has room for the non-null rows, whose number the caller
/// knows from the null count. 'src' and 'dst' are aligned to 'width' and do
/// not overlap. The values of the null rows do not matter.
void compactValidValues(
    const void* src,
    int32_t width,
    const cudf::bitmask_type* mask,
    int64_t firstBit,
    int32_t numRows,
    void* dst,
    cuda::stream_ref stream);

/// Converts timestamps to the milliseconds Presto carries: dst[i] is src[i],
/// a tick count in the unit of 'timestampType', scaled to milliseconds and
/// rounded towards negative infinity, as Timestamp::toMillis() rounds.
/// 'timestampType' is TIMESTAMP_SECONDS, TIMESTAMP_MILLISECONDS,
/// TIMESTAMP_MICROSECONDS or TIMESTAMP_NANOSECONDS; 'src' and 'dst' are 8-byte
/// aligned and may be the same. Returns false when a value in seconds does
/// not fit in int64 milliseconds, as Timestamp::toMillis() fails then; 'dst'
/// is unspecified in that case.
bool timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    cudf::type_id timestampType,
    int64_t* dst,
    cuda::stream_ref stream);

/// Rewrites two's complement decimals as the sign-magnitude values Presto Java
/// reads from an INT128_ARRAY block: a negative value becomes its magnitude
/// with the top bit set. 'src' and 'dst' are 16-byte aligned and may be the
/// same.
void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::detail
