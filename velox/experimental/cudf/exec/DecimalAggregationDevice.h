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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

// Size in bytes of each row's packed decimal SUM intermediate state in the
// strings payload (count, overflow placeholder, and 128-bit sum split into
// words).
constexpr size_t kDecimalSumStateSize = 32;

// Writes strings-style prefix offsets: offset[i] == i * kDecimalSumStateSize.
// @param use64BitOffsets whether offsets are INT64 (else INT32).
// @param offsetsMutable output buffer of numRows + 1 offset elements.
// @param numRows number of payload rows.
// @param stream CUDA stream for the launch.
void fillOffsetsForDecimalSumState(
    bool use64BitOffsets,
    void* offsetsMutable,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

// Encodes each row's partial sum and count into the fixed-width device layout
// used for VARBINARY interchange.
// @param sumType element type of sumPtr (DECIMAL64 or DECIMAL128).
// @param use64BitOffsets whether offsetsPtr is INT64 (else INT32).
// @param sumPtr per-row sums.
// @param countPtr per-row int64 counts.
// @param offsetsPtr per-row byte offsets into chars.
// @param chars output payload buffer.
// @param numRows number of rows.
// @param stream CUDA stream for the launch.
void packDecimalSumState(
    cudf::type_id sumType,
    bool use64BitOffsets,
    const void* sumPtr,
    const int64_t* countPtr,
    const void* offsetsPtr,
    uint8_t* chars,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

// Inverse of packDecimalSumState.
// @param offsets64 whether offsetsPtr is INT64 (else INT32).
// @param offsetsPtr per-row byte offsets into chars.
// @param chars packed payload buffer.
// @param sums output per-row DECIMAL128 sums.
// @param counts output per-row counts.
// @param numRows number of rows.
// @param stream CUDA stream for the launch.
void unpackDecimalSumState(
    bool offsets64,
    const void* offsetsPtr,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

// Per-row half-up integer divide of sum by count; count == 0 writes zero
// (validity is applied separately).
// @param sumType element type of sums/out (DECIMAL64 or DECIMAL128).
// @param sums per-row sums.
// @param counts per-row counts.
// @param out output per-row averages.
// @param numRows number of rows.
// @param stream CUDA stream for the launch.
void averageRoundDecimalSum(
    cudf::type_id sumType,
    const void* sums,
    const int64_t* counts,
    void* out,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

// Builds a null mask for rows where sum and count are both valid and count is
// non-zero, for serializing state or finalizing averages.
// @param sumCol decoded sum column.
// @param countCol decoded count column.
// @param stream CUDA stream for the launch.
// @param mr memory resource for the returned mask.
// @return {null mask buffer, null count}.
std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
