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
constexpr int32_t kDecimalSumStateSize = 32;

// Writes strings-style prefix offsets into offsetsMutable for numRows + 1
// entries: offset[i] == i * kDecimalSumStateSize (INT32 or INT64 elements).
void fillOffsetsForDecimalSumState(
    bool use64BitOffsets,
    void* offsetsMutable,
    int32_t numRows,
    rmm::cuda_stream_view stream);

// For each row, writes kDecimalSumStateSize bytes into chars at the byte offset
// given by offsetsPtr, encoding the partial sum (DECIMAL64 or DECIMAL128) and
// int64 count into the device struct layout used for VARBINARY interchange.
void packDecimalSumState(
    cudf::type_id sumType,
    bool use64BitOffsets,
    const void* sumPtr,
    const int64_t* countPtr,
    const void* offsetsPtr,
    uint8_t* chars,
    int32_t numRows,
    rmm::cuda_stream_view stream);

// Inverse of packDecimalSumState: reads fixed-width payloads via offsetsPtr
// (INT32 or INT64 string offsets) and fills per-row DECIMAL128 sums and counts.
void unpackDecimalSumState(
    bool offsets64,
    const void* offsetsPtr,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    int32_t numRows,
    rmm::cuda_stream_view stream);

// Per-row average from intermediate sum/count: integer divide of abs(sum) by
// count with half-up bias (add count/2 before dividing), then restore sign;
// count == 0 writes a numeric zero (validity is applied separately). Output
// element type matches sumType (DECIMAL64 or DECIMAL128).
void averageRoundDecimalSum(
    cudf::type_id sumType,
    const void* sums,
    const int64_t* counts,
    void* out,
    int32_t numRows,
    rmm::cuda_stream_view stream);

// Builds a bitmask for rows where both sum and count are valid and count is
// non-zero (via cudf::detail::valid_if), for use when serializing state or
// finalizing averages.
std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
