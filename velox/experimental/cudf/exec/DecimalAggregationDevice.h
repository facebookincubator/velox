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

#include <cstddef>
#include <cstdint>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

// Size in bytes of each row's packed decimal SUM intermediate state in the
// strings payload (count, overflow placeholder, and 128-bit sum split into
// words).
constexpr size_t kDecimalSumStateSize = 32;

/**
 * Writes strings-style prefix offsets: offset[i] == i * kDecimalSumStateSize.
 *
 * @param offsetType INT32 or INT64; selects offset storage width via
 *        cudf::type_dispatcher.
 * @param offsetsView output offsets column of numRows + 1 elements.
 * @param numRows number of payload rows.
 * @param stream CUDA stream for the launch.
 */
void fillOffsetsForDecimalSumState(
    cudf::type_id offsetType,
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

/**
 * Encodes each row's partial sum and count into the fixed-width device layout
 * used for VARBINARY interchange.
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width.
 * @param offsetType INT32 or INT64; selects offset storage width. sumType and
 *        offsetType are dispatched via cudf::double_type_dispatcher.
 * @param sumCol per-row sums.
 * @param counts per-row int64 counts.
 * @param offsetsView per-row byte offsets into chars.
 * @param chars output payload buffer.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

/**
 * Inverse of packDecimalSumState.
 *
 * @param offsetType INT32 or INT64; selects offset storage width via
 *        cudf::type_dispatcher.
 * @param offsetsView per-row byte offsets into chars.
 * @param chars packed payload buffer.
 * @param sumView output per-row DECIMAL128 sums.
 * @param countView output per-row counts.
 * @param numRows number of rows.
 * @param nullMask device null-mask bitmap; null rows are skipped to avoid
 *        out-of-bounds reads when Arrow compacts null payloads.  Pass nullptr
 *        when no mask is present.
 * @param stream CUDA stream for the launch.
 */
void unpackDecimalSumState(
    cudf::type_id offsetType,
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::size_type numRows,
    cudf::bitmask_type const* nullMask,
    rmm::cuda_stream_view stream);

/**
 * Per-row half-up integer divide of sum by count; count == 0 writes zero
 * (validity is applied separately).
 *
 * @param sumType DECIMAL64 or DECIMAL128; selects sum storage width via
 *        cudf::type_dispatcher<cudf::dispatch_storage_type>.
 * @param sumCol per-row sums.
 * @param counts per-row counts.
 * @param outView output per-row averages.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream);

/**
 * Builds a null mask for rows where sum and count are both valid and count is
 * non-zero, for serializing state or finalizing averages.
 *
 * @param sumCol decoded sum column.
 * @param countCol decoded count column.
 * @param stream CUDA stream for the launch.
 * @param mr memory resource for the returned mask.
 * @return {null mask buffer, null count}.
 */
std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
