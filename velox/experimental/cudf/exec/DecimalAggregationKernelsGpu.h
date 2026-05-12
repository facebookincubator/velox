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

/// Bytes per serialized row for decimal sum aggregation state.
constexpr int32_t kDecimalSumStateSize = 32;

void fillOffsetsForDecimalSumState(
    bool use64BitOffsets,
    void* offsetsMutable,
    int32_t numRows,
    rmm::cuda_stream_view stream);

void packDecimalSumState(
    cudf::type_id sumType,
    bool use64BitOffsets,
    const void* sumPtr,
    const int64_t* countPtr,
    const void* offsetsPtr,
    uint8_t* chars,
    int32_t numRows,
    rmm::cuda_stream_view stream);

void unpackDecimalSumState(
    bool offsets64,
    const void* offsetsPtr,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    int32_t numRows,
    rmm::cuda_stream_view stream);

void averageRoundDecimalSum(
    cudf::type_id sumType,
    const void* sums,
    const int64_t* counts,
    void* out,
    int32_t numRows,
    rmm::cuda_stream_view stream);

std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
