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

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>

namespace facebook::velox::cudf_velox {

template <typename Lhs, typename Rhs>
std::unique_ptr<cudf::column> decimalDivide(
    const Lhs& lhs,
    const Rhs& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

template <typename Lhs, typename Rhs>
std::unique_ptr<cudf::column> decimalBinaryOperation(
    const Lhs& lhs,
    const Rhs& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Helper function to scatter nulls at zero-divisor positions.
// Moved to .cpp file to allow use of VELOX_FAIL (incompatible with nvcc).
std::unique_ptr<cudf::column> scatterNullsAtZeroDivisor(
    std::unique_ptr<cudf::column> result,
    const cudf::column_view& divisor,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// CUDA implementations that return {result, didOverflow}. Overflow is tracked
// with a single device-side flag (set via atomicOr by any overflowing row),
// matching the fail-fast semantics of Presto / Velox CPU decimal arithmetic;
// no per-row (O(n)) overflow column is allocated.
std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
