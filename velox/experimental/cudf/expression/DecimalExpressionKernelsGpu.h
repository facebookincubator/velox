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

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <utility>

namespace facebook::velox::cudf_velox {

// Outcome of a checked decimal binary op. Division-by-zero (divide or modulo
// with a zero divisor) is kept distinct from arithmetic overflow so the host
// can raise the matching Velox/Presto CPU error (e.g. "Division by zero" /
// "Modulus by zero" vs "Decimal overflow in ...").
enum class DecimalBinaryOpStatus : int32_t {
  kOk = 0,
  kOverflow = 1,
  kDivisionByZero = 2,
};

// CUDA implementations that return {result, status}. The status is tracked with
// a single device-side flag (OR'd via a per-thread register, then one
// atomicOr per thread), matching the fail-fast semantics of Presto / Velox
// CPU decimal arithmetic; no per-row (O(n)) status column is allocated.
std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

namespace detail {

/**
 * @brief Decodes a cuDF decimal scalar into a raw __int128_t payload.
 *
 * Shared by the divide (host) and binary-op (device-launch) paths so the
 * DECIMAL64/DECIMAL128 scalar unpacking lives in exactly one place.
 *
 * @param s DECIMAL64 or DECIMAL128 fixed-point scalar.
 * @param stream CUDA stream used to read the device-resident scalar value.
 * @return The scalar's underlying integer representation as __int128_t.
 */
__int128_t getDecimalScalarValue(
    const cudf::scalar& s,
    cuda::stream_ref stream);

} // namespace detail
} // namespace facebook::velox::cudf_velox
