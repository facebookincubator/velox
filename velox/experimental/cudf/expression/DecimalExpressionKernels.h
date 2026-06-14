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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

// Element-wise decimal division of two columns (same DECIMAL64 or DECIMAL128
// input type). Builds the output null mask as the bitwise AND of lhs and rhs
// validity, runs the GPU divide into outputType, and applies
// scatterNullsAtZeroDivisor so rows with a zero divisor are null. aRescale is
// the fixed-point scale adjustment (Velox passes outScale - lhsScale +
// rhsScale) used inside the kernel as a power-of-ten factor.
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Like column/column decimalDivide, but rhs is a single decimal scalar. If the
// scalar is invalid, returns an all-null column of outputType; otherwise copies
// lhs nulls and divides without zero-divisor scattering (rhs is not per-row).
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Like column/column decimalDivide, but lhs is a scalar and rhs is a column.
// Invalid lhs yields all-null output; otherwise rhs nulls are propagated, then
// divide and scatterNullsAtZeroDivisor on rhs so division-by-zero rows are
// null.
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// After a decimal divide, forces output rows to null where the divisor column
// compares equal to zero (DECIMAL64 or DECIMAL128), using copy_if_else. Kept in
// the .cpp translation unit so it can use Velox checks alongside cuDF APIs
// without pulling those into CUDA compilation units.
std::unique_ptr<cudf::column> scatterNullsAtZeroDivisor(
    std::unique_ptr<cudf::column> result,
    const cudf::column_view& divisor,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
