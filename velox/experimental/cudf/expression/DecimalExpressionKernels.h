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

/**
 * @brief Element-wise decimal division of two columns.
 *
 * Runs the GPU divide into outputType. The kernel propagates input nulls and
 * nulls rows where the divisor is zero.
 *
 * @param lhs Left-hand decimal operand column (DECIMAL64 or DECIMAL128).
 * @param rhs Right-hand decimal operand column (same type as lhs).
 * @param outputType Output decimal type including precision and scale.
 * @param aRescale Fixed-point scale adjustment (Velox passes outScale -
 * lhsScale + rhsScale), used inside the kernel as a power-of-ten factor.
 * @param stream CUDA stream for GPU execution.
 * @param mr Memory resource for output allocation.
 * @return Column containing the divided decimal values.
 */
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Element-wise decimal division of a column by a scalar.
 *
 * If the scalar is invalid, returns an all-null column of outputType;
 * otherwise the kernel propagates lhs nulls and nulls rows when dividing by
 * zero.
 *
 * @param lhs Left-hand decimal operand column.
 * @param rhs Right-hand decimal operand scalar.
 * @param outputType Output decimal type including precision and scale.
 * @param aRescale Fixed-point scale adjustment (Velox passes outScale -
 * lhsScale + rhsScale), used inside the kernel as a power-of-ten factor.
 * @param stream CUDA stream for GPU execution.
 * @param mr Memory resource for output allocation.
 * @return Column containing the divided decimal values.
 */
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Element-wise decimal division of a scalar by a column.
 *
 * Invalid lhs yields all-null output; otherwise the kernel propagates rhs
 * nulls and nulls rows where the divisor is zero.
 *
 * @param lhs Left-hand decimal operand scalar.
 * @param rhs Right-hand decimal operand column.
 * @param outputType Output decimal type including precision and scale.
 * @param aRescale Fixed-point scale adjustment (Velox passes outScale -
 * lhsScale + rhsScale), used inside the kernel as a power-of-ten factor.
 * @param stream CUDA stream for GPU execution.
 * @param mr Memory resource for output allocation.
 * @return Column containing the divided decimal values.
 */
std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
