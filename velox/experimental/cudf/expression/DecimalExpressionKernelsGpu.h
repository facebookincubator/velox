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

#include <cstdint>

namespace facebook::velox::cudf_velox::detail {

/**
 * @brief Dispatches a per-row device loop for fixed-point decimal division.
 *
 * Computes (lhs * rescaleFactor) / rhs with half-away-from-zero rounding on the
 * remainder, writing into out. Input nulls and zero divisors are written as
 * null in the output null mask.
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhs Left-hand decimal operand column.
 * @param rhs Right-hand decimal operand column.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return True on success; false if any row overflowed (caller should
 * VELOX_USER_FAIL).
 */
bool decimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream);

/**
 * @brief Fixed-point decimal division with a column lhs and scalar rhs.
 *
 * Same kernel math as decimalDivideColumnColumn, but rhs is a single
 * __int128_t decimal payload (already decoded from a cuDF scalar).
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhs Left-hand decimal operand column.
 * @param rhsValue Right-hand decimal operand as a decoded __int128_t payload.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return True on success; false if any row overflowed (caller should
 * VELOX_USER_FAIL).
 */
bool decimalDivideColumnScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream);

/**
 * @brief Fixed-point decimal division with a scalar lhs and column rhs.
 *
 * Same kernel math as decimalDivideColumnColumn, but lhs is a single
 * __int128_t decimal payload and rhs is per-row.
 *
 * @param inType DECIMAL64 or DECIMAL128 input storage width selector.
 * @param outType DECIMAL64 or DECIMAL128 output storage width selector.
 * @param lhsValue Left-hand decimal operand as a decoded __int128_t payload.
 * @param rhs Right-hand decimal operand column.
 * @param out Mutable output column to write divided values into.
 * @param rescaleFactor Fixed-point scale factor, typically
 * DecimalUtil::kPowersOfTen[aRescale] from the caller.
 * @param stream CUDA stream for kernel execution.
 * @return True on success; false if any row overflowed (caller should
 * VELOX_USER_FAIL).
 */
bool decimalDivideScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox::detail
