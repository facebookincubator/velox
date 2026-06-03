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

// Dispatches a per-row device loop: fixed-point divide (lhs * 10^aRescale) /
// rhs with half-away-from-zero rounding on the remainder, writing into out.
// Zero divisors produce a numeric zero in out (callers patch nulls). inType /
// outType select DECIMAL64 vs DECIMAL128 storage widths for inputs and result.
void decimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream);

// Same kernel math as decimalDivideColumnColumn, but rhs is a single
// __int128_t decimal payload (already decoded from a cuDF scalar).
void decimalDivideColumnScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream);

// Same kernel math as decimalDivideColumnColumn, but lhs is a single
// __int128_t decimal payload and rhs is per-row.
void decimalDivideScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox::detail
