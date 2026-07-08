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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Applies the deletion bitmap to the current deletion row mask
/// @param bitmap Deletion bitmap
/// @param deleteMask Mutable view of deletion mask column
/// @param stream CUDA stream to use
/// @param temp_mr Memory resource for temporary allocations
void applyBitmapToMask(
    cudf::device_span<const cudf::bitmask_type> bitmap,
    cudf::mutable_column_view const& deleteMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

/// Scatters deletes to the deletion mask at positions indicated by `indices`
/// (anti-join semantics).
/// @param deleteMask Mutable view of deletion mask column
/// @param indices Indices to scatter deletes to
/// @param stream CUDA stream to use
/// @param temp_mr Memory resource for temporary allocations
void scatterDeletesToMask(
    cudf::mutable_column_view const& deleteMask,
    cudf::device_span<const cudf::size_type> indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

/// Fills a sequence of row indices into a column.
/// @param rowIndices Mutable view of row indices column
/// @param startRow Starting row index
/// @param numRows Number of rows
/// @param stream CUDA stream to use
/// @param temp_mr Memory resource for temporary allocations
template <typename ValueType>
void fillSequence(
    cudf::mutable_column_view const& rowIndices,
    ValueType startRow,
    int64_t numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
