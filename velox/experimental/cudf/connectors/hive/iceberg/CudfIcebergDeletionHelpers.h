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

/// Applies the deletion bitmap to the current surviving row mask.
/// @param deviceBitmap Deletion bitmap.
/// @param rowMask Mutable view of row mask column.
/// @param stream CUDA stream to use for the operation.
/// @param temp_mr Memory resource for temporary allocations.
void applyDeletionBitmapToRowMask(
    cudf::device_span<const cudf::bitmask_type> deviceBitmap,
    cudf::mutable_column_view const& rowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

/// Scatters deletes to the row mask at positions indicated by `indices`
/// (anti-join semantics).
/// @param rowMask Mutable view of row mask column.
/// @param indices The indices to scatter deletes to.
/// @param stream The stream to use for the operation.
/// @param temp_mr Memory resource for temporary allocations.
void scatterDeletesToRowMask(
    cudf::mutable_column_view const& rowMask,
    cudf::device_span<const cudf::size_type> indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
