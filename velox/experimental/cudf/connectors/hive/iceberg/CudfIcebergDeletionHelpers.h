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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// GPU-side helper: Transforms the host deletion bitmap to a device surviving
/// row mask, applies it to the input table, returning the new output table.
std::unique_ptr<cudf::table> applyDeleteBitmap(
    cudf::table_view input,
    const uint8_t* hostBitmap,
    cudf::bitmask_type* deviceBitmap,
    bool* deviceRowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr,
    rmm::device_async_resource_ref output_mr);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
