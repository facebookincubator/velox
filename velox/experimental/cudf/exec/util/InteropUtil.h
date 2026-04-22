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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

/// Convert Timestamp array to the configured cuDF timestamp representation on
/// GPU.
void convertTimestamps(
    const void* dTimestamps,
    const cudf::bitmask_type* dMask,
    int64_t* dOutput,
    cudf::size_type numRows,
    cudf::type_id timestampUnit,
    std::optional<std::string_view> timestampTimeZone,
    rmm::cuda_stream_view stream);

std::pair<std::unique_ptr<cudf::column>, cudf::size_type>
makeOffsetsColumnFromSizes(
    const int32_t* rawSizes,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox

// Made with Bob
