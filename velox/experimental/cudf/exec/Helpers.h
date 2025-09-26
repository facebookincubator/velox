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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <vector>

namespace facebook::velox::cudf_velox {

void sort_join_indices_inplace(
    cudf::mutable_column_view leftJoinIndices,
    cudf::mutable_column_view rightJoinIndices,
    rmm::cuda_stream_view stream);

[[nodiscard]]
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
filtered_indices_again(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightIndices,
    cudf::mutable_column_view& filterColumn,
    rmm::cuda_stream_view stream);
} // namespace facebook::velox::cudf_velox
