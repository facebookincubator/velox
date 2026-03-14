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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::shared_ptr<rmm::mr::device_memory_resource> mr_;
extern std::shared_ptr<rmm::mr::device_memory_resource> output_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] std::shared_ptr<rmm::mr::device_memory_resource>
createMemoryResource(std::string_view mode, int percent);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
