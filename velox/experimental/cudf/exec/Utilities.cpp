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

#include <cstdlib>
#include <memory>
#include <string_view>

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/utilities/error.hpp>

namespace facebook::velox::cudf_velox {

namespace {
auto make_cuda_mr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

auto make_pool_mr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_cuda_mr(), rmm::percent_of_free_device_memory(50));
}

auto make_async_mr() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

auto make_managed_mr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

auto make_arena_mr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      make_cuda_mr());
}

auto make_managed_pool_mr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_managed_mr(), rmm::percent_of_free_device_memory(50));
}
} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
    std::string_view mode) {
  if (mode == "cuda")
    return make_cuda_mr();
  if (mode == "pool")
    return make_pool_mr();
  if (mode == "async")
    return make_async_mr();
  if (mode == "arena")
    return make_arena_mr();
  if (mode == "managed")
    return make_managed_mr();
  if (mode == "managed_pool")
    return make_managed_pool_mr();
  throw cudf::logic_error(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, or managed_pool");
}

bool cudfDebugEnabled() {
  const char* env_cudf_debug = std::getenv("VELOX_CUDF_DEBUG");
  return env_cudf_debug != nullptr && std::stoi(env_cudf_debug);
}

} // namespace facebook::velox::cudf_velox
