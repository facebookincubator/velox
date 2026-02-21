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

#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <common/base/Exceptions.h>

#include <cstdlib>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

namespace {
/// \brief Makes a cuda resource
[[nodiscard]] auto makeCudaMr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

/// \brief Makes a pool<cuda> resource
[[nodiscard]] auto makePoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes an async resource
[[nodiscard]] auto makeAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

/// \brief Makes a managed resource
[[nodiscard]] auto makeManagedMr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

/// \brief Makes a prefetch<managed> resource
[[nodiscard]] auto makePrefetchManagedMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedMr());
}

/// \brief Makes an arena<cuda> resource
[[nodiscard]] auto makeArenaMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a pool<managed> resource
[[nodiscard]] auto makeManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a prefetch<pool<managed>> resource
[[nodiscard]] auto makePrefetchManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedPoolMr(percent));
}

/// \brief Makes a managed_async resource (experimental, requires CUDA 13+)
[[nodiscard]] auto makeManagedAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_managed_memory_resource>();
}

/// \brief Makes a prefetch<managed_async> resource (experimental, requires CUDA
/// 13+)
[[nodiscard]] auto makePrefetchManagedAsyncMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedAsyncMr());
}

void enablePrefetching() {
  cudf::prefetch::enable();
}

} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda")
    return makeCudaMr();
  if (mode == "pool")
    return makePoolMr(percent);
  if (mode == "async")
    return makeAsyncMr();
  if (mode == "arena")
    return makeArenaMr(percent);
  if (mode == "managed")
    return makeManagedMr();
  if (mode == "managed_pool")
    return makeManagedPoolMr(percent);
  if (mode == "managed_async")
    return makeManagedAsyncMr();
  if (mode == "prefetch_managed") {
    enablePrefetching();
    return makePrefetchManagedMr();
  }
  if (mode == "prefetch_managed_pool") {
    enablePrefetching();
    return makePrefetchManagedPoolMr(percent);
  }
  if (mode == "prefetch_managed_async") {
    enablePrefetching();
    return makePrefetchManagedAsyncMr();
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::shared_ptr<rmm::mr::device_memory_resource> mr_;
std::shared_ptr<rmm::mr::device_memory_resource> output_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.get();
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

struct allow_default_stream_t {};
constexpr allow_default_stream_t allow_default_stream{};

rmm::cuda_stream_view get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}
} // namespace cudf
