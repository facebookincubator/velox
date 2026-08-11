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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <common/base/Exceptions.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

thread_local std::optional<rmm::device_async_resource_ref> scopedTempMr;
thread_local std::optional<rmm::device_async_resource_ref> scopedOutputMr;

class ThreadLocalTemporaryMemoryResourceImpl {
 public:
  explicit ThreadLocalTemporaryMemoryResourceImpl(
      cuda::mr::any_resource<cuda::mr::device_accessible> fallback)
      : fallback_(std::move(fallback)) {}

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    auto resource = currentResource();
    auto* pointer = resource.allocate_sync(bytes, alignment);
    try {
      rememberAllocation(pointer, bytes, resource);
    } catch (...) {
      resource.deallocate_sync(pointer, bytes, alignment);
      throw;
    }
    return pointer;
  }

  void deallocate_sync(
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    resourceForDeallocation(pointer, bytes)
        .deallocate_sync(pointer, bytes, alignment);
  }

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    auto resource = currentResource();
    auto* pointer = resource.allocate(stream, bytes, alignment);
    try {
      rememberAllocation(pointer, bytes, resource);
    } catch (...) {
      resource.deallocate(stream, pointer, bytes, alignment);
      throw;
    }
    return pointer;
  }

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    resourceForDeallocation(pointer, bytes)
        .deallocate(stream, pointer, bytes, alignment);
  }

  bool operator==(
      const ThreadLocalTemporaryMemoryResourceImpl& other) const noexcept {
    return this == std::addressof(other);
  }

  bool operator!=(
      const ThreadLocalTemporaryMemoryResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const ThreadLocalTemporaryMemoryResourceImpl&,
      cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref currentResource() {
    return scopedTempMr.has_value() ? *scopedTempMr
                                    : rmm::device_async_resource_ref{fallback_};
  }

  void rememberAllocation(
      void* pointer,
      std::size_t bytes,
      rmm::device_async_resource_ref resource) {
    if (bytes == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto inserted = allocations_.emplace(pointer, resource).second;
    VELOX_CHECK(inserted, "Duplicate outstanding cuDF temporary allocation");
  }

  rmm::device_async_resource_ref resourceForDeallocation(
      void* pointer,
      std::size_t bytes) noexcept {
    if (bytes == 0) {
      return currentResource();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(pointer);
    VELOX_CHECK(
        it != allocations_.end(),
        "Unknown cuDF temporary allocation during deallocation");
    auto resource = it->second;
    allocations_.erase(it);
    return resource;
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> fallback_;
  std::mutex mutex_;
  std::unordered_map<void*, rmm::device_async_resource_ref> allocations_;
};

class ThreadLocalTemporaryMemoryResource
    : public cuda::mr::shared_resource<ThreadLocalTemporaryMemoryResourceImpl> {
  using SharedBase =
      cuda::mr::shared_resource<ThreadLocalTemporaryMemoryResourceImpl>;

 public:
  explicit ThreadLocalTemporaryMemoryResource(
      cuda::mr::any_resource<cuda::mr::device_accessible> fallback)
      : SharedBase(
            cuda::mr::make_shared_resource<
                ThreadLocalTemporaryMemoryResourceImpl>(std::move(fallback))) {}

  friend void get_property(
      const ThreadLocalTemporaryMemoryResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(cuda::mr::resource_with<
              ThreadLocalTemporaryMemoryResource,
              cuda::mr::device_accessible>);

} // namespace

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;

rmm::device_async_resource_ref get_temp_mr() {
  return scopedTempMr.has_value() ? *scopedTempMr
                                  : rmm::mr::get_current_device_resource_ref();
}

rmm::device_async_resource_ref get_output_mr() {
  if (scopedOutputMr.has_value()) {
    return *scopedOutputMr;
  }
  return rmm::device_async_resource_ref{output_mr_.value()};
}

cuda::mr::any_resource<cuda::mr::device_accessible>
createThreadLocalTemporaryMemoryResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> fallback) {
  return ThreadLocalTemporaryMemoryResource{std::move(fallback)};
}

ScopedCudfMemoryResources::ScopedCudfMemoryResources(
    rmm::device_async_resource_ref tempMr,
    rmm::device_async_resource_ref outputMr)
    : previousTempMr_(scopedTempMr), previousOutputMr_(scopedOutputMr) {
  scopedTempMr = tempMr;
  scopedOutputMr = outputMr;
}

ScopedCudfMemoryResources::~ScopedCudfMemoryResources() {
  scopedTempMr = previousTempMr_;
  scopedOutputMr = previousOutputMr_;
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
