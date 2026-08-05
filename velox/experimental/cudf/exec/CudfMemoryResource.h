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

#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"

#include "velox/common/memory/MemoryPool.h"

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace facebook::velox::core {
class QueryCtx;
}

namespace facebook::velox::cudf_velox {

inline constexpr std::string_view kCudfMemoryResourceRegistryKey{
    "cudfMemoryResource"};

namespace detail {

/// An RMM-compatible resource that delegates device allocation to 'upstream'
/// and reports logical live bytes to a Velox leaf MemoryPool.
///
/// The implementation is held by cuda::mr::shared_resource. As a result, RMM
/// containers that copy this resource through cuda::mr::any_resource retain
/// the pool, custom resource, and upstream resource until their final
/// deallocation.
class CudfMemoryResourceImpl {
 public:
  CudfMemoryResourceImpl(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::shared_ptr<memory::MemoryPool> pool,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner);

  CudfMemoryResourceImpl(const CudfMemoryResourceImpl&) = delete;
  CudfMemoryResourceImpl& operator=(const CudfMemoryResourceImpl&) = delete;

  void* allocate_sync(std::size_t bytes, std::size_t alignment);

  void deallocate_sync(
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept;

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept;

  bool operator==(const CudfMemoryResourceImpl& other) const noexcept {
    return this == std::addressof(other);
  }

  bool operator!=(const CudfMemoryResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const CudfMemoryResourceImpl&,
      cuda::mr::device_accessible) noexcept {}

 private:
  // Declared first so it is destroyed last. MemoryPool borrows its allocator
  // and arbitrator from this object.
  std::shared_ptr<memory::CustomMemoryResource> resourceOwner_;
  std::shared_ptr<memory::MemoryPool> pool_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

} // namespace detail

/// Copyable, owning RMM resource that accounts allocations in a Velox pool.
class CudfMemoryResource
    : public cuda::mr::shared_resource<detail::CudfMemoryResourceImpl> {
  using SharedBase = cuda::mr::shared_resource<detail::CudfMemoryResourceImpl>;

 public:
  CudfMemoryResource(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::shared_ptr<memory::MemoryPool> pool,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner = nullptr);

  friend void get_property(
      const CudfMemoryResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<CudfMemoryResource, cuda::mr::device_accessible>);

/// Query-scoped owner for reporting resources. cuDF APIs accept non-owning
/// resource refs, so the resource objects must outlive output columns that can
/// survive their producing operator.
class CudfMemoryResourceRegistry {
 public:
  struct ResourceRefs {
    rmm::device_async_resource_ref temp;
    rmm::device_async_resource_ref output;
  };

  ResourceRefs resourcesFor(
      const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
      const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream,
      std::shared_ptr<memory::MemoryPool> pool,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner);

 private:
  struct Resources {
    Resources(
        cuda::mr::any_resource<cuda::mr::device_accessible> tempUpstream,
        cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream,
        std::shared_ptr<memory::MemoryPool> pool,
        std::shared_ptr<memory::CustomMemoryResource> resourceOwner);

    bool matches(
        const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
        const cuda::mr::any_resource<cuda::mr::device_accessible>&
            outputUpstream,
        const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner)
        const;

    ResourceRefs refs();

    // Declared before reporting resources so the borrowed allocator and
    // arbitrator remain alive through pool and resource teardown.
    std::shared_ptr<memory::CustomMemoryResource> resourceOwner;
    cuda::mr::any_resource<cuda::mr::device_accessible> tempUpstream;
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream;
    CudfMemoryResource temp;
    std::optional<CudfMemoryResource> output;
  };

  std::mutex mutex_;
  std::unordered_map<memory::MemoryPool*, std::unique_ptr<Resources>>
      resources_;
};

std::optional<rmm::device_async_resource_ref> cudfExchangeOutputMemoryResource(
    core::QueryCtx& queryCtx);

std::shared_ptr<memory::MemoryPool> cudfExchangeMemoryPool(
    const core::QueryCtx& queryCtx);

/// Releases retired exchange resources that have no live packed buffers.
/// Returns true when no active or in-use exchange resources remain.
bool tryResetCudfExchangeMemoryResource();

/// Strict test/shutdown helper. Fails if any query or packed buffer is active.
void resetCudfExchangeMemoryResource();

} // namespace facebook::velox::cudf_velox
