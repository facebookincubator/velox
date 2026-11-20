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

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"

#include <rmm/cuda_device.hpp>

#include <fmt/format.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

CudfMemoryResourceImpl::CudfMemoryResourceImpl(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<memory::CustomMemoryResource> resourceOwner)
    : resourceOwner_(std::move(resourceOwner)),
      pool_(std::move(pool)),
      upstream_(std::move(upstream)) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK(pool_->isLeaf(), "cuDF memory accounting requires a leaf pool");
  VELOX_CHECK(
      pool_->threadSafe(),
      "cuDF memory accounting requires a thread-safe pool");
}

void* CudfMemoryResourceImpl::allocate_sync(
    std::size_t bytes,
    std::size_t alignment) {
  if (bytes == 0) {
    return upstream_.allocate_sync(bytes, alignment);
  }

  pool_->reportExternalAllocation(bytes);
  try {
    return upstream_.allocate_sync(bytes, alignment);
  } catch (...) {
    pool_->reportExternalFree(bytes);
    throw;
  }
}

void CudfMemoryResourceImpl::deallocate_sync(
    void* pointer,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  upstream_.deallocate_sync(pointer, bytes, alignment);
  if (bytes != 0) {
    pool_->reportExternalFree(bytes);
  }
}

void* CudfMemoryResourceImpl::allocate(
    cuda::stream_ref stream,
    std::size_t bytes,
    std::size_t alignment) {
  if (bytes == 0) {
    return upstream_.allocate(stream, bytes, alignment);
  }

  pool_->reportExternalAllocation(bytes);
  try {
    return upstream_.allocate(stream, bytes, alignment);
  } catch (...) {
    pool_->reportExternalFree(bytes);
    throw;
  }
}

void CudfMemoryResourceImpl::deallocate(
    cuda::stream_ref stream,
    void* pointer,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  upstream_.deallocate(stream, pointer, bytes, alignment);
  if (bytes != 0) {
    // reportExternalFree must not fail for a correctly paired RMM
    // allocation. Any failure is an accounting invariant violation and, as
    // required by the RMM resource contract, terminates this noexcept path.
    pool_->reportExternalFree(bytes);
  }
}

} // namespace facebook::velox::cudf_velox::detail

namespace facebook::velox::cudf_velox {

CudfMemoryResource::CudfMemoryResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<memory::CustomMemoryResource> resourceOwner)
    : SharedBase(
          cuda::mr::make_shared_resource<detail::CudfMemoryResourceImpl>(
              std::move(upstream),
              std::move(pool),
              std::move(resourceOwner))) {}

namespace {

constexpr std::string_view kCudfExchangeMemoryResourcesKey{
    "cudfExchangeMemoryResources"};

// Exchange resources of one query, one per CUDA device, owned by the
// QueryCtx's registries. Packed buffers hold copies of the resources, so a
// resource and its pool outlive the query until the last packed buffer is
// freed.
class CudfExchangeMemoryResources {
 public:
  explicit CudfExchangeMemoryResources(
      const std::shared_ptr<memory::MemoryPool>& queryRootPool)
      : queryRootPool_(queryRootPool) {}

  // Returns true if this was created for 'queryRootPool'. False for a copy
  // that QueryCtx::copyRegistriesFrom() made into a query with its own root.
  bool belongsTo(const memory::MemoryPool& queryRootPool) const {
    return queryRootPool_.lock().get() == &queryRootPool;
  }

  CudfMemoryResource resourceFor(
      const std::shared_ptr<memory::MemoryPool>& queryRootPool,
      int device,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner,
      const cuda::mr::any_resource<cuda::mr::device_accessible>& upstream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resources_.find(device);
    if (it == resources_.end()) {
      auto pool = queryRootPool->addLeafChild(
          fmt::format("ucxExchange.device.{}", device));
      CudfMemoryResource resource{upstream, pool, std::move(resourceOwner)};
      it =
          resources_.emplace(device, Resource{pool, std::move(resource)}).first;
    }
    return it->second.resource;
  }

  std::shared_ptr<memory::MemoryPool> pool(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resources_.find(device);
    return it == resources_.end() ? nullptr : it->second.pool.lock();
  }

 private:
  struct Resource {
    // Not owned: 'resource' owns the pool together with the
    // CustomMemoryResource the pool borrows from, in the order they must be
    // destroyed in.
    std::weak_ptr<memory::MemoryPool> pool;
    CudfMemoryResource resource;
  };

  // The `gpu` custom root of the query this was created for.
  const std::weak_ptr<memory::MemoryPool> queryRootPool_;
  // Guards 'resources_'.
  mutable std::mutex mutex_;
  // Keyed by CUDA device.
  std::unordered_map<int, Resource> resources_;
};

// Returns the exchange resources of the query whose `gpu` custom root is
// 'queryRootPool', creating them on first use.
std::shared_ptr<CudfExchangeMemoryResources> exchangeMemoryResources(
    core::QueryCtx& queryCtx,
    const std::shared_ptr<memory::MemoryPool>& queryRootPool) {
  auto ownResources = [&]() -> std::shared_ptr<CudfExchangeMemoryResources> {
    auto resources = queryCtx.registry<CudfExchangeMemoryResources>(
        kCudfExchangeMemoryResourcesKey);
    return resources && resources->belongsTo(*queryRootPool) ? resources
                                                             : nullptr;
  };
  if (auto resources = ownResources()) {
    return resources;
  }

  // QueryCtx has no atomic get-or-create for registries. Serialize the lazy
  // creation and recheck.
  static std::mutex initMutex;
  std::lock_guard<std::mutex> lock(initMutex);
  if (auto resources = ownResources()) {
    return resources;
  }
  auto resources = std::make_shared<CudfExchangeMemoryResources>(queryRootPool);
  queryCtx.setRegistry(
      kCudfExchangeMemoryResourcesKey, resources, /*overwrite=*/true);
  return resources;
}

} // namespace

std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource(
    const core::QueryCtx& queryCtx) {
  auto perQueryRegistry =
      queryCtx.registry<memory::CustomMemoryResourceRegistry::Registry>(
          memory::kCustomMemoryResourceRegistryKey);
  auto& registry = perQueryRegistry
      ? *perQueryRegistry
      : memory::CustomMemoryResourceRegistry::global();
  auto resourceOwner = registry.find(std::string{kCudfMemoryResourceTag});
  VELOX_CHECK_NOT_NULL(
      resourceOwner,
      "A query with a gpu custom pool must register a CustomMemoryResource "
      "for tag: {}",
      kCudfMemoryResourceTag);
  return resourceOwner;
}

std::optional<CudfMemoryResource> cudfExchangeOutputMemoryResource(
    core::QueryCtx& queryCtx) {
  auto queryRootPool = queryCtx.customPool(std::string{kCudfMemoryResourceTag});
  if (!queryRootPool) {
    return std::nullopt;
  }
  auto resourceOwner = cudfCustomMemoryResource(queryCtx);
  VELOX_CHECK(
      output_mr_.has_value(),
      "cuDF memory resources are not initialized; call registerCudf() first");
  return exchangeMemoryResources(queryCtx, queryRootPool)
      ->resourceFor(
          queryRootPool,
          rmm::get_current_cuda_device().value(),
          std::move(resourceOwner),
          *output_mr_);
}

std::shared_ptr<memory::MemoryPool> testingCudfExchangeMemoryPool(
    const core::QueryCtx& queryCtx) {
  auto queryRootPool = queryCtx.customPool(std::string{kCudfMemoryResourceTag});
  if (!queryRootPool) {
    return nullptr;
  }
  auto resources = queryCtx.registry<CudfExchangeMemoryResources>(
      kCudfExchangeMemoryResourcesKey);
  if (!resources || !resources->belongsTo(*queryRootPool)) {
    return nullptr;
  }
  return resources->pool(rmm::get_current_cuda_device().value());
}

} // namespace facebook::velox::cudf_velox
