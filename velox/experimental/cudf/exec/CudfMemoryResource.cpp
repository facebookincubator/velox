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
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/MemoryReclaimer.h"

#include <rmm/cuda_device.hpp>

#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
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

std::mutex registeredResourceMutex;
std::shared_ptr<memory::CustomMemoryResource> registeredResource;
bool ownsRegisteredResource{false};

} // namespace

std::shared_ptr<memory::CustomMemoryResource> createCudfCustomMemoryResource(
    int64_t capacity) {
  VELOX_USER_CHECK_GT(capacity, 0, "GPU memory capacity must be positive");

  // GPU bytes are allocated by RMM and charged through
  // reportExternalAllocation(). CustomMemoryResource still requires an
  // allocator, but this one never backs the charged GPU allocations.
  memory::MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = capacity;
  auto allocator = std::make_shared<memory::MallocAllocator>(allocatorOptions);

  memory::MemoryArbitrator::Config arbitratorConfig{
      .kind = "SHARED",
      .capacity = capacity,
      .arbitrationStateCheckCb = exec::memoryArbitrationStateCheck,
      .extraConfigs = {
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolInitialCapacity),
           "0B"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kGlobalArbitrationEnabled),
           "true"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinReclaimBytes),
           "1MB"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinReclaimPct),
           "0"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinFreeCapacity),
           "0B"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinFreeCapacityPct),
           "0"}}};
  auto arbitrator =
      std::make_shared<memory::SharedArbitrator>(arbitratorConfig);

  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::move(allocator),
      std::move(arbitrator),
      []() { return std::unique_ptr<memory::MemoryReclaimer>{}; },
      capacity);
}

std::shared_ptr<memory::CustomMemoryResource> registerCudfMemoryResource(
    int64_t capacity) {
  std::lock_guard<std::mutex> lock(registeredResourceMutex);
  if (registeredResource != nullptr) {
    return registeredResource;
  }

  auto& registry = memory::CustomMemoryResourceRegistry::global();
  registeredResource = registry.find(std::string{kCudfMemoryResourceTag});
  if (registeredResource != nullptr) {
    ownsRegisteredResource = false;
    return registeredResource;
  }

  registeredResource = createCudfCustomMemoryResource(capacity);
  registry.insert(std::string{kCudfMemoryResourceTag}, registeredResource);
  ownsRegisteredResource = true;
  return registeredResource;
}

void unregisterCudfMemoryResource() {
  std::lock_guard<std::mutex> lock(registeredResourceMutex);
  if (registeredResource == nullptr) {
    return;
  }
  auto& registry = memory::CustomMemoryResourceRegistry::global();
  if (ownsRegisteredResource &&
      registry.find(std::string{kCudfMemoryResourceTag}) ==
          registeredResource) {
    registry.erase(std::string{kCudfMemoryResourceTag});
  }
  registeredResource.reset();
  ownsRegisteredResource = false;
}

std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource() {
  std::lock_guard<std::mutex> lock(registeredResourceMutex);
  return registeredResource;
}

namespace {

class CudfExchangeMemoryResource {
 public:
  CudfExchangeMemoryResource(
      std::shared_ptr<memory::MemoryPool> queryRootPool,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner,
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      int device)
      : resourceOwner_(std::move(resourceOwner)),
        queryRootPool_(std::move(queryRootPool)),
        pool_(queryRootPool_->addLeafChild(
            "ucxExchange.device." + std::to_string(device))),
        upstream_(std::move(upstream)),
        resource_(upstream_, pool_, resourceOwner_),
        device_(device) {}

  bool matches(
      const std::shared_ptr<memory::MemoryPool>& queryRootPool,
      const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner,
      const cuda::mr::any_resource<cuda::mr::device_accessible>& upstream,
      int device) const {
    return queryRootPool_ == queryRootPool && resourceOwner_ == resourceOwner &&
        upstream_ == upstream && device_ == device;
  }

  rmm::device_async_resource_ref resource() {
    return rmm::device_async_resource_ref{resource_};
  }

  const std::shared_ptr<memory::MemoryPool>& pool() const {
    return pool_;
  }

 private:
  // The entry retains the query's custom root and backing resource after the
  // QueryCtx is destroyed so packed buffers can cross task boundaries safely.
  // Declaration order keeps borrowed allocator/arbitrator state alive last.
  std::shared_ptr<memory::CustomMemoryResource> resourceOwner_;
  std::shared_ptr<memory::MemoryPool> queryRootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  CudfMemoryResource resource_;
  const int device_;
};

struct ExchangeResourceKey {
  memory::MemoryPool* queryRootPool;
  memory::CustomMemoryResource* resourceOwner;
  int device;

  bool operator==(const ExchangeResourceKey& other) const {
    return queryRootPool == other.queryRootPool &&
        resourceOwner == other.resourceOwner && device == other.device;
  }
};

struct ExchangeResourceKeyHash {
  size_t operator()(const ExchangeResourceKey& key) const {
    const auto poolHash = std::hash<memory::MemoryPool*>{}(key.queryRootPool);
    const auto ownerHash =
        std::hash<memory::CustomMemoryResource*>{}(key.resourceOwner);
    const auto deviceHash = std::hash<int>{}(key.device);
    const auto poolAndOwnerHash =
        poolHash ^ (ownerHash + 0x9e3779b9 + (poolHash << 6) + (poolHash >> 2));
    return poolAndOwnerHash ^
        (deviceHash + 0x9e3779b9 + (poolAndOwnerHash << 6) +
         (poolAndOwnerHash >> 2));
  }
};

struct ExchangeResourceEntry {
  explicit ExchangeResourceEntry(
      std::unique_ptr<CudfExchangeMemoryResource> resource)
      : resource(std::move(resource)) {}

  std::unique_ptr<CudfExchangeMemoryResource> resource;
  std::unordered_set<const core::QueryCtx*> activeQueries;
};

std::mutex exchangeResourceMutex;
std::unordered_map<
    ExchangeResourceKey,
    std::unique_ptr<ExchangeResourceEntry>,
    ExchangeResourceKeyHash>
    exchangeResources;

void pruneInactiveExchangeResourcesLocked() {
  for (auto it = exchangeResources.begin(); it != exchangeResources.end();) {
    const auto& entry = *it->second;
    if (entry.activeQueries.empty() &&
        entry.resource->pool()->usedBytes() == 0) {
      it = exchangeResources.erase(it);
    } else {
      ++it;
    }
  }
}

void retireExchangeResource(
    ExchangeResourceKey key,
    const core::QueryCtx* queryCtx) {
  std::lock_guard<std::mutex> lock(exchangeResourceMutex);
  auto it = exchangeResources.find(key);
  if (it == exchangeResources.end()) {
    return;
  }
  it->second->activeQueries.erase(queryCtx);
  if (it->second->activeQueries.empty() &&
      it->second->resource->pool()->usedBytes() == 0) {
    exchangeResources.erase(it);
  }
}

} // namespace

CudfMemoryResourceRegistry::Resources::Resources(
    cuda::mr::any_resource<cuda::mr::device_accessible> tempUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream,
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<memory::CustomMemoryResource> resourceOwner)
    : resourceOwner{std::move(resourceOwner)},
      tempUpstream{std::move(tempUpstream)},
      outputUpstream{std::move(outputUpstream)},
      temp{this->tempUpstream, pool, this->resourceOwner} {
  if (this->tempUpstream != this->outputUpstream) {
    output.emplace(this->outputUpstream, std::move(pool), this->resourceOwner);
  }
}

bool CudfMemoryResourceRegistry::Resources::matches(
    const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
    const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream,
    const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner) const {
  return this->resourceOwner == resourceOwner &&
      this->tempUpstream == tempUpstream &&
      this->outputUpstream == outputUpstream;
}

CudfMemoryResourceRegistry::ResourceRefs
CudfMemoryResourceRegistry::Resources::refs() {
  auto tempRef = rmm::device_async_resource_ref{temp};
  return {
      tempRef,
      output.has_value() ? rmm::device_async_resource_ref{*output} : tempRef};
}

CudfMemoryResourceRegistry::ResourceRefs
CudfMemoryResourceRegistry::resourcesFor(
    const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
    const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream,
    std::shared_ptr<memory::MemoryPool> pool,
    std::shared_ptr<memory::CustomMemoryResource> resourceOwner) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* poolKey = pool.get();
  auto it = resources_.find(poolKey);
  if (it != resources_.end()) {
    VELOX_CHECK(
        it->second->matches(tempUpstream, outputUpstream, resourceOwner),
        "A cuDF pool cannot be reused with different upstream resources or "
        "a different CustomMemoryResource owner");
    return it->second->refs();
  }

  // Construct before inserting so a throwing constructor cannot leave a null
  // entry that poisons the next lookup for this pool.
  auto resources = std::make_unique<Resources>(
      tempUpstream, outputUpstream, std::move(pool), std::move(resourceOwner));
  auto [insertedIt, inserted] =
      resources_.emplace(poolKey, std::move(resources));
  VELOX_CHECK(inserted, "Duplicate cuDF memory resource pool");
  return insertedIt->second->refs();
}

std::optional<rmm::device_async_resource_ref> cudfExchangeOutputMemoryResource(
    core::QueryCtx& queryCtx) {
  auto queryRootPool = queryCtx.customPool(std::string{kCudfMemoryResourceTag});
  if (!queryRootPool) {
    return std::nullopt;
  }

  auto perQueryRegistry =
      queryCtx.registry<memory::CustomMemoryResourceRegistry::Registry>(
          memory::kCustomMemoryResourceRegistryKey);
  auto& registry = perQueryRegistry
      ? *perQueryRegistry
      : memory::CustomMemoryResourceRegistry::global();
  auto resourceOwner = registry.find(std::string{kCudfMemoryResourceTag});
  VELOX_CHECK_NOT_NULL(
      resourceOwner,
      "Query has a gpu custom pool but no registered gpu "
      "CustomMemoryResource");
  VELOX_CHECK(
      output_mr_.has_value(), "cuDF output memory resource is not initialized");

  const auto device = rmm::get_current_cuda_device().value();
  const ExchangeResourceKey key{
      queryRootPool.get(), resourceOwner.get(), device};
  std::lock_guard<std::mutex> lock(exchangeResourceMutex);
  pruneInactiveExchangeResourcesLocked();

  auto it = exchangeResources.find(key);
  if (it == exchangeResources.end()) {
    auto resource = std::make_unique<CudfExchangeMemoryResource>(
        queryRootPool, resourceOwner, output_mr_.value(), device);
    auto entry = std::make_unique<ExchangeResourceEntry>(std::move(resource));
    auto [insertedIt, inserted] =
        exchangeResources.emplace(key, std::move(entry));
    VELOX_CHECK(inserted, "Duplicate cuDF exchange memory resource");
    it = insertedIt;
  } else {
    VELOX_CHECK(
        it->second->resource->matches(
            queryRootPool, resourceOwner, output_mr_.value(), device),
        "A query GPU root and CUDA device cannot be reused with a different "
        "CustomMemoryResource owner or cuDF output upstream resource");
  }

  auto* queryCtxPtr = std::addressof(queryCtx);
  if (it->second->activeQueries.find(queryCtxPtr) ==
      it->second->activeQueries.end()) {
    queryCtx.addReleaseCallback(
        [key, queryCtxPtr]() { retireExchangeResource(key, queryCtxPtr); });
    it->second->activeQueries.insert(queryCtxPtr);
  }
  return it->second->resource->resource();
}

std::shared_ptr<memory::MemoryPool> cudfExchangeMemoryPool(
    const core::QueryCtx& queryCtx) {
  auto queryRootPool = queryCtx.customPool(std::string{kCudfMemoryResourceTag});
  if (!queryRootPool) {
    return nullptr;
  }

  auto perQueryRegistry =
      queryCtx.registry<memory::CustomMemoryResourceRegistry::Registry>(
          memory::kCustomMemoryResourceRegistryKey);
  auto& registry = perQueryRegistry
      ? *perQueryRegistry
      : memory::CustomMemoryResourceRegistry::global();
  auto resourceOwner = registry.find(std::string{kCudfMemoryResourceTag});
  VELOX_CHECK_NOT_NULL(
      resourceOwner,
      "Query has a gpu custom pool but no registered gpu "
      "CustomMemoryResource");

  const ExchangeResourceKey key{
      queryRootPool.get(),
      resourceOwner.get(),
      rmm::get_current_cuda_device().value()};
  std::lock_guard<std::mutex> lock(exchangeResourceMutex);
  auto it = exchangeResources.find(key);
  return it == exchangeResources.end() ? nullptr : it->second->resource->pool();
}

bool tryResetCudfExchangeMemoryResource() {
  std::lock_guard<std::mutex> lock(exchangeResourceMutex);
  pruneInactiveExchangeResourcesLocked();
  return exchangeResources.empty();
}

void resetCudfExchangeMemoryResource() {
  std::lock_guard<std::mutex> lock(exchangeResourceMutex);
  for (const auto& [_, entry] : exchangeResources) {
    VELOX_CHECK(
        entry->activeQueries.empty(),
        "Cannot tear down the cuDF UCX exchange resource while query contexts "
        "are still active");
    VELOX_CHECK_EQ(
        entry->resource->pool()->usedBytes(),
        0,
        "Cannot tear down the cuDF UCX exchange resource while packed buffers "
        "are still alive");
  }
  exchangeResources.clear();
}

} // namespace facebook::velox::cudf_velox
