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
#include "velox/common/base/Exceptions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::cudf_velox::detail {

CudfMemoryResourceImpl::CudfMemoryResourceImpl(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::shared_ptr<memory::MemoryPool> pool)
    : pool_(std::move(pool)), upstream_(std::move(upstream)) {
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
    std::shared_ptr<memory::MemoryPool> pool)
    : SharedBase(
          cuda::mr::make_shared_resource<detail::CudfMemoryResourceImpl>(
              std::move(upstream),
              std::move(pool))) {}

std::shared_ptr<CudfMemoryResourceRegistry> cudfMemoryResourceRegistry(
    core::QueryCtx& queryCtx) {
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }

  static std::mutex initMutex;
  std::lock_guard<std::mutex> lock(initMutex);
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }
  auto registry = std::make_shared<CudfMemoryResourceRegistry>();
  queryCtx.setRegistry(kCudfMemoryResourceRegistryKey, registry);
  return registry;
}

CudfMemoryResourceRegistry::Resources::Resources(
    cuda::mr::any_resource<cuda::mr::device_accessible> tempUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream,
    std::shared_ptr<memory::MemoryPool> pool)
    : tempUpstream{std::move(tempUpstream)},
      outputUpstream{std::move(outputUpstream)},
      temp{this->tempUpstream, pool} {
  if (this->tempUpstream != this->outputUpstream) {
    output.emplace(this->outputUpstream, std::move(pool));
  }
}

bool CudfMemoryResourceRegistry::Resources::matches(
    const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
    const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream)
    const {
  return this->tempUpstream == tempUpstream &&
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
    std::shared_ptr<memory::MemoryPool> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* poolKey = pool.get();
  auto it = resources_.find(poolKey);
  if (it != resources_.end()) {
    VELOX_CHECK(
        it->second->matches(tempUpstream, outputUpstream),
        "A cuDF pool cannot be reused with different upstream resources");
    return it->second->refs();
  }

  // Construct before inserting so a throwing constructor cannot leave a null
  // entry that poisons the next lookup for this pool.
  auto resources = std::make_unique<Resources>(
      tempUpstream, outputUpstream, std::move(pool));
  auto [insertedIt, inserted] =
      resources_.emplace(poolKey, std::move(resources));
  VELOX_CHECK(inserted, "Duplicate cuDF memory resource pool");
  return insertedIt->second->refs();
}

} // namespace facebook::velox::cudf_velox
