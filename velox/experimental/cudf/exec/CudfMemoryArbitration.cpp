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

#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"

#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Task.h"

#include <folly/Indestructible.h>

#include <mutex>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {
// One backend per process. Owns allocator/arbitrator infrastructure, never
// query pools or query buffers.
struct CudfBackend {
  std::mutex mutex;
  std::shared_ptr<memory::CustomMemoryResource> resource;
  std::optional<int64_t> requestedCapacity;
};

CudfBackend& cudfBackend() {
  static folly::Indestructible<CudfBackend> backend;
  return *backend;
}
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
               memory::SharedArbitrator::ExtraConfig::kMemoryPoolMinReclaimPct),
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
      [](const memory::ReclaimerContext& context) {
        if (auto* query =
                std::get_if<memory::QueryReclaimerContext>(&context)) {
          return core::QueryCtx::MemoryReclaimer::create(
              query->queryCtx, query->pool);
        }
        if (auto* task = std::get_if<memory::TaskReclaimerContext>(&context)) {
          return exec::Task::MemoryReclaimer::create(
              task->task, task->priority, std::string{task->resourceTag});
        }
        return exec::MemoryReclaimer::create();
      },
      capacity);
}

std::shared_ptr<memory::CustomMemoryResource> registerCudfMemoryResource(
    int64_t capacity) {
  auto& backend = cudfBackend();
  std::lock_guard lock(backend.mutex);
  auto& registry = memory::CustomMemoryResourceRegistry::global();
  auto registered = registry.find(std::string{kCudfMemoryResourceTag});
  if (backend.resource) {
    VELOX_USER_CHECK(
        capacity == backend.resource->maxCapacity() ||
            (backend.requestedCapacity &&
             capacity == *backend.requestedCapacity),
        "cuDF backend capacity is process-lived and cannot be changed");
    VELOX_USER_CHECK(
        !registered || registered == backend.resource,
        "Cannot replace the process-lived cuDF backend in the registry");
    if (!registered) {
      registry.insert(std::string{kCudfMemoryResourceTag}, backend.resource);
    }
    return backend.resource;
  }
  // An application-provided resource is retained just as permanently. Its
  // explicit capacity takes precedence over the requested capacity.
  auto resource =
      registered ? registered : createCudfCustomMemoryResource(capacity);
  if (!registered) {
    registry.insert(std::string{kCudfMemoryResourceTag}, resource);
  }
  backend.requestedCapacity = capacity;
  backend.resource = std::move(resource);
  return backend.resource;
}

std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource() {
  auto& backend = cudfBackend();
  std::lock_guard lock(backend.mutex);
  return backend.resource;
}

void attachCudfMemoryPool(core::QueryCtx& queryCtx) {
  if (auto resource = cudfCustomMemoryResource()) {
    const auto name = fmt::format(
        "{}.{}",
        core::QueryCtx::generatePoolName(queryCtx.queryId()),
        kCudfMemoryResourceTag);
    auto* manager = memory::memoryManager();
    auto root = manager->addCustomRootPool(
        name,
        resource->allocator(),
        resource->arbitrator(),
        resource->maxCapacity());
    auto reclaimer = resource->newReclaimer(
        memory::QueryReclaimerContext{&queryCtx, root.get()});
    if (reclaimer != nullptr) {
      root->setReclaimer(std::move(reclaimer));
    }
    queryCtx.addCustomPool(resource->tag(), std::move(root));
  }
}

} // namespace facebook::velox::cudf_velox
