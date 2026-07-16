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

#include "velox/experimental/cxl/CxlMemoryResource.h"

#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/experimental/cxl/CxlMemoryAllocator.h"

namespace facebook::velox::cxl {

std::shared_ptr<memory::CustomMemoryResource> CxlMemoryResource::create(
    int32_t numaNode,
    int64_t maxCapacity) {
  // Allocator: the default mmap allocator, with its pages bound to the CXL
  // NUMA node. The constructor rejects a non-CXL (CPU-bearing) node.
  memory::MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = static_cast<size_t>(maxCapacity);
  auto allocator =
      std::make_shared<CxlMemoryAllocator>(allocatorOptions, numaNode);

  // Arbitrator and reclaimer: the defaults from the CPU tier, reused unchanged.
  std::shared_ptr<memory::MemoryArbitrator> arbitrator =
      memory::MemoryArbitrator::create(
          {.kind = "SHARED", .capacity = maxCapacity});
  memory::CustomMemoryResource::ReclaimerFactory reclaimerFactory = [] {
    return memory::MemoryReclaimer::create();
  };

  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kTag},
      std::move(allocator),
      std::move(arbitrator),
      std::move(reclaimerFactory),
      maxCapacity);
}

} // namespace facebook::velox::cxl
