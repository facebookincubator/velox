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

#ifdef __linux__
#include <numa.h>
#endif

#include <glog/logging.h>

#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/experimental/cxl/CxlMemoryAllocator.h"

namespace facebook::velox::cxl {

bool numaNodeHasCpus(int32_t numaNode) {
#ifdef __linux__
  if (numa_available() < 0) {
    return false;
  }
  auto* cpus = numa_allocate_cpumask();
  if (cpus == nullptr) {
    return false;
  }
  const bool hasCpus =
      numa_node_to_cpus(numaNode, cpus) == 0 && numa_bitmask_weight(cpus) > 0;
  numa_free_cpumask(cpus);
  return hasCpus;
#else
  (void)numaNode;
  return false;
#endif
}

namespace {
// Warns when binding to a node that has CPUs attached: CXL "allocations" there
// would silently land on regular DRAM. See numaNodeHasCpus for why this is a
// heuristic rather than a hard check.
void warnIfNodeUnlikelyCxl(int32_t numaNode) {
  if (numaNodeHasCpus(numaNode)) {
    LOG(WARNING) << "NUMA node " << numaNode
                 << " has CPUs; it may be DRAM, not a CXL device.";
  }
}
} // namespace

std::shared_ptr<memory::CustomMemoryResource> makeCxlMemoryResource(
    int32_t numaNode,
    int64_t maxCapacity) {
  warnIfNodeUnlikelyCxl(numaNode);

  // Allocator: the default mmap allocator, with its pages bound to the CXL
  // NUMA node.
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
      std::string{kCxlResourceTag},
      std::move(allocator),
      std::move(arbitrator),
      std::move(reclaimerFactory),
      maxCapacity);
}

} // namespace facebook::velox::cxl
