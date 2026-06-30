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

#include <numa.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/experimental/cxl/CxlMemoryAllocator.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cxl {
namespace {
bool numaNodeHasCpus(int32_t numaNode) {
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
}

// Fails when binding to a node that has CPUs attached: CXL expanders surface as
// CPU-less NUMA nodes, so a node with CPUs is regular DRAM where a CXL resource
// would silently land allocations.
void checkNodeIsCxl(int32_t numaNode) {
  bool hasCpus = numaNodeHasCpus(numaNode);
  // Lets tests treat a DRAM node as CXL so the path runs on CXL-less CI hosts.
  TestValue::adjust("facebook::velox::cxl::checkNodeIsCxl", &hasCpus);
  VELOX_USER_CHECK(
      !hasCpus,
      "CXL memory resource requires a CPU-less NUMA node; node has CPUs and is "
      "likely DRAM: {}",
      numaNode);
}
} // namespace

std::shared_ptr<memory::CustomMemoryResource> CxlMemoryResource::create(
    int32_t numaNode,
    int64_t maxCapacity) {
  checkNodeIsCxl(numaNode);

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
      std::string{kTag},
      std::move(allocator),
      std::move(arbitrator),
      std::move(reclaimerFactory),
      maxCapacity);
}

} // namespace facebook::velox::cxl
