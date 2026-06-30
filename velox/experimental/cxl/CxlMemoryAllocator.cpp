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

#include "velox/experimental/cxl/CxlMemoryAllocator.h"

#include <numa.h>

#include "velox/common/testutil/TestValue.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cxl {
namespace {

// Returns whether 'numaNode' has any CPUs attached.
bool numaNodeHasCpus(int32_t numaNode) {
  auto* cpus = numa_allocate_cpumask();
  if (cpus == nullptr) {
    return false;
  }
  const bool hasCpus =
      numa_node_to_cpus(numaNode, cpus) == 0 && numa_bitmask_weight(cpus) > 0;
  numa_free_cpumask(cpus);
  return hasCpus;
}

// Verifies 'numaNode' is a CXL device. CXL expanders surface as CPU-less NUMA
// nodes, so a node with CPUs is regular DRAM where allocations would silently
// land off the CXL device.
void checkNodeIsCxl(int32_t numaNode) {
  bool hasCpus = numaNodeHasCpus(numaNode);
  // Lets tests treat a DRAM node as CXL so the path runs on CXL-less CI hosts.
  TestValue::adjust("facebook::velox::cxl::checkNodeIsCxl", &hasCpus);
  VELOX_USER_CHECK(
      !hasCpus,
      "CXL memory allocator requires a CPU-less NUMA node; node has CPUs and "
      "is likely DRAM: {}",
      numaNode);
}

// Validates 'numaNode', returning it unchanged.
int32_t validateNumaNode(int32_t numaNode) {
  VELOX_CHECK_GE(numa_available(), 0, "NUMA is not available on this host");
  VELOX_CHECK_GE(numaNode, 0, "Invalid CXL NUMA node: {}", numaNode);
  VELOX_CHECK_LE(
      numaNode, numa_max_node(), "CXL NUMA node is out of range: {}", numaNode);
  checkNodeIsCxl(numaNode);
  return numaNode;
}

// Binds [address, address + bytes) to 'numaNode'.
void bindToNumaNode(void* address, size_t bytes, int32_t numaNode) {
  numa_tonode_memory(address, bytes, numaNode);
}

// Returns 'options' with an onMap hook binding to 'numaNode'.
memory::MemoryAllocator::Options withNumaBinding(
    memory::MemoryAllocator::Options options,
    int32_t numaNode) {
  options.onMap = [numaNode](void* address, size_t bytes) {
    bindToNumaNode(address, bytes, numaNode);
  };
  return options;
}

} // namespace

CxlMemoryAllocator::CxlMemoryAllocator(const Options& options, int32_t numaNode)
    : MmapAllocator(withNumaBinding(options, validateNumaNode(numaNode))),
      numaNode_(numaNode) {}

} // namespace facebook::velox::cxl
