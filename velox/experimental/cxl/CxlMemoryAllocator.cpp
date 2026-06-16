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

#ifdef __linux__
#include <numa.h>
#endif

namespace facebook::velox::cxl {
namespace {

// Validates 'numaNode', returning it unchanged.
int32_t validateNumaNode(int32_t numaNode) {
#ifdef __linux__
  VELOX_CHECK_GE(numa_available(), 0, "NUMA is not available on this host");
  VELOX_CHECK_GE(numaNode, 0, "Invalid CXL NUMA node: {}", numaNode);
  VELOX_CHECK_LE(
      numaNode, numa_max_node(), "CXL NUMA node is out of range: {}", numaNode);
#endif
  return numaNode;
}

// Binds [address, address + bytes) to 'numaNode'.
void bindToNumaNode(void* address, size_t bytes, int32_t numaNode) {
#ifdef __linux__
  numa_tonode_memory(address, bytes, numaNode);
#else
  (void)address;
  (void)bytes;
  (void)numaNode;
#endif
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
