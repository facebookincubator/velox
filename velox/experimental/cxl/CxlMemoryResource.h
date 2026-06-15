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

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

#include "velox/common/memory/CustomMemoryResource.h"

namespace facebook::velox::cxl {

/// Tag identifying the CXL memory tier. Used when registering the resource
/// with memory::CustomMemoryResourceRegistry and when referencing it by tag
/// while building per-query pools via MemoryManager::addCustomRootPool.
inline constexpr std::string_view kCxlResourceTag{"cxl"};

/// Builds a memory::CustomMemoryResource backed by CXL-attached memory.
///
/// The resource bundles a CxlMemoryAllocator bound to 'numaNode' with the
/// default SHARED arbitrator and reclaimer, reused unchanged from the CPU
/// tier. Register the returned resource with
/// memory::CustomMemoryResourceRegistry and reference it by 'kCxlResourceTag'
/// when building per-query pools via MemoryManager::addCustomRootPool.
///
/// 'numaNode' is the virtual NUMA node id exposed by the CXL device.
/// 'maxCapacity' bounds, in bytes, both the allocator and the per-query root
/// pool created from this resource; it defaults to unbounded.
std::shared_ptr<memory::CustomMemoryResource> makeCxlMemoryResource(
    int32_t numaNode,
    int64_t maxCapacity = std::numeric_limits<int64_t>::max());

/// Returns true if 'numaNode' has CPUs attached. CXL memory expanders surface
/// to Linux as CPU-less (memory-only) NUMA nodes, so a node with CPUs is most
/// likely regular DRAM rather than a CXL device. makeCxlMemoryResource uses
/// this to warn on a likely-misconfigured node. Best-effort, not proof:
/// userspace cannot reliably confirm a node is CXL-backed. Always returns false
/// off Linux or when NUMA is unavailable.
bool numaNodeHasCpus(int32_t numaNode);

} // namespace facebook::velox::cxl
