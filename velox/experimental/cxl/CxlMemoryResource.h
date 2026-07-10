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

/// Factory and identity for the CXL memory tier built on the core
/// CustomMemoryResource framework.
class CxlMemoryResource {
 public:
  /// Tag identifying the CXL memory resource: used to register it with
  /// memory::CustomMemoryResourceRegistry and to build a per-query pool via
  /// MemoryManager::addCustomRootPool. Set this as the value of the
  /// 'relocation_resource_tag' query config to have an operator relocate its
  /// payload here; core resolves the tag generically and never references this
  /// symbol.
  static constexpr std::string_view kTag{"cxl"};

  /// Builds a memory::CustomMemoryResource backed by CXL-attached memory.
  ///
  /// The resource bundles a CxlMemoryAllocator bound to 'numaNode' with the
  /// default SHARED arbitrator and reclaimer, reused unchanged from the CPU
  /// tier. Register the returned resource with
  /// memory::CustomMemoryResourceRegistry and reference it by 'kTag' when
  /// building per-query pools via MemoryManager::addCustomRootPool.
  ///
  /// 'numaNode' is the virtual NUMA node id exposed by the CXL device.
  /// 'maxCapacity' bounds, in bytes, both the allocator and the per-query root
  /// pool created from this resource; it defaults to unbounded.
  static std::shared_ptr<memory::CustomMemoryResource> create(
      int32_t numaNode,
      int64_t maxCapacity = std::numeric_limits<int64_t>::max());
};

} // namespace facebook::velox::cxl
