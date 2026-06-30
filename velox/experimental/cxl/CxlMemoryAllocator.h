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

#include "velox/common/memory/MmapAllocator.h"

namespace facebook::velox::cxl {

/// An MmapAllocator that binds its mapped pages to a CXL NUMA node, so
/// allocations land in CXL-attached memory.
class CxlMemoryAllocator : public memory::MmapAllocator {
 public:
  /// Binds all mapped memory to 'numaNode', the virtual NUMA node of the CXL
  /// device. Throws if NUMA is unavailable or 'numaNode' is out of range.
  CxlMemoryAllocator(const Options& options, int32_t numaNode);

  /// Virtual NUMA node backing the CXL memory device.
  int32_t numaNode() const {
    return numaNode_;
  }

 private:
  const int32_t numaNode_;
};

} // namespace facebook::velox::cxl
