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

#include <unistd.h>

#include <gtest/gtest.h>

#include "velox/common/memory/MmapAllocator.h"

namespace facebook::velox::memory::test {

/// True if MmapAllocator cannot be constructed on this system, i.e. the
/// system's page size doesn't match AllocationTraits::kPageSize (see
/// MmapAllocator::isPageSizeSupported()). This is the single source of
/// truth other MmapAllocator-unsupported-system handling in tests should
/// query, rather than re-testing MmapAllocator::isPageSizeSupported()
/// directly at each call site.
///
/// NOTE: this is a real, currently-unresolved gap, not a quirk of this
/// particular host. 64KB pages are NVIDIA's documented recommended
/// *default* for Grace / Grace-Hopper systems (better TLB hit rates for
/// memory-intensive workloads), so any Grace deployment following that
/// guidance hits it. It also isn't Velox-specific: jemalloc (which
/// DuckDB and ArangoDB both bundle) has the identical "doesn't support
/// pages > 4KB" limitation -- see https://github.com/arangodb/arangodb/issues/22177,
/// still open. Properly supporting non-4KB pages means generalizing
/// AllocationTraits::kPageSize from a compile-time 4096 constant to a
/// runtime-queried value across ~80 call sites in MmapAllocator and
/// friends, which also changes the allocator's minimum granularity (and
/// therefore its memory-overhead characteristics) on affected hosts --
/// a real design change that deserves its own PR and sign-off from
/// whoever owns the allocator subsystem, not a fix bundled in here.
inline bool mmapAllocatorUnsupported() {
  return !MmapAllocator::isPageSizeSupported();
}

} // namespace facebook::velox::memory::test

/// Skips the current test if mmapAllocatorUnsupported() above is true.
/// Call this directly in SetUp() or directly in a test body -- NOT from a
/// helper function the test body calls, since GTEST_SKIP() only returns
/// from its immediate enclosing function, so a skip inside a helper does
/// not stop the calling test body from running afterward with
/// uninitialized state.
#define SKIP_IF_MMAP_ALLOCATOR_UNSUPPORTED()                               \
  do {                                                                     \
    if (facebook::velox::memory::test::mmapAllocatorUnsupported()) {       \
      GTEST_SKIP() << "MmapAllocator requires a system page size of "      \
                   << facebook::velox::memory::AllocationTraits::kPageSize \
                   << " bytes; this system's page size is "                \
                   << sysconf(_SC_PAGESIZE) << " bytes.";                  \
    }                                                                      \
  } while (0)
