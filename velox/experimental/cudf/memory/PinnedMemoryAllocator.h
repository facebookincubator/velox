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

#include "velox/common/memory/MemoryAllocator.h"

#include <memory>
#include <optional>

namespace facebook::velox::cudf_velox {

/// Shared host storage for AsyncDataCache and GPU spill buffers. Allocations
/// are CPU-ready on return. Callers must complete ALL GPU accesses before
/// freeing, and retain cache pins until GPU reads finish. Intended for one
/// process-wide instance which outlives the cache and every allocation.
///
/// Capacity includes page rounding and alignment padding. It is also passed to
/// CUDA as max_pool_size, which the driver may round up. retainedBytes()
/// exposes actual backing; unused backing is retained for reuse. Allow
/// approximately 32 MiB of backing-size rounding in process headroom (observed
/// on CUDA 12.9). No 32 MiB minimum or capacity-multiple restriction is
/// imposed. This is not a general-purpose operator allocator: contiguous page
/// allocation and growth are unsupported. Tiny cache entries still bypass this
/// allocator.
class PinnedMemoryAllocator final : public memory::MemoryAllocator {
 public:
  struct Options {
    uint64_t capacity;
    // Placement policy belongs to the launcher. CUDA 12.9 requires a node;
    // CUDA 13+ also supports unspecified host placement.
    std::optional<int> numaNode;
  };

  explicit PinnedMemoryAllocator(const Options& options);
  ~PinnedMemoryAllocator() override;

  Kind kind() const override {
    return Kind::kCustom;
  }
  void registerCache(const std::shared_ptr<memory::Cache>& cache) override;
  size_t capacity() const override;
  size_t totalUsedBytes() const override;
  memory::MachinePageCount numAllocated() const override;
  // No mmap allocations. CUDA backing is reported by retainedBytes().
  memory::MachinePageCount numMapped() const override {
    return 0;
  }
  memory::MachinePageCount numExternalMapped() const override {
    return 0;
  }
  uint64_t retainedBytes() const;
  bool checkConsistency() const override;
  std::string toString() const override;

  void freeBytes(void* ptr, uint64_t bytes) noexcept override;
  int64_t freeNonContiguous(memory::Allocation& allocation) override;
  void freeContiguous(memory::ContiguousAllocation& allocation) override;
  // Deliberately retain free backing, including on cache shrink.
  memory::MachinePageCount unmap(memory::MachinePageCount) override {
    return 0;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  // Register once, before concurrent use. Cache holds a non-owning allocator
  // ptr.
  std::shared_ptr<memory::Cache> cache_;

  memory::Cache* cache() const override {
    return cache_.get();
  }
  void* allocateBytesWithoutRetry(uint64_t bytes, uint16_t alignment) override;
  bool allocateNonContiguousWithoutRetry(
      const SizeMix& mix,
      memory::Allocation& out) override;
  bool allocateContiguousWithoutRetry(
      memory::MachinePageCount,
      memory::Allocation*,
      memory::ContiguousAllocation&,
      memory::MachinePageCount) override;
  bool growContiguousWithoutRetry(
      memory::MachinePageCount,
      memory::ContiguousAllocation&) override;
};

} // namespace facebook::velox::cudf_velox
