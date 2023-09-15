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

#include <atomic>
#include <chrono>
#include <limits>
#include <list>
#include <memory>
#include <string>

#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <velox/common/base/Exceptions.h>
#include "folly/CPortability.h"
#include "folly/GLog.h"
#include "folly/Likely.h"
#include "folly/Random.h"
#include "folly/SharedMutex.h"
#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryPool.h"

DECLARE_bool(velox_memory_leak_check_enabled);
DECLARE_bool(velox_memory_pool_debug_enabled);
DECLARE_bool(velox_enable_memory_usage_track_in_default_memory_pool);

namespace facebook::velox::memory {
#define VELOX_MEM_LOG_PREFIX "[MEM] "
#define VELOX_MEM_LOG(severity) LOG(severity) << VELOX_MEM_LOG_PREFIX
#define VELOX_MEM_LOG_EVERY_MS(severity, ms) \
  FB_LOG_EVERY_MS(severity, ms) << VELOX_MEM_LOG_PREFIX

#define VELOX_MEM_ALLOC_ERROR(errorMessage)                         \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemAllocError.c_str(),        \
      /* isRetriable */ true,                                       \
      "{}",                                                         \
      errorMessage);

struct MemoryManagerOptions {
  /// Specifies the default memory allocation alignment.
  uint16_t alignment{MemoryAllocator::kMaxAlignment};

  /// Specifies the max memory capacity in bytes. MemoryManager will not
  /// enforce capacity. This will be used by MemoryArbitrator
  int64_t capacity{MemoryAllocator::kDefaultCapacityBytes};

  /// Memory capacity for query/task memory pools. This capacity setting should
  /// be equal or smaller than 'capacity'. The difference between 'capacity' and
  /// 'queryMemoryCapacity' is reserved for system usage such as cache and
  /// spilling.
  ///
  /// NOTE:
  /// - if 'queryMemoryCapacity' is greater than 'capacity', the behavior
  /// will be equivalent to as if they are equal, meaning no reservation
  /// capacity for system usage.
  int64_t queryMemoryCapacity{kMaxMemory};

  /// If true, enable memory usage tracking in the default memory pool.
  bool trackDefaultUsage{
      FLAGS_velox_enable_memory_usage_track_in_default_memory_pool};

  /// If true, check the memory pool and usage leaks on destruction.
  ///
  /// TODO: deprecate this flag after all the existing memory leak use cases
  /// have been fixed.
  bool checkUsageLeak{FLAGS_velox_memory_leak_check_enabled};

  /// If true, the memory pool will be running in debug mode to track the
  /// allocation and free call stacks to detect the source of memory leak for
  /// testing purpose.
  bool debugEnabled{FLAGS_velox_memory_pool_debug_enabled};

  /// Specifies the backing memory allocator.
  MemoryAllocator* allocator{MemoryAllocator::getInstance()};

  /// ================== 'MemoryArbitrator' settings ==================

  /// The string kind of memory arbitrator used in the memory manager.
  ///
  /// NOTE: the arbitrator will only be created if its kind is set explicitly.
  /// Otherwise MemoryArbitrator::create returns a nullptr.
  std::string arbitratorKind{};

  /// The initial memory capacity to reserve for a newly created memory pool.
  uint64_t memoryPoolInitCapacity{256 << 20};

  /// The minimal memory capacity to transfer out of or into a memory pool
  /// during the memory arbitration.
  uint64_t memoryPoolTransferCapacity{32 << 20};
};

/// 'MemoryManager' is responsible for managing the memory pools. For now, users
/// wanting multiple different allocators would need to instantiate different
/// MemoryManager classes and manage them across static boundaries.
class MemoryManager {
 public:
  explicit MemoryManager(
      const MemoryManagerOptions& options = MemoryManagerOptions{});

  ~MemoryManager();

  /// Tries to get the singleton memory manager. If not previously initialized,
  /// the process singleton manager will be initialized.
  FOLLY_EXPORT static MemoryManager& getInstance(
      const MemoryManagerOptions& options = MemoryManagerOptions{});

  /// Returns the memory capacity of this memory manager which puts a hard cap
  /// on memory usage, and any allocation that exceeds this capacity throws.
  int64_t capacity() const;

  /// Returns the memory allocation alignment of this memory manager.
  uint16_t alignment() const;

  /// Creates a root memory pool with specified 'name' and 'capacity'. If 'name'
  /// is missing, the memory manager generates a default name internally to
  /// ensure uniqueness.
  std::shared_ptr<MemoryPool> addRootPool(
      const std::string& name = "",
      int64_t capacity = kMaxMemory,
      std::unique_ptr<MemoryReclaimer> reclaimer = nullptr);

  /// Creates a leaf memory pool for direct memory allocation use with specified
  /// 'name'. If 'name' is missing, the memory manager generates a default name
  /// internally to ensure uniqueness. The leaf memory pool is created as the
  /// child of the memory manager's default root memory pool. If 'threadSafe' is
  /// true, then we track its memory usage in a non-thread-safe mode to reduce
  /// its cpu cost.
  std::shared_ptr<MemoryPool> addLeafPool(
      const std::string& name = "",
      bool threadSafe = true);

  /// Invoked to grows a memory pool's free capacity with at least
  /// 'incrementBytes'. The function returns true on success, otherwise false.
  bool growPool(MemoryPool* pool, uint64_t incrementBytes);

  /// Invoked to shrink alive pools to free 'targetBytes' capacity. The function
  /// returns the actual freed memory capacity in bytes.
  uint64_t shrinkPools(uint64_t targetBytes);

  /// Default unmanaged leaf pool with no threadsafe stats support. Libraries
  /// using this method can get a pool that is shared with other threads. The
  /// goal is to minimize lock contention while supporting such use cases.
  ///
  /// TODO: deprecate this API after all the use cases are able to manage the
  /// lifecycle of the allocated memory pools properly.
  MemoryPool& deprecatedSharedLeafPool();

  /// Returns the current total memory usage under this memory manager.
  int64_t getTotalBytes() const;

  /// Returns the number of alive memory pools allocated from addRootPool() and
  /// addLeafPool().
  ///
  /// NOTE: this doesn't count the memory manager's internal default root and
  /// leaf memory pools.
  size_t numPools() const;

  MemoryAllocator& allocator();

  MemoryArbitrator* arbitrator();

  /// Returns debug string of this memory manager.
  std::string toString() const;

  /// Returns the memory manger's internal default root memory pool for testing
  /// purpose.
  MemoryPool& testingDefaultRoot() const {
    return *defaultRoot_;
  }

  const std::vector<std::shared_ptr<MemoryPool>>& testingSharedLeafPools() {
    return sharedLeafPools_;
  }

 private:
  void dropPool(MemoryPool* pool);

  //  Returns the shared references to all the alive memory pools in 'pools_'.
  std::vector<std::shared_ptr<MemoryPool>> getAlivePools() const;

  // Specifies the total memory capacity. Memory manager itself doesn't enforce
  // the capacity but relies on memory allocator and memory arbitrator to do the
  // enforcement. Memory allocator ensures physical memory allocations are
  // within capacity limit. Memory arbitrator ensures that total allocated
  // memory pool capacity is within the limit.
  const int64_t capacity_;
  const std::shared_ptr<MemoryAllocator> allocator_;
  // If not null, used to arbitrate the memory capacity among 'pools_'.
  const std::unique_ptr<MemoryArbitrator> arbitrator_;
  const uint16_t alignment_;
  const bool checkUsageLeak_;
  const bool debugEnabled_;
  // The destruction callback set for the allocated  root memory pools which are
  // tracked by 'pools_'. It is invoked on the root pool destruction and removes
  // the pool from 'pools_'.
  const MemoryPoolImpl::DestructionCallback poolDestructionCb_;

  const std::shared_ptr<MemoryPool> defaultRoot_;
  std::vector<std::shared_ptr<MemoryPool>> sharedLeafPools_;

  mutable folly::SharedMutex mutex_;
  std::unordered_map<std::string, std::weak_ptr<MemoryPool>> pools_;
};

MemoryManager& defaultMemoryManager();

/// Creates a leaf memory pool from the default memory manager for memory
/// allocation use. If 'threadSafe' is true, then creates a leaf memory pool
/// with thread-safe memory usage tracking.
std::shared_ptr<MemoryPool> addDefaultLeafMemoryPool(
    const std::string& name = "",
    bool threadSafe = true);

/// Default unmanaged leaf pool with no threadsafe stats support. Libraries
/// using this method can get a pool that is shared with other threads. The goal
/// is to minimize lock contention while supporting such use cases.
///
///
/// TODO: deprecate this API after all the use cases are able to manage the
/// lifecycle of the allocated memory pools properly.
MemoryPool& deprecatedSharedLeafPool();

FOLLY_ALWAYS_INLINE int32_t alignmentPadding(void* address, int32_t alignment) {
  auto extra = reinterpret_cast<uintptr_t>(address) % alignment;
  return extra == 0 ? 0 : alignment - extra;
}
} // namespace facebook::velox::memory
