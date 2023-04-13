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
#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/memory/MemoryUsage.h"

DECLARE_int32(memory_usage_aggregation_interval_millis);
DECLARE_bool(velox_memory_leak_check_enabled);

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

/// This class provides the interface of memory manager. The memory manager is
/// responsible for enforcing the memory usage quota as well as managing the
/// memory pools.
class IMemoryManager {
 public:
  struct Options {
    /// Specifies the default memory allocation alignment.
    uint16_t alignment{MemoryAllocator::kMaxAlignment};

    /// Specifies the max memory capacity in bytes.
    int64_t capacity{kMaxMemory};

    /// If true, check the memory pool and usage leaks on destruction.
    ///
    /// TODO: deprecate this flag after all the existing memory leak use cases
    /// have been fixed.
    bool checkUsageLeak{FLAGS_velox_memory_leak_check_enabled};

    /// Specifies the backing memory allocator.
    MemoryAllocator* allocator{MemoryAllocator::getInstance()};
  };

  virtual ~IMemoryManager() = default;

  /// Returns the total memory usage allowed under this memory manager.
  /// The memory manager maintains this quota as a hard cap, and any allocation
  /// that would exceed the quota throws.
  virtual int64_t getMemoryQuota() const = 0;

  /// Returns the memory allocation alignment of this memory manager.
  virtual uint16_t alignment() const = 0;

  /// Creates a memory pool for use with specified 'name', 'kind' and 'cap'. If
  /// 'name' is missing, the memory manager generates a default name internally
  /// to ensure uniqueness. If 'kind' is kAggregate, a root memory pool is
  /// created. Otherwise, a leaf memory pool is created as the child of the
  /// memory manager's default root memory pool. If 'kind' is kAggregate and
  /// 'trackUsage' is true, then set the memory usage tracker in the created
  /// memory pool.
  virtual std::shared_ptr<MemoryPool> getPool(
      const std::string& name = "",
      MemoryPool::Kind kind = MemoryPool::Kind::kAggregate,
      int64_t maxBytes = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr,
      bool trackUsage = true) = 0;

  /// Returns the number of alive memory pools allocated from getPool().
  ///
  /// NOTE: this doesn't count the memory manager's internal default root and
  /// leaf memory pools.
  virtual size_t numPools() const = 0;

  /// Returns a leaf memory pool for memory allocation use. The pool is
  /// created as the child of the memory manager's default root memory pool and
  /// is owned by the memory manager.
  ///
  /// TODO: deprecate this API after all the use cases are able to manage the
  /// lifecycle of the allocated memory pools properly.
  virtual MemoryPool& deprecatedGetPool() = 0;

  /// Returns the current total memory usage under this memory manager.
  virtual int64_t getTotalBytes() const = 0;

  /// Reserves size for the allocation. Returns true if the total usage remains
  /// under quota after the reservation. Caller is responsible for releasing the
  /// offending reservation.
  ///
  /// TODO: deprecate this and enforce the memory usage quota by memory pool.
  virtual bool reserve(int64_t size) = 0;

  /// Subtracts from current total and regain memory quota.
  ///
  /// TODO: deprecate this and enforce the memory usage quota by memory pool.
  virtual void release(int64_t size) = 0;

  /// Returns debug string of this memory manager.
  virtual std::string toString() const = 0;
};

/// For now, users wanting multiple different allocators would need to
/// instantiate different MemoryManager classes and manage them across static
/// boundaries.
class MemoryManager final : public IMemoryManager {
 public:
  /// Tries to get the singleton memory manager. If not previously initialized,
  /// the process singleton manager will be initialized with the given quota.
  FOLLY_EXPORT static MemoryManager& getInstance(
      const Options& options = Options{},
      bool ensureQuota = false) {
    static MemoryManager manager{options};
    auto actualCapacity = manager.getMemoryQuota();
    VELOX_USER_CHECK(
        !ensureQuota || actualCapacity == options.capacity,
        "Process level manager manager created with input capacity: {}, actual capacity: {}",
        options.capacity,
        actualCapacity);

    return manager;
  }

  explicit MemoryManager(const Options& options = Options{});

  ~MemoryManager();

  int64_t getMemoryQuota() const final;

  uint16_t alignment() const final;

  std::shared_ptr<MemoryPool> getPool(
      const std::string& name = "",
      MemoryPool::Kind kind = MemoryPool::Kind::kAggregate,
      int64_t maxBytes = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr,
      bool trackUsage = true) final;

  MemoryPool& deprecatedGetPool() final;

  int64_t getTotalBytes() const final;

  bool reserve(int64_t size) final;
  void release(int64_t size) final;

  size_t numPools() const final;

  std::string toString() const final;

  MemoryAllocator& getAllocator();

  /// Returns the memory manger's internal default root memory pool for testing
  /// purpose.
  MemoryPool& testingDefaultRoot() const {
    return *defaultRoot_;
  }

 private:
  void dropPool(MemoryPool* pool);

  const std::shared_ptr<MemoryAllocator> allocator_;
  const int64_t memoryQuota_;
  const uint16_t alignment_;
  const bool checkUsageLeak_;
  // The destruction callback set for the root memory pools created by getPool()
  // which are tracked by 'pools_'. It is invoked on the root pool destruction
  // and removes the pool from 'pools_'.
  const MemoryPoolImpl::DestructionCallback poolDestructionCb_;

  const std::shared_ptr<MemoryPool> defaultRoot_;
  // The leaf memory pool created as child of 'defaultRoot_' on memory manager
  // construction. This is for legacy use cases that can't manage the memory
  // pool's lifecycle properly, and will be deprecated later.
  const std::shared_ptr<MemoryPool> deprecatedDefaultLeafPool_;

  mutable folly::SharedMutex mutex_;
  std::atomic_long totalBytes_{0};
  std::vector<MemoryPool*> pools_;
};

IMemoryManager& getProcessDefaultMemoryManager();

/// Creates a leaf memory pool from the default memory manager for memory
/// allocation use.
std::shared_ptr<MemoryPool> getDefaultMemoryPool(const std::string& name = "");

FOLLY_ALWAYS_INLINE int32_t alignmentPadding(void* address, int32_t alignment) {
  auto extra = reinterpret_cast<uintptr_t>(address) % alignment;
  return extra == 0 ? 0 : alignment - extra;
}
} // namespace facebook::velox::memory
