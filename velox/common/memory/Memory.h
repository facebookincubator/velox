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
#include <mutex>
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
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/MemoryPool.h"

DECLARE_int32(memory_usage_aggregation_interval_millis);

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

class MemoryManager;

/// This function will be used when a MEM_CAP_EXCEEDED error is thrown during
/// a call to increment reservation. It returns a string that will be appended
/// to the error's message and should ideally be used to add additional
/// details to it. This may be called to add details when the error is thrown
/// or when encountered during a recursive call to increment reservation from
/// a parent tracker.
using MemoryCapExceededMessageMaker =
    std::function<std::string(MemoryPool& pool)>;

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

    MemoryArbitrator::Kind arbitratorKind{MemoryArbitrator::Kind::kFixed};

    /// Specifies the backing memory allocator.
    MemoryAllocator* allocator{MemoryAllocator::getInstance()};
  };

  virtual ~IMemoryManager() = default;

  /// Returns the total memory usage allowed under this memory manager.
  /// The memory manager maintains this quota as a hard cap, and any allocation
  /// that would exceed the quota throws.
  virtual uint64_t memoryQuota() const = 0;

  /// Returns the memory allocation alignment of this memory manager.
  virtual uint16_t alignment() const = 0;

  /// Allocates a memory pool for use.
  virtual std::shared_ptr<MemoryPool> getPool(
      int64_t capacity = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr) = 0;

  virtual std::shared_ptr<MemoryPool> getPool(
      const std::string& name,
      int64_t capacity = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr) = 0;

  virtual bool growPool(MemoryPool* pool, uint64_t targetBytes) = 0;

  /// Returns the number of alive pools created from this memory manager.
  virtual size_t numPools() const = 0;
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
      bool ensureQuota = false);

  explicit MemoryManager(const Options& options = Options{});

  ~MemoryManager();

  uint64_t memoryQuota() const final;

  uint16_t alignment() const final;

  std::shared_ptr<MemoryPool> getPool(
      int64_t capacity = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr);

  std::shared_ptr<MemoryPool> getPool(
      const std::string& name,
      int64_t capacity = kMaxMemory,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr) final;

  bool growPool(MemoryPool* pool, uint64_t targetBytes) final;

  size_t numPools() const final;

  MemoryAllocator& allocator();

  void testingVisitPools(std::function<void(MemoryPool*)> visitFn) const;

 private:
  void dropPool(MemoryPool* pool);

  const std::shared_ptr<MemoryAllocator> allocator_;
  const int64_t memoryQuota_;
  const uint16_t alignment_;
  const MemoryPoolImpl::DestructionCallback memoryPoolDtorCb_;
  const std::unique_ptr<MemoryArbitrator> memoryArbitrator_;

  mutable folly::SharedMutex mutex_;
  std::vector<MemoryPool*> pools_;
};

IMemoryManager& getProcessDefaultMemoryManager();

/// Adds a new child memory pool to the root. The new child pool memory cap is
/// set to the input value provided.
std::shared_ptr<MemoryPool> getDefaultMemoryPool(const std::string& name = "");
} // namespace facebook::velox::memory
