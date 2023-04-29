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

#include "velox/common/memory/Memory.h"

DECLARE_bool(velox_enable_memory_usage_track_in_default_memory_pool);

namespace facebook::velox::memory {
namespace {
#define VELOX_MEM_MANAGER_CAP_EXCEEDED(cap)                         \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemCapExceeded.c_str(),       \
      /* isRetriable */ true,                                       \
      "Exceeded memory manager cap of {} MB",                       \
      (cap) / 1024 / 1024);

constexpr folly::StringPiece kDefaultRootName{"__default_root__"};
constexpr folly::StringPiece kDefaultLeafName("__default_leaf__");
} // namespace

MemoryManager::MemoryManager(const Options& options)
    : memoryQuota_{options.capacity},
      allocator_{options.allocator->shared_from_this()},
      arbitrator_(MemoryArbitrator::create(MemoryArbitrator::Config{
          .kind = options.arbitratorConfig.kind,
          .capacity = memoryQuota_,
          .initMemoryPoolCapacity =
              options.arbitratorConfig.initMemoryPoolCapacity,
          .minMemoryPoolCapacityTransferSize =
              options.arbitratorConfig.minMemoryPoolCapacityTransferSize,
      })),
      alignment_(std::max(MemoryAllocator::kMinAlignment, options.alignment)),
      checkUsageLeak_(options.checkUsageLeak),
      poolDestructionCb_([&](MemoryPool* pool) { dropPool(pool); }),
      defaultRoot_{std::make_shared<MemoryPoolImpl>(
          this,
          kDefaultRootName.str(),
          MemoryPool::Kind::kAggregate,
          nullptr,
          nullptr,
          // NOTE: the default root memory pool has no quota limit, and it is
          // used for system usage in production such as disk spilling.
          MemoryPool::Options{
              .alignment = alignment_,
              .capacity = kMaxMemory,
              .trackUsage =
                  FLAGS_velox_enable_memory_usage_track_in_default_memory_pool})},
      deprecatedDefaultLeafPool_(
          defaultRoot_->addLeafChild(kDefaultLeafName.str())) {
  VELOX_CHECK_NOT_NULL(allocator_);
  VELOX_USER_CHECK_GE(memoryQuota_, 0);
  MemoryAllocator::alignmentCheck(0, alignment_);
}

MemoryManager::~MemoryManager() {
  if (checkUsageLeak_) {
    VELOX_CHECK_EQ(
        numPools(),
        0,
        "There are {} unexpected alive memory pools allocated by user on memory manager destruction:\n{}",
        numPools(),
        toString());

    const auto currentBytes = getTotalBytes();
    VELOX_CHECK_EQ(
        currentBytes,
        0,
        "Leaked total memory of {}",
        succinctBytes(currentBytes));
  }
}

int64_t MemoryManager::getMemoryQuota() const {
  return memoryQuota_;
}

uint16_t MemoryManager::alignment() const {
  return alignment_;
}

std::shared_ptr<MemoryPool> MemoryManager::addRootPool(
    const std::string& name,
    int64_t capacity,
    bool trackUsage,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  std::string poolName = name;
  if (poolName.empty()) {
    static std::atomic<int64_t> poolId{0};
    poolName = fmt::format("default_root_{}", poolId++);
  }

  MemoryPool::Options options;
  options.alignment = alignment_;
  if (arbitrator_ != nullptr) {
    options.capacity = 0;
  } else {
    options.capacity = capacity;
  }
  options.trackUsage = trackUsage;
  options.reclaimer = std::move(reclaimer);

  folly::SharedMutex::WriteHolder guard{mutex_};
  if (pools_.find(poolName) != pools_.end()) {
    VELOX_FAIL("Duplicate root pool name found: {}", poolName);
  }
  auto pool = std::make_shared<MemoryPoolImpl>(
      this,
      poolName,
      MemoryPool::Kind::kAggregate,
      nullptr,
      poolDestructionCb_,
      options);
  pools_.emplace(poolName, pool);
  if (arbitrator_ != nullptr) {
    VELOX_CHECK_EQ(pool->capacity(), 0);
    arbitrator_->reserveMemory(pool.get(), capacity);
  }
  return pool;
}

std::shared_ptr<MemoryPool> MemoryManager::addLeafPool(
    const std::string& name,
    bool threadSafe,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  std::string poolName = name;
  if (poolName.empty()) {
    static std::atomic<int64_t> poolId{0};
    poolName = fmt::format("default_leaf_{}", poolId++);
  }
  return defaultRoot_->addLeafChild(poolName, threadSafe, reclaimer);
}

bool MemoryManager::growPool(MemoryPool* pool, uint64_t incrementBytes) {
  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NE(pool->capacity(), kMaxMemory);
  if (arbitrator_ == nullptr) {
    return false;
  }
  // Holds a shared reference to each alive memory pool in 'pools_' to keep
  // their aliveness during the memory arbitration process.
  std::vector<std::shared_ptr<MemoryPool>> candidates;
  bool success;
  {
    folly::SharedMutex::ReadHolder guard{mutex_};
    candidates = getAlivePoolsLocked();
    success = arbitrator_->growMemory(pool, candidates, incrementBytes);
  }
  return success;
}

void MemoryManager::dropPool(MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(pool);
  folly::SharedMutex::WriteHolder guard{mutex_};
  auto it = pools_.find(pool->name());
  if (it == pools_.end()) {
    VELOX_FAIL("The dropped memory pool {} not found", pool->name());
  }
  pools_.erase(it);
  if (arbitrator_ != nullptr) {
    arbitrator_->releaseMemory(pool);
  }
}

MemoryPool& MemoryManager::deprecatedLeafPool() {
  return *deprecatedDefaultLeafPool_;
}

int64_t MemoryManager::getTotalBytes() const {
  return totalBytes_.load(std::memory_order_relaxed);
}

bool MemoryManager::reserve(int64_t size) {
  return totalBytes_.fetch_add(size, std::memory_order_relaxed) + size <=
      memoryQuota_;
}

void MemoryManager::release(int64_t size) {
  totalBytes_.fetch_sub(size, std::memory_order_relaxed);
}

size_t MemoryManager::numPools() const {
  // Don't count 'deprecatedDefaultLeafPool_' which is a child of
  // 'defaultRoot_'.
  size_t numPools = defaultRoot_->getChildCount() - 1;
  VELOX_CHECK_GE(numPools, 0);
  {
    folly::SharedMutex::ReadHolder guard{mutex_};
    numPools += pools_.size();
  }
  return numPools;
}

MemoryAllocator& MemoryManager::getAllocator() {
  return *allocator_;
}

MemoryArbitrator* MemoryManager::arbitrator() {
  return arbitrator_.get();
}

std::string MemoryManager::toString() const {
  std::stringstream out;
  out << "Memory Manager[limit " << succinctBytes(memoryQuota_) << " alignment "
      << succinctBytes(alignment_) << " usedBytes "
      << succinctBytes(totalBytes_) << " number of pools " << numPools()
      << "\n";
  out << "List of root pools:\n";
  out << "\t" << defaultRoot_->name() << "\n";
  std::vector<std::shared_ptr<MemoryPool>> pools = getAlivePools();
  for (const auto& pool : pools) {
    out << "\t" << pool->name() << "\n";
  }
  out << "]";
  return out.str();
}

std::vector<std::shared_ptr<MemoryPool>> MemoryManager::getAlivePools() const {
  folly::SharedMutex::ReadHolder guard{mutex_};
  return getAlivePoolsLocked();
}

std::vector<std::shared_ptr<MemoryPool>> MemoryManager::getAlivePoolsLocked()
    const {
  std::vector<std::shared_ptr<MemoryPool>> pools;
  pools.reserve(pools_.size());
  for (const auto& entry : pools_) {
    auto pool = entry.second.lock();
    if (pool != nullptr) {
      pools.push_back(std::move(pool));
    }
  }
  return pools;
}

IMemoryManager& defaultMemoryManager() {
  return MemoryManager::getInstance();
}

std::shared_ptr<MemoryPool> addDefaultLeafMemoryPool(
    const std::string& name,
    bool threadSafe) {
  auto& memoryManager = defaultMemoryManager();
  return memoryManager.addLeafPool(name, threadSafe);
}
} // namespace facebook::velox::memory
