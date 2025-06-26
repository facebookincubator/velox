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
#include "velox4j/memory/MemoryManager.h"

#include <fmt/core.h>
#include <folly/Conv.h>
#include <glog/logging.h>
#include <stdint.h>
#include <velox/common/base/BitUtil.h>
#include <velox/common/base/Exceptions.h>
#include <velox/common/base/SuccinctPrinter.h>
#include <velox/common/config/Config.h>
#include <velox/common/memory/Memory.h>
#include <velox/common/memory/MemoryAllocator.h>
#include <velox/common/memory/MemoryArbitrator.h>
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <ostream>
#include <string_view>
#include <utility>

#include "velox4j/memory/AllocationListener.h"
#include "velox4j/memory/ArrowMemoryPool.h"

namespace facebook::velox4j {
using namespace facebook;

namespace {

constexpr std::string_view kMemoryPoolInitialCapacity{
    "memory-pool-initial-capacity"};
constexpr std::string_view kDefaultMemoryPoolInitialCapacity{"256MB"};
constexpr std::string_view kMemoryPoolTransferCapacity{
    "memory-pool-transfer-capacity"};
constexpr std::string_view kDefaultMemoryPoolTransferCapacity{"128MB"};
constexpr std::string_view kMemoryReclaimMaxWaitMs{
    "memory-reclaim-max-wait-time"};
constexpr std::string_view kDefaultMemoryReclaimMaxWaitMs{"3600000ms"};

template <typename T>
T getConfig(
    const std::unordered_map<std::string, std::string>& configs,
    const std::string_view& key,
    const T& defaultValue) {
  if (configs.count(std::string(key)) > 0) {
    try {
      return folly::to<T>(configs.at(std::string(key)));
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed while parsing SharedArbitrator configs: {}", e.what());
    }
  }
  return defaultValue;
}

class ListenableArbitrator : public velox::memory::MemoryArbitrator {
 public:
  ListenableArbitrator(const Config& config, AllocationListener* listener)
      : MemoryArbitrator(config),
        listener_(listener),
        memoryPoolInitialCapacity_(velox::config::toCapacity(
            getConfig<std::string>(
                config.extraConfigs,
                kMemoryPoolInitialCapacity,
                std::string(kDefaultMemoryPoolInitialCapacity)),
            velox::config::CapacityUnit::BYTE)),
        memoryPoolTransferCapacity_(velox::config::toCapacity(
            getConfig<std::string>(
                config.extraConfigs,
                kMemoryPoolTransferCapacity,
                std::string(kDefaultMemoryPoolTransferCapacity)),
            velox::config::CapacityUnit::BYTE)),
        memoryReclaimMaxWaitMs_(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                velox::config::toDuration(getConfig<std::string>(
                    config.extraConfigs,
                    kMemoryReclaimMaxWaitMs,
                    std::string(kDefaultMemoryReclaimMaxWaitMs))))
                .count()) {}

  std::string kind() const override {
    return kind_;
  }

  void shutdown() override {}

  void addPool(
      const std::shared_ptr<velox::memory::MemoryPool>& pool) override {
    VELOX_CHECK_EQ(pool->capacity(), 0);

    std::unique_lock guard{mutex_};
    VELOX_CHECK_EQ(candidates_.count(pool.get()), 0);
    candidates_.emplace(pool.get(), pool->weak_from_this());
  }

  void removePool(velox::memory::MemoryPool* pool) override {
    VELOX_CHECK_EQ(pool->reservedBytes(), 0);
    shrinkCapacity(pool, pool->capacity());

    std::unique_lock guard{mutex_};
    const auto ret = candidates_.erase(pool);
    VELOX_CHECK_EQ(ret, 1);
  }

  void growCapacity(velox::memory::MemoryPool* pool, uint64_t targetBytes)
      override {
    // Set arbitration context to allow memory over-use during recursive
    // arbitration. See MemoryPoolImpl::maybeIncrementReservation.
    velox::memory::ScopedMemoryArbitrationContext ctx{};
    velox::memory::MemoryPool* candidate;
    {
      std::unique_lock guard{mutex_};
      VELOX_CHECK_EQ(
          candidates_.size(),
          1,
          "ListenableArbitrator should only be used within a single root pool");
      candidate = candidates_.begin()->first;
    }
    VELOX_CHECK(
        pool->root() == candidate, "Illegal state in ListenableArbitrator");

    growCapacityInternal(pool->root(), targetBytes);
  }

  uint64_t shrinkCapacity(
      uint64_t targetBytes,
      bool allowSpill,
      bool allowAbort) override {
    velox::memory::ScopedMemoryArbitrationContext ctx{};
    velox::memory::MemoryReclaimer::Stats status;
    velox::memory::MemoryPool* pool = nullptr;
    {
      std::unique_lock guard{mutex_};
      VELOX_CHECK_EQ(
          candidates_.size(),
          1,
          "ListenableArbitrator should only be used within a single root pool");
      pool = candidates_.begin()->first;
    }
    pool->reclaim(
        targetBytes,
        memoryReclaimMaxWaitMs_,
        status); // ignore the output
    return shrinkCapacityInternal(pool, 0);
  }

  uint64_t shrinkCapacity(velox::memory::MemoryPool* pool, uint64_t targetBytes)
      override {
    return shrinkCapacityInternal(pool, targetBytes);
  }

  Stats stats() const override {
    Stats stats; // no-op
    return stats;
  }

  std::string toString() const override {
    return fmt::format(
        "ARBITRATOR[{}] CAPACITY {} {}",
        kind_,
        velox::succinctBytes(capacity()),
        stats().toString());
  }

 private:
  void growCapacityInternal(velox::memory::MemoryPool* pool, uint64_t bytes) {
    // Since
    // https://github.com/facebookincubator/velox/pull/9557/files#diff-436e44b7374032f8f5d7eb45869602add6f955162daa2798d01cc82f8725724dL812-L820,
    // We should pass bytes as parameter "reservationBytes" when calling ::grow.
    auto freeByes = pool->freeBytes();
    if (freeByes > bytes) {
      if (growPool(pool, 0, bytes)) {
        return;
      }
    }
    auto reclaimedFreeBytes = shrinkPool(pool, 0);
    auto neededBytes = velox::bits::roundUp(
        bytes - reclaimedFreeBytes, memoryPoolTransferCapacity_);
    try {
      listener_->allocationChanged(neededBytes);
    } catch (const std::exception& e) {
      // if allocationChanged failed, we need to free the reclaimed bytes
      listener_->allocationChanged(-reclaimedFreeBytes);
      throw;
    }
    auto ret = growPool(pool, reclaimedFreeBytes + neededBytes, bytes);
    VELOX_CHECK(
        ret,
        "{} failed to grow {} bytes, current state {}",
        pool->name(),
        velox::succinctBytes(bytes),
        pool->toString());
  }

  uint64_t shrinkCapacityInternal(
      velox::memory::MemoryPool* pool,
      uint64_t bytes) {
    uint64_t freeBytes = shrinkPool(pool, bytes);
    listener_->allocationChanged(-freeBytes);
    return freeBytes;
  }

  // TODO: This variable is Unused.
  const uint64_t memoryPoolInitialCapacity_;
  const uint64_t memoryPoolTransferCapacity_;
  const uint64_t memoryReclaimMaxWaitMs_;
  AllocationListener* listener_{nullptr};

  mutable std::mutex mutex_;
  inline static std::string kind_ = "VELOX4J";
  std::unordered_map<
      velox::memory::MemoryPool*,
      std::weak_ptr<velox::memory::MemoryPool>>
      candidates_;
};

// A helper class to allow the memory manager find the listenable
// arbitrator to be assigned to the Velox memory manager.
class ArbitratorFactoryRegister {
 public:
  explicit ArbitratorFactoryRegister(AllocationListener* listener)
      : listener_(listener) {
    static std::atomic_uint32_t id{0UL};
    kind_ = "VELOX4J_ARBITRATOR_FACTORY_" + std::to_string(id++);
    velox::memory::MemoryArbitrator::registerFactory(
        kind_,
        [this](const velox::memory::MemoryArbitrator::Config& config)
            -> std::unique_ptr<velox::memory::MemoryArbitrator> {
          return std::make_unique<ListenableArbitrator>(config, listener_);
        });
  }

  virtual ~ArbitratorFactoryRegister() {
    velox::memory::MemoryArbitrator::unregisterFactory(kind_);
  }

  const std::string& getKind() const {
    return kind_;
  }

 private:
  std::string kind_;
  AllocationListener* listener_;
};
} // namespace

MemoryManager::MemoryManager(std::unique_ptr<AllocationListener> listener)
    : listener_(std::move(listener)) {
  arrowAllocator_ = std::make_unique<ListenableMemoryAllocator>(
      defaultMemoryAllocator().get(), listener_.get());
  std::unordered_map<std::string, std::string> extraArbitratorConfigs;
  ArbitratorFactoryRegister arbitratorFactoryRegister(listener_.get());
  velox::memory::MemoryManagerOptions mmOptions{
      .alignment = velox::memory::MemoryAllocator::kMaxAlignment,
      .trackDefaultUsage = true, // memory usage tracking
      .checkUsageLeak = true, // leak check
      .coreOnAllocationFailureEnabled = false,
      .allocatorCapacity = velox::memory::kMaxMemory,
      .arbitratorKind = arbitratorFactoryRegister.getKind(),
      .extraArbitratorConfigs = extraArbitratorConfigs};
  veloxMemoryManager_ =
      std::make_unique<velox::memory::MemoryManager>(mmOptions);
  veloxRootPool_ = veloxMemoryManager_->addRootPool(
      "root",
      velox::memory::kMaxMemory, // the 3rd capacity
      facebook::velox::memory::MemoryReclaimer::create());
}

namespace {

void detectAndLogMemoryLeak(
    const velox::memory::MemoryPool* pool,
    bool& leakFound) {
  if (pool->usedBytes() != 0) {
    LOG(ERROR)
        << "[Velox4J MemoryManager DTOR] "
        << "Memory leak found on Velox memory pool: " << pool->toString()
        << ". "
        << "Please make sure your code released all opened resources already.";
    leakFound = true;
  }
}

void detectAndLogMemoryLeak(const arrow::MemoryPool* pool, bool& leakFound) {
  if (pool->bytes_allocated() != 0) {
    LOG(ERROR)
        << "[Velox4J MemoryManager DTOR] "
        << "Memory leak found on Arrow memory pool: " << pool->backend_name()
        << ". "
        << "Please make sure your code released all opened resources already.";
    leakFound = true;
  }
}

} // namespace

bool MemoryManager::tryDestruct() {
  bool leakFound{false};
  // Velox memory pools considered safe to destruct when no alive allocations.
  for (const auto& pair : veloxPoolRefs_) {
    const auto& veloxPool = pair.second;
    VELOX_CHECK_NOT_NULL(veloxPool);
    detectAndLogMemoryLeak(veloxPool.get(), leakFound);
  }
  detectAndLogMemoryLeak(veloxRootPool_.get(), leakFound);
  veloxPoolRefs_.clear();
  veloxRootPool_.reset();

  // Velox memory manager considered safe to destruct when no alive pools.
  if (veloxMemoryManager_) {
    if (veloxMemoryManager_->numPools() > 3) {
      LOG(ERROR) << "[Velox4J MemoryManager DTOR] " << "There are "
                 << veloxMemoryManager_->numPools()
                 << " outstanding Velox memory pools.";
      leakFound = true;
    }
    if (veloxMemoryManager_->numPools() < 3) {
      VELOX_FAIL(
          "Fatal: Velox memory manager should have at least 3 built-in pools");
    }
    int32_t spillPoolCount = 0;
    int32_t cachePoolCount = 0;
    int32_t tracePoolCount = 0;
    veloxMemoryManager_->deprecatedSysRootPool().visitChildren(
        [&](velox::memory::MemoryPool* child) -> bool {
          if (child == veloxMemoryManager_->spillPool()) {
            spillPoolCount++;
          }
          if (child == veloxMemoryManager_->cachePool()) {
            cachePoolCount++;
          }
          if (child == veloxMemoryManager_->tracePool()) {
            tracePoolCount++;
          }
          return true;
        });
    VELOX_CHECK(
        spillPoolCount == 1,
        "Illegal pool count state: spillPoolCount: " +
            std::to_string(spillPoolCount));
    VELOX_CHECK(
        cachePoolCount == 1,
        "Illegal pool count state: cachePoolCount: " +
            std::to_string(cachePoolCount));
    VELOX_CHECK(
        tracePoolCount == 1,
        "Illegal pool count state: tracePoolCount: " +
            std::to_string(tracePoolCount));
  }
  veloxMemoryManager_.reset();

  // Applies similar rule for Arrow memory pool.
  for (const auto& pair : arrowPoolRefs_) {
    const auto& arrowPool = pair.second;
    VELOX_CHECK_NOT_NULL(arrowPool);
    detectAndLogMemoryLeak(arrowPool.get(), leakFound);
  }
  arrowPoolRefs_.clear();
  return !leakFound;
}

MemoryManager::~MemoryManager() {
  bool succeeded{false};
  try {
    succeeded = tryDestruct();
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[Velox4J MemoryManager DTOR] "
               << " Error occurred: " << ex.what();
  }
  if (!succeeded) {
    LOG(ERROR) << "[Velox4J MemoryManager DTOR] "
               << "Fatal: Memory leak found, aborting the destruction of "
                  "MemoryManager. This could cause the process to crash.";
    VELOX_FAIL("Memory leak found during destruction of MemoryManager");
  }
}

velox::memory::MemoryPool* MemoryManager::getVeloxPool(
    const std::string& name,
    const velox::memory::MemoryPool::Kind& kind) {
  if (const auto it = veloxPoolRefs_.find(name); it != veloxPoolRefs_.end()) {
    const auto& pool = it->second;
    VELOX_CHECK_EQ(
        pool->kind(),
        kind,
        "Pool [" + name + "] was already created but with different kind");
    return pool.get();
  }
  switch (kind) {
    case velox::memory::MemoryPool::Kind::kLeaf: {
      auto pool = veloxRootPool_->addLeafChild(
          name, true, velox::memory::MemoryReclaimer::create());
      veloxPoolRefs_[name] = pool;
      return pool.get();
    }
    case velox::memory::MemoryPool::Kind::kAggregate: {
      auto pool = veloxRootPool_->addAggregateChild(
          name, velox::memory::MemoryReclaimer::create());
      veloxPoolRefs_[name] = pool;
      return pool.get();
    }
    default: {
      VELOX_FAIL("Unexpected memory pool kind");
    }
  }
  VELOX_UNREACHABLE();
}

arrow::MemoryPool* MemoryManager::getArrowPool(const std::string& name) {
  if (const auto it = arrowPoolRefs_.find(name); it != arrowPoolRefs_.end()) {
    return it->second.get();
  }
  arrowPoolRefs_[name] =
      std::make_unique<ArrowMemoryPool>(arrowAllocator_.get());
  return arrowPoolRefs_[name].get();
}
} // namespace facebook::velox4j
