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

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/MemoryPool.h"

namespace facebook {
namespace velox {
namespace memory {
namespace {
constexpr folly::StringPiece kRootNodeName{"__root__"};

std::vector<std::string> getPoolNames(const std::vector<MemoryPool*>& pools) {
  std::vector<std::string> poolNames;
  poolNames.reserve(pools.size());
  for (const auto& pool : pools) {
    poolNames.push_back(pool->name());
  }
  return poolNames;
}
} // namespace

MemoryManager::MemoryManager(const Options& options)
    : allocator_{options.allocator->shared_from_this()},
      memoryQuota_{options.capacity},
      alignment_(std::max(MemoryAllocator::kMinAlignment, options.alignment)),
      memoryPoolDtorCb_([&](MemoryPool* pool) { dropPool(pool); }),
      memoryArbitrator_(MemoryArbitrator::create(MemoryArbitrator::Config{
          .kind = options.arbitratorKind,
          .capacity = options.capacity})) {
  VELOX_CHECK_NOT_NULL(allocator_);
  VELOX_USER_CHECK_GE(memoryQuota_, 0);
  MemoryAllocator::alignmentCheck(0, alignment_);
}

MemoryManager::~MemoryManager() {
  VELOX_CHECK_EQ(
      pools_.size(),
      0,
      "There are memory pools left on destruction: {}",
      folly::join(",", getPoolNames(pools_)));
}

MemoryManager& MemoryManager::getInstance(
    const Options& options,
    bool ensureQuota) {
  static MemoryManager manager{options};
  auto actualCapacity = manager.memoryQuota();
  VELOX_USER_CHECK(
      !ensureQuota || actualCapacity == options.capacity,
      "Process level manager manager created with input capacity: {}, actual capacity: {}",
      options.capacity,
      actualCapacity);
  return manager;
}

uint64_t MemoryManager::memoryQuota() const {
  return memoryQuota_;
}

uint16_t MemoryManager::alignment() const {
  return alignment_;
}

std::shared_ptr<MemoryPool> MemoryManager::getPool(
    int64_t capacity,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  static std::atomic<int64_t> poolId{0};
  return getPool(
      fmt::format("default_usage_node_{}", poolId++),
      capacity,
      std::move(reclaimer));
}

std::shared_ptr<MemoryPool> MemoryManager::getPool(
    const std::string& name,
    int64_t capacity,
    std::shared_ptr<MemoryReclaimer> reclaimer) {
  MemoryPool::Options options;
  options.alignment = alignment_;
  options.capacity = memoryArbitrator_->reserveMemory(capacity);
  options.reclaimer = reclaimer;
  auto pool = std::make_shared<MemoryPoolImpl>(
      this,
      name,
      MemoryPool::Kind::kAggregate,
      nullptr,
      memoryPoolDtorCb_,
      options);
  {
    folly::SharedMutex::WriteHolder guard{mutex_};
    pools_.push_back(pool.get());
  }
  return pool;
}

void MemoryManager::dropPool(MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(pool);
  memoryArbitrator_->releaseMemory(pool);
  folly::SharedMutex::WriteHolder guard{mutex_};
  auto it = pools_.begin();
  while (it != pools_.end()) {
    if (*it == pool) {
      pools_.erase(it);
      return;
    }
  }
  VELOX_UNREACHABLE("Memory pool{} is not found", pool->name());
}

bool MemoryManager::growPool(MemoryPool* pool, uint64_t targetBytes) {
  VELOX_CHECK_NE(targetBytes, kMaxMemory);
  VELOX_CHECK_NE(pool->capacity(), kMaxMemory);
  folly::SharedMutex::ReadHolder guard{mutex_};
  return memoryArbitrator_->growMemory(pool, pools_, targetBytes);
}

size_t MemoryManager::numPools() const {
  folly::SharedMutex::ReadHolder guard{mutex_};
  return pools_.size();
}

MemoryAllocator& MemoryManager::allocator() {
  return *allocator_;
}

void MemoryManager::testingVisitPools(
    std::function<void(MemoryPool*)> visitFn) const {
  folly::SharedMutex::ReadHolder guard{mutex_};
  for (auto* pool : pools_) {
    visitFn(pool);
  }
}

IMemoryManager& getProcessDefaultMemoryManager() {
  return MemoryManager::getInstance();
}

std::shared_ptr<MemoryPool> getDefaultMemoryPool(const std::string& name) {
  static std::shared_ptr<MemoryPool> pool =
      getProcessDefaultMemoryManager().getPool("default_pool_root");
  std::string poolName = name;
  if (poolName.empty()) {
    static std::atomic<int64_t> poolId{0};
    poolName = fmt::format("default_leaf_pool_{}", poolId++);
  }
  return pool->addChild(poolName, MemoryPool::Kind::kLeaf);
}

} // namespace memory
} // namespace velox
} // namespace facebook
