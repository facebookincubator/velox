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

#include <gtest/gtest.h>

#include <deque>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"

using namespace ::testing;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

namespace facebook::velox::memory {

class MemoryArbitrationTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }
};

TEST_F(MemoryArbitrationTest, stats) {
  MemoryArbitrator::Stats stats;
  stats.numRequests = 2;
  stats.numAborted = 3;
  stats.numFailures = 100;
  stats.queueTimeUs = 230'000;
  stats.arbitrationTimeUs = 1020;
  stats.numShrunkBytes = 100'000'000;
  stats.numReclaimedBytes = 10'000;
  stats.reclaimTimeUs = 1'000;
  stats.numNonReclaimableAttempts = 5;
  ASSERT_EQ(
      stats.toString(),
      "STATS[numRequests 2 numSucceeded 0 numAborted 3 numFailures 100 "
      "numNonReclaimableAttempts 5 numReserves 0 numReleases 0 "
      "queueTime 230.00ms arbitrationTime 1.02ms reclaimTime 1.00ms "
      "shrunkMemory 95.37MB reclaimedMemory 9.77KB "
      "maxCapacity 0B freeCapacity 0B]");
  ASSERT_EQ(
      fmt::format("{}", stats),
      "STATS[numRequests 2 numSucceeded 0 numAborted 3 numFailures 100 "
      "numNonReclaimableAttempts 5 numReserves 0 numReleases 0 "
      "queueTime 230.00ms arbitrationTime 1.02ms reclaimTime 1.00ms "
      "shrunkMemory 95.37MB reclaimedMemory 9.77KB "
      "maxCapacity 0B freeCapacity 0B]");
}

TEST_F(MemoryArbitrationTest, create) {
  const std::string unknownType = "UNKNOWN";
  const std::vector<std::string> kinds = {
      std::string(),
      unknownType // expect error on unregistered type
  };
  for (const auto& kind : kinds) {
    MemoryArbitrator::Config config;
    config.capacity = 1 * GB;
    config.kind = kind;
    if (kind.empty()) {
      auto arbitrator = MemoryArbitrator::create(config);
      ASSERT_EQ(arbitrator->kind(), "NOOP");
    } else if (kind == unknownType) {
      VELOX_ASSERT_THROW(
          MemoryArbitrator::create(config),
          "Arbitrator factory for kind UNKNOWN not registered");
    } else {
      FAIL();
    }
  }
}

TEST_F(MemoryArbitrationTest, createWithDefaultConf) {
  MemoryArbitrator::Config config;
  config.capacity = 1 * GB;
  const auto& arbitrator = MemoryArbitrator::create(config);
  ASSERT_EQ(arbitrator->kind(), "NOOP");
}

TEST_F(MemoryArbitrationTest, queryMemoryCapacity) {
  {
    // Reserved memory is not enforced when no arbitrator is provided.
    MemoryManagerOptions options;
    options.allocatorCapacity = 8L << 20;
    options.arbitratorCapacity = 4L << 20;
    MemoryManager manager(options);
    auto rootPool = manager.addRootPool("root-1", 8L << 20);
    auto leafPool = rootPool->addLeafChild("leaf-1.0");
    void* buffer;
    ASSERT_NO_THROW({
      buffer = leafPool->allocate(7L << 20);
      leafPool->free(buffer, 7L << 20);
    });
  }
  {
    // Reserved memory is e`nforced when SharedMemoryArbitrator is used.
    SharedArbitrator::registerFactory();
    MemoryManagerOptions options;
    options.allocatorCapacity = 8L << 20;
    options.arbitratorCapacity = 4L << 20;
    options.arbitratorKind = "SHARED";
    options.memoryPoolInitCapacity = 1 << 20;
    MemoryManager manager(options);
    auto rootPool =
        manager.addRootPool("root-1", 8L << 20, MemoryReclaimer::create());
    ASSERT_EQ(rootPool->capacity(), 1 << 20);
    ASSERT_EQ(
        manager.arbitrator()->growCapacity(rootPool.get(), 1 << 20), 1 << 20);
    ASSERT_EQ(
        manager.arbitrator()->growCapacity(rootPool.get(), 6 << 20), 2 << 20);
    auto leafPool = rootPool->addLeafChild("leaf-1.0");
    void* buffer;
    VELOX_ASSERT_THROW(
        buffer = leafPool->allocate(7L << 20),
        "Exceeded memory pool cap of 4.00MB");
    ASSERT_NO_THROW(buffer = leafPool->allocate(4L << 20));
    ASSERT_EQ(manager.arbitrator()->shrinkCapacity(rootPool.get(), 0), 0);
    ASSERT_EQ(manager.arbitrator()->shrinkCapacity(leafPool.get(), 0), 0);
    ASSERT_EQ(manager.arbitrator()->shrinkCapacity(leafPool.get(), 1), 0);
    ASSERT_EQ(manager.arbitrator()->shrinkCapacity(rootPool.get(), 1), 0);
    leafPool->free(buffer, 4L << 20);
    ASSERT_EQ(
        manager.arbitrator()->shrinkCapacity(leafPool.get(), 1 << 20), 1 << 20);
    ASSERT_EQ(
        manager.arbitrator()->shrinkCapacity(rootPool.get(), 1 << 20), 1 << 20);
    ASSERT_EQ(rootPool->capacity(), 2 << 20);
    ASSERT_EQ(leafPool->capacity(), 2 << 20);
    ASSERT_EQ(manager.arbitrator()->shrinkCapacity(leafPool.get(), 0), 2 << 20);
    ASSERT_EQ(rootPool->capacity(), 0);
    ASSERT_EQ(leafPool->capacity(), 0);
    memory::SharedArbitrator::unregisterFactory();
  }
}

TEST_F(MemoryArbitrationTest, arbitratorStats) {
  const MemoryArbitrator::Stats emptyStats;
  ASSERT_TRUE(emptyStats.empty());
  const MemoryArbitrator::Stats anchorStats(
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5);
  ASSERT_FALSE(anchorStats.empty());
  const MemoryArbitrator::Stats largeStats(
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
  ASSERT_FALSE(largeStats.empty());
  ASSERT_TRUE(!(anchorStats == largeStats));
  ASSERT_TRUE(anchorStats != largeStats);
  ASSERT_TRUE(anchorStats < largeStats);
  ASSERT_TRUE(!(anchorStats > largeStats));
  ASSERT_TRUE(anchorStats <= largeStats);
  ASSERT_TRUE(!(anchorStats >= largeStats));
  const auto delta = largeStats - anchorStats;
  ASSERT_EQ(
      delta, MemoryArbitrator::Stats(3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3, 3, 3, 3));

  const MemoryArbitrator::Stats smallStats(
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  ASSERT_TRUE(!(anchorStats == smallStats));
  ASSERT_TRUE(anchorStats != smallStats);
  ASSERT_TRUE(!(anchorStats < smallStats));
  ASSERT_TRUE(anchorStats > smallStats);
  ASSERT_TRUE(!(anchorStats <= smallStats));
  ASSERT_TRUE(anchorStats >= smallStats);

  const MemoryArbitrator::Stats invalidStats(
      2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 2, 8, 2);
  ASSERT_TRUE(!(anchorStats == invalidStats));
  ASSERT_TRUE(anchorStats != invalidStats);
  ASSERT_THROW(anchorStats < invalidStats, VeloxException);
  ASSERT_THROW(anchorStats > invalidStats, VeloxException);
  ASSERT_THROW(anchorStats <= invalidStats, VeloxException);
  ASSERT_THROW(anchorStats >= invalidStats, VeloxException);
}

namespace {
class FakeTestArbitrator : public MemoryArbitrator {
 public:
  explicit FakeTestArbitrator(const Config& config)
      : MemoryArbitrator(
            {.kind = config.kind,
             .capacity = config.capacity,
             .memoryPoolTransferCapacity = config.memoryPoolTransferCapacity}) {
  }

  std::string kind() const override {
    return "USER";
  }

  uint64_t growCapacity(MemoryPool* /*unused*/, uint64_t /*unused*/) override {
    VELOX_NYI();
    return 0;
  }

  bool growCapacity(
      MemoryPool* /*unused*/,
      const std::vector<std::shared_ptr<MemoryPool>>& /*unused*/,
      uint64_t /*unused*/) override {
    VELOX_NYI();
  }

  uint64_t shrinkCapacity(MemoryPool* /*unused*/, uint64_t /*unused*/)
      override {
    VELOX_NYI();
  }

  uint64_t shrinkCapacity(
      const std::vector<std::shared_ptr<MemoryPool>>& /*unused*/,
      uint64_t /*unused*/,
      bool /*unused*/,
      bool /*unused*/) override {
    VELOX_NYI();
  }

  Stats stats() const override {
    VELOX_NYI();
  }

  std::string toString() const override {
    VELOX_NYI();
  }
};
} // namespace

class MemoryArbitratorFactoryTest : public testing::Test {
 public:
 protected:
  static void SetUpTestCase() {
    MemoryArbitrator::registerFactory(kind_, factory_);
    memory::MemoryManager::testingSetInstance({});
  }

  static void TearDownTestCase() {
    MemoryArbitrator::unregisterFactory(kind_);
  }

 protected:
  inline static const std::string kind_{"USER"};
  inline static const MemoryArbitrator::Factory factory_{
      [](const MemoryArbitrator::Config& config) {
        return std::make_unique<FakeTestArbitrator>(config);
      }};
};

TEST_F(MemoryArbitratorFactoryTest, register) {
  VELOX_ASSERT_THROW(
      MemoryArbitrator::registerFactory(kind_, factory_),
      "Arbitrator factory for kind USER is already registered");
}

TEST_F(MemoryArbitratorFactoryTest, create) {
  MemoryArbitrator::Config config;
  config.kind = kind_;
  auto arbitrator = MemoryArbitrator::create(config);
  ASSERT_EQ(kind_, arbitrator->kind());
}

class MemoryReclaimerTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  MemoryReclaimerTest() {
    const auto seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
    LOG(INFO) << "Random seed: " << seed;
  }

  void SetUp() override {}

  void TearDown() override {}

  folly::Random::DefaultGenerator rng_;
  MemoryReclaimer::Stats stats_;
};

TEST_F(MemoryReclaimerTest, common) {
  struct {
    int numChildren;
    int numGrandchildren;
    bool doAllocation;

    std::string debugString() const {
      return fmt::format(
          "numChildren {} numGrandchildren {} doAllocation{}",
          numChildren,
          numGrandchildren,
          doAllocation);
    }
  } testSettings[] = {
      {0, 0, false},
      {0, 0, true},
      {1, 0, false},
      {1, 0, true},
      {100, 0, false},
      {100, 0, true},
      {1, 100, false},
      {1, 100, true},
      {10, 100, false},
      {10, 100, true},
      {100, 1, false},
      {100, 1, true},
      {0, 0, false},
      {0, 0, true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    std::vector<std::shared_ptr<MemoryPool>> pools;
    auto pool = memory::memoryManager()->addRootPool(
        "shrinkAPIs", kMaxMemory, memory::MemoryReclaimer::create());
    pools.push_back(pool);

    struct Allocation {
      void* buffer;
      size_t size;
      MemoryPool* pool;
    };
    std::vector<Allocation> allocations;
    for (int i = 0; i < testData.numChildren; ++i) {
      const bool isLeaf = testData.numGrandchildren == 0;
      auto childPool = isLeaf
          ? pool->addLeafChild(
                std::to_string(i), true, memory::MemoryReclaimer::create())
          : pool->addAggregateChild(
                std::to_string(i), memory::MemoryReclaimer::create());
      pools.push_back(childPool);
      for (int j = 0; j < testData.numGrandchildren; ++j) {
        auto grandchild = childPool->addLeafChild(
            std::to_string(j), true, memory::MemoryReclaimer::create());
        pools.push_back(grandchild);
      }
    }
    for (auto& pool : pools) {
      ASSERT_NE(pool->reclaimer(), nullptr);
      if (pool->kind() == MemoryPool::Kind::kLeaf) {
        const size_t size = 1 + folly::Random::rand32(rng_) % 1024;
        void* buffer = pool->allocate(size);
        allocations.push_back({buffer, size, pool.get()});
      }
    }
    for (auto& pool : pools) {
      ASSERT_FALSE(pool->reclaimableBytes().has_value());
      ASSERT_EQ(pool->reclaim(0, 0, stats_), 0);
      ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
      ASSERT_EQ(pool->reclaim(100, 0, stats_), 0);
      ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
      ASSERT_EQ(pool->reclaim(kMaxMemory, 0, stats_), 0);
      ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
    }
    ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
    for (const auto& allocation : allocations) {
      allocation.pool->free(allocation.buffer, allocation.size);
    }
  }
}

class MockLeafMemoryReclaimer : public MemoryReclaimer {
 public:
  explicit MockLeafMemoryReclaimer(std::atomic<uint64_t>& totalUsedBytes)
      : totalUsedBytes_(totalUsedBytes) {}

  ~MockLeafMemoryReclaimer() override {
    VELOX_CHECK(allocations_.empty());
  }

  bool reclaimableBytes(const MemoryPool& pool, uint64_t& bytes)
      const override {
    VELOX_CHECK_EQ(pool.name(), pool_->name());
    bytes = reclaimableBytes();
    return true;
  }

  uint64_t reclaim(
      MemoryPool* /*unused*/,
      uint64_t targetBytes,
      uint64_t /*unused*/,
      Stats& stats) noexcept override {
    std::lock_guard<std::mutex> l(mu_);
    uint64_t reclaimedBytes{0};
    while (!allocations_.empty() &&
           ((targetBytes == 0) || (reclaimedBytes < targetBytes))) {
      free(allocations_.front());
      reclaimedBytes += allocations_.front().size;
      allocations_.pop_front();
    }
    stats.reclaimedBytes += reclaimedBytes;
    return reclaimedBytes;
  }

  void setPool(MemoryPool* pool) {
    VELOX_CHECK_NOT_NULL(pool);
    VELOX_CHECK_NULL(pool_);
    pool_ = pool;
  }

  void addAllocation(void* buffer, size_t size) {
    std::lock_guard<std::mutex> l(mu_);
    allocations_.push_back({buffer, size});
    totalUsedBytes_ += size;
  }

 private:
  struct Allocation {
    void* buffer;
    size_t size;
  };

  void free(const Allocation& allocation) {
    pool_->free(allocation.buffer, allocation.size);
    totalUsedBytes_ -= allocation.size;
    VELOX_CHECK_GE(static_cast<int>(totalUsedBytes_), 0);
  }

  uint64_t reclaimableBytes() const {
    uint64_t sumBytes{0};
    std::lock_guard<std::mutex> l(mu_);
    for (const auto& allocation : allocations_) {
      sumBytes += allocation.size;
    }
    return sumBytes;
  }

  std::atomic<uint64_t>& totalUsedBytes_;
  mutable std::mutex mu_;
  MemoryPool* pool_{nullptr};
  std::deque<Allocation> allocations_;
};

TEST_F(MemoryReclaimerTest, mockReclaim) {
  const int numChildren = 10;
  const int numGrandchildren = 10;
  const int numAllocationsPerLeaf = 10;
  const int allocBytes = 10;
  std::atomic<uint64_t> totalUsedBytes{0};
  auto root = memory::memoryManager()->addRootPool(
      "mockReclaim", kMaxMemory, MemoryReclaimer::create());
  std::vector<std::shared_ptr<MemoryPool>> childPools;
  for (int i = 0; i < numChildren; ++i) {
    auto childPool =
        root->addAggregateChild(std::to_string(i), MemoryReclaimer::create());
    childPools.push_back(childPool);
    for (int j = 0; j < numGrandchildren; ++j) {
      auto grandchildPool = childPool->addLeafChild(
          std::to_string(j),
          true,
          std::make_unique<MockLeafMemoryReclaimer>(totalUsedBytes));
      childPools.push_back(grandchildPool);
      auto* reclaimer =
          static_cast<MockLeafMemoryReclaimer*>(grandchildPool->reclaimer());
      reclaimer->setPool(grandchildPool.get());
      for (int k = 0; k < numAllocationsPerLeaf; ++k) {
        void* buffer = grandchildPool->allocate(allocBytes);
        reclaimer->addAllocation(buffer, allocBytes);
      }
    }
  }
  ASSERT_EQ(
      numGrandchildren * numChildren * numAllocationsPerLeaf * allocBytes,
      totalUsedBytes);
  ASSERT_EQ(root->reclaimableBytes().value(), totalUsedBytes);
  const int numReclaims = 5;
  const int numBytesToReclaim = allocBytes * 3;
  for (int iter = 0; iter < numReclaims; ++iter) {
    const auto reclaimedBytes = root->reclaim(numBytesToReclaim, 0, stats_);
    ASSERT_EQ(reclaimedBytes, numBytesToReclaim);
    ASSERT_EQ(reclaimedBytes, stats_.reclaimedBytes);
    ASSERT_EQ(root->reclaimableBytes().value(), totalUsedBytes);
    stats_.reset();
  }

  ASSERT_EQ(totalUsedBytes, root->reclaimableBytes().value());
  ASSERT_EQ(root->reclaim(allocBytes + 1, 0, stats_), 2 * allocBytes);
  ASSERT_EQ(root->reclaim(allocBytes - 1, 0, stats_), allocBytes);
  ASSERT_EQ(3 * allocBytes, stats_.reclaimedBytes);

  const uint64_t expectedReclaimedBytes = totalUsedBytes;
  ASSERT_EQ(root->reclaim(0, 0, stats_), expectedReclaimedBytes);
  ASSERT_EQ(3 * allocBytes + expectedReclaimedBytes, stats_.reclaimedBytes);
  ASSERT_EQ(totalUsedBytes, 0);
  ASSERT_EQ(root->reclaimableBytes().value(), 0);

  stats_.reset();
  ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
}

TEST_F(MemoryReclaimerTest, mockReclaimMoreThanAvailable) {
  const int numChildren = 10;
  const int numAllocationsPerLeaf = 10;
  const int allocBytes = 100;
  std::atomic<uint64_t> totalUsedBytes{0};
  auto root = memory::memoryManager()->addRootPool(
      "mockReclaimMoreThanAvailable", kMaxMemory, MemoryReclaimer::create());
  std::vector<std::shared_ptr<MemoryPool>> childPools;
  for (int i = 0; i < numChildren; ++i) {
    auto childPool = root->addLeafChild(
        std::to_string(i),
        true,
        std::make_unique<MockLeafMemoryReclaimer>(totalUsedBytes));
    childPools.push_back(childPool);
    auto* reclaimer =
        static_cast<MockLeafMemoryReclaimer*>(childPool->reclaimer());
    reclaimer->setPool(childPool.get());
    for (int j = 0; j < numAllocationsPerLeaf; ++j) {
      void* buffer = childPool->allocate(allocBytes);
      reclaimer->addAllocation(buffer, allocBytes);
    }
  }
  ASSERT_EQ(numChildren * numAllocationsPerLeaf * allocBytes, totalUsedBytes);
  uint64_t reclaimableBytes = root->reclaimableBytes().value();
  ASSERT_EQ(reclaimableBytes, totalUsedBytes);
  const uint64_t expectedReclaimedBytes = totalUsedBytes;
  ASSERT_EQ(
      root->reclaim(totalUsedBytes + 100, 0, stats_), expectedReclaimedBytes);
  ASSERT_EQ(expectedReclaimedBytes, stats_.reclaimedBytes);
  ASSERT_EQ(totalUsedBytes, 0);
  reclaimableBytes = root->reclaimableBytes().value();
  ASSERT_EQ(reclaimableBytes, 0);
  stats_.reset();
  ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
}

TEST_F(MemoryReclaimerTest, orderedReclaim) {
  // Set 1MB unit to avoid memory pool quantized reservation effect.
  const int allocUnitBytes = 1L << 20;
  const int numChildren = 5;
  // The initial allocation units per each child pool.
  const std::vector<int> initAllocUnitsVec = {10, 11, 8, 16, 5};
  ASSERT_EQ(initAllocUnitsVec.size(), numChildren);
  std::atomic<uint64_t> totalUsedBytes{0};
  auto root = memory::memoryManager()->addRootPool(
      "orderedReclaim", kMaxMemory, MemoryReclaimer::create());
  int totalAllocUnits{0};
  std::vector<std::shared_ptr<MemoryPool>> childPools;
  for (int i = 0; i < numChildren; ++i) {
    auto childPool = root->addLeafChild(
        std::to_string(i),
        true,
        std::make_unique<MockLeafMemoryReclaimer>(totalUsedBytes));
    childPools.push_back(childPool);
    auto* reclaimer =
        static_cast<MockLeafMemoryReclaimer*>(childPool->reclaimer());
    reclaimer->setPool(childPool.get());

    const auto initAllocUnit = initAllocUnitsVec[i];
    totalAllocUnits += initAllocUnit;
    for (int j = 0; j < initAllocUnit; ++j) {
      void* buffer = childPool->allocate(allocUnitBytes);
      reclaimer->addAllocation(buffer, allocUnitBytes);
    }
  }

  uint64_t reclaimableBytes{0};
  // 'expectedReclaimableUnits' is the expected allocation unit per each child
  // pool after each round of memory reclaim. And we expect the memory reclaimer
  // always reclaim from the child with most meomry usage.
  auto verify = [&](const std::vector<int>& expectedReclaimableUnits) {
    root->reclaimer()->reclaimableBytes(*root, reclaimableBytes);
    ASSERT_EQ(reclaimableBytes, totalAllocUnits * allocUnitBytes) << "total";
    for (int i = 0; i < numChildren; ++i) {
      auto* reclaimer =
          static_cast<MockLeafMemoryReclaimer*>(childPools[i]->reclaimer());
      reclaimer->reclaimableBytes(*childPools[i], reclaimableBytes);
      ASSERT_EQ(reclaimableBytes, expectedReclaimableUnits[i] * allocUnitBytes)
          << " " << i;
    }
  };
  // No reclaim so far so just expect the initial allocation unit distribution.
  verify(initAllocUnitsVec);

  // Reclaim two units from {10, 11, 8, 16, 5}, and expect to reclaim from 3th
  // child.
  // So expected reclaimable allocation units are {10, 11, 8, *14*, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 2 * allocUnitBytes, 0, stats_),
      2 * allocUnitBytes);
  ASSERT_EQ(2 * allocUnitBytes, stats_.reclaimedBytes);
  totalAllocUnits -= 2;
  verify({10, 11, 8, 14, 5});

  // Reclaim two units from {10, 11, 8, 14, 5}, and expect to reclaim from 3th
  // child.
  // So expected reclaimable allocation units are {10, 11, 8, *12*, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 2 * allocUnitBytes, 0, stats_),
      2 * allocUnitBytes);
  ASSERT_EQ(4 * allocUnitBytes, stats_.reclaimedBytes);
  totalAllocUnits -= 2;
  verify({10, 11, 8, 12, 5});

  // Reclaim eight units from {10, 11, 8, 12, 5}, and expect to reclaim from 3th
  // child.
  // So expected reclaimable allocation units are {10, 11, 8, *4*, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 8 * allocUnitBytes, 0, stats_),
      8 * allocUnitBytes);
  ASSERT_EQ(12 * allocUnitBytes, stats_.reclaimedBytes);
  totalAllocUnits -= 8;
  verify({10, 11, 8, 4, 5});

  // Reclaim two unit from {10, 11, 8, 4, 5}, and expect to reclaim from 1th
  // child.
  // So expected reclaimable allocation gunits are {10, *9*, 8, 4, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 2 * allocUnitBytes, 0, stats_),
      2 * allocUnitBytes);
  totalAllocUnits -= 2;
  verify({10, 9, 8, 4, 5});

  // Reclaim three unit from {10, 9, 8, 4, 5}, and expect to reclaim from 0th
  // child.
  // So expected reclaimable allocation units are {*7*, 9, 8, 4, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 3 * allocUnitBytes, 0, stats_),
      3 * allocUnitBytes);
  totalAllocUnits -= 3;
  verify({7, 9, 8, 4, 5});

  // Reclaim eleven unit from {7, 9, 8, 4, 5}, and expect to reclaim 9 from 1th
  // child and two from 2nd child.
  // So expected reclaimable allocation units are {7, *0*, *6*, 4, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 11 * allocUnitBytes, 0, stats_),
      11 * allocUnitBytes);
  totalAllocUnits -= 11;
  verify({7, 0, 6, 4, 5});

  // Reclaim ten unit from {7, 0, 6, 4, 5}, and expect to reclaim 7 from 0th
  // child and three from 2nd child.
  // So expected reclaimable allocation units are {*0*, 0, *3*, 4, 5}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 10 * allocUnitBytes, 0, stats_),
      10 * allocUnitBytes);
  totalAllocUnits -= 10;
  verify({0, 0, 3, 4, 5});

  // Reclaim ten unit from {0, 0, 3, 4, 5}, and expect to reclaim 5 from 4th
  // child and 4 from 4th child and 1 from 2nd.
  // So expected reclaimable allocation units are {0, 0, 2, *0*, *0*}
  ASSERT_EQ(
      root->reclaimer()->reclaim(root.get(), 10 * allocUnitBytes, 0, stats_),
      10 * allocUnitBytes);
  totalAllocUnits -= 10;
  verify({0, 0, 2, 0, 0});

  // Reclaim all the remaining units and expect all reclaimable bytes got
  // cleared.
  ASSERT_EQ(
      root->reclaimer()->reclaim(
          root.get(), totalAllocUnits * allocUnitBytes, 0, stats_),
      totalAllocUnits * allocUnitBytes);
  totalAllocUnits = 0;
  verify({0, 0, 0, 0, 0});
  stats_.reset();
  ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
}

TEST_F(MemoryReclaimerTest, arbitrationContext) {
  auto root = memory::memoryManager()->addRootPool(
      "arbitrationContext", kMaxMemory, MemoryReclaimer::create());
  ASSERT_FALSE(isSpillMemoryPool(root.get()));
  ASSERT_TRUE(isSpillMemoryPool(spillMemoryPool()));
  auto leafChild1 = root->addLeafChild(spillMemoryPool()->name());
  ASSERT_FALSE(isSpillMemoryPool(leafChild1.get()));
  auto leafChild2 = root->addLeafChild("arbitrationContext");
  ASSERT_FALSE(isSpillMemoryPool(leafChild2.get()));
  ASSERT_TRUE(memoryArbitrationContext() == nullptr);
  {
    ScopedMemoryArbitrationContext arbitrationContext(leafChild1.get());
    ASSERT_TRUE(memoryArbitrationContext() != nullptr);
    ASSERT_EQ(memoryArbitrationContext()->requestor, leafChild1.get());
  }
  ASSERT_TRUE(memoryArbitrationContext() == nullptr);
  {
    ScopedMemoryArbitrationContext arbitrationContext(leafChild2.get());
    ASSERT_TRUE(memoryArbitrationContext() != nullptr);
    ASSERT_EQ(memoryArbitrationContext()->requestor, leafChild2.get());
  }
  ASSERT_TRUE(memoryArbitrationContext() == nullptr);
  std::thread nonAbitrationThread([&]() {
    ASSERT_TRUE(memoryArbitrationContext() == nullptr);
    {
      ScopedMemoryArbitrationContext arbitrationContext(leafChild1.get());
      ASSERT_TRUE(memoryArbitrationContext() != nullptr);
      ASSERT_EQ(memoryArbitrationContext()->requestor, leafChild1.get());
    }
    ASSERT_TRUE(memoryArbitrationContext() == nullptr);
    {
      ScopedMemoryArbitrationContext arbitrationContext(leafChild2.get());
      ASSERT_TRUE(memoryArbitrationContext() != nullptr);
      ASSERT_EQ(memoryArbitrationContext()->requestor, leafChild2.get());
    }
    ASSERT_TRUE(memoryArbitrationContext() == nullptr);
  });
  nonAbitrationThread.join();
  ASSERT_TRUE(memoryArbitrationContext() == nullptr);
}

TEST_F(MemoryReclaimerTest, concurrentRandomMockReclaims) {
  auto root = memory::memoryManager()->addRootPool(
      "concurrentRandomMockReclaims", kMaxMemory, MemoryReclaimer::create());

  std::atomic<uint64_t> totalUsedBytes{0};
  const int numLeafPools = 50;
  std::vector<std::shared_ptr<MemoryPool>> childPools;
  std::vector<std::shared_ptr<MemoryPool>> leafPools;
  for (int i = 0; i < numLeafPools / 5; ++i) {
    childPools.push_back(
        root->addAggregateChild(std::to_string(i), MemoryReclaimer::create()));
    for (int j = 0; j < 5; ++j) {
      auto leafPool = childPools.back()->addLeafChild(
          std::to_string(j),
          true,
          std::make_unique<MockLeafMemoryReclaimer>(totalUsedBytes));
      leafPools.push_back(leafPool);
      auto* reclaimer =
          static_cast<MockLeafMemoryReclaimer*>(leafPool->reclaimer());
      reclaimer->setPool(leafPool.get());
    }
  }

  const int32_t kNumReclaims = 100;
  uint64_t totalReclaimedBytes = 0;
  std::thread reclaimerThread([&]() {
    for (int i = 0; i < kNumReclaims; ++i) {
      const uint64_t oldUsedBytes = totalUsedBytes;
      if (oldUsedBytes == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        continue;
      }
      uint64_t bytesToReclaim = folly::Random::rand64() % oldUsedBytes;
      if (folly::Random::oneIn(10, rng_)) {
        if (folly::Random::oneIn(2, rng_)) {
          bytesToReclaim = totalUsedBytes + 1000;
        } else {
          bytesToReclaim = 0;
        }
      }
      const auto reclaimedBytes = root->reclaim(bytesToReclaim, 0, stats_);
      totalReclaimedBytes += reclaimedBytes;
      if (reclaimedBytes < bytesToReclaim) {
        ASSERT_GT(bytesToReclaim, oldUsedBytes);
      }
      if (bytesToReclaim == 0) {
        ASSERT_GE(reclaimedBytes, oldUsedBytes);
      }
    }
  });

  const int32_t kNumAllocThreads = numLeafPools;
  const int32_t kNumAllocsPerThread = 1'000;
  const int maxAllocSize = 256;

  std::vector<std::thread> allocThreads;
  allocThreads.reserve(kNumAllocThreads);
  for (size_t i = 0; i < kNumAllocThreads; ++i) {
    allocThreads.emplace_back([&, pool = leafPools[i]]() {
      auto* reclaimer =
          static_cast<MockLeafMemoryReclaimer*>(pool->reclaimer());
      for (int32_t op = 0; op < kNumAllocsPerThread; ++op) {
        const int allocSize =
            std::max<int>(1, folly::Random().rand32() % maxAllocSize);
        void* buffer = pool->allocate(allocSize);
        reclaimer->addAllocation(buffer, allocSize);
      }
    });
  }

  for (auto& th : allocThreads) {
    th.join();
  }
  reclaimerThread.join();

  uint64_t reclaimableBytes = root->reclaimableBytes().value();
  ASSERT_EQ(reclaimableBytes, totalUsedBytes);

  root->reclaim(0, 0, stats_);
  ASSERT_EQ(totalReclaimedBytes + reclaimableBytes, stats_.reclaimedBytes);

  reclaimableBytes = root->reclaimableBytes().value();
  ASSERT_EQ(reclaimableBytes, 0);
  ASSERT_EQ(totalUsedBytes, 0);
  stats_.reset();
  ASSERT_EQ(stats_, MemoryReclaimer::Stats{});
}

TEST_F(MemoryArbitrationTest, reclaimerStats) {
  MemoryReclaimer::Stats stats1;
  MemoryReclaimer::Stats stats2;
  ASSERT_EQ(stats1.numNonReclaimableAttempts, 0);
  ASSERT_EQ(stats1, stats2);
  stats1.numNonReclaimableAttempts = 2;
  ASSERT_NE(stats1, stats2);
  stats2.numNonReclaimableAttempts = 3;
  ASSERT_NE(stats1, stats2);
  stats2.reset();
  ASSERT_EQ(stats2.numNonReclaimableAttempts, 0);
  ASSERT_NE(stats1, stats2);
  stats1.reset();
  ASSERT_EQ(stats1, stats2);
}
} // namespace facebook::velox::memory
