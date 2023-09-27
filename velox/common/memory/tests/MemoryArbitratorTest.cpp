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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <deque>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/MemoryReclaimer.h"

using namespace ::testing;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

namespace facebook::velox::memory {

class MemoryArbitrationTest : public testing::Test {};

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
  ASSERT_EQ(
      stats.toString(),
      "STATS[numRequests 2 numSucceeded 0 numAborted 3 numFailures 100 queueTime 230.00ms arbitrationTime 1.02ms reclaimTime 1.00ms shrunkMemory 95.37MB reclaimedMemory 9.77KB maxCapacity 0B freeCapacity 0B]");
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
    auto allocator = std::make_shared<MallocAllocator>(8L << 20);
    MemoryManager manager{
        {.capacity = (int64_t)allocator->capacity(),
         .queryMemoryCapacity = 4L << 20,
         .allocator = allocator.get()}};
    auto rootPool = manager.addRootPool("root-1", 8L << 20);
    auto leafPool = rootPool->addLeafChild("leaf-1.0");
    void* buffer;
    ASSERT_NO_THROW({
      buffer = leafPool->allocate(7L << 20);
      leafPool->free(buffer, 7L << 20);
    });
  }
  {
    // Reserved memory is enforced when SharedMemoryArbitrator is used.
    SharedArbitrator::registerFactory();
    auto allocator = std::make_shared<MallocAllocator>(8L << 20);
    MemoryManager manager{
        {.capacity = (int64_t)allocator->capacity(),
         .queryMemoryCapacity = 4L << 20,
         .allocator = allocator.get(),
         .arbitratorKind = "SHARED"}};
    auto rootPool =
        manager.addRootPool("root-1", 8L << 20, MemoryReclaimer::create());
    auto leafPool = rootPool->addLeafChild("leaf-1.0");
    void* buffer;
    VELOX_ASSERT_THROW(
        buffer = leafPool->allocate(7L << 20),
        "Exceeded memory pool cap of 4.00MB");
    ASSERT_NO_THROW(buffer = leafPool->allocate(4L << 20));
    leafPool->free(buffer, 4L << 20);
    SharedArbitrator::unregisterFactory();
  }
}

TEST_F(MemoryArbitrationTest, arbitratorStats) {
  const MemoryArbitrator::Stats emptyStats;
  ASSERT_TRUE(emptyStats.empty());
  const MemoryArbitrator::Stats anchorStats(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5);
  ASSERT_FALSE(anchorStats.empty());
  const MemoryArbitrator::Stats largeStats(8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
  ASSERT_FALSE(largeStats.empty());
  ASSERT_TRUE(!(anchorStats == largeStats));
  ASSERT_TRUE(anchorStats != largeStats);
  ASSERT_TRUE(anchorStats < largeStats);
  ASSERT_TRUE(!(anchorStats > largeStats));
  ASSERT_TRUE(anchorStats <= largeStats);
  ASSERT_TRUE(!(anchorStats >= largeStats));
  const auto delta = largeStats - anchorStats;
  ASSERT_EQ(delta, MemoryArbitrator::Stats(3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3));

  const MemoryArbitrator::Stats smallStats(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  ASSERT_TRUE(!(anchorStats == smallStats));
  ASSERT_TRUE(anchorStats != smallStats);
  ASSERT_TRUE(!(anchorStats < smallStats));
  ASSERT_TRUE(anchorStats > smallStats);
  ASSERT_TRUE(!(anchorStats <= smallStats));
  ASSERT_TRUE(anchorStats >= smallStats);

  const MemoryArbitrator::Stats invalidStats(2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8);
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
             .memoryPoolInitCapacity = config.memoryPoolInitCapacity,
             .memoryPoolTransferCapacity = config.memoryPoolTransferCapacity}) {
  }

  void reserveMemory(MemoryPool* pool, uint64_t bytes) override {
    VELOX_NYI();
  }

  void releaseMemory(MemoryPool* pool) override {
    VELOX_NYI();
  }

  std::string kind() const override {
    return "USER";
  }

  bool growMemory(
      MemoryPool* pool,
      const std::vector<std::shared_ptr<MemoryPool>>& candidatePools,
      uint64_t targetBytes) override {
    VELOX_NYI();
  }

  uint64_t shrinkMemory(
      const std::vector<std::shared_ptr<MemoryPool>>& pools,
      uint64_t targetBytes) override {
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
      "Arbitrator factory for kind USER already registered");
}

TEST_F(MemoryArbitratorFactoryTest, create) {
  MemoryArbitrator::Config config;
  config.kind = kind_;
  auto arbitrator = MemoryArbitrator::create(config);
  ASSERT_EQ(kind_, arbitrator->kind());
}
} // namespace facebook::velox::memory
