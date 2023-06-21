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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/testutil/TestValue.h"

DECLARE_bool(velox_memory_leak_check_enabled);
DECLARE_bool(velox_memory_debug_mode_enabled);
DECLARE_int32(velox_memory_num_shared_leaf_pools);

using namespace ::testing;
using namespace facebook::velox::cache;
using namespace facebook::velox::common::testutil;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

namespace facebook {
namespace velox {
namespace memory {

struct TestParam {
  bool useMmap;
  bool useCache;
  bool threadSafe;

  TestParam(bool _useMmap, bool _useCache, bool _threadSafe)
      : useMmap(_useMmap), useCache(_useCache), threadSafe(_threadSafe) {}

  std::string toString() const {
    return fmt::format(
        "useMmap{} useCache{} threadSafe{}", useMmap, useCache, threadSafe);
  }
};

class MemoryPoolTest : public testing::TestWithParam<TestParam> {
 public:
  static const std::vector<TestParam> getTestParams() {
    std::vector<TestParam> params;
    params.push_back({true, true, false});
    params.push_back({true, false, false});
    params.push_back({false, true, false});
    params.push_back({false, false, false});
    params.push_back({true, true, true});
    params.push_back({true, false, true});
    params.push_back({false, true, true});
    params.push_back({false, false, true});
    return params;
  }

 protected:
  static void SetUpTestCase() {
    FLAGS_velox_memory_leak_check_enabled = true;
    FLAGS_velox_memory_debug_mode_enabled = true;
    TestValue::enable();
  }

  MemoryPoolTest()
      : useMmap_(GetParam().useMmap),
        useCache_(GetParam().useCache),
        isLeafThreadSafe_(GetParam().threadSafe) {}

  void SetUp() override {
    // For duration of the test, make a local MmapAllocator that will not be
    // seen by any other test.
    const uint64_t kCapacity = 8UL << 30;
    if (useMmap_) {
      MmapAllocator::Options opts{8UL << 30};
      allocator_ = std::make_shared<MmapAllocator>(opts);
      if (useCache_) {
        cache_ =
            std::make_shared<AsyncDataCache>(allocator_, kCapacity, nullptr);
        MemoryAllocator::setDefaultInstance(cache_.get());
      } else {
        MemoryAllocator::setDefaultInstance(allocator_.get());
      }
    } else {
      allocator_ = MemoryAllocator::createDefaultInstance();
      if (useCache_) {
        cache_ =
            std::make_shared<AsyncDataCache>(allocator_, kCapacity, nullptr);
        MemoryAllocator::setDefaultInstance(cache_.get());
      } else {
        MemoryAllocator::setDefaultInstance(allocator_.get());
      }
    }
    const auto seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
    LOG(INFO) << "Random seed: " << seed;
  }

  void TearDown() override {
    allocator_->testingClearFailureInjection();
    MmapAllocator::setDefaultInstance(nullptr);
  }

  void reset() {
    TearDown();
    SetUp();
  }

  std::shared_ptr<IMemoryManager> getMemoryManager(int64_t quota) {
    return std::make_shared<MemoryManager>(
        IMemoryManager::Options{.capacity = quota});
  }

  std::shared_ptr<IMemoryManager> getMemoryManager(
      const IMemoryManager::Options& options) {
    return std::make_shared<MemoryManager>(options);
  }

  const int32_t maxMallocBytes_ = 3072;
  const bool useMmap_;
  const bool useCache_;
  const bool isLeafThreadSafe_;
  folly::Random::DefaultGenerator rng_;
  std::shared_ptr<MemoryAllocator> allocator_;
  std::shared_ptr<AsyncDataCache> cache_;
};

// Class used to test operations on MemoryPool.
class MemoryPoolTester {
 public:
  MemoryPoolTester(int32_t id, int64_t maxMemory, memory::MemoryPool& pool)
      : id_(id), maxMemory_(maxMemory), pool_(pool) {}

  ~MemoryPoolTester() {
    for (auto& allocation : contiguousAllocations_) {
      pool_.freeContiguous(allocation);
    }
    for (auto& allocation : nonContiguiusAllocations_) {
      pool_.freeNonContiguous(allocation);
    }
    for (auto& bufferEntry : allocBuffers_) {
      pool_.free(bufferEntry.first, bufferEntry.second);
    }
    if (reservedBytes_ != 0) {
      pool_.release();
    }
  }

  void run() {
    const int32_t op = folly::Random().rand32() % 4;
    switch (op) {
      case 0: {
        // Allocate buffers.
        if ((folly::Random().rand32() % 2) && !allocBuffers_.empty()) {
          pool_.free(allocBuffers_.back().first, allocBuffers_.back().second);
          allocBuffers_.pop_back();
        } else {
          const int64_t allocateBytes =
              folly::Random().rand32() % (maxMemory_ / 64);
          try {
            void* buffer = pool_.allocate(allocateBytes);
            VELOX_CHECK_NOT_NULL(buffer);
            allocBuffers_.push_back({buffer, allocateBytes});
          } catch (VeloxException& e) {
            // Ignore memory limit exception.
            ASSERT_TRUE(e.message().find("Negative") == std::string::npos);
            return;
          }
        }
        break;
      }
      case 2: {
        // Contiguous allocations.
        if ((folly::Random().rand32() % 2) && !contiguousAllocations_.empty()) {
          pool_.freeContiguous(contiguousAllocations_.back());
          contiguousAllocations_.pop_back();
        } else {
          int64_t allocatePages = std::max<int64_t>(
              1,
              folly::Random().rand32() % Allocation::PageRun::kMaxPagesInRun);
          ContiguousAllocation allocation;
          if (folly::Random().oneIn(2) && !contiguousAllocations_.empty()) {
            allocation = std::move(contiguousAllocations_.back());
            contiguousAllocations_.pop_back();
            if (folly::Random().oneIn(2)) {
              allocatePages = std::max<int64_t>(1, allocation.numPages() - 4);
            }
          }
          try {
            pool_.allocateContiguous(allocatePages, allocation);
            contiguousAllocations_.push_back(std::move(allocation));
          } catch (VeloxException& e) {
            // Ignore memory limit exception.
            ASSERT_TRUE(e.message().find("Negative") == std::string::npos);
            return;
          }
        }
        break;
      }
      case 1: {
        // Non-contiguous allocations.
        if ((folly::Random().rand32() % 2) &&
            !nonContiguiusAllocations_.empty()) {
          pool_.freeNonContiguous(nonContiguiusAllocations_.back());
          nonContiguiusAllocations_.pop_back();
        } else {
          const int64_t allocatePages = std::max<int64_t>(
              1,
              folly::Random().rand32() % Allocation::PageRun::kMaxPagesInRun);
          Allocation allocation;
          try {
            pool_.allocateNonContiguous(allocatePages, allocation);
            nonContiguiusAllocations_.push_back(std::move(allocation));
          } catch (VeloxException& e) {
            // Ignore memory limit exception.
            ASSERT_TRUE(e.message().find("Negative") == std::string::npos);
            return;
          }
        }
        break;
      }
      case 3: {
        // maybe reserve.
        if (reservedBytes_ == 0) {
          const uint64_t reservedBytes =
              folly::Random().rand32() % (maxMemory_ / 32);
          if (pool_.maybeReserve(reservedBytes)) {
            reservedBytes_ = reservedBytes;
          }
        } else {
          pool_.release();
          reservedBytes_ = 0;
        }
        break;
      }
    }
  }

 private:
  const int32_t id_;
  const int64_t maxMemory_;
  memory::MemoryPool& pool_;
  uint64_t reservedBytes_{0};
  std::vector<ContiguousAllocation> contiguousAllocations_;
  std::vector<Allocation> nonContiguiusAllocations_;
  std::vector<std::pair<void*, uint64_t>> allocBuffers_;
};

TEST_P(MemoryPoolTest, concurrentUpdateToSharedPools) {
  // TODO(jtan6): Have a followup PR that fixes AsyncDataCache not freeing 'out'
  FLAGS_velox_memory_debug_mode_enabled = false;
  // under some conditions bug.
  constexpr int64_t kMaxMemory = 10 * GB;
  MemoryManager manager{{.capacity = kMaxMemory}};
  const int32_t kNumThreads = FLAGS_velox_memory_num_shared_leaf_pools;

  folly::Random::DefaultGenerator rng;
  rng.seed(1234);

  const int32_t kNumOpsPerThread = 1'000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      auto& sharedPool = manager.deprecatedSharedLeafPool();
      MemoryPoolTester tester(i, kMaxMemory, sharedPool);
      for (int32_t iter = 0; iter < kNumOpsPerThread; ++iter) {
        tester.run();
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  for (auto pool : manager.testingSharedLeafPools()) {
    EXPECT_EQ(pool->currentBytes(), 0);
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    MemoryPoolTestSuite,
    MemoryPoolTest,
    testing::ValuesIn(MemoryPoolTest::getTestParams()));

} // namespace memory
} // namespace velox
} // namespace facebook
