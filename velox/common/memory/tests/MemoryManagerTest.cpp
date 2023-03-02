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

#include "velox/common/base/VeloxException.h"
#include "velox/common/memory/Memory.h"

using namespace ::testing;

namespace facebook {
namespace velox {
namespace memory {

TEST(MemoryManagerTest, Ctor) {
  {
    MemoryManager manager{};
    ASSERT_EQ(manager.numPools(), 0);
    ASSERT_EQ(manager.memoryQuota(), std::numeric_limits<int64_t>::max());
    ASSERT_EQ(manager.alignment(), MemoryAllocator::kMaxAlignment);
  }
  {
    MemoryManager manager{{.capacity = 8L * 1024 * 1024}};

    ASSERT_EQ(manager.numPools(), 0);
    ASSERT_EQ(manager.memoryQuota(), 8L * 1024 * 1024);
  }
  {
    MemoryManager manager{{.alignment = 0, .capacity = 8L * 1024 * 1024}};

    ASSERT_EQ(manager.alignment(), MemoryAllocator::kMinAlignment);
    ASSERT_EQ(manager.numPools(), 0);
    ASSERT_EQ(manager.memoryQuota(), 8L * 1024 * 1024);
  }
  { ASSERT_ANY_THROW(MemoryManager manager{{.capacity = -1}}); }
}

// TODO: when run sequentially, e.g. `buck run dwio/memory/...`, this has side
// effects for other tests using process singleton memory manager. Might need to
// use folly::Singleton for isolation by tag.
TEST(MemoryManagerTest, GlobalMemoryManager) {
  auto& managerI = MemoryManager::getInstance();
  auto& managerII = MemoryManager::getInstance();

  const std::string poolName("GlobalMemoryManager");
  auto pool = managerI.getPool(poolName);
  ASSERT_EQ(managerI.numPools(), 1);

  ASSERT_EQ(managerII.numPools(), 1);
  {
    std::vector<MemoryPool*> pools;
    managerII.testingVisitPools(
        [&](MemoryPool* pool) { pools.emplace_back(pool); });
    ASSERT_EQ(pools.size(), 1);
    auto& pool = *pools.back();
    ASSERT_EQ(pools.back()->name(), poolName);
  }

  {
    std::vector<MemoryPool*> pools;
    managerI.testingVisitPools(
        [&](MemoryPool* pool) { pools.emplace_back(pool); });
    ASSERT_EQ(pools.size(), 1);
    auto& pool = *pools.back();
    ASSERT_EQ(pools.back()->name(), poolName);
  }
}

TEST(MemoryManagerTest, GlobalMemoryManagerQuota) {
  auto& manager = MemoryManager::getInstance();
  ASSERT_THROW(
      MemoryManager::getInstance({.capacity = 42}, true),
      velox::VeloxUserError);

  auto& coercedManager = MemoryManager::getInstance({.capacity = 42});
  ASSERT_EQ(manager.memoryQuota(), coercedManager.memoryQuota());
}

TEST(MemoryManagerTest, alignmentOptionCheck) {
  struct {
    uint16_t alignment;
    bool expectedSuccess;

    std::string debugString() const {
      return fmt::format(
          "alignment:{}, expectedSuccess:{}", alignment, expectedSuccess);
    }
  } testSettings[] = {
      {0, true},
      {MemoryAllocator::kMinAlignment - 1, true},
      {MemoryAllocator::kMinAlignment, true},
      {MemoryAllocator::kMinAlignment * 2, true},
      {MemoryAllocator::kMinAlignment + 1, false},
      {MemoryAllocator::kMaxAlignment - 1, false},
      {MemoryAllocator::kMaxAlignment, true},
      {MemoryAllocator::kMaxAlignment + 1, false},
      {MemoryAllocator::kMaxAlignment * 2, false}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    IMemoryManager::Options options;
    options.alignment = testData.alignment;
    if (!testData.expectedSuccess) {
      ASSERT_THROW(MemoryManager{options}, VeloxRuntimeError);
      continue;
    }
    MemoryManager manager{options};
    ASSERT_EQ(
        manager.alignment(),
        std::max(testData.alignment, MemoryAllocator::kMinAlignment));
    ASSERT_EQ(
        manager.getPool()->alignment(),
        std::max(testData.alignment, MemoryAllocator::kMinAlignment));
  }
}
} // namespace memory
} // namespace velox
} // namespace facebook
