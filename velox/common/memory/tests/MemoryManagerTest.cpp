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
    const auto& root = manager.getRoot();

    ASSERT_EQ(0, root.getChildCount());
    ASSERT_EQ(0, root.getCurrentBytes());
    ASSERT_EQ(std::numeric_limits<int64_t>::max(), manager.getMemoryQuota());
    ASSERT_EQ(0, manager.getTotalBytes());
    ASSERT_EQ(manager.alignment(), MemoryAllocator::kMaxAlignment);
  }
  {
    MemoryManager manager{{.capacity = 8L * 1024 * 1024}};
    const auto& root = manager.getRoot();

    ASSERT_EQ(0, root.getChildCount());
    ASSERT_EQ(0, root.getCurrentBytes());
    ASSERT_EQ(8L * 1024 * 1024, manager.getMemoryQuota());
    ASSERT_EQ(0, manager.getTotalBytes());
  }
  {
    MemoryManager manager{{.alignment = 0, .capacity = 8L * 1024 * 1024}};
    const auto& root = manager.getRoot();

    ASSERT_EQ(manager.alignment(), MemoryAllocator::kMinAlignment);
    // TODO: replace with root pool memory tracker quota check.
    // ASSERT_EQ(8L * 1024 * 1024, root.cap());
    ASSERT_EQ(0, root.getChildCount());
    ASSERT_EQ(0, root.getCurrentBytes());
    ASSERT_EQ(8L * 1024 * 1024, manager.getMemoryQuota());
    ASSERT_EQ(0, manager.getTotalBytes());
  }
  { ASSERT_ANY_THROW(MemoryManager manager{{.capacity = -1}}); }
}

// TODO: when run sequentially, e.g. `buck run dwio/memory/...`, this has side
// effects for other tests using process singleton memory manager. Might need to
// use folly::Singleton for isolation by tag.
TEST(MemoryManagerTest, GlobalMemoryManager) {
  auto& manager = MemoryManager::getInstance();
  auto& managerII = MemoryManager::getInstance();

  auto& root = manager.getRoot();
  auto child = root.addChild("some_child");
  ASSERT_EQ(1, root.getChildCount());

  auto& rootII = managerII.getRoot();
  ASSERT_EQ(1, rootII.getChildCount());
  std::vector<MemoryPool*> pools{};
  rootII.visitChildren(
      [&pools](MemoryPool* child) { pools.emplace_back(child); });
  ASSERT_EQ(1, pools.size());
  auto& pool = *pools.back();
  ASSERT_EQ("some_child", pool.name());
}

TEST(MemoryManagerTest, Reserve) {
  {
    MemoryManager manager{};
    ASSERT_TRUE(manager.reserve(0));
    ASSERT_EQ(0, manager.getTotalBytes());
    manager.release(0);
    ASSERT_TRUE(manager.reserve(42));
    ASSERT_EQ(42, manager.getTotalBytes());
    manager.release(42);
    ASSERT_TRUE(manager.reserve(std::numeric_limits<int64_t>::max()));
    ASSERT_EQ(std::numeric_limits<int64_t>::max(), manager.getTotalBytes());
    manager.release(std::numeric_limits<int64_t>::max());
    ASSERT_EQ(0, manager.getTotalBytes());
  }
  {
    MemoryManager manager{{.capacity = 42}};
    ASSERT_TRUE(manager.reserve(1));
    ASSERT_TRUE(manager.reserve(1));
    ASSERT_TRUE(manager.reserve(2));
    ASSERT_TRUE(manager.reserve(3));
    ASSERT_TRUE(manager.reserve(5));
    ASSERT_TRUE(manager.reserve(8));
    ASSERT_TRUE(manager.reserve(13));
    ASSERT_FALSE(manager.reserve(21));
    ASSERT_FALSE(manager.reserve(1));
    ASSERT_FALSE(manager.reserve(2));
    ASSERT_FALSE(manager.reserve(3));
    manager.release(20);
    ASSERT_TRUE(manager.reserve(1));
    ASSERT_FALSE(manager.reserve(2));
    manager.release(manager.getTotalBytes());
    ASSERT_EQ(manager.getTotalBytes(), 0);
  }
}

TEST(MemoryManagerTest, GlobalMemoryManagerQuota) {
  auto& manager = MemoryManager::getInstance();
  ASSERT_THROW(
      MemoryManager::getInstance({.capacity = 42}, true),
      velox::VeloxUserError);

  auto& coercedManager = MemoryManager::getInstance({.capacity = 42});
  ASSERT_EQ(manager.getMemoryQuota(), coercedManager.getMemoryQuota());
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
        manager.getRoot().getAlignment(),
        std::max(testData.alignment, MemoryAllocator::kMinAlignment));
  }
}
} // namespace memory
} // namespace velox
} // namespace facebook
