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
#include "velox/common/memory/AllocationPool.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;

class AllocationPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    memory::MemoryManager::Options options;
    options.allocatorCapacity = 8L << 30;
    manager_ = std::make_shared<memory::MemoryManager>(options);

    root_ = manager_->addRootPool("allocationPoolTestRoot");
    pool_ = root_->addLeafChild("leaf");

    TestValue::enable();
  }

  // Writes a byte at pointer so we see RSS change.
  void setByte(void* ptr) {
    *reinterpret_cast<char*>(ptr) = 1;
  }

  std::shared_ptr<memory::MemoryManager> manager_;
  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(AllocationPoolTest, hugePages) {
  constexpr int64_t kHugePageSize = memory::AllocationTraits::kHugePageSize;
  auto allocationPool = std::make_unique<memory::AllocationPool>(pool_.get());
  allocationPool->setHugePageThreshold(128 << 10);
  int32_t counter = 0;
  for (;;) {
    allocationPool->newRun(32 << 10);
    // Initial allocations round up to 64K
    EXPECT_EQ(1, allocationPool->numRanges());
    EXPECT_EQ(allocationPool->testingFreeAddressableBytes(), 64 << 10);
    allocationPool->newRun(64 << 10);
    EXPECT_LE(128 << 10, pool_->usedBytes());
    allocationPool->allocateFixed(64 << 10);
    // Now at end of second 64K range, next will go to huge pages.
    setByte(allocationPool->allocateFixed(11));
    EXPECT_LE((2 << 20) - 11, allocationPool->testingFreeAddressableBytes());
    // The first 2MB of the hugepage run are marked reserved.
    EXPECT_LE((2048 + 128) << 10, pool_->usedBytes());

    // The next allocation starts reserves the next 2MB of the mmapped range.
    setByte(allocationPool->allocateFixed(2 << 20));
    EXPECT_LE((4096 + 128) << 10, pool_->usedBytes());

    // Allocate the rest.
    allocationPool->allocateFixed(
        allocationPool->testingFreeAddressableBytes());

    // We expect 3 ranges, 2 small and one large.
    EXPECT_EQ(3, allocationPool->numRanges());

    // We allocate more, expect a larger mmap.
    allocationPool->allocateFixed(1);

    // The first is at least 15 huge pages. The next is at least 31. The mmaps
    // may have unused addresses at either end, so count one huge page less than
    // the nominal size.
    EXPECT_LE((62 << 20) - 1, allocationPool->testingFreeAddressableBytes());

    // We make a 5GB extra large allocation.
    allocationPool->allocateFixed(5UL << 30);
    EXPECT_EQ(5, allocationPool->numRanges());

    // 5G is an even multiple of huge page, no free space at end. But it can be
    // the mmap happens to start at 2MB boundary so we get another 2MB.
    EXPECT_GE(kHugePageSize, allocationPool->testingFreeAddressableBytes());

    EXPECT_LE(
        (5UL << 30) + (31 << 20) + (128 << 10),
        allocationPool->allocatedBytes());
    EXPECT_LE((5UL << 30) + (31 << 20) + (128 << 10), pool_->usedBytes());

    if (counter++ >= 1) {
      break;
    }

    // Repeat the above after a clear().
    allocationPool->clear();
    // Should be empty after clear().
    EXPECT_EQ(0, pool_->usedBytes());
  }
  allocationPool.reset();
  // Should be empty after destruction.
  EXPECT_EQ(0, pool_->usedBytes());
}

// This test relies on TestValue, so needs to be run in debug mode.
DEBUG_ONLY_TEST_F(AllocationPoolTest, oomCleanUp) {
  // Test that when an OOM happens while growing an allocation in the
  // AllocationPool, the AllocationPool is still in a valid state.
  auto test = [&](int32_t alignment) {
    auto allocationPool = std::make_unique<memory::AllocationPool>(pool_.get());
    // Ensure we're beyond the huge page threshod.
    allocationPool->setHugePageThreshold(32 << 10);
    allocationPool->allocateFixed(32 << 10, alignment);

    // Allocate some memory so we have a large allocation.
    allocationPool->allocateFixed(1 << 20, alignment);

    {
      static const std::string kErrorMessage = "Simulate OOM for testing.";
      // Trigger an OOM.
      SCOPED_TESTVALUE_SET(
          "facebook::velox::memory::MemoryPoolImpl::reserveThreadSafe",
          std::function<void(memory::MemoryPool*)>(
              [&](memory::MemoryPool* /*unused*/) {
                VELOX_FAIL(kErrorMessage);
              }));
      VELOX_ASSERT_THROW(
          allocationPool->allocateFixed(
              allocationPool->testingFreeAddressableBytes(), alignment),
          kErrorMessage);
    }

    // Ensure the last range in the pool is still consistent, e.g.
    // currentOffset_ isn't pointing into unallocated memory.
    ASSERT_EQ(
        allocationPool->rangeAt(allocationPool->numRanges() - 1).size(),
        1 << 20);
  };

  // Test with an alignment of 1.
  test(1);
  // Test with an alignment of 2 (this goes through a different code path in
  // allocateFixed).
  test(2);
}

TEST_F(AllocationPoolTest, clone) {
  auto source = std::make_unique<memory::AllocationPool>(pool_.get());
  // Force a switch to huge-page runs early so the payload spans both small and
  // large allocation runs, exercising both branches of clone().
  source->setHugePageThreshold(128 << 10);

  // Allocate a long run of fixed chunks, stamping each with its index so we can
  // check both content fidelity and address translation after the clone.
  constexpr int32_t kChunkBytes = 512;
  constexpr int32_t kNumChunks = 20'000; // ~10MB, well past the threshold.
  std::vector<char*> chunks;
  chunks.reserve(kNumChunks);
  for (int64_t i = 0; i < kNumChunks; ++i) {
    char* chunk = source->allocateFixed(kChunkBytes, 8);
    *reinterpret_cast<int64_t*>(chunk) = i;
    chunks.push_back(chunk);
  }
  const auto numRanges = source->numRanges();
  ASSERT_GT(numRanges, 1) << "test must span multiple runs";

  auto destPool = root_->addLeafChild("clone-dest");
  memory::AllocationPool dest(destPool.get());
  const auto relocations = dest.clone(*source);

  // The destination mirrors the source's run structure.
  EXPECT_EQ(dest.numRanges(), numRanges);
  EXPECT_EQ(dest.allocatedBytes(), source->allocatedBytes());

  // Relocations are sorted by source address and disjoint.
  for (size_t i = 1; i < relocations.size(); ++i) {
    EXPECT_LE(
        relocations[i - 1].sourceBegin + relocations[i - 1].numBytes,
        relocations[i].sourceBegin);
  }

  // Every chunk maps into the destination through exactly one relocation, lands
  // at a new address, and keeps its content.
  auto translate = [&](char* from) -> char* {
    for (const auto& run : relocations) {
      if (from >= run.sourceBegin && from < run.sourceBegin + run.numBytes) {
        return from + run.delta;
      }
    }
    return nullptr;
  };
  for (int64_t i = 0; i < kNumChunks; ++i) {
    char* moved = translate(chunks[i]);
    ASSERT_NE(moved, nullptr)
        << "chunk " << i << " not covered by a relocation";
    EXPECT_NE(moved, chunks[i]);
    EXPECT_EQ(*reinterpret_cast<int64_t*>(moved), i);
  }
}

TEST_F(AllocationPoolTest, cloneEmptyAndPreconditions) {
  memory::AllocationPool empty(pool_.get());

  auto destPool = root_->addLeafChild("clone-empty-dest");
  memory::AllocationPool dest(destPool.get());
  EXPECT_TRUE(dest.clone(empty).empty());
  EXPECT_EQ(dest.numRanges(), 0);

  // A non-empty destination is rejected.
  dest.allocateFixed(64, 8);
  VELOX_ASSERT_THROW(
      dest.clone(empty), "clone requires an empty destination pool");
}
