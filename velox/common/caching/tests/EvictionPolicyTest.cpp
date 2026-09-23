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

#include "velox/common/caching/EvictionPolicy.h"

#include "velox/common/caching/ApproxLrfuEvictionPolicy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/tests/CacheTestUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"

using namespace facebook::velox;
using namespace facebook::velox::cache;

class EvictionPolicyTest : public ::testing::Test {
 protected:
  static constexpr int32_t kEntrySize = 4096;

  void SetUp() override {
    memory::MemoryManager::Options options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = 1L << 28;
    options.arbitratorCapacity = 1L << 28;
    options.trackDefaultUsage = true;
    manager_ = std::make_unique<memory::MemoryManager>(options);
    allocator_ = static_cast<memory::MmapAllocator*>(manager_->allocator());
    auto policy = std::make_unique<ApproxLrfuEvictionPolicy>();
    policy_ = policy.get();
    AsyncDataCache::Options cacheOptions;
    cacheOptions.numShards = 1;
    cache_ = AsyncDataCache::create(
        allocator_, nullptr, cacheOptions, std::move(policy));
    fileName_ = StringIdLease(fileIds(), "eviction_policy_test_file");
  }

  void TearDown() override {
    pin_.clear();
    if (cache_) {
      cache_->shutdown();
    }
    cache_.reset();
    fileName_ = StringIdLease{};
    fileIds().testingReset();
  }

  AsyncDataCacheEntry* addEntry(
      uint64_t offset,
      AccessTime lastUse,
      int32_t numUses,
      bool leavePinned = false) {
    RawFileCacheKey key{fileName_.id(), offset};
    folly::SemiFuture<bool> wait(false);
    auto pin =
        cache_->findOrCreate(key, kEntrySize, /*contiguous=*/false, &wait);
    VELOX_CHECK(!pin.empty());
    auto* entry = pin.checkedEntry();
    VELOX_CHECK(entry->isExclusive());
    entry->setExclusiveToShared(/*ssdSavable=*/false);
    test::AsyncDataCacheEntryTestHelper(entry).setAccessStats(lastUse, numUses);
    if (leavePinned) {
      pin_.push_back(std::move(pin));
    }
    return entry;
  }

  test::CacheShardTestHelper shardHelper() {
    return test::CacheShardTestHelper(
        test::AsyncDataCacheTestHelper(cache_.get()).shard(0));
  }

  ApproxLrfuShardState& lrfuState() {
    return const_cast<ApproxLrfuShardState&>(
        static_cast<const ApproxLrfuShardState&>(*shardHelper().policyState()));
  }

  std::unique_ptr<EvictionCandidateCursor> makeCursor(
      uint32_t& clockHand,
      uint32_t& eventCounter,
      bool evictAllUnpinned) {
    return policy_->createEvictionCursorLocked(
        lrfuState(),
        shardHelper().entries(),
        clockHand,
        eventCounter,
        evictAllUnpinned);
  }

  static std::vector<AsyncDataCacheEntry*> drainCursor(
      EvictionCandidateCursor& cursor) {
    std::vector<AsyncDataCacheEntry*> yielded;
    while (true) {
      auto candidate = cursor.next();
      if (candidate.entry == nullptr) {
        break;
      }
      yielded.push_back(candidate.entry);
    }
    return yielded;
  }

  std::unique_ptr<memory::MemoryManager> manager_;
  memory::MmapAllocator* allocator_{nullptr};
  std::shared_ptr<AsyncDataCache> cache_;
  ApproxLrfuEvictionPolicy* policy_{nullptr};
  StringIdLease fileName_;
  std::vector<CachePin> pin_;
};

TEST_F(EvictionPolicyTest, cursorEvictAllUnpinnedYieldsAllUnpinnedEntries) {
  constexpr int32_t kNumUnpinned = 15;
  constexpr int32_t kNumPinned = 5;
  const auto now = accessTime();
  std::vector<AsyncDataCacheEntry*> expected;
  for (int32_t i = 0; i < kNumUnpinned + kNumPinned; ++i) {
    const bool pinned =
        (i % 4 == 0) && (static_cast<int32_t>(pin_.size()) < kNumPinned);
    auto* entry = addEntry(
        /*offset=*/i * kEntrySize, now, /*numUses=*/0, /*leavePinned=*/pinned);
    if (!pinned) {
      expected.push_back(entry);
    }
  }

  uint32_t clockHand = 0;
  uint32_t eventCounter = 0;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/true);
  EXPECT_THAT(
      drainCursor(*cursor), testing::UnorderedElementsAreArray(expected));
}

TEST_F(EvictionPolicyTest, cursorSkipsPinnedInThresholdMode) {
  constexpr int32_t kNumEntries = 10;
  const auto now = accessTime();
  std::vector<AsyncDataCacheEntry*> expected;
  for (int32_t i = 0; i < kNumEntries; ++i) {
    const bool pinned = (i == 2 || i == 5 || i == 8);
    auto* entry = addEntry(
        /*offset=*/i * kEntrySize,
        now - 1'000'000,
        /*numUses=*/0,
        /*leavePinned=*/pinned);
    if (!pinned) {
      expected.push_back(entry);
    }
  }
  lrfuState().evictionThreshold = 1;

  uint32_t clockHand = 0;
  uint32_t eventCounter = 0;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/false);
  EXPECT_THAT(
      drainCursor(*cursor), testing::UnorderedElementsAreArray(expected));
}

TEST_F(EvictionPolicyTest, cursorExhaustsAfterOnePass) {
  constexpr int32_t kNumEntries = 5;
  const auto now = accessTime();
  for (int32_t i = 0; i < kNumEntries; ++i) {
    addEntry(/*offset=*/i * kEntrySize, now - 1'000'000, /*numUses=*/0);
  }

  uint32_t clockHand = 0;
  uint32_t eventCounter = 0;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/true);
  int32_t yielded = 0;
  while (cursor->next().entry != nullptr) {
    ++yielded;
  }
  EXPECT_EQ(yielded, kNumEntries);
  EXPECT_EQ(cursor->next().entry, nullptr);
}

TEST_F(EvictionPolicyTest, emptyKeyEntriesAlwaysYielded) {
  constexpr int32_t kNumEntries = 5;
  const auto now = accessTime();
  for (int32_t i = 0; i < kNumEntries; ++i) {
    addEntry(/*offset=*/i * kEntrySize, now, /*numUses=*/1'000'000);
  }
  lrfuState().evictionThreshold = std::numeric_limits<int32_t>::max() - 1;
  auto* emptyKeyEntry = shardHelper().entries()[2].get();
  test::AsyncDataCacheEntryTestHelper(emptyKeyEntry).clearKey();

  uint32_t clockHand = 0;
  uint32_t eventCounter = 0;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/false);
  EXPECT_THAT(drainCursor(*cursor), testing::Contains(emptyKeyEntry));
}

TEST_F(EvictionPolicyTest, recalibrationOnEventCounter) {
  constexpr int32_t kNumEntries = 20;
  const auto now = accessTime();
  for (int32_t i = 0; i < kNumEntries; ++i) {
    addEntry(
        /*offset=*/i * kEntrySize,
        now - (i * 1'000),
        /*numUses=*/0);
  }
  auto& state = lrfuState();
  constexpr int32_t kPresetThreshold = 42;
  state.evictionThreshold = kPresetThreshold;

  uint32_t clockHand = 0;
  uint32_t eventCounter = kNumEntries / 4 + 1;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/false);
  cursor->next();
  EXPECT_EQ(eventCounter, 0);
  EXPECT_NE(state.evictionThreshold, kPresetThreshold);
}

TEST_F(EvictionPolicyTest, recalibrationOnInitialSentinel) {
  constexpr int32_t kNumEntries = 12;
  const auto now = accessTime();
  for (int32_t i = 0; i < kNumEntries; ++i) {
    addEntry(
        /*offset=*/i * kEntrySize,
        now - (i * 1'000),
        /*numUses=*/0);
  }
  auto& state = lrfuState();
  ASSERT_EQ(state.evictionThreshold, ApproxLrfuEvictionPolicy::kNoThreshold);

  uint32_t clockHand = 0;
  uint32_t eventCounter = 0;
  auto cursor = makeCursor(clockHand, eventCounter, /*evictAllUnpinned=*/false);
  cursor->next();
  EXPECT_NE(state.evictionThreshold, ApproxLrfuEvictionPolicy::kNoThreshold);
}
