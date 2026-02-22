/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/HashTableCache.h"

#include <gtest/gtest.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::exec::test {

class HashTableCacheTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addRootPool("HashTableCacheTest");
    queryCtx_ = core::QueryCtx::create();
  }

  void TearDown() override {
    // Clean up any cache entries created during tests.
    auto* cache = HashTableCache::instance();
    for (const auto& key : createdKeys_) {
      cache->drop(key);
    }
    createdKeys_.clear();
    queryCtx_.reset();
  }

  // Helper to track keys for cleanup.
  void trackKey(const std::string& key) {
    createdKeys_.push_back(key);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::vector<std::string> createdKeys_;
};

TEST_F(HashTableCacheTest, basicGet) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query1:node1";
  trackKey(key);

  ContinueFuture future = ContinueFuture::makeEmpty();
  auto entry = cache->get(key, "task1", queryCtx_.get(), &future);

  ASSERT_NE(entry, nullptr);
  ASSERT_NE(entry->tablePool, nullptr);
  EXPECT_EQ(entry->builderTaskId, "task1");
  EXPECT_FALSE(entry->buildComplete);
  // First caller should not get a future (they are the builder).
  EXPECT_FALSE(future.valid());
}

TEST_F(HashTableCacheTest, secondCallerGetsWaitFuture) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query2:node1";
  trackKey(key);

  // First caller (builder).
  ContinueFuture future1 = ContinueFuture::makeEmpty();
  auto entry1 = cache->get(key, "task1", queryCtx_.get(), &future1);
  EXPECT_FALSE(future1.valid());
  EXPECT_EQ(entry1->builderTaskId, "task1");

  // Second caller (waiter).
  ContinueFuture future2 = ContinueFuture::makeEmpty();
  auto entry2 = cache->get(key, "task2", queryCtx_.get(), &future2);

  // Should get the same entry.
  EXPECT_EQ(entry1, entry2);
  // Should get a valid future to wait on.
  EXPECT_TRUE(future2.valid());
  // Builder task ID should not change.
  EXPECT_EQ(entry2->builderTaskId, "task1");
}

TEST_F(HashTableCacheTest, putNotifiesWaiters) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query3:node1";
  trackKey(key);

  // First caller (builder).
  ContinueFuture future1 = ContinueFuture::makeEmpty();
  auto entry = cache->get(key, "task1", queryCtx_.get(), &future1);

  // Second caller (waiter).
  ContinueFuture future2 = ContinueFuture::makeEmpty();
  cache->get(key, "task2", queryCtx_.get(), &future2);
  ASSERT_TRUE(future2.valid());

  // Put the table.
  cache->put(key, nullptr, false);

  // Entry should now be marked complete.
  EXPECT_TRUE(entry->buildComplete);

  // The future should be fulfilled.
  EXPECT_TRUE(future2.isReady());
}

TEST_F(HashTableCacheTest, getAfterBuildComplete) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query4:node1";
  trackKey(key);

  // First caller creates and builds.
  ContinueFuture future1 = ContinueFuture::makeEmpty();
  cache->get(key, "task1", queryCtx_.get(), &future1);
  cache->put(key, nullptr, true);

  // Later caller should get completed entry without waiting.
  ContinueFuture future2 = ContinueFuture::makeEmpty();
  auto entry = cache->get(key, "task2", queryCtx_.get(), &future2);

  EXPECT_TRUE(entry->buildComplete);
  EXPECT_FALSE(future2.valid()); // No need to wait.
  EXPECT_TRUE(entry->hasNullKeys);
}

TEST_F(HashTableCacheTest, drop) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query5:node1";
  // Don't track - we're testing drop.

  ContinueFuture future = ContinueFuture::makeEmpty();
  auto entry1 = cache->get(key, "task1", queryCtx_.get(), &future);
  ASSERT_NE(entry1, nullptr);

  // Keep track of the original pool to verify it's different after flush.
  auto originalPool = entry1->tablePool;

  // Drop the entry.
  cache->drop(key);

  // Getting the same key should create a new entry.
  // Use a new queryCtx with different pool to avoid the leaf child name
  // collision.
  auto pool2 = memory::memoryManager()->addRootPool("HashTableCacheTest2");
  auto queryCtx2 = core::QueryCtx::create(
      nullptr, // executor
      core::QueryConfig{{}}, // queryConfig
      {}, // connectorConfigs
      cache::AsyncDataCache::getInstance(), // cache
      pool2);
  ContinueFuture future2 = ContinueFuture::makeEmpty();
  auto entry2 = cache->get(key, "task2", queryCtx2.get(), &future2);

  EXPECT_NE(entry1, entry2);
  EXPECT_EQ(entry2->builderTaskId, "task2");
  // The pool should be different (created under queryCtx2's pool).
  EXPECT_NE(entry1->tablePool, entry2->tablePool);

  // Cleanup.
  cache->drop(key);
}

TEST_F(HashTableCacheTest, concurrentWaiters) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query6:node1";
  trackKey(key);

  // First caller (builder).
  ContinueFuture builderFuture = ContinueFuture::makeEmpty();
  auto entry = cache->get(key, "builder", queryCtx_.get(), &builderFuture);

  // Multiple waiters.
  constexpr int kNumWaiters = 5;
  std::vector<ContinueFuture> waiterFutures(kNumWaiters);

  for (int i = 0; i < kNumWaiters; ++i) {
    waiterFutures[i] = ContinueFuture::makeEmpty();
    cache->get(
        key, fmt::format("waiter{}", i), queryCtx_.get(), &waiterFutures[i]);
    EXPECT_TRUE(waiterFutures[i].valid());
  }

  // Put the table.
  cache->put(key, nullptr, false);

  // All waiters should be notified.
  for (int i = 0; i < kNumWaiters; ++i) {
    EXPECT_TRUE(waiterFutures[i].isReady());
  }
}

TEST_F(HashTableCacheTest, builderTaskDriversDoNotWait) {
  auto* cache = HashTableCache::instance();
  const std::string key = "query7:node1";
  trackKey(key);

  // First driver of builder task.
  ContinueFuture future1 = ContinueFuture::makeEmpty();
  auto entry1 = cache->get(key, "task1", queryCtx_.get(), &future1);
  EXPECT_FALSE(future1.valid());

  // Second driver of the same builder task should not wait.
  ContinueFuture future2 = ContinueFuture::makeEmpty();
  auto entry2 = cache->get(key, "task1", queryCtx_.get(), &future2);

  EXPECT_EQ(entry1, entry2);
  // Same task should not get a future - they coordinate via JoinBridge.
  EXPECT_FALSE(future2.valid());
}

} // namespace facebook::velox::exec::test
