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

#include "velox/dwio/common/CachedBufferedInput.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/testutil/TestValue.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>

#include <thread>

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::cache;
using namespace facebook::velox::memory;
using facebook::velox::common::testutil::TestValue;

namespace {

std::optional<std::string> getNext(SeekableInputStream& input) {
  const void* buf = nullptr;
  int32_t size;
  if (input.Next(&buf, &size)) {
    return std::string(
        static_cast<const char*>(buf), static_cast<size_t>(size));
  } else {
    return std::nullopt;
  }
}

class CachedBufferedInputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    TestValue::enable();
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  void SetUp() override {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(10);
    allocator_ = std::make_shared<MallocAllocator>(
        512 << 20 /*capacity=*/, 0 /*reservationByteLimit=*/);
    cache_ = AsyncDataCache::create(allocator_.get());
    ioStatistics_ = std::make_shared<IoStatistics>();
    tracker_ = std::make_shared<ScanTracker>(
        "testTracker", nullptr, 256 << 10 /* 256KB */);
    pool_ = memoryManager()->addLeafPool();
  }

  void TearDown() override {
    executor_.reset();
    cache_->shutdown();
    cache_.reset();
    allocator_.reset();
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::shared_ptr<MemoryPool> pool_;
  std::shared_ptr<MallocAllocator> allocator_;
  std::shared_ptr<AsyncDataCache> cache_;
  std::shared_ptr<IoStatistics> ioStatistics_;
  std::shared_ptr<ScanTracker> tracker_;
};

TEST_F(CachedBufferedInputTest, reset) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  // First round: enqueue and load streams.
  constexpr int32_t kRegionSize = 8 << 10; // 8KB
  auto stream1 = input.enqueue({0, kRegionSize}, nullptr);
  auto stream2 = input.enqueue({kRegionSize, kRegionSize}, nullptr);
  ASSERT_NE(stream1, nullptr);
  ASSERT_NE(stream2, nullptr);

  // Verify cache is empty before load.
  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);

  input.load(LogType::TEST);

  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  // Wait until cache has two entries after load.
  while (cache_->refreshStats().numEntries != 2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);

  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  // Consume streams.
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), content.substr(0, kRegionSize));
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);

  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(kRegionSize, kRegionSize));
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  stream1.reset();
  stream2.reset();

  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 2);

  // Reset the input.
  input.reset();

  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 2);

  // Second round: enqueue different regions after reset.
  auto stream3 = input.enqueue({2 * kRegionSize, kRegionSize}, nullptr);
  auto stream4 = input.enqueue({3 * kRegionSize, kRegionSize}, nullptr);
  ASSERT_NE(stream3, nullptr);
  ASSERT_NE(stream4, nullptr);

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 2);

  input.load(LogType::TEST);

  // Wait until cache has two entries after load.
  while (cache_->refreshStats().numEntries != 4) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 4);

  auto next3 = getNext(*stream3);
  ASSERT_TRUE(next3.has_value());
  EXPECT_EQ(next3.value(), content.substr(2 * kRegionSize, kRegionSize));

  auto next4 = getNext(*stream4);
  ASSERT_TRUE(next4.has_value());
  EXPECT_EQ(next4.value(), content.substr(3 * kRegionSize, kRegionSize));

  // Reset the input.
  input.reset();

  stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 4);

  stream3.reset();
  stream4.reset();

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 4);
}

TEST_F(CachedBufferedInputTest, readAfterReset) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr int32_t kRegionSize = 8 << 10; // 8KB

  // Enqueue and load streams.
  auto stream1 = input.enqueue({0, kRegionSize}, nullptr);
  auto stream2 = input.enqueue({kRegionSize, kRegionSize}, nullptr);
  ASSERT_NE(stream1, nullptr);
  ASSERT_NE(stream2, nullptr);

  input.load(LogType::TEST);

  // Wait until cache has two entries after load.
  while (cache_->refreshStats().numEntries != 2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);

  // Reset the input before reading from streams.
  input.reset();
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);

  // Read from streams after reset - data should still be available from cache.
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), content.substr(0, kRegionSize));

  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(kRegionSize, kRegionSize));

  EXPECT_EQ(stats.sharedPinnedBytes, 0);
  EXPECT_EQ(stats.exclusivePinnedBytes, 0);
  EXPECT_EQ(stats.numEntries, 2);
}

DEBUG_ONLY_TEST_F(CachedBufferedInputTest, resetInputWithBeforeLoading) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr int32_t kRegionSize = 8 << 10; // 8KB

  // Block the coalesced load to verify cache references are held.
  folly::Baton<> loadStarted;
  folly::Baton<> loadAllowed;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::cache::CoalescedLoad::loadOrFuture",
      std::function<void(const CoalescedLoad*)>(
          [&](const CoalescedLoad* /*load*/) {
            loadStarted.post();
            loadAllowed.wait();
          }));

  // Enqueue and load streams.
  auto stream1 = input.enqueue({0, kRegionSize}, nullptr);
  auto stream2 = input.enqueue({kRegionSize, kRegionSize}, nullptr);
  ASSERT_NE(stream1, nullptr);
  ASSERT_NE(stream2, nullptr);

  input.load(LogType::TEST);

  // Wait for the load to start (but it's blocked).
  loadStarted.wait();

  // Verify cache is still empty (load is pending).
  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);
  EXPECT_EQ(stats.numExclusive, 0);

  // Verify coalesced load references are held.
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  // Reset the input while load is pending.
  input.reset();

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);
  EXPECT_EQ(stats.numExclusive, 0);

  // After reset, internal tracking should be cleared.
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

  // Allow the load to proceed but cancelled.
  loadAllowed.post();

  std::this_thread::sleep_for(std::chrono::seconds(1));

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);
  EXPECT_EQ(stats.numExclusive, 0);

  // Read from streams - data should be available from cache.
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), content.substr(0, kRegionSize));

  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(kRegionSize, kRegionSize));

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);
  EXPECT_EQ(stats.numExclusive, 0);
}

DEBUG_ONLY_TEST_F(CachedBufferedInputTest, resetInputWithAfterLoading) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr int32_t kRegionSize = 8 << 10; // 8KB

  // Block the coalesced load to verify cache references are held.
  folly::Baton<> loadStarted;
  folly::Baton<> loadAllowed;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::cache::CoalescedLoad::loadOrFuture::loading",
      std::function<void(const CoalescedLoad*)>(
          [&](const CoalescedLoad* /*load*/) {
            loadStarted.post();
            loadAllowed.wait();
          }));

  // Enqueue and load streams.
  auto stream1 = input.enqueue({0, kRegionSize}, nullptr);
  auto stream2 = input.enqueue({kRegionSize, kRegionSize}, nullptr);
  ASSERT_NE(stream1, nullptr);
  ASSERT_NE(stream2, nullptr);

  input.load(LogType::TEST);

  // Wait for the load to start (but it's blocked).
  loadStarted.wait();

  // Verify cache is still empty (load is pending).
  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);
  EXPECT_EQ(stats.numExclusive, 0);

  // Verify coalesced load references are held.
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

  // Reset the input while load is pending.
  input.reset();

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 0);
  EXPECT_EQ(stats.numExclusive, 0);

  // After reset, internal tracking should be cleared.
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
  EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

  // Allow the load to proceed without cancelling.
  loadAllowed.post();

  // Wait until cache has two entries after load.
  while (cache_->refreshStats().numEntries != 2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);
  EXPECT_EQ(stats.numExclusive, 0);

  // Read from streams - data should be available from cache.
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), content.substr(0, kRegionSize));

  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(kRegionSize, kRegionSize));

  stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);
  EXPECT_EQ(stats.numExclusive, 0);
}

TEST_F(CachedBufferedInputTest, hasCache) {
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  // CachedBufferedInput always has cache.
  EXPECT_TRUE(input.hasCache());
}

TEST_F(CachedBufferedInputTest, cacheRegionAndFindCachedRegion) {
  constexpr int32_t kContentSize = 1 << 20; // 1MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cacheRegion");
  StringIdLease groupId(ids, "testGroup_cacheRegion");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  // Initially no regions are cached.
  EXPECT_FALSE(input.findCachedRegion(0).has_value());
  EXPECT_FALSE(input.findCachedRegion(1000).has_value());

  // Cache a small region (fits in tinyData path).
  const std::string_view smallData = std::string_view(content).substr(0, 100);
  input.cacheRegion(0, 100, smallData);

  // Now the region should be found.
  EXPECT_TRUE(input.findCachedRegion(0).has_value());
  // Different offset is still not cached.
  EXPECT_FALSE(input.findCachedRegion(1000).has_value());

  // Verify cached data is readable via read().
  auto stream = input.read(0, 100, LogType::FILE);
  auto next = getNext(*stream);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content.substr(0, 100));

  // Cache a larger region.
  constexpr uint64_t kLargeOffset = 1000;
  constexpr uint64_t kLargeSize = 64 * 1024; // 64KB
  const std::string_view largeData =
      std::string_view(content).substr(kLargeOffset, kLargeSize);
  input.cacheRegion(kLargeOffset, kLargeSize, largeData);

  EXPECT_TRUE(input.findCachedRegion(kLargeOffset).has_value());

  // Verify large cached data is readable.
  auto stream2 = input.read(kLargeOffset, kLargeSize, LogType::FILE);
  std::string readBack;
  readBack.resize(kLargeSize);
  stream2->readFully(readBack.data(), kLargeSize);
  EXPECT_EQ(readBack, content.substr(kLargeOffset, kLargeSize));

  // Caching the same region again is a no-op (already cached).
  input.cacheRegion(0, 100, smallData);
  EXPECT_TRUE(input.findCachedRegion(0).has_value());

  // Verify cache entry count.
  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);
}

TEST_F(CachedBufferedInputTest, cachedRegionSmallEntry) {
  // Test CachedRegion with a tiny entry (< kTinyDataSize, uses tinyData path).
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cachedRegionSmall");
  StringIdLease groupId(ids, "testGroup_cachedRegionSmall");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kOffset = 0;
  constexpr uint64_t kSize = 100;
  static_assert(kSize < AsyncDataCacheEntry::kTinyDataSize);

  // No cached region before caching.
  auto result = input.findCachedRegion(kOffset);
  EXPECT_FALSE(result.has_value());

  // Populate cache via cacheRegion.
  const std::string_view data = std::string_view(content).substr(kOffset, kSize);
  input.cacheRegion(kOffset, kSize, data);

  // Now findCachedRegion should return a valid CachedRegion.
  result = input.findCachedRegion(kOffset);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), kSize);

  // Small entry should have exactly one contiguous range.
  const auto& ranges = result->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].size(), kSize);
  EXPECT_EQ(std::string_view(ranges[0].data(), ranges[0].size()), data);
}

TEST_F(CachedBufferedInputTest, cachedRegionLargeEntry) {
  // Test CachedRegion with a large entry (>= kTinyDataSize, uses allocation
  // runs).
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cachedRegionLarge");
  StringIdLease groupId(ids, "testGroup_cachedRegionLarge");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kOffset = 0;
  constexpr uint64_t kSize = 64 * 1024; // 64KB
  static_assert(kSize >= AsyncDataCacheEntry::kTinyDataSize);

  const std::string_view data = std::string_view(content).substr(kOffset, kSize);
  input.cacheRegion(kOffset, kSize, data);

  auto result = input.findCachedRegion(kOffset);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), kSize);

  // Large entry may have multiple ranges. Verify total size and content.
  const auto& ranges = result->ranges();
  ASSERT_GT(ranges.size(), 0);

  uint64_t totalSize = 0;
  std::string reassembled;
  reassembled.reserve(kSize);
  for (const auto& range : ranges) {
    EXPECT_GT(range.size(), 0);
    reassembled.append(range.data(), range.size());
    totalSize += range.size();
  }
  EXPECT_EQ(totalSize, kSize);
  EXPECT_EQ(reassembled, std::string(data));
}

TEST_F(CachedBufferedInputTest, cachedRegionMiss) {
  // findCachedRegion returns nullopt on cache miss.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content(kContentSize, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cachedRegionMiss");
  StringIdLease groupId(ids, "testGroup_cachedRegionMiss");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  EXPECT_FALSE(input.findCachedRegion(0).has_value());
  EXPECT_FALSE(input.findCachedRegion(12345).has_value());
}

TEST_F(CachedBufferedInputTest, cachedRegionPinKeepsDataAlive) {
  // CachedRegion holds a pin that prevents eviction.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cachedRegionPin");
  StringIdLease groupId(ids, "testGroup_cachedRegionPin");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kSize = 100;
  input.cacheRegion(0, kSize, std::string_view(content.data(), kSize));

  auto cached = input.findCachedRegion(0);
  ASSERT_TRUE(cached.has_value());

  auto stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);

  // Data remains valid and pinned while CachedRegion is alive.
  const auto& ranges = cached->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(content.data(), kSize));

  // Dropping the CachedRegion releases the pin.
  cached.reset();
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
}

TEST_F(CachedBufferedInputTest, cachedRegionMoveSemantics) {
  // CachedRegion is move-constructible and move-assignable.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cachedRegionMove");
  StringIdLease groupId(ids, "testGroup_cachedRegionMove");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kSize = 100;
  input.cacheRegion(0, kSize, std::string_view(content.data(), kSize));

  auto cached = input.findCachedRegion(0);
  ASSERT_TRUE(cached.has_value());

  // Move construct into a new optional.
  auto moved = std::move(cached);
  ASSERT_TRUE(moved.has_value());
  EXPECT_EQ(moved->size(), kSize);
  EXPECT_EQ(moved->ranges().size(), 1);
  EXPECT_EQ(
      std::string_view(moved->ranges()[0].data(), moved->ranges()[0].size()),
      std::string_view(content.data(), kSize));

  // Pin still held after move.
  auto stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);

  moved.reset();
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
}

TEST_F(CachedBufferedInputTest, cachedRegionToIOBufSmall) {
  // toIOBuf on a small (single-range) entry produces a single IOBuf.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_toIOBufSmall");
  StringIdLease groupId(ids, "testGroup_toIOBufSmall");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kSize = 100;
  input.cacheRegion(0, kSize, std::string_view(content.data(), kSize));

  auto cached = input.findCachedRegion(0);
  ASSERT_TRUE(cached.has_value());
  ASSERT_EQ(cached->ranges().size(), 1);

  auto iobuf = cached->toIOBuf();
  // Single range produces a single-element chain.
  EXPECT_FALSE(iobuf.isChained());
  EXPECT_EQ(iobuf.length(), kSize);
  EXPECT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(iobuf.data()), iobuf.length()),
      std::string_view(content.data(), kSize));
}

TEST_F(CachedBufferedInputTest, cachedRegionToIOBufLarge) {
  // toIOBuf on a large (multi-range) entry produces a chained IOBuf whose
  // total length matches the cached size and whose content matches the
  // original data.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_toIOBufLarge");
  StringIdLease groupId(ids, "testGroup_toIOBufLarge");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  constexpr uint64_t kSize = 64 * 1024; // 64KB
  const std::string_view data(content.data(), kSize);
  input.cacheRegion(0, kSize, data);

  auto cached = input.findCachedRegion(0);
  ASSERT_TRUE(cached.has_value());

  auto iobuf = cached->toIOBuf();
  EXPECT_EQ(iobuf.computeChainDataLength(), kSize);

  // Reassemble content from the IOBuf chain and verify.
  std::string reassembled;
  reassembled.reserve(kSize);
  const auto* current = &iobuf;
  do {
    reassembled.append(
        reinterpret_cast<const char*>(current->data()), current->length());
    current = current->next();
  } while (current != &iobuf);
  EXPECT_EQ(reassembled, std::string(data));
}

TEST_F(CachedBufferedInputTest, asyncDataCacheFind) {
  // Test AsyncDataCache::find() for miss, hit, and exclusive entry.
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_find");
  const RawFileCacheKey key{fileId.id(), /*offset=*/0};

  // Miss: entry does not exist.
  auto result = cache_->find(key);
  EXPECT_FALSE(result.has_value());

  // Create an entry via findOrCreate (returns exclusive pin).
  auto exclusivePin = cache_->findOrCreate(key, /*size=*/100, nullptr);
  EXPECT_FALSE(exclusivePin.empty());
  EXPECT_TRUE(exclusivePin.checkedEntry()->isExclusive());

  // While the entry is exclusive, find() returns an empty pin (not nullopt).
  result = cache_->find(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());

  // find() with waitFuture sets the future when entry is exclusive.
  folly::SemiFuture<bool> waitFuture(false);
  result = cache_->find(key, &waitFuture);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->empty());
  EXPECT_TRUE(waitFuture.valid());

  // Transition entry to shared mode and release the original pin.
  exclusivePin.checkedEntry()->setExclusiveToShared();
  exclusivePin.clear();

  // Hit: entry is now shared, find() returns a non-empty shared pin.
  result = cache_->find(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->empty());
  EXPECT_TRUE(result->checkedEntry()->isShared());
  EXPECT_EQ(result->checkedEntry()->size(), 100);

  // Pin keeps data alive.
  auto stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);

  result.reset();
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
}

TEST_F(CachedBufferedInputTest, cacheRegionFromIOBuf) {
  // Test cacheRegion(IOBuf) overload — populates cache from an IOBuf chain.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cacheIOBuf");
  StringIdLease groupId(ids, "testGroup_cacheIOBuf");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  // Build a chained IOBuf from two separate pieces.
  constexpr uint64_t kOffset = 100;
  constexpr uint64_t kSize = 200;
  constexpr uint64_t kSplit = 80;
  auto iobuf = folly::IOBuf::copyBuffer(content.data() + kOffset, kSplit);
  iobuf->appendToChain(
      folly::IOBuf::copyBuffer(content.data() + kOffset + kSplit, kSize - kSplit));
  ASSERT_TRUE(iobuf->isChained());
  ASSERT_EQ(iobuf->computeChainDataLength(), kSize);

  // Cache from IOBuf with iobufOffset=0 (data starts at beginning of chain).
  input.cacheRegion(kOffset, kSize, *iobuf, /*iobufOffset=*/0);

  // Verify data is readable and matches original content.
  auto cached = input.findCachedRegion(kOffset);
  ASSERT_TRUE(cached.has_value());
  EXPECT_EQ(cached->size(), kSize);

  std::string reassembled;
  for (const auto& range : cached->ranges()) {
    reassembled.append(range.data(), range.size());
  }
  EXPECT_EQ(reassembled, content.substr(kOffset, kSize));
}

TEST_F(CachedBufferedInputTest, cacheRegionFromIOBufWithOffset) {
  // Test cacheRegion(IOBuf) with non-zero iobufOffset.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cacheIOBufOffset");
  StringIdLease groupId(ids, "testGroup_cacheIOBufOffset");

  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  // Build an IOBuf with prefix data we want to skip.
  constexpr uint64_t kFileOffset = 500;
  constexpr uint64_t kSize = 150;
  constexpr uint64_t kIoBufOffset = 50; // skip first 50 bytes of IOBuf
  // Create IOBuf containing [0..kIoBufOffset + kSize) of content.
  auto iobuf = folly::IOBuf::copyBuffer(
      content.data(), kIoBufOffset + kSize);
  ASSERT_FALSE(iobuf->isChained());

  // Cache region at kFileOffset from IOBuf starting at kIoBufOffset.
  // The data at iobuf[kIoBufOffset..kIoBufOffset+kSize) should be cached
  // at file offset kFileOffset.
  input.cacheRegion(kFileOffset, kSize, *iobuf, kIoBufOffset);

  auto cached = input.findCachedRegion(kFileOffset);
  ASSERT_TRUE(cached.has_value());
  EXPECT_EQ(cached->size(), kSize);

  std::string reassembled;
  for (const auto& range : cached->ranges()) {
    reassembled.append(range.data(), range.size());
  }
  // The cached data should match content[kIoBufOffset..kIoBufOffset+kSize),
  // since that's what was at iobuf[kIoBufOffset..].
  EXPECT_EQ(reassembled, content.substr(kIoBufOffset, kSize));
}

TEST_F(CachedBufferedInputTest, baseBufferedInputHasNoCache) {
  constexpr int32_t kContentSize = 1 << 20;
  std::string content(kContentSize, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  BufferedInput input(readFile, *pool_);

  // Base BufferedInput has no cache.
  EXPECT_FALSE(input.hasCache());

  // cacheRegion is a no-op (should not crash).
  input.cacheRegion(0, 100, std::string_view(content.data(), 100));

  // findCachedRegion returns nullopt on base BufferedInput.
  EXPECT_FALSE(input.findCachedRegion(0).has_value());
}
} // namespace
