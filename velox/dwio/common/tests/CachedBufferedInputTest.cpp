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
#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/testutil/TestValue.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/IOBuf.h>
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
    allocator_ = std::make_shared<MallocAllocator>(MemoryAllocator::Options{
        .capacity = 512 << 20, .reservationByteLimit = 0});
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

enum class CacheRegionApi {
  kStringView,
  kIOBuf,
};

std::string cacheRegionApiString(CacheRegionApi api) {
  switch (api) {
    case CacheRegionApi::kStringView:
      return "StringView";
    case CacheRegionApi::kIOBuf:
      return "IOBuf";
  }
  VELOX_UNREACHABLE();
}

class CacheRegionTest : public CachedBufferedInputTest,
                        public testing::WithParamInterface<CacheRegionApi> {
 protected:
  // Calls cacheRegion using the API selected by the test parameter.
  // For IOBuf mode, splits the source data into multiple chained buffers
  // to exercise the multi-buffer path.
  void cacheRegionWithApi(
      CachedBufferedInput& input,
      uint64_t offset,
      uint64_t length,
      const char* data) {
    switch (GetParam()) {
      case CacheRegionApi::kStringView: {
        input.cacheRegion(offset, length, std::string_view(data, length));
        break;
      }
      case CacheRegionApi::kIOBuf: {
        // Split data into 3 chained IOBufs to exercise multi-buffer cursor
        // path. Also prepend a prefix to exercise non-zero bufferOffset.
        constexpr uint64_t kPrefix = 37;
        const std::string prefix(kPrefix, 'Z');
        const uint64_t chunk1 = length / 3;
        const uint64_t chunk2 = length / 3;
        const uint64_t chunk3 = length - chunk1 - chunk2;

        auto iobuf = folly::IOBuf::copyBuffer(prefix);
        iobuf->appendToChain(folly::IOBuf::copyBuffer(data, chunk1));
        iobuf->appendToChain(folly::IOBuf::copyBuffer(data + chunk1, chunk2));
        iobuf->appendToChain(
            folly::IOBuf::copyBuffer(data + chunk1 + chunk2, chunk3));
        ASSERT_EQ(iobuf->computeChainDataLength(), kPrefix + length);
        input.cacheRegion(offset, length, *iobuf, kPrefix);
        break;
      }
    }
  }
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

TEST_P(CacheRegionTest, cacheAndFind) {
  SCOPED_TRACE(cacheRegionApiString(GetParam()));
  constexpr int32_t kContentSize = 1 << 20; // 1MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_cacheAndFind");
  StringIdLease groupId(ids, "testGroup_cacheAndFind");

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
  constexpr uint64_t kSmallSize = 100;
  cacheRegionWithApi(input, 0, kSmallSize, content.data());

  EXPECT_TRUE(input.findCachedRegion(0).has_value());
  EXPECT_FALSE(input.findCachedRegion(1000).has_value());

  // Verify cached data is readable via read().
  auto stream = input.read(0, kSmallSize, LogType::FILE);
  auto next = getNext(*stream);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content.substr(0, kSmallSize));

  // Cache a larger region.
  constexpr uint64_t kLargeOffset = 1000;
  constexpr uint64_t kLargeSize = 64 * 1024; // 64KB
  cacheRegionWithApi(
      input, kLargeOffset, kLargeSize, content.data() + kLargeOffset);

  EXPECT_TRUE(input.findCachedRegion(kLargeOffset).has_value());

  // Verify large cached data is readable.
  auto stream2 = input.read(kLargeOffset, kLargeSize, LogType::FILE);
  std::string readBack;
  readBack.resize(kLargeSize);
  stream2->readFully(readBack.data(), kLargeSize);
  EXPECT_EQ(readBack, content.substr(kLargeOffset, kLargeSize));

  // Caching the same region again is a no-op (already cached).
  cacheRegionWithApi(input, 0, kSmallSize, content.data());
  EXPECT_TRUE(input.findCachedRegion(0).has_value());

  auto stats = cache_->refreshStats();
  EXPECT_EQ(stats.numEntries, 2);
}

TEST_P(CacheRegionTest, smallEntry) {
  SCOPED_TRACE(cacheRegionApiString(GetParam()));
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_smallEntry");
  StringIdLease groupId(ids, "testGroup_smallEntry");

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

  EXPECT_FALSE(input.findCachedRegion(kOffset).has_value());

  cacheRegionWithApi(input, kOffset, kSize, content.data() + kOffset);

  // First findCachedRegion after cacheRegion should count as a cache hit since
  // cacheRegion clears the first-use flag (data was populated externally, not
  // loaded on-demand).
  EXPECT_EQ(ioStatistics_->ramHit().count(), 0);
  auto result = input.findCachedRegion(kOffset);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), kSize);
  EXPECT_EQ(ioStatistics_->ramHit().count(), 1);
  EXPECT_EQ(ioStatistics_->ramHit().sum(), kSize);

  // Second findCachedRegion is also a cache hit.
  auto result2 = input.findCachedRegion(kOffset);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(ioStatistics_->ramHit().count(), 2);
  EXPECT_EQ(ioStatistics_->ramHit().sum(), 2 * kSize);

  // Small entry should have exactly one contiguous range.
  const auto& ranges = result->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].size(), kSize);
  EXPECT_EQ(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(content.data() + kOffset, kSize));

  // toIOBuf on a small (single-range) entry produces a single IOBuf.
  auto iobuf = result->toIOBuf();
  EXPECT_FALSE(iobuf.isChained());
  EXPECT_EQ(iobuf.length(), kSize);
  EXPECT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(iobuf.data()), iobuf.length()),
      std::string_view(content.data() + kOffset, kSize));
}

TEST_P(CacheRegionTest, largeEntry) {
  SCOPED_TRACE(cacheRegionApiString(GetParam()));
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_largeEntry");
  StringIdLease groupId(ids, "testGroup_largeEntry");

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

  cacheRegionWithApi(input, kOffset, kSize, content.data() + kOffset);

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
  EXPECT_EQ(reassembled, content.substr(kOffset, kSize));

  // toIOBuf total length matches and content matches.
  auto iobuf = result->toIOBuf();
  EXPECT_EQ(iobuf.computeChainDataLength(), kSize);

  std::string fromIobuf;
  fromIobuf.reserve(kSize);
  const auto* current = &iobuf;
  do {
    fromIobuf.append(
        reinterpret_cast<const char*>(current->data()), current->length());
    current = current->next();
  } while (current != &iobuf);
  EXPECT_EQ(fromIobuf, content.substr(kOffset, kSize));
}

TEST_P(CacheRegionTest, miss) {
  SCOPED_TRACE(cacheRegionApiString(GetParam()));
  constexpr int32_t kContentSize = 1 << 20;
  std::string content(kContentSize, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_miss");
  StringIdLease groupId(ids, "testGroup_miss");

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

TEST_P(CacheRegionTest, pinKeepsDataAlive) {
  SCOPED_TRACE(cacheRegionApiString(GetParam()));
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile_pinAlive");
  StringIdLease groupId(ids, "testGroup_pinAlive");

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
  cacheRegionWithApi(input, 0, kSize, content.data());

  auto cached = input.findCachedRegion(0);
  ASSERT_TRUE(cached.has_value());

  auto stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);

  const auto& ranges = cached->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(content.data(), kSize));

  // Move into a new optional — pin transfers, data stays valid.
  auto moved = std::move(cached);
  ASSERT_TRUE(moved.has_value());
  EXPECT_EQ(moved->size(), kSize);
  EXPECT_EQ(moved->ranges().size(), 1);
  EXPECT_EQ(
      std::string_view(moved->ranges()[0].data(), moved->ranges()[0].size()),
      std::string_view(content.data(), kSize));

  stats = cache_->refreshStats();
  EXPECT_GT(stats.sharedPinnedBytes, 0);

  // Dropping the moved CachedRegion releases the pin.
  moved.reset();
  stats = cache_->refreshStats();
  EXPECT_EQ(stats.sharedPinnedBytes, 0);
}

INSTANTIATE_TEST_SUITE_P(
    CacheRegionTest,
    CacheRegionTest,
    testing::Values(CacheRegionApi::kStringView, CacheRegionApi::kIOBuf),
    [](const testing::TestParamInfo<CacheRegionApi>& info) {
      return cacheRegionApiString(info.param);
    });

TEST_F(CachedBufferedInputTest, findCachedRegionExclusiveWithWait) {
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  const std::string fileName = "testFile_findWait";
  StringIdLease fileId(ids, fileName);
  StringIdLease groupId(ids, "testGroup_findWait");

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

  // Create an exclusive entry using the same file ID as the
  // CachedBufferedInput.
  StringIdLease sameFileId(ids, fileName);
  RawFileCacheKey key{sameFileId.id(), kOffset};
  auto exclusivePin = cache_->findOrCreate(key, kSize, nullptr);
  ASSERT_FALSE(exclusivePin.empty());
  ASSERT_TRUE(exclusivePin.entry()->isExclusive());

  // Verify latency counters are initially zero.
  EXPECT_EQ(ioStatistics_->cacheWaitLatencyUs().count(), 0);
  EXPECT_EQ(ioStatistics_->queryThreadIoLatencyUs().count(), 0);

  // Spawn a thread that calls findCachedRegion — it will block waiting for
  // the exclusive entry to become shared.
  std::atomic_bool findFinished{false};
  std::optional<CachedRegion> findResult;
  std::thread findThread([&]() {
    findResult = input.findCachedRegion(kOffset);
    findFinished.store(true, std::memory_order_release);
  });

  // Give the thread time to enter the wait state.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ASSERT_FALSE(findFinished.load(std::memory_order_acquire));

  // Populate the entry and transition to shared.
  auto* entry = exclusivePin.entry();
  ASSERT_LT(kSize, AsyncDataCacheEntry::kTinyDataSize);
  ::memcpy(entry->tinyData(), content.data() + kOffset, kSize);
  entry->setExclusiveToShared();
  exclusivePin.clear();

  // The findCachedRegion thread should now complete.
  findThread.join();
  ASSERT_TRUE(findFinished.load(std::memory_order_acquire));

  // Verify the latency counters were incremented.
  EXPECT_EQ(ioStatistics_->cacheWaitLatencyUs().count(), 1);
  EXPECT_GT(ioStatistics_->cacheWaitLatencyUs().sum(), 0);
  EXPECT_EQ(ioStatistics_->queryThreadIoLatencyUs().count(), 1);
  EXPECT_GT(ioStatistics_->queryThreadIoLatencyUs().sum(), 0);

  // Verify the returned CachedRegion has correct data.
  ASSERT_TRUE(findResult.has_value());
  EXPECT_EQ(findResult->size(), kSize);
  const auto& ranges = findResult->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(content.data() + kOffset, kSize));
}

TEST_F(CachedBufferedInputTest, cacheRegionSkipsOngoingInsert) {
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  const std::string fileName = "testFile_skipInsert";
  StringIdLease fileId(ids, fileName);
  StringIdLease groupId(ids, "testGroup_skipInsert");

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

  // Simulate an ongoing insert by creating an exclusive entry directly.
  StringIdLease sameFileId(ids, fileName);
  RawFileCacheKey key{sameFileId.id(), kOffset};
  auto exclusivePin = cache_->findOrCreate(key, kSize, nullptr);
  ASSERT_FALSE(exclusivePin.empty());
  ASSERT_TRUE(exclusivePin.entry()->isExclusive());

  // cacheRegion with different data should return immediately without blocking
  // or overwriting (the entry is already exclusively held by another
  // operation).
  const std::string differentData(kSize, 'X');
  input.cacheRegion(kOffset, kSize, std::string_view(differentData));

  // The entry should still be exclusive — cacheRegion gave up.
  ASSERT_TRUE(exclusivePin.entry()->isExclusive());

  // Complete the original insert with the real data.
  auto* entry = exclusivePin.entry();
  ASSERT_LT(kSize, AsyncDataCacheEntry::kTinyDataSize);
  ::memcpy(entry->tinyData(), content.data() + kOffset, kSize);
  entry->setExclusiveToShared();
  exclusivePin.clear();

  // findCachedRegion should return the original data, not the 'X' data
  // that cacheRegion tried to write.
  auto cached = input.findCachedRegion(kOffset);
  ASSERT_TRUE(cached.has_value());
  EXPECT_EQ(cached->size(), kSize);
  const auto& ranges = cached->ranges();
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(content.data() + kOffset, kSize));
  // Verify it's NOT the data passed to cacheRegion.
  EXPECT_NE(
      std::string_view(ranges[0].data(), ranges[0].size()),
      std::string_view(differentData));
}

TEST_F(CachedBufferedInputTest, preloadCalledTwice) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadTwice");
  StringIdLease groupId(ids, "preloadTwiceGroup");

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

  input.preload();
  ASSERT_TRUE(input.preloaded());
  VELOX_ASSERT_THROW(input.preload(), "preload() called more than once");
}

TEST_F(CachedBufferedInputTest, isBufferedWithPreload) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "isBufferedPreload");
  StringIdLease groupId(ids, "isBufferedPreloadGroup");

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

  // Before preload, isBuffered should return false.
  EXPECT_FALSE(input.isBuffered(0, 100));

  input.preload();
  ASSERT_TRUE(input.preloaded());

  // After preload, isBuffered should return true.
  EXPECT_TRUE(input.isBuffered(0, 100));
  EXPECT_TRUE(input.isBuffered(500, 200));
}

TEST_F(CachedBufferedInputTest, enqueueSkipsRequestsWhenPreloaded) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "enqueueSkipsRequests");
  StringIdLease groupId(ids, "enqueueSkipsRequestsGroup");

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

  ASSERT_EQ(readFile->numReads(), 0);

  input.preload();
  ASSERT_TRUE(input.preloaded());
  ASSERT_EQ(readFile->numReads(), 1);

  // After preload, enqueue should not add to requests or coalesced loads.
  auto stream = input.enqueue(common::Region{0, 100}, nullptr);
  ASSERT_NE(stream, nullptr);

  // Verify no coalesced loads were created (requests are skipped when
  // preloaded).
  EXPECT_EQ(input.testingCoalescedLoads().size(), 0);
  EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);

  // Stream should still be able to read data from preloaded content.
  auto next = getNext(*stream);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content.substr(0, 100));

  // No additional file reads - all served from preloaded cache.
  ASSERT_EQ(readFile->numReads(), 1);
}

TEST_F(CachedBufferedInputTest, readSetsPreloadedPinWhenPreloaded) {
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "readSetsPreloadedPin");
  StringIdLease groupId(ids, "readSetsPreloadedPinGroup");

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

  ASSERT_EQ(readFile->numReads(), 0);

  input.preload();
  ASSERT_TRUE(input.preloaded());
  ASSERT_EQ(readFile->numReads(), 1);

  // Use read() to create a stream (not enqueue).
  auto stream = input.read(0, 100, LogType::TEST);
  ASSERT_NE(stream, nullptr);

  // Stream should be able to read data from preloaded content.
  auto next = getNext(*stream);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content.substr(0, 100));

  // No additional file reads - all served from preloaded cache.
  ASSERT_EQ(readFile->numReads(), 1);
}

TEST_F(CachedBufferedInputTest, preloadAfterEnqueue) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadAfterEnqueue");
  StringIdLease groupId(ids, "preloadAfterEnqueueGroup");

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

  input.enqueue({0, 100}, nullptr);
  VELOX_ASSERT_THROW(
      input.preload(), "preload() must be called before enqueue()");
}

TEST_F(CachedBufferedInputTest, preloadedStreamSkipsEviction) {
  // When cacheable_=false, non-preloaded streams mark cache entries as
  // immediately evictable on destruction (via clearCachePin and
  // makeCacheEvictable). Preloaded streams must skip both because the
  // preloaded cache entry is shared across all streams.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setCacheable(false);
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadEvictionTest");
  StringIdLease groupId(ids, "preloadEvictionGroup");

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

  ASSERT_EQ(readFile->numReads(), 0);

  input.preload();
  ASSERT_TRUE(input.preloaded());
  ASSERT_EQ(readFile->numReads(), 1);

  auto statsBefore = cache_->refreshStats();
  ASSERT_GT(statsBefore.sharedPinnedBytes, 0);

  // Create a stream, read from it, and destroy it.
  {
    auto stream = input.enqueue(common::Region{0, 100}, nullptr);
    auto next = getNext(*stream);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), content.substr(0, 100));
  }

  // The preloaded cache entry must still be pinned and accessible after stream
  // destruction, even with cacheable_=false.
  auto statsAfter = cache_->refreshStats();
  EXPECT_GT(statsAfter.sharedPinnedBytes, 0);

  // A new stream should still be able to read from the preloaded entry.
  auto stream2 = input.enqueue(common::Region{0, 100}, nullptr);
  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(0, 100));

  // No additional file reads — all served from preloaded cache.
  ASSERT_EQ(readFile->numReads(), 1);
}

TEST_F(CachedBufferedInputTest, preloadRespectsNotCacheable) {
  // When cacheable_=false, the preloaded cache entry should be made evictable
  // (lastUse=0, numUses=0) when CachedBufferedInput is destroyed.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setCacheable(false);
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadNotCacheableTest");
  StringIdLease groupId(ids, "preloadNotCacheableGroup");
  const auto fileNumId = fileId.id();

  {
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

    input.preload();
    ASSERT_TRUE(input.preloaded());

    auto stats = cache_->refreshStats();
    ASSERT_GT(stats.sharedPinnedBytes, 0);
  }

  // After CachedBufferedInput destruction with cacheable_=false, the preloaded
  // entry should be marked as immediately evictable via makeEvictable().
  cache::RawFileCacheKey key{fileNumId, 0};
  EXPECT_TRUE(cache_->testingIsEvictable(key));
}

TEST_F(CachedBufferedInputTest, preloadRespectsCacheable) {
  // When cacheable_=true (default), the preloaded cache entry should not be
  // marked as immediately evictable after CachedBufferedInput is destroyed.
  constexpr int32_t kContentSize = 1 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setCacheable(true);
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadCacheableTest");
  StringIdLease groupId(ids, "preloadCacheableGroup");
  const auto fileNumId = fileId.id();

  {
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

    input.preload();
    ASSERT_TRUE(input.preloaded());
  }

  // After CachedBufferedInput destruction with cacheable_=true, the cache
  // entry should NOT be marked as immediately evictable.
  cache::RawFileCacheKey key{fileNumId, 0};
  EXPECT_FALSE(cache_->testingIsEvictable(key));
}

TEST_F(CachedBufferedInputTest, preload) {
  struct TestParam {
    uint64_t fileSize;
    std::string debugString() const {
      return fmt::format("fileSize {}", fileSize);
    }
  };
  std::vector<TestParam> testSettings = {
      // Small file (tinyData path in cache entry).
      {100},
      // File at tinyData boundary.
      {AsyncDataCacheEntry::kTinyDataSize},
      // Larger file (allocation path).
      {1 << 20},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::string content;
    content.resize(testData.fileSize);
    for (uint64_t i = 0; i < testData.fileSize; ++i) {
      content[i] = static_cast<char>('a' + (i % 26));
    }
    auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

    io::ReaderOptions readerOptions(pool_.get());
    readerOptions.setLoadQuantum(1 << 20);

    auto& ids = fileIds();
    StringIdLease fileId(ids, fmt::format("preloadTest_{}", testData.fileSize));
    StringIdLease groupId(
        ids, fmt::format("preloadGroup_{}", testData.fileSize));

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

    ASSERT_EQ(readFile->numReads(), 0);
    EXPECT_FALSE(input.preloaded());

    auto statsBefore = cache_->refreshStats();
    const auto readCountBefore = ioStatistics_->read().count();
    const auto readSumBefore = ioStatistics_->read().sum();
    const auto rawBytesBefore = ioStatistics_->rawBytesRead();

    input.preload();

    EXPECT_TRUE(input.preloaded());

    const auto readsAfterPreload = readFile->numReads();
    ASSERT_GT(readsAfterPreload, 0);

    // Cache should have one new entry for the whole file.
    auto statsAfter = cache_->refreshStats();
    EXPECT_EQ(statsAfter.numEntries, statsBefore.numEntries + 1);
    // preloadPin_ holds a shared pin.
    EXPECT_GT(statsAfter.sharedPinnedBytes, 0);
    // IO stats: one storage read of fileSize bytes.
    EXPECT_EQ(ioStatistics_->read().count(), readCountBefore + 1);
    EXPECT_EQ(ioStatistics_->read().sum(), readSumBefore + testData.fileSize);
    EXPECT_EQ(
        ioStatistics_->rawBytesRead(), rawBytesBefore + testData.fileSize);

    // Enqueue sub-region streams and read from the preloaded cache entry.
    const uint64_t regionSize =
        std::min<uint64_t>(testData.fileSize / 2, testData.fileSize);
    auto stream1 = input.enqueue(common::Region{0, regionSize}, nullptr);
    ASSERT_NE(stream1, nullptr);

    auto next1 = getNext(*stream1);
    ASSERT_TRUE(next1.has_value());
    EXPECT_EQ(next1.value(), content.substr(0, regionSize));

    if (testData.fileSize > regionSize) {
      auto stream2 = input.enqueue(
          common::Region{regionSize, testData.fileSize - regionSize}, nullptr);
      ASSERT_NE(stream2, nullptr);

      auto next2 = getNext(*stream2);
      ASSERT_TRUE(next2.has_value());
      EXPECT_EQ(
          next2.value(),
          content.substr(regionSize, testData.fileSize - regionSize));
    }

    // No additional file reads after preload.
    ASSERT_EQ(readFile->numReads(), readsAfterPreload);
  }
}

TEST_F(CachedBufferedInputTest, preloadCacheSharing) {
  constexpr int32_t kContentSize = 1 << 20; // 1MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  const std::string fileName = "preloadSharingTest";
  StringIdLease fileId1(ids, fileName);
  StringIdLease groupId1(ids, "preloadSharingGroup1");

  CachedBufferedInput input1(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId1),
      cache_.get(),
      tracker_,
      std::move(groupId1),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  ASSERT_EQ(readFile->numReads(), 0);

  // First preload reads from storage.
  input1.preload();
  EXPECT_TRUE(input1.preloaded());
  ASSERT_EQ(readFile->numReads(), 1);

  auto stats1 = cache_->refreshStats();
  auto readCount1 = ioStatistics_->read().count();

  // Second CachedBufferedInput with the same file should hit the cache.
  StringIdLease fileId2(ids, fileName);
  StringIdLease groupId2(ids, "preloadSharingGroup2");

  CachedBufferedInput input2(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId2),
      cache_.get(),
      tracker_,
      std::move(groupId2),
      ioStatistics_,
      nullptr,
      executor_.get(),
      readerOptions);

  input2.preload();
  EXPECT_TRUE(input2.preloaded());

  // No additional file read — second preload hits the cache.
  ASSERT_EQ(readFile->numReads(), 1);

  // No additional cache entry should be created.
  auto stats2 = cache_->refreshStats();
  EXPECT_EQ(stats2.numEntries, stats1.numEntries);

  // No additional storage read should happen (cache hit).
  EXPECT_EQ(ioStatistics_->read().count(), readCount1);

  // Both inputs should still be able to read data.
  auto stream1 = input1.enqueue(common::Region{0, 100}, nullptr);
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), content.substr(0, 100));

  auto stream2 = input2.enqueue(common::Region{0, 100}, nullptr);
  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), content.substr(0, 100));

  // No additional file reads — all served from cache.
  ASSERT_EQ(readFile->numReads(), 1);
}

TEST_F(CachedBufferedInputTest, prefetchScope) {
  constexpr int32_t kContentSize = 32 << 20;
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }
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

  // Enqueue non-prefetch requests (with stream IDs).
  constexpr int32_t kNumRequests = 3;
  constexpr uint64_t kRequestSize = 500;
  std::vector<std::unique_ptr<StreamIdentifier>> streamIds;
  for (int32_t i = 0; i < kNumRequests; ++i) {
    streamIds.push_back(std::make_unique<StreamIdentifier>(i));
  }
  for (int32_t i = 0; i < kNumRequests; ++i) {
    input.enqueue(
        {static_cast<uint64_t>(i * 1000), kRequestSize}, streamIds[i].get());
  }
  input.load(LogType::TEST);

  // There should be exactly 1 non-prefetch CoalescedLoad in kPlanned state.
  const auto& loadsBeforePrefetch = input.testingCoalescedLoads();
  ASSERT_EQ(loadsBeforePrefetch.size(), 1);
  ASSERT_EQ(loadsBeforePrefetch[0]->state(), CoalescedLoad::State::kPlanned);

  // Enqueue prefetch requests (without stream IDs).
  constexpr int32_t kMB = 1 << 20;
  for (int32_t i = 0; i < kNumRequests; ++i) {
    input.enqueue(
        {static_cast<uint64_t>(10 * kMB + i * 1000), kRequestSize}, nullptr);
  }
  input.load(LogType::TEST);

  // Wait for executor to complete all submitted tasks.
  executor_->join();

  const auto& loadsAfterPrefetch = input.testingCoalescedLoads();
  ASSERT_EQ(loadsAfterPrefetch.size(), 2);
  EXPECT_EQ(loadsAfterPrefetch[0]->state(), CoalescedLoad::State::kPlanned)
      << "Non-prefetch load should NOT be submitted to executor";
  EXPECT_EQ(loadsAfterPrefetch[1]->state(), CoalescedLoad::State::kLoaded)
      << "Prefetch load should be submitted and completed by executor";

  EXPECT_EQ(ioStatistics_->prefetch().sum(), kNumRequests * kRequestSize);
}

TEST_F(CachedBufferedInputTest, cloneNonCacheable) {
  constexpr int32_t kContentSize = 1 << 20; // 1MB
  std::string content(kContentSize, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(1 << 20);

  auto& ids = fileIds();
  StringIdLease fileId(ids, "testFile");
  StringIdLease groupId(ids, "testGroup");

  const auto fileNum = fileId.id();
  CachedBufferedInput input(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      tracker_,
      std::move(groupId),
      ioStatistics_,
      /*ioStats=*/nullptr,
      executor_.get(),
      readerOptions);

  ASSERT_TRUE(input.hasCache());

  auto nonCacheable = input.cloneNonCacheable();
  ASSERT_TRUE(nonCacheable->hasCache());

  // Read through the non-cacheable clone.
  constexpr uint64_t kOffset = 0;
  constexpr uint64_t kLength = 1'024;
  StreamIdentifier sid(0);
  auto stream = nonCacheable->enqueue({kOffset, kLength}, &sid);
  nonCacheable->load(LogType::FILE);

  auto result = getNext(*stream);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), kLength);

  // The cache should have an entry from the non-cacheable read.
  auto statsAfterLoad = cache_->refreshStats();
  EXPECT_GT(statsAfterLoad.numEntries, 0);

  // The entry should not be evictable while the non-cacheable clone is alive.
  const cache::RawFileCacheKey cacheKey{fileNum, kOffset};
  EXPECT_FALSE(cache_->testingIsEvictable(cacheKey));

  // Release the stream and destroy the non-cacheable input. Because
  // cacheable=false, the destructor marks preloaded entries as immediately
  // evictable (lastUse=0, numUses=0).
  stream.reset();
  nonCacheable.reset();

  // The entry is still in cache but now marked as immediately evictable.
  auto statsAfterDestroy = cache_->refreshStats();
  EXPECT_EQ(statsAfterDestroy.numEntries, statsAfterLoad.numEntries);
  EXPECT_TRUE(cache_->testingIsEvictable(cacheKey));

  // Read the same region through a cacheable input — should still work.
  StringIdLease fileId2(ids, "testFile2");
  StringIdLease groupId2(ids, "testGroup2");
  const auto fileNum2 = fileId2.id();
  CachedBufferedInput cacheableInput(
      readFile,
      MetricsLog::voidLog(),
      std::move(fileId2),
      cache_.get(),
      tracker_,
      std::move(groupId2),
      ioStatistics_,
      /*ioStats=*/nullptr,
      executor_.get(),
      readerOptions);

  auto stream2 = cacheableInput.enqueue({kOffset, kLength}, &sid);
  cacheableInput.load(LogType::FILE);
  auto result2 = getNext(*stream2);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(result2->size(), kLength);

  // The cacheable input's entry should persist and not be marked evictable.
  stream2.reset();
  const cache::RawFileCacheKey cacheKey2{fileNum2, kOffset};
  auto statsAfterCacheable = cache_->refreshStats();
  EXPECT_GT(statsAfterCacheable.numEntries, 0);
  EXPECT_FALSE(cache_->testingIsEvictable(cacheKey2));
}

} // namespace
