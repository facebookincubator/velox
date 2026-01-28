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
    ioStats_ = std::make_shared<IoStatistics>();
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
  std::shared_ptr<IoStatistics> ioStats_;
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
      ioStats_,
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
      ioStats_,
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
      ioStats_,
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
      ioStats_,
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
} // namespace
