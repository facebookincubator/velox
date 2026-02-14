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

#include "velox/dwio/common/DirectBufferedInput.h"

#include <fmt/core.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <thread>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/io/Options.h"
#include "velox/common/testutil/TestValue.h"

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

class DirectBufferedInputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    TestValue::enable();
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  void SetUp() override {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(10);
    ioStatistics_ = std::make_shared<IoStatistics>();
    tracker_ = std::make_shared<ScanTracker>(
        "testTracker", nullptr, 256 << 10 /* 256KB */);
    rootPool_ = memoryManager()->addRootPool();
    pool_ = rootPool_->addLeafChild("DirectBufferedInputTest");
  }

  void TearDown() override {
    executor_.reset();
  }

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::shared_ptr<MemoryPool> rootPool_;
  std::shared_ptr<MemoryPool> pool_;
  std::shared_ptr<IoStatistics> ioStatistics_;
  std::shared_ptr<ScanTracker> tracker_;
};

TEST_F(DirectBufferedInputTest, reset) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }

  // Test both tiny and non-tiny region sizes.
  for (const bool tinyRegion : {false, true}) {
    SCOPED_TRACE(fmt::format("tinyRegion: {}", tinyRegion));

    const uint64_t regionSize = tinyRegion
        ? DirectBufferedInput::kTinySize - 100
        : DirectBufferedInput::kTinySize + 1000;

    auto readFile = std::make_shared<InMemoryReadFile>(content);

    io::ReaderOptions readerOptions(pool_.get());
    readerOptions.setLoadQuantum(1 << 20);

    auto& ids = fileIds();
    StringIdLease fileId(ids, "testFile");
    StringIdLease groupId(ids, "testGroup");

    DirectBufferedInput input(
        readFile,
        MetricsLog::voidLog(),
        std::move(fileId),
        tracker_,
        std::move(groupId),
        ioStatistics_,
        nullptr,
        executor_.get(),
        readerOptions);

    // First round: enqueue and load streams.
    auto stream1 = input.enqueue(common::Region{0, regionSize}, nullptr);
    auto stream2 =
        input.enqueue(common::Region{regionSize, regionSize}, nullptr);
    ASSERT_NE(stream1, nullptr);
    ASSERT_NE(stream2, nullptr);

    input.load(LogType::TEST);

    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    // Consume streams.
    auto next1 = getNext(*stream1);
    ASSERT_TRUE(next1.has_value());
    EXPECT_EQ(next1.value(), content.substr(0, regionSize));
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 1);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    auto next2 = getNext(*stream2);
    ASSERT_TRUE(next2.has_value());
    EXPECT_EQ(next2.value(), content.substr(regionSize, regionSize));
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    stream1.reset();
    stream2.reset();

    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Reset the input.
    input.reset();
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 0);
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Second round: enqueue different regions after reset.
    auto stream3 =
        input.enqueue(common::Region{2 * regionSize, regionSize}, nullptr);
    auto stream4 =
        input.enqueue(common::Region{3 * regionSize, regionSize}, nullptr);
    ASSERT_NE(stream3, nullptr);
    ASSERT_NE(stream4, nullptr);

    input.load(LogType::TEST);

    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    auto next3 = getNext(*stream3);
    ASSERT_TRUE(next3.has_value());
    EXPECT_EQ(next3.value(), content.substr(2 * regionSize, regionSize));

    auto next4 = getNext(*stream4);
    ASSERT_TRUE(next4.has_value());
    EXPECT_EQ(next4.value(), content.substr(3 * regionSize, regionSize));

    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    // Reset the input.
    input.reset();

    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

    stream3.reset();
    stream4.reset();
    ASSERT_EQ(pool_->usedBytes(), 0);
  }
}

TEST_F(DirectBufferedInputTest, readAfterReset) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }

  // Test both tiny and non-tiny region sizes.
  for (const bool tinyRegion : {false, true}) {
    SCOPED_TRACE(fmt::format("tinyRegion: {}", tinyRegion));

    const uint64_t regionSize = tinyRegion
        ? DirectBufferedInput::kTinySize - 100
        : DirectBufferedInput::kTinySize + 1000;

    auto readFile = std::make_shared<InMemoryReadFile>(content);

    io::ReaderOptions readerOptions(pool_.get());
    readerOptions.setLoadQuantum(1 << 20);

    auto& ids = fileIds();
    StringIdLease fileId(ids, "testFile");
    StringIdLease groupId(ids, "testGroup");

    DirectBufferedInput input(
        readFile,
        MetricsLog::voidLog(),
        std::move(fileId),
        tracker_,
        std::move(groupId),
        ioStatistics_,
        nullptr,
        executor_.get(),
        readerOptions);

    // Enqueue and load streams.
    auto stream1 = input.enqueue(common::Region{0, regionSize}, nullptr);
    auto stream2 =
        input.enqueue(common::Region{regionSize, regionSize}, nullptr);
    ASSERT_NE(stream1, nullptr);
    ASSERT_NE(stream2, nullptr);

    input.load(LogType::TEST);

    // Reset the input before reading from streams.
    input.reset();

    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

    // Read from streams after reset - data should still be available.
    auto next1 = getNext(*stream1);
    ASSERT_TRUE(next1.has_value());
    EXPECT_EQ(next1.value(), content.substr(0, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    auto next2 = getNext(*stream2);
    ASSERT_TRUE(next2.has_value());
    EXPECT_EQ(next2.value(), content.substr(regionSize, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    stream1.reset();
    stream2.reset();
    while (pool_->usedBytes() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

DEBUG_ONLY_TEST_F(DirectBufferedInputTest, resetInputWithBeforeLoading) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }

  // Test both tiny and non-tiny region sizes.
  for (const bool tinyRegion : {false, true}) {
    SCOPED_TRACE(fmt::format("tinyRegion: {}", tinyRegion));

    const uint64_t regionSize = tinyRegion
        ? DirectBufferedInput::kTinySize - 100
        : DirectBufferedInput::kTinySize + 1000;

    auto readFile = std::make_shared<InMemoryReadFile>(content);

    io::ReaderOptions readerOptions(pool_.get());
    readerOptions.setLoadQuantum(1 << 20);

    auto& ids = fileIds();
    StringIdLease fileId(ids, "testFile");
    StringIdLease groupId(ids, "testGroup");

    DirectBufferedInput input(
        readFile,
        MetricsLog::voidLog(),
        std::move(fileId),
        tracker_,
        std::move(groupId),
        ioStatistics_,
        nullptr,
        executor_.get(),
        readerOptions);

    // Block the coalesced load to verify references are held.
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
    auto stream1 = input.enqueue(common::Region{0, regionSize}, nullptr);
    auto stream2 =
        input.enqueue(common::Region{regionSize, regionSize}, nullptr);
    ASSERT_NE(stream1, nullptr);
    ASSERT_NE(stream2, nullptr);

    ASSERT_EQ(pool_->usedBytes(), 0);
    input.load(LogType::TEST);
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Wait for the load to start (but it's blocked).
    loadStarted.wait();
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Verify coalesced load references are held.
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    // Reset the input while load is pending.
    input.reset();
    ASSERT_EQ(pool_->usedBytes(), 0);

    // After reset, internal tracking should be cleared.
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

    // Allow the load to proceed but cancelled.
    loadAllowed.post();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Read from streams - data should be available after load completes.
    auto next1 = getNext(*stream1);
    ASSERT_TRUE(next1.has_value());
    EXPECT_EQ(next1.value(), content.substr(0, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    auto next2 = getNext(*stream2);
    ASSERT_TRUE(next2.has_value());
    EXPECT_EQ(next2.value(), content.substr(regionSize, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }
    stream1.reset();
    stream2.reset();
    ASSERT_EQ(pool_->usedBytes(), 0);
  }
}

DEBUG_ONLY_TEST_F(DirectBufferedInputTest, resetInputWithAfterLoading) {
  constexpr int32_t kContentSize = 4 << 20; // 4MB
  std::string content;
  content.resize(kContentSize);
  for (int32_t i = 0; i < kContentSize; ++i) {
    content[i] = static_cast<char>('a' + (i % 26));
  }

  // Test both tiny and non-tiny region sizes.
  for (const bool tinyRegion : {false, true}) {
    SCOPED_TRACE(fmt::format("tinyRegion: {}", tinyRegion));

    const uint64_t regionSize = tinyRegion
        ? DirectBufferedInput::kTinySize - 100
        : DirectBufferedInput::kTinySize + 1000;

    auto readFile = std::make_shared<InMemoryReadFile>(content);

    io::ReaderOptions readerOptions(pool_.get());
    readerOptions.setLoadQuantum(1 << 20);

    auto& ids = fileIds();
    StringIdLease fileId(ids, "testFile");
    StringIdLease groupId(ids, "testGroup");

    DirectBufferedInput input(
        readFile,
        MetricsLog::voidLog(),
        std::move(fileId),
        tracker_,
        std::move(groupId),
        ioStatistics_,
        nullptr,
        executor_.get(),
        readerOptions);

    // Block the coalesced load to verify references are held.
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
    auto stream1 = input.enqueue(common::Region{0, regionSize}, nullptr);
    auto stream2 =
        input.enqueue(common::Region{regionSize, regionSize}, nullptr);
    ASSERT_NE(stream1, nullptr);
    ASSERT_NE(stream2, nullptr);

    ASSERT_EQ(pool_->usedBytes(), 0);
    input.load(LogType::TEST);
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Wait for the load to start (but it's blocked).
    loadStarted.wait();
    ASSERT_EQ(pool_->usedBytes(), 0);

    // Verify coalesced load references are held.
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 2);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 1);

    // Reset the input while load is pending.
    input.reset();
    ASSERT_EQ(pool_->usedBytes(), 0);

    // After reset, internal tracking should be cleared.
    EXPECT_EQ(input.testingStreamToCoalescedLoadSize(), 0);
    EXPECT_EQ(input.testingCoalescedLoads().size(), 0);

    // Allow the load to proceed without cancelling.
    loadAllowed.post();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Read from streams - data should be available after load completes.
    auto next1 = getNext(*stream1);
    ASSERT_TRUE(next1.has_value());
    EXPECT_EQ(next1.value(), content.substr(0, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }

    auto next2 = getNext(*stream2);
    ASSERT_TRUE(next2.has_value());
    EXPECT_EQ(next2.value(), content.substr(regionSize, regionSize));
    if (tinyRegion) {
      ASSERT_EQ(pool_->usedBytes(), 0);
    } else {
      ASSERT_GT(pool_->usedBytes(), 0);
    }
    stream1.reset();
    stream2.reset();
    while (pool_->usedBytes() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

} // namespace
