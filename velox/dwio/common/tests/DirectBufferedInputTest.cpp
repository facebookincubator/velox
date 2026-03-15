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

#include "velox/dwio/common/DirectBufferedInput.h"

#include <fmt/core.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <thread>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/tests/TestUtils.h"
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

TEST_F(DirectBufferedInputTest, preloadCalledTwice) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadTwice");
  StringIdLease groupId(ids, "preloadTwiceGroup");

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

  input.preload();
  ASSERT_TRUE(input.preloaded());
  VELOX_ASSERT_THROW(input.preload(), "preload() called more than once");
}

TEST_F(DirectBufferedInputTest, preloadAfterEnqueue) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadAfterEnqueue");
  StringIdLease groupId(ids, "preloadAfterEnqueueGroup");

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

  input.enqueue({0, 100}, nullptr);
  VELOX_ASSERT_THROW(
      input.preload(), "preload() must be called before enqueue()");
}

TEST_F(DirectBufferedInputTest, preloadedDataWithoutPreload) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadedDataNoPreload");
  StringIdLease groupId(ids, "preloadedDataNoPreloadGroup");

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

  VELOX_ASSERT_THROW(
      input.preloadedData(0, 100), "preloadedData() called without preload");
}

TEST_F(DirectBufferedInputTest, preloadedDataOffsetOutOfRange) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());
  auto& ids = fileIds();
  StringIdLease fileId(ids, "preloadedDataOOR");
  StringIdLease groupId(ids, "preloadedDataOORGroup");

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

  input.preload();
  VELOX_ASSERT_THROW(
      input.preloadedData(1024, 100), "Offset exceeds preloaded size");
}

TEST_F(DirectBufferedInputTest, preload) {
  struct TestParam {
    uint64_t fileSize;
    std::string debugString() const {
      return fmt::format("fileSize {}", fileSize);
    }
  };
  std::vector<TestParam> testSettings = {
      // Tiny file (below kTinySize).
      {DirectBufferedInput::kTinySize - 100},
      // Non-tiny file (above kTinySize).
      {DirectBufferedInput::kTinySize + 1000},
      // Larger file.
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

    ASSERT_EQ(readFile->numReads(), 0);
    EXPECT_FALSE(input.preloaded());

    input.preload();

    EXPECT_TRUE(input.preloaded());

    ASSERT_EQ(readFile->numReads(), 1);

    // Enqueue sub-region streams and read from preloaded data.
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
    ASSERT_EQ(readFile->numReads(), 1);
  }
}

TEST_F(DirectBufferedInputTest, preloadedData) {
  struct TestParam {
    uint64_t fileSize;
    uint64_t offset;
    uint64_t length;
    // Expected size of the returned range. For tiny files, this equals
    // min(length, fileSize - offset). For large files, it may be smaller
    // due to non-contiguous allocation run boundaries.
    bool expectTinyPath;
    std::string debugString() const {
      return fmt::format(
          "fileSize {}, offset {}, length {}, expectTinyPath {}",
          fileSize,
          offset,
          length,
          expectTinyPath);
    }
  };

  std::vector<TestParam> testSettings = {
      // Tiny file: read from beginning.
      {1000, 0, 500, true},
      // Tiny file: read from middle.
      {1000, 400, 200, true},
      // Tiny file: length exceeds remaining bytes.
      {1000, 800, 500, true},
      // Tiny file: length zero.
      {1000, 0, 0, true},
      // Tiny file: read last byte.
      {1000, 999, 100, true},
      // Tiny file: read entire file.
      {1000, 0, 1000, true},
      // Large file: read from beginning.
      {1 << 20, 0, 4096, false},
      // Large file: read from middle.
      {1 << 20, 100'000, 4096, false},
      // Large file: length exceeds remaining bytes.
      {1 << 20, (1 << 20) - 100, 4096, false},
      // Large file: length zero.
      {1 << 20, 0, 0, false},
      // Large file: read last byte.
      {1 << 20, (1 << 20) - 1, 100, false},
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
    auto& ids = fileIds();
    StringIdLease fileId(
        ids,
        fmt::format(
            "preloadedData_{}_{}_{}",
            testData.fileSize,
            testData.offset,
            testData.length));
    StringIdLease groupId(
        ids,
        fmt::format(
            "preloadedDataGroup_{}_{}_{}",
            testData.fileSize,
            testData.offset,
            testData.length));

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

    ASSERT_EQ(readFile->numReads(), 0);
    input.preload();
    ASSERT_EQ(readFile->numReads(), 1);

    const auto range = input.preloadedData(testData.offset, testData.length);

    // No additional file reads — preloadedData serves from memory.
    ASSERT_EQ(readFile->numReads(), 1);

    const auto expectedAvailable = std::min<uint64_t>(
        testData.length, testData.fileSize - testData.offset);

    if (testData.length == 0) {
      EXPECT_EQ(range.size(), 0);
      continue;
    }

    // For tiny files, the data is contiguous so we get all available bytes.
    // For large files, we may get fewer bytes due to allocation run boundaries.
    EXPECT_GT(range.size(), 0);
    EXPECT_LE(range.size(), expectedAvailable);
    if (testData.expectTinyPath) {
      EXPECT_EQ(range.size(), expectedAvailable);
    }

    // Verify data content matches original file.
    EXPECT_EQ(
        std::string_view(range.data(), range.size()),
        std::string_view(content.data() + testData.offset, range.size()));
  }
}

TEST_F(DirectBufferedInputTest, hasCache) {
  std::string content(1024, 'x');
  auto readFile = std::make_shared<InMemoryReadFile>(content);

  io::ReaderOptions readerOptions(pool_.get());

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

  ASSERT_FALSE(input.hasCache());
  VELOX_ASSERT_THROW(
      input.cacheRegion(0, 10, std::string_view("0123456789")),
      "cacheRegion requires a backing cache");
  VELOX_ASSERT_THROW(
      input.findCachedRegion(0), "findCachedRegion requires a backing cache");
}

} // namespace
