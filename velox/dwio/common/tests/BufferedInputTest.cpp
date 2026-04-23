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

#include "velox/dwio/common/BufferedInput.h"

#include <fmt/core.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/tests/TestUtils.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/dwrf/test/TestReadFile.h"

using namespace facebook::velox::dwio::common;
namespace cache = facebook::velox::cache;
using facebook::velox::StringIdLease;
using facebook::velox::common::Region;
using namespace facebook::velox::memory;
using namespace ::testing;

namespace {

class ReadFileMock : public ::facebook::velox::ReadFile {
 public:
  virtual ~ReadFileMock() override = default;

// On Centos9 the gtest mock header doesn't initialize the
// buffer_ member in MatcherBase correctly - the default constructor only
// initializes one: /usr/include/gtest/gtest-matchers.h:302:33 resulting in
// error:
// '<unnamed>.testing::Matcher<const
// facebook::velox::FileIoContext&>::<unnamed>.testing::internal::MatcherBase<const
// facebook::velox::FileIoContext&>::buffer_' is used uninitialized
// [-Werror=uninitialized]
//  302 |       : vtable_(other.vtable_), buffer_(other.buffer_) {
// Fix: https://github.com/google/googletest/pull/3797
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  MOCK_METHOD(
      std::string_view,
      pread,
      (uint64_t offset,
       uint64_t length,
       void* buf,
       (const facebook::velox::FileIoContext&)context),
      (const, override));

  MOCK_METHOD(bool, shouldCoalesce, (), (const, override));
  MOCK_METHOD(uint64_t, size, (), (const, override));
  MOCK_METHOD(uint64_t, memoryUsage, (), (const, override));
  MOCK_METHOD(std::string, getName, (), (const, override));
  MOCK_METHOD(uint64_t, getNaturalReadSize, (), (const, override));
  MOCK_METHOD(
      uint64_t,
      preadv,
      (folly::Range<const Region*> regions,
       folly::Range<folly::IOBuf*> iobufs,
       (const facebook::velox::FileIoContext&)context),
      (const, override));
};

void expectPreads(
    ReadFileMock& file,
    std::string_view content,
    std::vector<Region> reads) {
  EXPECT_CALL(file, getName()).WillRepeatedly(Return("mock_name"));
  EXPECT_CALL(file, size()).WillRepeatedly(Return(content.size()));
  for (auto& read : reads) {
    ASSERT_GE(content.size(), read.offset + read.length);
    EXPECT_CALL(file, pread(read.offset, read.length, _, _))
        .Times(1)
        .WillOnce(
            [content](
                uint64_t offset,
                uint64_t length,
                void* buf,
                const facebook::velox::FileIoContext& context)
                -> std::string_view {
              memcpy(buf, content.data() + offset, length);
              return {content.data() + offset, length};
            });
  }
}

void expectPreadvs(
    ReadFileMock& file,
    std::string_view content,
    std::vector<Region> reads) {
  EXPECT_CALL(file, getName()).WillRepeatedly(Return("mock_name"));
  EXPECT_CALL(file, size()).WillRepeatedly(Return(content.size()));
  EXPECT_CALL(file, preadv(_, _, _))
      .Times(1)
      .WillOnce(
          [content, reads](
              folly::Range<const Region*> regions,
              folly::Range<folly::IOBuf*> iobufs,
              const facebook::velox::FileIoContext& context) -> uint64_t {
            EXPECT_EQ(regions.size(), reads.size());
            uint64_t length = 0;
            for (size_t i = 0; i < reads.size(); ++i) {
              const auto& region = regions[i];
              const auto& read = reads[i];
              auto& iobuf = iobufs[i];
              length += region.length;
              EXPECT_EQ(region.offset, read.offset);
              EXPECT_EQ(region.length, read.length);
              if (!read.label.empty()) {
                EXPECT_EQ(read.label, region.label);
              }
              EXPECT_LE(region.offset + region.length, content.size());
              iobuf = folly::IOBuf(
                  folly::IOBuf::COPY_BUFFER,
                  content.data() + region.offset,
                  region.length);
            }

            return length;
          });
}
#pragma GCC diagnostic pop

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

class BufferedInputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  const std::shared_ptr<MemoryPool> pool_ = memoryManager()->addLeafPool();
};

TEST_F(BufferedInputTest, hasCache) {
  auto readFile =
      std::make_shared<facebook::velox::InMemoryReadFile>(std::string("test"));
  BufferedInput input(readFile, *pool_);
  // Base BufferedInput does not have cache.
  EXPECT_FALSE(input.hasCache());

  // Cache APIs throw when there is no backing cache.
  VELOX_ASSERT_THROW(
      input.cacheRegion(0, 4, std::string_view("test")),
      "cacheRegion requires a backing cache");
  VELOX_ASSERT_THROW(
      input.findCachedRegion(0), "findCachedRegion requires a backing cache");
}

TEST_F(BufferedInputTest, cachedRegion) {
  auto dataCache = cache::AsyncDataCache::create(memoryManager()->allocator());

  // Empty pin throws.
  VELOX_ASSERT_THROW(
      CachedRegion(cache::CachePin{}),
      "CachedRegion requires a non-empty cache pin");

  // Exclusive pin throws.
  auto& ids = facebook::velox::fileIds();
  {
    StringIdLease fileId(ids, "exclusiveTestFile");
    cache::RawFileCacheKey key{fileId.id(), 0};
    auto pin = dataCache->findOrCreate(key, 100, nullptr);
    ASSERT_FALSE(pin.empty());
    ASSERT_TRUE(pin.checkedEntry()->isExclusive());
    VELOX_ASSERT_THROW(
        CachedRegion(std::move(pin)),
        "CachedRegion requires a shared (non-exclusive) cache pin");
  }

  struct TestParam {
    uint64_t entrySize;
    bool expectTinyData;

    std::string debugString() const {
      return fmt::format(
          "entrySize {}, expectTinyData {}", entrySize, expectTinyData);
    }
  };
  std::vector<TestParam> testSettings = {
      // Small entry uses tinyData (single contiguous range).
      {100, true},
      // Entry just below tinyData boundary still uses tinyData.
      {cache::AsyncDataCacheEntry::kTinyDataSize - 1, true},
      // Entry at tinyData boundary uses allocation.
      {cache::AsyncDataCacheEntry::kTinyDataSize, false},
      // One page allocation (single run).
      {AllocationTraits::kPageSize, false},
      // Large allocation (possibly multiple runs).
      {128 << 10, false},
  };

  for (size_t i = 0; i < testSettings.size(); ++i) {
    const auto& testData = testSettings[i];
    SCOPED_TRACE(testData.debugString());

    const auto entrySize = testData.entrySize;
    std::string expected(entrySize, '\0');
    for (uint64_t j = 0; j < entrySize; ++j) {
      expected[j] = static_cast<char>('a' + (j % 26));
    }

    StringIdLease fileId(ids, fmt::format("cachedRegionTestFile_{}", i));
    cache::RawFileCacheKey key{fileId.id(), 0};
    auto pin = dataCache->findOrCreate(key, entrySize, nullptr);
    ASSERT_FALSE(pin.empty());
    auto* entry = pin.checkedEntry();
    ASSERT_TRUE(entry->isExclusive());

    // Populate the entry with test data.
    if (testData.expectTinyData) {
      ASSERT_NE(entry->tinyData(), nullptr);
      memcpy(entry->tinyData(), expected.data(), entrySize);
    } else {
      auto& allocation = entry->data();
      ASSERT_GT(allocation.numRuns(), 0);
      uint64_t offset = 0;
      for (int i = 0; i < allocation.numRuns() && offset < entrySize; ++i) {
        auto run = allocation.runAt(i);
        const uint64_t bytes = run.numPages() * AllocationTraits::kPageSize;
        const uint64_t copySize = std::min(bytes, entrySize - offset);
        memcpy(run.data<char>(), expected.data() + offset, copySize);
        offset += copySize;
      }
    }
    entry->setExclusiveToShared();

    CachedRegion region(std::move(pin));
    EXPECT_EQ(region.size(), entrySize);
    ASSERT_FALSE(region.ranges().empty());

    if (testData.expectTinyData) {
      EXPECT_EQ(region.ranges().size(), 1);
    }

    // Verify content through ranges.
    uint64_t verified = 0;
    for (const auto& range : region.ranges()) {
      EXPECT_EQ(
          std::string_view(range.data(), range.size()),
          std::string_view(expected.data() + verified, range.size()));
      verified += range.size();
    }
    EXPECT_EQ(verified, entrySize);

    // Verify toIOBuf produces identical content.
    auto iobuf = region.toIOBuf();
    EXPECT_EQ(iobuf.computeChainDataLength(), entrySize);
    iobuf.coalesce();
    EXPECT_EQ(
        std::string_view(
            reinterpret_cast<const char*>(iobuf.data()), iobuf.length()),
        expected);
  }
}

TEST_F(BufferedInputTest, zeroLengthStream) {
  auto readFile =
      std::make_shared<facebook::velox::InMemoryReadFile>(std::string());
  BufferedInput input(readFile, *pool_);
  auto ret = input.enqueue({0, 0});
  EXPECT_EQ(input.nextFetchSize(), 0);
  EXPECT_NE(ret, nullptr);
  const void* buf = nullptr;
  int32_t size = 1;
  EXPECT_FALSE(ret->Next(&buf, &size));
  EXPECT_EQ(size, 0);
}

TEST_F(BufferedInputTest, useRead) {
  std::string content = "hello";
  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreads(*readFileMock, content, {{0, 5}});
  // Use read
  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10,
      /* wsVRLoad = */ false);
  auto ret = input.enqueue({0, 5});
  ASSERT_NE(ret, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 5);
  input.load(LogType::TEST);

  auto next = getNext(*ret);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content);
}

TEST_F(BufferedInputTest, useVRead) {
  std::string content = "hello";
  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(*readFileMock, content, {{0, 5}});
  // Use vread
  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10,
      /* wsVRLoad = */ true);
  auto ret = input.enqueue({0, 5});
  ASSERT_NE(ret, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 5);
  input.load(LogType::TEST);

  auto next = getNext(*ret);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content);
}

TEST_F(BufferedInputTest, willMerge) {
  std::string content = "hello world";
  auto readFileMock = std::make_shared<ReadFileMock>();

  // Will merge because the distance is 1 and max distance to merge is 10.
  // Expect only one call.
  expectPreads(*readFileMock, content, {{0, 11}});

  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10, // Will merge if distance <= 10
      /* wsVRLoad = */ false);

  auto ret1 = input.enqueue({0, 5});
  auto ret2 = input.enqueue({6, 5});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 10);
  input.load(LogType::TEST);

  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "hello");

  auto next2 = getNext(*ret2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "world");
}

TEST_F(BufferedInputTest, wontMerge) {
  std::string content = "hello  world"; // two spaces
  auto readFileMock = std::make_shared<ReadFileMock>();

  // Won't merge because the distance is 2 and max distance to merge is 1.
  // Expect two calls
  expectPreads(*readFileMock, content, {{0, 5}, {7, 5}});

  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ false);

  auto ret1 = input.enqueue({0, 5});
  auto ret2 = input.enqueue({7, 5});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 10);
  input.load(LogType::TEST);

  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "hello");

  auto next2 = getNext(*ret2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "world");
}

TEST_F(BufferedInputTest, readSorting) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<Region> regions = {{6, 3}, {24, 3}, {3, 3}, {0, 3}, {29, 3}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreads(*readFileMock, content, {{0, 9}, {24, 3}, {29, 3}});
  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ false);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  int64_t bytesToRead = 0;
  for (auto& region : regions) {
    bytesToRead += region.length;
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  EXPECT_EQ(input.nextFetchSize(), bytesToRead);
  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}

TEST_F(BufferedInputTest, VreadSorting) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<Region> regions = {{6, 3}, {24, 3}, {3, 3}, {0, 3}, {29, 3}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(
      *readFileMock, content, {{0, 3}, {3, 3}, {6, 3}, {24, 3}, {29, 3}});
  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ true);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  int64_t bytesToRead = 0;
  for (auto& region : regions) {
    bytesToRead += region.length;
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  EXPECT_EQ(input.nextFetchSize(), bytesToRead);
  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}

TEST_F(BufferedInputTest, VreadSortingWithLabels) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<std::string> l = {"a", "b", "c", "d", "e"};
  std::vector<Region> regions = {
      {6, 3, l[2]}, {24, 3, l[3]}, {3, 3, l[1]}, {0, 3, l[0]}, {29, 3, l[4]}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(
      *readFileMock,
      content,
      {{0, 3, l[0]}, {3, 3, l[1]}, {6, 3, l[2]}, {24, 3, l[3]}, {29, 3, l[4]}});
  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ true);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  int64_t bytesToRead = 0;
  for (auto& region : regions) {
    bytesToRead += region.length;
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  EXPECT_EQ(input.nextFetchSize(), bytesToRead);
  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}

TEST_F(BufferedInputTest, resetAfterAllStreamsConsumed) {
  const std::string content = "aaabbbcccdddeee";
  auto readFileMock = std::make_shared<ReadFileMock>();

  // First round: enqueue and consume all streams.
  expectPreads(*readFileMock, content, {{0, 6}});

  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10,
      /* wsVRLoad = */ false);

  const auto ret1 = input.enqueue({0, 3});
  const auto ret2 = input.enqueue({3, 3});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);

  input.load(LogType::TEST);

  // Consume all streams.
  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "aaa");

  const auto next2 = getNext(*ret2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "bbb");

  // Reset and enqueue new streams.
  input.reset();
  EXPECT_EQ(input.nextFetchSize(), 0);

  // Second round: enqueue different regions after reset.
  expectPreads(*readFileMock, content, {{6, 9}});

  const auto ret3 = input.enqueue({6, 3});
  const auto ret4 = input.enqueue({9, 6});
  ASSERT_NE(ret3, nullptr);
  ASSERT_NE(ret4, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 9);
  input.load(LogType::TEST);

  const auto next3 = getNext(*ret3);
  ASSERT_TRUE(next3.has_value());
  EXPECT_EQ(next3.value(), "ccc");

  const auto next4 = getNext(*ret4);
  ASSERT_TRUE(next4.has_value());
  EXPECT_EQ(next4.value(), "dddeee");
}

TEST_F(BufferedInputTest, resetAfterPartialStreamsConsumed) {
  const std::string content = "aaabbbcccdddeee";
  auto readFileMock = std::make_shared<ReadFileMock>();

  // First round: enqueue streams but only consume some.
  expectPreads(*readFileMock, content, {{0, 9}});

  BufferedInput input(
      readFileMock,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10,
      /*wsVRLoad=*/false);

  const auto ret1 = input.enqueue({0, 3});
  const auto ret2 = input.enqueue({3, 3});
  const auto ret3 = input.enqueue({6, 3});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);
  ASSERT_NE(ret3, nullptr);

  input.load(LogType::TEST);

  // Only consume the first stream, leave ret2 and ret3 unconsumed.
  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "aaa");

  // Reset without consuming all streams.
  input.reset();
  EXPECT_EQ(input.nextFetchSize(), 0);

  // Second round: enqueue different regions after reset.
  expectPreads(*readFileMock, content, {{9, 6}});

  const auto ret4 = input.enqueue({9, 6});
  ASSERT_NE(ret4, nullptr);

  EXPECT_EQ(input.nextFetchSize(), 6);
  input.load(LogType::TEST);

  const auto next4 = getNext(*ret4);
  ASSERT_TRUE(next4.has_value());
  EXPECT_EQ(next4.value(), "dddeee");
}

TEST_F(BufferedInputTest, preload) {
  std::string content = "hello world, this is preload test data!";
  auto readFile =
      std::make_shared<facebook::velox::tests::utils::CountingReadFile>(
          content);

  BufferedInput input(
      readFile,
      *pool_,
      MetricsLog::voidLog(),
      nullptr,
      nullptr,
      10,
      /* wsVRLoad = */ false);

  ASSERT_EQ(readFile->numReads(), 0);
  EXPECT_FALSE(input.preloaded());

  input.preload();
  EXPECT_TRUE(input.preloaded());

  const auto readsAfterPreload = readFile->numReads();
  ASSERT_GT(readsAfterPreload, 0);

  // After preload, sub-region reads should be served from preloaded data.
  auto stream1 = input.read(0, 5, LogType::FILE);
  auto next1 = getNext(*stream1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "hello");

  auto stream2 = input.read(6, 5, LogType::FILE);
  auto next2 = getNext(*stream2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "world");

  // No additional file reads after preload.
  ASSERT_EQ(readFile->numReads(), readsAfterPreload);
}

class CustomDirectBufferedInput
    : public facebook::velox::dwio::common::DirectBufferedInput {
 public:
  CustomDirectBufferedInput(
      std::shared_ptr<facebook::velox::ReadFile> readFile,
      const facebook::velox::dwio::common::MetricsLogPtr& metricsLog,
      facebook::velox::StringIdLease fileNum,
      std::shared_ptr<facebook::velox::cache::ScanTracker> tracker,
      facebook::velox::StringIdLease groupId,
      std::shared_ptr<facebook::velox::io::IoStatistics> ioStatistics,
      std::shared_ptr<facebook::velox::IoStats> ioStats,
      folly::Executor* executor,
      const facebook::velox::io::ReaderOptions& readerOptions,
      folly::F14FastMap<std::string, std::string> fileReadOps = {})
      : DirectBufferedInput(
            std::move(readFile),
            metricsLog,
            std::move(fileNum),
            std::move(tracker),
            std::move(groupId),
            std::move(ioStatistics),
            std::move(ioStats),
            executor,
            readerOptions,
            std::move(fileReadOps)) {
    VELOX_NYI("Not implemented in CustomBufferedInputBuilder");
  }

  std::unique_ptr<facebook::velox::dwio::common::BufferedInput> clone()
      const override {
    return std::unique_ptr<facebook::velox::dwio::common::BufferedInput>(
        new CustomDirectBufferedInput(
            input_,
            fileNum_,
            tracker_,
            groupId_,
            ioStatistics_,
            ioStats_,
            executor_,
            options_));
  }

 protected:
  // Expose protected members to verify their accessibility.
  using facebook::velox::dwio::common::DirectBufferedInput::coalescedLoads_;
  using facebook::velox::dwio::common::DirectBufferedInput::requests_;

 private:
  CustomDirectBufferedInput(
      std::shared_ptr<facebook::velox::dwio::common::ReadFileInputStream> input,
      facebook::velox::StringIdLease fileNum,
      std::shared_ptr<facebook::velox::cache::ScanTracker> tracker,
      facebook::velox::StringIdLease groupId,
      std::shared_ptr<facebook::velox::io::IoStatistics> ioStatistics,
      std::shared_ptr<facebook::velox::IoStats> ioStats,
      folly::Executor* executor,
      const facebook::velox::io::ReaderOptions& readerOptions)
      : DirectBufferedInput(
            std::move(input),
            std::move(fileNum),
            std::move(tracker),
            std::move(groupId),
            std::move(ioStatistics),
            std::move(ioStats),
            executor,
            readerOptions) {}
};

class CustomBufferedInputBuilder
    : public facebook::velox::connector::hive::BufferedInputBuilder {
 public:
  std::unique_ptr<facebook::velox::dwio::common::BufferedInput> create(
      const facebook::velox::FileHandle& fileHandle,
      const facebook::velox::dwio::common::ReaderOptions& readerOpts,
      const facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<facebook::velox::io::IoStatistics> ioStatistics,
      std::shared_ptr<facebook::velox::IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override {
    auto file = std::make_shared<TestReadFile>(11, 100 << 20, ioStats);
    auto tracker = std::make_shared<facebook::velox::cache::ScanTracker>(
        "", nullptr, /*loadQuantum=*/8 << 20);
    return std::make_unique<CustomDirectBufferedInput>(
        std::move(file),
        facebook::velox::dwio::common::MetricsLog::voidLog(),
        fileHandle.uuid,
        std::move(tracker),
        fileHandle.groupId,
        std::move(ioStatistics),
        std::move(ioStats),
        executor,
        readerOpts,
        fileReadOps);
  }
};

class CustomBufferedInputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
    facebook::velox::connector::hive::BufferedInputBuilder::registerBuilder(
        std::make_shared<CustomBufferedInputBuilder>());
  }

  const std::shared_ptr<MemoryPool> pool_ = memoryManager()->addLeafPool();
};

} // namespace

TEST_F(CustomBufferedInputTest, basic) {
  facebook::velox::FileHandle fileHandle;
  facebook::velox::dwio::common::ReaderOptions readerOpts(pool_.get());
  auto ioStatistics = std::make_shared<facebook::velox::io::IoStatistics>();
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  auto executor = std::make_unique<folly::IOThreadPoolExecutor>(10, 10);

  VELOX_ASSERT_THROW(
      facebook::velox::connector::hive::BufferedInputBuilder::getInstance()
          ->create(
              fileHandle, readerOpts, nullptr, ioStatistics, ioStats, nullptr),
      "Not implemented in CustomBufferedInputBuilder");
}
