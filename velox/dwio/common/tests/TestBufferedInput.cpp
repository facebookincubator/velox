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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/dwio/common/BufferedInput.h"

using namespace facebook::velox::dwio::common;
using facebook::velox::common::Region;
using namespace ::testing;

namespace {

class ReadFileMock : public ::facebook::velox::ReadFile {
 public:
  virtual ~ReadFileMock() override = default;

  MOCK_METHOD(
      std::string_view,
      pread,
      (uint64_t offset, uint64_t length, void* FOLLY_NONNULL buf),
      (const, override));

  MOCK_METHOD(bool, shouldCoalesce, (), (const, override));
  MOCK_METHOD(uint64_t, size, (), (const, override));
  MOCK_METHOD(uint64_t, memoryUsage, (), (const, override));
  MOCK_METHOD(std::string, getName, (), (const, override));
  MOCK_METHOD(uint64_t, getNaturalReadSize, (), (const, override));
  MOCK_METHOD(
      void,
      preadv,
      (folly::Range<const Region*> regions, folly::Range<folly::IOBuf*> iobufs),
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
    EXPECT_CALL(file, pread(read.offset, read.length, _))
        .Times(1)
        .WillOnce(
            [content](uint64_t offset, uint64_t length, void* buf)
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
  EXPECT_CALL(file, preadv(_, _))
      .Times(1)
      .WillOnce([content, reads](
                    folly::Range<const Region*> regions,
                    folly::Range<folly::IOBuf*> iobufs) {
        ASSERT_EQ(regions.size(), reads.size());
        for (size_t i = 0; i < reads.size(); ++i) {
          const auto& region = regions[i];
          const auto& read = reads[i];
          auto& iobuf = iobufs[i];
          ASSERT_EQ(region.offset, read.offset);
          ASSERT_EQ(region.length, read.length);
          if (!read.label.empty()) {
            EXPECT_EQ(read.label, region.label);
          }
          ASSERT_LE(region.offset + region.length, content.size());
          iobuf = folly::IOBuf(
              folly::IOBuf::COPY_BUFFER,
              content.data() + region.offset,
              region.length);
        }
      });
}

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

} // namespace

TEST(TestBufferedInput, AllowMoveConstructor) {
  auto readFileMock = std::make_shared<ReadFileMock>();
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput a(readFileMock, *pool);
  BufferedInput b(std::move(a));
}

TEST(TestBufferedInput, ZeroLengthStream) {
  auto readFile =
      std::make_shared<facebook::velox::InMemoryReadFile>(std::string());
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(readFile, *pool);
  auto ret = input.enqueue({0, 0});
  EXPECT_NE(ret, nullptr);
  const void* buf = nullptr;
  int32_t size = 1;
  EXPECT_FALSE(ret->Next(&buf, &size));
  EXPECT_EQ(size, 0);
}

TEST(TestBufferedInput, UseRead) {
  std::string content = "hello";
  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreads(*readFileMock, content, {{0, 5}});
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  // Use read
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      10,
      /* wsVRLoad = */ false);
  auto ret = input.enqueue({0, 5});
  ASSERT_NE(ret, nullptr);

  input.load(LogType::TEST);

  auto next = getNext(*ret);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content);
}

TEST(TestBufferedInput, UseVRead) {
  std::string content = "hello";
  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(*readFileMock, content, {{0, 5}});
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  // Use vread
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      10,
      /* wsVRLoad = */ true);
  auto ret = input.enqueue({0, 5});
  ASSERT_NE(ret, nullptr);

  input.load(LogType::TEST);

  auto next = getNext(*ret);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next.value(), content);
}

TEST(TestBufferedInput, WillMerge) {
  std::string content = "hello world";
  auto readFileMock = std::make_shared<ReadFileMock>();

  // Will merge because the distance is 1 and max distance to merge is 10.
  // Expect only one call.
  expectPreads(*readFileMock, content, {{0, 11}});

  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      10, // Will merge if distance <= 10
      /* wsVRLoad = */ false);

  auto ret1 = input.enqueue({0, 5});
  auto ret2 = input.enqueue({6, 5});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);

  input.load(LogType::TEST);

  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "hello");

  auto next2 = getNext(*ret2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "world");
}

TEST(TestBufferedInput, WontMerge) {
  std::string content = "hello  world"; // two spaces
  auto readFileMock = std::make_shared<ReadFileMock>();

  // Won't merge because the distance is 2 and max distance to merge is 1.
  // Expect two calls
  expectPreads(*readFileMock, content, {{0, 5}, {7, 5}});

  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ false);

  auto ret1 = input.enqueue({0, 5});
  auto ret2 = input.enqueue({7, 5});
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret2, nullptr);

  input.load(LogType::TEST);

  auto next1 = getNext(*ret1);
  ASSERT_TRUE(next1.has_value());
  EXPECT_EQ(next1.value(), "hello");

  auto next2 = getNext(*ret2);
  ASSERT_TRUE(next2.has_value());
  EXPECT_EQ(next2.value(), "world");
}

TEST(TestBufferedInput, ReadSorting) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<Region> regions = {{6, 3}, {24, 3}, {3, 3}, {0, 3}, {29, 3}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreads(*readFileMock, content, {{0, 9}, {24, 3}, {29, 3}});
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ false);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  for (auto& region : regions) {
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}

TEST(TestBufferedInput, VReadSorting) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<Region> regions = {{6, 3}, {24, 3}, {3, 3}, {0, 3}, {29, 3}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(
      *readFileMock, content, {{0, 3}, {3, 3}, {6, 3}, {24, 3}, {29, 3}});
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ true);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  for (auto& region : regions) {
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}

TEST(TestBufferedInput, VReadSortingWithLabels) {
  std::string content = "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqq";
  std::vector<std::string> l = {"a", "b", "c", "d", "e"};
  std::vector<Region> regions = {
      {6, 3, l[2]}, {24, 3, l[3]}, {3, 3, l[1]}, {0, 3, l[0]}, {29, 3, l[4]}};

  auto readFileMock = std::make_shared<ReadFileMock>();
  expectPreadvs(
      *readFileMock,
      content,
      {{0, 3, l[0]}, {3, 3, l[1]}, {6, 3, l[2]}, {24, 3, l[3]}, {29, 3, l[4]}});
  auto pool = facebook::velox::memory::addDefaultLeafMemoryPool();
  BufferedInput input(
      readFileMock,
      *pool,
      MetricsLog::voidLog(),
      nullptr,
      1, // Will merge if distance <= 1
      /* wsVRLoad = */ true);

  std::vector<std::pair<std::unique_ptr<SeekableInputStream>, std::string>>
      result;
  result.reserve(regions.size());
  for (auto& region : regions) {
    auto ret = input.enqueue(region);
    ASSERT_NE(ret, nullptr);
    result.push_back(
        {std::move(ret), content.substr(region.offset, region.length)});
  }

  input.load(LogType::TEST);

  for (auto& r : result) {
    auto next = getNext(*r.first);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next.value(), r.second);
  }
}
