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
#include <gtest/gtest.h>

#include <sstream>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/File.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble::index::test {
namespace {

using LookupRequest = IndexLookup::LookupRequest;
using LookupResult = IndexLookup::LookupResult;
using LookupOptions = IndexLookup::LookupOptions;

class IndexLookupTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

 protected:
  static velox::serializer::EncodedKeyBounds makePointKey(
      const std::string& key) {
    return {.lowerKey = key, .upperKey = key};
  }

  static velox::serializer::EncodedKeyBounds makeRangeKey(
      const std::string& lower,
      const std::string& upper) {
    return {.lowerKey = lower, .upperKey = upper};
  }
};

TEST_F(IndexLookupTest, pointLookupRequest) {
  auto request = LookupRequest::pointLookup({"aaa", "bbb"});

  EXPECT_EQ(request.mode(), LookupRequest::Mode::PointLookup);
  EXPECT_EQ(request.size(), 2);
  EXPECT_EQ(request.pointKey(0), "aaa");
  EXPECT_EQ(request.pointKey(1), "bbb");
  EXPECT_FALSE(request.options().rowRange.has_value());
}

TEST_F(IndexLookupTest, rangeScanRequest) {
  std::vector<velox::serializer::EncodedKeyBounds> keys = {
      makeRangeKey("aaa", "zzz")};
  auto request = LookupRequest::rangeScan(keys);

  EXPECT_EQ(request.mode(), LookupRequest::Mode::RangeScan);
  EXPECT_EQ(request.size(), 1);
  EXPECT_EQ(request.rangeBound(0).lowerKey.value(), "aaa");
  EXPECT_EQ(request.rangeBound(0).upperKey.value(), "zzz");
}

TEST_F(IndexLookupTest, withOptions) {
  auto request =
      LookupRequest::pointLookup({"key"}, {.rowRange = RowRange(10, 100)});

  EXPECT_EQ(request.size(), 1);
  EXPECT_TRUE(request.options().rowRange.has_value());
  EXPECT_EQ(request.options().rowRange->startRow, 10);
  EXPECT_EQ(request.options().rowRange->endRow, 100);
}

TEST_F(IndexLookupTest, emptyKeyBoundsThrows) {
  NIMBLE_ASSERT_THROW(LookupRequest::pointLookup({}), "");
  NIMBLE_ASSERT_THROW(LookupRequest::rangeScan({}), "");
}

TEST_F(IndexLookupTest, singleKeyWithResult) {
  std::vector<RowRange> rowRanges = {RowRange(10, 20)};
  std::vector<uint32_t> resultOffsets = {0, 1};
  LookupResult result(std::move(rowRanges), std::move(resultOffsets));

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0].startRow, 10);
  EXPECT_EQ(result[0][0].endRow, 20);
}

TEST_F(IndexLookupTest, singleKeyMultiResults) {
  std::vector<RowRange> rowRanges = {
      RowRange(10, 20), RowRange(50, 60), RowRange(100, 110)};
  std::vector<uint32_t> resultOffsets = {0, 3};
  LookupResult result(std::move(rowRanges), std::move(resultOffsets));

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].size(), 3);
  EXPECT_EQ(result[0][0].startRow, 10);
  EXPECT_EQ(result[0][1].startRow, 50);
  EXPECT_EQ(result[0][2].startRow, 100);
}

TEST_F(IndexLookupTest, singleKeyNoResult) {
  std::vector<RowRange> rowRanges;
  std::vector<uint32_t> resultOffsets = {0, 0};
  LookupResult result(std::move(rowRanges), std::move(resultOffsets));

  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].empty());
}

TEST_F(IndexLookupTest, multipleKeys) {
  std::vector<RowRange> rowRanges = {
      RowRange(0, 10), RowRange(50, 60), RowRange(100, 110)};
  // key0 → 1 result, key1 → 0 results, key2 → 2 results
  std::vector<uint32_t> resultOffsets = {0, 1, 1, 3};
  LookupResult result(std::move(rowRanges), std::move(resultOffsets));

  EXPECT_EQ(result.size(), 3);

  EXPECT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0].startRow, 0);

  EXPECT_TRUE(result[1].empty());

  EXPECT_EQ(result[2].size(), 2);
  EXPECT_EQ(result[2][0].startRow, 50);
  EXPECT_EQ(result[2][1].startRow, 100);
}

TEST_F(IndexLookupTest, allKeysEmpty) {
  std::vector<RowRange> rowRanges;
  std::vector<uint32_t> resultOffsets = {0, 0, 0, 0};
  LookupResult result(std::move(rowRanges), std::move(resultOffsets));

  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].empty());
  EXPECT_TRUE(result[1].empty());
  EXPECT_TRUE(result[2].empty());
  NIMBLE_ASSERT_THROW(result[3], "");
}

TEST_F(IndexLookupTest, tooFewOffsets) {
  NIMBLE_ASSERT_THROW(LookupResult({}, {0}), "");
}

TEST_F(IndexLookupTest, emptyOffsets) {
  NIMBLE_ASSERT_THROW(LookupResult({}, {}), "");
}

TEST_F(IndexLookupTest, offsetsBeyondRowRanges) {
  std::vector<RowRange> rowRanges = {RowRange(0, 10)};
  // Offset 2 references beyond the single-element rowRanges vector.
  NIMBLE_ASSERT_THROW(LookupResult(std::move(rowRanges), {0, 2}), "");
}

TEST_F(IndexLookupTest, offsetsBeyondRowRangesMultiKey) {
  std::vector<RowRange> rowRanges = {RowRange(0, 10), RowRange(20, 30)};
  // Last offset 3 references beyond the 2-element rowRanges vector.
  NIMBLE_ASSERT_THROW(LookupResult(std::move(rowRanges), {0, 1, 3}), "");
}

TEST_F(IndexLookupTest, nonMonotonicOffsets) {
  std::vector<RowRange> rowRanges = {RowRange(0, 10), RowRange(20, 30)};
  // Offsets go backwards: 0, 2, 1 — violates monotonic invariant.
  LookupResult result(std::move(rowRanges), {0, 2, 1});
  EXPECT_EQ(result.size(), 2);
  // First key access is valid.
  EXPECT_EQ(result[0].size(), 2);
  // Second key access triggers DCHECK on non-monotonic offsets.
  NIMBLE_ASSERT_THROW(result[1], "");
}

TEST_F(IndexLookupTest, toString) {
  EXPECT_EQ(toString(IndexType::Cluster), "Cluster");
  EXPECT_EQ(toString(IndexType::Hash), "Hash");
}

TEST_F(IndexLookupTest, ostream) {
  std::ostringstream oss;
  oss << IndexType::Cluster << " " << IndexType::Hash;
  EXPECT_EQ(oss.str(), "Cluster Hash");
}

TEST_F(IndexLookupTest, fmtFormat) {
  EXPECT_EQ(fmt::format("{}", IndexType::Cluster), "Cluster");
  EXPECT_EQ(fmt::format("{}", IndexType::Hash), "Hash");
  EXPECT_EQ(
      fmt::format("type={}, count={}", IndexType::Hash, 42),
      "type=Hash, count=42");
}

TEST_F(IndexLookupTest, unknownIndexType) {
  const auto unknown = static_cast<IndexType>(99);
  NIMBLE_ASSERT_THROW(toString(unknown), "Unknown IndexType: 99");
}

TEST_F(IndexLookupTest, optionsValidateRequiredFields) {
  {
    SCOPED_TRACE("null file");
    IndexLookup::Options options;
    NIMBLE_ASSERT_THROW(options.validate(), "");
  }
  {
    SCOPED_TRACE("null ioOptions");
    auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
    IndexLookup::Options options{.file = file};
    NIMBLE_ASSERT_THROW(options.validate(), "");
  }
}

TEST_F(IndexLookupTest, optionsValidateIndexIoStats) {
  auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
  auto pool = velox::memory::memoryManager()->addLeafPool();
  velox::io::ReaderOptions ioOptions(pool.get());
  IndexLookup::Options options{.file = file, .ioOptions = &ioOptions};
  NIMBLE_ASSERT_THROW(options.validate(), "indexIoStats must be set");
}

TEST_F(IndexLookupTest, optionsValidateFileHandleAndCache) {
  auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
  auto pool = velox::memory::memoryManager()->addLeafPool();
  auto indexIoStats = std::make_shared<velox::io::IoStatistics>();
  velox::io::ReaderOptions ioOptions(pool.get());
  ioOptions.setIndexIoStats(indexIoStats);

  {
    SCOPED_TRACE("direct path (no fileHandle, no cache)");
    IndexLookup::Options options{.file = file, .ioOptions = &ioOptions};
    EXPECT_NO_THROW(options.validate());
  }
  {
    SCOPED_TRACE("fileHandle without cache");
    velox::FileHandle fileHandle;
    fileHandle.file = file;
    IndexLookup::Options options{
        .file = file, .ioOptions = &ioOptions, .fileHandle = &fileHandle};
    NIMBLE_ASSERT_THROW(
        options.validate(), "fileHandle and cache must both be set or neither");
  }
  {
    SCOPED_TRACE("cache without fileHandle");
    IndexLookup::Options options{
        .file = file,
        .ioOptions = &ioOptions,
        .cache = reinterpret_cast<velox::cache::AsyncDataCache*>(0x1)};
    NIMBLE_ASSERT_THROW(
        options.validate(), "fileHandle and cache must both be set or neither");
  }
  {
    SCOPED_TRACE("file mismatch with fileHandle->file");
    auto file2 = std::make_shared<velox::InMemoryReadFile>(std::string{});
    velox::FileHandle fileHandle2;
    fileHandle2.file = file2;
    IndexLookup::Options options{
        .file = file,
        .ioOptions = &ioOptions,
        .fileHandle = &fileHandle2,
        .cache = reinterpret_cast<velox::cache::AsyncDataCache*>(0x1)};
    NIMBLE_ASSERT_THROW(
        options.validate(),
        "file and fileHandle->file must point to the same file");
  }
}

TEST_F(IndexLookupTest, createIndexMetadataInput) {
  struct TestParam {
    bool cached;
    std::string debugString() const {
      return cached ? "cached" : "direct";
    }
  };
  std::vector<TestParam> testSettings = {{false}, {true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
    auto pool = velox::memory::memoryManager()->addLeafPool();
    velox::io::ReaderOptions ioOptions(pool.get());
    ioOptions.setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
    ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());

    std::shared_ptr<velox::memory::MallocAllocator> allocator;
    std::shared_ptr<velox::cache::AsyncDataCache> cache;
    std::unique_ptr<velox::FileHandle> fileHandle;
    if (testData.cached) {
      allocator = std::make_shared<velox::memory::MallocAllocator>(
          velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});
      cache = velox::cache::AsyncDataCache::create(allocator.get());
      fileHandle = std::make_unique<velox::FileHandle>();
      fileHandle->file = file;
      fileHandle->uuid = velox::StringIdLease(velox::fileIds(), "testFile");
    }

    IndexLookup::Options options{
        .file = file,
        .ioOptions = &ioOptions,
        .fileHandle = fileHandle.get(),
        .cache = cache.get()};
    auto metadataInput = createIndexMetadataInput(options);
    ASSERT_NE(metadataInput, nullptr);
    EXPECT_EQ(metadataInput->cached(), testData.cached);
  }
}

TEST_F(IndexLookupTest, createIndexDataInput) {
  for (bool cached : {false, true}) {
    SCOPED_TRACE(cached ? "cached" : "direct");
    auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
    auto pool = velox::memory::memoryManager()->addLeafPool();
    velox::io::ReaderOptions ioOptions(pool.get());
    ioOptions.setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
    ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());

    std::shared_ptr<velox::memory::MallocAllocator> allocator;
    std::shared_ptr<velox::cache::AsyncDataCache> cache;
    std::unique_ptr<velox::FileHandle> fileHandle;
    if (cached) {
      allocator = std::make_shared<velox::memory::MallocAllocator>(
          velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});
      cache = velox::cache::AsyncDataCache::create(allocator.get());
      fileHandle = std::make_unique<velox::FileHandle>();
      fileHandle->file = file;
      fileHandle->uuid = velox::StringIdLease(velox::fileIds(), "testFile");
    }

    IndexLookup::Options options{
        .file = file,
        .ioOptions = &ioOptions,
        .fileHandle = fileHandle.get(),
        .cache = cache.get()};
    auto dataInput = createIndexDataInput(options);
    ASSERT_NE(dataInput, nullptr);
  }
}

} // namespace
} // namespace facebook::nimble::index::test
