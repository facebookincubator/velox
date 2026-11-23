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
#include "velox/dwio/nimble/tablet/BlockCache.h"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include <folly/executors/IOThreadPoolExecutor.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/caching/SsdFile.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

DECLARE_bool(velox_ssd_odirect);

namespace facebook::nimble {
namespace {

// Largest entry an SsdRun can describe.
constexpr uint32_t kMaxSsdEntrySize{1U << velox::cache::SsdRun::kSizeBits};

// Runs the block cache that getBlockCache() builds over an AsyncDataCache with
// an SSD cache.
class BlockCacheTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
    velox::common::testutil::TestValue::enable();
    velox::filesystems::registerLocalFileSystem();
    FLAGS_velox_ssd_odirect = false;
  }

  void SetUp() override {
    ssdDirectory_ = velox::common::testutil::TempDirectoryPath::create();
    allocator_ = std::make_shared<velox::memory::MallocAllocator>(
        velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});

    // Entries reach SSD only when a test saves them.
    velox::cache::AsyncDataCache::Options options;
    options.minSsdSavableBytes = std::numeric_limits<int32_t>::max();
    asyncDataCache_ = velox::cache::AsyncDataCache::create(
        allocator_.get(),
        std::make_unique<velox::cache::SsdCache>(velox::cache::SsdCache::Config(
            fmt::format("{}/cache", ssdDirectory_->getPath()),
            /*_maxBytes=*/64UL << 20,
            /*_numShards=*/1,
            &ssdExecutor_)),
        options);
    testingResetBlockCache();
    blockCache_ = getBlockCache(asyncDataCache_.get());
  }

  void TearDown() override {
    testingResetBlockCache();
    asyncDataCache_->shutdown();
  }

  // Caches 'data' at 'offset'.
  void cacheEntry(uint64_t offset, std::string_view data) {
    velox::io::IoStatistics ioStats;
    auto entry = blockCache_->getOrCreate(
        fileId_.id(), offset, static_cast<uint32_t>(data.size()), ioStats);
    ASSERT_FALSE(entry.sealed());
    std::memcpy(entry.data(), data.data(), data.size());
    blockCache_->seal(entry);
  }

  // Saves the entries to SSD and clears them from memory.
  void moveToSsd() {
    ASSERT_TRUE(asyncDataCache_->ssdCache()->startWrite());
    asyncDataCache_->saveToSsd(/*saveAll=*/true);
    asyncDataCache_->ssdCache()->waitForWriteToFinish();
    asyncDataCache_->clear();
  }

  static std::string_view contents(const BlockCache::Entry& entry) {
    return {entry.data(), entry.size()};
  }

  folly::IOThreadPoolExecutor ssdExecutor_{1};
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> ssdDirectory_;
  std::shared_ptr<velox::memory::MemoryAllocator> allocator_;
  std::shared_ptr<velox::cache::AsyncDataCache> asyncDataCache_;
  BlockCache* blockCache_{nullptr};

  // AsyncDataCache keys need file ids leased from velox::fileIds().
  const velox::StringIdLease fileId_{velox::fileIds(), "BlockCacheTest"};
};

TEST_F(BlockCacheTest, getOrCreateLoadsEntryFromSsd) {
  cacheEntry(0, "abc");
  moveToSsd();

  velox::io::IoStatistics ioStats;
  {
    auto entry = blockCache_->getOrCreate(fileId_.id(), 0, 3, ioStats);
    ASSERT_TRUE(entry.sealed());
    EXPECT_EQ(contents(entry), "abc");
  }
  EXPECT_EQ(ioStats.ssdRead().count(), 1);
  EXPECT_EQ(ioStats.ssdRead().sum(), 3);
  EXPECT_EQ(ioStats.ssdCacheReadLatencyUs().count(), 1);
  EXPECT_EQ(ioStats.queryThreadIoLatencyUs().count(), 1);
  EXPECT_EQ(ioStats.ramHit().count(), 0);

  // The load left the entry in memory.
  EXPECT_TRUE(blockCache_->getOrCreate(fileId_.id(), 0, 3, ioStats).sealed());
  EXPECT_EQ(ioStats.ramHit().count(), 1);
  EXPECT_EQ(ioStats.ssdRead().count(), 1);
}

TEST_F(BlockCacheTest, getOrCreateSkipsSsdEntryOfOtherSize) {
  cacheEntry(0, "abcd");
  moveToSsd();

  velox::io::IoStatistics ioStats;
  for (const uint32_t size : {3U, 5U}) {
    SCOPED_TRACE(fmt::format("size {}", size));
    auto entry = blockCache_->getOrCreate(fileId_.id(), 0, size, ioStats);
    EXPECT_FALSE(entry.sealed());
    EXPECT_EQ(entry.size(), size);
  }
  EXPECT_EQ(ioStats.ssdRead().count(), 0);

  // The SSD entry is still there for a lookup of its size.
  EXPECT_TRUE(blockCache_->getOrCreate(fileId_.id(), 0, 4, ioStats).sealed());
  EXPECT_EQ(ioStats.ssdRead().count(), 1);
}

TEST_F(BlockCacheTest, savesEntriesUpToMaxSsdEntrySize) {
  // An SsdRun cannot describe the first entry. Saving it would also fail the
  // SSD write of the entries after it, such as the second.
  cacheEntry(0, std::string(kMaxSsdEntrySize + 1, 'a'));
  cacheEntry(kMaxSsdEntrySize + 1, std::string(kMaxSsdEntrySize, 'b'));
  moveToSsd();

  velox::io::IoStatistics ioStats;
  EXPECT_FALSE(
      blockCache_->getOrCreate(fileId_.id(), 0, kMaxSsdEntrySize + 1, ioStats)
          .sealed());
  EXPECT_TRUE(
      blockCache_
          ->getOrCreate(
              fileId_.id(), kMaxSsdEntrySize + 1, kMaxSsdEntrySize, ioStats)
          .sealed());
}

TEST_F(BlockCacheTest, findLoadsEntryFromSsd) {
  cacheEntry(0, "abc");
  moveToSsd();

  velox::io::IoStatistics ioStats;
  EXPECT_TRUE(blockCache_->find(fileId_.id(), 100, ioStats).empty());
  {
    auto entry = blockCache_->find(fileId_.id(), 0, ioStats);
    ASSERT_FALSE(entry.empty());
    EXPECT_TRUE(entry.sealed());
    EXPECT_EQ(contents(entry), "abc");
  }
  EXPECT_EQ(ioStats.ssdRead().count(), 1);
  EXPECT_EQ(ioStats.ssdRead().sum(), 3);
  EXPECT_EQ(ioStats.ramHit().count(), 0);

  // The load left the entry in memory.
  EXPECT_FALSE(blockCache_->find(fileId_.id(), 0, ioStats).empty());
  EXPECT_EQ(ioStats.ramHit().count(), 1);
  EXPECT_EQ(ioStats.ssdRead().count(), 1);
}

DEBUG_ONLY_TEST_F(BlockCacheTest, findReturnsEntryCachedDuringSsdLookup) {
  cacheEntry(0, "abc");
  moveToSsd();

  // Another reader caches a larger entry under the key after find() misses
  // memory, and before find() creates the entry to load from SSD.
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::BlockCacheImpl::findSsd",
      std::function<void(void*)>(
          [&](void* /*blockCache*/) { cacheEntry(0, "wxyz"); }));
  velox::io::IoStatistics ioStats;
  auto entry = blockCache_->find(fileId_.id(), 0, ioStats);
  ASSERT_FALSE(entry.empty());
  EXPECT_EQ(contents(entry), "wxyz");
  EXPECT_EQ(ioStats.ssdRead().count(), 0);
  EXPECT_EQ(ioStats.ramHit().count(), 1);
}

DEBUG_ONLY_TEST_F(BlockCacheTest, failedSsdLoadErasesSsdEntry) {
  cacheEntry(0, "abc");
  cacheEntry(100, "def");
  moveToSsd();

  velox::io::IoStatistics ioStats;
  {
    SCOPED_TESTVALUE_SET(
        "facebook::velox::cache::SsdFile::load",
        std::function<void(velox::cache::SsdFile*)>(
            [](velox::cache::SsdFile* /*ssdFile*/) {
              VELOX_FAIL("Injected SSD load failure");
            }));

    // getOrCreate() leaves the entry for the caller to fill from storage.
    auto entry = blockCache_->getOrCreate(fileId_.id(), 0, 3, ioStats);
    EXPECT_FALSE(entry.sealed());
    EXPECT_TRUE(blockCache_->find(fileId_.id(), 100, ioStats).empty());
  }
  EXPECT_EQ(ioStats.ssdRead().count(), 0);

  // Later lookups miss the SSD cache rather than fail to load again.
  auto& ssdFile = asyncDataCache_->ssdCache()->file(fileId_.id());
  EXPECT_TRUE(ssdFile.find({fileId_.id(), 0}).empty());
  EXPECT_TRUE(ssdFile.find({fileId_.id(), 100}).empty());
}

} // namespace
} // namespace facebook::nimble
