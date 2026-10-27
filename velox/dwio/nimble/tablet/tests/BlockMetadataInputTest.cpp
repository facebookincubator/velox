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
#include "velox/dwio/nimble/tablet/BlockMetadataInput.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <folly/Synchronized.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
#include "velox/dwio/nimble/tablet/Compression.h"

namespace facebook::nimble {
namespace {

constexpr uint64_t kAlignment{4'096};

// Offset and length of each region of one positioned vector read.
using Read = std::vector<std::pair<uint64_t, uint64_t>>;

// In-memory file that records each positioned vector read. With 'alignment'
// above 1 it reports direct I/O and rejects reads that a direct-I/O file
// could not serve without copying.
class RecordingReadFile final : public velox::InMemoryReadFile {
 public:
  RecordingReadFile(std::string content, uint64_t alignment)
      : velox::InMemoryReadFile{std::move(content)}, alignment_{alignment} {}

  using velox::InMemoryReadFile::preadv;

  uint64_t preadv(
      folly::Range<const velox::common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      const velox::FileIoContext& context = {}) const override {
    if (failReads_) {
      NIMBLE_FAIL("Injected preadv failure");
    }
    Read read;
    for (size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      const auto end = region.offset + region.length;
      NIMBLE_CHECK_LE(end, size(), "Read past the end of the file");
      NIMBLE_CHECK_EQ(region.offset % alignment_, 0, "Unaligned read offset");
      NIMBLE_CHECK(
          region.length % alignment_ == 0 || end == size(),
          "Unaligned read length");
      NIMBLE_CHECK_EQ(
          reinterpret_cast<uintptr_t>(buffers[i].data()) % alignment_,
          0,
          "Unaligned read buffer");
      read.emplace_back(region.offset, region.length);
    }
    reads_.wlock()->push_back(std::move(read));
    return velox::InMemoryReadFile::preadv(regions, buffers, context);
  }

  bool directIo(uint64_t& alignment) const override {
    alignment = alignment_;
    return alignment_ > 1;
  }

  std::vector<Read> reads() const {
    return *reads_.rlock();
  }

  void setFailReads(bool failReads) {
    failReads_ = failReads;
  }

 private:
  const uint64_t alignment_;
  std::atomic_bool failReads_{false};
  mutable folly::Synchronized<std::vector<Read>> reads_;
};

// Delegates to another BlockCache and records the offset of each
// getOrCreate() call. Entries come from the delegate and are released there.
class RecordingBlockCache final : public BlockCache {
 public:
  explicit RecordingBlockCache(BlockCache* cache) : cache_{cache} {}

  Entry getOrCreate(
      uint64_t fileId,
      uint64_t offset,
      uint32_t size,
      velox::io::IoStatistics& ioStats) override {
    offsets_.push_back(offset);
    return cache_->getOrCreate(fileId, offset, size, ioStats);
  }

  void seal(Entry& entry) override {
    cache_->seal(entry);
  }

  const std::vector<uint64_t>& offsets() const {
    return offsets_;
  }

 protected:
  void release(void* /*pin*/) noexcept override {}

 private:
  BlockCache* const cache_;
  std::vector<uint64_t> offsets_;
};

// Bytes of a section and where they sit in the file.
struct SectionSpec {
  uint64_t offset;
  std::string data;
  CompressionType compressionType{CompressionType::Uncompressed};
  bool hasUncompressedSize{true};
};

class BlockMetadataInputTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ =
        velox::memory::memoryManager()->addLeafPool("BlockMetadataInputTest");
    // An allocator accepts one cache for its lifetime, so each test brings its
    // own rather than registering another cache with the shared one.
    allocator_ = std::make_shared<velox::memory::MallocAllocator>(
        velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});
    asyncDataCache_ = velox::cache::AsyncDataCache::create(allocator_.get());
    testingResetBlockCache();
    blockCache_ = getBlockCache(asyncDataCache_.get());
  }

  void TearDown() override {
    testingResetBlockCache();
    asyncDataCache_->shutdown();
  }

  // Lays out 'specs' in a file of 'fileSize' bytes with gaps of 0xCC, and
  // returns the file with the sections that describe it.
  std::pair<std::shared_ptr<RecordingReadFile>, std::vector<MetadataSection>>
  makeFile(
      const std::vector<SectionSpec>& specs,
      uint64_t fileSize,
      uint64_t alignment) {
    std::string content(fileSize, '\xCC');
    std::vector<MetadataSection> sections;
    for (const auto& spec : specs) {
      std::string bytes = spec.data;
      if (spec.compressionType == CompressionType::Zstd) {
        auto compressed = ZstdCompression::compress(spec.data, pool_.get());
        NIMBLE_CHECK(compressed.has_value());
        bytes = {compressed.value()->as<char>(), compressed.value()->size()};
      }
      NIMBLE_CHECK_LE(spec.offset + bytes.size(), fileSize);
      std::memcpy(content.data() + spec.offset, bytes.data(), bytes.size());
      sections.emplace_back(
          spec.offset,
          static_cast<uint32_t>(bytes.size()),
          spec.compressionType,
          spec.hasUncompressedSize ? std::optional<uint32_t>(spec.data.size())
                                   : std::nullopt);
    }
    return {
        std::make_shared<RecordingReadFile>(std::move(content), alignment),
        std::move(sections)};
  }

  MetadataInput::Options makeOptions(
      BlockCache* blockCache,
      int32_t maxCoalesceDistance = 0,
      int64_t maxCoalesceBytes = 128 << 20) {
    return {
        .pool = pool_.get(),
        .ioStats = ioStats_,
        .maxCoalesceDistance = maxCoalesceDistance,
        .maxCoalesceBytes = maxCoalesceBytes,
        .blockCache = blockCache,
        .blockCacheFileId = fileId_.id(),
        .blockMetadataInput = true,
    };
  }

  static std::vector<std::string> contents(
      const std::vector<std::shared_ptr<MetadataBuffer>>& buffers) {
    std::vector<std::string> result;
    for (const auto& buffer : buffers) {
      result.emplace_back(buffer->content());
    }
    return result;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::io::IoStatistics> ioStats_{
      std::make_shared<velox::io::IoStatistics>()};
  std::shared_ptr<velox::memory::MemoryAllocator> allocator_;
  std::shared_ptr<velox::cache::AsyncDataCache> asyncDataCache_;
  BlockCache* blockCache_{nullptr};
  // AsyncDataCache keys need file ids leased from velox::fileIds().
  const velox::StringIdLease fileId_{
      velox::fileIds(),
      "BlockMetadataInputTest"};
};

TEST_F(BlockMetadataInputTest, createsBlockInputs) {
  auto [file, sections] = makeFile({}, 100, 1);
  {
    auto input = MetadataInput::create(file.get(), makeOptions(nullptr));
    EXPECT_NE(dynamic_cast<DirectBlockMetadataInput*>(input.get()), nullptr);
    EXPECT_FALSE(input->cached());
  }
  {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
    EXPECT_NE(dynamic_cast<CachedBlockMetadataInput*>(input.get()), nullptr);
    EXPECT_TRUE(input->cached());
  }
  {
    auto options = makeOptions(blockCache_);
    options.cache = asyncDataCache_.get();
    NIMBLE_ASSERT_THROW(
        MetadataInput::create(file.get(), options),
        "Block metadata input does not use an AsyncDataCache");
  }
  {
    auto options = makeOptions(blockCache_);
    options.blockCacheFileId = 0;
    NIMBLE_ASSERT_USER_THROW(
        MetadataInput::create(file.get(), options),
        "blockCacheFileId must be set together with blockCache");
  }
}

TEST_F(BlockMetadataInputTest, loadSections) {
  const std::vector<SectionSpec> specs{
      {0, "uncompressed-head"},
      {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
      {9'000, std::string(1'000, 'O'), CompressionType::Zstd, false},
      {12'300, "uncompressed-tail"},
  };
  const std::vector<std::string> expected{
      "uncompressed-tail",
      std::string(1'000, 'O'),
      std::string(1'000, 'Z'),
      "uncompressed-head",
  };
  for (const auto alignment : {uint64_t{1}, kAlignment}) {
    for (const bool cached : {false, true}) {
      SCOPED_TRACE(fmt::format("alignment {}, cached {}", alignment, cached));
      auto [file, sections] = makeFile(specs, 12'317, alignment);
      // A file id per file, so no file hits another file's entries.
      const velox::StringIdLease fileId{
          velox::fileIds(),
          fmt::format("BlockMetadataInputTest.loadSections.{}", alignment)};
      auto options = makeOptions(cached ? blockCache_ : nullptr);
      options.blockCacheFileId = fileId.id();
      auto input = MetadataInput::create(file.get(), options);

      // Requested in reverse file order.
      const std::array requested{
          sections[3], sections[2], sections[1], sections[0]};
      EXPECT_EQ(contents(input->load(requested)), expected);
      EXPECT_EQ(file->reads().size(), 1);
    }
  }
}

TEST_F(BlockMetadataInputTest, readsInOneAlignedBatch) {
  const uint64_t fileSize = 6 * kAlignment + 100;
  auto [file, sections] = makeFile(
      {
          {100, std::string(100, 'a')},
          {9'000, std::string(100, 'b')},
          {20'000, std::string(50, 'c')},
          {fileSize - 30, std::string(30, 'd')},
      },
      fileSize,
      kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));

  input->load(sections);

  // The last read stops at the end of the file.
  const std::vector<Read> expected{{
      {0, kAlignment},
      {2 * kAlignment, kAlignment},
      {4 * kAlignment, kAlignment},
      {6 * kAlignment, 100},
  }};
  EXPECT_EQ(file->reads(), expected);
}

TEST_F(BlockMetadataInputTest, mergesReadsSharingABlock) {
  auto [file, sections] = makeFile(
      {
          {100, std::string(100, 'a')},
          {300, std::string(100, 'b')},
          {4'000, std::string(200, 'c')},
      },
      3 * kAlignment,
      kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));

  EXPECT_EQ(
      contents(input->load(sections)),
      (std::vector<std::string>{
          std::string(100, 'a'),
          std::string(100, 'b'),
          std::string(200, 'c'),
      }));
  const std::vector<Read> expected{{{0, 2 * kAlignment}}};
  EXPECT_EQ(file->reads(), expected);
}

TEST_F(BlockMetadataInputTest, coalescesReads) {
  const std::vector<SectionSpec> specs{
      {0, std::string(10, 'a')},
      {50, std::string(10, 'b')},
      {500, std::string(10, 'c')},
  };
  {
    SCOPED_TRACE("Merges reads within the coalesce distance");
    auto [file, sections] = makeFile(specs, 1'000, 1);
    auto input = MetadataInput::create(
        file.get(), makeOptions(nullptr, /*maxCoalesceDistance=*/100));
    input->load(sections);
    const std::vector<Read> expected{{{0, 60}, {500, 10}}};
    EXPECT_EQ(file->reads(), expected);
  }
  {
    SCOPED_TRACE("Stops merging at the coalesce byte limit");
    auto [file, sections] = makeFile(specs, 1'000, 1);
    auto input = MetadataInput::create(
        file.get(),
        makeOptions(
            nullptr, /*maxCoalesceDistance=*/100, /*maxCoalesceBytes=*/15));
    input->load(sections);
    const std::vector<Read> expected{{{0, 10}, {50, 10}, {500, 10}}};
    EXPECT_EQ(file->reads(), expected);
  }
  {
    SCOPED_TRACE("Caps the coalesce distance");
    constexpr uint64_t kGap{BlockMetadataInput::kMaxCoalesceDistance + 1};
    auto [file, sections] = makeFile(
        {{0, std::string(10, 'a')}, {10 + kGap, std::string(10, 'b')}},
        20 + kGap,
        1);
    auto input = MetadataInput::create(
        file.get(),
        makeOptions(
            nullptr,
            /*maxCoalesceDistance=*/2 *
                BlockMetadataInput::kMaxCoalesceDistance));
    input->load(sections);
    const std::vector<Read> expected{{{0, 10}, {10 + kGap, 10}}};
    EXPECT_EQ(file->reads(), expected);
  }
}

TEST_F(BlockMetadataInputTest, loadsEmptySection) {
  auto [file, sections] = makeFile({{10, ""}}, 100, kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  EXPECT_EQ(contents(input->load(sections)), std::vector<std::string>{""});
  EXPECT_TRUE(file->reads().empty());
}

TEST_F(BlockMetadataInputTest, loadsEmptySectionSharingOffset) {
  // The writer gives an empty section the offset of the section after it.
  auto [file, sections] = makeFile({{10, ""}, {10, "data"}}, 100, kAlignment);
  for (auto* blockCache : {static_cast<BlockCache*>(nullptr), blockCache_}) {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache));
    EXPECT_EQ(
        contents(input->load(sections)),
        (std::vector<std::string>{"", "data"}));
  }
}

TEST_F(BlockMetadataInputTest, compressedSectionTooSmallThrows) {
  auto [file, sections] = makeFile({}, 100, 1);
  // Neither holds the 4-byte uncompressed size. The second ends the file, so
  // reading the size would run past the read buffer.
  const std::array tooSmall{
      MetadataSection{10, 0, CompressionType::Zstd, 100},
      MetadataSection{98, 2, CompressionType::Zstd, 100},
  };
  for (auto* blockCache : {static_cast<BlockCache*>(nullptr), blockCache_}) {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache));
    for (const auto& section : tooSmall) {
      NIMBLE_ASSERT_FILE_THROW(
          input->load({&section, 1}),
          "Compressed metadata section is too small");
    }
  }
}

TEST_F(BlockMetadataInputTest, sectionPastEndOfFileThrows) {
  auto [file, sections] = makeFile({}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));
  {
    SCOPED_TRACE("Offset past the end, with offset + size wrapping to 16");
    const MetadataSection section{
        std::numeric_limits<uint64_t>::max() - 15,
        32,
        CompressionType::Uncompressed,
        32};
    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}),
        "Metadata section starts past the end of the file");
  }
  {
    SCOPED_TRACE("Size past the end");
    const MetadataSection section{90, 32, CompressionType::Uncompressed, 32};
    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}),
        "Metadata section extends past the end of the file");
  }
}

TEST_F(BlockMetadataInputTest, overlappingSectionsRecordBytesRead) {
  auto [file, sections] = makeFile({}, 300, 1);
  const std::array overlapping{
      MetadataSection{0, 200, CompressionType::Uncompressed, 200},
      MetadataSection{100, 100, CompressionType::Uncompressed, 100},
  };
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));

  input->load(overlapping);

  EXPECT_EQ(ioStats_->read().sum(), 200);
  EXPECT_EQ(ioStats_->rawBytesRead(), 200);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), 0);
}

TEST_F(BlockMetadataInputTest, recordsReadStats) {
  auto [file, sections] =
      makeFile({{100, std::string(200, 'a')}}, kAlignment, kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));

  input->load(sections);

  EXPECT_EQ(ioStats_->read().sum(), kAlignment);
  EXPECT_EQ(ioStats_->rawBytesRead(), 200);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), kAlignment - 200);
  EXPECT_EQ(ioStats_->storageReadLatencyUs().count(), 1);
  EXPECT_EQ(ioStats_->queryThreadIoLatencyUs().count(), 1);
}

TEST_F(BlockMetadataInputTest, duplicateSectionsThrow) {
  auto [file, sections] = makeFile({{0, "duplicate"}}, 100, 1);
  for (auto* blockCache : {static_cast<BlockCache*>(nullptr), blockCache_}) {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache));
    NIMBLE_ASSERT_THROW(
        input->load(std::array{sections[0], sections[0]}),
        "Metadata input must not contain duplicate regions");
  }
}

TEST_F(BlockMetadataInputTest, cacheHitSkipsRead) {
  auto [file, sections] = makeFile(
      {
          {0, "uncompressed"},
          {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
      },
      2 * kAlignment,
      kAlignment);
  const std::vector<std::string> expected{
      "uncompressed", std::string(1'000, 'Z')};
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(file->reads().size(), 1);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);

  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(file->reads().size(), 1);
  EXPECT_EQ(ioStats_->ramHit().count(), 2);
}

TEST_F(BlockMetadataInputTest, sectionWithoutUncompressedSizeBypassesCache) {
  auto [file, sections] = makeFile(
      {
          {0, "cached"},
          {5'000, std::string(1'000, 'Z'), CompressionType::Zstd, false},
      },
      2 * kAlignment,
      kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  input->load(sections);

  EXPECT_EQ(
      contents(input->load(sections)),
      (std::vector<std::string>{"cached", std::string(1'000, 'Z')}));
  ASSERT_EQ(file->reads().size(), 2);
  const Read expected{{kAlignment, kAlignment}};
  EXPECT_EQ(file->reads()[1], expected);
}

TEST_F(BlockMetadataInputTest, oversizedSectionBypassesCache) {
  auto [file, sections] = makeFile(
      {{0, std::string(1'000, 'Z'), CompressionType::Zstd}},
      kAlignment,
      kAlignment);
  // A corrupt footer can claim an uncompressed size no cache entry can hold.
  const MetadataSection section{
      sections[0].offset(),
      sections[0].size(),
      sections[0].compressionType(),
      1U << 31};
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  const std::vector<std::string> expected{std::string(1'000, 'Z')};

  EXPECT_EQ(contents(input->load({&section, 1})), expected);
  EXPECT_EQ(contents(input->load({&section, 1})), expected);

  EXPECT_EQ(file->reads().size(), 2);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);
}

TEST_F(BlockMetadataInputTest, claimsCacheEntriesInFileOffsetOrder) {
  auto [file, sections] = makeFile(
      {
          {0, "first"},
          {5'000, "second"},
          {9'000, "third"},
      },
      3 * kAlignment,
      kAlignment);
  RecordingBlockCache recordingCache{blockCache_};
  auto input = MetadataInput::create(file.get(), makeOptions(&recordingCache));

  input->load(std::array{sections[2], sections[0], sections[1]});

  EXPECT_EQ(recordingCache.offsets(), (std::vector<uint64_t>{0, 5'000, 9'000}));
}

TEST_F(BlockMetadataInputTest, failedReadReleasesCacheEntries) {
  auto [file, sections] = makeFile({{0, "metadata"}}, kAlignment, kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  file->setFailReads(true);
  NIMBLE_ASSERT_THROW(input->load(sections), "Injected preadv failure");

  // A leaked unfilled entry would block this load forever.
  file->setFailReads(false);
  EXPECT_EQ(
      contents(input->load(sections)), std::vector<std::string>{"metadata"});
}

TEST_F(BlockMetadataInputTest, cacheAndFindMetadata) {
  auto [file, sections] = makeFile({}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  EXPECT_EQ(input->findCachedMetadata(100), nullptr);

  const std::array<std::string_view, 2> ranges{"foot", "er"};
  input->cacheMetadata(100, ranges);

  auto found = input->findCachedMetadata(100);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->content(), "footer");

  // Caching bytes that are already cached is not a RAM hit.
  const auto ramHits = ioStats_->ramHit().count();
  input->cacheMetadata(100, ranges);
  EXPECT_EQ(ioStats_->ramHit().count(), ramHits);

  const velox::StringIdLease otherFileId{
      velox::fileIds(), "BlockMetadataInputTest.otherFile"};
  auto options = makeOptions(blockCache_);
  options.blockCacheFileId = otherFileId.id();
  auto otherFileInput = MetadataInput::create(file.get(), options);
  EXPECT_EQ(otherFileInput->findCachedMetadata(100), nullptr);
}

TEST_F(BlockMetadataInputTest, findLeavesNoEntry) {
  velox::io::IoStatistics ioStats;
  auto [missing, missingSize] = blockCache_->find(fileId_.id(), 0, ioStats);
  EXPECT_TRUE(missing.empty());

  // find() created nothing, so this lookup creates the entry.
  auto created = blockCache_->getOrCreate(fileId_.id(), 0, 3, ioStats);
  EXPECT_FALSE(created.sealed());
  std::memcpy(created.data(), "abc", 3);
  blockCache_->seal(created);

  auto [found, foundSize] = blockCache_->find(fileId_.id(), 0, ioStats);
  ASSERT_FALSE(found.empty());
  EXPECT_TRUE(found.sealed());
  EXPECT_EQ(std::string_view(found.data(), foundSize), "abc");
}

TEST_F(BlockMetadataInputTest, uncachedInputRejectsCacheMethods) {
  auto [file, sections] = makeFile({}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(nullptr));
  NIMBLE_ASSERT_THROW(
      input->findCachedMetadata(0),
      "findCachedMetadata requires a cached MetadataInput");
  const std::array<std::string_view, 1> ranges{"data"};
  NIMBLE_ASSERT_THROW(
      input->cacheMetadata(0, ranges),
      "cacheMetadata requires a cached MetadataInput");
}

} // namespace
} // namespace facebook::nimble
