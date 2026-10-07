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

#include "velox/dwio/nimble/tablet/MetadataInput.h"

#include <gtest/gtest.h>
#include <array>

#include "velox/common/base/CoalesceIo.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/caching/SsdFile.h"
#include "velox/common/caching/tests/CacheTestUtil.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/tablet/Compression.h"

#include "folly/executors/IOThreadPoolExecutor.h"
#include "folly/synchronization/Baton.h"

#include <thread>

using namespace facebook;

namespace {

// Builds a flat file buffer containing sections at specified offsets.
// Each section's data is written at its offset in the file. Gaps are
// filled with 0xCC bytes so we can detect over-reads.
std::string buildFileContent(
    const std::vector<nimble::MetadataSection>& sections,
    const std::vector<std::string>& sectionData) {
  EXPECT_EQ(sections.size(), sectionData.size());
  uint64_t fileSize = 0;
  for (size_t i = 0; i < sections.size(); ++i) {
    fileSize = std::max(fileSize, sections[i].offset() + sectionData[i].size());
  }
  std::string fileContent(fileSize, '\xCC');
  for (size_t i = 0; i < sections.size(); ++i) {
    std::memcpy(
        fileContent.data() + sections[i].offset(),
        sectionData[i].data(),
        sectionData[i].size());
  }
  return fileContent;
}

class MetadataInputTestBase : public ::testing::Test {
 protected:
  nimble::MetadataInput::Options makeMetadataInputOptions(
      int32_t maxCoalesceDistance = 1 << 20,
      int64_t maxCoalesceBytes = 128 << 20) {
    return {
        .pool = pool_.get(),
        .ioStats = metadataIoStats_,
        .maxCoalesceDistance = maxCoalesceDistance,
        .maxCoalesceBytes = maxCoalesceBytes};
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::io::IoStatistics> dataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
};

struct SectionSpec {
  std::string data;
  nimble::CompressionType compressionType;
  bool hasUncompressedSize{true};
};

struct EnqueueLoadReadTestParam {
  std::vector<SectionSpec> specs;
  std::string debugString() const {
    return fmt::format(
        "{} sections, [{}]",
        specs.size(),
        fmt::join(
            specs | std::views::transform([](const auto& s) {
              return fmt::format(
                  "comp{}:sz{}:uncomp{}",
                  static_cast<int>(s.compressionType),
                  s.data.size(),
                  s.hasUncompressedSize);
            }),
            ", "));
  }
};

void buildSections(
    const std::vector<SectionSpec>& specs,
    velox::memory::MemoryPool* pool,
    std::vector<nimble::MetadataSection>& sections,
    std::vector<std::string>& fileDataParts) {
  uint64_t offset = 0;
  for (const auto& spec : specs) {
    std::string onDiskData;
    const uint32_t uncompressedSize = spec.data.size();
    if (spec.compressionType == nimble::CompressionType::Zstd) {
      auto compressed = nimble::ZstdCompression::compress(spec.data, pool);
      ASSERT_TRUE(compressed.has_value());
      onDiskData = {compressed.value()->as<char>(), compressed.value()->size()};
    } else {
      onDiskData = spec.data;
    }
    sections.emplace_back(
        offset,
        static_cast<uint32_t>(onDiskData.size()),
        spec.compressionType,
        spec.hasUncompressedSize ? std::optional<uint32_t>(uncompressedSize)
                                 : std::nullopt);
    fileDataParts.emplace_back(std::move(onDiskData));
    offset += fileDataParts.back().size();
  }
}

std::vector<EnqueueLoadReadTestParam> enqueueLoadReadTestSettings() {
  return {
      {{{"Hello, MetadataInput!", nimble::CompressionType::Uncompressed}}},
      {{{std::string(1'000, 'A'), nimble::CompressionType::Zstd}}},
      // Compressed without uncompressed size (old file format).
      {{{std::string(1'000, 'A'), nimble::CompressionType::Zstd, false}}},
      {{{"section-zero", nimble::CompressionType::Uncompressed},
        {"section-one", nimble::CompressionType::Uncompressed},
        {"section-two", nimble::CompressionType::Uncompressed}}},
      {{{std::string(1'000, 'B'), nimble::CompressionType::Zstd},
        {std::string(2'000, 'C'), nimble::CompressionType::Zstd}}},
      {{{"uncompressed-first", nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'D'), nimble::CompressionType::Zstd}}},
      {{{std::string(1'000, 'E'), nimble::CompressionType::Zstd},
        {"uncompressed-second", nimble::CompressionType::Uncompressed}}},
      {{{"head", nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'F'), nimble::CompressionType::Zstd},
        {"tail", nimble::CompressionType::Uncompressed}}},
      // Mixed: with and without uncompressed size.
      {{{"with-size", nimble::CompressionType::Uncompressed, true},
        {std::string(1'000, 'I'), nimble::CompressionType::Zstd, false},
        {std::string(2'000, 'J'), nimble::CompressionType::Zstd, true}}},
      // All without uncompressed size.
      {{{std::string(1'000, 'K'), nimble::CompressionType::Zstd, false},
        {std::string(2'000, 'L'), nimble::CompressionType::Zstd, false}}},
      {{{std::string(64 * 1'024, 'G'), nimble::CompressionType::Uncompressed}}},
      {{{std::string(64 * 1'024, 'H'), nimble::CompressionType::Zstd}}},
  };
}

class DirectMetadataInputTest : public MetadataInputTestBase {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
    velox::common::testutil::TestValue::enable();
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("DirectMetadataInputTest");
    pool_ = rootPool_->addLeafChild("leaf");
  }
};

TEST_F(DirectMetadataInputTest, loadSections) {
  for (const auto& testData : enqueueLoadReadTestSettings()) {
    SCOPED_TRACE(testData.debugString());

    std::vector<nimble::MetadataSection> sections;
    std::vector<std::string> fileDataParts;
    buildSections(testData.specs, pool_.get(), sections, fileDataParts);

    const auto fileContent = buildFileContent(sections, fileDataParts);
    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

    const auto options = makeMetadataInputOptions();
    auto input = nimble::MetadataInput::create(readFile.get(), options);

    auto results = input->load(sections);
    ASSERT_EQ(results.size(), sections.size());
    for (uint32_t i = 0; i < sections.size(); ++i) {
      ASSERT_NE(results[i], nullptr);
      EXPECT_EQ(results[i]->content(), testData.specs[i].data);
    }
  }
}

TEST_F(DirectMetadataInputTest, reverseSectionOrder) {
  const std::string data0 = "first-in-file";
  const std::string data1 = "second-in-file";

  const uint64_t offset0 = 0;
  const uint64_t offset1 = data0.size();

  nimble::MetadataSection section0{
      offset0,
      static_cast<uint32_t>(data0.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data0.size())};
  nimble::MetadataSection section1{
      offset1,
      static_cast<uint32_t>(data1.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data1.size())};

  const auto fileContent =
      buildFileContent({section0, section1}, {data0, data1});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  const auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(readFile.get(), options);

  // Sections in reverse file order — should still work.
  auto results = input->load(std::array{section1, section0});
  EXPECT_EQ(results[0]->content(), data1);
  EXPECT_EQ(results[1]->content(), data0);
}

TEST_F(DirectMetadataInputTest, duplicateSectionsThrow) {
  const std::string data = "duplicate-metadata";
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  const auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(readFile.get(), options);

  const std::array sections{section, section};
  NIMBLE_ASSERT_THROW(
      input->load(sections),
      "Metadata input must not contain duplicate regions");
}

TEST_F(DirectMetadataInputTest, repeatedLoad) {
  const std::string data = "repeat-test";
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  const auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(readFile.get(), options);

  // Stateless: can call load() multiple times without reset.
  auto results1 = input->load(std::array{section});
  EXPECT_EQ(results1[0]->content(), data);

  auto results2 = input->load(std::array{section});
  EXPECT_EQ(results2[0]->content(), data);

  // Empty input throws.
  NIMBLE_ASSERT_THROW(
      input->load(std::span<const nimble::MetadataSection>{}),
      "No sections to load");
}

TEST_F(DirectMetadataInputTest, ioStats) {
  constexpr uint64_t kInjectedDelayUs = 10'000;
  const std::string data = std::string(500, 'X');
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto innerFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
  auto readFile =
      std::make_shared<nimble::testing::TrackingReadFile>(innerFile);
  readFile->setReadDelayUs(kInjectedDelayUs);

  const auto readCountBefore = metadataIoStats_->read().count();
  const auto readSumBefore = metadataIoStats_->read().sum();
  const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
  const auto overreadBefore = metadataIoStats_->rawOverreadBytes();
  const auto totalScanTimeBefore = metadataIoStats_->totalScanTimeNs();
  const auto storageLatencySumBefore =
      metadataIoStats_->storageReadLatencyUs().sum();
  const auto storageLatencyCountBefore =
      metadataIoStats_->storageReadLatencyUs().count();
  const auto queryIoLatencyCountBefore =
      metadataIoStats_->queryThreadIoLatencyUs().count();

  const auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(readFile.get(), options);
  input->load(std::array{section});

  EXPECT_GT(metadataIoStats_->read().count(), readCountBefore);
  EXPECT_EQ(metadataIoStats_->read().sum() - readSumBefore, data.size());
  EXPECT_EQ(metadataIoStats_->rawBytesRead() - rawBytesBefore, data.size());
  EXPECT_EQ(metadataIoStats_->rawOverreadBytes(), overreadBefore);
  EXPECT_GE(
      metadataIoStats_->totalScanTimeNs() - totalScanTimeBefore,
      kInjectedDelayUs * 1'000);
  EXPECT_GT(
      metadataIoStats_->storageReadLatencyUs().count(),
      storageLatencyCountBefore);
  EXPECT_GE(
      metadataIoStats_->storageReadLatencyUs().sum() - storageLatencySumBefore,
      kInjectedDelayUs);
  EXPECT_GT(
      metadataIoStats_->queryThreadIoLatencyUs().count(),
      queryIoLatencyCountBefore);
}

struct SectionLayout {
  uint64_t offset;
  uint32_t size;
};

DEBUG_ONLY_TEST_F(DirectMetadataInputTest, ioCoalescing) {
  struct TestParam {
    std::vector<SectionLayout> sections;
    int32_t maxCoalesceDistance;
    int64_t maxCoalesceBytes;
    int32_t expectedNumIos;
    uint64_t expectedOverread;
    std::string debugString() const {
      return fmt::format(
          "{} sections, maxCoalesceDistance {}, maxCoalesceBytes {}, expectedNumIos {}, expectedOverread {}",
          sections.size(),
          maxCoalesceDistance,
          maxCoalesceBytes,
          expectedNumIos,
          expectedOverread);
    }
  };

  constexpr int64_t kLargeBytes = 128 << 20;

  std::vector<TestParam> testSettings = {
      // Single section → 1 IO.
      {{{0, 100}}, 0, kLargeBytes, 1, 0},
      // Two contiguous sections → 1 IO.
      {{{0, 100}, {100, 200}}, 0, kLargeBytes, 1, 0},
      // Three contiguous sections → 1 IO.
      {{{0, 50}, {50, 60}, {110, 90}}, 0, kLargeBytes, 1, 0},
      // Two sections with small gap, distance=0 → 2 IOs.
      {{{0, 100}, {150, 100}}, 0, kLargeBytes, 2, 0},
      // Two sections with gap within coalesce distance → 1 IO.
      {{{0, 100}, {150, 100}}, 100, kLargeBytes, 1, 50},
      // Two sections with gap exceeding coalesce distance → 2 IOs.
      {{{0, 100}, {250, 100}}, 100, kLargeBytes, 2, 0},
      // Three sections: first two coalesce, third separate.
      {{{0, 100}, {120, 80}, {500, 100}}, 50, kLargeBytes, 2, 20},
      // Three sections: all coalesce with large distance.
      {{{0, 100}, {200, 100}, {400, 100}}, 200, kLargeBytes, 1, 200},
      // Large gap, small distance → separate IOs.
      {{{0, 1'000}, {10'000, 1'000}}, 100, kLargeBytes, 2, 0},
      // maxCoalesceBytes splits contiguous sections.
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 150, 2, 0},
      // maxCoalesceBytes exactly fits two sections → third splits.
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 200, 2, 0},
      // maxCoalesceBytes fits all three.
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 300, 1, 0},
      // maxCoalesceBytes splits every section into its own IO.
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 50, 3, 0},
      // maxCoalesceBytes with gap: gap coalesced but bytes limit splits.
      {{{0, 100}, {150, 100}, {300, 100}}, 200, 200, 2, 50},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    // Build file with sections filled with deterministic data.
    uint64_t fileSize = 0;
    for (const auto& s : testData.sections) {
      fileSize = std::max(fileSize, s.offset + s.size);
    }
    std::string fileContent(fileSize, '\xCC');
    std::vector<std::string> expectedData;
    std::vector<nimble::MetadataSection> sections;
    for (uint32_t i = 0; i < testData.sections.size(); ++i) {
      const auto& layout = testData.sections[i];
      std::string data(layout.size, 'A' + i);
      std::memcpy(fileContent.data() + layout.offset, data.data(), data.size());
      sections.emplace_back(
          layout.offset,
          layout.size,
          nimble::CompressionType::Uncompressed,
          layout.size);
      expectedData.push_back(std::move(data));
    }

    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

    const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
    const auto overreadBefore = metadataIoStats_->rawOverreadBytes();

    velox::CoalesceIoStats capturedStats;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::MetadataInput::computeIoGroups",
        std::function<void(const velox::CoalesceIoStats*)>(
            [&](const velox::CoalesceIoStats* stats) {
              capturedStats = *stats;
            }));

    const auto options = makeMetadataInputOptions(
        testData.maxCoalesceDistance, testData.maxCoalesceBytes);
    auto input = nimble::MetadataInput::create(readFile.get(), options);
    input->load(sections);

    EXPECT_EQ(capturedStats.numIos, testData.expectedNumIos);
    EXPECT_EQ(capturedStats.extraBytes, testData.expectedOverread);

    uint64_t expectedPayloadBytes = 0;
    for (const auto& s : testData.sections) {
      expectedPayloadBytes += s.size;
    }
    EXPECT_EQ(
        metadataIoStats_->rawBytesRead() - rawBytesBefore,
        expectedPayloadBytes);
    EXPECT_EQ(
        metadataIoStats_->rawOverreadBytes() - overreadBefore,
        testData.expectedOverread);
  }
}

TEST_F(DirectMetadataInputTest, ioError) {
  struct TestParam {
    std::vector<SectionSpec> specs;
    int32_t maxCoalesceDistance;
    std::string debugString() const {
      return fmt::format(
          "{} sections, maxCoalesceDistance {}, [{}]",
          specs.size(),
          maxCoalesceDistance,
          fmt::join(
              specs | std::views::transform([](const auto& s) {
                return fmt::format(
                    "comp{}:sz{}:uncomp{}",
                    static_cast<int>(s.compressionType),
                    s.data.size(),
                    s.hasUncompressedSize);
              }),
              ", "));
    }
  };

  std::vector<TestParam> testSettings = {
      {{{std::string(100, 'A'), nimble::CompressionType::Uncompressed}}, 0},
      {{{std::string(1'000, 'B'), nimble::CompressionType::Zstd}}, 0},
      {{{std::string(1'000, 'C'), nimble::CompressionType::Zstd, false}}, 0},
      {{{std::string(100, 'D'), nimble::CompressionType::Uncompressed},
        {std::string(100, 'E'), nimble::CompressionType::Uncompressed}},
       0},
      {{{std::string(100, 'F'), nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'G'), nimble::CompressionType::Zstd, false}},
       0},
      {{{std::string(100, 'H'), nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'I'), nimble::CompressionType::Zstd},
        {std::string(100, 'J'), nimble::CompressionType::Uncompressed}},
       0},
      {{{std::string(100, 'K'), nimble::CompressionType::Uncompressed},
        {std::string(100, 'L'), nimble::CompressionType::Uncompressed}},
       1 << 20},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::vector<nimble::MetadataSection> sections;
    std::vector<std::string> fileDataParts;
    buildSections(testData.specs, pool_.get(), sections, fileDataParts);

    const auto fileContent = buildFileContent(sections, fileDataParts);
    auto innerFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
    auto readFile =
        std::make_shared<nimble::testing::TrackingReadFile>(innerFile);
    readFile->setReadError(
        std::make_exception_ptr(std::runtime_error("injected IO error")));

    const auto options = makeMetadataInputOptions(testData.maxCoalesceDistance);
    auto input = nimble::MetadataInput::create(readFile.get(), options);
    EXPECT_THROW(input->load(sections), std::runtime_error);

    // After IO error, retry with error cleared should succeed (stateless).
    readFile->clearReadError();
    auto results = input->load(sections);
    for (uint32_t i = 0; i < results.size(); ++i) {
      EXPECT_EQ(results[i]->content(), testData.specs[i].data);
    }
  }
}

TEST_F(DirectMetadataInputTest, readRawTracksIoStats) {
  const std::string fileContent(4096, 'Z');
  auto innerFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
  auto readFile =
      std::make_shared<nimble::testing::TrackingReadFile>(innerFile);
  constexpr uint64_t kInjectedDelayUs = 5'000;
  readFile->setReadDelayUs(kInjectedDelayUs);

  const auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(readFile.get(), options);

  const auto readCountBefore = metadataIoStats_->read().count();
  const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
  const auto totalScanTimeBefore = metadataIoStats_->totalScanTimeNs();
  const auto storageLatencyCountBefore =
      metadataIoStats_->storageReadLatencyUs().count();
  const auto queryIoLatencyCountBefore =
      metadataIoStats_->queryThreadIoLatencyUs().count();

  std::vector<char> dest(1024);
  input->readRaw(0, dest.size(), dest.data());

  EXPECT_EQ(metadataIoStats_->read().count() - readCountBefore, 1);
  EXPECT_EQ(metadataIoStats_->rawBytesRead() - rawBytesBefore, dest.size());
  EXPECT_GE(
      metadataIoStats_->totalScanTimeNs() - totalScanTimeBefore,
      kInjectedDelayUs * 1'000);
  EXPECT_GT(
      metadataIoStats_->storageReadLatencyUs().count(),
      storageLatencyCountBefore);
  EXPECT_GT(
      metadataIoStats_->queryThreadIoLatencyUs().count(),
      queryIoLatencyCountBefore);

  EXPECT_EQ(
      std::string(dest.data(), dest.size()),
      fileContent.substr(0, dest.size()));
}

// --- CachedMetadataInput tests ---
// Parameterized by whether SSD cache is enabled.

class CachedMetadataInputTest : public MetadataInputTestBase,
                                public ::testing::WithParamInterface<bool> {
 protected:
  static constexpr uint64_t kCacheSize = 64 << 20;
  static constexpr uint64_t kSsdSize = 64 << 20;

  static void SetUpTestCase() {
    velox::common::testutil::TestValue::enable();
  }

  bool enableSsd() const {
    return GetParam();
  }

  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::Options options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = kCacheSize;
    options.arbitratorCapacity = kCacheSize;
    options.trackDefaultUsage = true;
    manager_ = std::make_unique<velox::memory::MemoryManager>(options);
    allocator_ =
        static_cast<velox::memory::MmapAllocator*>(manager_->allocator());

    std::unique_ptr<velox::cache::SsdCache> ssdCache;
    if (enableSsd()) {
      tempDir_ = velox::common::testutil::TempDirectoryPath::create();
      ssdExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(1);
      ssdCache = std::make_unique<velox::cache::SsdCache>(
          velox::cache::SsdCache::Config(
              fmt::format("{}/cache", tempDir_->getPath()),
              kSsdSize,
              1,
              ssdExecutor_.get()));
    }
    cache_ =
        velox::cache::AsyncDataCache::create(allocator_, std::move(ssdCache));
    rootPool_ = manager_->addRootPool("CachedMetadataInputTest");
    pool_ = rootPool_->addLeafChild("leaf");
    fileId_ = std::make_unique<velox::StringIdLease>(
        velox::fileIds(), "CachedMetadataInputTest");
  }

  void TearDown() override {
    fileId_.reset();
    if (cache_ != nullptr) {
      cache_->shutdown();
    }
    cache_.reset();
    pool_.reset();
    rootPool_.reset();
    manager_.reset();
    ssdExecutor_.reset();
    tempDir_.reset();
  }

  std::unique_ptr<nimble::MetadataInput> createCachedInput(
      velox::ReadFile* readFile,
      int32_t maxCoalesceDistance = 1 << 20,
      int64_t maxCoalesceBytes = 128 << 20,
      std::optional<uint32_t> maxCacheEntrySize = std::nullopt) {
    fileHandle_ = std::make_unique<velox::FileHandle>();
    fileHandle_->file = std::shared_ptr<velox::ReadFile>(
        std::shared_ptr<velox::ReadFile>{}, readFile);
    fileHandle_->uuid =
        velox::StringIdLease(velox::fileIds(), "CachedMetadataInputTest");
    auto options =
        makeMetadataInputOptions(maxCoalesceDistance, maxCoalesceBytes);
    options.maxCacheEntrySize = maxCacheEntrySize;
    options.fileHandle = fileHandle_.get();
    options.cache = cache_.get();
    return nimble::MetadataInput::create(readFile, options);
  }

  std::unique_ptr<velox::memory::MemoryManager> manager_;
  velox::memory::MmapAllocator* allocator_{nullptr};
  std::shared_ptr<velox::cache::AsyncDataCache> cache_;
  std::unique_ptr<velox::FileHandle> fileHandle_;
  std::unique_ptr<velox::StringIdLease> fileId_;
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> tempDir_;
  std::unique_ptr<folly::IOThreadPoolExecutor> ssdExecutor_;
};

TEST_P(CachedMetadataInputTest, loadSections) {
  for (const auto& testData : enqueueLoadReadTestSettings()) {
    SCOPED_TRACE(testData.debugString());
    cache_->clear();

    std::vector<nimble::MetadataSection> sections;
    std::vector<std::string> fileDataParts;
    buildSections(testData.specs, pool_.get(), sections, fileDataParts);

    const auto fileContent = buildFileContent(sections, fileDataParts);
    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

    auto input = createCachedInput(readFile.get());
    auto results = input->load(sections);
    ASSERT_EQ(results.size(), sections.size());
    for (uint32_t i = 0; i < sections.size(); ++i) {
      ASSERT_NE(results[i], nullptr);
      EXPECT_EQ(results[i]->content(), testData.specs[i].data);
    }
  }
}

TEST_P(CachedMetadataInputTest, reverseSectionOrder) {
  const std::string data0 = "first-in-file";
  const std::string data1 = "second-in-file";

  const uint64_t offset0 = 0;
  const uint64_t offset1 = data0.size();

  nimble::MetadataSection section0{
      offset0,
      static_cast<uint32_t>(data0.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data0.size())};
  nimble::MetadataSection section1{
      offset1,
      static_cast<uint32_t>(data1.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data1.size())};

  const auto fileContent =
      buildFileContent({section0, section1}, {data0, data1});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  auto input = createCachedInput(readFile.get());
  auto results = input->load(std::array{section1, section0});
  EXPECT_EQ(results[0]->content(), data1);
  EXPECT_EQ(results[1]->content(), data0);
}

TEST_P(CachedMetadataInputTest, loadWithCache) {
  for (const auto& testData : enqueueLoadReadTestSettings()) {
    SCOPED_TRACE(testData.debugString());
    cache_->clear();
    if (enableSsd()) {
      cache_->ssdCache()->clear();
    }

    std::vector<nimble::MetadataSection> sections;
    std::vector<std::string> fileDataParts;
    buildSections(testData.specs, pool_.get(), sections, fileDataParts);

    const auto fileContent = buildFileContent(sections, fileDataParts);
    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

    // First load — populates cache, expect zero ram hits.
    {
      const auto ramHitBefore = metadataIoStats_->ramHit().count();
      const auto storageLatencyBefore =
          metadataIoStats_->storageReadLatencyUs().count();
      auto input = createCachedInput(readFile.get());
      auto results = input->load(sections);
      for (uint32_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i]->content(), testData.specs[i].data);
      }
      EXPECT_EQ(metadataIoStats_->ramHit().count(), ramHitBefore);
      EXPECT_GT(
          metadataIoStats_->storageReadLatencyUs().count(),
          storageLatencyBefore);
    }

    const uint32_t sectionsWithSize = std::count_if(
        testData.specs.begin(), testData.specs.end(), [](const auto& s) {
          return s.hasUncompressedSize;
        });

    // Second load — sections with uncompressed size hit cache directly.
    {
      const auto ramHitBefore = metadataIoStats_->ramHit().count();
      auto input = createCachedInput(readFile.get());
      auto results = input->load(sections);
      for (uint32_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i]->content(), testData.specs[i].data);
      }
      EXPECT_GE(
          metadataIoStats_->ramHit().count() - ramHitBefore, sectionsWithSize);
    }

    // Third load (SSD only) — flush RAM to SSD, clear RAM, read from SSD.
    if (enableSsd()) {
      ASSERT_TRUE(cache_->ssdCache()->startWrite());
      cache_->saveToSsd(/*saveAll=*/true);
      cache_->ssdCache()->waitForWriteToFinish();
      cache_->clear();

      const auto ramHitBefore = metadataIoStats_->ramHit().count();
      const auto ssdReadBefore = metadataIoStats_->ssdRead().count();
      const auto storageLatencyBefore =
          metadataIoStats_->storageReadLatencyUs().count();

      auto input = createCachedInput(readFile.get());
      auto results = input->load(sections);
      for (uint32_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i]->content(), testData.specs[i].data);
      }

      EXPECT_EQ(metadataIoStats_->ramHit().count(), ramHitBefore);
      EXPECT_GE(
          metadataIoStats_->ssdRead().count() - ssdReadBefore,
          sectionsWithSize);
      EXPECT_EQ(
          metadataIoStats_->storageReadLatencyUs().count(),
          storageLatencyBefore);
    }
  }
}

TEST_P(CachedMetadataInputTest, repeatedLoad) {
  const std::string data = "cached-repeat-test";
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  auto input = createCachedInput(readFile.get());

  // Stateless: can call load() multiple times without reset.
  auto results1 = input->load(std::array{section});
  EXPECT_EQ(results1[0]->content(), data);

  auto results2 = input->load(std::array{section});
  EXPECT_EQ(results2[0]->content(), data);

  // Empty input throws.
  NIMBLE_ASSERT_THROW(
      input->load(std::span<const nimble::MetadataSection>{}),
      "No sections to load");
}

DEBUG_ONLY_TEST_P(CachedMetadataInputTest, ioCoalescing) {
  struct TestParam {
    std::vector<SectionLayout> sections;
    int32_t maxCoalesceDistance;
    int64_t maxCoalesceBytes;
    int32_t expectedNumIos;
    uint64_t expectedOverread;
    std::string debugString() const {
      return fmt::format(
          "{} sections, maxCoalesceDistance {}, maxCoalesceBytes {}, expectedNumIos {}, expectedOverread {}",
          sections.size(),
          maxCoalesceDistance,
          maxCoalesceBytes,
          expectedNumIos,
          expectedOverread);
    }
  };

  constexpr int64_t kLargeBytes = 128 << 20;

  std::vector<TestParam> testSettings = {
      {{{0, 100}}, 0, kLargeBytes, 1, 0},
      {{{0, 100}, {100, 200}}, 0, kLargeBytes, 1, 0},
      {{{0, 50}, {50, 60}, {110, 90}}, 0, kLargeBytes, 1, 0},
      {{{0, 100}, {150, 100}}, 0, kLargeBytes, 2, 0},
      {{{0, 100}, {150, 100}}, 100, kLargeBytes, 1, 50},
      {{{0, 100}, {250, 100}}, 100, kLargeBytes, 2, 0},
      {{{0, 100}, {120, 80}, {500, 100}}, 50, kLargeBytes, 2, 20},
      {{{0, 100}, {200, 100}, {400, 100}}, 200, kLargeBytes, 1, 200},
      {{{0, 1'000}, {10'000, 1'000}}, 100, kLargeBytes, 2, 0},
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 150, 2, 0},
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 200, 2, 0},
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 300, 1, 0},
      {{{0, 100}, {100, 100}, {200, 100}}, 0, 50, 3, 0},
      {{{0, 100}, {150, 100}, {300, 100}}, 200, 200, 2, 50},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    cache_->clear();
    if (enableSsd()) {
      cache_->ssdCache()->clear();
    }

    uint64_t fileSize = 0;
    std::vector<std::string> sectionData;
    std::vector<nimble::MetadataSection> sections;
    for (uint32_t i = 0; i < testData.sections.size(); ++i) {
      const auto& layout = testData.sections[i];
      fileSize = std::max(fileSize, layout.offset + layout.size);
      sectionData.emplace_back(layout.size, 'A' + i);
      sections.emplace_back(
          layout.offset,
          layout.size,
          nimble::CompressionType::Uncompressed,
          layout.size);
    }

    std::string fileContent(fileSize, '\xCC');
    for (uint32_t i = 0; i < sections.size(); ++i) {
      std::memcpy(
          fileContent.data() + sections[i].offset(),
          sectionData[i].data(),
          sectionData[i].size());
    }

    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

    uint64_t expectedPayloadBytes = 0;
    for (const auto& s : testData.sections) {
      expectedPayloadBytes += s.size;
    }

    // First load — cold cache, expect file IO with coalescing.
    {
      const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
      const auto overreadBefore = metadataIoStats_->rawOverreadBytes();
      const auto ramHitBefore = metadataIoStats_->ramHit().count();

      velox::CoalesceIoStats capturedStats;
      SCOPED_TESTVALUE_SET(
          "facebook::nimble::MetadataInput::computeIoGroups",
          std::function<void(const velox::CoalesceIoStats*)>(
              [&](const velox::CoalesceIoStats* stats) {
                capturedStats = *stats;
              }));

      auto input = createCachedInput(
          readFile.get(),
          testData.maxCoalesceDistance,
          testData.maxCoalesceBytes);
      input->load(sections);

      EXPECT_EQ(capturedStats.numIos, testData.expectedNumIos);
      EXPECT_EQ(capturedStats.extraBytes, testData.expectedOverread);
      EXPECT_EQ(
          metadataIoStats_->rawBytesRead() - rawBytesBefore,
          expectedPayloadBytes);
      EXPECT_EQ(
          metadataIoStats_->rawOverreadBytes() - overreadBefore,
          testData.expectedOverread);
      EXPECT_EQ(metadataIoStats_->ramHit().count(), ramHitBefore);
    }

    // Second load — warm cache, expect all RAM hits, zero file IO.
    {
      const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
      const auto overreadBefore = metadataIoStats_->rawOverreadBytes();
      const auto ramHitBefore = metadataIoStats_->ramHit().count();
      const auto storageLatencyBefore =
          metadataIoStats_->storageReadLatencyUs().count();

      auto input = createCachedInput(readFile.get());
      input->load(sections);

      EXPECT_EQ(metadataIoStats_->rawBytesRead(), rawBytesBefore);
      EXPECT_EQ(metadataIoStats_->rawOverreadBytes(), overreadBefore);
      EXPECT_EQ(
          metadataIoStats_->ramHit().count() - ramHitBefore, sections.size());
      EXPECT_EQ(
          metadataIoStats_->storageReadLatencyUs().count(),
          storageLatencyBefore);
    }

    // Third load (SSD only) — flush RAM to SSD, clear RAM, read from SSD.
    if (enableSsd()) {
      ASSERT_TRUE(cache_->ssdCache()->startWrite());
      cache_->saveToSsd(/*saveAll=*/true);
      cache_->ssdCache()->waitForWriteToFinish();
      cache_->clear();

      const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
      const auto overreadBefore = metadataIoStats_->rawOverreadBytes();
      const auto ramHitBefore = metadataIoStats_->ramHit().count();
      const auto ssdReadBefore = metadataIoStats_->ssdRead().count();
      const auto storageLatencyBefore =
          metadataIoStats_->storageReadLatencyUs().count();

      auto input = createCachedInput(readFile.get());
      input->load(sections);

      EXPECT_EQ(metadataIoStats_->rawBytesRead(), rawBytesBefore);
      EXPECT_EQ(metadataIoStats_->rawOverreadBytes(), overreadBefore);
      EXPECT_EQ(metadataIoStats_->ramHit().count(), ramHitBefore);
      EXPECT_EQ(
          metadataIoStats_->ssdRead().count() - ssdReadBefore, sections.size());
      EXPECT_EQ(
          metadataIoStats_->storageReadLatencyUs().count(),
          storageLatencyBefore);
    }
  }
}

TEST_P(CachedMetadataInputTest, partialCacheHit) {
  const std::string data0 = "partial-hit-0";
  const std::string data1 = "partial-hit-1";

  const uint64_t offset0 = 0;
  const uint64_t offset1 = data0.size();

  nimble::MetadataSection section0{
      offset0,
      static_cast<uint32_t>(data0.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data0.size())};
  nimble::MetadataSection section1{
      offset1,
      static_cast<uint32_t>(data1.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data1.size())};

  const auto fileContent =
      buildFileContent({section0, section1}, {data0, data1});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
  cache_->clear();

  // Load only section0 to populate cache for it.
  {
    auto input = createCachedInput(readFile.get());
    auto results = input->load(std::array{section0});
    EXPECT_EQ(results[0]->content(), data0);
  }

  // Load both sections — section0 should be a cache hit, section1 a miss.
  {
    const auto ramHitBefore = metadataIoStats_->ramHit().count();
    auto input = createCachedInput(readFile.get());
    auto results = input->load(std::array{section0, section1});
    EXPECT_EQ(results[0]->content(), data0);
    EXPECT_EQ(results[1]->content(), data1);
    EXPECT_GT(metadataIoStats_->ramHit().count(), ramHitBefore);
  }

  // SSD partial hit: flush section0+section1 to SSD, clear RAM, then load
  // only section1 so section0 stays only on SSD. Reload both — section0
  // should come from SSD, section1 from RAM.
  if (enableSsd()) {
    ASSERT_TRUE(cache_->ssdCache()->startWrite());
    cache_->saveToSsd(/*saveAll=*/true);
    cache_->ssdCache()->waitForWriteToFinish();
    cache_->clear();

    // Reload only section1 into RAM.
    {
      auto input = createCachedInput(readFile.get());
      auto results = input->load(std::array{section1});
      EXPECT_EQ(results[0]->content(), data1);
    }

    const auto ssdReadBefore = metadataIoStats_->ssdRead().count();
    const auto ramHitBefore = metadataIoStats_->ramHit().count();
    const auto storageLatencyBefore =
        metadataIoStats_->storageReadLatencyUs().count();
    {
      auto input = createCachedInput(readFile.get());
      auto results = input->load(std::array{section0, section1});
      EXPECT_EQ(results[0]->content(), data0);
      EXPECT_EQ(results[1]->content(), data1);
    }
    // section0 from SSD, section1 from RAM, no file IO.
    EXPECT_GT(metadataIoStats_->ssdRead().count(), ssdReadBefore);
    EXPECT_GT(metadataIoStats_->ramHit().count(), ramHitBefore);
    EXPECT_EQ(
        metadataIoStats_->storageReadLatencyUs().count(), storageLatencyBefore);
  }
}

TEST_P(CachedMetadataInputTest, ioError) {
  struct TestParam {
    std::vector<SectionSpec> specs;
    int32_t maxCoalesceDistance;
    std::string debugString() const {
      return fmt::format(
          "{} sections, maxCoalesceDistance {}, [{}]",
          specs.size(),
          maxCoalesceDistance,
          fmt::join(
              specs | std::views::transform([](const auto& s) {
                return fmt::format(
                    "comp{}:sz{}:uncomp{}",
                    static_cast<int>(s.compressionType),
                    s.data.size(),
                    s.hasUncompressedSize);
              }),
              ", "));
    }
  };

  std::vector<TestParam> testSettings = {
      // Single uncompressed, 1 IO group.
      {{{std::string(100, 'A'), nimble::CompressionType::Uncompressed}}, 0},
      // Single compressed with known size.
      {{{std::string(1'000, 'B'), nimble::CompressionType::Zstd}}, 0},
      // Single compressed without known size.
      {{{std::string(1'000, 'C'), nimble::CompressionType::Zstd, false}}, 0},
      // Two sections, no coalescing → 2 IO groups (async dispatch).
      {{{std::string(100, 'D'), nimble::CompressionType::Uncompressed},
        {std::string(100, 'E'), nimble::CompressionType::Uncompressed}},
       0},
      // Mixed compression with/without known size.
      {{{std::string(100, 'F'), nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'G'), nimble::CompressionType::Zstd, false}},
       0},
      // Three sections, no coalescing → 3 IO groups.
      {{{std::string(100, 'H'), nimble::CompressionType::Uncompressed},
        {std::string(1'000, 'I'), nimble::CompressionType::Zstd},
        {std::string(100, 'J'), nimble::CompressionType::Uncompressed}},
       0},
      // Two sections coalesced → 1 IO group.
      {{{std::string(100, 'K'), nimble::CompressionType::Uncompressed},
        {std::string(100, 'L'), nimble::CompressionType::Uncompressed}},
       1 << 20},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    cache_->clear();

    std::vector<nimble::MetadataSection> sections;
    std::vector<std::string> fileDataParts;
    buildSections(testData.specs, pool_.get(), sections, fileDataParts);

    const auto fileContent = buildFileContent(sections, fileDataParts);
    auto innerFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
    auto readFile =
        std::make_shared<nimble::testing::TrackingReadFile>(innerFile);
    readFile->setReadError(
        std::make_exception_ptr(std::runtime_error("injected IO error")));

    auto input =
        createCachedInput(readFile.get(), testData.maxCoalesceDistance);
    EXPECT_THROW(input->load(sections), std::runtime_error);

    // After IO error, retry with error cleared should succeed (stateless).
    readFile->clearReadError();
    auto results = input->load(sections);
    for (uint32_t i = 0; i < results.size(); ++i) {
      EXPECT_EQ(results[i]->content(), testData.specs[i].data);
    }
  }
}

DEBUG_ONLY_TEST_P(CachedMetadataInputTest, ssdIoError) {
  if (!enableSsd()) {
    return;
  }

  const std::string data = "ssd-io-error-test";
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
  cache_->clear();

  // Populate cache, flush to SSD, clear RAM.
  {
    auto input = createCachedInput(readFile.get());
    auto results = input->load(std::array{section});
    EXPECT_EQ(results[0]->content(), data);
  }
  ASSERT_TRUE(cache_->ssdCache()->startWrite());
  cache_->saveToSsd(/*saveAll=*/true);
  cache_->ssdCache()->waitForWriteToFinish();
  cache_->clear();

  // Inject SSD read error — load() should propagate the exception.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cache::SsdFile::load",
      std::function<void(velox::cache::SsdFile*)>(
          [&](velox::cache::SsdFile* /*ssdFile*/) {
            VELOX_FAIL("injected SSD IO error");
          }));
  {
    auto input = createCachedInput(readFile.get());
    VELOX_ASSERT_THROW(
        input->load(std::array{section}), "injected SSD IO error");
  }
}

// SsdRun packs an entry's size into kSizeBits bits, so a larger metadata
// section can never be written to SSD. Queueing one anyway either spins in
// SsdFile::getSpace() above kRegionSize or throws when the SsdRun is
// constructed below it, so oversized sections must bypass the cache.
TEST_P(CachedMetadataInputTest, oversizedSectionIsNotCached) {
  velox::cache::test::AsyncDataCacheTestHelper cacheHelper{cache_.get()};

  // Counts entries of exactly 'size'. cache_->clear() leaves freed entry
  // shells behind, so a bare entry count would also see earlier sections.
  const auto countCachedOfSize = [&](uint32_t size) {
    const auto entries = cacheHelper.cacheEntries();
    return std::count_if(
        entries.begin(), entries.end(), [&](const auto* entry) {
          return entry != nullptr &&
              entry->size() == static_cast<int32_t>(size);
        });
  };

  const auto loadSectionOfSize = [&](uint32_t size, uint32_t limit) {
    const std::string data(size, 'x');
    const nimble::MetadataSection section{
        /*offset=*/0,
        size,
        nimble::CompressionType::Uncompressed,
        /*uncompressedSize=*/size,
    };
    const auto fileContent = buildFileContent({section}, {data});
    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
    auto input = createCachedInput(
        readFile.get(),
        /*maxCoalesceDistance=*/1 << 20,
        /*maxCoalesceBytes=*/128 << 20,
        limit);
    auto results = input->load(std::array{section});
    EXPECT_EQ(results[0]->content().size(), size);
    // 'results' still holds any pin, so a cached section is visible here.
    return countCachedOfSize(size);
  };

  constexpr uint32_t kLimit = 1U << velox::cache::SsdRun::kSizeBits;

  // Distinct sizes so each case is distinguishable from earlier residue.
  EXPECT_EQ(loadSectionOfSize(kLimit + (1 << 20), kLimit), 0);
  EXPECT_EQ(loadSectionOfSize(kLimit / 2, kLimit), 1);
}

TEST_P(CachedMetadataInputTest, cachePinRetention) {
  const std::string data = "pin-retention-test-data";
  nimble::MetadataSection section{
      0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      static_cast<uint32_t>(data.size())};
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);
  cache_->clear();

  // Load — MetadataBuffer holds the cache pin reference.
  auto input = createCachedInput(readFile.get());
  auto results = input->load(std::array{section});
  auto buffer = std::move(results[0]);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->content(), data);

  // Clear the cache — the entry is still pinned by the MetadataBuffer,
  // so it can't be evicted.
  cache_->clear();

  // The buffer still holds valid data via the cache pin.
  EXPECT_EQ(buffer->content(), data);

  // A second load should still hit RAM cache because the pin keeps the
  // entry alive even after clear().
  const auto ramHitBefore = metadataIoStats_->ramHit().count();
  {
    auto input2 = createCachedInput(readFile.get());
    auto results2 = input2->load(std::array{section});
    ASSERT_NE(results2[0], nullptr);
    EXPECT_EQ(results2[0]->content(), data);
  }
  EXPECT_GT(metadataIoStats_->ramHit().count(), ramHitBefore);

  // Drop the original buffer — releases the pin.
  buffer.reset();

  // Now clear should fully evict.
  cache_->clear();

  // Third load — cold miss, requires file IO.
  const auto ramHitBefore2 = metadataIoStats_->ramHit().count();
  const auto storageLatencyBefore =
      metadataIoStats_->storageReadLatencyUs().count();
  {
    auto input3 = createCachedInput(readFile.get());
    auto results3 = input3->load(std::array{section});
    ASSERT_NE(results3[0], nullptr);
    EXPECT_EQ(results3[0]->content(), data);
  }
  EXPECT_EQ(metadataIoStats_->ramHit().count(), ramHitBefore2);
  EXPECT_GT(
      metadataIoStats_->storageReadLatencyUs().count(), storageLatencyBefore);
}

TEST_F(DirectMetadataInputTest, cacheMethodsThrow) {
  auto file = std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));
  auto options = makeMetadataInputOptions();
  auto input = nimble::MetadataInput::create(file.get(), options);
  EXPECT_FALSE(input->cached());
  NIMBLE_ASSERT_THROW(input->findCachedMetadata(0), "");
  std::string_view data = "test";
  std::span<const std::string_view> ranges(&data, 1);
  NIMBLE_ASSERT_THROW(input->cacheMetadata(0, ranges), "");
}

TEST_P(CachedMetadataInputTest, cacheAndFindMetadata) {
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));
  auto input = createCachedInput(readFile.get());
  EXPECT_TRUE(input->cached());

  // Cache miss before caching.
  EXPECT_EQ(input->findCachedMetadata(42), nullptr);

  // Cache data at offset 42.
  std::string data1 = "hello";
  std::string data2 = "world";
  std::array<std::string_view, 2> ranges = {data1, data2};
  input->cacheMetadata(42, ranges);

  // Cache hit after caching.
  auto result = input->findCachedMetadata(42);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content().size(), data1.size() + data2.size());
  EXPECT_EQ(result->content(), std::string_view("helloworld"));
}

TEST_P(CachedMetadataInputTest, cacheAndFindMetadataRespectsSizeLimit) {
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));
  constexpr uint32_t kLimit = 1 << 10;
  auto input = createCachedInput(
      readFile.get(),
      /*maxCoalesceDistance=*/1 << 20,
      /*maxCoalesceBytes=*/128 << 20,
      kLimit);

  const std::string cacheableData(kLimit, 'a');
  std::array<std::string_view, 1> cacheableRanges{cacheableData};
  input->cacheMetadata(42, cacheableRanges);
  EXPECT_NE(input->findCachedMetadata(42), nullptr);

  const std::string oversizedData(kLimit + 1, 'b');
  std::array<std::string_view, 1> oversizedRanges{oversizedData};
  input->cacheMetadata(84, oversizedRanges);
  EXPECT_EQ(input->findCachedMetadata(84), nullptr);
}

TEST_P(CachedMetadataInputTest, loweredLimitIgnoresExistingRamEntry) {
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));
  constexpr uint32_t kLowerLimit = 1 << 10;
  const std::string data(kLowerLimit + 1, 'a');
  std::array<std::string_view, 1> ranges{data};

  {
    auto input = createCachedInput(readFile.get());
    input->cacheMetadata(42, ranges);
    EXPECT_NE(input->findCachedMetadata(42), nullptr);
  }

  auto input = createCachedInput(
      readFile.get(),
      /*maxCoalesceDistance=*/1 << 20,
      /*maxCoalesceBytes=*/128 << 20,
      kLowerLimit);
  EXPECT_EQ(input->findCachedMetadata(42), nullptr);
}

TEST_P(CachedMetadataInputTest, unsetLimitPreservesExistingBehavior) {
  constexpr uint32_t kSize = (1U << velox::cache::SsdRun::kSizeBits) + 1;
  const std::string data(kSize, 'x');
  const nimble::MetadataSection section{
      /*offset=*/0,
      kSize,
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/kSize,
  };
  auto readFile = std::make_shared<velox::InMemoryReadFile>(data);
  auto input = createCachedInput(readFile.get());

  auto results = input->load(std::array{section});
  EXPECT_EQ(results[0]->content(), data);
  auto cached = input->findCachedMetadata(0);
  ASSERT_NE(cached, nullptr);
  EXPECT_EQ(cached->content(), data);
}

// A limit above the SsdRun ceiling would queue entries that can never be
// written to SSD, so it is rejected when the SSD cache is enabled.
TEST_P(CachedMetadataInputTest, ssdCacheRejectsOversizedLimit) {
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));
  const auto createOversized = [&]() {
    return createCachedInput(
        readFile.get(),
        /*maxCoalesceDistance=*/1 << 20,
        /*maxCoalesceBytes=*/128 << 20,
        (1U << velox::cache::SsdRun::kSizeBits) + 1);
  };

  if (enableSsd()) {
    NIMBLE_ASSERT_THROW(
        createOversized(),
        "MetadataInput::Options::maxCacheEntrySize must not exceed the Velox "
        "SSD cache maximum entry size when an SSD cache is configured");
  } else {
    EXPECT_NE(createOversized(), nullptr);
  }
}

TEST_P(CachedMetadataInputTest, loweredLimitSkipsExistingSsdEntry) {
  if (!enableSsd()) {
    GTEST_SKIP() << "Requires SSD cache";
  }

  constexpr uint32_t kLowerLimit = 1 << 10;
  const std::string data(kLowerLimit + 1, 'a');
  const nimble::MetadataSection section{
      /*offset=*/0,
      static_cast<uint32_t>(data.size()),
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/static_cast<uint32_t>(data.size()),
  };
  const auto fileContent = buildFileContent({section}, {data});
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileContent);

  {
    auto input = createCachedInput(readFile.get());
    auto results = input->load(std::array{section});
    EXPECT_EQ(results[0]->content(), data);
  }
  ASSERT_TRUE(cache_->ssdCache()->startWrite());
  cache_->saveToSsd(/*saveAll=*/true);
  cache_->ssdCache()->waitForWriteToFinish();
  cache_->clear();

  const auto rawBytesBefore = metadataIoStats_->rawBytesRead();
  const auto ssdReadBefore = metadataIoStats_->ssdRead().count();
  auto input = createCachedInput(
      readFile.get(),
      /*maxCoalesceDistance=*/1 << 20,
      /*maxCoalesceBytes=*/128 << 20,
      kLowerLimit);
  auto results = input->load(std::array{section});
  EXPECT_EQ(results[0]->content(), data);
  EXPECT_GT(metadataIoStats_->rawBytesRead(), rawBytesBefore);
  EXPECT_EQ(metadataIoStats_->ssdRead().count(), ssdReadBefore);
}

DEBUG_ONLY_TEST_P(CachedMetadataInputTest, concurrentCacheAndFind) {
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string(100, 'x'));

  // Verifies that findCachedMetadata blocks while another thread holds an
  // exclusive cache pin via cacheMetadata, then returns the cached result
  // once the pin is promoted to shared.
  std::string cachedData(1'000, 'Z');
  std::array<std::string_view, 1> ranges = {cachedData};

  folly::Baton<> pinAcquired;
  folly::Baton<> writerResume;

  SCOPED_TESTVALUE_SET(
      "facebook::nimble::CachedMetadataInput::cacheMetadata",
      std::function<void(const velox::cache::CachePin*)>(
          [&](const velox::cache::CachePin*) {
            pinAcquired.post();
            writerResume.wait();
          }));

  auto input1 = createCachedInput(readFile.get());
  auto input2 = createCachedInput(readFile.get());

  std::atomic_bool finderDone{false};
  std::unique_ptr<nimble::MetadataBuffer> foundResult;

  // Start cacher thread — will block in TestValue after acquiring exclusive
  // pin.
  std::thread cacher([&] { input1->cacheMetadata(99, ranges); });

  pinAcquired.wait();

  // Start finder thread — will block in findCachedMetadata waiting for the
  // exclusive pin to be released.
  std::thread finder([&] {
    foundResult = input2->findCachedMetadata(99);
    finderDone.store(true);
  });

  // Verify finder is blocked while cacher holds the exclusive pin.
  EXPECT_EQ(foundResult, nullptr);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  EXPECT_FALSE(finderDone.load());

  // Resume cacher — writes data and promotes pin to shared.
  writerResume.post();
  cacher.join();
  finder.join();

  EXPECT_TRUE(finderDone.load());
  ASSERT_NE(foundResult, nullptr);
  EXPECT_EQ(foundResult->content().size(), cachedData.size());
  EXPECT_EQ(foundResult->content(), cachedData);
}

INSTANTIATE_TEST_SUITE_P(
    CachedMetadataInputTests,
    CachedMetadataInputTest,
    ::testing::Values(false, true),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "WithSsd" : "WithoutSsd";
    });

} // namespace
