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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include "velox/common/base/CoalesceIo.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
#include "velox/dwio/nimble/tablet/Compression.h"

#include "folly/executors/InlineExecutor.h"
#include "folly/executors/ManualExecutor.h"

DECLARE_bool(velox_ssd_odirect);

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

TEST_F(DirectMetadataInputTest, executorStartsNoTaskDuringLoad) {
  // The gap between the sections exceeds the coalesce distance, so they form
  // two IO groups and load() hands the second group to the executor.
  const std::string data0 = "first";
  const std::string data1 = "second";
  const nimble::MetadataSection section0{
      /*offset=*/0,
      /*size=*/static_cast<uint32_t>(data0.size()),
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/static_cast<uint32_t>(data0.size()),
  };
  const nimble::MetadataSection section1{
      /*offset=*/data0.size() + 100,
      /*size=*/static_cast<uint32_t>(data1.size()),
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/static_cast<uint32_t>(data1.size()),
  };
  const auto fileContent =
      buildFileContent({section0, section1}, {data0, data1});
  auto readFile = std::make_shared<nimble::testing::TrackingReadFile>(
      std::make_shared<velox::InMemoryReadFile>(fileContent));

  // The manual executor starts no task until drain(), so load() reads both
  // groups itself and returns while the task is still queued.
  folly::ManualExecutor executor;
  auto options = makeMetadataInputOptions(/*maxCoalesceDistance=*/0);
  options.executor = &executor;
  auto input = nimble::MetadataInput::create(readFile.get(), options);
  const auto results = input->load(std::array{section0, section1});
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->content(), data0);
  EXPECT_EQ(results[1]->content(), data1);

  // load() reads the unqueued group before it claims the queued one. That
  // gives a running executor time to start the queued one.
  const auto ioGroups = readFile->ioGroups();
  ASSERT_EQ(ioGroups.size(), 2);
  EXPECT_EQ(ioGroups[0].offset, section0.offset());
  EXPECT_EQ(ioGroups[1].offset, section1.offset());

  // The queued task runs after load() returns and after the MetadataInput is
  // destroyed, as when a cached tablet is evicted. ASAN reports any access to
  // the freed state.
  input.reset();
  EXPECT_EQ(executor.drain(), 1);
}

TEST_F(DirectMetadataInputTest, executorReadsAllButFirstGroup) {
  // The gap between the sections exceeds the coalesce distance, so they form
  // two IO groups.
  const std::string data0 = "first";
  const std::string data1 = "second";
  const nimble::MetadataSection section0{
      /*offset=*/0,
      /*size=*/static_cast<uint32_t>(data0.size()),
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/static_cast<uint32_t>(data0.size()),
  };
  const nimble::MetadataSection section1{
      /*offset=*/data0.size() + 100,
      /*size=*/static_cast<uint32_t>(data1.size()),
      nimble::CompressionType::Uncompressed,
      /*uncompressedSize=*/static_cast<uint32_t>(data1.size()),
  };
  const auto fileContent =
      buildFileContent({section0, section1}, {data0, data1});
  auto readFile = std::make_shared<nimble::testing::TrackingReadFile>(
      std::make_shared<velox::InMemoryReadFile>(fileContent));

  // The inline executor runs each task inside add(), so the first read is the
  // group load() queued. load() then reads the first group and takes the
  // queued group's finished read.
  auto options = makeMetadataInputOptions(/*maxCoalesceDistance=*/0);
  options.executor = &folly::InlineExecutor::instance();
  auto input = nimble::MetadataInput::create(readFile.get(), options);
  const auto results = input->load(std::array{section0, section1});
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->content(), data0);
  EXPECT_EQ(results[1]->content(), data1);

  const auto ioGroups = readFile->ioGroups();
  ASSERT_EQ(ioGroups.size(), 2);
  EXPECT_EQ(ioGroups[0].offset, section1.offset());
  EXPECT_EQ(ioGroups[1].offset, section0.offset());
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

} // namespace

// --- CachedMetadataInput tests ---

namespace facebook::nimble {
namespace {

constexpr uint64_t kAlignment{4'096};

// File offset and length of each range that one read fills.
using Read = std::vector<std::pair<uint64_t, uint64_t>>;

// In-memory file that records each positioned vector read, and each scatter
// read, which it accepts only without direct I/O. With 'alignment' above 1 it
// reports direct I/O and rejects unaligned reads. A direct-I/O read past the
// end of the file then returns only the bytes before the end, as on Linux, and
// leaves junk in the rest of its buffer. Without direct I/O, a read past the
// end of the file fails. Runs an optional hook before each read.
class RecordingReadFile final : public velox::InMemoryReadFile {
 public:
  RecordingReadFile(std::string content, uint64_t alignment)
      : velox::InMemoryReadFile{std::move(content)}, alignment_{alignment} {}

  using velox::InMemoryReadFile::preadv;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const velox::FileIoContext& context = {}) const override {
    NIMBLE_CHECK_EQ(alignment_, 1, "Scatter read of a direct-I/O file");
    if (failReads_) {
      NIMBLE_FAIL("Injected preadv failure");
    }
    if (beforeRead_) {
      beforeRead_();
    }
    const auto start = offset;
    Read read;
    for (const auto& buffer : buffers) {
      if (buffer.data() != nullptr) {
        NIMBLE_CHECK_LE(
            offset + buffer.size(), size(), "Read past the end of the file");
        pread(offset, buffer.size(), buffer.data(), context);
        read.emplace_back(offset, buffer.size());
        scatterDestinations_.wlock()->push_back(buffer.data());
      }
      offset += buffer.size();
    }
    scatterReads_.wlock()->push_back(std::move(read));
    return offset - start;
  }

  uint64_t preadv(
      folly::Range<const velox::common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      const velox::FileIoContext& context = {}) const override {
    if (failReads_) {
      NIMBLE_FAIL("Injected preadv failure");
    }
    if (beforeRead_) {
      beforeRead_();
    }
    Read read;
    uint64_t bytesRead{0};
    for (size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      if (alignment_ == 1) {
        NIMBLE_CHECK_LE(
            region.offset + region.length,
            size(),
            "Read past the end of the file");
      }
      NIMBLE_CHECK_EQ(region.offset % alignment_, 0, "Unaligned read offset");
      NIMBLE_CHECK_EQ(region.length % alignment_, 0, "Unaligned read length");
      NIMBLE_CHECK_EQ(
          reinterpret_cast<uintptr_t>(buffers[i].data()) % alignment_,
          0,
          "Unaligned read buffer");
      const auto bytesBeforeFileEnd =
          std::min(region.length, size() - std::min(region.offset, size()));
      pread(region.offset, bytesBeforeFileEnd, buffers[i].data(), context);
      std::memset(
          buffers[i].data() + bytesBeforeFileEnd,
          kJunk,
          region.length - bytesBeforeFileEnd);
      bytesRead += bytesBeforeFileEnd;
      read.emplace_back(region.offset, region.length);
    }
    reads_.wlock()->push_back(std::move(read));
    return bytesRead;
  }

  bool directIo(uint64_t& alignment) const override {
    alignment = alignment_;
    return alignment_ > 1;
  }

  std::vector<Read> reads() const {
    return *reads_.rlock();
  }

  std::vector<Read> scatterReads() const {
    return *scatterReads_.rlock();
  }

  // Buffers that the ranges of scatter reads filled, in the order filled.
  std::vector<const char*> scatterDestinations() const {
    return *scatterDestinations_.rlock();
  }

  void setFailReads(bool failReads) {
    failReads_ = failReads;
  }

  void setBeforeRead(std::function<void()> beforeRead) {
    beforeRead_ = std::move(beforeRead);
  }

 private:
  static constexpr char kJunk{'\xCC'};

  const uint64_t alignment_;
  std::atomic_bool failReads_{false};
  std::function<void()> beforeRead_;
  mutable folly::Synchronized<std::vector<Read>> reads_;
  mutable folly::Synchronized<std::vector<Read>> scatterReads_;
  mutable folly::Synchronized<std::vector<const char*>> scatterDestinations_;
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

  Entry find(uint64_t fileId, uint64_t offset, velox::io::IoStatistics& ioStats)
      override {
    return cache_->find(fileId, offset, ioStats);
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

class CachedMetadataInputTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ =
        velox::memory::memoryManager()->addLeafPool("CachedMetadataInputTest");

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
        .fileHandle = &fileHandle_,
        .blockCache = blockCache,
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

  // AsyncDataCache keys need file ids leased from velox::fileIds(). The inputs
  // read only the id of the handle.
  const velox::FileHandle fileHandle_{
      .uuid = {velox::fileIds(), "CachedMetadataInputTest"}};
};

TEST_F(CachedMetadataInputTest, create) {
  auto [file, sections] = makeFile({}, 100, 1);
  {
    auto input = MetadataInput::create(file.get(), makeOptions(nullptr));
    EXPECT_NE(dynamic_cast<DirectMetadataInput*>(input.get()), nullptr);
    EXPECT_FALSE(input->cached());
  }
  {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
    EXPECT_NE(dynamic_cast<CachedMetadataInput*>(input.get()), nullptr);
    EXPECT_TRUE(input->cached());
  }
  {
    NIMBLE_ASSERT_USER_THROW(
        MetadataInput::create(
            file.get(), makeOptions(blockCache_, /*maxCoalesceDistance=*/-1)),
        "maxCoalesceDistance must not be negative");
  }
  {
    NIMBLE_ASSERT_USER_THROW(
        MetadataInput::create(
            file.get(),
            makeOptions(
                blockCache_,
                /*maxCoalesceDistance=*/0,
                /*maxCoalesceBytes=*/-1)),
        "maxCoalesceBytes must not be negative");
  }
  {
    auto options = makeOptions(blockCache_);
    options.fileHandle = nullptr;
    NIMBLE_ASSERT_USER_THROW(
        MetadataInput::create(file.get(), options),
        "blockCache requires fileHandle");
  }
  {
    const velox::FileHandle fileHandle;
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    NIMBLE_ASSERT_USER_THROW(
        MetadataInput::create(file.get(), options),
        "blockCache requires fileHandle->uuid");
  }
}

// The input keeps its file id registered in velox::fileIds(), so it can create
// cache entries after the caller destroys the file handle, which held the only
// lease on the id.
TEST_F(CachedMetadataInputTest, keepsFileId) {
  auto [file, sections] = makeFile({{0, "section"}}, 100, 1);
  std::unique_ptr<MetadataInput> input;
  {
    const velox::FileHandle fileHandle{
        .uuid = {velox::fileIds(), "CachedMetadataInputTest.keepsFileId"}};
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    input = MetadataInput::create(file.get(), options);
  }
  EXPECT_EQ(
      contents(input->load(sections)), std::vector<std::string>{"section"});
}

TEST_F(CachedMetadataInputTest, loadSections) {
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
    SCOPED_TRACE(fmt::format("alignment {}", alignment));
    auto [file, sections] = makeFile(specs, 12'317, alignment);

    // A file id per file, so no file hits another file's entries.
    const velox::FileHandle fileHandle{
        .uuid = {
            velox::fileIds(),
            fmt::format("CachedMetadataInputTest.loadSections.{}", alignment)}};
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);

    // Requested in reverse file order.
    const std::array requested{
        sections[3], sections[2], sections[1], sections[0]};
    EXPECT_EQ(contents(input->load(requested)), expected);
    if (alignment == 1) {
      // A scatter read per section, since no gap is within the coalesce
      // distance of 0.
      EXPECT_TRUE(file->reads().empty());
      EXPECT_EQ(file->scatterReads().size(), 4);
    } else {
      EXPECT_EQ(file->reads().size(), 1);
    }
  }
}

TEST_F(CachedMetadataInputTest, readsInOneAlignedBatch) {
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
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  EXPECT_EQ(
      contents(input->load(sections)),
      (std::vector<std::string>{
          std::string(100, 'a'),
          std::string(100, 'b'),
          std::string(50, 'c'),
          std::string(30, 'd'),
      }));

  // The last read runs past the end of the file.
  const std::vector<Read> expected{{
      {0, kAlignment},
      {2 * kAlignment, kAlignment},
      {4 * kAlignment, kAlignment},
      {6 * kAlignment, kAlignment},
  }};
  EXPECT_EQ(file->reads(), expected);
}

// Without direct I/O, each merged range is one scatter read, which skips the
// gaps between its sections. An uncompressed section is read straight into its
// new cache entry, so caching it takes no copy.
TEST_F(CachedMetadataInputTest, readsIntoCacheEntriesWithoutDirectIo) {
  auto [file, sections] = makeFile(
      {
          {0, std::string(10, 'a')},
          {50, std::string(10, 'b')},
          {100, std::string(1'000, 'c'), CompressionType::Zstd},
          {5'000, std::string(10, 'd')},
          {5'100, std::string(2'000, 'e')},
      },
      8'000,
      1);
  auto options = makeOptions(blockCache_, /*maxCoalesceDistance=*/100);
  options.maxCacheEntrySize = 1'500;
  auto input = MetadataInput::create(file.get(), options);

  const std::vector<std::string> expected{
      std::string(10, 'a'),
      std::string(10, 'b'),
      std::string(1'000, 'c'),
      std::string(10, 'd'),
      std::string(2'000, 'e'),
  };
  const auto buffers = input->load(sections);
  EXPECT_EQ(contents(buffers), expected);
  const uint64_t compressedSize{sections[2].size()};
  const std::vector<Read> expectedReads{
      {{0, 10}, {50, 10}, {100, compressedSize}},
      {{5'000, 10}, {5'100, 2'000}},
  };
  EXPECT_EQ(file->scatterReads(), expectedReads);
  EXPECT_TRUE(file->reads().empty());

  // The compressed section is read into a buffer and decompressed into its
  // entry. The last section is above maxCacheEntrySize, so it is returned in
  // the buffer it was read into.
  const auto destinations = file->scatterDestinations();
  ASSERT_EQ(destinations.size(), 5);
  EXPECT_EQ(buffers[0]->content().data(), destinations[0]);
  EXPECT_EQ(buffers[1]->content().data(), destinations[1]);
  EXPECT_NE(buffers[2]->content().data(), destinations[2]);
  EXPECT_EQ(buffers[3]->content().data(), destinations[3]);
  EXPECT_EQ(buffers[4]->content().data(), destinations[4]);

  // The gaps of 40, 40, and 90 bytes are read and dropped.
  EXPECT_EQ(ioStats_->read().sum(), 2'200 + compressedSize);
  EXPECT_EQ(ioStats_->rawBytesRead(), 2'030 + compressedSize);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), 170);

  // The entries the reads filled are cached. Only the last section is read
  // again.
  const auto cachedBuffers = input->load(sections);
  EXPECT_EQ(contents(cachedBuffers), expected);
  ASSERT_EQ(file->scatterReads().size(), 3);
  const Read expectedRead{{5'100, 2'000}};
  EXPECT_EQ(file->scatterReads()[2], expectedRead);
  EXPECT_EQ(ioStats_->ramHit().count(), 4);
  EXPECT_EQ(cachedBuffers[0]->content().data(), destinations[0]);
  EXPECT_EQ(cachedBuffers[3]->content().data(), destinations[3]);
}

TEST_F(CachedMetadataInputTest, mergesReadsSharingABlock) {
  auto [file, sections] = makeFile(
      {
          {100, std::string(100, 'a')},
          {300, std::string(100, 'b')},
          {4'000, std::string(200, 'c')},
      },
      3 * kAlignment,
      kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

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

TEST_F(CachedMetadataInputTest, coalescesReads) {
  const std::vector<SectionSpec> specs{
      {0, std::string(10, 'a')},
      {50, std::string(10, 'b')},
      {500, std::string(10, 'c')},
  };
  {
    SCOPED_TRACE("Merges reads within the coalesce distance");
    auto [file, sections] = makeFile(specs, 1'000, 1);
    auto input = MetadataInput::create(
        file.get(), makeOptions(blockCache_, /*maxCoalesceDistance=*/100));
    input->load(sections);
    const std::vector<Read> expected{{{0, 10}, {50, 10}}, {{500, 10}}};
    EXPECT_EQ(file->scatterReads(), expected);
  }
  {
    SCOPED_TRACE("Stops merging at the coalesce byte limit");
    auto [file, sections] = makeFile(specs, 1'000, 1);

    // A file id per case, so no case hits another case's entries.
    const velox::FileHandle fileHandle{
        .uuid = {velox::fileIds(), "CachedMetadataInputTest.coalesceBytes"}};
    auto options = makeOptions(
        blockCache_, /*maxCoalesceDistance=*/100, /*maxCoalesceBytes=*/15);
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);
    input->load(sections);
    const std::vector<Read> expected{{{0, 10}}, {{50, 10}}, {{500, 10}}};
    EXPECT_EQ(file->scatterReads(), expected);
  }
  {
    SCOPED_TRACE("Merges a large gap within the coalesce distance");
    constexpr uint64_t kGap{1 << 20};
    auto [file, sections] = makeFile(
        {{0, std::string(10, 'a')}, {10 + kGap, std::string(10, 'b')}},
        20 + kGap,
        1);
    const velox::FileHandle fileHandle{
        .uuid = {velox::fileIds(), "CachedMetadataInputTest.coalesceDistance"}};
    auto options = makeOptions(
        blockCache_, /*maxCoalesceDistance=*/static_cast<int32_t>(kGap));
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);
    input->load(sections);
    const std::vector<Read> expected{{{0, 10}, {10 + kGap, 10}}};
    EXPECT_EQ(file->scatterReads(), expected);
  }
}

// Without direct I/O, Options::executor runs every scatter read of a load but
// the first, which load() runs itself.
TEST_F(CachedMetadataInputTest, executorRunsAllButFirstScatterRead) {
  const std::vector<SectionSpec> specs{{0, "first"}, {100, "second"}};
  const std::vector<std::string> expected{"first", "second"};
  {
    SCOPED_TRACE("The inline executor runs the queued read first");
    auto [file, sections] = makeFile(specs, 200, 1);
    auto options = makeOptions(blockCache_);
    options.executor = &folly::InlineExecutor::instance();
    auto input = MetadataInput::create(file.get(), options);

    EXPECT_EQ(contents(input->load(sections)), expected);
    const std::vector<Read> expectedReads{{{100, 6}}, {{0, 5}}};
    EXPECT_EQ(file->scatterReads(), expectedReads);

    // Each read records its own latency. The query thread records its wait for
    // all of them once.
    EXPECT_EQ(ioStats_->storageReadLatencyUs().count(), 2);
    EXPECT_EQ(ioStats_->queryThreadIoLatencyUs().count(), 1);
  }
  {
    SCOPED_TRACE("A queued read can start after the load returns");
    auto [file, sections] = makeFile(specs, 200, 1);

    // A file id of its own, so this case does not hit the entries of the first.
    const velox::FileHandle fileHandle{
        .uuid = {velox::fileIds(), "CachedMetadataInputTest.manualExecutor"}};
    folly::ManualExecutor executor;
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    options.executor = &executor;
    auto input = MetadataInput::create(file.get(), options);

    // The manual executor starts no task until drain(), so load() runs both
    // reads itself.
    EXPECT_EQ(contents(input->load(sections)), expected);
    const std::vector<Read> expectedReads{{{0, 5}}, {{100, 6}}};
    EXPECT_EQ(file->scatterReads(), expectedReads);

    // The queued task runs after the input is destroyed. ASAN reports any
    // access to the freed state.
    input.reset();
    EXPECT_EQ(executor.drain(), 1);
  }
}

TEST_F(CachedMetadataInputTest, loadsEmptySection) {
  auto [file, sections] = makeFile({{10, ""}}, 100, kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  EXPECT_EQ(contents(input->load(sections)), std::vector<std::string>{""});
  EXPECT_TRUE(file->reads().empty());
}

TEST_F(CachedMetadataInputTest, loadsEmptySectionSharingOffset) {
  // The writer gives an empty section the offset of the section after it.
  for (const auto alignment : {uint64_t{1}, kAlignment}) {
    SCOPED_TRACE(fmt::format("alignment {}", alignment));
    auto [file, sections] = makeFile({{10, ""}, {10, "data"}}, 100, alignment);
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
    EXPECT_EQ(
        contents(input->load(sections)),
        (std::vector<std::string>{"", "data"}));
  }
}

TEST_F(CachedMetadataInputTest, compressedSectionTooSmallThrows) {
  auto [file, sections] = makeFile({}, 100, 1);

  // Neither holds the 4-byte uncompressed size. The second ends the file, so
  // reading the size would run past the read buffer.
  const std::array tooSmall{
      MetadataSection{10, 0, CompressionType::Zstd, 100},
      MetadataSection{98, 2, CompressionType::Zstd, 100},
  };
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  for (const auto& section : tooSmall) {
    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}), "Compressed metadata section is too small");
  }
}

TEST_F(CachedMetadataInputTest, sectionPastEndOfFileThrows) {
  auto [file, sections] = makeFile({}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  {
    SCOPED_TRACE("Offset past the end, with offset + size wrapping to 16");
    const MetadataSection section{
        std::numeric_limits<uint64_t>::max() - 15,
        32,
        CompressionType::Uncompressed,
        32};
    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}),
        "Read region starts past the end of the file");
  }
  {
    SCOPED_TRACE("Size past the end");
    const MetadataSection section{90, 32, CompressionType::Uncompressed, 32};
    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}),
        "Read region extends past the end of the file");
  }
}

TEST_F(CachedMetadataInputTest, overlappingSectionsRecordBytesRead) {
  {
    SCOPED_TRACE("The second section starts a scatter read");
    const auto bytes = std::string(100, 'a') + std::string(100, 'b');
    auto [file, sections] = makeFile({{0, bytes}}, 300, 1);
    const std::array overlapping{
        MetadataSection{0, 200, CompressionType::Uncompressed, 200},
        MetadataSection{100, 100, CompressionType::Uncompressed, 100},
    };
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
    EXPECT_EQ(
        contents(input->load(overlapping)),
        (std::vector<std::string>{bytes, std::string(100, 'b')}));
    const std::vector<Read> expectedReads{{{0, 200}}, {{100, 100}}};
    EXPECT_EQ(file->scatterReads(), expectedReads);
    EXPECT_EQ(ioStats_->read().sum(), 300);
    EXPECT_EQ(ioStats_->rawBytesRead(), 300);
    EXPECT_EQ(ioStats_->rawOverreadBytes(), 0);
  }
  {
    SCOPED_TRACE("One direct-I/O read covers both sections");
    ioStats_ = std::make_shared<velox::io::IoStatistics>();
    const std::string bytes(kAlignment, 'a');
    auto [file, sections] = makeFile({{0, bytes}}, kAlignment, kAlignment);
    const std::array overlapping{
        MetadataSection{
            0, kAlignment, CompressionType::Uncompressed, kAlignment},
        MetadataSection{
            100,
            kAlignment - 100,
            CompressionType::Uncompressed,
            kAlignment - 100},
    };
    const velox::FileHandle fileHandle{
        .uuid = {velox::fileIds(), "CachedMetadataInputTest.overlapping"}};
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);
    EXPECT_EQ(
        contents(input->load(overlapping)),
        (std::vector<std::string>{bytes, std::string(kAlignment - 100, 'a')}));
    EXPECT_EQ(file->reads().size(), 1);
    EXPECT_EQ(ioStats_->read().sum(), kAlignment);
    EXPECT_EQ(ioStats_->rawBytesRead(), kAlignment);
    EXPECT_EQ(ioStats_->rawOverreadBytes(), 0);
  }
}

TEST_F(CachedMetadataInputTest, recordsReadStats) {
  auto [file, sections] =
      makeFile({{100, std::string(200, 'a')}}, kAlignment, kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  input->load(sections);

  EXPECT_EQ(ioStats_->read().sum(), kAlignment);
  EXPECT_EQ(ioStats_->rawBytesRead(), 200);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), kAlignment - 200);
  EXPECT_EQ(ioStats_->storageReadLatencyUs().count(), 1);
  EXPECT_EQ(ioStats_->queryThreadIoLatencyUs().count(), 1);
}

// A direct-I/O read that runs past the end of the file reads only the bytes
// before the end, so only those count as read.
TEST_F(CachedMetadataInputTest, readPastEndOfFileRecordsBytesBeforeEnd) {
  auto [file, sections] = makeFile(
      {{kAlignment + 100, std::string(200, 'a')}},
      kAlignment + 300,
      kAlignment);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  input->load(sections);

  const std::vector<Read> expectedReads{{{kAlignment, kAlignment}}};
  EXPECT_EQ(file->reads(), expectedReads);
  EXPECT_EQ(ioStats_->read().sum(), 300);
  EXPECT_EQ(ioStats_->rawBytesRead(), 200);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), 100);
}

// Every read of the file passes Options::fileIoContext, so its file options
// reach the storage layer.
TEST_F(CachedMetadataInputTest, readsPassFileIoContext) {
  const folly::F14FastMap<std::string, std::string> fileOpts{{"key", "value"}};
  auto file = std::make_shared<nimble::testing::FileOptsRecordingReadFile>(
      std::string(1'000, 'a'));
  const std::vector<MetadataSection> sections{
      {100, 200, CompressionType::Uncompressed}};
  auto options = makeOptions(blockCache_);
  options.fileIoContext.fileOpts = fileOpts;
  auto input = MetadataInput::create(file.get(), options);

  EXPECT_EQ(
      contents(input->load(sections)),
      std::vector<std::string>{std::string(200, 'a')});
  const std::vector<folly::F14FastMap<std::string, std::string>> expected{
      fileOpts,
  };
  EXPECT_EQ(file->fileOpts(), expected);
}

TEST_F(CachedMetadataInputTest, duplicateSectionsThrow) {
  auto [file, sections] = makeFile({{0, "duplicate"}}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  NIMBLE_ASSERT_THROW(
      input->load(std::array{sections[0], sections[0]}),
      "Metadata input must not contain duplicate regions");
}

TEST_F(CachedMetadataInputTest, cacheHitSkipsRead) {
  auto [file, sections] = makeFile(
      {
          {0, "uncompressed"},
          {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
      },
      2 * kAlignment,
      kAlignment);
  const std::vector<std::string> expected{
      "uncompressed", std::string(1'000, 'Z')};

  // Raw bytes read count each section at its size in the file, from the file
  // and from the cache alike.
  const uint64_t sectionBytes = sections[0].size() + sections[1].size();
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(file->reads().size(), 1);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);
  EXPECT_EQ(ioStats_->rawBytesRead(), sectionBytes);

  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(file->reads().size(), 1);
  EXPECT_EQ(ioStats_->ramHit().count(), 2);
  EXPECT_EQ(ioStats_->rawBytesRead(), 2 * sectionBytes);
}

TEST_F(CachedMetadataInputTest, cachedEntryOfOtherSizeThrows) {
  auto [file, sections] = makeFile({{0, "metadata"}}, 100, 1);

  // A larger entry under the same key, as a reused file id would leave. The
  // AsyncDataCache-backed cache returns it as a hit.
  {
    velox::io::IoStatistics ioStats;
    auto entry =
        blockCache_->getOrCreate(fileHandle_.uuid.id(), 0, 20, ioStats);
    ASSERT_FALSE(entry.sealed());
    std::memset(entry.data(), 'x', 20);
    blockCache_->seal(entry);
  }
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  NIMBLE_ASSERT_THROW(
      input->load(sections), "Cached entry size mismatch with expected size");
}

TEST_F(CachedMetadataInputTest, sectionWithoutUncompressedSizeCachedAfterRead) {
  auto [file, sections] = makeFile(
      {{5'000, std::string(1'000, 'Z'), CompressionType::Zstd, false}},
      2 * kAlignment,
      kAlignment);
  const std::vector<std::string> expected{std::string(1'000, 'Z')};
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);

  EXPECT_EQ(contents(input->load(sections)), expected);
  EXPECT_EQ(file->reads().size(), 1);
  EXPECT_EQ(ioStats_->ramHit().count(), 1);
  EXPECT_EQ(ioStats_->rawBytesRead(), 2 * sections[0].size());
}

TEST_F(
    CachedMetadataInputTest,
    sectionWithoutUncompressedSizeFilledDuringRead) {
  const std::string data(1'000, 'Z');
  {
    SCOPED_TRACE("Entry of the same size");
    auto [file, sections] = makeFile(
        {{5'000, data, CompressionType::Zstd, false}},
        2 * kAlignment,
        kAlignment);

    // Another reader fills the entry after this load probes for it. Its bytes
    // differ from the file's, so the result shows which bytes the load kept.
    const std::string otherBytes(1'000, 'Y');
    file->setBeforeRead([&]() {
      velox::io::IoStatistics ioStats;
      auto entry = blockCache_->getOrCreate(
          fileHandle_.uuid.id(), 5'000, 1'000, ioStats);
      std::memcpy(entry.data(), otherBytes.data(), otherBytes.size());
      blockCache_->seal(entry);
    });
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

    // The load returns the entry the other reader sealed rather than
    // refilling it.
    EXPECT_EQ(
        contents(input->load(sections)), std::vector<std::string>{otherBytes});

    // This load read the bytes itself, so the entry it found is not a hit.
    EXPECT_EQ(ioStats_->ramHit().count(), 0);
  }
  {
    SCOPED_TRACE("Larger entry");
    const velox::FileHandle fileHandle{
        .uuid = {
            velox::fileIds(), "CachedMetadataInputTest.largerEntryDuringRead"}};
    auto [file, sections] = makeFile(
        {{5'000, data, CompressionType::Zstd, false}},
        2 * kAlignment,
        kAlignment);
    file->setBeforeRead([&]() {
      velox::io::IoStatistics ioStats;
      auto entry =
          blockCache_->getOrCreate(fileHandle.uuid.id(), 5'000, 2'000, ioStats);
      std::memset(entry.data(), 'x', 2'000);
      blockCache_->seal(entry);
    });
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);

    NIMBLE_ASSERT_THROW(
        input->load(sections), "Cached entry size mismatch with expected size");
  }
}

TEST_F(
    CachedMetadataInputTest,
    sectionWithoutUncompressedSizeRespectsMaxCacheEntrySize) {
  const std::vector<std::string> expected{std::string(1'000, 'Z')};
  for (const uint32_t maxCacheEntrySize : {999U, 1'000U}) {
    SCOPED_TRACE(fmt::format("maxCacheEntrySize={}", maxCacheEntrySize));

    // A file id per case, so no case hits another case's entry.
    const velox::FileHandle fileHandle{
        .uuid = {
            velox::fileIds(),
            fmt::format(
                "CachedMetadataInputTest.maxCacheEntrySize.{}",
                maxCacheEntrySize)}};
    auto [file, sections] = makeFile(
        {{5'000, expected[0], CompressionType::Zstd, false}},
        2 * kAlignment,
        kAlignment);
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    options.maxCacheEntrySize = maxCacheEntrySize;
    auto input = MetadataInput::create(file.get(), options);
    input->load(sections);

    EXPECT_EQ(contents(input->load(sections)), expected);
    EXPECT_EQ(file->reads().size(), maxCacheEntrySize < 1'000 ? 2 : 1);
  }
}

TEST_F(CachedMetadataInputTest, oversizedSectionThrows) {
  {
    SCOPED_TRACE("Uncompressed size in the footer");
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

    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}), "Metadata section is too large");
    EXPECT_TRUE(file->reads().empty());
  }
  {
    SCOPED_TRACE("Uncompressed size in the section");

    // The 4-byte uncompressed size the section starts with is 1 << 31.
    const auto corrupt = std::string("\0\0\0\x80", 4) + "not a zstd frame";
    const auto file = makeFile({{0, corrupt}}, 100, 1).first;
    const MetadataSection section{
        0, static_cast<uint32_t>(corrupt.size()), CompressionType::Zstd};
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

    NIMBLE_ASSERT_FILE_THROW(
        input->load({&section, 1}), "Metadata section is too large");
  }
}

TEST_F(CachedMetadataInputTest, entryAboveMaxCacheEntrySizeBypassesCache) {
  auto options = makeOptions(blockCache_);
  options.maxCacheEntrySize = 100;
  {
    SCOPED_TRACE("load");
    auto [file, sections] = makeFile(
        {
            {0, std::string(100, 'a')},
            {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
        },
        2 * kAlignment,
        kAlignment);
    auto input = MetadataInput::create(file.get(), options);
    input->load(sections);

    // Only the compressed section, 1'000 bytes decompressed, is read again.
    EXPECT_EQ(
        contents(input->load(sections)),
        (std::vector<std::string>{
            std::string(100, 'a'), std::string(1'000, 'Z')}));
    ASSERT_EQ(file->reads().size(), 2);
    const Read expected{{kAlignment, kAlignment}};
    EXPECT_EQ(file->reads()[1], expected);
  }
  {
    SCOPED_TRACE("cacheMetadata");
    auto [file, sections] = makeFile({}, 1'000, 1);
    auto input = MetadataInput::create(file.get(), options);
    const std::string atLimit(100, 'a');
    const std::string aboveLimit(101, 'b');
    input->cacheMetadata(200, std::array<std::string_view, 1>{atLimit});
    input->cacheMetadata(500, std::array<std::string_view, 1>{aboveLimit});
    EXPECT_NE(input->findCachedMetadata(200), nullptr);
    EXPECT_EQ(input->findCachedMetadata(500), nullptr);
  }
}

// A lookup that does not know the entry size returns an entry above
// maxCacheEntrySize that an input without the limit cached.
TEST_F(
    CachedMetadataInputTest,
    unsizedLookupReturnsEntryAboveMaxCacheEntrySize) {
  const uint64_t fileSize = 2 * kAlignment;
  auto [file, sections] = makeFile(
      {{5'000, std::string(1'000, 'Z'), CompressionType::Zstd, false}},
      fileSize,
      kAlignment);
  const std::string footer(200, 'f');
  {
    auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
    input->load(sections);
    input->cacheMetadata(fileSize, std::array<std::string_view, 1>{footer});
  }

  auto options = makeOptions(blockCache_);
  options.maxCacheEntrySize = 100;
  auto input = MetadataInput::create(file.get(), options);
  EXPECT_EQ(
      contents(input->load(sections)),
      std::vector<std::string>{std::string(1'000, 'Z')});
  EXPECT_EQ(file->reads().size(), 1);
  const auto cachedFooter = input->findCachedMetadata(fileSize);
  ASSERT_NE(cachedFooter, nullptr);
  EXPECT_EQ(cachedFooter->content(), footer);
}

TEST_F(CachedMetadataInputTest, claimsCacheEntriesInFileOffsetOrder) {
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

// Two overlapping loads of the same sections, which list them in opposite
// orders, read the file once between them: the load that claims the entries
// first reads them, and the other waits for its fills. Both claim entries in
// file offset order, so neither holds an entry the other needs while it waits.
TEST_F(CachedMetadataInputTest, concurrentLoadsReadOnce) {
  const std::vector<SectionSpec> specs{
      {0, "first"},
      {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
      {9'000, "third"},
  };
  const std::vector<std::string> expected{
      "first", std::string(1'000, 'Z'), "third"};
  for (const auto alignment : {uint64_t{1}, kAlignment}) {
    SCOPED_TRACE(fmt::format("alignment {}", alignment));
    auto [file, sections] = makeFile(specs, 3 * kAlignment, alignment);

    // A file id per alignment, so no case hits another case's entries.
    const velox::FileHandle fileHandle{
        .uuid = {
            velox::fileIds(),
            fmt::format(
                "CachedMetadataInputTest.concurrentLoads.{}", alignment)}};
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;

    // The first read waits until both loads have started, then gives the other
    // load time to wait on the entries that this one holds.
    std::atomic_int numStarted{0};
    std::atomic_bool firstRead{true};
    file->setBeforeRead([&]() {
      if (firstRead.exchange(false)) {
        while (numStarted < 2) {
          std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });

    std::vector<std::string> forward;
    std::vector<std::string> reverse;
    std::thread forwardLoad([&]() {
      auto input = MetadataInput::create(file.get(), options);
      ++numStarted;
      forward = contents(input->load(sections));
    });
    std::thread reverseLoad([&]() {
      auto input = MetadataInput::create(file.get(), options);
      ++numStarted;
      reverse = contents(
          input->load(std::array{sections[2], sections[1], sections[0]}));
    });
    forwardLoad.join();
    reverseLoad.join();

    EXPECT_EQ(forward, expected);
    EXPECT_EQ(
        reverse,
        (std::vector<std::string>{expected.rbegin(), expected.rend()}));
    if (alignment == 1) {
      // A scatter read per section, since no gap is within the coalesce
      // distance of 0.
      EXPECT_EQ(file->scatterReads().size(), 3);
    } else {
      EXPECT_EQ(file->reads().size(), 1);
    }
  }
}

TEST_F(CachedMetadataInputTest, failedReadReleasesCacheEntries) {
  for (const auto alignment : {uint64_t{1}, kAlignment}) {
    SCOPED_TRACE(fmt::format("alignment {}", alignment));
    auto [file, sections] = makeFile({{0, "metadata"}}, kAlignment, alignment);

    // A file id per alignment, so no case hits another case's entries.
    const velox::FileHandle fileHandle{
        .uuid = {
            velox::fileIds(),
            fmt::format("CachedMetadataInputTest.failedRead.{}", alignment)}};
    auto options = makeOptions(blockCache_);
    options.fileHandle = &fileHandle;
    auto input = MetadataInput::create(file.get(), options);

    file->setFailReads(true);
    NIMBLE_ASSERT_THROW(input->load(sections), "Injected preadv failure");

    // A leaked unfilled entry would block this load forever.
    file->setFailReads(false);
    EXPECT_EQ(
        contents(input->load(sections)), std::vector<std::string>{"metadata"});
  }
}

TEST_F(CachedMetadataInputTest, failedDecompressionReleasesCacheEntry) {
  // A 4-byte uncompressed size of 64, followed by bytes that are not a zstd
  // frame.
  const auto corrupt = std::string("\x40\0\0\0", 4) + "not a zstd frame";
  auto [file, sections] = makeFile({{0, "metadata"}, {100, corrupt}}, 200, 1);
  const std::array loaded{
      sections[0],
      MetadataSection{
          100,
          static_cast<uint32_t>(corrupt.size()),
          CompressionType::Zstd,
          64},
  };
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));

  NIMBLE_ASSERT_THROW(input->load(loaded), "Error uncompressing data");

  // The load cached the first section before the second failed.
  const auto ramHits = ioStats_->ramHit().count();
  EXPECT_EQ(
      contents(input->load({&loaded[0], 1})),
      std::vector<std::string>{"metadata"});
  EXPECT_EQ(ioStats_->ramHit().count(), ramHits + 1);

  // A leaked unfilled entry would block this load forever, and a sealed one
  // would return bytes that failed to decompress.
  NIMBLE_ASSERT_THROW(input->load({&loaded[1], 1}), "Error uncompressing data");
}

TEST_F(CachedMetadataInputTest, cacheAndFindMetadata) {
  auto [file, sections] = makeFile({}, 100, 1);
  auto input = MetadataInput::create(file.get(), makeOptions(blockCache_));
  EXPECT_EQ(input->findCachedMetadata(100), nullptr);

  // Empty metadata is not cached.
  input->cacheMetadata(100, {});
  EXPECT_EQ(input->findCachedMetadata(100), nullptr);

  const std::array<std::string_view, 2> ranges{"foot", "er"};
  input->cacheMetadata(100, ranges);
  EXPECT_EQ(ioStats_->rawBytesRead(), 0);

  auto found = input->findCachedMetadata(100);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->content(), "footer");
  EXPECT_EQ(ioStats_->rawBytesRead(), found->content().size());
  EXPECT_EQ(ioStats_->ramHit().count(), 1);

  // Caching bytes that are already cached is not a RAM hit.
  const auto ramHits = ioStats_->ramHit().count();
  input->cacheMetadata(100, ranges);
  EXPECT_EQ(ioStats_->ramHit().count(), ramHits);

  const velox::FileHandle otherFileHandle{
      .uuid = {velox::fileIds(), "CachedMetadataInputTest.otherFile"}};
  auto options = makeOptions(blockCache_);
  options.fileHandle = &otherFileHandle;
  auto otherFileInput = MetadataInput::create(file.get(), options);
  EXPECT_EQ(otherFileInput->findCachedMetadata(100), nullptr);
}

TEST_F(CachedMetadataInputTest, loadsAndFindsFromSsd) {
  velox::filesystems::registerLocalFileSystem();
  FLAGS_velox_ssd_odirect = false;
  const auto ssdDirectory =
      velox::common::testutil::TempDirectoryPath::create();
  folly::IOThreadPoolExecutor ssdExecutor{1};
  const auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});
  const auto ssdAsyncDataCache = velox::cache::AsyncDataCache::create(
      allocator.get(),
      std::make_unique<velox::cache::SsdCache>(velox::cache::SsdCache::Config(
          fmt::format("{}/cache", ssdDirectory->getPath()),
          /*_maxBytes=*/64UL << 20,
          /*_numShards=*/1,
          &ssdExecutor)));
  testingResetBlockCache();
  auto* ssdBlockCache = getBlockCache(ssdAsyncDataCache.get());

  const uint64_t fileSize = 3 * kAlignment;
  auto [file, sections] = makeFile(
      {
          {0, "uncompressed"},
          {5'000, std::string(1'000, 'Z'), CompressionType::Zstd},
          {9'000, std::string(1'000, 'O'), CompressionType::Zstd, false},
      },
      fileSize,
      kAlignment);
  const std::vector<std::string> expected{
      "uncompressed", std::string(1'000, 'Z'), std::string(1'000, 'O')};
  {
    auto input = MetadataInput::create(file.get(), makeOptions(ssdBlockCache));
    input->load(sections);
    const std::array<std::string_view, 2> ranges{"foot", "er"};
    input->cacheMetadata(fileSize, ranges);
  }
  ASSERT_TRUE(ssdAsyncDataCache->ssdCache()->startWrite());
  ssdAsyncDataCache->saveToSsd(/*saveAll=*/true);
  ssdAsyncDataCache->ssdCache()->waitForWriteToFinish();
  ssdAsyncDataCache->clear();

  // Every read of the file fails, so the metadata must come from SSD.
  file->setFailReads(true);
  {
    auto input = MetadataInput::create(file.get(), makeOptions(ssdBlockCache));
    EXPECT_EQ(contents(input->load(sections)), expected);
    const auto footer = input->findCachedMetadata(fileSize);
    ASSERT_NE(footer, nullptr);
    EXPECT_EQ(footer->content(), "footer");
  }
  EXPECT_EQ(ioStats_->ssdRead().count(), 4);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);

  // Caching metadata that only SSD holds loads the entry from SSD, and records
  // that read.
  ssdAsyncDataCache->clear();
  {
    auto input = MetadataInput::create(file.get(), makeOptions(ssdBlockCache));
    const std::array<std::string_view, 2> ranges{"foot", "er"};
    input->cacheMetadata(fileSize, ranges);
  }
  EXPECT_EQ(ioStats_->ssdRead().count(), 5);
  EXPECT_EQ(ioStats_->ssdCacheReadLatencyUs().count(), 5);
  EXPECT_EQ(ioStats_->ramHit().count(), 0);

  testingResetBlockCache();
  ssdAsyncDataCache->shutdown();
}

TEST_F(CachedMetadataInputTest, findLeavesNoEntry) {
  velox::io::IoStatistics ioStats;
  EXPECT_TRUE(blockCache_->find(fileHandle_.uuid.id(), 0, ioStats).empty());

  // find() created nothing, so this lookup creates the entry.
  auto created = blockCache_->getOrCreate(fileHandle_.uuid.id(), 0, 3, ioStats);
  EXPECT_FALSE(created.sealed());
  EXPECT_EQ(created.size(), 3);
  std::memcpy(created.data(), "abc", 3);
  blockCache_->seal(created);

  auto found = blockCache_->find(fileHandle_.uuid.id(), 0, ioStats);
  ASSERT_FALSE(found.empty());
  EXPECT_TRUE(found.sealed());
  EXPECT_EQ(std::string_view(found.data(), found.size()), "abc");
}

TEST_F(CachedMetadataInputTest, blockCachePerAsyncDataCache) {
  EXPECT_EQ(getBlockCache(asyncDataCache_.get()), blockCache_);

  auto otherAllocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{.capacity = 1UL << 30});
  auto otherAsyncDataCache =
      velox::cache::AsyncDataCache::create(otherAllocator.get());
  auto* otherBlockCache = getBlockCache(otherAsyncDataCache.get());
  EXPECT_NE(otherBlockCache, blockCache_);
  {
    velox::io::IoStatistics ioStats;
    auto entry =
        otherBlockCache->getOrCreate(fileHandle_.uuid.id(), 0, 3, ioStats);
    std::memcpy(entry.data(), "abc", 3);
    otherBlockCache->seal(entry);
  }

  // Each BlockCache holds its entries in its own AsyncDataCache.
  velox::io::IoStatistics ioStats;
  EXPECT_TRUE(blockCache_->find(fileHandle_.uuid.id(), 0, ioStats).empty());
  EXPECT_FALSE(
      otherBlockCache->find(fileHandle_.uuid.id(), 0, ioStats).empty());
  EXPECT_EQ(otherAsyncDataCache->refreshStats().numEntries, 1);
  otherAsyncDataCache->shutdown();
}

TEST_F(CachedMetadataInputTest, findWaitsForFill) {
  velox::io::IoStatistics ioStats;
  auto created = blockCache_->getOrCreate(fileHandle_.uuid.id(), 0, 3, ioStats);
  ASSERT_FALSE(created.sealed());

  const auto numWaitsBefore = asyncDataCache_->refreshStats().numWaitExclusive;
  std::string found;
  std::thread finder([&]() {
    velox::io::IoStatistics finderIoStats;
    auto entry = blockCache_->find(fileHandle_.uuid.id(), 0, finderIoStats);
    if (!entry.empty()) {
      found.assign(entry.data(), entry.size());
    }
  });

  // Fills the entry once find() waits on it.
  while (asyncDataCache_->refreshStats().numWaitExclusive == numWaitsBefore) {
    std::this_thread::yield();
  }
  std::memcpy(created.data(), "abc", 3);
  blockCache_->seal(created);
  finder.join();

  EXPECT_EQ(found, "abc");
}

} // namespace
} // namespace facebook::nimble
