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
#include "velox/common/memory/ByteStream.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileStream.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

#include <gflags/gflags.h>

using namespace facebook::velox;
using namespace facebook::velox::memory;

class FileStreamTest : public testing::TestWithParam<bool> {
 protected:
  static void SetUpTestCase() {
    filesystems::registerLocalFileSystem();
  }

  void SetUp() override {
    constexpr uint64_t kMaxMappedMemory = 64 << 20;
    MemoryManagerOptions options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = kMaxMappedMemory;
    options.arbitratorCapacity = kMaxMappedMemory;
    memoryManager_ = std::make_unique<MemoryManager>(options);
    mmapAllocator_ = static_cast<MmapAllocator*>(memoryManager_->allocator());
    pool_ = memoryManager_->addLeafPool("ByteStreamTest");
    rng_.seed(124);
    tempDirPath_ = exec::test::TempDirectoryPath::create();
    fs_ = filesystems::getFileSystem(tempDirPath_->getPath(), nullptr);
  }

  void TearDown() override {}

  std::unique_ptr<common::FileReadStream> createStream(
      uint64_t streamSize,
      uint32_t bufferSize = 1024) {
    const auto filePath =
        fmt::format("{}/{}", tempDirPath_->getPath(), fileId_++);
    auto writeFile = fs_->openFileForWrite(filePath);
    std::uint8_t buffer[streamSize];
    for (int i = 0; i < streamSize; ++i) {
      buffer[i] = i % 256;
    }
    writeFile->append(
        std::string_view(reinterpret_cast<char*>(buffer), streamSize));
    writeFile->close();
    if (useFileInputStream_) {
      return std::make_unique<common::FileInputStream>(
          fs_->openFileForRead(filePath), bufferSize, pool_.get());
    } else {
      return std::make_unique<common::FileReadStream>(
          fs_->openFileForRead(filePath), pool_.get());
    }
  }

  const bool useFileInputStream_{GetParam()};

  folly::Random::DefaultGenerator rng_;
  std::unique_ptr<MemoryManager> memoryManager_;
  MmapAllocator* mmapAllocator_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::atomic_uint64_t fileId_{0};
  std::shared_ptr<exec::test::TempDirectoryPath> tempDirPath_;
  std::shared_ptr<filesystems::FileSystem> fs_;
};

TEST_P(FileStreamTest, stats) {
  struct {
    size_t streamSize;
    size_t bufferSize;

    std::string debugString() const {
      return fmt::format(
          "streamSize {}, bufferSize {}", streamSize, bufferSize);
    }
  } testSettings[] = {
      {4096, 1024}, {4096, 4096}, {4096, 8192}, {4096, 4096 + 1024}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    auto byteStream = createStream(testData.streamSize, testData.bufferSize);
    if (useFileInputStream_) {
      ASSERT_EQ(byteStream->stats().numReads, 1);
      ASSERT_EQ(
          byteStream->stats().readBytes,
          std::min(testData.streamSize, testData.bufferSize));
      ASSERT_GT(byteStream->stats().readTimeNs, 0);
    } else {
      ASSERT_EQ(byteStream->stats().numReads, 0);
      ASSERT_EQ(byteStream->stats().readBytes, 0);
      ASSERT_EQ(byteStream->stats().readTimeNs, 0);
    }
    uint8_t buffer[testData.streamSize / 8];
    for (int offset = 0; offset < testData.streamSize;) {
      byteStream->readBytes(buffer, testData.streamSize / 8);
      for (int i = 0; i < testData.streamSize / 8; ++i, ++offset) {
        if (useFileInputStream_) {
          ASSERT_EQ(buffer[i], offset % 256);
        }
      }
    }
    ASSERT_TRUE(byteStream->atEnd());
    if (useFileInputStream_) {
      ASSERT_EQ(
          byteStream->stats().numReads,
          bits::roundUp(testData.streamSize, testData.bufferSize) /
              testData.bufferSize);
    } else {
      ASSERT_EQ(byteStream->stats().numReads, 8);
    }
    ASSERT_EQ(byteStream->stats().readBytes, testData.streamSize);
    ASSERT_GT(byteStream->stats().readTimeNs, 0);
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    FileStreamTest,
    FileStreamTest,
    testing::ValuesIn({false, true}));
