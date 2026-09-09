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
#include <fstream>
#include <limits>

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

using namespace facebook;

class FileLayoutTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  FileLayoutTest()
      : rootPool_(
            velox::memory::memoryManager()->addRootPool("FileLayoutTest")),
        pool_(rootPool_->addLeafChild("FileLayoutTestLeaf")) {}

  const std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  const std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(FileLayoutTest, nonEmptyFile) {
  std::string file;
  velox::InMemoryWriteFile writeFile(&file);
  nimble::Buffer buffer(*pool_);

  auto tabletWriter = nimble::TabletWriter::create(
      &writeFile,
      *pool_,
      {.metadataCompressionThreshold = std::numeric_limits<uint32_t>::max()});

  // Write two stripes
  const uint32_t streamCount = 3;
  for (int stripe = 0; stripe < 2; ++stripe) {
    std::vector<nimble::Stream> streams;
    for (uint32_t i = 0; i < streamCount; ++i) {
      const auto size = 100 + stripe * 10 + i;
      auto pos = buffer.reserve(size);
      std::memset(pos, 'a' + i, size);
      streams.push_back({
          .offset = i,
          .chunks = {{.content = {std::string_view(pos, size)}}},
      });
    }
    tabletWriter->writeStripe(1000 + stripe, std::move(streams));
  }

  tabletWriter->close();
  writeFile.close();

  // Write to a temporary file for testing path-based API
  velox::filesystems::registerLocalFileSystem();
  auto tmpDir = velox::common::testutil::TempDirectoryPath::create();
  auto tmpPath = tmpDir->getPath() + "/test_file_layout.nimble";
  {
    std::ofstream ofs(tmpPath, std::ios::binary);
    ofs.write(file.data(), file.size());
  }

  // Test both FileLayout::create() APIs
  for (int apiIndex = 0; apiIndex < 2; ++apiIndex) {
    SCOPED_TRACE(apiIndex == 0 ? "ReadFile API" : "Path API");

    nimble::FileLayout layout;
    if (apiIndex == 0) {
      auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
      layout = nimble::FileLayout::create(readFile, pool_.get());
    } else {
      layout = nimble::FileLayout::create(tmpPath, pool_.get());
    }

    // Verify layout fields
    EXPECT_EQ(layout.fileSize, file.size());
    EXPECT_EQ(layout.postscript.majorVersion(), nimble::kVersionMajor);
    EXPECT_EQ(layout.postscript.minorVersion(), nimble::kVersionMinor);
    EXPECT_EQ(layout.postscript.checksumType(), nimble::ChecksumType::XXH3_64);
    EXPECT_GT(layout.footer.size(), 0);
    EXPECT_LT(layout.footer.offset(), layout.fileSize);
    EXPECT_EQ(layout.stripeGroups.size(), 1);
    EXPECT_TRUE(layout.indexPartitions.empty()); // No index configured

    // Verify stripe group metadata
    const auto& stripeGroup = layout.stripeGroups[0];
    EXPECT_GT(stripeGroup.size(), 0);
    EXPECT_LT(stripeGroup.offset(), layout.footer.offset());

    // Verify per-stripe info
    EXPECT_EQ(layout.stripesInfo.size(), 2);
    for (size_t i = 0; i < layout.stripesInfo.size(); ++i) {
      EXPECT_EQ(layout.stripesInfo[i].stripeGroupIndex, 0);
      EXPECT_GT(layout.stripesInfo[i].size, 0);
    }
  }
  // TempDirectoryPath automatically cleans up on destruction
}

TEST_F(FileLayoutTest, emptyFile) {
  // Test FileLayout::create() with empty file (no stripes).
  std::string file;
  velox::InMemoryWriteFile writeFile(&file);

  auto tabletWriter = nimble::TabletWriter::create(&writeFile, *pool_, {});
  tabletWriter->close();
  writeFile.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto layout = nimble::FileLayout::create(readFile, pool_.get());

  EXPECT_EQ(layout.fileSize, file.size());
  EXPECT_EQ(layout.postscript.majorVersion(), nimble::kVersionMajor);
  EXPECT_EQ(layout.postscript.minorVersion(), nimble::kVersionMinor);
  EXPECT_TRUE(layout.stripeGroups.empty());
  EXPECT_TRUE(layout.indexPartitions.empty());
  EXPECT_TRUE(layout.stripesInfo.empty());
}

TEST_F(FileLayoutTest, emptyFileWithChunkIndex) {
  // Test FileLayout::create() with empty file that has chunk index enabled.
  std::string file;
  velox::InMemoryWriteFile writeFile(&file);

  auto tabletWriter = nimble::TabletWriter::create(
      &writeFile,
      *pool_,
      {
          .enableChunkIndex = true,
      });
  tabletWriter->close();
  writeFile.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto layout = nimble::FileLayout::create(readFile, pool_.get());

  EXPECT_EQ(layout.fileSize, file.size());
  EXPECT_TRUE(layout.stripeGroups.empty());
  // Empty file with chunk index still has no index groups (no stripes to index)
  EXPECT_TRUE(layout.indexPartitions.empty());
  EXPECT_TRUE(layout.stripesInfo.empty());
}
