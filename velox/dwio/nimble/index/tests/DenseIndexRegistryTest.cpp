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
#include <deque>

#include "velox/dwio/nimble/index/DenseIndexRegistry.h"
#include "velox/dwio/nimble/index/HashIndex.h"
#include "velox/dwio/nimble/index/HashIndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/SortedIndex.h"
#include "velox/dwio/nimble/index/SortedIndexConfig.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index {
namespace {

class DenseIndexRegistryTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ =
        velox::memory::memoryManager()->addRootPool("DenseIndexRegistryTest");
    leafPool_ = pool_->addLeafChild("leaf");
    tempDir_ = velox::common::testutil::TempDirectoryPath::create();
  }

  static velox::RowTypePtr rowType() {
    return velox::ROW({{"id", velox::INTEGER()}, {"value", velox::VARCHAR()}});
  }

  velox::VectorPtr makeBatch(int32_t startRow, int32_t numRows) {
    auto ids =
        velox::BaseVector::create(velox::INTEGER(), numRows, leafPool_.get());
    auto values =
        velox::BaseVector::create(velox::VARCHAR(), numRows, leafPool_.get());
    auto* flatIds = ids->asFlatVector<int32_t>();
    auto* flatValues = values->asFlatVector<velox::StringView>();
    for (int i = 0; i < numRows; ++i) {
      const int32_t id = startRow + i;
      flatIds->set(i, id);
      valueStrings_.emplace_back("v_" + std::to_string(id));
      flatValues->set(i, velox::StringView(valueStrings_.back()));
    }
    return std::make_shared<velox::RowVector>(
        leafPool_.get(),
        rowType(),
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, values});
  }

  void writeFile(
      const std::string& filePath,
      const velox::RowTypePtr& schema,
      const std::vector<velox::VectorPtr>& batches,
      WriterOptions options) {
    auto fs = velox::filesystems::getFileSystem(filePath, {});
    auto file = fs->openFileForWrite(
        filePath,
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});

    Writer writer(schema, std::move(file), *pool_, std::move(options));
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  std::shared_ptr<TabletReader> openTablet(const std::string& filePath) {
    auto fs = velox::filesystems::getFileSystem(filePath, {});
    auto readFile = fs->openFileForRead(filePath);
    ioStats_ = std::make_shared<velox::io::IoStatistics>();
    readerOptions_ =
        std::make_unique<velox::io::ReaderOptions>(leafPool_.get());
    readerOptions_->setMetadataIoStats(ioStats_);
    readerOptions_->setIndexIoStats(
        std::make_shared<velox::io::IoStatistics>());
    TabletReader::Options options;
    options.loadDenseIndexes = true;
    options.ioOptions = *readerOptions_;
    return TabletReader::create(std::move(readFile), leafPool_.get(), options);
  }

  std::string tempFilePath(const std::string& name) {
    return tempDir_->getPath() + "/" + name;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> tempDir_;
  std::deque<std::string> valueStrings_;
  std::shared_ptr<velox::io::IoStatistics> ioStats_;
  std::unique_ptr<velox::io::ReaderOptions> readerOptions_;
};

TEST_F(DenseIndexRegistryTest, emptyRegistry) {
  auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
  auto ioStats = std::make_shared<velox::io::IoStatistics>();
  velox::io::ReaderOptions ioOptions(leafPool_.get());
  ioOptions.setMetadataIoStats(ioStats);
  ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  IndexLookup::Options options{.file = file, .ioOptions = &ioOptions};
  auto registry = DenseIndexRegistry::create({}, options, leafPool_.get());
  EXPECT_EQ(registry, nullptr);
}

TEST_F(DenseIndexRegistryTest, hashOnlyRegistry) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("hash_only");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  writeFile(filePath, rowType(), batches, std::move(options));

  auto tablet = openTablet(filePath);

  // Hash index on {"id"} should be found.
  auto* hashIdx = tablet->denseIndex({"id"});
  ASSERT_NE(hashIdx, nullptr);
  EXPECT_EQ(hashIdx->type(), IndexType::Hash);
  EXPECT_EQ(hashIdx->indexColumns(), std::vector<std::string>{"id"});

  // Non-existent columns should return null.
  EXPECT_EQ(tablet->denseIndex({"nonexistent"}), nullptr);
}

TEST_F(DenseIndexRegistryTest, sortedOnlyRegistry) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("sorted_only");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      SortedIndexConfigBuilder{}.withKeyColumns({"value"}).build());
  writeFile(filePath, rowType(), batches, std::move(options));

  auto tablet = openTablet(filePath);

  // Sorted index on {"value"} should be found.
  auto* sortedIdx = tablet->denseIndex({"value"});
  ASSERT_NE(sortedIdx, nullptr);
  EXPECT_EQ(sortedIdx->type(), IndexType::Sorted);
  EXPECT_EQ(sortedIdx->indexColumns(), std::vector<std::string>{"value"});

  // Non-existent columns should return null.
  EXPECT_EQ(tablet->denseIndex({"nonexistent"}), nullptr);
}

TEST_F(DenseIndexRegistryTest, hashAndSortedLookup) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("hash_and_sorted");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  options.denseIndexConfigs.push_back(
      SortedIndexConfigBuilder{}.withKeyColumns({"value"}).build());
  writeFile(filePath, rowType(), batches, std::move(options));

  auto tablet = openTablet(filePath);

  // Hash index on {"id"} should be found.
  auto* hashIdx = tablet->denseIndex({"id"});
  ASSERT_NE(hashIdx, nullptr);
  EXPECT_EQ(hashIdx->type(), IndexType::Hash);

  // Sorted index on {"value"} should be found.
  auto* sortedIdx = tablet->denseIndex({"value"});
  ASSERT_NE(sortedIdx, nullptr);
  EXPECT_EQ(sortedIdx->type(), IndexType::Sorted);

  // Non-existent columns should return null.
  EXPECT_EQ(tablet->denseIndex({"nonexistent"}), nullptr);
}

TEST_F(DenseIndexRegistryTest, indexCacheRetention) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("cache_retention");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  writeFile(filePath, rowType(), batches, std::move(options));

  auto tablet = openTablet(filePath);

  auto* hashIdx = tablet->denseIndex({"id"});
  ASSERT_NE(hashIdx, nullptr);
  EXPECT_EQ(hashIdx->type(), IndexType::Hash);
  EXPECT_EQ(hashIdx->indexColumns(), std::vector<std::string>{"id"});

  // Repeated lookup returns the same pinned index object.
  auto* hashIdx2 = tablet->denseIndex({"id"});
  EXPECT_EQ(hashIdx, hashIdx2);

  // Non-existent columns return null.
  EXPECT_EQ(tablet->denseIndex({"nonexistent"}), nullptr);
}

TEST_F(DenseIndexRegistryTest, firstIndexWinsForDuplicateColumns) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("cross_type_duplicate");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  options.denseIndexConfigs.push_back(
      SortedIndexConfigBuilder{}.withKeyColumns({"id"}).build());

  writeFile(filePath, rowType(), batches, std::move(options));

  auto tablet = openTablet(filePath);
  const auto* index = tablet->denseIndex({"id"});
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->type(), IndexType::Hash);

  const auto* sortedIndex = tablet->denseIndex(kDenseSortedIndexName, {"id"});
  ASSERT_NE(sortedIndex, nullptr);
  EXPECT_EQ(sortedIndex->type(), IndexType::Sorted);
}

} // namespace
} // namespace facebook::nimble::index
