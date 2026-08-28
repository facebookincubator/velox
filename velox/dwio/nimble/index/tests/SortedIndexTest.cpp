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
#include <thread>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/SortedIndex.h"
#include "velox/dwio/nimble/index/SortedIndexConfig.h"
#include "velox/dwio/nimble/index/SortedIndexWriter.h"
#include "velox/dwio/nimble/index/tests/SortedIndexTestUtils.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index {
namespace {

class SortedIndexTestBase {
 protected:
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

  void writeFile(
      const std::string& filePath,
      const std::vector<velox::VectorPtr>& batches,
      std::shared_ptr<const IndexConfig> indexConfig,
      std::optional<uint64_t> flushSize = std::nullopt) {
    WriterOptions options;
    options.denseIndexConfigs.push_back(std::move(indexConfig));
    if (flushSize.has_value()) {
      options.flushPolicyFactory = [size = flushSize.value()]() {
        return std::make_unique<StripeRawSizeFlushPolicy>(size);
      };
    }
    writeFile(filePath, rowType(), batches, std::move(options));
  }

  std::shared_ptr<TabletReader> openTablet(const std::string& filePath) {
    auto fs = velox::filesystems::getFileSystem(filePath, {});
    auto readFile = fs->openFileForRead(filePath);
    return TabletReader::create(
        std::move(readFile),
        leafPool_.get(),
        ::facebook::nimble::test::makeTestTabletOptions(leafPool_.get()));
  }

  std::string encodeLookupKey(
      const std::vector<std::string>& columns,
      int32_t idValue) {
    auto encoder = velox::serializer::KeyEncoder::create(
        columns,
        rowType(),
        std::vector<velox::core::SortOrder>(
            columns.size(), velox::core::SortOrder{true, false}),
        leafPool_.get());

    auto idCol =
        velox::BaseVector::create(velox::INTEGER(), 1, leafPool_.get());
    idCol->asFlatVector<int32_t>()->set(0, idValue);

    auto valStr = "v_" + std::to_string(idValue);
    auto valCol =
        velox::BaseVector::create(velox::VARCHAR(), 1, leafPool_.get());
    valCol->asFlatVector<velox::StringView>()->set(
        0, velox::StringView(valStr));

    auto lookupRow = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        rowType(),
        nullptr,
        1,
        std::vector<velox::VectorPtr>{idCol, valCol});

    Buffer buffer(*leafPool_);
    std::vector<std::string_view> encodedKeys;
    encoder->encode(lookupRow, encodedKeys, [&buffer](size_t size) {
      return buffer.reserve(size);
    });
    return std::string(encodedKeys[0]);
  }

  IndexLookup::LookupResult pointLookup(
      const IndexLookup* index,
      const std::vector<std::string>& columns,
      int32_t idValue) {
    auto key = encodeLookupKey(columns, idValue);
    return index->lookup(
        IndexLookup::LookupRequest::pointLookup({std::move(key)}));
  }

  IndexLookup::LookupResult rangeScan(
      const IndexLookup* index,
      const std::vector<std::string>& columns,
      int32_t lowerValue,
      int32_t upperValue) {
    auto lowerKey = encodeLookupKey(columns, lowerValue);
    auto upperKey = encodeLookupKey(columns, upperValue);
    return index->lookup(
        IndexLookup::LookupRequest::rangeScan(
            {velox::serializer::EncodedKeyBounds{
                std::move(lowerKey), std::move(upperKey)}}));
  }

  std::string tempFilePath(const std::string& name) {
    return tempDir_->getPath() + "/" + name;
  }

  static std::unique_ptr<SortedIndexWriter> createWriter(
      const std::shared_ptr<const IndexConfig>& config,
      const velox::TypePtr& type,
      velox::memory::MemoryPool* pool) {
    const IndexConfig* configs[] = {config.get()};
    return SortedIndexWriter::create(configs, type, pool);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> tempDir_;
  std::deque<std::string> valueStrings_;
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  std::shared_ptr<velox::io::IoStatistics> indexIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
};

class SortedIndexTest : public SortedIndexTestBase, public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("SortedIndexTest");
    leafPool_ = pool_->addLeafChild("leaf");
    tempDir_ = velox::common::testutil::TempDirectoryPath::create();
  }
};

struct SortedIndexTestParam {
  std::vector<std::string> columns;
  EncodingType encodingType{EncodingType::Prefix};

  std::string debugString() const {
    return fmt::format(
        "columns: [{}], encoding: {}", fmt::join(columns, ", "), encodingType);
  }
};

class SortedIndexParamTest
    : public SortedIndexTestBase,
      public ::testing::TestWithParam<SortedIndexTestParam> {
 protected:
  static void SetUpTestCase() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("SortedIndexParamTest");
    leafPool_ = pool_->addLeafChild("leaf");
    tempDir_ = velox::common::testutil::TempDirectoryPath::create();
  }

  const std::vector<std::string>& columns() const {
    return GetParam().columns;
  }

  std::shared_ptr<const IndexConfig> indexConfig() const {
    const auto encodingType = GetParam().encodingType;
    auto layout = encodingType == EncodingType::Prefix
        ? EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed}
        : EncodingLayout{
              EncodingType::Trivial,
              {},
              CompressionType::Uncompressed,
              {EncodingLayout{
                  EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
    return SortedIndexConfigBuilder{}
        .withKeyColumns(columns())
        .withEncodingLayout(std::move(layout))
        .build();
  }
};

INSTANTIATE_TEST_SUITE_P(
    SingleAndCompositeKey,
    SortedIndexParamTest,
    testing::Values(
        SortedIndexTestParam{{"id"}, EncodingType::Prefix},
        SortedIndexTestParam{{"id"}, EncodingType::Trivial},
        SortedIndexTestParam{{"id", "value"}, EncodingType::Prefix},
        SortedIndexTestParam{{"id", "value"}, EncodingType::Trivial}),
    [](const testing::TestParamInfo<SortedIndexTestParam>& info) {
      return fmt::format(
          "columns_{}_{}",
          fmt::join(info.param.columns, "_"),
          info.param.encodingType);
    });

TEST_P(SortedIndexParamTest, roundTrip) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 100));

  const auto filePath = tempFilePath("round_trip");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);
  EXPECT_EQ(sortedIdx->type(), IndexType::Sorted);
  EXPECT_EQ(sortedIdx->indexColumns(), columns());

  // Verify min/max keys are populated and ordered.
  const auto minKey = sortedIdx->minKey();
  const auto maxKey = sortedIdx->maxKey();
  EXPECT_FALSE(minKey.empty());
  EXPECT_FALSE(maxKey.empty());
  EXPECT_LE(minKey, maxKey);

  // Min key should match the encoded key for id=0, max for id=99.
  EXPECT_EQ(minKey, encodeLookupKey(columns(), 0));
  EXPECT_EQ(maxKey, encodeLookupKey(columns(), 99));
}

TEST_P(SortedIndexParamTest, pointLookup) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("point_lookup");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);

  {
    auto result = pointLookup(sortedIdx, columns(), 20);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 1);
  }

  {
    auto result = pointLookup(sortedIdx, columns(), 999);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }

  const auto stats = sortedIdx->stats();
  EXPECT_EQ(stats.at(SortedIndex::kNumPointLookups).sum, 2);
  EXPECT_EQ(stats.count(SortedIndex::kNumRangeScans), 0);
  EXPECT_EQ(stats.at(SortedIndex::kNumMinMaxSkips).sum, 1);
  EXPECT_EQ(stats.at(SortedIndex::kNumMatchedRows).sum, 1);
}

TEST_P(SortedIndexParamTest, rangeScan) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("range_scan");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);

  // Range [10, 20) should return 10 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 10, 20);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 10);
  }

  // Range [100, 200) is beyond max key — skipped by min/max filter.
  {
    auto result = rangeScan(sortedIdx, columns(), 100, 200);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }

  const auto stats = sortedIdx->stats();
  EXPECT_EQ(stats.count(SortedIndex::kNumPointLookups), 0);
  EXPECT_EQ(stats.at(SortedIndex::kNumRangeScans).sum, 2);
  EXPECT_EQ(stats.at(SortedIndex::kNumMinMaxSkips).sum, 1);
  EXPECT_EQ(stats.at(SortedIndex::kNumMatchedRows).sum, 10);
}

TEST_P(SortedIndexParamTest, rangeScanExclusiveUpperBound) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 10));

  const auto filePath = tempFilePath("range_exclusive");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);

  // Range [0, 10) should return all 10 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 0, 10);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 10);
  }

  // Range [0, 5) should return 5 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 0, 5);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 5);
  }

  // Range [5, 5) should return 0 rows (empty range).
  {
    auto result = rangeScan(sortedIdx, columns(), 5, 5);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
}

TEST_P(SortedIndexParamTest, multipleStripes) {
  std::vector<velox::VectorPtr> batches;
  for (int b = 0; b < 5; ++b) {
    batches.push_back(makeBatch(b * 20, 20));
  }

  const auto filePath = tempFilePath("multi_stripe");
  writeFile(filePath, batches, indexConfig(), /*flushSize=*/256);

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);
  EXPECT_GT(tablet->stripeCount(), 1);

  auto result = pointLookup(sortedIdx, columns(), 50);
  ASSERT_EQ(result.size(), 1);
  EXPECT_FALSE(result[0].empty());
}

TEST_F(SortedIndexTest, noSortedIndexConfig) {
  auto noIndexType = velox::ROW({{"x", velox::INTEGER()}});

  auto col = velox::BaseVector::create(velox::INTEGER(), 10, leafPool_.get());
  auto* flat = col->asFlatVector<int32_t>();
  for (int i = 0; i < 10; ++i) {
    flat->set(i, i);
  }
  auto batch = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      noIndexType,
      nullptr,
      10,
      std::vector<velox::VectorPtr>{col});

  const auto filePath = tempFilePath("no_index");
  writeFile(filePath, noIndexType, {batch}, WriterOptions{});

  auto tablet = openTablet(filePath);
  EXPECT_EQ(tablet->denseIndex({"x"}), nullptr);
}

TEST_F(SortedIndexTest, multipleIndices) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 20));

  const auto filePath = tempFilePath("multi_indices");
  {
    WriterOptions options;
    options.denseIndexConfigs.push_back(
        SortedIndexConfigBuilder{}.withKeyColumns({"id"}).build());
    options.denseIndexConfigs.push_back(
        SortedIndexConfigBuilder{}.withKeyColumns({"id", "value"}).build());
    writeFile(filePath, rowType(), batches, std::move(options));
  }

  auto tablet = openTablet(filePath);

  auto* index0 = tablet->denseIndex({"id"});
  ASSERT_NE(index0, nullptr);
  EXPECT_EQ(index0->indexColumns(), std::vector<std::string>{"id"});

  auto* index1 = tablet->denseIndex({"id", "value"});
  ASSERT_NE(index1, nullptr);
  const std::vector<std::string> expected{"id", "value"};
  EXPECT_EQ(index1->indexColumns(), expected);

  EXPECT_EQ(tablet->denseIndex({"value"}), nullptr);
  EXPECT_EQ(tablet->denseIndex({"value", "id"}), nullptr);
}

TEST_F(SortedIndexTest, rejectsNonSortedIndexName) {
  const auto config = std::make_shared<const SortedIndexConfig>(
      std::string{kDenseHashIndexName},
      std::vector<std::string>{"id"},
      EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
      0);
  const IndexConfig* configs[] = {config.get()};
  NIMBLE_ASSERT_THROW(
      SortedIndexWriter::create(configs, rowType(), leafPool_.get()),
      "Sorted index writer must use the built-in sorted index name");
}

TEST_P(SortedIndexParamTest, duplicateKeys) {
  // Write data with duplicate id values to verify they are all found.
  auto ids = velox::BaseVector::create(velox::INTEGER(), 6, leafPool_.get());
  auto values = velox::BaseVector::create(velox::VARCHAR(), 6, leafPool_.get());
  auto* flatIds = ids->asFlatVector<int32_t>();
  auto* flatValues = values->asFlatVector<velox::StringView>();
  // ids: [1, 2, 1, 3, 2, 1]
  std::vector<int32_t> idVals = {1, 2, 1, 3, 2, 1};
  for (int i = 0; i < 6; ++i) {
    flatIds->set(i, idVals[i]);
    valueStrings_.emplace_back("v_" + std::to_string(idVals[i]));
    flatValues->set(i, velox::StringView(valueStrings_.back()));
  }
  auto batch = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      rowType(),
      nullptr,
      6,
      std::vector<velox::VectorPtr>{ids, values});

  const auto filePath = tempFilePath("duplicate_keys");
  WriterOptions options;
  options.denseIndexConfigs.push_back(indexConfig());
  writeFile(filePath, rowType(), {batch}, std::move(options));

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);

  // Key=1 appears 3 times.
  {
    auto result = pointLookup(sortedIdx, columns(), 1);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 3);
  }

  // Key=2 appears 2 times.
  {
    auto result = pointLookup(sortedIdx, columns(), 2);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 2);
  }

  // Key=3 appears 1 time.
  {
    auto result = pointLookup(sortedIdx, columns(), 3);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 1);
  }

  // Key=999 doesn't exist.
  {
    auto result = pointLookup(sortedIdx, columns(), 999);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }

  // Range [1, 3) should return keys 1 and 2: 3 + 2 = 5 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 1, 3);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 5);
  }

  // Range [2, 4) should return keys 2 and 3: 2 + 1 = 3 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 2, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 3);
  }

  // Range [1, 4) should return all 6 rows.
  {
    auto result = rangeScan(sortedIdx, columns(), 1, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 6);
  }

  // Range [100, 200) is beyond max — empty.
  {
    auto result = rangeScan(sortedIdx, columns(), 100, 200);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
}

TEST_F(SortedIndexTest, duplicateKeysAcrossChunks) {
  // Write data where duplicate keys span chunk boundaries.
  // Use maxRowsPerKeyChunk=3 so that 6 entries with the same key span 2 chunks.
  const int32_t numRows = 10;
  auto ids =
      velox::BaseVector::create(velox::INTEGER(), numRows, leafPool_.get());
  auto values =
      velox::BaseVector::create(velox::VARCHAR(), numRows, leafPool_.get());
  auto* flatIds = ids->asFlatVector<int32_t>();
  auto* flatValues = values->asFlatVector<velox::StringView>();
  // ids: [1, 1, 1, 1, 1, 1, 2, 2, 3, 3]
  // After sorting by key, chunk boundary at row 3 splits key=1 across chunks.
  const std::vector<int32_t> idVals = {1, 1, 1, 1, 1, 1, 2, 2, 3, 3};
  for (int i = 0; i < numRows; ++i) {
    flatIds->set(i, idVals[i]);
    valueStrings_.emplace_back("v_" + std::to_string(idVals[i]));
    flatValues->set(i, velox::StringView(valueStrings_.back()));
  }
  auto batch = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      rowType(),
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, values});

  const auto filePath = tempFilePath("dup_across_chunks");
  WriterOptions options;
  options.denseIndexConfigs.push_back(
      SortedIndexConfigBuilder{}
          .withKeyColumns({"id"})
          .withMaxRowsPerKeyChunk(3)
          .build());
  writeFile(filePath, rowType(), {batch}, std::move(options));

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex({"id"});
  ASSERT_NE(sortedIdx, nullptr);

  // Key=1 appears 6 times, spanning at least 2 chunks (3 entries per chunk).
  {
    auto result = pointLookup(sortedIdx, {"id"}, 1);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 6);
  }

  // Key=2 appears 2 times.
  {
    auto result = pointLookup(sortedIdx, {"id"}, 2);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 2);
  }

  // Key=3 appears 2 times.
  {
    auto result = pointLookup(sortedIdx, {"id"}, 3);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 2);
  }

  // Non-existent key.
  {
    auto result = pointLookup(sortedIdx, {"id"}, 999);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }

  // Range [1, 2) spans 2 chunks of key=1: 6 rows.
  {
    auto result = rangeScan(sortedIdx, {"id"}, 1, 2);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 6);
  }

  // Range [1, 3) includes keys 1 and 2: 6 + 2 = 8 rows.
  {
    auto result = rangeScan(sortedIdx, {"id"}, 1, 3);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 8);
  }

  // Range [1, 4) includes all keys: 6 + 2 + 2 = 10 rows.
  {
    auto result = rangeScan(sortedIdx, {"id"}, 1, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 10);
  }

  // Range [2, 4) includes keys 2 and 3: 2 + 2 = 4 rows.
  {
    auto result = rangeScan(sortedIdx, {"id"}, 2, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 4);
  }

  // Range beyond max — empty.
  {
    auto result = rangeScan(sortedIdx, {"id"}, 100, 200);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
}

TEST_F(SortedIndexTest, createWithNullPool) {
  NIMBLE_ASSERT_THROW(
      createWriter(
          SortedIndexConfigBuilder{}.withKeyColumns({"id"}).build(),
          rowType(),
          nullptr),
      "memory pool must not be null");
}

TEST_F(SortedIndexTest, createWithEmptyConfigs) {
  EXPECT_EQ(SortedIndexWriter::create({}, rowType(), leafPool_.get()), nullptr);
}

TEST_F(SortedIndexTest, createWithEmptyColumns) {
  NIMBLE_ASSERT_THROW(
      createWriter(
          SortedIndexConfigBuilder{}.build(), rowType(), leafPool_.get()),
      "Sorted index must have at least one column");
}

TEST_F(SortedIndexTest, createWithNonExistentColumn) {
  VELOX_ASSERT_USER_THROW(
      createWriter(
          SortedIndexConfigBuilder{}.withKeyColumns({"non_existent"}).build(),
          rowType(),
          leafPool_.get()),
      "Field not found");
}

TEST_F(SortedIndexTest, rowCountOverflow) {
  auto overflowType = velox::ROW({{"key", velox::INTEGER()}});

  const auto config =
      SortedIndexConfigBuilder{}.withKeyColumns({"key"}).build();
  auto writer = createWriter(config, overflowType, leafPool_.get());
  ASSERT_NE(writer, nullptr);

  test::SortedIndexWriterTestHelper helper(writer.get());
  helper.setNumRows(std::numeric_limits<uint32_t>::max() - 5);

  auto keys = velox::BaseVector::create(velox::INTEGER(), 10, leafPool_.get());
  auto* flatKeys = keys->asFlatVector<int32_t>();
  for (int i = 0; i < 10; ++i) {
    flatKeys->set(i, i);
  }
  auto batch = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      overflowType,
      nullptr,
      10,
      std::vector<velox::VectorPtr>{keys});

  NIMBLE_ASSERT_USER_THROW(
      writer->write(batch), "Sorted index row count exceeds uint32 limit");
}

TEST_F(SortedIndexTest, indexDataReuseWithSameReader) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("same_reader_reuse");
  writeFile(
      filePath,
      batches,
      SortedIndexConfigBuilder{}.withKeyColumns({"value"}).build());

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());

  struct TestCase {
    bool cacheIndex;
    bool provideCache;
    bool expectRamHit;

    std::string debugString() const {
      return fmt::format(
          "cacheIndex={}, provideCache={}, expectRamHit={}",
          cacheIndex,
          provideCache,
          expectRamHit);
    }
  };

  std::vector<TestCase> testCases = {
      {true, true, true},
      {false, true, false},
      {false, false, false},
      {true, false, false},
  };

  auto fs = velox::filesystems::getFileSystem(filePath, {});

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    indexIoStats_ = std::make_shared<velox::io::IoStatistics>();

    auto readFile =
        std::shared_ptr<velox::ReadFile>(fs->openFileForRead(filePath));
    auto& ids = velox::fileIds();
    auto fileIdStr = fmt::format(
        "sortedSameReader_cache{}_provide{}",
        testCase.cacheIndex,
        testCase.provideCache);

    TabletReader::Options options;
    options.loadDenseIndexes = true;
    options.cacheIndex = testCase.cacheIndex;
    options.ioOptions.emplace(leafPool_.get())
        .setMetadataIoStats(metadataIoStats_)
        .setIndexIoStats(indexIoStats_);

    std::unique_ptr<velox::FileHandle> fileHandle;
    if (testCase.provideCache) {
      fileHandle = std::make_unique<velox::FileHandle>();
      fileHandle->file = readFile;
      fileHandle->uuid = velox::StringIdLease(ids, fileIdStr);
      fileHandle->groupId = velox::StringIdLease(ids, fileIdStr + "_group");
      options.cache = cache.get();
      options.fileHandle = fileHandle.get();
    }

    auto tablet =
        TabletReader::create(std::move(readFile), leafPool_.get(), options);
    ASSERT_NE(tablet->denseIndex({"value"}), nullptr);

    // Both lookups should return correct results. The pinned SortedIndex
    // object is reused across lookups within the same reader.
    auto firstResult = pointLookup(tablet->denseIndex({"value"}), {"value"}, 0);
    ASSERT_EQ(firstResult.size(), 1);
    ASSERT_FALSE(firstResult[0].empty());

    auto secondResult =
        pointLookup(tablet->denseIndex({"value"}), {"value"}, 25);
    ASSERT_EQ(secondResult.size(), 1);
    ASSERT_FALSE(secondResult[0].empty());

    // Verify the same index object is returned (pinned in MetadataCache).
    EXPECT_EQ(tablet->denseIndex({"value"}), tablet->denseIndex({"value"}));
  }

  cache->shutdown();
}

TEST_F(SortedIndexTest, indexDataReuseCrossReaders) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("cross_reader_reuse");
  writeFile(
      filePath,
      batches,
      SortedIndexConfigBuilder{}.withKeyColumns({"value"}).build());

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());

  struct TestCase {
    bool cacheIndex;
    bool provideCache;
    bool expectCrossReaderReuse;
    bool expectRamHit;

    std::string debugString() const {
      return fmt::format(
          "cacheIndex={}, provideCache={}, "
          "expectCrossReaderReuse={}, expectRamHit={}",
          cacheIndex,
          provideCache,
          expectCrossReaderReuse,
          expectRamHit);
    }
  };

  std::vector<TestCase> testCases = {
      {true, true, true, true},
      {false, true, false, false},
      {false, false, false, false},
      {true, false, false, false},
  };

  auto fs = velox::filesystems::getFileSystem(filePath, {});

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto& ids = velox::fileIds();
    auto fileIdStr = fmt::format(
        "sortedCrossReader_cache{}_provide{}",
        testCase.cacheIndex,
        testCase.provideCache);
    velox::StringIdLease fileId(ids, fileIdStr);
    velox::StringIdLease groupId(ids, fileIdStr + "_group");

    auto createReader = [&]() {
      metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
      indexIoStats_ = std::make_shared<velox::io::IoStatistics>();

      auto readFile =
          std::shared_ptr<velox::ReadFile>(fs->openFileForRead(filePath));

      TabletReader::Options options;
      options.loadDenseIndexes = true;
      options.cacheIndex = testCase.cacheIndex;
      options.ioOptions.emplace(leafPool_.get())
          .setMetadataIoStats(metadataIoStats_)
          .setIndexIoStats(indexIoStats_);

      std::unique_ptr<velox::FileHandle> fileHandle;
      if (testCase.provideCache) {
        fileHandle = std::make_unique<velox::FileHandle>();
        fileHandle->file = readFile;
        fileHandle->uuid = fileId;
        fileHandle->groupId = groupId;
        options.cache = cache.get();
        options.fileHandle = fileHandle.get();
      }

      auto reader =
          TabletReader::create(std::move(readFile), leafPool_.get(), options);
      return std::make_pair(std::move(reader), std::move(fileHandle));
    };

    // First reader: cold path reads from file.
    {
      auto [reader1, fh1] = createReader();
      ASSERT_NE(reader1->denseIndex({"value"}), nullptr);

      auto firstResult =
          pointLookup(reader1->denseIndex({"value"}), {"value"}, 0);
      ASSERT_EQ(firstResult.size(), 1);
      ASSERT_FALSE(firstResult[0].empty());
      EXPECT_GT(indexIoStats_->rawBytesRead(), 0)
          << "First reader should read index data from file";
    }

    // Second reader: verify functional correctness with fresh IO stats.
    {
      auto [reader2, fh2] = createReader();
      ASSERT_NE(reader2->denseIndex({"value"}), nullptr);

      auto secondResult =
          pointLookup(reader2->denseIndex({"value"}), {"value"}, 0);
      ASSERT_EQ(secondResult.size(), 1);
      ASSERT_FALSE(secondResult[0].empty());

      if (testCase.expectCrossReaderReuse) {
        EXPECT_GT(indexIoStats_->ramHit().count(), 0)
            << "Second reader should serve index data from cache";
      }
    }
  }

  cache->shutdown();
}

TEST_P(SortedIndexParamTest, concurrentLookup) {
  constexpr uint32_t kNumThreads = 16;
  constexpr uint32_t kLookupsPerThread = 200;
  constexpr int32_t kNumRows = 50;

  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, kNumRows));

  const auto filePath = tempFilePath("concurrent_lookup");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* sortedIdx = tablet->denseIndex(columns());
  ASSERT_NE(sortedIdx, nullptr);

  std::atomic_uint32_t errors{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (uint32_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(42 + t);
      std::uniform_int_distribution<int32_t> idDist(0, kNumRows - 1);
      for (uint32_t i = 0; i < kLookupsPerThread; ++i) {
        const auto id = idDist(rng);

        auto result = pointLookup(sortedIdx, columns(), id);
        if (result.size() != 1 || result[0].size() != 1) {
          ++errors;
          continue;
        }
        const auto& range = result[0][0];
        if (range.startRow != static_cast<uint32_t>(id) ||
            range.endRow != static_cast<uint32_t>(id) + 1) {
          ++errors;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(errors.load(), 0);
}

} // namespace
} // namespace facebook::nimble::index
