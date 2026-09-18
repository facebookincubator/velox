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

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/HashIndex.h"
#include "velox/dwio/nimble/index/HashIndexConfig.h"
#include "velox/dwio/nimble/index/HashIndexWriter.h"
#include "velox/dwio/nimble/index/IndexLookup.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index {
namespace {

class HashIndexTestBase {
 protected:
  // Schema: {id: INTEGER, value: VARCHAR}.
  // Both columns have unique values derived from row index.
  static velox::RowTypePtr rowType() {
    return velox::ROW({{"id", velox::INTEGER()}, {"value", velox::VARCHAR()}});
  }

  // Creates a batch of numRows with id=[startRow, startRow+numRows) and
  // value="v_{id}".
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

  // Writes batches to a file with the given schema and writer options.
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

  // Convenience overload using the default schema.
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

  // Opens a file and returns the TabletReader.
  std::shared_ptr<TabletReader> openTablet(const std::string& filePath) {
    auto fs = velox::filesystems::getFileSystem(filePath, {});
    auto readFile = fs->openFileForRead(filePath);
    return TabletReader::create(
        std::move(readFile),
        leafPool_.get(),
        ::facebook::nimble::test::makeTestTabletOptions(leafPool_.get()));
  }

  // Encodes a lookup key for a single row and returns the encoded key.
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

  // Performs a point lookup and returns the result.
  IndexLookup::LookupResult pointLookup(
      const IndexLookup* index,
      const std::vector<std::string>& columns,
      int32_t idValue) {
    auto key = encodeLookupKey(columns, idValue);
    return index->lookup(
        IndexLookup::LookupRequest::pointLookup({std::move(key)}));
  }

  std::string tempFilePath(const std::string& name) {
    return tempDir_->getPath() + "/" + name;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> tempDir_;
  // Deque for pointer stability across push_back calls, since StringViews
  // in FlatVector reference these strings.
  std::deque<std::string> valueStrings_;
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  std::shared_ptr<velox::io::IoStatistics> indexIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
};

// Non-parameterized fixture.
class HashIndexTest : public HashIndexTestBase, public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("HashIndexTest");
    leafPool_ = pool_->addLeafChild("leaf");
    tempDir_ = velox::common::testutil::TempDirectoryPath::create();
  }
};

// Parameterized fixture: single-column vs composite-key index.
struct HashIndexTestParam {
  std::vector<std::string> columns;

  std::string debugString() const {
    return fmt::format("columns: [{}]", fmt::join(columns, ", "));
  }
};

class HashIndexParamTest : public HashIndexTestBase,
                           public ::testing::TestWithParam<HashIndexTestParam> {
 protected:
  static void SetUpTestCase() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("HashIndexParamTest");
    leafPool_ = pool_->addLeafChild("leaf");
    tempDir_ = velox::common::testutil::TempDirectoryPath::create();
  }

  const std::vector<std::string>& columns() const {
    return GetParam().columns;
  }

  std::shared_ptr<const IndexConfig> indexConfig() const {
    return HashIndexConfigBuilder{}
        .withKeyColumns(columns())
        .withBloomFilter(10)
        .build();
  }
};

INSTANTIATE_TEST_SUITE_P(
    SingleAndCompositeKey,
    HashIndexParamTest,
    testing::Values(
        HashIndexTestParam{{"id"}},
        HashIndexTestParam{{"id", "value"}}),
    [](const testing::TestParamInfo<HashIndexTestParam>& info) {
      return fmt::format("columns_{}", fmt::join(info.param.columns, "_"));
    });

TEST_P(HashIndexParamTest, roundTrip) {
  std::vector<velox::VectorPtr> batches;
  for (int b = 0; b < 3; ++b) {
    batches.push_back(makeBatch(b * 100, 100));
  }

  const auto filePath = tempFilePath("round_trip");
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}
          .withKeyColumns(columns())
          .withLoadFactor(0.5f)
          .withBloomFilter(10)
          .build(),
      /*flushSize=*/512);

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);
  EXPECT_EQ(hashIndex->type(), IndexType::Hash);
  EXPECT_EQ(hashIndex->indexColumns(), columns());
  EXPECT_FALSE(hashIndex->minKey().empty());
  EXPECT_FALSE(hashIndex->maxKey().empty());
  EXPECT_LE(hashIndex->minKey(), hashIndex->maxKey());
}

TEST_P(HashIndexParamTest, pointLookup) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("point_lookup");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);
  EXPECT_FALSE(hashIndex->minKey().empty());
  EXPECT_FALSE(hashIndex->maxKey().empty());
  EXPECT_LE(hashIndex->minKey(), hashIndex->maxKey());

  {
    auto result = pointLookup(hashIndex, columns(), 20);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 1);
  }

  {
    auto result = pointLookup(hashIndex, columns(), 999);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }

  const auto stats = hashIndex->stats();
  EXPECT_EQ(stats.at(HashIndex::kNumLookups).sum, 2);
  EXPECT_EQ(stats.at(HashIndex::kNumMatchedRows).sum, 1);
  // Bloom filter may or may not skip the non-existent key depending on the
  // hash — just verify the stats entry is present if skips occurred.
  if (stats.count(HashIndex::kNumBloomFilterSkips) > 0) {
    EXPECT_LE(stats.at(HashIndex::kNumBloomFilterSkips).sum, 1);
  }
}

TEST_P(HashIndexParamTest, bloomFilterSkips) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("bloom_filter");
  // Use high bitsPerKey to eliminate false positives, making bloom filter
  // behavior fully deterministic for this test.
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}
          .withKeyColumns(columns())
          .withBloomFilter(40)
          .build());

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);

  const int32_t kNumExisting = 4;
  for (int32_t key : {0, 10, 25, 49}) {
    auto result = pointLookup(hashIndex, columns(), key);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 1);
  }
  {
    const auto stats = hashIndex->stats();
    EXPECT_EQ(stats.at(HashIndex::kNumLookups).sum, kNumExisting);
    EXPECT_EQ(stats.count(HashIndex::kNumBloomFilterSkips), 0);
    EXPECT_EQ(stats.at(HashIndex::kNumMatchedRows).sum, kNumExisting);
  }

  const int32_t kNumNonExistent = 100;
  for (int32_t key = 1'000; key < 1'000 + kNumNonExistent; ++key) {
    auto result = pointLookup(hashIndex, columns(), key);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
  {
    const auto stats = hashIndex->stats();
    EXPECT_EQ(
        stats.at(HashIndex::kNumLookups).sum, kNumExisting + kNumNonExistent);
    // Non-existent keys are skipped by min/max check and/or bloom filter.
    // All non-existent keys should be skipped by one of the two filters.
    const int64_t numMinMaxSkips = stats.count(HashIndex::kNumMinMaxSkips) > 0
        ? stats.at(HashIndex::kNumMinMaxSkips).sum
        : 0;
    const int64_t numBloomFilterSkips =
        stats.count(HashIndex::kNumBloomFilterSkips) > 0
        ? stats.at(HashIndex::kNumBloomFilterSkips).sum
        : 0;
    EXPECT_EQ(numMinMaxSkips + numBloomFilterSkips, kNumNonExistent);
    EXPECT_EQ(stats.at(HashIndex::kNumMatchedRows).sum, kNumExisting);
  }
}

TEST_P(HashIndexParamTest, rangeScanThrows) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 10));

  const auto filePath = tempFilePath("range_scan_throws");
  writeFile(filePath, batches, indexConfig());

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);

  auto key = encodeLookupKey(columns(), 5);
  NIMBLE_ASSERT_THROW(
      hashIndex->lookup(
          IndexLookup::LookupRequest::rangeScan(
              {velox::serializer::EncodedKeyBounds{key, key}})),
      "Hash index only supports point lookups");
}

TEST_P(HashIndexParamTest, multipleStripes) {
  std::vector<velox::VectorPtr> batches;
  for (int b = 0; b < 5; ++b) {
    batches.push_back(makeBatch(b * 20, 20));
  }

  const auto filePath = tempFilePath("multi_batch");
  writeFile(filePath, batches, indexConfig(), /*flushSize=*/256);

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);
  EXPECT_GT(tablet->stripeCount(), 1);

  auto result = pointLookup(hashIndex, columns(), 50);
  ASSERT_EQ(result.size(), 1);
  EXPECT_FALSE(result[0].empty());
}

TEST_P(HashIndexParamTest, partitionedIndex) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 200));

  const auto filePath = tempFilePath("partitioned");
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}
          .withKeyColumns(columns())
          .withBloomFilter(10)
          .withMaxPartitionSizeBytes(64)
          .build());

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);

  for (int32_t lookupValue : {0, 50, 99, 150, 199}) {
    SCOPED_TRACE(fmt::format("key={}", lookupValue));
    auto result = pointLookup(hashIndex, columns(), lookupValue);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 1);
  }

  // Non-existent key.
  auto result = pointLookup(hashIndex, columns(), 999);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].size(), 0);
}

TEST_P(HashIndexParamTest, partitionedMultipleBatches) {
  std::vector<velox::VectorPtr> batches;
  for (int b = 0; b < 5; ++b) {
    batches.push_back(makeBatch(b * 40, 40));
  }

  const auto filePath = tempFilePath("partitioned_multi_batch");
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}
          .withKeyColumns(columns())
          .withBloomFilter(10)
          .withMaxPartitionSizeBytes(128)
          .build(),
      /*flushSize=*/256);

  auto tablet = openTablet(filePath);
  auto* hashIndex = tablet->denseIndex(columns());
  ASSERT_NE(hashIndex, nullptr);
  EXPECT_GT(tablet->stripeCount(), 1);

  for (int32_t lookupValue : {0, 50, 100, 150, 199}) {
    SCOPED_TRACE(fmt::format("key={}", lookupValue));
    auto result = pointLookup(hashIndex, columns(), lookupValue);
    ASSERT_EQ(result.size(), 1);
    EXPECT_FALSE(result[0].empty());
  }
}

TEST_F(HashIndexTest, multipleIndices) {
  auto colAType = velox::ROW({
      {"col_a", velox::INTEGER()},
      {"col_b", velox::VARCHAR()},
      {"col_c", velox::BIGINT()},
  });

  const int32_t numRows = 20;
  auto colA =
      velox::BaseVector::create(velox::INTEGER(), numRows, leafPool_.get());
  auto colB =
      velox::BaseVector::create(velox::VARCHAR(), numRows, leafPool_.get());
  auto colC =
      velox::BaseVector::create(velox::BIGINT(), numRows, leafPool_.get());
  auto* flatA = colA->asFlatVector<int32_t>();
  auto* flatB = colB->asFlatVector<velox::StringView>();
  auto* flatC = colC->asFlatVector<int64_t>();
  std::vector<std::string> bStrings;
  for (int i = 0; i < numRows; ++i) {
    flatA->set(i, i);
    bStrings.emplace_back("b_" + std::to_string(i));
    flatB->set(i, velox::StringView(bStrings.back()));
    flatC->set(i, i * 1'000);
  }
  auto batch = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      colAType,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{colA, colB, colC});

  const auto filePath = tempFilePath("multi_indices");
  {
    WriterOptions options;
    options.denseIndexConfigs.push_back(
        HashIndexConfigBuilder{}.withKeyColumns({"col_a"}).build());
    options.denseIndexConfigs.push_back(
        HashIndexConfigBuilder{}.withKeyColumns({"col_b", "col_c"}).build());
    writeFile(filePath, colAType, {batch}, std::move(options));
  }

  {
    auto tablet = openTablet(filePath);

    auto* index0 = tablet->denseIndex({"col_a"});
    ASSERT_NE(index0, nullptr);
    EXPECT_EQ(index0->indexColumns(), std::vector<std::string>{"col_a"});

    auto* index1 = tablet->denseIndex({"col_b", "col_c"});
    ASSERT_NE(index1, nullptr);
    const std::vector<std::string> expected{"col_b", "col_c"};
    EXPECT_EQ(index1->indexColumns(), expected);

    EXPECT_EQ(tablet->denseIndex({"col_c"}), nullptr);
    EXPECT_EQ(tablet->denseIndex({"col_a", "col_b"}), nullptr);
  }
}

TEST_F(HashIndexTest, noHashIndexConfig) {
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

  {
    auto tablet = openTablet(filePath);
    EXPECT_EQ(tablet->denseIndex({"x"}), nullptr);
  }
}

TEST_F(HashIndexTest, indexDataReuseWithSameReader) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("same_reader_reuse");
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());

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
        "hashSameReader_cache{}_provide{}",
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
    ASSERT_NE(tablet->denseIndex({"id"}), nullptr);

    // Both lookups should return correct results. The pinned HashIndex
    // object is reused across lookups within the same reader.
    auto firstResult = pointLookup(tablet->denseIndex({"id"}), {"id"}, 0);
    ASSERT_EQ(firstResult.size(), 1);
    ASSERT_FALSE(firstResult[0].empty());

    auto secondResult = pointLookup(tablet->denseIndex({"id"}), {"id"}, 25);
    ASSERT_EQ(secondResult.size(), 1);
    ASSERT_FALSE(secondResult[0].empty());

    // Verify the same index object is returned (pinned in MetadataCache).
    EXPECT_EQ(tablet->denseIndex({"id"}), tablet->denseIndex({"id"}));
  }

  cache->shutdown();
}

TEST_F(HashIndexTest, indexDataReuseCrossReaders) {
  std::vector<velox::VectorPtr> batches;
  batches.push_back(makeBatch(0, 50));

  const auto filePath = tempFilePath("cross_reader_reuse");
  writeFile(
      filePath,
      batches,
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());

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
        "hashCrossReader_cache{}_provide{}",
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
      ASSERT_NE(reader1->denseIndex({"id"}), nullptr);

      auto firstResult = pointLookup(reader1->denseIndex({"id"}), {"id"}, 0);
      ASSERT_EQ(firstResult.size(), 1);
      ASSERT_FALSE(firstResult[0].empty());
    }

    // Second reader: verify functional correctness with fresh IO stats.
    {
      auto [reader2, fh2] = createReader();
      ASSERT_NE(reader2->denseIndex({"id"}), nullptr);

      auto secondResult = pointLookup(reader2->denseIndex({"id"}), {"id"}, 0);
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

} // namespace
} // namespace facebook::nimble::index
