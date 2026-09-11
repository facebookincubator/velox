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

#include "velox/dwio/nimble/tablet/ClusterIndexWriter.h"

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/Buffer.h"

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/ClusterIndexGroup.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/ChunkStatsWriter.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

namespace facebook::nimble::test {

using index::detail::ClusterIndexConfig;

using index::test::ChunkSpec;
using index::test::ChunkStatsTestHelper;
using index::test::createChunks;
using index::test::createKeyStream;
using index::test::KeyChunkSpec;

using index::test::ClusterIndexGroupTestHelper;

// Holds all the index data written during a test.
// Use with the callback methods to capture index data for verification.
struct TestFileIndex {
  // Key stream data written via writeStreamCallback().
  std::string keyStreamData;
  // Metadata sections for each stripe group, written via
  // createMetadataSectionCallback().
  std::vector<std::string> groupMetadataSections;
  // Root index data written via writeRootIndexCallback().
  std::string rootIndexData;
  // Chunk stats metadata sections for each stripe group.
  std::vector<std::string> chunkStatsGroupSections;
  // Root chunk stats data written via writeChunkRootIndexCallback().
  std::string chunkRootIndexData;
};

class ClusterIndexWriterTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
    chunkStatsWriter_ = std::make_unique<ChunkStatsWriter>(*pool_, 0);
  }

  // Returns a callback that appends key stream data to TestFileIndex.
  static auto writeStreamCallback(TestFileIndex& fileIndex) {
    return [&fileIndex](std::string_view data) {
      fileIndex.keyStreamData.append(data);
    };
  }

  // Returns a callback that stores metadata sections in TestFileIndex.
  static auto createMetadataSectionCallback(TestFileIndex& fileIndex) {
    return [&fileIndex](std::string_view metadata) -> MetadataSection {
      fileIndex.groupMetadataSections.emplace_back(metadata);
      return MetadataSection(0, metadata.size(), CompressionType::Uncompressed);
    };
  }

  // Returns a callback that stores root index data in TestFileIndex.
  static auto writeRootIndexCallback(TestFileIndex& fileIndex) {
    return [&fileIndex](const std::string& name, std::string_view content) {
      EXPECT_EQ(name, nimble::kIndexSection);
      fileIndex.rootIndexData = std::string(content);
    };
  }

  // Returns a callback that stores chunk stats metadata sections.
  static auto createChunkMetadataSectionCallback(TestFileIndex& fileIndex) {
    return [&fileIndex](std::string_view metadata) -> MetadataSection {
      fileIndex.chunkStatsGroupSections.emplace_back(metadata);
      return MetadataSection(0, metadata.size(), CompressionType::Uncompressed);
    };
  }

  // Returns a callback that stores chunk root index data.
  static auto writeChunkRootIndexCallback(TestFileIndex& fileIndex) {
    return [&fileIndex](const std::string& name, std::string_view content) {
      EXPECT_EQ(name, nimble::kChunkStatsSection);
      fileIndex.chunkRootIndexData = std::string(content);
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<ChunkStatsWriter> chunkStatsWriter_;
};

TEST_F(ClusterIndexWriterTest, basic) {
  // Test create with valid config
  {
    ClusterIndexConfig config{
        .columns = {"col1", "col2"},
        .sortOrders =
            {SortOrder{.ascending = true}, SortOrder{.ascending = false}},
        .enforceKeyOrder = true,
    };
    auto writer = ClusterIndexWriter::create(config, *pool_);
    EXPECT_NE(writer, nullptr);
  }

  // Test create with nullopt returns nullptr
  {
    auto writer = ClusterIndexWriter::create(std::nullopt, *pool_);
    EXPECT_EQ(writer, nullptr);
  }

  // Test create with empty columns throws
  {
    ClusterIndexConfig config{
        .columns = {},
        .sortOrders = {},
        .enforceKeyOrder = false,
    };
    NIMBLE_ASSERT_USER_THROW(
        ClusterIndexWriter::create(config, *pool_),
        "Index columns must not be empty in ClusterIndexConfig");
  }

  // Test create with empty sortOrders throws
  {
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {},
        .enforceKeyOrder = false,
    };
    NIMBLE_ASSERT_USER_THROW(
        ClusterIndexWriter::create(config, *pool_),
        "(1 vs. 0) Index columns and sort orders must have the same size in ClusterIndexConfig");
  }

  // Test create with mismatched columns and sortOrders size throws
  {
    ClusterIndexConfig config{
        .columns = {"col1", "col2"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = false,
    };
    NIMBLE_ASSERT_USER_THROW(
        ClusterIndexWriter::create(config, *pool_),
        "Index columns and sort orders must have the same size in ClusterIndexConfig");
  }
}

// Test all combinations of enforceKeyOrder, noDuplicateKey, and
// firstStripeSingleRow for stripe key enforcements validation.
TEST_F(ClusterIndexWriterTest, stripeKeyEnforcementValidation) {
  Buffer buffer{*pool_};

  // enforceKeyOrder=false: no checks are performed regardless of noDuplicateKey
  for (bool noDuplicateKey : {false, true}) {
    SCOPED_TRACE(
        fmt::format(
            "enforceKeyOrder=false, noDuplicateKey={}", noDuplicateKey));
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = false,
        .noDuplicateKey = noDuplicateKey};
    auto writer = ClusterIndexWriter::create(config, *pool_);

    // Multi-row first stripe with equal keys - pass (no checks)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{5, "aaa", "aaa"}})));

    // Duplicate key - pass (no checks)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})));

    // Out of order key - pass (no checks)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "000", "000"}})));
  }

  // enforceKeyOrder=true, noDuplicateKey=false, singleRowFirstStripe=false
  {
    SCOPED_TRACE(
        "enforceKeyOrder=true, noDuplicateKey=false, singleRowFirstStripe=false");
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
        .noDuplicateKey = false};
    auto writer = ClusterIndexWriter::create(config, *pool_);

    // Multi-row first stripe with equal keys - pass (duplicates allowed)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{5, "aaa", "aaa"}})));

    // Duplicate key - pass (duplicates allowed)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})));

    // Out of order key - fail
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    NIMBLE_ASSERT_USER_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "000", "000"}})),
        "Stripe keys must be in ascending order");
  }

  // enforceKeyOrder=true, noDuplicateKey=false, singleRowFirstStripe=true
  {
    SCOPED_TRACE(
        "enforceKeyOrder=true, noDuplicateKey=false, singleRowFirstStripe=true");
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
        .noDuplicateKey = false};
    auto writer = ClusterIndexWriter::create(config, *pool_);

    // Single-row first stripe with equal keys - pass
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})));

    // Duplicate key - pass (duplicates allowed)
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    EXPECT_NO_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})));

    // Out of order key - fail
    chunkStatsWriter_->newStripe(1);
    writer->newStripe();
    NIMBLE_ASSERT_USER_THROW(
        writer->addStripeKey(createKeyStream(buffer, {{1, "000", "000"}})),
        "Stripe keys must be in ascending order");
  }

  // enforceKeyOrder=true, noDuplicateKey=true, singleRowFirstStripe=false
  {
    SCOPED_TRACE(
        "enforceKeyOrder=true, noDuplicateKey=true, singleRowFirstStripe=false");
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
        .noDuplicateKey = true};

    // Multi-row first stripe with equal keys - fail (implies duplicates)
    {
      auto writer = ClusterIndexWriter::create(config, *pool_);
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      NIMBLE_ASSERT_USER_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "aaa", "aaa"}})),
          "Stripe keys must be in strictly ascending order");
    }

    // Multi-row first stripe with strictly ascending keys - pass
    {
      auto writer = ClusterIndexWriter::create(config, *pool_);
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "aaa", "bbb"}})));

      // Strictly greater key across stripes - pass
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "ccc", "ddd"}})));

      // Duplicate key across stripes (lastKey of prev stripe == lastKey of new
      // stripe) - fail
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      NIMBLE_ASSERT_USER_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "ddd", "ddd"}})),
          "Stripe keys must be in strictly ascending order");
    }

    // Out of order key across stripes - fail
    {
      auto writer = ClusterIndexWriter::create(config, *pool_);
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "bbb", "ccc"}})));

      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      NIMBLE_ASSERT_USER_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{5, "aaa", "aab"}})),
          "Stripe keys must be in strictly ascending order");
    }
  }

  // enforceKeyOrder=true, noDuplicateKey=true, singleRowFirstStripe=true
  {
    SCOPED_TRACE(
        "enforceKeyOrder=true, noDuplicateKey=true, singleRowFirstStripe=true");
    ClusterIndexConfig config{
        .columns = {"col1"},
        .sortOrders = {SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
        .noDuplicateKey = true};

    // Single-row first stripe with equal keys - pass
    // Then strictly greater key - pass
    // Then duplicate key - fail
    {
      auto writer = ClusterIndexWriter::create(config, *pool_);
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})));

      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{1, "bbb", "bbb"}})));

      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      NIMBLE_ASSERT_USER_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{1, "bbb", "bbb"}})),
          "Stripe keys must be in strictly ascending order");
    }

    // Single-row first stripe, then out of order key - fail
    {
      auto writer = ClusterIndexWriter::create(config, *pool_);
      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      EXPECT_NO_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{1, "ccc", "ccc"}})));

      chunkStatsWriter_->newStripe(1);
      writer->newStripe();
      NIMBLE_ASSERT_USER_THROW(
          writer->addStripeKey(createKeyStream(buffer, {{1, "aaa", "aaa"}})),
          "Stripe keys must be in strictly ascending order");
    }
  }
}

TEST_F(ClusterIndexWriterTest, checkNotFinalizedAfterWriteRootIndex) {
  ClusterIndexConfig config{
      .columns = {"col1"},
      .sortOrders = {SortOrder{.ascending = true}},
      .enforceKeyOrder = false};
  auto writer = ClusterIndexWriter::create(config, *pool_);

  Buffer buffer{*pool_};
  TestFileIndex fileIndex;
  chunkStatsWriter_->newStripe(2);
  writer->newStripe();

  auto keyStream = createKeyStream(buffer, {{100, "aaa", "bbb"}});
  writer->addStripeKey(keyStream);

  writer->writeKeyStream(0, keyStream.chunks, writeStreamCallback(fileIndex));

  auto chunks = createChunks(buffer, {{100, 20}});
  chunkStatsWriter_->addStream(0, chunks);

  // Write root index - this finalizes the writer
  writer->writeRoot({0}, writeRootIndexCallback(fileIndex));

  // After finalization, mutation operations should throw
  NIMBLE_ASSERT_THROW(
      writer->newStripe(), "ClusterIndexWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer->addStripeKey(keyStream), "ClusterIndexWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer->writeKeyStream(
          0, keyStream.chunks, writeStreamCallback(fileIndex)),
      "ClusterIndexWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer->writeRoot({0}, writeRootIndexCallback(fileIndex)),
      "ClusterIndexWriter has been finalized");
}

TEST_F(ClusterIndexWriterTest, emptyStripeKeys) {
  ClusterIndexConfig config{
      .columns = {"col1"},
      .sortOrders = {SortOrder{.ascending = true}},
      .enforceKeyOrder = false};
  auto writer = ClusterIndexWriter::create(config, *pool_);

  TestFileIndex fileIndex;
  // writeRootIndex should succeed but not write anything when stripe keys are
  // empty
  EXPECT_NO_THROW(writer->writeRoot({}, writeRootIndexCallback(fileIndex)));
  EXPECT_FALSE(fileIndex.rootIndexData.empty());
  EXPECT_TRUE(fileIndex.keyStreamData.empty());
  EXPECT_TRUE(fileIndex.groupMetadataSections.empty());
}

TEST_F(ClusterIndexWriterTest, emptyStreamInStripe) {
  ClusterIndexConfig config{
      .columns = {"col1"},
      .sortOrders = {SortOrder{.ascending = true}},
      .enforceKeyOrder = false};
  auto writer = ClusterIndexWriter::create(config, *pool_);

  Buffer buffer{*pool_};
  TestFileIndex fileIndex;
  chunkStatsWriter_->newStripe(3);
  writer->newStripe();

  auto keyStream = createKeyStream(buffer, {{100, "aaa", "bbb"}});
  writer->addStripeKey(keyStream);

  writer->writeKeyStream(0, keyStream.chunks, writeStreamCallback(fileIndex));

  // Stream 0 has data
  auto chunks0 = createChunks(buffer, {{50, 10}, {50, 12}});
  chunkStatsWriter_->addStream(0, chunks0);

  // Stream 1 is empty (no addStreamIndex call)

  // Stream 2 has data
  auto chunks2 = createChunks(buffer, {{100, 25}});
  chunkStatsWriter_->addStream(2, chunks2);

  EXPECT_NO_THROW(
      writer->writeGroup(1, createMetadataSectionCallback(fileIndex)));
  chunkStatsWriter_->writeGroup(
      3, 1, createChunkMetadataSectionCallback(fileIndex));
  EXPECT_EQ(fileIndex.groupMetadataSections.size(), 1);
}

// Test writing and reading back index data with a single stripe group.
// This test verifies the complete write/read cycle by:
// 1. Writing index data using ClusterIndexWriter with fake callbacks
// 2. Reading back the serialized data using ClusterIndex and ClusterIndexGroup
// 3. Verifying all index metadata matches expected values
// All stripes are in a single group, testing multi-stripe index within one
// group.
TEST_F(ClusterIndexWriterTest, writeAndReadWithSingleGroup) {
  ClusterIndexConfig config{
      .columns = {"col1", "col2"},
      .sortOrders =
          {SortOrder{.ascending = true}, SortOrder{.ascending = false}},
      .enforceKeyOrder = true,
  };
  auto writer = ClusterIndexWriter::create(config, *pool_);

  Buffer buffer{*pool_};
  TestFileIndex fileIndex;

  // ========== Stripe 0 ==========
  // Total rows: 100
  // Stream 0: 3 chunks (rows: 30, 45, 25)
  // Stream 1: 2 chunks (rows: 60, 40)
  // Key stream: 2 chunks (rows: 50, 50), keys: "aaa"->"bbb", "bbb"->"ccc"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "aaa", "bbb"},
            {50, "bbb", "ccc"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{30, 10}, {45, 15}, {25, 8}});
    auto chunks1 = createChunks(buffer, {{60, 20}, {40, 12}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
  }

  // ========== Stripe 1 ==========
  // Total rows: 200
  // Stream 0: 4 chunks (rows: 40, 55, 65, 40)
  // Stream 1: 1 chunk (rows: 200)
  // Key stream: 4 chunks (rows: 50, 50, 50, 50), keys: "ccc"->"ddd",
  // "ddd"->"eee", "eee"->"fff", "fff"->"ggg"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "ccc", "ddd"},
            {50, "ddd", "eee"},
            {50, "eee", "fff"},
            {50, "fff", "ggg"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{40, 5}, {55, 18}, {65, 22}, {40, 7}});
    auto chunks1 = createChunks(buffer, {{200, 50}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
  }

  // ========== Stripe 2 ==========
  // Total rows: 150
  // Stream 0: 2 chunks (rows: 100, 50)
  // Stream 1: 5 chunks (rows: 20, 35, 40, 25, 30)
  // Key stream: 1 chunk (rows: 150), keys: "ggg"->"hhh"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {150, "ggg", "hhh"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{100, 30}, {50, 14}});
    auto chunks1 =
        createChunks(buffer, {{20, 6}, {35, 9}, {40, 11}, {25, 4}, {30, 16}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
  }

  // Write single index group containing all 3 stripes
  writer->writeGroup(3, createMetadataSectionCallback(fileIndex));
  chunkStatsWriter_->writeGroup(
      2, 3, createChunkMetadataSectionCallback(fileIndex));

  // Write root index with all stripes in group 0
  writer->writeRoot({0, 0, 0}, writeRootIndexCallback(fileIndex));
  chunkStatsWriter_->writeRoot(writeChunkRootIndexCallback(fileIndex));

  // ========== Verify written data ==========
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  // Read and verify root index
  Section rootIndexSection{MetadataBuffer(
      MetadataBuffer::decompress(
          fileIndex.rootIndexData,
          CompressionType::Uncompressed,
          pool_.get()))};
  auto clusterIndex = index::ClusterIndex::create(std::move(rootIndexSection));

  // Verify basic properties
  EXPECT_EQ(clusterIndex->numStripes(), 3);
  EXPECT_EQ(clusterIndex->numPartitions(), 1);
  EXPECT_EQ(clusterIndex->indexColumns().size(), 2);
  EXPECT_EQ(clusterIndex->indexColumns()[0], "col1");
  EXPECT_EQ(clusterIndex->indexColumns()[1], "col2");

  // Verify sort orders
  ASSERT_EQ(clusterIndex->sortOrders().size(), 2);
  EXPECT_EQ(clusterIndex->sortOrders()[0], SortOrder{.ascending = true});
  EXPECT_EQ(clusterIndex->sortOrders()[1], SortOrder{.ascending = false});

  // Verify stripe keys
  EXPECT_EQ(clusterIndex->minKey(), "aaa");
  EXPECT_EQ(clusterIndex->stripeKey(0), "ccc");
  EXPECT_EQ(clusterIndex->stripeKey(1), "ggg");
  EXPECT_EQ(clusterIndex->stripeKey(2), "hhh");
  EXPECT_EQ(clusterIndex->maxKey(), "hhh");

  // Verify key lookups
  // Keys before minKey
  EXPECT_FALSE(clusterIndex->lookup("9").has_value());
  EXPECT_FALSE(clusterIndex->lookup("aa").has_value());

  // Stripe 0: ["aaa", "ccc"]
  EXPECT_EQ(clusterIndex->lookup("aaa")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("bbb")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("ccc")->stripeIndex, 0);

  // Stripe 1: ("ccc", "ggg"]
  EXPECT_EQ(clusterIndex->lookup("ccd")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("ddd")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("eee")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("fff")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("ggg")->stripeIndex, 1);

  // Stripe 2: ("ggg", "hhh"]
  EXPECT_EQ(clusterIndex->lookup("ggh")->stripeIndex, 2);
  EXPECT_EQ(clusterIndex->lookup("hhh")->stripeIndex, 2);

  // Keys after maxKey
  EXPECT_FALSE(clusterIndex->lookup("hhi").has_value());
  EXPECT_FALSE(clusterIndex->lookup("iii").has_value());

  // ========== Verify Single Group (all 3 stripes) ==========
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[0],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        0, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 0);
    EXPECT_EQ(helper.firstStripe(), 0);
    EXPECT_EQ(helper.stripeCount(), 3);

    // Verify key stream stats
    // Stripe 0: 2 chunks (rows: 50, 50), keys: "bbb", "ccc"
    // Stripe 1: 4 chunks (rows: 50, 50, 50, 50), keys: "ddd", "eee", "fff",
    // "ggg" Stripe 2: 1 chunk (rows: 150), keys: "hhh"
    const auto keyStats = helper.keyStreamStats();

    // Accumulated chunk counts per stripe: 2, 2+4=6, 6+1=7
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2, 6, 7}));

    // Accumulated row counts per chunk (within each stripe)
    // Stripe 0: 50, 100
    // Stripe 1: 50, 100, 150, 200
    // Stripe 2: 150
    EXPECT_EQ(
        keyStats.chunkRows,
        (std::vector<uint32_t>{50, 100, 50, 100, 150, 200, 150}));

    // Last key for each chunk
    EXPECT_EQ(
        keyStats.chunkKeys,
        (std::vector<std::string>{
            "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"}));
  }

  // Verify chunk index for stream position data
  {
    auto chunkStats = index::ChunkStatsGroup::create(
        0,
        3,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[0],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    EXPECT_EQ(chunkHelper.streamCount(), 2);

    // Verify stream 0 position index stats
    // Stripe 0: 3 chunks (rows: 30, 45, 25)
    // Stripe 1: 4 chunks (rows: 40, 55, 65, 40)
    // Stripe 2: 2 chunks (rows: 100, 50)
    {
      auto stream0Stats = chunkHelper.streamStats(0);

      // Per-stream accumulated chunk counts:
      // Stripe 0: 3 chunks -> accumulated = 3
      // Stripe 1: 4 chunks -> accumulated = 3 + 4 = 7
      // Stripe 2: 2 chunks -> accumulated = 7 + 2 = 9
      EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{3, 7, 9}));

      // Accumulated row counts per chunk
      // Stripe 0: 30, 75, 100
      // Stripe 1: 40, 95, 160, 200
      // Stripe 2: 100, 150
      EXPECT_EQ(
          stream0Stats.chunkRows,
          (std::vector<uint32_t>{30, 75, 100, 40, 95, 160, 200, 100, 150}));
    }

    // Verify stream 1 position index stats
    // Stripe 0: 2 chunks (rows: 60, 40)
    // Stripe 1: 1 chunk (rows: 200)
    // Stripe 2: 5 chunks (rows: 20, 35, 40, 25, 30)
    {
      auto stream1Stats = chunkHelper.streamStats(1);

      // Per-stream accumulated chunk counts for stream 1:
      // Stripe 0: 2 chunks -> accumulated = 2
      // Stripe 1: 1 chunk -> accumulated = 2 + 1 = 3
      // Stripe 2: 5 chunks -> accumulated = 3 + 5 = 8
      EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{2, 3, 8}));

      // Accumulated row counts per chunk
      // Stripe 0: 60, 100
      // Stripe 1: 200
      // Stripe 2: 20, 55, 95, 120, 150
      EXPECT_EQ(
          stream1Stats.chunkRows,
          (std::vector<uint32_t>{60, 100, 200, 20, 55, 95, 120, 150}));
    }
  }
}

// Test writing and reading back index data with multiple stripe groups.
// This test verifies the complete write/read cycle by:
// 1. Writing index data using ClusterIndexWriter with fake callbacks
// 2. Reading back the serialized data using ClusterIndex and ClusterIndexGroup
// 3. Verifying all index metadata matches expected values
TEST_F(ClusterIndexWriterTest, writeAndReadWithMultipleGroups) {
  ClusterIndexConfig config{
      .columns = {"col1", "col2"},
      .sortOrders =
          {SortOrder{.ascending = true}, SortOrder{.ascending = false}},
      .enforceKeyOrder = true,
  };
  auto writer = ClusterIndexWriter::create(config, *pool_);

  Buffer buffer{*pool_};
  TestFileIndex fileIndex;

  // ========== Stripe Group 0: Stripe 0 ==========
  // Total rows: 100
  // Stream 0: 2 chunks (rows: 50, 50)
  // Stream 1: 3 chunks (rows: 30, 40, 30)
  // Key stream: 2 chunks (rows: 50, 50), keys: "aaa"->"bbb", "bbb"->"ccc"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "aaa", "bbb"},
            {50, "bbb", "ccc"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{50, 10}, {50, 12}});
    auto chunks1 = createChunks(buffer, {{30, 8}, {40, 10}, {30, 6}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        2, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // ========== Stripe Group 1: Stripe 1 ==========
  // Total rows: 150
  // Stream 0: 3 chunks (rows: 40, 60, 50)
  // Stream 1: 2 chunks (rows: 80, 70)
  // Key stream: 3 chunks (rows: 50, 50, 50), keys: "ccc"->"ddd", "ddd"->"eee",
  // "eee"->"fff"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "ccc", "ddd"},
            {50, "ddd", "eee"},
            {50, "eee", "fff"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{40, 8}, {60, 14}, {50, 11}});
    auto chunks1 = createChunks(buffer, {{80, 18}, {70, 15}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        2, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // ========== Stripe Group 2: Stripe 2 ==========
  // Total rows: 120
  // Stream 0: 2 chunks (rows: 70, 50)
  // Stream 1: 4 chunks (rows: 25, 35, 30, 30)
  // Key stream: 2 chunks (rows: 60, 60), keys: "fff"->"ggg", "ggg"->"hhh"
  {
    chunkStatsWriter_->newStripe(2);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {60, "fff", "ggg"},
            {60, "ggg", "hhh"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{70, 16}, {50, 13}});
    auto chunks1 = createChunks(buffer, {{25, 5}, {35, 7}, {30, 6}, {30, 8}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        2, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // Write root index with stripe group indices {0, 1, 2}
  writer->writeRoot({0, 1, 2}, writeRootIndexCallback(fileIndex));
  chunkStatsWriter_->writeRoot(writeChunkRootIndexCallback(fileIndex));

  // ========== Verify written data ==========
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 3);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  // Read and verify root index
  Section rootIndexSection{MetadataBuffer(
      MetadataBuffer::decompress(
          fileIndex.rootIndexData,
          CompressionType::Uncompressed,
          pool_.get()))};
  auto clusterIndex = index::ClusterIndex::create(std::move(rootIndexSection));

  // Verify basic properties
  EXPECT_EQ(clusterIndex->numStripes(), 3);
  EXPECT_EQ(clusterIndex->numPartitions(), 3);
  EXPECT_EQ(clusterIndex->indexColumns().size(), 2);
  EXPECT_EQ(clusterIndex->indexColumns()[0], "col1");
  EXPECT_EQ(clusterIndex->indexColumns()[1], "col2");

  // Verify stripe keys
  EXPECT_EQ(clusterIndex->minKey(), "aaa");
  EXPECT_EQ(clusterIndex->stripeKey(0), "ccc");
  EXPECT_EQ(clusterIndex->stripeKey(1), "fff");
  EXPECT_EQ(clusterIndex->stripeKey(2), "hhh");
  EXPECT_EQ(clusterIndex->maxKey(), "hhh");

  // Verify key lookups
  // Keys before minKey
  EXPECT_FALSE(clusterIndex->lookup("9").has_value());
  EXPECT_FALSE(clusterIndex->lookup("aa").has_value());

  // Stripe 0: ["aaa", "ccc"]
  EXPECT_EQ(clusterIndex->lookup("aaa")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("bbb")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("ccc")->stripeIndex, 0);

  // Stripe 1: ("ccc", "fff"]
  EXPECT_EQ(clusterIndex->lookup("ccd")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("ddd")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("eee")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("fff")->stripeIndex, 1);

  // Stripe 2: ("fff", "hhh"]
  EXPECT_EQ(clusterIndex->lookup("ffg")->stripeIndex, 2);
  EXPECT_EQ(clusterIndex->lookup("ggg")->stripeIndex, 2);
  EXPECT_EQ(clusterIndex->lookup("hhh")->stripeIndex, 2);

  // Keys after maxKey
  EXPECT_FALSE(clusterIndex->lookup("hhi").has_value());
  EXPECT_FALSE(clusterIndex->lookup("iii").has_value());

  // ========== Verify Group 0 (Stripe 0) ==========
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[0],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        0, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 0);
    EXPECT_EQ(helper.firstStripe(), 0);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{50, 100}));
    EXPECT_EQ(keyStats.chunkKeys, (std::vector<std::string>{"bbb", "ccc"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        0,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[0],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    EXPECT_EQ(chunkHelper.streamCount(), 2);

    // Verify stream 0 stats: 2 chunks (rows: 50, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{50, 100}));

    // Verify stream 1 stats: 3 chunks (rows: 30, 40, 30)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{30, 70, 100}));
  }

  // ========== Verify Group 1 (Stripe 1) ==========
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[1],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        1, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 1);
    EXPECT_EQ(helper.firstStripe(), 1);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats: 3 chunks
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{50, 100, 150}));
    EXPECT_EQ(
        keyStats.chunkKeys, (std::vector<std::string>{"ddd", "eee", "fff"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        1,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[1],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    EXPECT_EQ(chunkHelper.streamCount(), 2);

    // Verify stream 0 stats: 3 chunks (rows: 40, 60, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{40, 100, 150}));

    // Verify stream 1 stats: 2 chunks (rows: 80, 70)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{80, 150}));
  }

  // ========== Verify Group 2 (Stripe 2) ==========
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[2],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        2, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 2);
    EXPECT_EQ(helper.firstStripe(), 2);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats: 2 chunks
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{60, 120}));
    EXPECT_EQ(keyStats.chunkKeys, (std::vector<std::string>{"ggg", "hhh"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        2,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[2],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    EXPECT_EQ(chunkHelper.streamCount(), 2);

    // Verify stream 0 stats: 2 chunks (rows: 70, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{70, 120}));

    // Verify stream 1 stats: 4 chunks (rows: 25, 35, 30, 30)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{4}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{25, 60, 90, 120}));
  }
}

// Test writing and reading back index data with multiple stripe groups and
// empty streams. This test verifies the position index handles missing streams
// correctly across group boundaries.
//
// Configuration (4 stripes, 4 streams, 4 groups - one group per stripe):
// - Stream 0: empty in first stripe (stripe 0 / group 0)
// - Stream 1: empty in middle stripe (stripe 1 / group 1)
// - Stream 2: empty in stripe 2 (group 2)
// - Stream 3: has data in all stripes
// - Stripe 3 / Group 3: ALL streams have data (no empty streams)
TEST_F(ClusterIndexWriterTest, writeAndReadWithMultipleGroupsAndEmptyStream) {
  ClusterIndexConfig config{
      .columns = {"col1"},
      .sortOrders = {SortOrder{.ascending = true}},
      .enforceKeyOrder = true,
  };
  auto writer = ClusterIndexWriter::create(config, *pool_);

  Buffer buffer{*pool_};
  TestFileIndex fileIndex;

  // ========== Stripe 0 (Group 0) ==========
  // Total rows: 100
  // Stream 0: EMPTY (no data in first stripe)
  // Stream 1: 2 chunks (rows: 50, 50)
  // Stream 2: 3 chunks (rows: 30, 40, 30)
  // Stream 3: 2 chunks (rows: 60, 40)
  // Key stream: 2 chunks (rows: 50, 50), keys: "aaa"->"bbb", "bbb"->"ccc"
  {
    chunkStatsWriter_->newStripe(4);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "aaa", "bbb"},
            {50, "bbb", "ccc"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    // Stream 0 is empty - no addStreamIndex call
    auto chunks1 = createChunks(buffer, {{50, 10}, {50, 12}});
    auto chunks2 = createChunks(buffer, {{30, 8}, {40, 10}, {30, 6}});
    auto chunks3 = createChunks(buffer, {{60, 15}, {40, 10}});
    chunkStatsWriter_->addStream(1, chunks1);
    chunkStatsWriter_->addStream(2, chunks2);
    chunkStatsWriter_->addStream(3, chunks3);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        4, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // ========== Stripe 1 (Group 1) ==========
  // Total rows: 150
  // Stream 0: 3 chunks (rows: 40, 60, 50)
  // Stream 1: EMPTY (no data in middle stripe)
  // Stream 2: 2 chunks (rows: 80, 70)
  // Stream 3: 1 chunk (rows: 150)
  // Key stream: 3 chunks (rows: 50, 50, 50), keys: "ccc"->"ddd", "ddd"->"eee",
  // "eee"->"fff"
  {
    chunkStatsWriter_->newStripe(4);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "ccc", "ddd"},
            {50, "ddd", "eee"},
            {50, "eee", "fff"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{40, 8}, {60, 14}, {50, 11}});
    // Stream 1 is empty - no addStreamIndex call
    auto chunks2 = createChunks(buffer, {{80, 18}, {70, 15}});
    auto chunks3 = createChunks(buffer, {{150, 30}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(2, chunks2);
    chunkStatsWriter_->addStream(3, chunks3);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        4, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // ========== Stripe 2 (Group 2) ==========
  // Total rows: 120
  // Stream 0: 2 chunks (rows: 70, 50)
  // Stream 1: 4 chunks (rows: 25, 35, 30, 30)
  // Stream 2: EMPTY (no data in this stripe)
  // Stream 3: 3 chunks (rows: 40, 40, 40)
  // Key stream: 2 chunks (rows: 60, 60), keys: "fff"->"ggg", "ggg"->"hhh"
  {
    chunkStatsWriter_->newStripe(4);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {60, "fff", "ggg"},
            {60, "ggg", "hhh"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{70, 16}, {50, 13}});
    auto chunks1 = createChunks(buffer, {{25, 5}, {35, 7}, {30, 6}, {30, 8}});
    // Stream 2 is empty - no addStreamIndex call
    auto chunks3 = createChunks(buffer, {{40, 10}, {40, 10}, {40, 10}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
    chunkStatsWriter_->addStream(3, chunks3);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        4, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // ========== Stripe 3 (Group 3) ==========
  // Total rows: 100 - ALL STREAMS HAVE DATA
  // Stream 0: 2 chunks (rows: 50, 50)
  // Stream 1: 2 chunks (rows: 40, 60)
  // Stream 2: 1 chunk (rows: 100)
  // Stream 3: 2 chunks (rows: 30, 70)
  // Key stream: 2 chunks (rows: 50, 50), keys: "hhh"->"iii", "iii"->"jjj"
  {
    chunkStatsWriter_->newStripe(4);
    writer->newStripe();
    auto keyStream = createKeyStream(
        buffer,
        {
            {50, "hhh", "iii"},
            {50, "iii", "jjj"},
        });
    writer->addStripeKey(keyStream);
    writer->writeKeyStream(
        fileIndex.keyStreamData.size(),
        keyStream.chunks,
        writeStreamCallback(fileIndex));

    auto chunks0 = createChunks(buffer, {{50, 12}, {50, 12}});
    auto chunks1 = createChunks(buffer, {{40, 10}, {60, 14}});
    auto chunks2 = createChunks(buffer, {{100, 25}});
    auto chunks3 = createChunks(buffer, {{30, 8}, {70, 18}});
    chunkStatsWriter_->addStream(0, chunks0);
    chunkStatsWriter_->addStream(1, chunks1);
    chunkStatsWriter_->addStream(2, chunks2);
    chunkStatsWriter_->addStream(3, chunks3);
    writer->writeGroup(1, createMetadataSectionCallback(fileIndex));
    chunkStatsWriter_->writeGroup(
        4, 1, createChunkMetadataSectionCallback(fileIndex));
  }

  // Write root index with stripe group indices {0, 1, 2, 3}
  writer->writeRoot({0, 1, 2, 3}, writeRootIndexCallback(fileIndex));
  chunkStatsWriter_->writeRoot(writeChunkRootIndexCallback(fileIndex));

  // ========== Verify written data ==========
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 4);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  // Read and verify root index
  Section rootIndexSection{MetadataBuffer(
      MetadataBuffer::decompress(
          fileIndex.rootIndexData,
          CompressionType::Uncompressed,
          pool_.get()))};
  auto clusterIndex = index::ClusterIndex::create(std::move(rootIndexSection));

  // Verify basic properties
  EXPECT_EQ(clusterIndex->numStripes(), 4);
  EXPECT_EQ(clusterIndex->numPartitions(), 4);

  // Verify key lookups
  EXPECT_EQ(clusterIndex->lookup("aaa")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("ccc")->stripeIndex, 0);
  EXPECT_EQ(clusterIndex->lookup("ddd")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("fff")->stripeIndex, 1);
  EXPECT_EQ(clusterIndex->lookup("ggg")->stripeIndex, 2);
  EXPECT_EQ(clusterIndex->lookup("hhh")->stripeIndex, 2);
  EXPECT_EQ(clusterIndex->lookup("iii")->stripeIndex, 3);
  EXPECT_EQ(clusterIndex->lookup("jjj")->stripeIndex, 3);
  EXPECT_FALSE(clusterIndex->lookup("kkk").has_value());

  // ========== Verify Group 0 (Stripe 0) ==========
  // Stream 0: EMPTY
  // Stream 1: 2 chunks (rows: 50, 50)
  // Stream 2: 3 chunks (rows: 30, 40, 30)
  // Stream 3: 2 chunks (rows: 60, 40)
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[0],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        0, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 0);
    EXPECT_EQ(helper.firstStripe(), 0);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{50, 100}));
    EXPECT_EQ(keyStats.chunkKeys, (std::vector<std::string>{"bbb", "ccc"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        0,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[0],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    // All 4 streams indexed (minAvgChunksPerStream = 0).
    EXPECT_EQ(chunkHelper.streamCount(), 4);

    // Stream 0: EMPTY → 0 chunks.
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{0}));
    EXPECT_TRUE(stream0Stats.chunkRows.empty());

    // Stream 1: 2 chunks (rows: 50, 50)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{50, 100}));

    // Stream 2: 3 chunks (rows: 30, 40, 30)
    auto stream2Stats = chunkHelper.streamStats(2);
    EXPECT_EQ(stream2Stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stream2Stats.chunkRows, (std::vector<uint32_t>{30, 70, 100}));

    // Stream 3: 2 chunks (rows: 60, 40)
    auto stream3Stats = chunkHelper.streamStats(3);
    EXPECT_EQ(stream3Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream3Stats.chunkRows, (std::vector<uint32_t>{60, 100}));
  }

  // ========== Verify Group 1 (Stripe 1) ==========
  // Stream 0: 3 chunks (rows: 40, 60, 50)
  // Stream 1: EMPTY
  // Stream 2: 2 chunks (rows: 80, 70)
  // Stream 3: 1 chunk (rows: 150)
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[1],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        1, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 1);
    EXPECT_EQ(helper.firstStripe(), 1);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{50, 100, 150}));
    EXPECT_EQ(
        keyStats.chunkKeys, (std::vector<std::string>{"ddd", "eee", "fff"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        1,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[1],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    // All 4 streams indexed (minAvgChunksPerStream = 0).
    EXPECT_EQ(chunkHelper.streamCount(), 4);

    // Stream 0: 3 chunks (rows: 40, 60, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{40, 100, 150}));

    // Stream 1: EMPTY → 0 chunks.
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{0}));
    EXPECT_TRUE(stream1Stats.chunkRows.empty());

    // Stream 2: 2 chunks (rows: 80, 70)
    auto stream2Stats = chunkHelper.streamStats(2);
    EXPECT_EQ(stream2Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream2Stats.chunkRows, (std::vector<uint32_t>{80, 150}));

    // Stream 3: 1 chunk (rows: 150)
    auto stream3Stats = chunkHelper.streamStats(3);
    EXPECT_EQ(stream3Stats.chunkCounts, (std::vector<uint32_t>{1}));
    EXPECT_EQ(stream3Stats.chunkRows, (std::vector<uint32_t>{150}));
  }

  // ========== Verify Group 2 (Stripe 2) ==========
  // Stream 0: 2 chunks (rows: 70, 50)
  // Stream 1: 4 chunks (rows: 25, 35, 30, 30)
  // Stream 2: EMPTY
  // Stream 3: 3 chunks (rows: 40, 40, 40)
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[2],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        2, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 2);
    EXPECT_EQ(helper.firstStripe(), 2);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{60, 120}));
    EXPECT_EQ(keyStats.chunkKeys, (std::vector<std::string>{"ggg", "hhh"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        2,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[2],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    // All 4 streams indexed (minAvgChunksPerStream = 0).
    EXPECT_EQ(chunkHelper.streamCount(), 4);

    // Stream 0: 2 chunks (rows: 70, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{70, 120}));

    // Stream 1: 4 chunks (rows: 25, 35, 30, 30)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{4}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{25, 60, 90, 120}));

    // Stream 2: EMPTY → 0 chunks.
    auto stream2Stats = chunkHelper.streamStats(2);
    EXPECT_EQ(stream2Stats.chunkCounts, (std::vector<uint32_t>{0}));
    EXPECT_TRUE(stream2Stats.chunkRows.empty());

    // Stream 3: 3 chunks (rows: 40, 40, 40)
    auto stream3Stats = chunkHelper.streamStats(3);
    EXPECT_EQ(stream3Stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stream3Stats.chunkRows, (std::vector<uint32_t>{40, 80, 120}));
  }

  // ========== Verify Group 3 (Stripe 3) - ALL STREAMS HAVE DATA ==========
  // Stream 0: 2 chunks (rows: 50, 50)
  // Stream 1: 2 chunks (rows: 40, 60)
  // Stream 2: 1 chunk (rows: 100)
  // Stream 3: 2 chunks (rows: 30, 70)
  {
    auto groupBuffer = std::make_unique<MetadataBuffer>(
        *pool_,
        fileIndex.groupMetadataSections[3],
        CompressionType::Uncompressed);
    auto clusterIndexGroup = index::ClusterIndexGroup::create(
        3, clusterIndex->rootIndex(), std::move(groupBuffer));

    index::test::ClusterIndexGroupTestHelper helper(clusterIndexGroup.get());
    EXPECT_EQ(helper.groupIndex(), 3);
    EXPECT_EQ(helper.firstStripe(), 3);
    EXPECT_EQ(helper.stripeCount(), 1);

    // Verify key stream stats
    const auto keyStats = helper.keyStreamStats();
    EXPECT_EQ(keyStats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(keyStats.chunkRows, (std::vector<uint32_t>{50, 100}));
    EXPECT_EQ(keyStats.chunkKeys, (std::vector<std::string>{"iii", "jjj"}));

    // Verify chunk index for stream position data
    auto chunkStats = index::ChunkStatsGroup::create(
        3,
        1,
        std::make_unique<MetadataBuffer>(
            *pool_,
            fileIndex.chunkStatsGroupSections[3],
            CompressionType::Uncompressed));
    ChunkStatsTestHelper chunkHelper(chunkStats.get());
    // All 4 streams indexed (minAvgChunksPerStream = 0).
    EXPECT_EQ(chunkHelper.streamCount(), 4);

    // Stream 0: 2 chunks (rows: 50, 50)
    auto stream0Stats = chunkHelper.streamStats(0);
    EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{50, 100}));

    // Stream 1: 2 chunks (rows: 40, 60)
    auto stream1Stats = chunkHelper.streamStats(1);
    EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{40, 100}));

    // Stream 2: 1 chunk (rows: 100)
    auto stream2Stats = chunkHelper.streamStats(2);
    EXPECT_EQ(stream2Stats.chunkCounts, (std::vector<uint32_t>{1}));
    EXPECT_EQ(stream2Stats.chunkRows, (std::vector<uint32_t>{100}));

    // Stream 3: 2 chunks (rows: 30, 70)
    auto stream3Stats = chunkHelper.streamStats(3);
    EXPECT_EQ(stream3Stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stream3Stats.chunkRows, (std::vector<uint32_t>{30, 100}));
  }
}

TEST_F(ClusterIndexWriterTest, writeRootIndexForEmptyFile) {
  // Test writeRootIndex handles empty file case: writes config only
  // (columns, sort orders) but no stripe keys or stripe index groups.
  ClusterIndexConfig config{
      .columns = {"col1", "col2", "col3"},
      .sortOrders =
          {SortOrder{.ascending = true},
           SortOrder{.ascending = false},
           SortOrder{.ascending = true}},
      .enforceKeyOrder = true,
      .noDuplicateKey = false,
  };

  auto writer = ClusterIndexWriter::create(config, *pool_);

  TestFileIndex fileIndex;
  // Call writeRootIndex with empty stripe group indices for empty file.
  writer->writeRoot({}, writeRootIndexCallback(fileIndex));

  // Verify root index was written.
  EXPECT_FALSE(fileIndex.rootIndexData.empty());

  // Parse and verify the root index content.
  auto* rootIndex = flatbuffers::GetRoot<serialization::ClusterIndex>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(rootIndex, nullptr);

  // Verify stripe keys are empty.
  EXPECT_EQ(rootIndex->stripe_keys()->size(), 0);

  // Verify index columns match config.
  ASSERT_EQ(rootIndex->index_columns()->size(), 3);
  EXPECT_EQ(rootIndex->index_columns()->Get(0)->str(), "col1");
  EXPECT_EQ(rootIndex->index_columns()->Get(1)->str(), "col2");
  EXPECT_EQ(rootIndex->index_columns()->Get(2)->str(), "col3");

  // Verify sort orders match config.
  ASSERT_EQ(rootIndex->sort_orders()->size(), 3);
  auto sortOrder0 = SortOrder::deserialize(
      folly::parseJson(rootIndex->sort_orders()->Get(0)->str()));
  EXPECT_TRUE(sortOrder0.ascending);
  auto sortOrder1 = SortOrder::deserialize(
      folly::parseJson(rootIndex->sort_orders()->Get(1)->str()));
  EXPECT_FALSE(sortOrder1.ascending);
  auto sortOrder2 = SortOrder::deserialize(
      folly::parseJson(rootIndex->sort_orders()->Get(2)->str()));
  EXPECT_TRUE(sortOrder2.ascending);

  // Verify stripe counts are empty.
  EXPECT_EQ(rootIndex->stripe_counts()->size(), 0);
  // Verify stripe indexes are empty.
  EXPECT_EQ(rootIndex->stripe_indexes()->size(), 0);
}

TEST_F(ClusterIndexWriterTest, chunkIndexOnlyWriteAndRead) {
  // Test standalone ChunkStatsWriter without ClusterIndexWriter.
  // This simulates the chunk-index-only mode where enableChunkIndex=true
  // but indexConfig is not set.
  TestFileIndex fileIndex;
  Buffer buffer{*pool_};

  // Stripe 0: 2 streams
  chunkStatsWriter_->newStripe(2);
  auto chunks0 = createChunks(buffer, {{50, 10}, {50, 12}});
  chunkStatsWriter_->addStream(0, chunks0);
  auto chunks1 = createChunks(buffer, {{100, 25}});
  chunkStatsWriter_->addStream(1, chunks1);

  // Write standalone index group (chunk index only, no value index).
  chunkStatsWriter_->writeGroup(
      2, 1, createChunkMetadataSectionCallback(fileIndex));

  // Write standalone root index (no stripe keys, no index columns).
  chunkStatsWriter_->writeRoot(writeChunkRootIndexCallback(fileIndex));

  // Verify data was written.
  ASSERT_EQ(fileIndex.chunkStatsGroupSections.size(), 1);
  ASSERT_FALSE(fileIndex.chunkRootIndexData.empty());

  // Verify ChunkStatsGroup can be loaded and used for chunk seeking.
  auto chunkStats = index::ChunkStatsGroup::create(
      0,
      1,
      std::make_unique<MetadataBuffer>(
          *pool_,
          fileIndex.chunkStatsGroupSections[0],
          CompressionType::Uncompressed));

  // Lookup via StreamIndex (public API).
  // Stream 0: chunks at rows {50, 50} -> accumulated {50, 100}
  // Stream 0: 2 chunks of {50 rows, 10 bytes} and {50 rows, 12 bytes}.
  // Total stream size = 10 + 12 = 22.
  auto stream0 = chunkStats->createStreamIndex(0, 0, 22);
  ASSERT_NE(stream0, nullptr);

  auto result00 = stream0->lookupChunk(0);
  EXPECT_EQ(result00.chunkOffset, 0);
  EXPECT_EQ(result00.chunkSize, 10);
  EXPECT_EQ(result00.rowOffset, 0);

  auto result01 = stream0->lookupChunk(50);
  EXPECT_EQ(result01.chunkOffset, 10);
  EXPECT_EQ(result01.chunkSize, 12);
  EXPECT_EQ(result01.rowOffset, 50);

  // Stream 1: 1 chunk → createStreamIndex returns nullptr.
  EXPECT_EQ(chunkStats->createStreamIndex(0, 1, 25), nullptr);
}

} // namespace facebook::nimble::test
