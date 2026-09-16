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

#include "velox/dwio/nimble/tablet/ChunkStatsWriter.h"

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/ChunkStats.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

namespace facebook::nimble::test {

using index::test::ChunkSpec;
using index::test::ChunkStatsTestHelper;
using index::test::createChunks;

// Holds all the index data written during a test.
struct TestChunkFileIndex {
  std::vector<std::string> groupMetadataSections;
  std::string rootIndexData;
};

class ChunkStatsWriterTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  static auto createMetadataSectionCallback(TestChunkFileIndex& fileIndex) {
    return [&fileIndex](std::string_view metadata) -> MetadataSection {
      fileIndex.groupMetadataSections.emplace_back(metadata);
      return MetadataSection(0, metadata.size(), CompressionType::Uncompressed);
    };
  }

  static auto writeRootCallback(TestChunkFileIndex& fileIndex) {
    return [&fileIndex](const std::string& name, std::string_view content) {
      EXPECT_EQ(name, nimble::kChunkStatsSection);
      fileIndex.rootIndexData = std::string(content);
    };
  }

  // Creates a ChunkStatsGroup reader from a serialized group metadata section.
  std::shared_ptr<index::ChunkStatsGroup> loadChunkStats(
      const std::string& groupData,
      uint32_t firstStripe,
      uint32_t stripeCount) {
    auto inputBuffer =
        velox::AlignedBuffer::allocate<char>(groupData.size(), pool_.get());
    std::memcpy(
        inputBuffer->asMutable<char>(), groupData.data(), groupData.size());
    auto buffer = std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
        std::move(inputBuffer), CompressionType::Uncompressed, pool_.get()));
    return index::ChunkStatsGroup::create(
        firstStripe, stripeCount, std::move(buffer));
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(ChunkStatsWriterTest, singleStripe) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // 1 stripe, 2 streams.
  // Stream 0: 3 chunks (rows: 30, 45, 25; sizes: 10, 15, 8)
  // Stream 1: 2 chunks (rows: 60, 40; sizes: 20, 12)
  writer.newStripe(2);
  auto chunks0 = createChunks(buffer, {{30, 10}, {45, 15}, {25, 8}});
  auto chunks1 = createChunks(buffer, {{60, 20}, {40, 12}});
  writer.addStream(0, chunks0);
  writer.addStream(1, chunks1);

  writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex));

  // Verify root index (ChunkStats flatbuffer).
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  auto* rootChunkStats = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(rootChunkStats, nullptr);
  ASSERT_NE(rootChunkStats->stripe_indexes(), nullptr);
  EXPECT_EQ(rootChunkStats->stripe_indexes()->size(), 1);

  // Load as ChunkStatsGroup reader.
  auto chunkStats = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 1);

  ChunkStatsTestHelper helper(chunkStats.get());
  EXPECT_EQ(helper.firstStripe(), 0);
  EXPECT_EQ(helper.stripeCount(), 1);
  EXPECT_EQ(helper.streamCount(), 2);

  // Stream 0: 3 chunks (rows: 30, 45, 25; sizes: 10, 15, 8)
  auto stream0Stats = helper.streamStats(0);
  EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{3}));
  EXPECT_EQ(stream0Stats.chunkRows, (std::vector<uint32_t>{30, 75, 100}));
  EXPECT_EQ(stream0Stats.chunkOffsets, (std::vector<uint32_t>{0, 10, 25}));

  // Stream 1: 2 chunks (rows: 60, 40; sizes: 20, 12)
  auto stream1Stats = helper.streamStats(1);
  EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{2}));
  EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{60, 100}));
  EXPECT_EQ(stream1Stats.chunkOffsets, (std::vector<uint32_t>{0, 20}));

  // Lookup via StreamIndex (public API).
  // Stream 0: 3 chunks, sizes {10, 15, 8}, total = 33.
  auto stream0 = chunkStats->createStreamIndex(0, 0, 33);
  ASSERT_NE(stream0, nullptr);

  auto r00 = stream0->lookupChunk(0);
  EXPECT_EQ(r00.chunkOffset, 0);
  EXPECT_EQ(r00.chunkSize, 10);
  EXPECT_EQ(r00.rowOffset, 0);

  auto r01 = stream0->lookupChunk(30);
  EXPECT_EQ(r01.chunkOffset, 10);
  EXPECT_EQ(r01.chunkSize, 15);
  EXPECT_EQ(r01.rowOffset, 30);

  auto r02 = stream0->lookupChunk(75);
  EXPECT_EQ(r02.chunkOffset, 25);
  EXPECT_EQ(r02.chunkSize, 8);
  EXPECT_EQ(r02.rowOffset, 75);

  // Stream 1: 2 chunks, sizes {20, 12}, total = 32.
  auto stream1 = chunkStats->createStreamIndex(0, 1, 32);
  ASSERT_NE(stream1, nullptr);

  auto r10 = stream1->lookupChunk(0);
  EXPECT_EQ(r10.chunkOffset, 0);
  EXPECT_EQ(r10.chunkSize, 20);
  EXPECT_EQ(r10.rowOffset, 0);

  auto r11 = stream1->lookupChunk(60);
  EXPECT_EQ(r11.chunkOffset, 20);
  EXPECT_EQ(r11.chunkSize, 12);
  EXPECT_EQ(r11.rowOffset, 60);
}

TEST_F(ChunkStatsWriterTest, perChunkNullCounts) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // 1 stripe, 2 streams. ChunkSpec = {rowCount, size, nullCount}.
  // Stream 0: 3 chunks with null counts 3, 0, 5.
  // Stream 1: 2 chunks with null counts 0, 40 (a fully-null chunk).
  writer.newStripe(2);
  auto chunks0 = createChunks(buffer, {{30, 10, 3}, {45, 15, 0}, {25, 8, 5}});
  auto chunks1 = createChunks(buffer, {{60, 20, 0}, {40, 12, 40}});
  writer.addStream(0, chunks0);
  writer.addStream(1, chunks1);

  writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex));

  auto chunkStats = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 1);

  // Null counts round-trip through the flatbuffer in flattened order.
  ChunkStatsTestHelper helper(chunkStats.get());
  EXPECT_EQ(
      helper.streamStats(0).chunkNullCounts, (std::vector<uint32_t>{3, 0, 5}));
  EXPECT_EQ(
      helper.streamStats(1).chunkNullCounts, (std::vector<uint32_t>{0, 40}));

  // Public reader accessor: lookupChunk() yields the absolute chunk index,
  // which chunkNullCount() maps to the per-chunk null statistic.
  auto stream0 = chunkStats->createStreamIndex(0, 0, 33);
  ASSERT_NE(stream0, nullptr);
  EXPECT_EQ(stream0->chunkNullCount(stream0->lookupChunk(0).chunkIndex), 3);
  EXPECT_EQ(stream0->chunkNullCount(stream0->lookupChunk(30).chunkIndex), 0);
  EXPECT_EQ(stream0->chunkNullCount(stream0->lookupChunk(75).chunkIndex), 5);

  auto stream1 = chunkStats->createStreamIndex(0, 1, 32);
  ASSERT_NE(stream1, nullptr);
  EXPECT_EQ(stream1->chunkNullCount(stream1->lookupChunk(60).chunkIndex), 40);
}

TEST_F(ChunkStatsWriterTest, multipleStripesInSingleGroup) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // Stripe 0: 2 streams
  // Stream 0: 2 chunks (rows: 50, 50; sizes: 10, 12)
  // Stream 1: 1 chunk (rows: 100; size: 25)
  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.addStream(1, createChunks(buffer, {{100, 25}}));

  // Stripe 1: 2 streams
  // Stream 0: 3 chunks (rows: 40, 60, 50; sizes: 8, 14, 11)
  // Stream 1: 2 chunks (rows: 80, 70; sizes: 18, 15)
  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{40, 8}, {60, 14}, {50, 11}}));
  writer.addStream(1, createChunks(buffer, {{80, 18}, {70, 15}}));

  writer.writeGroup(2, 2, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex));

  // Load and verify.
  auto chunkStats = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 2);

  ChunkStatsTestHelper helper(chunkStats.get());
  EXPECT_EQ(helper.stripeCount(), 2);
  EXPECT_EQ(helper.streamCount(), 2);

  // Stream 0: stripe 0 has 2 chunks, stripe 1 has 3 chunks.
  // Accumulated chunk counts: {2, 5}
  auto stream0Stats = helper.streamStats(0);
  EXPECT_EQ(stream0Stats.chunkCounts, (std::vector<uint32_t>{2, 5}));
  // Stripe 0 rows: 50, 100; Stripe 1 rows: 40, 100, 150
  EXPECT_EQ(
      stream0Stats.chunkRows, (std::vector<uint32_t>{50, 100, 40, 100, 150}));
  // Stripe 0 offsets: 0, 10; Stripe 1 offsets: 0, 8, 22
  EXPECT_EQ(
      stream0Stats.chunkOffsets, (std::vector<uint32_t>{0, 10, 0, 8, 22}));

  // Stream 1: stripe 0 has 1 chunk, stripe 1 has 2 chunks.
  // Accumulated chunk counts: {1, 3}
  auto stream1Stats = helper.streamStats(1);
  EXPECT_EQ(stream1Stats.chunkCounts, (std::vector<uint32_t>{1, 3}));
  // Stripe 0 rows: 100; Stripe 1 rows: 80, 150
  EXPECT_EQ(stream1Stats.chunkRows, (std::vector<uint32_t>{100, 80, 150}));
  EXPECT_EQ(stream1Stats.chunkOffsets, (std::vector<uint32_t>{0, 0, 18}));
}

TEST_F(ChunkStatsWriterTest, multipleStripeGroups) {
  ChunkStatsWriter writer(*pool_, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // Group 0: Stripe 0
  // Stream 0: 2 chunks (rows: 50, 50; sizes: 10, 12)
  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  // Group 1: Stripe 1
  // Stream 0: 3 chunks (rows: 40, 60, 50; sizes: 8, 14, 11)
  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{40, 8}, {60, 14}, {50, 11}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  // Group 2: Stripe 2
  // Stream 0: 1 chunk (rows: 200; size: 50)
  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{200, 50}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  writer.writeRoot(writeRootCallback(fileIndex));

  // All 3 groups are written (threshold=0, no skipping).
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 3);

  auto* rootChunkStats = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(rootChunkStats, nullptr);
  ASSERT_NE(rootChunkStats->stripe_indexes(), nullptr);
  EXPECT_EQ(rootChunkStats->stripe_indexes()->size(), 3);

  // Verify group 0 (stripe 0).
  {
    auto ci = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 1);
    ChunkStatsTestHelper helper(ci.get());
    EXPECT_EQ(helper.stripeCount(), 1);
    EXPECT_EQ(helper.streamCount(), 1);
    auto stats = helper.streamStats(0);
    EXPECT_EQ(stats.chunkCounts, (std::vector<uint32_t>{2}));
    EXPECT_EQ(stats.chunkRows, (std::vector<uint32_t>{50, 100}));
    EXPECT_EQ(stats.chunkOffsets, (std::vector<uint32_t>{0, 10}));
  }

  // Verify group 1 (stripe 1).
  {
    auto ci = loadChunkStats(fileIndex.groupMetadataSections[1], 1, 1);
    ChunkStatsTestHelper helper(ci.get());
    EXPECT_EQ(helper.stripeCount(), 1);
    EXPECT_EQ(helper.streamCount(), 1);
    auto stats = helper.streamStats(0);
    EXPECT_EQ(stats.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(stats.chunkRows, (std::vector<uint32_t>{40, 100, 150}));
    EXPECT_EQ(stats.chunkOffsets, (std::vector<uint32_t>{0, 8, 22}));
  }

  // Verify group 2 (stripe 2): 1 chunk, createStreamIndex returns nullptr.
  {
    auto ci = loadChunkStats(fileIndex.groupMetadataSections[2], 2, 1);
    ChunkStatsTestHelper helper(ci.get());
    EXPECT_EQ(helper.stripeCount(), 1);
    EXPECT_EQ(helper.streamCount(), 1);
    auto stats = helper.streamStats(0);
    EXPECT_EQ(stats.chunkCounts, (std::vector<uint32_t>{1}));
    EXPECT_EQ(stats.chunkRows, (std::vector<uint32_t>{200}));
    EXPECT_EQ(stats.chunkOffsets, (std::vector<uint32_t>{0}));
    // Single-chunk stream returns nullptr.
    EXPECT_EQ(ci->createStreamIndex(2, 0, 30), nullptr);
  }
}

TEST_F(ChunkStatsWriterTest, emptyStream) {
  ChunkStatsWriter writer(*pool_, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // 1 stripe, 3 streams. Stream 1 is empty (no addStream call).
  writer.newStripe(3);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  // Stream 1 is empty.
  writer.addStream(2, createChunks(buffer, {{100, 25}}));

  writer.writeGroup(3, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex));

  auto chunkStats = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 1);

  ChunkStatsTestHelper helper(chunkStats.get());
  // All 3 streams are indexed (dense layout).
  EXPECT_EQ(helper.streamCount(), 3);

  // Stream 0: 2 chunks.
  auto stream0 = helper.streamStats(0);
  EXPECT_EQ(stream0.chunkCounts, (std::vector<uint32_t>{2}));
  EXPECT_EQ(stream0.chunkRows, (std::vector<uint32_t>{50, 100}));
  EXPECT_EQ(stream0.chunkOffsets, (std::vector<uint32_t>{0, 10}));

  // Stream 1: 0 chunks (empty).
  auto stream1 = helper.streamStats(1);
  EXPECT_EQ(stream1.chunkCounts, (std::vector<uint32_t>{0}));
  EXPECT_TRUE(stream1.chunkRows.empty());

  // Stream 2: 1 chunk.
  auto stream2 = helper.streamStats(2);
  EXPECT_EQ(stream2.chunkCounts, (std::vector<uint32_t>{1}));
  EXPECT_EQ(stream2.chunkRows, (std::vector<uint32_t>{100}));
  EXPECT_EQ(stream2.chunkOffsets, (std::vector<uint32_t>{0}));

  // createStreamIndex returns nullptr for streams with ≤1 chunk.
  EXPECT_EQ(chunkStats->createStreamIndex(0, 1, 0), nullptr);
  EXPECT_EQ(chunkStats->createStreamIndex(0, 2, 25), nullptr);
  // streamId out of range returns nullptr.
  EXPECT_EQ(chunkStats->createStreamIndex(0, 3, 0), nullptr);
}

TEST_F(ChunkStatsWriterTest, emptyFileNoStripeGroups) {
  ChunkStatsWriter writer(*pool_);
  TestChunkFileIndex fileIndex;

  // No stripes written — writeGroup() is never called.
  writer.writeRoot(writeRootCallback(fileIndex));

  // No chunk stats section should be written.
  EXPECT_TRUE(fileIndex.rootIndexData.empty());
}

TEST_F(ChunkStatsWriterTest, finalization) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 10}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex));

  // After finalization, all mutation methods should throw.
  NIMBLE_ASSERT_THROW(
      writer.newStripe(1), "ChunkStatsWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer.addStream(0, createChunks(buffer, {{10, 5}})),
      "ChunkStatsWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex)),
      "ChunkStatsWriter has been finalized");
  NIMBLE_ASSERT_THROW(
      writer.writeRoot(writeRootCallback(fileIndex)),
      "ChunkStatsWriter has been finalized");
}

TEST_F(ChunkStatsWriterTest, addStreamIndexValidation) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};

  // addStream before newStripe should fail.
  NIMBLE_ASSERT_THROW(writer.addStream(0, createChunks(buffer, {{10, 5}})), "");

  // Out-of-range stream index should fail.
  writer.newStripe(2);
  NIMBLE_ASSERT_THROW(writer.addStream(2, createChunks(buffer, {{10, 5}})), "");
}

TEST_F(ChunkStatsWriterTest, multipleStripesInMultipleGroups) {
  ChunkStatsWriter writer(*pool_);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  // Group 0: 2 stripes, 2 streams.
  // Stripe 0:
  //   Stream 0: 2 chunks (rows: 50, 50; sizes: 10, 12)
  //   Stream 1: 1 chunk (rows: 100; size: 25)
  // Stripe 1:
  //   Stream 0: 1 chunk (rows: 80; size: 20)
  //   Stream 1: 2 chunks (rows: 40, 40; sizes: 8, 9)
  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.addStream(1, createChunks(buffer, {{100, 25}}));
  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{80, 20}}));
  writer.addStream(1, createChunks(buffer, {{40, 8}, {40, 9}}));
  writer.writeGroup(2, 2, createMetadataSectionCallback(fileIndex));

  // Group 1: 1 stripe, 2 streams.
  // Stripe 2:
  //   Stream 0: 3 chunks (rows: 30, 30, 40; sizes: 6, 7, 9)
  //   Stream 1: 1 chunk (rows: 100; size: 30)
  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{30, 6}, {30, 7}, {40, 9}}));
  writer.addStream(1, createChunks(buffer, {{100, 30}}));
  writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

  writer.writeRoot(writeRootCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 2);

  auto* rootChunkStats = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(rootChunkStats, nullptr);
  ASSERT_NE(rootChunkStats->stripe_indexes(), nullptr);
  EXPECT_EQ(rootChunkStats->stripe_indexes()->size(), 2);

  // Verify group 0 (2 stripes, firstStripe=0).
  {
    auto ci = loadChunkStats(fileIndex.groupMetadataSections[0], 0, 2);
    ChunkStatsTestHelper helper(ci.get());
    EXPECT_EQ(helper.stripeCount(), 2);

    auto s0 = helper.streamStats(0);
    EXPECT_EQ(s0.chunkCounts, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(s0.chunkRows, (std::vector<uint32_t>{50, 100, 80}));
    EXPECT_EQ(s0.chunkOffsets, (std::vector<uint32_t>{0, 10, 0}));

    auto s1 = helper.streamStats(1);
    EXPECT_EQ(s1.chunkCounts, (std::vector<uint32_t>{1, 3}));
    EXPECT_EQ(s1.chunkRows, (std::vector<uint32_t>{100, 40, 80}));
    EXPECT_EQ(s1.chunkOffsets, (std::vector<uint32_t>{0, 0, 8}));
  }

  // Verify group 1 (1 stripe, firstStripe=2).
  {
    auto ci = loadChunkStats(fileIndex.groupMetadataSections[1], 2, 1);
    ChunkStatsTestHelper helper(ci.get());
    EXPECT_EQ(helper.stripeCount(), 1);
    EXPECT_EQ(helper.streamCount(), 2);

    auto s0 = helper.streamStats(0);
    EXPECT_EQ(s0.chunkCounts, (std::vector<uint32_t>{3}));
    EXPECT_EQ(s0.chunkRows, (std::vector<uint32_t>{30, 60, 100}));
    EXPECT_EQ(s0.chunkOffsets, (std::vector<uint32_t>{0, 6, 13}));

    // Stream 1: 1 chunk (createStreamIndex returns nullptr).
    auto s1 = helper.streamStats(1);
    EXPECT_EQ(s1.chunkCounts, (std::vector<uint32_t>{1}));
    EXPECT_EQ(s1.chunkRows, (std::vector<uint32_t>{100}));
    EXPECT_EQ(s1.chunkOffsets, (std::vector<uint32_t>{0}));
    EXPECT_EQ(ci->createStreamIndex(2, 1, 20), nullptr);
  }
}

TEST_F(ChunkStatsWriterTest, minAvgChunksPerStream) {
  // Test that minAvgChunksPerStream controls group-level skipping.
  // Setup: 3 groups with different average chunks per stream.
  //   Group 0: 1 stripe, 2 streams. Stream 0: 3 chunks, Stream 1: 1 chunk.
  //            Total=4, avg=2.0
  //   Group 1: 1 stripe, 2 streams. Stream 0: 2 chunks, Stream 1: 1 chunk.
  //            Total=3, avg=1.5
  //   Group 2: 1 stripe, 2 streams. Stream 0: 1 chunk, Stream 1: 1 chunk.
  //            Total=2, avg=1.0

  struct TestParam {
    float threshold;
    // Expected number of groups that are actually written (callback invoked).
    uint32_t expectedWrittenGroups;
    // Expected total root index entries (0 when all groups are skipped,
    // otherwise equals the number of stripe groups including skipped ones).
    uint32_t expectedRootEntries;
    // Which groups are skipped (size=0 in root index).
    std::vector<bool> expectedSkipped;

    std::string debugString() const {
      return fmt::format(
          "threshold {}, expectedWrittenGroups {}, expectedRootEntries {}, expectedSkipped [{}]",
          threshold,
          expectedWrittenGroups,
          expectedRootEntries,
          fmt::join(expectedSkipped, ", "));
    }
  };

  std::vector<TestParam> testSettings = {
      // threshold=0: no skipping, all 3 groups written.
      {0.0f, 3, 3, {false, false, false}},
      // threshold=1.0: avg must be >= 1.0. All groups meet threshold.
      {1.0f, 3, 3, {false, false, false}},
      // threshold=1.5: group 2 (avg=1.0) is skipped.
      {1.5f, 2, 3, {false, false, true}},
      // threshold=2.0: groups 1 (avg=1.5) and 2 (avg=1.0) are skipped.
      {2.0f, 1, 3, {false, true, true}},
      // threshold=3.0: all groups skipped — no chunk stats section written.
      {3.0f, 0, 0, {true, true, true}},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    ChunkStatsWriter writer(*pool_, testData.threshold);
    Buffer buffer{*pool_};
    TestChunkFileIndex fileIndex;

    // Group 0: avg = 4/2 = 2.0
    writer.newStripe(2);
    writer.addStream(0, createChunks(buffer, {{30, 6}, {30, 7}, {40, 9}}));
    writer.addStream(1, createChunks(buffer, {{100, 30}}));
    writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

    // Group 1: avg = 3/2 = 1.5
    writer.newStripe(2);
    writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
    writer.addStream(1, createChunks(buffer, {{100, 25}}));
    writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

    // Group 2: avg = 2/2 = 1.0
    writer.newStripe(2);
    writer.addStream(0, createChunks(buffer, {{80, 20}}));
    writer.addStream(1, createChunks(buffer, {{120, 35}}));
    writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

    writer.writeRoot(writeRootCallback(fileIndex));

    ASSERT_EQ(
        fileIndex.groupMetadataSections.size(), testData.expectedWrittenGroups);

    if (testData.expectedRootEntries == 0) {
      // All groups were skipped — no chunk stats section written.
      EXPECT_TRUE(fileIndex.rootIndexData.empty());
      continue;
    }

    ASSERT_FALSE(fileIndex.rootIndexData.empty());

    auto* rootChunkStats = flatbuffers::GetRoot<serialization::ChunkStats>(
        fileIndex.rootIndexData.data());
    ASSERT_NE(rootChunkStats, nullptr);
    ASSERT_NE(rootChunkStats->stripe_indexes(), nullptr);
    EXPECT_EQ(
        rootChunkStats->stripe_indexes()->size(), testData.expectedRootEntries);

    for (uint32_t i = 0; i < testData.expectedRootEntries; ++i) {
      auto* entry = rootChunkStats->stripe_indexes()->Get(i);
      if (testData.expectedSkipped[i]) {
        EXPECT_EQ(entry->size(), 0) << "Group " << i << " should be skipped";
      } else {
        EXPECT_GT(entry->size(), 0) << "Group " << i << " should be written";
      }
    }
  }
}

TEST_F(ChunkStatsWriterTest, uncompressedSizeRoundtrip) {
  ChunkStatsWriter writer(*pool_, /*minAvgChunksPerStream=*/0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  uint64_t nextOffset = 1000;
  auto compressedCallback =
      [&fileIndex, &nextOffset](std::string_view metadata) -> MetadataSection {
    fileIndex.groupMetadataSections.emplace_back(metadata);
    const auto uncompressedSize = static_cast<uint32_t>(metadata.size());
    const auto compressedSize = uncompressedSize / 2;
    auto offset = nextOffset;
    nextOffset += compressedSize;
    return MetadataSection(
        offset, compressedSize, CompressionType::Zstd, uncompressedSize);
  };

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.addStream(1, createChunks(buffer, {{60, 20}, {40, 12}}));
  writer.writeGroup(2, 1, compressedCallback);

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{80, 20}, {20, 5}}));
  writer.addStream(1, createChunks(buffer, {{50, 15}, {50, 18}}));
  writer.writeGroup(2, 1, compressedCallback);

  writer.writeRoot(writeRootCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 2);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  auto* root = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(root, nullptr);
  ASSERT_NE(root->stripe_indexes(), nullptr);
  ASSERT_EQ(root->stripe_indexes()->size(), 2);

  for (uint32_t i = 0; i < 2; ++i) {
    auto* entry = root->stripe_indexes()->Get(i);
    EXPECT_EQ(entry->compression_type(), serialization::CompressionType_Zstd)
        << "Group " << i;
    EXPECT_GT(entry->uncompressed_size(), 0) << "Group " << i;
    EXPECT_GT(entry->uncompressed_size(), entry->size()) << "Group " << i;
  }

  auto rootBuffer = velox::AlignedBuffer::allocate<char>(
      fileIndex.rootIndexData.size(), pool_.get());
  std::memcpy(
      rootBuffer->asMutable<char>(),
      fileIndex.rootIndexData.data(),
      fileIndex.rootIndexData.size());
  Section rootSection{MetadataBuffer(
      MetadataBuffer::decompress(
          std::move(rootBuffer), CompressionType::Uncompressed, pool_.get()))};

  auto chunkStats = index::ChunkStats::create(std::move(rootSection));
  ASSERT_NE(chunkStats, nullptr);
  ASSERT_EQ(chunkStats->numGroups(), 2);

  for (uint32_t i = 0; i < 2; ++i) {
    const auto& section = chunkStats->groupMetadata(i);
    EXPECT_EQ(section.compressionType(), CompressionType::Zstd)
        << "Group " << i;
    EXPECT_TRUE(section.uncompressedSize().has_value()) << "Group " << i;
    EXPECT_GT(section.uncompressedSize().value(), section.size())
        << "Group " << i;
  }
}

// Verifies that index metadata without uncompressed_size (written before
// D108464890) is handled gracefully: the reader returns nullopt instead of
// throwing, restoring backward compatibility with old files.
TEST_F(ChunkStatsWriterTest, missingUncompressedSizeBackwardCompat) {
  flatbuffers::FlatBufferBuilder builder(256);

  // Simulate a pre-D108464890 FlatBuffer: no uncompressed_size field set.
  auto section0 = serialization::CreateMetadataSection(
      builder,
      /*offset=*/100,
      /*size=*/200,
      serialization::CompressionType_Zstd);

  auto section1 = serialization::CreateMetadataSection(
      builder,
      /*offset=*/300,
      /*size=*/150,
      serialization::CompressionType_Uncompressed);

  std::vector<flatbuffers::Offset<serialization::MetadataSection>> sections = {
      section0, section1};
  auto stripeIndexes = builder.CreateVector(sections);
  builder.Finish(serialization::CreateChunkStats(builder, stripeIndexes));

  auto rootBuffer =
      velox::AlignedBuffer::allocate<char>(builder.GetSize(), pool_.get());
  std::memcpy(
      rootBuffer->asMutable<char>(),
      builder.GetBufferPointer(),
      builder.GetSize());
  Section rootSection{MetadataBuffer(
      MetadataBuffer::decompress(
          std::move(rootBuffer), CompressionType::Uncompressed, pool_.get()))};

  auto chunkStats = index::ChunkStats::create(std::move(rootSection));
  ASSERT_NE(chunkStats, nullptr);
  ASSERT_EQ(chunkStats->numGroups(), 2);

  const auto& zstdSection = chunkStats->groupMetadata(0);
  EXPECT_EQ(zstdSection.compressionType(), CompressionType::Zstd);
  EXPECT_FALSE(zstdSection.uncompressedSize().has_value());
  EXPECT_EQ(zstdSection.size(), 200);

  const auto& uncompressedSection = chunkStats->groupMetadata(1);
  EXPECT_EQ(
      uncompressedSection.compressionType(), CompressionType::Uncompressed);
  EXPECT_FALSE(uncompressedSection.uncompressedSize().has_value());
  EXPECT_EQ(uncompressedSection.size(), 150);
}

// Verifies that uncompressed sections get uncompressedSize == size in the
// FlatBuffer, and the reader correctly recovers it.
TEST_F(ChunkStatsWriterTest, uncompressedSizeForUncompressedSections) {
  ChunkStatsWriter writer(*pool_, /*minAvgChunksPerStream=*/0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.addStream(1, createChunks(buffer, {{60, 20}, {40, 12}}));
  writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

  writer.writeRoot(writeRootCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  auto* root = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(root, nullptr);
  ASSERT_NE(root->stripe_indexes(), nullptr);
  ASSERT_EQ(root->stripe_indexes()->size(), 1);

  auto* entry = root->stripe_indexes()->Get(0);
  EXPECT_EQ(
      entry->compression_type(), serialization::CompressionType_Uncompressed);
  EXPECT_EQ(entry->uncompressed_size(), entry->size());

  auto rootBuffer = velox::AlignedBuffer::allocate<char>(
      fileIndex.rootIndexData.size(), pool_.get());
  std::memcpy(
      rootBuffer->asMutable<char>(),
      fileIndex.rootIndexData.data(),
      fileIndex.rootIndexData.size());
  Section rootSection{MetadataBuffer(
      MetadataBuffer::decompress(
          std::move(rootBuffer), CompressionType::Uncompressed, pool_.get()))};

  auto chunkStats = index::ChunkStats::create(std::move(rootSection));
  ASSERT_NE(chunkStats, nullptr);
  ASSERT_EQ(chunkStats->numGroups(), 1);

  const auto& section = chunkStats->groupMetadata(0);
  EXPECT_EQ(section.compressionType(), CompressionType::Uncompressed);
  EXPECT_TRUE(section.uncompressedSize().has_value());
  EXPECT_EQ(section.uncompressedSize().value(), section.size());
}

} // namespace facebook::nimble::test
