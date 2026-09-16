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
#include <limits>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestBase.h"

namespace facebook::nimble::index::test {

class ChunkStatsTest : public ClusterIndexTestBase {};

TEST_F(ChunkStatsTest, lookupChunkWithRowId) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Setup: 2 stripes, each with 2 streams with different chunk configurations
  std::vector<Stripe> stripes = {
      {.streams =
           {// Stream 0: 3 chunks at rows {100, 250, 400}, offsets {0, 50, 120}
            {.numChunks = 3,
             .chunkRows = {100, 250, 400},
             .chunkOffsets = {0, 50, 120}},
            // Stream 1: 2 chunks at rows {150, 300}, offsets {0, 80}
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 80}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {400}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      {.streams =
           {// Stream 0: 2 chunks at rows {200, 350}, offsets {0, 60}
            {.numChunks = 2, .chunkRows = {200, 350}, .chunkOffsets = {0, 60}},
            // Stream 1: 3 chunks at rows {100, 200, 350}, offsets {0, 40, 90}
            {.numChunks = 3,
             .chunkRows = {100, 200, 350},
             .chunkOffsets = {0, 40, 90}}},
       .keyStream = {
           .streamOffset = 100,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {350}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}}};
  std::vector<int> stripeGroups = {2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  struct {
    uint32_t stripeIndex;
    uint32_t streamId;
    uint32_t rowId;
    uint32_t expectedStreamOffset;
    uint32_t expectedRowOffset;
  } testCases[] = {
      // Stripe 0, Stream 0: chunks at rows {100, 250, 400}, offsets {0, 50,
      // 120}
      // Accumulated rows: {100, 350, 750}
      // Row 0 - first row of first chunk
      {0, 0, 0, 0, 0},
      // Row in first chunk [0, 100)
      {0, 0, 50, 0, 0},
      {0, 0, 99, 0, 0},
      // Row at first chunk boundary (100) - second chunk
      {0, 0, 100, 50, 100},
      // Row in second chunk [100, 350)
      {0, 0, 150, 50, 100},
      {0, 0, 249, 50, 100},
      {0, 0, 349, 50, 100},
      // Row at second chunk boundary (350) - third chunk
      {0, 0, 350, 120, 350},
      // Row in third chunk [350, 750)
      {0, 0, 351, 120, 350},
      {0, 0, 399, 120, 350},
      {0, 0, 749, 120, 350},

      // Stripe 0, Stream 1: chunks at rows {150, 300}, offsets {0, 80}
      // Accumulated rows: {150, 450}
      // Row 0 - first row of first chunk
      {0, 1, 0, 0, 0},
      // Row in first chunk [0, 150)
      {0, 1, 75, 0, 0},
      {0, 1, 149, 0, 0},
      // Row at first chunk boundary (150) - second chunk
      {0, 1, 150, 80, 150},
      // Row in second chunk [150, 450)
      {0, 1, 200, 80, 150},
      {0, 1, 299, 80, 150},
      {0, 1, 449, 80, 150},

      // Stripe 1, Stream 0: chunks at rows {200, 350}, offsets {0, 60}
      // Accumulated rows: {200, 550}
      // Row 0 - first row of first chunk
      {1, 0, 0, 0, 0},
      // Row in first chunk [0, 200)
      {1, 0, 100, 0, 0},
      {1, 0, 199, 0, 0},
      // Row at first chunk boundary (200) - second chunk
      {1, 0, 200, 60, 200},
      // Row in second chunk [200, 550)
      {1, 0, 275, 60, 200},
      {1, 0, 400, 60, 200},
      {1, 0, 549, 60, 200},

      // Stripe 1, Stream 1: chunks at rows {100, 200, 350}, offsets {0, 40, 90}
      // Accumulated rows: {100, 300, 650}
      // Row 0 - first row of first chunk
      {1, 1, 0, 0, 0},
      // Row in first chunk [0, 100)
      {1, 1, 50, 0, 0},
      {1, 1, 99, 0, 0},
      // Row at first chunk boundary (100) - second chunk
      {1, 1, 100, 40, 100},
      // Row in second chunk [100, 300)
      {1, 1, 150, 40, 100},
      {1, 1, 200, 40, 100},
      {1, 1, 299, 40, 100},
      // Row at second chunk boundary (300) - third chunk
      {1, 1, 300, 90, 300},
      // Row in third chunk [300, 650)
      {1, 1, 400, 90, 300},
      {1, 1, 500, 90, 300},
      {1, 1, 649, 90, 300}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "stripeIndex {} streamId {} rowId {}",
            testCase.stripeIndex,
            testCase.streamId,
            testCase.rowId));
    auto streamIndex = chunkStats->createStreamIndex(
        testCase.stripeIndex, testCase.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);
    auto result = streamIndex->lookupChunk(testCase.rowId);
    EXPECT_EQ(result.chunkOffset, testCase.expectedStreamOffset);
    EXPECT_EQ(result.rowOffset, testCase.expectedRowOffset);
  }

  // Row at/after last chunk boundary — throws
  auto s00 = chunkStats->createStreamIndex(0, 0, /*streamSize=*/1000);
  NIMBLE_ASSERT_THROW(s00->lookupChunk(750), "beyond the last chunk");
  NIMBLE_ASSERT_THROW(s00->lookupChunk(850), "beyond the last chunk");

  auto s01 = chunkStats->createStreamIndex(0, 1, /*streamSize=*/1000);
  NIMBLE_ASSERT_THROW(s01->lookupChunk(450), "beyond the last chunk");

  auto s10 = chunkStats->createStreamIndex(1, 0, /*streamSize=*/1000);
  NIMBLE_ASSERT_THROW(s10->lookupChunk(550), "beyond the last chunk");

  auto s11 = chunkStats->createStreamIndex(1, 1, /*streamSize=*/1000);
  NIMBLE_ASSERT_THROW(s11->lookupChunk(650), "beyond the last chunk");
}

TEST_F(ChunkStatsTest, lookupChunkPopulatesChunkIndex) {
  // Single stripe, two streams. chunkIndex is the absolute position in the
  // stripe's chunkRows FlatBuffers array — i.e., it is offset by the number
  // of chunks belonging to earlier streams in the stripe.
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  std::vector<Stripe> stripes = {
      {.streams =
           {// Stream 0: 3 chunks at rows {100, 250, 400}, offsets {0, 50, 120}
            {.numChunks = 3,
             .chunkRows = {100, 250, 400},
             .chunkOffsets = {0, 50, 120}},
            // Stream 1: 2 chunks at rows {150, 300}, offsets {0, 80}
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 80}}},
       .keyStream = {
           .streamOffset = 0,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {400}, .chunkOffsets = {0}},
           .chunkKeys = {"bbb"}}}};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  struct {
    uint32_t streamId;
    uint32_t rowId;
    uint32_t expectedChunkIndex;
  } testCases[] = {
      // Stream 0 has startChunkOffset_ == 0. Cumulative row counts
      // {100, 350, 750} → chunk 0 covers [0, 99], chunk 1 covers [100, 349],
      // chunk 2 covers [350, 749]. Stripe-absolute chunkIndex == 0, 1, 2.
      {0, 0, 0},
      {0, 99, 0},
      {0, 100, 1},
      {0, 349, 1},
      {0, 350, 2},
      {0, 749, 2},
      // Stream 1 has startChunkOffset_ == 3 (stream 0 contributed 3 chunks).
      // Cumulative row counts {150, 450} → chunk 0 covers [0, 149], chunk 1
      // covers [150, 449]. Stripe-absolute chunkIndex == 3, 4.
      {1, 0, 3},
      {1, 149, 3},
      {1, 150, 4},
      {1, 449, 4},
  };

  for (const auto& tc : testCases) {
    SCOPED_TRACE(fmt::format("streamId {} rowId {}", tc.streamId, tc.rowId));
    auto streamIndex = chunkStats->createStreamIndex(
        /*stripe=*/0, tc.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);
    const auto result = streamIndex->lookupChunk(tc.rowId);
    EXPECT_EQ(result.chunkIndex, tc.expectedChunkIndex);
  }
}

TEST_F(ChunkStatsTest, createStreamIndex) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  std::vector<Stripe> stripes = {
      {.streams =
           {{.numChunks = 3,
             .chunkRows = {100, 200, 300},
             .chunkOffsets = {0, 50, 120}},
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 80}}},
       .keyStream = {
           .streamOffset = 0,
           .streamSize = 100,
           .stream =
               {.numChunks = 3,
                .chunkRows = {100, 200, 300},
                .chunkOffsets = {0, 30, 70}},
           .chunkKeys = {"bbb", "ccc", "ddd"}}}};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  // Multi-chunk streams return non-null StreamIndex.
  EXPECT_NE(chunkStats->createStreamIndex(0, 0, /*streamSize=*/1000), nullptr);
  EXPECT_NE(chunkStats->createStreamIndex(0, 1, /*streamSize=*/1000), nullptr);
}

TEST_F(ChunkStatsTest, singleChunkStreamReturnsNullptr) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Mix of single-chunk and multi-chunk streams across stripes.
  std::vector<Stripe> stripes = {
      {.streams =
           {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            {.numChunks = 3,
             .chunkRows = {50, 100, 150},
             .chunkOffsets = {0, 20, 60}},
            {.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 0,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
           .chunkKeys = {"bbb"}}}};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  // Single-chunk streams return nullptr.
  EXPECT_EQ(chunkStats->createStreamIndex(0, 0, /*streamSize=*/1000), nullptr);
  EXPECT_EQ(chunkStats->createStreamIndex(0, 2, /*streamSize=*/1000), nullptr);

  // Multi-chunk stream returns non-null.
  auto streamIndex = chunkStats->createStreamIndex(0, 1, /*streamSize=*/1000);
  ASSERT_NE(streamIndex, nullptr);
  EXPECT_EQ(streamIndex->streamId(), 1);
}

TEST_F(ChunkStatsTest, streamIndexStreamId) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Each stream needs >1 chunk so createStreamIndex returns non-null.
  std::vector<Stripe> stripes = {
      {.streams =
           {{.numChunks = 2, .chunkRows = {50, 100}, .chunkOffsets = {0, 20}},
            {.numChunks = 2, .chunkRows = {75, 150}, .chunkOffsets = {0, 30}},
            {.numChunks = 2, .chunkRows = {100, 200}, .chunkOffsets = {0, 40}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      {.streams =
           {{.numChunks = 2, .chunkRows = {60, 120}, .chunkOffsets = {0, 25}},
            {.numChunks = 2, .chunkRows = {90, 180}, .chunkOffsets = {0, 35}},
            {.numChunks = 2, .chunkRows = {120, 240}, .chunkOffsets = {0, 45}}},
       .keyStream = {
           .streamOffset = 100,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {240}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}}};
  std::vector<int> stripeGroups = {2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  struct {
    uint32_t stripeIndex;
    uint32_t streamId;
  } testCases[] = {
      {0, 0},
      {0, 1},
      {0, 2},
      {1, 0},
      {1, 1},
      {1, 2},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "stripeIndex {} streamId {}",
            testCase.stripeIndex,
            testCase.streamId));
    auto streamIndex = chunkStats->createStreamIndex(
        testCase.stripeIndex, testCase.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);
    EXPECT_EQ(streamIndex->streamId(), testCase.streamId);
  }
}

TEST_F(ChunkStatsTest, streamIndexLookupChunk) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Setup: 4 stripes across 2 groups, each stripe with 2 streams
  std::vector<Stripe> stripes = {
      // Stripe 0 (Group 0)
      {.streams =
           {// Stream 0: 3 chunks at rows {100, 250, 400}, offsets {0, 50, 120}
            {.numChunks = 3,
             .chunkRows = {100, 250, 400},
             .chunkOffsets = {0, 50, 120}},
            // Stream 1: 2 chunks at rows {150, 300}, offsets {0, 80}
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 80}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {400}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      // Stripe 1 (Group 0)
      {.streams =
           {// Stream 0: 2 chunks at rows {200, 350}, offsets {0, 60}
            {.numChunks = 2, .chunkRows = {200, 350}, .chunkOffsets = {0, 60}},
            // Stream 1: 3 chunks at rows {100, 200, 350}, offsets {0, 40, 90}
            {.numChunks = 3,
             .chunkRows = {100, 200, 350},
             .chunkOffsets = {0, 40, 90}}},
       .keyStream =
           {.streamOffset = 100,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {350}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      // Stripe 2 (Group 1)
      {.streams =
           {// Stream 0: 2 chunks at rows {150, 300}, offsets {0, 70}
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 70}},
            // Stream 1: 2 chunks at rows {180, 360}, offsets {0, 55}
            {.numChunks = 2, .chunkRows = {180, 360}, .chunkOffsets = {0, 55}}},
       .keyStream =
           {.streamOffset = 200,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {300}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      // Stripe 3 (Group 1)
      {.streams =
           {// Stream 0: 3 chunks at rows {80, 160, 240}, offsets {0, 30, 65}
            {.numChunks = 3,
             .chunkRows = {80, 160, 240},
             .chunkOffsets = {0, 30, 65}},
            // Stream 1: 2 chunks at rows {120, 240}, offsets {0, 45}
            {.numChunks = 2, .chunkRows = {120, 240}, .chunkOffsets = {0, 45}}},
       .keyStream = {
           .streamOffset = 300,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {240}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  struct {
    uint32_t groupIndex;
    uint32_t stripeIndex;
    uint32_t streamId;
    uint32_t rowId;
    uint32_t expectedStreamOffset;
    uint32_t expectedRowOffset;
  } testCases[] = {
      // Group 0, Stripe 0, Stream 0: chunks at rows {100, 250, 400}, offsets
      // {0, 50, 120}
      // Accumulated rows: {100, 350, 750}
      {0, 0, 0, 0, 0, 0},
      {0, 0, 0, 50, 0, 0},
      {0, 0, 0, 99, 0, 0},
      {0, 0, 0, 100, 50, 100},
      {0, 0, 0, 349, 50, 100},
      {0, 0, 0, 350, 120, 350},
      {0, 0, 0, 749, 120, 350},

      // Group 0, Stripe 0, Stream 1: chunks at rows {150, 300}, offsets {0, 80}
      // Accumulated rows: {150, 450}
      {0, 0, 1, 0, 0, 0},
      {0, 0, 1, 149, 0, 0},
      {0, 0, 1, 150, 80, 150},
      {0, 0, 1, 449, 80, 150},

      // Group 0, Stripe 1, Stream 0: chunks at rows {200, 350}, offsets {0, 60}
      // Accumulated rows: {200, 550}
      {0, 1, 0, 0, 0, 0},
      {0, 1, 0, 199, 0, 0},
      {0, 1, 0, 200, 60, 200},
      {0, 1, 0, 549, 60, 200},

      // Group 0, Stripe 1, Stream 1: chunks at rows {100, 200, 350}, offsets
      // {0, 40, 90}
      // Accumulated rows: {100, 300, 650}
      {0, 1, 1, 0, 0, 0},
      {0, 1, 1, 99, 0, 0},
      {0, 1, 1, 100, 40, 100},
      {0, 1, 1, 299, 40, 100},
      {0, 1, 1, 300, 90, 300},
      {0, 1, 1, 649, 90, 300},

      // Group 1, Stripe 2, Stream 0: chunks at rows {150, 300}, offsets {0, 70}
      // Accumulated rows: {150, 450}
      {1, 2, 0, 0, 0, 0},
      {1, 2, 0, 149, 0, 0},
      {1, 2, 0, 150, 70, 150},
      {1, 2, 0, 449, 70, 150},

      // Group 1, Stripe 2, Stream 1: chunks at rows {180, 360}, offsets {0, 55}
      // Accumulated rows: {180, 540}
      {1, 2, 1, 0, 0, 0},
      {1, 2, 1, 179, 0, 0},
      {1, 2, 1, 180, 55, 180},
      {1, 2, 1, 539, 55, 180},

      // Group 1, Stripe 3, Stream 0: chunks at rows {80, 160, 240}, offsets {0,
      // 30, 65}
      // Accumulated rows: {80, 240, 480}
      {1, 3, 0, 0, 0, 0},
      {1, 3, 0, 79, 0, 0},
      {1, 3, 0, 80, 30, 80},
      {1, 3, 0, 239, 30, 80},
      {1, 3, 0, 240, 65, 240},
      {1, 3, 0, 479, 65, 240},

      // Group 1, Stripe 3, Stream 1: chunks at rows {120, 240}, offsets {0, 45}
      // Accumulated rows: {120, 360}
      {1, 3, 1, 0, 0, 0},
      {1, 3, 1, 119, 0, 0},
      {1, 3, 1, 120, 45, 120},
      {1, 3, 1, 359, 45, 120},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "groupIndex {} stripeIndex {} streamId {} rowId {}",
            testCase.groupIndex,
            testCase.stripeIndex,
            testCase.streamId,
            testCase.rowId));

    auto chunkStats = createChunkStats(indexBuffers, testCase.groupIndex);
    ASSERT_NE(chunkStats, nullptr);

    auto streamIndex = chunkStats->createStreamIndex(
        testCase.stripeIndex, testCase.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);

    auto result = streamIndex->lookupChunk(testCase.rowId);
    EXPECT_EQ(result.chunkOffset, testCase.expectedStreamOffset);
    EXPECT_EQ(result.rowOffset, testCase.expectedRowOffset);
  }

  // Row at/after last chunk boundary — throws
  auto ci0 = createChunkStats(indexBuffers, 0);
  NIMBLE_ASSERT_THROW(
      ci0->createStreamIndex(0, 0, /*streamSize=*/1000)->lookupChunk(750),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci0->createStreamIndex(0, 1, /*streamSize=*/1000)->lookupChunk(450),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci0->createStreamIndex(1, 0, /*streamSize=*/1000)->lookupChunk(550),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci0->createStreamIndex(1, 1, /*streamSize=*/1000)->lookupChunk(650),
      "beyond the last chunk");

  auto ci1 = createChunkStats(indexBuffers, 1);
  NIMBLE_ASSERT_THROW(
      ci1->createStreamIndex(2, 0, /*streamSize=*/1000)->lookupChunk(450),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci1->createStreamIndex(2, 1, /*streamSize=*/1000)->lookupChunk(540),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci1->createStreamIndex(3, 0, /*streamSize=*/1000)->lookupChunk(480),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      ci1->createStreamIndex(3, 1, /*streamSize=*/1000)->lookupChunk(360),
      "beyond the last chunk");
}

TEST_F(ChunkStatsTest, stripeIndexAndStreamIdOutOfBound) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Setup: 4 stripes across 2 groups, each stripe with 2 streams
  std::vector<Stripe> stripes = {
      // Stripe 0 (Group 0)
      {.streams =
           {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      // Stripe 1 (Group 0)
      {.streams =
           {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 100,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      // Stripe 2 (Group 1)
      {.streams =
           {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 200,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      // Stripe 3 (Group 1)
      {.streams =
           {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 300,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  // Test Group 0 (stripes 0, 1)
  auto chunkStats0 = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats0, nullptr);

  // Test stripe index after group's range
  NIMBLE_ASSERT_THROW(
      chunkStats0->createStreamIndex(2, 0, /*streamSize=*/1000),
      "Stripe offset is out of range for this chunk stats group");
  NIMBLE_ASSERT_THROW(
      chunkStats0->createStreamIndex(3, 0, /*streamSize=*/1000),
      "Stripe offset is out of range for this chunk stats group");

  // streamId >= streamCount returns nullptr.
  EXPECT_EQ(chunkStats0->createStreamIndex(0, 2, /*streamSize=*/1000), nullptr);
  EXPECT_EQ(
      chunkStats0->createStreamIndex(1, 10, /*streamSize=*/1000), nullptr);

  // Test Group 1 (stripes 2, 3)
  auto chunkStats1 = createChunkStats(indexBuffers, 1);
  ASSERT_NE(chunkStats1, nullptr);

  // Stripe index before group's range (group 1 starts at stripe 2)
  NIMBLE_ASSERT_THROW(
      chunkStats1->createStreamIndex(0, 0, /*streamSize=*/1000),
      "Stripe index is before this group's range");
  NIMBLE_ASSERT_THROW(
      chunkStats1->createStreamIndex(1, 0, /*streamSize=*/1000),
      "Stripe index is before this group's range");

  // Stripe index after group's range (group 1 has stripes 2, 3)
  NIMBLE_ASSERT_THROW(
      chunkStats1->createStreamIndex(4, 0, /*streamSize=*/1000),
      "Stripe offset is out of range for this chunk stats group");
  NIMBLE_ASSERT_THROW(
      chunkStats1->createStreamIndex(5, 0, /*streamSize=*/1000),
      "Stripe offset is out of range for this chunk stats group");

  // streamId >= streamCount returns nullptr.
  EXPECT_EQ(chunkStats1->createStreamIndex(2, 2, /*streamSize=*/1000), nullptr);
  EXPECT_EQ(chunkStats1->createStreamIndex(3, 5, /*streamSize=*/1000), nullptr);
}

TEST_F(ChunkStatsTest, streamIndexRowCount) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Setup: 4 stripes across 2 groups, each stripe with 2 streams
  std::vector<Stripe> stripes = {
      // Stripe 0 (Group 0)
      {.streams =
           {// Stream 0: 3 chunks at rows {100, 250, 400}, offsets {0, 50, 120}
            // Accumulated rows: 100, 350, 750 -> total rows = 750
            {.numChunks = 3,
             .chunkRows = {100, 250, 400},
             .chunkOffsets = {0, 50, 120}},
            // Stream 1: 2 chunks at rows {150, 300}, offsets {0, 80}
            // Accumulated rows: 150, 450 -> total rows = 450
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 80}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {750}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      // Stripe 1 (Group 0)
      {.streams =
           {// Stream 0: 2 chunks at rows {200, 350}, offsets {0, 60}
            // Accumulated rows: 200, 550 -> total rows = 550
            {.numChunks = 2, .chunkRows = {200, 350}, .chunkOffsets = {0, 60}},
            // Stream 1: 3 chunks at rows {100, 200, 350}, offsets {0, 40, 90}
            // Accumulated rows: 100, 300, 650 -> total rows = 650
            {.numChunks = 3,
             .chunkRows = {100, 200, 350},
             .chunkOffsets = {0, 40, 90}}},
       .keyStream =
           {.streamOffset = 100,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {550}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      // Stripe 2 (Group 1)
      {.streams =
           {// Stream 0: 2 chunks at rows {150, 300}, offsets {0, 70}
            // Accumulated rows: 150, 450 -> total rows = 450
            {.numChunks = 2, .chunkRows = {150, 300}, .chunkOffsets = {0, 70}},
            // Stream 1: 2 chunks at rows {180, 360}, offsets {0, 55}
            // Accumulated rows: 180, 540 -> total rows = 540
            {.numChunks = 2, .chunkRows = {180, 360}, .chunkOffsets = {0, 55}}},
       .keyStream =
           {.streamOffset = 200,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {450}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      // Stripe 3 (Group 1)
      {.streams =
           {// Stream 0: 3 chunks at rows {80, 160, 240}, offsets {0, 30, 65}
            // Accumulated rows: 80, 240, 480 -> total rows = 480
            {.numChunks = 3,
             .chunkRows = {80, 160, 240},
             .chunkOffsets = {0, 30, 65}},
            // Stream 1: 2 chunks at rows {120, 240}, offsets {0, 45}
            // Accumulated rows: 120, 360 -> total rows = 360
            {.numChunks = 2, .chunkRows = {120, 240}, .chunkOffsets = {0, 45}}},
       .keyStream = {
           .streamOffset = 300,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {480}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  struct {
    uint32_t groupIndex;
    uint32_t stripeIndex;
    uint32_t streamId;
    uint32_t expectedRowCount;
  } testCases[] = {
      // Group 0, Stripe 0
      {0, 0, 0, 750}, // Stream 0: accumulated rows = 100 + 250 + 400 = 750
      {0, 0, 1, 450}, // Stream 1: accumulated rows = 150 + 300 = 450

      // Group 0, Stripe 1
      {0, 1, 0, 550}, // Stream 0: accumulated rows = 200 + 350 = 550
      {0, 1, 1, 650}, // Stream 1: accumulated rows = 100 + 200 + 350 = 650

      // Group 1, Stripe 2
      {1, 2, 0, 450}, // Stream 0: accumulated rows = 150 + 300 = 450
      {1, 2, 1, 540}, // Stream 1: accumulated rows = 180 + 360 = 540

      // Group 1, Stripe 3
      {1, 3, 0, 480}, // Stream 0: accumulated rows = 80 + 160 + 240 = 480
      {1, 3, 1, 360}, // Stream 1: accumulated rows = 120 + 240 = 360
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "groupIndex {} stripeIndex {} streamId {}",
            testCase.groupIndex,
            testCase.stripeIndex,
            testCase.streamId));

    auto chunkStats = createChunkStats(indexBuffers, testCase.groupIndex);
    ASSERT_NE(chunkStats, nullptr);

    auto streamIndex = chunkStats->createStreamIndex(
        testCase.stripeIndex, testCase.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);
    EXPECT_EQ(streamIndex->rowCount(), testCase.expectedRowCount);
  }
}

TEST_F(ChunkStatsTest, chunkOnlyLookupByRowId) {
  // Chunk-index-only: no value index, no key stream.
  std::vector<ChunkOnlyStripe> stripes = {
      {.streams =
           {{.numChunks = 3,
             .chunkRows = {100, 250, 400},
             .chunkOffsets = {0, 50, 120}},
            {.numChunks = 2,
             .chunkRows = {150, 300},
             .chunkOffsets = {0, 80}}}},
      {.streams = {
           {.numChunks = 2, .chunkRows = {200, 350}, .chunkOffsets = {0, 60}},
           {.numChunks = 3,
            .chunkRows = {100, 200, 350},
            .chunkOffsets = {0, 40, 90}}}}};
  std::vector<int> stripeGroups = {2};

  auto indexBuffers = createChunkOnlyTestClusterIndex(stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  struct {
    uint32_t stripeIndex;
    uint32_t streamId;
    uint32_t rowId;
    uint32_t expectedStreamOffset;
    uint32_t expectedRowOffset;
  } testCases[] = {
      // Stripe 0, Stream 0: chunks at rows {100, 250, 400}, offsets {0, 50,
      // 120}. Accumulated rows: {100, 350, 750}
      {0, 0, 0, 0, 0},
      {0, 0, 99, 0, 0},
      {0, 0, 100, 50, 100},
      {0, 0, 350, 120, 350},
      // Stripe 0, Stream 1: chunks at rows {150, 300}, offsets {0, 80}.
      // Accumulated rows: {150, 450}
      {0, 1, 0, 0, 0},
      {0, 1, 150, 80, 150},
      // Stripe 1, Stream 0: chunks at rows {200, 350}, offsets {0, 60}.
      // Accumulated rows: {200, 550}
      {1, 0, 0, 0, 0},
      {1, 0, 200, 60, 200},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "stripeIndex {} streamId {} rowId {}",
            testCase.stripeIndex,
            testCase.streamId,
            testCase.rowId));
    auto streamIndex = chunkStats->createStreamIndex(
        testCase.stripeIndex, testCase.streamId, /*streamSize=*/1000);
    ASSERT_NE(streamIndex, nullptr);
    auto result = streamIndex->lookupChunk(testCase.rowId);
    EXPECT_EQ(result.chunkOffset, testCase.expectedStreamOffset);
    EXPECT_EQ(result.rowOffset, testCase.expectedRowOffset);
  }

  // Row at/after last chunk boundary — throws
  NIMBLE_ASSERT_THROW(
      chunkStats->createStreamIndex(0, 0, /*streamSize=*/1000)
          ->lookupChunk(750),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      chunkStats->createStreamIndex(0, 1, /*streamSize=*/1000)
          ->lookupChunk(450),
      "beyond the last chunk");
  NIMBLE_ASSERT_THROW(
      chunkStats->createStreamIndex(1, 0, /*streamSize=*/1000)
          ->lookupChunk(550),
      "beyond the last chunk");
}

TEST_F(ChunkStatsTest, chunkNullCountAbsentReturnsNullopt) {
  // Files written before per-chunk null statistics existed store the chunk
  // index without the appended null-count array (as do these fixtures, which
  // set only counts/rows/offsets). chunkNullCount() must report the count as
  // unknown (nullopt) rather than fabricating a value.
  std::vector<ChunkOnlyStripe> stripes = {
      {.streams = {
           {.numChunks = 3,
            .chunkRows = {100, 250, 400},
            .chunkOffsets = {0, 50, 120}}}}};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers = createChunkOnlyTestClusterIndex(stripes, stripeGroups);
  auto chunkStats = createChunkStats(indexBuffers, 0);
  ASSERT_NE(chunkStats, nullptr);

  auto streamIndex = chunkStats->createStreamIndex(0, 0, /*streamSize=*/1000);
  ASSERT_NE(streamIndex, nullptr);
  const auto location = streamIndex->lookupChunk(0);
  EXPECT_FALSE(streamIndex->chunkNullCount(location.chunkIndex).has_value());
}

} // namespace facebook::nimble::index::test
