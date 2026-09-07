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

#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/index/ClusterIndexGroup.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestBase.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

namespace facebook::nimble::index::test {

class ClusterIndexGroupTest : public ClusterIndexTestBase {};

TEST_F(ClusterIndexGroupTest, chunkLocation) {
  ChunkLocation loc1{100, 50, 200};
  EXPECT_EQ(loc1.chunkOffset, 100);
  EXPECT_EQ(loc1.chunkSize, 50);
  EXPECT_EQ(loc1.rowOffset, 200);

  ChunkLocation loc2{0, 0, 0};
  EXPECT_EQ(loc2.chunkOffset, 0);
  EXPECT_EQ(loc2.chunkSize, 0);
  EXPECT_EQ(loc2.rowOffset, 0);

  ChunkLocation loc3{UINT32_MAX, UINT32_MAX, UINT32_MAX};
  EXPECT_EQ(loc3.chunkOffset, UINT32_MAX);
  EXPECT_EQ(loc3.chunkSize, UINT32_MAX);
  EXPECT_EQ(loc3.rowOffset, UINT32_MAX);
}

TEST_F(ClusterIndexGroupTest, keyStreamRegion) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  std::vector<Stripe> stripes = {
      {.streams = {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 100,
            .streamSize = 150,
            .stream = {.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 250,
            .streamSize = 200,
            .stream = {.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 450,
           .streamSize = 50,
           .stream = {.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  // Test all key stream regions using parameterized test cases
  struct {
    uint32_t groupIndex;
    uint32_t stripeIndex;
    uint32_t expectedOffset;
    uint32_t expectedLength;
  } testCases[] = {
      // Group 0: stripes 0, 1
      {0, 0, 0, 100},
      {0, 1, 100, 150},
      // Group 1: stripes 2, 3
      {1, 2, 250, 200},
      {1, 3, 450, 50},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "groupIndex {} stripeIndex {}",
            testCase.groupIndex,
            testCase.stripeIndex));
    auto clusterIndexGroup =
        createClusterIndexGroup(indexBuffers, testCase.groupIndex);
    ASSERT_NE(clusterIndexGroup, nullptr);

    auto region = clusterIndexGroup->keyStreamRegion(testCase.stripeIndex, 0);
    EXPECT_EQ(region.offset, testCase.expectedOffset);
    EXPECT_EQ(region.length, testCase.expectedLength);
  }
}

TEST_F(ClusterIndexGroupTest, keyStreamRegionWithLargeBaseOffset) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  std::vector<Stripe> stripes = {
      {.streams = {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}},
            .chunkKeys = {"bbb"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 100,
            .streamSize = 150,
            .stream = {.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 250,
            .streamSize = 200,
            .stream = {.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 450,
           .streamSize = 50,
           .stream = {.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  // Test cases where stripe based relative offset is able to handle
  // large files (e.g., > 4 GB)
  struct {
    uint32_t groupIndex;
    uint32_t stripeIndex;
    uint64_t stripeBaseOffset;
    uint64_t expectedOffset;
    uint32_t expectedLength;
  } testCases[] = {
      // Base offset exactly at 4 GB boundary
      {0,
       0,
       static_cast<uint64_t>(UINT32_MAX) + 1,
       static_cast<uint64_t>(UINT32_MAX) + 1 + 0,
       100},
      {0,
       1,
       static_cast<uint64_t>(UINT32_MAX) + 1,
       static_cast<uint64_t>(UINT32_MAX) + 1 + 100,
       150},

      // Base offset at UINT32_MAX (just under 4 GB)
      {0, 0, UINT32_MAX, static_cast<uint64_t>(UINT32_MAX) + 0, 100},
      {0, 1, UINT32_MAX, static_cast<uint64_t>(UINT32_MAX) + 100, 150},
      // Large base offset with large relative offset to verify no truncation
      {1,
       3,
       static_cast<uint64_t>(UINT32_MAX) + 1000,
       static_cast<uint64_t>(UINT32_MAX) + 1000 + 450,
       50},
      // Zero base offset (existing behavior, sanity check)
      {0, 0, 0, 0, 100},
      {0, 1, 0, 100, 150},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "groupIndex {} stripeIndex {} stripeBaseOffset {}",
            testCase.groupIndex,
            testCase.stripeIndex,
            testCase.stripeBaseOffset));
    auto clusterIndexGroup =
        createClusterIndexGroup(indexBuffers, testCase.groupIndex);
    ASSERT_NE(clusterIndexGroup, nullptr);

    auto region = clusterIndexGroup->keyStreamRegion(
        testCase.stripeIndex, testCase.stripeBaseOffset);
    EXPECT_EQ(region.offset, testCase.expectedOffset);
    EXPECT_EQ(region.length, testCase.expectedLength);
  }
}

TEST_F(ClusterIndexGroupTest, lookupChunkWithEncodedKey) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  std::vector<Stripe> stripes = {
      {.streams = {{.numChunks = 1, .chunkRows = {100}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 100,
            .stream =
                {.numChunks = 3,
                 .chunkRows = {100, 200, 300},
                 .chunkOffsets = {0, 50, 120}},
            .chunkKeys = {"ccc", "fff", "iii"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 100,
           .streamSize = 150,
           .stream =
               {.numChunks = 2,
                .chunkRows = {150, 300},
                .chunkOffsets = {0, 80}},
           .chunkKeys = {"mmm", "ppp"}}}};
  std::vector<int> stripeGroups = {2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndexGroup = createClusterIndexGroup(indexBuffers, 0);
  ASSERT_NE(clusterIndexGroup, nullptr);

  struct {
    uint32_t stripeIndex;
    std::string encodedKey;
    std::optional<uint32_t> expectedChunkOffset;
    std::optional<uint32_t> expectedRowOffset;
  } testCases[] = {
      // Stripe 0: chunks with keys "ccc", "fff", "iii"
      // chunkRows = {100, 200, 300}
      // Key before first chunk locate at the first chunk as we
      // expect the caller will locate the right stripe with tablet index.
      {0, "aa", 0, 0},
      {0, "aaa", 0, 0},
      {0, "bbb", 0, 0},
      // Key at first chunk boundary
      {0, "ccc", 0, 0},
      // Key between first and second chunk
      {0, "ddd", 50, 100},
      {0, "eee", 50, 100},
      // Key at second chunk boundary
      {0, "fff", 50, 100},
      // Key between second and third chunk
      {0, "ggg", 120, 300},
      {0, "hhh", 120, 300},
      // Key at third chunk boundary
      {0, "iii", 120, 300},
      // Key after last chunk - not found
      {0, "jjj", std::nullopt, std::nullopt},
      {0, "zzz", std::nullopt, std::nullopt},

      // Stripe 1: chunks with keys "mmm", "ppp"
      // chunkRows = {150, 300}
      // Key before first chunk locate at the first chunk as we
      // expect the caller will locate the right stripe with tablet index.
      {1, "aaa", 0, 0},
      {1, "lll", 0, 0},
      // Key at first chunk boundary
      {1, "mmm", 0, 0},
      // Key between first and second chunk
      {1, "nnn", 80, 150},
      {1, "ooo", 80, 150},
      // Key at second chunk boundary
      {1, "ppp", 80, 150},
      // Key after last chunk - not found
      {1, "qqq", std::nullopt, std::nullopt},
      {1, "zzz", std::nullopt, std::nullopt}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "stripeIndex {} encodedKey {}",
            testCase.stripeIndex,
            testCase.encodedKey));
    auto result = clusterIndexGroup->lookupChunk(
        testCase.stripeIndex, testCase.encodedKey);
    if (testCase.expectedChunkOffset.has_value()) {
      ASSERT_TRUE(result.has_value())
          << "Expected chunk at streamOffset "
          << testCase.expectedChunkOffset.value() << ", rowOffset "
          << testCase.expectedRowOffset.value() << " for key '"
          << testCase.encodedKey << "', but lookup returned nullopt";
      EXPECT_EQ(result->chunkOffset, testCase.expectedChunkOffset.value());
      EXPECT_EQ(result->rowOffset, testCase.expectedRowOffset.value());
    } else {
      EXPECT_FALSE(result.has_value())
          << "Expected nullopt for key '" << testCase.encodedKey
          << "', but got streamOffset " << result->chunkOffset << ", rowOffset "
          << result->rowOffset;
    }
  }
}

} // namespace facebook::nimble::index::test
