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

// Thin gtest wrapper so CI can run the fuzzer loop. All setup and run logic is
// shared with the standalone binary via NimbleWriterFuzzerRunner.

#include <map>

#include <folly/init/Init.h>
#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzerRunner.h"

namespace facebook::nimble::fuzzer {

TEST(NimbleWriterFuzzerTest, cappingJsonShape) {
  bool sawMissing795{false};
  bool sawPresent795{false};
  bool sawMissing855{false};
  bool sawPresent855{false};
  bool sawEmptyT1{false};
  bool sawLongT1{false};
  bool sawPresent402{false};
  bool sawPresent921{false};

  for (uint64_t row = 0; row < 128; ++row) {
    const auto seed = 0xc0ffeeULL + row;
    const auto value = makeCappingJsonValue(seed, row);
    EXPECT_EQ(value, makeCappingJsonValue(seed, row));

    const auto parsed = folly::parseJson(value);
    ASSERT_TRUE(parsed.isObject());
    ASSERT_TRUE(parsed.count("t1"));
    const auto& t1 = parsed.at("t1");
    ASSERT_TRUE(t1.isArray());
    sawEmptyT1 |= t1.empty();
    sawLongT1 |= t1.size() >= 64;
    for (const auto& item : parsed.items()) {
      ASSERT_TRUE(item.second.isArray());
      for (const auto& element : item.second) {
        EXPECT_TRUE(element.isInt());
      }
    }
    sawMissing795 |= !parsed.count("795");
    sawPresent795 |= parsed.count("795");
    sawMissing855 |= !parsed.count("855");
    sawPresent855 |= parsed.count("855");
    sawPresent402 |= parsed.count("402");
    sawPresent921 |= parsed.count("921");
  }

  EXPECT_TRUE(sawMissing795);
  EXPECT_TRUE(sawPresent795);
  EXPECT_TRUE(sawMissing855);
  EXPECT_TRUE(sawPresent855);
  EXPECT_TRUE(sawEmptyT1);
  EXPECT_TRUE(sawLongT1);
  EXPECT_TRUE(sawPresent402);
  EXPECT_TRUE(sawPresent921);
}

TEST(NimbleWriterFuzzerTest, chunkStatsCoverageBreakdownsAddUp) {
  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "chunkStatsCoverageBreakdownsAddUp", velox::memory::kMaxMemory);
  NimbleWriterFuzzerOptions options;
  options.seed = 1;
  options.batchSize = 64;
  options.numBatches = 4;
  NimbleWriterFuzzer fuzzer(options, *rootPool);
  for (int iteration = 0; iteration < 3; ++iteration) {
    fuzzer.run();
    fuzzer.reSeed();
  }

  const auto& coverage = fuzzer.chunkStatsCoverage();
  ASSERT_GT(coverage.numV2IndexedStreams, 0U);

  const auto sumBuckets = [](const std::map<uint32_t, uint64_t>& buckets) {
    uint64_t sum{0};
    for (const auto& [bucket, count] : buckets) {
      sum += count;
    }
    return sum;
  };
  uint64_t numFilesByStripes{0};
  for (const auto& [flushRegime, buckets] : coverage.filesByStripes) {
    numFilesByStripes += sumBuckets(buckets);
  }
  uint64_t numFilesByStripeGroups{0};
  for (const auto& [threshold, buckets] : coverage.filesByStripeGroups) {
    numFilesByStripeGroups += sumBuckets(buckets);
  }
  const auto numFiles = coverage.numV2IndexedFiles + coverage.numUnindexedFiles;
  EXPECT_EQ(numFilesByStripes, numFiles);
  EXPECT_EQ(numFilesByStripeGroups, numFiles);
  EXPECT_EQ(sumBuckets(coverage.streamsByChunks), coverage.numV2IndexedStreams);

  // The chunk-per-batch regime never cuts a stripe before the writer closes.
  const auto chunkPerBatch = coverage.filesByStripes.find("chunkPerBatch");
  ASSERT_NE(chunkPerBatch, coverage.filesByStripes.end());
  ASSERT_EQ(chunkPerBatch->second.size(), 1U);
  EXPECT_EQ(chunkPerBatch->second.begin()->first, 1U);

  ChunkStatsVerificationCoverage::StreamKindCoverage kindTotals;
  const auto addKinds =
      [&](const std::map<
          std::string,
          ChunkStatsVerificationCoverage::StreamKindCoverage>& streamKinds) {
        uint64_t numStreams{0};
        for (const auto& [kind, kindCoverage] : streamKinds) {
          // Every chunk either carries min/max or has one reason for lacking
          // it.
          EXPECT_EQ(
              kindCoverage.numBoundedChunks +
                  kindCoverage.numAllNullUnboundedChunks +
                  kindCoverage.numNaNUnboundedChunks +
                  kindCoverage.numLongStringUnboundedChunks,
              kindCoverage.numChunks)
              << kind;
          EXPECT_LE(
              kindCoverage.numSingleValueChunks, kindCoverage.numBoundedChunks)
              << kind;
          numStreams += kindCoverage.numStreams;
          kindTotals.numMultiChunkStreams += kindCoverage.numMultiChunkStreams;
          kindTotals.numZeroNullChunks += kindCoverage.numZeroNullChunks;
          kindTotals.numPartiallyNullChunks +=
              kindCoverage.numPartiallyNullChunks;
          kindTotals.numFullyNullChunks += kindCoverage.numFullyNullChunks;
          kindTotals.numBoundedChunks += kindCoverage.numBoundedChunks;
        }
        return numStreams;
      };
  EXPECT_EQ(addKinds(coverage.scalarStreamKinds), coverage.numScalarStreams);
  EXPECT_EQ(
      addKinds(coverage.complexStreamKinds), coverage.numStructuralStreams);
  EXPECT_EQ(kindTotals.numMultiChunkStreams, coverage.numMultiChunkStreams);
  EXPECT_EQ(kindTotals.numZeroNullChunks, coverage.numZeroNullChunks);
  EXPECT_EQ(kindTotals.numPartiallyNullChunks, coverage.numPartiallyNullChunks);
  EXPECT_EQ(kindTotals.numFullyNullChunks, coverage.numFullyNullChunks);
  EXPECT_EQ(kindTotals.numBoundedChunks, coverage.numV2BoundedChunks);
  EXPECT_LE(
      coverage.numNullBatchIterationsWithFullyNullChunk,
      coverage.numNullBatchIterations);
}

TEST(NimbleWriterFuzzerTest, pointLookupsMatchInput) {
  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "pointLookupsMatchInput", velox::memory::kMaxMemory);
  NimbleWriterFuzzerOptions options;
  options.seed = 1;
  options.batchSize = 64;
  options.numBatches = 4;
  NimbleWriterFuzzer fuzzer(options, *rootPool);
  for (int iteration = 0; iteration < 2; ++iteration) {
    fuzzer.run();
    fuzzer.reSeed();
  }

  const auto& coverage = fuzzer.pointLookupCoverage();
  uint64_t numLookups{0};
  for (const auto& [numColumns, counts] : coverage.byNumColumns) {
    EXPECT_GE(numColumns, 1U);
    EXPECT_LE(numColumns, 3U);
    EXPECT_LE(counts.numLookupsWithChunkSkips, counts.numLookups);
    numLookups += counts.numLookups;
  }
  EXPECT_GT(numLookups, 0U);
  EXPECT_LE(numLookups, fuzzer.numUnfilteredFilesWritten());
  EXPECT_GT(coverage.byNumColumns.rbegin()->first, 1U);

  for (const auto& [type, byKind] : coverage.byType) {
    for (const auto& [kind, counts] : byKind) {
      EXPECT_LE(counts.numLookupsWithChunkSkips, counts.numLookups)
          << type << " " << kind;
    }
  }

  uint64_t numBucketedLookups{0};
  for (const auto& [anchoring, buckets] : coverage.lookupsByMatchedRows) {
    for (const auto& [bucket, count] : buckets) {
      numBucketedLookups += count;
    }
  }
  EXPECT_EQ(numBucketedLookups, numLookups);

  // Anchored lookups return whole blocks of consecutive rows.
  const auto anchored = coverage.lookupsByMatchedRows.find("anchored");
  ASSERT_NE(anchored, coverage.lookupsByMatchedRows.end());
  EXPECT_GT(anchored->second.rbegin()->first, 1U);
}

TEST(NimbleWriterFuzzerTest, run) {
  runNimbleWriterFuzzer();
}

} // namespace facebook::nimble::fuzzer

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::fuzzer::setUpFuzzerEnvironments();
  return RUN_ALL_TESTS();
}
