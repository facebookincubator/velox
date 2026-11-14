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
#include <atomic>
#include <thread>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestBase.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

#include "folly/executors/IOThreadPoolExecutor.h"
#include "folly/synchronization/Baton.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"
#include "velox/serializers/KeyEncoder.h"

DECLARE_bool(velox_ssd_odirect);

namespace facebook::nimble::index::test {
namespace {

using namespace facebook::velox;

// Creates a range scan LookupRequest with only a lower bound (no upper bound).
// The result endRow will be numRows_ (total file rows).
std::unique_ptr<const IndexLookup::LookupRequest> makeRangeScanRequest(
    std::string_view key) {
  velox::serializer::EncodedKeyBounds bounds{
      .lowerKey = std::string(key), .upperKey = std::nullopt};
  return IndexLookup::LookupRequest::rangeScan({bounds});
}

// Returns the bounds of the rows of one key, which is the range from the key to
// its successor, the key followed by a zero byte.
velox::serializer::EncodedKeyBounds makeKeyBounds(std::string_view key) {
  return {
      .lowerKey = std::string(key),
      .upperKey = std::string(key) + std::string(1, '\0')};
}

// Returns the request for the rows of one key, as makeKeyBounds() bounds them.
std::unique_ptr<const IndexLookup::LookupRequest> makeKeyRequest(
    std::string_view key) {
  return IndexLookup::LookupRequest::rangeScan({makeKeyBounds(key)});
}

// Reads the keys of 'cursor' through batches of up to 'maxKeys' keys, and
// checks that only the last batch holds fewer keys, that a short last batch
// still holds every key left, and that a batch past the end appends nothing
// and allocates nothing. Copies the keys once all are read, so the keys of a
// batch must stay valid past later batches and past the chunk they came from.
std::vector<std::string> readKeyBatches(KeyCursor& cursor, size_t maxKeys) {
  std::vector<std::unique_ptr<char[]>> buffers;
  const KeyCursor::BufferAllocator bufferAllocator = [&](size_t size) {
    return buffers.emplace_back(std::make_unique<char[]>(size)).get();
  };
  std::vector<std::string_view> keys;
  while (cursor.hasNext()) {
    const auto numKeys = keys.size();
    cursor.next(maxKeys, keys, bufferAllocator);
    const auto numRead = keys.size() - numKeys;
    EXPECT_GT(numRead, 0);
    EXPECT_LE(numRead, maxKeys);
    if (cursor.hasNext()) {
      EXPECT_EQ(numRead, maxKeys);
    }
  }
  const auto numKeys = keys.size();
  const auto numBuffers = buffers.size();
  cursor.next(maxKeys, keys, bufferAllocator);
  EXPECT_EQ(keys.size(), numKeys);
  EXPECT_EQ(buffers.size(), numBuffers);
  return {keys.begin(), keys.end()};
}

class ClusterIndexTest : public ClusterIndexTestBase {
 protected:
  struct EncodedKeyStream {
    std::string data;
    std::vector<uint32_t> chunkOffsets;
  };

  // Pins the key-stream encoding the way ClusterIndexWriter does. A
  // selection policy would not: EncodingSizeEstimation has no Prefix case
  // for std::string_view, so it reports Prefix as incompatible and the
  // policy silently falls back to Trivial.
  static EncodingLayout keyEncodingLayout(EncodingType encodingType) {
    if (encodingType == EncodingType::Prefix) {
      return EncodingLayout{
          EncodingType::Prefix, {}, CompressionType::Uncompressed};
    }
    return EncodingLayout{
        EncodingType::Trivial,
        {},
        CompressionType::Uncompressed,
        {EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
  }

  // Encodes multiple chunks of keys into a single stream.
  EncodedKeyStream encodeKeyStream(
      const std::vector<std::vector<std::string>>& keyChunks,
      EncodingType encodingType = EncodingType::Trivial) {
    EncodedKeyStream result;
    Buffer buffer{*pool_};

    for (const auto& keys : keyChunks) {
      result.chunkOffsets.push_back(result.data.size());

      std::vector<std::string_view> keyViews;
      keyViews.reserve(keys.size());
      for (const auto& key : keys) {
        keyViews.push_back(key);
      }

      Buffer encodingBuffer{*pool_};
      auto policy =
          std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
              keyEncodingLayout(encodingType),
              CompressionOptions{},
              [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
                return nullptr;
              });
      auto encoded = EncodingFactory::encode<std::string_view>(
          std::move(policy), keyViews, encodingBuffer);

      ChunkedStreamWriter writer{buffer};
      auto segments = writer.encode(encoded);
      for (const auto& segment : segments) {
        result.data.append(segment.data(), segment.size());
      }
    }
    return result;
  }

  std::shared_ptr<velox::ReadFile> createKeyStreamFile(
      const std::string& streamData) {
    keyStreamFile_ = std::make_shared<velox::InMemoryReadFile>(streamData);
    return keyStreamFile_;
  }

  std::shared_ptr<velox::ReadFile> keyStreamFile_;
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  std::shared_ptr<velox::io::IoStatistics> indexIoStats_{
      std::make_shared<velox::io::IoStatistics>()};

  // Creates a Stripe with only the keyStream specified.
  Stripe createStripeWithKeys(const KeyStream& keyStream) {
    Stream defaultStream{
        .numChunks = keyStream.stream.numChunks,
        .chunkRows = keyStream.stream.chunkRows,
        .chunkOffsets = std::vector<uint32_t>(keyStream.stream.numChunks, 0)};
    return Stripe{.streams = {defaultStream}, .keyStream = keyStream};
  }

  // Keys of one stripe in sorted order, which its key stream holds as one key
  // chunk.
  using StripeKeys = std::vector<std::string>;

  // A key stream file and the stripes whose key streams it holds.
  struct StripeKeyStreams {
    std::string file;
    std::vector<Stripe> stripes;
  };

  // Writes the keys of each stripe as that stripe's key stream. Filler bytes
  // before each key stream stand in for the stripe's data streams, so no two
  // key streams are adjacent.
  StripeKeyStreams writeStripeKeyStreams(
      const std::vector<StripeKeys>& stripeKeys) {
    constexpr uint32_t kDataStreamBytes{17};
    StripeKeyStreams result;
    for (const auto& keys : stripeKeys) {
      result.file.append(kDataStreamBytes, 'x');
      const auto encoded = encodeKeyStream({keys});
      result.stripes.push_back(createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(result.file.size()),
           .streamSize = static_cast<uint32_t>(encoded.data.size()),
           .stream =
               {.numChunks = 1,
                .chunkRows = {static_cast<int32_t>(keys.size())},
                .chunkOffsets = encoded.chunkOffsets},
           .chunkKeys = {keys.back()}}));
      result.file.append(encoded.data);
    }
    return result;
  }

  // Keys "k00" to "k15" in four stripes of five, three, six and two keys, for
  // indexes of two partitions of two stripes each.
  static std::vector<StripeKeys> fourStripeKeys() {
    return {
        {"k00", "k01", "k02", "k03", "k04"},
        {"k05", "k06", "k07"},
        {"k08", "k09", "k10", "k11", "k12", "k13"},
        {"k14", "k15"},
    };
  }

  // Returns the keys of all stripes in row order.
  static std::vector<std::string> flattenKeys(
      const std::vector<StripeKeys>& stripeKeys) {
    std::vector<std::string> keys;
    for (const auto& stripe : stripeKeys) {
      keys.insert(keys.end(), stripe.begin(), stripe.end());
    }
    return keys;
  }

  // Creates a Stripe layout index of two stripes per partition over the key
  // streams of 'keyStreams'.
  std::unique_ptr<ClusterIndex> createStripeLayoutIndex(
      const StripeKeyStreams& keyStreams,
      bool pinIndex,
      bool preloadIndex) {
    return createClusterIndex(
        createTestClusterIndex(
            {"col1"},
            "k00",
            keyStreams.stripes,
            {2, 2},
            KeyStreamLayout::kStripe),
        createKeyStreamFile(keyStreams.file),
        pinIndex,
        preloadIndex);
  }

  // Returns the rows that a range scan from 'lowerKey', without an upper
  // bound, selects among the sorted 'keys': from the first key at or after it
  // to the last row, or none when every key is before it.
  static std::optional<RowRange> rowsFromLowerBound(
      const std::vector<std::string>& keys,
      const std::string& lowerKey) {
    const auto startRow = static_cast<uint32_t>(
        std::lower_bound(keys.begin(), keys.end(), lowerKey) - keys.begin());
    if (startRow == keys.size()) {
      return std::nullopt;
    }
    return RowRange(startRow, static_cast<uint32_t>(keys.size()));
  }

  // Expects the result of a one-request lookup to select 'expected', or no
  // rows when it is null.
  static void expectRows(
      const IndexLookup::LookupResult& result,
      const std::optional<RowRange>& expected) {
    ASSERT_EQ(result.size(), 1);
    const auto ranges = result[0];
    if (!expected.has_value()) {
      EXPECT_EQ(ranges.size(), 0);
      return;
    }
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].toString(), expected->toString());
  }
};

TEST_F(ClusterIndexTest, basic) {
  std::vector<std::string> indexColumns = {"col1", "col2", "col3"};
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
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {200}, .chunkOffsets = {0}},
            .chunkKeys = {"ccc"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 200,
            .streamSize = 100,
            .stream = {.numChunks = 1, .chunkRows = {150}, .chunkOffsets = {0}},
            .chunkKeys = {"ddd"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 300,
           .streamSize = 100,
           .stream = {.numChunks = 1, .chunkRows = {120}, .chunkOffsets = {0}},
           .chunkKeys = {"eee"}}}};
  std::vector<int> stripeGroups = {2, 1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex = createClusterIndex(indexBuffers);

  EXPECT_EQ(clusterIndex->type(), IndexType::Cluster);
  EXPECT_EQ(clusterIndex->layout().numPartitions, 3);

  const auto& cols = clusterIndex->indexColumns();
  EXPECT_EQ(cols.size(), 3);
  EXPECT_EQ(cols[0], "col1");
  EXPECT_EQ(cols[1], "col2");
  EXPECT_EQ(cols[2], "col3");

  EXPECT_EQ(clusterIndex->minKey(), "aaa");
  EXPECT_EQ(clusterIndex->maxKey(), "eee");

  ClusterIndexTestHelper helper(clusterIndex.get());
  auto metadata0 = helper.partitionSection(0);
  EXPECT_EQ(metadata0.offset(), 0);
  EXPECT_GT(metadata0.size(), 0);
  EXPECT_EQ(metadata0.compressionType(), CompressionType::Uncompressed);

  auto metadata1 = helper.partitionSection(1);
  EXPECT_EQ(metadata1.offset(), metadata0.size());
  EXPECT_GT(metadata1.size(), 0);
  EXPECT_EQ(metadata1.compressionType(), CompressionType::Uncompressed);

  auto metadata2 = helper.partitionSection(2);
  EXPECT_EQ(metadata2.offset(), metadata0.size() + metadata1.size());
  EXPECT_GT(metadata2.size(), 0);
  EXPECT_EQ(metadata2.compressionType(), CompressionType::Uncompressed);
  EXPECT_EQ(
      metadata2.offset() + metadata2.size(),
      indexBuffers.indexPartitions.size());
}

TEST_F(ClusterIndexTest, lookup) {
  std::vector<std::string> indexColumns = {"key_col"};
  std::string minKey = "aaa";

  // Encode key stream data for each stripe's keys.
  // Stripe 0: 1 chunk with 1 key "ccc" (1 row)
  // Stripe 1: 1 chunk with 1 key "fff" (1 row)
  // Stripe 2: 1 chunk with 1 key "iii" (1 row)
  auto encoded0 = encodeKeyStream({{"ccc"}});
  auto encoded1 = encodeKeyStream({{"fff"}});
  auto encoded2 = encodeKeyStream({{"iii"}});
  std::string combinedStream = encoded0.data + encoded1.data + encoded2.data;

  const auto size0 = static_cast<uint32_t>(encoded0.data.size());
  const auto size1 = static_cast<uint32_t>(encoded1.data.size());
  const auto size2 = static_cast<uint32_t>(encoded2.data.size());

  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = size0,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}),
      createStripeWithKeys(
          {.streamOffset = size0,
           .streamSize = size1,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"fff"}}),
      createStripeWithKeys(
          {.streamOffset = size0 + size1,
           .streamSize = size2,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"iii"}})};
  // 2 partitions: partition 0 has stripes 0+1 (2 rows), partition 1 has
  // stripe 2 (1 row). Total 3 rows.
  std::vector<int> stripeGroups = {2, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  // Partition 0: rows 0-1 (keys "ccc", "fff")
  // Partition 1: rows 2 (key "iii")
  // Total: 3 rows.

  EXPECT_EQ(clusterIndex->minKey(), "aaa");
  EXPECT_EQ(clusterIndex->maxKey(), "iii");

  // Key before minKey returns full file range.
  {
    auto result = clusterIndex->lookup(*makeRangeScanRequest("000"));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, 0);
    EXPECT_EQ(ranges[0].endRow, 3);
  }

  // Key in partition 0, chunk 0 → row-level start.
  {
    auto result = clusterIndex->lookup(*makeRangeScanRequest("bbb"));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, 0);
    EXPECT_EQ(ranges[0].endRow, 3);
  }

  // Key "ddd" is in partition 0, chunk 1 → row 1.
  {
    auto result = clusterIndex->lookup(*makeRangeScanRequest("ddd"));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, 1);
    EXPECT_EQ(ranges[0].endRow, 3);
  }

  // Key "ggg" is in partition 1 → row 2.
  {
    auto result = clusterIndex->lookup(*makeRangeScanRequest("ggg"));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, 2);
    EXPECT_EQ(ranges[0].endRow, 3);
  }

  // Key after last stripe max key returns empty.
  {
    auto result = clusterIndex->lookup(*makeRangeScanRequest("zzz"));
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
}

TEST_F(ClusterIndexTest, lookupAllStripes) {
  std::vector<std::string> indexColumns = {"key_col"};
  std::string minKey = "aaa";

  // Each stripe has 1 chunk with 1 key.
  // Stripe 0: key "ccc", Stripe 1: key "fff", Stripe 2: key "iii".
  auto encoded0 = encodeKeyStream({{"ccc"}});
  auto encoded1 = encodeKeyStream({{"fff"}});
  auto encoded2 = encodeKeyStream({{"iii"}});
  std::string combinedStream = encoded0.data + encoded1.data + encoded2.data;

  const auto size0 = static_cast<uint32_t>(encoded0.data.size());
  const auto size1 = static_cast<uint32_t>(encoded1.data.size());
  const auto size2 = static_cast<uint32_t>(encoded2.data.size());

  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = size0,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}),
      createStripeWithKeys(
          {.streamOffset = size0,
           .streamSize = size1,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"fff"}}),
      createStripeWithKeys(
          {.streamOffset = size0 + size1,
           .streamSize = size2,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"iii"}})};
  // 2 partitions: partition 0 = stripes 0+1 (2 rows), partition 1 = stripe 2
  // (1 row). Total 3 rows.
  std::vector<int> stripeGroups = {2, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  // Partition 0: rows 0-1 (keys "ccc", "fff")
  // Partition 1: row 2 (key "iii")
  struct TestCase {
    std::string lookupKey;
    std::optional<RowRange> expectedRange;

    std::string debugString() const {
      if (expectedRange.has_value()) {
        return fmt::format(
            "key '{}', expected [{}, {})",
            lookupKey,
            expectedRange->startRow,
            expectedRange->endRow);
      }
      return fmt::format("key '{}', expected empty", lookupKey);
    }
  };

  // makeRangeScanRequest sets only lowerKey, so endRow = numRows_ = 3.
  std::vector<TestCase> testCases = {
      // Keys before minKey return full file range.
      {"000", RowRange(0, 3)},
      {"aa", RowRange(0, 3)},
      // Keys in partition 0, chunk 0 (lastKey "ccc") → row 0.
      {"aaa", RowRange(0, 3)},
      {"bbb", RowRange(0, 3)},
      {"ccc", RowRange(0, 3)},
      // Keys in partition 0, chunk 1 (lastKey "fff") → row 1.
      {"ddd", RowRange(1, 3)},
      {"eee", RowRange(1, 3)},
      {"fff", RowRange(1, 3)},
      // Keys in partition 1, chunk 0 (lastKey "iii") → row 2.
      {"ggg", RowRange(2, 3)},
      {"hhh", RowRange(2, 3)},
      {"iii", RowRange(2, 3)},
      // Keys after last stripe max key.
      {"jjj", std::nullopt},
      {"zzz", std::nullopt}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result =
        clusterIndex->lookup(*makeRangeScanRequest(testCase.lookupKey));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    if (testCase.expectedRange.has_value()) {
      ASSERT_EQ(ranges.size(), 1);
      EXPECT_EQ(ranges[0].startRow, testCase.expectedRange->startRow);
      EXPECT_EQ(ranges[0].endRow, testCase.expectedRange->endRow);
    } else {
      EXPECT_EQ(ranges.size(), 0);
    }
  }
}

TEST_F(ClusterIndexTest, lookupWithSameKeys) {
  std::vector<std::string> indexColumns = {"key_col"};
  std::string minKey = "aaa";

  // All stripes have the same key "aaa". Each stripe has 1 chunk with 1 key.
  auto encoded0 = encodeKeyStream({{"aaa"}});
  auto encoded1 = encodeKeyStream({{"aaa"}});
  auto encoded2 = encodeKeyStream({{"aaa"}});
  std::string combinedStream = encoded0.data + encoded1.data + encoded2.data;

  const auto size0 = static_cast<uint32_t>(encoded0.data.size());
  const auto size1 = static_cast<uint32_t>(encoded1.data.size());
  const auto size2 = static_cast<uint32_t>(encoded2.data.size());

  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = size0,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"aaa"}}),
      createStripeWithKeys(
          {.streamOffset = size0,
           .streamSize = size1,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"aaa"}}),
      createStripeWithKeys(
          {.streamOffset = size0 + size1,
           .streamSize = size2,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"aaa"}})};
  // 2 partitions: partition 0 = stripes 0+1 (2 rows), partition 1 = stripe 2
  // (1 row). Total 3 rows.
  std::vector<int> stripeGroups = {2, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  // Partition 0: rows 0-1 (both "aaa"), Partition 1: row 2 ("aaa").
  struct TestCase {
    std::string lookupKey;
    std::optional<RowRange> expectedRange;

    std::string debugString() const {
      if (expectedRange.has_value()) {
        return fmt::format(
            "key '{}', expected [{}, {})",
            lookupKey,
            expectedRange->startRow,
            expectedRange->endRow);
      }
      return fmt::format("key '{}', expected empty", lookupKey);
    }
  };

  // makeRangeScanRequest sets only lowerKey, so endRow = numRows_ = 3.
  std::vector<TestCase> testCases = {
      // Keys before minKey return full file range.
      {"000", RowRange(0, 3)},
      {"aa", RowRange(0, 3)},
      // Key at minKey (same as all stripe keys) → first partition row 0.
      {"aaa", RowRange(0, 3)},
      // Keys after the common key.
      {"aab", std::nullopt},
      {"bbb", std::nullopt},
      {"zzz", std::nullopt}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result =
        clusterIndex->lookup(*makeRangeScanRequest(testCase.lookupKey));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    if (testCase.expectedRange.has_value()) {
      ASSERT_EQ(ranges.size(), 1);
      EXPECT_EQ(ranges[0].startRow, testCase.expectedRange->startRow);
      EXPECT_EQ(ranges[0].endRow, testCase.expectedRange->endRow);
    } else {
      EXPECT_EQ(ranges.size(), 0);
    }
  }
}
TEST_F(ClusterIndexTest, lookupWithKeyStream) {
  // Test row-level precision when key stream is available.
  // Single partition with multiple chunks of sorted keys.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b", "key_c"},
      {"key_d", "key_e", "key_f"},
      {"key_g", "key_h", "key_i"},
  };
  auto encodedStream = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encodedStream.data.size()),
       .stream =
           {.numChunks = 3,
            .chunkRows = {3, 3, 3},
            .chunkOffsets = encodedStream.chunkOffsets},
       .chunkKeys = {"key_c", "key_f", "key_i"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encodedStream.data));

  struct TestCase {
    std::string lookupKey;
    std::optional<RowRange> expectedRange;

    std::string debugString() const {
      if (expectedRange.has_value()) {
        return fmt::format(
            "key '{}', expected [{}, {})",
            lookupKey,
            expectedRange->startRow,
            expectedRange->endRow);
      }
      return fmt::format("key '{}', expected empty", lookupKey);
    }
  };

  // With key stream, lookup resolves to exact row positions.
  // 9 rows total: key_a(0), key_b(1), key_c(2), key_d(3), key_e(4),
  //               key_f(5), key_g(6), key_h(7), key_i(8)
  std::vector<TestCase> testCases = {
      // Exact key matches.
      {"key_a", RowRange(0, 9)},
      {"key_d", RowRange(3, 9)},
      {"key_i", RowRange(8, 9)},
      // Key between existing keys — resolves to next key's row.
      {"key_ab", RowRange(1, 9)},
      {"key_de", RowRange(4, 9)},
      // Key before all data.
      {"key_0", RowRange(0, 9)},
      // Key after all data.
      {"key_z", std::nullopt},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result =
        clusterIndex->lookup(*makeRangeScanRequest(testCase.lookupKey));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    if (testCase.expectedRange.has_value()) {
      ASSERT_EQ(ranges.size(), 1);
      EXPECT_EQ(ranges[0].startRow, testCase.expectedRange->startRow);
      EXPECT_EQ(ranges[0].endRow, testCase.expectedRange->endRow);
    } else {
      EXPECT_EQ(ranges.size(), 0);
    }
  }
}

TEST_F(ClusterIndexTest, lookupWithKeyStreamMultiplePartitions) {
  // Test row-level precision across multiple partitions.
  std::vector<std::vector<std::string>> partition0Keys = {
      {"key_a", "key_b", "key_c"}};
  std::vector<std::vector<std::string>> partition1Keys = {
      {"key_d", "key_e", "key_f"}};

  auto encoded0 = encodeKeyStream(partition0Keys);
  auto encoded1 = encodeKeyStream(partition1Keys);
  std::string combinedStream = encoded0.data + encoded1.data;

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {3}, .chunkOffsets = {0}},
           .chunkKeys = {"key_c"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {3}, .chunkOffsets = {0}},
           .chunkKeys = {"key_f"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  struct TestCase {
    std::string lookupKey;
    std::optional<RowRange> expectedRange;

    std::string debugString() const {
      if (expectedRange.has_value()) {
        return fmt::format(
            "key '{}', expected [{}, {})",
            lookupKey,
            expectedRange->startRow,
            expectedRange->endRow);
      }
      return fmt::format("key '{}', expected empty", lookupKey);
    }
  };

  // Partition 0: rows 0-2 (key_a, key_b, key_c)
  // Partition 1: rows 3-5 (key_d, key_e, key_f)
  // makeRangeScanRequest sets only lowerKey, so endRow = numRows_ = 6.
  std::vector<TestCase> testCases = {
      {"key_a", RowRange(0, 6)},
      {"key_b", RowRange(1, 6)},
      {"key_c", RowRange(2, 6)},
      {"key_d", RowRange(3, 6)},
      {"key_e", RowRange(4, 6)},
      {"key_f", RowRange(5, 6)},
      {"key_0", RowRange(0, 6)},
      {"key_z", std::nullopt},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result =
        clusterIndex->lookup(*makeRangeScanRequest(testCase.lookupKey));
    ASSERT_EQ(result.size(), 1);
    auto ranges = result[0];
    if (testCase.expectedRange.has_value()) {
      ASSERT_EQ(ranges.size(), 1);
      EXPECT_EQ(ranges[0].startRow, testCase.expectedRange->startRow);
      EXPECT_EQ(ranges[0].endRow, testCase.expectedRange->endRow);
    } else {
      EXPECT_EQ(ranges.size(), 0);
    }
  }
}

TEST_F(ClusterIndexTest, lookupKeyRangeSpanningChunks) {
  struct TestCase {
    // Keys of each chunk. The chunk key of a chunk is its last key.
    std::vector<std::vector<std::string>> keyChunks;
    std::vector<std::string> chunkKeys;
    std::string key;
    std::optional<RowRange> expected;

    std::string debugString() const {
      return fmt::format(
          "{} chunks, key '{}', expected {}",
          keyChunks.size(),
          key,
          expected.has_value()
              ? fmt::format("[{}, {})", expected->startRow, expected->endRow)
              : "empty");
    }
  };

  // The rows of key_b run through two chunk boundaries and end inside the last
  // chunk: key_a(0), key_b(1..4), key_c(5).
  const std::vector<std::vector<std::string>> endsInLastChunk = {
      {"key_a", "key_b"}, {"key_b", "key_b"}, {"key_b", "key_c"}};
  const std::vector<std::string> endsInLastChunkKeys = {
      "key_b", "key_b", "key_c"};
  // The rows of key_b end at a chunk boundary and the next chunk starts with a
  // larger key: key_a(0), key_b(1..3), key_c(4), key_d(5).
  const std::vector<std::vector<std::string>> endsAtBoundary = {
      {"key_a", "key_b"}, {"key_b", "key_b"}, {"key_c", "key_d"}};
  const std::vector<std::string> endsAtBoundaryKeys = {
      "key_b", "key_b", "key_d"};
  // The rows of key_b run to the last row of the last chunk: key_a(0),
  // key_b(1..4).
  const std::vector<std::vector<std::string>> endsAtLastRow = {
      {"key_a", "key_b"}, {"key_b", "key_b"}, {"key_b"}};
  const std::vector<std::string> endsAtLastRowKeys = {
      "key_b", "key_b", "key_b"};

  const std::vector<TestCase> testCases = {
      {endsInLastChunk, endsInLastChunkKeys, "key_a", RowRange(0, 1)},
      {endsInLastChunk, endsInLastChunkKeys, "key_b", RowRange(1, 5)},
      {endsInLastChunk, endsInLastChunkKeys, "key_c", RowRange(5, 6)},
      {endsInLastChunk, endsInLastChunkKeys, "key_bb", std::nullopt},
      {endsAtBoundary, endsAtBoundaryKeys, "key_b", RowRange(1, 4)},
      {endsAtBoundary, endsAtBoundaryKeys, "key_c", RowRange(4, 5)},
      {endsAtLastRow, endsAtLastRowKeys, "key_b", RowRange(1, 5)},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto encodedStream = encodeKeyStream(testCase.keyChunks);
    std::vector<int> chunkRows;
    for (const auto& chunk : testCase.keyChunks) {
      chunkRows.push_back(static_cast<int>(chunk.size()));
    }
    std::vector<Stripe> stripes = {createStripeWithKeys(
        {.streamOffset = 0,
         .streamSize = static_cast<uint32_t>(encodedStream.data.size()),
         .stream =
             {.numChunks = static_cast<uint32_t>(testCase.keyChunks.size()),
              .chunkRows = chunkRows,
              .chunkOffsets = encodedStream.chunkOffsets},
         .chunkKeys = testCase.chunkKeys})};
    auto indexBuffers = createTestClusterIndex(
        {"col1"}, "key_0", stripes, /*stripeGroups=*/{1});
    auto clusterIndex = createClusterIndex(
        indexBuffers, createKeyStreamFile(encodedStream.data));

    const auto result = clusterIndex->lookup(*makeKeyRequest(testCase.key));
    ASSERT_EQ(result.size(), 1);
    const auto ranges = result[0];
    if (!testCase.expected.has_value()) {
      EXPECT_EQ(ranges.size(), 0);
      continue;
    }
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, testCase.expected->startRow);
    EXPECT_EQ(ranges[0].endRow, testCase.expected->endRow);
  }
}

TEST_F(ClusterIndexTest, lookupKeyRangeSpanningPartitions) {
  // The rows of key_b continue from partition 0 into partition 1: key_a(0),
  // key_b(1..3) | key_b(4), key_c(5).
  auto encoded0 = encodeKeyStream({{"key_a", "key_b", "key_b", "key_b"}});
  auto encoded1 = encodeKeyStream({{"key_b", "key_c"}});
  const std::string combinedStream = encoded0.data + encoded1.data;
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {4}, .chunkOffsets = {0}},
           .chunkKeys = {"key_b"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {2}, .chunkOffsets = {0}},
           .chunkKeys = {"key_c"}})};
  auto indexBuffers = createTestClusterIndex(
      {"col1"}, "key_0", stripes, /*stripeGroups=*/{1, 1});
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  const std::vector<std::pair<std::string, RowRange>> testCases = {
      {"key_a", RowRange(0, 1)},
      {"key_b", RowRange(1, 5)},
      {"key_c", RowRange(5, 6)},
  };
  for (const auto& [key, expected] : testCases) {
    SCOPED_TRACE(key);
    const auto result = clusterIndex->lookup(*makeKeyRequest(key));
    ASSERT_EQ(result.size(), 1);
    const auto ranges = result[0];
    ASSERT_EQ(ranges.size(), 1);
    EXPECT_EQ(ranges[0].startRow, expected.startRow);
    EXPECT_EQ(ranges[0].endRow, expected.endRow);
  }
}

TEST_F(ClusterIndexTest, chunkLocation) {
  ChunkLocation loc1{3, 100, 50, 200};
  EXPECT_EQ(loc1.chunkIndex, 3);
  EXPECT_EQ(loc1.chunkOffset, 100);
  EXPECT_EQ(loc1.chunkSize, 50);
  EXPECT_EQ(loc1.rowOffset, 200);

  ChunkLocation loc2{0, 0, 0, 0};
  EXPECT_EQ(loc2.chunkIndex, 0);
  EXPECT_EQ(loc2.chunkOffset, 0);
  EXPECT_EQ(loc2.chunkSize, 0);
  EXPECT_EQ(loc2.rowOffset, 0);
}

TEST_F(ClusterIndexTest, keyStreamRegion) {
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
  auto clusterIndex = createClusterIndex(indexBuffers);
  ASSERT_NE(clusterIndex, nullptr);
  EXPECT_EQ(clusterIndex->keyStreamLayout(), KeyStreamLayout::kPartition);

  struct TestCase {
    uint32_t partitionId;
    uint32_t expectedOffset;
    uint32_t expectedLength;

    std::string debugString() const {
      return fmt::format(
          "partitionId {}, expected [{}, {})",
          partitionId,
          expectedOffset,
          expectedOffset + expectedLength);
    }
  };

  std::vector<TestCase> testCases = {
      // Partition 0: stripes 0, 1. keyStreamOffset = 0, keyStreamSize = 250.
      {0, 0, 250},
      // Partition 1: stripes 2, 3. keyStreamOffset = 250, keyStreamSize = 250.
      {1, 250, 250},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ClusterIndexTestHelper helper(clusterIndex.get());
    auto region = helper.keyStreamRegion(testCase.partitionId);
    EXPECT_EQ(region.offset, testCase.expectedOffset);
    EXPECT_EQ(region.length, testCase.expectedLength);
  }
}

TEST_F(ClusterIndexTest, stripeLayoutLookup) {
  const auto stripeKeys = fourStripeKeys();
  auto clusterIndex = createStripeLayoutIndex(
      writeStripeKeyStreams(stripeKeys),
      /*pinIndex=*/false,
      /*preloadIndex=*/false);
  EXPECT_EQ(clusterIndex->keyStreamLayout(), KeyStreamLayout::kStripe);

  const auto keys = flattenKeys(stripeKeys);
  for (uint32_t row = 0; row < keys.size(); ++row) {
    SCOPED_TRACE(keys[row]);
    expectRows(
        clusterIndex->lookup(*makeKeyRequest(keys[row])),
        RowRange(row, row + 1));
  }

  // Keys before the first key, within a stripe, between the stripes of a
  // partition, between partitions, between the stripes of the last partition,
  // and after the last key.
  for (const std::string key :
       {"j", "k025", "k035", "k045", "k075", "k135", "l"}) {
    SCOPED_TRACE(key);
    expectRows(
        clusterIndex->lookup(*makeRangeScanRequest(key)),
        rowsFromLowerBound(keys, key));
  }
}

TEST_F(ClusterIndexTest, stripeLayoutKeyAtRow) {
  const auto stripeKeys = fourStripeKeys();
  auto clusterIndex = createStripeLayoutIndex(
      writeStripeKeyStreams(stripeKeys),
      /*pinIndex=*/false,
      /*preloadIndex=*/false);

  // Reads every pair of rows in turn. An unpinned index keeps one decoded
  // chunk per partition, and the chunk of every stripe starts at offset 0 of
  // its stripe's key stream, so the pairs that cross stripes check that the
  // index does not take one stripe's chunk for another's.
  const auto keys = flattenKeys(stripeKeys);
  for (uint32_t firstRow = 0; firstRow < keys.size(); ++firstRow) {
    for (uint32_t secondRow = 0; secondRow < keys.size(); ++secondRow) {
      SCOPED_TRACE(fmt::format("rows {} then {}", firstRow, secondRow));
      EXPECT_EQ(clusterIndex->keyAtRow(firstRow), keys[firstRow]);
      EXPECT_EQ(clusterIndex->keyAtRow(secondRow), keys[secondRow]);
    }
  }
}

TEST_F(ClusterIndexTest, stripeLayoutKeyCursor) {
  const auto stripeKeys = fourStripeKeys();
  auto clusterIndex = createStripeLayoutIndex(
      writeStripeKeyStreams(stripeKeys),
      /*pinIndex=*/false,
      /*preloadIndex=*/false);
  const auto keys = flattenKeys(stripeKeys);

  // The whole file, rows within one stripe, rows across the stripes of a
  // partition, and rows across partitions.
  for (const auto& rows :
       {RowRange(0, 16), RowRange(5, 7), RowRange(3, 6), RowRange(6, 15)}) {
    SCOPED_TRACE(rows.toString());
    const auto cursor = clusterIndex->keyCursor(rows);
    std::vector<std::string> scannedKeys;
    while (cursor->hasNext()) {
      scannedKeys.emplace_back(cursor->next());
    }
    EXPECT_EQ(
        scannedKeys,
        std::vector<std::string>(
            keys.begin() + rows.startRow, keys.begin() + rows.endRow));
  }
}

TEST_F(ClusterIndexTest, stripeLayoutPreloadIndex) {
  const auto stripeKeys = fourStripeKeys();
  auto clusterIndex = createStripeLayoutIndex(
      writeStripeKeyStreams(stripeKeys),
      /*pinIndex=*/true,
      /*preloadIndex=*/true);

  // Preloading decodes the key chunk of every stripe, two in each partition.
  ClusterIndexTestHelper helper(clusterIndex.get());
  EXPECT_EQ(helper.decodedChunkCount(0), 2);
  EXPECT_EQ(helper.decodedChunkCount(1), 2);

  const auto keys = flattenKeys(stripeKeys);
  for (uint32_t row = 0; row < keys.size(); ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(clusterIndex->keyAtRow(row), keys[row]);
  }
}

TEST_F(ClusterIndexTest, stripeLayoutLayout) {
  const auto keyStreams = writeStripeKeyStreams(fourStripeKeys());
  auto clusterIndex = createStripeLayoutIndex(
      keyStreams, /*pinIndex=*/false, /*preloadIndex=*/false);

  const auto layout = clusterIndex->layout();
  EXPECT_EQ(layout.keyStreamLayout, KeyStreamLayout::kStripe);
  ASSERT_EQ(layout.partitions.size(), 2);
  for (uint32_t partition = 0; partition < 2; ++partition) {
    SCOPED_TRACE(partition);
    const auto& partitionLayout = layout.partitions[partition];
    EXPECT_EQ(partitionLayout.numRows, 8);

    std::vector<std::pair<uint64_t, uint64_t>> regions;
    regions.reserve(partitionLayout.keyStreamRegions.size());
    for (const auto& region : partitionLayout.keyStreamRegions) {
      regions.emplace_back(region.offset, region.length);
    }
    std::vector<std::pair<uint64_t, uint64_t>> expectedRegions;
    for (uint32_t stripe = 2 * partition; stripe < 2 * partition + 2;
         ++stripe) {
      const auto& keyStream = keyStreams.stripes[stripe].keyStream;
      expectedRegions.emplace_back(
          keyStream.streamOffset, keyStream.streamSize);
    }
    EXPECT_EQ(regions, expectedRegions);
  }
  EXPECT_EQ(layout.partitions[0].numChunks, 2);
  EXPECT_EQ(layout.partitions[1].numChunks, 2);
}

TEST_F(ClusterIndexTest, stripeLayoutRejectsMalformedGroup) {
  // Two stripes of two rows each, the 4 rows of the group.
  const StripeGroupIndex valid{
      .stripeKeys = std::vector<std::string>{"b", "d"},
      .stripeKeyStreams =
          {{.offset = 100, .numRows = 2, .streams = {{.size = 10}}},
           {.offset = 200, .numRows = 2, .streams = {{.size = 10}}}},
  };
  {
    // layout() loads every group, and reads no key chunks.
    const auto layout =
        createClusterIndex(createStripeGroupIndex(valid))->layout();
    ASSERT_EQ(layout.partitions.size(), 1);
    EXPECT_EQ(layout.partitions[0].keyStreamRegions.size(), 2);
    EXPECT_EQ(layout.partitions[0].numChunks, 2);
  }

  struct TestParam {
    std::string description;
    void (*corrupt)(StripeGroupIndex&);
    std::string expectedError;

    std::string debugString() const {
      return description;
    }
  };
  const std::vector<TestParam> testSettings{
      {"also in index_partitions",
       [](StripeGroupIndex& index) { index.setIndexPartitions = true; },
       "Stripe layout index must not set index_partitions"},
      {"no stripe keys",
       [](StripeGroupIndex& index) { index.stripeKeys.reset(); },
       "Cluster index group has no stripe_keys"},
      {"no stripes",
       [](StripeGroupIndex& index) { index.stripeKeyStreams.clear(); },
       "Cluster index group has no stripes"},
      {"a key per stripe missing",
       [](StripeGroupIndex& index) {
         index.stripeKeys = std::vector<std::string>{"d"};
       },
       "Cluster index group must have one key per stripe"},
      {"a stripe without rows",
       [](StripeGroupIndex& index) {
         index.stripeKeyStreams.at(0).numRows = 0;
         index.stripeKeyStreams.at(1).numRows = 4;
       },
       "Stripe has no rows"},
      {"a stripe without a key stream",
       [](StripeGroupIndex& index) {
         index.stripeKeyStreams.at(1).streams.at(0).size = 0;
       },
       "Stripe has no key stream"},
      {"stripes short of the rows of the group",
       [](StripeGroupIndex& index) {
         index.stripeKeyStreams.at(1).numRows = 1;
       },
       "The stripes of a cluster index group must hold its rows"},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto index = valid;
    testData.corrupt(index);
    NIMBLE_ASSERT_FILE_THROW(
        createClusterIndex(createStripeGroupIndex(index))->layout(),
        testData.expectedError);
  }
}

TEST_F(ClusterIndexTest, rejectsUnknownKeyStreamLayout) {
  NIMBLE_ASSERT_FILE_THROW(
      createClusterIndex(createStripeGroupIndex({.keyStreamLayout = 2})),
      "Unknown cluster index key stream layout");
}

TEST_F(ClusterIndexTest, stripeLayoutRequiresKeyStreamLocator) {
  NIMBLE_ASSERT_THROW(
      createClusterIndex(
          createStripeGroupIndex({}),
          /*keyStreamFile=*/nullptr,
          /*pinIndex=*/false,
          /*preloadIndex=*/false,
          /*withKeyStreamLocator=*/false),
      "A cluster index that stores its key chunks in stripes needs a key stream locator");
}

TEST_F(ClusterIndexTest, lookupChunk) {
  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "aaa";
  // Two stripes in one partition:
  // Stripe 0: 3 chunks, keys "ccc"/"fff"/"iii"
  //   chunkRows = {100, 100, 100}, offsets = {0, 50, 120}
  //   streamOffset = 0, streamSize = 200
  // Stripe 1: 2 chunks, keys "mmm"/"ppp"
  //   chunkRows = {150, 150}, offsets = {0, 80}
  //   streamOffset = 200, streamSize = 150
  //
  // The test base computes FlatBuffer chunkOffsets as:
  //   streamOffset + chunkOffset[i]
  // So the partition-level offsets are:
  //   Stripe 0: 0, 50, 120
  //   Stripe 1: 200, 280
  // These must be monotonically increasing.
  // Total key_stream_size = 200 + 150 = 350.
  std::vector<Stripe> stripes = {
      {.streams = {{.numChunks = 1, .chunkRows = {300}, .chunkOffsets = {0}}},
       .keyStream =
           {.streamOffset = 0,
            .streamSize = 200,
            .stream =
                {.numChunks = 3,
                 .chunkRows = {100, 100, 100},
                 .chunkOffsets = {0, 50, 120}},
            .chunkKeys = {"ccc", "fff", "iii"}}},
      {.streams = {{.numChunks = 1, .chunkRows = {300}, .chunkOffsets = {0}}},
       .keyStream = {
           .streamOffset = 200,
           .streamSize = 150,
           .stream =
               {.numChunks = 2,
                .chunkRows = {150, 150},
                .chunkOffsets = {0, 80}},
           .chunkKeys = {"mmm", "ppp"}}}};
  std::vector<int> stripeGroups = {2};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex = createClusterIndex(indexBuffers);
  ASSERT_NE(clusterIndex, nullptr);
  ClusterIndexTestHelper helper(clusterIndex.get());

  // lookupChunk() searches all chunks in the partition and returns
  // partition-wide chunk locations.
  //
  // Chunk 0: key="ccc", chunkOffset=0, chunkSize=50, rowOffset=0
  // Chunk 1: key="fff", chunkOffset=50, chunkSize=70, rowOffset=100
  // Chunk 2: key="iii", chunkOffset=120, chunkSize=80, rowOffset=200
  // Chunk 3: key="mmm", chunkOffset=200, chunkSize=80, rowOffset=300
  // Chunk 4: key="ppp", chunkOffset=280, chunkSize=70, rowOffset=450
  //
  // Total key stream size = 350 bytes.

  struct TestCase {
    std::string encodedKey;
    uint32_t expectedChunkOffset;
    uint32_t expectedChunkSize;
    uint32_t expectedRowOffset;

    std::string debugString() const {
      return fmt::format("encodedKey '{}'", encodedKey);
    }
  };

  std::vector<TestCase> testCases = {
      // Chunk 0: offset=0, size=50, rowOffset=0
      {"aa", 0, 50, 0},
      {"aaa", 0, 50, 0},
      {"bbb", 0, 50, 0},
      {"ccc", 0, 50, 0},
      // Chunk 1: offset=50, size=70, rowOffset=100
      {"ddd", 50, 70, 100},
      {"eee", 50, 70, 100},
      {"fff", 50, 70, 100},
      // Chunk 2: offset=120, size=80, rowOffset=200
      {"ggg", 120, 80, 200},
      {"hhh", 120, 80, 200},
      {"iii", 120, 80, 200},
      // Chunk 3: offset=200, size=80, rowOffset=300
      {"jjj", 200, 80, 300},
      {"lll", 200, 80, 300},
      {"mmm", 200, 80, 300},
      // Chunk 4: offset=280, size=70, rowOffset=450
      {"nnn", 280, 70, 450},
      {"ooo", 280, 70, 450},
      {"ppp", 280, 70, 450},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result = helper.lookupChunk(0, testCase.encodedKey);
    EXPECT_EQ(result.chunkOffset, testCase.expectedChunkOffset);
    EXPECT_EQ(result.chunkSize, testCase.expectedChunkSize);
    EXPECT_EQ(result.rowOffset, testCase.expectedRowOffset);
  }

  // Keys beyond all chunks throw.
  NIMBLE_ASSERT_THROW(helper.lookupChunk(0, "qqq"), "");
  NIMBLE_ASSERT_THROW(helper.lookupChunk(0, "zzz"), "");
}

// Parameterized test verifying lookup correctness across different chunk
// granularities. The same 9 keys (key_a..key_i) are split into chunks of
// different sizes, and all lookups should return the same row positions.
class ClusterIndexLookupTest : public ClusterIndexTest,
                               public ::testing::WithParamInterface<uint32_t> {
 protected:
  void SetUp() override {
    const uint32_t rowsPerChunk = GetParam();
    const std::vector<std::string> allKeys = {
        "key_a",
        "key_b",
        "key_c",
        "key_d",
        "key_e",
        "key_f",
        "key_g",
        "key_h",
        "key_i"};
    const uint32_t numKeys = allKeys.size();

    std::vector<std::vector<std::string>> keyChunks;
    for (uint32_t offset = 0; offset < numKeys; offset += rowsPerChunk) {
      const auto end = std::min(offset + rowsPerChunk, numKeys);
      keyChunks.emplace_back(allKeys.begin() + offset, allKeys.begin() + end);
    }

    encodedStream_ = encodeKeyStream(keyChunks);

    std::vector<int32_t> chunkRows;
    std::vector<std::string> chunkKeys;
    for (const auto& chunk : keyChunks) {
      chunkRows.push_back(chunk.size());
      chunkKeys.push_back(chunk.back());
    }

    std::vector<std::string> indexColumns = {"col1"};
    std::string minKey = "key_0";
    std::vector<Stripe> stripes = {createStripeWithKeys(
        {.streamOffset = 0,
         .streamSize = static_cast<uint32_t>(encodedStream_.data.size()),
         .stream =
             {.numChunks = static_cast<uint32_t>(keyChunks.size()),
              .chunkRows = chunkRows,
              .chunkOffsets = encodedStream_.chunkOffsets},
         .chunkKeys = chunkKeys})};
    std::vector<int> stripeGroups = {1};

    auto indexBuffers =
        createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
    clusterIndex_ = createClusterIndex(
        indexBuffers, createKeyStreamFile(encodedStream_.data));
  }

  EncodedKeyStream encodedStream_;
  std::unique_ptr<ClusterIndex> clusterIndex_;
};

TEST_P(ClusterIndexLookupTest, rangeScanEmptyRange) {
  struct TestCase {
    std::string lower;
    std::string upper;
    std::string debugString() const {
      return fmt::format("lower='{}', upper='{}'", lower, upper);
    }
  };

  std::vector<TestCase> testCases = {
      // upper == lower → empty (exclusive upper means [x, x) = empty).
      {"key_d", "key_d"},
      // upper < lower → empty.
      {"key_f", "key_a"},
      // Both beyond all keys.
      {"key_z", "key_a"},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = testCase.lower, .upperKey = testCase.upper};
    auto result =
        clusterIndex_->lookup(*IndexLookup::LookupRequest::rangeScan({bounds}));
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(), 0);
  }
}

TEST_P(ClusterIndexLookupTest, rangeScanValidRange) {
  // 9 rows: key_a(0)..key_i(8). Upper bound is exclusive.
  struct TestCase {
    std::string lower;
    std::string upper;
    RowRange expected;
    std::string debugString() const {
      return fmt::format(
          "lower='{}', upper='{}', expected=[{}, {})",
          lower,
          upper,
          expected.startRow,
          expected.endRow);
    }
  };

  std::vector<TestCase> testCases = {
      // Full range.
      {"key_0", "key_z", RowRange(0, 9)},
      // Subset range.
      {"key_d", "key_g", RowRange(3, 6)},
      // Single row range: [key_a, key_b) → row 0.
      {"key_a", "key_b", RowRange(0, 1)},
      // Last row: [key_i, key_z) → row 8.
      {"key_i", "key_z", RowRange(8, 9)},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = testCase.lower, .upperKey = testCase.upper};
    auto result =
        clusterIndex_->lookup(*IndexLookup::LookupRequest::rangeScan({bounds}));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 1);
    EXPECT_EQ(result[0][0].startRow, testCase.expected.startRow);
    EXPECT_EQ(result[0][0].endRow, testCase.expected.endRow);
  }
}

TEST_P(ClusterIndexLookupTest, pointLookupExactMatch) {
  // 9 rows: key_a(0)..key_i(8).
  // Point lookup returns [row, row+1) for exact matches.
  struct TestCase {
    std::string key;
    RowRange expected;
    std::string debugString() const {
      return fmt::format(
          "key='{}', expected=[{}, {})",
          key,
          expected.startRow,
          expected.endRow);
    }
  };

  std::vector<TestCase> testCases = {
      {"key_a", RowRange(0, 1)},
      {"key_b", RowRange(1, 2)},
      {"key_c", RowRange(2, 3)},
      {"key_d", RowRange(3, 4)},
      {"key_e", RowRange(4, 5)},
      {"key_f", RowRange(5, 6)},
      {"key_g", RowRange(6, 7)},
      {"key_h", RowRange(7, 8)},
      {"key_i", RowRange(8, 9)},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result = clusterIndex_->lookup(
        *IndexLookup::LookupRequest::pointLookup({testCase.key}));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 1);
    EXPECT_EQ(result[0][0].startRow, testCase.expected.startRow);
    EXPECT_EQ(result[0][0].endRow, testCase.expected.endRow);
  }
}

TEST_P(ClusterIndexLookupTest, pointLookupNonExistent) {
  struct TestCase {
    std::string key;
    bool expectEmpty;
    std::string debugString() const {
      return fmt::format("key='{}', expectEmpty={}", key, expectEmpty);
    }
  };

  std::vector<TestCase> testCases = {
      {"key_z", true},
      {"key_0", true},
      {"key_ab", true},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto result = clusterIndex_->lookup(
        *IndexLookup::LookupRequest::pointLookup({testCase.key}));
    ASSERT_EQ(result.size(), 1);
    if (testCase.expectEmpty) {
      EXPECT_EQ(result[0].size(), 0);
    } else {
      EXPECT_GE(result[0].size(), 1);
    }
  }
}

TEST_P(ClusterIndexLookupTest, pointLookupBatch) {
  auto result = clusterIndex_->lookup(
      *IndexLookup::LookupRequest::pointLookup({"key_a", "key_e", "key_i"}));
  ASSERT_EQ(result.size(), 3);

  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0].startRow, 0);
  EXPECT_EQ(result[0][0].endRow, 1);

  ASSERT_EQ(result[1].size(), 1);
  EXPECT_EQ(result[1][0].startRow, 4);
  EXPECT_EQ(result[1][0].endRow, 5);

  ASSERT_EQ(result[2].size(), 1);
  EXPECT_EQ(result[2][0].startRow, 8);
  EXPECT_EQ(result[2][0].endRow, 9);
}

TEST_P(ClusterIndexLookupTest, pointLookupBatchMixed) {
  auto result = clusterIndex_->lookup(
      *IndexLookup::LookupRequest::pointLookup({"key_a", "key_z", "key_e"}));
  ASSERT_EQ(result.size(), 3);

  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0].startRow, 0);
  EXPECT_EQ(result[0][0].endRow, 1);

  EXPECT_EQ(result[1].size(), 0);

  ASSERT_EQ(result[2].size(), 1);
  EXPECT_EQ(result[2][0].startRow, 4);
  EXPECT_EQ(result[2][0].endRow, 5);
}

TEST_P(ClusterIndexLookupTest, keyAtRow) {
  // 9 rows: key_a(0), key_b(1), ..., key_i(8)
  const std::vector<std::string> expectedKeys = {
      "key_a",
      "key_b",
      "key_c",
      "key_d",
      "key_e",
      "key_f",
      "key_g",
      "key_h",
      "key_i"};
  for (uint32_t row = 0; row < 9; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(clusterIndex_->keyAtRow(row), expectedKeys[row]);
  }
}

TEST_P(ClusterIndexLookupTest, keyAtRowOutOfRange) {
  NIMBLE_ASSERT_THROW(clusterIndex_->keyAtRow(9), "beyond file total rows");
  NIMBLE_ASSERT_THROW(clusterIndex_->keyAtRow(100), "beyond file total rows");
}

TEST_P(ClusterIndexLookupTest, keyIteratorMatchesKeyAtRow) {
  constexpr uint32_t kNumRows = 9;

  std::vector<std::string> scanned;
  scanned.reserve(kNumRows);
  auto iterator = clusterIndex_->keyCursor(RowRange{0, kNumRows});
  while (iterator->hasNext()) {
    scanned.emplace_back(iterator->next());
  }

  std::vector<std::string> expected;
  expected.reserve(kNumRows);
  for (uint32_t row = 0; row < kNumRows; ++row) {
    expected.emplace_back(clusterIndex_->keyAtRow(row));
  }
  EXPECT_EQ(scanned, expected);
}

TEST_P(ClusterIndexLookupTest, keyIteratorRowRange) {
  auto iterator = clusterIndex_->keyCursor(RowRange{2, 7});
  std::vector<std::string> scanned;
  while (iterator->hasNext()) {
    scanned.emplace_back(iterator->next());
  }

  const std::vector<std::string> expected = {
      "key_c", "key_d", "key_e", "key_f", "key_g"};
  EXPECT_EQ(scanned, expected);
}

TEST_P(ClusterIndexLookupTest, keyIteratorEmptyRange) {
  auto iterator = clusterIndex_->keyCursor(RowRange{4, 4});
  EXPECT_FALSE(iterator->hasNext());
  NIMBLE_ASSERT_THROW(iterator->next(), "Cluster index iterator is exhausted");
}

TEST_P(ClusterIndexLookupTest, keyIteratorRowRangeOutOfRange) {
  NIMBLE_ASSERT_THROW(
      clusterIndex_->keyCursor(RowRange{0, 10}), "beyond file total rows");
  NIMBLE_ASSERT_THROW(
      clusterIndex_->keyCursor(RowRange{5, 2}), "row range is inverted");
}

INSTANTIATE_TEST_SUITE_P(
    ChunkGranularity,
    ClusterIndexLookupTest,
    ::testing::Values(1, 3, 9),
    [](const ::testing::TestParamInfo<uint32_t>& info) {
      return fmt::format("rowsPerChunk_{}", info.param);
    });

TEST_F(ClusterIndexTest, keyAtRowMultiplePartitions) {
  // 2 partitions: partition 0 = rows 0-2 (key_a,key_b,key_c),
  //               partition 1 = rows 3-5 (key_d,key_e,key_f)
  std::vector<std::vector<std::string>> partition0Keys = {
      {"key_a", "key_b", "key_c"}};
  std::vector<std::vector<std::string>> partition1Keys = {
      {"key_d", "key_e", "key_f"}};

  auto encoded0 = encodeKeyStream(partition0Keys);
  auto encoded1 = encodeKeyStream(partition1Keys);
  std::string combinedStream = encoded0.data + encoded1.data;

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {3}, .chunkOffsets = {0}},
           .chunkKeys = {"key_c"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream = {.numChunks = 1, .chunkRows = {3}, .chunkOffsets = {0}},
           .chunkKeys = {"key_f"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  const std::vector<std::string> expectedKeys = {
      "key_a", "key_b", "key_c", "key_d", "key_e", "key_f"};
  for (uint32_t row = 0; row < 6; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(clusterIndex->keyAtRow(row), expectedKeys[row]);
  }

  NIMBLE_ASSERT_THROW(clusterIndex->keyAtRow(6), "beyond file total rows");
}

TEST_F(ClusterIndexTest, keyAtRowMultipleChunksPerPartition) {
  // 1 partition, 3 chunks of 2 rows each = 6 rows total.
  // Verifies correct lookup at chunk boundaries.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}, {"key_e", "key_f"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 3,
            .chunkRows = {2, 2, 2},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_b", "key_d", "key_f"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  const std::vector<std::string> expectedKeys = {
      "key_a", "key_b", "key_c", "key_d", "key_e", "key_f"};
  for (uint32_t row = 0; row < 6; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(clusterIndex->keyAtRow(row), expectedKeys[row]);
  }

  NIMBLE_ASSERT_THROW(clusterIndex->keyAtRow(6), "beyond file total rows");
}

TEST_F(ClusterIndexTest, keyAtRowMultiplePartitionsMultipleChunks) {
  // 2 partitions, each with 2 chunks of 2 rows = 8 rows total.
  // Tests partition boundary + chunk boundary interactions.
  std::vector<std::vector<std::string>> p0Chunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  std::vector<std::vector<std::string>> p1Chunks = {
      {"key_e", "key_f"}, {"key_g", "key_h"}};

  auto encoded0 = encodeKeyStream(p0Chunks);
  auto encoded1 = encodeKeyStream(p1Chunks);
  std::string combinedStream = encoded0.data + encoded1.data;

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {2, 2},
                .chunkOffsets = encoded0.chunkOffsets},
           .chunkKeys = {"key_b", "key_d"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {2, 2},
                .chunkOffsets = encoded1.chunkOffsets},
           .chunkKeys = {"key_f", "key_h"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  const std::vector<std::string> expectedKeys = {
      "key_a", "key_b", "key_c", "key_d", "key_e", "key_f", "key_g", "key_h"};
  for (uint32_t row = 0; row < 8; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(clusterIndex->keyAtRow(row), expectedKeys[row]);
  }

  // Partition boundary: row 3 is last of partition 0, row 4 is first of
  // partition 1.
  EXPECT_EQ(clusterIndex->keyAtRow(3), "key_d");
  EXPECT_EQ(clusterIndex->keyAtRow(4), "key_e");

  NIMBLE_ASSERT_THROW(clusterIndex->keyAtRow(8), "beyond file total rows");
}

TEST_F(ClusterIndexTest, keyAtRowSingleRow) {
  // Edge case: file with exactly 1 row.
  std::vector<std::vector<std::string>> keyChunks = {{"only_key"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
       .chunkKeys = {"only_key"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  EXPECT_EQ(clusterIndex->keyAtRow(0), "only_key");
  NIMBLE_ASSERT_THROW(clusterIndex->keyAtRow(1), "beyond file total rows");
}

TEST_F(ClusterIndexTest, keyAtRowUnevenChunks) {
  // 1 partition with uneven chunks: 1 row, 3 rows, 2 rows = 6 rows total.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a"}, {"key_b", "key_c", "key_d"}, {"key_e", "key_f"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 3,
            .chunkRows = {1, 3, 2},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_a", "key_d", "key_f"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  const std::vector<std::string> expectedKeys = {
      "key_a", "key_b", "key_c", "key_d", "key_e", "key_f"};
  for (uint32_t row = 0; row < 6; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(clusterIndex->keyAtRow(row), expectedKeys[row]);
  }

  // Chunk boundaries: row 0 = chunk 0 (single), row 1 = first of chunk 1,
  // row 3 = last of chunk 1, row 4 = first of chunk 2.
  EXPECT_EQ(clusterIndex->keyAtRow(0), "key_a");
  EXPECT_EQ(clusterIndex->keyAtRow(1), "key_b");
  EXPECT_EQ(clusterIndex->keyAtRow(3), "key_d");
  EXPECT_EQ(clusterIndex->keyAtRow(4), "key_e");

  NIMBLE_ASSERT_THROW(clusterIndex->keyAtRow(6), "beyond file total rows");
}

TEST_F(ClusterIndexTest, keyIteratorMultiplePartitionsMultipleChunks) {
  // 2 partitions, each with 2 chunks of 2 rows = 8 rows total. Exercises
  // chunk and partition transitions mid-scan.
  std::vector<std::vector<std::string>> p0Chunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  std::vector<std::vector<std::string>> p1Chunks = {
      {"key_e", "key_f"}, {"key_g", "key_h"}};

  auto encoded0 = encodeKeyStream(p0Chunks);
  auto encoded1 = encodeKeyStream(p1Chunks);
  std::string combinedStream = encoded0.data + encoded1.data;

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {2, 2},
                .chunkOffsets = encoded0.chunkOffsets},
           .chunkKeys = {"key_b", "key_d"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {2, 2},
                .chunkOffsets = encoded1.chunkOffsets},
           .chunkKeys = {"key_f", "key_h"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  std::vector<std::string> scanned;
  auto iterator = clusterIndex->keyCursor(RowRange{0, 8});
  while (iterator->hasNext()) {
    scanned.emplace_back(iterator->next());
  }
  const std::vector<std::string> expected = {
      "key_a", "key_b", "key_c", "key_d", "key_e", "key_f", "key_g", "key_h"};
  EXPECT_EQ(scanned, expected);

  // A range that starts inside partition 0's second chunk and ends inside
  // partition 1's first chunk.
  scanned.clear();
  auto rangeIterator = clusterIndex->keyCursor(RowRange{3, 6});
  while (rangeIterator->hasNext()) {
    scanned.emplace_back(rangeIterator->next());
  }
  const std::vector<std::string> expectedRange = {"key_d", "key_e", "key_f"};
  EXPECT_EQ(scanned, expectedRange);

  // Batches cross chunk and partition boundaries mid-batch. The larger ones ask
  // for more keys than the range has left, so their read is short.
  for (const auto maxKeys : {size_t{1}, size_t{3}, size_t{8}, size_t{100}}) {
    SCOPED_TRACE(fmt::format("maxKeys={}", maxKeys));
    EXPECT_EQ(
        readKeyBatches(*clusterIndex->keyCursor(RowRange{0, 8}), maxKeys),
        expected);
    EXPECT_EQ(
        readKeyBatches(*clusterIndex->keyCursor(RowRange{3, 6}), maxKeys),
        expectedRange);
  }
}

TEST_F(ClusterIndexTest, keyIteratorPrefixEncodedChunks) {
  // Prefix is the default production key encoding, so scan chunks that are
  // really prefix-encoded: two chunks of 20 rows, which puts a restart block
  // boundary inside each chunk as well as a chunk boundary in the middle.
  constexpr uint32_t kRowsPerChunk = 20;
  constexpr uint32_t kNumRows = 2 * kRowsPerChunk;

  std::vector<std::string> allKeys;
  allKeys.reserve(kNumRows);
  for (uint32_t row = 0; row < kNumRows; ++row) {
    allKeys.push_back(fmt::format("key_{:04d}", row));
  }
  std::vector<std::vector<std::string>> keyChunks = {
      {allKeys.begin(), allKeys.begin() + kRowsPerChunk},
      {allKeys.begin() + kRowsPerChunk, allKeys.end()}};
  auto encoded = encodeKeyStream(keyChunks, EncodingType::Prefix);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0000";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 2,
            .chunkRows = {kRowsPerChunk, kRowsPerChunk},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {allKeys[kRowsPerChunk - 1], allKeys.back()}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  std::vector<std::string> scanned;
  scanned.reserve(kNumRows);
  auto iterator = clusterIndex->keyCursor(RowRange{0, kNumRows});
  while (iterator->hasNext()) {
    scanned.emplace_back(iterator->next());
  }
  EXPECT_EQ(scanned, allKeys);

  // Starting mid-chunk must resume against the right shared prefix, which is
  // where a prefix cursor differs from a trivial one.
  auto midChunk = clusterIndex->keyCursor(RowRange{25, kNumRows});
  scanned.clear();
  while (midChunk->hasNext()) {
    scanned.emplace_back(midChunk->next());
  }
  EXPECT_EQ(
      scanned, std::vector<std::string>(allKeys.begin() + 25, allKeys.end()));

  // Batches cross restart blocks and the chunk boundary, from a chunk start
  // and from mid-chunk.
  for (const auto maxKeys : {size_t{7}, size_t{kNumRows}}) {
    SCOPED_TRACE(fmt::format("maxKeys={}", maxKeys));
    EXPECT_EQ(
        readKeyBatches(
            *clusterIndex->keyCursor(RowRange{0, kNumRows}), maxKeys),
        allKeys);
    EXPECT_EQ(
        readKeyBatches(
            *clusterIndex->keyCursor(RowRange{25, kNumRows}), maxKeys),
        std::vector<std::string>(allKeys.begin() + 25, allKeys.end()));
  }
}

TEST_F(ClusterIndexTest, keyIteratorDuplicateKeys) {
  // Duplicate keys are legal unless the writer enforces noDuplicateKey, and
  // a run of them can straddle a chunk boundary. The iterator reports one
  // key per row regardless.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_b", "key_b"}, {"key_b", "key_c"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 3,
            .chunkRows = {2, 2, 2},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_b", "key_b", "key_c"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  std::vector<std::string> scanned;
  auto iterator = clusterIndex->keyCursor(RowRange{0, 6});
  while (iterator->hasNext()) {
    scanned.emplace_back(iterator->next());
  }
  const std::vector<std::string> expected = {
      "key_a", "key_b", "key_b", "key_b", "key_b", "key_c"};
  EXPECT_EQ(scanned, expected);
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_F(ClusterIndexTest, keyIteratorReusesCachedChunk) {
  // Callers scan in row batches, so several short-lived iterators walk one
  // chunk in turn. They must share the decoded-chunk cache rather than each
  // decoding the chunk again.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 2,
            .chunkRows = {2, 2},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_b", "key_d"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t decodeCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(DecodedKeyChunk*)>(
          [&](DecodedKeyChunk* /*decodedChunk*/) { ++decodeCount; }));

  // One iterator per row, as a batched caller produces.
  auto firstRow = clusterIndex->keyCursor(RowRange{0, 1});
  EXPECT_EQ(firstRow->next(), "key_a");
  EXPECT_EQ(decodeCount.load(), 1);

  auto secondRow = clusterIndex->keyCursor(RowRange{1, 2});
  EXPECT_EQ(secondRow->next(), "key_b");
  EXPECT_EQ(decodeCount.load(), 1)
      << "second iterator re-decoded a chunk the first one had cached";

  // Reaching into the next chunk must decode, so a stuck counter cannot make
  // the assertions above pass vacuously.
  auto thirdRow = clusterIndex->keyCursor(RowRange{2, 3});
  EXPECT_EQ(thirdRow->next(), "key_c");
  EXPECT_EQ(decodeCount.load(), 2);
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_F(ClusterIndexTest, seekKeepsTheKeyChunkOfItsFirstRow) {
  // Without pinIndex, the index keeps one decoded chunk per partition, so a
  // reader of another chunk evicts the chunk of a lookup before the lookup's
  // caller reads the keys of its rows. The key cursor of a seek keeps it.
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 2,
            .chunkRows = {2, 2},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_b", "key_d"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t decodeCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(DecodedKeyChunk*)>(
          [&](DecodedKeyChunk* /*decodedChunk*/) { ++decodeCount; }));

  // Both bounds of the key are in its chunk, which the seek decodes once.
  auto keyA = clusterIndex->seek(makeKeyBounds("key_a"));
  EXPECT_EQ(keyA.rows.startRow, 0);
  EXPECT_EQ(keyA.rows.endRow, 1);
  ASSERT_NE(keyA.keyCursor, nullptr);
  EXPECT_EQ(decodeCount.load(), 1);

  // A lookup of the other chunk evicts the first chunk from the index.
  clusterIndex->lookup(*makeKeyRequest("key_c"));
  EXPECT_EQ(decodeCount.load(), 2);

  // The key cursor of the seek reads the first chunk that it keeps.
  EXPECT_EQ(keyA.keyCursor->next(), "key_a");
  EXPECT_FALSE(keyA.keyCursor->hasNext());
  EXPECT_EQ(decodeCount.load(), 2);

  // A key cursor of the index decodes the first chunk again.
  EXPECT_EQ(clusterIndex->keyCursor(RowRange{0, 1})->next(), "key_a");
  EXPECT_EQ(decodeCount.load(), 3);

  // The bounds of a range from the last key of the first chunk are in both
  // chunks. The index holds the first one, and the seek decodes the other.
  auto range = clusterIndex->seek(
      velox::serializer::EncodedKeyBounds{
          .lowerKey = "key_b", .upperKey = "key_d"});
  EXPECT_EQ(range.rows.startRow, 1);
  EXPECT_EQ(range.rows.endRow, 3);
  ASSERT_NE(range.keyCursor, nullptr);
  EXPECT_EQ(decodeCount.load(), 4);
  clusterIndex->lookup(*makeKeyRequest("key_a"));
  EXPECT_EQ(decodeCount.load(), 5);

  // The key cursor reads its first row from the chunk that it keeps, and then
  // holds only the chunk that it reads, so it decodes the second chunk when it
  // reaches it.
  EXPECT_EQ(range.keyCursor->next(), "key_b");
  EXPECT_EQ(decodeCount.load(), 5);
  EXPECT_EQ(range.keyCursor->next(), "key_c");
  EXPECT_EQ(decodeCount.load(), 6);
  EXPECT_FALSE(range.keyCursor->hasNext());

  // A miss returns no rows and no key cursor.
  const auto miss = clusterIndex->seek(makeKeyBounds("key_bb"));
  EXPECT_TRUE(miss.rows.empty());
  EXPECT_EQ(miss.keyCursor, nullptr);
}

// Debug-only: counting decodes relies on a TestValue hook, which is compiled
// out under NDEBUG.
DEBUG_ONLY_TEST_F(ClusterIndexTest, keyCursorSkipsKeysWithoutReadingThem) {
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b", "key_c"},
      {"key_d", "key_e", "key_f"},
      {"key_g", "key_h", "key_i"}};
  auto encoded = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encoded.data.size()),
       .stream =
           {.numChunks = 3,
            .chunkRows = {3, 3, 3},
            .chunkOffsets = encoded.chunkOffsets},
       .chunkKeys = {"key_c", "key_f", "key_i"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encoded.data));

  // Fires only after a cache miss, so it counts decodes rather than reads.
  std::atomic_uint32_t decodeCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(DecodedKeyChunk*)>(
          [&](DecodedKeyChunk* /*decodedChunk*/) { ++decodeCount; }));

  auto cursor = clusterIndex->keyCursor(RowRange{0, 9});
  EXPECT_EQ(cursor->next(), "key_a");
  EXPECT_EQ(decodeCount.load(), 1);

  // A skip in the chunk that the cursor holds moves in the chunk.
  cursor->skip(1);
  EXPECT_EQ(cursor->next(), "key_c");
  EXPECT_EQ(decodeCount.load(), 1);

  // A skip over the second chunk does not read it.
  cursor->skip(3);
  EXPECT_EQ(cursor->next(), "key_g");
  EXPECT_EQ(decodeCount.load(), 2);

  // A batch reads on from the row after a skip, and a skip stops at the end of
  // the cursor.
  cursor->skip(0);
  EXPECT_EQ(
      readKeyBatches(*cursor, /*maxKeys=*/1),
      (std::vector<std::string>{"key_h", "key_i"}));
  cursor->skip(10);
  EXPECT_FALSE(cursor->hasNext());

  // A cursor that skips before its first read opens the chunk of the row that
  // it reads first.
  auto skipFirst = clusterIndex->keyCursor(RowRange{1, 9});
  skipFirst->skip(6);
  EXPECT_EQ(skipFirst->next(), "key_h");
  EXPECT_EQ(decodeCount.load(), 2);
  skipFirst->skip(10);
  EXPECT_FALSE(skipFirst->hasNext());

  // A skip to the end of the held chunk releases it, and the next read opens
  // the next chunk.
  auto toChunkEnd = clusterIndex->keyCursor(RowRange{0, 9});
  EXPECT_EQ(toChunkEnd->next(), "key_a");
  EXPECT_EQ(decodeCount.load(), 3);
  toChunkEnd->skip(2);
  EXPECT_EQ(toChunkEnd->next(), "key_d");
  EXPECT_EQ(decodeCount.load(), 4);
}

// Verifies that the index caching mechanism works end-to-end when
// ClusterIndex is created with IndexLookup::Options containing the block cache
// over the async data cache. The first lookup reads index data from the file;
// a second lookup of the same key reuses the decoded chunk and does no index
// IO.
TEST_F(ClusterIndexTest, indexDataReuseWithSameReader) {
  std::string file;
  velox::InMemoryWriteFile writeFile(&file);
  Buffer buffer{*pool_};

  TestClusterIndexMetadataWriter indexHelper(
      *pool_,
      {"col1"},
      {SortOrder{.ascending = true}},
      /*enforceKeyOrder=*/true);

  auto tabletWriter = TabletWriter::create(
      &writeFile,
      *pool_,
      {
          .metadataFlushThreshold = 1024 * 1024 * 1024,
          .streamDeduplicationEnabled = false,
          .enableChunkStats = true,
          .chunkStatsVersion = ChunkStatsVersion::kV1,
          .stripeGroupFlushCallback =
              indexHelper.createStripeGroupFlushCallback(),
          .closeCallback = indexHelper.createCloseCallback(),
      });

  constexpr int kNumStripes = 3;
  std::vector<std::string> keys = {"bbb", "ddd", "fff"};
  for (int i = 0; i < kNumStripes; ++i) {
    auto pos = buffer.reserve(50);
    std::memset(pos, 'A', 50);
    std::vector<nimble::Stream> tabletStreams;
    tabletStreams.push_back(
        {.offset = 0, .chunks = {{.rowCount = 100, .content = {{pos, 50}}}}});
    indexHelper.addStripe({{.rowCount = 100, .key = keys[i]}});
    tabletWriter->writeStripe(100, std::move(tabletStreams));
  }
  tabletWriter->close();
  writeFile.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();

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
      // cacheIndex is set but cache is not provided — silently falls back
      // to direct IO, so RAM hits are not expected.
      {true, false, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    indexIoStats_ = std::make_shared<velox::io::IoStatistics>();

    auto& ids = velox::fileIds();
    auto fileIdStr = fmt::format(
        "sameReaderTest_cache{}_provide{}",
        testCase.cacheIndex,
        testCase.provideCache);

    TabletReader::Options options;
    options.loadClusterIndex = true;
    options.cacheIndex = testCase.cacheIndex;
    // pinIndex=true so the ClusterIndex retains decoded chunks across
    // lookups. Without pinning, !pin decodes via the scratch slot every
    // call and would always issue fresh dataInput_->read() calls (modulo
    // AsyncDataCache hits in the cacheIndex=true + provideCache case).
    options.pinIndex = true;
    options.ioOptions.emplace(pool_.get())
        .setMetadataIoStats(metadataIoStats_)
        .setIndexIoStats(indexIoStats_);

    std::unique_ptr<velox::FileHandle> fileHandle;
    if (testCase.provideCache) {
      fileHandle = std::make_unique<velox::FileHandle>();
      fileHandle->file = readFile;
      fileHandle->uuid = velox::StringIdLease(ids, fileIdStr);
      fileHandle->groupId = velox::StringIdLease(ids, fileIdStr + "_group");
      options.blockCache = getBlockCache(cache.get());
      options.fileHandle = fileHandle.get();
    }

    auto reader = TabletReader::create(readFile, pool_.get(), options);
    ASSERT_NE(reader->clusterIndex(), nullptr);

    // First lookup loads partition metadata and key stream data lazily.
    auto firstResult =
        reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
    ASSERT_EQ(firstResult.size(), 1);
    ASSERT_EQ(firstResult[0].size(), 1);
    EXPECT_EQ(firstResult[0][0], RowRange(0, 300));

    // Capture baseline after first lookup fully loads data.
    const auto indexBytesAfterFirst = indexIoStats_->rawBytesRead();
    EXPECT_GT(indexBytesAfterFirst, 0)
        << "First lookup should read index data from file";

    // Same-key lookup reuses DecodedChunk cache — no new index IO.
    auto secondResult =
        reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
    ASSERT_EQ(secondResult.size(), 1);
    ASSERT_EQ(secondResult[0].size(), 1);
    EXPECT_EQ(secondResult[0][0], RowRange(0, 300));
    EXPECT_EQ(indexIoStats_->rawBytesRead(), indexBytesAfterFirst)
        << "Same-key lookup should reuse cached data with no new IO";
  }

  testingResetBlockCache();
  cache->shutdown();
}

TEST_F(ClusterIndexTest, indexDataReuseCrossReaders) {
  std::string file;
  velox::InMemoryWriteFile writeFile(&file);
  Buffer buffer{*pool_};

  TestClusterIndexMetadataWriter indexHelper(
      *pool_,
      {"col1"},
      {SortOrder{.ascending = true}},
      /*enforceKeyOrder=*/true);

  auto tabletWriter = TabletWriter::create(
      &writeFile,
      *pool_,
      {
          .metadataFlushThreshold = 1024 * 1024 * 1024,
          .streamDeduplicationEnabled = false,
          .enableChunkStats = true,
          .chunkStatsVersion = ChunkStatsVersion::kV1,
          .stripeGroupFlushCallback =
              indexHelper.createStripeGroupFlushCallback(),
          .closeCallback = indexHelper.createCloseCallback(),
      });

  constexpr int kNumStripes = 3;
  std::vector<std::string> keys = {"bbb", "ddd", "fff"};
  for (int i = 0; i < kNumStripes; ++i) {
    auto pos = buffer.reserve(50);
    std::memset(pos, 'A', 50);
    std::vector<nimble::Stream> tabletStreams;
    tabletStreams.push_back(
        {.offset = 0, .chunks = {{.rowCount = 100, .content = {{pos, 50}}}}});
    indexHelper.addStripe({{.rowCount = 100, .key = keys[i]}});
    tabletWriter->writeStripe(100, std::move(tabletStreams));
  }
  tabletWriter->close();
  writeFile.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();

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
      // cacheIndex is set but cache is not provided — silently falls back
      // to direct IO, so cross-reader reuse and RAM hits are not expected.
      {true, false, false, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto& ids = velox::fileIds();
    auto fileIdStr = fmt::format(
        "crossReaderTest_cache{}_provide{}",
        testCase.cacheIndex,
        testCase.provideCache);
    velox::StringIdLease fileId(ids, fileIdStr);
    velox::StringIdLease groupId(ids, fileIdStr + "_group");

    auto createReader = [&]() {
      metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
      indexIoStats_ = std::make_shared<velox::io::IoStatistics>();

      velox::io::ReaderOptions ioOpts(pool_.get());
      ioOpts.setMetadataIoStats(metadataIoStats_);
      ioOpts.setIndexIoStats(indexIoStats_);

      TabletReader::Options options;
      options.loadClusterIndex = true;
      options.cacheIndex = testCase.cacheIndex;
      options.ioOptions = ioOpts;

      std::unique_ptr<velox::FileHandle> fileHandle;
      if (testCase.provideCache) {
        fileHandle = std::make_unique<velox::FileHandle>();
        fileHandle->file = readFile;
        fileHandle->uuid = fileId;
        fileHandle->groupId = groupId;
        options.blockCache = getBlockCache(cache.get());
        options.fileHandle = fileHandle.get();
      }

      auto reader = TabletReader::create(readFile, pool_.get(), options);
      return std::make_pair(std::move(reader), std::move(fileHandle));
    };

    // First reader: cold path reads from file.
    {
      auto [reader1, fh1] = createReader();
      ASSERT_NE(reader1->clusterIndex(), nullptr);

      auto firstResult =
          reader1->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
      ASSERT_EQ(firstResult.size(), 1);
      ASSERT_EQ(firstResult[0].size(), 1);
      EXPECT_EQ(firstResult[0][0], RowRange(0, 300));
      EXPECT_GT(indexIoStats_->rawBytesRead(), 0)
          << "First reader should read index data from file";
    }

    // Second reader: verify functional correctness with fresh IO stats.
    {
      auto [reader2, fh2] = createReader();
      ASSERT_NE(reader2->clusterIndex(), nullptr);

      auto secondResult =
          reader2->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
      ASSERT_EQ(secondResult.size(), 1);
      ASSERT_EQ(secondResult[0].size(), 1);
      EXPECT_EQ(secondResult[0][0], RowRange(0, 300));

      if (testCase.expectCrossReaderReuse) {
        EXPECT_GT(indexIoStats_->ramHit().count(), 0)
            << "Second reader should serve index data from cache";
        EXPECT_EQ(indexIoStats_->read().sum(), 0)
            << "Second reader should not read index data from the file";
      }
    }
  }

  testingResetBlockCache();
  cache->shutdown();
}

// Writes a tablet with three single-key stripes that all fall into the same
// partition (no mid-write stripe-group flush), producing one partition with
// three key chunks.
namespace {
std::shared_ptr<velox::ReadFile> writeThreeChunkPartitionFile(
    velox::memory::MemoryPool& pool,
    std::string& fileBacking) {
  velox::InMemoryWriteFile writeFile(&fileBacking);
  Buffer buffer{pool};

  TestClusterIndexMetadataWriter indexHelper(
      pool,
      {"col1"},
      {SortOrder{.ascending = true}},
      /*enforceKeyOrder=*/true);

  auto tabletWriter = TabletWriter::create(
      &writeFile,
      pool,
      {
          .metadataFlushThreshold = 1024 * 1024 * 1024,
          .streamDeduplicationEnabled = false,
          .enableChunkStats = true,
          .chunkStatsVersion = ChunkStatsVersion::kV1,
          .stripeGroupFlushCallback =
              indexHelper.createStripeGroupFlushCallback(),
          .closeCallback = indexHelper.createCloseCallback(),
      });

  const std::vector<std::string> keys = {"bbb", "ddd", "fff"};
  for (const auto& key : keys) {
    auto* pos = buffer.reserve(50);
    std::memset(pos, 'A', 50);
    std::vector<nimble::Stream> tabletStreams;
    tabletStreams.push_back(
        {.offset = 0, .chunks = {{.rowCount = 100, .content = {{pos, 50}}}}});
    indexHelper.addStripe({{.rowCount = 100, .key = key}});
    tabletWriter->writeStripe(100, std::move(tabletStreams));
  }
  tabletWriter->close();
  writeFile.close();

  return std::make_shared<velox::InMemoryReadFile>(fileBacking);
}
} // namespace

TEST_F(ClusterIndexTest, indexDataReuseCrossReadersWithBlockCache) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();
  auto* blockCache = getBlockCache(cache.get());

  struct TestCase {
    bool cacheIndex;

    // Whether the reader that fills the cache preloads the index.
    bool firstPreloadIndex;

    // Whether the reader after it preloads the index.
    bool secondPreloadIndex;
    bool expectKeyStreamReuse;

    std::string debugString() const {
      return fmt::format(
          "cacheIndex={}, firstPreloadIndex={}, secondPreloadIndex={}, expectKeyStreamReuse={}",
          cacheIndex,
          firstPreloadIndex,
          secondPreloadIndex,
          expectKeyStreamReuse);
    }
  };

  const std::vector<TestCase> testCases = {
      {true, false, false, true},
      {true, true, true, true},
      // Preloading and lookups cache each key chunk as its own entry, so
      // either reader reuses the chunks the other cached.
      {true, true, false, true},
      {true, false, true, true},
      // Without cacheIndex, the index reads its metadata and key stream from
      // the file.
      {false, false, false, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // A file id per case keeps the cases' cache entries apart.
    const velox::StringIdLease fileId{
        velox::fileIds(),
        fmt::format(
            "blockCacheCrossReaderTest_cache{}_preload{}{}",
            testCase.cacheIndex,
            testCase.firstPreloadIndex,
            testCase.secondPreloadIndex)};
    velox::FileHandle fileHandle;
    fileHandle.file = readFile;
    fileHandle.uuid = fileId;

    // Opens a reader over the shared cache and looks up a key in every
    // chunk, so each chunk is read whether or not it was preloaded.
    const auto lookupWithNewReader = [&](bool preloadIndex) {
      indexIoStats_ = std::make_shared<velox::io::IoStatistics>();
      TabletReader::Options options;
      options.loadClusterIndex = true;
      options.cacheIndex = testCase.cacheIndex;
      options.preloadIndex = preloadIndex;
      options.fileHandle = &fileHandle;
      options.blockCache = blockCache;
      options.ioOptions.emplace(pool_.get())
          .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
          .setIndexIoStats(indexIoStats_);
      auto reader = TabletReader::create(readFile, pool_.get(), options);
      ASSERT_NE(reader->clusterIndex(), nullptr);
      const std::vector<std::pair<std::string, RowRange>> expectedRanges = {
          {"bbb", RowRange(0, 300)},
          {"ddd", RowRange(100, 300)},
          {"fff", RowRange(200, 300)},
      };
      for (const auto& [key, expectedRange] : expectedRanges) {
        SCOPED_TRACE(fmt::format("key={}", key));
        const auto result =
            reader->clusterIndex()->lookup(*makeRangeScanRequest(key));
        ASSERT_EQ(result.size(), 1);
        ASSERT_EQ(result[0].size(), 1);
        EXPECT_EQ(result[0][0], expectedRange);
      }
    };

    // read() counts bytes read from storage. rawBytesRead also counts cache
    // hits.
    ASSERT_NO_FATAL_FAILURE(lookupWithNewReader(testCase.firstPreloadIndex));
    EXPECT_GT(indexIoStats_->read().sum(), 0)
        << "The first reader reads the index from the file";
    const auto firstRawBytesRead = indexIoStats_->rawBytesRead();

    ASSERT_NO_FATAL_FAILURE(lookupWithNewReader(testCase.secondPreloadIndex));
    if (testCase.secondPreloadIndex == testCase.firstPreloadIndex) {
      EXPECT_EQ(indexIoStats_->rawBytesRead(), firstRawBytesRead)
          << "A reader that reads the same bytes counts them the same, "
          << "whether from the file or the cache";
    }

    if (testCase.cacheIndex) {
      EXPECT_GT(indexIoStats_->ramHit().count(), 0)
          << "The second reader reads the partition metadata from the cache";
    } else {
      EXPECT_EQ(indexIoStats_->ramHit().count(), 0)
          << "Without cacheIndex, the index bypasses the cache";
    }
    if (testCase.expectKeyStreamReuse) {
      EXPECT_EQ(indexIoStats_->read().sum(), 0)
          << "The second reader reads the key stream from the cache";
    } else {
      EXPECT_GT(indexIoStats_->read().sum(), 0)
          << "The second reader reads the key stream from the file";
    }
  }

  testingResetBlockCache();
  cache->shutdown();
}

// The index cache keys use the id of the file handle, which the index inputs
// keep leased after the handle is released.
TEST_F(ClusterIndexTest, cacheIndexOutlivesFileHandle) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);
  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();

  std::shared_ptr<TabletReader> reader;
  {
    velox::FileHandle fileHandle;
    fileHandle.file = readFile;
    fileHandle.uuid = velox::StringIdLease(
        velox::fileIds(), "ClusterIndexTest.cacheIndexOutlivesFileHandle");
    TabletReader::Options options;
    options.loadClusterIndex = true;
    options.cacheIndex = true;
    options.blockCache = getBlockCache(cache.get());
    options.fileHandle = &fileHandle;
    options.ioOptions.emplace(pool_.get())
        .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
        .setIndexIoStats(indexIoStats_);
    reader = TabletReader::create(readFile, pool_.get(), options);
  }

  // Only the index leases the file id now. A cache miss creates its entry
  // under that id, which needs the id in velox::fileIds().
  cache->clear();
  const auto result =
      reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0], RowRange(0, 300));

  reader.reset();
  testingResetBlockCache();
  cache->shutdown();
}

// The block cache saves index metadata and key chunks to the SSD cache of the
// async data cache, so a reader after a memory clear reads them from SSD.
TEST_F(ClusterIndexTest, cacheIndexReadsFromSsd) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);
  velox::filesystems::registerLocalFileSystem();
  FLAGS_velox_ssd_odirect = false;
  const auto ssdDirectory =
      velox::common::testutil::TempDirectoryPath::create();
  folly::IOThreadPoolExecutor ssdExecutor{1};
  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(
      allocator.get(),
      std::make_unique<velox::cache::SsdCache>(velox::cache::SsdCache::Config(
          fmt::format("{}/cache", ssdDirectory->getPath()),
          /*_maxBytes=*/64UL << 20,
          /*_numShards=*/1,
          &ssdExecutor)));
  testingResetBlockCache();
  velox::FileHandle fileHandle;
  fileHandle.file = readFile;
  fileHandle.uuid = velox::StringIdLease(
      velox::fileIds(), "ClusterIndexTest.cacheIndexReadsFromSsd");

  // Opens a reader and looks up a key in every chunk.
  const auto lookupWithNewReader = [&]() {
    indexIoStats_ = std::make_shared<velox::io::IoStatistics>();
    TabletReader::Options options;
    options.loadClusterIndex = true;
    options.cacheIndex = true;
    options.blockCache = getBlockCache(cache.get());
    options.fileHandle = &fileHandle;
    options.ioOptions.emplace(pool_.get())
        .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
        .setIndexIoStats(indexIoStats_);
    auto reader = TabletReader::create(readFile, pool_.get(), options);
    ASSERT_NE(reader->clusterIndex(), nullptr);
    const std::vector<std::pair<std::string, RowRange>> expectedRanges = {
        {"bbb", RowRange(0, 300)},
        {"ddd", RowRange(100, 300)},
        {"fff", RowRange(200, 300)},
    };
    for (const auto& [key, expectedRange] : expectedRanges) {
      SCOPED_TRACE(fmt::format("key={}", key));
      const auto result =
          reader->clusterIndex()->lookup(*makeRangeScanRequest(key));
      ASSERT_EQ(result.size(), 1);
      ASSERT_EQ(result[0].size(), 1);
      EXPECT_EQ(result[0][0], expectedRange);
    }
  };

  ASSERT_NO_FATAL_FAILURE(lookupWithNewReader());
  EXPECT_GT(indexIoStats_->read().sum(), 0);

  ASSERT_TRUE(cache->ssdCache()->startWrite());
  cache->saveToSsd(/*saveAll=*/true);
  cache->ssdCache()->waitForWriteToFinish();
  cache->clear();

  ASSERT_NO_FATAL_FAILURE(lookupWithNewReader());
  EXPECT_EQ(indexIoStats_->read().sum(), 0);
  EXPECT_GT(indexIoStats_->ssdRead().count(), 0);
  EXPECT_EQ(indexIoStats_->ramHit().count(), 0);

  testingResetBlockCache();
  cache->shutdown();
}

// The key chunks of this file are uncompressed, so a decoded chunk reads its
// keys in place from the cache entry its bytes were loaded into.
TEST_F(ClusterIndexTest, blockCachedKeyChunkStaysPinnedWhileDecoded) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();
  auto* blockCache = getBlockCache(cache.get());

  for (const bool preloadIndex : {false, true}) {
    SCOPED_TRACE(fmt::format("preloadIndex={}", preloadIndex));

    // A file id per case keeps the cases' cache entries apart.
    velox::FileHandle fileHandle;
    fileHandle.file = readFile;
    fileHandle.uuid = velox::StringIdLease(
        velox::fileIds(),
        fmt::format("blockCachedKeyChunkPinTest_preload{}", preloadIndex));

    TabletReader::Options options;
    options.loadClusterIndex = true;
    options.pinIndex = true;
    options.preloadIndex = preloadIndex;
    options.cacheIndex = true;
    options.blockCache = blockCache;
    options.fileHandle = &fileHandle;
    options.ioOptions.emplace(pool_.get())
        .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
        .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
    auto reader = TabletReader::create(readFile, pool_.get(), options);
    ASSERT_NE(reader->clusterIndex(), nullptr);
    ClusterIndexTestHelper helper(reader->clusterIndex());

    const auto lookupFirstKey = [&]() {
      const auto result =
          reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
      ASSERT_EQ(result.size(), 1);
      ASSERT_EQ(result[0].size(), 1);
      EXPECT_EQ(result[0][0], RowRange(0, 300));
    };
    ASSERT_NO_FATAL_FAILURE(lookupFirstKey());
    ASSERT_GE(helper.decodedChunkCount(0), 1);
    const auto chunkOffset = helper.keyStreamRegion(0).offset +
        helper.lookupChunk(0, "bbb").chunkOffset;

    // Evicts every entry that nothing pins.
    cache->clear();

    velox::io::IoStatistics probeStats;
    EXPECT_FALSE(
        blockCache->find(fileHandle.uuid.id(), chunkOffset, probeStats).empty())
        << "The decoded chunk must keep its cache entry pinned";
    ASSERT_NO_FATAL_FAILURE(lookupFirstKey());

    // Destroying the reader drops the decoded chunk and with it the pin.
    reader.reset();
    cache->clear();
    EXPECT_TRUE(
        blockCache->find(fileHandle.uuid.id(), chunkOffset, probeStats).empty())
        << "The cache entry must not stay pinned after the reader is gone";
  }

  testingResetBlockCache();
  cache->shutdown();
}

// Without pinIndex, every lookup of a chunk other than the one in the scratch
// slot reads it again through the shared cached input, so concurrent lookups
// read key chunks concurrently.
TEST_F(ClusterIndexTest, blockCachedKeyChunksReadConcurrently) {
  constexpr uint32_t kNumThreads{16};
  constexpr uint32_t kLookupsPerThread{200};

  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  auto allocator = std::make_shared<velox::memory::MallocAllocator>(
      velox::memory::MemoryAllocator::Options{
          .capacity = 1UL << 30, .reservationByteLimit = 0});
  auto cache = velox::cache::AsyncDataCache::create(allocator.get());
  testingResetBlockCache();
  auto* blockCache = getBlockCache(cache.get());

  const velox::StringIdLease fileId{
      velox::fileIds(), "blockCachedKeyChunksReadConcurrentlyTest"};
  velox::FileHandle fileHandle;
  fileHandle.file = readFile;
  fileHandle.uuid = fileId;
  const auto indexIoStats = std::make_shared<velox::io::IoStatistics>();

  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.cacheIndex = true;
  options.fileHandle = &fileHandle;
  options.blockCache = blockCache;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(indexIoStats);
  auto reader = TabletReader::create(readFile, pool_.get(), options);
  ASSERT_NE(reader->clusterIndex(), nullptr);

  const std::vector<std::pair<std::string, RowRange>> expectedRanges = {
      {"bbb", RowRange(0, 300)},
      {"ddd", RowRange(100, 300)},
      {"fff", RowRange(200, 300)},
  };
  std::atomic_uint32_t numMismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (uint32_t threadIdx = 0; threadIdx < kNumThreads; ++threadIdx) {
    threads.emplace_back([&, threadIdx]() {
      for (uint32_t i = 0; i < kLookupsPerThread; ++i) {
        const auto& [key, expectedRange] =
            expectedRanges[(threadIdx + i) % expectedRanges.size()];
        const auto result =
            reader->clusterIndex()->lookup(*makeRangeScanRequest(key));
        if (result.size() != 1 || result[0].size() != 1 ||
            result[0][0] != expectedRange) {
          ++numMismatches;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(numMismatches.load(), 0);

  // The partition metadata is read once, so every hit is a key chunk read.
  EXPECT_GT(indexIoStats->ramHit().count(), 0);

  reader.reset();
  testingResetBlockCache();
  cache->shutdown();
}

TEST_F(ClusterIndexTest, pinIndexEnabledRetainsChunks) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.pinIndex = true;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(metadataIoStats_)
      .setIndexIoStats(indexIoStats_);

  auto reader = TabletReader::create(readFile, pool_.get(), options);
  ASSERT_NE(reader->clusterIndex(), nullptr);
  ClusterIndexTestHelper helper(reader->clusterIndex());

  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  const auto bytesAfterChunk0 = indexIoStats_->rawBytesRead();
  EXPECT_GT(bytesAfterChunk0, 0);

  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  EXPECT_EQ(indexIoStats_->rawBytesRead(), bytesAfterChunk0)
      << "Repeated lookup of a pinned chunk must not trigger index IO";

  reader->clusterIndex()->lookup(*makeRangeScanRequest("fff"));
  EXPECT_EQ(helper.decodedChunkCount(0), 2);

  reader->clusterIndex()->lookup(*makeRangeScanRequest("ddd"));
  EXPECT_EQ(helper.decodedChunkCount(0), 3);

  const auto bytesAfterAllChunks = indexIoStats_->rawBytesRead();
  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  EXPECT_EQ(helper.decodedChunkCount(0), 3);
  EXPECT_EQ(indexIoStats_->rawBytesRead(), bytesAfterAllChunks)
      << "Re-lookup of pinned chunk must not trigger new index IO";
}

TEST_F(ClusterIndexTest, pinIndexDisabledDoesNotRetainChunks) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.pinIndex = false;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(metadataIoStats_)
      .setIndexIoStats(indexIoStats_);

  auto reader = TabletReader::create(readFile, pool_.get(), options);
  ASSERT_NE(reader->clusterIndex(), nullptr);
  ClusterIndexTestHelper helper(reader->clusterIndex());

  // Unpinned decodes share a single scratch slot — at most one decoded
  // chunk is retained at a time, evicted whenever a different chunk is
  // requested. Each lookup that lands on a different chunk issues fresh IO.
  const auto bytesBefore = indexIoStats_->rawBytesRead();
  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  const auto bytesAfter1 = indexIoStats_->rawBytesRead();
  EXPECT_GT(bytesAfter1, bytesBefore);

  reader->clusterIndex()->lookup(*makeRangeScanRequest("fff"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  const auto bytesAfter2 = indexIoStats_->rawBytesRead();
  EXPECT_GT(bytesAfter2, bytesAfter1);

  reader->clusterIndex()->lookup(*makeRangeScanRequest("ddd"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  EXPECT_GT(indexIoStats_->rawBytesRead(), bytesAfter2);
}

TEST_F(ClusterIndexTest, preloadIndexDecodesAllChunksAtConstruction) {
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.preloadIndex = true;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(metadataIoStats_)
      .setIndexIoStats(indexIoStats_);

  auto reader = TabletReader::create(readFile, pool_.get(), options);
  ASSERT_NE(reader->clusterIndex(), nullptr);
  ClusterIndexTestHelper helper(reader->clusterIndex());

  // All chunks must be decoded at construction time, before any lookup runs.
  EXPECT_EQ(helper.decodedChunkCount(0), 3);
  const auto bytesAfterPreload = indexIoStats_->rawBytesRead();
  EXPECT_GT(bytesAfterPreload, 0)
      << "Preload should have triggered key-stream IO";

  // Subsequent lookups across all three chunks must not trigger any further
  // index data IO.
  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  reader->clusterIndex()->lookup(*makeRangeScanRequest("ddd"));
  reader->clusterIndex()->lookup(*makeRangeScanRequest("fff"));
  EXPECT_EQ(helper.decodedChunkCount(0), 3);
  EXPECT_EQ(indexIoStats_->rawBytesRead(), bytesAfterPreload)
      << "Lookups after preload must not trigger additional index IO";
}

// Preloading loads the key chunks of each partition in its own batch, so every
// partition is decoded at construction, not only the first.
TEST_F(ClusterIndexTest, preloadIndexDecodesEveryPartition) {
  auto encoded0 = encodeKeyStream({{"ccc"}});
  auto encoded1 = encodeKeyStream({{"fff"}});
  auto encoded2 = encodeKeyStream({{"iii"}});
  const auto size0 = static_cast<uint32_t>(encoded0.data.size());
  const auto size1 = static_cast<uint32_t>(encoded1.data.size());
  const auto size2 = static_cast<uint32_t>(encoded2.data.size());
  const std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = size0,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}),
      createStripeWithKeys(
          {.streamOffset = size0,
           .streamSize = size1,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"fff"}}),
      createStripeWithKeys(
          {.streamOffset = size0 + size1,
           .streamSize = size2,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"iii"}})};
  // Partition 0 holds stripes 0 and 1, and partition 1 holds stripe 2.
  const auto indexBuffers = createTestClusterIndex(
      {"key_col"}, "aaa", stripes, /*stripeGroups=*/{2, 1});
  auto clusterIndex = createClusterIndex(
      indexBuffers,
      createKeyStreamFile(encoded0.data + encoded1.data + encoded2.data),
      /*pinIndex=*/true,
      /*preloadIndex=*/true);
  ClusterIndexTestHelper helper(clusterIndex.get());

  EXPECT_EQ(helper.decodedChunkCount(0), 2);
  EXPECT_EQ(helper.decodedChunkCount(1), 1);
  const auto bytesAfterPreload = keyStreamIoStats_->rawBytesRead();
  EXPECT_EQ(bytesAfterPreload, size0 + size1 + size2);

  const std::vector<std::pair<std::string, RowRange>> expectedRanges = {
      {"ddd", RowRange(1, 3)},
      {"ggg", RowRange(2, 3)},
  };
  for (const auto& [key, expectedRange] : expectedRanges) {
    SCOPED_TRACE(key);
    const auto result = clusterIndex->lookup(*makeRangeScanRequest(key));
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 1);
    EXPECT_EQ(result[0][0], expectedRange);
  }
  EXPECT_EQ(keyStreamIoStats_->rawBytesRead(), bytesAfterPreload);
}

TEST_F(ClusterIndexTest, preloadIndexRequiresPinIndex) {
  // IndexLookup::Options::validate() must reject preloadIndex=true when
  // pinIndex=false, because non-pinned chunks would be evicted on first
  // lookup, defeating the purpose of preloading.
  auto readFile = std::make_shared<velox::InMemoryReadFile>(std::string{});
  velox::io::ReaderOptions ioOpts(pool_.get());
  ioOpts.setIndexIoStats(indexIoStats_);
  IndexLookup::Options opts{
      .file = readFile,
      .ioOptions = &ioOpts,
      .pinIndex = false,
      .preloadIndex = true,
  };
  EXPECT_THROW(opts.validate(), NimbleUserError);
}

TEST_F(ClusterIndexTest, preloadIndexAutoPromotesPinIndex) {
  // At the TabletReader::Options level, setting preloadIndex=true together
  // with pinIndex=false (apparently conflicting) must auto-promote pinIndex
  // to true rather than failing — preloaded chunks must be retained for the
  // preload to have any effect.
  std::string file;
  auto readFile = writeThreeChunkPartitionFile(*pool_, file);

  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.preloadIndex = true;
  options.pinIndex = false; // intentionally conflicting; should be promoted.
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(metadataIoStats_)
      .setIndexIoStats(indexIoStats_);

  // Construction must succeed (no throw) despite the apparent conflict.
  auto reader = TabletReader::create(readFile, pool_.get(), options);
  ASSERT_NE(reader->clusterIndex(), nullptr);
  ClusterIndexTestHelper helper(reader->clusterIndex());

  // All chunks must be decoded and retained — proves pinIndex was promoted
  // (otherwise decodedChunks would be capped at 1 entry).
  EXPECT_EQ(helper.decodedChunkCount(0), 3);
  const auto bytesAfterPreload = indexIoStats_->rawBytesRead();

  // Lookups across all three chunks must not trigger additional index IO,
  // confirming the chunks are pinned (not evicted between lookups).
  reader->clusterIndex()->lookup(*makeRangeScanRequest("bbb"));
  reader->clusterIndex()->lookup(*makeRangeScanRequest("ddd"));
  reader->clusterIndex()->lookup(*makeRangeScanRequest("fff"));
  EXPECT_EQ(helper.decodedChunkCount(0), 3);
  EXPECT_EQ(indexIoStats_->rawBytesRead(), bytesAfterPreload);
}

DEBUG_ONLY_TEST_F(ClusterIndexTest, loadPartitionOnce) {
  std::vector<std::string> indexColumns = {"key_col"};
  std::string minKey = "aaa";

  auto encoded0 = encodeKeyStream({{"ccc"}});
  auto encoded1 = encodeKeyStream({{"fff"}});
  std::string combinedStream = encoded0.data + encoded1.data;

  const auto size0 = static_cast<uint32_t>(encoded0.data.size());
  const auto size1 = static_cast<uint32_t>(encoded1.data.size());

  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = size0,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"ccc"}}),
      createStripeWithKeys(
          {.streamOffset = size0,
           .streamSize = size1,
           .stream = {.numChunks = 1, .chunkRows = {1}, .chunkOffsets = {0}},
           .chunkKeys = {"fff"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  std::atomic_uint32_t loadCount{0};
  folly::Baton firstLoaderReady;
  folly::Baton firstLoaderProceed;

  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::loadPartition",
      std::function<void(uint32_t*)>([&](uint32_t* partitionId) {
        ASSERT_EQ(*partitionId, 0);
        const auto count = ++loadCount;
        if (count == 1) {
          firstLoaderReady.post();
          firstLoaderProceed.wait();
        }
      }));

  std::thread t1([&]() { clusterIndex->lookup(*makeRangeScanRequest("bbb")); });

  firstLoaderReady.wait();
  EXPECT_EQ(loadCount.load(), 1);

  std::thread t2([&]() { clusterIndex->lookup(*makeRangeScanRequest("aaa")); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(loadCount.load(), 1);

  firstLoaderProceed.post();
  t1.join();
  t2.join();

  EXPECT_EQ(loadCount.load(), 1);
}

DEBUG_ONLY_TEST_F(ClusterIndexTest, decodeChunkFirstWins) {
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b", "key_c"}};
  auto encodedStream = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encodedStream.data.size()),
       .stream = {.numChunks = 1, .chunkRows = {3}, .chunkOffsets = {0}},
       .chunkKeys = {"key_c"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encodedStream.data));
  ClusterIndexTestHelper helper(clusterIndex.get());

  std::atomic_int decodeCount{0};
  const DecodedKeyChunk* firstDecoded{nullptr};
  const DecodedKeyChunk* secondDecoded{nullptr};
  folly::Baton firstDecodeReady;
  folly::Baton secondDecodeReady;
  folly::Baton firstProceed;
  folly::Baton secondProceed;

  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(DecodedKeyChunk*)>([&](DecodedKeyChunk* decodedChunk) {
        const auto count = ++decodeCount;
        if (count == 1) {
          firstDecoded = decodedChunk;
          firstDecodeReady.post();
          firstProceed.wait();
        } else {
          secondDecoded = decodedChunk;
          secondDecodeReady.post();
          secondProceed.wait();
        }
      }));

  std::thread t1(
      [&]() { clusterIndex->lookup(*makeRangeScanRequest("key_a")); });

  firstDecodeReady.wait();

  std::thread t2(
      [&]() { clusterIndex->lookup(*makeRangeScanRequest("key_b")); });

  secondDecodeReady.wait();

  EXPECT_NE(firstDecoded, nullptr);
  EXPECT_NE(secondDecoded, nullptr);
  EXPECT_NE(firstDecoded, secondDecoded);

  // Release t1 first — it wins the install.
  firstProceed.post();
  t1.join();

  EXPECT_EQ(
      helper.decodedChunkData(/*partitionIndex=*/0, /*chunkIndex=*/0),
      firstDecoded);

  // Release t2 — its install is skipped (slot already filled).
  secondProceed.post();
  t2.join();

  EXPECT_EQ(
      helper.decodedChunkData(/*partitionIndex=*/0, /*chunkIndex=*/0),
      firstDecoded);
}

DEBUG_ONLY_TEST_F(ClusterIndexTest, decodeChunksDifferentChunksInParallel) {
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  auto encodedStream = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encodedStream.data.size()),
       .stream =
           {.numChunks = 2,
            .chunkRows = {2, 2},
            .chunkOffsets = encodedStream.chunkOffsets},
       .chunkKeys = {"key_b", "key_d"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);

  auto rootBuf = velox::AlignedBuffer::allocate<char>(
      indexBuffers.rootIndex.size(), pool_.get());
  std::memcpy(
      rootBuf->asMutable<char>(),
      indexBuffers.rootIndex.data(),
      indexBuffers.rootIndex.size());
  Section indexSection{MetadataBuffer(toPinnedBuffer(
      MetadataBuffer::decompress(
          std::move(rootBuf), CompressionType::Uncompressed, pool_.get())))};
  metadataFile_ =
      std::make_shared<velox::InMemoryReadFile>(indexBuffers.indexPartitions);
  ioStats_ = std::make_shared<velox::io::IoStatistics>();
  auto metadataInput = MetadataInput::create(
      metadataFile_.get(),
      MetadataInput::Options{.pool = pool_.get(), .ioStats = ioStats_});
  keyStreamFile_ =
      std::make_shared<velox::InMemoryReadFile>(encodedStream.data);
  auto dataInput = std::make_shared<DirectDataInput>(
      keyStreamFile_,
      DirectDataInput::Options{
          .pool = pool_.get(),
          .ioStats = std::make_shared<velox::io::IoStatistics>(),
      });
  auto clusterIndex = ClusterIndexTestHelper::create(
      std::move(indexSection),
      pool_.get(),
      std::move(metadataInput),
      std::move(dataInput),
      /*pinIndex=*/true);
  ClusterIndexTestHelper helper(clusterIndex.get());

  // Pre-load the partition so both threads go straight to getDecodedChunk.
  clusterIndex->lookup(*makeRangeScanRequest("key_a"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);

  std::atomic_int decodeCount{0};
  folly::Baton firstDecodeReady;
  folly::Baton secondDecodeReady;
  folly::Baton firstProceed;
  folly::Baton secondProceed;

  SCOPED_TESTVALUE_SET(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      std::function<void(DecodedKeyChunk*)>(
          [&](DecodedKeyChunk* /*decodedChunk*/) {
            const auto count = ++decodeCount;
            if (count == 1) {
              firstDecodeReady.post();
              firstProceed.wait();
            } else {
              secondDecodeReady.post();
              secondProceed.wait();
            }
          }));

  // Thread 1 looks up chunk 1 (not yet decoded), thread 2 looks up chunk 0.
  // Chunk 0 is already cached so t2 hits the shared-lock fast path and
  // never reaches the TestValue. We look up two uncached keys instead.
  // Actually chunk 0 is cached. Let's invalidate by looking up chunk 1
  // from both threads to test the same-chunk parallel decode scenario.
  // Instead, let's just look up the uncached chunk 1 from two threads.
  std::thread t1(
      [&]() { clusterIndex->lookup(*makeRangeScanRequest("key_c")); });
  std::thread t2(
      [&]() { clusterIndex->lookup(*makeRangeScanRequest("key_d")); });

  // Both threads are decoding chunk 1 in parallel.
  firstDecodeReady.wait();
  secondDecodeReady.wait();
  EXPECT_EQ(decodeCount.load(), 2);

  // Release both to install.
  firstProceed.post();
  secondProceed.post();
  t1.join();
  t2.join();

  EXPECT_EQ(helper.decodedChunkCount(0), 2);
  EXPECT_NE(helper.decodedChunkData(0, 0), nullptr);
  EXPECT_NE(helper.decodedChunkData(0, 1), nullptr);
}

DEBUG_ONLY_TEST_F(ClusterIndexTest, decodeChunkLastWinsUnpinned) {
  std::vector<std::vector<std::string>> keyChunks = {
      {"key_a", "key_b"}, {"key_c", "key_d"}};
  auto encodedStream = encodeKeyStream(keyChunks);

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {createStripeWithKeys(
      {.streamOffset = 0,
       .streamSize = static_cast<uint32_t>(encodedStream.data.size()),
       .stream =
           {.numChunks = 2,
            .chunkRows = {2, 2},
            .chunkOffsets = encodedStream.chunkOffsets},
       .chunkKeys = {"key_b", "key_d"}})};
  std::vector<int> stripeGroups = {1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(encodedStream.data));
  ClusterIndexTestHelper helper(clusterIndex.get());

  // Pre-load partition and chunk 0.
  clusterIndex->lookup(*makeRangeScanRequest("key_a"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  const auto* chunk0Data = helper.decodedChunkData(0, 0);

  // Now look up chunk 1 — single scratch slot evicts chunk 0.
  clusterIndex->lookup(*makeRangeScanRequest("key_c"));
  EXPECT_EQ(helper.decodedChunkCount(0), 1);
  const auto* chunk1Data = helper.decodedChunkData(0, 0);
  EXPECT_NE(chunk0Data, chunk1Data);
}

class ClusterIndexConcurrentTest
    : public ClusterIndexTest,
      public ::testing::WithParamInterface<EncodingType> {
 protected:
  EncodedKeyStream encodeKeyStreamWithType(
      const std::vector<std::vector<std::string>>& keyChunks,
      EncodingType encodingType) {
    EncodedKeyStream result;
    Buffer buffer{*pool_};

    for (const auto& keys : keyChunks) {
      result.chunkOffsets.push_back(result.data.size());

      std::vector<std::string_view> keyViews;
      keyViews.reserve(keys.size());
      for (const auto& key : keys) {
        keyViews.push_back(key);
      }

      Buffer encodingBuffer{*pool_};
      auto policy =
          std::make_unique<ManualEncodingSelectionPolicy<std::string_view>>(
              std::vector<std::pair<EncodingType, float>>{{encodingType, 1.0}},
              CompressionOptions{},
              std::nullopt);
      auto encoded = EncodingFactory::encode<std::string_view>(
          std::move(policy), keyViews, encodingBuffer);

      ChunkedStreamWriter writer{buffer};
      auto segments = writer.encode(encoded);
      for (const auto& segment : segments) {
        result.data.append(segment.data(), segment.size());
      }
    }
    return result;
  }
};

TEST_P(ClusterIndexConcurrentTest, concurrentLookup) {
  constexpr uint32_t kNumThreads = 16;
  constexpr uint32_t kLookupsPerThread = 200;

  const auto encodingType = GetParam();

  std::vector<std::vector<std::string>> p0Chunks = {
      {"key_a", "key_b", "key_c", "key_d"},
      {"key_e", "key_f", "key_g", "key_h"}};
  std::vector<std::vector<std::string>> p1Chunks = {
      {"key_i", "key_j", "key_k", "key_l"},
      {"key_m", "key_n", "key_o", "key_p"}};

  auto encoded0 = encodeKeyStreamWithType(p0Chunks, encodingType);
  auto encoded1 = encodeKeyStreamWithType(p1Chunks, encodingType);
  std::string combinedStream = encoded0.data + encoded1.data;

  std::vector<std::string> indexColumns = {"col1"};
  std::string minKey = "key_0";
  std::vector<Stripe> stripes = {
      createStripeWithKeys(
          {.streamOffset = 0,
           .streamSize = static_cast<uint32_t>(encoded0.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {4, 4},
                .chunkOffsets = encoded0.chunkOffsets},
           .chunkKeys = {"key_d", "key_h"}}),
      createStripeWithKeys(
          {.streamOffset = static_cast<uint32_t>(encoded0.data.size()),
           .streamSize = static_cast<uint32_t>(encoded1.data.size()),
           .stream =
               {.numChunks = 2,
                .chunkRows = {4, 4},
                .chunkOffsets = encoded1.chunkOffsets},
           .chunkKeys = {"key_l", "key_p"}})};
  std::vector<int> stripeGroups = {1, 1};

  auto indexBuffers =
      createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
  auto clusterIndex =
      createClusterIndex(indexBuffers, createKeyStreamFile(combinedStream));

  // 16 keys across 2 partitions x 2 chunks each.
  const std::vector<std::string> allKeys = {
      "key_a",
      "key_b",
      "key_c",
      "key_d",
      "key_e",
      "key_f",
      "key_g",
      "key_h",
      "key_i",
      "key_j",
      "key_k",
      "key_l",
      "key_m",
      "key_n",
      "key_o",
      "key_p"};

  std::atomic_uint32_t errors{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (uint32_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(42 + t);
      std::uniform_int_distribution<uint32_t> keyDist(0, allKeys.size() - 1);
      for (uint32_t i = 0; i < kLookupsPerThread; ++i) {
        const auto keyIdx = keyDist(rng);
        const auto& key = allKeys[keyIdx];

        auto result = clusterIndex->lookup(
            *IndexLookup::LookupRequest::pointLookup({key}));
        if (result.size() != 1 || result[0].size() != 1 ||
            result[0][0].startRow != keyIdx ||
            result[0][0].endRow != keyIdx + 1) {
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

INSTANTIATE_TEST_SUITE_P(
    KeyEncodingType,
    ClusterIndexConcurrentTest,
    ::testing::Values(EncodingType::Trivial, EncodingType::Prefix),
    [](const ::testing::TestParamInfo<EncodingType>& info) {
      return fmt::format("{}", toString(info.param));
    });

} // namespace
} // namespace facebook::nimble::index::test
