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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/ChunkStats.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"

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
      return MetadataSection(
          0,
          static_cast<uint32_t>(metadata.size()),
          CompressionType::Uncompressed);
    };
  }

  static auto writeRootCallback(
      TestChunkFileIndex& fileIndex,
      std::string_view expectedSectionName = nimble::kChunkStatsSection) {
    return [&fileIndex, expectedSectionName](
               const std::string& name, std::string_view content) {
      EXPECT_EQ(name, expectedSectionName);
      fileIndex.rootIndexData = std::string(content);
    };
  }

  static std::string_view sectionName(ChunkStatsVersion version) {
    return version == ChunkStatsVersion::kV1 ? nimble::kChunkStatsSection
                                             : nimble::kChunkStatsV2Section;
  }

  std::vector<uint32_t> decode(
      const serialization::EncodedStream& encodedStream) {
    const auto* data = encodedStream.data();
    NIMBLE_CHECK_NOT_NULL(data);
    auto encoding = EncodingFactory{}.create(
        *pool_,
        std::string_view{
            reinterpret_cast<const char*>(data->data()), data->size()},
        nullptr);
    const auto rowCount = encoding->rowCount();
    std::vector<uint32_t> values(rowCount);
    encoding->materialize(rowCount, values.data());
    return values;
  }

  std::unique_ptr<MetadataBuffer> copyMetadata(const std::string& data) {
    auto inputBuffer =
        velox::AlignedBuffer::allocate<char>(data.size(), pool_.get());
    std::memcpy(inputBuffer->asMutable<char>(), data.data(), data.size());
    return std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
        std::move(inputBuffer), CompressionType::Uncompressed, pool_.get()));
  }

  static std::vector<uint8_t> encodeTrivial(std::span<const uint32_t> values) {
    std::vector<uint8_t> encoded(
        EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) +
        values.size_bytes());
    auto* position = reinterpret_cast<char*>(encoded.data());
    EncodingPrefix::serialize(
        EncodingType::Trivial,
        DataType::Uint32,
        static_cast<uint32_t>(values.size()),
        /*useVarint=*/false,
        position);
    encoding::writeChar(
        static_cast<char>(CompressionType::Uncompressed), position);
    for (const auto value : values) {
      encoding::write(value, position);
    }
    return encoded;
  }

  template <typename T>
  static std::vector<uint8_t> encodeConstant(T value, uint32_t rowCount) {
    std::vector<uint8_t> encoded(EncodingPrefix::kFixedPrefixSize + sizeof(T));
    auto* position = reinterpret_cast<char*>(encoded.data());
    EncodingPrefix::serialize(
        EncodingType::Constant,
        TypeTraits<T>::dataType,
        rowCount,
        /*useVarint=*/false,
        position);
    std::memcpy(position, &value, sizeof(T));
    return encoded;
  }

  static std::vector<uint8_t> encodeConstantString(
      std::string_view value,
      uint32_t rowCount) {
    std::vector<uint8_t> encoded(
        EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t) + value.size());
    auto* position = reinterpret_cast<char*>(encoded.data());
    EncodingPrefix::serialize(
        EncodingType::Constant,
        DataType::String,
        rowCount,
        /*useVarint=*/false,
        position);
    encoding::writeString(value, position);
    return encoded;
  }

  static std::string createV2ConstantBoundsData(uint32_t chunkCount) {
    flatbuffers::FlatBufferBuilder builder;
    const auto createEncodedStream = [&](const std::vector<uint8_t>& encoded) {
      return serialization::CreateEncodedStream(
          builder, builder.CreateVector(encoded));
    };
    const auto rows =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/1, chunkCount));
    const auto offsets =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const auto nullCounts =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const auto mins =
        createEncodedStream(encodeConstant<int64_t>(/*value=*/1, chunkCount));
    const auto maxs =
        createEncodedStream(encodeConstant<int64_t>(/*value=*/2, chunkCount));
    const std::vector<uint32_t> chunkCounts{chunkCount};
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        minStreams{mins};
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        maxStreams{maxs};
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            builder.CreateVector(chunkCounts),
            rows,
            offsets,
            nullCounts,
            builder.CreateVector(minStreams),
            builder.CreateVector(maxStreams)));
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  static std::string createV2ConstantStringBoundsData(
      std::string_view min,
      std::string_view max,
      uint32_t chunkCount) {
    flatbuffers::FlatBufferBuilder builder;
    const auto createEncodedStream = [&](const std::vector<uint8_t>& encoded) {
      return serialization::CreateEncodedStream(
          builder, builder.CreateVector(encoded));
    };
    const auto rows =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/1, chunkCount));
    const auto offsets =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const auto nullCounts =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const auto mins =
        createEncodedStream(encodeConstantString(min, chunkCount));
    const auto maxs =
        createEncodedStream(encodeConstantString(max, chunkCount));
    const auto presence =
        createEncodedStream(encodeConstant<bool>(true, chunkCount));
    const std::vector<uint32_t> chunkCounts{chunkCount};
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        minStreams{mins};
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        maxStreams{maxs};
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            builder.CreateVector(chunkCounts),
            rows,
            offsets,
            nullCounts,
            builder.CreateVector(minStreams),
            builder.CreateVector(maxStreams),
            presence));
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  static std::string createV2BoundsData(
      std::span<const uint8_t> min,
      std::span<const uint8_t> max,
      std::span<const uint8_t> presence,
      uint32_t chunkCount = 1) {
    flatbuffers::FlatBufferBuilder builder;
    const auto createEncodedStream = [&](std::span<const uint8_t> encoded) {
      return encoded.empty()
          ? serialization::CreateEncodedStream(builder)
          : serialization::CreateEncodedStream(
                builder, builder.CreateVector(encoded.data(), encoded.size()));
    };
    const auto rows =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/1, chunkCount));
    const auto offsets =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const auto nullCounts =
        createEncodedStream(encodeConstant<uint32_t>(/*value=*/0, chunkCount));
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        minStreams{createEncodedStream(min)};
    const std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        maxStreams{createEncodedStream(max)};
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            builder.CreateVector(std::vector<uint32_t>{chunkCount}),
            rows,
            offsets,
            nullCounts,
            builder.CreateVector(minStreams),
            builder.CreateVector(maxStreams),
            createEncodedStream(presence)));
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  static std::string createV2GroupDataFromEncoded(
      std::span<const uint32_t> chunkCounts,
      std::span<const uint8_t> chunkRows,
      std::span<const uint8_t> chunkOffsets,
      std::span<const uint8_t> chunkNullCounts,
      uint32_t streamCount = 1) {
    flatbuffers::FlatBufferBuilder builder;
    const auto createEncodedStream = [&](std::span<const uint8_t> encoded) {
      return serialization::CreateEncodedStream(
          builder, builder.CreateVector(encoded.data(), encoded.size()));
    };
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            streamCount,
            builder.CreateVector(chunkCounts.data(), chunkCounts.size()),
            createEncodedStream(chunkRows),
            createEncodedStream(chunkOffsets),
            createEncodedStream(chunkNullCounts)));
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  static std::string createV2GroupData(
      std::span<const uint32_t> chunkCounts,
      std::span<const uint32_t> chunkRows,
      std::span<const uint32_t> chunkOffsets,
      std::span<const uint32_t> chunkNullCounts) {
    const auto encodedRows = encodeTrivial(chunkRows);
    const auto encodedOffsets = encodeTrivial(chunkOffsets);
    const auto encodedNullCounts = encodeTrivial(chunkNullCounts);
    return createV2GroupDataFromEncoded(
        chunkCounts, encodedRows, encodedOffsets, encodedNullCounts);
  }

  // Creates a ChunkStatsGroup reader from a serialized group metadata section.
  std::shared_ptr<index::ChunkStatsGroup> loadChunkStats(
      const std::string& groupData,
      uint32_t firstStripe,
      uint32_t stripeCount) {
    return index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV1,
        firstStripe,
        stripeCount,
        copyMetadata(groupData),
        *pool_);
  }

  ChunkStatsWriter& createWriter() {
    return createWriter(2);
  }

  ChunkStatsWriter& createWriter(float minAvgChunksPerStream) {
    return createWriter(ChunkStatsVersion::kV1, minAvgChunksPerStream);
  }

  ChunkStatsWriter& createWriter(
      ChunkStatsVersion version,
      float minAvgChunksPerStream = 2) {
    writer_ = ChunkStatsWriter::create(
        *pool_,
        {
            .version = version,
            .minAvgChunksPerStream = minAvgChunksPerStream,
        });
    return *writer_;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<ChunkStatsWriter> writer_;
};

class ChunkStatsReaderVersionTest
    : public ChunkStatsWriterTest,
      public ::testing::WithParamInterface<ChunkStatsVersion> {};

TEST_P(ChunkStatsReaderVersionTest, createRejectsNullMetadata) {
  NIMBLE_ASSERT_THROW(
      index::ChunkStatsGroup::create(
          GetParam(),
          /*firstStripe=*/0,
          /*stripeCount=*/1,
          std::unique_ptr<MetadataBuffer>{},
          *pool_),
      "must not be null");
}

TEST_P(ChunkStatsReaderVersionTest, readerContract) {
  auto& writer = createWriter(GetParam(), 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(3);
  writer.addStream(0, createChunks(buffer, {{10, 4, 1}, {20, 6, 2}}));
  writer.addStream(1, createChunks(buffer, {{30, 5, 3}}));

  writer.newStripe(3);
  writer.addStream(0, createChunks(buffer, {{40, 7, 4}}));
  writer.addStream(
      1, createChunks(buffer, {{10, 2, 0}, {15, 3, 5}, {15, 4, 1}}));

  writer.writeGroup(3, 2, createMetadataSectionCallback(fileIndex));
  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);

  auto chunkStats = index::ChunkStatsGroup::create(
      GetParam(),
      /*firstStripe=*/5,
      /*stripeCount=*/2,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);

  auto firstStream = chunkStats->createStreamIndex(
      /*stripe=*/5, /*streamId=*/0, /*streamSize=*/10);
  ASSERT_NE(firstStream, nullptr);
  EXPECT_EQ(firstStream->streamId(), 0);
  const auto initialLocation = firstStream->lookupChunk(0);
  EXPECT_EQ(initialLocation.chunkIndex, 0);
  EXPECT_EQ(initialLocation.chunkOffset, 0);
  EXPECT_EQ(initialLocation.chunkSize, 4);
  EXPECT_EQ(initialLocation.rowOffset, 0);
  EXPECT_EQ(firstStream->lookupChunk(9).chunkIndex, initialLocation.chunkIndex);
  const auto firstLocation = firstStream->lookupChunk(10);
  EXPECT_EQ(firstLocation.chunkIndex, 1);
  EXPECT_EQ(firstLocation.chunkOffset, 4);
  EXPECT_EQ(firstLocation.chunkSize, 6);
  EXPECT_EQ(firstLocation.rowOffset, 10);
  EXPECT_EQ(firstStream->chunkNullCount(firstLocation.chunkIndex), 2);
  EXPECT_EQ(firstStream->rowCount(), 30);
  const auto finalLocation = firstStream->lookupChunk(29);
  EXPECT_EQ(finalLocation.chunkIndex, firstLocation.chunkIndex);
  EXPECT_EQ(finalLocation.chunkOffset, firstLocation.chunkOffset);
  EXPECT_EQ(finalLocation.chunkSize, firstLocation.chunkSize);
  EXPECT_EQ(finalLocation.rowOffset, firstLocation.rowOffset);
  NIMBLE_ASSERT_THROW(firstStream->lookupChunk(30), "beyond the last chunk");

  EXPECT_EQ(
      chunkStats->createStreamIndex(
          /*stripe=*/5, /*streamId=*/1, /*streamSize=*/5),
      nullptr);
  EXPECT_EQ(
      chunkStats->createStreamIndex(
          /*stripe=*/6, /*streamId=*/0, /*streamSize=*/7),
      nullptr);
  EXPECT_EQ(
      chunkStats->createStreamIndex(
          /*stripe=*/5, /*streamId=*/2, /*streamSize=*/0),
      nullptr);
  EXPECT_EQ(
      chunkStats->createStreamIndex(
          /*stripe=*/5, /*streamId=*/3, /*streamSize=*/0),
      nullptr);
  NIMBLE_ASSERT_THROW(
      chunkStats->createStreamIndex(
          /*stripe=*/4, /*streamId=*/0, /*streamSize=*/10),
      "Stripe index is before this group's range");
  NIMBLE_ASSERT_THROW(
      chunkStats->createStreamIndex(
          /*stripe=*/7, /*streamId=*/0, /*streamSize=*/10),
      "Stripe offset is out of range for this chunk stats group");

  auto secondStream = chunkStats->createStreamIndex(
      /*stripe=*/6, /*streamId=*/1, /*streamSize=*/9);
  ASSERT_NE(secondStream, nullptr);
  EXPECT_EQ(secondStream->streamId(), 1);
  chunkStats.reset();

  const auto secondLocation = secondStream->lookupChunk(25);
  EXPECT_EQ(secondLocation.chunkIndex, 6);
  EXPECT_EQ(secondLocation.chunkOffset, 5);
  EXPECT_EQ(secondLocation.chunkSize, 4);
  EXPECT_EQ(secondLocation.rowOffset, 25);
  EXPECT_EQ(secondStream->chunkNullCount(secondLocation.chunkIndex), 1);
  EXPECT_EQ(secondStream->rowCount(), 40);
  NIMBLE_ASSERT_THROW(secondStream->lookupChunk(40), "beyond the last chunk");
}

TEST_P(ChunkStatsReaderVersionTest, multipleGroupsRoundTrip) {
  auto& writer = createWriter(GetParam(), 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{40, 8}, {60, 14}, {50, 11}}));
  writer.writeGroup(1, 2, createMetadataSectionCallback(fileIndex));

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{30, 6}, {30, 7}, {40, 9}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 2);
  const auto* root = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(root, nullptr);
  ASSERT_NE(root->stripe_indexes(), nullptr);
  EXPECT_EQ(root->stripe_indexes()->size(), 2);

  auto firstGroup = index::ChunkStatsGroup::create(
      GetParam(),
      0,
      2,
      copyMetadata(fileIndex.groupMetadataSections[0]),
      *pool_);
  auto firstStripe = firstGroup->createStreamIndex(0, 0, 22);
  ASSERT_NE(firstStripe, nullptr);
  const auto firstLocation = firstStripe->lookupChunk(50);
  EXPECT_EQ(firstLocation.chunkOffset, 10);
  EXPECT_EQ(firstLocation.chunkSize, 12);
  EXPECT_EQ(firstLocation.rowOffset, 50);

  auto secondStripe = firstGroup->createStreamIndex(1, 0, 33);
  ASSERT_NE(secondStripe, nullptr);
  const auto secondLocation = secondStripe->lookupChunk(40);
  EXPECT_EQ(secondLocation.chunkOffset, 8);
  EXPECT_EQ(secondLocation.chunkSize, 14);
  EXPECT_EQ(secondLocation.rowOffset, 40);

  auto secondGroup = index::ChunkStatsGroup::create(
      GetParam(),
      2,
      1,
      copyMetadata(fileIndex.groupMetadataSections[1]),
      *pool_);
  auto thirdStripe = secondGroup->createStreamIndex(2, 0, 22);
  ASSERT_NE(thirdStripe, nullptr);
  const auto thirdLocation = thirdStripe->lookupChunk(60);
  EXPECT_EQ(thirdLocation.chunkOffset, 13);
  EXPECT_EQ(thirdLocation.chunkSize, 9);
  EXPECT_EQ(thirdLocation.rowOffset, 60);
}

TEST_F(ChunkStatsWriterTest, singleStripe) {
  auto& writer = createWriter();
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
  auto& writer = createWriter();
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
  auto& writer = createWriter();
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

TEST_F(ChunkStatsWriterTest, v2MultipleStreamsAndStripes) {
  auto& writer = createWriter(ChunkStatsVersion::kV2);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{10, 4, 1}, {20, 6, 2}}));
  writer.addStream(1, createChunks(buffer, {{30, 5, 3}}));

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{40, 7, 4}}));
  writer.addStream(
      1, createChunks(buffer, {{10, 2, 0}, {15, 3, 5}, {15, 4, 1}}));

  writer.writeGroup(2, 2, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex, nimble::kChunkStatsV2Section));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  ASSERT_FALSE(fileIndex.rootIndexData.empty());

  const auto* rootChunkStats = flatbuffers::GetRoot<serialization::ChunkStats>(
      fileIndex.rootIndexData.data());
  ASSERT_NE(rootChunkStats, nullptr);
  ASSERT_NE(rootChunkStats->stripe_indexes(), nullptr);
  EXPECT_EQ(rootChunkStats->stripe_indexes()->size(), 1);

  const auto* group = flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
      fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->stream_count(), 2);
  ASSERT_NE(group->stream_chunk_counts(), nullptr);
  const std::vector<uint32_t> chunkCounts{
      group->stream_chunk_counts()->begin(),
      group->stream_chunk_counts()->end()};
  EXPECT_THAT(chunkCounts, testing::ElementsAre(2, 3, 1, 4));

  ASSERT_NE(group->stream_chunk_rows(), nullptr);
  ASSERT_NE(group->stream_chunk_offsets(), nullptr);
  ASSERT_NE(group->stream_chunk_null_counts(), nullptr);
  const auto chunkRows = decode(*group->stream_chunk_rows());
  EXPECT_THAT(chunkRows, testing::ElementsAre(10, 30, 40, 30, 10, 25, 40));
  EXPECT_THAT(
      decode(*group->stream_chunk_offsets()),
      testing::ElementsAre(0, 4, 0, 0, 0, 2, 5));
  EXPECT_THAT(
      decode(*group->stream_chunk_null_counts()),
      testing::ElementsAre(1, 2, 4, 3, 0, 5, 1));

  // Binary search requires cumulative row ends to be non-decreasing within
  // each stream and stripe.
  uint32_t streamBaseOffset{0};
  for (uint32_t streamId = 0; streamId < group->stream_count(); ++streamId) {
    uint32_t previousChunkCount{0};
    for (uint32_t stripeOffset = 0; stripeOffset < 2; ++stripeOffset) {
      const auto chunkCount = chunkCounts[streamId * 2 + stripeOffset];
      for (uint32_t chunkOffset = previousChunkCount + 1;
           chunkOffset < chunkCount;
           ++chunkOffset) {
        EXPECT_LE(
            chunkRows[streamBaseOffset + chunkOffset - 1],
            chunkRows[streamBaseOffset + chunkOffset]);
      }
      previousChunkCount = chunkCount;
    }
    streamBaseOffset += previousChunkCount;
  }
}

TEST_F(ChunkStatsWriterTest, v2EncodesDoubleBoundsAsDouble) {
  auto& writer = createWriter(ChunkStatsVersion::kV2);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  auto chunks = createChunks(buffer, {{10, 4}, {20, 6}});
  chunks[0].minValue = 1.5;
  chunks[0].maxValue = 2.5;
  chunks[1].minValue = -3.25;
  chunks[1].maxValue = 4.75;
  writer.addStream(0, chunks);
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  const auto* group = flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
      fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(group->chunk_min_values(), nullptr);
  ASSERT_NE(group->chunk_max_values(), nullptr);
  const auto* encodedMins = group->chunk_min_values()->Get(0)->data();
  const auto* encodedMaxs = group->chunk_max_values()->Get(0)->data();
  ASSERT_NE(encodedMins, nullptr);
  ASSERT_NE(encodedMaxs, nullptr);
  EXPECT_EQ(
      EncodingPrefix::dataType(
          std::string_view{
              reinterpret_cast<const char*>(encodedMins->data()),
              encodedMins->size()}),
      DataType::Double);
  EXPECT_EQ(
      EncodingPrefix::dataType(
          std::string_view{
              reinterpret_cast<const char*>(encodedMaxs->data()),
              encodedMaxs->size()}),
      DataType::Double);

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  auto stream = chunkStats->createStreamIndex(
      /*stripe=*/0, /*streamId=*/0, /*streamSize=*/10);
  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(std::get<double>(*stream->chunkMinValue(0)), 1.5);
  EXPECT_EQ(std::get<double>(*stream->chunkMaxValue(0)), 2.5);
  EXPECT_EQ(std::get<double>(*stream->chunkMinValue(1)), -3.25);
  EXPECT_EQ(std::get<double>(*stream->chunkMaxValue(1)), 4.75);
}

TEST_F(ChunkStatsWriterTest, v2EncodesFloatBoundsAsFloat) {
  auto& writer = createWriter(ChunkStatsVersion::kV2);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  auto chunks = createChunks(buffer, {{10, 4}, {20, 6}});
  chunks[0].minValue = 1.5F;
  chunks[0].maxValue = 2.5F;
  chunks[1].minValue = -3.25F;
  chunks[1].maxValue = 4.75F;
  writer.addStream(0, chunks);
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  const auto* group = flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
      fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(group->chunk_min_values(), nullptr);
  ASSERT_NE(group->chunk_max_values(), nullptr);
  const auto* encodedMins = group->chunk_min_values()->Get(0)->data();
  const auto* encodedMaxs = group->chunk_max_values()->Get(0)->data();
  ASSERT_NE(encodedMins, nullptr);
  ASSERT_NE(encodedMaxs, nullptr);
  EXPECT_EQ(
      EncodingPrefix::dataType(
          std::string_view{
              reinterpret_cast<const char*>(encodedMins->data()),
              encodedMins->size()}),
      DataType::Float);
  EXPECT_EQ(
      EncodingPrefix::dataType(
          std::string_view{
              reinterpret_cast<const char*>(encodedMaxs->data()),
              encodedMaxs->size()}),
      DataType::Float);

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  auto stream = chunkStats->createStreamIndex(
      /*stripe=*/0, /*streamId=*/0, /*streamSize=*/10);
  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(std::get<float>(*stream->chunkMinValue(0)), 1.5F);
  EXPECT_EQ(std::get<float>(*stream->chunkMaxValue(0)), 2.5F);
  EXPECT_EQ(std::get<float>(*stream->chunkMinValue(1)), -3.25F);
  EXPECT_EQ(std::get<float>(*stream->chunkMaxValue(1)), 4.75F);
}

TEST_F(ChunkStatsWriterTest, v2PreservesPhysicalBoundTypes) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  const std::vector<ChunkStatValue> mins{
      int8_t{-1},
      uint8_t{1},
      int16_t{-2},
      uint16_t{2},
      int32_t{-3},
      uint32_t{3},
      int64_t{-4},
      uint64_t{4},
      false,
  };
  const std::vector<ChunkStatValue> maxs{
      int8_t{1},
      uint8_t{2},
      int16_t{2},
      uint16_t{3},
      int32_t{3},
      uint32_t{4},
      int64_t{4},
      uint64_t{5},
      true,
  };
  const std::vector<DataType> expectedTypes{
      DataType::Int8,
      DataType::Uint8,
      DataType::Int16,
      DataType::Uint16,
      DataType::Int32,
      DataType::Uint32,
      DataType::Int64,
      DataType::Uint64,
      DataType::Bool,
  };

  writer.newStripe(mins.size());
  for (uint32_t streamId = 0; streamId < mins.size(); ++streamId) {
    auto chunks = createChunks(buffer, {{1, 1}});
    chunks.front().minValue = mins[streamId];
    chunks.front().maxValue = maxs[streamId];
    writer.addStream(streamId, chunks);
  }
  writer.writeGroup(mins.size(), 1, createMetadataSectionCallback(fileIndex));

  const auto* serializedGroup =
      flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
          fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(serializedGroup->chunk_min_values(), nullptr);
  for (uint32_t streamId = 0; streamId < expectedTypes.size(); ++streamId) {
    const auto* data =
        serializedGroup->chunk_min_values()->Get(streamId)->data();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(
        EncodingPrefix::dataType(
            std::string_view{
                reinterpret_cast<const char*>(data->data()), data->size()}),
        expectedTypes[streamId]);
  }

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  for (uint32_t streamId = 0; streamId < expectedTypes.size(); ++streamId) {
    auto stream = chunkStats->createStreamIndex(0, streamId, 1);
    ASSERT_NE(stream, nullptr);
    EXPECT_EQ(stream->chunkMinValue(streamId), mins[streamId]);
    EXPECT_EQ(stream->chunkMaxValue(streamId), maxs[streamId]);
  }
}

TEST_F(ChunkStatsWriterTest, v2RoundTripsTypedAndMissingBounds) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(3);
  auto integralChunks = createChunks(buffer, {{10, 4}, {20, 6}});
  integralChunks[0].minValue = int64_t{-7};
  integralChunks[0].maxValue = int64_t{11};
  integralChunks[1].minValue = std::numeric_limits<int64_t>::min();
  integralChunks[1].maxValue = std::numeric_limits<int64_t>::max();
  writer.addStream(0, integralChunks);

  auto stringChunks = createChunks(buffer, {{10, 4}, {20, 6}});
  stringChunks[0].minValue = std::string{};
  stringChunks[0].maxValue = std::string{"beta"};
  stringChunks[1].minValue = std::string{"a\0b", 3};
  stringChunks[1].maxValue = std::string{"z"};
  writer.addStream(1, stringChunks);

  auto partialChunks = createChunks(buffer, {{10, 4}, {20, 6}});
  partialChunks[0].minValue = -1.5;
  partialChunks[0].maxValue = 3.25;
  writer.addStream(2, partialChunks);

  writer.writeGroup(3, 1, createMetadataSectionCallback(fileIndex));
  const auto* serializedGroup =
      flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
          fileIndex.groupMetadataSections.front().data());
  const auto* encodedPresence = serializedGroup->chunk_min_max_present();
  ASSERT_NE(encodedPresence, nullptr);
  ASSERT_NE(encodedPresence->data(), nullptr);
  const std::string_view presenceData{
      reinterpret_cast<const char*>(encodedPresence->data()->data()),
      encodedPresence->data()->size()};
  EXPECT_EQ(EncodingPrefix::dataType(presenceData), DataType::Bool);
  EXPECT_EQ(EncodingPrefix::readRowCount(presenceData, /*useVarint=*/false), 6);

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);

  auto integral = chunkStats->createStreamIndex(0, 0, 10);
  ASSERT_NE(integral, nullptr);
  EXPECT_EQ(std::get<int64_t>(*integral->chunkMinValue(0)), -7);
  EXPECT_EQ(std::get<int64_t>(*integral->chunkMaxValue(0)), 11);
  EXPECT_EQ(
      std::get<int64_t>(*integral->chunkMinValue(1)),
      std::numeric_limits<int64_t>::min());
  EXPECT_EQ(
      std::get<int64_t>(*integral->chunkMaxValue(1)),
      std::numeric_limits<int64_t>::max());

  auto strings = chunkStats->createStreamIndex(0, 1, 10);
  ASSERT_NE(strings, nullptr);
  EXPECT_EQ(std::get<std::string>(*strings->chunkMinValue(2)), "");
  EXPECT_EQ(std::get<std::string>(*strings->chunkMaxValue(2)), "beta");
  EXPECT_EQ(
      std::get<std::string>(*strings->chunkMinValue(3)),
      std::string("a\0b", 3));
  EXPECT_EQ(std::get<std::string>(*strings->chunkMaxValue(3)), "z");

  auto partial = chunkStats->createStreamIndex(0, 2, 10);
  ASSERT_NE(partial, nullptr);
  EXPECT_EQ(std::get<double>(*partial->chunkMinValue(4)), -1.5);
  EXPECT_EQ(std::get<double>(*partial->chunkMaxValue(4)), 3.25);
  EXPECT_FALSE(partial->chunkMinValue(5).has_value());
  EXPECT_FALSE(partial->chunkMaxValue(5).has_value());
}

TEST_F(ChunkStatsWriterTest, v2UsesRleForChunkBoundPresence) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  constexpr size_t kRunLength{1'000};
  writer.newStripe(1);
  auto chunks = createChunks(
      buffer,
      std::vector<ChunkSpec>(2 * kRunLength, {.rowCount = 1, .size = 1}));
  for (size_t i = 0; i < kRunLength; ++i) {
    chunks[i].minValue = int64_t{1};
    chunks[i].maxValue = int64_t{2};
  }
  writer.addStream(0, chunks);
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  const auto* group = flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
      fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(group->chunk_min_max_present(), nullptr);
  ASSERT_NE(group->chunk_min_max_present()->data(), nullptr);
  const auto* encoded = group->chunk_min_max_present()->data();
  const std::string_view presenceData{
      reinterpret_cast<const char*>(encoded->data()), encoded->size()};
  EXPECT_EQ(EncodingPrefix::dataType(presenceData), DataType::Bool);
  EXPECT_EQ(EncodingPrefix::encodingType(presenceData), EncodingType::RLE);
}

TEST_F(ChunkStatsWriterTest, v2RoundTripsConstantStringBounds) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  auto chunks = createChunks(buffer, {{10, 4}, {20, 6}});
  for (auto& chunk : chunks) {
    chunk.minValue = std::string{"same"};
    chunk.maxValue = std::string{"same"};
  }
  writer.addStream(0, chunks);
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  auto stream = chunkStats->createStreamIndex(0, 0, 10);
  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(std::get<std::string>(*stream->chunkMinValue(0)), "same");
  EXPECT_EQ(std::get<std::string>(*stream->chunkMaxValue(1)), "same");
}

TEST_F(ChunkStatsWriterTest, v2RejectsOversizedStringBounds) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};

  writer.newStripe(1);
  auto chunks = createChunks(buffer, {{10, 4}});
  chunks[0].minValue = std::string(
      ChunkStatsWriter::Options::kDefaultMaxChunkStringStatSize + 1, 'a');
  chunks[0].maxValue = std::string{"z"};
  NIMBLE_ASSERT_THROW(
      writer.addStream(0, chunks),
      "Chunk minimum exceeds the maximum string statistic size");
}

TEST_F(ChunkStatsWriterTest, v2PreservesBoundsAcrossStripes) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  auto firstStripe = createChunks(buffer, {{10, 4}});
  firstStripe[0].minValue = int64_t{2};
  firstStripe[0].maxValue = int64_t{8};
  writer.addStream(0, firstStripe);

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{10, 4}}));
  writer.writeGroup(1, 2, createMetadataSectionCallback(fileIndex));

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/5,
      /*stripeCount=*/2,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  auto first = chunkStats->createStreamIndex(5, 0, 4);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(std::get<int64_t>(*first->chunkMinValue(0)), 2);
  EXPECT_EQ(std::get<int64_t>(*first->chunkMaxValue(0)), 8);
  EXPECT_EQ(chunkStats->createStreamIndex(6, 0, 4), nullptr);
}

TEST_F(ChunkStatsWriterTest, v2PreservesPackedLayoutWithAbsentStreams) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  const auto addBoundedStream = [&](uint32_t streamId,
                                    std::vector<Chunk> chunks,
                                    const ChunkStatValue& min,
                                    const ChunkStatValue& max) {
    for (auto& chunk : chunks) {
      chunk.minValue = min;
      chunk.maxValue = max;
    }
    writer.addStream(streamId, chunks);
  };

  writer.newStripe(1);
  addBoundedStream(
      0, createChunks(buffer, {{10, 4}, {20, 6}}), int8_t{-1}, int8_t{1});

  writer.newStripe(3);
  addBoundedStream(0, createChunks(buffer, {{30, 7}}), int8_t{-2}, int8_t{2});
  auto stream2Chunks = createChunks(buffer, {{60, 8}, {40, 9}});
  stream2Chunks[0].minValue = uint16_t{10};
  stream2Chunks[0].maxValue = uint16_t{20};
  writer.addStream(2, stream2Chunks);

  writer.newStripe(2);
  addBoundedStream(
      0, createChunks(buffer, {{40, 10}, {50, 11}}), int8_t{-3}, int8_t{3});
  addBoundedStream(1, createChunks(buffer, {{50, 12}}), false, true);

  writer.writeGroup(3, 3, createMetadataSectionCallback(fileIndex));
  const auto* serializedGroup =
      flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
          fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(serializedGroup->stream_chunk_counts(), nullptr);
  const std::vector<uint32_t> chunkCounts{
      serializedGroup->stream_chunk_counts()->begin(),
      serializedGroup->stream_chunk_counts()->end()};
  EXPECT_THAT(chunkCounts, testing::ElementsAre(2, 3, 5, 0, 0, 1, 0, 2, 2));
  EXPECT_THAT(
      decode(*serializedGroup->stream_chunk_rows()),
      testing::ElementsAre(10, 30, 30, 40, 90, 50, 60, 100));
  EXPECT_THAT(
      decode(*serializedGroup->stream_chunk_offsets()),
      testing::ElementsAre(0, 4, 0, 0, 10, 0, 0, 8));

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/3,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  EXPECT_EQ(chunkStats->createStreamIndex(0, 1, 0), nullptr);
  EXPECT_EQ(chunkStats->createStreamIndex(1, 1, 0), nullptr);
  auto stream1Stripe2 = chunkStats->createStreamIndex(2, 1, 12);
  ASSERT_NE(stream1Stripe2, nullptr);
  EXPECT_EQ(stream1Stripe2->lookupChunk(0).chunkIndex, 5);
  EXPECT_EQ(stream1Stripe2->chunkMinValue(5), ChunkStatValue{false});
  EXPECT_EQ(stream1Stripe2->chunkMaxValue(5), ChunkStatValue{true});

  EXPECT_EQ(chunkStats->createStreamIndex(0, 2, 0), nullptr);
  auto stream2Stripe1 = chunkStats->createStreamIndex(1, 2, 17);
  ASSERT_NE(stream2Stripe1, nullptr);
  EXPECT_EQ(stream2Stripe1->lookupChunk(0).chunkIndex, 6);
  EXPECT_EQ(stream2Stripe1->lookupChunk(60).chunkIndex, 7);
  EXPECT_EQ(stream2Stripe1->chunkMinValue(6), ChunkStatValue{uint16_t{10}});
  EXPECT_EQ(stream2Stripe1->chunkMaxValue(6), ChunkStatValue{uint16_t{20}});
  EXPECT_FALSE(stream2Stripe1->chunkMinValue(7).has_value());
  EXPECT_FALSE(stream2Stripe1->chunkMaxValue(7).has_value());
  EXPECT_EQ(chunkStats->createStreamIndex(2, 2, 0), nullptr);
}

TEST_F(ChunkStatsWriterTest, v2RejectsInvalidBounds) {
  Buffer buffer{*pool_};

  {
    auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
    writer.newStripe(1);
    auto chunks = createChunks(buffer, {{10, 4}});
    chunks[0].minValue = int64_t{1};
    chunks[0].maxValue = 2.0;
    NIMBLE_ASSERT_THROW(
        writer.addStream(0, chunks),
        "Chunk minimum and maximum must have the same type");
  }

  {
    auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
    writer.newStripe(1);
    auto chunks = createChunks(buffer, {{10, 4}});
    chunks[0].minValue = int64_t{2};
    chunks[0].maxValue = int64_t{1};
    NIMBLE_ASSERT_THROW(
        writer.addStream(0, chunks), "Chunk minimum must not exceed maximum");
  }

  {
    auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
    writer.newStripe(1);
    auto chunks = createChunks(buffer, {{10, 4}});
    chunks[0].minValue = std::numeric_limits<double>::quiet_NaN();
    chunks[0].maxValue = 1.0;
    NIMBLE_ASSERT_THROW(
        writer.addStream(0, chunks), "Chunk bounds must not be NaN");
  }
}

TEST_F(ChunkStatsWriterTest, v2RejectsCompactUnboundedBounds) {
  const auto groupData = createV2ConstantBoundsData(/*chunkCount=*/10'000);
  NIMBLE_ASSERT_THROW(
      index::ChunkStatsGroup::create(
          ChunkStatsVersion::kV2,
          /*firstStripe=*/0,
          /*stripeCount=*/1,
          copyMetadata(groupData),
          *pool_),
      "V2 chunk bounds metadata is incomplete or has an invalid stream count");
}

TEST_F(ChunkStatsWriterTest, v2ReadsOversizedStringBounds) {
  const std::string min(
      ChunkStatsWriter::Options::kDefaultMaxChunkStringStatSize + 1, 'a');
  const auto groupData =
      createV2ConstantStringBoundsData(min, "z", /*chunkCount=*/2);
  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(groupData),
      *pool_);
  auto stream = chunkStats->createStreamIndex(0, 0, 0);
  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(std::get<std::string>(*stream->chunkMinValue(0)), min);
  EXPECT_EQ(std::get<std::string>(*stream->chunkMaxValue(0)), "z");
}

TEST_F(ChunkStatsWriterTest, v2EmptyStream) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  writer.addStream(0, {});
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex, nimble::kChunkStatsV2Section));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  const auto* group = flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(
      fileIndex.groupMetadataSections.front().data());
  ASSERT_NE(group, nullptr);
  ASSERT_NE(group->stream_chunk_counts(), nullptr);
  const std::vector<uint32_t> chunkCounts{
      group->stream_chunk_counts()->begin(),
      group->stream_chunk_counts()->end()};
  EXPECT_THAT(chunkCounts, testing::ElementsAre(0));
  ASSERT_NE(group->stream_chunk_rows(), nullptr);
  ASSERT_NE(group->stream_chunk_offsets(), nullptr);
  ASSERT_NE(group->stream_chunk_null_counts(), nullptr);
  EXPECT_THAT(decode(*group->stream_chunk_rows()), testing::IsEmpty());
  EXPECT_THAT(decode(*group->stream_chunk_offsets()), testing::IsEmpty());
  EXPECT_THAT(decode(*group->stream_chunk_null_counts()), testing::IsEmpty());

  auto chunkStats = index::ChunkStatsGroup::create(
      ChunkStatsVersion::kV2,
      /*firstStripe=*/0,
      /*stripeCount=*/1,
      copyMetadata(fileIndex.groupMetadataSections.front()),
      *pool_);
  EXPECT_EQ(
      chunkStats->createStreamIndex(
          /*stripe=*/0, /*streamId=*/0, /*streamSize=*/0),
      nullptr);
}

TEST_F(ChunkStatsWriterTest, v2RejectsMismatchedChunkCounts) {
  auto& writer = createWriter(ChunkStatsVersion::kV2, 0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));

  ASSERT_EQ(fileIndex.groupMetadataSections.size(), 1);
  auto& groupData = fileIndex.groupMetadataSections.front();
  const auto* group =
      flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(groupData.data());
  ASSERT_NE(group->stream_chunk_counts(), nullptr);
  auto* chunkCounts =
      const_cast<flatbuffers::Vector<uint32_t>*>(group->stream_chunk_counts());
  chunkCounts->Mutate(0, 3);

  NIMBLE_ASSERT_THROW(
      index::ChunkStatsGroup::create(
          ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
      "row count");
}

TEST_F(ChunkStatsWriterTest, v2RejectsDecreasingChunkCounts) {
  const std::array<uint32_t, 2> chunkCounts{2, 1};
  const std::array<uint32_t, 1> rows{10};
  const std::array<uint32_t, 1> offsets{0};
  const std::array<uint32_t, 1> nullCounts{0};
  const auto groupData =
      createV2GroupData(chunkCounts, rows, offsets, nullCounts);

  NIMBLE_ASSERT_THROW(
      index::ChunkStatsGroup::create(
          ChunkStatsVersion::kV2,
          /*firstStripe=*/0,
          /*stripeCount=*/2,
          copyMetadata(groupData),
          *pool_),
      "stream chunk counts must be non-decreasing");
}

TEST_F(ChunkStatsWriterTest, v2RejectsInvalidChunkOffsets) {
  const std::array<uint32_t, 1> chunkCounts{2};
  const std::array<uint32_t, 2> rows{10, 20};
  const std::array<uint32_t, 2> nullCounts{0, 0};

  {
    const std::array<uint32_t, 2> offsets{5, 4};
    auto chunkStats = index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV2,
        /*firstStripe=*/0,
        /*stripeCount=*/1,
        copyMetadata(createV2GroupData(chunkCounts, rows, offsets, nullCounts)),
        *pool_);
    auto streamIndex = chunkStats->createStreamIndex(
        /*stripe=*/0, /*streamId=*/0, /*streamSize=*/10);
    ASSERT_NE(streamIndex, nullptr);
    NIMBLE_ASSERT_THROW(streamIndex->lookupChunk(0), "(5 vs. 4)");
  }

  {
    const std::array<uint32_t, 2> offsets{0, 11};
    auto chunkStats = index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV2,
        /*firstStripe=*/0,
        /*stripeCount=*/1,
        copyMetadata(createV2GroupData(chunkCounts, rows, offsets, nullCounts)),
        *pool_);
    auto streamIndex = chunkStats->createStreamIndex(
        /*stripe=*/0, /*streamId=*/0, /*streamSize=*/10);
    ASSERT_NE(streamIndex, nullptr);
    NIMBLE_ASSERT_THROW(streamIndex->lookupChunk(0), "(11 vs. 10)");
  }
}

TEST_F(ChunkStatsWriterTest, v2RejectsInvalidRootShape) {
  {
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata("invalid"), *pool_),
        "Invalid V2 chunk stats metadata");
  }

  const std::array<uint32_t, 0> emptyValues{};
  const auto emptyEncoded = encodeTrivial(emptyValues);
  {
    const auto groupData = createV2GroupDataFromEncoded(
        emptyValues,
        emptyEncoded,
        emptyEncoded,
        emptyEncoded,
        /*streamCount=*/0);
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        "V2 chunk stats has no streams");
  }

  {
    flatbuffers::FlatBufferBuilder builder;
    const auto encodedStream = serialization::CreateEncodedStream(
        builder, builder.CreateVector(emptyEncoded));
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            {},
            encodedStream,
            encodedStream,
            encodedStream));
    const std::string groupData{
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        "Missing V2 stream chunk counts");
  }

  {
    const std::array<uint32_t, 3> chunkCounts{0, 0, 0};
    const auto groupData = createV2GroupDataFromEncoded(
        chunkCounts,
        emptyEncoded,
        emptyEncoded,
        emptyEncoded,
        /*streamCount=*/2);
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 2, copyMetadata(groupData), *pool_),
        "stream chunk count size does not match");
  }
}

TEST_F(ChunkStatsWriterTest, v2RequiresEncodedStreams) {
  const std::array<uint32_t, 1> chunkCounts{2};
  const std::array<uint32_t, 2> values{0, 1};
  const auto encoded = encodeTrivial(values);

  constexpr std::array<std::string_view, 3> kFieldNames{
      "chunk rows", "chunk offsets", "chunk null counts"};
  for (size_t missingIndex = 0; missingIndex < kFieldNames.size();
       ++missingIndex) {
    SCOPED_TRACE(kFieldNames[missingIndex]);
    flatbuffers::FlatBufferBuilder builder;
    const auto encodedStream = serialization::CreateEncodedStream(
        builder, builder.CreateVector(encoded));
    std::array<flatbuffers::Offset<serialization::EncodedStream>, 3>
        encodedStreams{encodedStream, encodedStream, encodedStream};
    encodedStreams[missingIndex] = {};
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            builder.CreateVector(chunkCounts.data(), chunkCounts.size()),
            encodedStreams[0],
            encodedStreams[1],
            encodedStreams[2]));
    const std::string groupData{
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};

    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        fmt::format("Missing encoded {}", kFieldNames[missingIndex]));
  }

  {
    flatbuffers::FlatBufferBuilder builder;
    const auto missingDataStream = serialization::CreateEncodedStream(
        builder, flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{});
    const auto validStream = serialization::CreateEncodedStream(
        builder, builder.CreateVector(encoded));
    builder.Finish(
        serialization::CreateStripeChunkStatsV2(
            builder,
            /*stream_count=*/1,
            builder.CreateVector(chunkCounts.data(), chunkCounts.size()),
            missingDataStream,
            validStream,
            validStream));
    const std::string groupData{
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        "Missing encoded chunk rows data");
  }
}

TEST_F(ChunkStatsWriterTest, v2RejectsInvalidEncodedStreamPrefix) {
  const std::array<uint32_t, 1> chunkCounts{2};
  const std::array<uint32_t, 2> values{0, 1};
  const auto validData = encodeTrivial(values);
  const auto expectInvalidRows = [&](const std::vector<uint8_t>& rows,
                                     std::string_view error) {
    const auto groupData =
        createV2GroupDataFromEncoded(chunkCounts, rows, validData, validData);
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        error);
  };

  {
    std::vector<uint8_t> tooSmall(EncodingPrefix::kFixedPrefixSize - 1);
    expectInvalidRows(tooSmall, "Encoded chunk rows array is too small");
  }

  {
    auto invalidDataType = validData;
    invalidDataType[EncodingPrefix::kDataTypeOffset] =
        static_cast<uint8_t>(DataType::Int32);
    expectInvalidRows(invalidDataType, "invalid data type");
  }
}

TEST_F(ChunkStatsWriterTest, v2RejectsInvalidChunkBoundsMetadata) {
  const auto presence = encodeConstant<bool>(true, /*rowCount=*/1);

  {
    const std::array<uint8_t, 1> invalidMin{0};
    const auto max = encodeConstant<int8_t>(1, /*rowCount=*/1);
    const auto groupData = createV2BoundsData(invalidMin, max, presence);
    NIMBLE_ASSERT_THROW(
        index::ChunkStatsGroup::create(
            ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_),
        "Encoded chunk min values array is too small");
  }

  {
    const auto groupData = createV2BoundsData({}, {}, presence);
    auto chunkStats = index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_);
    NIMBLE_ASSERT_THROW(
        chunkStats->createStreamIndex(
            /*stripe=*/0, /*streamId=*/0, /*streamSize=*/1),
        "marked present but stream 0 has no bounds data");
  }

  {
    const auto min = encodeConstant<int8_t>(2, /*rowCount=*/1);
    const auto max = encodeConstant<int8_t>(1, /*rowCount=*/1);
    const auto groupData = createV2BoundsData(min, max, presence);
    auto chunkStats = index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_);
    auto stream = chunkStats->createStreamIndex(
        /*stripe=*/0, /*streamId=*/0, /*streamSize=*/1);
    ASSERT_NE(stream, nullptr);
    NIMBLE_ASSERT_THROW(
        stream->chunkMinValue(/*chunkIndex=*/0),
        "chunk minimum must not exceed maximum");
  }

  {
    const auto min = encodeConstant<float>(
        std::numeric_limits<float>::quiet_NaN(), /*rowCount=*/1);
    const auto max = encodeConstant<float>(1, /*rowCount=*/1);
    const auto groupData = createV2BoundsData(min, max, presence);
    auto chunkStats = index::ChunkStatsGroup::create(
        ChunkStatsVersion::kV2, 0, 1, copyMetadata(groupData), *pool_);
    auto stream = chunkStats->createStreamIndex(
        /*stripe=*/0, /*streamId=*/0, /*streamSize=*/1);
    ASSERT_NE(stream, nullptr);
    NIMBLE_ASSERT_THROW(
        stream->chunkMaxValue(/*chunkIndex=*/0),
        "chunk bounds must not be NaN");
  }
}

TEST_F(ChunkStatsWriterTest, multipleStripeGroups) {
  auto& writer = createWriter(0);
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
  auto& writer = createWriter(0);
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

TEST_P(ChunkStatsReaderVersionTest, emptyFileNoStripeGroups) {
  auto& writer = createWriter(GetParam());
  TestChunkFileIndex fileIndex;

  // No stripes written — writeGroup() is never called.
  writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

  // No chunk stats section should be written.
  EXPECT_TRUE(fileIndex.rootIndexData.empty());
}

TEST_P(ChunkStatsReaderVersionTest, finalization) {
  auto& writer = createWriter(GetParam());
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(1);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 10}}));
  writer.writeGroup(1, 1, createMetadataSectionCallback(fileIndex));
  writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

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

TEST_P(ChunkStatsReaderVersionTest, addStreamIndexValidation) {
  auto& writer = createWriter(GetParam());
  Buffer buffer{*pool_};

  // addStream before newStripe should fail.
  NIMBLE_ASSERT_THROW(writer.addStream(0, createChunks(buffer, {{10, 5}})), "");

  // Out-of-range stream index should fail.
  writer.newStripe(2);
  NIMBLE_ASSERT_THROW(writer.addStream(2, createChunks(buffer, {{10, 5}})), "");
}

TEST_F(ChunkStatsWriterTest, multipleStripesInMultipleGroups) {
  auto& writer = createWriter();
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

TEST_P(ChunkStatsReaderVersionTest, minAvgChunksPerStream) {
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

    auto& writer = createWriter(GetParam(), testData.threshold);
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

    writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

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

TEST_P(ChunkStatsReaderVersionTest, uncompressedSizeRoundtrip) {
  auto& writer = createWriter(GetParam(), /*minAvgChunksPerStream=*/0);
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

  writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

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
TEST_P(ChunkStatsReaderVersionTest, uncompressedSizeForUncompressedSections) {
  auto& writer = createWriter(GetParam(), /*minAvgChunksPerStream=*/0);
  Buffer buffer{*pool_};
  TestChunkFileIndex fileIndex;

  writer.newStripe(2);
  writer.addStream(0, createChunks(buffer, {{50, 10}, {50, 12}}));
  writer.addStream(1, createChunks(buffer, {{60, 20}, {40, 12}}));
  writer.writeGroup(2, 1, createMetadataSectionCallback(fileIndex));

  writer.writeRoot(writeRootCallback(fileIndex, sectionName(GetParam())));

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

INSTANTIATE_TEST_SUITE_P(
    ChunkStatsVersions,
    ChunkStatsReaderVersionTest,
    ::testing::Values(ChunkStatsVersion::kV1, ChunkStatsVersion::kV2),
    [](const ::testing::TestParamInfo<ChunkStatsVersion>& info) {
      return info.param == ChunkStatsVersion::kV1 ? "V1" : "V2";
    });

} // namespace facebook::nimble::test
