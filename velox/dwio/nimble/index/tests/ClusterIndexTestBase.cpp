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
#include "velox/dwio/nimble/index/tests/ClusterIndexTestBase.h"

#include "folly/json/json.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/io/Options.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble::index::test {

namespace {

velox::BufferPtr toBufferPtr(
    std::string_view data,
    velox::memory::MemoryPool* pool) {
  auto buffer = velox::AlignedBuffer::allocate<char>(data.size(), pool);
  std::memcpy(buffer->asMutable<char>(), data.data(), data.size());
  return buffer;
}

} // namespace

void ClusterIndexTestBase::SetUpTestCase() {
  velox::memory::MemoryManager::testingSetInstance({});
  velox::common::testutil::TestValue::enable();
}

ClusterIndexTestBase::IndexBuffers ClusterIndexTestBase::createTestClusterIndex(
    const std::vector<std::string>& indexColumns,
    const std::vector<std::string>& sortOrders,
    const std::string& minKey,
    const std::vector<Stripe>& stripes,
    const std::vector<int>& stripeGroups) {
  NIMBLE_CHECK_EQ(indexColumns.size(), sortOrders.size());

  flatbuffers::FlatBufferBuilder builder;
  IndexBuffers result;

  // Build stripeGroupIndices from stripeGroups.
  for (uint32_t groupIdx = 0; groupIdx < stripeGroups.size(); ++groupIdx) {
    for (int i = 0; i < stripeGroups[groupIdx]; ++i) {
      result.stripeGroupIndices.push_back(groupIdx);
    }
  }

  std::vector<flatbuffers::Offset<serialization::MetadataSection>>
      clusterIndexGroupMetadataOffsets;
  clusterIndexGroupMetadataOffsets.reserve(stripeGroups.size());

  uint32_t stripeIdx = 0;
  for (const auto& groupStripeCount : stripeGroups) {
    // Build ClusterIndexPartition with chunk and sub-partition data.
    flatbuffers::FlatBufferBuilder groupBuilder;

    std::vector<uint32_t> chunkRows;
    std::vector<uint32_t> chunkOffsets;
    std::vector<flatbuffers::Offset<flatbuffers::String>> chunkKeys;

    // Build standalone ChunkStatsGroup for this group
    flatbuffers::FlatBufferBuilder chunkBuilder;
    std::vector<uint32_t> streamChunkCounts;
    std::vector<uint32_t> streamChunkRows;
    std::vector<uint32_t> streamChunkOffsets;

    uint32_t accumulatedStreamChunkCount = 0;
    uint32_t keyDataSize = 0;
    // Base offset of this partition's key stream in the combined stream.
    const uint32_t partitionKeyStreamOffset =
        stripes[stripeIdx].keyStream.streamOffset;
    // Accumulated rows across all stripes within this partition.
    uint32_t accumulatedChunkRows = 0;
    for (int i = 0; i < groupStripeCount; ++i, ++stripeIdx) {
      NIMBLE_CHECK_LT(stripeIdx, stripes.size());
      const auto& stripe = stripes[stripeIdx];

      const auto& keyStream = stripe.keyStream;
      NIMBLE_CHECK_EQ(
          keyStream.stream.numChunks, keyStream.stream.chunkRows.size());
      NIMBLE_CHECK_EQ(
          keyStream.stream.numChunks, keyStream.stream.chunkOffsets.size());
      NIMBLE_CHECK_EQ(keyStream.stream.numChunks, keyStream.chunkKeys.size());

      // Accumulate chunk data per stripe. Chunk offsets are partition-relative.
      uint32_t stripeRowCount = 0;
      for (size_t c = 0; c < keyStream.stream.numChunks; ++c) {
        accumulatedChunkRows += keyStream.stream.chunkRows[c];
        chunkRows.push_back(accumulatedChunkRows);
        chunkOffsets.push_back(
            (keyStream.streamOffset - partitionKeyStreamOffset) +
            keyStream.stream.chunkOffsets[c]);
        chunkKeys.push_back(groupBuilder.CreateString(keyStream.chunkKeys[c]));
        stripeRowCount += keyStream.stream.chunkRows[c];
      }
      result.stripeRowCounts.push_back(stripeRowCount);
      keyDataSize += keyStream.streamSize;

      for (const auto& stream : stripe.streams) {
        accumulatedStreamChunkCount += stream.numChunks;
        streamChunkCounts.push_back(accumulatedStreamChunkCount);
        NIMBLE_CHECK_EQ(stream.numChunks, stream.chunkRows.size());
        uint32_t accumulatedRows = 0;
        for (const auto& row : stream.chunkRows) {
          accumulatedRows += row;
          streamChunkRows.push_back(accumulatedRows);
        }
        NIMBLE_CHECK_EQ(stream.numChunks, stream.chunkOffsets.size());
        for (const auto& offset : stream.chunkOffsets) {
          streamChunkOffsets.push_back(offset);
        }
      }
    }
    auto chunkRowsVec = groupBuilder.CreateVector(chunkRows);
    auto chunkOffsetsVec = groupBuilder.CreateVector(chunkOffsets);
    auto chunkKeysVec = groupBuilder.CreateVector(chunkKeys);

    groupBuilder.Finish(
        serialization::CreateClusterIndexPartition(
            groupBuilder,
            partitionKeyStreamOffset,
            keyDataSize,
            chunkRowsVec,
            chunkOffsetsVec,
            chunkKeysVec));

    uint64_t offset = result.indexPartitions.size();
    uint32_t size = groupBuilder.GetSize();
    result.indexPartitions.append(
        reinterpret_cast<const char*>(groupBuilder.GetBufferPointer()), size);

    auto metadataSection = serialization::CreateMetadataSection(
        builder,
        offset,
        size,
        serialization::CompressionType_Uncompressed,
        size);
    clusterIndexGroupMetadataOffsets.push_back(metadataSection);

    const uint32_t numStreams = streamChunkCounts.size() / groupStripeCount;

    // Build standalone StripeChunkStats flatbuffer
    auto stripeChunkStats = serialization::CreateStripeChunkStats(
        chunkBuilder,
        numStreams,
        chunkBuilder.CreateVector(streamChunkCounts),
        chunkBuilder.CreateVector(streamChunkRows),
        chunkBuilder.CreateVector(streamChunkOffsets));
    chunkBuilder.Finish(stripeChunkStats);

    result.chunkStatsGroupSizes.push_back(chunkBuilder.GetSize());
    result.chunkStatsGroups.append(
        reinterpret_cast<const char*>(chunkBuilder.GetBufferPointer()),
        chunkBuilder.GetSize());
  }

  std::vector<flatbuffers::Offset<flatbuffers::String>> subPartitionKeyOffsets;
  subPartitionKeyOffsets.reserve(stripeGroups.size() + 1);
  subPartitionKeyOffsets.push_back(builder.CreateString(minKey));
  uint32_t partitionStripeStart = 0;
  for (const auto& groupStripeCount : stripeGroups) {
    const auto& lastStripe =
        stripes[partitionStripeStart + groupStripeCount - 1];
    const auto& chunkKeys = lastStripe.keyStream.chunkKeys;
    NIMBLE_CHECK(!chunkKeys.empty());
    subPartitionKeyOffsets.push_back(builder.CreateString(chunkKeys.back()));
    partitionStripeStart += groupStripeCount;
  }

  std::vector<flatbuffers::Offset<flatbuffers::String>> indexColumnOffsets;
  indexColumnOffsets.reserve(indexColumns.size());
  for (const auto& column : indexColumns) {
    indexColumnOffsets.push_back(builder.CreateString(column));
  }

  std::vector<flatbuffers::Offset<flatbuffers::String>> sortOrderOffsets;
  sortOrderOffsets.reserve(sortOrders.size());
  for (const auto& sortOrder : sortOrders) {
    sortOrderOffsets.push_back(builder.CreateString(sortOrder));
  }

  // Build partition row counts from per-stripe row counts.
  std::vector<uint32_t> partitionRowCounts;
  partitionRowCounts.reserve(stripeGroups.size());
  uint32_t stripeStart = 0;
  for (const auto& groupStripeCount : stripeGroups) {
    uint32_t partitionRows = 0;
    for (int i = 0; i < groupStripeCount; ++i) {
      partitionRows += result.stripeRowCounts[stripeStart + i];
    }
    partitionRowCounts.push_back(partitionRows);
    stripeStart += groupStripeCount;
  }

  auto indexColumnsVector = builder.CreateVector(indexColumnOffsets);
  auto sortOrdersVector = builder.CreateVector(sortOrderOffsets);
  auto indexPartitionsVector =
      builder.CreateVector(clusterIndexGroupMetadataOffsets);
  auto partitionKeysVector = builder.CreateVector(subPartitionKeyOffsets);
  auto partitionRowCountsVector = builder.CreateVector(partitionRowCounts);

  auto index = serialization::CreateClusterIndex(
      builder,
      indexColumnsVector,
      sortOrdersVector,
      indexPartitionsVector,
      partitionKeysVector,
      partitionRowCountsVector);

  builder.Finish(index);

  result.rootIndex = std::string(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());

  return result;
}

ClusterIndexTestBase::IndexBuffers ClusterIndexTestBase::createTestClusterIndex(
    const std::vector<std::string>& indexColumns,
    const std::string& minKey,
    const std::vector<Stripe>& stripes,
    const std::vector<int>& stripeGroups) {
  std::vector<std::string> defaultSortOrders;
  defaultSortOrders.reserve(indexColumns.size());
  for (size_t i = 0; i < indexColumns.size(); ++i) {
    defaultSortOrders.push_back(
        folly::toJson(velox::core::kAscNullsFirst.serialize()));
  }
  return createTestClusterIndex(
      indexColumns, defaultSortOrders, minKey, stripes, stripeGroups);
}

ClusterIndexTestBase::IndexBuffers
ClusterIndexTestBase::createChunkOnlyTestClusterIndex(
    const std::vector<ChunkOnlyStripe>& stripes,
    const std::vector<int>& stripeGroups) {
  flatbuffers::FlatBufferBuilder builder;
  IndexBuffers result;

  // Build stripeGroupIndices from stripeGroups.
  for (uint32_t groupIdx = 0; groupIdx < stripeGroups.size(); ++groupIdx) {
    for (int i = 0; i < stripeGroups[groupIdx]; ++i) {
      result.stripeGroupIndices.push_back(groupIdx);
    }
  }

  std::vector<flatbuffers::Offset<serialization::MetadataSection>>
      clusterIndexGroupMetadataOffsets;
  clusterIndexGroupMetadataOffsets.reserve(stripeGroups.size());

  uint32_t stripeIdx = 0;
  for (const auto& groupStripeCount : stripeGroups) {
    // Build standalone ChunkStatsGroup only (no StripeClusterIndex needed)
    flatbuffers::FlatBufferBuilder chunkBuilder;

    std::vector<uint32_t> streamChunkCounts;
    std::vector<uint32_t> streamChunkRows;
    std::vector<uint32_t> streamChunkOffsets;

    uint32_t accumulatedStreamChunkCount = 0;
    for (int i = 0; i < groupStripeCount; ++i, ++stripeIdx) {
      NIMBLE_CHECK_LT(stripeIdx, stripes.size());
      const auto& stripe = stripes[stripeIdx];

      for (const auto& stream : stripe.streams) {
        accumulatedStreamChunkCount += stream.numChunks;
        streamChunkCounts.push_back(accumulatedStreamChunkCount);
        NIMBLE_CHECK_EQ(stream.numChunks, stream.chunkRows.size());
        uint32_t accumulatedRows = 0;
        for (const auto& row : stream.chunkRows) {
          accumulatedRows += row;
          streamChunkRows.push_back(accumulatedRows);
        }
        NIMBLE_CHECK_EQ(stream.numChunks, stream.chunkOffsets.size());
        for (const auto& offset : stream.chunkOffsets) {
          streamChunkOffsets.push_back(offset);
        }
      }
    }

    const uint32_t numStreams = streamChunkCounts.size() / groupStripeCount;

    auto stripeChunkStats = serialization::CreateStripeChunkStats(
        chunkBuilder,
        numStreams,
        chunkBuilder.CreateVector(streamChunkCounts),
        chunkBuilder.CreateVector(streamChunkRows),
        chunkBuilder.CreateVector(streamChunkOffsets));
    chunkBuilder.Finish(stripeChunkStats);

    result.chunkStatsGroupSizes.push_back(chunkBuilder.GetSize());
    result.chunkStatsGroups.append(
        reinterpret_cast<const char*>(chunkBuilder.GetBufferPointer()),
        chunkBuilder.GetSize());

    // Build empty IndexPartition for root ClusterIndex to work.
    flatbuffers::FlatBufferBuilder groupBuilder;
    groupBuilder.Finish(
        serialization::CreateClusterIndexPartition(groupBuilder));

    uint64_t offset = result.indexPartitions.size();
    uint32_t size = groupBuilder.GetSize();
    result.indexPartitions.append(
        reinterpret_cast<const char*>(groupBuilder.GetBufferPointer()), size);

    auto metadataSection = serialization::CreateMetadataSection(
        builder,
        offset,
        size,
        serialization::CompressionType_Uncompressed,
        size);
    clusterIndexGroupMetadataOffsets.push_back(metadataSection);
  }

  auto indexPartitionsVector =
      builder.CreateVector(clusterIndexGroupMetadataOffsets);

  auto index = serialization::CreateClusterIndex(
      builder,
      0 /* index_columns = null */,
      0 /* sort_orders = null */,
      indexPartitionsVector);

  builder.Finish(index);

  result.rootIndex = std::string(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());

  return result;
}

std::unique_ptr<ClusterIndex> ClusterIndexTestBase::createClusterIndex(
    const IndexBuffers& indexBuffers,
    velox::ReadFile* keyStreamFile) {
  Section indexSection{MetadataBuffer(
      MetadataBuffer::decompress(
          toBufferPtr(indexBuffers.rootIndex, pool_.get()),
          CompressionType::Uncompressed,
          pool_.get()))};

  metadataFile_ =
      std::make_shared<velox::InMemoryReadFile>(indexBuffers.indexPartitions);
  ioStats_ = std::make_shared<velox::io::IoStatistics>();
  auto metadataInput = MetadataInput::create(
      metadataFile_.get(),
      MetadataInput::Options{.pool = pool_.get(), .ioStats = ioStats_});

  auto dataFile = keyStreamFile != nullptr
      ? std::shared_ptr<velox::ReadFile>(
            std::shared_ptr<velox::ReadFile>{}, keyStreamFile)
      : metadataFile_;
  auto dataInput = std::make_unique<velox::dwio::common::BufferedInput>(
      std::move(dataFile), *pool_);
  return ClusterIndexTestHelper::create(
      std::move(indexSection),
      pool_.get(),
      std::move(metadataInput),
      std::move(dataInput));
}

std::shared_ptr<ChunkStatsGroup> ClusterIndexTestBase::createChunkStats(
    const IndexBuffers& indexBuffers,
    uint32_t stripeGroupIndex) {
  // Compute firstStripe and stripeCount from stripeGroupIndices.
  uint32_t firstStripe = 0;
  uint32_t stripeCount = 0;
  for (uint32_t s = 0; s < indexBuffers.stripeGroupIndices.size(); ++s) {
    if (indexBuffers.stripeGroupIndices[s] == stripeGroupIndex) {
      if (stripeCount == 0) {
        firstStripe = s;
      }
      ++stripeCount;
    }
  }

  // Find the ChunkStatsGroup data for this group using recorded sizes.
  NIMBLE_CHECK_LT(stripeGroupIndex, indexBuffers.chunkStatsGroupSizes.size());
  size_t offset = 0;
  for (uint32_t g = 0; g < stripeGroupIndex; ++g) {
    offset += indexBuffers.chunkStatsGroupSizes[g];
  }

  auto chunkStatsBuffer =
      std::make_unique<MetadataBuffer>(MetadataBuffer::decompress(
          toBufferPtr(
              std::string_view(
                  indexBuffers.chunkStatsGroups.data() + offset,
                  indexBuffers.chunkStatsGroupSizes[stripeGroupIndex]),
              pool_.get()),
          CompressionType::Uncompressed,
          pool_.get()));

  return ChunkStatsGroup::create(
      firstStripe, stripeCount, std::move(chunkStatsBuffer));
}

} // namespace facebook::nimble::index::test
