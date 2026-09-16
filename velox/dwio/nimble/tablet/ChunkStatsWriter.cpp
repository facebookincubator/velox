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

#include <algorithm>
#include <span>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tablet/Chunk.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"

namespace facebook::nimble {

namespace {

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

// Holds chunk-level index data for a single stream within a stripe.
struct StreamIndex {
  // Accumulated row counts per chunk.
  std::vector<uint32_t> chunkRows;
  // Byte offsets of each chunk within the stream.
  std::vector<uint32_t> chunkOffsets;
  // Per-chunk null-value count (statistic used for chunk skipping).
  std::vector<uint32_t> chunkNullCounts;
  // Number of chunks in this stripe for this stream.
  uint32_t chunkCount{0};
};

using StripeIndex = std::vector<StreamIndex>;
using GroupIndex = std::vector<StripeIndex>;

MetadataSection writeGroupV1(
    const GroupIndex& groupIndex,
    size_t streamCount,
    size_t stripeCount,
    uint32_t totalNumChunks,
    const CreateMetadataSectionFn& createMetadataSection);

MetadataSection writeGroupV2(
    const GroupIndex& groupIndex,
    velox::memory::MemoryPool& pool,
    size_t streamCount,
    size_t stripeCount,
    uint32_t totalNumChunks,
    const CreateMetadataSectionFn& createMetadataSection);

// Implements chunk collection and both on-disk formats.
class ChunkStatsWriterImpl final : public ChunkStatsWriter {
 public:
  ChunkStatsWriterImpl(
      ChunkStatsVersion version,
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream)
      : version_{version},
        pool_{pool},
        minAvgChunksPerStream_{minAvgChunksPerStream} {}

  void newStripe(size_t streamCount) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    if (groupIndex_ == nullptr) {
      groupIndex_ = std::make_unique<GroupIndex>();
    }
    groupIndex_->emplace_back().resize(streamCount);
  }

  void addStream(uint32_t streamIndex, const std::vector<Chunk>& chunks) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    NIMBLE_CHECK_NOT_NULL(groupIndex_);
    NIMBLE_CHECK(!groupIndex_->empty());
    NIMBLE_CHECK_LT(streamIndex, groupIndex_->back().size());

    auto& index = groupIndex_->back()[streamIndex];
    uint32_t accumulatedRows{0};
    uint32_t accumulatedOffset{0};
    for (const auto& chunk : chunks) {
      accumulatedRows += chunk.rowCount;
      index.chunkRows.emplace_back(accumulatedRows);
      index.chunkOffsets.emplace_back(accumulatedOffset);
      index.chunkNullCounts.emplace_back(chunk.nullCount);
      accumulatedOffset += chunk.contentSize();
      ++index.chunkCount;
    }
  }

  void writeGroup(
      size_t streamCount,
      size_t stripeCount,
      const CreateMetadataSectionFn& createMetadataSection) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    NIMBLE_CHECK_NOT_NULL(groupIndex_);
    NIMBLE_CHECK_EQ(stripeCount, groupIndex_->size());
    NIMBLE_CHECK_GT(stripeCount, 0);
    NIMBLE_CHECK_GT(streamCount, 0);

    SCOPE_EXIT {
      groupIndex_.reset();
    };

    // Compute total number of chunks across all streams and stripes.
    uint32_t totalNumChunks{0};
    for (const auto& stripe : *groupIndex_) {
      for (size_t streamId = 0; streamId < streamCount; ++streamId) {
        if (streamId < stripe.size()) {
          totalNumChunks += stripe[streamId].chunkCount;
        }
      }
    }

    // Check threshold: skip chunk stats for this group if average chunks
    // per stream is below threshold.
    if (minAvgChunksPerStream_ > 0) {
      const float avgChunks =
          static_cast<float>(totalNumChunks) / static_cast<float>(streamCount);
      if (avgChunks < minAvgChunksPerStream_) {
        chunkStatsSections_.emplace_back(0, 0, CompressionType::Uncompressed);
        return;
      }
    }

    switch (version_) {
      case ChunkStatsVersion::kV1:
        chunkStatsSections_.push_back(writeGroupV1(
            *groupIndex_,
            streamCount,
            stripeCount,
            totalNumChunks,
            createMetadataSection));
        return;
      case ChunkStatsVersion::kV2:
        chunkStatsSections_.push_back(writeGroupV2(
            *groupIndex_,
            pool_,
            streamCount,
            stripeCount,
            totalNumChunks,
            createMetadataSection));
        return;
    }
    NIMBLE_UNREACHABLE(
        "Unknown chunk stats version: {}", static_cast<int>(version_));
  }

  void writeRoot(const WriteOptionalSectionFn& writeOptionalSection) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    SCOPE_EXIT {
      finalized_ = true;
    };

    // Skip writing the chunk stats section if all groups were skipped or no
    // groups were written.
    const bool hasNonEmptyGroup = std::any_of(
        chunkStatsSections_.begin(),
        chunkStatsSections_.end(),
        [](const auto& section) { return section.size() > 0; });
    if (!hasNonEmptyGroup) {
      return;
    }

    flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
    auto stripeIndexesVector =
        builder
            .CreateVector<flatbuffers::Offset<serialization::MetadataSection>>(
                chunkStatsSections_.size(), [&builder, this](size_t i) {
                  return serialization::CreateMetadataSection(
                      builder,
                      chunkStatsSections_[i].offset(),
                      chunkStatsSections_[i].size(),
                      static_cast<serialization::CompressionType>(
                          chunkStatsSections_[i].compressionType()),
                      chunkStatsSections_[i].uncompressedSize().value_or(
                          chunkStatsSections_[i].size()));
                });

    builder.Finish(
        serialization::CreateChunkStats(builder, stripeIndexesVector));
    writeOptionalSection(
        std::string{
            version_ == ChunkStatsVersion::kV1 ? kChunkStatsSection
                                               : kChunkStatsV2Section},
        asView(builder));
  }

 private:
  const ChunkStatsVersion version_;
  velox::memory::MemoryPool& pool_;
  const float minAvgChunksPerStream_;
  std::unique_ptr<GroupIndex> groupIndex_;
  // Metadata sections for chunk stats flatbuffers (used by writeRoot).
  std::vector<MetadataSection> chunkStatsSections_;
  bool finalized_{false};
};

MetadataSection writeGroupV1(
    const GroupIndex& groupIndex,
    size_t streamCount,
    size_t stripeCount,
    uint32_t totalNumChunks,
    const CreateMetadataSectionFn& createMetadataSection) {
  std::vector<uint32_t> flattenedStreamChunkCounts;
  flattenedStreamChunkCounts.reserve(stripeCount * streamCount);

  uint32_t accumulatedChunkCount{0};
  for (const auto& stripe : groupIndex) {
    for (size_t streamId = 0; streamId < streamCount; ++streamId) {
      const uint32_t chunkCount =
          streamId < stripe.size() ? stripe[streamId].chunkCount : 0;
      accumulatedChunkCount += chunkCount;
      flattenedStreamChunkCounts.push_back(accumulatedChunkCount);
    }
  }

  std::vector<uint32_t> flattenedChunkRows;
  std::vector<uint32_t> flattenedChunkOffsets;
  std::vector<uint32_t> flattenedChunkNullCounts;
  flattenedChunkRows.reserve(totalNumChunks);
  flattenedChunkOffsets.reserve(totalNumChunks);
  flattenedChunkNullCounts.reserve(totalNumChunks);

  for (const auto& stripe : groupIndex) {
    for (size_t streamId = 0; streamId < streamCount; ++streamId) {
      if (streamId < stripe.size()) {
        const auto& stream = stripe[streamId];
        flattenedChunkRows.insert(
            flattenedChunkRows.end(),
            stream.chunkRows.begin(),
            stream.chunkRows.end());
        flattenedChunkOffsets.insert(
            flattenedChunkOffsets.end(),
            stream.chunkOffsets.begin(),
            stream.chunkOffsets.end());
        flattenedChunkNullCounts.insert(
            flattenedChunkNullCounts.end(),
            stream.chunkNullCounts.begin(),
            stream.chunkNullCounts.end());
      }
    }
  }

  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
  auto chunkStats = serialization::CreateStripeChunkStats(
      builder,
      static_cast<uint32_t>(streamCount),
      builder.CreateVector(flattenedStreamChunkCounts),
      builder.CreateVector(flattenedChunkRows),
      builder.CreateVector(flattenedChunkOffsets),
      builder.CreateVector(flattenedChunkNullCounts));
  builder.Finish(chunkStats);
  return createMetadataSection(asView(builder));
}

// Encodes values and resets the reusable scratch buffer.
flatbuffers::Offset<serialization::EncodedStream> encodeArray(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<uint32_t>& values,
    Buffer& encodingBuffer) {
  ManualEncodingSelectionPolicyFactory factory{
      std::vector<std::pair<EncodingType, float>>{
          {EncodingType::Constant, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::FixedBitWidth, 1.0},
      },
      /*compressionOptions=*/std::nullopt,
  };
  auto base = factory.createPolicy(TypeTraits<uint32_t>::dataType);
  auto policy = std::unique_ptr<EncodingSelectionPolicy<uint32_t>>(
      static_cast<EncodingSelectionPolicy<uint32_t>*>(base.release()));
  const auto encoded = EncodingFactory::encode<uint32_t>(
      std::move(policy), std::span<const uint32_t>(values), encodingBuffer);
  const auto result = serialization::CreateEncodedStream(
      builder,
      builder.CreateVector(
          reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size()));
  encodingBuffer.reset();
  return result;
}

MetadataSection writeGroupV2(
    const GroupIndex& groupIndex,
    velox::memory::MemoryPool& pool,
    size_t streamCount,
    size_t stripeCount,
    uint32_t totalNumChunks,
    const CreateMetadataSectionFn& createMetadataSection) {
  std::vector<uint32_t> streamChunkCounts(streamCount * stripeCount);
  for (size_t streamId = 0; streamId < streamCount; ++streamId) {
    uint32_t accumulatedChunkCount{0};
    for (size_t stripe = 0; stripe < stripeCount; ++stripe) {
      const auto& stripeIndex = groupIndex[stripe];
      const uint32_t chunkCount =
          streamId < stripeIndex.size() ? stripeIndex[streamId].chunkCount : 0;
      accumulatedChunkCount += chunkCount;
      streamChunkCounts[streamId * stripeCount + stripe] =
          accumulatedChunkCount;
    }
  }

  Buffer encodingBuffer{pool};
  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
  std::vector<uint32_t> flattenedValues;
  flattenedValues.reserve(totalNumChunks);
  const auto encodeStreamValues =
      [&](const std::vector<uint32_t> StreamIndex::* valuesMember) {
        flattenedValues.clear();
        for (size_t streamId = 0; streamId < streamCount; ++streamId) {
          for (const auto& stripe : groupIndex) {
            if (streamId < stripe.size()) {
              const auto& values = stripe[streamId].*valuesMember;
              flattenedValues.insert(
                  flattenedValues.end(), values.begin(), values.end());
            }
          }
        }
        return encodeArray(builder, flattenedValues, encodingBuffer);
      };

  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedRows{
      encodeStreamValues(&StreamIndex::chunkRows)};
  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedOffsets{
      encodeStreamValues(&StreamIndex::chunkOffsets)};
  std::vector<flatbuffers::Offset<serialization::EncodedStream>>
      encodedNullCounts{encodeStreamValues(&StreamIndex::chunkNullCounts)};

  builder.Finish(
      serialization::CreateStripeChunkStatsV2(
          builder,
          static_cast<uint32_t>(streamCount),
          builder.CreateVector(streamChunkCounts),
          builder.CreateVector(encodedRows),
          builder.CreateVector(encodedOffsets),
          builder.CreateVector(encodedNullCounts)));
  return createMetadataSection(asView(builder));
}

} // namespace

std::unique_ptr<ChunkStatsWriter> ChunkStatsWriter::create(
    ChunkStatsVersion version,
    velox::memory::MemoryPool& pool,
    float minAvgChunksPerStream) {
  switch (version) {
    case ChunkStatsVersion::kV1:
    case ChunkStatsVersion::kV2:
      return std::make_unique<ChunkStatsWriterImpl>(
          version, pool, minAvgChunksPerStream);
  }
  NIMBLE_UNREACHABLE(
      "Unknown chunk stats version: {}", static_cast<int>(version));
}

} // namespace facebook::nimble
