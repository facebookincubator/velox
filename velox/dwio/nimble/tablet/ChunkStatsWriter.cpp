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

// Shares chunk collection, group filtering, and root serialization between
// both on-disk formats.
class ChunkStatsWriterBase : public ChunkStatsWriter {
 public:
  ChunkStatsWriterBase(
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream,
      std::string_view sectionName)
      : pool_{pool},
        minAvgChunksPerStream_{minAvgChunksPerStream},
        sectionName_{sectionName} {}

  void newStripe(size_t streamCount) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    if (groupIndex_ == nullptr) {
      groupIndex_ = std::make_unique<GroupIndex>();
    }
    auto& stripeIndex = groupIndex_->stripes.emplace_back();
    stripeIndex.streams.resize(streamCount);
  }

  void addStream(uint32_t streamIndex, const std::vector<Chunk>& chunks) final {
    NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
    NIMBLE_CHECK_NOT_NULL(groupIndex_);
    NIMBLE_CHECK(!groupIndex_->empty());
    NIMBLE_CHECK_LT(streamIndex, groupIndex_->stripes.back().streams.size());

    auto& index = groupIndex_->stripes.back().streams[streamIndex];
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
    NIMBLE_CHECK_EQ(stripeCount, groupIndex_->stripes.size());
    NIMBLE_CHECK_GT(stripeCount, 0);
    NIMBLE_CHECK_GT(streamCount, 0);

    SCOPE_EXIT {
      groupIndex_.reset();
    };

    // Compute total number of chunks across all streams and stripes.
    uint32_t totalNumChunks{0};
    for (const auto& stripe : groupIndex_->stripes) {
      for (size_t streamId = 0; streamId < streamCount; ++streamId) {
        if (streamId < stripe.streams.size()) {
          totalNumChunks += stripe.streams[streamId].chunkCount;
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

    chunkStatsSections_.push_back(writeGroupImpl(
        streamCount, stripeCount, totalNumChunks, createMetadataSection));
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
    writeOptionalSection(std::string{sectionName_}, asView(builder));
  }

 protected:
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

  // Holds index data for all streams in a single stripe.
  struct StripeIndex {
    std::vector<StreamIndex> streams;
  };

  // Holds index data for stream chunks across all stripes in a stripe group.
  struct GroupIndex {
    std::vector<StripeIndex> stripes;

    bool empty() const {
      return stripes.empty();
    }
  };

  const GroupIndex& groupIndex() const {
    return *groupIndex_;
  }

  velox::memory::MemoryPool& pool() const {
    return pool_;
  }

 private:
  // Serializes the collected group and returns its metadata section.
  virtual MetadataSection writeGroupImpl(
      size_t streamCount,
      size_t stripeCount,
      uint32_t totalNumChunks,
      const CreateMetadataSectionFn& createMetadataSection) = 0;

  velox::memory::MemoryPool& pool_;
  const float minAvgChunksPerStream_;
  const std::string_view sectionName_;
  std::unique_ptr<GroupIndex> groupIndex_;
  // Metadata sections for chunk stats flatbuffers (used by writeRoot).
  std::vector<MetadataSection> chunkStatsSections_;
  bool finalized_{false};
};

// Writes the legacy stripe-major format using raw FlatBuffer arrays.
class ChunkStatsWriterV1 final : public ChunkStatsWriterBase {
 public:
  ChunkStatsWriterV1(
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream)
      : ChunkStatsWriterBase{pool, minAvgChunksPerStream, kChunkStatsSection} {}

 private:
  MetadataSection writeGroupImpl(
      size_t streamCount,
      size_t stripeCount,
      uint32_t totalNumChunks,
      const CreateMetadataSectionFn& createMetadataSection) override {
    std::vector<uint32_t> flattenedStreamChunkCounts;
    flattenedStreamChunkCounts.reserve(stripeCount * streamCount);

    uint32_t accumulatedChunkCount{0};
    for (const auto& stripe : groupIndex().stripes) {
      for (size_t streamId = 0; streamId < streamCount; ++streamId) {
        const uint32_t chunkCount = streamId < stripe.streams.size()
            ? stripe.streams[streamId].chunkCount
            : 0;
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

    for (const auto& stripe : groupIndex().stripes) {
      for (size_t streamId = 0; streamId < streamCount; ++streamId) {
        if (streamId < stripe.streams.size()) {
          const auto& stream = stripe.streams[streamId];
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
};

// Creates the point-readable encodings supported by V2 chunk stats.
template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> makeChunkStatsEncodingPolicy() {
  ManualEncodingSelectionPolicyFactory factory{
      std::vector<std::pair<EncodingType, float>>{
          {EncodingType::Constant, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::FixedBitWidth, 1.0},
      },
      /*compressionOptions=*/std::nullopt};
  auto base = factory.createPolicy(TypeTraits<T>::dataType);
  return std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(base.release()));
}

// Encodes values and resets the reusable scratch buffer.
flatbuffers::Offset<serialization::EncodedStream> encodeArray(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<uint32_t>& values,
    Buffer& encodingBuffer) {
  const auto encoded = EncodingFactory::encode<uint32_t>(
      makeChunkStatsEncodingPolicy<uint32_t>(),
      std::span<const uint32_t>(values),
      encodingBuffer);
  const auto result = serialization::CreateEncodedStream(
      builder,
      builder.CreateVector(
          reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size()));
  encodingBuffer.reset();
  return result;
}

// Writes the stream-major format using Nimble-encoded arrays.
class ChunkStatsWriterV2 final : public ChunkStatsWriterBase {
 public:
  ChunkStatsWriterV2(
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream)
      : ChunkStatsWriterBase{
            pool,
            minAvgChunksPerStream,
            kChunkStatsV2Section} {}

 private:
  MetadataSection writeGroupImpl(
      size_t streamCount,
      size_t stripeCount,
      uint32_t totalNumChunks,
      const CreateMetadataSectionFn& createMetadataSection) override {
    std::vector<uint32_t> streamChunkCounts(streamCount * stripeCount);
    for (size_t streamId = 0; streamId < streamCount; ++streamId) {
      uint32_t accumulatedChunkCount{0};
      for (size_t stripe = 0; stripe < stripeCount; ++stripe) {
        const auto& stripeIndex = groupIndex().stripes[stripe];
        const uint32_t chunkCount = streamId < stripeIndex.streams.size()
            ? stripeIndex.streams[streamId].chunkCount
            : 0;
        accumulatedChunkCount += chunkCount;
        streamChunkCounts[streamId * stripeCount + stripe] =
            accumulatedChunkCount;
      }
    }

    std::vector<uint32_t> chunkRows;
    std::vector<uint32_t> chunkOffsets;
    std::vector<uint32_t> chunkNullCounts;
    chunkRows.reserve(totalNumChunks);
    chunkOffsets.reserve(totalNumChunks);
    chunkNullCounts.reserve(totalNumChunks);

    for (size_t streamId = 0; streamId < streamCount; ++streamId) {
      for (const auto& stripe : groupIndex().stripes) {
        if (streamId < stripe.streams.size()) {
          const auto& stream = stripe.streams[streamId];
          chunkRows.insert(
              chunkRows.end(),
              stream.chunkRows.begin(),
              stream.chunkRows.end());
          chunkOffsets.insert(
              chunkOffsets.end(),
              stream.chunkOffsets.begin(),
              stream.chunkOffsets.end());
          chunkNullCounts.insert(
              chunkNullCounts.end(),
              stream.chunkNullCounts.begin(),
              stream.chunkNullCounts.end());
        }
      }
    }

    Buffer encodingBuffer{pool()};
    flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
    std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedRows{
        encodeArray(builder, chunkRows, encodingBuffer)};
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        encodedOffsets{encodeArray(builder, chunkOffsets, encodingBuffer)};
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>
        encodedNullCounts{
            encodeArray(builder, chunkNullCounts, encodingBuffer)};

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
};

} // namespace

std::unique_ptr<ChunkStatsWriter> ChunkStatsWriter::create(
    ChunkStatsVersion version,
    velox::memory::MemoryPool& pool,
    float minAvgChunksPerStream) {
  switch (version) {
    case ChunkStatsVersion::kV1:
      return std::make_unique<ChunkStatsWriterV1>(pool, minAvgChunksPerStream);
    case ChunkStatsVersion::kV2:
      return std::make_unique<ChunkStatsWriterV2>(pool, minAvgChunksPerStream);
  }
  NIMBLE_UNREACHABLE("Unknown chunk stats version.");
}

} // namespace facebook::nimble
