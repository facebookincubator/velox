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

#include "flatbuffers/flatbuffers.h"
#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

namespace facebook::nimble {

namespace {

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

} // namespace

ChunkStatsWriter::ChunkStatsWriter(
    velox::memory::MemoryPool& pool,
    float minAvgChunksPerStream)
    : pool_{&pool}, minAvgChunksPerStream_{minAvgChunksPerStream} {}

void ChunkStatsWriter::newStripe(size_t streamCount) {
  NIMBLE_CHECK(!finalized_, "ChunkStatsWriter has been finalized");
  if (groupIndex_ == nullptr) {
    groupIndex_ = std::make_unique<GroupIndex>();
  }
  auto& stripeIndex = groupIndex_->stripes.emplace_back();
  stripeIndex.streams.resize(streamCount);
}

void ChunkStatsWriter::addStream(
    uint32_t streamIndex,
    const std::vector<Chunk>& chunks) {
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

void ChunkStatsWriter::writeGroup(
    size_t streamCount,
    size_t stripeCount,
    const CreateMetadataSectionFn& createMetadataSection) {
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

  // Flatten chunk data for all streams.
  std::vector<uint32_t> flattenedStreamChunkCounts;
  flattenedStreamChunkCounts.reserve(stripeCount * streamCount);

  uint32_t accumulatedChunkCount{0};
  for (const auto& stripe : groupIndex_->stripes) {
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
  flattenedChunkRows.reserve(accumulatedChunkCount);
  flattenedChunkOffsets.reserve(accumulatedChunkCount);
  flattenedChunkNullCounts.reserve(accumulatedChunkCount);

  for (const auto& stripe : groupIndex_->stripes) {
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

  chunkStatsSections_.push_back(createMetadataSection(asView(builder)));
}

void ChunkStatsWriter::writeRoot(
    const WriteOptionalSectionFn& writeOptionalSection) {
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
      builder.CreateVector<flatbuffers::Offset<serialization::MetadataSection>>(
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

  builder.Finish(serialization::CreateChunkStats(builder, stripeIndexesVector));
  writeOptionalSection(std::string(kChunkStatsSection), asView(builder));
}

} // namespace facebook::nimble
