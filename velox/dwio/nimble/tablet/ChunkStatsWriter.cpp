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
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
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
  // Per-chunk bounds. Empty values preserve the positions of missing bounds.
  std::vector<std::optional<ChunkStatValue>> chunkMinValues;
  std::vector<std::optional<ChunkStatValue>> chunkMaxValues;
  // Number of chunks in this stripe for this stream.
  uint32_t chunkCount{0};
};

using StripeIndex = std::vector<StreamIndex>;
using GroupIndex = std::vector<StripeIndex>;

// Maps the active variant alternative to its on-disk bounds type.
DataType chunkBoundsDataType(const std::optional<ChunkStatValue>& value) {
  if (!value.has_value()) {
    return DataType::Undefined;
  }
  return std::visit(
      [](const auto& typedValue) {
        return TypeTraits<std::decay_t<decltype(typedValue)>>::dataType;
      },
      *value);
}

bool hasChunkBounds(const Chunk& chunk) {
  return chunk.minValue.has_value() && chunk.maxValue.has_value();
}

// Enforces the invariants required to encode a min/max pair.
void validateChunkBounds(const Chunk& chunk, uint32_t maxStringStatSize) {
  NIMBLE_CHECK_EQ(
      chunk.minValue.has_value(),
      chunk.maxValue.has_value(),
      "Chunk minimum and maximum must both be present or absent.");
  if (!hasChunkBounds(chunk)) {
    return;
  }
  NIMBLE_CHECK_EQ(
      chunkBoundsDataType(chunk.minValue),
      chunkBoundsDataType(chunk.maxValue),
      "Chunk minimum and maximum must have the same type.");
  std::visit(
      [&](const auto& min) {
        using T = std::decay_t<decltype(min)>;
        const auto& max = std::get<T>(*chunk.maxValue);
        if constexpr (std::is_floating_point_v<T>) {
          NIMBLE_CHECK(
              !std::isnan(min) && !std::isnan(max),
              "Chunk bounds must not be NaN.");
        } else if constexpr (std::is_same_v<T, std::string>) {
          NIMBLE_CHECK_LE(
              min.size(),
              maxStringStatSize,
              "Chunk minimum exceeds the maximum string statistic size.");
          NIMBLE_CHECK_LE(
              max.size(),
              maxStringStatSize,
              "Chunk maximum exceeds the maximum string statistic size.");
        }
        NIMBLE_CHECK_LE(min, max, "Chunk minimum must not exceed maximum.");
      },
      *chunk.minValue);
}

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
      velox::memory::MemoryPool& pool,
      ChunkStatsWriter::Options options)
      : version_{options.version},
        pool_{pool},
        minAvgChunksPerStream_{options.minAvgChunksPerStream},
        maxStringStatSize_{options.maxStringStatSize} {}

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

      if (version_ == ChunkStatsVersion::kV2) {
        validateChunkBounds(chunk, maxStringStatSize_);
        if (hasChunkBounds(chunk)) {
          index.chunkMinValues.emplace_back(chunk.minValue);
          index.chunkMaxValues.emplace_back(chunk.maxValue);
        } else {
          index.chunkMinValues.emplace_back(std::nullopt);
          index.chunkMaxValues.emplace_back(std::nullopt);
        }
      }
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
  const uint32_t maxStringStatSize_;
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
template <typename T>
flatbuffers::Offset<serialization::EncodedStream> encodeTypedArray(
    flatbuffers::FlatBufferBuilder& builder,
    std::span<const T> values,
    Buffer& encodingBuffer,
    std::vector<std::pair<EncodingType, float>> encodingReadFactors) {
  ManualEncodingSelectionPolicyFactory factory{
      std::move(encodingReadFactors),
      /*compressionOptions=*/std::nullopt,
  };
  auto base = factory.createPolicy(TypeTraits<T>::dataType);
  auto policy = std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(base.release()));
  const auto encoded =
      EncodingFactory::encode<T>(std::move(policy), values, encodingBuffer);
  const auto result = serialization::CreateEncodedStream(
      builder,
      builder.CreateVector(
          reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size()));
  encodingBuffer.reset();
  return result;
}

flatbuffers::Offset<serialization::EncodedStream> encodeArray(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<uint32_t>& values,
    Buffer& encodingBuffer) {
  return encodeTypedArray<uint32_t>(
      builder,
      values,
      encodingBuffer,
      {
          {EncodingType::Constant, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::FixedBitWidth, 1.0},
      });
}

flatbuffers::Offset<serialization::EncodedStream> encodeBooleanArray(
    flatbuffers::FlatBufferBuilder& builder,
    const Vector<bool>& values,
    Buffer& encodingBuffer) {
  return encodeTypedArray<bool>(
      builder,
      std::span<const bool>{values.data(), values.size()},
      encodingBuffer,
      {
          {EncodingType::Constant, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::RLE, 1.0},
      });
}

flatbuffers::Offset<serialization::EncodedStream> emptyEncodedStream(
    flatbuffers::FlatBufferBuilder& builder) {
  return serialization::CreateEncodedStream(builder);
}

// Collects aligned optional bounds and their shared physical type for a stream.
struct StreamBounds {
  std::vector<std::optional<ChunkStatValue>> mins;
  std::vector<std::optional<ChunkStatValue>> maxs;
  DataType dataType{DataType::Undefined};
};

bool hasAnyChunkBounds(const GroupIndex& groupIndex) {
  for (const auto& stripe : groupIndex) {
    for (const auto& stream : stripe) {
      if (std::any_of(
              stream.chunkMinValues.begin(),
              stream.chunkMinValues.end(),
              [](const auto& value) { return value.has_value(); })) {
        return true;
      }
    }
  }
  return false;
}

StreamBounds collectStreamBounds(
    const GroupIndex& groupIndex,
    size_t streamId,
    size_t streamChunkCount,
    Vector<bool>& presence) {
  StreamBounds bounds;
  bounds.mins.reserve(streamChunkCount);
  bounds.maxs.reserve(streamChunkCount);
  for (const auto& stripe : groupIndex) {
    if (streamId < stripe.size()) {
      const auto& stream = stripe[streamId];
      bounds.mins.insert(
          bounds.mins.end(),
          stream.chunkMinValues.begin(),
          stream.chunkMinValues.end());
      bounds.maxs.insert(
          bounds.maxs.end(),
          stream.chunkMaxValues.begin(),
          stream.chunkMaxValues.end());
    }
  }
  NIMBLE_CHECK_EQ(bounds.mins.size(), streamChunkCount);
  NIMBLE_CHECK_EQ(bounds.maxs.size(), streamChunkCount);

  for (size_t chunk = 0; chunk < streamChunkCount; ++chunk) {
    const auto minType = chunkBoundsDataType(bounds.mins.at(chunk));
    const auto maxType = chunkBoundsDataType(bounds.maxs.at(chunk));
    NIMBLE_CHECK_EQ(
        minType, maxType, "Chunk minimum and maximum must have the same type.");
    presence.push_back(minType != DataType::Undefined);
    if (minType != DataType::Undefined) {
      NIMBLE_CHECK(
          bounds.dataType == DataType::Undefined || bounds.dataType == minType,
          "All chunk bounds in a stream must have the same type.");
      bounds.dataType = minType;
    }
  }
  return bounds;
}

template <typename T>
T boundValue(const std::optional<ChunkStatValue>& value) {
  if (!value.has_value()) {
    return T{};
  }
  if constexpr (std::is_same_v<T, std::string_view>) {
    return std::get<std::string>(*value);
  } else {
    return std::get<T>(*value);
  }
}

template <typename T>
void encodeTypedStreamBounds(
    const StreamBounds& bounds,
    flatbuffers::FlatBufferBuilder& builder,
    velox::memory::MemoryPool& pool,
    Buffer& encodingBuffer,
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>& encodedMins,
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>&
        encodedMaxs) {
  Vector<T> mins{&pool};
  Vector<T> maxs{&pool};
  mins.reserve(bounds.mins.size());
  maxs.reserve(bounds.maxs.size());
  for (size_t chunk = 0; chunk < bounds.mins.size(); ++chunk) {
    mins.push_back(boundValue<T>(bounds.mins.at(chunk)));
    maxs.push_back(boundValue<T>(bounds.maxs.at(chunk)));
  }
  std::vector<std::pair<EncodingType, float>> encodingReadFactors{
      {EncodingType::Constant, 1.0},
      {EncodingType::Trivial, 1.0},
  };
  encodingReadFactors.emplace_back(
      std::is_same_v<T, bool> ? EncodingType::RLE : EncodingType::FixedBitWidth,
      1.0);
  encodedMins.push_back(
      encodeTypedArray<T>(
          builder,
          std::span<const T>{mins.data(), mins.size()},
          encodingBuffer,
          encodingReadFactors));
  encodedMaxs.push_back(
      encodeTypedArray<T>(
          builder,
          std::span<const T>{maxs.data(), maxs.size()},
          encodingBuffer,
          encodingReadFactors));
}

void encodeStreamBounds(
    const StreamBounds& bounds,
    flatbuffers::FlatBufferBuilder& builder,
    velox::memory::MemoryPool& pool,
    Buffer& encodingBuffer,
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>& encodedMins,
    std::vector<flatbuffers::Offset<serialization::EncodedStream>>&
        encodedMaxs) {
  NIMBLE_RETURN_BY_DATA_TYPE(
      bounds.dataType,
      T,
      encodeTypedStreamBounds<T>(
          bounds, builder, pool, encodingBuffer, encodedMins, encodedMaxs));
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

  const auto encodedRows = encodeStreamValues(&StreamIndex::chunkRows);
  const auto encodedOffsets = encodeStreamValues(&StreamIndex::chunkOffsets);
  const auto encodedNullCounts =
      encodeStreamValues(&StreamIndex::chunkNullCounts);

  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedMins;
  std::vector<flatbuffers::Offset<serialization::EncodedStream>> encodedMaxs;
  Vector<bool> minMaxPresent{&pool};
  flatbuffers::Offset<serialization::EncodedStream> encodedMinMaxPresent;
  const bool hasAnyMinMax = hasAnyChunkBounds(groupIndex);
  if (hasAnyMinMax) {
    minMaxPresent.reserve(totalNumChunks);
    encodedMins.reserve(streamCount);
    encodedMaxs.reserve(streamCount);
    for (size_t streamId = 0; streamId < streamCount; ++streamId) {
      const auto streamChunkCount =
          streamChunkCounts[streamId * stripeCount + stripeCount - 1];
      const auto bounds = collectStreamBounds(
          groupIndex, streamId, streamChunkCount, minMaxPresent);
      if (bounds.dataType == DataType::Undefined) {
        encodedMins.push_back(emptyEncodedStream(builder));
        encodedMaxs.push_back(emptyEncodedStream(builder));
      } else {
        encodeStreamBounds(
            bounds, builder, pool, encodingBuffer, encodedMins, encodedMaxs);
      }
    }
    encodedMinMaxPresent =
        encodeBooleanArray(builder, minMaxPresent, encodingBuffer);
  }

  builder.Finish(
      serialization::CreateStripeChunkStatsV2(
          builder,
          static_cast<uint32_t>(streamCount),
          builder.CreateVector(streamChunkCounts),
          encodedRows,
          encodedOffsets,
          encodedNullCounts,
          hasAnyMinMax ? builder.CreateVector(encodedMins) : 0,
          hasAnyMinMax ? builder.CreateVector(encodedMaxs) : 0,
          encodedMinMaxPresent));
  return createMetadataSection(asView(builder));
}

} // namespace

std::unique_ptr<ChunkStatsWriter> ChunkStatsWriter::create(
    velox::memory::MemoryPool& pool,
    Options options) {
  switch (options.version) {
    case ChunkStatsVersion::kV1:
    case ChunkStatsVersion::kV2:
      return std::make_unique<ChunkStatsWriterImpl>(pool, std::move(options));
  }
  NIMBLE_UNREACHABLE(
      "Unknown chunk stats version: {}", static_cast<int>(options.version));
}

} // namespace facebook::nimble
