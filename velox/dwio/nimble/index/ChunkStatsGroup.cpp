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
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"

#include "flatbuffers/flatbuffers.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace facebook::nimble::index {

namespace {

template <typename T>
const T* asFlatBuffersRoot(std::string_view content) {
  return flatbuffers::GetRoot<T>(content.data());
}

// Implements V1 chunk statistics backed by FlatBuffer metadata.
class ChunkStatsGroupV1 final : public ChunkStatsGroup {
 public:
  ChunkStatsGroupV1(
      uint32_t firstStripe,
      uint32_t stripeCount,
      uint32_t streamCount,
      std::unique_ptr<MetadataBuffer> metadata);

  std::shared_ptr<index::StreamIndex> createStreamIndex(
      uint32_t stripe,
      uint32_t streamId,
      uint32_t streamSize) const final;

  const MetadataBuffer& metadata() const {
    return *metadata_;
  }

 private:
  class StreamIndex;

  const std::unique_ptr<MetadataBuffer> metadata_;
};

// Implements V1 stream lookup while retaining its backing metadata.
class ChunkStatsGroupV1::StreamIndex final : public index::StreamIndex {
 public:
  StreamIndex(
      std::shared_ptr<const ChunkStatsGroupV1> owner,
      uint32_t streamId,
      uint32_t startChunkOffset,
      uint32_t endChunkOffset,
      uint32_t streamSize);

  ChunkLocation lookupChunk(uint32_t rowId) const final;

  std::optional<uint32_t> chunkNullCount(uint32_t chunkIndex) const final;

  uint32_t rowCount() const final;

 private:
  // Keeps the backing V1 metadata alive for the lifetime of this index.
  const std::shared_ptr<const ChunkStatsGroupV1> chunkStatsGroup_;
  // Identifies the range in the flattened chunk arrays for this stream.
  const uint32_t startChunkOffset_;
  const uint32_t endChunkOffset_;
  // Supplies the end offset of the last chunk.
  const uint32_t streamSize_;
};

} // namespace

namespace test {

const MetadataBuffer& chunkStatsGroupV1MetadataForTest(
    const ChunkStatsGroup& chunkStats) {
  const auto* chunkStatsV1 =
      dynamic_cast<const ChunkStatsGroupV1*>(&chunkStats);
  NIMBLE_CHECK_NOT_NULL(chunkStatsV1, "Expected V1 chunk statistics.");
  return chunkStatsV1->metadata();
}

} // namespace test

ChunkStatsGroup::ChunkStatsGroup(
    uint32_t firstStripe,
    uint32_t stripeCount,
    uint32_t streamCount)
    : firstStripe_{firstStripe},
      stripeCount_{stripeCount},
      streamCount_{streamCount} {}

ChunkStatsGroup::~ChunkStatsGroup() = default;

uint32_t ChunkStatsGroup::stripeOffset(uint32_t stripe) const {
  NIMBLE_CHECK_GE(
      stripe, firstStripe_, "Stripe index is before this group's range");
  const uint32_t offset = stripe - firstStripe_;
  NIMBLE_CHECK_LT(
      offset,
      stripeCount_,
      "Stripe offset is out of range for this chunk stats group");
  return offset;
}

ChunkStatsGroupV1::ChunkStatsGroupV1(
    uint32_t firstStripe,
    uint32_t stripeCount,
    uint32_t streamCount,
    std::unique_ptr<MetadataBuffer> metadata)
    : ChunkStatsGroup(firstStripe, stripeCount, streamCount),
      metadata_{std::move(metadata)} {}

std::shared_ptr<index::StreamIndex> ChunkStatsGroupV1::createStreamIndex(
    uint32_t stripe,
    uint32_t streamId,
    uint32_t streamSize) const {
  if (streamId >= numStreams()) {
    return nullptr;
  }

  const uint32_t stripeOff = stripeOffset(stripe);
  const auto* root =
      asFlatBuffersRoot<serialization::StripeChunkStats>(metadata_->content());

  const auto* streamChunkCounts = root->stream_chunk_counts();
  NIMBLE_CHECK_NOT_NULL(streamChunkCounts);

  const uint32_t streamChunkCountIndex = stripeOff * numStreams() + streamId;
  NIMBLE_CHECK_LT(streamChunkCountIndex, streamChunkCounts->size());

  const uint32_t endChunkOffset = streamChunkCounts->Get(streamChunkCountIndex);
  const uint32_t startChunkOffset = streamChunkCountIndex == 0
      ? 0
      : streamChunkCounts->Get(streamChunkCountIndex - 1);

  // Single-chunk streams don't need index lookup.
  if (endChunkOffset - startChunkOffset <= 1) {
    return nullptr;
  }

  return std::make_shared<StreamIndex>(
      std::static_pointer_cast<const ChunkStatsGroupV1>(shared_from_this()),
      streamId,
      startChunkOffset,
      endChunkOffset,
      streamSize);
}

ChunkStatsGroupV1::StreamIndex::StreamIndex(
    std::shared_ptr<const ChunkStatsGroupV1> owner,
    uint32_t streamId,
    uint32_t startChunkOffset,
    uint32_t endChunkOffset,
    uint32_t streamSize)
    : index::StreamIndex{streamId},
      chunkStatsGroup_{std::move(owner)},
      startChunkOffset_(startChunkOffset),
      endChunkOffset_(endChunkOffset),
      streamSize_(streamSize) {}

StreamIndex::~StreamIndex() = default;

std::optional<ChunkStatValue> StreamIndex::chunkMinValue(
    uint32_t /*chunkIndex*/) const {
  return std::nullopt;
}

std::optional<ChunkStatValue> StreamIndex::chunkMaxValue(
    uint32_t /*chunkIndex*/) const {
  return std::nullopt;
}

ChunkLocation ChunkStatsGroupV1::StreamIndex::lookupChunk(
    uint32_t rowId) const {
  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStatsGroup_->metadata().content());

  const auto* chunkRows = root->stream_chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_LE(endChunkOffset_, chunkRows->size());

  const auto beginIt = chunkRows->begin() + startChunkOffset_;
  const auto endIt = chunkRows->begin() + endChunkOffset_;
  const auto it = std::lower_bound(beginIt, endIt, rowId + 1);
  NIMBLE_CHECK(
      it != endIt,
      "Row ID {} is beyond the last chunk in stream {}",
      rowId,
      streamId());

  const auto chunkIndex = static_cast<uint32_t>(it - chunkRows->begin());
  const auto* chunkOffsets = root->stream_chunk_offsets();
  NIMBLE_CHECK_NOT_NULL(chunkOffsets);
  const uint32_t rowOffset =
      chunkIndex == startChunkOffset_ ? 0 : chunkRows->Get(chunkIndex - 1);
  const uint32_t streamOffset = chunkOffsets->Get(chunkIndex);
  const uint32_t nextOffset = (chunkIndex + 1 < endChunkOffset_)
      ? chunkOffsets->Get(chunkIndex + 1)
      : streamSize_;
  return ChunkLocation{
      chunkIndex, streamOffset, nextOffset - streamOffset, rowOffset};
}

std::optional<uint32_t> ChunkStatsGroupV1::StreamIndex::chunkNullCount(
    uint32_t chunkIndex) const {
  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStatsGroup_->metadata().content());

  const auto* nullCounts = root->stream_chunk_null_counts();
  if (nullCounts == nullptr) {
    // Absent in files written before per-chunk null statistics were added; the
    // null count is unknown.
    return std::nullopt;
  }
  // The array is present, so chunkIndex (derived from ChunkLocation) must be in
  // range. An out-of-range index is a programmer error or malformed metadata,
  // not a legacy file, so fail loudly rather than masking it as "unknown".
  NIMBLE_CHECK_LT(chunkIndex, nullCounts->size());
  return nullCounts->Get(chunkIndex);
}

uint32_t ChunkStatsGroupV1::StreamIndex::rowCount() const {
  if (endChunkOffset_ == startChunkOffset_) {
    return 0;
  }

  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStatsGroup_->metadata().content());

  const auto* chunkRows = root->stream_chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_LE(endChunkOffset_, chunkRows->size());

  // stream_chunk_rows stores per-stream accumulated row counts, so the last
  // entry in this stream's chunk range is the total row count.
  return chunkRows->Get(endChunkOffset_ - 1);
}

namespace {

// Creates a typed view after validating the chunk-stats stream contract.
template <typename T>
std::unique_ptr<TypedEncodingView<T>> createEncodedStreamView(
    const serialization::EncodedStream* encoded,
    uint32_t expectedRowCount,
    std::string_view fieldName,
    velox::memory::MemoryPool& pool) {
  NIMBLE_CHECK_FILE_NOT_NULL(encoded, "Missing encoded {} array.", fieldName);
  const auto* data = encoded->data();
  NIMBLE_CHECK_FILE_NOT_NULL(data, "Missing encoded {} data.", fieldName);
  const std::string_view encodedView{
      reinterpret_cast<const char*>(data->data()), data->size()};
  NIMBLE_CHECK_FILE_GE(
      encodedView.size(),
      EncodingPrefix::kFixedPrefixSize,
      "Encoded {} array is too small.",
      fieldName);
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(encodedView),
      TypeTraits<T>::dataType,
      "Encoded {} array has an invalid data type.",
      fieldName);
  const auto actualRowCount =
      EncodingPrefix::readRowCount(encodedView, /*useVarint=*/false);
  NIMBLE_CHECK_FILE_EQ(
      actualRowCount,
      expectedRowCount,
      "Encoded {} row count does not match chunk counts ({} vs. {}).",
      fieldName,
      actualRowCount,
      expectedRowCount);
  return detail::createTypedEncodingView<T>(
      encodedView, &pool, Encoding::Options{});
}

using ChunkBoundViews =
    std::pair<std::unique_ptr<EncodingView>, std::unique_ptr<EncodingView>>;

template <typename T>
ChunkBoundViews createTypedChunkBoundViews(
    const serialization::EncodedStream* encodedMin,
    const serialization::EncodedStream* encodedMax,
    uint32_t streamChunkCount,
    velox::memory::MemoryPool& pool) {
  return {
      createEncodedStreamView<T>(
          encodedMin, streamChunkCount, "chunk min values", pool),
      createEncodedStreamView<T>(
          encodedMax, streamChunkCount, "chunk max values", pool),
  };
}

ChunkBoundViews createChunkBoundViews(
    DataType dataType,
    const serialization::EncodedStream* encodedMin,
    const serialization::EncodedStream* encodedMax,
    uint32_t streamChunkCount,
    velox::memory::MemoryPool& pool) {
  NIMBLE_RETURN_BY_DATA_TYPE_OR(
      // The macro's default case handles Undefined and unsupported types.
      // @lint-ignore CLANGTIDY clang-diagnostic-switch-enum
      dataType,
      T,
      createTypedChunkBoundViews<T>(
          encodedMin, encodedMax, streamChunkCount, pool),
      NIMBLE_FILE_FAIL(
          "Unsupported V2 chunk bounds data type: {}.",
          static_cast<int>(dataType)));
}

struct StreamLayout {
  // Stores the absolute starting chunk index for each stream.
  std::vector<uint32_t> baseOffsets;
  // Stores the total number of chunks across all streams.
  uint32_t chunkCount;
};

// Converts stream-major prefix counts into absolute point-read offsets.
StreamLayout createStreamLayout(
    const std::vector<uint32_t>& streamChunkCounts,
    uint32_t streamCount,
    uint32_t stripeCount) {
  std::vector<uint32_t> baseOffsets(streamCount);
  uint64_t totalChunkCount{0};
  for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
    NIMBLE_CHECK_FILE_LE(totalChunkCount, std::numeric_limits<uint32_t>::max());
    baseOffsets[streamId] = static_cast<uint32_t>(totalChunkCount);
    uint32_t previousChunkCount{0};
    for (uint32_t stripeOffset = 0; stripeOffset < stripeCount;
         ++stripeOffset) {
      const auto chunkCount =
          streamChunkCounts.at(streamId * stripeCount + stripeOffset);
      NIMBLE_CHECK_FILE_GE(
          chunkCount,
          previousChunkCount,
          "V2 stream chunk counts must be non-decreasing within a stream.");
      previousChunkCount = chunkCount;
    }
    totalChunkCount += previousChunkCount;
  }
  NIMBLE_CHECK_FILE_LE(totalChunkCount, std::numeric_limits<uint32_t>::max());
  return {std::move(baseOffsets), static_cast<uint32_t>(totalChunkCount)};
}

bool hasEncodedData(const serialization::EncodedStream* encoded) {
  if (encoded == nullptr) {
    return false;
  }
  const auto* data = encoded->data();
  return data != nullptr && data->size() > 0;
}

// Reads V2 chunk statistics directly from their Nimble-encoded arrays. Relies
// on the writer to preserve row and offset ordering to keep index creation
// O(1).
class ChunkStatsGroupV2 final : public ChunkStatsGroup {
 public:
  static std::shared_ptr<ChunkStatsGroupV2> create(
      uint32_t firstStripe,
      uint32_t stripeCount,
      std::unique_ptr<MetadataBuffer> metadata,
      velox::memory::MemoryPool& pool);

  std::shared_ptr<index::StreamIndex> createStreamIndex(
      uint32_t stripe,
      uint32_t streamId,
      uint32_t streamSize) const final;

 private:
  // Owns lazy per-chunk bounds for one stream.
  struct StreamChunkBounds {
    DataType dataType{DataType::Undefined};
    std::unique_ptr<EncodingView> min;
    std::unique_ptr<EncodingView> max;
  };

  class StreamIndex;

  ChunkStatsGroupV2(
      uint32_t firstStripe,
      uint32_t stripeCount,
      uint32_t streamCount,
      std::unique_ptr<MetadataBuffer> metadata,
      std::vector<uint32_t> streamChunkCounts,
      std::vector<uint32_t> streamBaseOffsets,
      std::unique_ptr<TypedEncodingView<uint32_t>> chunkRows,
      std::unique_ptr<TypedEncodingView<uint32_t>> chunkOffsets,
      std::unique_ptr<TypedEncodingView<uint32_t>> chunkNullCounts,
      std::unique_ptr<TypedEncodingView<bool>> chunkMinMaxPresent,
      std::vector<StreamChunkBounds> streamChunkBounds);

  // Returns whether a chunk has usable bounds.
  bool hasChunkBounds(uint32_t streamId, uint32_t chunkIndex) const;

  // Keeps the encoded data referenced by the views alive.
  const std::unique_ptr<MetadataBuffer> metadata_;

  // Stores per-stream accumulated chunk counts in stream-major order.
  const std::vector<uint32_t> streamChunkCounts_;

  // Stores the number of chunks belonging to earlier streams.
  const std::vector<uint32_t> streamBaseOffsets_;

  // Provide point reads over stream-major per-chunk values.
  const std::unique_ptr<TypedEncodingView<uint32_t>> chunkRows_;
  const std::unique_ptr<TypedEncodingView<uint32_t>> chunkOffsets_;
  const std::unique_ptr<TypedEncodingView<uint32_t>> chunkNullCounts_;

  // Identifies which stream-major chunk positions have valid bounds.
  const std::unique_ptr<TypedEncodingView<bool>> chunkMinMaxPresent_;

  // Stores lazy typed bounds by stream because scalar types can differ.
  const std::vector<StreamChunkBounds> streamChunkBounds_;
};

} // namespace

// Provides a stream-scoped view over its owning V2 chunk stats group.
class ChunkStatsGroupV2::StreamIndex final : public index::StreamIndex {
 public:
  StreamIndex(
      std::shared_ptr<const ChunkStatsGroupV2> owner,
      uint32_t streamId,
      uint32_t startOffset,
      uint32_t endOffset,
      uint32_t streamSize)
      : index::StreamIndex{streamId},
        owner_{std::move(owner)},
        startOffset_{startOffset},
        endOffset_{endOffset},
        streamSize_{streamSize} {
    NIMBLE_CHECK_NOT_NULL(owner_);
    NIMBLE_CHECK_LE(startOffset_, endOffset_);
    NIMBLE_CHECK_LE(endOffset_, owner_->chunkRows_->rowCount());
  }

  ChunkLocation lookupChunk(uint32_t rowId) const override;

  std::optional<uint32_t> chunkNullCount(uint32_t chunkIndex) const override;

  uint32_t rowCount() const override;
  std::optional<ChunkStatValue> chunkMinValue(
      uint32_t chunkIndex) const override;
  std::optional<ChunkStatValue> chunkMaxValue(
      uint32_t chunkIndex) const override;

 private:
  // Converts an absolute chunk index to this stream's bounds index.
  uint32_t streamChunkIndex(uint32_t chunkIndex) const;

  // Keeps the owning group and its encoded arrays alive.
  const std::shared_ptr<const ChunkStatsGroupV2> owner_;
  // Delimit this stream's absolute range in the shared chunk arrays.
  const uint32_t startOffset_;
  const uint32_t endOffset_;
  // Supplies the end offset of the last chunk.
  const uint32_t streamSize_;
};

std::shared_ptr<ChunkStatsGroupV2> ChunkStatsGroupV2::create(
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata,
    velox::memory::MemoryPool& pool) {
  const auto* metadataPtr = metadata.get();
  NIMBLE_CHECK_NOT_NULL(metadataPtr, "Chunk stats metadata must not be null.");
  NIMBLE_CHECK_GT(stripeCount, 0);
  const auto content = metadataPtr->content();
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(content.data()), content.size()};
  NIMBLE_CHECK_FILE(
      verifier.VerifyBuffer<serialization::StripeChunkStatsV2>(nullptr),
      "Invalid V2 chunk stats metadata.");
  const auto* root =
      flatbuffers::GetRoot<serialization::StripeChunkStatsV2>(content.data());
  NIMBLE_CHECK_FILE(root != nullptr, "Missing V2 chunk stats root.");

  const uint32_t streamCount = root->stream_count();
  NIMBLE_CHECK_FILE_GT(streamCount, 0, "V2 chunk stats has no streams.");

  const auto* streamChunkCounts = root->stream_chunk_counts();
  NIMBLE_CHECK_FILE_NOT_NULL(
      streamChunkCounts, "Missing V2 stream chunk counts.");
  const auto expectedStreamChunkCountSize =
      static_cast<uint64_t>(streamCount) * stripeCount;
  NIMBLE_CHECK_FILE_EQ(
      streamChunkCounts->size(),
      expectedStreamChunkCountSize,
      "V2 stream chunk count size does not match stream and stripe counts "
      "({} vs. {}).",
      streamChunkCounts->size(),
      expectedStreamChunkCountSize);

  std::vector<uint32_t> streamChunkCountValues(
      streamChunkCounts->begin(), streamChunkCounts->end());

  // Create EncodingViews over the encoded arrays (O(1), no materialization).
  const auto* encodedRows = root->stream_chunk_rows();
  const auto* encodedOffsets = root->stream_chunk_offsets();
  const auto* encodedNullCounts = root->stream_chunk_null_counts();
  NIMBLE_CHECK_FILE_NOT_NULL(encodedRows, "Missing encoded chunk rows.");
  NIMBLE_CHECK_FILE_NOT_NULL(encodedOffsets, "Missing encoded chunk offsets.");
  NIMBLE_CHECK_FILE_NOT_NULL(
      encodedNullCounts, "Missing encoded chunk null counts.");

  auto streamLayout =
      createStreamLayout(streamChunkCountValues, streamCount, stripeCount);

  auto chunkRows = createEncodedStreamView<uint32_t>(
      encodedRows, streamLayout.chunkCount, "chunk rows", pool);
  auto chunkOffsets = createEncodedStreamView<uint32_t>(
      encodedOffsets, streamLayout.chunkCount, "chunk offsets", pool);
  auto chunkNullCounts = createEncodedStreamView<uint32_t>(
      encodedNullCounts, streamLayout.chunkCount, "chunk null counts", pool);

  const auto createGroup =
      [&](std::unique_ptr<TypedEncodingView<bool>> minMaxPresent,
          std::vector<StreamChunkBounds> streamChunkBounds) {
        return std::shared_ptr<ChunkStatsGroupV2>(new ChunkStatsGroupV2(
            firstStripe,
            stripeCount,
            streamCount,
            std::move(metadata),
            std::move(streamChunkCountValues),
            std::move(streamLayout.baseOffsets),
            std::move(chunkRows),
            std::move(chunkOffsets),
            std::move(chunkNullCounts),
            std::move(minMaxPresent),
            std::move(streamChunkBounds)));
      };

  // Create lazy views over per-stream min/max values.
  const auto* encodedMins = root->chunk_min_values();
  const auto* encodedMaxs = root->chunk_max_values();
  const auto* encodedMinMaxPresent = root->chunk_min_max_present();
  if (encodedMins == nullptr && encodedMaxs == nullptr &&
      encodedMinMaxPresent == nullptr) {
    return createGroup(nullptr, std::vector<StreamChunkBounds>(streamCount));
  }

  NIMBLE_CHECK_FILE(
      encodedMins != nullptr && encodedMaxs != nullptr &&
          encodedMinMaxPresent != nullptr &&
          encodedMins->size() == streamCount &&
          encodedMaxs->size() == streamCount,
      "V2 chunk bounds metadata is incomplete or has an invalid stream count.");
  auto minMaxPresent = createEncodedStreamView<bool>(
      encodedMinMaxPresent,
      streamLayout.chunkCount,
      "chunk min/max presence",
      pool);
  std::vector<StreamChunkBounds> streamChunkBounds(streamCount);
  for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
    const auto* encodedMin = encodedMins->Get(streamId);
    const auto* encodedMax = encodedMaxs->Get(streamId);
    const bool hasMin = hasEncodedData(encodedMin);
    const bool hasMax = hasEncodedData(encodedMax);
    NIMBLE_CHECK_FILE_EQ(
        hasMin,
        hasMax,
        "V2 chunk min and max data must both be present or absent.");
    if (!hasMin) {
      continue;
    }

    NIMBLE_CHECK_FILE_NOT_NULL(encodedMin, "Missing encoded chunk min values.");
    NIMBLE_CHECK_FILE_NOT_NULL(
        encodedMin->data(), "Missing encoded chunk min values data.");
    const auto streamChunkCount =
        streamChunkCountValues.at(streamId * stripeCount + stripeCount - 1);
    const std::string_view minView{
        reinterpret_cast<const char*>(encodedMin->data()->data()),
        encodedMin->data()->size()};
    NIMBLE_CHECK_FILE_GE(
        minView.size(),
        EncodingPrefix::kFixedPrefixSize,
        "Encoded chunk min values array is too small.");
    const auto dataType = EncodingPrefix::dataType(minView);
    streamChunkBounds[streamId].dataType = dataType;
    auto [min, max] = createChunkBoundViews(
        dataType, encodedMin, encodedMax, streamChunkCount, pool);
    streamChunkBounds[streamId].min = std::move(min);
    streamChunkBounds[streamId].max = std::move(max);
  }
  return createGroup(std::move(minMaxPresent), std::move(streamChunkBounds));
}

ChunkStatsGroupV2::ChunkStatsGroupV2(
    uint32_t firstStripe,
    uint32_t stripeCount,
    uint32_t streamCount,
    std::unique_ptr<MetadataBuffer> metadata,
    std::vector<uint32_t> streamChunkCounts,
    std::vector<uint32_t> streamBaseOffsets,
    std::unique_ptr<TypedEncodingView<uint32_t>> chunkRows,
    std::unique_ptr<TypedEncodingView<uint32_t>> chunkOffsets,
    std::unique_ptr<TypedEncodingView<uint32_t>> chunkNullCounts,
    std::unique_ptr<TypedEncodingView<bool>> chunkMinMaxPresent,
    std::vector<StreamChunkBounds> streamChunkBounds)
    : ChunkStatsGroup(firstStripe, stripeCount, streamCount),
      metadata_{std::move(metadata)},
      streamChunkCounts_{std::move(streamChunkCounts)},
      streamBaseOffsets_{std::move(streamBaseOffsets)},
      chunkRows_{std::move(chunkRows)},
      chunkOffsets_{std::move(chunkOffsets)},
      chunkNullCounts_{std::move(chunkNullCounts)},
      chunkMinMaxPresent_{std::move(chunkMinMaxPresent)},
      streamChunkBounds_{std::move(streamChunkBounds)} {
  NIMBLE_CHECK_NOT_NULL(metadata_);
  NIMBLE_CHECK_NOT_NULL(chunkRows_);
  NIMBLE_CHECK_NOT_NULL(chunkOffsets_);
  NIMBLE_CHECK_NOT_NULL(chunkNullCounts_);
}

std::shared_ptr<index::StreamIndex> ChunkStatsGroupV2::createStreamIndex(
    uint32_t stripe,
    uint32_t streamId,
    uint32_t streamSize) const {
  if (streamId >= numStreams()) {
    return nullptr;
  }

  const uint32_t stripeOffset = this->stripeOffset(stripe);

  // V2 stream_chunk_counts: stream-major per-stream prefix sum.
  // index = streamId * stripeCount + stripeOffset
  const uint32_t chunkCountIndex = streamId * numStripes() + stripeOffset;
  const uint32_t endChunkOffset = streamChunkCounts_[chunkCountIndex];
  const uint32_t startChunkOffset =
      (stripeOffset > 0) ? streamChunkCounts_[chunkCountIndex - 1] : 0;
  const uint32_t absoluteStartChunkOffset =
      streamBaseOffsets_[streamId] + startChunkOffset;

  const auto chunkCount = endChunkOffset - startChunkOffset;
  if (chunkCount == 0 ||
      (chunkCount == 1 &&
       !hasChunkBounds(streamId, absoluteStartChunkOffset))) {
    return nullptr;
  }

  const uint32_t absoluteEndChunkOffset =
      streamBaseOffsets_[streamId] + endChunkOffset;
  return std::make_shared<StreamIndex>(
      std::static_pointer_cast<const ChunkStatsGroupV2>(shared_from_this()),
      streamId,
      absoluteStartChunkOffset,
      absoluteEndChunkOffset,
      streamSize);
}

bool ChunkStatsGroupV2::hasChunkBounds(uint32_t streamId, uint32_t chunkIndex)
    const {
  if (chunkMinMaxPresent_ == nullptr) {
    return false;
  }
  bool present;
  chunkMinMaxPresent_->readAt(chunkIndex, &present);
  if (!present) {
    return false;
  }
  const auto& bounds = streamChunkBounds_[streamId];
  NIMBLE_CHECK_FILE(
      bounds.min != nullptr && bounds.max != nullptr,
      "V2 chunk bounds are marked present but stream {} has no bounds data.",
      streamId);
  return true;
}

ChunkLocation ChunkStatsGroupV2::StreamIndex::lookupChunk(
    uint32_t rowId) const {
  uint32_t begin = startOffset_;
  uint32_t end = endOffset_;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (owner_->chunkRows_->readAt(middle) <= rowId) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  NIMBLE_CHECK(
      begin != endOffset_,
      "Row ID {} is beyond the last chunk in stream {}",
      rowId,
      streamId());

  const uint32_t chunkIndex = begin;
  const uint32_t rowOffset = chunkIndex == startOffset_
      ? 0
      : owner_->chunkRows_->readAt(chunkIndex - 1);
  NIMBLE_CHECK_LE(rowOffset, rowId);
  const uint32_t streamOffset = owner_->chunkOffsets_->readAt(chunkIndex);
  const uint32_t nextOffset = (chunkIndex + 1 < endOffset_)
      ? owner_->chunkOffsets_->readAt(chunkIndex + 1)
      : streamSize_;
  NIMBLE_CHECK_LE(streamOffset, nextOffset);
  NIMBLE_CHECK_LE(nextOffset, streamSize_);
  return ChunkLocation{
      chunkIndex, streamOffset, nextOffset - streamOffset, rowOffset};
}

std::optional<uint32_t> ChunkStatsGroupV2::StreamIndex::chunkNullCount(
    uint32_t chunkIndex) const {
  NIMBLE_CHECK_LT(chunkIndex, owner_->chunkNullCounts_->rowCount());
  return owner_->chunkNullCounts_->readAt(chunkIndex);
}

uint32_t ChunkStatsGroupV2::StreamIndex::rowCount() const {
  if (endOffset_ == startOffset_) {
    return 0;
  }
  return owner_->chunkRows_->readAt(endOffset_ - 1);
}

uint32_t ChunkStatsGroupV2::StreamIndex::streamChunkIndex(
    uint32_t chunkIndex) const {
  NIMBLE_CHECK_GE(chunkIndex, startOffset_);
  NIMBLE_CHECK_LT(chunkIndex, endOffset_);
  return chunkIndex - owner_->streamBaseOffsets_[streamId()];
}

namespace {

struct ChunkBounds {
  ChunkStatValue min;
  ChunkStatValue max;
};

template <typename T>
ChunkBounds readTypedChunkBounds(
    const EncodingView& minView,
    const EncodingView& maxView,
    uint32_t index) {
  T min;
  T max;
  minView.readAt(index, &min);
  maxView.readAt(index, &max);
  if constexpr (std::is_floating_point_v<T>) {
    NIMBLE_CHECK_FILE(
        !std::isnan(min) && !std::isnan(max),
        "V2 chunk bounds must not be NaN.");
  }
  NIMBLE_CHECK_FILE_LE(min, max, "V2 chunk minimum must not exceed maximum.");
  if constexpr (std::is_same_v<T, std::string_view>) {
    return {std::string{min}, std::string{max}};
  } else {
    return {min, max};
  }
}

ChunkBounds readChunkBounds(
    DataType dataType,
    const EncodingView* minView,
    const EncodingView* maxView,
    uint32_t index) {
  NIMBLE_CHECK_FILE_NOT_NULL(minView, "Missing V2 chunk minimum values.");
  NIMBLE_CHECK_FILE_NOT_NULL(maxView, "Missing V2 chunk maximum values.");
  NIMBLE_RETURN_BY_DATA_TYPE(
      // The macro's default case handles Undefined.
      // @lint-ignore CLANGTIDY clang-diagnostic-switch-enum
      dataType,
      T,
      readTypedChunkBounds<T>(*minView, *maxView, index));
}

} // namespace

std::optional<ChunkStatValue> ChunkStatsGroupV2::StreamIndex::chunkMinValue(
    uint32_t chunkIndex) const {
  const auto index = streamChunkIndex(chunkIndex);
  if (!owner_->hasChunkBounds(streamId(), chunkIndex)) {
    return std::nullopt;
  }
  const auto& streamBounds = owner_->streamChunkBounds_[streamId()];
  return readChunkBounds(
             streamBounds.dataType,
             streamBounds.min.get(),
             streamBounds.max.get(),
             index)
      .min;
}

std::optional<ChunkStatValue> ChunkStatsGroupV2::StreamIndex::chunkMaxValue(
    uint32_t chunkIndex) const {
  const auto index = streamChunkIndex(chunkIndex);
  if (!owner_->hasChunkBounds(streamId(), chunkIndex)) {
    return std::nullopt;
  }
  const auto& streamBounds = owner_->streamChunkBounds_[streamId()];
  return readChunkBounds(
             streamBounds.dataType,
             streamBounds.min.get(),
             streamBounds.max.get(),
             index)
      .max;
}

namespace {

std::shared_ptr<ChunkStatsGroup> createChunkStatsGroupV1(
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata) {
  const auto* metadataPtr = metadata.get();
  NIMBLE_CHECK_NOT_NULL(metadataPtr, "Chunk stats metadata must not be null.");
  const auto streamCount =
      asFlatBuffersRoot<serialization::StripeChunkStats>(metadataPtr->content())
          ->stream_count();
  return std::make_shared<ChunkStatsGroupV1>(
      firstStripe, stripeCount, streamCount, std::move(metadata));
}

} // namespace

std::shared_ptr<ChunkStatsGroup> ChunkStatsGroup::create(
    ChunkStatsVersion version,
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata,
    velox::memory::MemoryPool& pool) {
  switch (version) {
    case ChunkStatsVersion::kV1:
      return createChunkStatsGroupV1(
          firstStripe, stripeCount, std::move(metadata));
    case ChunkStatsVersion::kV2:
      return ChunkStatsGroupV2::create(
          firstStripe, stripeCount, std::move(metadata), pool);
  }
  NIMBLE_UNREACHABLE(
      "Unsupported chunk stats version: {}.", static_cast<int>(version));
}

} // namespace facebook::nimble::index
