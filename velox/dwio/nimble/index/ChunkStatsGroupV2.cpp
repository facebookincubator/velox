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
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::nimble::index {

namespace {

// Rejects payloads that could make a point-read view access invalid bytes.
void validateEncodedStreamPayload(
    std::string_view encodedView,
    uint32_t rowCount,
    std::string_view fieldName) {
  const auto encodingType = EncodingPrefix::encodingType(encodedView);
  uint64_t expectedSize{EncodingPrefix::kFixedPrefixSize};
  if (encodingType == EncodingType::Constant) {
    NIMBLE_CHECK_FILE_GT(
        rowCount,
        0,
        "Encoded {} array uses Constant encoding for an empty array.",
        fieldName);
    expectedSize += sizeof(uint32_t);
  } else if (encodingType == EncodingType::Trivial) {
    constexpr uint64_t kCompressionTypeOffset =
        EncodingPrefix::kFixedPrefixSize;
    expectedSize += sizeof(uint8_t);
    NIMBLE_CHECK_FILE_GE(
        encodedView.size(),
        expectedSize,
        "Encoded {} array is truncated.",
        fieldName);
    const auto compressionType =
        static_cast<CompressionType>(encodedView[kCompressionTypeOffset]);
    NIMBLE_CHECK_FILE_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "Encoded {} array must be uncompressed.",
        fieldName);
    expectedSize += static_cast<uint64_t>(rowCount) * sizeof(uint32_t);
  } else if (encodingType == EncodingType::FixedBitWidth) {
    constexpr uint64_t kCompressionTypeOffset =
        EncodingPrefix::kFixedPrefixSize;
    constexpr uint64_t kBitWidthOffset =
        kCompressionTypeOffset + sizeof(uint8_t) + sizeof(uint32_t);
    constexpr uint64_t kHeaderSize = kBitWidthOffset + sizeof(uint8_t);
    expectedSize = kHeaderSize;
    NIMBLE_CHECK_FILE_GE(
        encodedView.size(),
        expectedSize,
        "Encoded {} array is truncated.",
        fieldName);
    const auto compressionType =
        static_cast<CompressionType>(encodedView[kCompressionTypeOffset]);
    const auto bitWidth = static_cast<uint8_t>(encodedView[kBitWidthOffset]);
    NIMBLE_CHECK_FILE_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "Encoded {} array must be uncompressed.",
        fieldName);
    NIMBLE_CHECK_FILE(
        (bitWidth > 0 || rowCount == 0) &&
            bitWidth <= std::numeric_limits<uint32_t>::digits,
        "Encoded {} array has an invalid bit width: {}.",
        fieldName,
        bitWidth);
    expectedSize += FixedBitArray::bufferSize(rowCount, bitWidth);
  } else {
    NIMBLE_CHECK_FILE(
        false,
        "Encoded {} array has an unsupported encoding type: {}.",
        fieldName,
        static_cast<int>(encodingType));
  }
  NIMBLE_CHECK_FILE_GE(
      encodedView.size(),
      expectedSize,
      "Encoded {} array is truncated.",
      fieldName);
  NIMBLE_CHECK_FILE_EQ(
      encodedView.size(),
      expectedSize,
      "Encoded {} array has trailing data.",
      fieldName);
}

// Ensures an encoded array can be safely point-read as uint32 values.
std::unique_ptr<TypedEncodingView<uint32_t>> createEncodedStreamView(
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
      DataType::Uint32,
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
  validateEncodedStreamPayload(encodedView, actualRowCount, fieldName);
  return detail::createTypedEncodingView<uint32_t>(
      encodedView, &pool, Encoding::Options{});
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
      std::unique_ptr<TypedEncodingView<uint32_t>> chunkNullCounts);

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

 private:
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

  const auto* encodedRows = root->stream_chunk_rows();
  const auto* encodedOffsets = root->stream_chunk_offsets();
  const auto* encodedNullCounts = root->stream_chunk_null_counts();
  NIMBLE_CHECK_FILE_NOT_NULL(encodedRows, "Missing encoded chunk rows.");
  NIMBLE_CHECK_FILE_NOT_NULL(encodedOffsets, "Missing encoded chunk offsets.");
  NIMBLE_CHECK_FILE_NOT_NULL(
      encodedNullCounts, "Missing encoded chunk null counts.");

  auto streamLayout =
      createStreamLayout(streamChunkCountValues, streamCount, stripeCount);

  auto chunkRows = createEncodedStreamView(
      encodedRows, streamLayout.chunkCount, "chunk rows", pool);
  auto chunkOffsets = createEncodedStreamView(
      encodedOffsets, streamLayout.chunkCount, "chunk offsets", pool);
  auto chunkNullCounts = createEncodedStreamView(
      encodedNullCounts, streamLayout.chunkCount, "chunk null counts", pool);

  return std::shared_ptr<ChunkStatsGroupV2>(new ChunkStatsGroupV2(
      firstStripe,
      stripeCount,
      streamCount,
      std::move(metadata),
      std::move(streamChunkCountValues),
      std::move(streamLayout.baseOffsets),
      std::move(chunkRows),
      std::move(chunkOffsets),
      std::move(chunkNullCounts)));
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
    std::unique_ptr<TypedEncodingView<uint32_t>> chunkNullCounts)
    : ChunkStatsGroup(firstStripe, stripeCount, streamCount),
      metadata_{std::move(metadata)},
      streamChunkCounts_{std::move(streamChunkCounts)},
      streamBaseOffsets_{std::move(streamBaseOffsets)},
      chunkRows_{std::move(chunkRows)},
      chunkOffsets_{std::move(chunkOffsets)},
      chunkNullCounts_{std::move(chunkNullCounts)} {
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

  NIMBLE_CHECK_GE(
      stripe, firstStripe(), "Stripe index is before this group's range");
  const uint32_t stripeOffset = stripe - firstStripe();
  NIMBLE_CHECK_LT(
      stripeOffset,
      numStripes(),
      "Stripe offset is out of range for this chunk stats group");

  // V2 stream_chunk_counts: stream-major per-stream prefix sum.
  // index = streamId * stripeCount + stripeOffset
  const uint32_t chunkCountIndex = streamId * numStripes() + stripeOffset;
  const uint32_t endChunkOffset = streamChunkCounts_[chunkCountIndex];
  const uint32_t startChunkOffset =
      (stripeOffset > 0) ? streamChunkCounts_[chunkCountIndex - 1] : 0;

  if (endChunkOffset - startChunkOffset <= 1) {
    return nullptr;
  }

  return std::make_shared<StreamIndex>(
      std::static_pointer_cast<const ChunkStatsGroupV2>(shared_from_this()),
      streamId,
      streamBaseOffsets_[streamId] + startChunkOffset,
      streamBaseOffsets_[streamId] + endChunkOffset,
      streamSize);
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

std::shared_ptr<ChunkStatsGroup> ChunkStatsGroup::create(
    ChunkStatsVersion version,
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata,
    velox::memory::MemoryPool& pool) {
  switch (version) {
    case ChunkStatsVersion::kV1:
      return createV1(firstStripe, stripeCount, std::move(metadata));
    case ChunkStatsVersion::kV2:
      return ChunkStatsGroupV2::create(
          firstStripe, stripeCount, std::move(metadata), pool);
  }
  NIMBLE_UNREACHABLE(
      "Unsupported chunk stats version: {}.", static_cast<int>(version));
}

} // namespace facebook::nimble::index
