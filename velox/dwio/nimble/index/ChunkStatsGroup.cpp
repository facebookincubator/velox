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

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <algorithm>

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

  std::shared_ptr<StreamIndex> createStreamIndex(
      uint32_t stripe,
      uint32_t streamId,
      uint32_t streamSize) const final;

  const MetadataBuffer& metadata() const {
    return *metadata_;
  }

 private:
  const std::unique_ptr<MetadataBuffer> metadata_;
};

// Implements V1 stream lookup while retaining its backing metadata.
class StreamIndexV1 final : public StreamIndex {
 public:
  StreamIndexV1(
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

std::shared_ptr<ChunkStatsGroup> ChunkStatsGroup::create(
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata) {
  NIMBLE_CHECK_NOT_NULL(metadata, "Chunk stats metadata must not be null.");
  const auto streamCount =
      asFlatBuffersRoot<serialization::StripeChunkStats>(metadata->content())
          ->stream_count();
  return std::make_shared<ChunkStatsGroupV1>(
      firstStripe, stripeCount, streamCount, std::move(metadata));
}

std::shared_ptr<StreamIndex> ChunkStatsGroupV1::createStreamIndex(
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

  return std::make_shared<StreamIndexV1>(
      std::static_pointer_cast<const ChunkStatsGroupV1>(shared_from_this()),
      streamId,
      startChunkOffset,
      endChunkOffset,
      streamSize);
}

StreamIndexV1::StreamIndexV1(
    std::shared_ptr<const ChunkStatsGroupV1> owner,
    uint32_t streamId,
    uint32_t startChunkOffset,
    uint32_t endChunkOffset,
    uint32_t streamSize)
    : StreamIndex{streamId},
      chunkStatsGroup_{std::move(owner)},
      startChunkOffset_(startChunkOffset),
      endChunkOffset_(endChunkOffset),
      streamSize_(streamSize) {}

StreamIndex::~StreamIndex() = default;

ChunkLocation StreamIndexV1::lookupChunk(uint32_t rowId) const {
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

  const uint32_t chunkIndex = it - chunkRows->begin();
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

std::optional<uint32_t> StreamIndexV1::chunkNullCount(
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

uint32_t StreamIndexV1::rowCount() const {
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

} // namespace facebook::nimble::index
