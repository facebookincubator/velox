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

#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <algorithm>

namespace facebook::nimble::index {

namespace {

template <typename T>
const T* asFlatBuffersRoot(std::string_view content) {
  return flatbuffers::GetRoot<T>(content.data());
}

} // namespace

std::shared_ptr<ChunkStatsGroup> ChunkStatsGroup::create(
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata) {
  return std::shared_ptr<ChunkStatsGroup>(
      new ChunkStatsGroup(firstStripe, stripeCount, std::move(metadata)));
}

ChunkStatsGroup::ChunkStatsGroup(
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata)
    : metadata_{std::move(metadata)},
      firstStripe_{firstStripe},
      stripeCount_{stripeCount},
      streamCount_{asFlatBuffersRoot<serialization::StripeChunkStats>(
                       metadata_->content())
                       ->stream_count()} {}

std::shared_ptr<StreamIndex> ChunkStatsGroup::createStreamIndex(
    uint32_t stripe,
    uint32_t streamId,
    uint32_t streamSize) const {
  if (streamId >= streamCount_) {
    return nullptr;
  }

  const uint32_t stripeOff = stripeOffset(stripe);
  const auto* root =
      asFlatBuffersRoot<serialization::StripeChunkStats>(metadata_->content());

  const auto* streamChunkCounts = root->stream_chunk_counts();
  NIMBLE_CHECK_NOT_NULL(streamChunkCounts);

  const uint32_t streamChunkCountIndex = stripeOff * streamCount_ + streamId;
  NIMBLE_CHECK_LT(streamChunkCountIndex, streamChunkCounts->size());

  const uint32_t endChunkOffset = streamChunkCounts->Get(streamChunkCountIndex);
  const uint32_t startChunkOffset = streamChunkCountIndex == 0
      ? 0
      : streamChunkCounts->Get(streamChunkCountIndex - 1);

  // Single-chunk streams don't need index lookup.
  if (endChunkOffset - startChunkOffset <= 1) {
    return nullptr;
  }

  return StreamIndex::create(
      shared_from_this(),
      streamId,
      startChunkOffset,
      endChunkOffset,
      streamSize);
}

std::shared_ptr<StreamIndex> StreamIndex::create(
    std::shared_ptr<const ChunkStatsGroup> chunkStats,
    uint32_t streamId,
    uint32_t startChunkOffset,
    uint32_t endChunkOffset,
    uint32_t streamSize) {
  return std::shared_ptr<StreamIndex>(new StreamIndex(
      std::move(chunkStats),
      streamId,
      startChunkOffset,
      endChunkOffset,
      streamSize));
}

StreamIndex::StreamIndex(
    std::shared_ptr<const ChunkStatsGroup> chunkStats,
    uint32_t streamId,
    uint32_t startChunkOffset,
    uint32_t endChunkOffset,
    uint32_t streamSize)
    : chunkStats_(std::move(chunkStats)),
      streamId_(streamId),
      startChunkOffset_(startChunkOffset),
      endChunkOffset_(endChunkOffset),
      streamSize_(streamSize) {
  NIMBLE_CHECK_NOT_NULL(chunkStats_);
}

ChunkLocation StreamIndex::lookupChunk(uint32_t rowId) const {
  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStats_->metadata_->content());

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
      streamId_);

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

std::optional<uint32_t> StreamIndex::chunkNullCount(uint32_t chunkIndex) const {
  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStats_->metadata_->content());

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

uint32_t StreamIndex::rowCount() const {
  if (endChunkOffset_ == startChunkOffset_) {
    return 0;
  }

  const auto* root = asFlatBuffersRoot<serialization::StripeChunkStats>(
      chunkStats_->metadata_->content());

  const auto* chunkRows = root->stream_chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_LE(endChunkOffset_, chunkRows->size());

  // stream_chunk_rows stores per-stream accumulated row counts, so the last
  // entry in this stream's chunk range is the total row count.
  return chunkRows->Get(endChunkOffset_ - 1);
}

} // namespace facebook::nimble::index
