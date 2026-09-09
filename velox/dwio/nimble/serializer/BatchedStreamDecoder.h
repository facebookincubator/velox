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

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/serializer/StreamData.h"
#include "velox/dwio/nimble/velox/Decoder.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"

namespace facebook::nimble {

class Type;

// Decoder for one logical stream assembled from per-batch segments.
class BatchedStreamDecoder : public Decoder {
 public:
  BatchedStreamDecoder(
      const Type* type,
      bool isInMapStream,
      size_t bufferPoolCapacity,
      velox::memory::MemoryPool* pool);

  uint32_t next(
      uint32_t count,
      void* output,
      std::vector<velox::BufferPtr>& stringBuffers,
      std::function<void*()> getOutputNulls = nullptr,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) override;

  void skip(uint32_t count) override;

  void reset() override;

  void clear();

  const Encoding* encoding() const override;

  static inline BatchedStreamDecoder* as(Decoder* d) {
    return static_cast<BatchedStreamDecoder*>(d);
  }

  // Queues a physical stream segment for one batch. `startRow` is the
  // top-level row where the batch begins in the concatenated run; it's
  // read back by FlatMap in-map reads to detect and fill gaps when
  // earlier batches omitted the stream. Other streams just concatenate
  // in payload order.
  //
  // The segment is stored as raw bytes. The encoding is constructed
  // lazily by `ensureStreamData` the first time this segment is decoded.
  //
  // Defined inline: called once per stream per batch from
  // `Deserializer::appendStreamSegments`, so the cross-translation-unit
  // call would show up on the per-batch hot path.
  void addBatch(
      uint32_t startRow,
      std::string_view data,
      SerializationVersion version,
      bool streamEncodingUsesVarintRowCount) {
    NIMBLE_CHECK(!data.empty(), "Physical stream segment must be non-empty");
    streamSegments_.emplace_back(
        StreamSegment{
            .startRow = startRow,
            .data = data,
            .version = version,
            .streamEncodingUsesVarintRowCount =
                streamEncodingUsesVarintRowCount});
  }

  // Records a batch range where this FlatMap key is present in every row.
  // Called for batches whose in-map stream was omitted on the wire
  // (writer's all-true optimization). Merges into the previous segment
  // when contiguous to keep `presentInMapSegments_` compact; `fillInMapGap`
  // clamps segments that extend past the current read's gap.
  void addPresentInMapBatch(uint32_t startRow, uint32_t rowCount) {
    NIMBLE_CHECK(isInMapStream(), "Expected FlatMap in-map stream");
    NIMBLE_CHECK_GT(
        rowCount, 0, "All-present in-map segment must be non-empty");
    const uint32_t endRow = startRow + rowCount;
    if (!presentInMapSegments_.empty() &&
        presentInMapSegments_.back().endRow == startRow) {
      presentInMapSegments_.back().endRow = endRow;
    } else {
      presentInMapSegments_.emplace_back(InMapSegment{startRow, endRow});
    }
  }

  // Records an all-present FlatMap key range for a null-barrier batch.
  // The read's effective end row (not any per-batch rowCount) determines
  // how many rows are present, so `endRow` stores the sentinel
  // `kPresentInMapEndRow`; the read side clamps as needed.
  void addPresentInMapBatch() {
    NIMBLE_CHECK(isInMapStream(), "Expected FlatMap in-map stream");
    NIMBLE_CHECK(
        streamSegments_.empty(),
        "All-present in-map segment must not be mixed with physical batches");
    presentInMapSegments_.emplace_back(
        InMapSegment{.startRow = 0, .endRow = kPresentInMapEndRow});
  }

 private:
  // Sentinel value stored in `InMapSegment::endRow` when the segment's
  // extent isn't known until the read side (i.e. the parameter-less
  // `addPresentInMapBatch()` used by null-barrier batches). The read side
  // clamps the segment to the current gap when it encounters this value.
  static constexpr uint32_t kPresentInMapEndRow =
      std::numeric_limits<uint32_t>::max();

  // Physical stream data for one batch.
  struct StreamSegment {
    // Top-level row where this batch starts. Only relevant for FlatMap in-map
    // streams to detect gaps when decoding across multiple chunks.
    uint32_t startRow;
    std::string_view data;
    SerializationVersion version;
    bool streamEncodingUsesVarintRowCount;
  };

  // Row range where a FlatMap key is present in every requested row and the
  // in-map stream was omitted from the physical payload.
  struct InMapSegment {
    uint32_t startRow;
    uint32_t endRow;
  };

  // True for the FlatMap child-presence stream, not for the FlatMap
  // value/null stream itself.
  bool isInMapStream() const {
    return isInMapStream_;
  }

  // Returns the `StreamData` for the current segment, creating it lazily
  // on first access. `stringBuffers` is where any string content decoded
  // out of this segment will be pushed; caller retains ownership.
  //
  // On a cache hit (a prior `skip` already built the encoding using our
  // scratch vector), transfer any buffers the skip path pushed into
  // `skipStringBuffers_` into the caller's vector so the output takes
  // shared ownership before the end-of-run `clear()` drops our scratch,
  // then redirect the encoding's factory target for lazy string
  // encodings whose future page allocations must land in the caller's
  // vector, not ours.
  serde::StreamData& ensureStreamData(
      std::vector<velox::BufferPtr>& stringBuffers);

  // Advances to the next segment. ensureStreamData() will create StreamData for
  // the new segment before decoding it.
  void advanceSegment();

  // Fills the in-map output for rows `[rowOffset, gapEndRow)` where
  // `gapEndRow = min(rowOffset + rowCount, next stream segment's startRow)`.
  // Defaults every row to `kInMapAbsent`, then overlays with `kInMapPresent`
  // for every presence segment in `presentInMapSegments_` that overlaps.
  // Returns the number of rows filled.
  //
  // Parameters:
  //   * `rowOffset`   — absolute row in the concatenated batch-run row
  //                     domain (matches `StreamSegment::startRow` and
  //                     `InMapSegment::startRow`). Drives all range math.
  //   * `rowCount`    — upper bound on rows to fill; the actual count is
  //                     capped at the next stream segment's start.
  //   * `outputOffset` — where in `output` to start writing, in element
  //                      slots (multiplied by `typeStorageWidth_`).
  //                      Independent of `rowOffset` because `skip()` moves
  //                      the row cursor without moving the output cursor.
  //   * `output`      — destination buffer.
  //
  // Presence-segment handling:
  //   * A segment fully inside the gap is written and consumed
  //     (`++presentSegmentIndex_`).
  //   * A segment that extends past `gapEndRow` is partially written
  //     (clamped to the gap) and its `startRow` is advanced to `gapEndRow`
  //     so the next call resumes where this one stopped.
  //   * A segment starting before `rowOffset` (a prior skip landed inside
  //     it) is clamped on the low end via `max(segment.startRow, rowOffset)`.
  uint32_t fillInMapGap(
      uint32_t rowOffset,
      uint32_t rowCount,
      uint32_t outputOffset,
      void* output);

  serde::StreamData::DecodeResult readLegacyStreamSegment(
      serde::StreamData& streamData,
      void* output,
      uint32_t offset,
      uint32_t count);

  serde::StreamData::DecodeResult readSegment(
      void* output,
      uint32_t offset,
      uint32_t count,
      const std::function<void*()>& getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap,
      std::vector<velox::BufferPtr>& stringBuffers);

  // Skips up to `numRows` rows from the segment at `streamSegmentIndex_`.
  // Advances `streamSegmentIndex_` past the segment when its remaining
  // rows fit inside `numRows`; otherwise leaves the cursor mid-segment.
  // Returns the number of rows actually skipped (never more than
  // `numRows`, may be less if the segment has fewer remaining).
  uint32_t skipSegment(
      uint32_t numRows,
      std::vector<velox::BufferPtr>& stringBuffers);

  // Skips `numRows` rows for non-in-map columns (dense scalars, FlatMap
  // value / scattered columns). Walks `streamSegments_` from the current
  // cursor, consuming each segment fully until `numRows` are covered.
  void skipEncoded(
      uint32_t numRows,
      std::vector<velox::BufferPtr>& stringBuffers);

  // Skips `numRows` rows for FlatMap in-map streams. Alternates between
  // presence-gap regions (advance `presentSegmentIndex_` via
  // `advanceInMapPresentSegmentIndex`) and encoded segments (advance via
  // `skipSegment`), mirroring the read-side control flow in
  // `inMapRead` but writing nothing.
  void skipInMap(
      uint32_t numRows,
      std::vector<velox::BufferPtr>& stringBuffers);

  // Advances `presentSegmentIndex_` past every presence segment fully
  // contained in `[..., targetRow]` (i.e. `segment.endRow <= targetRow`).
  // A segment that extends past `targetRow` stays as the current segment;
  // partial consumption is tracked implicitly (the read side will clamp
  // it when needed).
  void advanceInMapPresentSegmentIndex(uint32_t targetRow);

  // Reads `count` non-in-map values into dense output row positions.
  uint32_t denseRead(
      uint32_t count,
      void* output,
      const std::function<void*()>& getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers);

  // FlatMap in-map streams still materialize dense bool output. Their physical
  // stream can be omitted for all-absent/all-present batch ranges, so this path
  // reconstructs those gaps while normal dense reads avoid the in-map branches.
  uint32_t inMapRead(
      uint32_t count,
      void* output,
      std::vector<velox::BufferPtr>& stringBuffers);

  // Decode directly to positions where scatterOutputBitmap bits are set. Used
  // for FlatMap value columns where some rows don't have certain keys
  // (inMap=false).
  uint32_t scatteredRead(
      uint32_t count,
      void* output,
      const std::function<void*()>& getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap,
      std::vector<velox::BufferPtr>& stringBuffers);

  // --- Const members (set at construction, never modified) ---
  const Type* const type_;
  velox::memory::MemoryPool* const pool_;
  // True when this decoder reads a FlatMap child in-map presence stream rather
  // than the FlatMap value/null stream.
  const bool isInMapStream_;
  // Cached from type at construction to avoid per-call dispatch.
  const ScalarKind scalarKind_;
  const uint32_t typeStorageWidth_;
  // Pool for encoding scratch buffers (e.g. MainlyConstant's isCommon and
  // otherValues buffers). Persists across reset()/addBatch() cycles so buffers
  // are reused instead of being allocated/freed through MemoryPool each time.
  // Null when buffer pooling is disabled via DeserializerOptions.
  const std::unique_ptr<velox::BufferPool> bufferPool_;
  // Decompression buffer reused across StreamData lifetimes. Persists across
  // reset()/addBatch() cycles so the buffer capacity is reused instead of
  // freed and re-allocated on each segment transition.
  velox::BufferPtr decompressionBuffer_;

  // --- Stream decode state (cleared by reset()) ---
  size_t streamSegmentIndex_{0};
  std::vector<StreamSegment> streamSegments_;

  // --- FlatMap in-map state (cleared by reset()) ---
  size_t presentSegmentIndex_{0};
  std::vector<InMapSegment> presentInMapSegments_;

  // Lazily-created StreamData wrapper reused across physical segments for this
  // stream decoder.
  std::optional<serde::StreamData> streamData_;

  // Row cursor in this decoder's row domain (matches
  // `StreamSegment::startRow` for in-map streams; encoded-row domain
  // otherwise). Bumped by every `next()` and `skip()`. Read by
  // `fillInMapGap` and `skipInMap` for presence-position math. Reset to
  // 0 in `clear()`.
  uint32_t currentRow_{0};

  // Backing storage for `ensureStreamData` when called from `skip*`.
  // Some string encodings allocate their content buffers into this vector
  // at construction time and keep `string_view`s into them; the vector
  // must outlive the encoding that references it, so it lives with the
  // decoder and is cleared inside `clear()` AFTER `streamData_` (and
  // hence the encoding) is destroyed.
  std::vector<velox::BufferPtr> skipStringBuffers_;
};

} // namespace facebook::nimble
