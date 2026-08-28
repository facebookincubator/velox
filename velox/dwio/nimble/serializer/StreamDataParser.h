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
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/io/IOBuf.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/ChunkedStreamPayload.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"

namespace facebook::nimble::serde {

namespace detail {

/// Reads the trailer size (u32) from the last 4 bytes of the buffer.
inline uint32_t readTrailerSize(const char* end) {
  const char* pos = end - sizeof(uint32_t);
  return encoding::readUint32(pos);
}

/// Reads the two-array sparse stream-sizes trailer from the end of a
/// contiguous buffer. Fills `streamIds` (non-zero stream slot ids, sorted
/// ascending) and `streamSizes` (their byte sizes), parallel arrays of
/// identical length. Both vectors are reusable buffers owned by
/// the caller (e.g. members on `StreamDataParser`) to keep the per-blob hot
/// path alloc-free across invocations.
void readTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamSizes);

/// Value-returning convenience overload for cold-path consumers
/// (tests, dump tools). Returns parallel (streamIds, streamSizes) arrays.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readTrailerStreamMetadata(const char* end);

/// IOBuf overload: reads the trailer from a (possibly chained) IOBuf.
/// Tries the fast path first: if the tail segment contains the entire
/// trailer, delegates to the contiguous overload. Falls back to
/// cursor + pull() when the trailer spans a chain boundary.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readTrailerStreamMetadata(const folly::IOBuf& input);

/// Reads the kTablet dedup trailer written by the three-section `writeTrailer`
/// overload and reconstructs the same parallel `streamIds`,
/// `streamOffsets`, and `streamSizes` arrays used by the kTablet decode path.
/// `uniqueStreamSizesScratch` is a caller-owned reusable buffer: it receives
/// the decoded unique-size table and is then transformed in place into the
/// unique-offset (prefix-sum) table. Keeping it caller-owned keeps the per-blob
/// hot path alloc-free.
void readTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamOffsets,
    std::vector<uint32_t>& streamSizes,
    std::vector<uint32_t>& uniqueStreamSizesScratch);

} // namespace detail

class StreamDataParser {
 public:
  StreamDataParser(
      velox::memory::MemoryPool* pool,
      const DeserializerOptions& options);

  /// Returns number of rows serialized.
  /// Validates that the version in serialized data matches options.
  ///
  /// PRECONDITION (kTablet only): the per-slice header
  /// (`[version][rowCount:varint][startRow:varint][endRow:varint]`
  /// `[resumeKeyLength:varint][resumeKey]`) must live contiguously within
  /// `data`. The projector emits the header in a single allocation; consumers
  /// that hand chunk slices to this method should coalesce the IOBuf chain
  /// first (`folly::IOBuf::cloneCoalescedAsValue()`).
  uint32_t initialize(std::string_view data);

  /// Walks every non-empty stream in the current blob, invoking
  /// `callback(offset, data)` per stream. For kTablet, `data` is the
  /// chunk-stripped (and decompressed, if needed) payload; for kLegacyCompact
  /// and kLegacy it is the raw stream bytes. Empty streams are skipped.
  /// Must be called at most once per `initialize()` (consumes the per-blob
  /// cursor).
  ///
  /// PERFORMANCE: This is a function template (rather than a function taking
  /// `std::function`/`folly::FunctionRef`) on purpose. The callback's call
  /// type is concrete at the call site, so `callback(offset, data)` becomes
  /// a direct call the compiler can inline into the loop body. The earlier
  /// `std::function`-based variant paid an indirect-call dispatch
  /// (`_Function_handler::_M_invoke`) per stream — measurable on per-batch
  /// hot paths with many streams (the Deserializer's flatmap deserialization
  /// over hundreds of keys).
  template <typename Callback>
  void iterateStreams(Callback&& callback) {
    if (isTabletVersion(version_)) {
      // kTablet: the trailer stores stream ids, per-stream size indices, and
      // unique sizes. Per-slot sizes and body offsets are reconstructed from
      // (sizeIndex, uniqueSizes); duplicate slots resolve to a single body
      // extent, so streams are addressed by the reconstructed offset rather
      // than walked sequentially.
      detail::readTrailerStreamMetadata(
          end_,
          streamIds_,
          streamOffsets_,
          streamSizes_,
          uniqueStreamSizesScratch_);
      const char* const bodyBase = pos_;
      const size_t numStreams = streamIds_.size();
      for (size_t entryIdx = 0; entryIdx < numStreams; ++entryIdx) {
        const uint32_t streamId = streamIds_[entryIdx];
        const uint32_t streamSize = streamSizes_[entryIdx];
        std::string_view streamData(
            bodyBase + streamOffsets_[entryIdx], streamSize);
        if (streamHasChunkHeader_) {
          callback(
              streamId,
              stripChunkHeaders(streamData, ensureStrippedStreamBuffer()));
        } else {
          callback(streamId, streamData);
        }
      }
      pos_ = end_; // Skip past trailer.
    } else if (nonLegacyFormat(version_)) {
      // kLegacyCompact/kLegacySerialization/kSerialization/kProjection:
      // sizes-only sparse trailer.
      // Each stream's body offset is the prefix sum of preceding sizes, so
      // streams are walked sequentially. Production blobs at kLegacyCompact are
      // decoded via the legacy reader (frozen snapshot of the pre-two-array
      // wire format); newer versions use the two-array sparse trailer reader.
      // Reusable member buffers keep this path alloc-free across blobs.
      if (usesLegacyTrailer(version_)) {
        legacy::readLegacyTrailerStreamMetadata(end_, streamIds_, streamSizes_);
      } else {
        detail::readTrailerStreamMetadata(end_, streamIds_, streamSizes_);
      }
      const size_t numStreams = streamIds_.size();
      for (size_t entryIdx = 0; entryIdx < numStreams; ++entryIdx) {
        const uint32_t streamId = streamIds_[entryIdx];
        const uint32_t streamSize = streamSizes_[entryIdx];
        // Writer invariant: the sparse trailer only encodes non-zero stream
        // slots (StreamDataWriter.h writeTrailer skips streamSizes[i]==0).
        // Debug-only assert documents that invariant; release elides it.
        NIMBLE_DCHECK_GT(
            streamSize, 0, "Sparse trailer must not encode zero-sized stream");
        std::string_view streamData(pos_, streamSize);
        pos_ += streamSize;
        callback(streamId, streamData);
      }
      pos_ = end_; // Skip past trailer.
    } else {
      // kLegacy: streams in order with inline u32 sizes.
      uint32_t offset = 0;
      while (pos_ < end_) {
        uint32_t size = encoding::readUint32(pos_);
        std::string_view streamData(pos_, size);
        pos_ += size;
        if (!streamData.empty()) {
          callback(offset, streamData);
        }
        ++offset;
      }
    }

    NIMBLE_CHECK(
        pos_ == end_,
        "Unexpected trailing data: pos={} end={}",
        reinterpret_cast<uintptr_t>(pos_),
        reinterpret_cast<uintptr_t>(end_));
  }

  /// Returns the auto-detected serialization version.
  /// Only valid after initialize() has been called.
  SerializationVersion version() const {
    return version_;
  }

  /// Returns true when encoding stream prefixes store row counts as varints.
  /// Only valid after initialize() has been called.
  bool streamEncodingUsesVarintRowCount() const {
    return streamEncodingUsesVarintRowCount_;
  }

  /// Returns true when Row/FlatMap null streams contain real nulls, forcing
  /// this batch to be deserialized via the per-batch barrier path. Always
  /// false for versions without a flags byte. Only valid after initialize().
  bool requiresNullBarrier() const {
    return requiresNullBarrier_;
  }

  /// Releases owned kTablet stream payload buffers after a decode run consumes
  /// the string_views returned by iterateStreams(). This does not reset the
  /// current initialized blob cursor/header because callers may initialize the
  /// next batch before flushing the previous run.
  void reset() {
    strippedStreamBuffer_.reset();
  }

  /// Returns the row range embedded in the per-slice header for kTablet
  /// chunks (always present on the wire for kTablet). nullopt for all
  /// other serialization versions. Only valid after initialize() has been
  /// called.
  const std::optional<RowRange>& rowRange() const {
    return rowRange_;
  }

 private:
  // Lazily acquires the arena backing stripped tablet stream payloads.
  Buffer& ensureStrippedStreamBuffer();

  const DeserializerOptions& options_;
  velox::memory::MemoryPool* const pool_;

  // Serialization version detected from data. If the data has a version
  // header, this is read from the first byte; otherwise defaults to kLegacy.
  // When options specify a version, the data version is validated against it.
  SerializationVersion version_{SerializationVersion::kLegacy};
  // True when Row/FlatMap null streams contain real nulls (read from the header
  // flags byte). Defaults false for versions without a flags byte.
  bool requiresNullBarrier_{false};
  // Encoding stream row-count format read from the serialization header.
  bool streamEncodingUsesVarintRowCount_{true};
  // True when kTablet streams retain their storage chunk framing.
  bool streamHasChunkHeader_{false};
  // Per-request row range embedded in the kTablet header (post-rowCount,
  // before stream data). nullopt for non-kTablet formats or when the
  // producer did not embed a row range.
  std::optional<RowRange> rowRange_;
  const char* pos_{nullptr};
  const char* end_{nullptr};
  // Owns slow-stripped kTablet stream payloads until the current decode run is
  // materialized. The arena may span several appended batches.
  EncodingBufferPool strippedStreamBufferPool_;
  std::unique_ptr<ScopedEncodingBuffer> strippedStreamBuffer_;
  // Reusable parallel buffers for the per-blob trailer. Refilled by
  // iterateStreams(): streamIds_ holds the ids of non-zero stream slots (sorted
  // ascending) and streamSizes_ their byte sizes. For kTablet,
  // streamOffsets_ additionally holds each slot's body offset; with the
  // kTablet dedup trailer those offsets and sizes are reconstructed from the
  // per-slot size indices and the unique-size table (duplicate slots resolve
  // to a shared offset). The sizes-only formats leave streamOffsets_ empty and
  // derive offsets by prefix-summing sizes. uniqueStreamSizesScratch_ is reused
  // across blobs for the unique stream table: decoded as sizes, then
  // transformed in place into unique-offset prefix sums to keep the hot path
  // alloc-free.
  std::vector<uint32_t> streamIds_;
  std::vector<uint32_t> streamOffsets_;
  std::vector<uint32_t> streamSizes_;
  std::vector<uint32_t> uniqueStreamSizesScratch_;
};

} // namespace facebook::nimble::serde
