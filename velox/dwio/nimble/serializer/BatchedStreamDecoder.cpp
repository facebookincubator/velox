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

#include "velox/dwio/nimble/serializer/BatchedStreamDecoder.h"

#include <algorithm>
#include <cstring>

#include "folly/Likely.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble {

namespace {

inline uint32_t getTypeStorageWidth(const Type& type) {
  switch (type.kind()) {
    case Kind::Scalar: {
      const auto scalarKind = type.asScalar().scalarDescriptor().scalarKind();
      switch (scalarKind) {
        case ScalarKind::Bool:
        case ScalarKind::Int8:
        case ScalarKind::UInt8:
          return 1;
        case ScalarKind::Int16:
        case ScalarKind::UInt16:
          return 2;
        case ScalarKind::Int32:
        case ScalarKind::Float:
        case ScalarKind::UInt32:
          return 4;
        case ScalarKind::Int64:
        case ScalarKind::UInt64:
        case ScalarKind::Double:
          return 8;
        case ScalarKind::String:
        case ScalarKind::Binary:
        case ScalarKind::Undefined:
          // Variable-length types return 0 to signal special handling path.
          return 0;
      }
      break;
    }
    case Kind::TimestampMicroNano:
      return 10;
    case Kind::Row:
    case Kind::FlatMap:
      return 1;
    case Kind::Array:
    case Kind::ArrayWithOffsets:
    case Kind::Map:
    case Kind::SlidingWindowMap:
      return 4;
  }
  NIMBLE_UNREACHABLE("Unsupported type kind: {}.", toString(type.kind()));
}

// Get the ScalarKind for a type based on its storage format.
inline ScalarKind getScalarKindForType(const Type& type) {
  if (type.isScalar()) {
    return type.asScalar().scalarDescriptor().scalarKind();
  } else if (type.isRow() || type.isFlatMap()) {
    // Row/FlatMap nulls streams are boolean.
    return ScalarKind::Bool;
  } else if (type.isArray() || type.isMap()) {
    // Array/Map lengths streams are uint32_t.
    return ScalarKind::UInt32;
  }
  NIMBLE_UNSUPPORTED("Unsupported type: {}", toString(type.kind()));
}

// Empty scattered reads still need to mark every output row as absent.
inline void markEmptyScatteredOutputNulls(
    const std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  if (scatterOutputBitmap == nullptr) {
    return;
  }
  NIMBLE_CHECK_EQ(
      velox::bits::countBits(
          static_cast<const uint64_t*>(scatterOutputBitmap->bits()),
          0,
          scatterOutputBitmap->size()),
      0,
      "Empty scattered reads require an empty scatterOutputBitmap");
  NIMBLE_CHECK_NOT_NULL(
      getOutputNulls, "Scattered reads require output nulls callback");
  velox::bits::fillBits(
      static_cast<uint64_t*>(getOutputNulls()),
      0,
      scatterOutputBitmap->size(),
      velox::bits::kNull);
}

} // namespace

BatchedStreamDecoder::BatchedStreamDecoder(
    const Type* type,
    bool isInMapStream,
    size_t bufferPoolCapacity,
    velox::memory::MemoryPool* pool)
    : type_{type},
      pool_{pool},
      isInMapStream_{isInMapStream},
      scalarKind_{getScalarKindForType(*type)},
      typeStorageWidth_{getTypeStorageWidth(*type)},
      bufferPool_{
          bufferPoolCapacity > 0
              ? std::make_unique<velox::BufferPool>(bufferPoolCapacity)
              : nullptr} {
  NIMBLE_CHECK(
      !isInMapStream_ || typeStorageWidth_ == sizeof(bool),
      "FlatMap in-map stream should be bool");
}

uint32_t BatchedStreamDecoder::next(
    uint32_t count,
    void* output,
    std::vector<velox::BufferPtr>& stringBuffers,
    std::function<void*()> getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  NIMBLE_CHECK(
      scatterOutputBitmap == nullptr || !isInMapStream(),
      "scatterOutputBitmap not used for FlatMap in-map streams");

  if (count == 0) {
    markEmptyScatteredOutputNulls(getOutputNulls, scatterOutputBitmap);
    return 0;
  }

  uint32_t nonNullCount;
  if (scatterOutputBitmap != nullptr) {
    nonNullCount = scatteredRead(
        count, output, getOutputNulls, scatterOutputBitmap, stringBuffers);
  } else if (isInMapStream()) {
    nonNullCount = inMapRead(count, output, stringBuffers);
  } else {
    nonNullCount = denseRead(count, output, getOutputNulls, stringBuffers);
  }
  currentRow_ += count;
  return nonNullCount;
}

void BatchedStreamDecoder::skip(uint32_t count) {
  if (count == 0) {
    return;
  }

  // For non-in-map streams, an empty `streamSegments_` is only valid
  // for Row/FlatMap null streams that the writer omitted (all-non-null).
  // Nothing decoded → nothing to advance; just bump the cursor.
  if (FOLLY_UNLIKELY(!isInMapStream() && streamSegments_.empty())) {
    NIMBLE_CHECK(
        type_->isRow() || type_->isFlatMap(),
        "Empty streamSegments_ only valid for Row/FlatMap null streams");
    currentRow_ += count;
    return;
  }

  // `skipStringBuffers_` is a persistent per-decoder buffer given to
  // `ensureStreamData`. It must outlive the encoding created by skip,
  // which may hold `string_view`s into it and be re-used by a
  // subsequent `next()`.
  if (isInMapStream()) {
    skipInMap(count, skipStringBuffers_);
  } else {
    skipEncoded(count, skipStringBuffers_);
  }
  currentRow_ += count;
}

void BatchedStreamDecoder::reset() {
  clear();
}

void BatchedStreamDecoder::clear() {
  streamSegments_.clear();
  presentInMapSegments_.clear();
  // Reset streamData_ (and hence its encoding_) BEFORE dropping
  // skipStringBuffers_ — the encoding may hold string_views into buffers
  // stored there, and those views must not outlive the buffers.
  streamData_.reset();
  skipStringBuffers_.clear();
  streamSegmentIndex_ = 0;
  presentSegmentIndex_ = 0;
  currentRow_ = 0;
}

const Encoding* BatchedStreamDecoder::encoding() const {
  NIMBLE_UNREACHABLE("unexpected call");
}

serde::StreamData& BatchedStreamDecoder::ensureStreamData(
    std::vector<velox::BufferPtr>& stringBuffers) {
  if (streamData_.has_value()) {
    if (!skipStringBuffers_.empty() && &stringBuffers != &skipStringBuffers_) {
      stringBuffers.insert(
          stringBuffers.end(),
          std::make_move_iterator(skipStringBuffers_.begin()),
          std::make_move_iterator(skipStringBuffers_.end()));
      skipStringBuffers_.clear();
    }
    streamData_->setStringBuffers(&stringBuffers);
    return *streamData_;
  }

  NIMBLE_CHECK_LT(streamSegmentIndex_, streamSegments_.size());
  const auto& segment = streamSegments_[streamSegmentIndex_];
  streamData_.emplace(
      scalarKind_,
      segment.data,
      stringBuffers,
      pool_,
      serde::StreamData::Options{
          .version = segment.version,
          .streamEncodingUsesVarintRowCount =
              segment.streamEncodingUsesVarintRowCount,
          .bufferPool = bufferPool_.get(),
          .decompressionBuffer = &decompressionBuffer_});
  return *streamData_;
}

void BatchedStreamDecoder::advanceSegment() {
  streamData_.reset();
  ++streamSegmentIndex_;
}

uint32_t BatchedStreamDecoder::fillInMapGap(
    uint32_t rowOffset,
    uint32_t rowCount,
    uint32_t outputOffset,
    void* output) {
  NIMBLE_CHECK(isInMapStream(), "Expected FlatMap in-map stream");
  const auto requestEndRow = rowOffset + rowCount;
  const auto gapEndRow = streamSegmentIndex_ < streamSegments_.size()
      ? std::min(requestEndRow, streamSegments_[streamSegmentIndex_].startRow)
      : requestEndRow;
  NIMBLE_CHECK_GT(
      gapEndRow,
      rowOffset,
      "FlatMap in-map gap fill requires a non-empty output range");
  const auto numGapRows = gapEndRow - rowOffset;
  auto* const outputBools =
      static_cast<char*>(output) + outputOffset * typeStorageWidth_;
  constexpr char kInMapAbsent = 0;
  constexpr char kInMapPresent = 1;
  std::memset(outputBools, kInMapAbsent, numGapRows * typeStorageWidth_);
  while (presentSegmentIndex_ < presentInMapSegments_.size()) {
    auto& segment = presentInMapSegments_[presentSegmentIndex_];
    if (segment.startRow >= gapEndRow) {
      break;
    }
    // Clamp both ends. A segment can start before `rowOffset` if a
    // prior skip landed inside it, and can end past `gapEndRow` when
    // the current read only covers a prefix (or when the segment is
    // the null-barrier sentinel `kPresentInMapEndRow`).
    const auto presentStartRow = std::max(segment.startRow, rowOffset);
    const auto presentEndRow = std::min(segment.endRow, gapEndRow);
    std::memset(
        outputBools + (presentStartRow - rowOffset) * typeStorageWidth_,
        kInMapPresent,
        (presentEndRow - presentStartRow) * typeStorageWidth_);
    if (segment.endRow > gapEndRow) {
      // Not fully consumed — advance its start so the next call resumes
      // where this one left off. `kPresentInMapEndRow` sentinel stays
      // in `endRow`.
      segment.startRow = gapEndRow;
      break;
    }
    ++presentSegmentIndex_;
  }
  return numGapRows;
}

serde::StreamData::DecodeResult BatchedStreamDecoder::readLegacyStreamSegment(
    serde::StreamData& streamData,
    void* output,
    uint32_t offset,
    uint32_t count) {
  const auto width = typeStorageWidth_;
  if (width > 0) {
    return streamData.decodeLegacy(output, offset, count, width);
  }

  auto* dest = static_cast<std::string_view*>(output) + offset;
  return streamData.decodeStrings(count, dest);
}

serde::StreamData::DecodeResult BatchedStreamDecoder::readSegment(
    void* output,
    uint32_t offset,
    uint32_t count,
    const std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap,
    std::vector<velox::BufferPtr>& stringBuffers) {
  NIMBLE_CHECK(
      scatterOutputBitmap == nullptr || !isInMapStream(),
      "scatterOutputBitmap not used for FlatMap in-map streams");

  NIMBLE_CHECK_LT(streamSegmentIndex_, streamSegments_.size());
  auto& streamData = ensureStreamData(stringBuffers);
  if (!streamData.hasEncoding()) {
    NIMBLE_CHECK_NULL(
        scatterOutputBitmap,
        "scatterOutputBitmap is only used for encoded streams");
    return readLegacyStreamSegment(streamData, output, offset, count);
  }

  const auto width = typeStorageWidth_;
  return streamData.decode(
      output, offset, count, width, getOutputNulls, scatterOutputBitmap);
}

uint32_t BatchedStreamDecoder::skipSegment(
    uint32_t numRows,
    std::vector<velox::BufferPtr>& stringBuffers) {
  NIMBLE_CHECK_LT(
      streamSegmentIndex_,
      streamSegments_.size(),
      "BatchedStreamDecoder::skip past end of decoder queue");
  auto& streamData = ensureStreamData(stringBuffers);
  NIMBLE_CHECK(
      streamData.hasEncoding(),
      "BatchedStreamDecoder::skip requires encoded segments");
  const auto remainingRows = streamData.remainingRows();
  NIMBLE_CHECK_GT(remainingRows, 0, "Current segment has no rows");
  const auto toSkip = std::min<uint32_t>(remainingRows, numRows);
  streamData.skip(toSkip);
  if (toSkip == remainingRows) {
    advanceSegment();
  }
  return toSkip;
}

void BatchedStreamDecoder::skipEncoded(
    uint32_t numRows,
    std::vector<velox::BufferPtr>& stringBuffers) {
  uint32_t skippedRows = 0;
  while (skippedRows < numRows) {
    skippedRows += skipSegment(numRows - skippedRows, stringBuffers);
  }
}

void BatchedStreamDecoder::skipInMap(
    uint32_t numRows,
    std::vector<velox::BufferPtr>& stringBuffers) {
  const uint32_t targetRow = currentRow_ + numRows;
  uint32_t skippedRows = 0;
  while (skippedRows < numRows) {
    const uint32_t currentRow = currentRow_ + skippedRows;
    if (streamSegmentIndex_ >= streamSegments_.size()) {
      // No more encoded segments — everything left is presence-gap.
      advanceInMapPresentSegmentIndex(targetRow);
      break;
    }
    const auto nextStreamStartRow =
        streamSegments_[streamSegmentIndex_].startRow;
    if (nextStreamStartRow > currentRow) {
      // Presence gap up to the next encoded segment (or the skip
      // target, whichever comes first).
      const uint32_t gapEndRow = std::min(targetRow, nextStreamStartRow);
      advanceInMapPresentSegmentIndex(gapEndRow);
      skippedRows += gapEndRow - currentRow;
      continue;
    }
    skippedRows += skipSegment(numRows - skippedRows, stringBuffers);
  }
}

void BatchedStreamDecoder::advanceInMapPresentSegmentIndex(uint32_t targetRow) {
  NIMBLE_CHECK(isInMapStream(), "Expected FlatMap in-map stream");
  while (presentSegmentIndex_ < presentInMapSegments_.size()) {
    const auto& segment = presentInMapSegments_[presentSegmentIndex_];
    if (segment.endRow > targetRow) {
      break;
    }
    ++presentSegmentIndex_;
  }
}

uint32_t BatchedStreamDecoder::denseRead(
    uint32_t count,
    void* output,
    const std::function<void*()>& getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers) {
  const auto width = typeStorageWidth_;
  if (FOLLY_UNLIKELY(streamSegments_.empty())) {
    NIMBLE_CHECK(
        type_->isRow() || type_->isFlatMap(),
        "streamSegments_ is empty for unexpected stream type={}",
        type_->kind());
    NIMBLE_CHECK_EQ(
        width, sizeof(bool), "Row/FlatMap null stream should be bool");
    // All-non-null Row/FlatMap null streams are omitted on the wire and
    // reconstructed as all-true here (no null rows).
    std::fill_n(static_cast<bool*>(output), count, true);
    return count;
  }

  uint32_t rowsRead{0};
  uint32_t nonNullCount{0};
  bool nullsInitialized{false};
  while (rowsRead < count) {
    NIMBLE_CHECK_LT(
        streamSegmentIndex_,
        streamSegments_.size(),
        "Non-in-map stream ended before requested rows were decoded");
    const uint32_t rowsToRead = count - rowsRead;
    const auto result = readSegment(
        output,
        rowsRead,
        rowsToRead,
        getOutputNulls,
        /*scatterOutputBitmap=*/nullptr,
        stringBuffers);
    NIMBLE_CHECK_GT(
        result.numOutputRows, 0, "Current segment returned no rows");
    NIMBLE_CHECK_LE(
        result.nonNullOutputRows,
        result.numOutputRows,
        "non-null row count exceeds row count");
    const bool segmentAllNonNull =
        result.nonNullOutputRows == result.numOutputRows;
    const bool needsNullHandling = !segmentAllNonNull || nullsInitialized;
    if (FOLLY_UNLIKELY(needsNullHandling)) {
      NIMBLE_CHECK_NOT_NULL(
          getOutputNulls, "nullable segment requires output nulls callback");
      if (!segmentAllNonNull && !nullsInitialized) {
        velox::bits::fillBits(
            static_cast<uint64_t*>(getOutputNulls()),
            0,
            rowsRead,
            velox::bits::kNotNull);
        nullsInitialized = true;
      } else if (segmentAllNonNull && nullsInitialized) {
        // Nullable decoding does not touch the null bitmap for all-non-null
        // segments, so keep the stitched output range explicitly non-null.
        velox::bits::fillBits(
            static_cast<uint64_t*>(getOutputNulls()),
            rowsRead,
            rowsRead + result.numOutputRows,
            velox::bits::kNotNull);
      }
    }
    rowsRead += result.numOutputRows;
    nonNullCount += result.nonNullOutputRows;
    if (FOLLY_LIKELY(result.segmentExhausted)) {
      advanceSegment();
    }
  }

  NIMBLE_CHECK_EQ(
      rowsRead,
      count,
      "Incomplete read: typeKind={} inMap={} segments={} streamSegmentIndex={}",
      toString(type_->kind()),
      isInMapStream_,
      streamSegments_.size(),
      streamSegmentIndex_);
  return nonNullCount;
}

uint32_t BatchedStreamDecoder::inMapRead(
    uint32_t count,
    void* output,
    std::vector<velox::BufferPtr>& stringBuffers) {
  uint32_t rowsRead{0};
  uint32_t nonNullCount{0};
  while (rowsRead < count) {
    const uint32_t currentRow = currentRow_ + rowsRead;
    if (streamSegmentIndex_ >= streamSegments_.size()) {
      const auto rows =
          fillInMapGap(currentRow, count - rowsRead, rowsRead, output);
      rowsRead += rows;
      nonNullCount += rows;
      break;
    }

    const auto nextStreamStartRow =
        streamSegments_[streamSegmentIndex_].startRow;
    if (nextStreamStartRow > currentRow) {
      const auto rows =
          fillInMapGap(currentRow, count - rowsRead, rowsRead, output);
      NIMBLE_CHECK_EQ(
          rows,
          std::min(count + currentRow_, nextStreamStartRow) - currentRow,
          "FlatMap in-map gap fill returned unexpected row count");
      rowsRead += rows;
      nonNullCount += rows;
      continue;
    }

    const uint32_t rowsToRead = count - rowsRead;
    const auto result = readSegment(
        output,
        rowsRead,
        rowsToRead,
        /*getOutputNulls=*/nullptr,
        /*scatterOutputBitmap=*/nullptr,
        stringBuffers);
    NIMBLE_CHECK_GT(
        result.numOutputRows, 0, "Current in-map segment returned no rows");
    NIMBLE_CHECK_EQ(
        result.nonNullOutputRows,
        result.numOutputRows,
        "FlatMap in-map stream must not contain nulls");
    rowsRead += result.numOutputRows;
    nonNullCount += result.numOutputRows;
    if (FOLLY_LIKELY(result.segmentExhausted)) {
      advanceSegment();
    }
  }

  NIMBLE_CHECK_EQ(
      rowsRead,
      count,
      "Incomplete in-map read: segments={} streamSegmentIndex={}",
      streamSegments_.size(),
      streamSegmentIndex_);
  return nonNullCount;
}

uint32_t BatchedStreamDecoder::scatteredRead(
    uint32_t count,
    void* output,
    const std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap,
    std::vector<velox::BufferPtr>& stringBuffers) {
  NIMBLE_CHECK(
      !type_->isFlatMap(),
      "scatterOutputBitmap not used for FlatMap null streams");

  const auto outputSize = scatterOutputBitmap->size();
  // Fast path: if bitmap is dense (all bits set), read directly to output.
  // This avoids temp buffer allocation and scatter overhead.
  if (count == outputSize) {
    return denseRead(count, output, getOutputNulls, stringBuffers);
  }

  uint32_t rowsRead = 0;
  uint32_t nonNullCount = 0;

  NIMBLE_CHECK_NOT_NULL(
      getOutputNulls, "Output nulls callback is required for scattered reads");
  uint32_t offset = 0;
  bool hasNulls = false;

  while (rowsRead < count && streamSegmentIndex_ < streamSegments_.size()) {
    auto& streamData = ensureStreamData(stringBuffers);
    NIMBLE_CHECK(
        streamData.hasEncoding(),
        "Scattered reads require encoded stream data");
    const auto requestRows = count - rowsRead;
    const auto rowsToRead = std::min(requestRows, streamData.remainingRows());
    NIMBLE_CHECK_GT(rowsToRead, 0, "Current scattered segment has no rows");

    const auto endOffset = velox::bits::findSetBit(
        static_cast<const char*>(scatterOutputBitmap->bits()),
        offset,
        outputSize,
        rowsToRead + 1);
    velox::bits::Bitmap segmentScatterBitmap{
        scatterOutputBitmap->bits(), endOffset};
    const auto result = readSegment(
        output,
        offset,
        rowsToRead,
        getOutputNulls,
        &segmentScatterBitmap,
        stringBuffers);
    NIMBLE_CHECK_EQ(
        result.numOutputRows, rowsToRead, "Incomplete scattered segment read");

    const auto segmentRows = endOffset - offset;
    const bool segmentHasNulls = result.nonNullOutputRows != segmentRows;
    if (segmentHasNulls && !hasNulls) {
      velox::bits::BitmapBuilder nullBits{getOutputNulls(), offset};
      nullBits.set(0, offset);
    }
    if (hasNulls && !segmentHasNulls) {
      velox::bits::BitmapBuilder nullBits{getOutputNulls(), endOffset};
      nullBits.set(offset, endOffset);
    }
    hasNulls |= segmentHasNulls;

    rowsRead += result.numOutputRows;
    nonNullCount += result.nonNullOutputRows;
    offset = endOffset;
    if (FOLLY_LIKELY(result.segmentExhausted)) {
      advanceSegment();
    }
  }

  NIMBLE_CHECK_EQ(
      rowsRead,
      count,
      "Incomplete scattered read: typeKind={} segments={} streamSegmentIndex={}",
      toString(type_->kind()),
      streamSegments_.size(),
      streamSegmentIndex_);
  return nonNullCount;
}

} // namespace facebook::nimble
