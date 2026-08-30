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

#include "velox/dwio/nimble/serializer/StreamDataParser.h"

#include <cstring>
#include <limits>

#include <folly/io/Cursor.h>

#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"

namespace facebook::nimble::serde {

namespace detail {

namespace {

// Sanity cap on the trailing trailer-size u32 at the public reader entries.
// Production trailers are well under this (typical low-MB at worst). Anything
// past this is treated as buffer corruption rather than a legitimate
// allocation request, so a single bit flip in the trailing u32 can't drive
// a multi-GB allocation before the post-decode mismatch check fires.
constexpr uint32_t kMaxTrailerBytes = 1u << 26; // 64 MiB.

// Per-section decoders for the two-array sparse trailer layout. Each consumes
// the encoding-specific payload (no encoding-type byte) and advances `payload`
// past the bytes it read. The caller is responsible for reading the
// encoding-type byte and dispatching to the right decoder.
//
// `remainingBytes` is the upper bound on payload bytes the decoder may
// consume (i.e. `trailerEnd - payload` at entry). It is used only for a
// single pre-loop sanity check on `count` so a corrupted count varint
// cannot drive a multi-GB `resize`. Loop bodies remain free of
// per-iteration bound checks to keep the hot path branch-free.

void decodeTrivialSection(
    const char*& payload,
    uint32_t remainingBytes,
    std::vector<uint32_t>& values) {
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      count,
      remainingBytes,
      "Trivial section count exceeds remaining trailer bytes");
  values.resize(count);
  if (count > 0) {
    std::memcpy(values.data(), payload, count * sizeof(uint32_t));
    payload += count * sizeof(uint32_t);
  }
}

void decodeVarintSection(
    const char*& payload,
    uint32_t remainingBytes,
    std::vector<uint32_t>& values) {
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      count,
      remainingBytes,
      "Varint section count exceeds remaining trailer bytes");
  values.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = varint::readVarint32(&payload);
  }
}

void decodeDeltaSection(
    const char*& payload,
    uint32_t remainingBytes,
    std::vector<uint32_t>& values) {
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      count,
      remainingBytes,
      "Delta section count exceeds remaining trailer bytes");
  values.resize(count);
  if (count > 0) {
    values[0] = varint::readVarint32(&payload);
    for (uint32_t i = 1; i < count; ++i) {
      const auto delta = varint::readVarint32(&payload);
      values[i] = values[i - 1] + delta;
    }
  }
}

void decodeFixedBitWidthSection(
    const char*& payload,
    uint32_t remainingBytes,
    std::vector<uint32_t>& values) {
  const uint8_t bitWidth = static_cast<uint8_t>(*payload++);
  NIMBLE_CHECK_LE(
      bitWidth, 32, "FixedBitWidth section bitWidth exceeds 32 bits");
  const uint32_t count = varint::readVarint32(&payload);
  // Each value occupies >= 1 bit on the wire, so count <= remainingBytes * 8
  // is a generous sanity ceiling. Cast to uint64_t to avoid overflow when
  // remainingBytes is large.
  NIMBLE_CHECK_LE(
      static_cast<uint64_t>(count),
      static_cast<uint64_t>(remainingBytes) * 8u,
      "FixedBitWidth section count exceeds remaining trailer bit-capacity");
  values.assign(count, 0);
  if (bitWidth > 0 && count > 0) {
    const uint32_t packedBytes =
        static_cast<uint32_t>(FixedBitArray::bufferSize(count, bitWidth));
    FixedBitArray arr{const_cast<char*>(payload), static_cast<int>(bitWidth)};
    arr.bulkGet32(0, count, values.data());
    payload += packedBytes;
  }
}

void decodeSection(
    EncodingType encodingType,
    const char*& payload,
    const char* trailerEnd,
    std::vector<uint32_t>& values) {
  const auto remainingBytes = static_cast<uint32_t>(trailerEnd - payload);
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (getTrailerEncodingType(encodingType)) {
    case EncodingType::Trivial:
      decodeTrivialSection(payload, remainingBytes, values);
      break;
    case EncodingType::Varint:
      decodeVarintSection(payload, remainingBytes, values);
      break;
    case EncodingType::Delta:
      decodeDeltaSection(payload, remainingBytes, values);
      break;
    case EncodingType::FixedBitWidth:
      decodeFixedBitWidthSection(payload, remainingBytes, values);
      break;
    default:
      NIMBLE_FAIL(
          "Unsupported EncodingType for stream sizes trailer section: {}",
          encodingType);
  }
}

// Decodes a two-array sparse trailer into parallel `streamIds` and
// `streamSizes` arrays. Validates that the per-section decoders consume
// exactly the trailer payload and that `streamIds.size() ==
// streamSizes.size()`.
void decodeTrailerStreamMetadata(
    const char* trailerStart,
    uint32_t trailerSize,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamSizes) {
  const auto* trailerEnd = trailerStart + trailerSize;
  const char* payload = trailerStart;
  const auto indicesEncodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*payload++));
  decodeSection(indicesEncodingType, payload, trailerEnd, streamIds);
  const auto sizesEncodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*payload++));
  decodeSection(sizesEncodingType, payload, trailerEnd, streamSizes);
  NIMBLE_CHECK_EQ(
      payload,
      trailerEnd,
      "Trailer size mismatch: read {} bytes, expected {}",
      payload - trailerStart,
      trailerSize);
  NIMBLE_CHECK_EQ(
      streamIds.size(),
      streamSizes.size(),
      "Trailer indices/sizes section count mismatch");
}

// Decodes the kTablet dedup trailer layout (streamIds, sizeIndices,
// uniqueSizes) and reconstructs the parallel `streamIds`, `streamOffsets`,
// and `streamSizes` arrays. The sizeIndices section stores, for each present
// stream slot, the unique body stream it aliases. `streamOffsets` is reused as
// scratch for those unique-stream indices before being overwritten with
// reconstructed body offsets; `uniqueStreamSizes` receives the decoded
// unique-size table and is then transformed in place into exclusive prefix-sum
// offsets.
void decodeTrailerStreamMetadata(
    const char* trailerStart,
    uint32_t trailerSize,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamOffsets,
    std::vector<uint32_t>& streamSizes,
    std::vector<uint32_t>& uniqueStreamSizes) {
  const auto* trailerEnd = trailerStart + trailerSize;
  const char* payload = trailerStart;
  const auto streamIdsEncodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*payload++));
  decodeSection(streamIdsEncodingType, payload, trailerEnd, streamIds);
  const auto sizeIndicesEncodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*payload++));
  // streamOffsets temporarily holds the per-slot unique stream index; it is
  // overwritten with the reconstructed body offset below.
  decodeSection(sizeIndicesEncodingType, payload, trailerEnd, streamOffsets);
  const auto uniqueSizesEncodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*payload++));
  decodeSection(
      uniqueSizesEncodingType, payload, trailerEnd, uniqueStreamSizes);
  NIMBLE_CHECK_EQ(
      payload,
      trailerEnd,
      "Trailer size mismatch: read {} bytes, expected {}",
      payload - trailerStart,
      trailerSize);
  NIMBLE_CHECK_EQ(
      streamIds.size(),
      streamOffsets.size(),
      "Trailer streamIds/sizeIndices section count mismatch");

  const auto numStreams = streamIds.size();
  const auto numUniqueStreams = uniqueStreamSizes.size();
  streamSizes.resize(numStreams);
  // Expand per-slot sizes from the unique-size table.
  for (size_t i = 0; i < numStreams; ++i) {
    const auto uniqueStreamIndex = streamOffsets[i];
    NIMBLE_CHECK_LT(
        uniqueStreamIndex,
        numUniqueStreams,
        "Unique stream index out of range");
    streamSizes[i] = uniqueStreamSizes[uniqueStreamIndex];
  }
  // Transform uniqueStreamSizes in place into exclusive prefix-sum body
  // offsets.
  uint32_t nextBodyOffset{0};
  for (size_t i = 0; i < numUniqueStreams; ++i) {
    const auto sizeValue = uniqueStreamSizes[i];
    uniqueStreamSizes[i] = nextBodyOffset;
    nextBodyOffset += sizeValue;
  }
  // Expand per-slot body offsets from the unique-offset table.
  // streamOffsets[i] currently holds the unique stream index; read it, then
  // overwrite with the resolved offset.
  for (size_t i = 0; i < numStreams; ++i) {
    streamOffsets[i] = uniqueStreamSizes[streamOffsets[i]];
  }
}

} // namespace

void readTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamSizes) {
  const uint32_t trailerSize = readTrailerSize(end);
  NIMBLE_CHECK_LE(
      trailerSize,
      kMaxTrailerBytes,
      "Trailer size sanity cap exceeded (likely buffer corruption)");
  const char* trailerStart = end - sizeof(uint32_t) - trailerSize;
  decodeTrailerStreamMetadata(
      trailerStart, trailerSize, streamIds, streamSizes);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readTrailerStreamMetadata(const char* end) {
  std::vector<uint32_t> streamIds;
  std::vector<uint32_t> streamSizes;
  readTrailerStreamMetadata(end, streamIds, streamSizes);
  return {std::move(streamIds), std::move(streamSizes)};
}

void readTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIds,
    std::vector<uint32_t>& streamOffsets,
    std::vector<uint32_t>& streamSizes,
    std::vector<uint32_t>& uniqueStreamSizesScratch) {
  const uint32_t trailerSize = readTrailerSize(end);
  NIMBLE_CHECK_LE(
      trailerSize,
      kMaxTrailerBytes,
      "Trailer size sanity cap exceeded (likely buffer corruption)");
  const char* trailerStart = end - sizeof(uint32_t) - trailerSize;
  decodeTrailerStreamMetadata(
      trailerStart,
      trailerSize,
      streamIds,
      streamOffsets,
      streamSizes,
      uniqueStreamSizesScratch);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readTrailerStreamMetadata(const folly::IOBuf& input) {
  // Fast path: trailer fits in the tail segment.
  const auto* tail = input.prev();
  if (tail->length() >= sizeof(uint32_t)) {
    const auto* tailEnd =
        reinterpret_cast<const char*>(tail->data()) + tail->length();
    const uint32_t trailerSize = readTrailerSize(tailEnd);
    if (tail->length() >= sizeof(uint32_t) + trailerSize) {
      return readTrailerStreamMetadata(tailEnd);
    }
  }

  // Fallback: trailer spans chain boundary — use cursor + pull().
  const auto totalLength = input.computeChainDataLength();
  folly::io::Cursor trailerCursor(&input);
  trailerCursor.skip(totalLength - sizeof(uint32_t));
  const uint32_t trailerSize = trailerCursor.read<uint32_t>();
  NIMBLE_CHECK_LE(
      trailerSize,
      kMaxTrailerBytes,
      "Trailer size sanity cap exceeded (likely buffer corruption)");
  // Bound trailerSize against the actual IOBuf chain length so a corrupted
  // value cannot drive a huge std::string allocation.
  NIMBLE_CHECK_LE(
      static_cast<uint64_t>(trailerSize) + sizeof(uint32_t),
      totalLength,
      "Trailer size exceeds IOBuf chain length");

  folly::io::Cursor sizesCursor(&input);
  sizesCursor.skip(totalLength - sizeof(uint32_t) - trailerSize);
  std::string trailerBuf(trailerSize, '\0');
  sizesCursor.pull(trailerBuf.data(), trailerSize);
  std::vector<uint32_t> streamIds;
  std::vector<uint32_t> streamSizes;
  decodeTrailerStreamMetadata(
      trailerBuf.data(), trailerSize, streamIds, streamSizes);
  return {std::move(streamIds), std::move(streamSizes)};
}

} // namespace detail

StreamDataParser::StreamDataParser(
    velox::memory::MemoryPool* pool,
    const DeserializerOptions& options)
    : options_{options},
      pool_{pool},
      strippedStreamBufferPool_{pool, /*maxCachedBuffers=*/1} {
  NIMBLE_CHECK_NOT_NULL(pool_);
}

Buffer& StreamDataParser::ensureStrippedStreamBuffer() {
  if (strippedStreamBuffer_ == nullptr) {
    strippedStreamBuffer_ = std::make_unique<ScopedEncodingBuffer>(
        pool_, &strippedStreamBufferPool_);
  }
  return strippedStreamBuffer_->get();
}

uint32_t StreamDataParser::initialize(std::string_view data) {
  pos_ = data.data();
  end_ = data.end();
  auto header = readSerializationHeader(pos_, end_, options_.hasHeader);
  version_ = header.version;
  requiresNullBarrier_ = header.flags.requiresNullBarrier;
  streamEncodingUsesVarintRowCount_ =
      header.flags.streamEncodingUsesVarintRowCount;
  streamHasChunkHeader_ = header.flags.streamHasChunkHeader;
  rowRange_ = header.rowRange;
  return header.rowCount;
}

} // namespace facebook::nimble::serde
