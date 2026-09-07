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

#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"

#include <cstring>
#include <string>

#include <folly/io/Cursor.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/Options.h"

namespace facebook::nimble::serde::legacy {

namespace {

// Defensive sanity ceiling on the encoded trailer-size suffix. Production
// stripe trailers are well under this (typical low-MB at worst). Anything
// past this is treated as buffer corruption rather than a legitimate
// allocation request.
constexpr uint32_t kMaxLegacyTrailerBytes = 1u << 26; // 64 MiB.

// Reads the trailing u32 trailer-size suffix from the last 4 bytes of the
// buffer.
inline uint32_t readTrailerSize(const char* end) {
  const char* pos = end - sizeof(uint32_t);
  return encoding::readUint32(pos);
}

void decodeTrivialTrailer(
    const char*& payload,
    uint32_t payloadSize,
    std::vector<uint32_t>& sizes) {
  const uint32_t count = payloadSize / sizeof(uint32_t);
  sizes.resize(count);
  if (count > 0) {
    std::memcpy(sizes.data(), payload, count * sizeof(uint32_t));
    payload += count * sizeof(uint32_t);
  }
}

// Bound: each varint is at least one byte, so `count <= payloadSize` is a
// sufficient sanity ceiling against trailer corruption.
void decodeVarintTrailer(
    const char*& payload,
    uint32_t payloadSize,
    std::vector<uint32_t>& sizes) {
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      count, payloadSize, "Legacy Varint trailer count exceeds payload size");
  sizes.resize(count);
  // NOLINTBEGIN(facebook-hte-ParameterUncheckedArrayBounds)
  for (uint32_t i = 0; i < count; ++i) {
    sizes[i] = varint::readVarint32(&payload);
  }
  // NOLINTEND(facebook-hte-ParameterUncheckedArrayBounds)
}

// Bound: each varint is at least one byte, so `count <= payloadSize` is a
// sufficient sanity ceiling against trailer corruption.
void decodeDeltaTrailer(
    const char*& payload,
    uint32_t payloadSize,
    std::vector<uint32_t>& sizes) {
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      count, payloadSize, "Legacy Delta trailer count exceeds payload size");
  sizes.resize(count);
  if (count > 0) {
    // NOLINTBEGIN(facebook-hte-ParameterUncheckedArrayBounds)
    sizes[0] = varint::readVarint32(&payload);
    for (uint32_t i = 1; i < count; ++i) {
      const auto delta = varint::readVarint32(&payload);
      sizes[i] = sizes[i - 1] + delta;
    }
    // NOLINTEND(facebook-hte-ParameterUncheckedArrayBounds)
  }
}

// Frozen snapshot of master's `decodeFixedBitWidthTrailer`.
// Bound: total packed bits `count * bitWidth` must fit in `payloadSize * 8`.
// When `bitWidth == 0` (all stream sizes are 0) the packed array is empty,
// so any `count` is valid and the check trivially passes (0 <= anything).
void decodeFixedBitWidthTrailer(
    const char*& payload,
    uint32_t payloadSize,
    std::vector<uint32_t>& sizes) {
  const uint8_t bitWidth = static_cast<uint8_t>(*payload++);
  NIMBLE_CHECK_LE(
      bitWidth, 32, "Legacy FixedBitWidth trailer bitWidth exceeds 32 bits");
  const uint32_t count = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      static_cast<uint64_t>(count) * bitWidth,
      static_cast<uint64_t>(payloadSize) * 8u,
      "Legacy FixedBitWidth trailer count*bitWidth exceeds payload bit-capacity");
  sizes.assign(count, 0);
  if (bitWidth > 0 && count > 0) {
    const uint32_t packedBytes =
        static_cast<uint32_t>(FixedBitArray::bufferSize(count, bitWidth));
    FixedBitArray arr{const_cast<char*>(payload), static_cast<int>(bitWidth)};
    arr.bulkGet32(0, count, sizes.data());
    payload += packedBytes;
  }
}

// Sparse-native MainlyConstant decoder: fills parallel (indices, sizes)
// arrays for only the non-zero stream slots.
// Bound: `nonZeroCount` must be <= `streamCount` (existing master invariant)
// AND <= `payloadSize` (sanity against trailer corruption that would force
// huge allocations even with a consistent streamCount).
void decodeMainlyConstantTrailerSparse(
    const char*& payload,
    uint32_t payloadSize,
    std::vector<uint32_t>& indices,
    std::vector<uint32_t>& sizes) {
  const uint32_t streamCount = varint::readVarint32(&payload);
  const uint32_t nonZeroCount = varint::readVarint32(&payload);
  NIMBLE_CHECK_LE(
      nonZeroCount,
      streamCount,
      "MainlyConstant nonZeroCount exceeds streamCount");
  NIMBLE_CHECK_LE(
      nonZeroCount,
      payloadSize,
      "MainlyConstant nonZeroCount exceeds payload size");
  indices.resize(nonZeroCount);
  sizes.resize(nonZeroCount);
  if (nonZeroCount == 0) {
    return;
  }

  const uint8_t idxBitWidth = static_cast<uint8_t>(*payload++);
  NIMBLE_CHECK_LE(
      idxBitWidth, 32, "MainlyConstant idxBitWidth exceeds 32 bits");
  const uint32_t packedIndicesBytes = static_cast<uint32_t>(
      FixedBitArray::bufferSize(nonZeroCount, idxBitWidth));
  FixedBitArray idxArr{
      const_cast<char*>(payload), static_cast<int>(idxBitWidth)};
  idxArr.bulkGet32(0, nonZeroCount, indices.data());
  payload += packedIndicesBytes;

  // Validate decoded indices against streamCount. Catches trailer corruption.
  for (uint32_t i = 0; i < nonZeroCount; ++i) {
    NIMBLE_CHECK_LT(
        indices[i], streamCount, "MainlyConstant trailer index out of range");
  }

  const uint8_t valBitWidth = static_cast<uint8_t>(*payload++);
  NIMBLE_CHECK_LE(
      valBitWidth, 32, "MainlyConstant valBitWidth exceeds 32 bits");
  const uint32_t packedValuesBytes = static_cast<uint32_t>(
      FixedBitArray::bufferSize(nonZeroCount, valBitWidth));
  FixedBitArray valArr{
      const_cast<char*>(payload), static_cast<int>(valBitWidth)};
  valArr.bulkGet32(0, nonZeroCount, sizes.data());
  payload += packedValuesBytes;
}

// Decodes the dense path (everything except MainlyConstant) into a dense
// vector, then the public API scatters it into sparse (streamIndices,
// streamSizes) pairs.
void decodeDenseTrailer(
    EncodingType encodingType,
    const char*& payload,
    uint32_t payloadSize,
    const char* trailerEnd,
    std::vector<uint32_t>& denseSizes) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (encodingType) {
    case EncodingType::Trivial:
      decodeTrivialTrailer(payload, payloadSize, denseSizes);
      break;
    case EncodingType::Varint:
      decodeVarintTrailer(payload, payloadSize, denseSizes);
      break;
    case EncodingType::Delta:
      decodeDeltaTrailer(payload, payloadSize, denseSizes);
      break;
    case EncodingType::FixedBitWidth:
      decodeFixedBitWidthTrailer(payload, payloadSize, denseSizes);
      break;
    default:
      NIMBLE_FAIL(
          "Unsupported EncodingType for legacy trailer dense path: {}",
          encodingType);
  }
  NIMBLE_CHECK_EQ(
      payload,
      trailerEnd,
      "Legacy trailer size mismatch (dense): read {} bytes, expected {}",
      payload - (trailerEnd - payloadSize - sizeof(uint8_t)),
      payloadSize + sizeof(uint8_t));
}

// Walks `denseSizes` once and emits `(i, denseSizes[i])` for every non-zero
// entry into the output sparse pair. Used to normalize the dense decoder
// output into the same shape MainlyConstant naturally produces.
void scatterDenseToSparse(
    const std::vector<uint32_t>& denseSizes,
    std::vector<uint32_t>& streamIndices,
    std::vector<uint32_t>& streamSizes) {
  streamIndices.clear();
  streamSizes.clear();
  streamIndices.reserve(denseSizes.size());
  streamSizes.reserve(denseSizes.size());
  for (uint32_t i = 0; i < denseSizes.size(); ++i) {
    if (denseSizes[i] != 0) {
      streamIndices.emplace_back(i);
      streamSizes.emplace_back(denseSizes[i]);
    }
  }
}

// Decodes the trailer bytes between [trailerStart, trailerStart+trailerSize)
// into normalized sparse (streamIndices, streamSizes) pairs regardless of the
// underlying encoding type.
void decodeTrailerToSparse(
    const char* trailerStart,
    uint32_t trailerSize,
    std::vector<uint32_t>& streamIndices,
    std::vector<uint32_t>& streamSizes) {
  NIMBLE_CHECK_GT(
      trailerSize, 0, "Legacy trailer must contain at least an encoding byte");
  const auto* trailerEnd = trailerStart + trailerSize;
  const auto encodingType =
      static_cast<EncodingType>(static_cast<uint8_t>(*trailerStart));
  const char* payload = trailerStart + sizeof(uint8_t);
  const uint32_t payloadSize = trailerSize - sizeof(uint8_t);

  if (encodingType == EncodingType::MainlyConstant) {
    decodeMainlyConstantTrailerSparse(
        payload, payloadSize, streamIndices, streamSizes);
    NIMBLE_CHECK_EQ(
        payload,
        trailerEnd,
        "Legacy trailer size mismatch (MainlyConstant): read {} bytes, expected {}",
        payload - trailerStart,
        trailerSize);
    return;
  }

  // Dense path: decode into a temp dense vector then scatter to sparse.
  std::vector<uint32_t> denseSizes;
  decodeDenseTrailer(
      encodingType, payload, payloadSize, trailerEnd, denseSizes);
  scatterDenseToSparse(denseSizes, streamIndices, streamSizes);
}

} // namespace

void readLegacyTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIndices,
    std::vector<uint32_t>& streamSizes) {
  const uint32_t trailerSize = readTrailerSize(end);
  NIMBLE_CHECK_LE(
      trailerSize,
      kMaxLegacyTrailerBytes,
      "Legacy trailer size sanity cap exceeded (likely buffer corruption): {} > {}",
      trailerSize,
      kMaxLegacyTrailerBytes);
  const char* trailerStart = end - sizeof(uint32_t) - trailerSize;
  decodeTrailerToSparse(trailerStart, trailerSize, streamIndices, streamSizes);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readLegacyTrailerStreamMetadata(const char* end) {
  std::vector<uint32_t> streamIndices;
  std::vector<uint32_t> streamSizes;
  readLegacyTrailerStreamMetadata(end, streamIndices, streamSizes);
  return {std::move(streamIndices), std::move(streamSizes)};
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readLegacyTrailerStreamMetadata(const folly::IOBuf& input) {
  // Fast path: trailer fits in the tail segment.
  const auto* tail = input.prev();
  if (tail->length() >= sizeof(uint32_t)) {
    const auto* tailEnd =
        reinterpret_cast<const char*>(tail->data()) + tail->length();
    const uint32_t trailerSize = readTrailerSize(tailEnd);
    if (tail->length() >= sizeof(uint32_t) + trailerSize) {
      // Tail fast path: defer the kMaxLegacyTrailerBytes bound check to the
      // raw-pointer overload below.
      return readLegacyTrailerStreamMetadata(tailEnd);
    }
  }

  // Fallback: trailer spans chain boundary — use cursor + pull().
  const auto totalLength = input.computeChainDataLength();
  folly::io::Cursor trailerCursor(&input);
  trailerCursor.skip(totalLength - sizeof(uint32_t));
  const uint32_t trailerSize = trailerCursor.read<uint32_t>();
  NIMBLE_CHECK_LE(
      trailerSize,
      kMaxLegacyTrailerBytes,
      "Legacy trailer size sanity cap exceeded (likely buffer corruption): {} > {}",
      trailerSize,
      kMaxLegacyTrailerBytes);
  // Bound trailerSize against the actual IOBuf chain length so a corrupted
  // value cannot drive a huge std::string allocation.
  NIMBLE_CHECK_LE(
      static_cast<uint64_t>(trailerSize) + sizeof(uint32_t),
      totalLength,
      "Legacy trailer size exceeds IOBuf chain length: trailerSize={}, chain={}",
      trailerSize,
      totalLength);

  folly::io::Cursor sizesCursor(&input);
  sizesCursor.skip(totalLength - sizeof(uint32_t) - trailerSize);
  std::string trailerBuf(trailerSize, '\0');
  sizesCursor.pull(trailerBuf.data(), trailerSize);
  std::vector<uint32_t> streamIndices;
  std::vector<uint32_t> streamSizes;
  decodeTrailerToSparse(
      trailerBuf.data(), trailerSize, streamIndices, streamSizes);
  return {std::move(streamIndices), std::move(streamSizes)};
}

namespace {

std::vector<uint32_t> scatterSparseToDense(
    const std::vector<uint32_t>& indices,
    const std::vector<uint32_t>& sizes) {
  std::vector<uint32_t> dense;
  if (!indices.empty()) {
    dense.assign(indices.back() + 1, 0);
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    for (size_t i = 0; i < indices.size(); ++i) {
      dense[indices[i]] = sizes[i];
    }
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
  }
  return dense;
}

} // namespace

std::vector<uint32_t> readLegacyTrailerStreamSizesDense(const char* end) {
  auto [indices, sizes] = readLegacyTrailerStreamMetadata(end);
  return scatterSparseToDense(indices, sizes);
}

std::vector<uint32_t> readLegacyTrailerStreamSizesDense(
    const folly::IOBuf& input) {
  auto [indices, sizes] = readLegacyTrailerStreamMetadata(input);
  return scatterSparseToDense(indices, sizes);
}

} // namespace facebook::nimble::serde::legacy
