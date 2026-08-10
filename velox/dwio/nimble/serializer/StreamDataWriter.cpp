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

#include "velox/dwio/nimble/serializer/StreamDataWriter.h"

#include <limits>

#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"

#include "velox/dwio/nimble/serializer/SerializationHeader.h"

#include <lz4.h>
#include <lz4hc.h>
#include <zstd.h>
#include <zstd_errors.h>

#include <folly/io/Cursor.h>

namespace facebook::nimble::serde::detail {

namespace {

// Upper-bound payload-size estimators (one per trailer encoding). Each
// returns the payload bytes between the encoding-type byte and the trailing
// trailer_size u32. estimateTrailerSize() adds the encoding-type byte and
// the trailer_size suffix.

// Compresses 'input' into 'compPos' (which already has the compression-type
// byte written). Returns the number of bytes written if compression was
// beneficial; std::nullopt if the caller should fall back to uncompressed.

std::optional<uint32_t> compressZstd(
    const SerializerOptions& options,
    std::string_view input,
    char* compPos,
    uint32_t size) {
  const auto ret = ZSTD_compress(
      compPos, size, input.data(), size, options.compressionLevel);
  if (ZSTD_isError(ret)) {
    NIMBLE_CHECK_EQ(
        static_cast<int>(ZSTD_getErrorCode(ret)),
        static_cast<int>(ZSTD_error_dstSize_tooSmall),
        "zstd error");
    return std::nullopt;
  }
  return static_cast<uint32_t>(ret);
}

std::optional<uint32_t> compressLz4(
    const SerializerOptions& options,
    std::string_view input,
    char* compPos,
    uint32_t size) {
  // LZ4 block mode is not self-descriptive (unlike ZSTD frames), so we
  // prepend the uncompressed size as a uint32 before the compressed data.
  // Wire format: [size:u32][compType:i8=3][origSize:u32][lz4_data...]
  const auto origSize = static_cast<uint32_t>(input.size());
  encoding::writeUint32(origSize, compPos);
  constexpr int kMinHcLevel = LZ4HC_CLEVEL_MIN;
  const auto compressedSize = options.compressionLevel >= kMinHcLevel
      ? LZ4_compress_HC(
            input.data(),
            compPos,
            static_cast<int>(input.size()),
            static_cast<int>(size),
            options.compressionLevel)
      : LZ4_compress_default(
            input.data(),
            compPos,
            static_cast<int>(input.size()),
            static_cast<int>(size));
  if (compressedSize == 0 ||
      static_cast<uint32_t>(compressedSize) >= origSize) {
    return std::nullopt;
  }
  return sizeof(uint32_t) + static_cast<uint32_t>(compressedSize);
}

// Upper-bound estimators for the encoded payload of one section. `count` is the
// worst-case number of values to emit; `maxValue` is the largest value that
// can appear (used by Delta/FixedBitWidth; Trivial/Varint ignore it).
size_t estimateTrivialSectionSize(size_t count) {
  // count varint (max 5 bytes) + N u32s.
  return 5 + count * sizeof(uint32_t);
}

size_t estimateVarintSectionSize(size_t count) {
  // count varint + N varints (max 5 bytes each).
  return 5 + count * 5;
}

size_t estimateDeltaSectionSize(size_t count) {
  // count varint + first value varint + (N-1) delta varints (max 5 each).
  return 5 + count * 5;
}

size_t estimateFixedBitWidthSectionSize(size_t count) {
  // bitWidth:1B + count varint + bufferSize (32 bits per element max).
  return 1 + 5 + FixedBitArray::bufferSize(count, /*bitWidth=*/32);
}

size_t estimateSectionSize(EncodingType encodingType, size_t count) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (getTrailerEncodingType(encodingType)) {
    case EncodingType::Trivial:
      return estimateTrivialSectionSize(count);
    case EncodingType::Varint:
      return estimateVarintSectionSize(count);
    case EncodingType::Delta:
      return estimateDeltaSectionSize(count);
    case EncodingType::FixedBitWidth:
      return estimateFixedBitWidthSectionSize(count);
    default:
      NIMBLE_FAIL(
          "Unsupported EncodingType for stream sizes trailer section: {}",
          encodingType);
  }
}

} // namespace

uint32_t getStringsTotalSize(std::string_view input) {
  const auto strData = reinterpret_cast<const std::string_view*>(input.data());
  const auto strDataEnd =
      reinterpret_cast<const std::string_view*>(input.end());
  uint32_t size = 0;
  for (auto sv = strData; sv < strDataEnd; ++sv) {
    size += sizeof(uint32_t);
    size += sv->size();
  }
  return size;
}

void encodeStrings(std::string_view input, uint32_t size, char* output) {
  const auto strData = reinterpret_cast<const std::string_view*>(input.data());
  const auto strDataEnd =
      reinterpret_cast<const std::string_view*>(input.end());
  encoding::writeUint32(size, output);
  for (auto sv = strData; sv < strDataEnd; ++sv) {
    encoding::writeString(*sv, output);
  }
}

uint32_t
encode(const SerializerOptions& options, std::string_view input, char* output) {
  // Size prefix + compression type + actual content
  uint32_t size = input.size();
  const auto compression = options.compressionType;
  bool writeUncompressed{true};
  if (compression != CompressionType::Uncompressed &&
      size >= options.compressionThreshold) {
    auto* compPos = output + sizeof(uint32_t);
    encoding::writeChar(static_cast<int8_t>(compression), compPos);
    std::optional<uint32_t> compressedSize;
    switch (compression) {
      case CompressionType::Zstd:
        compressedSize = compressZstd(options, input, compPos, size);
        break;
      case CompressionType::Lz4:
        compressedSize = compressLz4(options, input, compPos, size);
        break;
      default:
        NIMBLE_UNSUPPORTED("Unsupported compression {}", toString(compression));
    }
    if (compressedSize.has_value()) {
      size = *compressedSize;
      writeUncompressed = false;
    }
  }

  if (writeUncompressed) {
    auto* compPos = output + sizeof(uint32_t);
    encoding::writeChar(
        static_cast<int8_t>(CompressionType::Uncompressed), compPos);
    std::copy(input.data(), input.end(), compPos);
  }
  encoding::writeUint32(size + 1, output);
  return size + sizeof(uint32_t) + 1;
}

size_t estimateTrailerSize(
    size_t numStreams,
    EncodingType indicesEncodingType,
    EncodingType sizesEncodingType) {
  // [indicesEncType:1B][indicesPayload][sizesEncType:1B][sizesPayload]
  // [trailer_size:u32]
  // Worst case: every stream slot is non-zero, so both axes carry numStreams
  // entries.
  return sizeof(uint8_t) +
      estimateSectionSize(indicesEncodingType, numStreams) + sizeof(uint8_t) +
      estimateSectionSize(sizesEncodingType, numStreams) + sizeof(uint32_t);
}

size_t estimateTrailerSize(
    size_t numPresentStreams,
    size_t numUniqueStreams,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType) {
  // [streamIdsEncType:1B][streamIdsPayload]
  // [sizeIndicesEncType:1B][sizeIndicesPayload]
  // [uniqueSizesEncType:1B][uniqueSizesPayload][trailer_size:u32]
  return sizeof(uint8_t) +
      estimateSectionSize(streamIdsEncodingType, numPresentStreams) +
      sizeof(uint8_t) +
      estimateSectionSize(sizeIndicesEncodingType, numPresentStreams) +
      sizeof(uint8_t) +
      estimateSectionSize(uniqueSizesEncodingType, numUniqueStreams) +
      sizeof(uint32_t);
}

} // namespace facebook::nimble::serde::detail
