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

#include "velox/dwio/nimble/serializer/ChunkedStreamPayload.h"

#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include <lz4.h>
#include <zstd.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/ZstdContext.h"

namespace facebook::nimble::serde {
namespace {

size_t decodedChunkSize(
    CompressionType compression,
    const char* data,
    uint32_t length) {
  switch (compression) {
    case CompressionType::Uncompressed:
      return length;
    case CompressionType::Zstd: {
      NIMBLE_CHECK_GE(
          length, sizeof(uint32_t), "Truncated ZSTD chunk size header");
      const auto* pos = data;
      return encoding::readUint32(pos);
    }
    case CompressionType::Lz4: {
      NIMBLE_CHECK_GE(
          length, sizeof(uint32_t), "Truncated LZ4 chunk size header");
      const auto* pos = data;
      return encoding::readUint32(pos);
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported chunk compression {}", compression);
  }
}

size_t strippedStreamSize(const char* pos, const char* end) {
  size_t size{0};
  while (pos < end) {
    NIMBLE_CHECK_GE(
        static_cast<size_t>(end - pos),
        kChunkHeaderSize,
        "Truncated chunk header in stream");
    const auto [chunkLength, compressionType] = readChunkHeader(pos);
    NIMBLE_CHECK_LE(
        chunkLength,
        static_cast<uint32_t>(end - pos),
        "Chunk data exceeds stream boundary");
    size += decodedChunkSize(compressionType, pos, chunkLength);
    pos += chunkLength;
  }
  return size;
}

void appendChunkData(
    CompressionType compression,
    const char* data,
    uint32_t length,
    char*& output) {
  switch (compression) {
    case CompressionType::Uncompressed: {
      std::memcpy(output, data, length);
      output += length;
      break;
    }
    case CompressionType::Zstd: {
      const auto decompressedSize = decodedChunkSize(compression, data, length);
      const auto* pos = data;
      encoding::readUint32(pos);
      const auto compressedSize = length - sizeof(uint32_t);
      const auto ret = ZSTD_decompressDCtx(
          detail::getThreadLocalDCtx(),
          output,
          decompressedSize,
          pos,
          compressedSize);
      NIMBLE_CHECK(!ZSTD_isError(ret), "Error decompressing chunk data");
      NIMBLE_CHECK_EQ(
          ret, decompressedSize, "ZSTD chunk decompressed size mismatch");
      output += decompressedSize;
      break;
    }
    case CompressionType::Lz4: {
      const auto decompressedSize = decodedChunkSize(compression, data, length);
      const auto* pos = data;
      encoding::readUint32(pos);
      const auto compressedSize =
          static_cast<size_t>(length - sizeof(uint32_t));
      const auto ret = LZ4_decompress_safe(
          pos,
          output,
          static_cast<int>(compressedSize),
          static_cast<int>(decompressedSize));
      NIMBLE_CHECK_EQ(
          ret,
          static_cast<int>(decompressedSize),
          "LZ4 chunk decompressed size mismatch");
      output += decompressedSize;
      break;
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported chunk compression {}", compression);
  }
}

std::optional<std::string_view> tryStripUncompressedChunks(
    std::string_view streamData,
    Buffer& outputBuffer) {
  NIMBLE_CHECK_GE(
      streamData.size(), kChunkHeaderSize, "Truncated chunk header in stream");

  auto* pos = streamData.data();
  const auto* const end = pos + streamData.size();
  const auto [firstChunkLength, firstCompressionType] = readChunkHeader(pos);
  NIMBLE_CHECK_LE(
      firstChunkLength,
      static_cast<uint32_t>(end - pos),
      "Chunk data exceeds stream boundary");
  if (firstCompressionType != CompressionType::Uncompressed) {
    return std::nullopt;
  }
  if (pos + firstChunkLength == end) {
    NIMBLE_CHECK_GT(
        firstChunkLength, 0, "Chunked stream must have a non-empty payload");
    return std::string_view{pos, firstChunkLength};
  }

  std::vector<std::string_view> chunks{{pos, firstChunkLength}};
  size_t payloadSize{firstChunkLength};
  pos += firstChunkLength;

  while (pos < end) {
    NIMBLE_CHECK_GE(
        static_cast<size_t>(end - pos),
        kChunkHeaderSize,
        "Truncated chunk header in stream");
    const auto [chunkLength, compressionType] = readChunkHeader(pos);
    NIMBLE_CHECK_LE(
        chunkLength,
        static_cast<uint32_t>(end - pos),
        "Chunk data exceeds stream boundary");
    if (compressionType != CompressionType::Uncompressed) {
      return std::nullopt;
    }
    chunks.push_back({pos, chunkLength});
    payloadSize += chunkLength;
    pos += chunkLength;
  }

  NIMBLE_CHECK_GT(
      payloadSize, 0, "Chunked stream must have a non-empty payload");
  auto* const data = outputBuffer.reserve(payloadSize);
  auto* output = data;
  auto* const outputEnd = output + payloadSize;
  for (const auto chunk : chunks) {
    std::memcpy(output, chunk.data(), chunk.size());
    output += chunk.size();
  }
  NIMBLE_CHECK_EQ(output, outputEnd, "Stripped chunk size mismatch");
  return std::string_view{data, payloadSize};
}

} // namespace

std::string_view stripChunkHeaders(
    std::string_view streamData,
    Buffer& outputBuffer) {
  if (auto uncompressedData =
          tryStripUncompressedChunks(streamData, outputBuffer)) {
    return *uncompressedData;
  }

  const char* pos = streamData.data();
  const auto* const end = pos + streamData.size();
  const auto payloadSize = strippedStreamSize(streamData.data(), end);
  NIMBLE_CHECK_GT(
      payloadSize, 0, "Chunked stream must have a non-empty payload");
  auto* const data = outputBuffer.reserve(payloadSize);
  auto* output = data;
  auto* const outputEnd = output + payloadSize;
  pos = streamData.data();
  while (pos < end) {
    NIMBLE_CHECK_GE(
        static_cast<size_t>(end - pos),
        kChunkHeaderSize,
        "Truncated chunk header in stream");
    const auto [chunkLength, compressionType] = readChunkHeader(pos);
    NIMBLE_CHECK_LE(
        chunkLength,
        static_cast<uint32_t>(end - pos),
        "Chunk data exceeds stream boundary");
    appendChunkData(compressionType, pos, chunkLength, output);
    pos += chunkLength;
  }
  NIMBLE_CHECK_EQ(output, outputEnd, "Stripped chunk size mismatch");
  return {data, payloadSize};
}

} // namespace facebook::nimble::serde
