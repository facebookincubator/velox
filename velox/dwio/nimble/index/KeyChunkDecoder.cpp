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
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/compression/Compression.h"

namespace facebook::nimble::index {

std::shared_ptr<DecodedKeyChunk> decodeKeyChunk(
    std::unique_ptr<velox::dwio::common::SeekableInputStream> inputStream,
    velox::memory::MemoryPool& pool,
    velox::BufferPtr& dataBuffer) {
  const void* buf;
  int bufLen{0};
  NIMBLE_CHECK(inputStream->Next(&buf, &bufLen));
  NIMBLE_CHECK_GE(bufLen, kChunkHeaderSize);
  const auto* header = static_cast<const char*>(buf);
  const auto chunkHeader = readChunkHeader(header);
  NIMBLE_CHECK(
      chunkHeader.compressionType == CompressionType::Uncompressed ||
          chunkHeader.compressionType == CompressionType::Zstd ||
          chunkHeader.compressionType == CompressionType::Lz4,
      "Unsupported key chunk compression type: {}",
      chunkHeader.compressionType);

  const auto dataLen = chunkHeader.length;
  auto result = std::make_shared<DecodedKeyChunk>();

  // Determine the contiguous data pointer. If the first buffer contains all
  // the data, use it directly (zero-copy). Otherwise, copy across buffers.
  // readChunkHeader advanced 'header' past the chunk header, so the remaining
  // available bytes are bufLen - kChunkHeaderSize.
  const char* chunkData;
  bufLen -= kChunkHeaderSize;
  if (bufLen >= static_cast<int>(dataLen)) {
    result->dataStream = std::move(inputStream);
    chunkData = header;
  } else {
    // Reuse the caller's buffer when it has sufficient capacity; otherwise
    // allocate fresh and write it back into the slot. The destination buffer
    // is appended to stringBuffers so the returned DecodedKeyChunk holds its
    // own reference (co-owned with the caller's slot).
    if (dataBuffer == nullptr || dataBuffer->capacity() < dataLen) {
      dataBuffer = velox::AlignedBuffer::allocate<char>(dataLen, &pool);
    }
    dataBuffer->setSize(dataLen);
    auto* dest = dataBuffer->asMutable<char>();
    std::memcpy(dest, header, bufLen);
    int copied = bufLen;
    while (copied < static_cast<int>(dataLen)) {
      NIMBLE_CHECK(inputStream->Next(&buf, &bufLen));
      const int toCopy = std::min(bufLen, static_cast<int>(dataLen) - copied);
      std::memcpy(dest + copied, buf, toCopy);
      copied += toCopy;
    }
    chunkData = dest;
    result->stringBuffers.push_back(dataBuffer);
  }

  // chunkData/dataLen is the raw chunk payload. For a compressed chunk it must
  // be decompressed into an owned buffer before decoding keys, so the returned
  // (and cached) DecodedKeyChunk holds uncompressed key bytes. Uses the same
  // Compression::uncompress path as the general chunked-stream reader.
  std::string_view encodedData{chunkData, dataLen};
  if (chunkHeader.compressionType != CompressionType::Uncompressed) {
    auto uncompressed = Compression::uncompress(
        pool,
        chunkHeader.compressionType,
        DataType::String,
        encodedData,
        /*decompressCounter=*/nullptr);
    // Drop the returned chunk's references to the compressed payload (input
    // stream or staging copy) so the cached DecodedKeyChunk holds only the
    // uncompressed key bytes. The caller's dataBuffer scratch may still hold
    // the staging copy, but callers use only the returned chunk and discard it.
    result->dataStream.reset();
    result->stringBuffers.clear();
    encodedData =
        std::string_view(uncompressed->as<char>(), uncompressed->size());
    result->stringBuffers.push_back(std::move(uncompressed));
  }

  auto* raw = result.get();
  result->encoding = KeyEncoding::create(
      pool, encodedData, [raw, &pool](uint32_t totalLength) {
        auto& buffer = raw->stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, &pool));
        return buffer->asMutable<void>();
      });
  NIMBLE_CHECK(
      result->encoding->encodingType() == EncodingType::Trivial ||
          result->encoding->encodingType() == EncodingType::Prefix,
      "Unsupported encoding type: {}",
      result->encoding->encodingType());
  return result;
}

} // namespace facebook::nimble::index
