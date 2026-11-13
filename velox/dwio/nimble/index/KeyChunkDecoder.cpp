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

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/compression/Compression.h"

namespace facebook::nimble::index {

std::shared_ptr<DecodedKeyChunk> decodeKeyChunk(
    std::string_view chunk,
    DataInput::Handle handle,
    const KeyReaderFactory& readerFactory,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_FILE_GE(
      chunk.size(),
      static_cast<size_t>(kChunkHeaderSize),
      "Key chunk is smaller than a chunk header");
  const auto* header = chunk.data();
  const auto chunkHeader = readChunkHeader(header);
  NIMBLE_CHECK_FILE(
      chunkHeader.compressionType == CompressionType::Uncompressed ||
          chunkHeader.compressionType == CompressionType::Zstd ||
          chunkHeader.compressionType == CompressionType::Lz4,
      "Unsupported key chunk compression type: {}",
      chunkHeader.compressionType);

  const auto dataLen = chunkHeader.length;
  NIMBLE_CHECK_FILE_GE(
      chunk.size() - kChunkHeaderSize,
      dataLen,
      "Key chunk is shorter than its header length");
  auto result = std::make_shared<DecodedKeyChunk>();
  result->handle = std::move(handle);

  // readChunkHeader advanced 'header' past the chunk header, so header/dataLen
  // is the raw chunk payload. For a compressed chunk it must be decompressed
  // into an owned buffer before decoding keys, so the returned (and cached)
  // DecodedKeyChunk holds uncompressed key bytes. Uses the same
  // Compression::uncompress path as the general chunked-stream reader.
  std::string_view encodedData{header, dataLen};
  if (chunkHeader.compressionType != CompressionType::Uncompressed) {
    auto uncompressed = Compression::uncompress(
        *pool,
        chunkHeader.compressionType,
        DataType::String,
        encodedData,
        /*decompressCounter=*/nullptr);
    // Drop the returned chunk's reference to the compressed payload so the
    // cached DecodedKeyChunk holds only the uncompressed key bytes.
    result->handle.reset();
    encodedData =
        std::string_view(uncompressed->as<char>(), uncompressed->size());
    result->stringBuffers.push_back(std::move(uncompressed));
  }

  auto* raw = result.get();
  result->reader = readerFactory(
      encodedData,
      [raw, pool](uint32_t totalLength) {
        auto& buffer = velox::checkedNotNull(raw)->stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, pool));
        return buffer->asMutable<void>();
      },
      pool);
  NIMBLE_CHECK_NOT_NULL(result->reader);
  return result;
}

} // namespace facebook::nimble::index
