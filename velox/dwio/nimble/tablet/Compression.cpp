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
#include <zstd.h>
#include <zstd_errors.h>
#include <optional>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/tablet/Compression.h"

namespace facebook::nimble {

std::optional<velox::BufferPtr> ZstdCompression::compress(
    std::string_view source,
    velox::memory::MemoryPool* pool,
    int32_t compressionLevel) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_GE(
      compressionLevel, ZSTD_minCLevel(), "Compression level below minimum");
  NIMBLE_CHECK_LE(
      compressionLevel, ZSTD_maxCLevel(), "Compression level above maximum");
  const auto maxSize = source.size() + sizeof(uint32_t);
  auto buffer = velox::AlignedBuffer::allocateExact<char>(maxSize, pool);
  auto* pos = buffer->asMutable<char>();
  encoding::writeUint32(source.size(), pos);
  auto ret = ZSTD_compress(
      pos,
      source.size() - sizeof(uint32_t),
      source.data(),
      source.size(),
      compressionLevel);
  if (ZSTD_isError(ret)) {
    NIMBLE_CHECK(
        ZSTD_getErrorCode(ret) == ZSTD_ErrorCode::ZSTD_error_dstSize_tooSmall,
        "Error while compressing data");
    return std::nullopt;
  }

  buffer->setSize(ret + sizeof(uint32_t));
  return buffer;
}

velox::BufferPtr ZstdCompression::uncompress(
    std::string_view source,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  auto pos = source.data();
  const uint32_t uncompressedSize = encoding::readUint32(pos);
  auto buffer =
      velox::AlignedBuffer::allocateExact<char>(uncompressedSize, pool);
  auto ret = ZSTD_decompress(
      buffer->asMutable<char>(),
      buffer->size(),
      pos,
      source.size() - sizeof(uint32_t));
  NIMBLE_CHECK(
      !ZSTD_isError(ret),
      "Error uncompressing data: {}",
      ZSTD_getErrorName(ret));
  return buffer;
}

void ZstdCompression::uncompress(
    std::string_view source,
    void* destination,
    uint64_t destinationSize) {
  auto pos = source.data();
  const uint32_t uncompressedSize = encoding::readUint32(pos);
  NIMBLE_CHECK_EQ(
      destinationSize,
      uncompressedSize,
      "Destination size mismatch with uncompressed size");
  auto ret = ZSTD_decompress(
      destination, destinationSize, pos, source.size() - sizeof(uint32_t));
  NIMBLE_CHECK(
      !ZSTD_isError(ret),
      "Error uncompressing data: {}",
      ZSTD_getErrorName(ret));
}

} // namespace facebook::nimble
