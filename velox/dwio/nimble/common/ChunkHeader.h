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

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble {

/// Chunk header: 4 bytes for compressed chunk length + 1 byte for compression
/// type. This constant must remain 5 to maintain backward compatibility with
/// existing Nimble files.
constexpr int kChunkHeaderSize = 5;

struct ChunkHeader {
  uint32_t length;
  CompressionType compressionType;
};

/// Reads a chunk header from 'pos', advancing 'pos' by kChunkHeaderSize bytes.
inline ChunkHeader readChunkHeader(const char*& pos) {
  return {
      encoding::readUint32(pos),
      static_cast<CompressionType>(encoding::readChar(pos))};
}

/// Writes a chunk header to 'pos', advancing 'pos' by kChunkHeaderSize bytes.
inline void
writeChunkHeader(uint32_t length, CompressionType compressionType, char*& pos) {
  encoding::writeUint32(length, pos);
  encoding::write(compressionType, pos);
}

} // namespace facebook::nimble
