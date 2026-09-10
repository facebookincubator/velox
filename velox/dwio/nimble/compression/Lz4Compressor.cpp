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

#include "velox/dwio/nimble/compression/Lz4Compressor.h"

#include <lz4.h>
#include <optional>

namespace facebook::nimble {

CompressionResult Lz4Compressor::compress(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    DataType /* dataType */,
    int /* bitWidth */,
    const CompressionPolicy& compressionPolicy) {
  auto parameters = compressionPolicy.config().parameters.lz4;
  Vector<char> buffer{&memoryPool, data.size() + sizeof(uint32_t)};
  auto pos = buffer.data();
  encoding::writeUint32(data.size(), pos);
  auto ret = LZ4_compress_fast(
      data.data(), pos, data.size(), data.size(), parameters.accelerationLevel);
  const auto compressedSize = static_cast<size_t>(ret) + sizeof(uint32_t);
  if (ret == 0 ||
      !compressionPolicy.shouldAccept(
          CompressionType::Lz4, data.size(), compressedSize)) {
    return {
        .compressionType = CompressionType::Uncompressed,
        .buffer = std::nullopt,
    };
  }

  buffer.resize(ret + sizeof(uint32_t));
  return {
      .compressionType = CompressionType::Lz4,
      .buffer = std::move(buffer),
  };
}

velox::BufferPtr Lz4Compressor::uncompress(
    velox::memory::MemoryPool& memoryPool,
    const CompressionType /* compressionType */,
    const DataType /* dataType */,
    std::string_view data,
    velox::BufferPool* bufferPool) {
  auto pos = data.data();
  const uint32_t uncompressedSize = encoding::readUint32(pos);
  auto buffer = allocateBuffer(memoryPool, bufferPool, uncompressedSize);
  auto ret = LZ4_decompress_safe(
      pos,
      buffer->asMutable<char>(),
      data.size() - sizeof(uint32_t),
      uncompressedSize);
  NIMBLE_CHECK(ret >= 0, "Error decompressing LZ4 data.");
  return buffer;
}

std::optional<size_t> Lz4Compressor::uncompressedSize(
    std::string_view data) const {
  auto* pos = data.data();
  return encoding::readUint32(pos);
}

CompressionType Lz4Compressor::compressionType() {
  return CompressionType::Lz4;
}

} // namespace facebook::nimble
