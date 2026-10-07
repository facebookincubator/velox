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

#include "velox/dwio/nimble/compression/ZstdCompressor.h"

#include <zstd.h>
#include <zstd_errors.h>
#include <memory>
#include <optional>

namespace facebook::nimble {

CompressionResult ZstdCompressor::compress(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    DataType dataType,
    int /* bitWidth */,
    const CompressionPolicy& compressionPolicy) {
  auto parameters = compressionPolicy.config().parameters.zstd;
  Vector<char> buffer{&pool, data.size() + sizeof(uint32_t)};
  auto pos = buffer.data();
  encoding::writeUint32(data.size(), pos);
  auto ret = ZSTD_compress(
      pos, data.size(), data.data(), data.size(), parameters.compressionLevel);
  if (ZSTD_isError(ret)) {
    NIMBLE_CHECK(
        ZSTD_getErrorCode(ret) == ZSTD_ErrorCode::ZSTD_error_dstSize_tooSmall,
        "Error while compressing data: {}",
        ZSTD_getErrorName(ret));
    return {
        .compressionType = CompressionType::Uncompressed,
        .buffer = std::nullopt,
    };
  }
  const auto compressedSize = ret + sizeof(uint32_t);
  const bool shouldAccept = compressionPolicy.shouldAccept(
      CompressionType::Zstd, data.size(), compressedSize);
  if (!shouldAccept) {
    return {
        .compressionType = CompressionType::Uncompressed,
        .buffer = std::nullopt,
    };
  }

  buffer.resize(compressedSize);
  return {
      .compressionType = CompressionType::Zstd,
      .buffer = std::move(buffer),
  };
}

namespace {
ZSTD_DCtx* getThreadLocalDCtx() {
  struct DCtxDeleter {
    void operator()(ZSTD_DCtx* ctx) const {
      ZSTD_freeDCtx(ctx);
    }
  };
  static thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> ctx{
      ZSTD_createDCtx()};
  NIMBLE_CHECK(ctx != nullptr, "Failed to create ZSTD decompression context");
  return ctx.get();
}
} // namespace

velox::BufferPtr ZstdCompressor::uncompress(
    velox::memory::MemoryPool& pool,
    const CompressionType compressionType,
    const DataType /* dataType */,
    std::string_view data,
    velox::BufferPool* bufferPool) {
  auto pos = data.data();
  const uint32_t uncompressedSize = encoding::readUint32(pos);
  auto buffer = allocateBuffer(pool, bufferPool, uncompressedSize);
  auto ret = ZSTD_decompressDCtx(
      getThreadLocalDCtx(),
      buffer->asMutable<char>(),
      buffer->size(),
      pos,
      data.size() - sizeof(uint32_t));
  NIMBLE_CHECK(
      !ZSTD_isError(ret),
      "Error uncompressing data: {}",
      ZSTD_getErrorName(ret));
  return buffer;
}

std::optional<size_t> ZstdCompressor::uncompressedSize(
    std::string_view data) const {
  auto* pos = data.data();
  return encoding::readUint32(pos);
}

CompressionType ZstdCompressor::compressionType() {
  return CompressionType::Zstd;
}

} // namespace facebook::nimble
