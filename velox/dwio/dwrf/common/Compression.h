/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/common/compression/Compression.h"
#include "velox/dwio/common/OutputStream.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/compression/Compression.h"
#include "velox/dwio/common/compression/CompressionBufferPool.h"
#include "velox/dwio/common/compression/PagedOutputStream.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/dwio/dwrf/common/Decryption.h"

#include "velox/common/compression/Lz4Compression.h"
#include "velox/common/compression/LzoCompression.h"
#include "velox/common/compression/ZlibCompression.h"

namespace facebook::velox::dwrf {

using namespace dwio::common::compression;

constexpr uint8_t PAGE_HEADER_SIZE = 3;

inline std::shared_ptr<common::CodecOptions> getDwrfOrcCompressionOptions(
    common::CompressionKind kind,
    int32_t zlibCompressionLevel,
    int32_t zstdCompressionLevel) {
  if (kind == common::CompressionKind_ZLIB) {
    return std::make_shared<common::ZlibCodecOptions>(
        common::ZlibFormat::kDeflate, zlibCompressionLevel);
  }
  if (kind == common::CompressionKind_ZSTD) {
    return std::make_shared<common::CodecOptions>(zstdCompressionLevel);
  }
  return std::make_shared<common::CodecOptions>();
}

/**
 * Create a compressor for the given compression kind.
 * @param kind The compression type to implement
 * @param bufferPool Pool for compression buffer
 * @param bufferHolder Buffer holder that handles buffer allocation and
 * collection
 * @param config The compression options to use
 */
inline std::unique_ptr<dwio::common::BufferedOutputStream> createCompressor(
    common::CompressionKind kind,
    CompressionBufferPool& bufferPool,
    dwio::common::DataBufferHolder& bufferHolder,
    const Config& config,
    const dwio::common::encryption::Encrypter* encrypter = nullptr) {
  switch (kind) {
    case common::CompressionKind_NONE: {
      if (encrypter == nullptr) {
        return std::make_unique<dwio::common::BufferedOutputStream>(
            bufferHolder);
      }
      return std::make_unique<PagedOutputStream>(
          bufferPool, bufferHolder, 0, nullptr, PAGE_HEADER_SIZE, encrypter);
    }
    case common::CompressionKind_ZLIB:
    case common::CompressionKind_ZSTD: {
      const auto options = getDwrfOrcCompressionOptions(
          kind,
          config.get(Config::ZLIB_COMPRESSION_LEVEL),
          config.get(Config::ZSTD_COMPRESSION_LEVEL));
      auto codec = common::Codec::create(kind, *options);
      if (codec.hasError()) {
        VELOX_USER_FAIL(codec.error().message());
      }
      return std::make_unique<PagedOutputStream>(
          bufferPool,
          bufferHolder,
          config.get(Config::COMPRESSION_THRESHOLD),
          std::move(codec.value()),
          PAGE_HEADER_SIZE,
          encrypter);
    }
    default:
      VELOX_UNSUPPORTED(
          "Unsupported compression type: {}", compressionKindToString(kind));
  }
}

inline std::shared_ptr<common::CodecOptions> getDwrfOrcDecompressionOptions(
    common::CompressionKind kind) {
  if (kind == common::CompressionKind_ZLIB ||
      kind == common::CompressionKind_GZIP) {
    return std::make_shared<common::ZlibCodecOptions>(
        common::ZlibFormat::kDeflate);
  }
  if (kind == common::CompressionKind_LZ4) {
    return std::make_shared<common::Lz4CodecOptions>(common::Lz4Type::kLz4Raw);
  }
  if (kind == common::CompressionKind_LZO) {
    return std::make_shared<common::LzoCodecOptions>(common::LzoType::kLzo);
  }

  return std::make_shared<common::CodecOptions>();
}

/**
 * Create a decompressor for the given compression kind.
 * @param kind The compression type to implement
 * @param input The input stream that is the underlying source
 * @param bufferSize The maximum size of the buffer
 * @param pool The memory pool
 */
inline std::unique_ptr<dwio::common::SeekableInputStream> createDecompressor(
    facebook::velox::common::CompressionKind kind,
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    uint64_t bufferSize,
    memory::MemoryPool& pool,
    const std::string& streamDebugInfo,
    const dwio::common::encryption::Decrypter* decryptr = nullptr) {
  return dwio::common::compression::createDecompressor(
      kind,
      std::move(input),
      bufferSize,
      pool,
      getDwrfOrcDecompressionOptions(kind),
      streamDebugInfo,
      decryptr);
}

} // namespace facebook::velox::dwrf
