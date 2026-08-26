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
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/compression/Lz4Compressor.h"
#include "velox/dwio/nimble/compression/OpenZLCompressor.h"
#include "velox/dwio/nimble/compression/ZstdCompressor.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

#ifndef DISABLE_META_INTERNAL_COMPRESSOR
#include "velox/dwio/nimble/compression/fb/MetaInternalCompressor.h"
#endif

namespace facebook::nimble {

namespace {

struct CompressorRegistry {
  CompressorRegistry() {
    compressors.reserve(4);
    compressors.emplace(
        CompressionType::Zstd, std::make_unique<ZstdCompressor>());
    compressors.emplace(
        CompressionType::Lz4, std::make_unique<Lz4Compressor>());
    compressors.emplace(
        CompressionType::OpenZL, std::make_unique<OpenZLCompressor>());
#ifndef DISABLE_META_INTERNAL_COMPRESSOR
    compressors.emplace(
        CompressionType::MetaInternal,
        std::make_unique<MetaInternalCompressor>());
#endif
  }

  std::unordered_map<CompressionType, std::unique_ptr<ICompressor>> compressors;
};

ICompressor& getCompressor(CompressionType compressionType) {
  static CompressorRegistry registry;
  auto it = registry.compressors.find(compressionType);
  NIMBLE_CHECK(
      it != registry.compressors.end(),
      "Compressor for type {} is not registered.",
      toString(compressionType));
  return *it->second;
}
} // namespace

/* static */ CompressionResult Compression::compress(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    DataType dataType,
    int bitWidth,
    const CompressionPolicy& compressionPolicy) {
  auto compression = compressionPolicy.config();

  return getCompressor(compression.compressionType)
      .compress(pool, data, dataType, bitWidth, compressionPolicy);
}

/* static */ velox::BufferPtr Compression::uncompress(
    velox::memory::MemoryPool& pool,
    CompressionType compressionType,
    DataType dataType,
    std::string_view data,
    velox::io::IoCounter* decompressCounter,
    velox::BufferPool* bufferPool) {
  auto& compressor = getCompressor(compressionType);
  return velox::dwio::common::withDecompressStats(decompressCounter, [&] {
    return compressor.uncompress(
        pool, compressionType, dataType, data, bufferPool);
  });
}

/* static */ std::optional<size_t> Compression::uncompressedSize(
    CompressionType compressionType,
    std::string_view data) {
  return getCompressor(compressionType).uncompressedSize(data);
}

} // namespace facebook::nimble
