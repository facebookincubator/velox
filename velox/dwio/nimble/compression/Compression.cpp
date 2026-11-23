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
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/compression/Lz4Compressor.h"
#include "velox/dwio/nimble/compression/OpenZLCompressor.h"
#include "velox/dwio/nimble/compression/ZstdCompressor.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

#ifndef DISABLE_META_INTERNAL_COMPRESSOR
#include "fb_velox/dwio/nimble/compression/MetaInternalCompressor.h"
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

// Forwards to 'policy' with its MetaInternal compression level replaced.
class TypedLanesLevelPolicy final : public CompressionPolicy {
 public:
  TypedLanesLevelPolicy(const CompressionPolicy& policy, int32_t level)
      : policy_{policy}, level_{level} {}

  CompressionConfig config() const override {
    auto config = policy_.config();
    config.parameters.metaInternal.compressionLevel =
        static_cast<int16_t>(level_);
    return config;
  }

  bool shouldAccept(
      CompressionType compressionType,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return policy_.shouldAccept(
        compressionType, uncompressedSize, compressedSize);
  }

 private:
  const CompressionPolicy& policy_;
  const int32_t level_;
};

CompressionResult compressWith(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    DataType dataType,
    int bitWidth,
    CompressionType compressionType,
    const CompressionPolicy& compressionPolicy) {
#ifndef NDEBUG
  CompressCall call{.dataType = dataType, .config = compressionPolicy.config()};
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::Compression::compress", &call);
#endif
  return getCompressor(compressionType)
      .compress(pool, data, dataType, bitWidth, compressionPolicy);
}
} // namespace

/* static */ CompressionResult Compression::compress(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    DataType dataType,
    int bitWidth,
    const CompressionPolicy& compressionPolicy,
    std::optional<int32_t> typedLanesLevel) {
  auto compression = compressionPolicy.config();

  // The one place the typed-lanes level applies: a level is passed, the payload
  // is typed, and MetaInternal compresses it.
  if (typedLanesLevel.has_value() && dataType != DataType::Undefined &&
      compression.compressionType == CompressionType::MetaInternal) {
    const TypedLanesLevelPolicy typedLanesLevelPolicy{
        compressionPolicy, *typedLanesLevel};
    return compressWith(
        pool,
        data,
        dataType,
        bitWidth,
        compression.compressionType,
        typedLanesLevelPolicy);
  }
  return compressWith(
      pool,
      data,
      dataType,
      bitWidth,
      compression.compressionType,
      compressionPolicy);
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
