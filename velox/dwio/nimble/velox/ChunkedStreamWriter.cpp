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
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"

namespace facebook::nimble {

namespace {

class ChunkCompressionPolicy : public CompressionPolicy {
 public:
  explicit ChunkCompressionPolicy(CompressionParams params) : params_{params} {
    NIMBLE_CHECK(
        params_.type == CompressionType::Zstd ||
            params_.type == CompressionType::Lz4,
        "Unsupported chunk compression type: {}",
        params_.type);
  }

  CompressionConfig config() const override {
    CompressionConfig config{.compressionType = params_.type};
    if (params_.type == CompressionType::Zstd) {
      config.parameters.zstd.compressionLevel = params_.zstdLevel;
    }
    return config;
  }

  bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return compressedSize <= uncompressedSize * params_.acceptRatio;
  }

 private:
  const CompressionParams params_;
};

} // namespace

ChunkedStreamWriter::ChunkedStreamWriter(
    Buffer& buffer,
    CompressionParams compressionParams)
    : buffer_{buffer}, compressionParams_{compressionParams} {
  NIMBLE_CHECK(
      compressionParams_.type == CompressionType::Uncompressed ||
          compressionParams_.type == CompressionType::Zstd ||
          compressionParams_.type == CompressionType::Lz4,
      "Unsupported chunked stream compression type: {}",
      compressionParams_.type);
}

std::vector<std::string_view> ChunkedStreamWriter::encode(
    std::string_view chunk) {
  auto* header = buffer_.reserve(kChunkHeaderSize);
  auto* pos = header;

  if (compressionParams_.type != CompressionType::Uncompressed) {
    ChunkCompressionPolicy policy{compressionParams_};
    auto result = Compression::compress(
        buffer_.getMemoryPool(),
        chunk,
        DataType::String,
        /*bitWidth=*/0,
        policy);
    if (result.buffer.has_value() &&
        result.compressionType == compressionParams_.type) {
      auto view =
          buffer_.writeString({result.buffer->data(), result.buffer->size()});
      writeChunkHeader(
          static_cast<uint32_t>(view.size()), result.compressionType, pos);
      return {{header, kChunkHeaderSize}, view};
    }
  }

  writeChunkHeader(
      static_cast<uint32_t>(chunk.size()), CompressionType::Uncompressed, pos);
  return {{header, kChunkHeaderSize}, chunk};
}

} // namespace facebook::nimble
