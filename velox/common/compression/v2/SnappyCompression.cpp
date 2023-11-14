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

#include "velox/common/compression/v2/SnappyCompression.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {

uint64_t SnappyCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  size_t decompressedSize;
  VELOX_CHECK(
      snappy::GetUncompressedLength(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          &decompressedSize),
      "Corrupt snappy compressed data.");
  VELOX_CHECK_GE(outputLength, decompressedSize, "Output length is too small");
  VELOX_CHECK(
      snappy::RawUncompress(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          reinterpret_cast<char*>(output)),
      "Corrupt snappy compressed data.");
  return static_cast<uint64_t>(decompressedSize);
}

uint64_t SnappyCodec::maxCompressedLength(uint64_t inputLength) {
  DCHECK_GE(inputLength, 0);
  return static_cast<uint64_t>(
      snappy::MaxCompressedLength(static_cast<size_t>(inputLength)));
}

uint64_t SnappyCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  size_t output_size;
  snappy::RawCompress(
      reinterpret_cast<const char*>(input),
      static_cast<size_t>(inputLength),
      reinterpret_cast<char*>(output),
      &output_size);
  return static_cast<uint64_t>(output_size);
}

std::shared_ptr<Compressor> SnappyCodec::makeCompressor() {
  VELOX_UNSUPPORTED("Streaming compression unsupported with Snappy");
}

std::shared_ptr<Decompressor> SnappyCodec::makeDecompressor() {
  VELOX_UNSUPPORTED("Streaming decompression unsupported with Snappy");
}

CompressionKind SnappyCodec::compressionKind() const {
  return CompressionKind_SNAPPY;
}

int32_t SnappyCodec::minimumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t SnappyCodec::maximumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t SnappyCodec::defaultCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

std::optional<uint64_t> SnappyCodec::doGetUncompressedLength(
    uint64_t inputLength,
    const uint8_t* input,
    std::optional<uint64_t> uncompressedLength) const {
  size_t decompressedSize;
  if (!snappy::GetUncompressedLength(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          &decompressedSize)) {
    return uncompressedLength;
  }
  return static_cast<uint64_t>(decompressedSize);
}

std::unique_ptr<Codec> makeSnappyCodec() {
  return std::make_unique<SnappyCodec>();
}
} // namespace facebook::velox::common
