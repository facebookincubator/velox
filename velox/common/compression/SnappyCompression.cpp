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

#include "velox/common/compression/SnappyCompression.h"
#include "velox/common/base/Exceptions.h"

#include <snappy.h>

namespace facebook::velox::common {

class SnappyCodec : public Codec {
 public:
  uint64_t maxCompressedLength(uint64_t inputLength) override;

  Expected<uint64_t> compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  Expected<uint64_t> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  CompressionKind compressionKind() const override;

  int32_t minCompressionLevel() const override;

  int32_t maxCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  Expected<uint64_t> getUncompressedLength(
      const uint8_t* input,
      uint64_t inputLength) const override;

  std::string_view name() const override;
};

uint64_t SnappyCodec::maxCompressedLength(uint64_t inputLength) {
  DCHECK_GE(inputLength, 0);
  return static_cast<uint64_t>(
      snappy::MaxCompressedLength(static_cast<size_t>(inputLength)));
}

Expected<uint64_t> SnappyCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  size_t output_size;
  snappy::RawCompress(
      reinterpret_cast<const char*>(input),
      static_cast<size_t>(inputLength),
      reinterpret_cast<char*>(output),
      &output_size);
  return static_cast<uint64_t>(output_size);
}

Expected<uint64_t> SnappyCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  size_t decompressedSize;
  VELOX_RETURN_UNEXPECTED_IF(
      !snappy::GetUncompressedLength(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          &decompressedSize),
      Status::IOError("Corrupt snappy compressed data."));
  VELOX_RETURN_UNEXPECTED_IF(
      outputLength < decompressedSize,
      Status::IOError("Output length is too small"));
  VELOX_RETURN_UNEXPECTED_IF(
      !snappy::RawUncompress(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          reinterpret_cast<char*>(output)),
      Status::IOError("Corrupt snappy compressed data."));
  return static_cast<uint64_t>(decompressedSize);
}

CompressionKind SnappyCodec::compressionKind() const {
  return CompressionKind_SNAPPY;
}

int32_t SnappyCodec::minCompressionLevel() const {
  return kDefaultCompressionLevel;
}

int32_t SnappyCodec::maxCompressionLevel() const {
  return kDefaultCompressionLevel;
}

int32_t SnappyCodec::defaultCompressionLevel() const {
  return kDefaultCompressionLevel;
}

Expected<uint64_t> SnappyCodec::getUncompressedLength(
    const uint8_t* input,
    uint64_t inputLength) const {
  size_t decompressedSize;
  if (!snappy::GetUncompressedLength(
          reinterpret_cast<const char*>(input),
          static_cast<size_t>(inputLength),
          &decompressedSize)) {
    return folly::makeUnexpected(
        Status::IOError("Invalid ZSTD compressed data."));
  }
  return static_cast<uint64_t>(decompressedSize);
}

std::string_view SnappyCodec::name() const {
  return "snappy";
}

std::unique_ptr<Codec> makeSnappyCodec() {
  return std::make_unique<SnappyCodec>();
}
} // namespace facebook::velox::common
