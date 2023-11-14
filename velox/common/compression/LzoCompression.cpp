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

#include "velox/common/compression/LzoCompression.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/LzoDecompressor.h"

namespace facebook::velox::common {

uint64_t LzoCodec::maxCompressedLength(uint64_t inputLength) {
  VELOX_UNSUPPORTED("LZO compression is not supported.");
}

Expected<uint64_t> LzoCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  return folly::makeUnexpected(
      Status::NotImplemented("LZO compression is not supported."));
}

Expected<uint64_t> LzoCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  const char* inputAddress = reinterpret_cast<const char*>(input);
  char* outputAddress = reinterpret_cast<char*>(output);
  return velox::common::compression::lzoDecompress(
      inputAddress,
      inputAddress + inputLength,
      outputAddress,
      outputAddress + outputLength);
}

CompressionKind LzoCodec::compressionKind() const {
  return CompressionKind_LZO;
}

int32_t LzoCodec::minCompressionLevel() const {
  return kDefaultCompressionLevel;
}

int32_t LzoCodec::maxCompressionLevel() const {
  return kDefaultCompressionLevel;
}

int32_t LzoCodec::defaultCompressionLevel() const {
  return kDefaultCompressionLevel;
}

std::string_view LzoCodec::name() const {
  return "lzo";
}

Expected<uint64_t> LzoHadoopCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);
  uint64_t decompressedSize;
  if (tryDecompressHadoop(
          input, inputLength, output, outputLength, decompressedSize)) {
    return decompressedSize;
  }

  return folly::makeUnexpected(
      Status::Invalid("Failed to decompress LZO Hadoop data."));
}

Expected<uint64_t> LzoHadoopCodec::decompressInternal(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  return LzoCodec::decompress(input, inputLength, output, outputLength);
}

std::string_view LzoHadoopCodec::name() const {
  return "lzo_hadoop";
}

std::unique_ptr<Codec> makeLzoCodec() {
  return std::make_unique<LzoCodec>();
}

std::unique_ptr<Codec> makeLzoHadoopCodec() {
  return std::make_unique<LzoHadoopCodec>();
}
} // namespace facebook::velox::common
