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

#include "velox/common/compression/v2/LzoCompression.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/LzoDecompressor.h"

namespace facebook::velox::common {

LzoCodec::LzoCodec() = default;

uint64_t LzoCodec::maxCompressedLength(uint64_t inputLength) {
  VELOX_UNSUPPORTED("LZO compression is not supported.");
}

uint64_t LzoCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_UNSUPPORTED("LZO compression is not supported.");
}

uint64_t LzoCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  const char* inputAddress =
      reinterpret_cast<const char*>(const_cast<uint8_t*>(input));
  char* outputAddress = reinterpret_cast<char*>(output);
  return velox::common::compression::lzoDecompress(
      inputAddress,
      inputAddress + inputLength,
      outputAddress,
      outputAddress + outputLength);
}

std::shared_ptr<Compressor> LzoCodec::makeCompressor() {
  VELOX_UNSUPPORTED("Streaming compression unsupported with LZO");
}

std::shared_ptr<Decompressor> LzoCodec::makeDecompressor() {
  VELOX_UNSUPPORTED("Streaming decompression unsupported with LZO");
}

CompressionKind LzoCodec::compressionKind() const {
  return CompressionKind_LZO;
}

int32_t LzoCodec::minimumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t LzoCodec::maximumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t LzoCodec::defaultCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

LzoHadoopCodec::LzoHadoopCodec() = default;

CompressionKind LzoHadoopCodec::compressionKind() const {
  return CompressionKind_LZOHADOOP;
}

uint64_t LzoHadoopCodec::decompressInternal(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  return LzoCodec::decompress(inputLength, input, outputLength, output);
}

std::unique_ptr<Codec> makeLzoCodec() {
  return std::make_unique<LzoCodec>();
}

std::unique_ptr<Codec> makeLzoHadoopCodec() {
  return std::make_unique<LzoHadoopCodec>();
}
} // namespace facebook::velox::common
