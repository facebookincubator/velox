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

#include "velox/common/compression/v2/ZstdCompression.h"
#include <zstd_errors.h>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {
namespace {
void zstdError(size_t errorCode, const char* prefixMessage) {
  VELOX_FAIL(prefixMessage, ZSTD_getErrorName(errorCode));
}
} // namespace

class ZstdDecompressor : public Decompressor {
 public:
  ZstdDecompressor();

  ~ZstdDecompressor() override;

  void init();

  DecompressResult decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  void reset() override;

  bool isFinished() override;

 private:
  ZSTD_DStream* stream_;
  bool finished_{false};
};

class ZstdCompressor : public Compressor {
 public:
  explicit ZstdCompressor(int32_t compressionLevel);

  ~ZstdCompressor() override;

  void init();

  CompressResult compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  FlushResult flush(uint64_t outputLength, uint8_t* output) override;

  EndResult end(uint64_t outputLength, uint8_t* output) override;

 private:
  ZSTD_CStream* stream_;
  int32_t compressionLevel_;
};

ZstdDecompressor::ZstdDecompressor() : stream_(ZSTD_createDStream()) {}

ZstdDecompressor::~ZstdDecompressor() {
  ZSTD_freeDStream(stream_);
}

void ZstdDecompressor::init() {
  finished_ = false;
  size_t ret = ZSTD_initDStream(stream_);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD init failed: ");
  }
}

Decompressor::DecompressResult ZstdDecompressor::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  ZSTD_inBuffer inBuffer;
  ZSTD_outBuffer outBuffer;

  inBuffer.src = input;
  inBuffer.size = static_cast<size_t>(inputLength);
  inBuffer.pos = 0;
  outBuffer.dst = output;
  outBuffer.size = static_cast<size_t>(outputLength);
  outBuffer.pos = 0;

  auto ret = ZSTD_decompressStream(stream_, &outBuffer, &inBuffer);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD decompress failed: ");
  }
  finished_ = (ret == 0);
  return DecompressResult{
      static_cast<uint64_t>(inBuffer.pos),
      static_cast<uint64_t>(outBuffer.pos),
      inBuffer.pos == 0 && outBuffer.pos == 0};
}

void ZstdDecompressor::reset() {
  return init();
}

bool ZstdDecompressor::isFinished() {
  return finished_;
}

ZstdCompressor::ZstdCompressor(int32_t compressionLevel)
    : stream_(ZSTD_createCStream()), compressionLevel_(compressionLevel) {}

ZstdCompressor::~ZstdCompressor() {
  ZSTD_freeCStream(stream_);
}

void ZstdCompressor::init() {
  auto ret = ZSTD_initCStream(stream_, compressionLevel_);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD init failed: ");
  }
}

Compressor::CompressResult ZstdCompressor::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  ZSTD_inBuffer inBuffer;
  ZSTD_outBuffer outBuffer;

  inBuffer.src = input;
  inBuffer.size = static_cast<size_t>(inputLength);
  inBuffer.pos = 0;
  outBuffer.dst = output;
  outBuffer.size = static_cast<size_t>(outputLength);
  outBuffer.pos = 0;

  auto ret = ZSTD_compressStream(stream_, &outBuffer, &inBuffer);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD compress failed: ");
  }
  return CompressResult{
      static_cast<uint64_t>(inBuffer.pos),
      static_cast<uint64_t>(outBuffer.pos),
      inBuffer.pos == 0};
}

Compressor::FlushResult ZstdCompressor::flush(
    uint64_t outputLength,
    uint8_t* output) {
  ZSTD_outBuffer outBuffer;

  outBuffer.dst = output;
  outBuffer.size = static_cast<size_t>(outputLength);
  outBuffer.pos = 0;

  auto ret = ZSTD_flushStream(stream_, &outBuffer);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD flush failed: ");
  }
  return FlushResult{static_cast<uint64_t>(outBuffer.pos), ret > 0};
}

Compressor::EndResult ZstdCompressor::end(
    uint64_t outputLength,
    uint8_t* output) {
  ZSTD_outBuffer outBuffer;

  outBuffer.dst = output;
  outBuffer.size = static_cast<size_t>(outputLength);
  outBuffer.pos = 0;

  auto ret = ZSTD_endStream(stream_, &outBuffer);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD end failed: ");
  }
  return EndResult{static_cast<uint64_t>(outBuffer.pos), ret > 0};
}

ZstdCodec::ZstdCodec(int32_t compressionLevel)
    : compressionLevel_(
          compressionLevel == kUseDefaultCompressionLevel
              ? kZSTDDefaultCompressionLevel
              : compressionLevel) {}

uint64_t ZstdCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  if (output == nullptr) {
    // We may pass a NULL 0-byte output buffer but some zstd versions demand
    // a valid pointer: https://github.com/facebook/zstd/issues/1385
    static uint8_t emptyBuffer;
    VELOX_DCHECK_EQ(outputLength, 0);
    output = &emptyBuffer;
  }

  auto ret = ZSTD_decompress(
      output,
      static_cast<size_t>(outputLength),
      input,
      static_cast<size_t>(inputLength));
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD decompression failed: ");
  }
  VELOX_CHECK_EQ(
      static_cast<uint64_t>(ret),
      outputLength,
      "Corrupt ZSTD compressed data.");
  return static_cast<uint64_t>(ret);
}

uint64_t ZstdCodec::maxCompressedLength(uint64_t inputLength) {
  return ZSTD_compressBound(static_cast<size_t>(inputLength));
}

std::optional<uint64_t> ZstdCodec::doGetUncompressedLength(
    uint64_t inputLength,
    const uint8_t* input,
    std::optional<uint64_t> uncompressedLength) const {
  // Read decompressed size from frame if available in input.
  auto decompressedSize = ZSTD_getFrameContentSize(input, inputLength);
  if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN ||
      decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
    return uncompressedLength;
  }
  return decompressedSize;
}

uint64_t ZstdCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputBufferLength,
    uint8_t* output) {
  auto ret = ZSTD_compress(
      output,
      static_cast<size_t>(outputBufferLength),
      input,
      static_cast<size_t>(inputLength),
      compressionLevel_);
  if (ZSTD_isError(ret)) {
    zstdError(ret, "ZSTD compression failed: ");
  }
  return static_cast<uint64_t>(ret);
}

uint64_t ZstdCodec::compressPartial(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputBufferLength,
    uint8_t* output) {
  auto ret = ZSTD_compress(
      output,
      static_cast<size_t>(outputBufferLength),
      input,
      static_cast<size_t>(inputLength),
      compressionLevel_);
  if (ZSTD_isError(ret)) {
    // It's fine to hit dest size too small.
    if (ZSTD_getErrorCode(ret) == ZSTD_ErrorCode::ZSTD_error_dstSize_tooSmall) {
      return outputBufferLength;
    }
    zstdError(ret, "ZSTD compression failed: ");
  }
  return static_cast<uint64_t>(ret);
}

std::shared_ptr<Compressor> ZstdCodec::makeCompressor() {
  auto ptr = std::make_shared<ZstdCompressor>(compressionLevel_);
  ptr->init();
  return ptr;
}

std::shared_ptr<Decompressor> ZstdCodec::makeDecompressor() {
  auto ptr = std::make_shared<ZstdDecompressor>();
  ptr->init();
  return ptr;
}

CompressionKind ZstdCodec::compressionKind() const {
  return CompressionKind_ZSTD;
}

int32_t ZstdCodec::minimumCompressionLevel() const {
  return ZSTD_minCLevel();
}

int32_t ZstdCodec::maximumCompressionLevel() const {
  return ZSTD_maxCLevel();
}

int32_t ZstdCodec::defaultCompressionLevel() const {
  return kZSTDDefaultCompressionLevel;
}

int32_t ZstdCodec::compressionLevel() const {
  return compressionLevel_;
}

std::unique_ptr<Codec> makeZstdCodec(int32_t compressionLevel) {
  return std::make_unique<ZstdCodec>(compressionLevel);
}
} // namespace facebook::velox::common
