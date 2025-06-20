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

#include "velox/common/compression/ZstdCompression.h"
#include "velox/common/base/Exceptions.h"

#include <zstd.h>
#include <zstd_errors.h>

namespace facebook::velox::common {
namespace {
constexpr int32_t kZstdDefaultCompressionLevel = 1;

Status zstdError(const char* prefixMessage, size_t errorCode) {
  return Status::IOError(prefixMessage, ZSTD_getErrorName(errorCode));
}
} // namespace

class ZstdCodec : public Codec {
 public:
  explicit ZstdCodec(int32_t compressionLevel);

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

  Expected<uint64_t> compressFixedLength(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  bool supportsStreamingCompression() const override;

  Expected<std::shared_ptr<StreamingCompressor>> makeStreamingCompressor()
      override;

  Expected<std::shared_ptr<StreamingDecompressor>> makeStreamingDecompressor()
      override;

  int32_t minCompressionLevel() const override;

  int32_t maxCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  int32_t compressionLevel() const override;

  CompressionKind compressionKind() const override;

  Expected<uint64_t> getUncompressedLength(
      const uint8_t* input,
      uint64_t inputLength) const override;

  std::string_view name() const override;

 private:
  int32_t compressionLevel_;
};

class ZstdCompressor : public StreamingCompressor {
 public:
  explicit ZstdCompressor(int32_t compressionLevel);

  ~ZstdCompressor() override;

  Status init() const;

  Expected<CompressResult> compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  Expected<FlushResult> flush(uint8_t* output, uint64_t outputLength) override;

  Expected<EndResult> finalize(uint8_t* output, uint64_t outputLength) override;

 private:
  ZSTD_CStream* stream_;
  int32_t compressionLevel_;
};

class ZstdDecompressor : public StreamingDecompressor {
 public:
  ZstdDecompressor();

  ~ZstdDecompressor() override;

  Status init();

  Expected<DecompressResult> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  Status reset() override;

  bool isFinished() override;

 private:
  ZSTD_DStream* stream_;
  bool finished_{false};
};

ZstdDecompressor::ZstdDecompressor() : stream_(ZSTD_createDStream()) {}

ZstdDecompressor::~ZstdDecompressor() {
  ZSTD_freeDStream(stream_);
}

Status ZstdDecompressor::init() {
  finished_ = false;
  const auto ret = ZSTD_initDStream(stream_);
  VELOX_RETURN_IF(ZSTD_isError(ret), zstdError("ZSTD init failed: ", ret));
  return Status::OK();
}

Expected<StreamingDecompressor::DecompressResult> ZstdDecompressor::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);
  ZSTD_inBuffer inBuffer{input, inputLength, 0};
  ZSTD_outBuffer outBuffer{output, outputLength, 0};

  const auto ret = ZSTD_decompressStream(stream_, &outBuffer, &inBuffer);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(ret), zstdError("ZSTD decompression failed: ", ret));
  finished_ = ret == 0;
  return DecompressResult{
      static_cast<uint64_t>(inBuffer.pos),
      static_cast<uint64_t>(outBuffer.pos),
      inBuffer.pos == 0 && outBuffer.pos == 0};
}

Status ZstdDecompressor::reset() {
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

Status ZstdCompressor::init() const {
  const auto ret = ZSTD_initCStream(stream_, compressionLevel_);
  VELOX_RETURN_IF(ZSTD_isError(ret), zstdError("ZSTD init failed: ", ret));
  return Status::OK();
}

Expected<StreamingCompressor::CompressResult> ZstdCompressor::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);
  ZSTD_inBuffer inBuffer{input, inputLength, 0};
  ZSTD_outBuffer outBuffer{output, outputLength, 0};

  const auto ret = ZSTD_compressStream(stream_, &outBuffer, &inBuffer);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(ret), zstdError("ZSTD compression failed: ", ret));
  return CompressResult{
      static_cast<uint64_t>(inBuffer.pos),
      static_cast<uint64_t>(outBuffer.pos),
      inBuffer.pos == 0};
}

Expected<StreamingCompressor::FlushResult> ZstdCompressor::flush(
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(output);
  ZSTD_outBuffer outBuffer{output, outputLength, 0};

  const auto ret = ZSTD_flushStream(stream_, &outBuffer);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(ret), zstdError("ZSTD flush failed: ", ret));
  return FlushResult{static_cast<uint64_t>(outBuffer.pos), ret > 0};
}

Expected<StreamingCompressor::EndResult> ZstdCompressor::finalize(
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(output);
  ZSTD_outBuffer outBuffer{output, outputLength, 0};

  const auto ret = ZSTD_endStream(stream_, &outBuffer);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(ret), zstdError("ZSTD end failed: ", ret));
  return EndResult{static_cast<uint64_t>(outBuffer.pos), ret > 0};
}

ZstdCodec::ZstdCodec(int32_t compressionLevel)
    : compressionLevel_(
          compressionLevel == kDefaultCompressionLevel
              ? kZstdDefaultCompressionLevel
              : compressionLevel) {}

uint64_t ZstdCodec::maxCompressedLength(uint64_t inputLength) {
  return ZSTD_compressBound(inputLength);
}

Expected<uint64_t> ZstdCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);

  auto compressedSize = ZSTD_compress(
      output, outputLength, input, inputLength, compressionLevel_);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(compressedSize),
      zstdError("ZSTD compression failed: ", compressedSize));
  return compressedSize;
}

Expected<uint64_t> ZstdCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);

  auto decompressedSize =
      ZSTD_decompress(output, outputLength, input, inputLength);
  VELOX_RETURN_UNEXPECTED_IF(
      ZSTD_isError(decompressedSize),
      zstdError("ZSTD decompression failed: ", decompressedSize));
  return decompressedSize;
}

Expected<uint64_t> ZstdCodec::compressFixedLength(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(output);

  auto compressedSize = ZSTD_compress(
      output, outputLength, input, inputLength, compressionLevel_);
  if (ZSTD_isError(compressedSize)) {
    // It's fine to hit dest size too small.
    if (ZSTD_getErrorCode(compressedSize) ==
        ZSTD_ErrorCode::ZSTD_error_dstSize_tooSmall) {
      return outputLength;
    }
    return folly::makeUnexpected(
        zstdError("ZSTD compression failed: ", compressedSize));
  }
  return compressedSize;
}

bool ZstdCodec::supportsStreamingCompression() const {
  return true;
}

Expected<std::shared_ptr<StreamingCompressor>>
ZstdCodec::makeStreamingCompressor() {
  auto ptr = std::make_shared<ZstdCompressor>(compressionLevel_);
  VELOX_RETURN_UNEXPECTED_NOT_OK(ptr->init());
  return ptr;
}

Expected<std::shared_ptr<StreamingDecompressor>>
ZstdCodec::makeStreamingDecompressor() {
  auto ptr = std::make_shared<ZstdDecompressor>();
  VELOX_RETURN_UNEXPECTED_NOT_OK(ptr->init());
  return ptr;
}

int32_t ZstdCodec::minCompressionLevel() const {
  return ZSTD_minCLevel();
}

int32_t ZstdCodec::maxCompressionLevel() const {
  return ZSTD_maxCLevel();
}

int32_t ZstdCodec::defaultCompressionLevel() const {
  return kZstdDefaultCompressionLevel;
}

int32_t ZstdCodec::compressionLevel() const {
  return compressionLevel_;
}

CompressionKind ZstdCodec::compressionKind() const {
  return CompressionKind_ZSTD;
}

Expected<uint64_t> ZstdCodec::getUncompressedLength(
    const uint8_t* input,
    uint64_t inputLength) const {
  VELOX_CHECK_NOT_NULL(input);

  // Read decompressed size from the frame if available in input.
  auto decompressedSize = ZSTD_getFrameContentSize(input, inputLength);
  if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN ||
      decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
    return folly::makeUnexpected(
        Status::IOError("Invalid ZSTD compressed data."));
  }
  return decompressedSize;
}

std::string_view ZstdCodec::name() const {
  return "zstd";
}

std::unique_ptr<Codec> makeZstdCodec(int32_t compressionLevel) {
  return std::make_unique<ZstdCodec>(compressionLevel);
}
} // namespace facebook::velox::common
