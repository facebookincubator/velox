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

#include "velox/common/compression/Lz4Compression.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {
namespace {

constexpr int32_t kLz4DefaultCompressionLevel = 1;
constexpr int32_t kLz4MinCompressionLevel = 1;

void lz4Error(const char* prefixMessage, LZ4F_errorCode_t errorCode) {
  VELOX_FAIL(prefixMessage, LZ4F_getErrorName(errorCode));
}

LZ4F_preferences_t defaultPreferences() {
  LZ4F_preferences_t prefs;
  memset(&prefs, 0, sizeof(prefs));
  return prefs;
}

LZ4F_preferences_t defaultPreferences(int compressionLevel) {
  LZ4F_preferences_t prefs = defaultPreferences();
  prefs.compressionLevel = compressionLevel;
  return prefs;
}
} // namespace

class LZ4Compressor : public StreamingCompressor {
 public:
  explicit LZ4Compressor(int32_t compressionLevel);

  ~LZ4Compressor() override;

  void init();

  CompressResult compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  FlushResult flush(uint8_t* output, uint64_t outputLength) override;

  EndResult end(uint8_t* output, uint64_t outputLength) override;

 protected:
  void
  compressBegin(uint8_t* output, size_t& outputLen, uint64_t& bytesWritten);

  int compressionLevel_;
  LZ4F_compressionContext_t ctx_{nullptr};
  LZ4F_preferences_t prefs_;
  bool firstTime_;
};

class LZ4Decompressor : public StreamingDecompressor {
 public:
  LZ4Decompressor() {}

  ~LZ4Decompressor() override {
    if (ctx_ != nullptr) {
      LZ4F_freeDecompressionContext(ctx_);
    }
  }

  void init();

  void reset() override;

  DecompressResult decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  bool isFinished() override;

 protected:
  LZ4F_decompressionContext_t ctx_{nullptr};
  bool finished_{false};
};

LZ4Compressor::LZ4Compressor(int32_t compressionLevel)
    : compressionLevel_(compressionLevel) {}

LZ4Compressor::~LZ4Compressor() {
  if (ctx_ != nullptr) {
    LZ4F_freeCompressionContext(ctx_);
  }
}

void LZ4Compressor::init() {
  LZ4F_errorCode_t ret;
  prefs_ = defaultPreferences(compressionLevel_);
  firstTime_ = true;

  ret = LZ4F_createCompressionContext(&ctx_, LZ4F_VERSION);
  if (LZ4F_isError(ret)) {
    lz4Error("LZ4 init failed: ", ret);
  }
}

StreamingCompressor::CompressResult LZ4Compressor::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  auto inputSize = static_cast<size_t>(inputLength);
  auto outputSize = static_cast<size_t>(outputLength);
  uint64_t bytesWritten = 0;

  if (firstTime_) {
    // Output too small to write LZ4F header.
    if (outputLength < LZ4F_HEADER_SIZE_MAX) {
      return CompressResult{0, 0, true};
    }
    compressBegin(output, outputSize, bytesWritten);
  }

  if (outputSize < LZ4F_compressBound(inputSize, &prefs_)) {
    // Output too small to compress into.
    return CompressResult{0, bytesWritten, true};
  }
  auto numBytesOrError = LZ4F_compressUpdate(
      ctx_, output, outputSize, input, inputSize, nullptr /* options */);
  if (LZ4F_isError(numBytesOrError)) {
    lz4Error("LZ4 compress update failed: ", numBytesOrError);
  }
  bytesWritten += static_cast<int64_t>(numBytesOrError);
  VELOX_DCHECK_LE(bytesWritten, outputSize);
  return CompressResult{inputLength, bytesWritten, false};
}

StreamingCompressor::FlushResult LZ4Compressor::flush(
    uint8_t* output,
    uint64_t outputLength) {
  auto outputSize = static_cast<size_t>(outputLength);
  uint64_t bytesWritten = 0;

  if (firstTime_) {
    // Output too small to write LZ4F header.
    if (outputLength < LZ4F_HEADER_SIZE_MAX) {
      return FlushResult{0, true};
    }
    compressBegin(output, outputSize, bytesWritten);
  }

  if (outputSize < LZ4F_compressBound(0, &prefs_)) {
    // Output too small to flush into.
    return FlushResult{bytesWritten, true};
  }

  auto numBytesOrError =
      LZ4F_flush(ctx_, output, outputSize, nullptr /* options */);
  if (LZ4F_isError(numBytesOrError)) {
    lz4Error("LZ4 flush failed: ", numBytesOrError);
  }
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
  VELOX_DCHECK_LE(bytesWritten, outputLength);
  return FlushResult{bytesWritten, false};
}

StreamingCompressor::EndResult LZ4Compressor::end(
    uint8_t* output,
    uint64_t outputLength) {
  auto outputSize = static_cast<size_t>(outputLength);
  uint64_t bytesWritten = 0;

  if (firstTime_) {
    // Output too small to write LZ4F header.
    if (outputLength < LZ4F_HEADER_SIZE_MAX) {
      return EndResult{0, true};
    }
    compressBegin(output, outputSize, bytesWritten);
  }

  if (outputSize < LZ4F_compressBound(0, &prefs_)) {
    // Output too small to end frame into.
    return EndResult{bytesWritten, true};
  }

  auto numBytesOrError =
      LZ4F_compressEnd(ctx_, output, outputSize, nullptr /* options */);
  if (LZ4F_isError(numBytesOrError)) {
    lz4Error("LZ4 end failed: ", numBytesOrError);
  }
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
  VELOX_DCHECK_LE(bytesWritten, outputLength);
  return EndResult{bytesWritten, false};
}

void LZ4Compressor::compressBegin(
    uint8_t* output,
    size_t& outputLen,
    uint64_t& bytesWritten) {
  auto numBytesOrError = LZ4F_compressBegin(ctx_, output, outputLen, &prefs_);
  if (LZ4F_isError(numBytesOrError)) {
    lz4Error("LZ4 compress begin failed: ", numBytesOrError);
  }
  firstTime_ = false;
  output += numBytesOrError;
  outputLen -= numBytesOrError;
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
}

void common::LZ4Decompressor::init() {
  finished_ = false;
  auto ret = LZ4F_createDecompressionContext(&ctx_, LZ4F_VERSION);
  if (LZ4F_isError(ret)) {
    lz4Error("LZ4 init failed: ", ret);
  }
}

void LZ4Decompressor::reset() {
#if defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER >= 10800
  // LZ4F_resetDecompressionContext appeared in 1.8.0
  VELOX_CHECK_NOT_NULL(ctx_);
  LZ4F_resetDecompressionContext(ctx_);
  finished_ = false;
#else
  if (ctx_ != nullptr) {
    LZ4F_freeDecompressionContext(ctx_);
  }
  init();
#endif
}

StreamingDecompressor::DecompressResult LZ4Decompressor::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  auto inputSize = static_cast<size_t>(inputLength);
  auto outputSize = static_cast<size_t>(outputLength);

  auto ret = LZ4F_decompress(
      ctx_, output, &outputSize, input, &inputSize, nullptr /* options */);
  if (LZ4F_isError(ret)) {
    lz4Error("LZ4 decompress failed: ", ret);
  }
  finished_ = (ret == 0);
  return DecompressResult{
      static_cast<uint64_t>(inputSize),
      static_cast<uint64_t>(outputSize),
      (inputSize == 0 && outputSize == 0)};
}

bool LZ4Decompressor::isFinished() {
  return finished_;
}

Lz4CodecBase::Lz4CodecBase(int32_t compressionLevel)
    : compressionLevel_(
          compressionLevel == kUseDefaultCompressionLevel
              ? kLz4DefaultCompressionLevel
              : compressionLevel) {}

int32_t Lz4CodecBase::minimumCompressionLevel() const {
  return kLz4MinCompressionLevel;
}

int32_t Lz4CodecBase::maximumCompressionLevel() const {
#if (defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER < 10800)
  return 12;
#else
  return LZ4F_compressionLevel_max();
#endif
}

int32_t Lz4CodecBase::defaultCompressionLevel() const {
  return kLz4DefaultCompressionLevel;
}

int32_t Lz4CodecBase::compressionLevel() const {
  return compressionLevel_;
}

CompressionKind Lz4CodecBase::compressionKind() const {
  return CompressionKind::CompressionKind_LZ4;
}

Lz4FrameCodec::Lz4FrameCodec(int32_t compressionLevel)
    : Lz4CodecBase(compressionLevel),
      prefs_(defaultPreferences(compressionLevel_)) {}

uint64_t Lz4FrameCodec::maxCompressedLength(uint64_t inputLen) {
  return static_cast<int64_t>(
      LZ4F_compressFrameBound(static_cast<size_t>(inputLen), &prefs_));
}

uint64_t Lz4FrameCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  auto ret = LZ4F_compressFrame(
      output,
      static_cast<size_t>(outputLength),
      input,
      static_cast<size_t>(inputLength),
      &prefs_);
  if (LZ4F_isError(ret)) {
    lz4Error("Lz4 compression failure: ", ret);
  }
  return static_cast<uint64_t>(ret);
}

uint64_t Lz4FrameCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  auto decompressor = makeStreamingDecompressor();

  uint64_t bytesWritten = 0;
  while (!decompressor->isFinished() && inputLength != 0) {
    auto result =
        decompressor->decompress(input, inputLength, output, outputLength);
    input += result.bytesRead;
    inputLength -= result.bytesRead;
    output += result.bytesWritten;
    outputLength -= result.bytesWritten;
    bytesWritten += result.bytesWritten;
    if (result.outputTooSmall) {
      VELOX_FAIL("Lz4 decompression buffer too small.");
    }
  }
  if (!decompressor->isFinished()) {
    VELOX_FAIL("Lz4 compressed input contains less than one frame.");
  }
  if (inputLength != 0) {
    VELOX_FAIL("Lz4 compressed input contains more than one frame.");
  }
  return bytesWritten;
}

std::shared_ptr<StreamingCompressor> Lz4FrameCodec::makeStreamingCompressor() {
  auto ptr = std::make_shared<LZ4Compressor>(compressionLevel_);
  ptr->init();
  return ptr;
}

std::shared_ptr<StreamingDecompressor>
Lz4FrameCodec::makeStreamingDecompressor() {
  auto ptr = std::make_shared<LZ4Decompressor>();
  ptr->init();
  return ptr;
}

Lz4RawCodec::Lz4RawCodec(int32_t compressionLevel)
    : Lz4CodecBase(compressionLevel) {}

uint64_t Lz4RawCodec::maxCompressedLength(uint64_t inputLength) {
  return static_cast<uint64_t>(
      LZ4_compressBound(static_cast<int>(inputLength)));
}

uint64_t Lz4RawCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  auto decompressedSize = LZ4_decompress_safe(
      reinterpret_cast<const char*>(input),
      reinterpret_cast<char*>(output),
      static_cast<int>(inputLength),
      static_cast<int>(outputLength));
  if (decompressedSize < 0) {
    VELOX_FAIL("Corrupt Lz4 compressed data.");
  }
  return static_cast<uint64_t>(decompressedSize);
}

uint64_t Lz4RawCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  uint64_t compressedSize;
#ifdef LZ4HC_CLEVEL_MIN
  constexpr int kMinHcClevel = LZ4HC_CLEVEL_MIN;
#else // For older versions of the lz4 library.
  constexpr int kMinHcClevel = 3;
#endif
  if (compressionLevel_ < kMinHcClevel) {
    compressedSize = LZ4_compress_default(
        reinterpret_cast<const char*>(input),
        reinterpret_cast<char*>(output),
        static_cast<int>(inputLength),
        static_cast<int>(outputLength));
  } else {
    compressedSize = LZ4_compress_HC(
        reinterpret_cast<const char*>(input),
        reinterpret_cast<char*>(output),
        static_cast<int>(inputLength),
        static_cast<int>(outputLength),
        compressionLevel_);
  }
  if (compressedSize == 0) {
    VELOX_FAIL("Lz4 compression failure.");
  }
  return static_cast<uint64_t>(compressedSize);
}

std::shared_ptr<StreamingCompressor> Lz4RawCodec::makeStreamingCompressor() {
  VELOX_UNSUPPORTED(
      "Streaming compression unsupported with LZ4 raw format. "
      "Try using LZ4 frame format instead.");
}

std::shared_ptr<StreamingDecompressor>
Lz4RawCodec::makeStreamingDecompressor() {
  VELOX_UNSUPPORTED(
      "Streaming decompression unsupported with LZ4 raw format. "
      "Try using LZ4 frame format instead.");
}

Lz4HadoopCodec::Lz4HadoopCodec() : Lz4RawCodec(kLz4DefaultCompressionLevel) {}

uint64_t Lz4HadoopCodec::maxCompressedLength(uint64_t inputLength) {
  return kPrefixLength + Lz4RawCodec::maxCompressedLength(inputLength);
}

uint64_t Lz4HadoopCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  if (outputLength < kPrefixLength) {
    VELOX_FAIL("Output buffer too small for Lz4HadoopCodec compression.");
  }

  uint64_t compressedSize = Lz4RawCodec::compress(
      input, inputLength, output + kPrefixLength, outputLength - kPrefixLength);

  // Prepend decompressed size in bytes and compressed size in bytes
  // to be compatible with Hadoop Lz4RawCodec.
  const uint32_t decompressedLength =
      folly::Endian::big(static_cast<uint32_t>(inputLength));
  const uint32_t compressedLength =
      folly::Endian::big(static_cast<uint32_t>(compressedSize));
  folly::storeUnaligned(output, decompressedLength);
  folly::storeUnaligned(output + sizeof(uint32_t), compressedLength);

  return kPrefixLength + compressedSize;
}

uint64_t Lz4HadoopCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  uint64_t decompressedSize;
  if (tryDecompressHadoop(
          input, inputLength, output, outputLength, decompressedSize)) {
    return decompressedSize;
  }
  // Fall back on raw LZ4 codec (for files produces by earlier versions of
  // Parquet C++).
  return Lz4RawCodec::decompress(input, inputLength, output, outputLength);
}

std::shared_ptr<StreamingCompressor> Lz4HadoopCodec::makeStreamingCompressor() {
  VELOX_UNSUPPORTED(
      "Streaming compression unsupported with LZ4 Hadoop raw format. "
      "Try using LZ4 frame format instead.");
}

std::shared_ptr<StreamingDecompressor>
Lz4HadoopCodec::makeStreamingDecompressor() {
  VELOX_UNSUPPORTED(
      "Streaming decompression unsupported with LZ4 Hadoop raw format. "
      "Try using LZ4 frame format instead.");
}

int32_t Lz4HadoopCodec::minimumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t Lz4HadoopCodec::maximumCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

int32_t Lz4HadoopCodec::defaultCompressionLevel() const {
  return kUseDefaultCompressionLevel;
}

uint64_t Lz4HadoopCodec::decompressInternal(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  return Lz4RawCodec::decompress(input, inputLength, output, outputLength);
}

std::unique_ptr<Codec> makeLz4FrameCodec(int32_t compressionLevel) {
  return std::make_unique<Lz4FrameCodec>(compressionLevel);
}

std::unique_ptr<Codec> makeLz4RawCodec(int32_t compressionLevel) {
  return std::make_unique<Lz4RawCodec>(compressionLevel);
}

std::unique_ptr<Codec> makeLz4HadoopCodec() {
  return std::make_unique<Lz4HadoopCodec>();
}
} // namespace facebook::velox::common
