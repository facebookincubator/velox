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

#include "velox/common/compression/v2/Lz4Compression.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {

namespace {

void lz4Error(LZ4F_errorCode_t errorCode, const char* prefixMessage) {
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

class LZ4Compressor : public Compressor {
 public:
  explicit LZ4Compressor(int32_t compressionLevel);

  ~LZ4Compressor() override;

  void init();

  CompressResult compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  FlushResult flush(uint64_t outputLength, uint8_t* output) override;

  EndResult end(uint64_t outputLength, uint8_t* output) override;

 protected:
  void
  compressBegin(uint8_t* output, size_t& outputLen, uint64_t& bytesWritten);

  int compressionLevel_;
  LZ4F_compressionContext_t ctx_{nullptr};
  LZ4F_preferences_t prefs_;
  bool firstTime_;
};

class LZ4Decompressor : public Decompressor {
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
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  bool isFinished() override;

 protected:
  LZ4F_decompressionContext_t ctx_ = nullptr;
  bool finished_;
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
    lz4Error(ret, "LZ4 init failed: ");
  }
}

Compressor::CompressResult LZ4Compressor::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
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
    lz4Error(numBytesOrError, "LZ4 compress update failed: ");
  }
  bytesWritten += static_cast<int64_t>(numBytesOrError);
  DCHECK_LE(bytesWritten, outputSize);
  return CompressResult{inputLength, bytesWritten, false};
}

Compressor::FlushResult LZ4Compressor::flush(
    uint64_t outputLength,
    uint8_t* output) {
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
    lz4Error(numBytesOrError, "LZ4 flush failed: ");
  }
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
  DCHECK_LE(bytesWritten, outputLength);
  return FlushResult{bytesWritten, false};
}

Compressor::EndResult LZ4Compressor::end(
    uint64_t outputLength,
    uint8_t* output) {
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
    lz4Error(numBytesOrError, "LZ4 end failed: ");
  }
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
  DCHECK_LE(bytesWritten, outputLength);
  return EndResult{bytesWritten, false};
}

void LZ4Compressor::compressBegin(
    uint8_t* output,
    size_t& outputLen,
    uint64_t& bytesWritten) {
  auto numBytesOrError = LZ4F_compressBegin(ctx_, output, outputLen, &prefs_);
  if (LZ4F_isError(numBytesOrError)) {
    lz4Error(numBytesOrError, "LZ4 compress begin failed: ");
  }
  firstTime_ = false;
  output += numBytesOrError;
  outputLen -= numBytesOrError;
  bytesWritten += static_cast<uint64_t>(numBytesOrError);
}

void common::LZ4Decompressor::init() {
  LZ4F_errorCode_t ret;
  finished_ = false;

  ret = LZ4F_createDecompressionContext(&ctx_, LZ4F_VERSION);
  if (LZ4F_isError(ret)) {
    lz4Error(ret, "LZ4 init failed: ");
  }
}

void LZ4Decompressor::reset() {
#if defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER >= 10800
  // LZ4F_resetDecompressionContext appeared in 1.8.0
  DCHECK_NE(ctx_, nullptr);
  LZ4F_resetDecompressionContext(ctx_);
  finished_ = false;
#else
  if (ctx_ != nullptr) {
    LZ4F_freeDecompressionContext(ctx_);
  }
  init();
#endif
}

Decompressor::DecompressResult LZ4Decompressor::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  auto inputSize = static_cast<size_t>(inputLength);
  auto outputSize = static_cast<size_t>(outputLength);

  auto ret = LZ4F_decompress(
      ctx_, output, &outputSize, input, &inputSize, nullptr /* options */);
  if (LZ4F_isError(ret)) {
    lz4Error(ret, "LZ4 decompress failed: ");
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

Lz4FrameCodec::Lz4FrameCodec(int32_t compressionLevel)
    : Lz4CodecBase(compressionLevel),
      prefs_(defaultPreferences(compressionLevel_)) {}

uint64_t Lz4FrameCodec::maxCompressedLength(uint64_t inputLen) {
  return static_cast<int64_t>(
      LZ4F_compressFrameBound(static_cast<size_t>(inputLen), &prefs_));
}

uint64_t Lz4FrameCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  auto ret = LZ4F_compressFrame(
      output,
      static_cast<size_t>(outputLength),
      input,
      static_cast<size_t>(inputLength),
      &prefs_);
  if (LZ4F_isError(ret)) {
    lz4Error(ret, "Lz4 compression failure: ");
  }
  return static_cast<uint64_t>(ret);
}

uint64_t Lz4FrameCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  auto decompressor = makeDecompressor();

  uint64_t bytesWritten = 0;
  while (!decompressor->isFinished() && inputLength != 0) {
    auto result =
        decompressor->decompress(inputLength, input, outputLength, output);
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

std::shared_ptr<Compressor> Lz4FrameCodec::makeCompressor() {
  auto ptr = std::make_shared<LZ4Compressor>(compressionLevel_);
  ptr->init();
  return ptr;
}

std::shared_ptr<Decompressor> Lz4FrameCodec::makeDecompressor() {
  auto ptr = std::make_shared<LZ4Decompressor>();
  ptr->init();
  return ptr;
}

CompressionKind Lz4FrameCodec::compressionKind() const {
  return CompressionKind::CompressionKind_LZ4;
}

Lz4RawCodec::Lz4RawCodec(int32_t compressionLevel)
    : Lz4CodecBase(compressionLevel) {}

uint64_t Lz4RawCodec::maxCompressedLength(uint64_t inputLength) {
  return static_cast<uint64_t>(
      LZ4_compressBound(static_cast<int>(inputLength)));
}

uint64_t Lz4RawCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
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
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
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

std::shared_ptr<Compressor> Lz4RawCodec::makeCompressor() {
  VELOX_UNSUPPORTED(
      "Streaming compression unsupported with LZ4 raw format. "
      "Try using LZ4 frame format instead.");
}

std::shared_ptr<Decompressor> Lz4RawCodec::makeDecompressor() {
  VELOX_UNSUPPORTED(
      "Streaming decompression unsupported with LZ4 raw format. "
      "Try using LZ4 frame format instead.");
}

CompressionKind Lz4RawCodec::compressionKind() const {
  return CompressionKind::CompressionKind_LZ4RAW;
}

Lz4HadoopCodec::Lz4HadoopCodec() : Lz4RawCodec(kUseDefaultCompressionLevel) {}

uint64_t Lz4HadoopCodec::maxCompressedLength(uint64_t inputLength) {
  return kPrefixLength + Lz4RawCodec::maxCompressedLength(inputLength);
}

uint64_t Lz4HadoopCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  if (outputLength < kPrefixLength) {
    VELOX_FAIL("Output buffer too small for Lz4HadoopCodec compression.");
  }

  uint64_t compressedSize = Lz4RawCodec::compress(
      inputLength, input, outputLength - kPrefixLength, output + kPrefixLength);

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
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  uint64_t decompressedSize;
  if (tryDecompressHadoop(
          inputLength, input, outputLength, output, decompressedSize)) {
    return decompressedSize;
  }
  // Fall back on raw LZ4 codec (for files produces by earlier versions of
  // Parquet C++).
  return Lz4RawCodec::decompress(inputLength, input, outputLength, output);
}

std::shared_ptr<Compressor> Lz4HadoopCodec::makeCompressor() {
  VELOX_UNSUPPORTED(
      "Streaming compression unsupported with LZ4 Hadoop raw format. "
      "Try using LZ4 frame format instead.");
}

std::shared_ptr<Decompressor> Lz4HadoopCodec::makeDecompressor() {
  VELOX_UNSUPPORTED(
      "Streaming decompression unsupported with LZ4 Hadoop raw format. "
      "Try using LZ4 frame format instead.");
}

CompressionKind Lz4HadoopCodec::compressionKind() const {
  return CompressionKind::CompressionKind_LZ4HADOOP;
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
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  return Lz4RawCodec::decompress(inputLength, input, outputLength, output);
}

std::unique_ptr<Codec> makeLz4FrameCodec(int32_t compressionLevel) {
  return std::make_unique<Lz4FrameCodec>(compressionLevel);
}

std::unique_ptr<Codec> makeLz4RawCodec(int32_t compressionLevel) {
  return std::make_unique<Lz4RawCodec>(compressionLevel);
}

std::unique_ptr<Codec> makeLz4HadoopRawCodec() {
  return std::make_unique<Lz4HadoopCodec>();
}
} // namespace facebook::velox::common