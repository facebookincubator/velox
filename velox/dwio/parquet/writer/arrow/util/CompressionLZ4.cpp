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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/util/CompressionInternal.h"

#include <memory>

#include <lz4.h>
#include <lz4frame.h>
#include <lz4hc.h>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"

#include "velox/common/base/Exceptions.h"

using std::size_t;

namespace facebook::velox::parquet::arrow::util::internal {
namespace {

constexpr int kLz4MinCompressionLevel = 1;

static Status lZ4Error(LZ4F_errorCode_t ret, const char* prefixMsg) {
  return Status::IOError(prefixMsg, LZ4F_getErrorName(ret));
}

static LZ4F_preferences_t defaultPreferences() {
  LZ4F_preferences_t prefs;
  memset(&prefs, 0, sizeof(prefs));
  return prefs;
}

static LZ4F_preferences_t preferencesWithCompressionLevel(
    int compressionLevel) {
  LZ4F_preferences_t prefs = defaultPreferences();
  prefs.compressionLevel = compressionLevel;
  return prefs;
}

// ----------------------------------------------------------------------.
// Lz4 frame Decompressor implementation.

class LZ4Decompressor : public Decompressor {
 public:
  LZ4Decompressor() {}

  ~LZ4Decompressor() override {
    if (ctx_ != nullptr) {
      ARROW_UNUSED(LZ4F_freeDecompressionContext(ctx_));
    }
  }

  Status init() {
    LZ4F_errorCode_t ret;
    finished_ = false;

    ret = LZ4F_createDecompressionContext(&ctx_, LZ4F_VERSION);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 init failed: ");
    } else {
      return Status::OK();
    }
  }

  Status reset() override {
#if defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER >= 10800
    // LZ4F_resetDecompressionContext appeared in 1.8.0.
    VELOX_DCHECK_NOT_NULL(ctx_);
    LZ4F_resetDecompressionContext(ctx_);
    finished_ = false;
    return Status::OK();
#else
    if (ctx_ != nullptr) {
      ARROW_UNUSED(LZ4F_freeDecompressionContext(ctx_));
    }
    return init();
#endif
  }

  Result<DecompressResult> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    auto src = input;
    auto dst = output;
    auto srcSize = static_cast<size_t>(inputLen);
    auto dstCapacity = static_cast<size_t>(outputLen);
    size_t ret;

    ret = LZ4F_decompress(
        ctx_, dst, &dstCapacity, src, &srcSize, nullptr /* options */);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 decompress failed: ");
    }
    finished_ = (ret == 0);
    return DecompressResult{
        static_cast<int64_t>(srcSize),
        static_cast<int64_t>(dstCapacity),
        (srcSize == 0 && dstCapacity == 0)};
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  LZ4F_decompressionContext_t ctx_ = nullptr;
  bool finished_;
};

// ----------------------------------------------------------------------.
// Lz4 frame Compressor implementation.

class LZ4Compressor : public Compressor {
 public:
  explicit LZ4Compressor(int compressionLevel)
      : compressionLevel_(compressionLevel) {}

  ~LZ4Compressor() override {
    if (ctx_ != nullptr) {
      ARROW_UNUSED(LZ4F_freeCompressionContext(ctx_));
    }
  }

  Status init() {
    LZ4F_errorCode_t ret;
    prefs_ = preferencesWithCompressionLevel(compressionLevel_);
    firstTime_ = true;

    ret = LZ4F_createCompressionContext(&ctx_, LZ4F_VERSION);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 init failed: ");
    } else {
      return Status::OK();
    }
  }

#define BEGIN_COMPRESS(dst, dstCapacity, outputTooSmall)       \
  if (firstTime_) {                                            \
    if (dstCapacity < LZ4F_HEADER_SIZE_MAX) {                  \
      /* Output too small to write LZ4F header */              \
      return (outputTooSmall);                                 \
    }                                                          \
    ret = LZ4F_compressBegin(ctx_, dst, dstCapacity, &prefs_); \
    if (LZ4F_isError(ret)) {                                   \
      return lZ4Error(ret, "LZ4 compress begin failed: ");     \
    }                                                          \
    firstTime_ = false;                                        \
    dst += ret;                                                \
    dstCapacity -= ret;                                        \
    bytesWritten += static_cast<int64_t>(ret);                 \
  }

  Result<CompressResult> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    auto src = input;
    auto dst = output;
    auto srcSize = static_cast<size_t>(inputLen);
    auto dstCapacity = static_cast<size_t>(outputLen);
    size_t ret;
    int64_t bytesWritten = 0;

    BEGIN_COMPRESS(dst, dstCapacity, (CompressResult{0, 0}));

    if (dstCapacity < LZ4F_compressBound(srcSize, &prefs_)) {
      // Output too small to compress into.
      return CompressResult{0, bytesWritten};
    }
    ret = LZ4F_compressUpdate(
        ctx_, dst, dstCapacity, src, srcSize, nullptr /* options */);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 compress update failed: ");
    }
    bytesWritten += static_cast<int64_t>(ret);
    VELOX_DCHECK_LE(bytesWritten, outputLen);
    return CompressResult{inputLen, bytesWritten};
  }

  Result<FlushResult> flush(int64_t outputLen, uint8_t* output) override {
    auto dst = output;
    auto dstCapacity = static_cast<size_t>(outputLen);
    size_t ret;
    int64_t bytesWritten = 0;

    BEGIN_COMPRESS(dst, dstCapacity, (FlushResult{0, true}));

    if (dstCapacity < LZ4F_compressBound(0, &prefs_)) {
      // Output too small to flush into.
      return FlushResult{bytesWritten, true};
    }

    ret = LZ4F_flush(ctx_, dst, dstCapacity, nullptr /* options */);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 flush failed: ");
    }
    bytesWritten += static_cast<int64_t>(ret);
    VELOX_DCHECK_LE(bytesWritten, outputLen);
    return FlushResult{bytesWritten, false};
  }

  Result<EndResult> end(int64_t outputLen, uint8_t* output) override {
    auto dst = output;
    auto dstCapacity = static_cast<size_t>(outputLen);
    size_t ret;
    int64_t bytesWritten = 0;

    BEGIN_COMPRESS(dst, dstCapacity, (EndResult{0, true}));

    if (dstCapacity < LZ4F_compressBound(0, &prefs_)) {
      // Output too small to end frame into.
      return EndResult{bytesWritten, true};
    }

    ret = LZ4F_compressEnd(ctx_, dst, dstCapacity, nullptr /* options */);
    if (LZ4F_isError(ret)) {
      return lZ4Error(ret, "LZ4 end failed: ");
    }
    bytesWritten += static_cast<int64_t>(ret);
    VELOX_DCHECK_LE(bytesWritten, outputLen);
    return EndResult{bytesWritten, false};
  }

#undef BEGIN_COMPRESS

 protected:
  int compressionLevel_;
  LZ4F_compressionContext_t ctx_ = nullptr;
  LZ4F_preferences_t prefs_;
  bool firstTime_;
};

// ----------------------------------------------------------------------.
// Lz4 frame codec implementation.

class Lz4FrameCodec : public Codec {
 public:
  explicit Lz4FrameCodec(int compressionLevel)
      : compressionLevel_(
            compressionLevel == kUseDefaultCompressionLevel
                ? kLz4DefaultCompressionLevel
                : compressionLevel),
        prefs_(preferencesWithCompressionLevel(compressionLevel_)) {}

  int64_t maxCompressedLen(
      int64_t inputLen,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    return static_cast<int64_t>(
        LZ4F_compressFrameBound(static_cast<size_t>(inputLen), &prefs_));
  }

  Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    auto outputLen = LZ4F_compressFrame(
        outputBuffer,
        static_cast<size_t>(outputBufferLen),
        input,
        static_cast<size_t>(inputLen),
        &prefs_);
    if (LZ4F_isError(outputLen)) {
      return lZ4Error(outputLen, "Lz4 compression failure: ");
    }
    return static_cast<int64_t>(outputLen);
  }

  Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    ARROW_ASSIGN_OR_RAISE(auto decomp, makeDecompressor());

    int64_t totalBytesWritten = 0;
    while (!decomp->isFinished() && inputLen != 0) {
      ARROW_ASSIGN_OR_RAISE(
          auto res,
          decomp->decompress(inputLen, input, outputBufferLen, outputBuffer));
      input += res.bytesRead;
      inputLen -= res.bytesRead;
      outputBuffer += res.bytesWritten;
      outputBufferLen -= res.bytesWritten;
      totalBytesWritten += res.bytesWritten;
      if (res.needMoreOutput) {
        return Status::IOError("Lz4 decompression buffer too small");
      }
    }
    if (!decomp->isFinished()) {
      return Status::IOError(
          "Lz4 compressed input contains less than one frame");
    }
    if (inputLen != 0) {
      return Status::IOError(
          "Lz4 compressed input contains more than one frame");
    }
    return totalBytesWritten;
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    auto ptr = std::make_shared<LZ4Compressor>(compressionLevel_);
    RETURN_NOT_OK(ptr->init());
    return ptr;
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    auto ptr = std::make_shared<LZ4Decompressor>();
    RETURN_NOT_OK(ptr->init());
    return ptr;
  }

  Compression::type compressionType() const override {
    return Compression::LZ4_FRAME;
  }
  int minimumCompressionLevel() const override {
    return kLz4MinCompressionLevel;
  }
#if (defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER < 10800)
  int maximumCompressionLevel() const override {
    return 12;
  }
#else
  int maximumCompressionLevel() const override {
    return LZ4F_compressionLevel_max();
  }
#endif
  int defaultCompressionLevel() const override {
    return kLz4DefaultCompressionLevel;
  }

  int compressionLevel() const override {
    return compressionLevel_;
  }

 protected:
  const int compressionLevel_;
  const LZ4F_preferences_t prefs_;
};

// ----------------------------------------------------------------------.
// Lz4 "raw" codec implementation.

class Lz4Codec : public Codec {
 public:
  explicit Lz4Codec(int compressionLevel)
      : compressionLevel_(
            compressionLevel == kUseDefaultCompressionLevel
                ? kLz4DefaultCompressionLevel
                : compressionLevel) {}

  Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    int64_t decompressedSize = LZ4_decompress_safe(
        reinterpret_cast<const char*>(input),
        reinterpret_cast<char*>(outputBuffer),
        static_cast<int>(inputLen),
        static_cast<int>(outputBufferLen));
    if (decompressedSize < 0) {
      return Status::IOError("Corrupt Lz4 compressed data.");
    }
    return decompressedSize;
  }

  int64_t maxCompressedLen(
      int64_t inputLen,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    return LZ4_compressBound(static_cast<int>(inputLen));
  }

  Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    int64_t outputLen;
#ifdef LZ4HC_CLEVEL_MIN
    constexpr int minHcClevel = LZ4HC_CLEVEL_MIN;
#else // For older versions of the lz4 library
    constexpr int minHcClevel = 3;
#endif
    if (compressionLevel_ < minHcClevel) {
      outputLen = LZ4_compress_default(
          reinterpret_cast<const char*>(input),
          reinterpret_cast<char*>(outputBuffer),
          static_cast<int>(inputLen),
          static_cast<int>(outputBufferLen));
    } else {
      outputLen = LZ4_compress_HC(
          reinterpret_cast<const char*>(input),
          reinterpret_cast<char*>(outputBuffer),
          static_cast<int>(inputLen),
          static_cast<int>(outputBufferLen),
          compressionLevel_);
    }
    if (outputLen == 0) {
      return Status::IOError("Lz4 compression failure.");
    }
    return outputLen;
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    return Status::NotImplemented(
        "Streaming compression unsupported with LZ4 raw format. "
        "Try using LZ4 frame format instead.");
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    return Status::NotImplemented(
        "Streaming decompression unsupported with LZ4 raw format. "
        "Try using LZ4 frame format instead.");
  }

  Compression::type compressionType() const override {
    return Compression::LZ4;
  }
  int minimumCompressionLevel() const override {
    return kLz4MinCompressionLevel;
  }
#if (defined(LZ4_VERSION_NUMBER) && LZ4_VERSION_NUMBER < 10800)
  int maximumCompressionLevel() const override {
    return 12;
  }
#else
  int maximumCompressionLevel() const override {
    return LZ4F_compressionLevel_max();
  }
#endif
  int defaultCompressionLevel() const override {
    return kLz4DefaultCompressionLevel;
  }

 protected:
  int compressionLevel_;
};

// ----------------------------------------------------------------------.
// Lz4 Hadoop "raw" codec implementation.

class Lz4HadoopCodec : public Lz4Codec {
 public:
  Lz4HadoopCodec() : Lz4Codec(kUseDefaultCompressionLevel) {}

  Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    const int64_t decompressedSize =
        tryDecompressHadoop(inputLen, input, outputBufferLen, outputBuffer);
    if (decompressedSize != kNotHadoop) {
      return decompressedSize;
    }
    // Fall back on raw LZ4 codec (for files produces by earlier versions of.
    // Parquet C++)
    return Lz4Codec::decompress(inputLen, input, outputBufferLen, outputBuffer);
  }

  int64_t maxCompressedLen(
      int64_t inputLen,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    return kPrefixLength + Lz4Codec::maxCompressedLen(inputLen, nullptr);
  }

  Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    if (outputBufferLen < kPrefixLength) {
      return Status::Invalid(
          "Output buffer too small for Lz4HadoopCodec compression");
    }

    ARROW_ASSIGN_OR_RAISE(
        int64_t outputLen,
        Lz4Codec::compress(
            inputLen,
            input,
            outputBufferLen - kPrefixLength,
            outputBuffer + kPrefixLength));

    // Prepend decompressed size in bytes and compressed size in bytes.
    // To be compatible with Hadoop Lz4Codec.
    const uint32_t decompressedSize =
        bit_util::ToBigEndian(static_cast<uint32_t>(inputLen));
    const uint32_t compressedSize =
        bit_util::ToBigEndian(static_cast<uint32_t>(outputLen));
    ::arrow::util::SafeStore(outputBuffer, decompressedSize);
    ::arrow::util::SafeStore(outputBuffer + sizeof(uint32_t), compressedSize);

    return kPrefixLength + outputLen;
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    return Status::NotImplemented(
        "Streaming compression unsupported with LZ4 Hadoop raw format. "
        "Try using LZ4 frame format instead.");
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    return Status::NotImplemented(
        "Streaming decompression unsupported with LZ4 Hadoop raw format. "
        "Try using LZ4 frame format instead.");
  }

  Compression::type compressionType() const override {
    return Compression::LZ4_HADOOP;
  }

 protected:
  // Offset starting at which page data can be read/written.
  static const int64_t kPrefixLength = sizeof(uint32_t) * 2;

  static const int64_t kNotHadoop = -1;

  int64_t tryDecompressHadoop(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) {
    // Parquet files written with the Hadoop Lz4Codec use their own framing.
    // The input buffer can contain an arbitrary number of "frames", each.
    // With the following structure:
    // - Bytes 0..3: big-endian uint32_t representing the frame decompressed.
    // Size.
    // - Bytes 4..7: big-endian uint32_t representing the frame compressed size.
    // - Bytes 8...: frame compressed data.
    //
    // The Hadoop Lz4Codec source code can be found here:
    // https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-nativetask/src/main/native/src/codec/Lz4Codec.cc
    int64_t totalDecompressedSize = 0;

    while (inputLen >= kPrefixLength) {
      const uint32_t expectedDecompressedSize =
          bit_util::FromBigEndian(::arrow::util::SafeLoadAs<uint32_t>(input));
      const uint32_t expectedCompressedSize = bit_util::FromBigEndian(
          ::arrow::util::SafeLoadAs<uint32_t>(input + sizeof(uint32_t)));
      input += kPrefixLength;
      inputLen -= kPrefixLength;

      if (inputLen < expectedCompressedSize) {
        // Not enough bytes for Hadoop "frame".
        return kNotHadoop;
      }
      if (outputBufferLen < expectedDecompressedSize) {
        // Not enough bytes to hold advertised output => probably not Hadoop.
        return kNotHadoop;
      }
      // Try decompressing and compare with expected decompressed length.
      auto maybeDecompressedSize = Lz4Codec::decompress(
          expectedCompressedSize, input, outputBufferLen, outputBuffer);
      if (!maybeDecompressedSize.ok() ||
          *maybeDecompressedSize != expectedDecompressedSize) {
        return kNotHadoop;
      }
      input += expectedCompressedSize;
      inputLen -= expectedCompressedSize;
      outputBuffer += expectedDecompressedSize;
      outputBufferLen -= expectedDecompressedSize;
      totalDecompressedSize += expectedDecompressedSize;
    }

    if (inputLen == 0) {
      return totalDecompressedSize;
    } else {
      return kNotHadoop;
    }
  }

  int minimumCompressionLevel() const override {
    return kUseDefaultCompressionLevel;
  }
  int maximumCompressionLevel() const override {
    return kUseDefaultCompressionLevel;
  }
  int defaultCompressionLevel() const override {
    return kUseDefaultCompressionLevel;
  }
};

} // namespace

std::unique_ptr<Codec> makeLz4FrameCodec(int compressionLevel) {
  return std::make_unique<Lz4FrameCodec>(compressionLevel);
}

std::unique_ptr<Codec> makeLz4HadoopRawCodec() {
  return std::make_unique<Lz4HadoopCodec>();
}

std::unique_ptr<Codec> makeLz4RawCodec(int compressionLevel) {
  return std::make_unique<Lz4Codec>(compressionLevel);
}

} // namespace facebook::velox::parquet::arrow::util::internal
