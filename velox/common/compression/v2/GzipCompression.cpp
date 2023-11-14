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

#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/v2/GzipCompression.h"

namespace facebook::velox::common {

namespace {

constexpr uint64_t kGzipBufferLimit =
    static_cast<uint64_t>(std::numeric_limits<uInt>::max());

// Determine if this is zlib or gzip from header.
int32_t getCompressionWindowBits(GzipFormat format, int32_t windowBits) {
  switch (format) {
    case GzipFormat::kDeflate:
      windowBits = -windowBits;
      break;
    case GzipFormat::kGzip:
      windowBits += 16;
      break;
    case GzipFormat::kZlib:
      break;
  }
  return windowBits;
}

int32_t getDecompressionWindowBits(GzipFormat format, int32_t windowBits) {
  if (format == GzipFormat::kDeflate) {
    return -windowBits;
  } else {
    // If not deflate, autodetect format from header.
    return windowBits | 32;
  }
}

void zlibError(const char* prefix, const char* detail) {
  std::string msg(detail);
  VELOX_FAIL(prefix, msg.empty() ? msg : "(unknown error)");
}
} // namespace

class GZipDecompressor : public Decompressor {
 public:
  explicit GZipDecompressor(GzipFormat format, int32_t windowBits);

  ~GZipDecompressor() override;

  void init();

  void reset() override;

  DecompressResult decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  bool isFinished() {
    return finished_;
  }

 protected:
  z_stream stream_{};
  GzipFormat format_;
  int32_t windowBits_;
  bool initialized_{false};
  bool finished_{false};
};

class GzipCompressor : public Compressor {
 public:
  explicit GzipCompressor(int32_t compressionLevel);

  ~GzipCompressor() override;

  void init(GzipFormat format, int32_t windowBits);

  CompressResult compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  FlushResult flush(uint64_t outputLength, uint8_t* output) override;

  EndResult end(uint64_t outputLength, uint8_t* output) override;

 protected:
  z_stream stream_{};
  int32_t compressionLevel_;
  bool initialized_{false};
};

GZipDecompressor::GZipDecompressor(GzipFormat format, int32_t windowBits)
    : format_(format), windowBits_(windowBits) {}

GZipDecompressor::~GZipDecompressor() {
  if (initialized_) {
    inflateEnd(&stream_);
  }
}

void GZipDecompressor::init() {
  VELOX_CHECK(!initialized_, "Called on initialized stream.");
  memset(&stream_, 0, sizeof(stream_));
  finished_ = false;

  auto windowBits = getDecompressionWindowBits(format_, windowBits_);
  auto ret = inflateInit2(&stream_, windowBits);
  if (ret != Z_OK) {
    zlibError("zlib inflateInit failed: ", stream_.msg);
  }
  initialized_ = true;
}

void GZipDecompressor::reset() {
  DCHECK(initialized_);
  finished_ = false;
  auto ret = inflateReset(&stream_);
  if (ret != Z_OK) {
    zlibError("zlib inflateReset failed: ", stream_.msg);
  }
}

Decompressor::DecompressResult GZipDecompressor::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_CHECK(initialized_, "Called on non-initialized stream.");

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(std::min(inputLength, kGzipBufferLimit));
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = inflate(&stream_, Z_SYNC_FLUSH);
  if (ret == Z_DATA_ERROR || ret == Z_STREAM_ERROR || ret == Z_MEM_ERROR) {
    zlibError("Zlib inflate failed: ", stream_.msg);
  }
  if (ret == Z_NEED_DICT) {
    zlibError("Zlib inflate failed (need preset dictionary): ", stream_.msg);
  }
  if (ret == Z_BUF_ERROR) {
    // No progress was possible or output is too small.
    return DecompressResult{0, 0, true};
  }
  VELOX_CHECK(
      ret == Z_OK || ret == Z_STREAM_END,
      "Invalid return code from zlib: {}",
      ret);
  finished_ = (ret == Z_STREAM_END);
  return DecompressResult{
      inputLength - stream_.avail_in, outputLength - stream_.avail_out, false};
}

GzipCompressor::GzipCompressor(int32_t compressionLevel)
    : compressionLevel_(compressionLevel) {}

GzipCompressor::~GzipCompressor() {
  if (initialized_) {
    deflateEnd(&stream_);
  }
}

void GzipCompressor::init(GzipFormat format, int32_t windowBits) {
  VELOX_CHECK(!initialized_, "Called on initialized stream.");
  memset(&stream_, 0, sizeof(stream_));

  // Initialize to run specified format
  int32_t windowBitsForFormat = getCompressionWindowBits(format, windowBits);
  auto ret = deflateInit2(
      &stream_,
      Z_DEFAULT_COMPRESSION,
      Z_DEFLATED,
      windowBitsForFormat,
      compressionLevel_,
      Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) {
    zlibError("Zlib deflateInit failed: ", stream_.msg);
  }
  initialized_ = true;
}

Compressor::CompressResult GzipCompressor::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_CHECK(initialized_, "Called on non-initialized stream.");

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(std::min(inputLength, kGzipBufferLimit));
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = deflate(&stream_, Z_NO_FLUSH);
  if (ret == Z_STREAM_ERROR) {
    zlibError("Zlib compress failed: ", stream_.msg);
  }
  if (ret == Z_OK) {
    // Some progress has been made.
    // If deflate returns Z_OK and with zero avail_out, it must be called again
    // after making room in the output buffer because there might be more output
    // pending.
    return CompressResult{
        inputLength - stream_.avail_in,
        outputLength - stream_.avail_out,
        stream_.avail_out == 0};
  }
  // No progress was possible, need to increase output buffer size.
  VELOX_CHECK_EQ(ret, Z_BUF_ERROR, "Invalid return code from zlib: {}", ret);
  return CompressResult{0, 0, true};
}

Compressor::FlushResult GzipCompressor::flush(
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_CHECK(initialized_, "Called on non-initialized stream.");

  static constexpr auto kInputLimit =
      static_cast<uint64_t>(std::numeric_limits<uInt>::max());

  stream_.avail_in = 0;
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(std::min(outputLength, kInputLimit));

  auto ret = deflate(&stream_, Z_SYNC_FLUSH);
  if (ret == Z_STREAM_ERROR) {
    zlibError("Zlib flush failed: ", stream_.msg);
  }
  uint64_t bytesWritten;
  if (ret == Z_OK) {
    bytesWritten = outputLength - stream_.avail_out;
  } else {
    VELOX_CHECK_EQ(ret, Z_BUF_ERROR, "Invalid return code from zlib: {}", ret);
    bytesWritten = 0;
  }
  // "If deflate returns with avail_out == 0, this function must be called
  //  again with the same value of the flush parameter and more output space
  //  (updated avail_out), until the flush is complete (deflate returns
  //  with non-zero avail_out)."
  // "Note that Z_BUF_ERROR is not fatal, and deflate() can be called again
  //  with more input and more output space to continue compressing."
  return FlushResult{bytesWritten, stream_.avail_out == 0};
}

Compressor::EndResult GzipCompressor::end(
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_CHECK(initialized_, "Called on non-initialized stream");

  stream_.avail_in = 0;
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = deflate(&stream_, Z_FINISH);
  if (ret == Z_STREAM_ERROR) {
    zlibError("Zlib flush failed: ", stream_.msg);
  }
  uint64_t bytesWritten = outputLength - stream_.avail_out;
  if (ret == Z_STREAM_END) {
    // Flush complete, we can now end the stream
    initialized_ = false;
    ret = deflateEnd(&stream_);
    if (ret == Z_OK) {
      return EndResult{bytesWritten, false};
    }
    zlibError("Zlib end failed: ", stream_.msg);
  }
  // Not everything could be flushed, need to increase output buffer size.
  return EndResult{bytesWritten, true};
}

GzipCodec::GzipCodec(
    int32_t compressionLevel,
    GzipFormat format,
    int32_t windowBits)
    : format_(format), windowBits_(windowBits) {
  compressionLevel_ = compressionLevel == kUseDefaultCompressionLevel
      ? kGzipDefaultCompressionLevel
      : compressionLevel;
}

GzipCodec::~GzipCodec() {
  endCompressor();
  endDecompressor();
}

std::shared_ptr<Compressor> GzipCodec::makeCompressor() {
  auto ptr = std::make_shared<GzipCompressor>(compressionLevel_);
  ptr->init(format_, windowBits_);
  return ptr;
}

std::shared_ptr<Decompressor> GzipCodec::makeDecompressor() {
  auto ptr = std::make_shared<GZipDecompressor>(format_, windowBits_);
  ptr->init();
  return ptr;
}

void GzipCodec::initCompressor() {
  endDecompressor();
  memset(&stream_, 0, sizeof(stream_));

  // Initialize to run specified format
  int32_t windowBits = getCompressionWindowBits(format_, windowBits_);
  auto ret = deflateInit2(
      &stream_,
      Z_DEFAULT_COMPRESSION,
      Z_DEFLATED,
      windowBits,
      compressionLevel_,
      Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) {
    zlibError("zlib deflateInit failed: ", stream_.msg);
  }
  compressorInitialized_ = true;
}

void GzipCodec::endCompressor() {
  if (compressorInitialized_) {
    (void)deflateEnd(&stream_);
  }
  compressorInitialized_ = false;
}

void GzipCodec::initDecompressor() {
  endCompressor();
  memset(&stream_, 0, sizeof(stream_));

  // Initialize to run either deflate or zlib/gzip format
  int32_t windowBits = getDecompressionWindowBits(format_, windowBits_);
  auto ret = inflateInit2(&stream_, windowBits);
  if (ret != Z_OK) {
    zlibError("zlib inflateInit failed: ", stream_.msg);
  }
  decompressorInitialized_ = true;
}

void GzipCodec::endDecompressor() {
  if (decompressorInitialized_) {
    (void)inflateEnd(&stream_);
  }
  decompressorInitialized_ = false;
}

uint64_t GzipCodec::decompress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  if (!decompressorInitialized_) {
    initDecompressor();
  }
  if (outputLength == 0) {
    // If input doesn't contain compressed data, outputLength is 0.
    // The zlib library does not allow *output to be NULL, even when
    // outputLength is 0 (inflate() will return Z_STREAM_ERROR). We
    // don't consider this an error, so bail early if no output is expected.
    // Note that we don't signal an error if the input actually contains
    // compressed data.
    return 0;
  }

  // Reset the stream for this block
  if (inflateReset(&stream_) != Z_OK) {
    zlibError("zlib inflateReset failed: ", stream_.msg);
  }

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = inflate(&stream_, Z_FINISH);
  if (ret != Z_STREAM_END) {
    if (ret == Z_OK) {
      // Z_OK (and stream.msg NOT set) indicates stream.avail_out is too
      // small.
      VELOX_FAIL("zlib inflate failed, output buffer too small");
    }
    zlibError("Zlib inflate failed: ", stream_.msg);
  }

  return stream_.total_out;
}

uint64_t GzipCodec::maxCompressedLength(uint64_t inputLength) {
  // Must be in compression mode.
  if (!compressorInitialized_) {
    initCompressor();
  }
  uint64_t maxLength = deflateBound(&stream_, static_cast<uLong>(inputLength));
  // ARROW-3514: return a more pessimistic estimate to account for bugs
  // in old zlib versions.
  return maxLength + 12;
}

uint64_t GzipCodec::compress(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  if (!compressorInitialized_) {
    initCompressor();
  }

  if (deflateReset(&stream_) != Z_OK) {
    zlibError("Zlib deflateReset failed: ", stream_.msg);
  }

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = deflate(&stream_, Z_FINISH);
  if (ret != Z_STREAM_END) {
    if (ret == Z_OK) {
      // Z_OK (and stream.msg NOT set) indicates stream.avail_out is too
      // small.
      VELOX_FAIL("zlib deflate failed, output buffer too small");
    }
    zlibError("Zlib deflate failed: ", stream_.msg);
  }

  return stream_.total_out;
}

uint64_t GzipCodec::compressPartial(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  if (!compressorInitialized_) {
    initCompressor();
  }

  if (deflateReset(&stream_) != Z_OK) {
    zlibError("Zlib deflateReset failed: ", stream_.msg);
  }

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = deflate(&stream_, Z_FINISH);
  if (ret != Z_STREAM_END && ret != Z_OK && ret != Z_BUF_ERROR) {
    zlibError("Zlib deflate failed: ", stream_.msg);
  }

  return stream_.total_out;
}

void GzipCodec::init() {
  if (windowBits_ < kGzipMinWindowBits || windowBits_ > kGzipMaxWindowBits) {
    VELOX_USER_FAIL(
        "GZip window bits should be between {} and {}",
        kGzipMinWindowBits,
        kGzipMaxWindowBits);
  }
  initCompressor();
  initDecompressor();
}

CompressionKind GzipCodec::compressionKind() const {
  if (format_ == GzipFormat::kZlib) {
    return CompressionKind_ZLIB;
  }
  return CompressionKind_GZIP;
}

int32_t GzipCodec::compressionLevel() const {
  return compressionLevel_;
}

int32_t GzipCodec::minimumCompressionLevel() const {
  return kGZipMinCompressionLevel;
}

int32_t GzipCodec::maximumCompressionLevel() const {
  return kGZipMaxCompressionLevel;
}

int32_t GzipCodec::defaultCompressionLevel() const {
  return kGzipDefaultCompressionLevel;
}

std::unique_ptr<Codec> makeGzipCodec(
    int compressionLevel,
    GzipFormat format,
    std::optional<int32_t> windowBits) {
  return std::make_unique<GzipCodec>(
      compressionLevel, format, windowBits.value_or(kGzipDefaultWindowBits));
}

std::unique_ptr<Codec> makeZlibCodec(
    int compressionLevel,
    std::optional<int32_t> windowBits) {
  return makeGzipCodec(compressionLevel, GzipFormat::kZlib, windowBits);
}
} // namespace facebook::velox::common

