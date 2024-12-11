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

#include "velox/common/compression/ZlibCompression.h"
#include "velox/common/base/Exceptions.h"

#include <zconf.h>
#include <zlib.h>

namespace facebook::velox::common {

namespace {

constexpr uint64_t kGzipBufferLimit = std::numeric_limits<uInt>::max();

// Output Gzip.
constexpr int32_t kGzipCodec = 16;

// Determine if this is libz or gzip from header.
constexpr int32_t kDetectCodec = 32;

// Compression levels.
constexpr int32_t kGzipDefaultCompressionLevel = 9;
constexpr int32_t kGZipMinCompressionLevel = 1;
constexpr int32_t kGZipMaxCompressionLevel = 9;

// Determine if this is zlib or gzip from header.
int32_t getCompressionWindowBits(ZlibFormat format, int32_t windowBits) {
  switch (format) {
    case ZlibFormat::kDeflate:
      windowBits = -windowBits;
      break;
    case ZlibFormat::kGzip:
      windowBits += kGzipCodec;
      break;
    case ZlibFormat::kZlib:
      break;
  }
  return windowBits;
}

int32_t getDecompressionWindowBits(ZlibFormat format, int32_t windowBits) {
  if (format == ZlibFormat::kDeflate) {
    return -windowBits;
  }
  // If not deflate, autodetect format from header.
  return windowBits | kDetectCodec;
}

Status zlibError(const char* prefix, const char* detail) {
  std::string msg(detail);
  return Status::Invalid(prefix, msg.empty() ? "(unknown error)" : msg);
}
} // namespace

class ZlibCodec : public Codec {
 public:
  ZlibCodec(ZlibFormat format, int32_t compressionLevel, int32_t windowBits);

  ~ZlibCodec() override;

  Expected<std::shared_ptr<StreamingCompressor>> makeStreamingCompressor()
      override;

  Expected<std::shared_ptr<StreamingDecompressor>> makeStreamingDecompressor()
      override;

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

  uint64_t maxCompressedLength(uint64_t inputLength) override;

  Status init() override;

  CompressionKind compressionKind() const override;

  int32_t compressionLevel() const override;

  int32_t minCompressionLevel() const override;

  int32_t maxCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  std::string_view name() const override;

 private:
  Status initCompressor();

  Status initDecompressor();

  void endCompressor();

  void endDecompressor();

  // Zlib is stateful and the z_stream state variable must be initialized
  // before.
  z_stream stream_{};

  // Realistically, this will always be GZIP, but we leave the option open to
  // configure.
  ZlibFormat format_;

  int32_t windowBits_;
  int32_t compressionLevel_;

  // These variables are mutually exclusive. When the codec is in "compressor"
  // state, compressorInitialized_ is true while decompressorInitialized_ is
  // false. When it's decompressing, the opposite is true.
  bool compressorInitialized_{false};
  bool decompressorInitialized_{false};
};

class GzipCompressor : public StreamingCompressor {
 public:
  explicit GzipCompressor(int32_t compressionLevel);

  ~GzipCompressor() override;

  Expected<CompressResult> compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  Expected<FlushResult> flush(uint8_t* output, uint64_t outputLength) override;

  Expected<EndResult> finalize(uint8_t* output, uint64_t outputLength) override;

 private:
  Status init(ZlibFormat format, int32_t windowBits);

  z_stream stream_{};
  int32_t compressionLevel_;
  bool initialized_{false};

  friend class ZlibCodec;
};

class GZipDecompressor : public StreamingDecompressor {
 public:
  explicit GZipDecompressor(ZlibFormat format, int32_t windowBits);

  ~GZipDecompressor() override;

  Status reset() override;

  Expected<DecompressResult> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

  bool isFinished() override;

 private:
  Status init();

  z_stream stream_{};
  ZlibFormat format_;
  int32_t windowBits_;
  bool initialized_{false};
  bool finished_{false};

  friend class ZlibCodec;
};

GzipCompressor::GzipCompressor(int32_t compressionLevel)
    : compressionLevel_(compressionLevel) {}

GzipCompressor::~GzipCompressor() {
  if (initialized_) {
    deflateEnd(&stream_);
  }
}

Status GzipCompressor::init(ZlibFormat format, int32_t windowBits) {
  VELOX_DCHECK(!initialized_, "Called on initialized stream.");
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
  VELOX_RETURN_IF(
      ret != Z_OK, zlibError("Zlib deflateInit failed: ", stream_.msg));
  initialized_ = true;
  return Status::OK();
}

Expected<StreamingCompressor::CompressResult> GzipCompressor::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_RETURN_UNEXPECTED_IF(
      !initialized_, Status::UserError("Called on non-initialized stream."));

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(std::min(inputLength, kGzipBufferLimit));
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = deflate(&stream_, Z_NO_FLUSH);
  VELOX_RETURN_UNEXPECTED_IF(
      ret == Z_STREAM_ERROR, zlibError("Zlib compress failed: ", stream_.msg));
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
  VELOX_RETURN_UNEXPECTED_IF(
      ret != Z_BUF_ERROR,
      Status::IOError("Invalid return code from zlib: ", ret));
  // No progress was possible, need to increase output buffer size.
  return CompressResult{0, 0, true};
}

Expected<StreamingCompressor::FlushResult> GzipCompressor::flush(
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_RETURN_UNEXPECTED_IF(
      !initialized_, Status::UserError("Called on non-initialized stream."));

  static constexpr auto kInputLimit =
      static_cast<uint64_t>(std::numeric_limits<uInt>::max());

  stream_.avail_in = 0;
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(std::min(outputLength, kInputLimit));

  auto ret = deflate(&stream_, Z_SYNC_FLUSH);
  VELOX_RETURN_UNEXPECTED_IF(
      ret == Z_STREAM_ERROR, zlibError("Zlib flush failed: ", stream_.msg));

  uint64_t bytesWritten;
  if (ret == Z_OK) {
    bytesWritten = outputLength - stream_.avail_out;
  } else {
    VELOX_RETURN_UNEXPECTED_IF(
        ret != Z_BUF_ERROR,
        Status::IOError("Invalid return code from zlib: ", ret));
    bytesWritten = 0;
  }
  // Quoted from zlib.h:
  // "If deflate returns with avail_out == 0, this function must be called
  // again with the same value of the flush parameter and more output space
  // (updated avail_out), until the flush is complete (deflate returns
  // with non-zero avail_out)."
  // "Note that Z_BUF_ERROR is not fatal, and deflate() can be called again
  // with more input and more output space to continue compressing."
  return FlushResult{bytesWritten, stream_.avail_out == 0};
}

Expected<StreamingCompressor::EndResult> GzipCompressor::finalize(
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_RETURN_UNEXPECTED_IF(
      !initialized_, Status::UserError("Called on non-initialized stream."));

  stream_.avail_in = 0;
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = deflate(&stream_, Z_FINISH);
  VELOX_RETURN_UNEXPECTED_IF(
      ret == Z_STREAM_ERROR, zlibError("Zlib flush failed: ", stream_.msg));

  uint64_t bytesWritten = outputLength - stream_.avail_out;
  if (ret == Z_STREAM_END) {
    // Flush complete, we can now end the stream.
    initialized_ = false;
    ret = deflateEnd(&stream_);
    VELOX_RETURN_UNEXPECTED_IF(
        ret != Z_OK, zlibError("Zlib end failed: ", stream_.msg));
    return EndResult{bytesWritten, false};
  }
  // Not everything could be flushed, need to increase output buffer size.
  return EndResult{bytesWritten, true};
}

GZipDecompressor::GZipDecompressor(ZlibFormat format, int32_t windowBits)
    : format_(format), windowBits_(windowBits) {}

GZipDecompressor::~GZipDecompressor() {
  if (initialized_) {
    inflateEnd(&stream_);
  }
}

Status GZipDecompressor::reset() {
  VELOX_RETURN_IF(
      !initialized_, Status::UserError("Called on non-initialized stream."));
  finished_ = false;

  VELOX_RETURN_IF(
      inflateReset(&stream_) != Z_OK,
      zlibError("zlib inflateReset failed: ", stream_.msg));
  return Status::OK();
}

Expected<StreamingDecompressor::DecompressResult> GZipDecompressor::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  VELOX_RETURN_UNEXPECTED_IF(
      !initialized_, Status::UserError("Called on non-initialized stream."));

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(std::min(inputLength, kGzipBufferLimit));
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out =
      static_cast<uInt>(std::min(outputLength, kGzipBufferLimit));

  auto ret = inflate(&stream_, Z_SYNC_FLUSH);
  VELOX_RETURN_UNEXPECTED_IF(
      ret == Z_DATA_ERROR || ret == Z_STREAM_ERROR || ret == Z_MEM_ERROR,
      zlibError("Zlib inflate failed: ", stream_.msg));
  VELOX_RETURN_UNEXPECTED_IF(
      ret == Z_NEED_DICT,
      zlibError("Zlib inflate failed (need preset dictionary): ", stream_.msg));
  if (ret == Z_BUF_ERROR) {
    // No progress was possible or output is too small.
    return DecompressResult{0, 0, true};
  }
  VELOX_RETURN_UNEXPECTED_IF(
      ret != Z_OK && ret != Z_STREAM_END,
      Status::IOError("Invalid return code from zlib: ", ret));

  finished_ = (ret == Z_STREAM_END);
  return DecompressResult{
      inputLength - stream_.avail_in, outputLength - stream_.avail_out, false};
}

bool GZipDecompressor::isFinished() {
  return finished_;
}

Status GZipDecompressor::init() {
  VELOX_DCHECK(!initialized_, "Called on initialized stream.");
  memset(&stream_, 0, sizeof(stream_));
  finished_ = false;

  auto windowBits = getDecompressionWindowBits(format_, windowBits_);
  VELOX_RETURN_IF(
      inflateInit2(&stream_, windowBits) != Z_OK,
      zlibError("zlib inflateInit failed: ", stream_.msg));
  initialized_ = true;
  return Status::OK();
}

ZlibCodec::ZlibCodec(
    ZlibFormat format,
    int32_t compressionLevel,
    int32_t windowBits)
    : format_(format), windowBits_(windowBits) {
  compressionLevel_ = compressionLevel == kDefaultCompressionLevel
      ? kGzipDefaultCompressionLevel
      : compressionLevel;
}

ZlibCodec::~ZlibCodec() {
  endCompressor();
  endDecompressor();
}

Expected<std::shared_ptr<StreamingCompressor>>
ZlibCodec::makeStreamingCompressor() {
  auto ptr = std::make_shared<GzipCompressor>(compressionLevel_);
  VELOX_RETURN_UNEXPECTED_NOT_OK(ptr->init(format_, windowBits_));
  return ptr;
}

Expected<std::shared_ptr<StreamingDecompressor>>
ZlibCodec::makeStreamingDecompressor() {
  auto ptr = std::make_shared<GZipDecompressor>(format_, windowBits_);
  VELOX_RETURN_UNEXPECTED_NOT_OK(ptr->init());
  return ptr;
}

Status ZlibCodec::initCompressor() {
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
  VELOX_RETURN_IF(
      ret != Z_OK, zlibError("zlib deflateInit failed: ", stream_.msg));
  compressorInitialized_ = true;
  return Status::OK();
}

Status ZlibCodec::initDecompressor() {
  endCompressor();
  memset(&stream_, 0, sizeof(stream_));

  // Initialize to run either deflate or zlib/gzip format
  int32_t windowBits = getDecompressionWindowBits(format_, windowBits_);
  auto ret = inflateInit2(&stream_, windowBits);
  VELOX_RETURN_IF(
      ret != Z_OK, zlibError("zlib inflateInit failed: ", stream_.msg));
  decompressorInitialized_ = true;
  return Status::OK();
}

void ZlibCodec::endCompressor() {
  if (compressorInitialized_) {
    (void)deflateEnd(&stream_);
  }
  compressorInitialized_ = false;
}

void ZlibCodec::endDecompressor() {
  if (decompressorInitialized_) {
    (void)inflateEnd(&stream_);
  }
  decompressorInitialized_ = false;
}

Expected<uint64_t> ZlibCodec::compress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  if (!compressorInitialized_) {
    VELOX_RETURN_UNEXPECTED_NOT_OK(initCompressor());
  }

  VELOX_RETURN_UNEXPECTED_IF(
      deflateReset(&stream_) != Z_OK,
      zlibError("Zlib deflateReset failed: ", stream_.msg));

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = deflate(&stream_, Z_FINISH);
  if (ret != Z_STREAM_END) {
    // Z_OK (and stream.msg NOT set) indicates stream.avail_out is too
    // small.
    VELOX_RETURN_UNEXPECTED_IF(
        ret == Z_OK,
        Status::IOError("zlib deflate failed, output buffer too small"));
    return folly::makeUnexpected(
        zlibError("Zlib deflate failed: ", stream_.msg));
  }
  return stream_.total_out;
}

Expected<uint64_t> ZlibCodec::decompress(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  if (!decompressorInitialized_) {
    VELOX_RETURN_UNEXPECTED_NOT_OK(initDecompressor());
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

  // Reset the stream for this block.
  VELOX_RETURN_UNEXPECTED_IF(
      inflateReset(&stream_) != Z_OK,
      zlibError("zlib inflateReset failed: ", stream_.msg));

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = inflate(&stream_, Z_FINISH);
  if (ret != Z_STREAM_END) {
    // Z_OK (and stream.msg NOT set) indicates stream.avail_out is too
    // small.
    VELOX_RETURN_UNEXPECTED_IF(
        ret == Z_OK,
        Status::IOError("zlib inflate failed, output buffer too small"));
    return folly::makeUnexpected(
        zlibError("Zlib inflate failed: ", stream_.msg));
  }
  return stream_.total_out;
}

Expected<uint64_t> ZlibCodec::compressFixedLength(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  if (!compressorInitialized_) {
    VELOX_RETURN_UNEXPECTED_NOT_OK(initCompressor());
  }

  VELOX_RETURN_UNEXPECTED_IF(
      deflateReset(&stream_) != Z_OK,
      zlibError("Zlib deflateReset failed: ", stream_.msg));

  stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
  stream_.avail_in = static_cast<uInt>(inputLength);
  stream_.next_out = reinterpret_cast<Bytef*>(output);
  stream_.avail_out = static_cast<uInt>(outputLength);

  auto ret = deflate(&stream_, Z_FINISH);
  VELOX_RETURN_UNEXPECTED_IF(
      ret != Z_STREAM_END && ret != Z_OK && ret != Z_BUF_ERROR,
      zlibError("Zlib deflate failed: ", stream_.msg));

  return stream_.total_out;
}

uint64_t ZlibCodec::maxCompressedLength(uint64_t inputLength) {
  // Must be in compression mode.
  if (!compressorInitialized_) {
    initCompressor();
  }
  uint64_t maxLength = deflateBound(&stream_, static_cast<uLong>(inputLength));
  // ARROW-3514: return a more pessimistic estimate to account for bugs
  // in old zlib versions.
  return maxLength + 12;
}

Status ZlibCodec::init() {
  if (windowBits_ < kZlibMinWindowBits || windowBits_ > kZlibMaxWindowBits) {
    return Status::UserError(
        "zlib window bits should be between {} and {}",
        kZlibMinWindowBits,
        kZlibMaxWindowBits);
  }
  VELOX_RETURN_NOT_OK(initCompressor());
  VELOX_RETURN_NOT_OK(initDecompressor());
  return Status::OK();
}

CompressionKind ZlibCodec::compressionKind() const {
  if (format_ == ZlibFormat::kZlib) {
    return CompressionKind_ZLIB;
  }
  return CompressionKind_GZIP;
}

int32_t ZlibCodec::compressionLevel() const {
  return compressionLevel_;
}

int32_t ZlibCodec::minCompressionLevel() const {
  return kGZipMinCompressionLevel;
}

int32_t ZlibCodec::maxCompressionLevel() const {
  return kGZipMaxCompressionLevel;
}

int32_t ZlibCodec::defaultCompressionLevel() const {
  return kGzipDefaultCompressionLevel;
}

std::string_view ZlibCodec::name() const {
  switch (format_) {
    case ZlibFormat::kDeflate:
      return "deflate";
    case ZlibFormat::kZlib:
      return "zlib";
    case ZlibFormat::kGzip:
      return "gzip";
  }
  VELOX_UNREACHABLE();
}

std::unique_ptr<Codec> makeZlibCodec(
    ZlibFormat format,
    int32_t compressionLevel,
    std::optional<int32_t> windowBits) {
  return std::make_unique<ZlibCodec>(
      format, compressionLevel, windowBits.value_or(kZlibDefaultWindowBits));
}
} // namespace facebook::velox::common
