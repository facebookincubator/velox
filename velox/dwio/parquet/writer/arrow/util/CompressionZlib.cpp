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

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

#include <zlib.h>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::parquet::arrow::util::internal {
namespace {

// ----------------------------------------------------------------------.
// Gzip implementation.

// These are magic numbers from zlib.h.  Not clear why they are not defined.
// There.

// Maximum window size.
constexpr int kGZipMaxWindowBits = 15;

// Minimum window size.
constexpr int kGZipMinWindowBits = 9;

// Default window size.
constexpr int kGZipDefaultWindowBits = 15;

// Output Gzip.
constexpr int GZIP_CODEC = 16;

// Determine if this is libz or gzip from header.
constexpr int DETECT_CODEC = 32;

constexpr int kGZipMinCompressionLevel = 1;
constexpr int kGZipMaxCompressionLevel = 9;

int compressionWindowBitsForFormat(GZipFormat format, int windowBits) {
  switch (format) {
    case GZipFormat::DEFLATE:
      windowBits = -windowBits;
      break;
    case GZipFormat::GZIP:
      windowBits += GZIP_CODEC;
      break;
    case GZipFormat::ZLIB:
      break;
  }
  return windowBits;
}

int decompressionWindowBitsForFormat(GZipFormat format, int windowBits) {
  if (format == GZipFormat::DEFLATE) {
    return -windowBits;
  } else {
    /* If not deflate, autodetect format from header */
    return windowBits | DETECT_CODEC;
  }
}

Status zlibErrorPrefix(const char* prefixMsg, const char* msg) {
  return Status::IOError(prefixMsg, (msg) ? msg : "(unknown error)");
}

// ----------------------------------------------------------------------.
// Gzip Decompressor implementation.

class GZipDecompressor : public Decompressor {
 public:
  explicit GZipDecompressor(GZipFormat format, int windowBits)
      : format_(format),
        windowBits_(windowBits),
        initialized_(false),
        finished_(false) {}

  ~GZipDecompressor() override {
    if (initialized_) {
      inflateEnd(&stream_);
    }
  }

  Status init() {
    VELOX_DCHECK(!initialized_);
    memset(&stream_, 0, sizeof(stream_));
    finished_ = false;

    int ret;
    int windowBits = decompressionWindowBitsForFormat(format_, windowBits_);
    if ((ret = inflateInit2(&stream_, windowBits)) != Z_OK) {
      return zlibError("zlib inflateInit failed: ");
    } else {
      initialized_ = true;
      return Status::OK();
    }
  }

  Status reset() override {
    VELOX_DCHECK(initialized_);
    finished_ = false;
    int ret;
    if ((ret = inflateReset(&stream_)) != Z_OK) {
      return zlibError("zlib inflateReset failed: ");
    } else {
      return Status::OK();
    }
  }

  Result<DecompressResult> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    static constexpr auto inputLimit =
        static_cast<int64_t>(std::numeric_limits<uInt>::max());
    stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
    stream_.avail_in = static_cast<uInt>(std::min(inputLen, inputLimit));
    stream_.next_out = reinterpret_cast<Bytef*>(output);
    stream_.avail_out = static_cast<uInt>(std::min(outputLen, inputLimit));
    int ret;

    ret = inflate(&stream_, Z_SYNC_FLUSH);
    if (ret == Z_DATA_ERROR || ret == Z_STREAM_ERROR || ret == Z_MEM_ERROR) {
      return zlibError("zlib inflate failed: ");
    }
    if (ret == Z_NEED_DICT) {
      return zlibError("zlib inflate failed (need preset dictionary): ");
    }
    finished_ = (ret == Z_STREAM_END);
    if (ret == Z_BUF_ERROR) {
      // No progress was possible.
      return DecompressResult{0, 0, true};
    } else {
      VELOX_DCHECK(ret == Z_OK || ret == Z_STREAM_END);
      // Some progress has been made.
      return DecompressResult{
          inputLen - stream_.avail_in, outputLen - stream_.avail_out, false};
    }
    return Status::OK();
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  Status zlibError(const char* prefixMsg) {
    return zlibErrorPrefix(prefixMsg, stream_.msg);
  }

  z_stream stream_;
  GZipFormat format_;
  int windowBits_;
  bool initialized_;
  bool finished_;
};

// ----------------------------------------------------------------------.
// Gzip Compressor implementation.

class GZipCompressor : public Compressor {
 public:
  explicit GZipCompressor(int compressionLevel)
      : initialized_(false), compressionLevel_(compressionLevel) {}

  ~GZipCompressor() override {
    if (initialized_) {
      deflateEnd(&stream_);
    }
  }

  Status init(GZipFormat format, int inputWindowBits) {
    VELOX_DCHECK(!initialized_);
    memset(&stream_, 0, sizeof(stream_));

    int ret;
    // Initialize to run specified format.
    int windowBits = compressionWindowBitsForFormat(format, inputWindowBits);
    if ((ret = deflateInit2(
             &stream_,
             Z_DEFAULT_COMPRESSION,
             Z_DEFLATED,
             windowBits,
             compressionLevel_,
             Z_DEFAULT_STRATEGY)) != Z_OK) {
      return zlibError("zlib deflateInit failed: ");
    } else {
      initialized_ = true;
      return Status::OK();
    }
  }

  Result<CompressResult> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    VELOX_DCHECK(initialized_, "Called on non-initialized stream");

    static constexpr auto inputLimit =
        static_cast<int64_t>(std::numeric_limits<uInt>::max());

    stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
    stream_.avail_in = static_cast<uInt>(std::min(inputLen, inputLimit));
    stream_.next_out = reinterpret_cast<Bytef*>(output);
    stream_.avail_out = static_cast<uInt>(std::min(outputLen, inputLimit));

    int64_t ret = 0;
    ret = deflate(&stream_, Z_NO_FLUSH);
    if (ret == Z_STREAM_ERROR) {
      return zlibError("zlib compress failed: ");
    }
    if (ret == Z_OK) {
      // Some progress has been made.
      return CompressResult{
          inputLen - stream_.avail_in, outputLen - stream_.avail_out};
    } else {
      // No progress was possible.
      VELOX_DCHECK_EQ(ret, Z_BUF_ERROR);
      return CompressResult{0, 0};
    }
  }

  Result<FlushResult> flush(int64_t outputLen, uint8_t* output) override {
    VELOX_DCHECK(initialized_, "Called on non-initialized stream");

    static constexpr auto inputLimit =
        static_cast<int64_t>(std::numeric_limits<uInt>::max());

    stream_.avail_in = 0;
    stream_.next_out = reinterpret_cast<Bytef*>(output);
    stream_.avail_out = static_cast<uInt>(std::min(outputLen, inputLimit));

    int64_t ret = 0;
    ret = deflate(&stream_, Z_SYNC_FLUSH);
    if (ret == Z_STREAM_ERROR) {
      return zlibError("zlib flush failed: ");
    }
    int64_t bytesWritten;
    if (ret == Z_OK) {
      bytesWritten = outputLen - stream_.avail_out;
    } else {
      VELOX_DCHECK_EQ(ret, Z_BUF_ERROR);
      bytesWritten = 0;
    }
    // "If deflate returns with avail_out == 0, this function must be called.
    //  Again with the same value of the flush parameter and more output space.
    //  (Updated avail_out), until the flush is complete (deflate returns.
    //  With non-zero avail_out).".
    // "Note that Z_BUF_ERROR is not fatal, and deflate() can be called again.
    //  With more input and more output space to continue compressing.".
    return FlushResult{bytesWritten, stream_.avail_out == 0};
  }

  Result<EndResult> end(int64_t outputLen, uint8_t* output) override {
    VELOX_DCHECK(initialized_, "Called on non-initialized stream");

    static constexpr auto inputLimit =
        static_cast<int64_t>(std::numeric_limits<uInt>::max());

    stream_.avail_in = 0;
    stream_.next_out = reinterpret_cast<Bytef*>(output);
    stream_.avail_out = static_cast<uInt>(std::min(outputLen, inputLimit));

    int64_t ret = 0;
    ret = deflate(&stream_, Z_FINISH);
    if (ret == Z_STREAM_ERROR) {
      return zlibError("zlib flush failed: ");
    }
    int64_t bytesWritten = outputLen - stream_.avail_out;
    if (ret == Z_STREAM_END) {
      // Flush complete, we can now end the stream.
      initialized_ = false;
      ret = deflateEnd(&stream_);
      if (ret == Z_OK) {
        return EndResult{bytesWritten, false};
      } else {
        return zlibError("zlib end failed: ");
      }
    } else {
      // Not everything could be flushed,.
      return EndResult{bytesWritten, true};
    }
  }

 protected:
  Status zlibError(const char* prefixMsg) {
    return zlibErrorPrefix(prefixMsg, stream_.msg);
  }

  z_stream stream_;
  bool initialized_;
  int compressionLevel_;
};

// ----------------------------------------------------------------------.
// Gzip codec implementation.

class GZipCodec : public Codec {
 public:
  explicit GZipCodec(int compressionLevel, GZipFormat format, int windowBits)
      : format_(format),
        windowBits_(windowBits),
        compressorInitialized_(false),
        decompressorInitialized_(false) {
    compressionLevel_ = compressionLevel == kUseDefaultCompressionLevel
        ? kGZipDefaultCompressionLevel
        : compressionLevel;
  }

  ~GZipCodec() override {
    endCompressor();
    endDecompressor();
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    auto ptr = std::make_shared<GZipCompressor>(compressionLevel_);
    RETURN_NOT_OK(ptr->init(format_, windowBits_));
    return ptr;
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    auto ptr = std::make_shared<GZipDecompressor>(format_, windowBits_);
    RETURN_NOT_OK(ptr->init());
    return ptr;
  }

  Status initCompressor() {
    endDecompressor();
    memset(&stream_, 0, sizeof(stream_));

    int ret;
    // Initialize to run specified format.
    int windowBits = compressionWindowBitsForFormat(format_, windowBits_);
    if ((ret = deflateInit2(
             &stream_,
             Z_DEFAULT_COMPRESSION,
             Z_DEFLATED,
             windowBits,
             compressionLevel_,
             Z_DEFAULT_STRATEGY)) != Z_OK) {
      return zlibErrorPrefix("zlib deflateInit failed: ", stream_.msg);
    }
    compressorInitialized_ = true;
    return Status::OK();
  }

  void endCompressor() {
    if (compressorInitialized_) {
      static_cast<void>(deflateEnd(&stream_));
    }
    compressorInitialized_ = false;
  }

  Status initDecompressor() {
    endCompressor();
    memset(&stream_, 0, sizeof(stream_));
    int ret;

    // Initialize to run either deflate or zlib/gzip format.
    int windowBits = decompressionWindowBitsForFormat(format_, windowBits_);
    if ((ret = inflateInit2(&stream_, windowBits)) != Z_OK) {
      return zlibErrorPrefix("zlib inflateInit failed: ", stream_.msg);
    }
    decompressorInitialized_ = true;
    return Status::OK();
  }

  void endDecompressor() {
    if (decompressorInitialized_) {
      static_cast<void>(inflateEnd(&stream_));
    }
    decompressorInitialized_ = false;
  }

  Result<int64_t> decompress(
      int64_t inputLength,
      const uint8_t* input,
      int64_t outputBufferLength,
      uint8_t* output) override {
    if (!decompressorInitialized_) {
      RETURN_NOT_OK(initDecompressor());
    }
    if (outputBufferLength == 0) {
      // The zlib library does not allow *output to be NULL, even when.
      // Output_buffer_length is 0 (inflate() will return Z_STREAM_ERROR). We.
      // Don't consider this an error, so bail early if no output is expected.
      // Note that we don't signal an error if the input actually contains.
      // Compressed data.
      return 0;
    }

    // Reset the stream for this block.
    if (inflateReset(&stream_) != Z_OK) {
      return zlibErrorPrefix("zlib inflateReset failed: ", stream_.msg);
    }

    int ret = 0;
    // Gzip can run in streaming mode or non-streaming mode.  We only.
    // Support the non-streaming use case where we present it the entire.
    // Compressed input and a buffer big enough to contain the entire.
    // Compressed output.  In the case where we don't know the output,.
    // We just make a bigger buffer and try the non-streaming mode.
    // From the beginning again.
    while (ret != Z_STREAM_END) {
      stream_.next_in =
          const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
      stream_.avail_in = static_cast<uInt>(inputLength);
      stream_.next_out = reinterpret_cast<Bytef*>(output);
      stream_.avail_out = static_cast<uInt>(outputBufferLength);

      // We know the output size.  In this case, we can use Z_FINISH.
      // Which is more efficient.
      ret = inflate(&stream_, Z_FINISH);
      if (ret == Z_STREAM_END || ret != Z_OK) {
        break;
      }

      // Failure, buffer was too small.
      return Status::IOError(
          "Too small a buffer passed to GZipCodec. InputLength=",
          inputLength,
          " OutputLength=",
          outputBufferLength);
    }

    // Failure for some other reason.
    if (ret != Z_STREAM_END) {
      return zlibErrorPrefix("GZipCodec failed: ", stream_.msg);
    }

    return stream_.total_out;
  }

  int64_t maxCompressedLen(
      int64_t inputLength,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    // Must be in compression mode.
    if (!compressorInitialized_) {
      Status s = initCompressor();
      VELOX_DCHECK(s.ok(), s.ToString());
    }
    int64_t maxLen = deflateBound(&stream_, static_cast<uLong>(inputLength));
    // ARROW-3514: return a more pessimistic estimate to account for bugs.
    // In old zlib versions.
    return maxLen + 12;
  }

  Result<int64_t> compress(
      int64_t inputLength,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* output) override {
    if (!compressorInitialized_) {
      RETURN_NOT_OK(initCompressor());
    }
    stream_.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
    stream_.avail_in = static_cast<uInt>(inputLength);
    stream_.next_out = reinterpret_cast<Bytef*>(output);
    stream_.avail_out = static_cast<uInt>(outputBufferLen);

    int64_t ret = 0;
    if ((ret = deflate(&stream_, Z_FINISH)) != Z_STREAM_END) {
      if (ret == Z_OK) {
        // Will return Z_OK (and stream.msg NOT set) if stream.avail_out is too.
        // Small.
        return Status::IOError("zlib deflate failed, output buffer too small");
      }

      return zlibErrorPrefix("zlib deflate failed: ", stream_.msg);
    }

    if (deflateReset(&stream_) != Z_OK) {
      return zlibErrorPrefix("zlib deflateReset failed: ", stream_.msg);
    }

    // Actual output length.
    return outputBufferLen - stream_.avail_out;
  }

  Status init() override {
    if (windowBits_ < kGZipMinWindowBits || windowBits_ > kGZipMaxWindowBits) {
      return Status::Invalid(
          "GZip window_bits should be between ",
          kGZipMinWindowBits,
          " and ",
          kGZipMaxWindowBits);
    }
    const Status initCompressorStatus = initCompressor();
    if (!initCompressorStatus.ok()) {
      return initCompressorStatus;
    }
    return initDecompressor();
  }

  Compression::type compressionType() const override {
    return Compression::GZIP;
  }

  int compressionLevel() const override {
    return compressionLevel_;
  }
  int minimumCompressionLevel() const override {
    return kGZipMinCompressionLevel;
  }
  int maximumCompressionLevel() const override {
    return kGZipMaxCompressionLevel;
  }
  int defaultCompressionLevel() const override {
    return kGZipDefaultCompressionLevel;
  }

 private:
  // Zlib is stateful and the z_stream state variable must be initialized.
  // Before.
  z_stream stream_;

  // Realistically, this will always be GZIP, but we leave the option open to.
  // Configure.
  GZipFormat format_;

  // These variables are mutually exclusive. When the codec is in "Compressor".
  // State, compressor_initialized_ is true while decompressor_initialized_ is.
  // False. When it's decompressing, the opposite is true.
  //
  // Indeed, this is slightly hacky, but the alternative is having separate.
  // Compressor and Decompressor classes. If this ever becomes an issue, we can.
  // Perform the refactoring then.
  int windowBits_;
  bool compressorInitialized_;
  bool decompressorInitialized_;
  int compressionLevel_;
};

} // namespace

std::unique_ptr<Codec> makeGZipCodec(
    int compressionLevel,
    GZipFormat format,
    std::optional<int> windowBits) {
  return std::make_unique<GZipCodec>(
      compressionLevel, format, windowBits.value_or(kGZipDefaultWindowBits));
}

} // namespace facebook::velox::parquet::arrow::util::internal
