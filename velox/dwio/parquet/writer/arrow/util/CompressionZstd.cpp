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

#include <cstddef>
#include <cstdint>
#include <memory>

#include <zstd.h>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"

#include "velox/common/base/Exceptions.h"

using std::size_t;

namespace facebook::velox::parquet::arrow::util::internal {
namespace {

Status zSTDError(size_t ret, const char* prefixMsg) {
  return Status::IOError(prefixMsg, ZSTD_getErrorName(ret));
}

// ----------------------------------------------------------------------.
// ZSTD decompressor implementation.

class ZSTDDecompressor : public Decompressor {
 public:
  ZSTDDecompressor() : stream_(ZSTD_createDStream()) {}

  ~ZSTDDecompressor() override {
    ZSTD_freeDStream(stream_);
  }

  Status init() {
    finished_ = false;
    size_t ret = ZSTD_initDStream(stream_);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD init failed: ");
    } else {
      return Status::OK();
    }
  }

  Result<DecompressResult> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    ZSTD_inBuffer inBuf;
    ZSTD_outBuffer outBuf;

    inBuf.src = input;
    inBuf.size = static_cast<size_t>(inputLen);
    inBuf.pos = 0;
    outBuf.dst = output;
    outBuf.size = static_cast<size_t>(outputLen);
    outBuf.pos = 0;

    size_t ret;
    ret = ZSTD_decompressStream(stream_, &outBuf, &inBuf);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD decompress failed: ");
    }
    finished_ = (ret == 0);
    return DecompressResult{
        static_cast<int64_t>(inBuf.pos),
        static_cast<int64_t>(outBuf.pos),
        inBuf.pos == 0 && outBuf.pos == 0};
  }

  Status reset() override {
    return init();
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  ZSTD_DStream* stream_;
  bool finished_;
};

// ----------------------------------------------------------------------.
// ZSTD compressor implementation.

class ZSTDCompressor : public Compressor {
 public:
  explicit ZSTDCompressor(int compressionLevel)
      : stream_(ZSTD_createCStream()), compressionLevel_(compressionLevel) {}

  ~ZSTDCompressor() override {
    ZSTD_freeCStream(stream_);
  }

  Status init() {
    size_t ret = ZSTD_initCStream(stream_, compressionLevel_);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD init failed: ");
    } else {
      return Status::OK();
    }
  }

  Result<CompressResult> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    ZSTD_inBuffer inBuf;
    ZSTD_outBuffer outBuf;

    inBuf.src = input;
    inBuf.size = static_cast<size_t>(inputLen);
    inBuf.pos = 0;
    outBuf.dst = output;
    outBuf.size = static_cast<size_t>(outputLen);
    outBuf.pos = 0;

    size_t ret;
    ret = ZSTD_compressStream(stream_, &outBuf, &inBuf);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD compress failed: ");
    }
    return CompressResult{
        static_cast<int64_t>(inBuf.pos), static_cast<int64_t>(outBuf.pos)};
  }

  Result<FlushResult> flush(int64_t outputLen, uint8_t* output) override {
    ZSTD_outBuffer outBuf;

    outBuf.dst = output;
    outBuf.size = static_cast<size_t>(outputLen);
    outBuf.pos = 0;

    size_t ret;
    ret = ZSTD_flushStream(stream_, &outBuf);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD flush failed: ");
    }
    return FlushResult{static_cast<int64_t>(outBuf.pos), ret > 0};
  }

  Result<EndResult> end(int64_t outputLen, uint8_t* output) override {
    ZSTD_outBuffer outBuf;

    outBuf.dst = output;
    outBuf.size = static_cast<size_t>(outputLen);
    outBuf.pos = 0;

    size_t ret;
    ret = ZSTD_endStream(stream_, &outBuf);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD end failed: ");
    }
    return EndResult{static_cast<int64_t>(outBuf.pos), ret > 0};
  }

 protected:
  ZSTD_CStream* stream_;

 private:
  int compressionLevel_;
};

// ----------------------------------------------------------------------.
// ZSTD codec implementation.

class ZSTDCodec : public Codec {
 public:
  explicit ZSTDCodec(int compressionLevel)
      : compressionLevel_(
            compressionLevel == kUseDefaultCompressionLevel
                ? kZSTDDefaultCompressionLevel
                : compressionLevel) {}

  Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    if (outputBuffer == nullptr) {
      // We may pass a NULL 0-byte output buffer but some zstd versions demand a
      // valid pointer: https://github.com/facebook/zstd/issues/1385.
      static uint8_t emptyBuffer;
      VELOX_DCHECK_EQ(outputBufferLen, 0);
      outputBuffer = &emptyBuffer;
    }

    size_t ret = ZSTD_decompress(
        outputBuffer,
        static_cast<size_t>(outputBufferLen),
        input,
        static_cast<size_t>(inputLen));
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD decompression failed: ");
    }
    if (static_cast<int64_t>(ret) != outputBufferLen) {
      return Status::IOError("Corrupt ZSTD compressed data.");
    }
    return static_cast<int64_t>(ret);
  }

  int64_t maxCompressedLen(
      int64_t inputLen,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    VELOX_DCHECK_GE(inputLen, 0);
    return ZSTD_compressBound(static_cast<size_t>(inputLen));
  }

  Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    size_t ret = ZSTD_compress(
        outputBuffer,
        static_cast<size_t>(outputBufferLen),
        input,
        static_cast<size_t>(inputLen),
        compressionLevel_);
    if (ZSTD_isError(ret)) {
      return zSTDError(ret, "ZSTD compression failed: ");
    }
    return static_cast<int64_t>(ret);
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    auto ptr = std::make_shared<ZSTDCompressor>(compressionLevel_);
    RETURN_NOT_OK(ptr->init());
    return ptr;
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    auto ptr = std::make_shared<ZSTDDecompressor>();
    RETURN_NOT_OK(ptr->init());
    return ptr;
  }

  Compression::type compressionType() const override {
    return Compression::ZSTD;
  }
  int minimumCompressionLevel() const override {
    return ZSTD_minCLevel();
  }
  int maximumCompressionLevel() const override {
    return ZSTD_maxCLevel();
  }
  int defaultCompressionLevel() const override {
    return kZSTDDefaultCompressionLevel;
  }

  int compressionLevel() const override {
    return compressionLevel_;
  }

 private:
  const int compressionLevel_;
};

} // namespace

std::unique_ptr<Codec> makeZSTDCodec(int compressionLevel) {
  return std::make_unique<ZSTDCodec>(compressionLevel);
}
} // namespace facebook::velox::parquet::arrow::util::internal
