/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <snappy.h>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"

#include "velox/common/base/Exceptions.h"

using std::size_t;

namespace facebook::velox::parquet::arrow::util::internal {
namespace {

using ::arrow::Result;

// ----------------------------------------------------------------------.
// Snappy implementation.

class SnappyCodec : public Codec {
 public:
  Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) override {
    size_t decompressedSize;
    if (!snappy::GetUncompressedLength(
            reinterpret_cast<const char*>(input),
            static_cast<size_t>(inputLen),
            &decompressedSize)) {
      return Status::IOError("Corrupt snappy compressed data.");
    }
    if (outputBufferLen < static_cast<int64_t>(decompressedSize)) {
      return Status::Invalid(
          "Output buffer size (",
          outputBufferLen,
          ") must be ",
          decompressedSize,
          " or larger.");
    }
    if (!snappy::RawUncompress(
            reinterpret_cast<const char*>(input),
            static_cast<size_t>(inputLen),
            reinterpret_cast<char*>(outputBuffer))) {
      return Status::IOError("Corrupt snappy compressed data.");
    }
    return static_cast<int64_t>(decompressedSize);
  }

  int64_t maxCompressedLen(
      int64_t inputLen,
      const uint8_t* ARROW_ARG_UNUSED(input)) override {
    VELOX_DCHECK_GE(inputLen, 0);
    return snappy::MaxCompressedLength(static_cast<size_t>(inputLen));
  }

  Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t ARROW_ARG_UNUSED(outputBufferLen),
      uint8_t* outputBuffer) override {
    size_t outputSize;
    snappy::RawCompress(
        reinterpret_cast<const char*>(input),
        static_cast<size_t>(inputLen),
        reinterpret_cast<char*>(outputBuffer),
        &outputSize);
    return static_cast<int64_t>(outputSize);
  }

  Result<std::shared_ptr<Compressor>> makeCompressor() override {
    return Status::NotImplemented(
        "Streaming compression unsupported with Snappy");
  }

  Result<std::shared_ptr<Decompressor>> makeDecompressor() override {
    return Status::NotImplemented(
        "Streaming decompression unsupported with Snappy");
  }

  Compression::type compressionType() const override {
    return Compression::SNAPPY;
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

std::unique_ptr<Codec> makeSnappyCodec() {
  return std::make_unique<SnappyCodec>();
}

} // namespace facebook::velox::parquet::arrow::util::internal
