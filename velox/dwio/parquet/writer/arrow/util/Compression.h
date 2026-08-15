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

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace facebook::velox::parquet::arrow {

struct Compression {
  /// \brief Compression algorithm.
  enum type {
    UNCOMPRESSED,
    SNAPPY,
    GZIP,
    BROTLI,
    ZSTD,
    LZ4,
    LZ4_FRAME,
    LZO,
    BZ2,
    LZ4_HADOOP
  };
};
} // namespace facebook::velox::parquet::arrow

namespace facebook::velox::parquet::arrow::util {

using namespace ::arrow;

constexpr int kUseDefaultCompressionLevel = std::numeric_limits<int>::min();

/// \brief Streaming compressor interface.
///
class ARROW_EXPORT Compressor {
 public:
  virtual ~Compressor() = default;

  struct CompressResult {
    int64_t bytesRead;
    int64_t bytesWritten;
  };
  struct FlushResult {
    int64_t bytesWritten;
    bool shouldRetry;
  };
  struct EndResult {
    int64_t bytesWritten;
    bool shouldRetry;
  };

  /// \brief Compress some input.
  ///
  /// If bytes_read is 0 on return, then a larger output buffer should be
  /// supplied.
  virtual Result<CompressResult> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) = 0;

  /// \brief Flush part of the compressed output.
  ///
  /// If should_retry is true on return, Flush() should be called again with a
  /// larger buffer.
  virtual Result<FlushResult> flush(int64_t outputLen, uint8_t* output) = 0;

  /// \brief End compressing, doing whatever is necessary to end the stream.
  ///
  /// If should_retry is true on return, End() should be called again with a
  /// larger buffer. Otherwise, the Compressor should not be used anymore.
  ///
  /// End() implies Flush().
  virtual Result<EndResult> end(int64_t outputLen, uint8_t* output) = 0;

  // XXX add methods for buffer size heuristics?
};

/// \brief Streaming decompressor interface.
///
class ARROW_EXPORT Decompressor {
 public:
  virtual ~Decompressor() = default;

  struct DecompressResult {
    // XXX is need_more_output necessary? (Brotli?)
    int64_t bytesRead;
    int64_t bytesWritten;
    bool needMoreOutput;
  };

  /// \brief Decompress some input.
  ///
  /// If need_more_output is true on return, a larger output buffer needs to be
  /// supplied.
  virtual Result<DecompressResult> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) = 0;

  /// \brief Return whether the compressed stream is finished.
  ///
  /// This is a heuristic. If true is returned, then it is guaranteed that the
  /// stream is finished. If false is returned, however, it may simply be that
  /// the underlying library isn't able to provide the information.
  virtual bool isFinished() = 0;

  /// \brief Reinitialize decompressor, making it ready for a new compressed
  /// stream.
  virtual Status reset() = 0;

  // XXX add methods for buffer size heuristics?
};

/// \brief Compression codec options.
class ARROW_EXPORT CodecOptions {
 public:
  explicit CodecOptions(int compressionLevel = kUseDefaultCompressionLevel)
      : compressionLevel(compressionLevel) {}

  virtual ~CodecOptions() = default;

  int compressionLevel;
};

// ----------------------------------------------------------------------.
// GZip codec options implementation.

enum class GZipFormat {
  ZLIB,
  DEFLATE,
  GZIP,
};

class ARROW_EXPORT GZipCodecOptions : public CodecOptions {
 public:
  GZipFormat gzipFormat = GZipFormat::GZIP;
  std::optional<int> windowBits;
};

// ----------------------------------------------------------------------.
// Brotli codec options implementation.

class ARROW_EXPORT BrotliCodecOptions : public CodecOptions {
 public:
  std::optional<int> windowBits;
};

/// \brief Compression codec.
class ARROW_EXPORT Codec {
 public:
  virtual ~Codec() = default;

  /// \brief Return special value to indicate that a codec implementation
  /// should use its default compression level.
  static int useDefaultCompressionLevel();

  /// \brief Return a string name for compression type.
  static const std::string& getCodecAsString(Compression::type t);

  /// \brief Return compression type for name (all lower case).
  static Result<Compression::type> getCompressionType(const std::string& name);

  /// \brief Create a codec for the given compression algorithm with codec
  /// options.
  static Result<std::unique_ptr<Codec>> create(
      Compression::type codec,
      const CodecOptions& codecOptions = CodecOptions{});

  /// \brief Create a codec for the given compression algorithm.
  static Result<std::unique_ptr<Codec>> create(
      Compression::type codec,
      int compressionLevel);

  /// \brief Return true if support for indicated codec has been enabled.
  static bool isAvailable(Compression::type codec);

  /// \brief Return true if indicated codec supports setting a compression
  /// level.
  static bool supportsCompressionLevel(Compression::type codec);

  /// \brief Return the smallest supported compression level for the codec.
  /// Note: This function creates a temporary Codec instance.
  static Result<int> minimumCompressionLevel(Compression::type codec);

  /// \brief Return the largest supported compression level for the codec.
  /// Note: This function creates a temporary Codec instance.
  static Result<int> maximumCompressionLevel(Compression::type codec);

  /// \brief Return the default compression level.
  /// Note: This function creates a temporary Codec instance.
  static Result<int> defaultCompressionLevel(Compression::type codec);

  /// \brief Return the smallest supported compression level.
  virtual int minimumCompressionLevel() const = 0;

  /// \brief Return the largest supported compression level.
  virtual int maximumCompressionLevel() const = 0;

  /// \brief Return the default compression level.
  virtual int defaultCompressionLevel() const = 0;

  /// \brief One-shot decompression function.
  ///
  /// Output_buffer_len must be correct and therefore be obtained in advance.
  /// The actual decompressed length is returned.
  ///
  /// \note One-shot decompression is not always compatible with streaming
  /// compression.  Depending on the codec (e.g. LZ4), different formats may
  /// be used.
  virtual Result<int64_t> decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) = 0;

  /// \brief One-shot compression function.
  ///
  /// Output_buffer_len must first have been computed using MaxCompressedLen().
  /// The actual compressed length is returned.
  ///
  /// \note One-shot compression is not always compatible with streaming
  /// decompression. Depending on the codec (e.g. LZ4), different formats may be
  /// used.
  virtual Result<int64_t> compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputBufferLen,
      uint8_t* outputBuffer) = 0;

  virtual int64_t maxCompressedLen(int64_t inputLen, const uint8_t* input) = 0;

  /// \brief Create a streaming compressor instance.
  virtual Result<std::shared_ptr<Compressor>> makeCompressor() = 0;

  /// \brief Create a streaming compressor instance.
  virtual Result<std::shared_ptr<Decompressor>> makeDecompressor() = 0;

  /// \brief This Codec's compression type.
  virtual Compression::type compressionType() const = 0;

  /// \brief The name of this Codec's compression type.
  const std::string& name() const {
    return getCodecAsString(compressionType());
  }

  /// \brief This Codec's compression level, if applicable.
  virtual int compressionLevel() const {
    return useDefaultCompressionLevel();
  }

 private:
  /// \brief Initializes the codec's resources.
  virtual Status init();
};

} // namespace facebook::velox::parquet::arrow::util
