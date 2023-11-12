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

// Derived from Apache Arrow.

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include "velox/common/compression/Compression.h"

namespace facebook::velox::common {

static constexpr int32_t kUseDefaultCompressionLevel =
    std::numeric_limits<int32_t>::min();

/// Streaming compressor interface.
class Compressor {
 public:
  virtual ~Compressor() = default;

  struct CompressResult {
    uint64_t bytesRead;
    uint64_t bytesWritten;
    bool outputTooSmall;
  };
  struct FlushResult {
    uint64_t bytesWritten;
    bool outputTooSmall;
  };
  struct EndResult {
    uint64_t bytesWritten;
    bool outputTooSmall;
  };

  /// Compress some input.
  /// If bytes_read is 0 on return, then a larger output buffer should be
  /// supplied.
  virtual CompressResult compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) = 0;

  /// Flush part of the compressed output.
  /// If outputTooSmall is true on return, flush() should be called again
  /// with a larger buffer.
  virtual FlushResult flush(uint64_t outputLength, uint8_t* output) = 0;

  /// End compressing, doing whatever is necessary to end the stream.
  /// If outputTooSmall is true on return, end() should be called again
  /// with a larger buffer. Otherwise, the Compressor should not be used
  /// anymore.
  /// end() implies flush().
  virtual EndResult end(uint64_t outputLength, uint8_t* output) = 0;
};

/// Streaming decompressor interface
class Decompressor {
 public:
  virtual ~Decompressor() = default;

  struct DecompressResult {
    uint64_t bytesRead;
    uint64_t bytesWritten;
    bool outputTooSmall;
  };

  /// Decompress some input.
  /// If outputTooSmall is true on return, a larger output buffer needs
  /// to be supplied.
  virtual DecompressResult decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) = 0;

  /// Return whether the compressed stream is finished.
  virtual bool isFinished() = 0;

  /// Reinitialize decompressor, making it ready for a new compressed stream.
  virtual void reset() = 0;
};

/// Compression codec options
class CodecOptions {
 public:
  explicit CodecOptions(int32_t compressionLevel = kUseDefaultCompressionLevel)
      : compressionLevel(compressionLevel) {}

  virtual ~CodecOptions() = default;

  int32_t compressionLevel;
};

/// Compression codec
class Codec {
 public:
  virtual ~Codec() = default;

  /// Return special value to indicate that a codec implementation
  /// should use its default compression level.
  static int32_t useDefaultCompressionLevel();

  /// Create a kind for the given compression algorithm with CodecOptions.
  static std::unique_ptr<Codec> create(
      CompressionKind kind,
      const CodecOptions& codecOptions = CodecOptions{});

  /// Create a kind for the given compression algorithm.
  static std::unique_ptr<Codec> create(
      CompressionKind kind,
      int32_t compressionLevel);

  /// Return true if support for indicated kind has been enabled.
  static bool isAvailable(CompressionKind kind);

  /// Return true if indicated kind supports extracting uncompressed length
  /// from compressed data.
  static bool supportsGetUncompressedLength(CompressionKind kind);

  /// Return true if indicated kind supports setting a compression level.
  static bool supportsCompressionLevel(CompressionKind kind);

  /// Return true if indicated kind supports creating streaming de/compressor.
  static bool supportsStreamingCompression(CompressionKind kind);

  /// Return the smallest supported compression level for the kind
  /// Note: This function creates a temporary Codec instance.
  static int32_t minimumCompressionLevel(CompressionKind kind);

  /// Return the largest supported compression level for the kind
  /// Note: This function creates a temporary Codec instance.
  static int32_t maximumCompressionLevel(CompressionKind kind);

  /// Return the default compression level.
  /// Note: This function creates a temporary Codec instance.
  static int32_t defaultCompressionLevel(CompressionKind kind);

  /// Return the smallest supported compression level.
  virtual int32_t minimumCompressionLevel() const = 0;

  /// Return the largest supported compression level.
  virtual int32_t maximumCompressionLevel() const = 0;

  /// Return the default compression level.
  virtual int32_t defaultCompressionLevel() const = 0;

  /// One-shot decompression function.
  /// `outputLength` must be correct and therefore be obtained in advance.
  /// The actual decompressed length is returned.
  /// Note: One-shot decompression is not always compatible with streaming
  /// compression. Depending on the codec (e.g. LZ4), different formats may
  /// be used.
  virtual uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) = 0;

  /// Performs one-shot compression.
  /// `outputLength` must first have been computed using maxCompressedLength().
  /// The actual compressed length is returned.
  /// Note: One-shot compression is not always compatible with streaming
  /// decompression. Depending on the codec (e.g. LZ4), different formats may
  /// be used.
  virtual uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) = 0;

  /// Performs one-shot compression.
  /// This function compresses data and writes the output up to the specified
  /// outputLength. If outputLength is too small to hold all the compressed
  /// data, the function doesn't fail. Instead, it returns the number of bytes
  /// actually written to the output buffer. Any remaining data that couldn't
  /// be written in this call will be written in subsequent calls to this
  /// function. This is useful when fixed-size compression blocks are required
  /// by the caller.
  /// Note: Only Gzip and Zstd codec supports this function.
  virtual uint64_t compressPartial(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output);

  /// Maximum compressed length of given input length.
  virtual uint64_t maxCompressedLength(uint64_t inputLength) = 0;

  /// Extracts the uncompressed length from the compressed data if possible.
  /// If the codec doesn't store the uncompressed length, or the data is
  /// corrupted it returns the given uncompressedLength.
  /// If the uncompressed length is stored in the compressed data and
  /// uncompressedLength is not none and they do not match a std::runtime_error
  /// is thrown.
  std::optional<uint64_t> getUncompressedLength(
      uint64_t inputLength,
      const uint8_t* input,
      std::optional<uint64_t> uncompressedLength = std::nullopt) const;

  /// Create a streaming compressor instance.
  virtual std::shared_ptr<Compressor> makeCompressor() = 0;

  /// Create a streaming compressor instance.
  virtual std::shared_ptr<Decompressor> makeDecompressor() = 0;

  /// This Codec's compression type.
  virtual CompressionKind compressionKind() const = 0;

  /// The name of this Codec's compression type.
  std::string name() const {
    return compressionKindToString(compressionKind());
  }

  /// This Codec's compression level, if applicable.
  virtual int32_t compressionLevel() const {
    return kUseDefaultCompressionLevel;
  }

 private:
  /// Initializes the codec's resources.
  virtual void init();

  virtual std::optional<uint64_t> doGetUncompressedLength(
      uint64_t inputLength,
      const uint8_t* input,
      std::optional<uint64_t> uncompressedLength) const;
};
} // namespace facebook::velox::common
