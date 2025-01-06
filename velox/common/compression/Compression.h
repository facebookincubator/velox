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

#pragma once

#include <fmt/format.h>
#include <folly/Expected.h>
#include <folly/compression/Compression.h>
#include <string>

#include "velox/common/base/Status.h"

namespace facebook::velox::common {

enum CompressionKind {
  CompressionKind_NONE = 0,
  CompressionKind_ZLIB = 1,
  CompressionKind_SNAPPY = 2,
  CompressionKind_LZO = 3,
  CompressionKind_ZSTD = 4,
  CompressionKind_LZ4 = 5,
  CompressionKind_GZIP = 6,
  CompressionKind_MAX = INT64_MAX
};

std::unique_ptr<folly::io::Codec> compressionKindToCodec(CompressionKind kind);

CompressionKind codecTypeToCompressionKind(folly::io::CodecType type);

/// Get the name of the CompressionKind.
std::string compressionKindToString(CompressionKind kind);

CompressionKind stringToCompressionKind(const std::string& kind);

constexpr uint64_t DEFAULT_COMPRESSION_BLOCK_SIZE = 256 * 1024;

static constexpr int32_t kUseDefaultCompressionLevel =
    std::numeric_limits<int32_t>::min();

class StreamingCompressor {
 public:
  virtual ~StreamingCompressor() = default;

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
  /// If CompressResult.outputTooSmall is true on return, then a larger output
  /// buffer should be supplied.
  virtual Expected<CompressResult> compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) = 0;

  /// Flush part of the compressed output.
  /// If FlushResult.outputTooSmall is true on return, flush() should be called
  /// again with a larger buffer.
  virtual Expected<FlushResult> flush(
      uint8_t* output,
      uint64_t outputLength) = 0;

  /// End compressing, doing whatever is necessary to end the stream.
  /// If EndResult.outputTooSmall is true on return, end() should be called
  /// again with a larger buffer. Otherwise, the StreamingCompressor should not
  /// be used anymore. end() will flush the compressed output.
  virtual Expected<EndResult> end(uint8_t* output, uint64_t outputLength) = 0;
};

class StreamingDecompressor {
 public:
  virtual ~StreamingDecompressor() = default;

  struct DecompressResult {
    uint64_t bytesRead;
    uint64_t bytesWritten;
    bool outputTooSmall;
  };

  /// Decompress some input.
  /// If outputTooSmall is true on return, a larger output buffer needs
  /// to be supplied.
  virtual Expected<DecompressResult> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) = 0;

  // Return whether the compressed stream is finished.
  virtual bool isFinished() = 0;

  // Reinitialize decompressor, making it ready for a new compressed stream.
  virtual Status reset() = 0;
};

struct CodecOptions {
  int32_t compressionLevel;

  CodecOptions(int32_t compressionLevel = kUseDefaultCompressionLevel)
      : compressionLevel(compressionLevel) {}

  virtual ~CodecOptions() = default;
};

class Codec {
 public:
  virtual ~Codec() = default;

  // Create a kind for the given compression algorithm with CodecOptions.
  static Expected<std::unique_ptr<Codec>> create(
      CompressionKind kind,
      const CodecOptions& codecOptions = CodecOptions{});

  // Create a kind for the given compression algorithm.
  static Expected<std::unique_ptr<Codec>> create(
      CompressionKind kind,
      int32_t compressionLevel);

  // Return true if support for indicated kind has been enabled.
  static bool isAvailable(CompressionKind kind);

  /// Return true if indicated kind supports extracting uncompressed length
  /// from compressed data.
  static bool supportsGetUncompressedLength(CompressionKind kind);

  /// Return true if indicated kind supports one-shot compression with fixed
  /// compressed length.
  static bool supportsCompressFixedLength(CompressionKind kind);

  // Return true if indicated kind supports creating streaming de/compressor.
  static bool supportsStreamingCompression(CompressionKind kind);

  /// Return the smallest supported compression level.
  /// If the codec doesn't support compression level,
  /// `kUseDefaultCompressionLevel` will be returned.
  virtual int32_t minimumCompressionLevel() const = 0;

  /// Return the largest supported compression level.
  /// If the codec doesn't support compression level,
  /// `kUseDefaultCompressionLevel` will be returned.
  virtual int32_t maximumCompressionLevel() const = 0;

  /// Return the default compression level.
  /// If the codec doesn't support compression level,
  /// `kUseDefaultCompressionLevel` will be returned.
  virtual int32_t defaultCompressionLevel() const = 0;

  /// Performs one-shot compression.
  /// `outputLength` must first have been computed using maxCompressedLength().
  /// The actual compressed length will be written to actualOutputLength.
  /// Note: One-shot compression is not always compatible with streaming
  /// decompression. Depending on the codec (e.g. LZ4), different formats may
  /// be used.
  virtual Expected<uint64_t> compress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) = 0;

  /// One-shot decompression function.
  /// `outputLength` must be correct and therefore be obtained in advance.
  /// The actual decompressed length is returned.
  /// Note: One-shot decompression is not always compatible with streaming
  /// compression. Depending on the codec (e.g. LZ4), different formats may
  /// be used.
  virtual Expected<uint64_t> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) = 0;

  /// Performs one-shot compression.
  /// This function compresses data and writes the output up to the specified
  /// outputLength. If outputLength is too small to hold all the compressed
  /// data, the function doesn't fail. Instead, it returns the number of bytes
  /// actually written to the output buffer. Any remaining data that couldn't
  /// be written in this call will be written in subsequent calls to this
  /// function. This is useful when fixed-length compression blocks are required
  /// by the caller.
  /// Note: Only Gzip and Zstd codec supports this function.
  virtual Expected<uint64_t> compressFixedLength(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength);

  // Maximum compressed length of given input length.
  virtual uint64_t maxCompressedLength(uint64_t inputLength) = 0;

  /// Retrieves the actual uncompressed length of data using the specified
  /// compression library.
  /// Note: This functionality is not universally supported by all compression
  /// libraries. If not supported, `std::nullopt` will be returned.
  virtual std::optional<uint64_t> getUncompressedLength(
      const uint8_t* input,
      uint64_t inputLength) const;

  // Create a streaming compressor instance.
  virtual Expected<std::shared_ptr<StreamingCompressor>>
  makeStreamingCompressor();

  // Create a streaming compressor instance.
  virtual Expected<std::shared_ptr<StreamingDecompressor>>
  makeStreamingDecompressor();

  // This Codec's compression type.
  virtual CompressionKind compressionKind() const = 0;

  // This Codec's compression level, if applicable.
  virtual int32_t compressionLevel() const;

  // The name of this Codec's compression type.
  std::string name() const;

 private:
  // Initializes the codec's resources.
  virtual Status init();
};
} // namespace facebook::velox::common

template <>
struct fmt::formatter<facebook::velox::common::CompressionKind>
    : fmt::formatter<std::string> {
  auto format(
      const facebook::velox::common::CompressionKind& s,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::common::compressionKindToString(s), ctx);
  }
};
