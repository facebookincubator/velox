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

#include <zconf.h>
#include <zlib.h>
#include <memory>
#include "velox/common/compression/v2/Compression.h"

namespace facebook::velox::common {

enum GzipFormat {
  kZlib,
  kDeflate,
  kGzip,
};

// Compression levels.
constexpr int32_t kGzipDefaultCompressionLevel = 9;
constexpr int32_t kGZipMinCompressionLevel = 1;
constexpr int32_t kGZipMaxCompressionLevel = 9;

// Maximum window size.
static constexpr int32_t kGzipMaxWindowBits = 15;
// Minimum window size.
static constexpr int32_t kGzipMinWindowBits = 9;
// Default window size.
static constexpr int32_t kGzipDefaultWindowBits = 15;
// 4KB window size.
static constexpr int32_t kGzip4KBWindowBits = 12;

class GzipCodecOptions : public CodecOptions {
 public:
  explicit GzipCodecOptions(
      int32_t compressionLevel = kUseDefaultCompressionLevel,
      GzipFormat format = GzipFormat::kGzip,
      std::optional<int32_t> windowBits = kGzipDefaultWindowBits)
      : CodecOptions(compressionLevel),
        format(format),
        windowBits(windowBits) {}

  GzipFormat format;
  std::optional<int32_t> windowBits;
};

class GzipCodec : public Codec {
 public:
  GzipCodec(int32_t compressionLevel, GzipFormat format, int32_t windowBits);

  ~GzipCodec() override;

  std::shared_ptr<Compressor> makeCompressor() override;

  std::shared_ptr<Decompressor> makeDecompressor() override;

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  uint64_t maxCompressedLength(uint64_t inputLength) override;

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  uint64_t compressPartial(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  void init() override;

  CompressionKind compressionKind() const override;

  int32_t compressionLevel() const override;

  int32_t minimumCompressionLevel() const override;

  int32_t maximumCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

 private:
  void initCompressor();

  void endCompressor();

  void initDecompressor();

  void endDecompressor();

  // zlib is stateful and the z_stream state variable must be initialized
  // before
  z_stream stream_{};

  // Realistically, this will always be GZIP, but we leave the option open to
  // configure
  GzipFormat format_;

  int32_t windowBits_;
  int32_t compressionLevel_;
  // These variables are mutually exclusive. When the codec is in "compressor"
  // state, compressorInitialized_ is true while decompressorInitialized_ is
  // false. When it's decompressing, the opposite is true.
  bool compressorInitialized_{false};
  bool decompressorInitialized_{false};
};

std::unique_ptr<Codec> makeGzipCodec(
    int compressionLevel = kGzipDefaultCompressionLevel,
    GzipFormat format = GzipFormat::kGzip,
    std::optional<int32_t> windowBits = std::nullopt);

std::unique_ptr<Codec> makeZlibCodec(
    int compressionLevel = kGzipDefaultCompressionLevel,
    std::optional<int32_t> windowBits = std::nullopt);

} // namespace facebook::velox::common
