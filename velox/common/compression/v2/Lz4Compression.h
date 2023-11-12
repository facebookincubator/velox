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

#include <lz4.h>
#include <lz4frame.h>
#include <lz4hc.h>
#include <memory>
#include "velox/common/compression/v2/Compression.h"
#include "velox/common/compression/v2/HadoopCompressionFormat.h"

namespace facebook::velox::common {

static constexpr int32_t kLz4DefaultCompressionLevel = 1;
static constexpr int32_t kLz4MinCompressionLevel = 1;

class Lz4CodecBase : public Codec {
 public:
  explicit Lz4CodecBase(int32_t compressionLevel);

  int32_t minimumCompressionLevel() const override;

  int32_t maximumCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  int32_t compressionLevel() const override;

 protected:
  const int compressionLevel_;
};

class Lz4FrameCodec : public Lz4CodecBase {
 public:
  explicit Lz4FrameCodec(int32_t compressionLevel);

  uint64_t maxCompressedLength(uint64_t inputLength) override;

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  std::shared_ptr<Compressor> makeCompressor() override;

  std::shared_ptr<Decompressor> makeDecompressor() override;

  CompressionKind compressionKind() const override;

 protected:
  const LZ4F_preferences_t prefs_;
};

class Lz4RawCodec : public Lz4CodecBase {
 public:
  explicit Lz4RawCodec(int32_t compressionLevel);

  uint64_t maxCompressedLength(uint64_t inputLength) override;

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  std::shared_ptr<Compressor> makeCompressor() override;

  std::shared_ptr<Decompressor> makeDecompressor() override;

  CompressionKind compressionKind() const override;
};

class Lz4HadoopCodec : public Lz4RawCodec, public HadoopCompressionFormat {
 public:
  Lz4HadoopCodec();

  uint64_t maxCompressedLength(uint64_t inputLength) override;

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;

  std::shared_ptr<Compressor> makeCompressor() override;

  std::shared_ptr<Decompressor> makeDecompressor() override;

  CompressionKind compressionKind() const override;

  int32_t minimumCompressionLevel() const override;

  int32_t maximumCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

 private:
  uint64_t decompressInternal(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override;
};

// Lz4 frame format codec.
std::unique_ptr<Codec> makeLz4FrameCodec(
    int32_t compressionLevel = kLz4DefaultCompressionLevel);

// Lz4 "raw" format codec.
std::unique_ptr<Codec> makeLz4RawCodec(
    int32_t compressionLevel = kLz4DefaultCompressionLevel);

// Lz4 "Hadoop" format codec (== Lz4 raw codec prefixed with lengths header)
std::unique_ptr<Codec> makeLz4HadoopRawCodec();
} // namespace facebook::velox::common
