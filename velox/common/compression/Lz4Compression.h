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
#include "velox/common/compression/Compression.h"
#include "velox/common/compression/HadoopCompressionFormat.h"

namespace facebook::velox::common {

struct Lz4CodecOptions : CodecOptions {
  enum Type { kLz4Frame, kLz4Raw, kLz4Hadoop };

  Lz4CodecOptions(
      Lz4CodecOptions::Type type,
      int32_t compressionLevel = kUseDefaultCompressionLevel)
      : CodecOptions(compressionLevel), type(type) {}

  Lz4CodecOptions::Type type;
};

class Lz4CodecBase : public Codec {
 public:
  explicit Lz4CodecBase(int32_t compressionLevel);

  int32_t minimumCompressionLevel() const override;

  int32_t maximumCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  int32_t compressionLevel() const override;

  CompressionKind compressionKind() const override;

 protected:
  const int32_t compressionLevel_;
};

class Lz4FrameCodec : public Lz4CodecBase {
 public:
  explicit Lz4FrameCodec(int32_t compressionLevel);

  uint64_t maxCompressedLength(uint64_t inputLength) override;

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

  Expected<std::shared_ptr<StreamingCompressor>> makeStreamingCompressor()
      override;

  Expected<std::shared_ptr<StreamingDecompressor>> makeStreamingDecompressor()
      override;

 protected:
  const LZ4F_preferences_t prefs_;
};

class Lz4RawCodec : public Lz4CodecBase {
 public:
  explicit Lz4RawCodec(int32_t compressionLevel);

  uint64_t maxCompressedLength(uint64_t inputLength) override;

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
};

/// The Hadoop Lz4Codec source code can be found here:
/// https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-nativetask/src/main/native/src/codec/Lz4Codec.cc
class Lz4HadoopCodec : public Lz4RawCodec, public HadoopCompressionFormat {
 public:
  Lz4HadoopCodec();

  uint64_t maxCompressedLength(uint64_t inputLength) override;

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

  int32_t minimumCompressionLevel() const override;

  int32_t maximumCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

 private:
  Expected<uint64_t> decompressInternal(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;
};

// Lz4 frame format codec.
std::unique_ptr<Codec> makeLz4FrameCodec(
    int32_t compressionLevel = kUseDefaultCompressionLevel);

// Lz4 "raw" format codec.
std::unique_ptr<Codec> makeLz4RawCodec(
    int32_t compressionLevel = kUseDefaultCompressionLevel);

// Lz4 "Hadoop" format codec (Lz4 raw codec prefixed with lengths header).
std::unique_ptr<Codec> makeLz4HadoopCodec();
} // namespace facebook::velox::common
