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

#include <memory>
#include "velox/common/compression/Compression.h"
#include "velox/common/compression/HadoopCompressionFormat.h"

namespace facebook::velox::common {

enum class LzoType { kLzo, kLzoHadoop };

struct LzoCodecOptions : CodecOptions {
  explicit LzoCodecOptions(
      LzoType lzoType,
      int32_t compressionLevel = kDefaultCompressionLevel)
      : CodecOptions(compressionLevel), lzoType(lzoType) {}

  LzoType lzoType;
};

class LzoCodec : public Codec {
 public:
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

  CompressionKind compressionKind() const override;

  int32_t minCompressionLevel() const override;

  int32_t maxCompressionLevel() const override;

  int32_t defaultCompressionLevel() const override;

  std::string_view name() const override;
};

class LzoHadoopCodec : public LzoCodec, public HadoopCompressionFormat {
 public:
  std::string_view name() const override;

  Expected<uint64_t> decompress(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;

 private:
  Expected<uint64_t> decompressInternal(
      const uint8_t* input,
      uint64_t inputLength,
      uint8_t* output,
      uint64_t outputLength) override;
};

// Lzo format codec.
std::unique_ptr<Codec> makeLzoCodec();

// Lzo "Hadoop" format codec.
std::unique_ptr<Codec> makeLzoHadoopCodec();
} // namespace facebook::velox::common
