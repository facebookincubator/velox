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
#include "velox/common/compression/v2/Compression.h"
#include "velox/common/compression/v2/HadoopCompressionFormat.h"

namespace facebook::velox::common {

class LzoCodec : public Codec {
 public:
  LzoCodec();

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
};

class LzoHadoopCodec : public LzoCodec, public HadoopCompressionFormat {
 public:
  LzoHadoopCodec();

  CompressionKind compressionKind() const;

 private:
  uint64_t decompressInternal(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output);
};

// Lzo format codec.
std::unique_ptr<Codec> makeLzoCodec();

// Lzo "Hadoop" format codec.
std::unique_ptr<Codec> makeLzoHadoopCodec();
} // namespace facebook::velox::common
