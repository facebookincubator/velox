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

#include "velox/common/compression/v2/Compression.h"
#include <memory>
#include <string>
#include <utility>
#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/v2/GzipCompression.h"
#include "velox/common/compression/v2/Lz4Compression.h"
#include "velox/common/compression/v2/LzoCompression.h"
#include "velox/common/compression/v2/SnappyCompression.h"
#include "velox/common/compression/v2/ZstdCompression.h"

#ifdef VELOX_ENABLE_IAA
#include "velox/common/compression/v2/iaa/IaaCompression.h"
#endif

namespace facebook::velox::common {

namespace {
void checkSupportsCompressionLevel(CompressionKind kind) {
  VELOX_USER_CHECK(
      Codec::supportsCompressionLevel(kind),
      "Codec '" + compressionKindToString(kind) +
          "' doesn't support setting a compression level.");
}
} // namespace

int32_t Codec::useDefaultCompressionLevel() {
  return kUseDefaultCompressionLevel;
}

void Codec::init() {}

bool Codec::supportsGetUncompressedLength(CompressionKind kind) {
  switch (kind) {
    case CompressionKind_ZSTD:
    case CompressionKind_SNAPPY:
      return true;
    default:
      return false;
  }
}

bool Codec::supportsCompressionLevel(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_LZ4:
    case CompressionKind::CompressionKind_LZ4RAW:
    case CompressionKind::CompressionKind_GZIP:
    case CompressionKind::CompressionKind_ZLIB:
    case CompressionKind::CompressionKind_ZSTD:
      return true;
    default:
      return false;
  }
}

bool Codec::supportsStreamingCompression(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_LZ4:
    case CompressionKind::CompressionKind_GZIP:
    case CompressionKind::CompressionKind_ZLIB:
    case CompressionKind::CompressionKind_ZSTD:
      return true;
    default:
      return false;
  }
}

int32_t Codec::maximumCompressionLevel(CompressionKind kind) {
  checkSupportsCompressionLevel(kind);
  auto codec = Codec::create(kind);
  return codec->maximumCompressionLevel();
}

int32_t Codec::minimumCompressionLevel(CompressionKind kind) {
  checkSupportsCompressionLevel(kind);
  auto codec = Codec::create(kind);
  return codec->minimumCompressionLevel();
}

int32_t Codec::defaultCompressionLevel(CompressionKind kind) {
  checkSupportsCompressionLevel(kind);
  auto codec = Codec::create(kind);
  return codec->defaultCompressionLevel();
}

std::unique_ptr<Codec> Codec::create(
    CompressionKind kind,
    const CodecOptions& codecOptions) {
  if (!isAvailable(kind)) {
    auto name = compressionKindToString(kind);
    if (folly::StringPiece({name}).startsWith("unknown")) {
      VELOX_UNSUPPORTED("Unrecognized codec '{}'", name);
    }
    VELOX_UNSUPPORTED("Support for codec '{}' not implemented.", name);
  }

  auto compressionLevel = codecOptions.compressionLevel;
  if (compressionLevel != kUseDefaultCompressionLevel) {
    checkSupportsCompressionLevel(kind);
  }

  std::unique_ptr<Codec> codec;
  switch (kind) {
    case CompressionKind::CompressionKind_NONE:
      return nullptr;
    case CompressionKind::CompressionKind_LZ4:
      codec = makeLz4FrameCodec(compressionLevel);
      break;
    case CompressionKind::CompressionKind_LZ4RAW:
      codec = makeLz4RawCodec(compressionLevel);
      break;
    case CompressionKind::CompressionKind_LZ4HADOOP:
      codec = makeLz4HadoopRawCodec();
      break;
    case CompressionKind::CompressionKind_GZIP: {
      if (auto opt = dynamic_cast<const GzipCodecOptions*>(&codecOptions)) {
        codec = makeGzipCodec(compressionLevel, opt->format, opt->windowBits);
        break;
      }
#ifdef VELOX_ENABLE_IAA
      if (auto opt =
              dynamic_cast<const iaa::IaaGzipCodecOptions*>(&codecOptions)) {
        codec = iaa::makeIaaGzipCodec(compressionLevel, opt->maxJobNumber);
        break;
      }
#endif
      codec = makeGzipCodec(compressionLevel);
      break;
    }
    case CompressionKind::CompressionKind_ZLIB: {
      auto opt = dynamic_cast<const GzipCodecOptions*>(&codecOptions);
      if (opt) {
        codec = makeZlibCodec(compressionLevel, opt->windowBits);
        break;
      }
      codec = makeZlibCodec(compressionLevel);
      break;
    }
    case CompressionKind::CompressionKind_ZSTD:
      codec = makeZstdCodec(compressionLevel);
      break;
    case CompressionKind::CompressionKind_SNAPPY:
      codec = makeSnappyCodec();
      break;
    case CompressionKind::CompressionKind_LZO:
      codec = makeLzoCodec();
    default:
      break;
  }

  codec->init();

  return codec;
}

// use compression level to create Codec
std::unique_ptr<Codec> Codec::create(
    CompressionKind kind,
    int32_t compressionLevel) {
  return create(kind, CodecOptions{compressionLevel});
}

bool Codec::isAvailable(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_NONE:
    case CompressionKind::CompressionKind_LZ4:
    case CompressionKind::CompressionKind_LZ4RAW:
    case CompressionKind::CompressionKind_LZ4HADOOP:
    case CompressionKind::CompressionKind_GZIP:
    case CompressionKind::CompressionKind_ZLIB:
    case CompressionKind::CompressionKind_ZSTD:
    case CompressionKind::CompressionKind_SNAPPY:
    case CompressionKind::CompressionKind_LZO:
      return true;
    default:
      return false;
  }
}

std::optional<uint64_t> Codec::getUncompressedLength(
    uint64_t inputLength,
    const uint8_t* input,
    std::optional<uint64_t> uncompressedLength) const {
  if (inputLength == 0) {
    if (uncompressedLength.value_or(0) != 0) {
      VELOX_USER_CHECK_EQ(
          uncompressedLength.value_or(0),
          0,
          "Invalid uncompressed length: {}.",
          *uncompressedLength);
    }
    return 0;
  }
  auto actualLength =
      doGetUncompressedLength(inputLength, input, uncompressedLength);
  if (actualLength) {
    if (uncompressedLength) {
      VELOX_USER_CHECK_EQ(
          *actualLength,
          *uncompressedLength,
          "Invalid uncompressed length: {}.",
          *uncompressedLength);
    }
    return actualLength;
  }
  return uncompressedLength;
}

std::optional<uint64_t> Codec::doGetUncompressedLength(
    uint64_t inputLength,
    const uint8_t* input,
    std::optional<uint64_t> uncompressedLength) const {
  return uncompressedLength;
}

uint64_t Codec::compressPartial(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output) {
  VELOX_UNSUPPORTED("'{}' doesn't support partial compression", name());
}

bool AsyncCodec::isAvailable(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_NONE:
    case CompressionKind::CompressionKind_GZIP:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<AsyncCodec> AsyncCodec::create(
    CompressionKind kind,
    const CodecOptions& codecOptions) {
  if (!isAvailable(kind)) {
    auto name = compressionKindToString(kind);
    if (folly::StringPiece({name}).startsWith("unknown")) {
      VELOX_UNSUPPORTED("Unrecognized codec '{}'", name);
    }
    VELOX_UNSUPPORTED("Support for codec '{}' not implemented.", name);
  }

  auto compressionLevel = codecOptions.compressionLevel;
  if (compressionLevel != kUseDefaultCompressionLevel) {
    checkSupportsCompressionLevel(kind);
  }

  std::unique_ptr<AsyncCodec> codec;
  switch (kind) {
    case CompressionKind::CompressionKind_NONE:
      return nullptr;
    case CompressionKind::CompressionKind_GZIP:
#ifdef VELOX_ENABLE_IAA
      if (auto opt =
              dynamic_cast<const iaa::IaaGzipCodecOptions*>(&codecOptions)) {
        codec = iaa::makeIaaGzipAsyncCodec(compressionLevel, opt->maxJobNumber);
        break;
      }
#endif
      return nullptr;
    default:
      VELOX_UNREACHABLE("Unknown compression kind: {}", kind);
  }

  codec->init();
  return codec;
}
} // namespace facebook::velox::common
