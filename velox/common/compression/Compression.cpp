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

#include "velox/common/compression/Compression.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/Lz4Compression.h"

#include <folly/Conv.h>

namespace facebook::velox::common {

std::unique_ptr<folly::io::Codec> compressionKindToCodec(CompressionKind kind) {
  switch (static_cast<int32_t>(kind)) {
    case CompressionKind_NONE:
      return getCodec(folly::io::CodecType::NO_COMPRESSION);
    case CompressionKind_ZLIB:
      return getCodec(folly::io::CodecType::ZLIB);
    case CompressionKind_SNAPPY:
      return getCodec(folly::io::CodecType::SNAPPY);
    case CompressionKind_ZSTD:
      return getCodec(folly::io::CodecType::ZSTD);
    case CompressionKind_LZ4:
      return getCodec(folly::io::CodecType::LZ4);
    case CompressionKind_GZIP:
      return getCodec(folly::io::CodecType::GZIP);
    default:
      VELOX_UNSUPPORTED(
          "Not support {} in folly", compressionKindToString(kind));
  }
}

CompressionKind codecTypeToCompressionKind(folly::io::CodecType type) {
  switch (type) {
    case folly::io::CodecType::NO_COMPRESSION:
      return CompressionKind_NONE;
    case folly::io::CodecType::ZLIB:
      return CompressionKind_ZLIB;
    case folly::io::CodecType::SNAPPY:
      return CompressionKind_SNAPPY;
    case folly::io::CodecType::ZSTD:
      return CompressionKind_ZSTD;
    case folly::io::CodecType::LZ4:
      return CompressionKind_LZ4;
    case folly::io::CodecType::GZIP:
      return CompressionKind_GZIP;
    default:
      VELOX_UNSUPPORTED(
          "Not support folly codec type {}", folly::to<std::string>(type));
  }
}

std::string compressionKindToString(CompressionKind kind) {
  switch (static_cast<int32_t>(kind)) {
    case CompressionKind_NONE:
      return "none";
    case CompressionKind_ZLIB:
      return "zlib";
    case CompressionKind_SNAPPY:
      return "snappy";
    case CompressionKind_LZO:
      return "lzo";
    case CompressionKind_ZSTD:
      return "zstd";
    case CompressionKind_LZ4:
      return "lz4";
    case CompressionKind_GZIP:
      return "gzip";
  }
  return folly::to<std::string>("unknown - ", kind);
}

CompressionKind stringToCompressionKind(const std::string& kind) {
  static const std::unordered_map<std::string, CompressionKind>
      stringToCompressionKindMap = {
          {"none", CompressionKind_NONE},
          {"zlib", CompressionKind_ZLIB},
          {"snappy", CompressionKind_SNAPPY},
          {"lzo", CompressionKind_LZO},
          {"zstd", CompressionKind_ZSTD},
          {"lz4", CompressionKind_LZ4},
          {"gzip", CompressionKind_GZIP}};
  auto iter = stringToCompressionKindMap.find(kind);
  if (iter != stringToCompressionKindMap.end()) {
    return iter->second;
  } else {
    VELOX_UNSUPPORTED("Not support compression kind {}", kind);
  }
}

void Codec::init() {}

bool Codec::supportsGetUncompressedLength(CompressionKind kind) {
  // TODO: Return true if it's supported by compression kind.
  return false;
}

bool Codec::supportsStreamingCompression(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_LZ4:
      return true;
    default:
      return false;
  }
}

bool Codec::supportsCompressFixedLength(CompressionKind kind) {
  // TODO: Return true if it's supported by compression kind.
  return false;
}

folly::Expected<int32_t, Status> Codec::maximumCompressionLevel(
    CompressionKind kind) {
  return Codec::create(kind).then(
      [](auto&& codec) { return codec->maximumCompressionLevel(); });
}

folly::Expected<int32_t, Status> Codec::minimumCompressionLevel(
    CompressionKind kind) {
  return Codec::create(kind).then(
      [](auto&& codec) { return codec->minimumCompressionLevel(); });
}

folly::Expected<int32_t, Status> Codec::defaultCompressionLevel(
    CompressionKind kind) {
  return Codec::create(kind).then(
      [](auto&& codec) { return codec->defaultCompressionLevel(); });
}

folly::Expected<std::unique_ptr<Codec>, Status> Codec::create(
    CompressionKind kind,
    const CodecOptions& codecOptions) {
  if (!isAvailable(kind)) {
    auto name = compressionKindToString(kind);
    if (folly::StringPiece({name}).startsWith("unknown")) {
      return folly::makeUnexpected(
          Status::Invalid("Unrecognized codec '{}'", name));
    }
    return folly::makeUnexpected(
        Status::Invalid("Support for codec '{}' not implemented.", name));
  }

  auto compressionLevel = codecOptions.compressionLevel;
  std::unique_ptr<Codec> codec;
  switch (kind) {
    case CompressionKind::CompressionKind_LZ4:
      if (auto options = dynamic_cast<const Lz4CodecOptions*>(&codecOptions)) {
        switch (options->type) {
          case Lz4CodecOptions::kLz4Frame:
            codec = makeLz4FrameCodec(compressionLevel);
            break;
          case Lz4CodecOptions::kLz4Raw:
            codec = makeLz4RawCodec(compressionLevel);
            break;
          case Lz4CodecOptions::kLz4Hadoop:
            codec = makeLz4HadoopCodec();
            break;
        }
      }
      // By default, create LZ4 Frame codec.
      codec = makeLz4FrameCodec(compressionLevel);
      break;
    default:
      break;
  }

  if (codec == nullptr) {
    return folly::makeUnexpected(Status::Invalid(
        "Support for codec '{}' not implemented.",
        compressionKindToString(kind)));
  }

  codec->init();

  return codec;
}

folly::Expected<std::unique_ptr<Codec>, Status> Codec::create(
    CompressionKind kind,
    int32_t compressionLevel) {
  return create(kind, CodecOptions{compressionLevel});
}

bool Codec::isAvailable(CompressionKind kind) {
  switch (kind) {
    case CompressionKind::CompressionKind_NONE:
    case CompressionKind::CompressionKind_LZ4:
      return true;
    case CompressionKind::CompressionKind_SNAPPY:
    case CompressionKind::CompressionKind_GZIP:
    case CompressionKind::CompressionKind_ZLIB:
    case CompressionKind::CompressionKind_ZSTD:
    case CompressionKind::CompressionKind_LZO:
    default:
      return false;
  }
}

std::optional<uint64_t> Codec::getUncompressedLength(
    uint64_t inputLength,
    const uint8_t* input) const {
  return std::nullopt;
}

folly::Expected<uint64_t, Status> Codec::compressFixedLength(
    const uint8_t* input,
    uint64_t inputLength,
    uint8_t* output,
    uint64_t outputLength) {
  return folly::makeUnexpected(
      Status::Invalid("'{}' doesn't support fixed-length compression", name()));
}

folly::Expected<std::shared_ptr<StreamingCompressor>, Status>
Codec::makeStreamingCompressor() {
  return folly::makeUnexpected(Status::Invalid(
      "Streaming compression is unsupported with {} format.", name()));
}

folly::Expected<std::shared_ptr<StreamingDecompressor>, Status>
Codec::makeStreamingDecompressor() {
  return folly::makeUnexpected(Status::Invalid(
      "Streaming decompression is unsupported with {} format.", name()));
}

std::string Codec::name() const {
  return compressionKindToString(compressionKind());
}

int32_t Codec::compressionLevel() const {
  return kUseDefaultCompressionLevel;
}
} // namespace facebook::velox::common
