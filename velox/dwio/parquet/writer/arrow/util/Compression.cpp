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

#include "velox/dwio/parquet/writer/arrow/util/Compression.h"

#include <memory>
#include <string>
#include <utility>

#include "arrow/result.h"
#include "arrow/status.h"
#include "velox/dwio/parquet/writer/arrow/util/CompressionInternal.h"

namespace facebook::velox::parquet::arrow::util {

namespace {

Status checkSupportsCompressionLevel(Compression::type type) {
  if (!Codec::supportsCompressionLevel(type)) {
    return Status::Invalid(
        "The specified codec does not support the compression level parameter");
  }
  return Status::OK();
}

} // namespace

int Codec::useDefaultCompressionLevel() {
  return kUseDefaultCompressionLevel;
}

Status Codec::init() {
  return Status::OK();
}

const std::string& Codec::getCodecAsString(Compression::type t) {
  static const std::string uncompressed = "uncompressed", snappy = "snappy",
                           gzip = "gzip", lzo = "lzo", brotli = "brotli",
                           lz4Raw = "lz4_raw", lz4 = "lz4",
                           lz4Hadoop = "lz4_hadoop", zstd = "zstd", bz2 = "bz2",
                           unknown = "unknown";

  switch (t) {
    case Compression::UNCOMPRESSED:
      return uncompressed;
    case Compression::SNAPPY:
      return snappy;
    case Compression::GZIP:
      return gzip;
    case Compression::LZO:
      return lzo;
    case Compression::BROTLI:
      return brotli;
    case Compression::LZ4:
      return lz4Raw;
    case Compression::LZ4_FRAME:
      return lz4;
    case Compression::LZ4_HADOOP:
      return lz4Hadoop;
    case Compression::ZSTD:
      return zstd;
    case Compression::BZ2:
      return bz2;
    default:
      return unknown;
  }
}

Result<Compression::type> Codec::getCompressionType(const std::string& name) {
  if (name == "uncompressed") {
    return Compression::UNCOMPRESSED;
  } else if (name == "gzip") {
    return Compression::GZIP;
  } else if (name == "snappy") {
    return Compression::SNAPPY;
  } else if (name == "lzo") {
    return Compression::LZO;
  } else if (name == "brotli") {
    return Compression::BROTLI;
  } else if (name == "lz4_raw") {
    return Compression::LZ4;
  } else if (name == "lz4") {
    return Compression::LZ4_FRAME;
  } else if (name == "lz4_hadoop") {
    return Compression::LZ4_HADOOP;
  } else if (name == "zstd") {
    return Compression::ZSTD;
  } else if (name == "bz2") {
    return Compression::BZ2;
  } else {
    return Status::Invalid("Unrecognized compression type: ", name);
  }
}

bool Codec::supportsCompressionLevel(Compression::type codec) {
  switch (codec) {
    case Compression::GZIP:
    case Compression::BROTLI:
    case Compression::ZSTD:
    case Compression::BZ2:
    case Compression::LZ4_FRAME:
    case Compression::LZ4:
      return true;
    default:
      return false;
  }
}

Result<int> Codec::maximumCompressionLevel(Compression::type codecType) {
  RETURN_NOT_OK(checkSupportsCompressionLevel(codecType));
  ARROW_ASSIGN_OR_RAISE(auto codec, Codec::create(codecType));
  return codec->maximumCompressionLevel();
}

Result<int> Codec::minimumCompressionLevel(Compression::type codecType) {
  RETURN_NOT_OK(checkSupportsCompressionLevel(codecType));
  ARROW_ASSIGN_OR_RAISE(auto codec, Codec::create(codecType));
  return codec->minimumCompressionLevel();
}

Result<int> Codec::defaultCompressionLevel(Compression::type codecType) {
  RETURN_NOT_OK(checkSupportsCompressionLevel(codecType));
  ARROW_ASSIGN_OR_RAISE(auto codec, Codec::create(codecType));
  return codec->defaultCompressionLevel();
}

Result<std::unique_ptr<Codec>> Codec::create(
    Compression::type codecType,
    const CodecOptions& codecOptions) {
  if (!isAvailable(codecType)) {
    if (codecType == Compression::LZO) {
      return Status::NotImplemented("LZO codec not implemented");
    }

    auto name = getCodecAsString(codecType);
    if (name == "unknown") {
      return Status::Invalid("Unrecognized codec");
    }

    return Status::NotImplemented(
        "Support for codec '", getCodecAsString(codecType), "' not built");
  }

  auto compressionLevel = codecOptions.compressionLevel;
  if (compressionLevel != kUseDefaultCompressionLevel &&
      !supportsCompressionLevel(codecType)) {
    return Status::Invalid(
        "Codec '",
        getCodecAsString(codecType),
        "' doesn't support setting a compression level.");
  }

  std::unique_ptr<Codec> codec;
  switch (codecType) {
    case Compression::UNCOMPRESSED:
      return nullptr;
    case Compression::SNAPPY:
      codec = internal::makeSnappyCodec();
      break;
    case Compression::GZIP: {
      auto opt = dynamic_cast<const GZipCodecOptions*>(&codecOptions);
      codec = internal::makeGZipCodec(
          compressionLevel,
          opt ? opt->gzipFormat : GZipFormat::GZIP,
          opt ? opt->windowBits : std::nullopt);
      break;
    }
    case Compression::BROTLI: {
#ifdef ARROW_WITH_BROTLI
      auto opt = dynamic_cast<const BrotliCodecOptions*>(&codecOptions);
      codec = internal::makeBrotliCodec(
          compressionLevel, opt ? opt->windowBits : std::nullopt);
#endif
      break;
    }
    case Compression::LZ4:
      codec = internal::makeLz4RawCodec(compressionLevel);
      break;
    case Compression::LZ4_FRAME:
      codec = internal::makeLz4FrameCodec(compressionLevel);
      break;
    case Compression::LZ4_HADOOP:
      codec = internal::makeLz4HadoopRawCodec();
      break;
    case Compression::ZSTD:
      codec = internal::makeZSTDCodec(compressionLevel);
      break;
    case Compression::BZ2:
#ifdef ARROW_WITH_BZ2
      codec = internal::makeBZ2Codec(compressionLevel);
#endif
      break;
    default:
      break;
  }

  if (codec == nullptr) {
    return Status::NotImplemented("LZO codec not implemented");
  }

  RETURN_NOT_OK(codec->init());
  return std::move(codec);
}

// Use compression level to create Codec.
Result<std::unique_ptr<Codec>> Codec::create(
    Compression::type codecType,
    int compressionLevel) {
  return Codec::create(codecType, CodecOptions{compressionLevel});
}

bool Codec::isAvailable(Compression::type codecType) {
  switch (codecType) {
    case Compression::UNCOMPRESSED:
    case Compression::SNAPPY:
    case Compression::GZIP:
      return true;
    case Compression::LZO:
      return false;
    case Compression::BROTLI:
#ifdef ARROW_WITH_BROTLI
      return true;
#else
      return false;
#endif
    case Compression::LZ4:
    case Compression::LZ4_FRAME:
    case Compression::LZ4_HADOOP:
      return true;
    case Compression::ZSTD:
      return true;
    case Compression::BZ2:
#ifdef ARROW_WITH_BZ2
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

} // namespace facebook::velox::parquet::arrow::util
