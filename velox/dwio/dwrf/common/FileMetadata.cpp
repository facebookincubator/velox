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
#include "velox/dwio/dwrf/common/FileMetadata.h"

namespace facebook::velox::dwrf {
namespace detail {
using dwio::common::CompressionKind;

CompressionKind orcCompressionToCompressionKind(
    proto::orc::CompressionKind compression) {
  switch (compression) {
    case proto::orc::CompressionKind::NONE:
      return CompressionKind::CompressionKind_NONE;
    case proto::orc::CompressionKind::ZLIB:
      return CompressionKind::CompressionKind_ZLIB;
    case proto::orc::CompressionKind::SNAPPY:
      return CompressionKind::CompressionKind_SNAPPY;
    case proto::orc::CompressionKind::LZO:
      return CompressionKind::CompressionKind_LZO;
    case proto::orc::CompressionKind::LZ4:
      return CompressionKind::CompressionKind_LZ4;
    case proto::orc::CompressionKind::ZSTD:
      return CompressionKind::CompressionKind_ZSTD;
  }
  return CompressionKind::CompressionKind_NONE;
}
} // namespace detail

PostScript::PostScript(const proto::PostScript& ps)
    : footerLength_{ps.footerlength()},
      compression_{
          ps.has_compression()
              ? static_cast<dwio::common::CompressionKind>(ps.compression())
              : dwio::common::CompressionKind::CompressionKind_NONE},
      compressionBlockSize_{
          ps.has_compressionblocksize()
              ? ps.compressionblocksize()
              : dwio::common::DEFAULT_COMPRESSION_BLOCK_SIZE},
      writerVersion_{
          ps.has_writerversion()
              ? static_cast<WriterVersion>(ps.writerversion())
              : WriterVersion::ORIGINAL},
      cacheMode_{static_cast<StripeCacheMode>(ps.cachemode())},
      cacheSize_{ps.cachesize()} {}

PostScript::PostScript(const proto::orc::PostScript& ps)
    : fileFormat_{dwio::common::FileFormat::ORC},
      footerLength_{ps.footerlength()},
      compression_{detail::orcCompressionToCompressionKind(ps.compression())},
      compressionBlockSize_{ps.compressionblocksize()},
      writerVersion_{static_cast<WriterVersion>(ps.writerversion())},
      metadataLength_{ps.metadatalength()},
      stripeStatisticsLength_{ps.stripestatisticslength()} {}

TypeKind ProtoType::kind() const {
  if (format_ == DwrfFormat::kDwrf) {
    switch (dwrfPtr()->kind()) {
      case proto::Type_Kind_BOOLEAN:
      case proto::Type_Kind_BYTE:
      case proto::Type_Kind_SHORT:
      case proto::Type_Kind_INT:
      case proto::Type_Kind_LONG:
      case proto::Type_Kind_FLOAT:
      case proto::Type_Kind_DOUBLE:
      case proto::Type_Kind_STRING:
      case proto::Type_Kind_BINARY:
      case proto::Type_Kind_TIMESTAMP:
        return static_cast<TypeKind>(dwrfPtr()->kind());
      case proto::Type_Kind_LIST:
        return TypeKind::ARRAY;
      case proto::Type_Kind_MAP:
        return TypeKind::MAP;
      case proto::Type_Kind_UNION: {
        DWIO_RAISE("Union type is deprecated!");
      }
      case proto::Type_Kind_STRUCT:
        return TypeKind::ROW;
      default:
        VELOX_FAIL("Unknown type kind");
    }
  }

  switch (orcPtr()->kind()) {
    case proto::orc::Type_Kind_BOOLEAN:
    case proto::orc::Type_Kind_BYTE:
    case proto::orc::Type_Kind_SHORT:
    case proto::orc::Type_Kind_INT:
    case proto::orc::Type_Kind_LONG:
    case proto::orc::Type_Kind_FLOAT:
    case proto::orc::Type_Kind_DOUBLE:
    case proto::orc::Type_Kind_STRING:
    case proto::orc::Type_Kind_BINARY:
    case proto::orc::Type_Kind_TIMESTAMP:
      return static_cast<TypeKind>(orcPtr()->kind());
    case proto::orc::Type_Kind_LIST:
      return TypeKind::ARRAY;
    case proto::orc::Type_Kind_MAP:
      return TypeKind::MAP;
    case proto::orc::Type_Kind_UNION: {
      DWIO_RAISE("Union type is deprecated!");
    }
    case proto::orc::Type_Kind_STRUCT:
      return TypeKind::ROW;
    case proto::orc::Type_Kind_VARCHAR:
      return TypeKind::VARCHAR;
    case proto::orc::Type_Kind_DECIMAL:
    case proto::orc::Type_Kind_DATE:
    case proto::orc::Type_Kind_CHAR:
    case proto::orc::Type_Kind_TIMESTAMP_INSTANT:
      DWIO_RAISE(
          "{} not supported yet.",
          proto::orc::Type_Kind_Name(orcPtr()->kind()));
    default:
      VELOX_FAIL("Unknown type kind");
  }
}

} // namespace facebook::velox::dwrf
