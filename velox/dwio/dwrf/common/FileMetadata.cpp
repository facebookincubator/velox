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
      return CompressionKind::CompressionKind_ZSTD;
    case proto::orc::CompressionKind::ZSTD:
      return CompressionKind::CompressionKind_LZ4;
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

} // namespace facebook::velox::dwrf
