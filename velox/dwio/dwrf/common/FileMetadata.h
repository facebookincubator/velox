#pragma once

#include "velox/dwio/common/Common.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/common/wrap/orc-proto-wrapper.h"

namespace facebook::velox::dwrf {

class PostScript {
 public:
  PostScript(
      uint64_t footerLength,
      dwio::common::CompressionKind compression,
      uint64_t compressionBlockSize,
      uint32_t writerVersion)
      : footerLength_{footerLength},
        compression_{compression},
        compressionBlockSize_{compressionBlockSize},
        writerVersion_{static_cast<WriterVersion>(writerVersion)} {}

  explicit PostScript(const proto::PostScript& ps);

  explicit PostScript(const proto::orc::PostScript& ps);

  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  // General methods
  uint64_t footerLength() const {
    return footerLength_;
  }

  dwio::common::CompressionKind compression() const {
    return compression_;
  }

  uint64_t compressionBlockSize() const {
    return compressionBlockSize_;
  }

  uint32_t writerVersion() const {
    return writerVersion_;
  }

  // DWRF-specific methods
  StripeCacheMode cacheMode() const {
    return cacheMode_;
  }

  uint32_t cacheSize() const {
    return cacheSize_;
  }

 private:
  // General attributes
  dwio::common::FileFormat fileFormat_ = dwio::common::FileFormat::DWRF;
  uint64_t footerLength_;
  dwio::common::CompressionKind compression_ =
      dwio::common::CompressionKind::CompressionKind_NONE;
  uint64_t compressionBlockSize_ = dwio::common::DEFAULT_COMPRESSION_BLOCK_SIZE;
  WriterVersion writerVersion_ = WriterVersion::ORIGINAL;

  // DWRF-specific attributes
  StripeCacheMode cacheMode_;
  uint32_t cacheSize_ = 0;

  // ORC-specific attributes
  uint64_t metadataLength_;
  uint64_t stripeStatisticsLength_;
};
} // namespace facebook::velox::dwrf
