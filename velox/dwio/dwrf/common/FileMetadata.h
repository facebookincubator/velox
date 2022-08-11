#pragma once

#include "velox/dwio/common/Common.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/common/wrap/orc-proto-wrapper.h"

namespace facebook::velox::dwrf {

enum dwrfFormat { kDwrf, kOrc };

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

class Footer {
  dwrfFormat format_;
  void* impl_;

 public:
  explicit Footer(proto::Footer* footer)
      : format_{dwrfFormat::kDwrf},
        impl_{footer},
        headerLength_{footer->has_headerlength() ? footer->headerlength() : 0},
        contentLength_{
            footer->has_contentlength() ? footer->contentlength() : 0},
        numberOfRows_{footer->has_numberofrows() ? footer->numberofrows() : 0},
        rowIndexStride_{
            footer->has_rowindexstride() ? footer->rowindexstride() : 0},
        rawDataSize_{footer->has_rawdatasize() ? footer->rawdatasize() : 0},
        checksumAlgorithm_{
            footer->has_checksumalgorithm() ? footer->checksumalgorithm()
                                            : proto::ChecksumAlgorithm::NULL_},
        dwrfFooter_{footer} {
    stripeCacheOffsets_.reserve(footer->stripecacheoffsets_size());
    for (const auto offset : footer->stripecacheoffsets()) {
      stripeCacheOffsets_.push_back(offset);
    }
  }

  dwrfFormat format() const {
    return format_;
  }

  const void* rawProtoPtr() const {
    return impl_;
  }

  const proto::Footer* getDwrfPtr() const {
    DWIO_ENSURE(format_ == kDwrf);
    return reinterpret_cast<proto::Footer*>(impl_);
  }

  const proto::orc::Footer* getOrcPtr() const {
    DWIO_ENSURE(format_ == kOrc);
    return reinterpret_cast<proto::orc::Footer*>(impl_);
  }

  bool hasHeaderLength() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_headerlength()
                                        : orcPtr()->has_headerlength();
  }

  uint64_t headerLength() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->headerlength()
                                        : orcPtr()->headerlength();
  }

  bool hasContentLength() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_contentlength()
                                        : orcPtr()->has_contentlength();
  }

  uint64_t contentLength() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->contentlength()
                                        : orcPtr()->contentlength();
  }

  int stripesSize() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->stripes_size()
                                        : orcPtr()->stripes_size();
  }

  bool hasNumberOfRows() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_numberofrows()
                                        : orcPtr()->has_numberofrows();
  }

  uint64_t numberOfRows() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->numberofrows()
                                        : orcPtr()->numberofrows();
  }

  bool hasRawDataSize() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_rawdatasize() : false;
  }

  uint64_t rawDataSize() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  bool hasChecksumAlgorithm() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_checksumalgorithm()
                                        : false;
  }

  const proto::ChecksumAlgorithm checksumAlgorithm() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfPtr()->checksumalgorithm();
  }

  bool hasRowIndexStride() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_rowindexstride()
                                        : false;
  }

  uint32_t rowIndexStride() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfPtr()->rowindexstride();
  }

  int stripeCacheOffsetsSize() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets_size();
  }

  const ::google::protobuf::RepeatedField<::google::protobuf::uint32>&
  stripeCacheOffsets() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets();
  }

  // TODO: ORC has not supported column statistics yet
  int statisticsSize() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->statistics_size() : 0;
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::ColumnStatistics>&
  statistics() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfFooter_->statistics();
  }

  const ::facebook::velox::dwrf::proto::ColumnStatistics& statistics(
      int index) const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfFooter_->statistics(index);
  }

  // TODO: ORC has not supported encryption yet
  bool hasEncryption() const {
    return format_ == dwrfFormat::kDwrf ? dwrfPtr()->has_encryption() : false;
  }

  const ::facebook::velox::dwrf::proto::Encryption& encryption() const {
    DWIO_ENSURE(format_ == dwrfFormat::kDwrf);
    return dwrfFooter_->encryption();
  }

  /***
   * TODO
   ***/
  const ::facebook::velox::dwrf::proto::StripeInformation& stripes(
      int index) const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->stripes(index);
  }

  int typesSize() const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->types_size();
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::Type>&
  types() const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->types();
  }

  const ::facebook::velox::dwrf::proto::Type& types(int index) const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->types(index);
  }

  int metadataSize() const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->metadata_size();
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::UserMetadataItem>&
  metadata() const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->metadata();
  }

  const ::facebook::velox::dwrf::proto::UserMetadataItem& metadata(
      int index) const {
    DWIO_ENSURE(dwrfFooter_);
    return dwrfFooter_->metadata(index);
  }

 private:
  // private helper with no format checking
  inline const proto::Footer* dwrfPtr() const {
    return reinterpret_cast<proto::Footer*>(impl_);
  }
  inline const proto::orc::Footer* orcPtr() const {
    return reinterpret_cast<proto::orc::Footer*>(impl_);
  }

 private:
  uint64_t headerLength_;
  uint64_t contentLength_;
  uint64_t numberOfRows_;
  uint32_t rowIndexStride_;
  // TODO: wrap stripes, types, metadata, column statistics

  // DWRF-specific
  uint64_t rawDataSize_;
  std::vector<uint32_t> stripeCacheOffsets_;
  proto::ChecksumAlgorithm checksumAlgorithm_;
  // TODO: encryption fallback to dwrfFooter_

  // ORC-specific
  // TODO: getter
  uint32_t writer_;
  // TODO: encryption
  proto::orc::CalendarKind calendarKind_;
  std::string softwareVersion_;

  // pointers to format-specific footers
  proto::Footer* dwrfFooter_ = nullptr;
  proto::orc::Footer* orcFooter_ = nullptr;
};

} // namespace facebook::velox::dwrf
