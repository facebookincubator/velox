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

#include <string>

#include "velox/dwio/common/Common.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/common/wrap/orc-proto-wrapper.h"

namespace facebook::velox::dwrf {

enum DwrfFormat : uint8_t { kDwrf, kOrc };

class ProtoWrapperBase {
 protected:
  DwrfFormat format_;
  void* impl_;

 public:
  ProtoWrapperBase() = delete;

  ProtoWrapperBase(DwrfFormat format, void* impl)
      : format_{format}, impl_{impl} {}

  DwrfFormat format() const {
    return format_;
  }

  inline void* rawProtoPtr() const {
    return impl_;
  }
};

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

class ProtoStripeInformation : public ProtoWrapperBase {
 public:
  explicit ProtoStripeInformation(proto::StripeInformation* si)
      : ProtoWrapperBase(DwrfFormat::kDwrf, si) {}

  explicit ProtoStripeInformation(proto::orc::StripeInformation* si)
      : ProtoWrapperBase(DwrfFormat::kOrc, si) {}

  const proto::StripeInformation* getDwrfPtr() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr();
  }

  const proto::orc::StripeInformation* getOrcPtr() const {
    DWIO_ENSURE(format_ == DwrfFormat::kOrc);
    return orcPtr();
  }

  uint64_t offset() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->offset()
                                        : orcPtr()->offset();
  }

  uint64_t indexLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->indexlength()
                                        : orcPtr()->indexlength();
  }

  uint64_t dataLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->datalength()
                                        : orcPtr()->datalength();
  }

  uint64_t footerLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->footerlength()
                                        : orcPtr()->footerlength();
  }

  uint64_t numberOfRows() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->numberofrows()
                                        : orcPtr()->numberofrows();
  }

  // DWRF-specific fields
  uint64_t rawDataSize() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  int64_t checksum() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->checksum();
  }

  uint64_t groupSize() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->groupsize();
  }

  int keyMetadataSize() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata_size();
  }

  const std::string& keyMetadata(int index) const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata(index);
  }

  const ::google::protobuf::RepeatedPtrField<std::string>& keyMetadata() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata();
  }

 private:
  // private helper with no format checking
  inline const proto::StripeInformation* dwrfPtr() const {
    return reinterpret_cast<proto::StripeInformation*>(rawProtoPtr());
  }

  inline const proto::orc::StripeInformation* orcPtr() const {
    return reinterpret_cast<proto::orc::StripeInformation*>(rawProtoPtr());
  }
};

class ProtoType : public ProtoWrapperBase {
 public:
  explicit ProtoType(proto::Type* t) : ProtoWrapperBase(DwrfFormat::kDwrf, t) {}
  explicit ProtoType(proto::orc::Type* t)
      : ProtoWrapperBase(DwrfFormat::kOrc, t) {}

  const proto::Type* getDwrfPtr() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr();
  }

  const proto::orc::Type* getOrcPtr() const {
    DWIO_ENSURE(format_ == DwrfFormat::kOrc);
    return orcPtr();
  }

  TypeKind kind() const;

  int subtypesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->subtypes_size()
                                        : orcPtr()->subtypes_size();
  }

  const ::google::protobuf::RepeatedField<::google::protobuf::uint32>&
  subtypes() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->subtypes()
                                        : orcPtr()->subtypes();
  }

  uint32_t subtypes(int index) const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->subtypes(index)
                                        : orcPtr()->subtypes(index);
  }

  int fieldNamesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->fieldnames_size()
                                        : orcPtr()->fieldnames_size();
  }

  const ::google::protobuf::RepeatedPtrField<std::string>& fieldNames() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->fieldnames()
                                        : orcPtr()->fieldnames();
  }

  const std::string& fieldNames(int index) const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->fieldnames(index)
                                        : orcPtr()->fieldnames(index);
  }

 private:
  // private helper with no format checking
  inline proto::Type* dwrfPtr() const {
    return reinterpret_cast<proto::Type*>(rawProtoPtr());
  }
  inline proto::orc::Type* orcPtr() const {
    return reinterpret_cast<proto::orc::Type*>(rawProtoPtr());
  }
};

class ProtoUserMetadataItem : public ProtoWrapperBase {
 public:
  explicit ProtoUserMetadataItem(proto::UserMetadataItem* item)
      : ProtoWrapperBase(DwrfFormat::kDwrf, item) {}

  explicit ProtoUserMetadataItem(proto::orc::UserMetadataItem* item)
      : ProtoWrapperBase(DwrfFormat::kOrc, item) {}

  const std::string& name() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->name() : orcPtr()->name();
  }

  const std::string& value() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->value()
                                        : orcPtr()->value();
  }

 private:
  // private helper with no format checking
  inline proto::UserMetadataItem* dwrfPtr() const {
    return reinterpret_cast<proto::UserMetadataItem*>(rawProtoPtr());
  }
  inline proto::orc::UserMetadataItem* orcPtr() const {
    return reinterpret_cast<proto::orc::UserMetadataItem*>(rawProtoPtr());
  }
};

class Footer : public ProtoWrapperBase {
 public:
  explicit Footer(proto::Footer* footer)
      : ProtoWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit Footer(proto::orc::Footer* footer)
      : ProtoWrapperBase(DwrfFormat::kOrc, footer) {}

  const proto::Footer* getDwrfPtr() const {
    DWIO_ENSURE(format_ == kDwrf);
    return reinterpret_cast<proto::Footer*>(rawProtoPtr());
  }

  const proto::orc::Footer* getOrcPtr() const {
    DWIO_ENSURE(format_ == kOrc);
    return reinterpret_cast<proto::orc::Footer*>(rawProtoPtr());
  }

  bool hasHeaderLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_headerlength()
                                        : orcPtr()->has_headerlength();
  }

  uint64_t headerLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->headerlength()
                                        : orcPtr()->headerlength();
  }

  bool hasContentLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_contentlength()
                                        : orcPtr()->has_contentlength();
  }

  uint64_t contentLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->contentlength()
                                        : orcPtr()->contentlength();
  }

  bool hasNumberOfRows() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_numberofrows()
                                        : orcPtr()->has_numberofrows();
  }

  uint64_t numberOfRows() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->numberofrows()
                                        : orcPtr()->numberofrows();
  }

  bool hasRawDataSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_rawdatasize() : false;
  }

  uint64_t rawDataSize() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  bool hasChecksumAlgorithm() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_checksumalgorithm()
                                        : false;
  }

  const proto::ChecksumAlgorithm checksumAlgorithm() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->checksumalgorithm();
  }

  bool hasRowIndexStride() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_rowindexstride()
                                        : false;
  }

  uint32_t rowIndexStride() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->rowindexstride() : 0;
  }

  int stripeCacheOffsetsSize() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets_size();
  }

  const ::google::protobuf::RepeatedField<::google::protobuf::uint32>&
  stripeCacheOffsets() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets();
  }

  // TODO: ORC has not supported column statistics yet
  int statisticsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->statistics_size() : 0;
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::ColumnStatistics>&
  statistics() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->statistics();
  }

  const ::facebook::velox::dwrf::proto::ColumnStatistics& statistics(
      int index) const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->statistics(index);
  }

  // TODO: ORC has not supported encryption yet
  bool hasEncryption() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_encryption() : false;
  }

  const ::facebook::velox::dwrf::proto::Encryption& encryption() const {
    DWIO_ENSURE(format_ == DwrfFormat::kDwrf);
    return dwrfPtr()->encryption();
  }

  int stripesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->stripes_size()
                                        : orcPtr()->stripes_size();
  }

  ProtoStripeInformation stripes(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? ProtoStripeInformation(dwrfPtr()->mutable_stripes(index))
        : ProtoStripeInformation(orcPtr()->mutable_stripes(index));
  }

  int typesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->types_size()
                                        : orcPtr()->types_size();
  }

  ProtoType types(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? ProtoType(dwrfPtr()->mutable_types(index))
        : ProtoType(orcPtr()->mutable_types(index));
  }

  int metadataSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->metadata_size()
                                        : orcPtr()->metadata_size();
  }

  ProtoUserMetadataItem metadata(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? ProtoUserMetadataItem(dwrfPtr()->mutable_metadata(index))
        : ProtoUserMetadataItem(orcPtr()->mutable_metadata(index));
  }

 private:
  // private helper with no format checking
  inline proto::Footer* dwrfPtr() const {
    return reinterpret_cast<proto::Footer*>(rawProtoPtr());
  }
  inline proto::orc::Footer* orcPtr() const {
    return reinterpret_cast<proto::orc::Footer*>(rawProtoPtr());
  }
};

} // namespace facebook::velox::dwrf
