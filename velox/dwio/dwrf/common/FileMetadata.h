/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/Compression.h"
#include "velox/dwio/common/OutputStream.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/common/wrap/orc-proto-wrapper.h"

namespace facebook::velox::dwrf {

class ProtoWrapperBase {
 public:
  DwrfFormat format() const {
    return format_;
  }

  inline const void* rawProtoPtr() const {
    return impl_;
  }

 protected:
  ProtoWrapperBase(DwrfFormat format, const void* impl)
      : format_{format}, impl_{impl} {}

  const DwrfFormat format_;
  const void* const impl_;
};

class ProtoWriteWrapperBase {
 protected:
  ProtoWriteWrapperBase(DwrfFormat format, void* impl)
      : format_{format}, impl_{impl} {}

  DwrfFormat format_;
  void* impl_;

 public:
  DwrfFormat format() const {
    return format_;
  }

  inline void* rawProtoPtr() const {
    return impl_;
  }
};

/***
 * PostScript that takes the ownership of proto::PostScript /
 *proto::orc::PostScript and provides access to the attributes
 ***/
class PostScript {
  DwrfFormat format_;
  std::unique_ptr<google::protobuf::Message> impl_;

 public:
  PostScript() = delete;

  explicit PostScript(std::unique_ptr<proto::PostScript> ps)
      : format_{DwrfFormat::kDwrf}, impl_{std::move(ps)} {}

  explicit PostScript(proto::PostScript&& ps)
      : format_{DwrfFormat::kDwrf},
        impl_{std::make_unique<proto::PostScript>(std::move(ps))} {}

  explicit PostScript(std::unique_ptr<proto::orc::PostScript> ps)
      : format_{DwrfFormat::kOrc}, impl_{std::move(ps)} {}

  explicit PostScript(proto::orc::PostScript&& ps)
      : format_{DwrfFormat::kOrc},
        impl_{std::make_unique<proto::orc::PostScript>(std::move(ps))} {}

  const proto::PostScript* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr();
  }

  const proto::orc::PostScript* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr();
  }

  DwrfFormat format() const {
    return format_;
  }

  // General methods
  bool hasFooterLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_footerlength()
                                        : orcPtr()->has_footerlength();
  }

  uint64_t footerLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->footerlength()
                                        : orcPtr()->footerlength();
  }

  uint64_t metadataLength() const {
    return format_ == DwrfFormat::kDwrf ? 0 : orcPtr()->metadatalength();
  }

  bool hasCompression() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_compression()
                                        : orcPtr()->has_compression();
  }

  common::CompressionKind compression() const;

  bool hasCompressionBlockSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_compressionblocksize()
                                        : orcPtr()->has_compressionblocksize();
  }

  uint64_t compressionBlockSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->compressionblocksize()
                                        : orcPtr()->compressionblocksize();
  }

  bool hasWriterVersion() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_writerversion()
                                        : orcPtr()->has_writerversion();
  }

  uint32_t writerVersion() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->writerversion()
                                        : orcPtr()->writerversion();
  }

  // DWRF-specific methods
  bool hasCacheMode() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_cachemode() : false;
  }

  StripeCacheMode cacheMode() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return static_cast<StripeCacheMode>(dwrfPtr()->cachemode());
  }

  bool hasCacheSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_cachesize() : false;
  }

  uint32_t cacheSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->cachesize();
  }

  PostScript copy() const {
    switch (format_) {
      case DwrfFormat::kDwrf:
        return PostScript(proto::PostScript(*dwrfPtr()));
      case DwrfFormat::kOrc:
        return PostScript(proto::orc::PostScript(*orcPtr()));
    }
  }

 private:
  inline const proto::PostScript* dwrfPtr() const {
    return reinterpret_cast<proto::PostScript*>(impl_.get());
  }

  inline const proto::orc::PostScript* orcPtr() const {
    return reinterpret_cast<proto::orc::PostScript*>(impl_.get());
  }
};

class StripeInformationWrapper : public ProtoWrapperBase {
 public:
  explicit StripeInformationWrapper(const proto::StripeInformation* si)
      : ProtoWrapperBase(DwrfFormat::kDwrf, si) {}

  explicit StripeInformationWrapper(const proto::orc::StripeInformation* si)
      : ProtoWrapperBase(DwrfFormat::kOrc, si) {}

  const proto::StripeInformation* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr();
  }

  const proto::orc::StripeInformation* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
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
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  int64_t checksum() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->checksum();
  }

  uint64_t groupSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->groupsize();
  }

  int keyMetadataSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata_size();
  }

  const std::string& keyMetadata(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata(index);
  }

  const ::google::protobuf::RepeatedPtrField<std::string>& keyMetadata() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->keymetadata();
  }

 private:
  // private helper with no format checking
  inline const proto::StripeInformation* dwrfPtr() const {
    return reinterpret_cast<const proto::StripeInformation*>(rawProtoPtr());
  }

  inline const proto::orc::StripeInformation* orcPtr() const {
    return reinterpret_cast<const proto::orc::StripeInformation*>(
        rawProtoPtr());
  }
};

class ColumnEncodingKindWrapper : public ProtoWrapperBase {
 public:
  explicit ColumnEncodingKindWrapper(proto::ColumnEncoding_Kind* stream)
      : ProtoWrapperBase(DwrfFormat::kDwrf, stream) {}

  explicit ColumnEncodingKindWrapper(proto::orc::ColumnEncoding_Kind* stream)
      : ProtoWrapperBase(DwrfFormat::kOrc, stream) {}
};

class ColumnEncodingWrapper : public ProtoWrapperBase {
 public:
  explicit ColumnEncodingWrapper(const proto::ColumnEncoding* columnEncoding)
      : ProtoWrapperBase(DwrfFormat::kDwrf, columnEncoding) {}
  explicit ColumnEncodingWrapper(
      const proto::orc::ColumnEncoding* columnEncoding)
      : ProtoWrapperBase(DwrfFormat::kOrc, columnEncoding) {}

  void Clear() {}

  proto::ColumnEncoding_Kind kind() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->kind();
  }

  uint32_t node() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->node();
  }

 private:
  // private helper with no format checking
  inline const proto::ColumnEncoding* dwrfPtr() const {
    return reinterpret_cast<const proto::ColumnEncoding*>(rawProtoPtr());
  }

  inline const proto::orc::ColumnEncoding* orcPtr() const {
    return reinterpret_cast<const proto::orc::ColumnEncoding*>(rawProtoPtr());
  }
};

class TypeWrapper : public ProtoWrapperBase {
 public:
  explicit TypeWrapper(const proto::Type* t)
      : ProtoWrapperBase(DwrfFormat::kDwrf, t) {}
  explicit TypeWrapper(const proto::orc::Type* t)
      : ProtoWrapperBase(DwrfFormat::kOrc, t) {}

  const proto::Type* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr();
  }

  const proto::orc::Type* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
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
  inline const proto::Type* dwrfPtr() const {
    return reinterpret_cast<const proto::Type*>(rawProtoPtr());
  }
  inline const proto::orc::Type* orcPtr() const {
    return reinterpret_cast<const proto::orc::Type*>(rawProtoPtr());
  }
};

class UserMetadataItemWrapper : public ProtoWrapperBase {
 public:
  explicit UserMetadataItemWrapper(const proto::UserMetadataItem* item)
      : ProtoWrapperBase(DwrfFormat::kDwrf, item) {}

  explicit UserMetadataItemWrapper(const proto::orc::UserMetadataItem* item)
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
  inline const proto::UserMetadataItem* dwrfPtr() const {
    return reinterpret_cast<const proto::UserMetadataItem*>(rawProtoPtr());
  }
  inline const proto::orc::UserMetadataItem* orcPtr() const {
    return reinterpret_cast<const proto::orc::UserMetadataItem*>(rawProtoPtr());
  }
};

class IntegerStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit IntegerStatisticsWrapper(
      const proto::IntegerStatistics* intStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, intStatistics) {}

  explicit IntegerStatisticsWrapper(
      const proto::orc::IntegerStatistics* intStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, intStatistics) {}

  bool hasMinimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_minimum()
                                        : orcPtr()->has_minimum();
  }

  int64_t minimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->minimum()
                                        : orcPtr()->minimum();
  }

  bool hasMaximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_maximum()
                                        : orcPtr()->has_maximum();
  }

  int64_t maximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->maximum()
                                        : orcPtr()->maximum();
  }

  bool hasSum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_sum()
                                        : orcPtr()->has_sum();
  }

  int64_t sum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->sum() : orcPtr()->sum();
  }

 private:
  // private helper with no format checking
  inline const proto::IntegerStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::IntegerStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::IntegerStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::IntegerStatistics*>(
        rawProtoPtr());
  }
};

class DoubleStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit DoubleStatisticsWrapper(
      const proto::DoubleStatistics* doubleStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, doubleStatistics) {}

  explicit DoubleStatisticsWrapper(
      const proto::orc::DoubleStatistics* doubleStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, doubleStatistics) {}

  bool hasMinimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_minimum()
                                        : orcPtr()->has_minimum();
  }

  double minimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->minimum()
                                        : orcPtr()->minimum();
  }

  bool hasMaximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_maximum()
                                        : orcPtr()->has_maximum();
  }

  double maximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->maximum()
                                        : orcPtr()->maximum();
  }

  bool hasSum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_sum()
                                        : orcPtr()->has_sum();
  }

  double sum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->sum() : orcPtr()->sum();
  }

 private:
  // private helper with no format checking
  inline const proto::DoubleStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::DoubleStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::DoubleStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::DoubleStatistics*>(rawProtoPtr());
  }
};

class StringStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit StringStatisticsWrapper(
      const proto::StringStatistics* stringStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, stringStatistics) {}

  explicit StringStatisticsWrapper(
      const proto::orc::StringStatistics* stringStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, stringStatistics) {}

  bool hasMinimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_minimum()
                                        : orcPtr()->has_minimum();
  }

  const std::string& minimum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->minimum()
                                        : orcPtr()->minimum();
  }

  bool hasMaximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_maximum()
                                        : orcPtr()->has_maximum();
  }

  const std::string& maximum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->maximum()
                                        : orcPtr()->maximum();
  }

  bool hasSum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_sum()
                                        : orcPtr()->has_sum();
  }

  int64_t sum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->sum() : orcPtr()->sum();
  }

 private:
  // private helper with no format checking
  inline const proto::StringStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::StringStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::StringStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::StringStatistics*>(rawProtoPtr());
  }
};

class BucketStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit BucketStatisticsWrapper(
      const proto::BucketStatistics* bucketStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, bucketStatistics) {}

  explicit BucketStatisticsWrapper(
      const proto::orc::BucketStatistics* bucketStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, bucketStatistics) {}

  int countSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->count_size()
                                        : orcPtr()->count_size();
  }

  uint64_t count(int index) const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->count(index)
                                        : orcPtr()->count(index);
  }

 private:
  // private helper with no format checking
  inline const proto::BucketStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::BucketStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::BucketStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::BucketStatistics*>(rawProtoPtr());
  }
};

class BinaryStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit BinaryStatisticsWrapper(
      const proto::BinaryStatistics* binaryStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, binaryStatistics) {}

  explicit BinaryStatisticsWrapper(
      const proto::orc::BinaryStatistics* binaryStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, binaryStatistics) {}

  bool hasSum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_sum()
                                        : orcPtr()->has_sum();
  }

  int64_t sum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->sum() : orcPtr()->sum();
  }

 private:
  // private helper with no format checking
  inline const proto::BinaryStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::BinaryStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::BinaryStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::BinaryStatistics*>(rawProtoPtr());
  }
};

class ColumnStatisticsWrapper : public ProtoWrapperBase {
 public:
  explicit ColumnStatisticsWrapper(
      const proto::ColumnStatistics* columnStatistics)
      : ProtoWrapperBase(DwrfFormat::kDwrf, columnStatistics) {}

  explicit ColumnStatisticsWrapper(
      const proto::orc::ColumnStatistics* columnStatistics)
      : ProtoWrapperBase(DwrfFormat::kOrc, columnStatistics) {}

  bool hasNumberOfValues() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_numberofvalues()
                                        : orcPtr()->has_numberofvalues();
  }

  uint64_t numberOfValues() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->numberofvalues()
                                        : orcPtr()->numberofvalues();
  }

  bool hasHasNull() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_hasnull()
                                        : orcPtr()->has_hasnull();
  }

  bool hasNull() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->hasnull()
                                        : orcPtr()->hasnull();
  }

  bool hasRawSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_rawsize() : false;
  }

  uint64_t rawSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->rawsize() : 0;
  }

  bool hasSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_size() : false;
  }

  uint64_t size() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->size() : 0;
  }

  bool hasIntStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_intstatistics()
                                        : orcPtr()->has_intstatistics();
  }

  IntegerStatisticsWrapper intStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? IntegerStatisticsWrapper(&dwrfPtr()->intstatistics())
        : IntegerStatisticsWrapper(&orcPtr()->intstatistics());
  }

  bool hasDoubleStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_doublestatistics()
                                        : orcPtr()->has_doublestatistics();
  }

  DoubleStatisticsWrapper doubleStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? DoubleStatisticsWrapper(&dwrfPtr()->doublestatistics())
        : DoubleStatisticsWrapper(&orcPtr()->doublestatistics());
  }

  bool hasStringStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_stringstatistics()
                                        : orcPtr()->has_stringstatistics();
  }

  StringStatisticsWrapper stringStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? StringStatisticsWrapper(&dwrfPtr()->stringstatistics())
        : StringStatisticsWrapper(&orcPtr()->stringstatistics());
  }

  bool hasBucketStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_bucketstatistics()
                                        : orcPtr()->has_bucketstatistics();
  }

  BucketStatisticsWrapper bucketStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? BucketStatisticsWrapper(&dwrfPtr()->bucketstatistics())
        : BucketStatisticsWrapper(&orcPtr()->bucketstatistics());
  }

  bool hasBinaryStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_binarystatistics()
                                        : orcPtr()->has_binarystatistics();
  }

  BinaryStatisticsWrapper binaryStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? BinaryStatisticsWrapper(&dwrfPtr()->binarystatistics())
        : BinaryStatisticsWrapper(&orcPtr()->binarystatistics());
  }

  bool hasMapStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_mapstatistics()
                                        : false;
  }

  const ::facebook::velox::dwrf::proto::MapStatistics& mapStatistics() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->mapstatistics();
  }

 private:
  // private helper with no format checking
  inline const proto::ColumnStatistics* dwrfPtr() const {
    return reinterpret_cast<const proto::ColumnStatistics*>(rawProtoPtr());
  }
  inline const proto::orc::ColumnStatistics* orcPtr() const {
    return reinterpret_cast<const proto::orc::ColumnStatistics*>(rawProtoPtr());
  }
};

class FooterWrapper : public ProtoWrapperBase {
 public:
  explicit FooterWrapper(const proto::Footer* footer)
      : ProtoWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit FooterWrapper(const proto::orc::Footer* footer)
      : ProtoWrapperBase(DwrfFormat::kOrc, footer) {}

  const proto::Footer* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return reinterpret_cast<const proto::Footer*>(rawProtoPtr());
  }

  const proto::orc::Footer* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return reinterpret_cast<const proto::orc::Footer*>(rawProtoPtr());
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
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  bool hasChecksumAlgorithm() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_checksumalgorithm()
                                        : false;
  }

  const proto::ChecksumAlgorithm checksumAlgorithm() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->checksumalgorithm();
  }

  bool hasRowIndexStride() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_rowindexstride()
                                        : orcPtr()->has_rowindexstride();
  }

  uint32_t rowIndexStride() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->rowindexstride()
                                        : orcPtr()->rowindexstride();
  }

  int stripeCacheOffsetsSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets_size();
  }

  const ::google::protobuf::RepeatedField<::google::protobuf::uint32>&
  stripeCacheOffsets() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->stripecacheoffsets();
  }

  int statisticsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->statistics_size()
                                        : orcPtr()->statistics_size();
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::ColumnStatistics>&
  statistics() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->statistics();
  }

  const ::facebook::velox::dwrf::proto::ColumnStatistics& dwrfStatistics(
      int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->statistics(index);
  }

  ColumnStatisticsWrapper statistics(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? ColumnStatisticsWrapper(&dwrfPtr()->statistics(index))
        : ColumnStatisticsWrapper(&orcPtr()->statistics(index));
  }

  // TODO: ORC has not supported encryption yet
  bool hasEncryption() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_encryption() : false;
  }

  const ::facebook::velox::dwrf::proto::Encryption& encryption() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->encryption();
  }

  int stripesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->stripes_size()
                                        : orcPtr()->stripes_size();
  }

  StripeInformationWrapper stripes(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? StripeInformationWrapper(&dwrfPtr()->stripes(index))
        : StripeInformationWrapper(&orcPtr()->stripes(index));
  }

  int typesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->types_size()
                                        : orcPtr()->types_size();
  }

  TypeWrapper types(int index) const {
    return format_ == DwrfFormat::kDwrf ? TypeWrapper(&dwrfPtr()->types(index))
                                        : TypeWrapper(&orcPtr()->types(index));
  }

  int metadataSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->metadata_size()
                                        : orcPtr()->metadata_size();
  }

  UserMetadataItemWrapper metadata(int index) const {
    return format_ == DwrfFormat::kDwrf
        ? UserMetadataItemWrapper(&dwrfPtr()->metadata(index))
        : UserMetadataItemWrapper(&orcPtr()->metadata(index));
  }

 private:
  // private helper with no format checking
  inline const proto::Footer* dwrfPtr() const {
    return reinterpret_cast<const proto::Footer*>(rawProtoPtr());
  }
  inline const proto::orc::Footer* orcPtr() const {
    return reinterpret_cast<const proto::orc::Footer*>(rawProtoPtr());
  }
};

class StripeFooterWrapper : public ProtoWrapperBase {
  // Supporting the two following proto definitions:
  //  kOrc
  //  message StripeFooter {
  //    repeated Stream streams = 1;
  //    repeated ColumnEncoding columns = 2;
  //    optional string writerTimezone = 3;
  //    repeated StripeEncryptionVariant encryption = 4;
  //  }
  //
  //  kDwrf
  //  message StripeFooter {
  //    repeated Stream streams = 1;
  //    repeated ColumnEncoding encoding = 2;
  //    repeated bytes encryptionGroups = 3;
  //  }

 public:
  explicit StripeFooterWrapper(
      std::shared_ptr<const proto::StripeFooter> stripeFooter)
      : ProtoWrapperBase(DwrfFormat::kDwrf, stripeFooter.get()),
        dwrfStripeFooter_(std::move(stripeFooter)) {}

  explicit StripeFooterWrapper(
      std::shared_ptr<const proto::orc::StripeFooter> stripeFooter)
      : ProtoWrapperBase(DwrfFormat::kOrc, stripeFooter.get()),
        orcStripeFooter_(std::move(stripeFooter)) {}

  const proto::StripeFooter& getStripeFooterDwrf() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    VELOX_CHECK_NOT_NULL(rawProtoPtr());
    return *reinterpret_cast<const proto::StripeFooter*>(rawProtoPtr());
  }

  const proto::orc::StripeFooter& getStripeFooterOrc() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    VELOX_CHECK_NOT_NULL(rawProtoPtr());
    return *reinterpret_cast<const proto::orc::StripeFooter*>(rawProtoPtr());
  }

  int streamsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->streams_size()
                                        : orcPtr()->streams_size();
  }

  const proto::Stream& streamDwrf(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->streams(index);
  }

  const proto::orc::Stream& streamOrc(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->streams(index);
  }

  const ::google::protobuf::RepeatedPtrField<proto::Stream>& streamsDwrf()
      const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->streams();
  }

  const ::google::protobuf::RepeatedPtrField<proto::orc::Stream>& streamsOrc()
      const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->streams();
  }

  int columnEncodingSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->encoding_size()
                                        : orcPtr()->columns_size();
  }

  const ::google::protobuf::RepeatedPtrField<proto::ColumnEncoding>&
  columnEncodingsDwrf() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->encoding();
  }

  const ::google::protobuf::RepeatedPtrField<proto::orc::ColumnEncoding>&
  columnEncodingsOrc() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->columns();
  }

  const proto::ColumnEncoding& columnEncodingDwrf(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->encoding(index);
  }

  const proto::orc::ColumnEncoding& columnEncodingOrc(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->columns(index);
  }

  int encryptiongroupsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->encryptiongroups_size()
                                        : 0;
  }

  const std::string& encryptiongroupsDwrf(int index) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->encryptiongroups(index);
  }

 private:
  // private helper with no format checking
  inline const proto::StripeFooter* dwrfPtr() const {
    return reinterpret_cast<const proto::StripeFooter*>(rawProtoPtr());
  }
  inline const proto::orc::StripeFooter* orcPtr() const {
    return reinterpret_cast<const proto::orc::StripeFooter*>(rawProtoPtr());
  }

  std::shared_ptr<const proto::StripeFooter> dwrfStripeFooter_ = nullptr;
  std::shared_ptr<const proto::orc::StripeFooter> orcStripeFooter_ = nullptr;
};

class StripeInformationWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit StripeInformationWriteWrapper(
      proto::StripeInformation* stripeInformation)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stripeInformation) {}

  explicit StripeInformationWriteWrapper(
      proto::orc::StripeInformation* stripeInformation)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stripeInformation) {}

  uint64_t numberOfRows() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->numberofrows()
                                        : orcPtr()->numberofrows();
  }

  uint64_t rawDataSize() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->rawdatasize() : 0;
  }

  bool hasChecksum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_checksum() : false;
  }

  uint64_t checksum() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->checksum() : 0;
  }

  void setNumberOfRows(uint64_t stripeRowCount) {
    return format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_numberofrows(stripeRowCount)
        : orcPtr()->set_numberofrows(stripeRowCount);
  }

  void setRawDataSize(uint64_t rawDataSize) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_rawdatasize(rawDataSize);
    }
  }

  void setChecksum(int64_t checksum) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_checksum(checksum);
  }

  void setGroupSize(uint64_t groupSize) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_groupsize(groupSize);
  }

  uint64_t groupSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->groupsize();
  }

  void setOffset(uint64_t offset) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_offset(offset)
                                 : orcPtr()->set_offset(offset);
  }

  uint64_t offset() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->offset()
                                        : orcPtr()->offset();
  }

  void setIndexLength(uint64_t indexLength) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_indexlength(indexLength)
                                 : orcPtr()->set_indexlength(indexLength);
  }

  uint64_t indexLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->indexlength()
                                        : orcPtr()->indexlength();
  }

  void setDataLength(uint64_t dataLength) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_datalength(dataLength)
                                 : orcPtr()->set_datalength(dataLength);
  }

  uint64_t dataLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->datalength()
                                        : orcPtr()->datalength();
  }

  void setFooterLength(uint64_t footerLength) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_footerlength(footerLength)
                                 : orcPtr()->set_footerlength(footerLength);
  }

  uint64_t footerLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->footerlength()
                                        : orcPtr()->footerlength();
  }

  std::string* addKeyMetadata() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->add_keymetadata();
  }

 private:
  // private helper with no format checking
  inline proto::StripeInformation* dwrfPtr() const {
    return reinterpret_cast<proto::StripeInformation*>(rawProtoPtr());
  }
  inline proto::orc::StripeInformation* orcPtr() const {
    return reinterpret_cast<proto::orc::StripeInformation*>(rawProtoPtr());
  }
};

class TypeKindWrapper : public ProtoWriteWrapperBase {
 public:
  explicit TypeKindWrapper(proto::Type_Kind* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit TypeKindWrapper(proto::orc::Type_Kind* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, footer) {}
};

class TypeWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit TypeWriteWrapper(proto::Type* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit TypeWriteWrapper(proto::orc::Type* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, footer) {}

  const proto::Type* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return reinterpret_cast<proto::Type*>(rawProtoPtr());
  }

  const proto::orc::Type* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return reinterpret_cast<proto::orc::Type*>(rawProtoPtr());
  }

  void setKind(TypeKindWrapper typeKindWrapper) {
    format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_kind(*reinterpret_cast<proto::Type_Kind*>(
              typeKindWrapper.rawProtoPtr()))
        : orcPtr()->set_kind(*reinterpret_cast<proto::orc::Type_Kind*>(
              typeKindWrapper.rawProtoPtr()));
  }

  void setScale(uint32_t scale) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    orcPtr()->set_scale(scale);
  }

  void setPrecision(uint32_t precision) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    orcPtr()->set_precision(precision);
  }

  void addFieldnames(const std::string& fieldName) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->add_fieldnames(fieldName)
                                 : orcPtr()->add_fieldnames(fieldName);
  }

  void addSubtypes(int fieldName) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->add_subtypes(fieldName)
                                 : orcPtr()->add_subtypes(fieldName);
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

class UserMetadataItemWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit UserMetadataItemWriteWrapper(
      proto::UserMetadataItem* userMetadataItem)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, userMetadataItem) {}

  explicit UserMetadataItemWriteWrapper(
      proto::orc::UserMetadataItem* userMetadataItem)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, userMetadataItem) {}

  void setName(const std::string& name) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_name(name)
                                 : orcPtr()->set_name(name);
  }

  void setValue(const std::string& value) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_value(value)
                                 : orcPtr()->set_value(value);
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

class BucketStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit BucketStatisticsWriteWrapper(
      proto::BucketStatistics* bucketStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, bucketStatistics) {}

  explicit BucketStatisticsWriteWrapper(
      proto::orc::BucketStatistics* bucketStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, bucketStatistics) {}

  int countSize() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->count_size()
                                        : orcPtr()->count_size();
  }

  void addCount(uint64_t count) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->add_count(count)
                                 : orcPtr()->add_count(count);
  }

 private:
  // private helper with no format checking
  inline proto::BucketStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::BucketStatistics*>(rawProtoPtr());
  }
  inline proto::orc::BucketStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::BucketStatistics*>(rawProtoPtr());
  }
};

class IntegerStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit IntegerStatisticsWriteWrapper(
      proto::IntegerStatistics* integerStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, integerStatistics) {}

  explicit IntegerStatisticsWriteWrapper(
      proto::orc::IntegerStatistics* integerStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, integerStatistics) {}

  void setSum(int64_t sum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_sum(sum)
                                 : orcPtr()->set_sum(sum);
  }

  void setMinimum(int64_t minimum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_minimum(minimum)
                                 : orcPtr()->set_minimum(minimum);
  }

  void setMaximum(int64_t maximum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_maximum(maximum)
                                 : orcPtr()->set_maximum(maximum);
  }

 private:
  // private helper with no format checking
  inline proto::IntegerStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::IntegerStatistics*>(rawProtoPtr());
  }
  inline proto::orc::IntegerStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::IntegerStatistics*>(rawProtoPtr());
  }
};

class DoubleStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit DoubleStatisticsWriteWrapper(
      proto::DoubleStatistics* doubleStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, doubleStatistics) {}

  explicit DoubleStatisticsWriteWrapper(
      proto::orc::DoubleStatistics* doubleStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, doubleStatistics) {}

  void setSum(double sum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_sum(sum)
                                 : orcPtr()->set_sum(sum);
  }

  void setMinimum(double minimum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_minimum(minimum)
                                 : orcPtr()->set_minimum(minimum);
  }

  void setMaximum(double maximum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_maximum(maximum)
                                 : orcPtr()->set_maximum(maximum);
  }

 private:
  // private helper with no format checking
  inline proto::DoubleStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::DoubleStatistics*>(rawProtoPtr());
  }
  inline proto::orc::DoubleStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::DoubleStatistics*>(rawProtoPtr());
  }
};

class StringStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit StringStatisticsWriteWrapper(
      proto::StringStatistics* stringStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stringStatistics) {}

  explicit StringStatisticsWriteWrapper(
      proto::orc::StringStatistics* stringStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stringStatistics) {}

  void setSum(uint64_t sum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_sum(sum)
                                 : orcPtr()->set_sum(sum);
  }

  void setMinimum(std::string minimum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_minimum(minimum)
                                 : orcPtr()->set_minimum(minimum);
  }

  void setMaximum(std::string maximum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_maximum(maximum)
                                 : orcPtr()->set_maximum(maximum);
  }

 private:
  // private helper with no format checking
  inline proto::StringStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::StringStatistics*>(rawProtoPtr());
  }
  inline proto::orc::StringStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::StringStatistics*>(rawProtoPtr());
  }
};

class BinaryStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit BinaryStatisticsWriteWrapper(
      proto::BinaryStatistics* binaryStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, binaryStatistics) {}

  explicit BinaryStatisticsWriteWrapper(
      proto::orc::BinaryStatistics* binaryStatistics)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, binaryStatistics) {}

  void setSum(uint64_t sum) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_sum(sum)
                                 : orcPtr()->set_sum(sum);
  }

 private:
  // private helper with no format checking
  inline proto::BinaryStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::BinaryStatistics*>(rawProtoPtr());
  }
  inline proto::orc::BinaryStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::BinaryStatistics*>(rawProtoPtr());
  }
};

class ColumnStatisticsWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit ColumnStatisticsWriteWrapper(proto::ColumnStatistics* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit ColumnStatisticsWriteWrapper(proto::orc::ColumnStatistics* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, footer) {}

  void setSize(uint64_t size) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_size(size);
    }
  }

  void setHasNull(bool hasNull) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_hasnull(hasNull)
                                 : orcPtr()->set_hasnull(hasNull);
  }

  void setNumberOfValues(uint64_t numberOfValues) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_numberofvalues(numberOfValues)
                                 : orcPtr()->set_numberofvalues(numberOfValues);
  }

  void setRawSize(uint64_t rawSize) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_rawsize(rawSize);
    }
  }

  uint64_t getRawSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->rawsize();
  }

  uint64_t getSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->size();
  }

  bool hasMapStatistics() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->has_mapstatistics();
  }

  proto::MapStatistics* mutableMapStatistics() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->mutable_mapstatistics();
  }

  BinaryStatisticsWriteWrapper mutableBinaryStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? BinaryStatisticsWriteWrapper(dwrfPtr()->mutable_binarystatistics())
        : BinaryStatisticsWriteWrapper(orcPtr()->mutable_binarystatistics());
  }

  StringStatisticsWriteWrapper mutableStringStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? StringStatisticsWriteWrapper(dwrfPtr()->mutable_stringstatistics())
        : StringStatisticsWriteWrapper(orcPtr()->mutable_stringstatistics());
  }

  DoubleStatisticsWriteWrapper mutableDoubleStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? DoubleStatisticsWriteWrapper(dwrfPtr()->mutable_doublestatistics())
        : DoubleStatisticsWriteWrapper(orcPtr()->mutable_doublestatistics());
  }

  IntegerStatisticsWriteWrapper mutableIntegerStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? IntegerStatisticsWriteWrapper(dwrfPtr()->mutable_intstatistics())
        : IntegerStatisticsWriteWrapper(orcPtr()->mutable_intstatistics());
  }

  proto::orc::DateStatistics* mutableDateStatistics() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->mutable_datestatistics();
  }

  proto::orc::TimestampStatistics* mutableTimestampStatistics() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->mutable_timestampstatistics();
  }

  proto::orc::DecimalStatistics* mutableDecimalStatistics() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return orcPtr()->mutable_decimalstatistics();
  }

  BucketStatisticsWriteWrapper mutableBucketStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? BucketStatisticsWriteWrapper(dwrfPtr()->mutable_bucketstatistics())
        : BucketStatisticsWriteWrapper(orcPtr()->mutable_bucketstatistics());
  }

  void reset(const proto::ColumnStatistics* dwrfStatistics) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    VELOX_CHECK_NOT_NULL(dwrfStatistics);
    dwrfPtr()->CopyFrom(*dwrfStatistics);
  }

 private:
  // private helper with no format checking
  inline proto::ColumnStatistics* dwrfPtr() const {
    return reinterpret_cast<proto::ColumnStatistics*>(rawProtoPtr());
  }
  inline proto::orc::ColumnStatistics* orcPtr() const {
    return reinterpret_cast<proto::orc::ColumnStatistics*>(rawProtoPtr());
  }
};

class FooterWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit FooterWriteWrapper(proto::Footer* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, footer) {}

  explicit FooterWriteWrapper(proto::orc::Footer* footer)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, footer) {}

  const proto::Footer* getDwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return reinterpret_cast<proto::Footer*>(rawProtoPtr());
  }

  proto::Footer* getMutableDwrfPtr() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return reinterpret_cast<proto::Footer*>(rawProtoPtr());
  }

  const proto::orc::Footer* getOrcPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kOrc);
    return reinterpret_cast<proto::orc::Footer*>(rawProtoPtr());
  }

  const StripeInformationWriteWrapper addStripes() const {
    return format_ == DwrfFormat::kDwrf
        ? StripeInformationWriteWrapper(dwrfPtr()->add_stripes())
        : StripeInformationWriteWrapper(orcPtr()->add_stripes());
  }

  void setHeaderLength(uint64_t headerLength) const {
    return format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_headerlength(headerLength)
        : orcPtr()->set_headerlength(headerLength);
  }

  void setContentLength(uint64_t contentLength) const {
    return format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_contentlength(contentLength)
        : orcPtr()->set_contentlength(contentLength);
  }

  void setRowIndexStride(uint32_t rowIndexStride) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_rowindexstride(rowIndexStride)
                                 : orcPtr()->set_rowindexstride(rowIndexStride);
  }

  void setNumberOfRows(uint64_t numberOfRows) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_numberofrows(numberOfRows)
                                 : orcPtr()->set_numberofrows(numberOfRows);
  }

  void setRawDataSize(uint64_t numberOfRows) const {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_rawdatasize(numberOfRows);
    }
  }

  void setWriter(uint32_t writer) const {
    if (format_ == DwrfFormat::kOrc) {
      orcPtr()->set_writer(writer);
    }
  }

  void setCheckSumAlgorithm(proto::ChecksumAlgorithm checksum) const {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_checksumalgorithm(checksum);
    }
  }

  void addStripeCacheOffsets(uint32_t stripeCacheOffsets) const {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->add_stripecacheoffsets(stripeCacheOffsets);
    } else {
      //
    }
  }

  TypeWriteWrapper addTypes() const {
    return format_ == DwrfFormat::kDwrf
        ? TypeWriteWrapper(dwrfPtr()->add_types())
        : TypeWriteWrapper(orcPtr()->add_types());
  }

  UserMetadataItemWriteWrapper addMetadata() const {
    return format_ == DwrfFormat::kDwrf
        ? UserMetadataItemWriteWrapper(dwrfPtr()->add_metadata())
        : UserMetadataItemWriteWrapper(orcPtr()->add_metadata());
  }

  ColumnStatisticsWriteWrapper addStatistics() const {
    return format_ == DwrfFormat::kDwrf
        ? ColumnStatisticsWriteWrapper(dwrfPtr()->add_statistics())
        : ColumnStatisticsWriteWrapper(orcPtr()->add_statistics());
  }

  const ::google::protobuf::RepeatedPtrField<
      ::facebook::velox::dwrf::proto::ColumnStatistics>&
  statistics() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->statistics();
  }

  int typesSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->types_size()
                                        : orcPtr()->types_size();
  }

  int statisticsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->statistics_size()
                                        : orcPtr()->statistics_size();
  }

  uint64_t contentLength() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->contentlength()
                                        : orcPtr()->contentlength();
  }

  uint64_t numberOfRows() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->numberofrows()
                                        : orcPtr()->numberofrows();
  }

  // DWRF-specific fields
  inline uint64_t rawDataSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->rawdatasize();
  }

  inline int stripesSize() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->stripes_size();
  }

  inline proto::Encryption* mutableEncryption() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->mutable_encryption();
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

class RowIndexEntryWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit RowIndexEntryWriteWrapper(proto::RowIndexEntry* rowIndexEntry)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, rowIndexEntry) {}

  explicit RowIndexEntryWriteWrapper(proto::orc::RowIndexEntry* rowIndexEntry)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, rowIndexEntry) {}

  ColumnStatisticsWriteWrapper mutableStatistics() {
    return format_ == DwrfFormat::kDwrf
        ? ColumnStatisticsWriteWrapper(dwrfPtr()->mutable_statistics())
        : ColumnStatisticsWriteWrapper(orcPtr()->mutable_statistics());
  }

  bool hasStatistics() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->has_statistics()
                                        : orcPtr()->has_statistics();
  }

  void mutablePositions(int start, int num) {
    return format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->mutable_positions()->ExtractSubrange(start, num, nullptr)
        : orcPtr()->mutable_positions()->ExtractSubrange(start, num, nullptr);
  }

  void addPositions(uint64_t pos) {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->add_positions(pos)
                                        : orcPtr()->add_positions(pos);
  }

  uint64_t positionsSize() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->positions_size()
                                        : orcPtr()->positions_size();
  }

  const ::google::protobuf::RepeatedField<uint64_t> positions() const {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->positions()
                                        : orcPtr()->positions();
  }

  void clear() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->Clear()
                                        : orcPtr()->Clear();
  }

 private:
  // private helper with no format checking
  inline proto::RowIndexEntry* dwrfPtr() const {
    return reinterpret_cast<proto::RowIndexEntry*>(rawProtoPtr());
  }
  inline proto::orc::RowIndexEntry* orcPtr() const {
    return reinterpret_cast<proto::orc::RowIndexEntry*>(rawProtoPtr());
  }
};

class RowIndexWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit RowIndexWriteWrapper(proto::RowIndex* rowIndex)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, rowIndex) {}

  explicit RowIndexWriteWrapper(proto::orc::RowIndex* rowIndex)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, rowIndex) {}

  void addEntry(std::unique_ptr<RowIndexEntryWriteWrapper>& entry) {
    if (format_ == DwrfFormat::kDwrf) {
      auto e = reinterpret_cast<proto::RowIndexEntry*>(entry->rawProtoPtr());
      *dwrfPtr()->add_entry() = *e;
    } else {
      auto e =
          reinterpret_cast<proto::orc::RowIndexEntry*>(entry->rawProtoPtr());
      *orcPtr()->add_entry() = *e;
    }
  }

  int32_t entrySize() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->entry_size()
                                        : orcPtr()->entry_size();
  }

  RowIndexEntryWriteWrapper mutableEntry(int32_t index) {
    return format_ == DwrfFormat::kDwrf
        ? RowIndexEntryWriteWrapper(dwrfPtr()->mutable_entry(index))
        : RowIndexEntryWriteWrapper(orcPtr()->mutable_entry(index));
  }

  void SerializeToZeroCopyStream(
      dwio::common::BufferedOutputStream* out) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->SerializeToZeroCopyStream(out)
                                 : orcPtr()->SerializeToZeroCopyStream(out);
  }

  void clear() {
    return format_ == DwrfFormat::kDwrf ? dwrfPtr()->Clear()
                                        : orcPtr()->Clear();
  }

 private:
  // private helper with no format checking
  inline proto::RowIndex* dwrfPtr() const {
    return reinterpret_cast<proto::RowIndex*>(rawProtoPtr());
  }
  inline proto::orc::RowIndex* orcPtr() const {
    return reinterpret_cast<proto::orc::RowIndex*>(rawProtoPtr());
  }
};

class ColumnEncodingWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit ColumnEncodingWriteWrapper(proto::ColumnEncoding* stream)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stream) {}

  explicit ColumnEncodingWriteWrapper(proto::orc::ColumnEncoding* stream)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stream) {}

  void setKind(ColumnEncodingKindWrapper columnEncodingKindWrapper) {
    format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_kind(
              *reinterpret_cast<const proto::ColumnEncoding_Kind*>(
                  columnEncodingKindWrapper.rawProtoPtr()))
        : orcPtr()->set_kind(
              *reinterpret_cast<const proto::orc::ColumnEncoding_Kind*>(
                  columnEncodingKindWrapper.rawProtoPtr()));
  }

  void setDictionarySize(uint32_t dictionarySize) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_dictionarysize(dictionarySize)
                                 : orcPtr()->set_dictionarysize(dictionarySize);
  }

  void setNode(uint32_t node) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_node(node);
    }
  }

  void setSequence(uint32_t sequence) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_sequence(sequence);
    }
  }

  proto::KeyInfo* mutableKey() {
    return dwrfPtr()->mutable_key();
  }

  void Clear() {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->Clear() : orcPtr()->Clear();
  }

  void reset(const proto::ColumnEncoding* dwrfEncoding) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    VELOX_CHECK_NOT_NULL(dwrfEncoding);
    dwrfPtr()->CopyFrom(*dwrfEncoding);
  }

 private:
  // private helper with no format checking
  inline proto::ColumnEncoding* dwrfPtr() const {
    return reinterpret_cast<proto::ColumnEncoding*>(rawProtoPtr());
  }
  inline proto::orc::ColumnEncoding* orcPtr() const {
    return reinterpret_cast<proto::orc::ColumnEncoding*>(rawProtoPtr());
  }
};

class StreamWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit StreamWriteWrapper(proto::Stream* stream)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stream) {}

  explicit StreamWriteWrapper(proto::orc::Stream* stream)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stream) {}

  void setOffset(uint64_t offset) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_offset(offset);
  }

  void setKind(const StreamKind& kind) {
    format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_kind(static_cast<proto::Stream_Kind>(kind))
        : orcPtr()->set_kind(static_cast<proto::orc::Stream_Kind>(kind));
  }

  void setColumn(uint32_t column) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_column(column)
                                 : orcPtr()->set_column(column);
  }

  void setLength(uint64_t length) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_length(length)
                                 : orcPtr()->set_length(length);
  }

  void setNode(uint32_t node) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_node(node);
  }

  void setSequence(uint32_t sequence) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_sequence(sequence);
  }

  void setUseVints(bool useVints) {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    dwrfPtr()->set_usevints(useVints);
  }

 private:
  // private helper with no format checking
  inline proto::Stream* dwrfPtr() const {
    return reinterpret_cast<proto::Stream*>(rawProtoPtr());
  }
  inline proto::orc::Stream* orcPtr() const {
    return reinterpret_cast<proto::orc::Stream*>(rawProtoPtr());
  }
};

class StripeEncryptionGroupWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit StripeEncryptionGroupWriteWrapper(
      proto::StripeEncryptionGroup* stripeFooter = nullptr)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stripeFooter) {}

  // See https://orc.apache.org/specification/ORCv1/
  explicit StripeEncryptionGroupWriteWrapper(
      proto::orc::StripeEncryptionVariant* stripeFooter)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stripeFooter) {}

  void encoding(
      std::vector<ColumnEncodingWrapper>& columnEncodingWrappers) const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    for (const proto::ColumnEncoding& encoding : dwrfPtr()->encoding()) {
      auto ce = ColumnEncodingWrapper(&encoding);
      columnEncodingWrappers.emplace_back(ce);
    }
  }

  ColumnEncodingWriteWrapper addEncoding() {
    return format_ == DwrfFormat::kDwrf
        ? ColumnEncodingWriteWrapper(dwrfPtr()->add_encoding())
        : ColumnEncodingWriteWrapper(orcPtr()->add_encoding());
  }

  StreamWriteWrapper addStreams() {
    return format_ == DwrfFormat::kDwrf
        ? StreamWriteWrapper(dwrfPtr()->add_streams())
        : StreamWriteWrapper(orcPtr()->add_streams());
  }

  void SerializeToZeroCopyStream(
      dwio::common::BufferedOutputStream* output) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->SerializeToZeroCopyStream(output)
                                 : orcPtr()->SerializeToZeroCopyStream(output);
  }

 private:
  // private helper with no format checking
  inline proto::StripeEncryptionGroup* dwrfPtr() const {
    return reinterpret_cast<proto::StripeEncryptionGroup*>(rawProtoPtr());
  }
  inline proto::orc::StripeEncryptionVariant* orcPtr() const {
    return reinterpret_cast<proto::orc::StripeEncryptionVariant*>(
        rawProtoPtr());
  }
};

class StripeFooterWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit StripeFooterWriteWrapper(proto::StripeFooter* stripeFooter)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, stripeFooter) {}

  explicit StripeFooterWriteWrapper(proto::orc::StripeFooter* stripeFooter)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, stripeFooter) {}

  void encoding(
      std::vector<ColumnEncodingWrapper>& columnEncodingWrappers) const {
    if (format_ == DwrfFormat::kDwrf) {
      for (const proto::ColumnEncoding& encoding : dwrfPtr()->encoding()) {
        auto ce = ColumnEncodingWrapper(&encoding);
        columnEncodingWrappers.emplace_back(ce);
      }
    } else {
      for (const proto::orc::ColumnEncoding& encoding : orcPtr()->columns()) {
        auto ce = ColumnEncodingWrapper(&encoding);
        columnEncodingWrappers.emplace_back(ce);
      }
    }
  }

  void setWriterTimezone() const {
    if (format_ == DwrfFormat::kOrc) {
      // orcPtr()->set_writertimezone("Asia/Shanghai");
    }
  }

  StreamWriteWrapper addStreams() {
    return format_ == DwrfFormat::kDwrf
        ? StreamWriteWrapper(dwrfPtr()->add_streams())
        : StreamWriteWrapper(orcPtr()->add_streams());
  }

  std::string* addEncryptionGroups() {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return dwrfPtr()->add_encryptiongroups();
  }

  ColumnEncodingWriteWrapper addEncoding() {
    return format_ == DwrfFormat::kDwrf
        ? ColumnEncodingWriteWrapper(dwrfPtr()->add_encoding())
        : ColumnEncodingWriteWrapper(orcPtr()->add_columns());
  }

  void SerializeToZeroCopyStream(
      dwio::common::BufferedOutputStream* output) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->SerializeToZeroCopyStream(output)
                                 : orcPtr()->SerializeToZeroCopyStream(output);
  }

  inline proto::StripeFooter* dwrfPtr() const {
    VELOX_CHECK_EQ(format_, DwrfFormat::kDwrf);
    return reinterpret_cast<proto::StripeFooter*>(rawProtoPtr());
  }

 private:
  // private helper with no format checking
  inline proto::orc::StripeFooter* orcPtr() const {
    return reinterpret_cast<proto::orc::StripeFooter*>(rawProtoPtr());
  }
};

class PostScriptWriteWrapper : public ProtoWriteWrapperBase {
 public:
  explicit PostScriptWriteWrapper(proto::PostScript* postScript)
      : ProtoWriteWrapperBase(DwrfFormat::kDwrf, postScript) {}

  explicit PostScriptWriteWrapper(proto::orc::PostScript* postScript)
      : ProtoWriteWrapperBase(DwrfFormat::kOrc, postScript) {}

  void setWriterVersion(uint32_t writerVersion) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_writerversion(writerVersion)
                                 : orcPtr()->set_writerversion(6);
  }

  void addVersion(uint32_t version) {
    if (format_ == DwrfFormat::kOrc) {
      orcPtr()->add_version(version);
    }
  }

  void setFooterLength(uint64_t footerLength) {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->set_footerlength(footerLength)
                                 : orcPtr()->set_footerlength(footerLength);
  }

  void setCompression(common::CompressionKind compressionKind);

  void setCompressionBlockSize(uint64_t compressionBlockSize) {
    format_ == DwrfFormat::kDwrf
        ? dwrfPtr()->set_compressionblocksize(compressionBlockSize)
        : orcPtr()->set_compressionblocksize(compressionBlockSize);
  }

  void setCacheMode(StripeCacheMode cacheMode) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_cachemode(static_cast<proto::StripeCacheMode>(cacheMode));
    }
  }

  void setCacheSize(uint32_t cacheSize) {
    if (format_ == DwrfFormat::kDwrf) {
      dwrfPtr()->set_cachesize(cacheSize);
    }
  }

  void setMetaDataLength(uint64_t metaDataLength) {
    if (format_ == DwrfFormat::kOrc) {
      orcPtr()->set_metadatalength(metaDataLength);
    }
  }

  void SerializeToZeroCopyStream(
      dwio::common::BufferedOutputStream* out) const {
    format_ == DwrfFormat::kDwrf ? dwrfPtr()->SerializeToZeroCopyStream(out)
                                 : orcPtr()->SerializeToZeroCopyStream(out);
  }

 private:
  // private helper with no format checking
  inline proto::PostScript* dwrfPtr() const {
    return reinterpret_cast<proto::PostScript*>(rawProtoPtr());
  }
  inline proto::orc::PostScript* orcPtr() const {
    return reinterpret_cast<proto::orc::PostScript*>(rawProtoPtr());
  }
};

} // namespace facebook::velox::dwrf

template <>
struct fmt::formatter<facebook::velox::dwrf::DwrfFormat> : formatter<int> {
  auto format(facebook::velox::dwrf::DwrfFormat s, format_context& ctx) const {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};
