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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "arrow/io/caching.h"
#include "arrow/type.h"
#include "arrow/util/type_fwd.h"
#include "velox/dwio/parquet/writer/arrow/Encryption.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/util/Compression.h"

// Define the Parquet created by version.
#define CREATED_BY_VERSION "parquet-cpp-velox"
// Velox has no versioning yet. Set default 0.0.0.
#define VELOX_VERSION "0.0.0"

namespace facebook::velox::parquet::arrow {

using facebook::velox::parquet::arrow::util::CodecOptions;

/// \brief Feature selection when writing Parquet files.
///
/// `ParquetVersion::type` governs which data types are allowed and how they
/// are represented. For example, uint32_t data will be written differently
/// depending on this value (as INT64 for PARQUET_1_0, as UINT32 for other
/// versions).
///
/// However, some features - such as compression algorithms, encryption,
/// or the improved "v2" data page format - must be enabled separately in
/// ArrowWriterProperties.
struct ParquetVersion {
  enum type : int {
    /// Enable only pre-2.2 Parquet format features when writing.
    ///
    /// This setting is useful for maximum compatibility with legacy readers.
    /// Corresponds to a converted type.
    PARQUET_1_0,

    /// DEPRECATED: Enable Parquet format 2.6 features.
    ///
    /// This misleadingly named enum value is roughly similar to PARQUET_2_6.
    PARQUET_2_0 ARROW_DEPRECATED_ENUM_VALUE(
        "use PARQUET_2_4 or PARQUET_2_6 "
        "for fine-grained feature selection"),

    /// Enable Parquet format 2.4 and earlier features when writing.
    ///
    /// This enables UINT32 as well as logical types which don't have
    /// a corresponding converted type.
    ///
    /// Note: Parquet format 2.4.0 was released in October 2017.
    PARQUET_2_4,

    /// Enable Parquet format 2.6 and earlier features when writing.
    ///
    /// units in addition to the PARQUET_2_4 features.
    ///
    /// Note: Parquet format 2.6.0 was released in September 2018.
    PARQUET_2_6,

    /// Enable latest Parquet format 2.x features.
    ///
    /// The version supported by this library.
    PARQUET_2_LATEST = PARQUET_2_6
  };
};

/// Controls serialization format of data pages. Parquet-format v2.0.0
/// introduced a new data page metadata type DataPageV2 and serialized page
/// structure (for example, encoded levels are no longer compressed). Prior to
/// the completion of PARQUET-457 in 2020, this library did not implement
/// DataPageV2 correctly, so if you use the V2 data page format, you may have
/// forward compatibility issues (older versions of the library will be unable
/// to read the files). Note that some Parquet implementations do not implement
/// DataPageV2 at all.
enum class ParquetDataPageVersion { V1, V2 };

/// Align the default buffer size to a small multiple of a page size.
constexpr int64_t kDefaultBufferSize = 4096 * 4;

constexpr int32_t kDefaultThriftStringSizeLimit = 100 * 1000 * 1000;
// Structs in the thrift definition are relatively large (at least 300 bytes).
// This limits total memory to the same order of magnitude as
// kDefaultStringSizeLimit.
constexpr int32_t kDefaultThriftContainerSizeLimit = 1000 * 1000;

class PARQUET_EXPORT ReaderProperties {
 public:
  explicit ReaderProperties(MemoryPool* pool = ::arrow::default_memory_pool())
      : pool_(pool) {}

  MemoryPool* memoryPool() const {
    return pool_;
  }

  std::shared_ptr<ArrowInputStream> getStream(
      std::shared_ptr<ArrowInputFile> source,
      int64_t start,
      int64_t numBytes);

  /// Buffered stream reading allows the user to control the memory usage of
  /// Parquet readers. This ensure that all `RandomAccessFile::ReadAt` calls
  /// are wrapped in a buffered reader that uses a fix sized buffer (of size
  /// `bufferSize()`) instead of the full size of the ReadAt.
  ///
  /// The primary reason for this control knob is for resource control and not
  /// performance.
  bool isBufferedStreamEnabled() const {
    return bufferedStreamEnabled_;
  }
  /// Enable buffered stream reading.
  void enableBufferedStream() {
    bufferedStreamEnabled_ = true;
  }
  /// Disable buffered stream reading.
  void disableBufferedStream() {
    bufferedStreamEnabled_ = false;
  }

  /// Return the size of the buffered stream buffer.
  int64_t bufferSize() const {
    return bufferSize_;
  }
  /// Set the size of the buffered stream buffer in bytes.
  void setBufferSize(int64_t size) {
    bufferSize_ = size;
  }

  /// \brief Return the size limit on thrift strings.
  ///
  /// This limit helps prevent space and time bombs in files, but may need to
  /// be increased in order to read files with especially large headers.
  int32_t thriftStringSizeLimit() const {
    return thriftStringSizeLimit_;
  }
  /// Set the size limit on thrift strings.
  void setThriftStringSizeLimit(int32_t size) {
    thriftStringSizeLimit_ = size;
  }

  /// \brief Return the size limit on thrift containers.
  ///
  /// This limit helps prevent space and time bombs in files, but may need to
  /// be increased in order to read files with especially large headers.
  int32_t thriftContainerSizeLimit() const {
    return thriftContainerSizeLimit_;
  }
  /// Set the size limit on thrift containers.
  void setThriftContainerSizeLimit(int32_t size) {
    thriftContainerSizeLimit_ = size;
  }

  /// Set the decryption properties.
  void setFileDecryptionProperties(
      std::shared_ptr<FileDecryptionProperties> decryption) {
    fileDecryptionProperties_ = std::move(decryption);
  }
  /// Return the decryption properties.
  const std::shared_ptr<FileDecryptionProperties>& fileDecryptionProperties()
      const {
    return fileDecryptionProperties_;
  }

  bool pageChecksumVerification() const {
    return pageChecksumVerification_;
  }
  void setPageChecksumVerification(bool checkCrc) {
    pageChecksumVerification_ = checkCrc;
  }

 private:
  MemoryPool* pool_;
  int64_t bufferSize_ = kDefaultBufferSize;
  int32_t thriftStringSizeLimit_ = kDefaultThriftStringSizeLimit;
  int32_t thriftContainerSizeLimit_ = kDefaultThriftContainerSizeLimit;
  bool bufferedStreamEnabled_ = false;
  bool pageChecksumVerification_ = false;
  std::shared_ptr<FileDecryptionProperties> fileDecryptionProperties_;
};

ReaderProperties PARQUET_EXPORT defaultReaderProperties();

static constexpr int64_t kDefaultDataPageSize = 1024 * 1024;
static constexpr bool DEFAULT_IS_DICTIONARY_ENABLED = true;
static constexpr int64_t DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT =
    kDefaultDataPageSize;
static constexpr int64_t DEFAULT_WRITE_BATCH_SIZE = 1024;
static constexpr int64_t DEFAULT_MAX_ROW_GROUP_LENGTH = 1024 * 1024;
static constexpr bool DEFAULT_ARE_STATISTICS_ENABLED = true;
static constexpr int64_t DEFAULT_MAX_STATISTICS_SIZE = 4096;
static constexpr Encoding::type DEFAULT_ENCODING = Encoding::kUnknown;
static const char DEFAULT_CREATED_BY[] = CREATED_BY_VERSION;
static constexpr Compression::type DEFAULT_COMPRESSION_TYPE =
    Compression::UNCOMPRESSED;
static constexpr bool DEFAULT_IS_PAGE_INDEX_ENABLED = false;

class PARQUET_EXPORT ColumnProperties {
 public:
  ColumnProperties(
      Encoding::type encoding = DEFAULT_ENCODING,
      Compression::type codec = DEFAULT_COMPRESSION_TYPE,
      bool dictionaryEnabled = DEFAULT_IS_DICTIONARY_ENABLED,
      bool statisticsEnabled = DEFAULT_ARE_STATISTICS_ENABLED,
      size_t maxStatsSize = DEFAULT_MAX_STATISTICS_SIZE,
      bool pageIndexEnabled = DEFAULT_IS_PAGE_INDEX_ENABLED)
      : encoding_(encoding),
        codec_(codec),
        dictionaryEnabled_(dictionaryEnabled),
        statisticsEnabled_(statisticsEnabled),
        maxStatsSize_(maxStatsSize),
        pageIndexEnabled_(pageIndexEnabled) {}

  void setEncoding(Encoding::type encoding) {
    encoding_ = encoding;
  }

  void setCompression(Compression::type codec) {
    codec_ = codec;
  }

  void setDictionaryEnabled(bool dictionaryEnabled) {
    dictionaryEnabled_ = dictionaryEnabled;
  }

  void setStatisticsEnabled(bool statisticsEnabled) {
    statisticsEnabled_ = statisticsEnabled;
  }

  void setMaxStatisticsSize(size_t maxStatsSize) {
    maxStatsSize_ = maxStatsSize;
  }

  void setCompressionLevel(int compressionLevel) {
    if (!codecOptions_) {
      codecOptions_ = std::make_shared<CodecOptions>();
    }
    codecOptions_->compressionLevel = compressionLevel;
  }

  void setCodecOptions(const std::shared_ptr<CodecOptions>& codecOptions) {
    codecOptions_ = codecOptions;
  }

  void setPageIndexEnabled(bool pageIndexEnabled) {
    pageIndexEnabled_ = pageIndexEnabled;
  }

  Encoding::type encoding() const {
    return encoding_;
  }

  Compression::type compression() const {
    return codec_;
  }

  bool dictionaryEnabled() const {
    return dictionaryEnabled_;
  }

  bool statisticsEnabled() const {
    return statisticsEnabled_;
  }

  size_t maxStatisticsSize() const {
    return maxStatsSize_;
  }

  int compressionLevel() const {
    return codecOptions_->compressionLevel;
  }

  const std::shared_ptr<CodecOptions>& codecOptions() const {
    return codecOptions_;
  }

  bool pageIndexEnabled() const {
    return pageIndexEnabled_;
  }

 private:
  Encoding::type encoding_;
  Compression::type codec_;
  bool dictionaryEnabled_;
  bool statisticsEnabled_;
  size_t maxStatsSize_;
  std::shared_ptr<CodecOptions> codecOptions_;
  bool pageIndexEnabled_;
};

class PARQUET_EXPORT WriterProperties {
 public:
  class Builder {
   public:
    Builder()
        : pool_(::arrow::default_memory_pool()),
          dictionaryPagesizeLimit_(DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT),
          writeBatchSize_(DEFAULT_WRITE_BATCH_SIZE),
          maxRowGroupLength_(DEFAULT_MAX_ROW_GROUP_LENGTH),
          pagesize_(kDefaultDataPageSize),
          version_(ParquetVersion::PARQUET_2_6),
          dataPageVersion_(ParquetDataPageVersion::V1),
          createdBy_(
              DEFAULT_CREATED_BY + std::string(" version ") + VELOX_VERSION),
          storeDecimalAsInteger_(false),
          pageChecksumEnabled_(false) {}
    virtual ~Builder() {}

    /// Specify the memory pool for the writer. Default default_memory_pool.
    Builder* memoryPool(MemoryPool* pool) {
      pool_ = pool;
      return this;
    }

    /// Enable dictionary encoding in general for all columns. Default enabled.
    Builder* enableDictionary() {
      defaultColumnProperties_.setDictionaryEnabled(true);
      return this;
    }

    /// Disable dictionary encoding in general for all columns. Default enabled.
    Builder* disableDictionary() {
      defaultColumnProperties_.setDictionaryEnabled(false);
      return this;
    }

    /// Enable dictionary encoding for column specified by `path`. Default
    /// enabled.
    Builder* enableDictionary(const std::string& path) {
      dictionaryEnabled_[path] = true;
      return this;
    }

    /// Enable dictionary encoding for column specified by `path`. Default
    /// enabled.
    Builder* enableDictionary(const std::shared_ptr<schema::ColumnPath>& path) {
      return this->enableDictionary(path->toDotString());
    }

    /// Disable dictionary encoding for column specified by `path`. Default
    /// enabled.
    Builder* disableDictionary(const std::string& path) {
      dictionaryEnabled_[path] = false;
      return this;
    }

    /// Disable dictionary encoding for column specified by `path`. Default
    /// enabled.
    Builder* disableDictionary(
        const std::shared_ptr<schema::ColumnPath>& path) {
      return this->disableDictionary(path->toDotString());
    }

    /// Specify the dictionary page size limit per row group. Default 1MB.
    Builder* dictionaryPagesizeLimit(int64_t dictionaryPsizeLimit) {
      dictionaryPagesizeLimit_ = dictionaryPsizeLimit;
      return this;
    }

    /// Specify the write batch size while writing batches of Arrow values into
    /// Parquet. Default 1024.
    Builder* writeBatchSize(int64_t writeBatchSize) {
      writeBatchSize_ = writeBatchSize;
      return this;
    }

    /// Specify the max number of rows to put in a single row group.
    /// Default 1Mi rows.
    Builder* maxRowGroupLength(int64_t maxRowGroupLength) {
      maxRowGroupLength_ = maxRowGroupLength;
      return this;
    }

    /// Specify the data page size.
    /// Default 1MB.
    Builder* dataPagesize(int64_t pgSize) {
      pagesize_ = pgSize;
      return this;
    }

    /// Specify the data page version.
    /// Default V1.
    Builder* dataPageVersion(ParquetDataPageVersion dataPageVersion) {
      dataPageVersion_ = dataPageVersion;
      return this;
    }

    /// Specify the Parquet file version.
    /// Default PARQUET_2_6.
    Builder* version(ParquetVersion::type version) {
      version_ = version;
      return this;
    }

    Builder* createdBy(const std::string& createdBy) {
      createdBy_ = createdBy;
      return this;
    }

    Builder* enablePageChecksum() {
      pageChecksumEnabled_ = true;
      return this;
    }

    Builder* disablePageChecksum() {
      pageChecksumEnabled_ = false;
      return this;
    }

    /// \brief Define the encoding that is used when we don't utilise
    /// dictionary encoding.
    //
    /// This either applies if dictionary encoding is disabled or if we
    /// fallback because the dictionary grew too large.
    Builder* encoding(Encoding::type encodingType) {
      if (encodingType == Encoding::kPlainDictionary ||
          encodingType == Encoding::kRleDictionary) {
        throw ParquetException(
            "Can't use dictionary encoding as fallback encoding");
      }

      defaultColumnProperties_.setEncoding(encodingType);
      return this;
    }

    /// \brief Define the encoding that is used when we don't utilise
    /// dictionary encoding.
    //
    /// This either applies if dictionary encoding is disabled or if we
    /// fallback because the dictionary grew too large.
    Builder* encoding(const std::string& path, Encoding::type encodingType) {
      if (encodingType == Encoding::kPlainDictionary ||
          encodingType == Encoding::kRleDictionary) {
        throw ParquetException(
            "Can't use dictionary encoding as fallback encoding");
      }

      encodings_[path] = encodingType;
      return this;
    }

    /// \brief Define the encoding that is used when we don't utilise
    /// dictionary encoding.
    //
    /// This either applies if dictionary encoding is disabled or if we
    /// fallback because the dictionary grew too large.
    Builder* encoding(
        const std::shared_ptr<schema::ColumnPath>& path,
        Encoding::type encodingType) {
      return this->encoding(path->toDotString(), encodingType);
    }

    /// Specify compression codec in general for all columns.
    /// Default UNCOMPRESSED.
    Builder* compression(Compression::type codec) {
      defaultColumnProperties_.setCompression(codec);
      return this;
    }

    /// Specify max statistics size to store min max value.
    /// Default 4KB.
    Builder* maxStatisticsSize(size_t maxStatsSz) {
      defaultColumnProperties_.setMaxStatisticsSize(maxStatsSz);
      return this;
    }

    /// Specify compression codec for the column specified by `path`.
    /// Default UNCOMPRESSED.
    Builder* compression(const std::string& path, Compression::type codec) {
      codecs_[path] = codec;
      return this;
    }

    /// Specify compression codec for the column specified by `path`.
    /// Default UNCOMPRESSED.
    Builder* compression(
        const std::shared_ptr<schema::ColumnPath>& path,
        Compression::type codec) {
      return this->compression(path->toDotString(), codec);
    }

    /// \brief Specify the default compression level for the compressor
    /// in every column. In case a column does not have an explicitly
    /// specified compression level, the default one would be used.
    ///
    /// The provided compression level is compressor specific. The user
    /// would have to familiarize oneself with the available levels for
    /// the selected compressor. If the compressor does not allow for
    /// selecting different compression levels, calling this function
    /// would not have any effect. Parquet and Arrow do not validate the
    /// passed compression level. If no level is selected by the user or
    /// if the special std::numeric_limits<int>::min() value is passed,
    /// then Arrow selects the compression level.
    ///
    /// If other compressor-specific options need to be set in addition
    /// to the compression level, use the codec_options method.
    Builder* compressionLevel(int compressionLevel) {
      defaultColumnProperties_.setCompressionLevel(compressionLevel);
      return this;
    }

    /// \brief Specify a compression level for the compressor for the
    /// column described by path.
    ///
    /// The provided compression level is compressor specific. The user
    /// would have to familiarize oneself with the available levels for
    /// the selected compressor. If the compressor does not allow for
    /// selecting different compression levels, calling this function
    /// would not have any effect. Parquet and Arrow do not validate the
    /// passed compression level. If no level is selected by the user or
    /// if the special std::numeric_limits<int>::min() value is passed,
    /// then Arrow selects the compression level.
    Builder* compressionLevel(const std::string& path, int compressionLevel) {
      if (!codecOptions_[path]) {
        codecOptions_[path] = std::make_shared<CodecOptions>();
      }
      codecOptions_[path]->compressionLevel = compressionLevel;
      return this;
    }

    /// \brief Specify a compression level for the compressor for the
    /// column described by path.
    ///
    /// The provided compression level is compressor specific. The user
    /// would have to familiarize oneself with the available levels for
    /// the selected compressor. If the compressor does not allow for
    /// selecting different compression levels, calling this function
    /// would not have any effect. Parquet and Arrow do not validate the
    /// passed compression level. If no level is selected by the user or
    /// if the special std::numeric_limits<int>::min() value is passed,
    /// then Arrow selects the compression level.
    Builder* compressionLevel(
        const std::shared_ptr<schema::ColumnPath>& path,
        int compressionLevel) {
      return this->compressionLevel(path->toDotString(), compressionLevel);
    }

    /// \brief Specify the default codec options for the compressor in
    /// every column.
    ///
    /// The codec options allow configuring the compression level as
    /// well as other codec-specific options.
    Builder* codecOptions(const std::shared_ptr<CodecOptions>& codecOptions) {
      defaultColumnProperties_.setCodecOptions(codecOptions);
      return this;
    }

    /// \brief Specify the codec options for the compressor for the
    /// column described by path.
    Builder* codecOptions(
        const std::string& path,
        const std::shared_ptr<CodecOptions>& codecOptions) {
      codecOptions_[path] = codecOptions;
      return this;
    }

    /// \brief Specify the codec options for the compressor for the
    /// column described by path.
    Builder* codecOptions(
        const std::shared_ptr<schema::ColumnPath>& path,
        const std::shared_ptr<CodecOptions>& codecOptions) {
      return this->codecOptions(path->toDotString(), codecOptions);
    }

    /// Define the file encryption properties.
    /// Default NULL.
    Builder* encryption(
        std::shared_ptr<FileEncryptionProperties> fileEncryptionProperties) {
      fileEncryptionProperties_ = std::move(fileEncryptionProperties);
      return this;
    }

    /// Enable statistics in general.
    /// Default enabled.
    Builder* enableStatistics() {
      defaultColumnProperties_.setStatisticsEnabled(true);
      return this;
    }

    /// Disable statistics in general.
    /// Default enabled.
    Builder* disableStatistics() {
      defaultColumnProperties_.setStatisticsEnabled(false);
      return this;
    }

    /// Enable statistics for the column specified by `path`.
    /// Default enabled.
    Builder* enableStatistics(const std::string& path) {
      statisticsEnabled_[path] = true;
      return this;
    }

    /// Enable statistics for the column specified by `path`.
    /// Default enabled.
    Builder* enableStatistics(const std::shared_ptr<schema::ColumnPath>& path) {
      return this->enableStatistics(path->toDotString());
    }

    /// Define the sorting columns.
    /// Default empty.
    ///
    /// If sorting columns are set, user should ensure that records
    /// are sorted by sorting columns. Otherwise, the storing data
    /// will be inconsistent with sorting_columns metadata.
    Builder* setSortingColumns(std::vector<SortingColumn> sortingColumns) {
      sortingColumns_ = std::move(sortingColumns);
      return this;
    }

    /// Disable statistics for the column specified by `path`.
    /// Default enabled.
    Builder* disableStatistics(const std::string& path) {
      statisticsEnabled_[path] = false;
      return this;
    }

    /// Disable statistics for the column specified by `path`.
    /// Default enabled.
    Builder* disableStatistics(
        const std::shared_ptr<schema::ColumnPath>& path) {
      return this->disableStatistics(path->toDotString());
    }

    /// Allow decimals with 1 <= precision <= 18 to be stored as
    /// integers.
    ///
    /// In Parquet, DECIMAL can be stored in any of the following
    /// physical types:
    /// - Int32: For 1 <= precision <= 9.
    /// - Int64: For 10 <= precision <= 18.
    /// - Fixed_len_byte_array: Precision is limited by the array size.
    ///   Length n can store <= floor(log_10(2^(8*n - 1) - 1)) base-10
    ///   digits.
    /// - Binary: Precision is unlimited. The minimum number of bytes to
    /// store
    ///   the unscaled value is used.
    ///
    /// By default, this is DISABLED and all decimal types annotate.
    /// Fixed_len_byte_array.
    ///
    /// When enabled, the C++ writer will use following physical types
    /// to store decimals:
    /// - Int32: For 1 <= precision <= 9.
    /// - Int64: For 10 <= precision <= 18.
    /// - Fixed_len_byte_array: For precision > 18.
    ///
    /// As a consequence, decimal columns stored in integer types are
    /// more compact.
    Builder* enableStoreDecimalAsInteger() {
      storeDecimalAsInteger_ = true;
      return this;
    }

    /// Disable decimal logical type with 1 <= precision <= 18 to be
    /// stored as integer physical type.
    ///
    /// Default disabled.
    Builder* disableStoreDecimalAsInteger() {
      storeDecimalAsInteger_ = false;
      return this;
    }

    /// Enable writing page index in general for all columns. Default
    /// disabled.
    ///
    /// Writing statistics to the page index disables the old method of
    /// writing statistics to each data page header. The page index
    /// makes filtering more efficient than the page header, as it
    /// gathers all the statistics for a Parquet file in a single place,
    /// avoiding scattered I/O.
    ///
    /// Please check the link below for more details:
    /// https://github.com/apache/parquet-format/blob/master/PageIndex.md
    Builder* enableWritePageIndex() {
      defaultColumnProperties_.setPageIndexEnabled(true);
      return this;
    }

    /// Disable writing page index in general for all columns. Default
    /// disabled.
    Builder* disableWritePageIndex() {
      defaultColumnProperties_.setPageIndexEnabled(false);
      return this;
    }

    /// Enable writing page index for column specified by `path`.
    /// Default disabled.
    Builder* enableWritePageIndex(const std::string& path) {
      pageIndexEnabled_[path] = true;
      return this;
    }

    /// Enable writing page index for column specified by `path`.
    /// Default disabled.
    Builder* enableWritePageIndex(
        const std::shared_ptr<schema::ColumnPath>& path) {
      return this->enableWritePageIndex(path->toDotString());
    }

    /// Disable writing page index for column specified by `path`.
    /// Default disabled.
    Builder* disableWritePageIndex(const std::string& path) {
      pageIndexEnabled_[path] = false;
      return this;
    }

    /// Disable writing page index for column specified by `path`.
    /// Default disabled.
    Builder* disableWritePageIndex(
        const std::shared_ptr<schema::ColumnPath>& path) {
      return this->disableWritePageIndex(path->toDotString());
    }

    /// \brief Build the WriterProperties with the builder parameters.
    /// \return The WriterProperties defined by the builder.
    std::shared_ptr<WriterProperties> build() {
      std::unordered_map<std::string, ColumnProperties> columnProperties;
      auto get = [&](const std::string& key) -> ColumnProperties& {
        auto it = columnProperties.find(key);
        if (it == columnProperties.end())
          return columnProperties[key] = defaultColumnProperties_;
        else
          return it->second;
      };

      for (const auto& item : encodings_)
        get(item.first).setEncoding(item.second);
      for (const auto& item : codecs_)
        get(item.first).setCompression(item.second);
      for (const auto& item : codecOptions_)
        get(item.first).setCodecOptions(item.second);
      for (const auto& item : dictionaryEnabled_)
        get(item.first).setDictionaryEnabled(item.second);
      for (const auto& item : statisticsEnabled_)
        get(item.first).setStatisticsEnabled(item.second);
      for (const auto& item : pageIndexEnabled_)
        get(item.first).setPageIndexEnabled(item.second);

      return std::shared_ptr<WriterProperties>(new WriterProperties(
          pool_,
          dictionaryPagesizeLimit_,
          writeBatchSize_,
          maxRowGroupLength_,
          pagesize_,
          version_,
          createdBy_,
          pageChecksumEnabled_,
          std::move(fileEncryptionProperties_),
          defaultColumnProperties_,
          columnProperties,
          dataPageVersion_,
          storeDecimalAsInteger_,
          std::move(sortingColumns_)));
    }

   private:
    MemoryPool* pool_;
    int64_t dictionaryPagesizeLimit_;
    int64_t writeBatchSize_;
    int64_t maxRowGroupLength_;
    int64_t pagesize_;
    ParquetVersion::type version_;
    ParquetDataPageVersion dataPageVersion_;
    std::string createdBy_;
    bool storeDecimalAsInteger_;
    bool pageChecksumEnabled_;

    std::shared_ptr<FileEncryptionProperties> fileEncryptionProperties_;

    // If empty, there is no sorting columns.
    std::vector<SortingColumn> sortingColumns_;

    // Settings used for each column unless overridden in any of the
    // maps below.
    ColumnProperties defaultColumnProperties_;
    std::unordered_map<std::string, Encoding::type> encodings_;
    std::unordered_map<std::string, Compression::type> codecs_;
    std::unordered_map<std::string, std::shared_ptr<CodecOptions>>
        codecOptions_;
    std::unordered_map<std::string, bool> dictionaryEnabled_;
    std::unordered_map<std::string, bool> statisticsEnabled_;
    std::unordered_map<std::string, bool> pageIndexEnabled_;
  };

  inline MemoryPool* memoryPool() const {
    return pool_;
  }

  inline int64_t dictionaryPagesizeLimit() const {
    return dictionaryPagesizeLimit_;
  }

  inline int64_t writeBatchSize() const {
    return writeBatchSize_;
  }

  inline int64_t maxRowGroupLength() const {
    return maxRowGroupLength_;
  }

  inline int64_t dataPagesize() const {
    return pagesize_;
  }

  inline ParquetDataPageVersion dataPageVersion() const {
    return parquetDataPageVersion_;
  }

  inline ParquetVersion::type version() const {
    return parquetVersion_;
  }

  inline std::string createdBy() const {
    return parquetCreatedBy_;
  }

  inline bool storeDecimalAsInteger() const {
    return storeDecimalAsInteger_;
  }

  inline bool pageChecksumEnabled() const {
    return pageChecksumEnabled_;
  }

  inline Encoding::type dictionaryIndexEncoding() const {
    if (parquetVersion_ == ParquetVersion::PARQUET_1_0) {
      return Encoding::kPlainDictionary;
    } else {
      return Encoding::kRleDictionary;
    }
  }

  inline Encoding::type dictionaryPageEncoding() const {
    if (parquetVersion_ == ParquetVersion::PARQUET_1_0) {
      return Encoding::kPlainDictionary;
    } else {
      return Encoding::kPlain;
    }
  }

  const ColumnProperties& columnProperties(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    auto it = columnProperties_.find(path->toDotString());
    if (it != columnProperties_.end())
      return it->second;
    return defaultColumnProperties_;
  }

  Encoding::type encoding(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).encoding();
  }

  Compression::type compression(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).compression();
  }

  int compressionLevel(const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).compressionLevel();
  }

  const std::shared_ptr<CodecOptions> codecOptions(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).codecOptions();
  }

  bool dictionaryEnabled(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).dictionaryEnabled();
  }

  const std::vector<SortingColumn>& sortingColumns() const {
    return sortingColumns_;
  }

  bool statisticsEnabled(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).statisticsEnabled();
  }

  size_t maxStatisticsSize(
      const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).maxStatisticsSize();
  }

  bool pageIndexEnabled(const std::shared_ptr<schema::ColumnPath>& path) const {
    return columnProperties(path).pageIndexEnabled();
  }

  bool pageIndexEnabled() const {
    if (defaultColumnProperties_.pageIndexEnabled()) {
      return true;
    }
    for (const auto& item : columnProperties_) {
      if (item.second.pageIndexEnabled()) {
        return true;
      }
    }
    return false;
  }

  inline FileEncryptionProperties* fileEncryptionProperties() const {
    return fileEncryptionProperties_.get();
  }

  std::shared_ptr<ColumnEncryptionProperties> columnEncryptionProperties(
      const std::string& path) const {
    if (fileEncryptionProperties_) {
      return fileEncryptionProperties_->columnEncryptionProperties(path);
    } else {
      return NULLPTR;
    }
  }

 private:
  explicit WriterProperties(
      MemoryPool* pool,
      int64_t dictionaryPagesizeLimit,
      int64_t writeBatchSize,
      int64_t maxRowGroupLength,
      int64_t pagesize,
      ParquetVersion::type version,
      const std::string& createdBy,
      bool pageWriteChecksumEnabled,
      std::shared_ptr<FileEncryptionProperties> fileEncryptionProperties,
      const ColumnProperties& defaultColumnProperties,
      const std::unordered_map<std::string, ColumnProperties>& columnProperties,
      ParquetDataPageVersion dataPageVersion,
      bool storeShortDecimalAsInteger,
      std::vector<SortingColumn> sortingColumns)
      : pool_(pool),
        dictionaryPagesizeLimit_(dictionaryPagesizeLimit),
        writeBatchSize_(writeBatchSize),
        maxRowGroupLength_(maxRowGroupLength),
        pagesize_(pagesize),
        parquetDataPageVersion_(dataPageVersion),
        parquetVersion_(version),
        parquetCreatedBy_(createdBy),
        storeDecimalAsInteger_(storeShortDecimalAsInteger),
        pageChecksumEnabled_(pageWriteChecksumEnabled),
        fileEncryptionProperties_(std::move(fileEncryptionProperties)),
        sortingColumns_(std::move(sortingColumns)),
        defaultColumnProperties_(defaultColumnProperties),
        columnProperties_(columnProperties) {}

  MemoryPool* pool_;
  int64_t dictionaryPagesizeLimit_;
  int64_t writeBatchSize_;
  int64_t maxRowGroupLength_;
  int64_t pagesize_;
  ParquetDataPageVersion parquetDataPageVersion_;
  ParquetVersion::type parquetVersion_;
  std::string parquetCreatedBy_;
  bool storeDecimalAsInteger_;
  bool pageChecksumEnabled_;

  std::shared_ptr<FileEncryptionProperties> fileEncryptionProperties_;

  std::vector<SortingColumn> sortingColumns_;

  ColumnProperties defaultColumnProperties_;
  std::unordered_map<std::string, ColumnProperties> columnProperties_;
};

PARQUET_EXPORT const std::shared_ptr<WriterProperties>&
defaultWriterProperties();

// ----------------------------------------------------------------------.
// Properties specific to Apache Arrow columnar read and write.

static constexpr bool kArrowDefaultUseThreads = false;

// Default number of rows to read when using ::arrow::RecordBatchReader.
static constexpr int64_t kArrowDefaultBatchSize = 64 * 1024;

/// EXPERIMENTAL: Properties for configuring FileReader behavior.
class PARQUET_EXPORT ArrowReaderProperties {
 public:
  explicit ArrowReaderProperties(bool useThreads = kArrowDefaultUseThreads)
      : useThreads_(useThreads),
        readDictIndices_(),
        batchSize_(kArrowDefaultBatchSize),
        preBuffer_(false),
        cacheOptions_(::arrow::io::CacheOptions::Defaults()),
        coerceInt96TimestampUnit_(::arrow::TimeUnit::NANO) {}

  /// \brief Set whether to use the IO thread pool to parse columns in
  /// parallel.
  ///
  /// Default is false.
  void setUseThreads(bool useThreads) {
    useThreads_ = useThreads;
  }
  /// Return whether will use multiple threads.
  bool useThreads() const {
    return useThreads_;
  }

  /// \brief Set whether to read a particular column as dictionary
  /// encoded.
  ///
  /// If the file metadata contains a serialized Arrow schema, then this
  /// is only supported for columns with a Parquet physical type of
  /// BYTE_ARRAY, such as string or binary types.
  void setReadDictionary(int columnIndex, bool readDict) {
    if (readDict) {
      readDictIndices_.insert(columnIndex);
    } else {
      readDictIndices_.erase(columnIndex);
    }
  }
  /// Return whether the column at the index will be read as dictionary.
  bool readDictionary(int columnIndex) const {
    if (readDictIndices_.find(columnIndex) != readDictIndices_.end()) {
      return true;
    } else {
      return false;
    }
  }

  /// \brief Set the maximum number of rows to read into a record batch.
  ///
  /// Will only be fewer rows when there are no more rows in the file.
  /// Note that some APIs such as ReadTable may ignore this setting.
  void setBatchSize(int64_t batchSize) {
    batchSize_ = batchSize;
  }
  /// Return the batch size in rows.
  ///
  /// Note that some APIs such as ReadTable may ignore this setting.
  int64_t batchSize() const {
    return batchSize_;
  }

  /// Enable read coalescing (default false).
  ///
  /// When enabled, the Arrow reader will pre-buffer necessary regions.
  /// Of the file in-memory. This is intended to improve performance on.
  /// High-latency filesystems (e.g. Amazon S3).
  void setPreBuffer(bool preBuffer) {
    preBuffer_ = preBuffer;
  }
  /// Return whether read coalescing is enabled.
  bool preBuffer() const {
    return preBuffer_;
  }

  /// Set options for read coalescing. This can be used to tune the.
  /// Implementation for characteristics of different filesystems.
  void setCacheOptions(::arrow::io::CacheOptions options) {
    cacheOptions_ = options;
  }
  /// Return the options for read coalescing.
  const ::arrow::io::CacheOptions& cacheOptions() const {
    return cacheOptions_;
  }

  /// Set execution context for read coalescing.
  void setIoContext(const ::arrow::io::IOContext& ctx) {
    ioContext_ = ctx;
  }
  /// Return the execution context used for read coalescing.
  const ::arrow::io::IOContext& ioContext() const {
    return ioContext_;
  }

  /// Set timestamp unit to use for deprecated INT96-encoded timestamps.
  /// (Default is NANO).
  void setCoerceInt96TimestampUnit(::arrow::TimeUnit::type unit) {
    coerceInt96TimestampUnit_ = unit;
  }

  ::arrow::TimeUnit::type coerceInt96TimestampUnit() const {
    return coerceInt96TimestampUnit_;
  }

 private:
  bool useThreads_;
  std::unordered_set<int> readDictIndices_;
  int64_t batchSize_;
  bool preBuffer_;
  ::arrow::io::IOContext ioContext_;
  ::arrow::io::CacheOptions cacheOptions_;
  ::arrow::TimeUnit::type coerceInt96TimestampUnit_;
};

/// EXPERIMENTAL: Constructs the default ArrowReaderProperties.
PARQUET_EXPORT
ArrowReaderProperties defaultArrowReaderProperties();

class PARQUET_EXPORT ArrowWriterProperties {
 public:
  enum EngineVersion {
    V1, // Supports only nested lists.
    V2 // Full support for all nesting combinations
  };
  class Builder {
   public:
    Builder()
        : writeTimestampsAsInt96_(false),
          coerceTimestampsEnabled_(false),
          coerceTimestampsUnit_(::arrow::TimeUnit::SECOND),
          truncatedTimestampsAllowed_(false),
          storeSchema_(false),
          compliantNestedTypes_(true),
          engineVersion_(V2),
          useThreads_(kArrowDefaultUseThreads),
          executor_(NULLPTR) {}
    virtual ~Builder() = default;

    /// \brief Disable writing legacy int96 timestamps (default
    /// disabled).
    Builder* disableDeprecatedInt96Timestamps() {
      writeTimestampsAsInt96_ = false;
      return this;
    }

    /// \brief Enable writing legacy int96 timestamps (default
    /// disabled).
    ///
    /// May be turned on to write timestamps compatible with older
    /// Parquet writers. This takes precedent over coerceTimestamps.
    Builder* enableDeprecatedInt96Timestamps() {
      writeTimestampsAsInt96_ = true;
      return this;
    }

    /// \brief Coerce all timestamps to the specified time unit.
    /// \param unit time unit to truncate to.
    /// For Parquet versions 1.0 and 2.4, nanoseconds are casted to.
    /// Microseconds.
    Builder* coerceTimestamps(::arrow::TimeUnit::type unit) {
      coerceTimestampsEnabled_ = true;
      coerceTimestampsUnit_ = unit;
      return this;
    }

    /// \brief Allow loss of data when truncating timestamps.
    ///
    /// This is disallowed by default and an error will be returned.
    Builder* allowTruncatedTimestamps() {
      truncatedTimestampsAllowed_ = true;
      return this;
    }

    /// \brief Disallow loss of data when truncating timestamps
    /// (default).
    Builder* disallowTruncatedTimestamps() {
      truncatedTimestampsAllowed_ = false;
      return this;
    }

    /// \brief EXPERIMENTAL: Write binary serialized Arrow schema to the
    /// file, to enable certain read options (like "read_dictionary") to
    /// be set automatically.
    Builder* storeSchema() {
      storeSchema_ = true;
      return this;
    }

    /// \brief When enabled, will not preserve Arrow field names for
    /// list types.
    ///
    /// Instead of using the field names Arrow uses for the values array
    /// of. List types (default "item"), will use "element", as is
    /// specified in. The Parquet spec.
    ///
    /// This is enabled by default.
    Builder* enableCompliantNestedTypes() {
      compliantNestedTypes_ = true;
      return this;
    }

    /// Preserve Arrow list field name.
    Builder* disableCompliantNestedTypes() {
      compliantNestedTypes_ = false;
      return this;
    }

    /// Set the version of the Parquet writer engine.
    Builder* setEngineVersion(EngineVersion version) {
      engineVersion_ = version;
      return this;
    }

    /// \brief Set whether to use multiple threads to write columns.
    /// In parallel in the buffered row group mode.
    ///
    /// WARNING: If writing multiple files in parallel in the same.
    /// Executor, deadlock may occur if use_threads is true. Please.
    /// Disable it in this case.
    ///
    /// Default is false.
    Builder* setUseThreads(bool useThreads) {
      useThreads_ = useThreads;
      return this;
    }

    /// \brief Set the executor to write columns in parallel in the.
    /// Buffered row group mode.
    ///
    /// Default is nullptr and the default cpu executor will be used.
    Builder* setExecutor(::arrow::internal::Executor* executor) {
      executor_ = executor;
      return this;
    }

    /// Create the final properties.
    std::shared_ptr<ArrowWriterProperties> build() {
      return std::shared_ptr<ArrowWriterProperties>(new ArrowWriterProperties(
          writeTimestampsAsInt96_,
          coerceTimestampsEnabled_,
          coerceTimestampsUnit_,
          truncatedTimestampsAllowed_,
          storeSchema_,
          compliantNestedTypes_,
          engineVersion_,
          useThreads_,
          executor_));
    }

   private:
    bool writeTimestampsAsInt96_;

    bool coerceTimestampsEnabled_;
    ::arrow::TimeUnit::type coerceTimestampsUnit_;
    bool truncatedTimestampsAllowed_;

    bool storeSchema_;
    bool compliantNestedTypes_;
    EngineVersion engineVersion_;

    bool useThreads_;
    ::arrow::internal::Executor* executor_;
  };

  bool supportDeprecatedInt96Timestamps() const {
    return writeTimestampsAsInt96_;
  }

  bool coerceTimestampsEnabled() const {
    return coerceTimestampsEnabled_;
  }
  ::arrow::TimeUnit::type coerceTimestampsUnit() const {
    return coerceTimestampsUnit_;
  }

  bool truncatedTimestampsAllowed() const {
    return truncatedTimestampsAllowed_;
  }

  bool storeSchema() const {
    return storeSchema_;
  }

  /// \brief Enable nested type naming according to the parquet
  /// specification.
  ///
  /// Older versions of arrow wrote out field names for nested lists
  /// based on the name of the field. According to the Parquet
  /// specification they should always be "element".
  bool compliantNestedTypes() const {
    return compliantNestedTypes_;
  }

  /// \brief The underlying engine version to use when writing Arrow
  /// data.
  ///
  /// V2 is currently the latest V1 is considered deprecated but left
  /// in. Place in case there are bugs detected in V2.
  EngineVersion engineVersion() const {
    return engineVersion_;
  }

  /// \brief Returns whether the writer will use multiple threads.
  /// To write columns in parallel in the buffered row group mode.
  bool useThreads() const {
    return useThreads_;
  }

  /// \brief Returns the executor used to write columns in parallel.
  ::arrow::internal::Executor* executor() const;

 private:
  explicit ArrowWriterProperties(
      bool writeNanosAsInt96,
      bool coerceTimestampsEnabled,
      ::arrow::TimeUnit::type coerceTimestampsUnit,
      bool truncatedTimestampsAllowed,
      bool storeSchema,
      bool compliantNestedTypes,
      EngineVersion engineVersion,
      bool useThreads,
      ::arrow::internal::Executor* executor)
      : writeTimestampsAsInt96_(writeNanosAsInt96),
        coerceTimestampsEnabled_(coerceTimestampsEnabled),
        coerceTimestampsUnit_(coerceTimestampsUnit),
        truncatedTimestampsAllowed_(truncatedTimestampsAllowed),
        storeSchema_(storeSchema),
        compliantNestedTypes_(compliantNestedTypes),
        engineVersion_(engineVersion),
        useThreads_(useThreads),
        executor_(executor) {}

  const bool writeTimestampsAsInt96_;
  const bool coerceTimestampsEnabled_;
  const ::arrow::TimeUnit::type coerceTimestampsUnit_;
  const bool truncatedTimestampsAllowed_;
  const bool storeSchema_;
  const bool compliantNestedTypes_;
  const EngineVersion engineVersion_;
  const bool useThreads_;
  ::arrow::internal::Executor* executor_;
};

/// \brief State object used for writing Arrow data directly to a
/// Parquet. Column chunk. API possibly not stable.
struct ArrowWriteContext {
  ArrowWriteContext(MemoryPool* memoryPool, ArrowWriterProperties* properties)
      : memoryPool(memoryPool),
        properties(properties),
        dataBuffer(allocateBuffer(memoryPool)),
        defLevelsBuffer(allocateBuffer(memoryPool)) {}

  template <typename T>
  ::arrow::Status getScratchData(const int64_t numValues, T** out) {
    ARROW_RETURN_NOT_OK(this->dataBuffer->Resize(numValues * sizeof(T), false));
    *out = reinterpret_cast<T*>(this->dataBuffer->mutable_data());
    return ::arrow::Status::OK();
  }

  MemoryPool* memoryPool;
  const ArrowWriterProperties* properties;

  // Buffer used for storing the data of an array converted to the
  // physical type. As expected by parquet-cpp.
  std::shared_ptr<ResizableBuffer> dataBuffer;

  // We use the shared ownership of this buffer.
  std::shared_ptr<ResizableBuffer> defLevelsBuffer;
};

PARQUET_EXPORT
std::shared_ptr<ArrowWriterProperties> defaultArrowWriterProperties();

} // namespace facebook::velox::parquet::arrow
