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

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

class ColumnDescriptor;
class EncodedStatistics;
class Statistics;
class SchemaDescriptor;

class FileCryptoMetaData;
class InternalFileDecryptor;
class Decryptor;
class Encryptor;
class FooterSigningEncryptor;

namespace schema {

class ColumnPath;

} // namespace schema

using KeyValueMetadata = ::arrow::KeyValueMetadata;

class PARQUET_EXPORT ApplicationVersion {
 public:
  // Known versions with issues.
  static const ApplicationVersion& PARQUET_251_FIXED_VERSION();
  static const ApplicationVersion& PARQUET_816_FIXED_VERSION();
  static const ApplicationVersion& PARQUET_CPP_FIXED_STATS_VERSION();
  static const ApplicationVersion& PARQUET_MR_FIXED_STATS_VERSION();
  static const ApplicationVersion& PARQUET_CPP_10353_FIXED_VERSION();

  // Application that wrote the file, e.g., "IMPALA".
  std::string application_;
  // Build name.
  std::string build_;

  // Version of the application that wrote the file, expressed as
  // (<major>.<minor>.<patch>). Unmatched parts default to 0.
  // "1.2.3"    => {1, 2, 3}
  // "1.2"      => {1, 2, 0}
  // "1.2-cdh5" => {1, 2, 0}
  struct {
    int major;
    int minor;
    int patch;
    std::string unknown;
    std::string preRelease;
    std::string buildInfo;
  } version;

  ApplicationVersion() = default;
  explicit ApplicationVersion(const std::string& createdBy);
  ApplicationVersion(std::string application, int major, int minor, int patch);

  // Returns true if version is strictly less than otherVersion.
  bool versionLt(const ApplicationVersion& otherVersion) const;

  // Returns true if version is strictly equal with otherVersion.
  bool versionEq(const ApplicationVersion& otherVersion) const;

  // Checks if the Version has the correct statistics for a given column.
  bool hasCorrectStatistics(
      Type::type primitive,
      EncodedStatistics& statistics,
      SortOrder::type sortOrder = SortOrder::kSigned) const;
};

class PARQUET_EXPORT ColumnCryptoMetaData {
 public:
  static std::unique_ptr<ColumnCryptoMetaData> make(const uint8_t* metadata);
  ~ColumnCryptoMetaData();

  bool equals(const ColumnCryptoMetaData& other) const;

  std::shared_ptr<schema::ColumnPath> pathInSchema() const;
  bool encryptedWithFooterKey() const;
  const std::string& keyMetadata() const;

 private:
  explicit ColumnCryptoMetaData(const uint8_t* metadata);

  class ColumnCryptoMetaDataImpl;
  std::unique_ptr<ColumnCryptoMetaDataImpl> impl_;
};

/// \brief Public struct for Thrift PageEncodingStats in ColumnChunkMetaData.
struct PageEncodingStats {
  PageType::type pageType;
  Encoding::type encoding;
  int32_t count;
};

/// \brief Public struct for location to page index in ColumnChunkMetaData.
struct IndexLocation {
  /// File offset of the given index, in bytes.
  int64_t offset;
  /// Length of the given index, in bytes.
  int32_t length;
};

/// \brief ColumnChunkMetaData is a proxy around
/// facebook::velox::parquet::thrift::ColumnChunkMetaData.
class PARQUET_EXPORT ColumnChunkMetaData {
 public:
  // API convenience to get a MetaData accessor.

  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::unique_ptr<ColumnChunkMetaData> make(
      const void* metadata,
      const ColumnDescriptor* descr,
      const ApplicationVersion* writerVersion,
      int16_t rowGroupOrdinal = -1,
      int16_t columnOrdinal = -1,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  static std::unique_ptr<ColumnChunkMetaData> make(
      const void* metadata,
      const ColumnDescriptor* descr,
      const ReaderProperties& properties = defaultReaderProperties(),
      const ApplicationVersion* writerVersion = NULLPTR,
      int16_t rowGroupOrdinal = -1,
      int16_t columnOrdinal = -1,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  ~ColumnChunkMetaData();

  bool equals(const ColumnChunkMetaData& other) const;

  // Column chunk.
  int64_t fileOffset() const;

  // Parameter is only used when a dataset is spread across multiple files.
  const std::string& filePath() const;

  // Column metadata.
  bool isMetadataSet() const;
  Type::type type() const;
  int64_t numValues() const;
  std::shared_ptr<schema::ColumnPath> pathInSchema() const;
  bool isStatsSet() const;
  std::shared_ptr<Statistics> statistics() const;

  Compression::type compression() const;
  // Indicate if the ColumnChunk compression is supported by the current
  // compiled Parquet library.
  bool canDecompress() const;

  const std::vector<Encoding::type>& encodings() const;
  const std::vector<PageEncodingStats>& encodingStats() const;
  std::optional<int64_t> bloomFilterOffset() const;
  bool hasDictionaryPage() const;
  int64_t dictionaryPageOffset() const;
  int64_t dataPageOffset() const;
  bool hasIndexPage() const;
  int64_t indexPageOffset() const;
  int64_t totalCompressedSize() const;
  int64_t totalUncompressedSize() const;
  int32_t fieldId() const;
  std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata() const;
  std::optional<IndexLocation> getColumnIndexLocation() const;
  std::optional<IndexLocation> getOffsetIndexLocation() const;

 private:
  explicit ColumnChunkMetaData(
      const void* metadata,
      const ColumnDescriptor* descr,
      int16_t rowGroupOrdinal,
      int16_t columnOrdinal,
      const ReaderProperties& properties,
      const ApplicationVersion* writerVersion = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);
  // PIMPL Idiom.
  class ColumnChunkMetaDataImpl;
  std::unique_ptr<ColumnChunkMetaDataImpl> impl_;
};

/// \brief RowGroupMetaData is a proxy around
/// facebook::velox::parquet::thrift::RowGroupMetaData.
class PARQUET_EXPORT RowGroupMetaData {
 public:
  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::unique_ptr<RowGroupMetaData> make(
      const void* metadata,
      const SchemaDescriptor* schema,
      const ApplicationVersion* writerVersion,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  /// \brief Create a RowGroupMetaData from a serialized thrift message.
  static std::unique_ptr<RowGroupMetaData> make(
      const void* metadata,
      const SchemaDescriptor* schema,
      const ReaderProperties& properties = defaultReaderProperties(),
      const ApplicationVersion* writerVersion = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  ~RowGroupMetaData();

  bool equals(const RowGroupMetaData& other) const;

  /// \brief The number of columns in this row group. The order must match the
  /// parent's column ordering.
  int numColumns() const;

  /// \brief Return the ColumnChunkMetaData of the corresponding column ordinal.
  ///
  /// WARNING: The returned object references memory location in its parent
  /// (RowGroupMetaData) object. Hence, the parent must outlive the returned
  /// object.
  ///
  /// \param[in] index Index of the ColumnChunkMetaData to retrieve.
  ///
  /// \throws ParquetException if the index is out of bound.
  std::unique_ptr<ColumnChunkMetaData> columnChunk(int index) const;

  /// \brief Number of rows in this row group.
  int64_t numRows() const;

  /// \brief Total byte size of all the uncompressed column data in this row
  /// group.
  int64_t totalByteSize() const;

  /// \brief Total byte size of all the compressed (and potentially encrypted)
  /// column data in this row group.
  ///
  /// This information is optional and may be 0 if omitted.
  int64_t totalCompressedSize() const;

  /// \brief Byte offset from beginning of file to first page (data or
  /// dictionary) in this row group.
  ///
  /// The file_offset field that this method exposes is optional. This method
  /// will return 0 if that field is not set to a meaningful value.
  int64_t fileOffset() const;
  // Return const pointer to make it clear that this object is not to be copied.
  const SchemaDescriptor* schema() const;
  // Indicate if all of the RowGroup's ColumnChunks can be decompressed.
  bool canDecompress() const;
  // Sorting columns of the row group if any.
  std::vector<SortingColumn> sortingColumns() const;

 private:
  explicit RowGroupMetaData(
      const void* metadata,
      const SchemaDescriptor* schema,
      const ReaderProperties& properties,
      const ApplicationVersion* writerVersion = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);
  // PIMPL Idiom.
  class RowGroupMetaDataImpl;
  std::unique_ptr<RowGroupMetaDataImpl> impl_;
};

class FileMetaDataBuilder;

/// \brief FileMetaData is a proxy around
/// facebook::velox::parquet::thrift::FileMetaData.
class PARQUET_EXPORT FileMetaData {
 public:
  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::shared_ptr<FileMetaData> make(
      const void* serializedMetadata,
      uint32_t* inoutMetadataLen,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor);

  /// \brief Create a FileMetaData from a serialized thrift message.
  static std::shared_ptr<FileMetaData> make(
      const void* serializedMetadata,
      uint32_t* inoutMetadataLen,
      const ReaderProperties& properties = defaultReaderProperties(),
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  ~FileMetaData();

  bool equals(const FileMetaData& other) const;

  /// \brief The number of Parquet "leaf" columns.
  ///
  /// Parquet thrift definition requires that nested schema elements are
  /// flattened. This method returns the number of columns in the flattened
  /// version.
  /// For instance, if the schema looks like this:
  /// 0 Foo.bar
  ///       Foo.bar.baz           0
  ///       Foo.bar.baz2          1
  ///   Foo.qux                   2
  /// 1 Foo2                      3
  /// 2 Foo3                      4
  /// This method will return 5, because there are 5 "leaf" fields (so 5
  /// flattened fields).
  int numColumns() const;

  /// \brief The number of flattened schema elements.
  ///
  /// Parquet thrift definition requires that nested schema elements are
  /// flattened. This method returns the total number of elements in the
  /// flattened list.
  int numSchemaElements() const;

  /// \brief The total number of rows.
  int64_t numRows() const;

  /// \brief The number of row groups in the file.
  int numRowGroups() const;

  /// \brief Return the RowGroupMetaData of the corresponding row group ordinal.
  ///
  /// WARNING: The returned object references memory location in its parent
  /// (FileMetaData) object. Hence, the parent must outlive the returned object.
  ///
  /// \param[in] index Index of the RowGroup to retrieve.
  ///
  /// \throws ParquetException if the index is out of bound.
  std::unique_ptr<RowGroupMetaData> rowGroup(int index) const;

  /// \brief Return the "version" of the file.
  ///
  /// WARNING: The value returned by this method is unreliable as 1) the
  /// Parquet file metadata stores the version as a single integer and 2) some
  /// producers are known to always write a hardcoded value. Therefore, you
  /// cannot use this value to know which features are used in the file.
  ParquetVersion::type version() const;

  /// \brief Return the application's user-agent string of the writer.
  const std::string& createdBy() const;

  /// \brief Return the application's version of the writer.
  const ApplicationVersion& writerVersion() const;

  /// \brief Size of the original thrift encoded metadata footer.
  uint32_t size() const;

  /// \brief Indicate if all of the FileMetadata's RowGroups can be
  /// decompressed.
  ///
  /// This will return false if any of the RowGroup's page is compressed with a
  /// compression format which is not compiled in the current Parquet library.
  bool canDecompress() const;

  bool isEncryptionAlgorithmSet() const;
  EncryptionAlgorithm encryptionAlgorithm() const;
  const std::string& footerSigningKeyMetadata() const;

  /// \brief Verify signature of FileMetaData when file is encrypted but footer
  /// is not encrypted (plaintext footer).
  bool verifySignature(const void* signature);

  void writeTo(
      ::arrow::io::OutputStream* dst,
      const std::shared_ptr<Encryptor>& encryptor = NULLPTR) const;

  /// \brief Return Thrift-serialized representation of the metadata as a
  /// string.
  std::string serializeToString() const;

  // Return const pointer to make it clear that this object is not to be copied.
  const SchemaDescriptor* schema() const;

  const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata() const;

  /// \brief Set a path to all ColumnChunk for all RowGroups.
  ///
  /// Commonly used by systems (Dask, Spark) who generate a metadata-only
  /// Parquet file. The path is usually relative to said index file.
  ///
  /// \param[in] path Path to set.
  void setFilePath(const std::string& path);

  /// \brief Merge row groups from another metadata file into this one.
  ///
  /// The schema of the input FileMetaData must be equal to the
  /// schema of this object.
  ///
  /// This is used by systems who create an aggregate metadata-only file by
  /// concatenating the row groups of multiple files. This newly created
  /// metadata file acts as an index of all available row groups.
  ///
  /// \param[in] other Other FileMetaData to merge the row groups from.
  ///
  /// \throws ParquetException if schemas are not equal.
  void appendRowGroups(const FileMetaData& other);

  /// \brief Return a FileMetaData containing a subset of the row groups in
  /// this FileMetaData.
  std::shared_ptr<FileMetaData> subset(const std::vector<int>& rowGroups) const;

  /// \brief Get total NaN count for a specific field ID across all row groups.
  /// Returns a pair of (nanCount, hasNanCount).
  /// NaN counts are collected during writing but not written to the Parquet
  /// file.
  std::pair<int64_t, bool> getNaNCount(int32_t fieldId) const;

 private:
  friend FileMetaDataBuilder;
  friend class SerializedFile;

  explicit FileMetaData(
      const void* serializedMetadata,
      uint32_t* metadataLen,
      const ReaderProperties& properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  void setFileDecryptor(std::shared_ptr<InternalFileDecryptor> fileDecryptor);

  // PIMPL Idiom.
  FileMetaData();
  class FileMetaDataImpl;
  std::unique_ptr<FileMetaDataImpl> impl_;
};

class PARQUET_EXPORT FileCryptoMetaData {
 public:
  // API convenience to get a MetaData accessor.
  static std::shared_ptr<FileCryptoMetaData> make(
      const uint8_t* serializedMetadata,
      uint32_t* metadataLen,
      const ReaderProperties& properties = defaultReaderProperties());
  ~FileCryptoMetaData();

  EncryptionAlgorithm encryptionAlgorithm() const;
  const std::string& keyMetadata() const;

  void writeTo(::arrow::io::OutputStream* dst) const;

 private:
  friend FileMetaDataBuilder;
  FileCryptoMetaData(
      const uint8_t* serializedMetadata,
      uint32_t* metadataLen,
      const ReaderProperties& properties);

  // PIMPL Idiom.
  FileCryptoMetaData();
  class FileCryptoMetaDataImpl;
  std::unique_ptr<FileCryptoMetaDataImpl> impl_;
};

// Builder API.
class PARQUET_EXPORT ColumnChunkMetaDataBuilder {
 public:
  // API convenience to get a MetaData reader.
  static std::unique_ptr<ColumnChunkMetaDataBuilder> make(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column);

  static std::unique_ptr<ColumnChunkMetaDataBuilder> make(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column,
      void* Contents);

  ~ColumnChunkMetaDataBuilder();

  // Column chunk.
  // Used when a dataset is spread across multiple files.
  void setFilePath(const std::string& path);
  // Column metadata.
  void setStatistics(const EncodedStatistics& stats);
  // Get the column descriptor.
  const ColumnDescriptor* descr() const;

  int64_t totalCompressedSize() const;

  // NaN count accessors - NaN counts are collected during writing but not
  // written to the parquet file.
  int64_t nanCount() const;

  bool hasNanCount() const;

  // Commit the metadata.
  void finish(
      int64_t numValues,
      int64_t dictionaryPageOffset,
      int64_t indexPageOffset,
      int64_t dataPageOffset,
      int64_t compressedSize,
      int64_t uncompressedSize,
      bool hasDictionary,
      bool dictionaryFallback,
      const std::map<Encoding::type, int32_t>& dictEncodingStats,
      const std::map<Encoding::type, int32_t>& dataEncodingStats,
      const std::shared_ptr<Encryptor>& encryptor = NULLPTR);

  // The metadata contents, suitable for passing to ColumnChunkMetaData::Make.
  const void* Contents() const;

  // For writing metadata at end of column chunk.
  void writeTo(::arrow::io::OutputStream* sink);

 private:
  explicit ColumnChunkMetaDataBuilder(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column);
  explicit ColumnChunkMetaDataBuilder(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column,
      void* Contents);
  // PIMPL Idiom.
  class ColumnChunkMetaDataBuilderImpl;
  std::unique_ptr<ColumnChunkMetaDataBuilderImpl> impl_;
};

class PARQUET_EXPORT RowGroupMetaDataBuilder {
 public:
  // API convenience to get a MetaData reader.
  static std::unique_ptr<RowGroupMetaDataBuilder> make(
      std::shared_ptr<WriterProperties> props,
      const SchemaDescriptor* schema_,
      void* Contents);

  ~RowGroupMetaDataBuilder();

  ColumnChunkMetaDataBuilder* nextColumnChunk();
  int numColumns();
  int64_t numRows();
  int currentColumn() const;

  void setNumRows(int64_t numRows);

  // Get NaN counts for all columns in current row group.
  // Returns a map of field_id -> (nan_count, has_nan_count).
  std::unordered_map<int32_t, std::pair<int64_t, bool>> nanCounts() const;

  // Commit the metadata.
  void finish(int64_t totalBytesWritten, int16_t rowGroupOrdinal = -1);

 private:
  explicit RowGroupMetaDataBuilder(
      std::shared_ptr<WriterProperties> props,
      const SchemaDescriptor* schema_,
      void* Contents);
  // PIMPL Idiom.
  class RowGroupMetaDataBuilderImpl;
  std::unique_ptr<RowGroupMetaDataBuilderImpl> impl_;
};

/// \brief Public struct for location to all page indexes in a Parquet file.
struct PageIndexLocation {
  /// Alias type of page index location of a row group. The index location
  /// is located by column ordinal. If the column does not have the page index,
  /// its value is set to std::nullopt.
  using RowGroupIndexLocation = std::vector<std::optional<IndexLocation>>;
  /// Alias type of page index location of a Parquet file. The index location
  /// is located by the row group ordinal.
  using FileIndexLocation = std::map<size_t, RowGroupIndexLocation>;
  /// Row group column index locations which use row group ordinal as the key.
  FileIndexLocation columnIndexLocation;
  /// Row group offset index locations which use row group ordinal as the key.
  FileIndexLocation offsetIndexLocation;
};

class PARQUET_EXPORT FileMetaDataBuilder {
 public:
  ARROW_DEPRECATED(
      "Deprecated in 12.0.0. Use overload without KeyValueMetadata instead.")
  static std::unique_ptr<FileMetaDataBuilder> make(
      const SchemaDescriptor* schema,
      std::shared_ptr<WriterProperties> props,
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata);

  // API convenience to get a MetaData builder.
  static std::unique_ptr<FileMetaDataBuilder> make(
      const SchemaDescriptor* schema,
      std::shared_ptr<WriterProperties> props);

  ~FileMetaDataBuilder();

  // The prior RowGroupMetaDataBuilder (if any) is destroyed.
  RowGroupMetaDataBuilder* appendRowGroup();

  // Update location to all page indexes in the Parquet file.
  void setPageIndexLocation(const PageIndexLocation& location);

  // Complete the Thrift structure.
  std::unique_ptr<FileMetaData> finish(
      const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata =
          NULLPTR);

  // Crypto metadata.
  std::unique_ptr<FileCryptoMetaData> getCryptoMetaData();

 private:
  explicit FileMetaDataBuilder(
      const SchemaDescriptor* schema,
      std::shared_ptr<WriterProperties> props,
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata = NULLPTR);
  // PIMPL Idiom.
  class FileMetaDataBuilderImpl;
  std::unique_ptr<FileMetaDataBuilderImpl> impl_;
};

PARQUET_EXPORT std::string parquetVersionToString(ParquetVersion::type ver);

} // namespace facebook::velox::parquet::arrow
