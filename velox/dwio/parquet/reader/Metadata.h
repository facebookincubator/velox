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

#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/compression/Compression.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

/// Parquet encoding types abstracted from thrift
enum class EncodingType {
  PLAIN = 0,
  PLAIN_DICTIONARY = 2,
  RLE = 3,
  BIT_PACKED = 4,
  DELTA_BINARY_PACKED = 5,
  DELTA_LENGTH_BYTE_ARRAY = 6,
  DELTA_BYTE_ARRAY = 7,
  RLE_DICTIONARY = 8,
  BYTE_STREAM_SPLIT = 9
};

/// Parquet page types abstracted from thrift
enum class PageType {
  DATA_PAGE = 0,
  INDEX_PAGE = 1,
  DICTIONARY_PAGE = 2,
  DATA_PAGE_V2 = 3
};

/// Page encoding statistics abstracted from thrift
struct PageEncodingStats {
  PageType pageType;
  EncodingType encoding;
  int32_t count;

  PageEncodingStats(PageType pt, EncodingType enc, int32_t cnt)
      : pageType(pt), encoding(enc), count(cnt) {}
};

/// ColumnChunkMetaDataPtr is a proxy around pointer to thrift::ColumnChunk.
class ColumnChunkMetaDataPtr {
 public:
  explicit ColumnChunkMetaDataPtr(const void* metadata);

  ~ColumnChunkMetaDataPtr();

  /// Check the presence of ColumnChunk metadata.
  bool hasMetadata() const;

  /// Check the presence of statistics in the ColumnChunk metadata.
  bool hasStatistics() const;

  /// Check the presence of the dictionary page offset in ColumnChunk metadata.
  bool hasDictionaryPageOffset() const;

  bool hasEncodingStats() const;

  std::vector<PageEncodingStats> getEncodingStats() const;

  std::vector<EncodingType> getEncodings() const;

  /// Return the ColumnChunk statistics.
  std::unique_ptr<dwio::common::ColumnStatistics> getColumnStatistics(
      const TypePtr type,
      int64_t numRows);

  /// Return the Column Metadata Statistics Min Value
  std::string getColumnMetadataStatsMinValue();

  /// Return the Column Metadata Statistics Max Value
  std::string getColumnMetadataStatsMaxValue();

  /// Return the Column Metadata Statistics Null Count
  int64_t getColumnMetadataStatsNullCount();

  /// Number of values.
  int64_t numValues() const;

  /// The data page offset.
  int64_t dataPageOffset() const;

  /// The dictionary page offset.
  /// Must check for its presence using hasDictionaryPageOffset().
  int64_t dictionaryPageOffset() const;

  /// The compression.
  common::CompressionKind compression() const;

  /// Total byte size of all the compressed (and potentially encrypted)
  /// column data in this row group.
  /// This information is optional and may be 0 if omitted.
  int64_t totalCompressedSize() const;

  /// Total byte size of all the uncompressed (and potentially encrypted)
  /// column data in this row group.
  /// This information is optional and may be 0 if omitted.
  int64_t totalUncompressedSize() const;

 private:
  const void* ptr_;
};

/// RowGroupMetaDataPtr is a proxy around pointer to thrift::RowGroup.
class RowGroupMetaDataPtr {
 public:
  explicit RowGroupMetaDataPtr(const void* metadata);

  ~RowGroupMetaDataPtr();

  /// The number of columns in this row group. The order must match the
  /// parent's column ordering.
  int numColumns() const;

  /// Return the ColumnChunkMetaData pointer of the corresponding column index.
  ColumnChunkMetaDataPtr columnChunk(int index) const;

  /// Number of rows in this row group.
  int64_t numRows() const;

  /// Total byte size of all the uncompressed column data in this row
  /// group.
  int64_t totalByteSize() const;

  /// Check the presence of file offset.
  bool hasFileOffset() const;

  /// Byte offset from beginning of file to first page (data or dictionary)
  /// in this row group
  int64_t fileOffset() const;

  /// Check the presence of total compressed size.
  bool hasTotalCompressedSize() const;

  /// The sorting column, column index in this row group
  int32_t sortingColumnIdx(int i) const;

  /// If true, indicates this column is sorted in descending order.
  bool sortingColumnDescending(int i) const;

  /// If true, nulls will come before non-null values, otherwise, nulls go at
  /// the end
  bool sortingColumnNullsFirst(int i) const;

  /// Total byte size of all the compressed (and potentially encrypted)
  /// column data in this row group.
  /// This information is optional and may be 0 if omitted.
  int64_t totalCompressedSize() const;

 private:
  const void* ptr_;
};

/// FileMetaData is a proxy around pointer to thrift::FileMetaData.
class FileMetaDataPtr {
 public:
  explicit FileMetaDataPtr(const void* metadata);

  ~FileMetaDataPtr();

  /// The total number of rows.
  int64_t numRows() const;

  /// The number of row groups in the file.
  int numRowGroups() const;

  /// Return the RowGroupMetaData pointer of the corresponding row group index.
  RowGroupMetaDataPtr rowGroup(int index) const;

  /// The key/value metadata size.
  int64_t keyValueMetadataSize() const;

  /// Returns True if the key/value metadata contains the key input.
  bool keyValueMetadataContains(const std::string_view key) const;

  /// Returns the value inside the key/value metadata if the key is present.
  std::string keyValueMetadataValue(const std::string_view key) const;

  /// Return the Parquet writer created_by string.
  std::string createdBy() const;

 private:
  const void* ptr_;
};

} // namespace facebook::velox::parquet
