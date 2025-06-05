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

// returns the size of the thrift::ColumnChunk
static size_t calculateColumnMetadataSize(const thrift::ColumnChunk& column) {
  size_t size = 0;
  // Add size of column metadata
  size += sizeof(thrift::ColumnChunk);
  // Add size of column metadata
  size += sizeof(thrift::ColumnMetaData);
  // Add size of encodings
  size += column.meta_data.encodings.size() * sizeof(thrift::Encoding::type);
  // Add size of path in schema
  size += column.meta_data.path_in_schema.size() * sizeof(std::string);
  for (const auto& path : column.meta_data.path_in_schema) {
    size += path.capacity();
  }
  // Add size of key_value_metadata
  size += column.meta_data.key_value_metadata.size() * sizeof(thrift::KeyValue);
  for (const auto& kv : column.meta_data.key_value_metadata) {
    size += kv.key.capacity();
    size += kv.value.capacity();
  }

  if (column.meta_data.__isset.statistics) {
    const auto& stats = column.meta_data.statistics;
    size += sizeof(thrift::Statistics);
    if (stats.__isset.min)
      size += stats.min.capacity();
    if (stats.__isset.max)
      size += stats.max.capacity();
    if (stats.__isset.min_value)
      size += stats.min_value.capacity();
    if (stats.__isset.max_value)
      size += stats.max_value.capacity();
    if (stats.__isset.null_count)
      size += sizeof(int64_t);
    if (stats.__isset.distinct_count)
      size += sizeof(int64_t);
  }
  return size;
}

// returns the size of the thrift object thrift::FileMetaData
static size_t analyzeFileMetadata(
    const thrift::FileMetaData& metadata,
    const std::string& filename,
    uint64_t footerLength) {
  size_t totalSize = sizeof(thrift::FileMetaData);

  // Statistics tracking
  size_t numSchemaElements = metadata.schema.size();
  size_t maxSchemaDepth = 0;
  size_t currentDepth = 0;
  size_t numLeafNodes = 0;
  size_t numGroupNodes = 0;
  size_t totalStringLength = 0;
  size_t maxStringLength = 0;
  size_t maxDepth = 0;
  size_t maxDepthElementSize = 0;
  std::string maxDepthElementName;
  size_t maxElementSize = 0;
  std::string maxElementName;
  currentDepth = 0;

  for (const auto& schema : metadata.schema) {
    size_t elementSize = sizeof(schema) + schema.name.capacity() +
        sizeof(schema.type) + sizeof(schema.type_length) +
        sizeof(schema.repetition_type) + sizeof(schema.num_children) +
        sizeof(schema.converted_type) + sizeof(schema.scale) +
        sizeof(schema.precision) + sizeof(schema.field_id);
    if (schema.__isset.logicalType) {
      elementSize += sizeof(schema.logicalType);
    }
    if (elementSize > maxElementSize) {
      maxElementSize = elementSize;
      maxElementName = schema.name;
    }
    if (schema.__isset.num_children) {
      currentDepth++;
      if (currentDepth > maxDepth) {
        maxDepth = currentDepth;
        maxDepthElementName = schema.name;
        maxDepthElementSize = elementSize;
      }
    }
  }

  // Add size of row groups
  size_t numRowGroups = metadata.row_groups.size();
  size_t totalColumns = 0;
  size_t totalEncodings = 0;
  size_t totalPaths = 0;
  size_t totalKeyValues = 0;

  totalSize += metadata.row_groups.size() * sizeof(thrift::RowGroup);
  for (const auto& rowGroup : metadata.row_groups) {
    // Add size of columns in each row group
    totalColumns += rowGroup.columns.size();
    totalSize += rowGroup.columns.size() * sizeof(thrift::ColumnChunk);
    for (const auto& column : rowGroup.columns) {
      totalSize += calculateColumnMetadataSize(column);
      // Add size of encodings
      totalEncodings += column.meta_data.encodings.size();
      // Add size of path in schema
      totalPaths += column.meta_data.path_in_schema.size();
      // Add size of key_value_metadata
      totalKeyValues += column.meta_data.key_value_metadata.size();
    }
  }

  // Add size of key_value_metadata
  size_t numKeyValues = metadata.key_value_metadata.size();
  totalSize += metadata.key_value_metadata.size() * sizeof(thrift::KeyValue);
  for (const auto& kv : metadata.key_value_metadata) {
    totalSize += kv.key.capacity();
    totalSize += kv.value.capacity();
  }

  // Add size of created_by
  totalSize += metadata.created_by.capacity();

  // Add size of column_orders
  totalSize += metadata.column_orders.size() * sizeof(thrift::ColumnOrder);

  // Add size of encryption_algorithm
  totalSize += sizeof(thrift::EncryptionAlgorithm);

  // Add size of footer_signing_key_metadata
  totalSize += metadata.footer_signing_key_metadata.capacity();

  auto encodingToString = [](thrift::Encoding::type encoding) -> std::string {
    switch (encoding) {
      case thrift::Encoding::PLAIN:
        return "PLAIN";
      case thrift::Encoding::PLAIN_DICTIONARY:
        return "PLAIN_DICTIONARY";
      case thrift::Encoding::RLE:
        return "RLE";
      case thrift::Encoding::BIT_PACKED:
        return "BIT_PACKED";
      case thrift::Encoding::DELTA_BINARY_PACKED:
        return "DELTA_BINARY_PACKED";
      case thrift::Encoding::DELTA_LENGTH_BYTE_ARRAY:
        return "DELTA_LENGTH_BYTE_ARRAY";
      case thrift::Encoding::DELTA_BYTE_ARRAY:
        return "DELTA_BYTE_ARRAY";
      case thrift::Encoding::RLE_DICTIONARY:
        return "RLE_DICTIONARY";
      case thrift::Encoding::BYTE_STREAM_SPLIT:
        return "BYTE_STREAM_SPLIT";
      default:
        return "UNKNOWN";
    }
  };

  // Track distinct encodings
  std::set<thrift::Encoding::type> distinctEncodings;
  std::map<thrift::Encoding::type, size_t> encodingCounts;

  for (const auto& rowGroup : metadata.row_groups) {
    for (const auto& column : rowGroup.columns) {
      for (const auto& encoding : column.meta_data.encodings) {
        distinctEncodings.insert(encoding);
        encodingCounts[encoding]++;
      }
      totalEncodings += column.meta_data.encodings.size();
      totalSize +=
          column.meta_data.encodings.size() * sizeof(thrift::Encoding::type);
    }
  }

  // Build encoding stats string and record metrics in one loop
  std::string encodingStats;
  for (const auto& [encoding, count] : encodingCounts) {
    encodingStats += fmt::format(
        "      {}: {} occurrences\n", encodingToString(encoding), count);

    // Record metric for each encoding type
  }

  // Only log if totalSize is more than 8x footerLength
  if (totalSize > 8 * footerLength) {
    VLOG(1) << fmt::format(
        "FileMetaData Statistics for {} \n"
        "  Footer length: {}):\n"
        "  FileMetaData object size: {}\n"
        "  Schema Statistics:\n"
        "    Number of schema elements: {}\n"
        "    Maximum schema depth: {} (Element: '{}', Size: {})\n"
        "    Largest schema element: '{}' (Size: {})\n"
        "    Number of leaf nodes: {}\n"
        "    Number of group nodes: {}\n"
        "    Average string length: {}\n"
        "    Maximum string length: {}\n"
        "  Row Group Statistics:\n"
        "    Number of row groups: {}\n"
        "    Total columns: {}\n"
        "    Average columns per row group: {}\n"
        "    Total encodings: {}\n"
        "    Distinct encodings: {}\n"
        "    Encoding distribution:\n{}"
        "    Total paths: {}\n"
        "    Total key-value pairs: {}",
        filename,
        succinctBytes(footerLength),
        succinctBytes(totalSize),
        numSchemaElements,
        maxDepth,
        maxDepthElementName,
        succinctBytes(maxDepthElementSize),
        maxElementName,
        succinctBytes(maxElementSize),
        numLeafNodes,
        numGroupNodes,
        (numSchemaElements > 0 ? totalStringLength / numSchemaElements : 0),
        maxStringLength,
        numRowGroups,
        totalColumns,
        (numRowGroups > 0 ? totalColumns / numRowGroups : 0),
        totalEncodings,
        distinctEncodings.size(),
        encodingStats,
        totalPaths,
        (totalKeyValues + numKeyValues));
  }
  return totalSize;
}

} // namespace facebook::velox::parquet
