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

#include "velox/dwio/parquet/reader/Metadata.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

template <typename T>
inline const T load(const char* ptr) {
  T ret;
  std::memcpy(&ret, ptr, sizeof(ret));
  return ret;
}

template <typename T>
inline std::optional<T> getMin(const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.min_value
      ? load<T>(columnChunkStats.min_value.data())
      : (columnChunkStats.__isset.min
             ? std::optional<T>(load<T>(columnChunkStats.min.data()))
             : std::nullopt);
}

template <typename T>
inline std::optional<T> getMax(const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.max_value
      ? std::optional<T>(load<T>(columnChunkStats.max_value.data()))
      : (columnChunkStats.__isset.max
             ? std::optional<T>(load<T>(columnChunkStats.max.data()))
             : std::nullopt);
}

template <>
inline std::optional<std::string> getMin(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.min_value
      ? std::optional(columnChunkStats.min_value)
      : (columnChunkStats.__isset.min ? std::optional(columnChunkStats.min)
                                      : std::nullopt);
}

template <>
inline std::optional<std::string> getMax(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.max_value
      ? std::optional(columnChunkStats.max_value)
      : (columnChunkStats.__isset.max ? std::optional(columnChunkStats.max)
                                      : std::nullopt);
}

std::unique_ptr<dwio::common::ColumnStatistics> buildColumnStatisticsFromThrift(
    const thrift::Statistics& columnChunkStats,
    const velox::Type& type,
    uint64_t numRowsInRowGroup) {
  std::optional<uint64_t> nullCount = columnChunkStats.__isset.null_count
      ? std::optional<uint64_t>(columnChunkStats.null_count)
      : std::nullopt;
  std::optional<uint64_t> valueCount = nullCount.has_value()
      ? std::optional<uint64_t>(numRowsInRowGroup - nullCount.value())
      : std::nullopt;
  std::optional<bool> hasNull = columnChunkStats.__isset.null_count
      ? std::optional<bool>(columnChunkStats.null_count > 0)
      : std::nullopt;

  switch (type.kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<dwio::common::BooleanColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt, std::nullopt);
    case TypeKind::TINYINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int8_t>(columnChunkStats),
          getMax<int8_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::SMALLINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int16_t>(columnChunkStats),
          getMax<int16_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::INTEGER:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int32_t>(columnChunkStats),
          getMax<int32_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::BIGINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int64_t>(columnChunkStats),
          getMax<int64_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::REAL:
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<float>(columnChunkStats),
          getMax<float>(columnChunkStats),
          std::nullopt);
    case TypeKind::DOUBLE:
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<double>(columnChunkStats),
          getMax<double>(columnChunkStats),
          std::nullopt);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return std::make_unique<dwio::common::StringColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<std::string>(columnChunkStats),
          getMax<std::string>(columnChunkStats),
          std::nullopt);

    default:
      return std::make_unique<dwio::common::ColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt);
  }
}

common::CompressionKind thriftCodecToCompressionKind(
    thrift::CompressionCodec::type codec) {
  switch (codec) {
    case thrift::CompressionCodec::UNCOMPRESSED:
      return common::CompressionKind::CompressionKind_NONE;
      break;
    case thrift::CompressionCodec::SNAPPY:
      return common::CompressionKind::CompressionKind_SNAPPY;
      break;
    case thrift::CompressionCodec::GZIP:
      return common::CompressionKind::CompressionKind_GZIP;
      break;
    case thrift::CompressionCodec::LZO:
      return common::CompressionKind::CompressionKind_LZO;
      break;
    case thrift::CompressionCodec::LZ4:
      return common::CompressionKind::CompressionKind_LZ4;
      break;
    case thrift::CompressionCodec::ZSTD:
      return common::CompressionKind::CompressionKind_ZSTD;
      break;
    case thrift::CompressionCodec::LZ4_RAW:
      return common::CompressionKind::CompressionKind_LZ4;
    default:
      VELOX_UNSUPPORTED(
          "Unsupported compression type: " +
          facebook::velox::parquet::thrift::to_string(codec));
      break;
  }
}

ColumnChunkMetaDataPtr::ColumnChunkMetaDataPtr(const void* metadata)
    : ptr_(metadata) {}

ColumnChunkMetaDataPtr::~ColumnChunkMetaDataPtr() = default;

FOLLY_ALWAYS_INLINE const thrift::ColumnChunk* thriftColumnChunkPtr(
    const void* metadata) {
  return reinterpret_cast<const thrift::ColumnChunk*>(metadata);
}

int64_t ColumnChunkMetaDataPtr::numValues() const {
  return thriftColumnChunkPtr(ptr_)->meta_data.num_values;
}

bool ColumnChunkMetaDataPtr::hasMetadata() const {
  return thriftColumnChunkPtr(ptr_)->__isset.meta_data;
}

bool ColumnChunkMetaDataPtr::hasStatistics() const {
  return hasMetadata() &&
      thriftColumnChunkPtr(ptr_)->meta_data.__isset.statistics;
}

bool ColumnChunkMetaDataPtr::hasDictionaryPageOffset() const {
  return hasMetadata() &&
      thriftColumnChunkPtr(ptr_)->meta_data.__isset.dictionary_page_offset;
}

std::unique_ptr<dwio::common::ColumnStatistics>
ColumnChunkMetaDataPtr::getColumnStatistics(
    const TypePtr type,
    int64_t numRows) {
  VELOX_CHECK(hasStatistics());
  return buildColumnStatisticsFromThrift(
      thriftColumnChunkPtr(ptr_)->meta_data.statistics, *type, numRows);
};

std::string ColumnChunkMetaDataPtr::getColumnMetadataStatsMinValue() {
  VELOX_CHECK(hasStatistics());
  return thriftColumnChunkPtr(ptr_)->meta_data.statistics.min_value;
}

std::string ColumnChunkMetaDataPtr::getColumnMetadataStatsMaxValue() {
  VELOX_CHECK(hasStatistics());
  return thriftColumnChunkPtr(ptr_)->meta_data.statistics.max_value;
}

int64_t ColumnChunkMetaDataPtr::getColumnMetadataStatsNullCount() {
  VELOX_CHECK(hasStatistics());
  return thriftColumnChunkPtr(ptr_)->meta_data.statistics.null_count;
}

int64_t ColumnChunkMetaDataPtr::dataPageOffset() const {
  return thriftColumnChunkPtr(ptr_)->meta_data.data_page_offset;
}

int64_t ColumnChunkMetaDataPtr::dictionaryPageOffset() const {
  VELOX_CHECK(hasDictionaryPageOffset());
  return thriftColumnChunkPtr(ptr_)->meta_data.dictionary_page_offset;
}

common::CompressionKind ColumnChunkMetaDataPtr::compression() const {
  return thriftCodecToCompressionKind(
      thriftColumnChunkPtr(ptr_)->meta_data.codec);
}

int64_t ColumnChunkMetaDataPtr::totalCompressedSize() const {
  return thriftColumnChunkPtr(ptr_)->meta_data.total_compressed_size;
}

int64_t ColumnChunkMetaDataPtr::totalUncompressedSize() const {
  return thriftColumnChunkPtr(ptr_)->meta_data.total_uncompressed_size;
}

FOLLY_ALWAYS_INLINE const thrift::RowGroup* thriftRowGroupPtr(
    const void* metadata) {
  return reinterpret_cast<const thrift::RowGroup*>(metadata);
}

RowGroupMetaDataPtr::RowGroupMetaDataPtr(const void* metadata)
    : ptr_(metadata) {}

RowGroupMetaDataPtr::~RowGroupMetaDataPtr() = default;

int RowGroupMetaDataPtr::numColumns() const {
  return thriftRowGroupPtr(ptr_)->columns.size();
}

int32_t RowGroupMetaDataPtr::sortingColumnIdx(int i) const {
  return thriftRowGroupPtr(ptr_)->sorting_columns[i].column_idx;
}

bool RowGroupMetaDataPtr::sortingColumnDescending(int i) const {
  return thriftRowGroupPtr(ptr_)->sorting_columns[i].descending;
}

bool RowGroupMetaDataPtr::sortingColumnNullsFirst(int i) const {
  return thriftRowGroupPtr(ptr_)->sorting_columns[i].nulls_first;
}

int64_t RowGroupMetaDataPtr::numRows() const {
  return thriftRowGroupPtr(ptr_)->num_rows;
}

int64_t RowGroupMetaDataPtr::totalByteSize() const {
  return thriftRowGroupPtr(ptr_)->total_byte_size;
}

bool RowGroupMetaDataPtr::hasFileOffset() const {
  return thriftRowGroupPtr(ptr_)->__isset.file_offset;
}

int64_t RowGroupMetaDataPtr::fileOffset() const {
  return thriftRowGroupPtr(ptr_)->file_offset;
}

bool RowGroupMetaDataPtr::hasTotalCompressedSize() const {
  return thriftRowGroupPtr(ptr_)->__isset.total_compressed_size;
}

int64_t RowGroupMetaDataPtr::totalCompressedSize() const {
  return thriftRowGroupPtr(ptr_)->total_compressed_size;
}

ColumnChunkMetaDataPtr RowGroupMetaDataPtr::columnChunk(int i) const {
  return ColumnChunkMetaDataPtr(
      reinterpret_cast<const void*>(&thriftRowGroupPtr(ptr_)->columns[i]));
}

FOLLY_ALWAYS_INLINE const thrift::FileMetaData* thriftFileMetaDataPtr(
    const void* metadata) {
  return reinterpret_cast<const thrift::FileMetaData*>(metadata);
}

FileMetaDataPtr::FileMetaDataPtr(const void* metadata) : ptr_(metadata) {}

FileMetaDataPtr::~FileMetaDataPtr() = default;

RowGroupMetaDataPtr FileMetaDataPtr::rowGroup(int i) const {
  return RowGroupMetaDataPtr(reinterpret_cast<const void*>(
      &thriftFileMetaDataPtr(ptr_)->row_groups[i]));
}

int64_t FileMetaDataPtr::numRows() const {
  return thriftFileMetaDataPtr(ptr_)->num_rows;
}

int FileMetaDataPtr::numRowGroups() const {
  return thriftFileMetaDataPtr(ptr_)->row_groups.size();
}

int64_t FileMetaDataPtr::keyValueMetadataSize() const {
  return thriftFileMetaDataPtr(ptr_)->key_value_metadata.size();
}

bool FileMetaDataPtr::keyValueMetadataContains(
    const std::string_view key) const {
  auto thriftKeyValueMeta = thriftFileMetaDataPtr(ptr_)->key_value_metadata;
  for (const auto& kv : thriftKeyValueMeta) {
    if (kv.key == key) {
      return true;
    }
  }
  return false;
}

std::string FileMetaDataPtr::keyValueMetadataValue(
    const std::string_view key) const {
  int thriftKeyValueMetaSize = keyValueMetadataSize();
  for (size_t i = 0; i < thriftKeyValueMetaSize; i++) {
    if (key == thriftFileMetaDataPtr(ptr_)->key_value_metadata[i].key) {
      return thriftFileMetaDataPtr(ptr_)->key_value_metadata[i].value;
    }
  }
  VELOX_FAIL(fmt::format("Input key {} is not in the key value metadata", key));
}

std::string FileMetaDataPtr::createdBy() const {
  return thriftFileMetaDataPtr(ptr_)->created_by;
}

size_t calculateColumnMetadataSize(const thrift::ColumnChunk& column) {
  size_t size = 0;
  size += sizeof(thrift::ColumnChunk);
  size += sizeof(thrift::ColumnMetaData);
  size += column.meta_data.encodings.size() * sizeof(thrift::Encoding::type);
  size += column.meta_data.path_in_schema.size() * sizeof(std::string);
  for (const auto& path : column.meta_data.path_in_schema) {
    size += path.capacity();
  }
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

size_t analyzeFileMetadata(
    const thrift::FileMetaData& metadata,
    const std::string& filename,
    uint64_t footerLength) {
  size_t totalSize = sizeof(thrift::FileMetaData);

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
  size_t numRowGroups = metadata.row_groups.size();
  size_t totalColumns = 0;
  size_t totalEncodings = 0;
  size_t totalPaths = 0;
  size_t totalKeyValues = 0;

  totalSize += metadata.row_groups.size() * sizeof(thrift::RowGroup);
  for (const auto& rowGroup : metadata.row_groups) {
    totalColumns += rowGroup.columns.size();
    totalSize += rowGroup.columns.size() * sizeof(thrift::ColumnChunk);
    for (const auto& column : rowGroup.columns) {
      totalSize += calculateColumnMetadataSize(column);
      totalEncodings += column.meta_data.encodings.size();
      totalPaths += column.meta_data.path_in_schema.size();
      totalKeyValues += column.meta_data.key_value_metadata.size();
    }
  }

  size_t numKeyValues = metadata.key_value_metadata.size();
  totalSize += metadata.key_value_metadata.size() * sizeof(thrift::KeyValue);
  for (const auto& kv : metadata.key_value_metadata) {
    totalSize += kv.key.capacity();
    totalSize += kv.value.capacity();
  }

  totalSize += metadata.created_by.capacity();
  totalSize += metadata.column_orders.size() * sizeof(thrift::ColumnOrder);
  totalSize += sizeof(thrift::EncryptionAlgorithm);
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

  std::string encodingStats;
  for (const auto& [encoding, count] : encodingCounts) {
    encodingStats += fmt::format(
        "      {}: {} occurrences\n", encodingToString(encoding), count);
  }
  return totalSize;
}

} // namespace facebook::velox::parquet
