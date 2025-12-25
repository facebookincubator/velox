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
#include "arrow/type.h"
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

// Retrieves the minimum or maximum decimal value from column chunk statistics
// based on the specified physical type and precision. Supports three physical
// types to meet the requirements of Arrow: INT32, INT64, and
// FIXED_LEN_BYTE_ARRAY.
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/arrow/schema.cc#L360-L365
// When min is true, it retrieves the minimum value; otherwise, it retrieves the
// maximum value.
template <typename T, bool min>
inline std::optional<T> getDecimalMinMax(
    const thrift::Statistics& columnChunkStats,
    thrift::Type::type physicalType,
    uint8_t precision) {
  const char* data;
  if constexpr (min) {
    data = columnChunkStats.__isset.min_value
        ? columnChunkStats.min_value.data()
        : (columnChunkStats.__isset.min ? columnChunkStats.min.data()
                                        : nullptr);
  } else {
    data = columnChunkStats.__isset.max_value
        ? columnChunkStats.max_value.data()
        : (columnChunkStats.__isset.max ? columnChunkStats.max.data()
                                        : nullptr);
  }

  switch (physicalType) {
    case thrift::Type::type::INT32:
      return data ? std::optional<T>(load<int32_t>(data)) : std::nullopt;
    case thrift::Type::type::INT64:
      return data ? std::optional<T>(load<int64_t>(data)) : std::nullopt;
    case thrift::Type::type::FIXED_LEN_BYTE_ARRAY: {
      if (data == nullptr) {
        return std::nullopt;
      }
      const auto size = arrow::DecimalType::DecimalSize(precision);
      // Extracts min or max from big-endian string.
      T value = 0;
      for (auto i = 0; i < size; ++i) {
        value = (value << 8) | (data[i] & 0xFF);
      }

      // If the value is negative, extend the sign bit.
      const bool isNegative = (data[0] & 0x80) != 0;
      if (isNegative) {
        value -= (static_cast<T>(1) << (size * 8));
      }
      return std::optional<T>(value);
    }
    default:
      VELOX_UNREACHABLE();
  }
}

std::unique_ptr<dwio::common::ColumnStatistics> buildColumnStatisticsFromThrift(
    const thrift::Statistics& columnChunkStats,
    const velox::Type& type,
    uint64_t numRowsInRowGroup,
    thrift::Type::type physicalType) {
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
      return std::make_unique<dwio::common::IntegerColumnStatistics<>>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int8_t>(columnChunkStats),
          getMax<int8_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::SMALLINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics<>>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int16_t>(columnChunkStats),
          getMax<int16_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::INTEGER:
      return std::make_unique<dwio::common::IntegerColumnStatistics<>>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getMin<int32_t>(columnChunkStats),
          getMax<int32_t>(columnChunkStats),
          std::nullopt);
    case TypeKind::BIGINT: {
      std::optional<int64_t> min = type.isDecimal()
          ? getDecimalMinMax<int64_t, true>(
                columnChunkStats,
                physicalType,
                getDecimalPrecisionScale(type).first)
          : getMin<int64_t>(columnChunkStats);
      std::optional<int64_t> max = type.isDecimal()
          ? getDecimalMinMax<int64_t, false>(
                columnChunkStats,
                physicalType,
                getDecimalPrecisionScale(type).first)
          : getMax<int64_t>(columnChunkStats);
      return std::make_unique<dwio::common::IntegerColumnStatistics<>>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          min,
          max,
          std::nullopt);
    }
    case TypeKind::HUGEINT: {
      const auto precision = getDecimalPrecisionScale(type).first;
      return std::make_unique<dwio::common::IntegerColumnStatistics<int128_t>>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          getDecimalMinMax<int128_t, true>(
              columnChunkStats, physicalType, precision),
          getDecimalMinMax<int128_t, false>(
              columnChunkStats, physicalType, precision),
          std::nullopt);
    }
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
    case thrift::CompressionCodec::SNAPPY:
      return common::CompressionKind::CompressionKind_SNAPPY;
    case thrift::CompressionCodec::GZIP:
      return common::CompressionKind::CompressionKind_GZIP;
    case thrift::CompressionCodec::LZO:
      return common::CompressionKind::CompressionKind_LZO;
    case thrift::CompressionCodec::LZ4:
      return common::CompressionKind::CompressionKind_LZ4;
    case thrift::CompressionCodec::ZSTD:
      return common::CompressionKind::CompressionKind_ZSTD;
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
  const auto physicalType = thriftColumnChunkPtr(ptr_)->meta_data.type;
  return buildColumnStatisticsFromThrift(
      thriftColumnChunkPtr(ptr_)->meta_data.statistics,
      *type,
      numRows,
      physicalType);
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
  return RowGroupMetaDataPtr(
      reinterpret_cast<const void*>(
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

} // namespace facebook::velox::parquet
