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

namespace {

// Returns the heap bytes held by `s`, skipping SSO strings whose buffer is
// inline within the std::string object and already counted by the containing
// struct's sizeof. Detects SSO by checking whether s.data() points inside the
// std::string itself. This relies on the standard library placing the SSO
// buffer inside the object, which is true on libc++, libstdc++, and MSVC —
// the implementations Velox supports — but is not guaranteed by the C++
// standard. On a hypothetical implementation that stored SSO data outside
// the object, the check would simply treat short strings as heap-allocated
// and slightly over-report, never under-report.
size_t heapStringSize(const std::string& s) {
  const auto* data = reinterpret_cast<const char*>(s.data());
  const auto* begin = reinterpret_cast<const char*>(&s);
  if (data >= begin && data < begin + sizeof(std::string)) {
    return 0;
  }
  return s.capacity();
}

// Returns the dynamically-allocated byte size held by a thrift key/value
// metadata vector — the inline thrift::KeyValue array bytes plus the heap
// storage backing each key and value string.
size_t keyValueMetadataSize(const std::vector<thrift::KeyValue>& keyValues) {
  size_t size = keyValues.size() * sizeof(thrift::KeyValue);
  for (const auto& kv : keyValues) {
    size += heapStringSize(kv.key);
    size += heapStringSize(kv.value);
  }
  return size;
}

// Returns the estimated total bytes held by `column`:
// sizeof(thrift::ColumnChunk) plus every dynamically allocated vector and
// string reachable through it. Inline sub-structs (thrift::ColumnMetaData,
// thrift::Statistics, thrift::ColumnCryptoMetaData) live inside
// sizeof(ColumnChunk) and are NOT counted again here; only their dynamically
// allocated payloads (vectors and string buffers) are added.
size_t columnMetadataSize(const thrift::ColumnChunk& column) {
  size_t size = sizeof(thrift::ColumnChunk);
  // Optional heap-backed strings on ColumnChunk itself.
  if (column.__isset.file_path) {
    size += heapStringSize(column.file_path);
  }
  if (column.__isset.encrypted_column_metadata) {
    size += heapStringSize(column.encrypted_column_metadata);
  }
  // Optional crypto metadata. The union's inner structs are inline within
  // ColumnChunk via __isset, so only their heap-backed payloads are added.
  if (column.__isset.crypto_metadata &&
      column.crypto_metadata.__isset.ENCRYPTION_WITH_COLUMN_KEY) {
    const auto& key = column.crypto_metadata.ENCRYPTION_WITH_COLUMN_KEY;
    size += key.path_in_schema.size() * sizeof(std::string);
    for (const auto& path : key.path_in_schema) {
      size += heapStringSize(path);
    }
    if (key.__isset.key_metadata) {
      size += heapStringSize(key.key_metadata);
    }
  }
  // Heap-backed vectors and the strings they contain inside ColumnMetaData.
  size += column.meta_data.encodings.size() * sizeof(thrift::Encoding::type);
  size += column.meta_data.path_in_schema.size() * sizeof(std::string);
  for (const auto& path : column.meta_data.path_in_schema) {
    size += heapStringSize(path);
  }
  size += keyValueMetadataSize(column.meta_data.key_value_metadata);
  if (column.meta_data.__isset.encoding_stats) {
    size += column.meta_data.encoding_stats.size() *
        sizeof(thrift::PageEncodingStats);
  }

  // thrift::Statistics is an inline member of thrift::ColumnMetaData. Its POD
  // fields (null_count, distinct_count) are already in sizeof(ColumnChunk).
  // Only the heap-backed string payloads need to be added here.
  if (column.meta_data.__isset.statistics) {
    const auto& stats = column.meta_data.statistics;
    if (stats.__isset.min) {
      size += heapStringSize(stats.min);
    }
    if (stats.__isset.max) {
      size += heapStringSize(stats.max);
    }
    if (stats.__isset.min_value) {
      size += heapStringSize(stats.min_value);
    }
    if (stats.__isset.max_value) {
      size += heapStringSize(stats.max_value);
    }
  }
  return size;
}

// Estimates the heap memory held by `metadata` after thrift deserialization.
// Returns sizeof(thrift::FileMetaData) plus the bytes of every dynamically
// allocated vector and string reachable through it. Inline POD members and
// inline thrift sub-structs (e.g. thrift::EncryptionAlgorithm,
// thrift::SchemaElement::logicalType) are part of the parent sizeof and are
// not double-counted here.
size_t fileMetadataSize(const thrift::FileMetaData& metadata) {
  size_t totalSize = sizeof(thrift::FileMetaData);
  // Schema vector heap allocation plus per-element name strings.
  totalSize += metadata.schema.size() * sizeof(thrift::SchemaElement);
  for (const auto& schema : metadata.schema) {
    totalSize += heapStringSize(schema.name);
  }
  // Row groups vector heap allocation plus the columns vectors it owns.
  totalSize += metadata.row_groups.size() * sizeof(thrift::RowGroup);
  for (const auto& rowGroup : metadata.row_groups) {
    for (const auto& column : rowGroup.columns) {
      totalSize += columnMetadataSize(column);
    }
  }
  totalSize += keyValueMetadataSize(metadata.key_value_metadata);
  totalSize += heapStringSize(metadata.created_by);
  totalSize += metadata.column_orders.size() * sizeof(thrift::ColumnOrder);
  totalSize += heapStringSize(metadata.footer_signing_key_metadata);
  return totalSize;
}

} // namespace

template <typename T>
inline const T load(const char* ptr) {
  T ret;
  std::memcpy(&ret, ptr, sizeof(ret));
  return ret;
}

inline std::optional<int64_t> decodeInt64Stat(const std::string& bytes) {
  switch (bytes.size()) {
    case sizeof(int64_t):
      return load<int64_t>(bytes.data());
    case sizeof(int32_t):
      // Parquet stores Time types as int32_t (for milliseconds), but Velox's
      // Time type is always int64_t, so we need to load sizeof(int32_t) and
      // cast to int64_t.
      return static_cast<int64_t>(load<int32_t>(bytes.data()));
    default:
      return std::nullopt;
  }
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
inline std::optional<int64_t> getMin(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.min_value
      ? decodeInt64Stat(columnChunkStats.min_value)
      : (columnChunkStats.__isset.min ? decodeInt64Stat(columnChunkStats.min)
                                      : std::nullopt);
}

template <>
inline std::optional<int64_t> getMax(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.__isset.max_value
      ? decodeInt64Stat(columnChunkStats.max_value)
      : (columnChunkStats.__isset.max ? decodeInt64Stat(columnChunkStats.max)
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

std::vector<thrift::Encoding::type> ColumnChunkMetaDataPtr::encodings() const {
  return thriftColumnChunkPtr(ptr_)->meta_data.encodings;
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
  VELOX_FAIL("Input key {} is not in the key value metadata", key);
}

std::string FileMetaDataPtr::createdBy() const {
  return thriftFileMetaDataPtr(ptr_)->created_by;
}

size_t ColumnChunkMetaDataPtr::estimateColumnMetadataSize() const {
  return columnMetadataSize(*thriftColumnChunkPtr(ptr_));
}

size_t FileMetaDataPtr::estimateFileMetadataSize() const {
  return fileMetadataSize(*thriftFileMetaDataPtr(ptr_));
}

} // namespace facebook::velox::parquet
