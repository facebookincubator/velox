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
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

#include <thrift/lib/cpp2/FieldRef.h>

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
    size += heapStringSize(*kv.key());
    if (kv.value()) {
      size += heapStringSize(*kv.value());
    }
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
  if (column.file_path()) {
    size += heapStringSize(*column.file_path());
  }
  if (column.encrypted_column_metadata()) {
    size += heapStringSize(*column.encrypted_column_metadata());
  }
  // Optional crypto metadata. Only the heap-backed payloads of the active
  // union arm are added.
  if (column.crypto_metadata() &&
      column.crypto_metadata()->ENCRYPTION_WITH_COLUMN_KEY()) {
    const auto& key = *column.crypto_metadata()->ENCRYPTION_WITH_COLUMN_KEY();
    size += key.path_in_schema()->size() * sizeof(std::string);
    for (const auto& path : *key.path_in_schema()) {
      size += heapStringSize(path);
    }
    if (key.key_metadata()) {
      size += heapStringSize(*key.key_metadata());
    }
  }
  if (!column.meta_data()) {
    return size;
  }
  const auto& metaData = *column.meta_data();
  // Heap-backed vectors and the strings they contain inside ColumnMetaData.
  size += metaData.encodings()->size() * sizeof(thrift::Encoding);
  size += metaData.path_in_schema()->size() * sizeof(std::string);
  for (const auto& path : *metaData.path_in_schema()) {
    size += heapStringSize(path);
  }
  if (metaData.key_value_metadata()) {
    size += keyValueMetadataSize(*metaData.key_value_metadata());
  }
  if (metaData.encoding_stats()) {
    size +=
        metaData.encoding_stats()->size() * sizeof(thrift::PageEncodingStats);
  }

  // thrift::Statistics is an optional member of thrift::ColumnMetaData. Its
  // POD fields (null_count, distinct_count) are already in sizeof(ColumnChunk)
  // when present. Only the heap-backed string payloads need to be added here.
  if (metaData.statistics()) {
    const auto& stats = *metaData.statistics();
    if (stats.min()) {
      size += heapStringSize(*stats.min());
    }
    if (stats.max()) {
      size += heapStringSize(*stats.max());
    }
    if (stats.min_value()) {
      size += heapStringSize(*stats.min_value());
    }
    if (stats.max_value()) {
      size += heapStringSize(*stats.max_value());
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
  totalSize += metadata.schema()->size() * sizeof(thrift::SchemaElement);
  for (const auto& schema : *metadata.schema()) {
    totalSize += heapStringSize(*schema.name());
  }
  // Row groups vector heap allocation plus the columns vectors it owns.
  totalSize += metadata.row_groups()->size() * sizeof(thrift::RowGroup);
  for (const auto& rowGroup : *metadata.row_groups()) {
    for (const auto& column : *rowGroup.columns()) {
      totalSize += columnMetadataSize(column);
    }
  }
  if (metadata.key_value_metadata()) {
    totalSize += keyValueMetadataSize(*metadata.key_value_metadata());
  }
  if (metadata.created_by()) {
    totalSize += heapStringSize(*metadata.created_by());
  }
  if (metadata.column_orders()) {
    totalSize += metadata.column_orders()->size() * sizeof(thrift::ColumnOrder);
  }
  if (metadata.footer_signing_key_metadata()) {
    totalSize += heapStringSize(*metadata.footer_signing_key_metadata());
  }
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
  return columnChunkStats.min_value()
      ? load<T>(columnChunkStats.min_value()->data())
      : (columnChunkStats.min()
             ? std::optional<T>(load<T>(columnChunkStats.min()->data()))
             : std::nullopt);
}

template <typename T>
inline std::optional<T> getMax(const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.max_value()
      ? std::optional<T>(load<T>(columnChunkStats.max_value()->data()))
      : (columnChunkStats.max()
             ? std::optional<T>(load<T>(columnChunkStats.max()->data()))
             : std::nullopt);
}

template <>
inline std::optional<int64_t> getMin(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.min_value()
      ? decodeInt64Stat(*columnChunkStats.min_value())
      : (columnChunkStats.min() ? decodeInt64Stat(*columnChunkStats.min())
                                : std::nullopt);
}

template <>
inline std::optional<int64_t> getMax(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.max_value()
      ? decodeInt64Stat(*columnChunkStats.max_value())
      : (columnChunkStats.max() ? decodeInt64Stat(*columnChunkStats.max())
                                : std::nullopt);
}

template <>
inline std::optional<std::string> getMin(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.min_value()
      ? columnChunkStats.min_value().to_optional()
      : columnChunkStats.min().to_optional();
}

template <>
inline std::optional<std::string> getMax(
    const thrift::Statistics& columnChunkStats) {
  return columnChunkStats.max_value()
      ? columnChunkStats.max_value().to_optional()
      : columnChunkStats.max().to_optional();
}

std::optional<Timestamp> int64ToTimestamp(
    std::optional<int64_t> value,
    std::optional<thrift::ConvertedType> convertedType,
    const std::optional<thrift::LogicalType>& logicalType) {
  if (!value.has_value()) {
    return std::nullopt;
  }
  if (logicalType.has_value() &&
      logicalType->getType() == thrift::LogicalType::Type::TIMESTAMP) {
    const auto& unit = logicalType->get_TIMESTAMP().unit();
    if (unit->getType() == thrift::TimeUnit::Type::MILLIS) {
      return Timestamp::fromMillis(value.value());
    } else if (unit->getType() == thrift::TimeUnit::Type::NANOS) {
      return Timestamp::fromNanos(value.value());
    }
    return Timestamp::fromMicros(value.value());
  }
  if (convertedType == thrift::ConvertedType::TIMESTAMP_MILLIS) {
    return Timestamp::fromMillis(value.value());
  }
  return Timestamp::fromMicros(value.value());
}

std::unique_ptr<dwio::common::ColumnStatistics> buildColumnStatisticsFromThrift(
    const thrift::Statistics& columnChunkStats,
    const velox::Type& type,
    uint64_t numRowsInRowGroup,
    thrift::Type physicalType,
    std::optional<thrift::ConvertedType> convertedType,
    const std::optional<thrift::LogicalType>& logicalType) {
  std::optional<uint64_t> nullCount =
      columnChunkStats.null_count().to_optional();
  std::optional<uint64_t> valueCount = nullCount
      ? std::optional<uint64_t>(numRowsInRowGroup - nullCount.value())
      : std::nullopt;
  std::optional<bool> hasNull = columnChunkStats.null_count()
      ? std::optional<bool>(*columnChunkStats.null_count() > 0)
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
    case TypeKind::TIMESTAMP:
      if (physicalType == thrift::Type::INT64 &&
          (convertedType.has_value() || logicalType.has_value())) {
        return std::make_unique<dwio::common::TimestampColumnStatistics>(
            valueCount,
            hasNull,
            std::nullopt,
            std::nullopt,
            int64ToTimestamp(
                getMin<int64_t>(columnChunkStats), convertedType, logicalType),
            int64ToTimestamp(
                getMax<int64_t>(columnChunkStats), convertedType, logicalType));
      }
      return std::make_unique<dwio::common::ColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt);

    default:
      return std::make_unique<dwio::common::ColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt);
  }
}

common::CompressionKind thriftCodecToCompressionKind(
    thrift::CompressionCodec codec) {
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
      return common::CompressionKind::CompressionKind_LZ4_HADOOP;
    case thrift::CompressionCodec::ZSTD:
      return common::CompressionKind::CompressionKind_ZSTD;
    case thrift::CompressionCodec::LZ4_RAW:
      return common::CompressionKind::CompressionKind_LZ4;
    default: {
      std::string_view name;
      apache::thrift::TEnumTraits<thrift::CompressionCodec>::findName(
          codec, &name);
      VELOX_UNSUPPORTED("Unsupported compression type: {}", name);
    }
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
  return apache::thrift::can_throw(
      *thriftColumnChunkPtr(ptr_)->meta_data()->num_values());
}

bool ColumnChunkMetaDataPtr::hasMetadata() const {
  return thriftColumnChunkPtr(ptr_)->meta_data().has_value();
}

bool ColumnChunkMetaDataPtr::hasStatistics() const {
  return hasMetadata() &&
      apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
          ->statistics()
          .has_value();
}

bool ColumnChunkMetaDataPtr::hasDictionaryPageOffset() const {
  return hasMetadata() &&
      apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
          ->dictionary_page_offset()
          .has_value();
}

std::unique_ptr<dwio::common::ColumnStatistics>
ColumnChunkMetaDataPtr::getColumnStatistics(
    const TypePtr type,
    int64_t numRows,
    std::optional<thrift::ConvertedType> convertedType,
    const std::optional<thrift::LogicalType>& logicalType) {
  VELOX_CHECK(hasStatistics());
  const auto& metaData =
      apache::thrift::can_throw(*thriftColumnChunkPtr(ptr_)->meta_data());
  return buildColumnStatisticsFromThrift(
      apache::thrift::can_throw(*metaData.statistics()),
      *type,
      numRows,
      apache::thrift::can_throw(*metaData.type()),
      convertedType,
      logicalType);
}

std::string ColumnChunkMetaDataPtr::getColumnMetadataStatsMinValue() {
  VELOX_CHECK(hasStatistics());
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(
           apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
               ->statistics())
           ->min_value());
}

std::string ColumnChunkMetaDataPtr::getColumnMetadataStatsMaxValue() {
  VELOX_CHECK(hasStatistics());
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(
           apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
               ->statistics())
           ->max_value());
}

int64_t ColumnChunkMetaDataPtr::getColumnMetadataStatsNullCount() {
  VELOX_CHECK(hasStatistics());
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(
           apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
               ->statistics())
           ->null_count());
}

int64_t ColumnChunkMetaDataPtr::dataPageOffset() const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
           ->data_page_offset());
}

int64_t ColumnChunkMetaDataPtr::dictionaryPageOffset() const {
  VELOX_CHECK(hasDictionaryPageOffset());
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
           ->dictionary_page_offset());
}

common::CompressionKind ColumnChunkMetaDataPtr::compression() const {
  return thriftCodecToCompressionKind(
      apache::thrift::can_throw(
          *apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
               ->codec()));
}

int64_t ColumnChunkMetaDataPtr::totalCompressedSize() const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
           ->total_compressed_size());
}

std::vector<thrift::Encoding> ColumnChunkMetaDataPtr::encodings() const {
  return *apache::thrift::can_throw(
      thriftColumnChunkPtr(ptr_)->meta_data()->encodings());
}

std::vector<std::string> ColumnChunkMetaDataPtr::pathInSchema() const {
  return *apache::thrift::can_throw(
      thriftColumnChunkPtr(ptr_)->meta_data()->path_in_schema());
}

int64_t ColumnChunkMetaDataPtr::totalUncompressedSize() const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(thriftColumnChunkPtr(ptr_)->meta_data())
           ->total_uncompressed_size());
}

FOLLY_ALWAYS_INLINE const thrift::RowGroup* thriftRowGroupPtr(
    const void* metadata) {
  return reinterpret_cast<const thrift::RowGroup*>(metadata);
}

RowGroupMetaDataPtr::RowGroupMetaDataPtr(const void* metadata)
    : ptr_(metadata) {}

RowGroupMetaDataPtr::~RowGroupMetaDataPtr() = default;

int RowGroupMetaDataPtr::numColumns() const {
  return thriftRowGroupPtr(ptr_)->columns()->size();
}

int32_t RowGroupMetaDataPtr::sortingColumnIdx(int i) const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(*thriftRowGroupPtr(ptr_)->sorting_columns())[i]
           .column_idx());
}

bool RowGroupMetaDataPtr::sortingColumnDescending(int i) const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(*thriftRowGroupPtr(ptr_)->sorting_columns())[i]
           .descending());
}

bool RowGroupMetaDataPtr::sortingColumnNullsFirst(int i) const {
  return apache::thrift::can_throw(
      *apache::thrift::can_throw(*thriftRowGroupPtr(ptr_)->sorting_columns())[i]
           .nulls_first());
}

int64_t RowGroupMetaDataPtr::numRows() const {
  return *thriftRowGroupPtr(ptr_)->num_rows();
}

int64_t RowGroupMetaDataPtr::totalByteSize() const {
  return *thriftRowGroupPtr(ptr_)->total_byte_size();
}

bool RowGroupMetaDataPtr::hasFileOffset() const {
  return thriftRowGroupPtr(ptr_)->file_offset().has_value();
}

int64_t RowGroupMetaDataPtr::fileOffset() const {
  return apache::thrift::can_throw(*thriftRowGroupPtr(ptr_)->file_offset());
}

bool RowGroupMetaDataPtr::hasTotalCompressedSize() const {
  return thriftRowGroupPtr(ptr_)->total_compressed_size().has_value();
}

int64_t RowGroupMetaDataPtr::totalCompressedSize() const {
  return apache::thrift::can_throw(
      *thriftRowGroupPtr(ptr_)->total_compressed_size());
}

ColumnChunkMetaDataPtr RowGroupMetaDataPtr::columnChunk(int i) const {
  return ColumnChunkMetaDataPtr(
      reinterpret_cast<const void*>(&(*thriftRowGroupPtr(ptr_)->columns())[i]));
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
          &(*thriftFileMetaDataPtr(ptr_)->row_groups())[i]));
}

int64_t FileMetaDataPtr::numRows() const {
  return *thriftFileMetaDataPtr(ptr_)->num_rows();
}

int FileMetaDataPtr::numRowGroups() const {
  return thriftFileMetaDataPtr(ptr_)->row_groups()->size();
}

int64_t FileMetaDataPtr::keyValueMetadataSize() const {
  return apache::thrift::can_throw(
             thriftFileMetaDataPtr(ptr_)->key_value_metadata())
      ->size();
}

bool FileMetaDataPtr::keyValueMetadataContains(
    const std::string_view key) const {
  auto thriftKeyValueMeta = apache::thrift::can_throw(
      *thriftFileMetaDataPtr(ptr_)->key_value_metadata());
  for (const auto& kv : thriftKeyValueMeta) {
    if (*kv.key() == key) {
      return true;
    }
  }
  return false;
}

std::string FileMetaDataPtr::keyValueMetadataValue(
    const std::string_view key) const {
  int thriftKeyValueMetaSize = keyValueMetadataSize();
  for (size_t i = 0; i < thriftKeyValueMetaSize; i++) {
    if (key ==
        apache::thrift::can_throw(
            *apache::thrift::can_throw(
                 *thriftFileMetaDataPtr(ptr_)->key_value_metadata())[i]
                 .key())) {
      return apache::thrift::can_throw(
          *apache::thrift::can_throw(
               *thriftFileMetaDataPtr(ptr_)->key_value_metadata())[i]
               .value());
    }
  }
  VELOX_FAIL("Input key {} is not in the key value metadata", key);
}

std::string FileMetaDataPtr::createdBy() const {
  return apache::thrift::can_throw(*thriftFileMetaDataPtr(ptr_)->created_by());
}

size_t ColumnChunkMetaDataPtr::estimateColumnMetadataSize() const {
  return columnMetadataSize(*thriftColumnChunkPtr(ptr_));
}

size_t FileMetaDataPtr::estimateFileMetadataSize() const {
  return fileMetadataSize(*thriftFileMetaDataPtr(ptr_));
}

} // namespace facebook::velox::parquet
