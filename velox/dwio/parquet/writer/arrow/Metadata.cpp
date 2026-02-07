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

#include "velox/dwio/parquet/writer/arrow/Metadata.h"

#include <algorithm>
#include <cinttypes>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/io/memory.h"
#include "arrow/util/key_value_metadata.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileDecryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/SchemaInternal.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"

namespace facebook::velox::parquet::arrow {

const ApplicationVersion& ApplicationVersion::PARQUET_251_FIXED_VERSION() {
  static ApplicationVersion version("parquet-mr", 1, 8, 0);
  return version;
}

const ApplicationVersion& ApplicationVersion::PARQUET_816_FIXED_VERSION() {
  static ApplicationVersion version("parquet-mr", 1, 2, 9);
  return version;
}

const ApplicationVersion&
ApplicationVersion::PARQUET_CPP_FIXED_STATS_VERSION() {
  static ApplicationVersion version("parquet-cpp", 1, 3, 0);
  return version;
}

const ApplicationVersion& ApplicationVersion::PARQUET_MR_FIXED_STATS_VERSION() {
  static ApplicationVersion version("parquet-mr", 1, 10, 0);
  return version;
}

const ApplicationVersion&
ApplicationVersion::PARQUET_CPP_10353_FIXED_VERSION() {
  // Parquet-cpp versions released prior to Arrow 3.0 would write DataPageV2.
  // Pages with is_compressed==0 but still write compressed data. (See:
  // ARROW-10353). Parquet 1.5.1 had this problem, and after that we switched
  // to. The application name "parquet-cpp-arrow", so this version is fake.
  static ApplicationVersion version("parquet-cpp", 2, 0, 0);
  return version;
}

std::string parquetVersionToString(ParquetVersion::type ver) {
  switch (ver) {
    case ParquetVersion::PARQUET_1_0:
      return "1.0";
      ARROW_SUPPRESS_DEPRECATION_WARNING
    case ParquetVersion::PARQUET_2_0:
      return "pseudo-2.0";
      ARROW_UNSUPPRESS_DEPRECATION_WARNING
    case ParquetVersion::PARQUET_2_4:
      return "2.4";
    case ParquetVersion::PARQUET_2_6:
      return "2.6";
  }

  // This should be unreachable.
  return "UNKNOWN";
}

template <typename DType>
static std::shared_ptr<Statistics> makeTypedColumnStats(
    const facebook::velox::parquet::thrift::ColumnMetaData& metadata,
    const ColumnDescriptor* descr) {
  // If ColumnOrder is defined, return max_value and min_value.
  const auto& stats = metadata.statistics;
  if (descr->columnOrder().order() == ColumnOrder::kTypeDefinedOrder) {
    return makeStatistics<DType>(
        descr,
        stats.min_value,
        stats.max_value,
        metadata.num_values - stats.null_count,
        stats.null_count,
        stats.distinct_count,
        stats.__isset.max_value || stats.__isset.min_value,
        stats.__isset.null_count,
        stats.__isset.distinct_count,
        /*hasNaNCount=*/false,
        /*nanCount=*/0);
  }
  // Default behavior.
  return makeStatistics<DType>(
      descr,
      stats.min,
      stats.max,
      metadata.num_values - stats.null_count,
      stats.null_count,
      stats.distinct_count,
      stats.__isset.max || stats.__isset.min,
      stats.__isset.null_count,
      stats.__isset.distinct_count,
      /*hasNaNCount=*/false,
      /*nanCount=*/0);
}

std::shared_ptr<Statistics> makeColumnStats(
    const facebook::velox::parquet::thrift::ColumnMetaData& meta_data,
    const ColumnDescriptor* descr) {
  switch (static_cast<Type::type>(meta_data.type)) {
    case Type::kBoolean:
      return makeTypedColumnStats<BooleanType>(meta_data, descr);
    case Type::kInt32:
      return makeTypedColumnStats<Int32Type>(meta_data, descr);
    case Type::kInt64:
      return makeTypedColumnStats<Int64Type>(meta_data, descr);
    case Type::kInt96:
      return makeTypedColumnStats<Int96Type>(meta_data, descr);
    case Type::kDouble:
      return makeTypedColumnStats<DoubleType>(meta_data, descr);
    case Type::kFloat:
      return makeTypedColumnStats<FloatType>(meta_data, descr);
    case Type::kByteArray:
      return makeTypedColumnStats<ByteArrayType>(meta_data, descr);
    case Type::kFixedLenByteArray:
      return makeTypedColumnStats<FLBAType>(meta_data, descr);
    case Type::kUndefined:
      break;
  }
  throw ParquetException(
      "Can't decode page statistics for selected column type.");
}

// MetaData Accessor.

// ColumnCryptoMetaData.
class ColumnCryptoMetaData::ColumnCryptoMetaDataImpl {
 public:
  explicit ColumnCryptoMetaDataImpl(
      const facebook::velox::parquet::thrift::ColumnCryptoMetaData*
          cryptoMetadata)
      : cryptoMetadata_(cryptoMetadata) {}

  bool encryptedWithFooterKey() const {
    return cryptoMetadata_->__isset.ENCRYPTION_WITH_FOOTER_KEY;
  }
  bool encryptedWithColumnKey() const {
    return cryptoMetadata_->__isset.ENCRYPTION_WITH_COLUMN_KEY;
  }
  std::shared_ptr<schema::ColumnPath> pathInSchema() const {
    return std::make_shared<schema::ColumnPath>(
        cryptoMetadata_->ENCRYPTION_WITH_COLUMN_KEY.path_in_schema);
  }
  const std::string& keyMetadata() const {
    return cryptoMetadata_->ENCRYPTION_WITH_COLUMN_KEY.key_metadata;
  }

 private:
  const facebook::velox::parquet::thrift::ColumnCryptoMetaData* cryptoMetadata_;
};

std::unique_ptr<ColumnCryptoMetaData> ColumnCryptoMetaData::make(
    const uint8_t* metadata) {
  return std::unique_ptr<ColumnCryptoMetaData>(
      new ColumnCryptoMetaData(metadata));
}

ColumnCryptoMetaData::ColumnCryptoMetaData(const uint8_t* metadata)
    : impl_(
          std::make_unique<ColumnCryptoMetaDataImpl>(
              reinterpret_cast<const facebook::velox::parquet::thrift::
                                   ColumnCryptoMetaData*>(metadata))) {}

ColumnCryptoMetaData::~ColumnCryptoMetaData() = default;

std::shared_ptr<schema::ColumnPath> ColumnCryptoMetaData::pathInSchema() const {
  return impl_->pathInSchema();
}
bool ColumnCryptoMetaData::encryptedWithFooterKey() const {
  return impl_->encryptedWithFooterKey();
}
const std::string& ColumnCryptoMetaData::keyMetadata() const {
  return impl_->keyMetadata();
}

// ColumnChunk metadata.
class ColumnChunkMetaData::ColumnChunkMetaDataImpl {
 public:
  explicit ColumnChunkMetaDataImpl(
      const facebook::velox::parquet::thrift::ColumnChunk* column,
      const ColumnDescriptor* descr,
      int16_t rowGroupOrdinal,
      int16_t columnOrdinal,
      const ReaderProperties& properties,
      const ApplicationVersion* writerVersion,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor)
      : column_(column),
        descr_(descr),
        properties_(properties),
        writerVersion_(writerVersion) {
    columnMetadata_ = &column->meta_data;
    if (column->__isset.crypto_metadata) { // column metadata is encrypted
      facebook::velox::parquet::thrift::ColumnCryptoMetaData ccmd =
          column->crypto_metadata;

      if (ccmd.__isset.ENCRYPTION_WITH_COLUMN_KEY) {
        if (fileDecryptor != nullptr &&
            fileDecryptor->properties() != nullptr) {
          // Should decrypt metadata.
          std::shared_ptr<schema::ColumnPath> path =
              std::make_shared<schema::ColumnPath>(
                  ccmd.ENCRYPTION_WITH_COLUMN_KEY.path_in_schema);
          std::string key_metadata =
              ccmd.ENCRYPTION_WITH_COLUMN_KEY.key_metadata;

          std::string aadColumnMetadata = encryption::createModuleAad(
              fileDecryptor->fileAad(),
              encryption::kColumnMetaData,
              rowGroupOrdinal,
              columnOrdinal,
              static_cast<int16_t>(-1));
          auto Decryptor = fileDecryptor->getColumnMetaDecryptor(
              path->toDotString(), key_metadata, aadColumnMetadata);
          auto len =
              static_cast<uint32_t>(column->encrypted_column_metadata.size());
          ThriftDeserializer deserializer(properties_);
          deserializer.deserializeMessage(
              reinterpret_cast<const uint8_t*>(
                  column->encrypted_column_metadata.c_str()),
              &len,
              &decryptedMetadata_,
              Decryptor);
          columnMetadata_ = &decryptedMetadata_;
        } else {
          throw ParquetException(
              "Cannot decrypt ColumnMetadata."
              " FileDecryption is not setup correctly.");
        }
      }
    }
    for (const auto& encoding : columnMetadata_->encodings) {
      encodings_.push_back(loadenumSafe(&encoding));
    }
    for (const auto& encodingStats : columnMetadata_->encoding_stats) {
      encodingStats_.push_back(
          {loadenumSafe(&encodingStats.page_type),
           loadenumSafe(&encodingStats.encoding),
           encodingStats.count});
    }
    possibleStats_ = nullptr;
  }

  bool equals(const ColumnChunkMetaDataImpl& other) const {
    return *columnMetadata_ == *other.columnMetadata_;
  }

  // Column chunk.
  inline int64_t fileOffset() const {
    return column_->file_offset;
  }
  inline const std::string& filePath() const {
    return column_->file_path;
  }

  inline Type::type type() const {
    return loadenumSafe(&columnMetadata_->type);
  }

  inline int64_t numValues() const {
    return columnMetadata_->num_values;
  }

  std::shared_ptr<schema::ColumnPath> pathInSchema() {
    return std::make_shared<schema::ColumnPath>(
        columnMetadata_->path_in_schema);
  }

  // Check if statistics are set and are valid.
  // 1) Must be set in the metadata.
  // 2) Statistics must not be corrupted.
  inline bool isStatsSet() const {
    VELOX_DCHECK_NOT_NULL(writerVersion_);
    // If the column statistics don't exist or column sort order is unknown,
    // we cannot use the column stats.
    if (!columnMetadata_->__isset.statistics ||
        descr_->sortOrder() == SortOrder::kUnknown) {
      return false;
    }
    if (possibleStats_ == nullptr) {
      possibleStats_ = makeColumnStats(*columnMetadata_, descr_);
    }
    EncodedStatistics encodedStats = possibleStats_->encode();
    return writerVersion_->hasCorrectStatistics(
        type(), encodedStats, descr_->sortOrder());
  }

  inline std::shared_ptr<::facebook::velox::parquet::arrow::Statistics>
  statistics() const {
    return isStatsSet() ? possibleStats_ : nullptr;
  }

  inline Compression::type compression() const {
    return loadenumSafe(&columnMetadata_->codec);
  }

  const std::vector<Encoding::type>& encodings() const {
    return encodings_;
  }

  const std::vector<PageEncodingStats>& encodingStats() const {
    return encodingStats_;
  }

  inline std::optional<int64_t> bloomFilterOffset() const {
    if (columnMetadata_->__isset.bloom_filter_offset) {
      return columnMetadata_->bloom_filter_offset;
    }
    return std::nullopt;
  }

  inline bool hasDictionaryPage() const {
    return columnMetadata_->__isset.dictionary_page_offset;
  }

  inline int64_t dictionaryPageOffset() const {
    return columnMetadata_->dictionary_page_offset;
  }

  inline int64_t dataPageOffset() const {
    return columnMetadata_->data_page_offset;
  }

  inline bool hasIndexPage() const {
    return columnMetadata_->__isset.index_page_offset;
  }

  inline int64_t indexPageOffset() const {
    return columnMetadata_->index_page_offset;
  }

  inline int64_t totalCompressedSize() const {
    return columnMetadata_->total_compressed_size;
  }

  inline int64_t totalUncompressedSize() const {
    return columnMetadata_->total_uncompressed_size;
  }

  inline std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata() const {
    if (column_->__isset.crypto_metadata) {
      return ColumnCryptoMetaData::make(
          reinterpret_cast<const uint8_t*>(&column_->crypto_metadata));
    } else {
      return nullptr;
    }
  }

  std::optional<IndexLocation> getColumnIndexLocation() const {
    if (column_->__isset.column_index_offset &&
        column_->__isset.column_index_length) {
      return IndexLocation{
          column_->column_index_offset, column_->column_index_length};
    }
    return std::nullopt;
  }

  std::optional<IndexLocation> getOffsetIndexLocation() const {
    if (column_->__isset.offset_index_offset &&
        column_->__isset.offset_index_length) {
      return IndexLocation{
          column_->offset_index_offset, column_->offset_index_length};
    }
    return std::nullopt;
  }

  inline int32_t fieldId() const {
    return descr_->schemaNode()->fieldId();
  }

 private:
  mutable std::shared_ptr<::facebook::velox::parquet::arrow::Statistics>
      possibleStats_;
  std::vector<Encoding::type> encodings_;
  std::vector<PageEncodingStats> encodingStats_;
  const facebook::velox::parquet::thrift::ColumnChunk* column_;
  const facebook::velox::parquet::thrift::ColumnMetaData* columnMetadata_;
  facebook::velox::parquet::thrift::ColumnMetaData decryptedMetadata_;
  const ColumnDescriptor* descr_;
  const ReaderProperties properties_;
  const ApplicationVersion* writerVersion_;
};

std::unique_ptr<ColumnChunkMetaData> ColumnChunkMetaData::make(
    const void* metadata,
    const ColumnDescriptor* descr,
    const ReaderProperties& properties,
    const ApplicationVersion* writerVersion,
    int16_t rowGroupOrdinal,
    int16_t columnOrdinal,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::unique_ptr<ColumnChunkMetaData>(new ColumnChunkMetaData(
      metadata,
      descr,
      rowGroupOrdinal,
      columnOrdinal,
      properties,
      writerVersion,
      std::move(fileDecryptor)));
}

std::unique_ptr<ColumnChunkMetaData> ColumnChunkMetaData::make(
    const void* metadata,
    const ColumnDescriptor* descr,
    const ApplicationVersion* writerVersion,
    int16_t rowGroupOrdinal,
    int16_t columnOrdinal,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::unique_ptr<ColumnChunkMetaData>(new ColumnChunkMetaData(
      metadata,
      descr,
      rowGroupOrdinal,
      columnOrdinal,
      defaultReaderProperties(),
      writerVersion,
      std::move(fileDecryptor)));
}

ColumnChunkMetaData::ColumnChunkMetaData(
    const void* metadata,
    const ColumnDescriptor* descr,
    int16_t rowGroupOrdinal,
    int16_t columnOrdinal,
    const ReaderProperties& properties,
    const ApplicationVersion* writerVersion,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor)
    : impl_{new ColumnChunkMetaDataImpl(
          reinterpret_cast<
              const facebook::velox::parquet::thrift::ColumnChunk*>(metadata),
          descr,
          rowGroupOrdinal,
          columnOrdinal,
          properties,
          writerVersion,
          std::move(fileDecryptor))} {}

ColumnChunkMetaData::~ColumnChunkMetaData() = default;

// Column chunk.
int64_t ColumnChunkMetaData::fileOffset() const {
  return impl_->fileOffset();
}

const std::string& ColumnChunkMetaData::filePath() const {
  return impl_->filePath();
}

Type::type ColumnChunkMetaData::type() const {
  return impl_->type();
}

int64_t ColumnChunkMetaData::numValues() const {
  return impl_->numValues();
}

std::shared_ptr<schema::ColumnPath> ColumnChunkMetaData::pathInSchema() const {
  return impl_->pathInSchema();
}

std::shared_ptr<Statistics> ColumnChunkMetaData::statistics() const {
  return impl_->statistics();
}

bool ColumnChunkMetaData::isStatsSet() const {
  return impl_->isStatsSet();
}

std::optional<int64_t> ColumnChunkMetaData::bloomFilterOffset() const {
  return impl_->bloomFilterOffset();
}

bool ColumnChunkMetaData::hasDictionaryPage() const {
  return impl_->hasDictionaryPage();
}

int64_t ColumnChunkMetaData::dictionaryPageOffset() const {
  return impl_->dictionaryPageOffset();
}

int64_t ColumnChunkMetaData::dataPageOffset() const {
  return impl_->dataPageOffset();
}

bool ColumnChunkMetaData::hasIndexPage() const {
  return impl_->hasIndexPage();
}

int64_t ColumnChunkMetaData::indexPageOffset() const {
  return impl_->indexPageOffset();
}

Compression::type ColumnChunkMetaData::compression() const {
  return impl_->compression();
}

bool ColumnChunkMetaData::canDecompress() const {
  return util::Codec::isAvailable(compression());
}

const std::vector<Encoding::type>& ColumnChunkMetaData::encodings() const {
  return impl_->encodings();
}

const std::vector<PageEncodingStats>& ColumnChunkMetaData::encodingStats()
    const {
  return impl_->encodingStats();
}

int64_t ColumnChunkMetaData::totalUncompressedSize() const {
  return impl_->totalUncompressedSize();
}

int64_t ColumnChunkMetaData::totalCompressedSize() const {
  return impl_->totalCompressedSize();
}

int32_t ColumnChunkMetaData::fieldId() const {
  return impl_->fieldId();
}

std::unique_ptr<ColumnCryptoMetaData> ColumnChunkMetaData::cryptoMetadata()
    const {
  return impl_->cryptoMetadata();
}

std::optional<IndexLocation> ColumnChunkMetaData::getColumnIndexLocation()
    const {
  return impl_->getColumnIndexLocation();
}

std::optional<IndexLocation> ColumnChunkMetaData::getOffsetIndexLocation()
    const {
  return impl_->getOffsetIndexLocation();
}

bool ColumnChunkMetaData::equals(const ColumnChunkMetaData& other) const {
  return impl_->equals(*other.impl_);
}

// Row-group metadata.
class RowGroupMetaData::RowGroupMetaDataImpl {
 public:
  explicit RowGroupMetaDataImpl(
      const facebook::velox::parquet::thrift::RowGroup* rowGroup,
      const SchemaDescriptor* schema,
      const ReaderProperties& properties,
      const ApplicationVersion* writerVersion,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor)
      : rowGroup_(rowGroup),
        schema_(schema),
        properties_(properties),
        writerVersion_(writerVersion),
        fileDecryptor_(std::move(fileDecryptor)) {
    if (ARROW_PREDICT_FALSE(
            rowGroup_->columns.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()))) {
      throw ParquetException(
          "Row group had too many columns: ", rowGroup_->columns.size());
    }
  }

  bool equals(const RowGroupMetaDataImpl& other) const {
    return *rowGroup_ == *other.rowGroup_;
  }

  inline int numColumns() const {
    return static_cast<int>(rowGroup_->columns.size());
  }

  inline int64_t numRows() const {
    return rowGroup_->num_rows;
  }

  inline int64_t totalByteSize() const {
    return rowGroup_->total_byte_size;
  }

  inline int64_t totalCompressedSize() const {
    return rowGroup_->total_compressed_size;
  }

  inline int64_t fileOffset() const {
    return rowGroup_->file_offset;
  }

  inline const SchemaDescriptor* schema() const {
    return schema_;
  }

  std::unique_ptr<ColumnChunkMetaData> columnChunk(int i) {
    if (i >= 0 && i < numColumns()) {
      return ColumnChunkMetaData::make(
          &rowGroup_->columns[i],
          schema_->column(i),
          properties_,
          writerVersion_,
          rowGroup_->ordinal,
          i,
          fileDecryptor_);
    }
    throw ParquetException(
        "The file only has ",
        numColumns(),
        " columns, requested metadata for column: ",
        i);
  }

  std::vector<SortingColumn> sortingColumns() const {
    std::vector<SortingColumn> sortingColumns;
    if (!rowGroup_->__isset.sorting_columns) {
      return sortingColumns;
    }
    sortingColumns.resize(rowGroup_->sorting_columns.size());
    for (size_t i = 0; i < sortingColumns.size(); ++i) {
      sortingColumns[i] = fromThrift(rowGroup_->sorting_columns[i]);
    }
    return sortingColumns;
  }

 private:
  const facebook::velox::parquet::thrift::RowGroup* rowGroup_;
  const SchemaDescriptor* schema_;
  const ReaderProperties properties_;
  const ApplicationVersion* writerVersion_;
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;
};

std::unique_ptr<RowGroupMetaData> RowGroupMetaData::make(
    const void* metadata,
    const SchemaDescriptor* schema,
    const ApplicationVersion* writerVersion,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::unique_ptr<RowGroupMetaData>(new RowGroupMetaData(
      metadata,
      schema,
      defaultReaderProperties(),
      writerVersion,
      std::move(fileDecryptor)));
}

std::unique_ptr<RowGroupMetaData> RowGroupMetaData::make(
    const void* metadata,
    const SchemaDescriptor* schema,
    const ReaderProperties& properties,
    const ApplicationVersion* writerVersion,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::unique_ptr<RowGroupMetaData>(new RowGroupMetaData(
      metadata, schema, properties, writerVersion, std::move(fileDecryptor)));
}

RowGroupMetaData::RowGroupMetaData(
    const void* metadata,
    const SchemaDescriptor* schema,
    const ReaderProperties& properties,
    const ApplicationVersion* writerVersion,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor)
    : impl_{new RowGroupMetaDataImpl(
          reinterpret_cast<const facebook::velox::parquet::thrift::RowGroup*>(
              metadata),
          schema,
          properties,
          writerVersion,
          std::move(fileDecryptor))} {}

RowGroupMetaData::~RowGroupMetaData() = default;

bool RowGroupMetaData::equals(const RowGroupMetaData& other) const {
  return impl_->equals(*other.impl_);
}

int RowGroupMetaData::numColumns() const {
  return impl_->numColumns();
}

int64_t RowGroupMetaData::numRows() const {
  return impl_->numRows();
}

int64_t RowGroupMetaData::totalByteSize() const {
  return impl_->totalByteSize();
}

int64_t RowGroupMetaData::totalCompressedSize() const {
  return impl_->totalCompressedSize();
}

int64_t RowGroupMetaData::fileOffset() const {
  return impl_->fileOffset();
}

const SchemaDescriptor* RowGroupMetaData::schema() const {
  return impl_->schema();
}

std::unique_ptr<ColumnChunkMetaData> RowGroupMetaData::columnChunk(
    int i) const {
  return impl_->columnChunk(i);
}

bool RowGroupMetaData::canDecompress() const {
  int nColumns = numColumns();
  for (int i = 0; i < nColumns; i++) {
    if (!columnChunk(i)->canDecompress()) {
      return false;
    }
  }
  return true;
}

std::vector<SortingColumn> RowGroupMetaData::sortingColumns() const {
  return impl_->sortingColumns();
}

// File metadata.
class FileMetaData::FileMetaDataImpl {
 public:
  FileMetaDataImpl() = default;

  explicit FileMetaDataImpl(
      const void* metadata,
      uint32_t* metadataLen,
      ReaderProperties properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = nullptr)
      : properties_(std::move(properties)),
        fileDecryptor_(std::move(fileDecryptor)) {
    metadata_ =
        std::make_unique<facebook::velox::parquet::thrift::FileMetaData>();

    auto footerDecryptor = fileDecryptor_ != nullptr
        ? fileDecryptor_->getFooterDecryptor()
        : nullptr;

    ThriftDeserializer deserializer(properties_);
    deserializer.deserializeMessage(
        reinterpret_cast<const uint8_t*>(metadata),
        metadataLen,
        metadata_.get(),
        footerDecryptor);
    metadataLen_ = *metadataLen;

    if (metadata_->__isset.created_by) {
      writerVersion_ = ApplicationVersion(metadata_->created_by);
    } else {
      writerVersion_ = ApplicationVersion("unknown 0.0.0");
    }

    initSchema();
    initColumnOrders();
    initKeyValueMetadata();
  }

  bool verifySignature(const void* signature) {
    // Verify decryption properties are set.
    if (fileDecryptor_ == nullptr) {
      throw ParquetException(
          "Decryption not set properly. Cannot verify signature.");
    }
    // Serialize the footer.
    uint8_t* serializedData;
    uint32_t serializedLen = metadataLen_;
    ThriftSerializer serializer;
    serializer.serializeToBuffer(
        metadata_.get(), &serializedLen, &serializedData);

    // Encrypt with nonce.
    auto nonce =
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(signature));
    auto tag =
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(signature)) +
        encryption::kNonceLength;

    std::string key = fileDecryptor_->getFooterKey();
    std::string aad = encryption::createFooterAad(fileDecryptor_->fileAad());

    auto aesEncryptor = encryption::AesEncryptor::make(
        fileDecryptor_->algorithm(),
        static_cast<int>(key.size()),
        true,
        false /*write_length*/,
        nullptr);

    std::shared_ptr<Buffer> encryptedBuffer =
        std::static_pointer_cast<ResizableBuffer>(allocateBuffer(
            fileDecryptor_->pool(),
            aesEncryptor->ciphertextSizeDelta() + serializedLen));
    uint32_t encryptedLen = aesEncryptor->signedFooterEncrypt(
        serializedData,
        serializedLen,
        str2bytes(key),
        static_cast<int>(key.size()),
        str2bytes(aad),
        static_cast<int>(aad.size()),
        nonce,
        encryptedBuffer->mutable_data());
    // Delete AES encryptor object. It was created only to verify the footer.
    // Signature.
    aesEncryptor->wipeOut();
    delete aesEncryptor;
    return 0 ==
        memcmp(encryptedBuffer->data() + encryptedLen -
                   encryption::kGcmTagLength,
               tag,
               encryption::kGcmTagLength);
  }

  inline uint32_t size() const {
    return metadataLen_;
  }
  inline int numColumns() const {
    return schema_.numColumns();
  }
  inline int64_t numRows() const {
    return metadata_->num_rows;
  }
  inline int numRowGroups() const {
    return static_cast<int>(metadata_->row_groups.size());
  }
  inline int32_t version() const {
    return metadata_->version;
  }
  inline const std::string& createdBy() const {
    return metadata_->created_by;
  }
  inline int numSchemaElements() const {
    return static_cast<int>(metadata_->schema.size());
  }

  inline bool isEncryptionAlgorithmSet() const {
    return metadata_->__isset.encryption_algorithm;
  }
  inline EncryptionAlgorithm encryptionAlgorithm() {
    return fromThrift(metadata_->encryption_algorithm);
  }
  inline const std::string& footerSigningKeyMetadata() {
    return metadata_->footer_signing_key_metadata;
  }

  const ApplicationVersion& writerVersion() const {
    return writerVersion_;
  }

  void writeTo(
      ::arrow::io::OutputStream* dst,
      const std::shared_ptr<Encryptor>& encryptor) const {
    ThriftSerializer serializer;
    // Only in encrypted files with plaintext footers the.
    // Encryption_algorithm is set in footer.
    if (isEncryptionAlgorithmSet()) {
      uint8_t* serializedData;
      uint32_t serializedLen;
      serializer.serializeToBuffer(
          metadata_.get(), &serializedLen, &serializedData);

      // Encrypt the footer key.
      std::vector<uint8_t> encryptedData(
          encryptor->ciphertextSizeDelta() + serializedLen);
      unsigned encryptedLen = encryptor->encrypt(
          serializedData, serializedLen, encryptedData.data());

      // Write unencrypted footer.
      PARQUET_THROW_NOT_OK(dst->Write(serializedData, serializedLen));
      // Write signature (nonce and tag)
      PARQUET_THROW_NOT_OK(
          dst->Write(encryptedData.data() + 4, encryption::kNonceLength));
      PARQUET_THROW_NOT_OK(dst->Write(
          encryptedData.data() + encryptedLen - encryption::kGcmTagLength,
          encryption::kGcmTagLength));
    } else { // either plaintext file (when encryptor is null)
      // Or encrypted file with encrypted footer.
      serializer.serialize(metadata_.get(), dst, encryptor);
    }
  }

  std::unique_ptr<RowGroupMetaData> rowGroup(int i) {
    if (!(i >= 0 && i < numRowGroups())) {
      std::stringstream ss;
      ss << "The file only has " << numRowGroups()
         << " row groups, requested metadata for row group: " << i;
      throw ParquetException(ss.str());
    }
    return RowGroupMetaData::make(
        &metadata_->row_groups[i],
        &schema_,
        properties_,
        &writerVersion_,
        fileDecryptor_);
  }

  bool equals(const FileMetaDataImpl& other) const {
    return *metadata_ == *other.metadata_;
  }

  const SchemaDescriptor* schema() const {
    return &schema_;
  }

  const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata() const {
    return keyValueMetadata_;
  }

  void setFilePath(const std::string& path) {
    for (facebook::velox::parquet::thrift::RowGroup& rowGroup :
         metadata_->row_groups) {
      for (facebook::velox::parquet::thrift::ColumnChunk& chunk :
           rowGroup.columns) {
        chunk.__set_file_path(path);
      }
    }
  }

  facebook::velox::parquet::thrift::RowGroup& thriftRowGroup(int i) {
    if (!(i >= 0 && i < numRowGroups())) {
      std::stringstream ss;
      ss << "The file only has " << numRowGroups()
         << " row groups, requested metadata for row group: " << i;
      throw ParquetException(ss.str());
    }
    return metadata_->row_groups[i];
  }

  void appendRowGroups(const std::unique_ptr<FileMetaDataImpl>& other) {
    std::ostringstream diffOutput;
    if (!schema()->equals(*other->schema(), &diffOutput)) {
      auto msg = "AppendRowGroups requires equal schemas.\n" + diffOutput.str();
      throw ParquetException(msg);
    }

    // ARROW-13654: `other` may point to self, be careful not to enter an.
    // Infinite loop.
    const int n = other->numRowGroups();
    // ARROW-16613: do not use reserve() as that may suppress overallocation.
    // And incur O(n²) behavior on repeated calls to AppendRowGroups().
    // (See https://en.cppreference.com/w/cpp/container/vector/reserve.
    //  About inappropriate uses of reserve()).
    const auto start = metadata_->row_groups.size();
    metadata_->row_groups.resize(start + n);
    for (int i = 0; i < n; i++) {
      metadata_->row_groups[start + i] = other->thriftRowGroup(i);
      metadata_->num_rows += metadata_->row_groups[start + i].num_rows;
    }
  }

  std::shared_ptr<FileMetaData> subset(const std::vector<int>& rowGroups) {
    for (int i : rowGroups) {
      if (i < numRowGroups())
        continue;

      throw ParquetException(
          "The file only has ",
          numRowGroups(),
          " row groups, but requested a subset including row group: ",
          i);
    }

    std::shared_ptr<FileMetaData> out(new FileMetaData());
    out->impl_ = std::make_unique<FileMetaDataImpl>();
    out->impl_->metadata_ =
        std::make_unique<facebook::velox::parquet::thrift::FileMetaData>();

    auto metadata = out->impl_->metadata_.get();
    metadata->version = metadata_->version;
    metadata->schema = metadata_->schema;

    metadata->row_groups.resize(rowGroups.size());
    int i = 0;
    for (int selectedIndex : rowGroups) {
      metadata->num_rows += thriftRowGroup(selectedIndex).num_rows;
      metadata->row_groups[i++] = thriftRowGroup(selectedIndex);
    }

    metadata->key_value_metadata = metadata_->key_value_metadata;
    metadata->created_by = metadata_->created_by;
    metadata->column_orders = metadata_->column_orders;
    metadata->encryption_algorithm = metadata_->encryption_algorithm;
    metadata->footer_signing_key_metadata =
        metadata_->footer_signing_key_metadata;
    metadata->__isset = metadata_->__isset;

    out->impl_->schema_ = schema_;
    out->impl_->writerVersion_ = writerVersion_;
    out->impl_->keyValueMetadata_ = keyValueMetadata_;
    out->impl_->fileDecryptor_ = fileDecryptor_;

    return out;
  }

  void setFileDecryptor(std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
    fileDecryptor_ = fileDecryptor;
  }

  // Set NaN counts from the builder (called during Finish)
  // This stores total NaN counts per field ID across all row groups.
  void setNaNCounts(
      std::unordered_map<int32_t, std::pair<int64_t, bool>> nan_counts) {
    fieldNanCounts_ = std::move(nan_counts);
  }

  // Get total NaN count for a specific field ID across all row groups.
  std::pair<int64_t, bool> getNaNCount(int32_t fieldId) const {
    auto it = fieldNanCounts_.find(fieldId);
    if (it != fieldNanCounts_.end()) {
      return it->second;
    }
    return {0, false};
  }

 private:
  friend FileMetaDataBuilder;
  uint32_t metadataLen_ = 0;
  std::unique_ptr<facebook::velox::parquet::thrift::FileMetaData> metadata_;
  SchemaDescriptor schema_;
  ApplicationVersion writerVersion_;
  std::shared_ptr<const KeyValueMetadata> keyValueMetadata_;
  const ReaderProperties properties_;
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;
  // Total NaN counts per field ID across all row groups: field_id ->
  // (nan_count, has_nan_count).
  std::unordered_map<int32_t, std::pair<int64_t, bool>> fieldNanCounts_;

  void initSchema() {
    if (metadata_->schema.empty()) {
      throw ParquetException("Empty file schema (no root)");
    }
    schema_.init(
        schema::unflatten(
            &metadata_->schema[0], static_cast<int>(metadata_->schema.size())));
  }

  void initColumnOrders() {
    // Update ColumnOrder.
    std::vector<ColumnOrder> columnOrders;
    if (metadata_->__isset.column_orders) {
      columnOrders.reserve(metadata_->column_orders.size());
      for (auto columnOrder : metadata_->column_orders) {
        if (columnOrder.__isset.TYPE_ORDER) {
          columnOrders.push_back(ColumnOrder::typeDefined_);
        } else {
          columnOrders.push_back(ColumnOrder::undefined_);
        }
      }
    } else {
      columnOrders.resize(schema_.numColumns(), ColumnOrder::undefined_);
    }

    schema_.updateColumnOrders(columnOrders);
  }

  void initKeyValueMetadata() {
    std::shared_ptr<KeyValueMetadata> metadata = nullptr;
    if (metadata_->__isset.key_value_metadata) {
      metadata = std::make_shared<KeyValueMetadata>();
      for (const auto& it : metadata_->key_value_metadata) {
        metadata->Append(it.key, it.value);
      }
    }
    keyValueMetadata_ = std::move(metadata);
  }
};

std::shared_ptr<FileMetaData> FileMetaData::make(
    const void* metadata,
    uint32_t* metadataLen,
    const ReaderProperties& properties,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  // This FileMetaData ctor is private, not compatible with std::make_shared.
  return std::shared_ptr<FileMetaData>(new FileMetaData(
      metadata, metadataLen, properties, std::move(fileDecryptor)));
}

std::shared_ptr<FileMetaData> FileMetaData::make(
    const void* metadata,
    uint32_t* metadataLen,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::shared_ptr<FileMetaData>(new FileMetaData(
      metadata, metadataLen, defaultReaderProperties(), fileDecryptor));
}

FileMetaData::FileMetaData(
    const void* metadata,
    uint32_t* metadataLen,
    const ReaderProperties& properties,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor)
    : impl_(new FileMetaDataImpl(
          metadata,
          metadataLen,
          properties,
          fileDecryptor)) {}

FileMetaData::FileMetaData() : impl_(new FileMetaDataImpl()) {}

FileMetaData::~FileMetaData() = default;

bool FileMetaData::equals(const FileMetaData& other) const {
  return impl_->equals(*other.impl_);
}

std::unique_ptr<RowGroupMetaData> FileMetaData::rowGroup(int i) const {
  return impl_->rowGroup(i);
}

bool FileMetaData::verifySignature(const void* signature) {
  return impl_->verifySignature(signature);
}

uint32_t FileMetaData::size() const {
  return impl_->size();
}

int FileMetaData::numColumns() const {
  return impl_->numColumns();
}

int64_t FileMetaData::numRows() const {
  return impl_->numRows();
}

int FileMetaData::numRowGroups() const {
  return impl_->numRowGroups();
}

bool FileMetaData::canDecompress() const {
  int nRowGroups = numRowGroups();
  for (int i = 0; i < nRowGroups; i++) {
    if (!rowGroup(i)->canDecompress()) {
      return false;
    }
  }
  return true;
}

bool FileMetaData::isEncryptionAlgorithmSet() const {
  return impl_->isEncryptionAlgorithmSet();
}

EncryptionAlgorithm FileMetaData::encryptionAlgorithm() const {
  return impl_->encryptionAlgorithm();
}

const std::string& FileMetaData::footerSigningKeyMetadata() const {
  return impl_->footerSigningKeyMetadata();
}

void FileMetaData::setFileDecryptor(
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  impl_->setFileDecryptor(fileDecryptor);
}

ParquetVersion::type FileMetaData::version() const {
  switch (impl_->version()) {
    case 1:
      return ParquetVersion::PARQUET_1_0;
    case 2:
      return ParquetVersion::PARQUET_2_LATEST;
    default:
      // Improperly set version, assuming Parquet 1.0.
      break;
  }
  return ParquetVersion::PARQUET_1_0;
}

const ApplicationVersion& FileMetaData::writerVersion() const {
  return impl_->writerVersion();
}

const std::string& FileMetaData::createdBy() const {
  return impl_->createdBy();
}

int FileMetaData::numSchemaElements() const {
  return impl_->numSchemaElements();
}

const SchemaDescriptor* FileMetaData::schema() const {
  return impl_->schema();
}

const std::shared_ptr<const KeyValueMetadata>& FileMetaData::keyValueMetadata()
    const {
  return impl_->keyValueMetadata();
}

void FileMetaData::setFilePath(const std::string& path) {
  impl_->setFilePath(path);
}

void FileMetaData::appendRowGroups(const FileMetaData& other) {
  impl_->appendRowGroups(other.impl_);
}

std::shared_ptr<FileMetaData> FileMetaData::subset(
    const std::vector<int>& rowGroups) const {
  return impl_->subset(rowGroups);
}

std::pair<int64_t, bool> FileMetaData::getNaNCount(int32_t fieldId) const {
  return impl_->getNaNCount(fieldId);
}

void FileMetaData::writeTo(
    ::arrow::io::OutputStream* dst,
    const std::shared_ptr<Encryptor>& encryptor) const {
  return impl_->writeTo(dst, encryptor);
}

class FileCryptoMetaData::FileCryptoMetaDataImpl {
 public:
  FileCryptoMetaDataImpl() = default;

  explicit FileCryptoMetaDataImpl(
      const uint8_t* metadata,
      uint32_t* metadataLen,
      const ReaderProperties& properties) {
    ThriftDeserializer deserializer(properties);
    deserializer.deserializeMessage(metadata, metadataLen, &metadata_);
    metadataLen_ = *metadataLen;
  }

  EncryptionAlgorithm encryptionAlgorithm() const {
    return fromThrift(metadata_.encryption_algorithm);
  }

  const std::string& keyMetadata() const {
    return metadata_.key_metadata;
  }

  void writeTo(::arrow::io::OutputStream* dst) const {
    ThriftSerializer serializer;
    serializer.serialize(&metadata_, dst);
  }

 private:
  friend FileMetaDataBuilder;
  facebook::velox::parquet::thrift::FileCryptoMetaData metadata_;
  uint32_t metadataLen_;
};

EncryptionAlgorithm FileCryptoMetaData::encryptionAlgorithm() const {
  return impl_->encryptionAlgorithm();
}

const std::string& FileCryptoMetaData::keyMetadata() const {
  return impl_->keyMetadata();
}

std::shared_ptr<FileCryptoMetaData> FileCryptoMetaData::make(
    const uint8_t* serializedMetadata,
    uint32_t* metadataLen,
    const ReaderProperties& properties) {
  return std::shared_ptr<FileCryptoMetaData>(
      new FileCryptoMetaData(serializedMetadata, metadataLen, properties));
}

FileCryptoMetaData::FileCryptoMetaData(
    const uint8_t* serializedMetadata,
    uint32_t* metadataLen,
    const ReaderProperties& properties)
    : impl_(new FileCryptoMetaDataImpl(
          serializedMetadata,
          metadataLen,
          properties)) {}

FileCryptoMetaData::FileCryptoMetaData()
    : impl_(new FileCryptoMetaDataImpl()) {}

FileCryptoMetaData::~FileCryptoMetaData() = default;

void FileCryptoMetaData::writeTo(::arrow::io::OutputStream* dst) const {
  impl_->writeTo(dst);
}

std::string FileMetaData::serializeToString() const {
  // We need to pass in an initial size. Since it will automatically.
  // Increase the buffer size to hold the metadata, we just leave it 0.
  PARQUET_ASSIGN_OR_THROW(
      auto serializer, ::arrow::io::BufferOutputStream::Create(0));
  writeTo(serializer.get());
  PARQUET_ASSIGN_OR_THROW(auto metadataBuffer, serializer->Finish());
  return metadataBuffer->ToString();
}

ApplicationVersion::ApplicationVersion(
    std::string application,
    int major,
    int minor,
    int patch)
    : application_(std::move(application)),
      version{major, minor, patch, "", "", ""} {}

namespace {
// Parse the application version format and set parsed values to.
// ApplicationVersion.
//
// The application version format must be compatible parquet-mr's.
// One. See also:
//   * Https://github.com/apache/parquet-mr/blob/master/parquet-common/src/main/java/org/apache/parquet/VersionParser.java.
//   * Https://github.com/apache/parquet-mr/blob/master/parquet-common/src/main/java/org/apache/parquet/SemanticVersion.java.
//
// The application version format:
//   "${APPLICATION_NAME}".
//   "${APPLICATION_NAME} version ${VERSION}".
//   "${APPLICATION_NAME} version ${VERSION} (build ${BUILD_NAME})".
//
// Eg:
//   Parquet-cpp.
//   Parquet-cpp version 1.5.0ab-xyz5.5.0+cd.
//   parquet-cpp version 1.5.0ab-xyz5.5.0+cd (build abcd)
//
// The VERSION format:
//   "${MAJOR}".
//   "${MAJOR}.${MINOR}".
//   "${MAJOR}.${MINOR}.${PATCH}".
//   "${MAJOR}.${MINOR}.${PATCH}${UNKNOWN}".
//   "${MAJOR}.${MINOR}.${PATCH}${UNKNOWN}-${PRE_RELEASE}".
//   "${MAJOR}.${MINOR}.${PATCH}${UNKNOWN}-${PRE_RELEASE}+${BUILD_INFO}".
//   "${MAJOR}.${MINOR}.${PATCH}${UNKNOWN}+${BUILD_INFO}".
//   "${MAJOR}.${MINOR}.${PATCH}-${PRE_RELEASE}".
//   "${MAJOR}.${MINOR}.${PATCH}-${PRE_RELEASE}+${BUILD_INFO}".
//   "${MAJOR}.${MINOR}.${PATCH}+${BUILD_INFO}".
//
// Eg:
//   1.
//   1.5.
//   1.5.0.
//   1.5.0Ab.
//   1.5.0Ab-cdh5.5.0.
//   1.5.0Ab-cdh5.5.0+cd.
//   1.5.0Ab+cd.
//   1.5.0-Cdh5.5.0.
//   1.5.0-Cdh5.5.0+cd.
//   1.5.0+Cd.
class ApplicationVersionParser {
 public:
  ApplicationVersionParser(
      const std::string& createdBy,
      ApplicationVersion& ApplicationVersion)
      : createdBy_(createdBy),
        ApplicationVersion_(ApplicationVersion),
        spaces_(" \t\v\r\n\f"),
        digits_("0123456789") {}

  void parse() {
    ApplicationVersion_.application_ = "unknown";
    ApplicationVersion_.version = {0, 0, 0, "", "", ""};

    if (!parseApplicationName()) {
      return;
    }
    if (!parseVersion()) {
      return;
    }
    if (!parseBuildName()) {
      return;
    }
  }

 private:
  bool isSpace(const std::string& text, const size_t& offset) {
    auto target = ::std::string_view(text).substr(offset, 1);
    return target.find_first_of(spaces_) != ::std::string_view::npos;
  }

  void removePrecedingSpaces(
      const std::string& text,
      size_t& start,
      const size_t& end) {
    while (start < end && isSpace(text, start)) {
      ++start;
    }
  }

  void removeTrailingSpaces(
      const std::string& text,
      const size_t& start,
      size_t& end) {
    while (start < (end - 1) && (end - 1) < text.size() &&
           isSpace(text, end - 1)) {
      --end;
    }
  }

  bool parseApplicationName() {
    std::string versionMark(" version ");
    auto versionMarkPosition = createdBy_.find(versionMark);
    size_t applicationNameEnd;
    // No VERSION and BUILD_NAME.
    if (versionMarkPosition == std::string::npos) {
      versionStart_ = std::string::npos;
      applicationNameEnd = createdBy_.size();
    } else {
      versionStart_ = versionMarkPosition + versionMark.size();
      applicationNameEnd = versionMarkPosition;
    }

    size_t applicationNameStart = 0;
    removePrecedingSpaces(createdBy_, applicationNameStart, applicationNameEnd);
    removeTrailingSpaces(createdBy_, applicationNameStart, applicationNameEnd);
    ApplicationVersion_.application_ = createdBy_.substr(
        applicationNameStart, applicationNameEnd - applicationNameStart);

    return true;
  }

  bool parseVersion() {
    // No VERSION.
    if (versionStart_ == std::string::npos) {
      return false;
    }

    removePrecedingSpaces(createdBy_, versionStart_, createdBy_.size());
    versionEnd_ = createdBy_.find(" (", versionStart_);
    // No BUILD_NAME.
    if (versionEnd_ == std::string::npos) {
      versionEnd_ = createdBy_.size();
    }
    removeTrailingSpaces(createdBy_, versionStart_, versionEnd_);
    // No VERSION.
    if (versionStart_ == versionEnd_) {
      return false;
    }
    versionString_ =
        createdBy_.substr(versionStart_, versionEnd_ - versionStart_);

    if (!parseVersionMajor()) {
      return false;
    }
    if (!parseVersionMinor()) {
      return false;
    }
    if (!parseVersionPatch()) {
      return false;
    }
    if (!parseVersionUnknown()) {
      return false;
    }
    if (!parseVersionPreRelease()) {
      return false;
    }
    if (!parseVersionBuildInfo()) {
      return false;
    }

    return true;
  }

  bool parseVersionMajor() {
    size_t versionMajorStart = 0;
    auto versionMajorEnd = versionString_.find_first_not_of(digits_);
    // MAJOR only.
    if (versionMajorEnd == std::string::npos) {
      versionMajorEnd = versionString_.size();
      versionParsingPosition_ = versionMajorEnd;
    } else {
      // No ".".
      if (versionString_[versionMajorEnd] != '.') {
        return false;
      }
      // No MAJOR.
      if (versionMajorEnd == versionMajorStart) {
        return false;
      }
      versionParsingPosition_ = versionMajorEnd + 1; // +1 is for '.'.
    }
    auto versionMajorString = versionString_.substr(
        versionMajorStart, versionMajorEnd - versionMajorStart);
    ApplicationVersion_.version.major = atoi(versionMajorString.c_str());
    return true;
  }

  bool parseVersionMinor() {
    auto versionMinorStart = versionParsingPosition_;
    auto versionMinorEnd =
        versionString_.find_first_not_of(digits_, versionMinorStart);
    // MAJOR.MINOR only.
    if (versionMinorEnd == std::string::npos) {
      versionMinorEnd = versionString_.size();
      versionParsingPosition_ = versionMinorEnd;
    } else {
      // No ".".
      if (versionString_[versionMinorEnd] != '.') {
        return false;
      }
      // No MINOR.
      if (versionMinorEnd == versionMinorStart) {
        return false;
      }
      versionParsingPosition_ = versionMinorEnd + 1; // +1 is for '.'.
    }
    auto versionMinorString = versionString_.substr(
        versionMinorStart, versionMinorEnd - versionMinorStart);
    ApplicationVersion_.version.minor = atoi(versionMinorString.c_str());
    return true;
  }

  bool parseVersionPatch() {
    auto versionPatchStart = versionParsingPosition_;
    auto versionPatchEnd =
        versionString_.find_first_not_of(digits_, versionPatchStart);
    // No UNKNOWN, PRE_RELEASE and BUILD_INFO.
    if (versionPatchEnd == std::string::npos) {
      versionPatchEnd = versionString_.size();
    }
    // No PATCH.
    if (versionPatchEnd == versionPatchStart) {
      return false;
    }
    auto versionPatchString = versionString_.substr(
        versionPatchStart, versionPatchEnd - versionPatchStart);
    ApplicationVersion_.version.patch = atoi(versionPatchString.c_str());
    versionParsingPosition_ = versionPatchEnd;
    return true;
  }

  bool parseVersionUnknown() {
    // No UNKNOWN.
    if (versionParsingPosition_ == versionString_.size()) {
      return true;
    }
    auto versionUnknownStart = versionParsingPosition_;
    auto versionUnknownEnd =
        versionString_.find_first_of("-+", versionUnknownStart);
    // No PRE_RELEASE and BUILD_INFO.
    if (versionUnknownEnd == std::string::npos) {
      versionUnknownEnd = versionString_.size();
    }
    ApplicationVersion_.version.unknown = versionString_.substr(
        versionUnknownStart, versionUnknownEnd - versionUnknownStart);
    versionParsingPosition_ = versionUnknownEnd;
    return true;
  }

  bool parseVersionPreRelease() {
    // No PRE_RELEASE.
    if (versionParsingPosition_ == versionString_.size() ||
        versionString_[versionParsingPosition_] != '-') {
      return true;
    }

    auto versionPreReleaseStart = versionParsingPosition_ + 1; // +1 is for '-'.
    auto versionPreReleaseEnd =
        versionString_.find_first_of("+", versionPreReleaseStart);
    // No BUILD_INFO.
    if (versionPreReleaseEnd == std::string::npos) {
      versionPreReleaseEnd = versionString_.size();
    }
    ApplicationVersion_.version.preRelease = versionString_.substr(
        versionPreReleaseStart, versionPreReleaseEnd - versionPreReleaseStart);
    versionParsingPosition_ = versionPreReleaseEnd;
    return true;
  }

  bool parseVersionBuildInfo() {
    // No BUILD_INFO.
    if (versionParsingPosition_ == versionString_.size() ||
        versionString_[versionParsingPosition_] != '+') {
      return true;
    }

    auto versionBuildInfoStart = versionParsingPosition_ + 1; // +1 is for '+'.
    ApplicationVersion_.version.buildInfo =
        versionString_.substr(versionBuildInfoStart);
    return true;
  }

  bool parseBuildName() {
    std::string buildMark(" (build ");
    auto buildMarkPosition = createdBy_.find(buildMark, versionEnd_);
    // No BUILD_NAME.
    if (buildMarkPosition == std::string::npos) {
      return false;
    }
    auto buildNameStart = buildMarkPosition + buildMark.size();
    removePrecedingSpaces(createdBy_, buildNameStart, createdBy_.size());
    auto buildNameEnd = createdBy_.find_first_of(")", buildNameStart);
    // No end ")".
    if (buildNameEnd == std::string::npos) {
      return false;
    }
    removeTrailingSpaces(createdBy_, buildNameStart, buildNameEnd);
    ApplicationVersion_.build_ =
        createdBy_.substr(buildNameStart, buildNameEnd - buildNameStart);

    return true;
  }

  const std::string& createdBy_;
  ApplicationVersion& ApplicationVersion_;

  // For parsing.
  std::string spaces_;
  std::string digits_;
  size_t versionParsingPosition_;
  size_t versionStart_;
  size_t versionEnd_;
  std::string versionString_;
};
} // namespace

ApplicationVersion::ApplicationVersion(const std::string& createdBy) {
  ApplicationVersionParser parser(createdBy, *this);
  parser.parse();
}

bool ApplicationVersion::versionLt(
    const ApplicationVersion& otherVersion) const {
  if (application_ != otherVersion.application_)
    return false;

  if (version.major < otherVersion.version.major)
    return true;
  if (version.major > otherVersion.version.major)
    return false;
  VELOX_DCHECK_EQ(version.major, otherVersion.version.major);
  if (version.minor < otherVersion.version.minor)
    return true;
  if (version.minor > otherVersion.version.minor)
    return false;
  VELOX_DCHECK_EQ(version.minor, otherVersion.version.minor);
  return version.patch < otherVersion.version.patch;
}

bool ApplicationVersion::versionEq(
    const ApplicationVersion& otherVersion) const {
  return application_ == otherVersion.application_ &&
      version.major == otherVersion.version.major &&
      version.minor == otherVersion.version.minor &&
      version.patch == otherVersion.version.patch;
}

// Reference:
// Parquet-mr/parquet-column/src/main/java/org/apache/parquet/CorruptStatistics.java.
// PARQUET-686 has more discussion on statistics.
bool ApplicationVersion::hasCorrectStatistics(
    Type::type colType,
    EncodedStatistics& statistics,
    SortOrder::type sortOrder) const {
  // Parquet-cpp version 1.3.0 and parquet-mr 1.10.0 onwards stats are computed
  // correctly for all types.
  if ((application_ == "parquet-cpp" &&
       versionLt(PARQUET_CPP_FIXED_STATS_VERSION())) ||
      (application_ == "parquet-mr" &&
       versionLt(PARQUET_MR_FIXED_STATS_VERSION()))) {
    // Only SIGNED are valid unless max and min are the same.
    // (in which case the sort order does not matter)
    bool maxEqualsMin = statistics.hasMin && statistics.hasMax
        ? statistics.min() == statistics.max()
        : false;
    if (SortOrder::kSigned != sortOrder && !maxEqualsMin) {
      return false;
    }

    // Statistics of other types are OK.
    if (colType != Type::kFixedLenByteArray && colType != Type::kByteArray) {
      return true;
    }
  }
  // Created_by is not populated, which could have been caused by
  // Parquet-mr during the same time as PARQUET-251, see PARQUET-297.
  if (application_ == "unknown") {
    return true;
  }

  // Unknown sort order has incorrect stats.
  if (SortOrder::kUnknown == sortOrder) {
    return false;
  }

  // PARQUET-251.
  if (versionLt(PARQUET_251_FIXED_VERSION())) {
    return false;
  }

  return true;
}

// MetaData Builders.
// Row-group metadata.
class ColumnChunkMetaDataBuilder::ColumnChunkMetaDataBuilderImpl {
 public:
  explicit ColumnChunkMetaDataBuilderImpl(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column)
      : ownedColumnChunk_(new facebook::velox::parquet::thrift::ColumnChunk),
        properties_(std::move(props)),
        column_(column) {
    init(ownedColumnChunk_.get());
  }

  explicit ColumnChunkMetaDataBuilderImpl(
      std::shared_ptr<WriterProperties> props,
      const ColumnDescriptor* column,
      facebook::velox::parquet::thrift::ColumnChunk* columnChunk)
      : properties_(std::move(props)), column_(column) {
    init(columnChunk);
  }

  const void* Contents() const {
    return columnChunk_;
  }

  // Column chunk.
  void setFilePath(const std::string& val) {
    columnChunk_->__set_file_path(val);
  }

  // Column metadata.
  void setStatistics(const EncodedStatistics& val) {
    columnChunk_->meta_data.__set_statistics(toThrift(val));
    // Store NaN count separately since it's not written to the parquet file.
    if (val.hasNanCount) {
      nanCount_ = val.nanCount;
      hasNanCount_ = true;
    }
  }

  int64_t nanCount() const {
    return nanCount_;
  }

  bool hasNanCount() const {
    return hasNanCount_;
  }

  void finish(
      int64_t num_values,
      int64_t dictionary_page_offset,
      int64_t index_page_offset,
      int64_t data_page_offset,
      int64_t compressedSize,
      int64_t uncompressedSize,
      bool hasDictionary,
      bool dictionaryFallback,
      const std::map<Encoding::type, int32_t>& dictEncodingStats,
      const std::map<Encoding::type, int32_t>& dataEncodingStats,
      const std::shared_ptr<Encryptor>& encryptor) {
    if (dictionary_page_offset > 0) {
      columnChunk_->meta_data.__set_dictionary_page_offset(
          dictionary_page_offset);
      columnChunk_->__set_file_offset(dictionary_page_offset + compressedSize);
    } else {
      columnChunk_->__set_file_offset(data_page_offset + compressedSize);
    }
    columnChunk_->__isset.meta_data = true;
    columnChunk_->meta_data.__set_num_values(num_values);
    if (index_page_offset >= 0) {
      columnChunk_->meta_data.__set_index_page_offset(index_page_offset);
    }
    columnChunk_->meta_data.__set_data_page_offset(data_page_offset);
    columnChunk_->meta_data.__set_total_uncompressed_size(uncompressedSize);
    columnChunk_->meta_data.__set_total_compressed_size(compressedSize);

    std::vector<facebook::velox::parquet::thrift::Encoding::type>
        thriftEncodings;
    std::vector<facebook::velox::parquet::thrift::PageEncodingStats>
        thriftEncodingStats;
    auto addEncoding =
        [&thriftEncodings](
            facebook::velox::parquet::thrift::Encoding::type value) {
          auto it = std::find(
              thriftEncodings.cbegin(), thriftEncodings.cend(), value);
          if (it == thriftEncodings.cend()) {
            thriftEncodings.push_back(value);
          }
        };
    // Add dictionary page encoding stats.
    if (hasDictionary) {
      for (const auto& entry : dictEncodingStats) {
        facebook::velox::parquet::thrift::PageEncodingStats dictEncStat;
        dictEncStat.__set_page_type(
            facebook::velox::parquet::thrift::PageType::DICTIONARY_PAGE);
        // Dictionary encoding would be PLAIN_DICTIONARY in v1 and
        // PLAIN in v2.
        facebook::velox::parquet::thrift::Encoding::type dictEncoding =
            toThrift(entry.first);
        dictEncStat.__set_encoding(dictEncoding);
        dictEncStat.__set_count(entry.second);
        thriftEncodingStats.push_back(dictEncStat);
        addEncoding(dictEncoding);
      }
    }
    // Always add encoding for RL/DL.
    // BIT_PACKED is supported in `LevelEncoder`, but would only be used.
    // In benchmark and testing.
    // And for now, we always add RLE even if there are no levels at all,
    // while parquet-mr is more fine-grained.
    addEncoding(facebook::velox::parquet::thrift::Encoding::RLE);
    // Add data page encoding stats.
    for (const auto& entry : dataEncodingStats) {
      facebook::velox::parquet::thrift::PageEncodingStats dataEncStat;
      dataEncStat.__set_page_type(
          facebook::velox::parquet::thrift::PageType::DATA_PAGE);
      facebook::velox::parquet::thrift::Encoding::type dataEncoding =
          toThrift(entry.first);
      dataEncStat.__set_encoding(dataEncoding);
      dataEncStat.__set_count(entry.second);
      thriftEncodingStats.push_back(dataEncStat);
      addEncoding(dataEncoding);
    }
    columnChunk_->meta_data.__set_encodings(thriftEncodings);
    columnChunk_->meta_data.__set_encoding_stats(thriftEncodingStats);

    const auto& encryptMd =
        properties_->columnEncryptionProperties(column_->path()->toDotString());
    // Column is encrypted.
    if (encryptMd != nullptr && encryptMd->isEncrypted()) {
      columnChunk_->__isset.crypto_metadata = true;
      facebook::velox::parquet::thrift::ColumnCryptoMetaData ccmd;
      if (encryptMd->isEncryptedWithFooterKey()) {
        // Encrypted with footer key.
        ccmd.__isset.ENCRYPTION_WITH_FOOTER_KEY = true;
        ccmd.__set_ENCRYPTION_WITH_FOOTER_KEY(
            facebook::velox::parquet::thrift::EncryptionWithFooterKey());
      } else { // encrypted with column key
        facebook::velox::parquet::thrift::EncryptionWithColumnKey eck;
        eck.__set_key_metadata(encryptMd->keyMetadata());
        eck.__set_path_in_schema(column_->path()->toDotVector());
        ccmd.__isset.ENCRYPTION_WITH_COLUMN_KEY = true;
        ccmd.__set_ENCRYPTION_WITH_COLUMN_KEY(eck);
      }
      columnChunk_->__set_crypto_metadata(ccmd);

      bool encryptedFooter =
          properties_->fileEncryptionProperties()->encryptedFooter();
      bool encryptMetadata =
          !encryptedFooter || !encryptMd->isEncryptedWithFooterKey();
      if (encryptMetadata) {
        ThriftSerializer serializer;
        // Serialize and encrypt ColumnMetadata separately.
        // Thrift-serialize the ColumnMetaData structure,
        // encrypt it with the column key, and write to
        // encrypted_column_metadata.
        uint8_t* serializedData;
        uint32_t serializedLen;

        serializer.serializeToBuffer(
            &columnChunk_->meta_data, &serializedLen, &serializedData);

        std::vector<uint8_t> encryptedData(
            encryptor->ciphertextSizeDelta() + serializedLen);
        unsigned encryptedLen = encryptor->encrypt(
            serializedData, serializedLen, encryptedData.data());

        const char* temp = const_cast<const char*>(
            reinterpret_cast<char*>(encryptedData.data()));
        std::string encrypted_column_metadata(temp, encryptedLen);
        columnChunk_->__set_encrypted_column_metadata(
            encrypted_column_metadata);

        if (encryptedFooter) {
          columnChunk_->__isset.meta_data = false;
        } else {
          // Keep redacted metadata version for old readers.
          columnChunk_->__isset.meta_data = true;
          columnChunk_->meta_data.__isset.statistics = false;
          columnChunk_->meta_data.__isset.encoding_stats = false;
        }
      }
    }
  }

  void writeTo(::arrow::io::OutputStream* sink) {
    ThriftSerializer serializer;
    serializer.serialize(columnChunk_, sink);
  }

  const ColumnDescriptor* descr() const {
    return column_;
  }
  int64_t totalCompressedSize() const {
    return columnChunk_->meta_data.total_compressed_size;
  }

 private:
  void init(facebook::velox::parquet::thrift::ColumnChunk* columnChunk) {
    columnChunk_ = columnChunk;

    columnChunk_->meta_data.__set_type(toThrift(column_->physicalType()));
    columnChunk_->meta_data.__set_path_in_schema(
        column_->path()->toDotVector());
    columnChunk_->meta_data.__set_codec(
        toThrift(properties_->compression(column_->path())));
  }

  facebook::velox::parquet::thrift::ColumnChunk* columnChunk_;
  std::unique_ptr<facebook::velox::parquet::thrift::ColumnChunk>
      ownedColumnChunk_;
  const std::shared_ptr<WriterProperties> properties_;
  const ColumnDescriptor* column_;
  // NaN count is stored separately since it's not written to the parquet file.
  int64_t nanCount_ = 0;
  bool hasNanCount_ = false;
};

std::unique_ptr<ColumnChunkMetaDataBuilder> ColumnChunkMetaDataBuilder::make(
    std::shared_ptr<WriterProperties> props,
    const ColumnDescriptor* column,
    void* Contents) {
  return std::unique_ptr<ColumnChunkMetaDataBuilder>(
      new ColumnChunkMetaDataBuilder(std::move(props), column, Contents));
}

std::unique_ptr<ColumnChunkMetaDataBuilder> ColumnChunkMetaDataBuilder::make(
    std::shared_ptr<WriterProperties> props,
    const ColumnDescriptor* column) {
  return std::unique_ptr<ColumnChunkMetaDataBuilder>(
      new ColumnChunkMetaDataBuilder(std::move(props), column));
}

ColumnChunkMetaDataBuilder::ColumnChunkMetaDataBuilder(
    std::shared_ptr<WriterProperties> props,
    const ColumnDescriptor* column)
    : impl_{std::unique_ptr<ColumnChunkMetaDataBuilderImpl>(
          new ColumnChunkMetaDataBuilderImpl(std::move(props), column))} {}

ColumnChunkMetaDataBuilder::ColumnChunkMetaDataBuilder(
    std::shared_ptr<WriterProperties> props,
    const ColumnDescriptor* column,
    void* Contents)
    : impl_{std::unique_ptr<ColumnChunkMetaDataBuilderImpl>(
          new ColumnChunkMetaDataBuilderImpl(
              std::move(props),
              column,
              reinterpret_cast<facebook::velox::parquet::thrift::ColumnChunk*>(
                  Contents)))} {}

ColumnChunkMetaDataBuilder::~ColumnChunkMetaDataBuilder() = default;

const void* ColumnChunkMetaDataBuilder::Contents() const {
  return impl_->Contents();
}

void ColumnChunkMetaDataBuilder::setFilePath(const std::string& path) {
  impl_->setFilePath(path);
}

void ColumnChunkMetaDataBuilder::finish(
    int64_t num_values,
    int64_t dictionary_page_offset,
    int64_t index_page_offset,
    int64_t data_page_offset,
    int64_t compressedSize,
    int64_t uncompressedSize,
    bool hasDictionary,
    bool dictionaryFallback,
    const std::map<Encoding::type, int32_t>& dictEncodingStats,
    const std::map<Encoding::type, int32_t>& dataEncodingStats,
    const std::shared_ptr<Encryptor>& encryptor) {
  impl_->finish(
      num_values,
      dictionary_page_offset,
      index_page_offset,
      data_page_offset,
      compressedSize,
      uncompressedSize,
      hasDictionary,
      dictionaryFallback,
      dictEncodingStats,
      dataEncodingStats,
      encryptor);
}

void ColumnChunkMetaDataBuilder::writeTo(::arrow::io::OutputStream* sink) {
  impl_->writeTo(sink);
}

const ColumnDescriptor* ColumnChunkMetaDataBuilder::descr() const {
  return impl_->descr();
}

void ColumnChunkMetaDataBuilder::setStatistics(
    const EncodedStatistics& result) {
  impl_->setStatistics(result);
}

int64_t ColumnChunkMetaDataBuilder::totalCompressedSize() const {
  return impl_->totalCompressedSize();
}

int64_t ColumnChunkMetaDataBuilder::nanCount() const {
  return impl_->nanCount();
}

bool ColumnChunkMetaDataBuilder::hasNanCount() const {
  return impl_->hasNanCount();
}

class RowGroupMetaDataBuilder::RowGroupMetaDataBuilderImpl {
 public:
  explicit RowGroupMetaDataBuilderImpl(
      std::shared_ptr<WriterProperties> props,
      const SchemaDescriptor* schema,
      void* Contents)
      : properties_(std::move(props)), schema_(schema), nextColumn_(0) {
    rowGroup_ =
        reinterpret_cast<facebook::velox::parquet::thrift::RowGroup*>(Contents);
    initializeColumns(schema->numColumns());
  }

  ColumnChunkMetaDataBuilder* nextColumnChunk() {
    if (!(nextColumn_ < numColumns())) {
      std::stringstream ss;
      ss << "The schema only has " << numColumns()
         << " columns, requested metadata for column: " << nextColumn_;
      throw ParquetException(ss.str());
    }
    auto column = schema_->column(nextColumn_);
    auto columnBuilder = ColumnChunkMetaDataBuilder::make(
        properties_, column, &rowGroup_->columns[nextColumn_++]);
    auto columnBuilderPtr = columnBuilder.get();
    columnBuilders_.push_back(std::move(columnBuilder));
    return columnBuilderPtr;
  }

  int currentColumn() {
    return nextColumn_ - 1;
  }

  void finish(int64_t totalBytesWritten, int16_t rowGroupOrdinal) {
    if (!(nextColumn_ == schema_->numColumns())) {
      std::stringstream ss;
      ss << "Only " << nextColumn_ - 1 << " out of " << schema_->numColumns()
         << " columns are initialized";
      throw ParquetException(ss.str());
    }

    int64_t fileOffset = 0;
    int64_t total_compressed_size = 0;
    for (int i = 0; i < schema_->numColumns(); i++) {
      if (!(rowGroup_->columns[i].file_offset >= 0)) {
        std::stringstream ss;
        ss << "Column " << i << " is not complete.";
        throw ParquetException(ss.str());
      }
      if (i == 0) {
        const facebook::velox::parquet::thrift::ColumnMetaData& firstCol =
            rowGroup_->columns[0].meta_data;
        // As per spec, file_offset for the row group points to the first
        // dictionary or data page of the column.
        if (firstCol.__isset.dictionary_page_offset &&
            firstCol.dictionary_page_offset > 0) {
          fileOffset = firstCol.dictionary_page_offset;
        } else {
          fileOffset = firstCol.data_page_offset;
        }
      }
      // Sometimes column metadata is encrypted and not available to read,
      // so we must get total_compressed_size from column builder.
      total_compressed_size += columnBuilders_[i]->totalCompressedSize();
    }

    const auto& sortingColumns = properties_->sortingColumns();
    if (!sortingColumns.empty()) {
      std::vector<facebook::velox::parquet::thrift::SortingColumn>
          thriftSortingColumns(sortingColumns.size());
      for (size_t i = 0; i < sortingColumns.size(); ++i) {
        thriftSortingColumns[i] = toThrift(sortingColumns[i]);
      }
      rowGroup_->__set_sorting_columns(std::move(thriftSortingColumns));
    }

    rowGroup_->__set_file_offset(fileOffset);
    rowGroup_->__set_total_compressed_size(total_compressed_size);
    rowGroup_->__set_total_byte_size(totalBytesWritten);
    rowGroup_->__set_ordinal(rowGroupOrdinal);
  }

  void setNumRows(int64_t numRows) {
    rowGroup_->num_rows = numRows;
  }

  int numColumns() {
    return static_cast<int>(rowGroup_->columns.size());
  }

  int64_t numRows() {
    return rowGroup_->num_rows;
  }

  // Returns a map of field_id -> (nan_count, has_nan_count).
  std::unordered_map<int32_t, std::pair<int64_t, bool>> nanCounts() const {
    std::unordered_map<int32_t, std::pair<int64_t, bool>> result;
    for (const auto& builder : columnBuilders_) {
      int32_t field_id = builder->descr()->schemaNode()->fieldId();
      result[field_id] = {builder->nanCount(), builder->hasNanCount()};
    }
    return result;
  }

 private:
  void initializeColumns(int ncols) {
    rowGroup_->columns.resize(ncols);
  }

  facebook::velox::parquet::thrift::RowGroup* rowGroup_;
  const std::shared_ptr<WriterProperties> properties_;
  const SchemaDescriptor* schema_;
  std::vector<std::unique_ptr<ColumnChunkMetaDataBuilder>> columnBuilders_;
  int nextColumn_;
};

std::unique_ptr<RowGroupMetaDataBuilder> RowGroupMetaDataBuilder::make(
    std::shared_ptr<WriterProperties> props,
    const SchemaDescriptor* schema_,
    void* Contents) {
  return std::unique_ptr<RowGroupMetaDataBuilder>(
      new RowGroupMetaDataBuilder(std::move(props), schema_, Contents));
}

RowGroupMetaDataBuilder::RowGroupMetaDataBuilder(
    std::shared_ptr<WriterProperties> props,
    const SchemaDescriptor* schema_,
    void* Contents)
    : impl_{new RowGroupMetaDataBuilderImpl(
          std::move(props),
          schema_,
          Contents)} {}

RowGroupMetaDataBuilder::~RowGroupMetaDataBuilder() = default;

ColumnChunkMetaDataBuilder* RowGroupMetaDataBuilder::nextColumnChunk() {
  return impl_->nextColumnChunk();
}

int RowGroupMetaDataBuilder::currentColumn() const {
  return impl_->currentColumn();
}

int RowGroupMetaDataBuilder::numColumns() {
  return impl_->numColumns();
}

int64_t RowGroupMetaDataBuilder::numRows() {
  return impl_->numRows();
}

void RowGroupMetaDataBuilder::setNumRows(int64_t numRows) {
  impl_->setNumRows(numRows);
}

void RowGroupMetaDataBuilder::finish(
    int64_t totalBytesWritten,
    int16_t rowGroupOrdinal) {
  impl_->finish(totalBytesWritten, rowGroupOrdinal);
}

std::unordered_map<int32_t, std::pair<int64_t, bool>>
RowGroupMetaDataBuilder::nanCounts() const {
  return impl_->nanCounts();
}

// File metadata.
class FileMetaDataBuilder::FileMetaDataBuilderImpl {
 public:
  explicit FileMetaDataBuilderImpl(
      const SchemaDescriptor* schema,
      std::shared_ptr<WriterProperties> props,
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata)
      : metadata_(new facebook::velox::parquet::thrift::FileMetaData()),
        properties_(std::move(props)),
        schema_(schema),
        keyValueMetadata_(std::move(keyValueMetadata)) {
    if (properties_->fileEncryptionProperties() != nullptr &&
        properties_->fileEncryptionProperties()->encryptedFooter()) {
      crypto_metadata_.reset(
          new facebook::velox::parquet::thrift::FileCryptoMetaData());
    }
  }

  RowGroupMetaDataBuilder* appendRowGroup() {
    // Accumulate NaN counts from the previous row group before creating a new
    // one.
    accumulateNaNCountsFromCurrentRowGroup();
    rowGroups_.emplace_back();
    currentRowGroupBuilder_ =
        RowGroupMetaDataBuilder::make(properties_, schema_, &rowGroups_.back());
    return currentRowGroupBuilder_.get();
  }

  void setPageIndexLocation(const PageIndexLocation& location) {
    auto setIndexLocation = [this](
                                size_t rowGroupOrdinal,
                                const PageIndexLocation::FileIndexLocation&
                                    fileIndexLocation,
                                bool columnIndex) {
      auto& rowGroupMetadata = this->rowGroups_.at(rowGroupOrdinal);
      auto iter = fileIndexLocation.find(rowGroupOrdinal);
      if (iter != fileIndexLocation.cend()) {
        const auto& rowGroupIndexLocation = iter->second;
        for (size_t i = 0; i < rowGroupIndexLocation.size(); ++i) {
          if (i >= rowGroupMetadata.columns.size()) {
            throw ParquetException(
                "Cannot find metadata for column ordinal ", i);
          }
          auto& columnMetadata = rowGroupMetadata.columns.at(i);
          const auto& indexLocation = rowGroupIndexLocation.at(i);
          if (indexLocation.has_value()) {
            if (columnIndex) {
              columnMetadata.__set_column_index_offset(indexLocation->offset);
              columnMetadata.__set_column_index_length(indexLocation->length);
            } else {
              columnMetadata.__set_offset_index_offset(indexLocation->offset);
              columnMetadata.__set_offset_index_length(indexLocation->length);
            }
          }
        }
      }
    };

    for (size_t i = 0; i < rowGroups_.size(); ++i) {
      setIndexLocation(i, location.columnIndexLocation, true);
      setIndexLocation(i, location.offsetIndexLocation, false);
    }
  }

  std::unique_ptr<FileMetaData> finish(
      const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata) {
    // Accumulate NaN counts from the last row group.
    accumulateNaNCountsFromCurrentRowGroup();

    int64_t totalRows = 0;
    for (auto rowGroup : rowGroups_) {
      totalRows += rowGroup.num_rows;
    }
    metadata_->__set_num_rows(totalRows);
    metadata_->__set_row_groups(rowGroups_);

    if (keyValueMetadata_ || keyValueMetadata) {
      if (!keyValueMetadata_) {
        keyValueMetadata_ = keyValueMetadata;
      } else if (keyValueMetadata) {
        keyValueMetadata_ = keyValueMetadata_->Merge(*keyValueMetadata);
      }
      metadata_->key_value_metadata.clear();
      metadata_->key_value_metadata.reserve(keyValueMetadata_->size());
      for (int64_t i = 0; i < keyValueMetadata_->size(); ++i) {
        facebook::velox::parquet::thrift::KeyValue kvPair;
        kvPair.__set_key(keyValueMetadata_->key(i));
        kvPair.__set_value(keyValueMetadata_->value(i));
        metadata_->key_value_metadata.push_back(kvPair);
      }
      metadata_->__isset.key_value_metadata = true;
    }

    int32_t fileVersion = 0;
    switch (properties_->version()) {
      case ParquetVersion::PARQUET_1_0:
        fileVersion = 1;
        break;
      default:
        fileVersion = 2;
        break;
    }
    metadata_->__set_version(fileVersion);
    metadata_->__set_created_by(properties_->createdBy());

    // Users cannot set the `ColumnOrder` since we do not have user-defined
    // sort order in the spec yet. We always default to `TYPE_DEFINED_ORDER`.
    // We can expose it in the API once we have user-defined sort orders in the
    // Parquet format. TypeDefinedOrder implies choose SortOrder based on
    // convertedType/physicalType.
    facebook::velox::parquet::thrift::TypeDefinedOrder typeDefinedOrder;
    facebook::velox::parquet::thrift::ColumnOrder columnOrder;
    columnOrder.__set_TYPE_ORDER(typeDefinedOrder);
    columnOrder.__isset.TYPE_ORDER = true;
    metadata_->column_orders.resize(schema_->numColumns(), columnOrder);
    metadata_->__isset.column_orders = true;

    // If plaintext footer, set footer signing algorithm.
    auto fileEncryptionProperties = properties_->fileEncryptionProperties();
    if (fileEncryptionProperties &&
        !fileEncryptionProperties->encryptedFooter()) {
      EncryptionAlgorithm signingAlgorithm;
      EncryptionAlgorithm algo = fileEncryptionProperties->algorithm();
      signingAlgorithm.aad.aadFileUnique = algo.aad.aadFileUnique;
      signingAlgorithm.aad.supplyAadPrefix = algo.aad.supplyAadPrefix;
      if (!algo.aad.supplyAadPrefix) {
        signingAlgorithm.aad.aadPrefix = algo.aad.aadPrefix;
      }
      signingAlgorithm.algorithm = ParquetCipher::kAesGcmV1;

      metadata_->__set_encryption_algorithm(toThrift(signingAlgorithm));
      const std::string& footerSigningKeyMetadata =
          fileEncryptionProperties->footerKeyMetadata();
      if (footerSigningKeyMetadata.size() > 0) {
        metadata_->__set_footer_signing_key_metadata(footerSigningKeyMetadata);
      }
    }

    toParquet(
        static_cast<schema::GroupNode*>(schema_->schemaRoot().get()),
        &metadata_->schema);
    auto fileMetaData = std::unique_ptr<FileMetaData>(new FileMetaData());
    fileMetaData->impl_->metadata_ = std::move(metadata_);
    fileMetaData->impl_->initSchema();
    fileMetaData->impl_->initKeyValueMetadata();
    // Pass total NaN counts per field ID to FileMetaData.
    fileMetaData->impl_->setNaNCounts(std::move(fieldNanCounts_));
    return fileMetaData;
  }

  std::unique_ptr<FileCryptoMetaData> buildFileCryptoMetaData() {
    if (crypto_metadata_ == nullptr) {
      return nullptr;
    }

    auto fileEncryptionProperties = properties_->fileEncryptionProperties();

    crypto_metadata_->__set_encryption_algorithm(
        toThrift(fileEncryptionProperties->algorithm()));
    std::string keyMetadata = fileEncryptionProperties->footerKeyMetadata();

    if (!keyMetadata.empty()) {
      crypto_metadata_->__set_key_metadata(keyMetadata);
    }

    std::unique_ptr<FileCryptoMetaData> fileCryptoMetadata(
        new FileCryptoMetaData());
    fileCryptoMetadata->impl_->metadata_ = std::move(*crypto_metadata_);
    return fileCryptoMetadata;
  }

 protected:
  std::unique_ptr<facebook::velox::parquet::thrift::FileMetaData> metadata_;
  std::unique_ptr<facebook::velox::parquet::thrift::FileCryptoMetaData>
      crypto_metadata_;

 private:
  // Helper to accumulate NaN counts from the current row group builder.
  void accumulateNaNCountsFromCurrentRowGroup() {
    if (!currentRowGroupBuilder_) {
      return;
    }
    auto rgNaNCounts = currentRowGroupBuilder_->nanCounts();
    // Accumulate NaN counts from this row group (keyed by field ID).
    for (const auto& [fieldId, countPair] : rgNaNCounts) {
      const auto& [count, has_count] = countPair;
      if (has_count) {
        fieldNanCounts_[fieldId].first += count;
        fieldNanCounts_[fieldId].second = true;
      }
    }
  }

  const std::shared_ptr<WriterProperties> properties_;
  std::vector<facebook::velox::parquet::thrift::RowGroup> rowGroups_;

  std::unique_ptr<RowGroupMetaDataBuilder> currentRowGroupBuilder_;
  const SchemaDescriptor* schema_;
  std::shared_ptr<const KeyValueMetadata> keyValueMetadata_;
  // Total NaN counts per field ID across all row groups: field_id ->
  // (nan_count, has_nan_count).
  std::unordered_map<int32_t, std::pair<int64_t, bool>> fieldNanCounts_;
};

std::unique_ptr<FileMetaDataBuilder> FileMetaDataBuilder::make(
    const SchemaDescriptor* schema,
    std::shared_ptr<WriterProperties> props,
    std::shared_ptr<const KeyValueMetadata> keyValueMetadata) {
  return std::unique_ptr<FileMetaDataBuilder>(new FileMetaDataBuilder(
      schema, std::move(props), std::move(keyValueMetadata)));
}

std::unique_ptr<FileMetaDataBuilder> FileMetaDataBuilder::make(
    const SchemaDescriptor* schema,
    std::shared_ptr<WriterProperties> props) {
  return std::unique_ptr<FileMetaDataBuilder>(
      new FileMetaDataBuilder(schema, std::move(props)));
}

FileMetaDataBuilder::FileMetaDataBuilder(
    const SchemaDescriptor* schema,
    std::shared_ptr<WriterProperties> props,
    std::shared_ptr<const KeyValueMetadata> keyValueMetadata)
    : impl_{
          std::unique_ptr<FileMetaDataBuilderImpl>(new FileMetaDataBuilderImpl(
              schema,
              std::move(props),
              std::move(keyValueMetadata)))} {}

FileMetaDataBuilder::~FileMetaDataBuilder() = default;

RowGroupMetaDataBuilder* FileMetaDataBuilder::appendRowGroup() {
  return impl_->appendRowGroup();
}

void FileMetaDataBuilder::setPageIndexLocation(
    const PageIndexLocation& location) {
  impl_->setPageIndexLocation(location);
}

std::unique_ptr<FileMetaData> FileMetaDataBuilder::finish(
    const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata) {
  return impl_->finish(keyValueMetadata);
}

std::unique_ptr<FileCryptoMetaData> FileMetaDataBuilder::getCryptoMetaData() {
  return impl_->buildFileCryptoMetaData();
}

} // namespace facebook::velox::parquet::arrow
