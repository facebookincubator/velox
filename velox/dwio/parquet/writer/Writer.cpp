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

#include "velox/dwio/parquet/writer/Writer.h"

#include <algorithm>
#include <exception>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include "velox/common/Casts.h"
#include "velox/common/base/Pointers.h"
#include "velox/common/config/Config.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/parquet/common/ParquetConfig.h"
#include "velox/dwio/parquet/writer/arrow/ArrowSchema.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Writer.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::parquet {

using facebook::velox::parquet::arrow::ArrowWriterProperties;
using facebook::velox::parquet::arrow::Compression;
using facebook::velox::parquet::arrow::WriterProperties;
using facebook::velox::parquet::arrow::arrow::FileWriter;

// Utility for buffering Arrow output with a DataBuffer.
class ArrowDataBufferSink : public ::arrow::io::OutputStream {
 public:
  /// @param growRatio Growth factor used when invoking the reserve() method of
  /// DataSink, thereby helping to minimize frequent memcpy operations.
  ArrowDataBufferSink(
      std::unique_ptr<dwio::common::FileSink> sink,
      memory::MemoryPool& pool,
      double growRatio)
      : sink_(std::move(sink)), growRatio_(growRatio), buffer_(pool) {}

  ::arrow::Status Write(const std::shared_ptr<::arrow::Buffer>& data) override {
    auto requestCapacity = buffer_.size() + data->size();
    if (requestCapacity > buffer_.capacity()) {
      buffer_.reserve(growRatio_ * (requestCapacity));
    }
    buffer_.append(
        buffer_.size(),
        reinterpret_cast<const char*>(data->data()),
        data->size());
    return ::arrow::Status::OK();
  }

  ::arrow::Status Write(const void* data, int64_t nbytes) override {
    auto requestCapacity = buffer_.size() + nbytes;
    if (requestCapacity > buffer_.capacity()) {
      buffer_.reserve(growRatio_ * (requestCapacity));
    }
    buffer_.append(buffer_.size(), reinterpret_cast<const char*>(data), nbytes);
    return ::arrow::Status::OK();
  }

  ::arrow::Status Flush() override {
    bytesFlushed_ += buffer_.size();
    sink_->write(std::move(buffer_));
    return ::arrow::Status::OK();
  }

  ::arrow::Result<int64_t> Tell() const override {
    return bytesFlushed_ + buffer_.size();
  }

  int64_t bufferedBytes() const {
    return buffer_.size();
  }

  ::arrow::Status Close() override {
    ARROW_RETURN_NOT_OK(Flush());
    sink_->close();
    return ::arrow::Status::OK();
  }

  bool closed() const override {
    return sink_->isClosed();
  }

  void abort() {
    sink_.reset();
    buffer_.clear();
  }

 private:
  std::unique_ptr<dwio::common::FileSink> sink_;
  const double growRatio_;
  dwio::common::DataBuffer<char> buffer_;
  int64_t bytesFlushed_ = 0;
};

struct ArrowContext {
  std::unique_ptr<FileWriter> writer;
  std::shared_ptr<::arrow::Schema> schema;
  std::shared_ptr<WriterProperties> properties;
};

Compression::type getArrowParquetCompression(
    common::CompressionKind compression) {
  if (compression == common::CompressionKind_SNAPPY) {
    return Compression::SNAPPY;
  } else if (compression == common::CompressionKind_GZIP) {
    return Compression::GZIP;
  } else if (compression == common::CompressionKind_ZSTD) {
    return Compression::ZSTD;
  } else if (compression == common::CompressionKind_NONE) {
    return Compression::UNCOMPRESSED;
  } else if (compression == common::CompressionKind_LZ4) {
    return Compression::LZ4;
  } else if (compression == common::CompressionKind_LZ4_HADOOP) {
    return Compression::LZ4_HADOOP;
  } else {
    VELOX_FAIL("Unsupported compression {}", compression);
  }
}

namespace {

std::optional<TimestampPrecision> toTimestampPrecision(
    std::optional<uint8_t> unit) {
  if (!unit) {
    return std::nullopt;
  }
  VELOX_CHECK(
      *unit == 3 /*milli*/ || *unit == 6 /*micro*/ || *unit == 9 /*nano*/,
      "Invalid timestamp unit: {}",
      *unit);
  return static_cast<TimestampPrecision>(*unit);
}

// Converts a string to TimestampPrecision. Accepts numeric values "3" (milli),
// "6" (micro), or "9" (nano).
TimestampPrecision stringToTimestampPrecision(const std::string& value) {
  return toTimestampPrecision(std::optional{folly::to<uint8_t>(value)}).value();
}

const ParquetWriterOptions& getFormatOptions(
    const dwio::common::WriterOptions& options) {
  static const ParquetWriterOptions kDefaultOptions;
  if (!options.formatSpecificOptions) {
    return kDefaultOptions;
  }
  return *checkedPointerCast<ParquetWriterOptions>(
      options.formatSpecificOptions);
}

std::shared_ptr<WriterProperties> getArrowParquetWriterOptions(
    const dwio::common::WriterOptions& options,
    const ParquetWriterOptions& parquetOptions,
    const std::unique_ptr<DefaultFlushPolicy>& flushPolicy) {
  auto builder = WriterProperties::Builder();
  WriterProperties::Builder* properties = &builder;
  if (parquetOptions.enableDictionary.value_or(
          facebook::velox::parquet::arrow::DEFAULT_IS_DICTIONARY_ENABLED)) {
    properties = properties->enableDictionary();
    properties = properties->dictionaryPagesizeLimit(
        parquetOptions.dictionaryPageSizeLimit.value_or(
            facebook::velox::parquet::arrow::
                DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT));
  } else {
    properties = properties->disableDictionary();
  }
  properties = properties->compression(getArrowParquetCompression(
      options.compressionKind.value_or(common::CompressionKind_NONE)));
  for (const auto& columnCompressionValues :
       parquetOptions.columnCompressionsMap) {
    properties->compression(
        columnCompressionValues.first,
        getArrowParquetCompression(columnCompressionValues.second));
  }
  properties = properties->encoding(parquetOptions.encoding);
  properties = properties->dataPagesize(parquetOptions.dataPageSize.value_or(
      facebook::velox::parquet::arrow::kDefaultDataPageSize));
  properties = properties->writeBatchSize(parquetOptions.batchSize.value_or(
      facebook::velox::parquet::arrow::DEFAULT_WRITE_BATCH_SIZE));
  properties = properties->maxRowGroupLength(
      static_cast<int64_t>(flushPolicy->rowsInRowGroup()));
  properties = properties->codecOptions(parquetOptions.codecOptions);
  if (parquetOptions.enableStoreDecimalAsInteger.value_or(true)) {
    properties = properties->enableStoreDecimalAsInteger();
  } else {
    properties = properties->disableStoreDecimalAsInteger();
  }
  if (parquetOptions.useParquetDataPageV2.value_or(false)) {
    properties = properties->dataPageVersion(arrow::ParquetDataPageVersion::V2);
  } else {
    properties = properties->dataPageVersion(arrow::ParquetDataPageVersion::V1);
  }
  if (parquetOptions.createdBy.has_value()) {
    properties = properties->createdBy(parquetOptions.createdBy.value());
  }
  return properties->build();
}

void validateSchemaRecursive(
    const RowTypePtr& schema,
    const std::vector<ParquetFieldId>& parquetFieldIds) {
  // Check the schema's field names are not empty and unique.
  VELOX_USER_CHECK_NOT_NULL(schema, "Schema must not be empty.");
  const auto& fieldNames = schema->names();

  folly::F14FastSet<std::string> uniqueNames;
  for (const auto& name : fieldNames) {
    VELOX_USER_CHECK(!name.empty(), "Field name must not be empty.");
    auto result = uniqueNames.insert(name);
    VELOX_USER_CHECK(
        result.second,
        "File schema should not have duplicate columns: {}",
        name);
  }

  if (!parquetFieldIds.empty()) {
    VELOX_USER_CHECK_EQ(parquetFieldIds.size(), schema->size());
  }

  for (auto i = 0; i < schema->size(); ++i) {
    const auto& childType = schema->childAt(i);
    const auto& childFieldIds =
        parquetFieldIds.empty() ? parquetFieldIds : parquetFieldIds[i].children;

    if (childType->isRow()) {
      validateSchemaRecursive(
          std::dynamic_pointer_cast<const RowType>(childType), childFieldIds);
    } else if (childType->isArray()) {
      if (!parquetFieldIds.empty()) {
        VELOX_USER_CHECK_EQ(parquetFieldIds[i].children.size(), 1);
      }
      const auto& elementType = childType->asArray().elementType();
      if (elementType->isRow()) {
        validateSchemaRecursive(
            std::dynamic_pointer_cast<const RowType>(elementType),
            childFieldIds.empty() ? childFieldIds : childFieldIds[0].children);
      }
    } else if (childType->isMap()) {
      if (!parquetFieldIds.empty()) {
        VELOX_USER_CHECK_EQ(parquetFieldIds[i].children.size(), 2);
      }
      const auto& mapType = childType->asMap();
      if (mapType.keyType()->isRow()) {
        validateSchemaRecursive(
            std::dynamic_pointer_cast<const RowType>(mapType.keyType()),
            childFieldIds.empty() ? childFieldIds : childFieldIds[0].children);
      }
      if (mapType.valueType()->isRow()) {
        validateSchemaRecursive(
            std::dynamic_pointer_cast<const RowType>(mapType.valueType()),
            childFieldIds.empty() ? childFieldIds : childFieldIds[1].children);
      }
    }
  }
}

std::shared_ptr<::arrow::Field> updateFieldNameAndIdRecursive(
    const std::shared_ptr<::arrow::Field>& field,
    const Type& type,
    const ParquetFieldId* fieldId,
    const std::string& name = "") {
  auto newField = name.empty() ? field : field->WithName(name);

  if (fieldId) {
    newField =
        newField->WithMetadata(arrow::arrow::fieldIdMetadata(fieldId->fieldId));
  }

  if (type.isRow()) {
    auto& rowType = type.asRow();
    auto structType =
        std::dynamic_pointer_cast<::arrow::StructType>(newField->type());
    auto childrenSize = rowType.size();
    VELOX_CHECK(!fieldId || childrenSize <= fieldId->children.size());
    std::vector<std::shared_ptr<::arrow::Field>> newFields;
    newFields.reserve(childrenSize);
    for (auto i = 0; i < childrenSize; ++i) {
      const auto* childSetting = fieldId ? &fieldId->children.at(i) : nullptr;
      newFields.push_back(updateFieldNameAndIdRecursive(
          structType->fields()[i],
          *rowType.childAt(i),
          childSetting,
          rowType.nameOf(i)));
    }
    newField = newField->WithType(::arrow::struct_(newFields));
  } else if (type.isArray()) {
    auto listType =
        std::dynamic_pointer_cast<::arrow::BaseListType>(newField->type());
    auto elementType = type.asArray().elementType();
    auto elementField = listType->value_field();
    const auto* childSetting = fieldId ? &fieldId->children.at(0) : nullptr;
    auto updatedElementField =
        updateFieldNameAndIdRecursive(elementField, *elementType, childSetting);
    newField = newField->WithType(::arrow::list(updatedElementField));
  } else if (type.isMap()) {
    auto mapType = type.asMap();
    auto arrowMapType =
        std::dynamic_pointer_cast<::arrow::MapType>(newField->type());
    const auto* keySetting = fieldId ? &fieldId->children.at(0) : nullptr;
    const auto* valueSetting = fieldId ? &fieldId->children.at(1) : nullptr;
    auto newKeyField = updateFieldNameAndIdRecursive(
        arrowMapType->key_field(), *mapType.keyType(), keySetting);
    auto newValueField = updateFieldNameAndIdRecursive(
        arrowMapType->item_field(), *mapType.valueType(), valueSetting);
    newField = newField->WithType(
        std::make_shared<::arrow::MapType>(newKeyField, newValueField));
  }

  return newField;
}

std::optional<bool> isParquetV2(std::optional<std::string> version) {
  if (!version) {
    return std::nullopt;
  }
  if (version == "V1") {
    return false;
  }
  if (version == "V2") {
    return true;
  }
  VELOX_FAIL("Unsupported parquet datapage version {}", *version);
}

std::optional<int64_t> toParquetPageSize(std::optional<std::string> pageSize) {
  if (!pageSize) {
    return std::nullopt;
  }
  return config::toCapacity(*pageSize, config::CapacityUnit::BYTE);
}

std::optional<bool> toBoolConfigValue(
    std::optional<std::string> value,
    const char* optionName) {
  if (!value) {
    return std::nullopt;
  }
  try {
    return folly::to<bool>(*value);
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Invalid parquet writer {} option: {}", optionName, e.what());
  }
}

std::optional<int64_t> toParquetBatchSize(
    std::optional<std::string> batchSize) {
  if (!batchSize) {
    return std::nullopt;
  }
  try {
    return folly::to<int64_t>(*batchSize);
  } catch (const std::exception& e) {
    VELOX_USER_FAIL("Invalid parquet writer batch size: {}", e.what());
  }
}

} // namespace

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool,
    RowTypePtr schema)
    : pool_(std::move(pool)),
      generalPool_{pool_->addLeafChild(".general")},
      stream_(
          std::make_shared<ArrowDataBufferSink>(
              std::move(sink),
              *generalPool_,
              getFormatOptions(options).bufferGrowRatio)),
      arrowContext_(std::make_shared<ArrowContext>()),
      schema_(std::move(schema)) {
  const auto& parquetWriterOptions = getFormatOptions(options);
  validateSchemaRecursive(schema_, parquetWriterOptions.parquetFieldIds);
  auto parquetWriteTimestampUnit =
      parquetWriterOptions.parquetWriteTimestampUnit;
  if (const auto serdeTimestampUnitIt = options.serdeParameters.find(
          std::string(ParquetConfig::kWriterSerdeTimestampUnit));
      serdeTimestampUnitIt != options.serdeParameters.end()) {
    parquetWriteTimestampUnit =
        stringToTimestampPrecision(serdeTimestampUnitIt->second);
  }
  auto parquetWriteTimestampTimeZone =
      parquetWriterOptions.parquetWriteTimestampTimeZone;
  if (const auto serdeTimestampTimezoneIt = options.serdeParameters.find(
          std::string(ParquetConfig::kWriterSerdeTimestampTimezone));
      serdeTimestampTimezoneIt != options.serdeParameters.end()) {
    parquetWriteTimestampTimeZone = serdeTimestampTimezoneIt->second.empty()
        ? std::optional<std::string>{std::nullopt}
        : std::optional<std::string>{serdeTimestampTimezoneIt->second};
  } else if (
      !parquetWriteTimestampTimeZone.has_value() &&
      !options.sessionTimezoneName.empty()) {
    parquetWriteTimestampTimeZone = options.sessionTimezoneName;
  }

  if (options.flushPolicyFactory) {
    castUniquePointer(options.flushPolicyFactory(), flushPolicy_);
  } else if (options.maxTargetFileSizeBytes > 0) {
    auto bytesInRowGroup = static_cast<int64_t>(std::min<uint64_t>(
        DefaultFlushPolicy::kDefaultBytesInRowGroup,
        options.maxTargetFileSizeBytes));
    flushPolicy_ = std::make_unique<DefaultFlushPolicy>(
        DefaultFlushPolicy::kDefaultRowsInGroup, bytesInRowGroup);
  } else {
    flushPolicy_ = std::make_unique<DefaultFlushPolicy>();
  }
  options_.timestampUnit = static_cast<TimestampUnit>(
      parquetWriteTimestampUnit.value_or(TimestampPrecision::kNanoseconds));
  options_.timestampTimeZone = parquetWriteTimestampTimeZone;
  common::testutil::TestValue::adjust(
      "facebook::velox::parquet::Writer::Writer", &options_);
  arrowContext_->properties =
      getArrowParquetWriterOptions(options, parquetWriterOptions, flushPolicy_);
  setMemoryReclaimers();
  writeInt96AsTimestamp_ = parquetWriterOptions.writeInt96AsTimestamp;
  arrowMemoryPool_ = parquetWriterOptions.arrowMemoryPool;
  parquetFieldIds_ = parquetWriterOptions.parquetFieldIds;
}

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options,
    RowTypePtr schema)
    : Writer{
          std::move(sink),
          options,
          options.memoryPool->addAggregateChild(
              fmt::format(
                  "writer_node_{}",
                  folly::to<std::string>(folly::Random::rand64()))),
          std::move(schema)} {}

void Writer::flush() {
  if (arrowContext_->writer) {
    PARQUET_THROW_NOT_OK(arrowContext_->writer->finishRowGroup());
  }
  PARQUET_THROW_NOT_OK(stream_->Flush());
}

dwio::common::StripeProgress getStripeProgress(int64_t bufferedBytes) {
  // Arrow Parquet FileWriter will new row group based on the row number, so
  // we only check buffered bytes to flush row group here.
  return dwio::common::StripeProgress{.stripeSizeEstimate = bufferedBytes};
}

/**
 * This method assumes each input `ColumnarBatch` have same schema.
 */
void Writer::write(const VectorPtr& data) {
  VELOX_USER_CHECK(
      data->type()->equivalent(*schema_),
      "The file schema type should be equal with the input rowvector type.");

  VectorPtr exportData = data;
  if (needFlatten(exportData)) {
    BaseVector::flattenVector(exportData);
  }

  ArrowArray array;
  exportToArrow(exportData, array, generalPool_.get(), options_);

  if (!arrowContext_->schema) {
    // First batch: export and fix up the Arrow schema, then cache it.
    ArrowSchema schema;
    exportToArrow(exportData, schema, options_);

    auto arrowSchema = ::arrow::ImportSchema(&schema).ValueOrDie();
    common::testutil::TestValue::adjust(
        "facebook::velox::parquet::Writer::write", arrowSchema.get());
    std::vector<std::shared_ptr<::arrow::Field>> newFields;
    auto childSize = schema_->size();
    if (!parquetFieldIds_.empty()) {
      VELOX_CHECK(childSize == parquetFieldIds_.size());
    }
    for (auto i = 0; i < childSize; i++) {
      newFields.push_back(updateFieldNameAndIdRecursive(
          arrowSchema->fields()[i],
          *schema_->childAt(i),
          !parquetFieldIds_.empty() ? &parquetFieldIds_.at(i) : nullptr,
          schema_->nameOf(i)));
    }

    arrowContext_->schema = ::arrow::schema(newFields);
  }

  // Import the data array using the cached schema.
  PARQUET_ASSIGN_OR_THROW(
      auto recordBatch,
      ::arrow::ImportRecordBatch(&array, arrowContext_->schema));

  if (recordBatch->num_rows() == 0) {
    return;
  }

  if (!arrowContext_->writer) {
    ArrowWriterProperties::Builder builder;
    if (writeInt96AsTimestamp_) {
      builder.enableDeprecatedInt96Timestamps();
    }
    auto arrowProperties = builder.build();
    PARQUET_ASSIGN_OR_THROW(
        arrowContext_->writer,
        FileWriter::open(
            *recordBatch->schema(),
            arrowMemoryPool_.get(),
            stream_,
            arrowContext_->properties,
            arrowProperties));
  }

  PARQUET_THROW_NOT_OK(arrowContext_->writer->writeRecordBatch(*recordBatch));

  // Flush as soon as the current write pushes the staged row group past the
  // policy threshold. Otherwise callers that rotate files based on raw written
  // bytes won't observe the row group until the next write.
  if (flushPolicy_->shouldFlush(getStripeProgress(
          arrowContext_->writer->currentRowGroupTotalBytes()))) {
    flush();
  } else if (flushPolicy_->bytesInRowGroup() <= stream_->bufferedBytes()) {
    // Flush the sink separately so completed row groups don't keep accumulating
    // in the stream buffer when Arrow keeps starting new row groups before the
    // current one hits the byte threshold.
    PARQUET_THROW_NOT_OK(stream_->Flush());
  }
}

bool Writer::isCodecAvailable(common::CompressionKind compression) {
  return arrow::util::Codec::isAvailable(
      getArrowParquetCompression(compression));
}

void Writer::newRowGroup(int32_t numRows) {
  PARQUET_THROW_NOT_OK(arrowContext_->writer->newRowGroup(numRows));
}

std::unique_ptr<dwio::common::FileMetadata> Writer::close() {
  std::unique_ptr<ParquetFileMetadata> parquetFileMetadata;
  if (arrowContext_->writer) {
    PARQUET_THROW_NOT_OK(arrowContext_->writer->close());
    parquetFileMetadata = std::make_unique<ParquetFileMetadata>(
        arrowContext_->writer->metadata());
    arrowContext_->writer.reset();
  }

  PARQUET_THROW_NOT_OK(stream_->Close());

  return parquetFileMetadata;
}

void Writer::abort() {
  stream_->abort();
  arrowContext_.reset();
}

void Writer::setMemoryReclaimers() {
  VELOX_CHECK(
      !pool_->isLeaf(),
      "The root memory pool for parquet writer can't be leaf: {}",
      pool_->name());
  VELOX_CHECK_NULL(pool_->reclaimer());

  if ((pool_->parent() == nullptr) ||
      (pool_->parent()->reclaimer() == nullptr)) {
    return;
  }

  // TODO https://github.com/facebookincubator/velox/issues/8190
  pool_->setReclaimer(exec::MemoryReclaimer::create());
  generalPool_->setReclaimer(exec::MemoryReclaimer::create());
}

bool Writer::needFlatten(const VectorPtr& data) const {
  auto rowVector = std::dynamic_pointer_cast<RowVector>(data);
  VELOX_CHECK_NOT_NULL(
      rowVector, "Arrow export expects a RowVector as input data.");

  const auto& children = rowVector->children();
  return std::any_of(children.begin(), children.end(), [](const auto& child) {
    bool isNestedWrapped =
        (child->encoding() == VectorEncoding::Simple::DICTIONARY ||
         child->encoding() == VectorEncoding::Simple::CONSTANT) &&
        child->valueVector() && !child->wrappedVector()->isFlatEncoding();
    bool isComplex = !child->isScalar();
    return isNestedWrapped || isComplex;
  });
}

std::unique_ptr<dwio::common::Writer> ParquetWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const std::shared_ptr<dwio::common::WriterOptions>& options) {
  return std::make_unique<Writer>(
      std::move(sink), *options, asRowType(options->schema));
}

std::unique_ptr<dwio::common::WriterOptions>
ParquetWriterFactory::createWriterOptions() {
  return std::make_unique<dwio::common::WriterOptions>();
}

std::shared_ptr<dwio::common::FormatSpecificOptions>
ParquetWriterFactory::createFormatOptions(
    const config::ConfigBase& connectorConfig,
    const config::ConfigBase& session) const {
  auto parquetOptions = std::make_shared<ParquetWriterOptions>();
  parquetOptions->parquetWriteTimestampUnit = toTimestampPrecision(
      ParquetConfig::writerTimestampUnit(connectorConfig, session));
  parquetOptions->enableDictionary = toBoolConfigValue(
      ParquetConfig::writerEnableDictionary(connectorConfig, session),
      "enable dictionary");
  parquetOptions->enableStoreDecimalAsInteger = toBoolConfigValue(
      ParquetConfig::writerEnableStoreDecimalAsInteger(
          connectorConfig, session),
      "enable store decimal as integer");
  parquetOptions->dictionaryPageSizeLimit = toParquetPageSize(
      ParquetConfig::writerDictionaryPageSizeLimit(connectorConfig, session));
  parquetOptions->useParquetDataPageV2 = isParquetV2(
      ParquetConfig::writerDataPageVersion(connectorConfig, session));
  parquetOptions->dataPageSize = toParquetPageSize(
      ParquetConfig::writerPageSize(connectorConfig, session));
  parquetOptions->batchSize = toParquetBatchSize(
      ParquetConfig::writerBatchSize(connectorConfig, session));
  parquetOptions->createdBy = ParquetConfig::writerCreatedBy(connectorConfig);
  return parquetOptions;
}

} // namespace facebook::velox::parquet
