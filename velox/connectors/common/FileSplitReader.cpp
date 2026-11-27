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

#include "velox/connectors/common/FileSplitReader.h"

#include <fmt/format.h>
#include <folly/Conv.h>

#include "velox/common/caching/CacheTTLController.h"
#include "velox/connectors/common/BufferedInputBuilder.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/connectors/common/FileConnectorSplit.h"
#include "velox/connectors/common/PartitionValue.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"

namespace facebook::velox::connector {

VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch) {
  if (!value.has_value()) {
    return BaseVector::createNullConstant(type, 1, pool);
  }
  return BaseVector::createConstant(
      type,
      PartitionValue::fromString(
          value.value(),
          *type,
          isLocalTimestamp ? PartitionValue::TimestampMode::kLocalTime
                           : PartitionValue::TimestampMode::kUtc,
          isDaysSinceEpoch ? PartitionValue::DateMode::kDaysSinceEpoch
                           : PartitionValue::DateMode::kIsoString),
      1,
      pool);
}

std::unique_ptr<FileSplitReader> FileSplitReader::create(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const common::SubfieldFilters* subfieldFiltersForValidation) {
  return std::unique_ptr<FileSplitReader>(new FileSplitReader(
      fileSplit,
      tableHandle,
      partitionKeys,
      connectorQueryCtx,
      fileConfig,
      readerOutputType,
      dataIoStats,
      metadataIoStats,
      ioStats,
      fileHandleFactory,
      ioExecutor,
      scanSpec,
      subfieldFiltersForValidation));
}

FileSplitReader::FileSplitReader(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const common::SubfieldFilters* subfieldFiltersForValidation)
    : tableHandle_(tableHandle),
      partitionKeys_(partitionKeys),
      fileConfig_(fileConfig),
      dataIoStats_(dataIoStats),
      metadataIoStats_(metadataIoStats),
      ioStats_(ioStats),
      fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      pool_(connectorQueryCtx->memoryPool()),
      scanSpec_(scanSpec),
      subfieldFiltersForValidation_(subfieldFiltersForValidation),
      fileSplit_(fileSplit),
      connectorQueryCtx_(connectorQueryCtx),
      readerOutputType_(readerOutputType),
      baseReaderOpts_(connectorQueryCtx->memoryPool()),
      emptySplit_(false) {
  baseReaderOpts_.setDataIoStats(dataIoStats_);
  baseReaderOpts_.setMetadataIoStats(metadataIoStats_);
}

void FileSplitReader::initialize(
    std::shared_ptr<velox::random::RandomSkipTracker> randomSkip,
    const folly::F14FastSet<std::string>& remainingFilterColumns) {
  configureBaseReaderOptions();
  baseReaderOpts_.setRandomSkip(std::move(randomSkip));
  baseReaderOpts_.setScanSpec(scanSpec_);
  baseRowReaderOpts_.setRemainingFilterColumns(remainingFilterColumns);
}

void FileSplitReader::configureBaseReaderOptions() {
  configureReaderOptions(
      fileConfig_,
      connectorQueryCtx_,
      fileSplit_,
      tableHandle_->dataColumns(),
      /*tableParameters=*/{},
      /*serdeParameters=*/{},
      baseReaderOpts_);
}

void FileSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  createReader(fileReadOps);
  if (emptySplit_) {
    return;
  }
  auto rowType = getAdaptedRowType();

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  createRowReader(std::move(metadataFilter), std::move(rowType), std::nullopt);
}

uint64_t FileSplitReader::next(uint64_t size, VectorPtr& output) {
  if (!baseReaderOpts_.randomSkip()) {
    return baseRowReader_->next(size, output);
  }
  dwio::common::Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  return baseRowReader_->next(size, output, &mutation);
}

void FileSplitReader::resetFilterCaches() {
  if (baseRowReader_) {
    baseRowReader_->resetFilterCaches();
  }
}

bool FileSplitReader::emptySplit() const {
  return emptySplit_;
}

void FileSplitReader::resetSplit() {
  fileSplit_.reset();
}

int64_t FileSplitReader::nextRowNumber() const {
  if (!baseRowReader_) {
    return dwio::common::RowReader::kAtEnd;
  }
  return baseRowReader_->nextRowNumber();
}

int64_t FileSplitReader::estimatedRowSize() const {
  if (!baseRowReader_) {
    return DataSource::kUnknownRowSize;
  }

  const auto size = baseRowReader_->estimatedRowSize();
  return size.value_or(DataSource::kUnknownRowSize);
}

void FileSplitReader::updateRuntimeStats(
    dwio::common::RuntimeStats& stats) const {
  if (baseRowReader_) {
    baseRowReader_->updateRuntimeStats(stats);
  }
}

bool FileSplitReader::allPrefetchIssued() const {
  return baseRowReader_ && baseRowReader_->allPrefetchIssued();
}

void FileSplitReader::setConnectorQueryCtx(
    const ConnectorQueryCtx* connectorQueryCtx) {
  connectorQueryCtx_ = connectorQueryCtx;
}

std::string FileSplitReader::toString() const {
  std::string partitionKeys;
  std::for_each(
      partitionKeys_->begin(), partitionKeys_->end(), [&](const auto& column) {
        partitionKeys += " " + column.second->toString();
      });
  return fmt::format(
      "FileSplitReader: fileSplit_{} scanSpec_{} readerOutputType_{} partitionKeys_{} reader{} rowReader{}",
      fileSplit_->toString(),
      scanSpec_->toString(),
      readerOutputType_->toString(),
      partitionKeys,
      static_cast<const void*>(baseReader_.get()),
      static_cast<const void*>(baseRowReader_.get()));
}

FileProperties FileSplitReader::makeFileProperties(
    const folly::F14FastMap<std::string, std::string>& fileReadOps) const {
  // The split's own fileReadOps only add keys; the reader's keys win. Like the
  // reader's, they take effect only when the file is opened: the file handle
  // cache key ignores them, so a cached handle keeps the ops it was opened
  // with.
  auto fileProperties = fileSplit_->properties.value_or(FileProperties{});
  for (const auto& [key, value] : fileReadOps) {
    fileProperties.fileReadOps.insert_or_assign(key, value);
  }
  // The table's identity authorizes the file unless the reader's ops name
  // another table's, as for a file of a side table read in place of the base.
  if (!tableHandle_->dbName().empty() && !fileReadOps.contains(kDbNameKey)) {
    fileProperties.fileReadOps[kDbNameKey] = tableHandle_->dbName();
  }
  if (!tableHandle_->name().empty() && !fileReadOps.contains(kTableNameKey)) {
    fileProperties.fileReadOps[kTableNameKey] = tableHandle_->name();
  }
  // Per-operation counters are attributed to the data source that opened the
  // file, so they are only meaningful while file handles are not reused across
  // data sources. A cached handle would both outlive these statistics and
  // report one data source's reads against another's.
  fileProperties.ioStatistics =
      fileHandleFactory_->maxSize() == 0 ? dataIoStats_.get() : nullptr;
  return fileProperties;
}

void FileSplitReader::createReader(
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  VELOX_CHECK_NE(
      baseReaderOpts_.fileFormat(), dwio::common::FileFormat::UNKNOWN);

  FileHandleCachedPtr fileHandleCachePtr;
  FileHandleKey fileHandleKey{
      .filename = fileSplit_->readPath(),
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};

  auto fileProperties = makeFileProperties(fileReadOps);

  try {
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    if (e.errorCode() == error_code::kFileNotFound &&
        fileConfig_->ignoreMissingFiles(
            connectorQueryCtx_->sessionProperties())) {
      emptySplit_ = true;
      return;
    }
    throw;
  }

  // Here we keep adding new entries to CacheTTLController when new fileHandles
  // are generated, if CacheTTLController was created. Creator of
  // CacheTTLController needs to make sure a size control strategy was available
  // such as removing aged out entries.
  if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
    cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
  }
  if (auto* cache = connectorQueryCtx_->cache()) {
    baseReaderOpts_.setFileHandle(&(*fileHandleCachePtr));
    baseReaderOpts_.setCache(cache);
  }

  auto baseFileInput = BufferedInputBuilder::getInstance()->create(
      *fileHandleCachePtr,
      baseReaderOpts_,
      connectorQueryCtx_,
      dataIoStats_,
      ioStats_,
      ioExecutor_,
      fileReadOps);

  baseReader_ = dwio::common::getReaderFactory(baseReaderOpts_.fileFormat())
                    ->createReader(std::move(baseFileInput), baseReaderOpts_);
  if (!baseReader_) {
    emptySplit_ = true;
  }
}

RowTypePtr FileSplitReader::getAdaptedRowType() const {
  auto& fileType = baseReader_->rowType();
  auto columnTypes = adaptColumns(fileType, baseReaderOpts_.fileSchema());
  auto columnNames = fileType->names();
  return ROW(std::move(columnNames), std::move(columnTypes));
}

namespace {

bool applyPartitionFilter(
    const TypePtr& type,
    const std::string& partitionValue,
    bool isPartitionDateDaysSinceEpoch,
    const common::Filter* filter,
    bool asLocalTime) {
  const auto value = PartitionValue::fromString(
      partitionValue,
      *type,
      asLocalTime ? PartitionValue::TimestampMode::kLocalTime
                  : PartitionValue::TimestampMode::kUtc,
      isPartitionDateDaysSinceEpoch ? PartitionValue::DateMode::kDaysSinceEpoch
                                    : PartitionValue::DateMode::kIsoString);
  return applyFilter(*filter, value);
}

template <TypeKind kind>
bool testFilterTyped(const common::Filter* filter, const VectorPtr& vec) {
  using T = typename TypeTraits<kind>::NativeType;
  return applyFilter(*filter, vec->as<SimpleVector<T>>()->valueAt(0));
}

// Tests a filter against a non-null constant vector value (e.g., an
// initial-default column missing from the data file).
bool testFilterOnConstantVector(
    const common::Filter* filter,
    const VectorPtr& constantVec) {
  VELOX_CHECK_EQ(constantVec->size(), 1);
  VELOX_CHECK(!constantVec->isNullAt(0));
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
      testFilterTyped, constantVec->typeKind(), filter, constantVec);
}

} // namespace

// static
bool FileSplitReader::testPartitionAndConstantFilters(
    const common::ScanSpec& scanSpec,
    const RowType& fileType,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>&
        partitionKeyHandles,
    bool asLocalTime) {
  for (const auto& child : scanSpec.children()) {
    if (child->filter() == nullptr) {
      continue;
    }
    const auto& name = child->fieldName();
    const auto partitionIt = partitionKeys.find(name);
    // By design, the partition key columns for Iceberg tables are included in
    // the data files to facilitate partition transform and partition
    // evolution, so a column can be both a partition key and a file column.
    if (partitionIt == partitionKeys.end() && !child->isConstant() &&
        fileType.containsChild(name)) {
      continue;
    }
    if (partitionIt != partitionKeys.end() && partitionIt->second.has_value()) {
      const auto handleIt = partitionKeyHandles.find(name);
      VELOX_CHECK(handleIt != partitionKeyHandles.end());

      // A later column may still exclude the split, so keep going.
      if (!applyPartitionFilter(
              handleIt->second->dataType(),
              partitionIt->second.value(),
              handleIt->second->isPartitionDateValueDaysSinceEpoch(),
              child->filter(),
              asLocalTime)) {
        return false;
      }
      continue;
    }
    // The value does not come from the file. Filter on the constant if there
    // is one (an initial-default from schema evolution), otherwise treat the
    // column as NULL. Nothing downstream re-checks it: testFilterOnConstant()
    // accepts any non-null constant.
    bool filterMatchedConstant = false;
    if (child->isConstant()) {
      auto constantVec = child->constantValue();
      if (!constantVec->isNullAt(0)) {
        if (!testFilterOnConstantVector(child->filter(), constantVec)) {
          return false;
        }
        filterMatchedConstant = true;
      }
    }
    if (!filterMatchedConstant && child->filter()->isDeterministic() &&
        !child->filter()->testNull()) {
      return false;
    }
  }
  return true;
}

bool FileSplitReader::filterOnStats(
    dwio::common::RuntimeStats& runtimeStats) const {
  if (testPartitionAndConstantFilters(
          *scanSpec_,
          *baseReader_->rowType(),
          fileSplit_->partitionKeys,
          *partitionKeys_,
          fileConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties())) &&
      common::testFilters(*scanSpec_, *baseReader_)) {
    ++runtimeStats.processedSplits;
    return true;
  }
  VLOG(1) << "Skipping " << fileSplit_->filePath
          << " based on partition values, constants or file statistics";
  ++runtimeStats.skippedSplits;
  runtimeStats.skippedSplitBytes += fileSplit_->length;
  return false;
}

bool FileSplitReader::checkIfSplitIsEmpty(
    dwio::common::RuntimeStats& runtimeStats) {
  // emptySplit_ may already be set if the data file is not found. In this case
  // we don't need to test further.
  if (emptySplit_) {
    return true;
  }
  if (!baseReader_ || baseReader_->numberOfRows() == 0 ||
      !filterOnStats(runtimeStats)) {
    emptySplit_ = true;
  }
  return emptySplit_;
}

void FileSplitReader::createRowReader(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    RowTypePtr rowType,
    std::optional<bool> rowSizeTrackingEnabled) {
  VELOX_CHECK_NULL(baseRowReader_);
  configureBaseRowReaderOptions(std::move(metadataFilter), std::move(rowType));
  baseRowReaderOpts_.setNimblePreserveDictionaryEncoding(
      fileConfig_->nimblePreserveDictionaryEncoding(
          connectorQueryCtx_->sessionProperties()));
  baseRowReaderOpts_.setTrackRowSize(
      rowSizeTrackingEnabled.has_value()
          ? *rowSizeTrackingEnabled
          : connectorQueryCtx_->rowSizeTrackingMode() !=
              core::QueryConfig::RowSizeTrackingMode::kDisabled);
  baseRowReader_ = baseReader_->createRowReader(baseRowReaderOpts_);
}

void FileSplitReader::configureBaseRowReaderOptions(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    RowTypePtr rowType) {
  configureRowReaderOptions(
      fileConfig_,
      connectorQueryCtx_,
      fileSplit_,
      scanSpec_,
      std::move(metadataFilter),
      std::move(rowType),
      ioExecutor_,
      tableHandle_->tableParameters(),
      /*serdeParameters=*/{},
      baseRowReaderOpts_);
}

std::vector<TypePtr> FileSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const std::shared_ptr<const velox::RowType>& tableSchema) const {
  // Keep track of schema types for columns in file, used by ColumnSelector.
  std::vector<TypePtr> columnTypes = fileType->children();

  auto& childrenSpecs = scanSpec_->children();
  for (size_t i = 0; i < childrenSpecs.size(); ++i) {
    auto* childSpec = childrenSpecs[i].get();
    const std::string& fieldName = childSpec->fieldName();

    if (auto partitionIt = fileSplit_->partitionKeys.find(fieldName);
        partitionIt != fileSplit_->partitionKeys.end()) {
      setPartitionValue(childSpec, fieldName, partitionIt->second);
    } else if (
        childSpec->columnType() == common::ScanSpec::ColumnType::kRegular) {
      auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
      if (!fileTypeIdx.has_value()) {
        // Column is missing. Most likely due to schema evolution.
        VELOX_CHECK(tableSchema, "Unable to resolve column '{}'", fieldName);
        childSpec->setConstantValue(
            BaseVector::createNullConstant(
                tableSchema->findChild(fieldName),
                1,
                connectorQueryCtx_->memoryPool()));
      } else {
        // Column no longer missing, reset constant value set on the spec.
        childSpec->setConstantValue(nullptr);
        auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);
        if (outputTypeIdx.has_value()) {
          auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
          auto& columnType = columnTypes[*fileTypeIdx];
          if (childSpec->isFlatMapAsStruct()) {
            // Flat map column read as struct.  Leave the schema type as MAP.
            VELOX_CHECK(outputType->isRow() && columnType->isMap());
          } else {
            // We know the fieldName exists in the file, make the type at that
            // position match what we expect in the output.
            columnType = outputType;
          }
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

void FileSplitReader::setPartitionValue(
    common::ScanSpec* spec,
    const std::string& partitionKey,
    const std::optional<std::string>& value) const {
  const auto it = partitionKeys_->find(partitionKey);
  VELOX_CHECK(
      it != partitionKeys_->end(),
      "ColumnHandle is missing for partition key {}",
      partitionKey);
  const auto type = it->second->dataType();
  const auto constant = newConstantFromString(
      type,
      value,
      connectorQueryCtx_->memoryPool(),
      fileConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties()),
      it->second->isPartitionDateValueDaysSinceEpoch());
  spec->setConstantValue(constant);
}

namespace {

void validateColumnMappingMode(
    dwio::common::ColumnMappingMode mode,
    dwio::common::FileFormat fileFormat) {
  // kParquetFieldId is format-specific: it matches requested columns against
  // physical Parquet schema field_id metadata. Other readers don't have that
  // metadata, so reject it at split setup time instead of letting a later
  // reader path interpret it as a generic field-id or name/position mode.
  VELOX_USER_CHECK(
      mode != dwio::common::ColumnMappingMode::kParquetFieldId ||
          fileFormat == dwio::common::FileFormat::PARQUET,
      "Column mapping mode {} is not supported for file format {}",
      mode,
      fileFormat);
}

uint8_t parseDelimiter(const std::string& delim) {
  for (char const& ch : delim) {
    if (!std::isdigit(ch)) {
      return delim[0];
    }
  }
  return stoi(delim);
}

std::unique_ptr<dwio::common::SerDeOptions> parseSerdeParameters(
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::unordered_map<std::string, std::string>& tableParameters) {
  auto fieldIt = serdeParameters.find(dwio::common::SerDeOptions::kFieldDelim);
  if (fieldIt == serdeParameters.end()) {
    fieldIt = serdeParameters.find("serialization.format");
  }
  auto collectionIt =
      serdeParameters.find(dwio::common::SerDeOptions::kCollectionDelim);
  if (collectionIt == serdeParameters.end()) {
    // For collection delimiter, Hive 1.x, 2.x uses "colelction.delim", but
    // Hive 3.x uses "collection.delim".
    // See: https://issues.apache.org/jira/browse/HIVE-16922)
    collectionIt = serdeParameters.find("colelction.delim");
  }
  auto mapKeyIt =
      serdeParameters.find(dwio::common::SerDeOptions::kMapKeyDelim);

  auto escapeCharIt =
      serdeParameters.find(dwio::common::SerDeOptions::kEscapeChar);

  auto nullStringIt = tableParameters.find(
      dwio::common::TableParameter::kSerializationNullFormat);

  if (fieldIt == serdeParameters.end() &&
      collectionIt == serdeParameters.end() &&
      mapKeyIt == serdeParameters.end() &&
      escapeCharIt == serdeParameters.end() &&
      nullStringIt == tableParameters.end()) {
    return nullptr;
  }

  uint8_t fieldDelim = '\1';
  uint8_t collectionDelim = '\2';
  uint8_t mapKeyDelim = '\3';
  // Treat an empty value as absent so parseDelimiter never sees "".
  if (fieldIt != serdeParameters.end() && !fieldIt->second.empty()) {
    fieldDelim = parseDelimiter(fieldIt->second);
  }
  if (collectionIt != serdeParameters.end() && !collectionIt->second.empty()) {
    collectionDelim = parseDelimiter(collectionIt->second);
  }
  if (mapKeyIt != serdeParameters.end() && !mapKeyIt->second.empty()) {
    mapKeyDelim = parseDelimiter(mapKeyIt->second);
  }

  // If escape character is specified then we use it, unless it is empty - in
  // which case we default to '\\'.
  // If escape character is not specified (not in the map) we turn escaping off.
  // Logic is based on apache hive java code:
  // https://github.com/apache/hive/blob/3f6f940af3f60cc28834268e5d7f5612e3b13c30/serde/src/java/org/apache/hadoop/hive/serde2/lazy/LazySerDeParameters.java#L105-L108
  uint8_t escapeChar = '\\';
  const bool hasEscapeChar = (escapeCharIt != serdeParameters.end());
  if (hasEscapeChar) {
    if (!escapeCharIt->second.empty()) {
      // If delim is convertible to uint8_t then we use it as character code,
      // otherwise we use the 1st character of the string.
      escapeChar = folly::tryTo<uint8_t>(escapeCharIt->second)
                       .value_or(escapeCharIt->second[0]);
    }
  }

  auto serDeOptions = hasEscapeChar
      ? std::make_unique<dwio::common::SerDeOptions>(
            fieldDelim, collectionDelim, mapKeyDelim, escapeChar, true)
      : std::make_unique<dwio::common::SerDeOptions>(
            fieldDelim, collectionDelim, mapKeyDelim);
  if (nullStringIt != tableParameters.end()) {
    serDeOptions->nullString = nullStringIt->second;
  }
  return serDeOptions;
}

dwio::common::ColumnMappingMode sessionColumnMappingMode(
    const FileConfig& fileConfig,
    const config::ConfigBase* sessionProperties) {
  return fileConfig.useColumnNames(sessionProperties)
      ? dwio::common::ColumnMappingMode::kName
      : dwio::common::ColumnMappingMode::kPosition;
}

} // namespace

// static
void FileSplitReader::configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const RowTypePtr& fileSchema,
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::ReaderOptions& readerOptions) {
  auto sessionProperties = connectorQueryCtx->sessionProperties();
  VELOX_CHECK_NOT_NULL(sessionProperties, "Session properties are null");
  readerOptions.setLoadQuantum(fileConfig->loadQuantum(sessionProperties));
  readerOptions.setMaxCoalesceBytes(
      fileConfig->maxCoalescedBytes(sessionProperties));
  readerOptions.setMaxCoalesceDistance(
      fileConfig->maxCoalescedDistanceBytes(sessionProperties));
  readerOptions.setFileColumnNamesReadAsLowerCase(
      fileConfig->isFileColumnNamesReadAsLowerCase(sessionProperties));
  readerOptions.setAllowEmptyFile(true);
  const auto columnMappingMode = fileSplit->columnMappingMode.value_or(
      sessionColumnMappingMode(*fileConfig, sessionProperties));
  validateColumnMappingMode(columnMappingMode, fileSplit->fileFormat);
  readerOptions.setColumnMappingMode(columnMappingMode);
  readerOptions.setFileSchema(fileSchema);
  readerOptions.setFilePreloadThreshold(fileConfig->filePreloadThreshold());
  readerOptions.setPrefetchRowGroups(fileConfig->prefetchRowGroups());
  readerOptions.setCacheable(fileSplit->cacheable);
  const auto& sessionTzName = connectorQueryCtx->sessionTimezone();
  if (!sessionTzName.empty()) {
    const auto timezone = tz::locateZone(sessionTzName);
    readerOptions.setSessionTimezone(timezone);
  }
  readerOptions.setAdjustTimestampToTimezone(
      connectorQueryCtx->adjustTimestampToTimezone());
  // Prefer connector session property (FileConfig). Fall back to
  // ConnectorQueryCtx (threaded from QueryConfig) for backward compatibility
  // with callers that set it as a query config.
  if (sessionProperties->valueExists(
          FileConfig::kSelectiveNimbleReaderEnabledSession)) {
    readerOptions.setSelectiveNimbleReaderEnabled(
        fileConfig->selectiveNimbleReaderEnabled(sessionProperties));
  } else {
    readerOptions.setSelectiveNimbleReaderEnabled(
        connectorQueryCtx->selectiveNimbleReaderEnabled());
  }
  readerOptions.setNimbleDirectBufferedInputEnabled(
      fileConfig->nimbleDirectBufferedInputEnabled(sessionProperties));
  readerOptions.setNimbleLazyFileStats(
      fileConfig->nimbleLazyFileStats(sessionProperties));
  readerOptions.setCacheMetadata(
      fileConfig->cacheMetadata(sessionProperties) && fileSplit->cacheable);
  readerOptions.setPinMetadata(fileConfig->pinMetadata(sessionProperties));
  readerOptions.setCacheIndex(
      fileConfig->cacheIndex(sessionProperties) && fileSplit->cacheable);
  readerOptions.setPinIndex(fileConfig->pinIndex(sessionProperties));

  // Set footer speculative IO size based on file format.
  switch (fileSplit->fileFormat) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->orcFooterSpeculativeIoSize(sessionProperties));
      break;
    case dwio::common::FileFormat::PARQUET:
      break;
    case dwio::common::FileFormat::NIMBLE:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->nimbleFooterSpeculativeIoSize(sessionProperties));
      break;
    default:
      // Use ORC default for unknown formats.
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->orcFooterSpeculativeIoSize(sessionProperties));
      break;
  }

  if (readerOptions.fileFormat() != dwio::common::FileFormat::UNKNOWN) {
    VELOX_CHECK_EQ(
        readerOptions.fileFormat(),
        fileSplit->fileFormat,
        "All splits of a data source must have the same file format");
  } else {
    readerOptions.setFileFormat(fileSplit->fileFormat);
    // Serde options apply only on the first call, which sets the format.
    if (auto serDeOptions =
            parseSerdeParameters(serdeParameters, tableParameters)) {
      readerOptions.setSerDeOptions(*serDeOptions);
    }
  }

  if (!dwio::common::hasReaderFactory(fileSplit->fileFormat)) {
    readerOptions.setFormatSpecificOptions(nullptr);
    return;
  }

  auto formatScopedConfigs = fileConfig->formatScopedConfigs(
      fileSplit->fileFormat, *sessionProperties);
  readerOptions.setFormatSpecificOptions(
      dwio::common::getReaderFactory(fileSplit->fileFormat)
          ->createFormatOptions(
              formatScopedConfigs.connectorConfig,
              formatScopedConfigs.sessionProperties));
}

// static
void FileSplitReader::configureRowReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& requestedType,
    folly::Executor* const ioExecutor,
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::RowReaderOptions& rowReaderOptions) {
  auto skipRowsIt =
      tableParameters.find(dwio::common::TableParameter::kSkipHeaderLineCount);
  if (skipRowsIt != tableParameters.end()) {
    rowReaderOptions.setSkipRows(folly::to<uint64_t>(skipRowsIt->second));
  }
  rowReaderOptions.setSerdeParameters(serdeParameters);
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setIOExecutor(ioExecutor);
  rowReaderOptions.setMetadataFilter(std::move(metadataFilter));
  rowReaderOptions.setRequestedType(requestedType);
  rowReaderOptions.range(fileSplit->start, fileSplit->length);
  if (fileSplit->splitRowRange.has_value()) {
    VELOX_CHECK(
        fileSplit->fileFormat == dwio::common::FileFormat::NIMBLE ||
            fileSplit->fileFormat == dwio::common::FileFormat::DWRF ||
            fileSplit->fileFormat == dwio::common::FileFormat::ORC ||
            fileSplit->fileFormat == dwio::common::FileFormat::PARQUET ||
            fileSplit->fileFormat == dwio::common::FileFormat::TEXT,
        "Split row ranges are only supported for Nimble, DWRF, ORC, Parquet and text files: {}",
        fileSplit->toString());
    rowReaderOptions.setSplitRowRange(fileSplit->splitRowRange);
  }
  if (fileConfig && connectorQueryCtx) {
    const auto* sessionProperties = connectorQueryCtx->sessionProperties();
    rowReaderOptions.setTimestampPrecision(
        static_cast<TimestampPrecision>(
            fileConfig->readTimestampUnit(sessionProperties)));
    rowReaderOptions.setPreserveFlatMapsInMemory(
        fileConfig->preserveFlatMapsInMemory(sessionProperties));
    rowReaderOptions.setParallelUnitLoadCount(
        fileConfig->parallelUnitLoadCount(sessionProperties));
    rowReaderOptions.setIndexEnabled(
        fileConfig->indexEnabled(sessionProperties));
    rowReaderOptions.setLazyColumnIo(
        fileConfig->nimbleLazyColumnIo(sessionProperties));
    rowReaderOptions.setCollectColumnCpuMetrics(
        fileConfig->readerCollectColumnCpuMetrics(sessionProperties));
    rowReaderOptions.setNimbleFsstDecodeRepeatedPayloads(
        fileConfig->nimbleFsstDecodeRepeatedPayloads(sessionProperties));
    rowReaderOptions.setNimblePreserveDictionaryEncoding(
        fileConfig->nimblePreserveDictionaryEncoding(sessionProperties));
    rowReaderOptions.setNimbleDictionaryAwareReads(
        fileConfig->nimbleDictionaryAwareReads(sessionProperties));
    rowReaderOptions.setNimbleScanFastPath(
        fileConfig->nimbleScanFastPath(sessionProperties));
  }
}

} // namespace facebook::velox::connector
