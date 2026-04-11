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

#include "velox/connectors/hive/FileSplitReader.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/FileConnectorUtil.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive {
namespace {

template <TypeKind kind>
VectorPtr newConstantFromStringImpl(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch) {
  using T = typename TypeTraits<kind>::NativeType;
  if (!value.has_value()) {
    return std::make_shared<ConstantVector<T>>(pool, 1, true, type, T());
  }

  if (type->isDate()) {
    int32_t days = 0;
    // For Iceberg, the date partition values are already in daysSinceEpoch
    // form.
    if (isDaysSinceEpoch) {
      days = folly::to<int32_t>(value.value());
    } else {
      days = DATE()->toDays(value.value());
    }
    return std::make_shared<ConstantVector<int32_t>>(
        pool, 1, false, type, std::move(days));
  }

  if constexpr (std::is_same_v<T, StringView>) {
    return std::make_shared<ConstantVector<StringView>>(
        pool, 1, false, type, StringView(value.value()));
  } else {
    auto copy = velox::util::Converter<kind>::tryCast(value.value())
                    .thenOrThrow(folly::identity, [&](const Status& status) {
                      VELOX_USER_FAIL("{}", status.message());
                    });
    if constexpr (kind == TypeKind::TIMESTAMP) {
      if (isLocalTimestamp) {
        copy.toGMT(Timestamp::defaultTimezone());
      }
    }
    return std::make_shared<ConstantVector<T>>(
        pool, 1, false, type, std::move(copy));
  }
}
} // namespace

VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
      newConstantFromStringImpl,
      type->kind(),
      type,
      value,
      pool,
      isLocalTimestamp,
      isDaysSinceEpoch);
}

std::unique_ptr<FileSplitReader> FileSplitReader::create(
    const std::shared_ptr<const hive::FileConnectorSplit>& fileSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
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
      ioStatistics,
      ioStats,
      fileHandleFactory,
      ioExecutor,
      scanSpec,
      subfieldFiltersForValidation));
}

FileSplitReader::FileSplitReader(
    const std::shared_ptr<const hive::FileConnectorSplit>& fileSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const common::SubfieldFilters* subfieldFiltersForValidation)
    : fileSplit_(fileSplit),
      tableHandle_(tableHandle),
      partitionKeys_(partitionKeys),
      connectorQueryCtx_(connectorQueryCtx),
      fileConfig_(fileConfig),
      readerOutputType_(readerOutputType),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      pool_(connectorQueryCtx->memoryPool()),
      scanSpec_(scanSpec),
      subfieldFiltersForValidation_(subfieldFiltersForValidation),
      baseReaderOpts_(connectorQueryCtx->memoryPool()),
      emptySplit_(false) {}

void FileSplitReader::configureReaderOptions(
    std::shared_ptr<velox::random::RandomSkipTracker> randomSkip) {
  configureBaseReaderOptions();
  baseReaderOpts_.setRandomSkip(std::move(randomSkip));
  baseReaderOpts_.setScanSpec(scanSpec_);
  baseReaderOpts_.setFileFormat(fileSplit_->fileFormat);
}

void FileSplitReader::configureBaseReaderOptions() {
  hive::configureReaderOptions(
      fileConfig_,
      connectorQueryCtx_,
      tableHandle_,
      fileSplit_,
      baseReaderOpts_);
}

void FileSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
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

int64_t FileSplitReader::estimatedRowSize() const {
  if (!baseRowReader_) {
    return DataSource::kUnknownRowSize;
  }

  const auto size = baseRowReader_->estimatedRowSize();
  return size.value_or(DataSource::kUnknownRowSize);
}

void FileSplitReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& stats) const {
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

void FileSplitReader::createReader(
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  VELOX_CHECK_NE(
      baseReaderOpts_.fileFormat(), dwio::common::FileFormat::UNKNOWN);

  FileHandleCachedPtr fileHandleCachePtr;
  FileHandleKey fileHandleKey{
      .filename = fileSplit_->filePath,
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};

  auto fileProperties = fileSplit_->properties.value_or(FileProperties{});
  fileProperties.fileReadOps = fileReadOps;
  if (!tableHandle_->dbName().empty()) {
    fileProperties.fileReadOps[kDbNameKey] = tableHandle_->dbName();
  }
  if (!tableHandle_->name().empty()) {
    fileProperties.fileReadOps[kTableNameKey] = tableHandle_->name();
  }

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
  auto baseFileInput = BufferedInputBuilder::getInstance()->create(
      *fileHandleCachePtr,
      baseReaderOpts_,
      connectorQueryCtx_,
      ioStatistics_,
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

bool FileSplitReader::filterOnStats(
    dwio::common::RuntimeStatistics& runtimeStats) const {
  if (testFilters(
          scanSpec_.get(),
          baseReader_.get(),
          fileSplit_->filePath,
          fileSplit_->partitionKeys,
          *partitionKeys_,
          fileConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties()))) {
    ++runtimeStats.processedSplits;
    return true;
  }
  ++runtimeStats.skippedSplits;
  runtimeStats.skippedSplitBytes += fileSplit_->length;
  return false;
}

bool FileSplitReader::checkIfSplitIsEmpty(
    dwio::common::RuntimeStatistics& runtimeStats) {
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
  baseRowReaderOpts_.setTrackRowSize(
      rowSizeTrackingEnabled.has_value()
          ? *rowSizeTrackingEnabled
          : connectorQueryCtx_->rowSizeTrackingMode() !=
              core::QueryConfig::RowSizeTrackingMode::DISABLED);
  baseRowReader_ = baseReader_->createRowReader(baseRowReaderOpts_);
}

void FileSplitReader::configureBaseRowReaderOptions(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    RowTypePtr rowType) {
  hive::configureRowReaderOptions(
      tableHandle_->tableParameters(),
      scanSpec_,
      std::move(metadataFilter),
      std::move(rowType),
      fileSplit_,
      fileConfig_,
      connectorQueryCtx_->sessionProperties(),
      ioExecutor_,
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
  auto it = partitionKeys_->find(partitionKey);
  VELOX_CHECK(
      it != partitionKeys_->end(),
      "ColumnHandle is missing for partition key {}",
      partitionKey);
  auto type = it->second->dataType();
  auto constant = newConstantFromString(
      type,
      value,
      connectorQueryCtx_->memoryPool(),
      fileConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties()),
      it->second->isPartitionDateValueDaysSinceEpoch());
  spec->setConstantValue(constant);
}

} // namespace facebook::velox::connector::hive
