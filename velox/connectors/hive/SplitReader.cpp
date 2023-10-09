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

#include "velox/connectors/hive/SplitReader.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"

#include <folly/Conv.h>
#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

namespace facebook::velox::connector::hive {

namespace {

bool applyPartitionFilter(
    TypeKind kind,
    const std::string& partitionValue,
    common::Filter* filter) {
  switch (kind) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT: {
      return applyFilter(*filter, folly::to<int64_t>(partitionValue));
    }
    case TypeKind::REAL:
    case TypeKind::DOUBLE: {
      return applyFilter(*filter, folly::to<double>(partitionValue));
    }
    case TypeKind::BOOLEAN: {
      return applyFilter(*filter, folly::to<bool>(partitionValue));
    }
    case TypeKind::VARCHAR: {
      return applyFilter(*filter, partitionValue);
    }
    default:
      VELOX_FAIL("Bad type {} for partition value: {}", kind, partitionValue);
      break;
  }
}

bool testFilters(
    common::ScanSpec* scanSpec,
    dwio::common::Reader* reader,
    const std::string& filePath,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKey,
    std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeysHandle) {
  auto totalRows = reader->numberOfRows();
  const auto& fileTypeWithId = reader->typeWithId();
  const auto& rowType = reader->rowType();
  for (const auto& child : scanSpec->children()) {
    if (child->filter()) {
      const auto& name = child->fieldName();
      if (!rowType->containsChild(name)) {
        // If missing column is partition key.
        auto iter = partitionKey.find(name);
        if (iter != partitionKey.end() && iter->second.has_value()) {
          return applyPartitionFilter(
              (*partitionKeysHandle)[name]->dataType()->kind(),
              iter->second.value(),
              child->filter());
        }
        // Column is missing. Most likely due to schema evolution.
        if (child->filter()->isDeterministic() &&
            !child->filter()->testNull()) {
          return false;
        }
      } else {
        const auto& typeWithId = fileTypeWithId->childByName(name);
        auto columnStats = reader->columnStatistics(typeWithId->id());
        if (columnStats != nullptr &&
            !testFilter(
                child->filter(),
                columnStats.get(),
                totalRows.value(),
                typeWithId->type())) {
          VLOG(1) << "Skipping " << filePath
                  << " based on stats and filter for column "
                  << child->fieldName();
          return false;
        }
      }
    }
  }

  return true;
}

template <TypeKind ToKind>
velox::variant convertFromString(const std::optional<std::string>& value) {
  if (value.has_value()) {
    if constexpr (ToKind == TypeKind::VARCHAR) {
      return velox::variant(value.value());
    }
    if constexpr (ToKind == TypeKind::VARBINARY) {
      return velox::variant::binary((value.value()));
    }
    auto result = velox::util::Converter<ToKind>::cast(value.value());

    return velox::variant(result);
  }
  return velox::variant(ToKind);
}

inline uint8_t parseDelimiter(const std::string& delim) {
  for (char const& ch : delim) {
    if (!std::isdigit(ch)) {
      return delim[0];
    }
  }
  return stoi(delim);
}

} // namespace

std::unique_ptr<SplitReader> SplitReader::create(
    std::shared_ptr<velox::connector::hive::HiveConnectorSplit> hiveSplit,
    std::shared_ptr<HiveTableHandle> hiveTableHandle,
    std::shared_ptr<common::ScanSpec> scanSpec,
    const RowTypePtr readerOutputType,
    std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeys,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    std::shared_ptr<io::IoStatistics> ioStats) {
  return std::make_unique<SplitReader>(
      hiveSplit,
      hiveTableHandle,
      scanSpec,
      readerOutputType,
      partitionKeys,
      fileHandleFactory,
      executor,
      connectorQueryCtx,
      ioStats);
}

SplitReader::SplitReader(
    std::shared_ptr<velox::connector::hive::HiveConnectorSplit> hiveSplit,
    std::shared_ptr<HiveTableHandle> hiveTableHandle,
    std::shared_ptr<common::ScanSpec> scanSpec,
    const RowTypePtr readerOutputType,
    std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeys,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    std::shared_ptr<io::IoStatistics> ioStats)
    : hiveSplit_(hiveSplit),
      hiveTableHandle_(hiveTableHandle),
      scanSpec_(scanSpec),
      readerOutputType_(readerOutputType),
      partitionKeys_(partitionKeys),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStats_(ioStats),
      baseReaderOpts_(connectorQueryCtx->memoryPool()) {}

void SplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats) {
  configureReaderOptions();

  auto fileHandle = fileHandleFactory_->generate(hiveSplit_->filePath).second;
  auto baseFileInput = createBufferedInput(*fileHandle, baseReaderOpts_);

  baseReader_ = dwio::common::getReaderFactory(baseReaderOpts_.getFileFormat())
                    ->createReader(std::move(baseFileInput), baseReaderOpts_);

  // Note that this doesn't apply to Hudi tables.
  emptySplit_ = false;
  if (baseReader_->numberOfRows() == 0) {
    emptySplit_ = true;
    return;
  }

  // Check filters and see if the whole split can be skipped. Note that this
  // doesn't apply to Hudi tables.
  if (!testFilters(
          scanSpec_.get(),
          baseReader_.get(),
          hiveSplit_->filePath,
          hiveSplit_->partitionKeys,
          partitionKeys_)) {
    emptySplit_ = true;
    ++runtimeStats.skippedSplits;
    runtimeStats.skippedSplitBytes += hiveSplit_->length;
    return;
  }

  auto& fileType = baseReader_->rowType();
  auto columnTypes = adaptColumns(fileType, baseReaderOpts_.getFileSchema());

  auto skipRowsIt = hiveTableHandle_->tableParameters().find(
      dwio::common::TableParameter::kSkipHeaderLineCount);
  if (skipRowsIt != hiveTableHandle_->tableParameters().end()) {
    baseRowReaderOpts_.setSkipRows(folly::to<uint64_t>(skipRowsIt->second));
  }

  baseRowReaderOpts_.setScanSpec(scanSpec_);
  baseRowReaderOpts_.setMetadataFilter(metadataFilter);
  configureRowReaderOptions(
      baseRowReaderOpts_,
      ROW(std::vector<std::string>(fileType->names()), std::move(columnTypes)));
  // NOTE: we firstly reset the finished 'baseRowReader_' of previous split
  // before setting up for the next one to avoid doubling the peak memory usage.
  baseRowReader_.reset();
  baseRowReader_ = baseReader_->createRowReader(baseRowReaderOpts_);
}

std::vector<TypePtr> SplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const std::shared_ptr<const velox::RowType>& tableSchema) {
  // Keep track of schema types for columns in file, used by ColumnSelector.
  std::vector<TypePtr> columnTypes = fileType->children();

  auto& childrenSpecs = scanSpec_->children();
  for (size_t i = 0; i < childrenSpecs.size(); ++i) {
    auto* childSpec = childrenSpecs[i].get();
    const std::string& fieldName = childSpec->fieldName();

    auto iter = hiveSplit_->partitionKeys.find(fieldName);
    if (iter != hiveSplit_->partitionKeys.end()) {
      setPartitionValue(childSpec, fieldName, iter->second);
    } else if (fieldName == kPath) {
      setConstantValue(
          childSpec, VARCHAR(), velox::variant(hiveSplit_->filePath));
    } else if (fieldName == kBucket) {
      if (hiveSplit_->tableBucketNumber.has_value()) {
        setConstantValue(
            childSpec,
            INTEGER(),
            velox::variant(hiveSplit_->tableBucketNumber.value()));
      }
    } else {
      auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
      if (!fileTypeIdx.has_value()) {
        // Column is missing. Most likely due to schema evolution.
        VELOX_CHECK(tableSchema);
        setNullConstantValue(childSpec, tableSchema->findChild(fieldName));
      } else {
        // Column no longer missing, reset constant value set on the spec.
        childSpec->setConstantValue(nullptr);
        auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);
        if (outputTypeIdx.has_value()) {
          // We know the fieldName exists in the file, make the type at that
          // position match what we expect in the output.
          columnTypes[fileTypeIdx.value()] =
              readerOutputType_->childAt(*outputTypeIdx);
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

uint64_t SplitReader::next(int64_t size, VectorPtr& output) {
  return baseRowReader_->next(size, output);
}

void SplitReader::resetFilterCaches() {
  if (baseRowReader_) {
    baseRowReader_->resetFilterCaches();
  }
}

bool SplitReader::emptySplit() const {
  return emptySplit_;
}

void SplitReader::resetSplit() {
  hiveSplit_.reset();
}

int64_t SplitReader::estimatedRowSize() const {
  if (!baseRowReader_) {
    return DataSource::kUnknownRowSize;
  }

  auto size = baseRowReader_->estimatedRowSize();
  if (size.has_value()) {
    return size.value();
  }
  return DataSource::kUnknownRowSize;
}

void SplitReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& stats) const {
  if (baseRowReader_) {
    baseRowReader_->updateRuntimeStats(stats);
  }
}

bool SplitReader::allPrefetchIssued() const {
  return baseRowReader_ && baseRowReader_->allPrefetchIssued();
}

void SplitReader::setConstantValue(
    common::ScanSpec* spec,
    const TypePtr& type,
    const velox::variant& value) const {
  spec->setConstantValue(BaseVector::createConstant(
      type, value, 1, connectorQueryCtx_->memoryPool()));
}

void SplitReader::setNullConstantValue(
    common::ScanSpec* spec,
    const TypePtr& type) const {
  spec->setConstantValue(BaseVector::createNullConstant(
      type, 1, connectorQueryCtx_->memoryPool()));
}

void SplitReader::setPartitionValue(
    common::ScanSpec* spec,
    const std::string& partitionKey,
    const std::optional<std::string>& value) const {
  auto it = partitionKeys_->find(partitionKey);
  VELOX_CHECK(
      it != partitionKeys_->end(),
      "ColumnHandle is missing for partition key {}",
      partitionKey);
  auto constValue = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      convertFromString, it->second->dataType()->kind(), value);
  setConstantValue(spec, it->second->dataType(), constValue);
}

void SplitReader::configureReaderOptions() {
  baseReaderOpts_.setMaxCoalesceBytes(
      HiveConfig::maxCoalescedBytes(connectorQueryCtx_->config()));
  baseReaderOpts_.setMaxCoalesceDistance(
      HiveConfig::maxCoalescedDistanceBytes(connectorQueryCtx_->config()));
  baseReaderOpts_.setFileColumnNamesReadAsLowerCase(
      HiveConfig::isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx_->config()));
  baseReaderOpts_.setUseColumnNamesForColumnMapping(
      HiveConfig::isOrcUseColumnNames(connectorQueryCtx_->config()));
  baseReaderOpts_.setFileSchema(hiveTableHandle_->dataColumns());

  if (baseReaderOpts_.getFileFormat() != dwio::common::FileFormat::UNKNOWN) {
    VELOX_CHECK(
        baseReaderOpts_.getFileFormat() == hiveSplit_->fileFormat,
        "HiveDataSource received splits of different formats: {} and {}",
        dwio::common::toString(baseReaderOpts_.getFileFormat()),
        dwio::common::toString(hiveSplit_->fileFormat));
  } else {
    parseSerdeParameters(hiveSplit_->serdeParameters);
    baseReaderOpts_.setFileFormat(hiveSplit_->fileFormat);
  }
}

void SplitReader::configureRowReaderOptions(
    dwio::common::RowReaderOptions& options,
    const RowTypePtr& rowType) {
  std::vector<std::string> columnNames;
  for (auto& spec : scanSpec_->children()) {
    if (!spec->isConstant()) {
      columnNames.push_back(spec->fieldName());
    }
  }
  std::shared_ptr<dwio::common::ColumnSelector> cs;
  if (columnNames.empty()) {
    static const RowTypePtr kEmpty{ROW({}, {})};
    cs = std::make_shared<dwio::common::ColumnSelector>(kEmpty);
  } else {
    cs = std::make_shared<dwio::common::ColumnSelector>(rowType, columnNames);
  }
  options.select(cs).range(hiveSplit_->start, hiveSplit_->length);
}

void SplitReader::parseSerdeParameters(
    const std::unordered_map<std::string, std::string>& serdeParameters) {
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

  if (fieldIt == serdeParameters.end() &&
      collectionIt == serdeParameters.end() &&
      mapKeyIt == serdeParameters.end()) {
    return;
  }

  uint8_t fieldDelim = '\1';
  uint8_t collectionDelim = '\2';
  uint8_t mapKeyDelim = '\3';
  if (fieldIt != serdeParameters.end()) {
    fieldDelim = parseDelimiter(fieldIt->second);
  }
  if (collectionIt != serdeParameters.end()) {
    collectionDelim = parseDelimiter(collectionIt->second);
  }
  if (mapKeyIt != serdeParameters.end()) {
    mapKeyDelim = parseDelimiter(mapKeyIt->second);
  }
  dwio::common::SerDeOptions serDeOptions(
      fieldDelim, collectionDelim, mapKeyDelim);
  baseReaderOpts_.setSerDeOptions(serDeOptions);
}

std::unique_ptr<dwio::common::BufferedInput> SplitReader::createBufferedInput(
    const FileHandle& fileHandle,
    const dwio::common::ReaderOptions& readerOpts) {
  if (connectorQueryCtx_->cache()) {
    return std::make_unique<dwio::common::CachedBufferedInput>(
        fileHandle.file,
        dwio::common::MetricsLog::voidLog(),
        fileHandle.uuid.id(),
        connectorQueryCtx_->cache(),
        Connector::getTracker(
            connectorQueryCtx_->scanId(), readerOpts.loadQuantum()),
        fileHandle.groupId.id(),
        ioStats_,
        executor_,
        readerOpts);
  }
  return std::make_unique<dwio::common::DirectBufferedInput>(
      fileHandle.file,
      dwio::common::MetricsLog::voidLog(),
      fileHandle.uuid.id(),
      Connector::getTracker(
          connectorQueryCtx_->scanId(), readerOpts.loadQuantum()),
      fileHandle.groupId.id(),
      ioStats_,
      executor_,
      readerOpts);
}

std::string SplitReader::toString() const {
  std::string partitionKeys;
  std::for_each(
      partitionKeys_->begin(),
      partitionKeys_->end(),
      [&](std::pair<
          const std::string,
          std::shared_ptr<facebook::velox::connector::hive::HiveColumnHandle>>
              column) { partitionKeys += " " + column.second->toString(); });
  return fmt::format(
      "SplitReader: hiveSplit_{} scanSpec_{} readerOutputType_{} partitionKeys_{} reader{} rowReader{}",
      hiveSplit_->toString(),
      scanSpec_->toString(),
      readerOutputType_->toString(),
      partitionKeys,
      static_cast<const void*>(baseReader_.get()),
      static_cast<const void*>(baseRowReader_.get()));
}

} // namespace facebook::velox::connector::hive
