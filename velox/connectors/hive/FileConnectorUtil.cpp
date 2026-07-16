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

#include "velox/connectors/hive/FileConnectorUtil.h"

#include <fmt/format.h>
#include <unordered_map>

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive {

FormatScopedConfigs makeFormatScopedConfigs(
    const FileConfig& fileConfig,
    const config::ConfigBase& sessionProperties,
    dwio::common::FileFormat fileFormat) {
  VELOX_CHECK_NE(
      fileFormat,
      dwio::common::FileFormat::UNKNOWN,
      "Cannot build format-specific configs for unknown file format");

  return {
      config::ConfigBase(fileConfig.config()->rawConfigsWithPrefix(
          fmt::format(
              "{}{}",
              fileConfig.connectorConfigPrefix(),
              dwio::common::formatConfigPrefix(fileFormat, ".")))),
      config::ConfigBase(sessionProperties.rawConfigsWithPrefix(
          dwio::common::formatConfigPrefix(fileFormat, "_")))};
}

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    dwio::common::ReaderOptions& readerOptions) {
  configureReaderOptions(
      fileConfig,
      connectorQueryCtx,
      tableHandle->dataColumns(),
      fileSplit,
      tableHandle->tableParameters(),
      readerOptions);
}

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& /*tableParameters*/,
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
  auto columnMappingMode = dwio::common::ColumnMappingMode::kPosition;
  switch (fileSplit->fileFormat) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC: {
      columnMappingMode = fileConfig->isOrcUseColumnNames(sessionProperties)
          ? dwio::common::ColumnMappingMode::kName
          : dwio::common::ColumnMappingMode::kPosition;
      break;
    }
    default:
      columnMappingMode = dwio::common::ColumnMappingMode::kPosition;
  }

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
    VELOX_CHECK(
        readerOptions.fileFormat() == fileSplit->fileFormat,
        "HiveDataSource received splits of different formats: {} and {}",
        dwio::common::FileFormatName::toName(readerOptions.fileFormat()),
        dwio::common::FileFormatName::toName(fileSplit->fileFormat));
  } else {
    readerOptions.setFileFormat(fileSplit->fileFormat);
  }

  if (!dwio::common::hasReaderFactory(fileSplit->fileFormat)) {
    readerOptions.setFormatSpecificOptions(nullptr);
    return;
  }

  auto formatScopedConfigs = makeFormatScopedConfigs(
      *fileConfig, *sessionProperties, fileSplit->fileFormat);
  readerOptions.setFormatSpecificOptions(
      dwio::common::getReaderFactory(fileSplit->fileFormat)
          ->createFormatOptions(
              formatScopedConfigs.connectorConfig,
              formatScopedConfigs.sessionProperties));
}

void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* const ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions) {
  auto skipRowsIt =
      tableParameters.find(dwio::common::TableParameter::kSkipHeaderLineCount);
  if (skipRowsIt != tableParameters.end()) {
    rowReaderOptions.setSkipRows(folly::to<uint64_t>(skipRowsIt->second));
  }
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setIOExecutor(ioExecutor);
  rowReaderOptions.setMetadataFilter(std::move(metadataFilter));
  rowReaderOptions.setRequestedType(rowType);
  rowReaderOptions.range(fileSplit->start, fileSplit->length);
  if (fileConfig && sessionProperties) {
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
    rowReaderOptions.setStringDecoderZeroCopy(
        fileConfig->nimbleStringDecoderZeroCopy(sessionProperties));
    rowReaderOptions.setNimblePreserveDictionaryEncoding(
        fileConfig->nimblePreserveDictionaryEncoding(sessionProperties));
  }
}

namespace {

bool applyPartitionFilter(
    const TypePtr& type,
    const std::string& partitionValue,
    bool isPartitionDateDaysSinceEpoch,
    const common::Filter* filter,
    bool asLocalTime) {
  if (type->isDate()) {
    int32_t result = 0;
    // days_since_epoch partition values are integers in string format. Eg.
    // Iceberg partition values.
    if (isPartitionDateDaysSinceEpoch) {
      result = folly::to<int32_t>(partitionValue);
    } else {
      result = DATE()->toDays(partitionValue);
    }
    return applyFilter(*filter, result);
  }

  switch (type->kind()) {
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
    case TypeKind::TIMESTAMP: {
      VELOX_DCHECK(type->equivalent(*TIMESTAMP()));
      auto result = util::fromTimestampString(
          StringView(partitionValue), util::TimestampParseMode::kPrestoCast);
      VELOX_CHECK(!result.hasError());
      if (asLocalTime) {
        result.value().toGMT(Timestamp::defaultTimezone());
      }
      return applyFilter(*filter, result.value());
    }
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      return applyFilter(*filter, partitionValue);
    }
    default:
      VELOX_FAIL(
          "Bad type {} for partition value: {}", type->kind(), partitionValue);
  }
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

bool testFilters(
    const common::ScanSpec* scanSpec,
    const dwio::common::Reader* reader,
    const std::string& filePath,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>&
        partitionKeysHandle,
    bool asLocalTime) {
  const auto totalRows = reader->numberOfRows();
  const auto& fileTypeWithId = reader->typeWithId();
  const auto& rowType = reader->rowType();
  for (const auto& child : scanSpec->children()) {
    if (child->filter()) {
      const auto& name = child->fieldName();
      auto iter = partitionKeys.find(name);
      // By design, the partition key columns for Iceberg tables are included in
      // the data files to facilitate partition transform and partition
      // evolution, so we need to test both cases.
      if (!rowType->containsChild(name) || iter != partitionKeys.end()) {
        if (iter != partitionKeys.end() && iter->second.has_value()) {
          const auto handlesIter = partitionKeysHandle.find(name);
          VELOX_CHECK(handlesIter != partitionKeysHandle.end());

          // This is a non-null partition key
          return applyPartitionFilter(
              handlesIter->second->dataType(),
              iter->second.value(),
              handlesIter->second->isPartitionDateValueDaysSinceEpoch(),
              child->filter(),
              asLocalTime);
        }
        // Column is missing from the file. If it has a constant value (e.g.,
        // an initial-default from schema evolution), test the filter against
        // it. Otherwise treat the column as NULL.
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
          VLOG(1) << "Skipping " << filePath
                  << " because the filter testNull() failed for column "
                  << child->fieldName();
          return false;
        }
      } else {
        const auto& typeWithId = fileTypeWithId->childByName(name);
        const auto columnStats = reader->columnStatistics(typeWithId->id());
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

} // namespace facebook::velox::connector::hive
