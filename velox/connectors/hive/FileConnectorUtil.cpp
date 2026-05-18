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

#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/FileTableHandle.h"

namespace facebook::velox::connector::hive {

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
  readerOptions.setLoadQuantum(fileConfig->loadQuantum(sessionProperties));
  readerOptions.setMaxCoalesceBytes(
      fileConfig->maxCoalescedBytes(sessionProperties));
  readerOptions.setMaxCoalesceDistance(
      fileConfig->maxCoalescedDistanceBytes(sessionProperties));
  readerOptions.setFileColumnNamesReadAsLowerCase(
      fileConfig->isFileColumnNamesReadAsLowerCase(sessionProperties));
  readerOptions.setAllowEmptyFile(true);
  bool useColumnNamesForColumnMapping = false;
  switch (fileSplit->fileFormat) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC: {
      useColumnNamesForColumnMapping =
          fileConfig->isOrcUseColumnNames(sessionProperties);
      break;
    }
    case dwio::common::FileFormat::PARQUET: {
      useColumnNamesForColumnMapping =
          fileConfig->isParquetUseColumnNames(sessionProperties);
      readerOptions.setAllowInt32Narrowing(
          fileConfig->allowInt32Narrowing(sessionProperties));
      break;
    }
    default:
      useColumnNamesForColumnMapping = false;
  }

  readerOptions.setUseColumnNamesForColumnMapping(
      useColumnNamesForColumnMapping);
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
  readerOptions.setFileMetadataCacheEnabled(
      fileConfig->fileMetadataCacheEnabled(sessionProperties));
  readerOptions.setPinFileMetadata(
      fileConfig->pinFileMetadata(sessionProperties));

  // Set footer speculative IO size based on file format.
  switch (fileSplit->fileFormat) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->orcFooterSpeculativeIoSize(sessionProperties));
      break;
    case dwio::common::FileFormat::PARQUET:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->parquetFooterSpeculativeIoSize(sessionProperties));
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
        dwio::common::toString(readerOptions.fileFormat()),
        dwio::common::toString(fileSplit->fileFormat));
  } else {
    readerOptions.setFileFormat(fileSplit->fileFormat);
  }
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
    rowReaderOptions.setCollectColumnCpuMetrics(
        fileConfig->readerCollectColumnCpuMetrics(sessionProperties));
  }
}

namespace {

// Parse timestamp partition value and return the Timestamp object.
// Tries bigint parsing first (with optional precision from columnHandle),
// then falls back to string parsing (ISO 8601, then PrestoCast).
Timestamp parseTimestampPartitionValue(
    const std::string& partitionValue,
    bool asLocalTime,
    const std::shared_ptr<const FileColumnHandle>& columnHandle) {
  // Try to parse as bigint timestamp first.
  auto bigintResult = folly::tryTo<int64_t>(partitionValue);
  if (bigintResult.hasValue()) {
    Timestamp ts;
    
    // Check if column handle provides explicit timestamp precision
    if (columnHandle) {
      auto precision = columnHandle->getPartitionTimestampPrecision();
      if (precision.has_value()) {
        switch (precision.value()) {
          case TimestampPrecision::kMicroseconds:
            ts = Timestamp::fromMicros(bigintResult.value());
            break;
          case TimestampPrecision::kMilliseconds:
            ts = Timestamp::fromMillis(bigintResult.value());
            break;
          case TimestampPrecision::kNanoseconds:
            ts = Timestamp::fromNanos(bigintResult.value());
            break;
        }
        return ts;
      }
    }
    
    // Default behavior: treat as microseconds (Iceberg default)
    // Note: Traditional Presto/Hive uses string format for timestamp partitions.
    // Iceberg identity transform stores timestamps as microseconds in UTC.
    ts = Timestamp::fromMicros(bigintResult.value());
    return ts;
  }
  
  // Fall back to string parsing - try ISO 8601 first (for Iceberg), then PrestoCast.
  auto result = util::fromTimestampString(
      StringView(partitionValue), util::TimestampParseMode::kIso8601);
  
  // If ISO 8601 fails, try PrestoCast for backward compatibility.
  if (result.hasError()) {
    result = util::fromTimestampString(
        StringView(partitionValue), util::TimestampParseMode::kPrestoCast);
  }
  
  VELOX_CHECK(
      !result.hasError(),
      "Failed to parse TIMESTAMP partition value '{}': {}",
      partitionValue,
      result.error().message());
  
  if (asLocalTime) {
    result.value().toGMT(Timestamp::defaultTimezone());
  }
  
  return result.value();
}

bool applyPartitionFilter(
    const TypePtr& type,
    const std::string& partitionValue,
    bool isPartitionDateDaysSinceEpoch,
    const common::Filter* filter,
    bool asLocalTime,
    const std::shared_ptr<const FileColumnHandle>& columnHandle) {
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
      Timestamp ts = parseTimestampPartitionValue(
          partitionValue, asLocalTime, columnHandle);
      return applyFilter(*filter, ts);
    }
    case TypeKind::VARCHAR: {
      return applyFilter(*filter, partitionValue);
    }
    default:
      VELOX_FAIL(
          "Bad type {} for partition value: {}", type->kind(), partitionValue);
  }
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
              asLocalTime,
              handlesIter->second);
        }
        // Column is missing, most likely due to schema evolution. Or it's a
        // partition key but the partition value is NULL.
        if (child->filter()->isDeterministic() &&
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
