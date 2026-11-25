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

#include "velox/connectors/common/FileConnectorUtil.h"

#include <fmt/format.h>
#include <unordered_map>

#include "velox/common/config/Config.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/connectors/common/FileConnectorSplit.h"
#include "velox/connectors/common/FileTableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/common/Config.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/common/ParquetConfig.h"
#endif

namespace facebook::velox::connector {

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
      dwio::common::FileFormatName::toName(fileFormat));
}

dwio::common::ColumnMappingMode sessionColumnMappingMode(
    const FileConfig& fileConfig,
    const config::ConfigBase* sessionProperties) {
  return fileConfig.useColumnNames(sessionProperties)
      ? dwio::common::ColumnMappingMode::kName
      : dwio::common::ColumnMappingMode::kPosition;
}

} // namespace

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

  auto formatScopedConfigs = fileConfig->formatScopedConfigs(
      fileSplit->fileFormat, *sessionProperties);
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
