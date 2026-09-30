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

#include "velox/connectors/hive/delta/DeltaSplitReader.h"

#include "velox/connectors/hive/ConstantFromString.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/HiveSplitReader.h"
#include "velox/connectors/hive/delta/HiveDeltaSplit.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::delta {

DeltaSplitReader::DeltaSplitReader(
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
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
    const std::unordered_map<std::string, FileColumnHandlePtr>* infoColumns,
    std::vector<column_index_t> bucketChannels,
    const common::SubfieldFilters* subfieldFiltersForValidation)
    : HiveSplitReader(
          hiveSplit,
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
          infoColumns,
          std::move(bucketChannels),
          subfieldFiltersForValidation) {}

void DeltaSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  const auto* deltaSplit = dynamic_cast<const HiveDeltaSplit*>(hiveSplit_.get());

  // Reading files with logically deleted rows is not yet supported. Reject
  // early rather than reading the file and returning the deleted rows.
  VELOX_USER_CHECK(
      deltaSplit == nullptr || !deltaSplit->hasDeletionVector,
      "Reading Delta files with a deletion vector is not supported.");

  // Read by name. The coordinator resolves each column to its physical
  // Parquet name on the HiveColumnHandle (see
  // DeltaPrestoToVeloxConnector::sourceName), so this works uniformly for
  // all three modes ('none', 'name', 'id'): in each case the column handle
  // already carries the name that appears in the Parquet file. The mode
  // on the split is informational -- the worker does not gate on it
  // today.
  //
  // TODO(delta): The Delta spec strictly requires resolving id-mode
  // columns by Parquet field_id (not by name), and rejecting or nulling
  // when a file has no field_ids. Route kId through
  // dwio::common::ColumnMappingMode::kParquetFieldId with field_ids
  // provided on the column handles once we surface field_ids on the wire.
  // Until then, name-based resolution works for delta.io writers (they
  // stamp physicalName as the Parquet column name alongside field_id) but
  // is not spec-strict against a spec-only-field_id writer.
  baseReaderOpts_.setColumnMappingMode(dwio::common::ColumnMappingMode::kName);

  // Delegate to the base Hive read path; adaptColumns() is virtual and picks
  // up this class's override.
  HiveSplitReader::prepareSplit(
      std::move(metadataFilter), runtimeStats, fileReadOps);
}

uint64_t DeltaSplitReader::next(uint64_t size, VectorPtr& output) {
  // Mutation::deletedRows defaults to nullptr; DV-carrying splits are
  // rejected in prepareSplit(), so there are no logically deleted rows to
  // pass here.
  Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();

  const auto actualSize = baseRowReader_->nextReadSize(size);
  if (actualSize == dwio::common::RowReader::kAtEnd) {
    return 0;
  }

  auto rowsScanned = baseRowReader_->next(actualSize, output, &mutation);

  return rowsScanned;
}

std::vector<TypePtr> DeltaSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& /*tableSchema*/) const {
  // Delta ignores the tableSchema parameter (baseReaderOpts_.fileSchema()) and
  // uses readerOutputType_ as the source of truth. The Presto coordinator sends
  // the output projection as the table schema for Delta scans, so
  // readerOutputType_ carries every logical column the reader must resolve.
  std::vector<TypePtr> columnTypes = fileType->children();
  auto& childrenSpecs = scanSpec_->children();
  const bool readTimestampAsLocalTime =
      fileConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());

  for (const auto& childSpec : childrenSpecs) {
    const std::string& fieldName = childSpec->fieldName();

    // 1. Info column ($path, $file_size, ...): install the split's provided
    // metadata value as a constant.
    if (auto infoIt = hiveSplit_->infoColumns.find(fieldName);
        infoIt != hiveSplit_->infoColumns.end()) {
      childSpec->setConstantValue(newConstantFromString(
          readerOutputType_->findChild(fieldName),
          infoIt->second,
          connectorQueryCtx_->memoryPool(),
          readTimestampAsLocalTime,
          /*isDaysSinceEpoch=*/false,
          adjustTimestampToTimezone_ ? sessionTimezone_ : nullptr));
      continue;
    }

    // 2. Partition column: install the split's partition value as a constant.
    // Partition columns are never stored in the data file for Delta tables.
    if (auto partitionIt = hiveSplit_->partitionKeys.find(fieldName);
        partitionIt != hiveSplit_->partitionKeys.end()) {
      setPartitionValue(childSpec.get(), fieldName, partitionIt->second);
      continue;
    }

    // 3. Data column present in the file: read it. Clear any stale constant
    // left behind by a previous split's adaptColumns pass so this column reads
    // from the file rather than returning a cached constant.
    const auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
    const auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);
    if (fileTypeIdx.has_value() && outputTypeIdx.has_value()) {
      if (childSpec->isConstant()) {
        childSpec->setConstantValue(nullptr);
      }
      columnTypes[*fileTypeIdx] = readerOutputType_->childAt(*outputTypeIdx);
      continue;
    }

    // 4. Column missing from the data file (Delta schema evolution — a column
    // was added after this file was written). Materialize as a null constant
    // of the logical type.
    if (!fileTypeIdx.has_value()) {
      VELOX_CHECK_NOT_NULL(
          readerOutputType_,
          "Unable to resolve missing column '{}'",
          fieldName);
      childSpec->setConstantValue(
          BaseVector::createNullConstant(
              readerOutputType_->findChild(fieldName),
              1,
              connectorQueryCtx_->memoryPool()));
    }
  }

  // The ScanSpec is reused across splits within a DataSource; child specs
  // above may have flipped between constant and non-constant, so invalidate
  // the cached hasFilter_ derivation before the next split evaluates it.
  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

void registerHiveDeltaSplitReader() {
  HiveSplitReader::registerFactory(
      "hive-delta",
      [](const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
         const FileTableHandlePtr& tableHandle,
         const std::unordered_map<std::string, FileColumnHandlePtr>*
             partitionKeys,
         const ConnectorQueryCtx* connectorQueryCtx,
         const std::shared_ptr<const FileConfig>& fileConfig,
         const RowTypePtr& readerOutputType,
         const std::shared_ptr<io::IoStatistics>& dataIoStats,
         const std::shared_ptr<io::IoStatistics>& metadataIoStats,
         const std::shared_ptr<IoStats>& ioStats,
         FileHandleFactory* fileHandleFactory,
         folly::Executor* ioExecutor,
         const std::shared_ptr<common::ScanSpec>& scanSpec,
         const std::unordered_map<std::string, FileColumnHandlePtr>*
             infoColumns,
         std::vector<column_index_t> bucketChannels,
         const common::SubfieldFilters* subfieldFiltersForValidation)
          -> std::unique_ptr<FileSplitReader> {
        auto deltaSplit =
            std::dynamic_pointer_cast<const HiveDeltaSplit>(hiveSplit);
        VELOX_CHECK_NOT_NULL(
            deltaSplit, "Expected HiveDeltaSplit for table_format=hive-delta");
        return std::make_unique<DeltaSplitReader>(
            deltaSplit,
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
            infoColumns,
            std::move(bucketChannels),
            subfieldFiltersForValidation);
      });
}

} // namespace facebook::velox::connector::hive::delta
