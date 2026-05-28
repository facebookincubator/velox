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

#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/delta/HiveDeltaSplit.h"
#include "velox/core/VectorUtil.h"

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
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  baseReaderOpts_.setColumnMappingMode(dwio::common::ColumnMappingMode::kName);
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

uint64_t DeltaSplitReader::next(uint64_t size, VectorPtr& output) {
  Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  mutation.deletedRows = nullptr;

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
  auto tableSchema = readerOutputType_;
  std::vector<TypePtr> columnTypes = fileType->children();
  auto& childrenSpecs = scanSpec_->children();

  for (const auto& childSpec : childrenSpecs) {
    const std::string& fieldName = childSpec->fieldName();

    if (auto iter = hiveSplit_->infoColumns.find(fieldName);
        iter != hiveSplit_->infoColumns.end()) {
      // Handle info columns (e.g., $path, $file_size)
      auto infoColumnType = readerOutputType_->findChild(fieldName);
      auto constant = core::newConstantFromString(
          infoColumnType,
          iter->second,
          connectorQueryCtx_->memoryPool(),
          fileConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties()),
          false,
          adjustTimestampToTimezone_ ? sessionTimezone_ : nullptr);
      childSpec->setConstantValue(constant);
    } else {
      // Check if this is a partition column first
      if (auto it = hiveSplit_->partitionKeys.find(fieldName);
          it != hiveSplit_->partitionKeys.end()) {
        // Partition column - set constant value from partition metadata
        // Partition columns should never be in the file
        setPartitionValue(childSpec.get(), fieldName, it->second);
      } else {
        // Not a partition column - check if it exists in the file
        auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
        auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);

        if (outputTypeIdx.has_value() && fileTypeIdx.has_value()) {
          // Column exists in both file and output - read from file
          // Clear any placeholder constant that may have been set
          if (childSpec->isConstant()) {
            childSpec->setConstantValue(nullptr);
          }
          auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
          columnTypes[*fileTypeIdx] = outputType;
        } else if (!fileTypeIdx.has_value()) {
          // Column missing from file - schema evolution case
          VELOX_CHECK(tableSchema, "Unable to resolve column '{}'", fieldName);
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  tableSchema->findChild(fieldName),
                  1,
                  connectorQueryCtx_->memoryPool()));
        }
      }
    }
  }

  return columnTypes;
}

} // namespace facebook::velox::connector::hive::delta

