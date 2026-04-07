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

#include "velox/connectors/hive/HiveSplitReader.h"

#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"

namespace facebook::velox::connector::hive {

HiveSplitReader::HiveSplitReader(
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
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
    const std::unordered_map<std::string, FileColumnHandlePtr>* infoColumns,
    std::vector<column_index_t> bucketChannels,
    const common::SubfieldFilters* subfieldFiltersForValidation)
    : FileSplitReader(
          hiveSplit,
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
          subfieldFiltersForValidation),
      hiveSplit_(hiveSplit),
      infoColumns_(infoColumns) {
  if (!bucketChannels.empty()) {
    bucketChannels_ = {bucketChannels.begin(), bucketChannels.end()};
    partitionFunction_ = std::make_unique<HivePartitionFunction>(
        hiveSplit_->bucketConversion->tableBucketCount,
        std::move(bucketChannels));
  }
}

void HiveSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  validateSynthesizedColumnFilters();
  FileSplitReader::prepareSplit(
      std::move(metadataFilter), runtimeStats, fileReadOps);
}

void HiveSplitReader::validateSynthesizedColumnFilters() const {
  if (!subfieldFiltersForValidation_ || !infoColumns_) {
    return;
  }
  const auto& splitInfoColumns = hiveSplit_->infoColumns;
  for (const auto& [subfield, filter] : *subfieldFiltersForValidation_) {
    const auto& fieldName = subfield.toString();
    auto infoColIter = splitInfoColumns.find(fieldName);
    if (infoColIter == splitInfoColumns.end()) {
      continue;
    }
    bool passed = false;
    const auto& value = infoColIter->second;
    auto handleIter = infoColumns_->find(fieldName);
    VELOX_CHECK(
        handleIter != infoColumns_->end(),
        "Column handle for synthesized column '{}' not found in infoColumns",
        fieldName);
    TypeKind typeKind = handleIter->second->dataType()->kind();
    switch (typeKind) {
      case TypeKind::BIGINT:
      case TypeKind::INTEGER:
        passed = common::applyFilter(*filter, folly::to<int64_t>(value));
        break;
      case TypeKind::VARCHAR:
        passed = common::applyFilter(*filter, value);
        break;
      default:
        VELOX_FAIL("Unexpected type for synthesized column '{}'.", fieldName);
    }
    VELOX_CHECK(
        passed,
        "Synthesized column '{}' failed filter validation. "
        "Filter: {}, Value: '{}'. Split: {}",
        fieldName,
        filter->toString(),
        value,
        fileSplit_->toString());
  }
}

std::vector<TypePtr> HiveSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& tableSchema) const {
  std::vector<TypePtr> columnTypes = fileType->children();

  auto& childrenSpecs = scanSpec_->children();
  for (size_t i = 0; i < childrenSpecs.size(); ++i) {
    auto* childSpec = childrenSpecs[i].get();
    const std::string& fieldName = childSpec->fieldName();

    auto partitionIt = fileSplit_->partitionKeys.find(fieldName);
    if (partitionIt != fileSplit_->partitionKeys.end()) {
      setPartitionValue(childSpec, fieldName, partitionIt->second);
    } else if (
        hiveSplit_->infoColumns.find(fieldName) !=
        hiveSplit_->infoColumns.end()) {
      auto iter = hiveSplit_->infoColumns.find(fieldName);
      auto infoColumnType =
          readerOutputType_->childAt(readerOutputType_->getChildIdx(fieldName));
      auto constant = newConstantFromString(
          infoColumnType,
          iter->second,
          connectorQueryCtx_->memoryPool(),
          fileConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties()),
          false);
      childSpec->setConstantValue(constant);
    } else if (
        childSpec->columnType() == common::ScanSpec::ColumnType::kRegular) {
      auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
      if (!fileTypeIdx.has_value()) {
        VELOX_CHECK(tableSchema, "Unable to resolve column '{}'", fieldName);
        childSpec->setConstantValue(
            BaseVector::createNullConstant(
                tableSchema->findChild(fieldName),
                1,
                connectorQueryCtx_->memoryPool()));
      } else {
        childSpec->setConstantValue(nullptr);
        auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);
        if (outputTypeIdx.has_value()) {
          auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
          auto& columnType = columnTypes[*fileTypeIdx];
          if (childSpec->isFlatMapAsStruct()) {
            VELOX_CHECK(outputType->isRow() && columnType->isMap());
          } else {
            columnType = outputType;
          }
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

void HiveSplitReader::configureBaseReaderOptions() {
  hive::configureReaderOptions(
      fileConfig_,
      connectorQueryCtx_,
      tableHandle_,
      fileSplit_,
      hiveSplit_->serdeParameters,
      baseReaderOpts_);
}

void HiveSplitReader::configureBaseRowReaderOptions(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    RowTypePtr rowType) {
  hive::configureRowReaderOptions(
      tableHandle_->tableParameters(),
      scanSpec_,
      std::move(metadataFilter),
      rowType,
      fileSplit_,
      hiveSplit_->serdeParameters,
      fileConfig_,
      connectorQueryCtx_->sessionProperties(),
      ioExecutor_,
      baseRowReaderOpts_);
}

std::vector<BaseVector::CopyRange> HiveSplitReader::bucketConversionRows(
    const RowVector& vector) {
  partitions_.clear();
  partitionFunction_->partition(vector, partitions_);
  const auto bucketToKeep = *hiveSplit_->tableBucketNumber;
  const auto partitionBucketCount =
      hiveSplit_->bucketConversion->partitionBucketCount;
  std::vector<BaseVector::CopyRange> ranges;
  for (vector_size_t i = 0; i < vector.size(); ++i) {
    VELOX_CHECK_EQ((partitions_[i] - bucketToKeep) % partitionBucketCount, 0);
    if (partitions_[i] == bucketToKeep) {
      auto& r = ranges.emplace_back();
      r.sourceIndex = i;
      r.targetIndex = ranges.size() - 1;
      r.count = 1;
    }
  }
  return ranges;
}

void HiveSplitReader::applyBucketConversion(
    VectorPtr& output,
    const std::vector<BaseVector::CopyRange>& ranges) {
  auto filtered =
      BaseVector::create(output->type(), ranges.size(), output->pool());
  filtered->copyRanges(output.get(), ranges);
  output = std::move(filtered);
}

uint64_t HiveSplitReader::next(uint64_t size, VectorPtr& output) {
  auto numScanned = FileSplitReader::next(size, output);
  if (numScanned > 0 && output->size() > 0 && partitionFunction_) {
    applyBucketConversion(
        output, bucketConversionRows(*output->asChecked<RowVector>()));
  }
  return numScanned;
}

} // namespace facebook::velox::connector::hive
