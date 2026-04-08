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

#include "velox/connectors/hive/HiveDataSource.h"

#include <utility>

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/HiveSplitReader.h"

namespace facebook::velox::connector::hive {

HiveDataSource::HiveDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig)
    : FileDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          hiveConfig),
      hiveConfig_(hiveConfig) {}

std::vector<column_index_t> HiveDataSource::setupBucketConversion() {
  auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split_);
  VELOX_CHECK_NE(
      hiveSplit->bucketConversion->tableBucketCount,
      hiveSplit->bucketConversion->partitionBucketCount);
  VELOX_CHECK(hiveSplit->tableBucketNumber.has_value());
  VELOX_CHECK_NOT_NULL(tableHandle_->dataColumns());
  ++numBucketConversion_;
  bool rebuildScanSpec = false;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<column_index_t> bucketChannels;
  for (auto& handle : hiveSplit->bucketConversion->bucketColumnHandles) {
    VELOX_CHECK(handle->columnType() == FileColumnHandle::ColumnType::kRegular);
    if (subfields_.erase(handle->name()) > 0) {
      rebuildScanSpec = true;
    }
    auto index = readerOutputType_->getChildIdxIfExists(handle->name());
    if (!index.has_value()) {
      if (names.empty()) {
        names = readerOutputType_->names();
        types = readerOutputType_->children();
      }
      index = names.size();
      names.push_back(handle->name());
      types.push_back(tableHandle_->dataColumns()->findChild(handle->name()));
      rebuildScanSpec = true;
    }
    bucketChannels.push_back(*index);
  }
  if (!names.empty()) {
    readerOutputType_ = ROW(std::move(names), std::move(types));
  }
  if (rebuildScanSpec) {
    auto newScanSpec = makeScanSpec(
        readerOutputType_,
        subfields_,
        filters_,
        /*indexColumns=*/{},
        tableHandle_->dataColumns(),
        partitionKeys_,
        infoColumns_,
        specialColumns_,
        fileConfig_->readStatsBasedFilterReorderDisabled(
            connectorQueryCtx_->sessionProperties()),
        pool_);
    newScanSpec->moveAdaptationFrom(*scanSpec_);
    scanSpec_ = std::move(newScanSpec);
  }
  return bucketChannels;
}

void HiveDataSource::setupRowIdColumn() {
  auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split_);
  VELOX_CHECK(hiveSplit->rowIdProperties.has_value());
  const auto& props = *hiveSplit->rowIdProperties;
  auto* rowId = scanSpec_->childByName(*specialColumns_.rowId);
  VELOX_CHECK_NOT_NULL(rowId);
  auto& rowIdType =
      readerOutputType_->findChild(*specialColumns_.rowId)->asRow();
  auto rowGroupId = split_->getFileName();
  rowId->childByName(rowIdType.nameOf(1))
      ->setConstantValue<StringView>(
          StringView(rowGroupId), VARCHAR(), connectorQueryCtx_->memoryPool());
  rowId->childByName(rowIdType.nameOf(2))
      ->setConstantValue<int64_t>(
          props.metadataVersion, BIGINT(), connectorQueryCtx_->memoryPool());
  rowId->childByName(rowIdType.nameOf(3))
      ->setConstantValue<int64_t>(
          props.partitionId, BIGINT(), connectorQueryCtx_->memoryPool());
  rowId->childByName(rowIdType.nameOf(4))
      ->setConstantValue<StringView>(
          StringView(props.tableGuid),
          VARCHAR(),
          connectorQueryCtx_->memoryPool());
}

std::vector<column_index_t> HiveDataSource::prepareSplit() {
  auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split_);
  ++numSplitsByFileFormat_[split_->fileFormat];
  std::vector<column_index_t> bucketChannels;
  if (hiveSplit->bucketConversion.has_value()) {
    bucketChannels = setupBucketConversion();
  }
  if (specialColumns_.rowId.has_value()) {
    setupRowIdColumn();
  }
  return bucketChannels;
}

std::unique_ptr<FileSplitReader> HiveDataSource::createSplitReader() {
  auto bucketChannels = prepareSplit();
  auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split_);

  return std::make_unique<HiveSplitReader>(
      hiveSplit,
      tableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      ioStatistics_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_,
      &infoColumns_,
      std::move(bucketChannels),
      /*subfieldFiltersForValidation=*/&filters_);
}

std::unordered_map<std::string, RuntimeMetric>
HiveDataSource::getRuntimeStats() {
  auto result = FileDataSource::getRuntimeStats();
  if (numBucketConversion_ > 0) {
    result.insert(
        {std::string(kNumBucketConversion),
         RuntimeMetric(numBucketConversion_)});
  }
  for (const auto& [format, count] : numSplitsByFileFormat_) {
    result.insert(
        {fmt::format("{}{}", kFileFormat, dwio::common::toString(format)),
         RuntimeMetric(count)});
  }
  return result;
}

void HiveDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto* source = checkedPointerCast<HiveDataSource>(sourceUnique.get());
  numBucketConversion_ += source->numBucketConversion_;
  for (const auto& [format, count] : source->numSplitsByFileFormat_) {
    numSplitsByFileFormat_[format] += count;
  }
  FileDataSource::setFromDataSource(std::move(sourceUnique));
}

HiveDataSource::WaveDelegateHookFunction HiveDataSource::waveDelegateHook_;

std::shared_ptr<wave::WaveDataSource> HiveDataSource::toWaveDataSource() {
  VELOX_CHECK_NOT_NULL(waveDelegateHook_);
  if (!waveDataSource_) {
    auto hiveTableHandle =
        checkedPointerCast<const HiveTableHandle>(tableHandle_);
    waveDataSource_ = waveDelegateHook_(
        hiveTableHandle,
        scanSpec_,
        readerOutputType_,
        &partitionKeys_,
        fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx_,
        hiveConfig_,
        ioStatistics_,
        remainingFilterExprSet(),
        metadataFilter());
  }
  return waveDataSource_;
}

//  static
void HiveDataSource::registerWaveDelegateHook(WaveDelegateHookFunction hook) {
  waveDelegateHook_ = hook;
}

} // namespace facebook::velox::connector::hive
