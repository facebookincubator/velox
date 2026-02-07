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

#include <fmt/ranges.h>
#include <string>
#include <unordered_map>

#include "velox/common/Casts.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConfig.h"

#include "velox/expression/FieldReference.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector::hive {

void HiveDataSource::processColumnHandle(const HiveColumnHandlePtr& handle) {
  switch (handle->columnType()) {
    case HiveColumnHandle::ColumnType::kRegular:
      break;
    case HiveColumnHandle::ColumnType::kPartitionKey:
      partitionKeys_.emplace(handle->name(), handle);
      break;
    case HiveColumnHandle::ColumnType::kSynthesized:
      infoColumns_.emplace(handle->name(), handle);
      break;
    case HiveColumnHandle::ColumnType::kRowIndex:
      specialColumns_.rowIndex = handle->name();
      break;
    case HiveColumnHandle::ColumnType::kRowId:
      specialColumns_.rowId = handle->name();
      break;
  }
}

HiveDataSource::HiveDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig)
    : fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      connectorQueryCtx_(connectorQueryCtx),
      hiveConfig_(hiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  hiveTableHandle_ = checkedPointerCast<const HiveTableHandle>(tableHandle);

  folly::F14FastMap<std::string_view, const HiveColumnHandle*> columnHandles;
  // Column handled keyed on the column alias, the name used in the query.
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const HiveColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    if (!unique) {
      // This should not happen normally, but there is some bug in Presto DELETE
      // queries that sometimes we do get duplicate assignments for partitioning
      // columns.
      checkColumnHandleConsistent(*handle, *it->second);
      VELOX_CHECK_EQ(
          handle->columnType(),
          HiveColumnHandle::ColumnType::kPartitionKey,
          "Cannot map from same table column to different outputs in table scan; a project node should be used instead: {}",
          handle->name());
      continue;
    }
    processColumnHandle(handle);
  }
  for (auto& handle : hiveTableHandle_->filterColumnHandles()) {
    auto it = columnHandles.find(handle->name());
    if (it != columnHandles.end()) {
      checkColumnHandleConsistent(*handle, *it->second);
      continue;
    }
    processColumnHandle(handle);
  }

  std::vector<std::string> readColumnNames;
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = assignments.find(outputName);
    VELOX_CHECK(
        it != assignments.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const HiveColumnHandle*>(it->second.get());
    readColumnNames.push_back(handle->name());
    for (auto& subfield : handle->requiredSubfields()) {
      VELOX_USER_CHECK_EQ(
          getColumnName(subfield),
          handle->name(),
          "Required subfield does not match column name");
      subfields_[handle->name()].push_back(&subfield);
    }
    columnPostProcessors_.push_back(handle->postProcessor());
  }

  if (hiveConfig_->isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx->sessionProperties())) {
    checkColumnNameLowerCase(outputType_);
    checkColumnNameLowerCase(hiveTableHandle_->subfieldFilters(), infoColumns_);
    checkColumnNameLowerCase(hiveTableHandle_->remainingFilter());
  }

  for (const auto& [k, v] : hiveTableHandle_->subfieldFilters()) {
    filters_.emplace(k.clone(), v);
  }
  double sampleRate = hiveTableHandle_->sampleRate();
  auto remainingFilter = extractFiltersFromRemainingFilter(
      hiveTableHandle_->remainingFilter(),
      expressionEvaluator_,
      filters_,
      sampleRate);
  if (sampleRate != 1) {
    randomSkip_ = std::make_shared<random::RandomSkipTracker>(sampleRate);
  }

  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    auto& remainingFilterExpr = remainingFilterExprSet_->expr(0);
    folly::F14FastMap<std::string, column_index_t> columnNames;
    for (int i = 0; i < readColumnNames.size(); ++i) {
      columnNames[readColumnNames[i]] = i;
    }
    for (auto& input : remainingFilterExpr->distinctFields()) {
      auto it = columnNames.find(input->field());
      if (it != columnNames.end()) {
        if (shouldEagerlyMaterialize(*remainingFilterExpr, *input)) {
          multiReferencedFields_.push_back(it->second);
        }
        continue;
      }
      // Remaining filter may reference columns that are not used otherwise,
      // e.g. are not being projected out and are not used in range filters.
      // Make sure to add these columns to readerOutputType_.
      readColumnNames.push_back(input->field());
      readColumnTypes.push_back(input->type());
    }
    remainingFilterSubfields_ = remainingFilterExpr->extractSubfields();
    if (VLOG_IS_ON(1)) {
      VLOG(1) << fmt::format(
          "Extracted subfields from remaining filter: [{}]",
          fmt::join(remainingFilterSubfields_, ", "));
    }
    for (auto& subfield : remainingFilterSubfields_) {
      const auto& name = getColumnName(subfield);
      auto it = subfields_.find(name);
      if (it != subfields_.end()) {
        // Some subfields of the column are already projected out, we append the
        // remainingFilter subfield
        it->second.push_back(&subfield);
      } else if (columnNames.count(name) == 0) {
        // remainingFilter subfield's column is not projected out, we add the
        // column and append the subfield
        subfields_[name].push_back(&subfield);
      }
    }
  }

  readerOutputType_ =
      ROW(std::move(readColumnNames), std::move(readColumnTypes));
  scanSpec_ = makeScanSpec(
      readerOutputType_,
      subfields_,
      filters_,
      /*indexColumns=*/{},
      hiveTableHandle_->dataColumns(),
      partitionKeys_,
      infoColumns_,
      specialColumns_,
      hiveConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);
  if (remainingFilter) {
    metadataFilter_ = std::make_shared<common::MetadataFilter>(
        *scanSpec_, *remainingFilter, expressionEvaluator_);
  }

  ioStats_ = std::make_shared<io::IoStatistics>();
  fsStats_ = std::make_shared<filesystems::File::IoStats>();
}

std::unique_ptr<SplitReader> HiveDataSource::createSplitReader() {
  return SplitReader::create(
      split_,
      hiveTableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      hiveConfig_,
      readerOutputType_,
      ioStats_,
      fsStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_,
      /*subfieldFiltersForValidation=*/&filters_);
}

std::vector<column_index_t> HiveDataSource::setupBucketConversion() {
  VELOX_CHECK_NE(
      split_->bucketConversion->tableBucketCount,
      split_->bucketConversion->partitionBucketCount);
  VELOX_CHECK(split_->tableBucketNumber.has_value());
  VELOX_CHECK_NOT_NULL(hiveTableHandle_->dataColumns());
  ++numBucketConversion_;
  bool rebuildScanSpec = false;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<column_index_t> bucketChannels;
  for (auto& handle : split_->bucketConversion->bucketColumnHandles) {
    VELOX_CHECK(handle->columnType() == HiveColumnHandle::ColumnType::kRegular);
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
      types.push_back(
          hiveTableHandle_->dataColumns()->findChild(handle->name()));
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
        hiveTableHandle_->dataColumns(),
        partitionKeys_,
        infoColumns_,
        specialColumns_,
        hiveConfig_->readStatsBasedFilterReorderDisabled(
            connectorQueryCtx_->sessionProperties()),
        pool_);
    newScanSpec->moveAdaptationFrom(*scanSpec_);
    scanSpec_ = std::move(newScanSpec);
  }
  return bucketChannels;
}

void HiveDataSource::setupRowIdColumn() {
  VELOX_CHECK(split_->rowIdProperties.has_value());
  const auto& props = *split_->rowIdProperties;
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

void HiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK_NULL(
      split_,
      "Previous split has not been processed yet. Call next to process the split.");
  split_ = checkedPointerCast<HiveConnectorSplit>(split);

  VLOG(1) << "Adding split " << split_->toString();

  if (splitReader_) {
    splitReader_.reset();
  }

  std::vector<column_index_t> bucketChannels;
  if (split_->bucketConversion.has_value()) {
    bucketChannels = setupBucketConversion();
  }
  if (specialColumns_.rowId.has_value()) {
    setupRowIdColumn();
  }

  splitReader_ = createSplitReader();
  splitReader_->setInfoColumns(&infoColumns_);
  if (!bucketChannels.empty()) {
    splitReader_->setBucketConversion(std::move(bucketChannels));
  }
  // Split reader subclasses may need to use the reader options in prepareSplit
  // so we initialize it beforehand.
  splitReader_->configureReaderOptions(randomSkip_);
  splitReader_->prepareSplit(metadataFilter_, runtimeStats_);
  readerOutputType_ = splitReader_->readerOutputType();
}

std::optional<RowVectorPtr> HiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK(split_ != nullptr, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(splitReader_, "No split reader present");

  TestValue::adjust(
      "facebook::velox::connector::hive::HiveDataSource::next", this);

  if (splitReader_->emptySplit()) {
    resetSplit();
    return nullptr;
  }

  // Bucket conversion or delta update could add extra column to reader output.
  auto needsExtraColumn = [&] {
    return output_->asUnchecked<RowVector>()->childrenSize() <
        readerOutputType_->size();
  };
  if (!output_ || needsExtraColumn()) {
    output_ = BaseVector::create(readerOutputType_, 0, pool_);
  }

  const auto rowsScanned = splitReader_->next(size, output_);
  completedRows_ += rowsScanned;
  if (rowsScanned == 0) {
    splitReader_->updateRuntimeStats(runtimeStats_);
    resetSplit();
    return nullptr;
  }

  VELOX_CHECK(
      !output_->mayHaveNulls(), "Top-level row vector cannot have nulls");
  auto rowsRemaining = output_->size();
  if (rowsRemaining == 0) {
    // no rows passed the pushed down filters.
    return getEmptyOutput();
  }

  auto rowVector = std::dynamic_pointer_cast<RowVector>(output_);

  // In case there is a remaining filter that excludes some but not all
  // rows, collect the indices of the passing rows. If there is no filter,
  // or it passes on all rows, leave this as null and let exec::wrap skip
  // wrapping the results.
  BufferPtr remainingIndices;
  filterRows_.resize(rowVector->size());

  if (remainingFilterExprSet_) {
    rowsRemaining = evaluateRemainingFilter(rowVector);
    VELOX_CHECK_LE(rowsRemaining, rowsScanned);
    if (rowsRemaining == 0) {
      // No rows passed the remaining filter.
      return getEmptyOutput();
    }

    if (rowsRemaining < rowVector->size()) {
      // Some, but not all rows passed the remaining filter.
      remainingIndices = filterEvalCtx_.selectedIndices;
    }
  }

  if (outputType_->size() == 0) {
    return exec::wrap(rowsRemaining, remainingIndices, rowVector);
  }

  std::vector<VectorPtr> outputColumns;
  outputColumns.reserve(outputType_->size());
  for (int i = 0; i < outputType_->size(); ++i) {
    auto& child = rowVector->childAt(i);
    if (remainingIndices) {
      // Disable dictionary values caching in expression eval so that we
      // don't need to reallocate the result for every batch.
      child->disableMemo();
    }
    auto column = exec::wrapChild(rowsRemaining, remainingIndices, child);
    if (columnPostProcessors_[i]) {
      columnPostProcessors_[i](column);
    }
    outputColumns.push_back(std::move(column));
  }

  return std::make_shared<RowVector>(
      pool_, outputType_, BufferPtr(nullptr), rowsRemaining, outputColumns);
}

void HiveDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  auto& fieldSpec = scanSpec_->getChildByChannel(outputChannel);
  fieldSpec.setFilter(filter);
  scanSpec_->resetCachedValues(true);
  if (splitReader_) {
    splitReader_->resetFilterCaches();
  }
}

std::unordered_map<std::string, RuntimeMetric>
HiveDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  res.insert(
      {Connector::kIoWaitWallNanos,
       RuntimeMetric(
           ioStats_->queryThreadIoLatencyUs().sum() * 1'000,
           ioStats_->queryThreadIoLatencyUs().count(),
           ioStats_->queryThreadIoLatencyUs().min() * 1'000,
           ioStats_->queryThreadIoLatencyUs().max() * 1'000,
           RuntimeCounter::Unit::kNanos)});
  // Breakdown of ioWaitWallNanos by I/O type
  if (ioStats_->storageReadLatencyUs().count() > 0) {
    res.insert(
        {Connector::kStorageReadWallNanos,
         RuntimeMetric(
             ioStats_->storageReadLatencyUs().sum() * 1'000,
             ioStats_->storageReadLatencyUs().count(),
             ioStats_->storageReadLatencyUs().min() * 1'000,
             ioStats_->storageReadLatencyUs().max() * 1'000,
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStats_->ssdCacheReadLatencyUs().count() > 0) {
    res.insert(
        {Connector::kSsdCacheReadWallNanos,
         RuntimeMetric(
             ioStats_->ssdCacheReadLatencyUs().sum() * 1'000,
             ioStats_->ssdCacheReadLatencyUs().count(),
             ioStats_->ssdCacheReadLatencyUs().min() * 1'000,
             ioStats_->ssdCacheReadLatencyUs().max() * 1'000,
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStats_->cacheWaitLatencyUs().count() > 0) {
    res.insert(
        {Connector::kCacheWaitWallNanos,
         RuntimeMetric(
             ioStats_->cacheWaitLatencyUs().sum() * 1'000,
             ioStats_->cacheWaitLatencyUs().count(),
             ioStats_->cacheWaitLatencyUs().min() * 1'000,
             ioStats_->cacheWaitLatencyUs().max() * 1'000,
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStats_->coalescedSsdLoadLatencyUs().count() > 0) {
    res.insert(
        {Connector::kCoalescedSsdLoadWallNanos,
         RuntimeMetric(
             ioStats_->coalescedSsdLoadLatencyUs().sum() * 1'000,
             ioStats_->coalescedSsdLoadLatencyUs().count(),
             ioStats_->coalescedSsdLoadLatencyUs().min() * 1'000,
             ioStats_->coalescedSsdLoadLatencyUs().max() * 1'000,
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStats_->coalescedStorageLoadLatencyUs().count() > 0) {
    res.insert(
        {Connector::kCoalescedStorageLoadWallNanos,
         RuntimeMetric(
             ioStats_->coalescedStorageLoadLatencyUs().sum() * 1'000,
             ioStats_->coalescedStorageLoadLatencyUs().count(),
             ioStats_->coalescedStorageLoadLatencyUs().min() * 1'000,
             ioStats_->coalescedStorageLoadLatencyUs().max() * 1'000,
             RuntimeCounter::Unit::kNanos)});
  }
  res.insert(
      {{"numPrefetch", RuntimeMetric(ioStats_->prefetch().count())},
       {"prefetchBytes",
        RuntimeMetric(
            ioStats_->prefetch().sum(),
            ioStats_->prefetch().count(),
            ioStats_->prefetch().min(),
            ioStats_->prefetch().max(),
            RuntimeCounter::Unit::kBytes)},
       {"totalScanTime",
        RuntimeMetric(ioStats_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
       {Connector::kTotalRemainingFilterTime,
        RuntimeMetric(
            totalRemainingFilterTime_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos)},
       {"overreadBytes",
        RuntimeMetric(
            ioStats_->rawOverreadBytes(), RuntimeCounter::Unit::kBytes)}});
  if (ioStats_->read().count() > 0) {
    res.insert(
        {"storageReadBytes",
         RuntimeMetric(
             ioStats_->read().sum(),
             ioStats_->read().count(),
             ioStats_->read().min(),
             ioStats_->read().max(),
             RuntimeCounter::Unit::kBytes)});
  }
  if (ioStats_->ssdRead().count() > 0) {
    res.insert({"numLocalRead", RuntimeMetric(ioStats_->ssdRead().count())});
    res.insert(
        {"localReadBytes",
         RuntimeMetric(
             ioStats_->ssdRead().sum(),
             ioStats_->ssdRead().count(),
             ioStats_->ssdRead().min(),
             ioStats_->ssdRead().max(),
             RuntimeCounter::Unit::kBytes)});
  }
  if (ioStats_->ramHit().count() > 0) {
    res.insert({"numRamRead", RuntimeMetric(ioStats_->ramHit().count())});
    res.insert(
        {"ramReadBytes",
         RuntimeMetric(
             ioStats_->ramHit().sum(),
             ioStats_->ramHit().count(),
             ioStats_->ramHit().min(),
             ioStats_->ramHit().max(),
             RuntimeCounter::Unit::kBytes)});
  }
  if (numBucketConversion_ > 0) {
    res.insert({"numBucketConversion", RuntimeMetric(numBucketConversion_)});
  }

  const auto fsStats = fsStats_->stats();
  for (const auto& storageStats : fsStats) {
    res.emplace(storageStats.first, storageStats.second);
  }
  return res;
}

void HiveDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto source = dynamic_cast<HiveDataSource*>(sourceUnique.get());
  VELOX_CHECK_NOT_NULL(source, "Bad DataSource type");

  split_ = std::move(source->split_);
  runtimeStats_.skippedSplits += source->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits += source->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes += source->runtimeStats_.skippedSplitBytes;
  readerOutputType_ = std::move(source->readerOutputType_);
  source->scanSpec_->moveAdaptationFrom(*scanSpec_);
  scanSpec_ = std::move(source->scanSpec_);
  metadataFilter_ = std::move(source->metadataFilter_);
  splitReader_ = std::move(source->splitReader_);
  splitReader_->setConnectorQueryCtx(connectorQueryCtx_);
  // New io will be accounted on the stats of 'source'. Add the existing
  // balance to that.
  source->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(source->ioStats_);
  source->fsStats_->merge(*fsStats_);
  fsStats_ = std::move(source->fsStats_);

  numBucketConversion_ += source->numBucketConversion_;
}

int64_t HiveDataSource::estimatedRowSize() {
  if (splitReader_ == nullptr) {
    return kUnknownRowSize;
  }
  auto rowSize = splitReader_->estimatedRowSize();
  TestValue::adjust(
      "facebook::velox::connector::hive::HiveDataSource::estimatedRowSize",
      &rowSize);
  return rowSize;
}

vector_size_t HiveDataSource::evaluateRemainingFilter(RowVectorPtr& rowVector) {
  for (auto fieldIndex : multiReferencedFields_) {
    LazyVector::ensureLoadedRows(
        rowVector->childAt(fieldIndex),
        filterRows_,
        filterLazyDecoded_,
        filterLazyBaseRows_);
  }
  uint64_t filterTimeUs{0};
  vector_size_t rowsRemaining{0};
  {
    MicrosecondTimer timer(&filterTimeUs);
    expressionEvaluator_->evaluate(
        remainingFilterExprSet_.get(), filterRows_, *rowVector, filterResult_);
    rowsRemaining = exec::processFilterResults(
        filterResult_, filterRows_, filterEvalCtx_, pool_);
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);
  return rowsRemaining;
}

void HiveDataSource::resetSplit() {
  split_.reset();
  splitReader_->resetSplit();
  // Keep readers around to hold adaptation.
}

HiveDataSource::WaveDelegateHookFunction HiveDataSource::waveDelegateHook_;

std::shared_ptr<wave::WaveDataSource> HiveDataSource::toWaveDataSource() {
  VELOX_CHECK_NOT_NULL(waveDelegateHook_);
  if (!waveDataSource_) {
    waveDataSource_ = waveDelegateHook_(
        hiveTableHandle_,
        scanSpec_,
        readerOutputType_,
        &partitionKeys_,
        fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx_,
        hiveConfig_,
        ioStats_,
        remainingFilterExprSet_.get(),
        metadataFilter_);
  }
  return waveDataSource_;
}

//  static
void HiveDataSource::registerWaveDelegateHook(WaveDelegateHookFunction hook) {
  waveDelegateHook_ = hook;
}
std::shared_ptr<wave::WaveDataSource> toWaveDataSource();

} // namespace facebook::velox::connector::hive
