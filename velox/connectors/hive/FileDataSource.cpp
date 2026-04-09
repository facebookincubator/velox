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

#include "velox/connectors/hive/FileDataSource.h"

#include <fmt/ranges.h>
#include <string>
#include <unordered_map>

#include "velox/common/Casts.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/expression/FieldReference.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector::hive {

void FileDataSource::processColumnHandle(const FileColumnHandlePtr& handle) {
  switch (handle->columnType()) {
    case FileColumnHandle::ColumnType::kRegular:
      break;
    case FileColumnHandle::ColumnType::kPartitionKey:
      partitionKeys_.emplace(handle->name(), handle);
      break;
    case FileColumnHandle::ColumnType::kSynthesized:
      infoColumns_.emplace(handle->name(), handle);
      break;
    case FileColumnHandle::ColumnType::kRowIndex:
      specialColumns_.rowIndex = handle->name();
      break;
    case FileColumnHandle::ColumnType::kRowId:
      specialColumns_.rowId = handle->name();
      break;
  }
}

FileDataSource::FileDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<FileConfig>& fileConfig)
    : fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      connectorQueryCtx_(connectorQueryCtx),
      fileConfig_(fileConfig),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  tableHandle_ = checkedPointerCast<const FileTableHandle>(tableHandle);

  folly::F14FastMap<std::string_view, const FileColumnHandle*> columnHandles;
  // Column handled keyed on the column alias, the name used in the query.
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const FileColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    if (!unique) {
      // This should not happen normally, but there are cases where we get
      // duplicate assignments for partitioning columns.
      checkColumnHandleConsistent(*handle, *it->second);
      VELOX_CHECK_EQ(
          handle->columnType(),
          FileColumnHandle::ColumnType::kPartitionKey,
          "Cannot map from same table column to different outputs in table scan; a project node should be used instead: {}",
          handle->name());
      continue;
    }
    processColumnHandle(handle);
  }
  for (auto& handle : tableHandle_->filterColumnHandles()) {
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

    auto* handle = static_cast<const FileColumnHandle*>(it->second.get());
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

  if (fileConfig_->isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx->sessionProperties())) {
    checkColumnNameLowerCase(outputType_);
    checkColumnNameLowerCase(tableHandle_->subfieldFilters(), infoColumns_);
    checkColumnNameLowerCase(tableHandle_->remainingFilter());
  }

  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    filters_.emplace(k.clone(), v);
  }
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter = extractFiltersFromRemainingFilter(
      tableHandle_->remainingFilter(),
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
      tableHandle_->dataColumns(),
      partitionKeys_,
      infoColumns_,
      specialColumns_,
      fileConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);
  if (remainingFilter) {
    metadataFilter_ = std::make_shared<common::MetadataFilter>(
        *scanSpec_, *remainingFilter, expressionEvaluator_);
  }

  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<IoStats>();
}

std::unique_ptr<FileSplitReader> FileDataSource::createSplitReader() {
  return FileSplitReader::create(
      split_,
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
      /*subfieldFiltersForValidation=*/&filters_);
}

void FileDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK_NULL(
      split_,
      "Previous split has not been processed yet. Call next to process the split.");
  split_ = checkedPointerCast<FileConnectorSplit>(split);

  VLOG(1) << "Adding split " << split_->toString();

  if (splitReader_) {
    splitReader_.reset();
  }

  splitReader_ = createSplitReader();

  // Split reader subclasses may need to use the reader options in prepareSplit
  // so we initialize it beforehand.
  splitReader_->configureReaderOptions(randomSkip_);
  splitReader_->prepareSplit(metadataFilter_, runtimeStats_);
  readerOutputType_ = splitReader_->readerOutputType();
}

std::optional<RowVectorPtr> FileDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK(split_ != nullptr, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(splitReader_, "No split reader present");

  TestValue::adjust(
      "facebook::velox::connector::hive::FileDataSource::next", this);

  if (splitReader_->emptySplit()) {
    resetSplit();
    return nullptr;
  }

  // Subclass reader may add extra columns to reader output (e.g. for bucket
  // conversion or delta update).
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

void FileDataSource::addDynamicFilter(
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
FileDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  res.insert(
      {std::string(Connector::kIoWaitWallNanos),
       RuntimeMetric(
           saturateCast(ioStatistics_->queryThreadIoLatencyUs().sum() * 1'000),
           ioStatistics_->queryThreadIoLatencyUs().count(),
           saturateCast(ioStatistics_->queryThreadIoLatencyUs().min() * 1'000),
           saturateCast(ioStatistics_->queryThreadIoLatencyUs().max() * 1'000),
           RuntimeCounter::Unit::kNanos)});
  // Breakdown of ioWaitWallNanos by I/O type
  if (ioStatistics_->storageReadLatencyUs().count() > 0) {
    res.insert(
        {std::string(Connector::kStorageReadWallNanos),
         RuntimeMetric(
             saturateCast(ioStatistics_->storageReadLatencyUs().sum() * 1'000),
             ioStatistics_->storageReadLatencyUs().count(),
             saturateCast(ioStatistics_->storageReadLatencyUs().min() * 1'000),
             saturateCast(ioStatistics_->storageReadLatencyUs().max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStatistics_->ssdCacheReadLatencyUs().count() > 0) {
    res.insert(
        {std::string(Connector::kSsdCacheReadWallNanos),
         RuntimeMetric(
             saturateCast(ioStatistics_->ssdCacheReadLatencyUs().sum() * 1'000),
             ioStatistics_->ssdCacheReadLatencyUs().count(),
             saturateCast(ioStatistics_->ssdCacheReadLatencyUs().min() * 1'000),
             saturateCast(ioStatistics_->ssdCacheReadLatencyUs().max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStatistics_->cacheWaitLatencyUs().count() > 0) {
    res.insert(
        {std::string(Connector::kCacheWaitWallNanos),
         RuntimeMetric(
             saturateCast(ioStatistics_->cacheWaitLatencyUs().sum() * 1'000),
             ioStatistics_->cacheWaitLatencyUs().count(),
             saturateCast(ioStatistics_->cacheWaitLatencyUs().min() * 1'000),
             saturateCast(ioStatistics_->cacheWaitLatencyUs().max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStatistics_->coalescedSsdLoadLatencyUs().count() > 0) {
    res.insert(
        {std::string(Connector::kCoalescedSsdLoadWallNanos),
         RuntimeMetric(
             saturateCast(
                 ioStatistics_->coalescedSsdLoadLatencyUs().sum() * 1'000),
             ioStatistics_->coalescedSsdLoadLatencyUs().count(),
             saturateCast(
                 ioStatistics_->coalescedSsdLoadLatencyUs().min() * 1'000),
             saturateCast(
                 ioStatistics_->coalescedSsdLoadLatencyUs().max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
  if (ioStatistics_->coalescedStorageLoadLatencyUs().count() > 0) {
    res.insert(
        {std::string(Connector::kCoalescedStorageLoadWallNanos),
         RuntimeMetric(
             saturateCast(
                 ioStatistics_->coalescedStorageLoadLatencyUs().sum() * 1'000),
             ioStatistics_->coalescedStorageLoadLatencyUs().count(),
             saturateCast(
                 ioStatistics_->coalescedStorageLoadLatencyUs().min() * 1'000),
             saturateCast(
                 ioStatistics_->coalescedStorageLoadLatencyUs().max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
  res.insert(
      {{std::string(kNumPrefetch),
        RuntimeMetric(ioStatistics_->prefetch().count())},
       {std::string(kPrefetchBytes),
        RuntimeMetric(
            saturateCast(ioStatistics_->prefetch().sum()),
            ioStatistics_->prefetch().count(),
            saturateCast(ioStatistics_->prefetch().min()),
            saturateCast(ioStatistics_->prefetch().max()),
            RuntimeCounter::Unit::kBytes)},
       {std::string(kTotalScanTime),
        RuntimeMetric(
            ioStatistics_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
       {std::string(Connector::kTotalRemainingFilterTime),
        RuntimeMetric(
            totalRemainingFilterTime_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos)},
       {Connector::kTotalRemainingFilterCpuTime,
        RuntimeMetric(
            totalRemainingFilterCpuTime_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos)},
       {std::string(kOverreadBytes),
        RuntimeMetric(
            ioStatistics_->rawOverreadBytes(), RuntimeCounter::Unit::kBytes)}});
  if (ioStatistics_->read().count() > 0) {
    res.insert(
        {std::string(kStorageReadBytes),
         RuntimeMetric(
             saturateCast(ioStatistics_->read().sum()),
             ioStatistics_->read().count(),
             saturateCast(ioStatistics_->read().min()),
             saturateCast(ioStatistics_->read().max()),
             RuntimeCounter::Unit::kBytes)});
  }
  if (ioStatistics_->ssdRead().count() > 0) {
    res.insert(
        {std::string(kNumLocalRead),
         RuntimeMetric(ioStatistics_->ssdRead().count())});
    res.insert(
        {std::string(kLocalReadBytes),
         RuntimeMetric(
             saturateCast(ioStatistics_->ssdRead().sum()),
             ioStatistics_->ssdRead().count(),
             saturateCast(ioStatistics_->ssdRead().min()),
             saturateCast(ioStatistics_->ssdRead().max()),
             RuntimeCounter::Unit::kBytes)});
  }
  if (ioStatistics_->ramHit().count() > 0) {
    res.insert(
        {std::string(kNumRamRead),
         RuntimeMetric(ioStatistics_->ramHit().count())});
    res.insert(
        {std::string(kRamReadBytes),
         RuntimeMetric(
             saturateCast(ioStatistics_->ramHit().sum()),
             ioStatistics_->ramHit().count(),
             saturateCast(ioStatistics_->ramHit().min()),
             saturateCast(ioStatistics_->ramHit().max()),
             RuntimeCounter::Unit::kBytes)});
  }

  const auto ioStatsMap = ioStats_->stats();
  for (const auto& [key, value] : ioStatsMap) {
    // IoStats may carry a ReadFile-layer storageReadBytes that reflects the
    // actual bytes fetched from remote storage. Use it to override the
    // DWIO-level estimate (IoStatistics).
    if (key == kStorageReadBytes) {
      res[std::string(key)] = value;
    } else {
      res.emplace(key, value);
    }
  }
  return res;
}

void FileDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto source = dynamic_cast<FileDataSource*>(sourceUnique.get());
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
  source->ioStatistics_->merge(*ioStatistics_);
  ioStatistics_ = std::move(source->ioStatistics_);
  source->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(source->ioStats_);
}

int64_t FileDataSource::estimatedRowSize() {
  if (splitReader_ == nullptr) {
    return kUnknownRowSize;
  }
  auto rowSize = splitReader_->estimatedRowSize();
  TestValue::adjust(
      "facebook::velox::connector::hive::FileDataSource::estimatedRowSize",
      &rowSize);
  return rowSize;
}

vector_size_t FileDataSource::evaluateRemainingFilter(RowVectorPtr& rowVector) {
  for (auto fieldIndex : multiReferencedFields_) {
    LazyVector::ensureLoadedRows(
        rowVector->childAt(fieldIndex),
        filterRows_,
        filterLazyDecoded_,
        filterLazyBaseRows_);
  }
  CpuWallTiming filterTiming;
  vector_size_t rowsRemaining{0};
  {
    CpuWallTimer timer(filterTiming);
    expressionEvaluator_->evaluate(
        remainingFilterExprSet_.get(), filterRows_, *rowVector, filterResult_);
    rowsRemaining = exec::processFilterResults(
        filterResult_, filterRows_, filterEvalCtx_, pool_);
  }
  totalRemainingFilterTime_.fetch_add(
      filterTiming.wallNanos, std::memory_order_relaxed);
  totalRemainingFilterCpuTime_.fetch_add(
      filterTiming.cpuNanos, std::memory_order_relaxed);
  return rowsRemaining;
}

void FileDataSource::resetSplit() {
  split_.reset();
  splitReader_->resetSplit();
  // Keep readers around to hold adaptation.
}

} // namespace facebook::velox::connector::hive
