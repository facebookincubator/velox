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

#include <string>
#include <unordered_map>

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileScanState.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector::hive {

namespace {

void addOperationStatsToRuntimeStats(
    io::IoStatistics& ioStats,
    std::unordered_map<std::string, RuntimeMetric>& res) {
  for (const auto& [operation, counters] : ioStats.operationStats()) {
    // Capturing a structured binding is legal in C++20, but clang-15 predates
    // P1091 and rejects it, and the OSS Ubuntu debug job builds with clang-15.
    const auto& operationName = operation;
    const auto add = [&](std::string_view counter, uint64_t value) {
      if (value == 0) {
        return;
      }
      res[fmt::format("storage.{}.{}", operationName, counter)] =
          RuntimeMetric(value, RuntimeCounter::Unit::kNone);
    };
    add("requestCount", counters.requestCount);
    add("localThrottleCount", counters.localThrottleCount);
    add("globalThrottleCount", counters.globalThrottleCount);
    add("resourceThrottleCount", counters.resourceThrottleCount);
    add("retryCount", counters.retryCount);
    // Cumulative across requests, so consumers must divide by requestCount to
    // recover a per-request mean.
    add("latencyInMs", counters.latencyInMs);
  }
}

} // namespace

FileDataSource::FileDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<FileConfig>& fileConfig,
    FileScanConfig::Options options)
    : FileDataSource(
          std::make_shared<const FileScanConfig>(
              outputType,
              checkedPointerCast<const FileTableHandle>(tableHandle),
              assignments,
              connectorQueryCtx,
              fileConfig,
              options),
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          fileConfig) {}

FileDataSource::FileDataSource(
    std::shared_ptr<const FileScanConfig> fileScanConfig,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<FileConfig>& fileConfig)
    : fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      connectorQueryCtx_(connectorQueryCtx),
      fileConfig_(fileConfig),
      pool_(connectorQueryCtx->memoryPool()),
      fileScanConfig_(std::move(fileScanConfig)),
      tableHandle_(fileScanConfig_->tableHandle()),
      readerOutputType_(fileScanConfig_->readerOutputType()),
      partitionKeys_(fileScanConfig_->partitionKeys()),
      infoColumns_(fileScanConfig_->infoColumns()),
      specialColumns_(fileScanConfig_->specialColumns()),
      subfields_(fileScanConfig_->subfields()),
      filters_(fileScanConfig_->filters()),
      extractionColumns_(fileScanConfig_->extractionColumns()),
      outputType_(fileScanConfig_->outputType()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      columnPostProcessors_(fileScanConfig_->columnPostProcessors()),
      multiReferencedFields_(fileScanConfig_->multiReferencedFields()),
      remainingFilterColumns_(fileScanConfig_->remainingFilterColumns()) {
  if (fileScanConfig_->remainingFilter()) {
    remainingFilterExprSet_ =
        expressionEvaluator_->compile(fileScanConfig_->remainingFilter());
  }
  if (fileScanConfig_->sampleRate() != 1) {
    randomSkip_ = std::make_shared<random::RandomSkipTracker>(
        fileScanConfig_->sampleRate());
  }
  resetScanSpec();
  dataIoStats_ = std::make_shared<io::IoStatistics>();
  metadataIoStats_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<IoStats>();
}

void FileDataSource::resetScanSpec(bool buildMetadataFilter) {
  VELOX_CHECK_NULL(splitReader_, "Cannot rebuild an active reader scan state");
  auto state =
      std::make_shared<FileScanState>(fileScanConfig_->newFileScanState(
          readerOutputType_,
          subfields_,
          filters_,
          connectorQueryCtx_,
          buildMetadataFilter));
  if (scanSpec_) {
    state->scanSpec->moveAdaptationFrom(*scanSpec_);
  }
  fileScanState_ = std::move(state);
  scanSpec_ = fileScanState_->scanSpec;
  metadataFilter_ = fileScanState_->metadataFilter;
  readerProducedType_ = fileScanState_->readerProducedType;
  applyDynamicFilters();
}

void FileDataSource::applyDynamicFilters() {
  for (const auto& [channel, filter] : dynamicFilters_) {
    scanSpec_->getChildByChannel(channel).setFilter(filter->clone());
  }
  scanSpec_->resetCachedValues(true);
}

void FileDataSource::resetOutputIfTypeChanged() {
  const auto& type =
      readerProducedType_ ? readerProducedType_ : readerOutputType_;
  // Different files can need different auxiliary columns at the same
  // channel. Retain reusable output only when both names and types match.
  if (output_ && *output_->type() != *type) {
    output_.reset();
  }
}

std::unique_ptr<FileSplitReader> FileDataSource::createSplitReader() {
  return FileSplitReader::create(
      split_,
      tableHandle_,
      &fileScanConfig_->partitionKeys(),
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_,
      /*subfieldFiltersForValidation=*/&fileScanState_->filters);
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

  // Start from logical demands. The previous file may have added reader-only
  // columns or installed constants for missing fields and partition keys.
  readerOutputType_ = fileScanConfig_->readerOutputType();
  subfields_ = fileScanConfig_->subfields();
  extractionColumns_ = fileScanConfig_->extractionColumns();
  resetScanSpec();
  splitReader_ = createSplitReader();

  // Split reader subclasses may need to use the reader options in prepareSplit
  // so we initialize it beforehand.
  splitReader_->configureReaderOptions(randomSkip_);
  splitReader_->setRemainingFilterColumns(remainingFilterColumns_);
  splitReader_->prepareSplit(metadataFilter_, runtimeStats_);

  auto splitReaderOutputType = splitReader_->readerOutputType();
  if (readerProducedType_ != nullptr &&
      splitReaderOutputType->size() > readerProducedType_->size()) {
    // The split reader appended columns, e.g. an unselected Iceberg
    // equality-delete or lineage column. next() allocates the output from
    // 'readerProducedType_', so it has to grow by the same columns.
    auto names = readerProducedType_->names();
    auto types = readerProducedType_->children();
    for (auto i = readerProducedType_->size();
         i < splitReaderOutputType->size();
         ++i) {
      names.push_back(splitReaderOutputType->nameOf(i));
      types.push_back(splitReaderOutputType->childAt(i));
    }
    readerProducedType_ = ROW(std::move(names), std::move(types));
  }
  readerOutputType_ = std::move(splitReaderOutputType);
  resetOutputIfTypeChanged();
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
  auto& outputRowType =
      readerProducedType_ ? readerProducedType_ : readerOutputType_;
  auto needsExtraColumn = [&] {
    return output_->asUnchecked<RowVector>()->childrenSize() <
        outputRowType->size();
  };
  if (!output_ || needsExtraColumn()) {
    output_ = BaseVector::create(outputRowType, 0, pool_);
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
    // Bound by the filter input, not rowsScanned: merge-on-read splits
    // legitimately emit more rows than scanned when deltas insert.
    VELOX_CHECK_LE(rowsRemaining, rowVector->size());
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
  dynamicFilters_[outputChannel] = filter->clone();
  auto& fieldSpec = scanSpec_->getChildByChannel(outputChannel);
  fieldSpec.setFilter(filter->clone());
  scanSpec_->resetCachedValues(true);
  if (splitReader_) {
    splitReader_->resetFilterCaches();
  }
}

void FileDataSource::fireScanBatchCallback(core::ScanBatchEvent event) {
  // Bytes are read when the reader loads a stripe, which for small files is
  // entirely inside addSplit() and for large ones is spread across next()
  // calls. Reporting the delta since the previous event captures them either
  // way; a window around a single next() would not.
  const uint64_t totalStorageReadBytes = dataIoStats_->read().sum();
  const uint64_t storageReadBytesDelta =
      totalStorageReadBytes - lastEventStorageReadBytes_;
  lastEventStorageReadBytes_ = totalStorageReadBytes;
  if (!scanBatchCallback_) {
    return;
  }
  FileScanBatchEvent fileEvent;
  fileEvent.numRows = event.numRows;
  fileEvent.wallTimeMicros = event.wallTimeMicros;
  fileEvent.planNodeId = event.planNodeId;
  fileEvent.storageReadBytes = storageReadBytesDelta;
  if (tableHandle_) {
    fileEvent.tableName = tableHandle_->name();
    fileEvent.dbName = tableHandle_->dbName();
  }
  if (split_) {
    fileEvent.filePath = split_->filePath;
    fileEvent.fileFormat = split_->fileFormat;
    if (!split_->partitionKeys.empty()) {
      fileEvent.partitionKeys = &split_->partitionKeys;
    }
  }
  scanBatchCallback_(fileEvent);
}

std::unordered_map<std::string, RuntimeMetric>
FileDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  io::addIoStatsToRuntimeStats(*dataIoStats_, "", res);
  io::addIoStatsToRuntimeStats(*metadataIoStats_, kMetadataPrefix, res);
  res.insert(
      {{std::string(Connector::kTotalRemainingFilterTime),
        RuntimeMetric(
            totalRemainingFilterTime_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos)},
       {Connector::kTotalRemainingFilterCpuTime,
        RuntimeMetric(
            totalRemainingFilterCpuTime_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos)}});

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

  addOperationStatsToRuntimeStats(*dataIoStats_, res);
  return res;
}

void FileDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto source = dynamic_cast<FileDataSource*>(sourceUnique.get());
  VELOX_CHECK_NOT_NULL(source, "Bad DataSource type");

  // Release the previous reader before replacing any of its borrowed inputs.
  splitReader_.reset();
  split_ = std::move(source->split_);
  runtimeStats_.skippedSplits += source->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits += source->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes += source->runtimeStats_.skippedSplitBytes;
  readerOutputType_ = std::move(source->readerOutputType_);
  readerProducedType_ = std::move(source->readerProducedType_);
  resetOutputIfTypeChanged();
  extractionColumns_ = std::move(source->extractionColumns_);
  source->scanSpec_->moveAdaptationFrom(*scanSpec_);
  // The reader borrows partition/info handles from the configuration and
  // validation filters from the state. Keep both owners after source dies.
  fileScanConfig_ = std::move(source->fileScanConfig_);
  subfields_ = std::move(source->subfields_);
  fileScanState_ = std::move(source->fileScanState_);
  scanSpec_ = std::move(source->scanSpec_);
  metadataFilter_ = std::move(source->metadataFilter_);
  splitReader_ = std::move(source->splitReader_);
  splitReader_->setConnectorQueryCtx(connectorQueryCtx_);
  // New io will be accounted on the stats of 'source'. Add the existing
  // balance to that.
  source->dataIoStats_->merge(*dataIoStats_);
  dataIoStats_ = std::move(source->dataIoStats_);
  source->metadataIoStats_->merge(*metadataIoStats_);
  metadataIoStats_ = std::move(source->metadataIoStats_);
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
