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

#include <folly/ScopeGuard.h>
#include <string>
#include <unordered_map>

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileSplitReaderAdapter.h"

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
    FileScanOptions options)
    : FileDataSource(
          std::make_shared<const FileScanSpec>(
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
    std::shared_ptr<const FileScanSpec> fileScanSpec,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<FileConfig>& fileConfig)
    : fileHandleFactory_(fileHandleFactory),
      ioExecutor_(ioExecutor),
      connectorQueryCtx_(connectorQueryCtx),
      fileConfig_(fileConfig),
      pool_(connectorQueryCtx->memoryPool()),
      fileScanSpec_(std::move(fileScanSpec)),
      tableHandle_(fileScanSpec_->tableHandle()),
      readerOutputType_(fileScanSpec_->readerOutputType()),
      partitionKeys_(fileScanSpec_->partitionKeys()),
      infoColumns_(fileScanSpec_->infoColumns()),
      specialColumns_(fileScanSpec_->specialColumns()),
      subfields_(fileScanSpec_->subfields()),
      filters_(fileScanSpec_->filters()),
      extractionColumns_(fileScanSpec_->extractionColumns()),
      outputType_(fileScanSpec_->outputType()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      columnPostProcessors_(fileScanSpec_->columnPostProcessors()),
      multiReferencedFields_(fileScanSpec_->multiReferencedFields()),
      remainingFilterColumns_(fileScanSpec_->remainingFilterColumns()) {
  if (fileScanSpec_->remainingFilter()) {
    remainingFilterExprSet_ =
        expressionEvaluator_->compile(fileScanSpec_->remainingFilter());
  }
  if (fileScanSpec_->sampleRate() != 1) {
    randomSkip_ = std::make_shared<random::RandomSkipTracker>(
        fileScanSpec_->sampleRate());
  }
  resetScanSpec();
  dataIoStats_ = std::make_shared<io::IoStatistics>();
  metadataIoStats_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<IoStats>();
}

void FileDataSource::resetScanSpec() {
  auto state = std::make_shared<FileScanState>(fileScanSpec_->newFileScanState(
      readerOutputType_, subfields_, filters_, connectorQueryCtx_));
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

std::unique_ptr<FileSplitReader> FileDataSource::createSplitReader() {
  return FileSplitReader::create(
      split_,
      tableHandle_,
      &fileScanSpec_->partitionKeys(),
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

std::unique_ptr<FileScanReader> FileDataSource::createScanReader() {
  split_ = checkedPointerCast<FileConnectorSplit>(activeSplit_);
  // Start from logical demands; a previous file may have added reader-only
  // columns, changed pruning, or installed constants for absent fields.
  readerOutputType_ = fileScanSpec_->readerOutputType();
  subfields_ = fileScanSpec_->subfields();
  extractionColumns_ = fileScanSpec_->extractionColumns();
  resetScanSpec();
  auto physicalReader = createSplitReader();
  auto reader = std::make_unique<FileSplitReaderAdapter>(
      std::move(physicalReader), split_, pool_, fileScanSpec_, fileScanState_);
  reader->prepare(
      randomSkip_,
      remainingFilterColumns_,
      metadataFilter_,
      readerProducedType_);
  readerOutputType_ = reader->readerOutputType();
  return reader;
}

FileDataSource::~FileDataSource() {
  // Do not collect statistics from a destructor. Cancellation also releases
  // pending work for callers that do not explicitly call DataSource::cancel().
  if (scanReader_) {
    scanReader_->cancel();
  }
}

void FileDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK_NULL(
      activeSplit_,
      "Previous split has not been processed yet. Call next to process the split.");
  VELOX_CHECK_NOT_NULL(split);
  activeSplit_ = std::move(split);
  try {
    VLOG(1) << "Adding split " << activeSplit_->toString();
    scanReader_ = createScanReader();
    VELOX_CHECK_NOT_NULL(scanReader_);
  } catch (...) {
    resetSplit();
    throw;
  }
}

std::optional<RowVectorPtr> FileDataSource::next(
    uint64_t size,
    ContinueFuture& future) {
  try {
    return nextImpl(size, future);
  } catch (...) {
    // Preserve the original read/validation error even if taking the final
    // statistics snapshot fails. resetSplit releases resources in either case.
    try {
      resetSplit();
    } catch (...) {
    }
    throw;
  }
}

std::optional<RowVectorPtr> FileDataSource::nextImpl(
    uint64_t size,
    ContinueFuture& future) {
  VELOX_CHECK_NOT_NULL(
      activeSplit_, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(scanReader_, "No scan reader present");
  VELOX_CHECK_GT(size, 0);

  TestValue::adjust(
      "facebook::velox::connector::hive::FileDataSource::next", this);

  // A previous ready future must not hide a reader that fails to supply one.
  future = ContinueFuture::makeEmpty();
  const auto result = scanReader_->next(size, output_, future);
  VELOX_CHECK_LE(
      result.physicalRowsScanned,
      std::numeric_limits<uint64_t>::max() - completedRows_,
      "Physical row count overflow");
  completedRows_ += result.physicalRowsScanned;
  switch (result.state) {
    case ScanReadResult::State::kBlocked:
      VELOX_CHECK(future.valid(), "Blocked scan reader must provide a future");
      return std::nullopt;
    case ScanReadResult::State::kEnd:
      VELOX_CHECK(!future.valid(), "Finished scan reader returned a future");
      resetSplit();
      return nullptr;
    case ScanReadResult::State::kData:
      VELOX_CHECK(!future.valid(), "Scan reader returned data and a future");
      break;
    default:
      VELOX_FAIL("Invalid scan reader state");
  }
  VELOX_CHECK_NOT_NULL(output_, "Scan reader returned data without a vector");
  VELOX_CHECK_NOT_NULL(output_->as<RowVector>(), "Expected a row vector");
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
    return std::make_shared<RowVector>(
        pool_, outputType_, nullptr, rowsRemaining, std::vector<VectorPtr>{});
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
  fieldSpec.setFilter(filter);
  scanSpec_->resetCachedValues(true);
  if (scanReader_) {
    scanReader_->resetFilterCaches();
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
  if (const auto* file =
          scanReader_ ? scanReader_->currentFileSplit() : nullptr) {
    fileEvent.filePath = file->filePath;
    fileEvent.fileFormat = file->fileFormat;
    if (!file->partitionKeys.empty()) {
      fileEvent.partitionKeys = &file->partitionKeys;
    }
  }
  scanBatchCallback_(fileEvent);
}

std::unordered_map<std::string, RuntimeMetric>
FileDataSource::getRuntimeStats() {
  auto stats = readerStats_;
  if (scanReader_) {
    stats.mergeFrom(scanReader_->getRuntimeStats());
  }
  auto res = stats.toRuntimeMetricMap();
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

  VELOX_CHECK_NULL(activeSplit_, "Cannot replace an active split");
  VELOX_CHECK_NOT_NULL(source->scanReader_);
  VELOX_CHECK_LE(
      source->completedRows_,
      std::numeric_limits<uint64_t>::max() - completedRows_,
      "Physical row count overflow");
  // Check support before moving any state out of the source.
  source->scanReader_->setConnectorQueryCtx(connectorQueryCtx_);
  activeSplit_ = std::move(source->activeSplit_);
  split_ = std::move(source->split_);
  readerStats_.mergeFrom(source->readerStats_);
  completedRows_ += source->completedRows_;
  lastEventStorageReadBytes_ += source->lastEventStorageReadBytes_;
  totalRemainingFilterTime_.fetch_add(
      source->totalRemainingFilterTime_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  totalRemainingFilterCpuTime_.fetch_add(
      source->totalRemainingFilterCpuTime_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  output_ = std::move(source->output_);
  readerOutputType_ = std::move(source->readerOutputType_);
  readerProducedType_ = std::move(source->readerProducedType_);
  // Immutable column demands retain the destination specification's ownership.
  source->scanSpec_->moveAdaptationFrom(*scanSpec_);
  scanSpec_ = std::move(source->scanSpec_);
  fileScanState_ = std::move(source->fileScanState_);
  for (const auto& [channel, filter] : source->dynamicFilters_) {
    if (auto it = dynamicFilters_.find(channel); it != dynamicFilters_.end()) {
      it->second = it->second->mergeWith(filter.get());
    } else {
      dynamicFilters_.emplace(channel, filter->clone());
    }
  }
  applyDynamicFilters();
  metadataFilter_ = std::move(source->metadataFilter_);
  scanReader_ = std::move(source->scanReader_);
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
  if (scanReader_ == nullptr) {
    return kUnknownRowSize;
  }
  auto rowSize = scanReader_->estimatedRowSize();
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
  SCOPE_EXIT {
    if (scanReader_) {
      scanReader_->cancel();
      scanReader_.reset();
    }
    output_.reset();
    split_.reset();
    activeSplit_.reset();
  };
  if (scanReader_) {
    readerStats_.mergeFrom(scanReader_->getRuntimeStats());
  }
}

void FileDataSource::cancel() {
  resetSplit();
  filterResult_.reset();
}

} // namespace facebook::velox::connector::hive
