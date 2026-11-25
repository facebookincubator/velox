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

#include "velox/connectors/common/FileDataSource.h"

#include <fmt/ranges.h>
#include <string>
#include <unordered_map>

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/connectors/common/ExtractionUtils.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/expression/FieldReference.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector {

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
    case FileColumnHandle::ColumnType::kReadRange:
      specialColumns_.readRange = handle->name();
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
      fileScanState_(std::make_unique<FileScanState>()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  tableHandle_ = checkedPointerCast<const FileTableHandle>(tableHandle);

  folly::F14FastMap<std::string_view, const FileColumnHandle*> columnHandles;
  // Column handles keyed on the table column name.
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const FileColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    if (!unique) {
      // This should not happen normally, but there are cases where we get
      // duplicate assignments for partitioning columns.
      handle->checkConsistentWith(*it->second);
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
      handle->checkConsistentWith(*it->second);
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
          subfield.baseName(),
          handle->name(),
          "Required subfield does not match column name");
      subfields_[handle->name()].push_back(&subfield);
    }
    columnPostProcessors_.push_back(handle->postProcessor());
  }

  if (fileConfig_->isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx->sessionProperties())) {
    tableHandle_->checkColumnNamesLowerCase(*outputType_);
  }

  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    fileScanState_->filters.emplace(k.clone(), v);
  }
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter = exec::ExprToSubfieldFilterParser::extractFilters(
      tableHandle_->remainingFilter(),
      expressionEvaluator_,
      fileScanState_->filters,
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
    // Capture top-level column names referenced by the remaining filter.
    // These columns must be loaded eagerly (not lazily) so the filter
    // can evaluate before lazy columns are accessed.
    folly::F14FastSet<std::string> remainingFilterColumns;
    const auto fieldsReadForAllPassingRows =
        remainingFilterExpr->fieldsReadForAllPassingRows();
    for (auto& input : remainingFilterExpr->distinctFields()) {
      remainingFilterColumns.insert(input->field());
      auto it = columnNames.find(input->field());
      if (it != columnNames.end()) {
        if (std::find(
                fieldsReadForAllPassingRows.begin(),
                fieldsReadForAllPassingRows.end(),
                input) == fieldsReadForAllPassingRows.end()) {
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
    remainingFilterColumns_ = std::move(remainingFilterColumns);
    remainingFilterSubfields_ = remainingFilterExpr->extractSubfields();
    if (VLOG_IS_ON(1)) {
      VLOG(1) << fmt::format(
          "Extracted subfields from remaining filter: [{}]",
          fmt::join(remainingFilterSubfields_, ", "));
    }
    for (auto& subfield : remainingFilterSubfields_) {
      const auto& name = subfield.baseName();
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
  fileScanState_->scanSpec = makeScanSpec(
      readerOutputType_,
      subfields_,
      fileScanState_->filters,
      tableHandle_->dataColumns(),
      partitionKeys_,
      infoColumns_,
      specialColumns_,
      fileConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);
  if (remainingFilter) {
    fileScanState_->metadataFilter = std::make_shared<common::MetadataFilter>(
        *fileScanState_->scanSpec, *remainingFilter, expressionEvaluator_);
  }

  // Detect extraction columns and reconfigure the ScanSpec if needed.
  bool hasExtractions = false;
  readColumnTypes = readerOutputType_->children();
  for (int outputIdx = 0; outputIdx < outputType->size(); ++outputIdx) {
    const auto& outputName = outputType->nameOf(outputIdx);
    auto it = assignments.find(outputName);
    if (it == assignments.end()) {
      continue;
    }
    auto* handle = static_cast<const FileColumnHandle*>(it->second.get());
    if (!handle->extractions().empty()) {
      // Column has extraction chains.  Read with schemaType from file, then
      // apply extraction post-read.  Extractions and requiredSubfields are
      // mutually exclusive (enforced by the column handle constructor).
      auto readerIdx = readerOutputType_->getChildIdxIfExists(handle->name());
      if (readerIdx.has_value()) {
        readColumnTypes[*readerIdx] = handle->schemaType();
        extractionColumns_[*readerIdx] = handle;
        hasExtractions = true;
      }
    }
  }

  if (hasExtractions) {
    // Rebuild readerOutputType_ with schemaType for extraction columns.
    readerOutputType_ =
        ROW(std::vector<std::string>(
                readerOutputType_->names().begin(),
                readerOutputType_->names().end()),
            std::move(readColumnTypes));
    // Rebuild the ScanSpec with the updated readerOutputType_.
    fileScanState_->scanSpec = makeScanSpec(
        readerOutputType_,
        subfields_,
        fileScanState_->filters,
        tableHandle_->dataColumns(),
        partitionKeys_,
        infoColumns_,
        specialColumns_,
        fileConfig_->readStatsBasedFilterReorderDisabled(
            connectorQueryCtx->sessionProperties()),
        pool_);
    configureExtractionColumns();
  }

  dataIoStats_ = std::make_shared<io::IoStatistics>();
  metadataIoStats_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<IoStats>();
}

void FileDataSource::configureExtractionColumns() {
  // Configure extraction columns on the ScanSpec.  For each column with
  // extractions, this:
  // 1. Sets pruning hints so DWRF/Nimble readers skip unneeded sub-streams.
  // 2. Sets a transform function on the ScanSpec node so the reader applies
  //    extraction chains and produces the output type directly.
  for (auto& [colIdx, handle] : extractionColumns_) {
    auto* fieldSpec = fileScanState_->scanSpec->childByName(
        readerOutputType_->nameOf(colIdx));
    if (!fieldSpec) {
      continue;
    }
    const auto& extractions = handle->extractions();
    auto extractionOutputType = handle->dataType();

    // For multiple extractions, do NOT call configureExtractionScanSpec --
    // keep ExtractionType as kNone and use full chains in the transform.
    // This ensures the text reader (which does not handle ExtractionType
    // natively) produces correct results.
    if (extractions.size() == 1) {
      configureExtractionScanSpec(
          handle->schemaType(), extractions, *fieldSpec, pool_);
    }
    if (extractions.size() == 1) {
      // Store a full-chain transform so hasTransform() returns true.  This
      // signals to the delta update path that extraction is configured.
      // The full chain is captured for PrismSplitReader to replace it.
      fieldSpec->setTransform(
          [fullChain = extractions[0].chain](
              const VectorPtr& input, memory::MemoryPool* pool) -> VectorPtr {
            return applyExtractionChain(input, fullChain, pool);
          },
          extractionOutputType);
    } else {
      // Multiple extractions: do NOT set ExtractionType on the ScanSpec.
      // Use full chains in the transform so the text reader (which does
      // not handle ExtractionType natively) produces correct results.
      // TODO: Optimization: for agreeing multiple extractions, set
      // ExtractionType and use remaining chains.  Requires text reader
      // to handle ExtractionType natively.
      struct ExtractionInfo {
        std::string outputName;
        std::vector<ExtractionPathElementPtr> chain;
      };

      std::vector<ExtractionInfo> infos;
      for (const auto& extraction : extractions) {
        infos.push_back({extraction.outputName, extraction.chain});
      }
      // Always need a transform for multiple extractions to assemble ROW.
      fieldSpec->setTransform(
          [infos = std::move(infos)](
              const VectorPtr& input, memory::MemoryPool* pool) -> VectorPtr {
            std::vector<VectorPtr> children;
            std::vector<std::string> names;
            std::vector<TypePtr> types;
            children.reserve(infos.size());
            names.reserve(infos.size());
            types.reserve(infos.size());
            for (const auto& info : infos) {
              VectorPtr extracted;
              if (info.chain.empty()) {
                extracted = input;
              } else {
                extracted = applyExtractionChain(input, info.chain, pool);
              }
              names.push_back(info.outputName);
              types.push_back(extracted->type());
              children.push_back(std::move(extracted));
            }
            return std::make_shared<RowVector>(
                pool,
                ROW(std::move(names), std::move(types)),
                nullptr,
                input->size(),
                std::move(children));
          },
          extractionOutputType);
    }
  }

  // Build the state's readerProducedType: the actual type the reader produces.
  // For extraction columns where the reader handles extraction natively
  // (ExtractionType != kNone), the output type differs from schemaType.
  {
    auto names = readerOutputType_->names();
    auto types = readerOutputType_->children();
    bool needsSeparateType = false;
    for (auto& [colIdx, handle] : extractionColumns_) {
      auto* fieldSpec = fileScanState_->scanSpec->childByName(
          readerOutputType_->nameOf(colIdx));
      if (fieldSpec &&
          fieldSpec->extractionType() !=
              common::ScanSpec::ExtractionType::kNone) {
        VELOX_CHECK_LT(static_cast<size_t>(colIdx), types.size());
        types[colIdx] = handle->dataType();
        needsSeparateType = true;
      }
    }
    if (needsSeparateType) {
      fileScanState_->readerProducedType =
          ROW(std::vector<std::string>(names.begin(), names.end()),
              std::move(types));
    }
  }
}

std::unique_ptr<FileSplitReader> FileDataSource::createSplitReader() {
  return FileSplitReader::create(
      split_,
      tableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      fileScanState_->scanSpec,
      /*subfieldFiltersForValidation=*/&fileScanState_->filters);
}

void FileDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK_NULL(
      split_,
      "Previous split has not been processed yet. Call next to process the split.");
  split_ = checkedPointerCast<FileConnectorSplit>(split);

  VLOG(1) << "Adding split " << split_->toString();

  splitReader_.reset();

  if (skipSplit()) {
    ++runtimeStats_.skippedSplits;
    runtimeStats_.skippedSplitBytes += split_->length;
    return;
  }

  splitReader_ = createSplitReader();

  // Split reader subclasses may need to use the reader options in prepareSplit
  // so we initialize it beforehand.
  splitReader_->configureReaderOptions(randomSkip_);
  splitReader_->setRemainingFilterColumns(remainingFilterColumns_);
  splitReader_->prepareSplit(
      fileScanState_->metadataFilter, runtimeStats_, fileReadOps_);

  auto splitReaderOutputType = splitReader_->readerOutputType();
  if (fileScanState_->readerProducedType != nullptr &&
      splitReaderOutputType->size() >
          fileScanState_->readerProducedType->size()) {
    // The split reader appended columns, e.g. an unselected Iceberg
    // equality-delete or lineage column. next() allocates the output from
    // the state's readerProducedType, so it must grow by the same columns.
    auto names = fileScanState_->readerProducedType->names();
    auto types = fileScanState_->readerProducedType->children();
    for (auto i = fileScanState_->readerProducedType->size();
         i < splitReaderOutputType->size();
         ++i) {
      names.push_back(splitReaderOutputType->nameOf(i));
      types.push_back(splitReaderOutputType->childAt(i));
    }
    fileScanState_->readerProducedType =
        ROW(std::move(names), std::move(types));
  }
  readerOutputType_ = std::move(splitReaderOutputType);
}

std::optional<RowVectorPtr> FileDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK(split_ != nullptr, "No split to process. Call addSplit first.");

  TestValue::adjust("facebook::velox::connector::FileDataSource::next", this);

  // A skipped split has no reader.
  if (splitReader_ == nullptr || splitReader_->emptySplit()) {
    resetSplit();
    return nullptr;
  }

  // Subclass reader may add extra columns to reader output (e.g. for bucket
  // conversion or delta update).
  auto& outputRowType = fileScanState_->readerProducedType
      ? fileScanState_->readerProducedType
      : readerOutputType_;
  auto needsExtraColumn = [&] {
    return output_->asUnchecked<RowVector>()->childrenSize() <
        outputRowType->size();
  };
  if (!output_ || needsExtraColumn()) {
    output_ = BaseVector::create(outputRowType, 0, pool_);
  }

  if (scanBatchCallback_) {
    lastReadStart_ = splitReader_->nextRowNumber();
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
  // or it passes on all rows, leave this as null and let exec::wrapChild skip
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
  auto& fieldSpec = fileScanState_->scanSpec->getChildByChannel(outputChannel);
  fieldSpec.setFilter(filter);
  fileScanState_->scanSpec->resetCachedValues(true);
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
    fileEvent.split = split_.get();
    fileEvent.rowStart = lastReadStart_;
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

  // The old reader may borrow validation filters from the state being replaced.
  splitReader_.reset();
  split_ = std::move(source->split_);
  runtimeStats_.skippedSplits += source->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits += source->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes += source->runtimeStats_.skippedSplitBytes;
  readerOutputType_ = std::move(source->readerOutputType_);
  extractionColumns_ = std::move(source->extractionColumns_);
  source->fileScanState_->scanSpec->moveAdaptationFrom(
      *fileScanState_->scanSpec);
  fileScanState_ = std::move(source->fileScanState_);
  splitReader_ = std::move(source->splitReader_);
  if (splitReader_) {
    splitReader_->setConnectorQueryCtx(connectorQueryCtx_);
  }
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
      "facebook::velox::connector::FileDataSource::estimatedRowSize", &rowSize);
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
  if (splitReader_) {
    splitReader_->resetSplit();
  }
  // Keep readers around to hold adaptation.
}

namespace {

struct SubfieldSpec {
  const common::Subfield* subfield;
  bool filterOnly;
};

template <typename T>
void deduplicate(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Floating point map key subscripts are truncated toward 0 in Presto.  For
// example given `a' as a map with floating point key, if user queries a[0.99],
// Presto coordinator will generate a required subfield a[0]; for a[-1.99] it
// will generate a[-1]; for anything larger than 9223372036854775807, it
// generates a[9223372036854775807]; for anything smaller than
// -9223372036854775808 it generates a[-9223372036854775808].
template <typename T>
std::unique_ptr<common::Filter> makeFloatingPointMapKeyFilter(
    const std::vector<int64_t>& subscripts) {
  std::vector<std::unique_ptr<common::Filter>> filters;
  for (auto subscript : subscripts) {
    T lower = subscript;
    T upper = subscript;
    bool lowerUnbounded = subscript == std::numeric_limits<int64_t>::min();
    bool upperUnbounded = subscript == std::numeric_limits<int64_t>::max();
    bool lowerExclusive = false;
    bool upperExclusive = false;
    if (lower <= 0 && !lowerUnbounded) {
      if (lower > subscript - 1) {
        lower = subscript - 1;
      } else {
        lower = std::nextafter(lower, -std::numeric_limits<T>::infinity());
      }
      lowerExclusive = true;
    }
    if (upper >= 0 && !upperUnbounded) {
      if (upper < subscript + 1) {
        upper = subscript + 1;
      } else {
        upper = std::nextafter(upper, std::numeric_limits<T>::infinity());
      }
      upperExclusive = true;
    }
    if (lowerUnbounded && upperUnbounded) {
      continue;
    }
    filters.push_back(
        std::make_unique<common::FloatingPointRange<T>>(
            lower,
            lowerUnbounded,
            lowerExclusive,
            upper,
            upperUnbounded,
            upperExclusive,
            false));
  }
  if (filters.size() == 1) {
    return std::move(filters[0]);
  }
  return std::make_unique<common::MultiRange>(std::move(filters), false);
}

// Recursively add subfields to scan spec.
void addSubfields(
    const Type& type,
    std::vector<SubfieldSpec>& subfields,
    int level,
    memory::MemoryPool* pool,
    common::ScanSpec& spec) {
  int newSize = 0;
  for (int i = 0; i < subfields.size(); ++i) {
    if (level < subfields[i].subfield->path().size()) {
      subfields[newSize++] = subfields[i];
    } else if (!subfields[i].filterOnly) {
      spec.addAllChildFields(type);
      return;
    }
  }
  subfields.resize(newSize);

  switch (type.kind()) {
    case TypeKind::ROW: {
      folly::F14FastMap<std::string, std::vector<SubfieldSpec>> required;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        auto* nestedField = element->asChecked<common::Subfield::NestedField>();
        required[nestedField->name()].push_back(subfield);
      }
      const auto& rowType = type.asRow();
      for (int i = 0; i < rowType.size(); ++i) {
        auto& childName = rowType.nameOf(i);
        auto& childType = rowType.childAt(i);
        auto* child = spec.addField(childName, i);
        auto it = required.find(childName);
        if (it == required.end()) {
          child->setConstantValue(
              BaseVector::createNullConstant(childType, 1, pool));
        } else {
          addSubfields(*childType, it->second, level + 1, pool, *child);
        }
      }
      break;
    }
    case TypeKind::MAP: {
      auto& keyType = type.childAt(0);
      auto* keys = spec.addMapKeyFieldRecursively(*keyType);
      addSubfields(
          *type.childAt(1),
          subfields,
          level + 1,
          pool,
          *spec.addMapValueField());
      if (subfields.empty()) {
        return;
      }
      bool stringKey = keyType->isVarchar() || keyType->isVarbinary();
      std::vector<std::string> stringSubscripts;
      std::vector<int64_t> longSubscripts;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        if (element->is(common::SubfieldKind::kAllSubscripts)) {
          return;
        }
        if (stringKey) {
          auto* subscript = element->as<common::Subfield::StringSubscript>();
          VELOX_CHECK(
              subscript,
              "Unsupported for string map pruning: {}",
              element->toString());
          stringSubscripts.push_back(subscript->index());
        } else {
          auto* subscript = element->as<common::Subfield::LongSubscript>();
          VELOX_CHECK(
              subscript,
              "Unsupported for long map pruning: {}",
              element->toString());
          longSubscripts.push_back(subscript->index());
        }
      }
      std::unique_ptr<common::Filter> filter;
      if (stringKey) {
        deduplicate(stringSubscripts);
        filter = std::make_unique<common::BytesValues>(stringSubscripts, false);
        spec.setFlatMapFeatureSelection(std::move(stringSubscripts));
      } else {
        deduplicate(longSubscripts);
        if (keyType->isReal()) {
          filter = makeFloatingPointMapKeyFilter<float>(longSubscripts);
        } else if (keyType->isDouble()) {
          filter = makeFloatingPointMapKeyFilter<double>(longSubscripts);
        } else {
          filter = common::createBigintValues(longSubscripts, false);
        }
        std::vector<std::string> features;
        for (auto num : longSubscripts) {
          features.push_back(std::to_string(num));
        }
        spec.setFlatMapFeatureSelection(std::move(features));
      }
      keys->setFilter(std::move(filter));
      break;
    }
    case TypeKind::ARRAY: {
      addSubfields(
          *type.childAt(0),
          subfields,
          level + 1,
          pool,
          *spec.addArrayElementField());
      if (subfields.empty()) {
        return;
      }
      constexpr long kMaxIndex = std::numeric_limits<vector_size_t>::max();
      long maxIndex = -1;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        if (dynamic_cast<const common::Subfield::AllSubscripts*>(element)) {
          return;
        }
        auto* subscript =
            dynamic_cast<const common::Subfield::LongSubscript*>(element);
        VELOX_CHECK(
            subscript,
            "Unsupported for array pruning: {}",
            element->toString());
        VELOX_USER_CHECK_GT(
            subscript->index(),
            0,
            "Non-positive array subscript cannot be push down");
        maxIndex = std::max(maxIndex, std::min(kMaxIndex, subscript->index()));
      }
      spec.setMaxArrayElementsCount(maxIndex);
      break;
    }
    default:
      break;
  }
}

inline bool isSynthesizedColumn(
    const std::string& name,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns) {
  return infoColumns.count(name) != 0;
}

bool isSpecialColumn(
    const std::string& name,
    const std::optional<std::string>& specialName) {
  return specialName.has_value() && name == *specialName;
}

} // namespace

namespace {

void processFieldSpec(
    const RowTypePtr& dataColumns,
    const TypePtr& outputType,
    common::ScanSpec& fieldSpec) {
  fieldSpec.visit(*outputType, [](const Type& type, common::ScanSpec& spec) {
    if (type.isMap() && !spec.isConstant()) {
      auto* keys = spec.childByName(common::ScanSpec::kMapKeysFieldName);
      VELOX_CHECK_NOT_NULL(keys);
      if (keys->filter()) {
        VELOX_CHECK(!keys->filter()->testNull());
      } else {
        keys->setFilter(std::make_shared<common::IsNotNull>());
      }
    }
  });
  if (dataColumns) {
    const auto childIdxOpt =
        dataColumns->getChildIdxIfExists(fieldSpec.fieldName());
    if (childIdxOpt.has_value()) {
      if (dataColumns->childAt(*childIdxOpt)->isMap() && outputType->isRow()) {
        fieldSpec.setFlatMapAsStruct(true);
      }
    }
  }
}

} // namespace

// static
std::shared_ptr<common::ScanSpec> FileDataSource::makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, FileColumnHandlePtr>& partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool) {
  auto spec = std::make_shared<common::ScanSpec>("root");
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      filterSubfields;
  std::vector<SubfieldSpec> subfieldSpecs;
  for (const auto& [subfield, _] : subfieldFilters) {
    if (auto name = subfield.toString();
        !isSynthesizedColumn(name, infoColumns) &&
        partitionKeys.count(name) == 0) {
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.rowIndex));
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.rowId));
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.readRange));
      filterSubfields[subfield.baseName()].push_back(&subfield);
    }
  }

  // Process columns that will be projected out.
  for (int i = 0; i < rowType->size(); ++i) {
    auto& name = rowType->nameOf(i);
    auto& type = rowType->childAt(i);
    if (isSpecialColumn(name, specialColumns.rowIndex)) {
      VELOX_CHECK(type->isBigint());
      auto* fieldSpec = spec->addField(name, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kRowIndex);
      continue;
    }
    if (isSpecialColumn(name, specialColumns.rowId)) {
      VELOX_CHECK(type->isRow() && type->size() == 5);
      auto& rowIdType = type->asRow();
      auto* fieldSpec = spec->addFieldRecursively(name, rowIdType, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kComposite);
      fieldSpec->childByName(rowIdType.nameOf(0))
          ->setColumnType(common::ScanSpec::ColumnType::kRowIndex);
      continue;
    }
    if (isSpecialColumn(name, specialColumns.readRange)) {
      VELOX_USER_CHECK(
          type->kindEquals(ROW({BIGINT(), INTEGER()})),
          "Read range column must be ROW(BIGINT, INTEGER): {}",
          type->toString());
      auto* fieldSpec = spec->addField(name, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kReadRange);
      continue;
    }
    auto it = outputSubfields.find(name);
    if (it == outputSubfields.end()) {
      auto* fieldSpec = spec->addFieldRecursively(name, *type, i);
      processFieldSpec(dataColumns, type, *fieldSpec);
      filterSubfields.erase(name);
      continue;
    }
    for (auto* subfield : it->second) {
      subfieldSpecs.push_back({subfield, false});
    }
    it = filterSubfields.find(name);
    if (it != filterSubfields.end()) {
      for (auto* subfield : it->second) {
        subfieldSpecs.push_back({subfield, true});
      }
      filterSubfields.erase(it);
    }
    auto* fieldSpec = spec->addField(name, i);
    addSubfields(*type, subfieldSpecs, 1, pool, *fieldSpec);
    processFieldSpec(dataColumns, type, *fieldSpec);
    subfieldSpecs.clear();
  }

  // Now process the columns that will not be projected out.
  if (!filterSubfields.empty()) {
    VELOX_CHECK_NOT_NULL(dataColumns);
    for (auto& [fieldName, subfields] : filterSubfields) {
      for (auto* subfield : subfields) {
        subfieldSpecs.push_back({subfield, true});
      }
      auto& type = dataColumns->findChild(fieldName);
      auto* fieldSpec = spec->getOrCreateChild(fieldName);
      addSubfields(*type, subfieldSpecs, 1, pool, *fieldSpec);
      processFieldSpec(dataColumns, type, *fieldSpec);
      subfieldSpecs.clear();
    }
  }

  for (auto& pair : subfieldFilters) {
    const auto name = pair.first.toString();
    if (isSynthesizedColumn(name, infoColumns)) {
      continue;
    }
    auto fieldSpec = spec->getOrCreateChild(pair.first);
    VELOX_CHECK_NULL(spec->filter());
    fieldSpec->setFilter(pair.second);
  }

  if (disableStatsBasedFilterReorder) {
    spec->disableStatsBasedFilterReorder();
  }
  return spec;
}

} // namespace facebook::velox::connector
