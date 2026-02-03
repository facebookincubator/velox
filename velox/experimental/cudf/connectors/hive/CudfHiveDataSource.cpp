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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.hpp"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/FieldReference.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      baseReaderOpts_(pool_),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
    // Add fields in the filter to the columns to read if not there
    for (const auto& [field, _] : subfieldFilters_) {
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field.toString()) == readColumnNames_.end()) {
        readColumnNames_.push_back(field.toString());
      }
    }
  }

  // Create remaining filter
  auto remainingFilter = tableHandle_->remainingFilter();
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    for (const auto& field : remainingFilterExprSet_->distinctFields()) {
      // Add fields in the filter to the columns to read if not there
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field->name()) == readColumnNames_.end()) {
        readColumnNames_.push_back(field->name());
      }
    }

    const RowTypePtr remainingFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    cudfExpressionEvaluator_ = velox::cudf_velox::createCudfExpression(
        remainingFilterExprSet_->exprs()[0], remainingFilterType_);
    // TODO(kn): Get column names and subfields from remaining filter and add to
    // readColumnNames_
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStats_ = std::make_shared<io::IoStatistics>();
  fsStats_ = std::make_shared<filesystems::File::IoStats>();

  // Whether to use the old split reader
  useOldSplitReader_ = cudfHiveConfig_->useOldCudfReaderSession(
      connectorQueryCtx_->sessionProperties());
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /* future */) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  // Basic sanity checks
  VELOX_CHECK_NOT_NULL(split_, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(
      splitReader_ or exptSplitReader_, "No split reader present");

  std::unique_ptr<cudf::table> cudfTable;
  cudf::io::table_metadata metadata;

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  if (useOldSplitReader_) {
    // Read table using the old cudf parquet reader
    VELOX_CHECK_NOT_NULL(splitReader_, "Old cudf split reader not present");

    if (not splitReader_->has_next()) {
      return nullptr;
    }

    auto tableWithMetadata = splitReader_->read_chunk();
    cudfTable = std::move(tableWithMetadata.tbl);
    metadata = std::move(tableWithMetadata.metadata);
  } else {
    // Read table using the experimental parquet reader
    VELOX_CHECK_NOT_NULL(
        exptSplitReader_, "Experimental cudf split reader not present");

    // TODO(mh): Replace this with chunked hybrid scan APIs when available in
    // the pinned cuDF version
    std::call_once(*tableMaterialized_, [&]() {
      auto rowGroupIndices = exptSplitReader_->all_row_groups(readerOptions_);

      if (readerOptions_.get_filter().has_value()) {
        rowGroupIndices = exptSplitReader_->filter_row_groups_with_stats(
            rowGroupIndices, readerOptions_, stream_);
      }

      // Get column chunk byte ranges to fetch
      const auto columnChunkByteRanges =
          exptSplitReader_->all_column_chunks_byte_ranges(
              rowGroupIndices, readerOptions_);

      // Fetch column chunk byte ranges
      const auto columnChunkBuffers = fetchByteRanges(
          dataSource_,
          columnChunkByteRanges,
          stream_,
          cudf::get_current_device_resource_ref());

      // Convert buffers to device spans
      const auto columnChunkData = makeDeviceSpans<uint8_t>(columnChunkBuffers);

      // Read table
      auto tableWithMetadata = exptSplitReader_->materialize_all_columns(
          rowGroupIndices, columnChunkData, readerOptions_, stream_);
      cudfTable = std::move(tableWithMetadata.tbl);
      metadata = std::move(tableWithMetadata.metadata);
    });

    if (cudfTable == nullptr) {
      return nullptr;
    }
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStats_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(),
      &CudfHiveDataSource::totalScanTimeCalculator,
      callbackData);

  uint64_t filterTimeUs{0};
  // Apply remaining filter if present
  if (remainingFilterExprSet_) {
    MicrosecondTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    const auto originalNumColumns = cudfTableColumns.size();
    // Filter may need addtional computed columns which are added to
    // cudfTableColumns
    auto filterResult = cudfExpressionEvaluator_->eval(
        cudfTableColumns, stream_, cudf::get_current_device_resource_ref());
    // discard computed columns
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(originalNumColumns);
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + originalNumColumns,
        std::back_inserter(originalColumns));
    auto originalTable =
        std::make_unique<cudf::table>(std::move(originalColumns));
    // Keep only rows where the filter is true
    cudfTable = cudf::apply_boolean_mask(
        *originalTable,
        asView(filterResult),
        stream_,
        cudf::get_current_device_resource_ref());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  // Output RowVectorPtr
  const auto nRows = cudfTable->num_rows();

  // keep only outputType_.size() columns in cudfTable_
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(originalColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(originalColumns));
  }

  // TODO (dm): Should we only enable table scan if cudf is registered?
  // Earlier we could enable cudf table scans without using other cudf operators
  // We still can, but I'm wondering if this is the right thing to do
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_->names(), stream_);
  stream_.synchronize();

  // Check if conversion yielded a nullptr
  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  // Update completedRows_.
  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

void CudfHiveDataSource::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStats->incTotalScanTime(elapsedNs);

  delete data;
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = [&]() {
    // Dynamic cast split to `CudfHiveConnectorSplit`
    if (std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
      return std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split);
      // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
    } else if (std::dynamic_pointer_cast<hive::HiveConnectorSplit>(split)) {
      const auto hiveSplit =
          std::dynamic_pointer_cast<hive::HiveConnectorSplit>(split);
      VELOX_CHECK_EQ(
          hiveSplit->fileFormat,
          dwio::common::FileFormat::PARQUET,
          "Unsupported file format for conversion from HiveConnectorSplit to CudfHiveConnectorSplit");
      VELOX_CHECK_EQ(
          hiveSplit->start,
          0,
          "CudfHiveDataSource cannot process splits with non-zero offset");
      // Remove "file:" prefix from the file path if present
      std::string cleanedPath = hiveSplit->filePath;
      constexpr std::string_view kFilePrefix = "file:";
      constexpr std::string_view kS3APrefix = "s3a:";
      if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
        cleanedPath = cleanedPath.substr(kFilePrefix.size());
      } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
        // KvikIO does not support "s3a:" prefix. We need to translate it to
        // "s3:".
        cleanedPath.erase(kS3APrefix.size() - 2, 1);
      }
      return CudfHiveConnectorSplitBuilder(cleanedPath)
          .connectorId(hiveSplit->connectorId)
          .splitWeight(hiveSplit->splitWeight)
          .build();
    } else {
      VELOX_FAIL("Unsupported split type: {}", split->toString());
    }
  }();

  VLOG(1) << "Adding split " << split_->toString();

  // Split reader already exists, reset
  if (splitReader_ or exptSplitReader_) {
    splitReader_.reset();
    exptSplitReader_.reset();
    tableMaterialized_.reset();
  }

  // Create a cudf split reader
  if (useOldSplitReader_) {
    splitReader_ = createSplitReader();
  } else {
    exptSplitReader_ = createExperimentalSplitReader();
  }

  tableMaterialized_ = std::make_unique<std::once_flag>();

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, fsStats_ ? fsStats_.get() : nullptr);
    if (fileHandleCachePtr.get() and fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    // Unable to get the file size, log a warning and continue
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

void CudfHiveDataSource::setupCudfDataSourceAndOptions() {
  // Build source info for the chunked parquet reader
  auto sourceInfo = [&]() {
    // Use file data source if we don't want to use the BufferedInput source
    if (not cudfHiveConfig_->useBufferedInputSession(
            connectorQueryCtx_->sessionProperties())) {
      LOG(INFO) << "Using file data source for CudfHiveDataSource";
      return cudf::io::source_info{split_->filePath};
    }

    auto fileHandleCachePtr = FileHandleCachedPtr{};
    try {
      const auto fileHandleKey = FileHandleKey{
          .filename = split_->filePath,
          .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
      auto fileProperties = FileProperties{};
      fileHandleCachePtr = fileHandleFactory_->generate(
          fileHandleKey, &fileProperties, fsStats_ ? fsStats_.get() : nullptr);
      VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
    } catch (const VeloxRuntimeError& e) {
      LOG(WARNING) << fmt::format(
          "Failed to generate file handle cache for file {}, falling back to file data source for CudfHiveDataSource",
          split_->filePath);
      return cudf::io::source_info{split_->filePath};
    }

    // Here we keep adding new entries to CacheTTLController when new
    // fileHandles are generated, if CacheTTLController was created. Creator of
    // CacheTTLController needs to make sure a size control strategy was
    // available such as removing aged out entries.
    if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
      cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
    }

    auto bufferedInput =
        velox::connector::hive::BufferedInputBuilder::getInstance()->create(
            *fileHandleCachePtr,
            baseReaderOpts_,
            connectorQueryCtx_,
            ioStats_,
            fsStats_,
            executor_);
    if (not bufferedInput) {
      LOG(WARNING) << fmt::format(
          "Failed to create buffered input source for file {}, falling back to file data source for CudfHiveDataSource",
          split_->filePath);
      return cudf::io::source_info{split_->filePath};
    }
    dataSource_ =
        std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
    return cudf::io::source_info{dataSource_.get()};
  }();

  if (dataSource_ == nullptr) {
    dataSource_ = std::move(cudf::io::make_datasources(sourceInfo).front());
  }

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .skip_rows(cudfHiveConfig_->skipRows())
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  // Set num_rows only if available
  if (cudfHiveConfig_->numRows().has_value()) {
    readerOptions_.set_num_rows(cudfHiveConfig_->numRows().value());
  }

  if (subfieldFilters_.size()) {
    const RowTypePtr readerFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> newNames;
        std::vector<TypePtr> newTypes;

        for (const auto& name : readColumnNames_) {
          // Ensure all columns being read are available to the filter
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          newNames.emplace_back(std::move(name));
          newTypes.push_back(parsedType);
        }

        return ROW(std::move(newNames), std::move(newTypes));
      } else {
        return outputType_;
      }
    }();

    // Build a combined AST for all subfield filters.
    auto const& combinedExpr = createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);

    readerOptions_.set_filter(combinedExpr);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }
}

CudfParquetReaderPtr CudfHiveDataSource::createSplitReader() {
  setupCudfDataSourceAndOptions();
  stream_ = cudfGlobalStreamPool().get_stream();

  // Create a parquet reader
  return std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      readerOptions_,
      stream_,
      cudf::get_current_device_resource_ref());
}

CudfHybridScanReaderPtr CudfHiveDataSource::createExperimentalSplitReader() {
  setupCudfDataSourceAndOptions();
  stream_ = cudfGlobalStreamPool().get_stream();

  // Create a hybrid scan reader
  auto const footerBytes = fetchFooterBytes(dataSource_);
  auto exptSplitReader = std::make_unique<CudfHybridScanReader>(
      cudf::host_span<uint8_t const>{footerBytes->data(), footerBytes->size()},
      readerOptions_);

  // Setup page index if available
  auto const pageIndexByteRange = exptSplitReader->page_index_byte_range();
  if (not pageIndexByteRange.is_empty()) {
    auto const pageIndexBytes = dataSource_->host_read(
        pageIndexByteRange.offset(), pageIndexByteRange.size());
    exptSplitReader->setup_page_index(cudf::host_span<uint8_t const>{
        pageIndexBytes->data(), pageIndexBytes->size()});
  }

  return exptSplitReader;
}

void CudfHiveDataSource::resetSplit() {
  split_.reset();
  splitReader_.reset();
  exptSplitReader_.reset();
  tableMaterialized_.reset();
  dataSource_.reset();
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  res.insert({
      {"totalScanTime",
       RuntimeMetric(ioStats_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
      {"totalRemainingFilterTime",
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  const auto& fsStats = fsStats_->stats();
  for (const auto& storageStats : fsStats) {
    res.emplace(storageStats.first, storageStats.second);
  }
  return res;
}

} // namespace facebook::velox::cudf_velox::connector::hive
