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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

CudfSplitReader::CudfSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    bool useExperimentalCudfReader,
    cudf::ast::expression const* subfieldFilterExpr)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241},
          std::nullopt,
          fmt::format("[split:{}]", split ? split->filePath : "unknown")),
      split_(std::move(split)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      cudfHiveConfig_(cudfHiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      useExperimentalCudfReader_(useExperimentalCudfReader),
      baseReaderOpts_(pool_),
      subfieldFilterExpr_(subfieldFilterExpr) {}

void CudfSplitReader::prepareSplit(
    dwio::common::RuntimeStatistics& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool
  stream_ = cudfGlobalStreamPool().get_stream();

  // Create a cuDF split reader
  if (useExperimentalCudfReader_) {
    createExperimentalReader();
    hybridScanState_ = std::make_unique<HybridScanState>();
  } else {
    createCudfReader(get_output_mr());
  }

  // Update runtime stats
  runtimeStats.processedSplits++;
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::next(
    uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  VELOX_CHECK(
      splitReader_ or exptSplitReader_,
      "No Cudf Split reader present. Call prepareSplit() first.");

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  auto chunkOpt = readNextChunk(get_output_mr());
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }
  auto cudfTable = std::move(chunkOpt.value());

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return cudfTable;
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::readNextChunk(
    rmm::device_async_resource_ref output_mr) {
  if (!useExperimentalCudfReader_) {
    // Read table using the regular cudf parquet reader
    VELOX_CHECK_NOT_NULL(splitReader_, "cudf parquet reader not present");

    if (!splitReader_->has_next()) {
      return std::nullopt;
    }

    auto tableWithMetadata = splitReader_->read_chunk();
    return std::move(tableWithMetadata.tbl);
  }

  // Read table using the experimental parquet reader
  VELOX_CHECK_NOT_NULL(exptSplitReader_, "cuDF hybrid scan reader not present");
  VELOX_CHECK_NOT_NULL(hybridScanState_, "hybrid scan state not present");

  std::call_once(*hybridScanState_->isHybridScanSetup_, [&]() {
    auto rowGroupIndices = exptSplitReader_->all_row_groups(readerOptions_);

    // Filter row groups using row group byte ranges
    if (readerOptions_.get_skip_bytes() > 0 or
        readerOptions_.get_num_bytes().has_value()) {
      rowGroupIndices = exptSplitReader_->filter_row_groups_with_byte_range(
          rowGroupIndices, readerOptions_);
    }

    // Filter row groups using column chunk statistics
    if (readerOptions_.get_filter().has_value()) {
      rowGroupIndices = exptSplitReader_->filter_row_groups_with_stats(
          rowGroupIndices, readerOptions_, stream_);
    }

    // Get column chunk byte ranges to fetch
    const auto columnChunkByteRanges =
        exptSplitReader_->all_column_chunks_byte_ranges(
            rowGroupIndices, readerOptions_);

    // Fetch column chunk byte ranges
    nvtxRangePush("fetchByteRanges");

    // Tuple containing a vector of device buffers, a vector of device spans
    // for each input byte range, and a future to wait for all reads to
    // complete
    auto ioData = fetchByteRangesAsync(
        dataSource_, columnChunkByteRanges, stream_, get_temp_mr());

    // Wait for all pending reads to complete
    std::get<2>(ioData).wait();
    nvtxRangePop();

    // Save state for hybrid scan reader for future calls to `next()`
    hybridScanState_->columnChunkBuffers_ = std::move(std::get<0>(ioData));
    hybridScanState_->columnChunkData_ = std::move(std::get<1>(ioData));

    exptSplitReader_->setup_chunking_for_all_columns(
        cudfHiveConfig_->maxChunkReadLimit(),
        cudfHiveConfig_->maxPassReadLimit(),
        rowGroupIndices,
        hybridScanState_->columnChunkData_,
        readerOptions_,
        stream_,
        output_mr);
    // TODO: check remainingFilterExprSet_ flag here to choose mr
  });

  if (!exptSplitReader_->has_next_table_chunk()) {
    return std::nullopt;
  }

  auto tableWithMetadata = exptSplitReader_->materialize_all_columns_chunk();
  return std::move(tableWithMetadata.tbl);
}

void CudfSplitReader::resetSplit() {
  splitReader_.reset();
  exptSplitReader_.reset();
  hybridScanState_.reset();
  dataSource_.reset();
}

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_) {
    return;
  }

  // Use file data source if we don't want to use the BufferedInput source
  if (not cudfHiveConfig_->useBufferedInputSession(
          connectorQueryCtx_->sessionProperties())) {
    VLOG(1) << "Using file data source for CudfSplitReader";
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
  }

  auto fileHandleCachePtr = FileHandleCachedPtr{};
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    LOG(WARNING) << fmt::format(
        "Failed to generate file handle cache for file {}, falling back to kvikio data source for CudfSplitReader",
        split_->filePath);
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
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
          ioStatistics_,
          ioStats_,
          executor_);
  if (not bufferedInput) {
    LOG(WARNING) << fmt::format(
        "Failed to create buffered input source for file {}, falling back to kvikio data source for CudfSplitReader",
        split_->filePath);
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
  }
  dataSource_ =
      std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
}

void CudfSplitReader::setupReaderOptions() {
  VELOX_CHECK_NOT_NULL(
      dataSource_, "CudfSplitReader failed to setup a cuDF datasource");
  auto sourceInfo = cudf::io::source_info{dataSource_.get()};

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  // Set skip_bytes and num_bytes if available
  if (split_->start != 0) {
    readerOptions_.set_skip_bytes(split_->start);
  }
  if (split_->size() != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(split_->size());
  }

  if (subfieldFilterExpr_ != nullptr) {
    readerOptions_.set_filter(*subfieldFilterExpr_);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }
}

void CudfSplitReader::createCudfReader(
    rmm::device_async_resource_ref output_mr) {
  // Setup datasource and reader options
  setupCudfDataSource();
  setupReaderOptions();

  // Create a parquet reader
  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      readerOptions_,
      stream_,
      output_mr);
}

void CudfSplitReader::createExperimentalReader() {
  // Setup datasource and reader options
  setupCudfDataSource();
  setupReaderOptions();

  // Create a hybrid scan reader
  nvtxRangePush("hybridScanReader");
  auto const footerBuffer = fetchFooterBytes(dataSource_);
  auto reader =
      std::make_unique<CudfHybridScanReader>(*footerBuffer, readerOptions_);
  nvtxRangePop();

  // Setup page index if available
  auto const pageIndexByteRange = reader->page_index_byte_range();
  if (not pageIndexByteRange.is_empty()) {
    nvtxRangePush("setupPageIndex");
    auto const pageIndexBuffer =
        fetchPageIndexBytes(dataSource_, pageIndexByteRange);
    reader->setup_page_index(*pageIndexBuffer);
    nvtxRangePop();
  }

  exptSplitReader_ = std::move(reader);
}

void CudfSplitReader::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTime(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
