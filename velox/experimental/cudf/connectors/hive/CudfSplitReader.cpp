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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#ifdef VELOX_ENABLE_ABFS
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#endif

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <memory>
#include <numeric>
#include <ranges>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

// Checks whether the `path` uses an ABFS scheme
bool isAbfsPath([[maybe_unused]] const std::string_view path) {
#ifdef VELOX_ENABLE_ABFS
  return ::facebook::velox::filesystems::isAbfsFile(path);
#else
  return false;
#endif
}

// Rebuilds a struct/list column in-place after possibly transforming (e.g.,
// decimal-casting) its children.
template <typename TransformChildrenFn>
std::unique_ptr<cudf::column> rebuildWithTransformedChildren(
    std::unique_ptr<cudf::column> col,
    TransformChildrenFn&& transformFn) {
  auto const type = col->type();
  auto const size = col->size();
  auto const nullCount = col->null_count();
  auto contents = col->release();
  transformFn(contents.children);
  return std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount,
      std::move(contents.children));
}

// Recursively casts columns to the expected Velox type iff the column is:
//  - Decimal type but not the expected Velox type.
//  - Struct type: with any of its children being decimal type but not the
//  expected Velox type. Rebuilt in place with the casted children.
//  - List type: with its `child` being decimal type but not the expected Velox
//  type. Rebuilt in place with the casted children.
std::unique_ptr<cudf::column> castDecimalColumns(
    std::unique_ptr<cudf::column> col,
    const TypePtr& veloxType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Decimal type (base case)
  if (veloxType->isDecimal()) {
    auto const targetType = veloxToCudfDataType(veloxType);
    if (col->type() != targetType) {
      return cudf::cast(col->view(), targetType, stream, mr);
    }
    return col;
  }

  // Struct type
  if (veloxType->kind() == TypeKind::ROW) {
    auto const& rowType = veloxType->asRow();
    auto const numChildren = static_cast<size_t>(col->num_children());
    VELOX_CHECK_EQ(
        numChildren,
        rowType.size(),
        "Scanned STRUCT column has {} fields but the expected schema has {}.",
        numChildren,
        rowType.size());
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      for (size_t i = 0; i < numChildren; ++i) {
        children[i] = castDecimalColumns(
            std::move(children[i]), rowType.childAt(i), stream, mr);
      }
    });
  }

  // List type
  if (veloxType->kind() == TypeKind::ARRAY) {
    // A LIST column stores [offsets, child]; only the child may hold decimal
    // data.
    VELOX_CHECK_EQ(
        col->num_children(),
        2,
        "LIST column must have exactly 2 children: [offsets, child]");
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      auto const childIdx = cudf::lists_column_view::child_column_index;
      children[childIdx] = castDecimalColumns(
          std::move(children[childIdx]), veloxType->childAt(0), stream, mr);
    });
  }

  return col;
}

std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    const RowTypePtr& rowType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numColumns =
      std::min<size_t>(table->view().num_columns(), rowType->size());
  auto columns = table->release();
  for (size_t i = 0; i < numColumns; ++i) {
    columns[i] = castDecimalColumns(
        std::move(columns[i]), rowType->childAt(i), stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace

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
      baseReaderOpts_(pool_),
      subfieldFilterExpr_(subfieldFilterExpr),
      pushdownFilterExpr_(subfieldFilterExpr) {
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);
}

CudfSplitReader::~CudfSplitReader() {
  // A split abandoned before it is read, e.g. when the task is cancelled while
  // the preloader prepares it, can still have reads in flight.
  if (passState_ == nullptr) {
    return;
  }
  try {
    releaseCurrentPassData();
  } catch (const std::exception& e) {
    // The data of a failed read is being dropped anyway, so the failure must
    // not propagate out of the destructor.
    LOG(ERROR) << fmt::format(
        "Failed to drain the reads of an abandoned split. Path: {}. Error: {}.",
        split_ != nullptr ? split_->filePath : "unknown",
        e.what());
  }
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  createCudfReader();
}

void CudfSplitReader::prepareSplit(dwio::common::RuntimeStats& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool
  stream_ = cudfGlobalStreamPool().get_stream();

  // Perform split-specific setup.
  prepareSplitInternal(runtimeStats);

  // Update runtime stats
  runtimeStats.processedSplits++;
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::next(
    uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  auto chunkOpt = readNextChunk();
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return std::move(chunkOpt.value());
}

void CudfSplitReader::setConnectorQueryCtx(
    const ConnectorQueryCtx* connectorQueryCtx) {
  VELOX_CHECK_NOT_NULL(connectorQueryCtx);
  // The reader keeps buffers and options tied to the memory pool of the
  // context it was created with, so only contexts sharing that pool can adopt
  // it.
  VELOX_CHECK(
      connectorQueryCtx->memoryPool() == pool_,
      "Cannot rebind a cuDF split reader to a query context with a different memory pool");
  connectorQueryCtx_ = connectorQueryCtx;
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::readNextChunk() {
  VELOX_CHECK_NOT_NULL(splitReader_, "cuDF parquet reader not present");
  VELOX_CHECK_NOT_NULL(passState_, "Row group pass state not present");

  auto outputMr = determineCudfMemoryResource();

  if (passState_->currentPass >= passState_->passes.size()) {
    return std::nullopt;
  }

  if (not passState_->isChunkingSetup) {
    setupChunkingForCurrentPass(outputMr);
    VELOX_CHECK(
        splitReader_->has_next_table_chunk(),
        "cuDF row group pass did not produce a table chunk");
  }

  auto tableWithMetadata = splitReader_->materialize_all_columns_chunk();
  auto table = castDecimalColumnsToVeloxTypes(
      std::move(tableWithMetadata.tbl), outputType_, stream_, outputMr);

  // This was the last chunk of the pass. Drop its fetch buffers and begin
  // I/O for the next pass while the caller consumes this table.
  if (not splitReader_->has_next_table_chunk()) {
    releaseCurrentPassData();
    passState_->isChunkingSetup = false;
    ++passState_->currentPass;
    if (passState_->currentPass < passState_->passes.size()) {
      startCurrentPassFetch();
    } else {
      passState_->passes.clear();
    }
  }

  return table;
}

void CudfSplitReader::startCurrentPassFetch() {
  if (passState_->currentPass >= passState_->passes.size() or
      passState_->fetch.pending.valid() or passState_->isChunkingSetup) {
    return;
  }

  const auto& rowGroupIndices = passState_->passes[passState_->currentPass];

  // Byte ranges are flattened across sources; the source index map is only
  // needed once a reader spans multiple data sources.
  const auto columnChunkByteRanges =
      splitReader_
          ->all_column_chunks_byte_ranges(rowGroupIndices, readerOptions_)
          .first;

  nvtxRangePush("fetchByteRanges");
  passState_->fetch = fetchByteRangesAsync(
      dataSource_, columnChunkByteRanges, stream_, get_temp_mr());
  nvtxRangePop();
}

void CudfSplitReader::setupChunkingForCurrentPass(
    rmm::device_async_resource_ref mr) {
  // A no-op when the fetch was already started while preparing the split.
  startCurrentPassFetch();

  // Wait for all reads of the pass to complete.
  passState_->fetch.pending.get();

  splitReader_->setup_chunking_for_all_columns(
      chunkReadLimit_,
      passReadLimit_,
      passState_->passes[passState_->currentPass],
      passState_->fetch.data,
      readerOptions_,
      stream_,
      mr);

  passState_->isChunkingSetup = true;
}

void CudfSplitReader::releaseCurrentPassData() {
  auto& fetch = passState_->fetch;
  if (fetch.pending.valid() and fetch.writesInFlight) {
    // Reads still in flight write into the buffers about to be released. A
    // fetch whose reads have not started yet is dropped instead: waiting on it
    // would read the whole pass only to discard it.
    fetch.pending.get();
  }
  passState_->fetch = {};
}

void CudfSplitReader::resetSplit() {
  if (passState_ != nullptr) {
    releaseCurrentPassData();
  }
  splitReader_.reset();
  passState_.reset();
  dataSource_.reset();
  fileMetaData_.clear();
  pushdownFilterExpr_ = subfieldFilterExpr_;
  hasSplitSpecificPushdownFilter_ = false;
}

cudf::ast::expression const* CudfSplitReader::pushdownFilter() const {
  return pushdownFilterExpr_;
}

cudf::ast::expression const* CudfSplitReader::subfieldFilter() const {
  return subfieldFilterExpr_;
}

bool CudfSplitReader::hasSplitSpecificPushdownFilter() const {
  return hasSplitSpecificPushdownFilter_;
}

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_) {
    return;
  }

  const auto useBufferedInput = cudfHiveConfig_->useBufferedInputSession(
      connectorQueryCtx_->sessionProperties());

  VELOX_CHECK(
      not isAbfsPath(split_->filePath) or useBufferedInput,
      "ABFS blobs require buffered input data source. "
      "Set the session property '{}' (or connector property '{}') to 'true'. "
      "Blob Path: {}.",
      CudfHiveConfig::kUseBufferedInputSession,
      CudfHiveConfig::kUseBufferedInput,
      split_->filePath);

  // Use KvikIO data source if we don't want to use the BufferedInput source
  if (not useBufferedInput) {
    VLOG(1) << fmt::format(
        "Using KvikIO data source for file: {}", split_->filePath);
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
    // ABFS blobs can not fall back to KvikIO. Throw the original error.
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to generate file handle cache for ABFS blob. Ensure "
          "registerAbfsFileSystem() and registerAzureClientProvider() have "
          "been called and the connector config provides Azure credentials. "
          "Blob path: {}. Error: {}.",
          split_->filePath,
          e.what());
    }

    LOG(WARNING) << fmt::format(
        "Failed to generate file handle cache for file. Falling back to KvikIO. Path: {}",
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
    // ABFS blobs can not fall back to KvikIO
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to create buffered input data source for the ABFS blob. Ensure that the registered "
          "BufferedInputBuilder is ABFS-aware. Blob path: {}.",
          split_->filePath);
    }

    LOG(WARNING) << fmt::format(
        "Failed to create buffered input data source for file. Falling back to the KvikIO. Path: {}",
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
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");
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

  if (auto* filter = pushdownFilter(); filter != nullptr) {
    readerOptions_.set_filter(*filter);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  if (prependRowIndex_) {
    readerOptions_.enable_prepend_row_index_column(true);
  }
}

rmm::device_async_resource_ref CudfSplitReader::determineCudfMemoryResource()
    const {
  return get_output_mr();
}

void CudfSplitReader::fileMetaDatas() {
  if (not fileMetaData_.empty()) {
    return;
  }

  // Setup the datasource
  setupCudfDataSource();

  // Check that the datasource is set up
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  // Wrap the existing datasource without transferring ownership.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));
  fileMetaData_ = cudf::io::read_parquet_footers(sources);
  VELOX_CHECK_GE(
      fileMetaData_.size(),
      1,
      "CudfSplitReader failed to read any parquet metadatas");

  if (pushdownFilterBuilder_) {
    VELOX_CHECK_EQ(
        fileMetaData_.size(),
        1,
        "Split-specific pushdown filters require exactly one Parquet metadata");
    pushdownFilterExpr_ = pushdownFilterBuilder_(fileMetaData_.front());
    VELOX_CHECK_NOT_NULL(
        pushdownFilterExpr_,
        "Split-specific pushdown filter builder must return an expression");
    hasSplitSpecificPushdownFilter_ = true;
  }
}

void CudfSplitReader::createCudfReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  // A reader spanning multiple sources also needs the byte range source index
  // map to fetch column chunks from the matching data source.
  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "cuDF parquet reader supports a single parquet metadata per split");

  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  chunkReadLimit_ =
      cudfHiveConfig_->maxChunkReadLimitSession(sessionProperties);
  passReadLimit_ = cudfHiveConfig_->maxPassReadLimitSession(sessionProperties);

  // Create a hybrid scan reader over all sources of the split
  nvtxRangePush("hybridScanMultifileReader");
  splitReader_ =
      std::make_unique<CudfParquetReader>(fileMetaData_, readerOptions_);
  nvtxRangePop();

  // Metadata ingested
  fileMetaData_.clear();

  setupPageIndexes();

  passState_ = std::make_unique<RowGroupPassState>();
  passState_->passes = selectRowGroupPasses();

  // Issue the reads of the first pass without waiting for them. When the split
  // is prepared by the preloader, this overlaps its I/O with the work the
  // driver is still doing on the previous split, at the cost of holding the
  // buffers of one pass per preloaded split rather than per driver.
  startCurrentPassFetch();
}

void CudfSplitReader::setupPageIndexes() {
  const auto pageIndexByteRanges = splitReader_->page_index_byte_ranges();

  // Parquet files written without a page index cannot be page pruned.
  if (std::ranges::any_of(pageIndexByteRanges, [](const auto& byteRange) {
        return byteRange.is_empty();
      })) {
    return;
  }

  [[maybe_unused]] auto [pageIndexBuffers, pageIndexData] =
      fetchPageIndexes(dataSource_, pageIndexByteRanges);
  splitReader_->setup_page_indexes(pageIndexData);
}

std::vector<std::vector<std::vector<cudf::size_type>>>
CudfSplitReader::selectRowGroupPasses() const {
  auto rowGroupIndices = splitReader_->all_row_groups(readerOptions_);

  // Filter row groups using row group byte ranges
  if (readerOptions_.get_skip_bytes() > 0 or
      readerOptions_.get_num_bytes().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_byte_range(
        rowGroupIndices, readerOptions_);
  }

  // Filter row groups using column chunk statistics
  if (readerOptions_.get_filter().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_stats(
        rowGroupIndices, readerOptions_, stream_);
  }

  const auto numRowGroups = std::accumulate(
      rowGroupIndices.begin(),
      rowGroupIndices.end(),
      std::size_t{0},
      [](auto sum, const auto& sourceRowGroups) {
        return sum + sourceRowGroups.size();
      });

  // Constructing passes requires at least one row group.
  if (numRowGroups == 0) {
    return {};
  }

  return splitReader_->construct_row_group_passes(
      rowGroupIndices, passReadLimit_);
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
  data->ioStatistics->incTotalScanTimeNs(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
