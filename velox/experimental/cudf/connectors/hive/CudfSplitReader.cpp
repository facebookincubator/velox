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
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <memory>
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
      subfieldFilterExpr_(subfieldFilterExpr),
      pushdownFilterExpr_(subfieldFilterExpr) {
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);
}

void CudfSplitReader::setupReader() {
  if (useExperimentalCudfReader_) {
    createExperimentalReader();
  } else {
    createCudfReader();
  }
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  setupReader();
}

void CudfSplitReader::prepareSplit(dwio::common::RuntimeStats& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool
  stream_ = cudfGlobalStreamPool().get_stream();

  useDecodedColumnCache_ = shouldUseDecodedColumnCache();
  if (useDecodedColumnCache_) {
    prepareDecodedColumnCache();
  } else {
    // Perform split-specific setup.
    prepareSplitInternal(runtimeStats);
  }

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

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::readNextChunk() {
  if (useDecodedColumnCache_) {
    return readNextDecodedColumnCacheFileRange();
  }

  auto output_mr = determineCudfMemoryResource();

  if (!useExperimentalCudfReader_) {
    // Read table using the regular cudf parquet reader
    VELOX_CHECK_NOT_NULL(splitReader_, "cudf parquet reader not present");

    if (!splitReader_->has_next()) {
      return std::nullopt;
    }

    auto tableWithMetadata = splitReader_->read_chunk();
    return castDecimalColumnsToVeloxTypes(
        std::move(tableWithMetadata.tbl), outputType_, stream_, output_mr);
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
        cudfHiveConfig_->maxChunkReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        cudfHiveConfig_->maxPassReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        rowGroupIndices,
        hybridScanState_->columnChunkData_,
        readerOptions_,
        stream_,
        output_mr);
  });

  if (!exptSplitReader_->has_next_table_chunk()) {
    return std::nullopt;
  }

  auto tableWithMetadata = exptSplitReader_->materialize_all_columns_chunk();
  return castDecimalColumnsToVeloxTypes(
      std::move(tableWithMetadata.tbl), outputType_, stream_, output_mr);
}

void CudfSplitReader::resetSplit() {
  splitReader_.reset();
  exptSplitReader_.reset();
  hybridScanState_.reset();
  dataSource_.reset();
  fileMetaData_.clear();
  pushdownFilterExpr_ = subfieldFilterExpr_;
  hasSplitSpecificPushdownFilter_ = false;
  useDecodedColumnCache_ = false;
  isFullyDecodedColumnCacheHit_ = false;
  decodedColumnCacheCompression_ =
      CudfDecodedColumnCache::CompressionMode::kNone;
  decodedColumnCacheMetadata_.reset();
  decodedColumnCacheRowGroups_.clear();
  decodedColumnCacheRowOffsets_.clear();
  decodedColumnCacheRowGroupRuns_.clear();
  decodedColumnCacheRowGroupIndex_ = 0;
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
  readerOptions_ = makeReaderOptions(
      cudf::io::source_info{dataSource_.get()}, readColumnNames_, true, true);
}

cudf::io::parquet_reader_options CudfSplitReader::makeReaderOptions(
    cudf::io::source_info sourceInfo,
    const std::vector<std::string>& columnNames,
    bool applySplitByteRange,
    bool applyFilter) const {
  auto options =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  if (applySplitByteRange) {
    if (split_->start != 0) {
      options.set_skip_bytes(split_->start);
    }
    if (split_->size() != std::numeric_limits<uint64_t>::max()) {
      options.set_num_bytes(split_->size());
    }
  }

  if (applyFilter) {
    if (auto* filter = pushdownFilter(); filter != nullptr) {
      options.set_filter(*filter);
    }
  }

  if (not columnNames.empty()) {
    options.set_column_names(columnNames);
  }

  if (prependRowIndex_) {
    options.enable_prepend_row_index_column(true);
  }

  return options;
}

bool CudfSplitReader::shouldUseDecodedColumnCache() const {
  if (not cudfHiveConfig_->experimentalDecodedColumnCacheEnabledSession(
          connectorQueryCtx_->sessionProperties())) {
    return false;
  }

  // Immutable files make path + absolute row range a stable identity. The
  // cache stores unfiltered top-level columns and applies any subfield filter
  // after restoration. Zero chunk limits allow one materialization for all
  // selected row groups in a file split.
  return cudfHiveConfig_->immutableFiles() and useExperimentalCudfReader_ and
      supportsDecodedColumnCache() and not readColumnNames_.empty() and
      not prependRowIndex_ and
      cudfHiveConfig_->maxChunkReadLimitSession(
          connectorQueryCtx_->sessionProperties()) == 0 and
      cudfHiveConfig_->maxPassReadLimitSession(
          connectorQueryCtx_->sessionProperties()) == 0;
}

void CudfSplitReader::prepareDecodedColumnCache() {
  CUDF_CUDA_TRY(cudaGetDevice(&cudaDeviceId_));
  decodedColumnCacheCompression_ =
      CudfDecodedColumnCache::compressionModeFromString(
          cudfHiveConfig_->experimentalDecodedColumnCacheCompressionSession(
              connectorQueryCtx_->sessionProperties()));
  decodedColumnCacheFileKey_ = {
      .connectorId = split_->connectorId, .filePath = split_->filePath};

  auto& cache = CudfDecodedColumnCache::instance();
  const bool metadataHit =
      cache.findMetadata(decodedColumnCacheFileKey_) != nullptr;
  decodedColumnCacheMetadata_ = cache.findMetadata(decodedColumnCacheFileKey_);
  if (not decodedColumnCacheMetadata_) {
    fileMetaDatas();
    VELOX_CHECK_EQ(
        fileMetaData_.size(),
        1,
        "Decoded column cache requires exactly one Parquet metadata");
    auto metadata = std::make_shared<const cudf::io::parquet::FileMetaData>(
        std::move(fileMetaData_.front()));
    fileMetaData_.clear();
    decodedColumnCacheMetadata_ = cache.insertMetadataIfAbsent(
        decodedColumnCacheFileKey_, std::move(metadata));
  }

  readerOptions_ = makeReaderOptions(
      cudf::io::source_info{split_->filePath}, readColumnNames_, true, true);
  exptSplitReader_ = std::make_unique<CudfHybridScanReader>(
      *decodedColumnCacheMetadata_, readerOptions_);

  decodedColumnCacheRowOffsets_.reserve(
      decodedColumnCacheMetadata_->row_groups.size() + 1);
  decodedColumnCacheRowOffsets_.push_back(0);
  for (const auto& rowGroup : decodedColumnCacheMetadata_->row_groups) {
    VELOX_CHECK_GE(rowGroup.num_rows, 0);
    decodedColumnCacheRowOffsets_.push_back(
        decodedColumnCacheRowOffsets_.back() + rowGroup.num_rows);
  }

  decodedColumnCacheRowGroups_ =
      exptSplitReader_->all_row_groups(readerOptions_);
  if (readerOptions_.get_skip_bytes() > 0 or
      readerOptions_.get_num_bytes().has_value()) {
    decodedColumnCacheRowGroups_ =
        exptSplitReader_->filter_row_groups_with_byte_range(
            decodedColumnCacheRowGroups_, readerOptions_);
  }

  if (readerOptions_.get_filter().has_value()) {
    decodedColumnCacheRowGroups_ =
        exptSplitReader_->filter_row_groups_with_stats(
            decodedColumnCacheRowGroups_, readerOptions_, stream_);
  }

  int64_t outputRow = 0;
  std::optional<cudf::size_type> previousRowGroup;
  for (const auto rowGroupIndex : decodedColumnCacheRowGroups_) {
    const auto [firstRow, lastRow] = decodedColumnRowRange(rowGroupIndex);
    VELOX_CHECK_LE(firstRow, lastRow);
    const auto outputLastRow = outputRow + (lastRow - firstRow);
    if (firstRow != lastRow) {
      if (previousRowGroup.has_value() and
          rowGroupIndex == previousRowGroup.value() + 1 and
          not decodedColumnCacheRowGroupRuns_.empty()) {
        auto& run = decodedColumnCacheRowGroupRuns_.back();
        run.lastRow = lastRow;
        run.outputLastRow = outputLastRow;
      } else {
        decodedColumnCacheRowGroupRuns_.push_back(
            {.firstRow = firstRow,
             .lastRow = lastRow,
             .outputFirstRow = outputRow,
             .outputLastRow = outputLastRow});
      }
    }
    outputRow = outputLastRow;
    previousRowGroup = rowGroupIndex;
  }

  isFullyDecodedColumnCacheHit_ = metadataHit;
  for (const auto& run : decodedColumnCacheRowGroupRuns_) {
    for (const auto& columnName : readColumnNames_) {
      const auto type = readColumnType(columnName);
      if (not cache.containsColumnRange(
              makeDecodedColumnCacheKey(columnName, type),
              run.firstRow,
              run.lastRow)) {
        isFullyDecodedColumnCacheHit_ = false;
        return;
      }
    }
  }
}

std::optional<std::unique_ptr<cudf::table>>
CudfSplitReader::readNextDecodedColumnCacheFileRange() {
  if (decodedColumnCacheRowGroupIndex_ >= decodedColumnCacheRowGroups_.size() or
      decodedColumnCacheRowGroupRuns_.empty()) {
    decodedColumnCacheRowGroupIndex_ = decodedColumnCacheRowGroups_.size();
    return std::nullopt;
  }

  struct ColumnState {
    std::string name;
    TypePtr veloxType;
    std::unique_ptr<cudf::column> output;
  };

  std::vector<ColumnState> columnStates;
  std::vector<size_t> missingColumnIndices;
  std::vector<std::string> missingColumnNames;
  std::vector<TypePtr> missingColumnTypes;
  columnStates.reserve(readColumnNames_.size());
  for (const auto& columnName : readColumnNames_) {
    const auto veloxType = readColumnType(columnName);
    auto key = makeDecodedColumnCacheKey(columnName, veloxType);
    auto output = materializeDecodedColumnCacheRuns(key);
    if (output) {
      ++decodedColumnCacheHits_;
    } else {
      ++decodedColumnCacheMisses_;
      missingColumnIndices.push_back(columnStates.size());
      missingColumnNames.push_back(columnName);
      missingColumnTypes.push_back(veloxType);
    }
    columnStates.push_back(
        ColumnState{columnName, std::move(veloxType), std::move(output)});
  }

  if (not missingColumnNames.empty()) {
    auto decodedColumns = decodeAndCacheFileColumns(
        decodedColumnCacheRowGroups_, missingColumnNames, missingColumnTypes);
    VELOX_CHECK_EQ(decodedColumns.size(), missingColumnIndices.size());
    for (size_t missingIndex = 0; missingIndex < missingColumnIndices.size();
         ++missingIndex) {
      columnStates[missingColumnIndices[missingIndex]].output =
          std::move(decodedColumns[missingIndex]);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(columnStates.size());
  for (auto& state : columnStates) {
    VELOX_CHECK_NOT_NULL(state.output);
    outputColumns.push_back(std::move(state.output));
  }
  decodedColumnCacheRowGroupIndex_ = decodedColumnCacheRowGroups_.size();
  return std::make_unique<cudf::table>(std::move(outputColumns));
}

std::unique_ptr<cudf::column>
CudfSplitReader::materializeDecodedColumnCacheRuns(
    const CudfDecodedColumnCache::ColumnKey& key) const {
  auto& cache = CudfDecodedColumnCache::instance();
  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(decodedColumnCacheRowGroupRuns_.size());
  for (const auto& run : decodedColumnCacheRowGroupRuns_) {
    auto piece = cache.materializeColumnRange(
        key,
        run.firstRow,
        run.lastRow,
        stream_,
        decodedColumnCacheRowGroupRuns_.size() == 1
            ? determineCudfMemoryResource()
            : get_temp_mr(),
        get_temp_mr());
    if (not piece) {
      return nullptr;
    }
    pieces.push_back(std::move(piece));
  }

  if (pieces.empty()) {
    return nullptr;
  }
  if (pieces.size() == 1) {
    return std::move(pieces.front());
  }
  std::vector<cudf::column_view> pieceViews;
  pieceViews.reserve(pieces.size());
  for (const auto& piece : pieces) {
    pieceViews.push_back(piece->view());
  }
  return cudf::concatenate(pieceViews, stream_, determineCudfMemoryResource());
}

std::vector<std::unique_ptr<cudf::column>>
CudfSplitReader::decodeAndCacheFileColumns(
    const std::vector<cudf::size_type>& rowGroupIndices,
    const std::vector<std::string>& columnNames,
    const std::vector<TypePtr>& veloxTypes) {
  VELOX_CHECK_EQ(columnNames.size(), veloxTypes.size());
  VELOX_CHECK(not columnNames.empty());
  VELOX_CHECK(not rowGroupIndices.empty());
  setupCudfDataSource();

  auto columnOptions = makeReaderOptions(
      cudf::io::source_info{split_->filePath}, columnNames, false, false);
  exptSplitReader_->reset_column_selection();
  const auto columnChunkByteRanges =
      exptSplitReader_->all_column_chunks_byte_ranges(
          rowGroupIndices, columnOptions);
  auto ioData = fetchByteRangesAsync(
      dataSource_, columnChunkByteRanges, stream_, get_temp_mr());
  std::get<2>(ioData).wait();

  auto tableWithMetadata = exptSplitReader_->materialize_all_columns(
      rowGroupIndices,
      std::get<1>(ioData),
      columnOptions,
      stream_,
      get_output_mr());
  ++decodedColumnCacheDecodeCalls_;
  VELOX_CHECK_NOT_NULL(tableWithMetadata.tbl);
  VELOX_CHECK_EQ(
      tableWithMetadata.tbl->num_columns(),
      columnNames.size(),
      "Expected {} decoded columns across {} row groups",
      columnNames.size(),
      rowGroupIndices.size());

  auto columns = tableWithMetadata.tbl->release();
  VELOX_CHECK_EQ(tableWithMetadata.metadata.schema_info.size(), columns.size());
  std::vector<std::unique_ptr<cudf::column>> result;
  result.reserve(columnNames.size());
  auto& cache = CudfDecodedColumnCache::instance();
  for (size_t requestedIndex = 0; requestedIndex < columnNames.size();
       ++requestedIndex) {
    const auto& columnName = columnNames[requestedIndex];
    const auto metadataIt = std::find_if(
        tableWithMetadata.metadata.schema_info.begin(),
        tableWithMetadata.metadata.schema_info.end(),
        [&](const auto& info) { return info.name == columnName; });
    VELOX_CHECK(
        metadataIt != tableWithMetadata.metadata.schema_info.end(),
        "Decoded table metadata is missing column '{}'",
        columnName);
    const auto decodedIndex = static_cast<size_t>(std::distance(
        tableWithMetadata.metadata.schema_info.begin(), metadataIt));
    VELOX_CHECK_NOT_NULL(columns[decodedIndex]);
    auto column = castDecimalColumns(
        std::move(columns[decodedIndex]),
        veloxTypes[requestedIndex],
        stream_,
        get_output_mr());
    for (const auto& run : decodedColumnCacheRowGroupRuns_) {
      VELOX_CHECK_GE(run.outputFirstRow, 0);
      VELOX_CHECK_LE(
          run.outputLastRow,
          static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
      const auto slices = cudf::slice(
          column->view(),
          {static_cast<cudf::size_type>(run.outputFirstRow),
           static_cast<cudf::size_type>(run.outputLastRow)},
          stream_);
      VELOX_CHECK_EQ(slices.size(), 1);
      cache.insertColumnRangeIfAbsent(
          makeDecodedColumnCacheKey(columnName, veloxTypes[requestedIndex]),
          run.firstRow,
          run.lastRow,
          slices.front(),
          stream_,
          get_temp_mr(),
          decodedColumnCacheCompression_);
    }
    result.push_back(std::move(column));
  }

  stream_.synchronize();
  return result;
}

CudfDecodedColumnCache::ColumnKey CudfSplitReader::makeDecodedColumnCacheKey(
    const std::string& columnName,
    const TypePtr& veloxType) const {
  return {
      .file = decodedColumnCacheFileKey_,
      .deviceId = cudaDeviceId_,
      .columnName = columnName,
      .veloxType = veloxType->toString(),
      .timestampType = cudfHiveConfig_->timestampType().id(),
      .usePandasMetadata = cudfHiveConfig_->isUsePandasMetadata(),
      .useArrowSchema = cudfHiveConfig_->isUseArrowSchema(),
      .allowMismatchedSchemas =
          cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas(),
  };
}

std::pair<int64_t, int64_t> CudfSplitReader::decodedColumnRowRange(
    cudf::size_type rowGroupIndex) const {
  VELOX_CHECK_GE(rowGroupIndex, 0);
  const auto index = static_cast<size_t>(rowGroupIndex);
  VELOX_CHECK_LT(index + 1, decodedColumnCacheRowOffsets_.size());
  return {
      decodedColumnCacheRowOffsets_[index],
      decodedColumnCacheRowOffsets_[index + 1]};
}

TypePtr CudfSplitReader::readColumnType(const std::string& columnName) const {
  if (tableHandle_->dataColumns() and
      tableHandle_->dataColumns()->containsChild(columnName)) {
    return tableHandle_->dataColumns()->findChild(columnName);
  }
  VELOX_CHECK(
      outputType_->containsChild(columnName),
      "No Velox type available for decoded cached column '{}'",
      columnName);
  return outputType_->findChild(columnName);
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

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));

  // Create a parquet reader
  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimitSession(
          connectorQueryCtx_->sessionProperties()),
      cudfHiveConfig_->maxPassReadLimitSession(
          connectorQueryCtx_->sessionProperties()),
      std::move(sources),
      std::move(fileMetaData_),
      readerOptions_,
      stream_,
      determineCudfMemoryResource());

  // Metadata ingested
  fileMetaData_.clear();
}

void CudfSplitReader::createExperimentalReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "cuDF experimental reader requires exactly one parquet metadata");

  // Create a hybrid scan reader
  nvtxRangePush("hybridScanReader");
  auto reader = std::make_unique<CudfHybridScanReader>(
      std::move(fileMetaData_.front()), readerOptions_);
  nvtxRangePop();

  exptSplitReader_ = std::move(reader);
  hybridScanState_ = std::make_unique<HybridScanState>();

  // Metadata ingested
  fileMetaData_.clear();
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
