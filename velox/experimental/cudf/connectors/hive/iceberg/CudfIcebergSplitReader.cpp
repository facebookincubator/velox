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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/encode/Base64.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"
#include "velox/dwio/common/BufferUtil.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <nvtx3/nvtx3.hpp>

#include <folly/lang/Bits.h>

#include <cstring>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

bool shouldSkipBySequenceNumber(
    int64_t fileSeqNum,
    int64_t dataSeqNum,
    bool isEqualityDelete) {
  if (fileSeqNum <= 0 || dataSeqNum <= 0) {
    return false;
  }
  return isEqualityDelete ? (fileSeqNum <= dataSeqNum)
                          : (fileSeqNum < dataSeqNum);
}

} // namespace

CudfIcebergSplitReader::CudfIcebergSplitReader(
    std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit,
    std::shared_ptr<const velox_hive::HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats)
    : NvtxHelper(
          nvtx3::rgb{80, 200, 120},
          std::nullopt,
          fmt::format(
              "[iceberg-split:{}]",
              icebergSplit ? icebergSplit->filePath : "unknown")),
      icebergSplit_(std::move(icebergSplit)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      cudfHiveConfig_(cudfHiveConfig),
      hiveConfig_(hiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      stream_(cudfGlobalStreamPool().get_stream()),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats) {}

void CudfIcebergSplitReader::prepareSplit() {
  splitReader_.reset();
  dataSource_.reset();
  dvReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  baseReadOffset_ = 0;

  loadDeletionVector();
  setupDeleteFileReaders();
  setupCudfDataSourceAndOptions();
  createCudfReader();
}

void CudfIcebergSplitReader::loadDeletionVector() {
  for (const auto& deleteFile : icebergSplit_->deleteFiles) {
    if (deleteFile.content == velox_iceberg::FileContent::kDeletionVector) {
      if (deleteFile.recordCount == 0) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              false)) {
        continue;
      }

      dvReader_ = std::make_unique<CudfDeletionVectorReader>(
          deleteFile.filePath,
          deleteFile.fileSizeInBytes,
          deleteFile.lowerBounds,
          deleteFile.upperBounds);
      dvReader_->loadAndInitialize(stream_);
      break;
    }
  }
}

void CudfIcebergSplitReader::setupDeleteFileReaders() {
  for (const auto& deleteFile : icebergSplit_->deleteFiles) {
    if (deleteFile.content == velox_iceberg::FileContent::kPositionalDeletes) {
      if (deleteFile.recordCount == 0) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/false)) {
        continue;
      }

      constexpr uint64_t splitOffset = 0;
      if (auto iter = deleteFile.upperBounds.find(
              velox_iceberg::IcebergMetadataColumn::kPosId);
          iter != deleteFile.upperBounds.end()) {
        auto decodedBound = encoding::Base64::decode(iter->second);
        VELOX_CHECK_EQ(
            decodedBound.size(),
            sizeof(uint64_t),
            "Unexpected decoded size for positional delete upper bound.");
        uint64_t posDeleteUpperBound;
        std::memcpy(
            &posDeleteUpperBound, decodedBound.data(), sizeof(uint64_t));
        posDeleteUpperBound = folly::Endian::little(posDeleteUpperBound);
        if (posDeleteUpperBound < splitOffset) {
          continue;
        }
      }

      positionalDeleteFileReaders_.push_back(
          std::make_unique<velox_iceberg::PositionalDeleteFileReader>(
              deleteFile,
              icebergSplit_->filePath,
              fileHandleFactory_,
              connectorQueryCtx_,
              executor_,
              hiveConfig_,
              ioStatistics_,
              ioStats_,
              runtimeStats_,
              splitOffset,
              icebergSplit_->connectorId));
    } else if (
        deleteFile.content == velox_iceberg::FileContent::kEqualityDeletes) {
      if (deleteFile.recordCount == 0 || deleteFile.equalityFieldIds.empty()) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/true)) {
        continue;
      }

      std::vector<std::string> equalityColumnNames;
      std::vector<TypePtr> equalityColumnTypes;

      const auto& dataColumns = tableHandle_->dataColumns();
      if (dataColumns) {
        for (const auto& eqFieldId : deleteFile.equalityFieldIds) {
          auto colIdx = static_cast<uint32_t>(eqFieldId - 1);
          VELOX_CHECK_LT(
              colIdx,
              dataColumns->size(),
              "Equality delete field ID out of range: {}",
              eqFieldId);
          equalityColumnNames.push_back(dataColumns->nameOf(colIdx));
          equalityColumnTypes.push_back(dataColumns->childAt(colIdx));
        }
      }

      if (!equalityColumnNames.empty()) {
        equalityDeleteFileReaders_.push_back(
            std::make_unique<velox_iceberg::EqualityDeleteFileReader>(
                deleteFile,
                equalityColumnNames,
                equalityColumnTypes,
                icebergSplit_->filePath,
                fileHandleFactory_,
                connectorQueryCtx_,
                executor_,
                hiveConfig_,
                ioStatistics_,
                ioStats_,
                runtimeStats_,
                icebergSplit_->connectorId));
      }
    } else if (
        deleteFile.content == velox_iceberg::FileContent::kDeletionVector) {
      // Handled via loadDeletionVectorBlob() and applyDeletionVector().
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

void CudfIcebergSplitReader::setupCudfDataSourceAndOptions() {
  std::string cleanedPath = icebergSplit_->filePath;
  constexpr std::string_view kFilePrefix = "file:";
  if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    cleanedPath = cleanedPath.substr(kFilePrefix.size());
  }

  auto sourceInfo = [&]() {
    if (!cudfHiveConfig_->useBufferedInputSession(
            connectorQueryCtx_->sessionProperties())) {
      VLOG(1) << "Using file data source for CudfIcebergSplitReader";
      return cudf::io::source_info{cleanedPath};
    }

    auto fileHandleCachePtr = FileHandleCachedPtr{};
    try {
      const auto fileHandleKey = FileHandleKey{
          .filename = icebergSplit_->filePath,
          .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
      auto fileProperties = FileProperties{};
      fileHandleCachePtr = fileHandleFactory_->generate(
          fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
      VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
    } catch (const VeloxRuntimeError& e) {
      LOG(WARNING) << fmt::format(
          "Failed to generate file handle cache for file {}, "
          "falling back to file data source for CudfIcebergSplitReader",
          icebergSplit_->filePath);
      return cudf::io::source_info{cleanedPath};
    }

    if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
      cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
    }

    dwio::common::ReaderOptions baseReaderOpts(pool_);
    auto bufferedInput =
        velox_hive::BufferedInputBuilder::getInstance()->create(
            *fileHandleCachePtr,
            baseReaderOpts,
            connectorQueryCtx_,
            ioStatistics_,
            ioStats_,
            executor_);
    if (!bufferedInput) {
      LOG(WARNING) << fmt::format(
          "Failed to create buffered input source for file {}, "
          "falling back to file data source for CudfIcebergSplitReader",
          icebergSplit_->filePath);
      return cudf::io::source_info{cleanedPath};
    }
    dataSource_ =
        std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
    return cudf::io::source_info{dataSource_.get()};
  }();

  if (dataSource_ == nullptr) {
    dataSource_ = std::move(cudf::io::make_datasources(sourceInfo).front());
  }

  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  if (icebergSplit_->start != 0) {
    readerOptions_.set_skip_bytes(icebergSplit_->start);
  }
  if (icebergSplit_->length != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(icebergSplit_->length);
  }

  if (!readColumnNames_.empty()) {
    readerOptions_.set_column_names(readColumnNames_);
  }
}

void CudfIcebergSplitReader::createCudfReader() {
  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      readerOptions_,
      stream_,
      get_output_mr());
}

std::optional<RowVectorPtr> CudfIcebergSplitReader::next(uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  VELOX_CHECK(splitReader_, "No split reader present");

  if (!splitReader_->has_next()) {
    return std::nullopt;
  }

  auto tableWithMetadata = splitReader_->read_chunk();
  auto cudfTable = std::move(tableWithMetadata.tbl);
  auto nRows = cudfTable->num_rows();

  const bool hasPositionalDeletes = !positionalDeleteFileReaders_.empty();

  if (dvReader_ && nRows > 0) {
    cudfTable = dvReader_->applyDeletionVector(
        cudfTable->view(), baseReadOffset_, stream_, get_output_mr());
  }

  if (cudfTable->num_rows() > 0 && !positionalDeleteFileReaders_.empty()) {
    cudfTable =
        applyPositionalDeletes(cudfTable->view(), stream_, get_temp_mr());
  }

  if (cudfTable->num_rows() > 0 && !equalityDeleteFileReaders_.empty()) {
    auto rowVector = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_, outputType_->names(), stream_, get_temp_mr());
    stream_.synchronize();
    VELOX_CHECK_NOT_NULL(
        rowVector, "Failed to get RowVector for equality delete application.");
    cudfTable = applyEqualityDeletes(
        cudfTable->view(), rowVector, stream_, get_output_mr());
  }

  auto nRowsAfterDeletes = cudfTable->num_rows();
  auto output = cudfIsRegistered() ? std::make_shared<CudfVector>(
                                         pool_,
                                         outputType_,
                                         nRowsAfterDeletes,
                                         std::move(cudfTable),
                                         stream_)
                                   : with_arrow::toVeloxColumn(
                                         cudfTable->view(),
                                         pool_,
                                         outputType_->names(),
                                         stream_,
                                         get_temp_mr());
  stream_.synchronize();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");
  completedRows_ += output->size();
  baseReadOffset_ += nRows;
  return output;
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::applyPositionalDeletes(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = input.num_rows();

  const auto numWords = cudf::num_bitmask_words(numRows);
  auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);
  BufferPtr deleteBitmap = AlignedBuffer::allocate<int8_t>(
      numBitmaskBytes, connectorQueryCtx_->memoryPool(), /*initValue=*/0);

  for (auto iter = positionalDeleteFileReaders_.begin();
       iter != positionalDeleteFileReaders_.end();) {
    (*iter)->readDeletePositions(baseReadOffset_, numRows, deleteBitmap);

    if ((*iter)->noMoreData()) {
      iter = positionalDeleteFileReaders_.erase(iter);
    } else {
      ++iter;
    }
  }

  // Copy the delete bitmap to device, convert to bools, then negate:
  // readDeletePositions sets bits for deleted rows, but apply_boolean_mask
  // keeps rows where the mask is true (surviving rows).
  auto* hostBits = deleteBitmap->as<uint8_t>();

  rmm::device_buffer deviceMask(numBitmaskBytes, stream, get_temp_mr());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceMask.data(),
      hostBits,
      numBitmaskBytes,
      cudaMemcpyHostToDevice,
      stream.value()));

  auto deletedBools = cudf::mask_to_bools(
      static_cast<cudf::bitmask_type const*>(deviceMask.data()),
      0,
      numRows,
      stream,
      get_temp_mr());

  auto survivingBools = cudf::unary_operation(
      deletedBools->view(), cudf::unary_operator::NOT, stream_, get_temp_mr());

  return cudf::apply_boolean_mask(input, survivingBools->view(), stream, mr);
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::applyEqualityDeletes(
    cudf::table_view input,
    const RowVectorPtr& rowVector,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = input.num_rows();
  const auto numWords = cudf::num_bitmask_words(numRows);
  auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);
  BufferPtr deleteBitmap = AlignedBuffer::allocate<int8_t>(
      numBitmaskBytes, connectorQueryCtx_->memoryPool(), /*initValue=*/0);

  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(rowVector, deleteBitmap);
  }

  auto* hostBits = deleteBitmap->as<uint8_t>();

  rmm::device_buffer deviceMask(numBitmaskBytes, stream, get_temp_mr());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceMask.data(),
      hostBits,
      numBitmaskBytes,
      cudaMemcpyHostToDevice,
      stream.value()));

  auto deletedBools = cudf::mask_to_bools(
      static_cast<cudf::bitmask_type const*>(deviceMask.data()),
      0,
      numRows,
      stream,
      get_temp_mr());

  auto survivingBools = cudf::unary_operation(
      deletedBools->view(), cudf::unary_operator::NOT, stream, get_temp_mr());

  return cudf::apply_boolean_mask(input, survivingBools->view(), stream, mr);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
