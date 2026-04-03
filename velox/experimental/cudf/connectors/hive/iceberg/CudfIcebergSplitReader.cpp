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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <folly/lang/Bits.h>
#include <nvtx3/nvtx3.hpp>

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
      if (deleteFile.recordCount == 0 ||
          deleteFile.equalityFieldIds.empty()) {
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

  if (dvReader_ && nRows > 0) {
    cudfTable = dvReader_->applyDeletionVector(
        cudfTable->view(), baseReadOffset_, stream_, get_output_mr());
  }

  auto nRowsAfterDv = cudfTable->num_rows();

  // Trim extra columns if the parquet file has more than we need.
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(outputColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  }

  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRowsAfterDv, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(),
            pool_,
            outputType_->names(),
            stream_,
            get_temp_mr());
  stream_.synchronize();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  bool hasPositionalDeletes = !positionalDeleteFileReaders_.empty();
  bool hasEqualityDeletes = !equalityDeleteFileReaders_.empty();

  if (hasPositionalDeletes || hasEqualityDeletes) {
    auto rowVector = std::dynamic_pointer_cast<RowVector>(output);
    if (!rowVector) {
      auto cudfVec = std::dynamic_pointer_cast<CudfVector>(output);
      if (cudfVec) {
        rowVector = with_arrow::toVeloxColumn(
            cudfVec->getTableView(),
            pool_,
            outputType_->names(),
            stream_,
            get_temp_mr());
        stream_.synchronize();
      }
    }
    VELOX_CHECK_NOT_NULL(
        rowVector,
        "Failed to get RowVector for delete application.");

    if (hasPositionalDeletes) {
      rowVector = applyPositionalDeletes(rowVector);
    }
    if (hasEqualityDeletes) {
      rowVector = applyEqualityDeletes(rowVector);
    }

    completedRows_ += rowVector->size();
    baseReadOffset_ += nRows;
    return rowVector;
  }

  completedRows_ += output->size();
  baseReadOffset_ += nRows;
  return output;
}

RowVectorPtr CudfIcebergSplitReader::applyPositionalDeletes(
    RowVectorPtr output) {
  auto numRows = output->size();
  if (numRows == 0) {
    return output;
  }

  auto numBytes = bits::nbytes(numRows);
  BufferPtr deleteBitmap = AlignedBuffer::allocate<int8_t>(
      numBytes, connectorQueryCtx_->memoryPool(), /*initValue=*/0);

  for (auto iter = positionalDeleteFileReaders_.begin();
       iter != positionalDeleteFileReaders_.end();) {
    (*iter)->readDeletePositions(baseReadOffset_, numRows, deleteBitmap);

    if ((*iter)->noMoreData()) {
      iter = positionalDeleteFileReaders_.erase(iter);
    } else {
      ++iter;
    }
  }

  auto* bitmap = deleteBitmap->as<uint8_t>();
  vector_size_t numDeleted = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (bits::isBitSet(bitmap, i)) {
      ++numDeleted;
    }
  }

  if (numDeleted == 0) {
    return output;
  }

  vector_size_t numSurviving = numRows - numDeleted;
  if (numSurviving == 0) {
    return std::dynamic_pointer_cast<RowVector>(BaseVector::create(
        output->type(), 0, connectorQueryCtx_->memoryPool()));
  }

  std::vector<BaseVector::CopyRange> ranges;
  ranges.reserve(numSurviving);
  vector_size_t targetIdx = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (!bits::isBitSet(bitmap, i)) {
      ranges.push_back({i, targetIdx++, 1});
    }
  }

  auto newOutput = BaseVector::create(
      output->type(), numSurviving, connectorQueryCtx_->memoryPool());
  auto newRowOutput = std::dynamic_pointer_cast<RowVector>(newOutput);
  for (auto i = 0; i < output->childrenSize(); ++i) {
    newRowOutput->childAt(i)->resize(numSurviving);
    newRowOutput->childAt(i)->copyRanges(output->childAt(i).get(), ranges);
  }
  newRowOutput->resize(numSurviving);

  return newRowOutput;
}

RowVectorPtr CudfIcebergSplitReader::applyEqualityDeletes(
    RowVectorPtr output) {
  auto numRows = output->size();
  if (numRows == 0) {
    return output;
  }

  BufferPtr eqDeleteBitmap = AlignedBuffer::allocate<bool>(
      numRows, connectorQueryCtx_->memoryPool());
  std::memset(
      eqDeleteBitmap->asMutable<uint8_t>(), 0, eqDeleteBitmap->size());

  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(output, eqDeleteBitmap);
  }

  auto* eqBitmap = eqDeleteBitmap->as<uint8_t>();
  vector_size_t numDeleted = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (bits::isBitSet(eqBitmap, i)) {
      ++numDeleted;
    }
  }

  if (numDeleted == 0) {
    return output;
  }

  vector_size_t numSurviving = numRows - numDeleted;
  if (numSurviving == 0) {
    return std::dynamic_pointer_cast<RowVector>(BaseVector::create(
        output->type(), 0, connectorQueryCtx_->memoryPool()));
  }

  std::vector<BaseVector::CopyRange> ranges;
  ranges.reserve(numSurviving);
  vector_size_t targetIdx = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (!bits::isBitSet(eqBitmap, i)) {
      ranges.push_back({i, targetIdx++, 1});
    }
  }

  auto newOutput = BaseVector::create(
      output->type(), numSurviving, connectorQueryCtx_->memoryPool());
  auto newRowOutput = std::dynamic_pointer_cast<RowVector>(newOutput);
  for (auto i = 0; i < output->childrenSize(); ++i) {
    newRowOutput->childAt(i)->resize(numSurviving);
    newRowOutput->childAt(i)->copyRanges(output->childAt(i).get(), ranges);
  }
  newRowOutput->resize(numSurviving);

  return newRowOutput;
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
