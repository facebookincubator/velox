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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"

#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/lang/Bits.h>

#include <cstring>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

/// Returns true if a delete/update file should be skipped based on sequence
/// number conflict resolution. Per the Iceberg spec (V2+):
///   - Equality deletes apply when deleteSeqNum > dataSeqNum (i.e., skip when
///     deleteSeqNum <= dataSeqNum).
///   - Positional deletes, deletion vectors, and positional updates apply when
///     deleteSeqNum >= dataSeqNum (i.e., skip when deleteSeqNum < dataSeqNum),
///     because same-snapshot positional deletes SHOULD apply.
///   - A sequence number of 0 means "unassigned" (legacy V1 tables) and
///     disables filtering (never skip).
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
    std::shared_ptr<CudfHiveConnectorSplit> split,
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
    const std::shared_ptr<IoStats>& ioStats,
    bool useExperimentalCudfReader,
    cudf::ast::expression const* subfieldFilterExpr,
    std::unique_ptr<exec::ExprSet>* remainingFilterExprSet,
    std::shared_ptr<CudfExpression> cudfExpressionEvaluator,
    std::atomic<uint64_t>* totalRemainingFilterTime)
    : CudfSplitReader(
          std::move(split),
          std::move(tableHandle),
          outputType,
          readColumnNames,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig,
          ioStatistics,
          ioStats,
          useExperimentalCudfReader,
          subfieldFilterExpr,
          remainingFilterExprSet,
          std::move(cudfExpressionEvaluator),
          totalRemainingFilterTime),
      icebergSplit_(std::move(icebergSplit)),
      hiveConfig_(hiveConfig) {}

void CudfIcebergSplitReader::prepareSplit() {
  dvReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  baseReadOffset_ = 0;

  // Must load deletion vector and setup delete file readers before calling base
  // method so that `prepareSplit` can use the correct memory resource for the
  // cudf reader
  loadDeletionVector();
  setupDeleteFileReaders();

  // Call base to setup stream, datasource, options, and the cudf reader
  CudfSplitReader::prepareSplit();
}

rmm::device_async_resource_ref
CudfIcebergSplitReader::determineCudfMemoryResource() {
  // If we will be applying any deletes, use temporary mr to create the
  // `chunked_parquet_reader` and when calling `readNextChunk` for the hybrid
  // scan reader.
  return (dvReader_ or positionalDeleteFileReaders_.size() or
          equalityDeleteFileReaders_.size())
      ? get_temp_mr()
      : get_output_mr();
}

void CudfIcebergSplitReader::createCudfReader(
    rmm::device_async_resource_ref output_mr) {
  CudfSplitReader::createCudfReader(determineCudfMemoryResource());
}

std::optional<std::unique_ptr<cudf::table>>
CudfIcebergSplitReader::readNextChunk(
    rmm::device_async_resource_ref output_mr) {
  // Determine the memory resource to use for `readNextChunk`
  auto mr = determineCudfMemoryResource();

  // Read the next table chunk from the cuDF reader
  auto cudfTable = CudfSplitReader::readNextChunk(mr).value_or(nullptr);
  if (not cudfTable) {
    return std::nullopt;
  }

  // Apply deletion vector, if any
  if (dvReader_ and cudfTable->num_rows() > 0) {
    // If we will be applying positional or equality deletes later, use the
    // temporary mr. Otherwise, use the output mr.
    mr =
        positionalDeleteFileReaders_.size() or equalityDeleteFileReaders_.size()
        ? get_temp_mr()
        : output_mr;
    cudfTable = dvReader_->applyDeletionVector(
        cudfTable->view(), baseReadOffset_, stream_, mr);
  }

  // Apply positional deletes, if any
  if (cudfTable->num_rows() > 0 and not positionalDeleteFileReaders_.empty()) {
    // If we will be applying equality deletes later, use the temporary mr.
    // Otherwise, use the output mr.
    mr = equalityDeleteFileReaders_.size() ? get_temp_mr() : output_mr;
    cudfTable = applyPositionalDeletes(cudfTable->view(), stream_, mr);
  }

  // Apply equality deletes, if any
  if (cudfTable->num_rows() > 0 and not equalityDeleteFileReaders_.empty()) {
    // Convert the cudf table to a velox row vector
    auto rowVector = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_, outputType_->names(), stream_, get_temp_mr());
    stream_.synchronize();
    VELOX_CHECK_NOT_NULL(
        rowVector,
        "Failed to convert cudf table to row vector for equality delete application.");
    // Use the output mr as there will be no more deletes after this
    cudfTable =
        applyEqualityDeletes(cudfTable->view(), rowVector, stream_, output_mr);
  }

  // Update the base read offset
  baseReadOffset_ += cudfTable->num_rows();

  return cudfTable;
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
