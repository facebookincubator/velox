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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/BufferUtil.h"

#include <cudf/null_mask.hpp>
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
  deletionVectorReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  baseReadOffset_ = 0;

  // Note: Must setup delete file readers before calling base `prepareSplit` so
  // that it can correctly determine the memory resource to construct the cuDF
  // reader.
  setupDeleteFileReaders();

  // Call base to setup stream, datasource, options, and the cudf reader
  CudfSplitReader::prepareSplit();
}

rmm::device_async_resource_ref
CudfIcebergSplitReader::determineCudfMemoryResource() {
  // If we will be applying any deletes, use temporary mr to read table chunks
  // from cuDF readers. Otherwise, use the output mr.
  return (deletionVectorReader_ or positionalDeleteFileReaders_.size() or
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

  // Number of table rows read by cuDF
  const auto numRows = cudfTable->num_rows();

  // Determine if we will be applying any deletes
  const auto willApplyDeletes = deletionVectorReader_ or
      positionalDeleteFileReaders_.size() or equalityDeleteFileReaders_.size();
  const auto rowMaskUnallocated = not rowMask_ or rowMask_->size() < numRows;
  if (willApplyDeletes and rowMaskUnallocated) {
    rowMask_ =
        std::make_shared<rmm::device_buffer>(numRows, stream_, get_temp_mr());
  }

  // Apply deletion vector, if any
  if (deletionVectorReader_ and cudfTable->num_rows() > 0) {
    // If we will be applying positional or equality deletes later, use the
    // temporary mr. Otherwise, use the output mr.
    mr =
        positionalDeleteFileReaders_.size() or equalityDeleteFileReaders_.size()
        ? get_temp_mr()
        : output_mr;
    cudfTable = deletionVectorReader_->applyDeletionVector(
        cudfTable->view(), baseReadOffset_, rowMask_, stream_, mr);

    // Reset the deletion vector reader if we have read the entire bitmap.
    if (deletionVectorReader_->noMoreData()) {
      deletionVectorReader_.reset();
    }
  }

  // Initialize the delete bitmap if we will be applying positional or equality
  // deletes now
  const auto willApplyPositionOrEqualityDeletes = cudfTable->num_rows() > 0 and
      (positionalDeleteFileReaders_.size() or
       equalityDeleteFileReaders_.size());
  if (willApplyPositionOrEqualityDeletes) {
    const auto numWords = cudf::num_bitmask_words(numRows);
    const auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);
    dwio::common::ensureCapacity<int8_t>(
        deleteBitmap_,
        numBitmaskBytes,
        connectorQueryCtx_->memoryPool(),
        false,
        true);

    if (not deviceDeleteBitmap_ or
        deviceDeleteBitmap_->size() < numBitmaskBytes) {
      deviceDeleteBitmap_ = std::make_shared<rmm::device_buffer>(
          numBitmaskBytes, stream_, get_temp_mr());
    }
  }

  // Apply positional deletes, if any
  if (cudfTable->num_rows() > 0 and not positionalDeleteFileReaders_.empty()) {
    // If we will be applying equality deletes later, use the temporary mr.
    // Otherwise, use the output mr.
    mr = equalityDeleteFileReaders_.size() ? get_temp_mr() : output_mr;
    cudfTable = applyPositionalDeletes(cudfTable->view(), mr);
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
    cudfTable = applyEqualityDeletes(cudfTable->view(), rowVector, output_mr);
  }

  // Update the base read offset
  baseReadOffset_ += numRows;

  return cudfTable;
}

void CudfIcebergSplitReader::setupDeleteFileReaders() {
  // TODO(mh): We currently read a data files as a single split.
  constexpr uint64_t splitOffset = 0;

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

      // Skip the delete file if all delete positions are before this split.
      // TODO: Skip delete files where all positions are after the split, if
      // split row count becomes available.
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

      // Resolve equalityFieldIds to column names and types. In Iceberg,
      // field IDs for top-level columns are assigned sequentially starting
      // from 1, matching the column order in the table schema.
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
            std::make_unique<CudfEqualityDeleteFileReader>(
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
      if (deleteFile.recordCount == 0) {
        continue;
      }
      if (shouldSkipBySequenceNumber(
              deleteFile.dataSequenceNumber,
              icebergSplit_->dataSequenceNumber,
              /*isEqualityDelete=*/false)) {
        continue;
      }
      deletionVectorReader_ =
          std::make_unique<CudfDeletionVectorReader>(deleteFile, splitOffset);
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::applyPositionalDeletes(
    cudf::table_view input,
    rmm::device_async_resource_ref output_mr) {
  // Read positional delete positions into the deleteBitmap_
  const auto numRows = input.num_rows();
  for (auto iter = positionalDeleteFileReaders_.begin();
       iter != positionalDeleteFileReaders_.end();) {
    (*iter)->readDeletePositions(baseReadOffset_, numRows, deleteBitmap_);
    if ((*iter)->noMoreData()) {
      iter = positionalDeleteFileReaders_.erase(iter);
    } else {
      ++iter;
    }
  }

  VELOX_CHECK_NOT_NULL(deleteBitmap_->as<uint8_t>());

  VELOX_CHECK_NOT_NULL(deviceDeleteBitmap_->data());
  VELOX_CHECK_GE(
      deviceDeleteBitmap_->size(),
      cudf::num_bitmask_words(numRows) * sizeof(cudf::bitmask_type));
  VELOX_CHECK_NOT_NULL(rowMask_->data());
  VELOX_CHECK_GE(rowMask_->size(), numRows);

  return applyDeleteBitmap(
      input,
      deleteBitmap_->as<uint8_t>(),
      deviceDeleteBitmap_,
      rowMask_,
      stream_,
      get_temp_mr(),
      output_mr);
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::applyEqualityDeletes(
    cudf::table_view input,
    const RowVectorPtr& rowVector,
    rmm::device_async_resource_ref output_mr) {
  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(rowVector, deleteBitmap_);
  }

  VELOX_CHECK_NOT_NULL(deleteBitmap_->as<uint8_t>());

  const auto numRows = input.num_rows();
  VELOX_CHECK_NOT_NULL(deviceDeleteBitmap_->data());
  VELOX_CHECK_GE(
      deviceDeleteBitmap_->size(),
      cudf::num_bitmask_words(numRows) * sizeof(cudf::bitmask_type));
  VELOX_CHECK_NOT_NULL(rowMask_->data());
  VELOX_CHECK_GE(rowMask_->size(), numRows);

  return applyDeleteBitmap(
      input,
      deleteBitmap_->as<uint8_t>(),
      deviceDeleteBitmap_,
      rowMask_,
      stream_,
      get_temp_mr(),
      output_mr);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
