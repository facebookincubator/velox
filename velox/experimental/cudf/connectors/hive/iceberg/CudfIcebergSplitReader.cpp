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
#include "velox/experimental/cudf/expression/AstUtils.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/type/Type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/Conv.h>
#include <folly/lang/Bits.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

// Returns true if a delete/update file should be skipped based on sequence
// number conflict resolution. Per the Iceberg spec (V2+):
//   - Equality deletes apply when deleteSeqNum > dataSeqNum (i.e., skip when
//     deleteSeqNum <= dataSeqNum).
//   - Positional deletes, deletion vectors, and positional updates apply when
//     deleteSeqNum >= dataSeqNum (i.e., skip when deleteSeqNum < dataSeqNum),
//     because same-snapshot positional deletes SHOULD apply.
//   - A sequence number of 0 means "unassigned" (legacy V1 tables) and
//     disables filtering (never skip).
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
    cudf::ast::expression const* subfieldFilterExpr)
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
          subfieldFilterExpr),
      icebergSplit_(std::move(icebergSplit)),
      hiveConfig_(hiveConfig) {}

void CudfIcebergSplitReader::resetSplit() {
  deletionVectorReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  extraEqualityColumns_.clear();
  injectedColumns_.clear();
  fileColumnNames_.clear();
  splitRowCount_ = 0;
  noColumnsToRead_ = false;
  syntheticTableProduced_ = false;
  deferSubfieldFilter_ = false;
  pushdownFilter_.reset();
  baseReadOffset_ = 0;
  deleteBitmap_ = nullptr;
  deviceBitmap_.reset();
  deleteMask_.reset();
}

void CudfIcebergSplitReader::setupReader() {
  if (not noColumnsToRead_) {
    CudfSplitReader::setupReader();
  }
}

cudf::ast::expression const* CudfIcebergSplitReader::pushdownFilter() const {
  if (pushdownFilter_) {
    return pushdownFilter_->expression();
  }
  return deferSubfieldFilter_ ? nullptr : subfieldFilter();
}

void CudfIcebergSplitReader::prepareSplit(
    dwio::common::RuntimeStatistics& runtimeStats) {
  // Reset delete readers and column injection
  resetSplit();

  // Read file metadata and cache schema information
  cacheSchemaFromMetadata();

  // Must setup delete file readers before reader construction so that it
  // can correctly determine the memory resource to construct the cuDF reader.
  setupDeleteFileReaders(runtimeStats);

  // Include any extra equality delete key columns that are not already in the
  // output projection.
  setupEqualityColumnKeys();

  // Detect info, (Hive-migrated) partition, and schema-evolution columns;
  // remove them from `readColumnNames_` so the parquet reader skips them, and
  // record them for post-read injection.
  adaptColumns();

  // Determine if there are no columns to read.
  noColumnsToRead_ = readColumnNames_.empty();

  prepareSubfieldFilter();

  // Evaluate after the pushed subfield filter is prepared.
  prependRowIndex_ = needPrependedRowIndex();

  if (deferSubfieldFilter_) {
    VLOG(1)
        << "Subfield filter is deferred to post table read due to missing column references.";
  }

  // Base prepare split. Call at the very end.
  CudfSplitReader::prepareSplit(runtimeStats);
}

rmm::device_async_resource_ref
CudfIcebergSplitReader::determineCudfMemoryResource() const {
  // Use temporary mr when there are filters beyond table read.
  const auto needsTempMr = deferSubfieldFilter_ or deletionVectorReader_ or
      positionalDeleteFileReaders_.size() or equalityDeleteFileReaders_.size();
  return needsTempMr ? get_temp_mr() : get_output_mr();
}

bool CudfIcebergSplitReader::needPrependedRowIndex() const {
  // Not needed when all projected columns are injected.
  if (noColumnsToRead_) {
    return false;
  }

  // Not needed if there are no positional deletes.
  const bool hasPositionalDeletes =
      deletionVectorReader_ or positionalDeleteFileReaders_.size();
  if (not hasPositionalDeletes) {
    return false;
  }

  // Needed if a filter is pushed into the data-file reader.
  return pushdownFilter() != nullptr;
}

void CudfIcebergSplitReader::prepareSubfieldFilter() {
  auto* originalFilter = CudfSplitReader::subfieldFilter();
  if (originalFilter == nullptr or injectedColumns_.empty()) {
    return;
  }

  if (noColumnsToRead_) {
    deferSubfieldFilter_ = true;
    return;
  }

  std::vector<cudf::size_type> injectedColumnIndices;
  injectedColumnIndices.reserve(injectedColumns_.size());
  for (const auto& column : injectedColumns_) {
    injectedColumnIndices.push_back(
        static_cast<cudf::size_type>(column.outputIndex));
  }

  auto transformed = std::make_unique<CudfIcebergExpressionTransformer>(
      *originalFilter, std::move(injectedColumnIndices));
  deferSubfieldFilter_ = transformed->referencesInjectedColumn();
  if (transformed->changed() and transformed->expression() != nullptr) {
    pushdownFilter_ = std::move(transformed);
  }
}

std::unique_ptr<cudf::column> CudfIcebergSplitReader::extractRowIndex(
    std::unique_ptr<cudf::table>& table) const {
  VELOX_USER_CHECK_GE(
      table->num_columns(),
      2,
      "cuDF table must contain a row-index column and at least one data column");
  auto columns = table->release();
  auto rowIndex = std::move(columns.front());
  columns.erase(columns.begin());
  table = std::make_unique<cudf::table>(std::move(columns));
  return rowIndex;
}

std::unique_ptr<cudf::column> CudfIcebergSplitReader::makeRowIndex(
    cudf::size_type numRows) const {
  auto rowIndex = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT64},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream_,
      get_temp_mr());
  fillSequence<uint64_t>(
      rowIndex->mutable_view(),
      baseReadOffset_,
      numRows,
      stream_,
      get_temp_mr());
  return rowIndex;
}

std::pair<std::size_t, std::size_t> CudfIcebergSplitReader::rowRange(
    cudf::column_view rowIndex) const {
  if (not prependRowIndex_) {
    return {baseReadOffset_, static_cast<std::size_t>(rowIndex.size())};
  }

  std::size_t startRow{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &startRow,
      rowIndex.begin<uint64_t>(),
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost,
      stream_.value()));

  if (pushdownFilter() == nullptr) {
    stream_.synchronize();
    return {startRow, static_cast<std::size_t>(rowIndex.size())};
  }

  std::size_t endRow{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &endRow,
      rowIndex.end<uint64_t>() - 1,
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost,
      stream_.value()));
  stream_.synchronize();

  VELOX_CHECK_LE(startRow, endRow);
  return {startRow, static_cast<std::size_t>(endRow - startRow + 1)};
}

std::optional<std::unique_ptr<cudf::table>>
CudfIcebergSplitReader::readNextChunk() {
  std::unique_ptr<cudf::table> cudfTable;
  if (noColumnsToRead_) {
    if (syntheticTableProduced_) {
      return std::nullopt;
    }
    syntheticTableProduced_ = true;
    cudfTable = std::make_unique<cudf::table>(
        std::vector<std::unique_ptr<cudf::column>>{});
  } else {
    // Read the next table chunk from the cuDF reader
    auto chunkOpt = CudfSplitReader::readNextChunk();
    if (not chunkOpt.has_value()) {
      return std::nullopt;
    }
    cudfTable = std::move(chunkOpt.value());
  }

  // Number of table rows before deletes.
  const auto numRows = [&]() {
    // For synthetic tables, return at most 2 billion rows at a time.
    if (noColumnsToRead_) {
      if (std::cmp_less_equal(
              splitRowCount_, std::numeric_limits<cudf::size_type>::max())) {
        return static_cast<cudf::size_type>(splitRowCount_);
      } else {
        // Reset the synthetic table produced flag to allow another chunk.
        syntheticTableProduced_ = false;
        splitRowCount_ -= std::numeric_limits<cudf::size_type>::max();
        return static_cast<cudf::size_type>(
            std::numeric_limits<cudf::size_type>::max());
      }
    } else {
      return cudfTable->num_rows();
    }
  }();

  auto rowIndexColumn = std::unique_ptr<cudf::column>{};
  if (prependRowIndex_) {
    rowIndexColumn = extractRowIndex(cudfTable);
  } else if (
      numRows > 0 and
      (deletionVectorReader_ or positionalDeleteFileReaders_.size())) {
    rowIndexColumn = makeRowIndex(numRows);
  }

  const auto isApplyingDeletes = numRows > 0 and
      (deletionVectorReader_ or positionalDeleteFileReaders_.size() or
       equalityDeleteFileReaders_.size());

  if (isApplyingDeletes) {
    // Allocate deletion mask column if needed (nothing is deleted)
    if (not deleteMask_ or std::cmp_less(deleteMask_->size(), numRows)) {
      auto false_scalar =
          cudf::numeric_scalar<bool>(false, true, stream_, get_temp_mr());
      deleteMask_ = cudf::make_column_from_scalar(
          false_scalar, numRows, stream_, get_temp_mr());
    } else {
      // Clear the bitmap
      CUDF_CUDA_TRY(cudaMemsetAsync(
          deleteMask_->mutable_view().data<bool>(),
          false,
          numRows * sizeof(bool),
          stream_));
    }

    // Set the current mutable view into the deleteMask_ column.
    deleteMaskView_ = cudf::mutable_column_view(
        deleteMask_->type(),
        numRows,
        deleteMask_->mutable_view().data<bool>(),
        nullptr,
        0);

    // Apply deletion vector
    if (deletionVectorReader_) {
      applyDeletionVector(rowIndexColumn->view());
    }
    // Apply positional deletes
    if (positionalDeleteFileReaders_.size()) {
      const auto [startRow, numFileRows] = rowRange(rowIndexColumn->view());
      applyPositionalDeletes(startRow, numFileRows, rowIndexColumn->view());
    }
    // Apply equality deletes
    if (equalityDeleteFileReaders_.size()) {
      VELOX_CHECK_GE(
          cudfTable->num_columns(),
          extraEqualityColumns_.size(),
          "cuDF table must at least contain the extra equality delete key columns");
      applyEqualityDeletes(cudfTable->view());
      // Drop any extra equality-delete key columns.
      if (extraEqualityColumns_.size()) {
        auto columns = cudfTable->release();
        columns.resize(columns.size() - extraEqualityColumns_.size());
        cudfTable = std::make_unique<cudf::table>(std::move(columns));
      }
    }

    // Apply the delete mask if there are remaining physical columns.
    if (cudfTable->num_columns() > 0) {
      const auto deleteMr =
          deferSubfieldFilter_ ? get_temp_mr() : get_output_mr();
      cudfTable = cudf::apply_deletion_mask(
          cudfTable->view(), deleteMaskView_, stream_, deleteMr);
    }
  }

  // Compute the row count override if there are no remaining physical columns.
  std::optional<cudf::size_type> rowCountOverride = std::nullopt;
  if (cudfTable->num_columns() == 0) {
    if (isApplyingDeletes) {
      const auto deletedRows = static_cast<std::size_t>(
          countDeletedRows(deleteMaskView_, stream_, get_temp_mr()));
      VELOX_CHECK_LE(
          deletedRows,
          static_cast<std::size_t>(numRows),
          "Deleted rows exceed input rows for synthetic table. numRows: {}, deletedRows: {}",
          numRows,
          deletedRows);
      rowCountOverride = numRows - static_cast<cudf::size_type>(deletedRows);
    } else {
      rowCountOverride = numRows;
    }
  }

  // Build output table by injecting missing columns at appropriate indices
  const auto injectMr = deferSubfieldFilter_ ? get_temp_mr() : get_output_mr();
  cudfTable =
      buildOutputTable(std::move(cudfTable), injectMr, rowCountOverride);

  // Apply the deferred subfield filter.
  if (deferSubfieldFilter_) {
    auto* filter = CudfSplitReader::subfieldFilter();
    VELOX_CHECK_NOT_NULL(filter);
    auto filterMask = cudf::compute_column(
        cudfTable->view(), *filter, stream_, get_temp_mr());
    cudfTable = cudf::apply_boolean_mask(
        cudfTable->view(), filterMask->view(), stream_, get_output_mr());
  }

  // Update the base read offset
  baseReadOffset_ += numRows;

  return cudfTable;
}

void CudfIcebergSplitReader::setupDeleteFileReaders(
    dwio::common::RuntimeStatistics& runtimeStats) {
  const auto splitOffset = baseReadOffset_;

  // Decodes the position bound from an Iceberg bounds map.
  const auto decodePositionalDeleteBound =
      [](const auto& bounds) -> std::optional<uint64_t> {
    if (auto iter = bounds.find(velox_iceberg::IcebergMetadataColumn::kPosId);
        iter != bounds.end()) {
      auto decodedBound = encoding::Base64::decode(iter->second);
      VELOX_CHECK_EQ(
          decodedBound.size(),
          sizeof(uint64_t),
          "Unexpected decoded size for positional delete bound.");
      uint64_t bound;
      std::memcpy(&bound, decodedBound.data(), sizeof(uint64_t));
      return folly::Endian::little(bound);
    }
    return std::nullopt;
  };

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

      // Prune positional delete files that are before or after the split
      if (auto upperBound = decodePositionalDeleteBound(deleteFile.upperBounds);
          upperBound.has_value() && *upperBound < splitOffset) {
        continue;
      }
      if (auto lowerBound = decodePositionalDeleteBound(deleteFile.lowerBounds);
          lowerBound.has_value() && *lowerBound >= splitOffset &&
          std::cmp_greater_equal(*lowerBound - splitOffset, splitRowCount_)) {
        continue;
      }

      // Pass `splitOffset == 0` as `readDeletePositions` now receives absolute
      // file row positions
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
              runtimeStats,
              /*splitOffset=*/0,
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
                runtimeStats,
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

      // If 'referencedDataFile' is set and does not match the split's
      // data file, log a warning. Do NOT skip the DV: silently dropping
      // a coordinator-shipped DV could mask deletes if the planner and
      // worker disagree on path normalization (trailing slash, scheme
      // prefix like s3:// vs s3a://, percent-encoding). The coordinator
      // already filters per-split, so the worker-side check is purely
      // diagnostic.
      if (!deleteFile.referencedDataFile.empty() &&
          deleteFile.referencedDataFile != split_->filePath) {
        LOG(WARNING)
            << "Iceberg DV referencedDataFile does not match split path. "
            << "Applying DV anyway. referencedDataFile='"
            << deleteFile.referencedDataFile << "' splitPath='"
            << split_->filePath << "'";
      }

      VELOX_USER_CHECK(
          not deletionVectorReader_,
          "CudfIcebergSplitReader encountered multiple deletion vector files for the split");
      deletionVectorReader_ =
          std::make_unique<CudfDeletionVectorReader>(deleteFile);
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

void CudfIcebergSplitReader::applyDeletionVector(cudf::column_view rowIndex) {
  deletionVectorReader_->applyDeletes(
      deleteMaskView_, rowIndex, stream_, get_temp_mr());
}

void CudfIcebergSplitReader::readPositionalDeleteBitmap(
    std::size_t startRow,
    std::size_t numRows) {
  VELOX_CHECK_LE(
      numRows,
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
      "Positional-delete bitmap range exceeds cudf::size_type");
  const auto numWords =
      cudf::num_bitmask_words(static_cast<cudf::size_type>(numRows));
  const auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);
  dwio::common::ensureCapacity<int8_t>(
      deleteBitmap_,
      numBitmaskBytes,
      connectorQueryCtx_->memoryPool(),
      false,
      true);
  // Reset the host delete bitmap
  std::memset(deleteBitmap_->asMutable<uint8_t>(), 0, numBitmaskBytes);
  if (not deviceBitmap_ or deviceBitmap_->size() < numBitmaskBytes) {
    deviceBitmap_ = std::make_shared<rmm::device_buffer>(
        numBitmaskBytes, stream_, get_temp_mr());
  }

  VELOX_CHECK_NOT_NULL(deleteBitmap_->as<uint8_t>());
  VELOX_CHECK_GE(deleteBitmap_->size(), numBitmaskBytes);
  VELOX_CHECK_NOT_NULL(deviceBitmap_->data());
  VELOX_CHECK_GE(deviceBitmap_->size(), numBitmaskBytes);

  for (auto iter = positionalDeleteFileReaders_.begin();
       iter != positionalDeleteFileReaders_.end();) {
    (*iter)->readDeletePositions(startRow, numRows, deleteBitmap_);
    if ((*iter)->noMoreData()) {
      iter = positionalDeleteFileReaders_.erase(iter);
    } else {
      ++iter;
    }
  }

  // Copy the deletion bitmap to device
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceBitmap_->data(),
      deleteBitmap_->as<uint8_t>(),
      numBitmaskBytes,
      cudaMemcpyHostToDevice,
      stream_.value()));
}

void CudfIcebergSplitReader::applyPositionalDeletes(
    std::size_t startRow,
    std::size_t numRows,
    cudf::column_view rowIndex) {
  readPositionalDeleteBitmap(startRow, numRows);
  const auto numWords =
      cudf::num_bitmask_words(static_cast<cudf::size_type>(numRows));
  const auto bitmap = cudf::device_span<const cudf::bitmask_type>(
      static_cast<const cudf::bitmask_type*>(deviceBitmap_->data()), numWords);
  applyBitmapToMask(
      bitmap,
      startRow,
      numRows,
      rowIndex,
      deleteMaskView_,
      stream_,
      get_temp_mr());
}

void CudfIcebergSplitReader::applyEqualityDeletes(cudf::table_view input) {
  // Apply equality deletes against the running deleteMask_.
  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(input, readColumnNames_, deleteMaskView_, stream_);
  }
}

void CudfIcebergSplitReader::setupEqualityColumnKeys() {
  if (equalityDeleteFileReaders_.empty()) {
    return;
  }

  std::unordered_set<std::string> readColumnSet(
      readColumnNames_.begin(), readColumnNames_.end());
  const auto& dataColumns = tableHandle_->dataColumns();

  // For each equality delete file, find and append any columns that are not
  // already in the readColumnSet
  std::for_each(
      icebergSplit_->deleteFiles.begin(),
      icebergSplit_->deleteFiles.end(),
      [&](const auto& deleteFile) {
        if (deleteFile.content !=
                velox_iceberg::FileContent::kEqualityDeletes or
            deleteFile.equalityFieldIds.empty()) {
          return;
        }
        std::for_each(
            deleteFile.equalityFieldIds.begin(),
            deleteFile.equalityFieldIds.end(),
            [&](const auto& equalityFieldId) {
              const auto columnIdx = static_cast<uint32_t>(equalityFieldId - 1);
              if (dataColumns and columnIdx < dataColumns->size()) {
                const auto& columnName = dataColumns->nameOf(columnIdx);
                if (icebergSplit_->partitionKeys.contains(columnName) or
                    not fileColumnNames_.contains(columnName)) {
                  VELOX_NYI(
                      "Equality deletes on partition columns or columns "
                      "missing from the data file are not yet supported: {}",
                      columnName);
                }
                // Insert column name into readColumnSet if not already present
                if (readColumnSet.insert(columnName).second) {
                  extraEqualityColumns_.push_back(columnName);
                }
              }
            });
      });

  // Append extra columns to readColumnNames_ so the Parquet reader fetches
  // them.
  readColumnNames_.insert(
      readColumnNames_.end(),
      extraEqualityColumns_.begin(),
      extraEqualityColumns_.end());
}

void CudfIcebergSplitReader::cacheSchemaFromMetadata() {
  // Read file metadatas if not already
  fileMetaDatas();

  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "Expected a single parquet footer for Iceberg data file");
  const auto& meta = fileMetaData_.front();
  VELOX_CHECK(not meta.schema.empty(), "Parquet footer schema is empty");
  VELOX_CHECK_GE(meta.num_rows, 0, "Parquet footer reports negative row count");
  std::tie(baseReadOffset_, splitRowCount_) = computeSplitRowRange();

  const auto& root = meta.schema.front();
  fileColumnNames_.clear();
  fileColumnNames_.reserve(root.children_idx.size());
  for (const auto childIdx : root.children_idx) {
    VELOX_CHECK_LT(
        childIdx,
        meta.schema.size(),
        "Parquet schema child index out of range");
    fileColumnNames_.insert(meta.schema[childIdx].name);
  }
}

std::pair<std::size_t, std::size_t>
CudfIcebergSplitReader::computeSplitRowRange() const {
  // Note: This function implements the same logic as cuDF's hybrid scan
  // reader's `filter_row_groups_with_byte_range()` API
  const auto rowGroupOffset = [](const auto& rowGroup) {
    if (rowGroup.file_offset.has_value()) {
      return rowGroup.file_offset.value();
    }
    if (rowGroup.columns.front().file_offset != 0) {
      return rowGroup.columns.front().file_offset;
    }
    const auto& column = rowGroup.columns.front().meta_data;
    return column.dictionary_page_offset != 0
        ? std::min(column.dictionary_page_offset, column.data_page_offset)
        : column.data_page_offset;
  };

  std::size_t startRow{0};
  std::size_t numRows{0};
  for (const auto& rowGroup : fileMetaData_.front().row_groups) {
    const auto offset = rowGroupOffset(rowGroup);
    if (offset < split_->start) {
      startRow += rowGroup.num_rows;
    } else if (offset - split_->start < split_->size()) {
      numRows += rowGroup.num_rows;
    }
  }
  return {startRow, numRows};
}

void CudfIcebergSplitReader::adaptColumns() {
  // Skip trailing equality-delete keys and classify output + filter-only
  // columns only.
  VELOX_CHECK_GE(
      readColumnNames_.size(),
      extraEqualityColumns_.size(),
      "Column projection must at least include the equality delete keys");
  const size_t schemaSize =
      readColumnNames_.size() - extraEqualityColumns_.size();

  std::unordered_set<std::string> injectedNames;
  for (size_t i = 0; i < schemaSize; ++i) {
    const auto& fieldName = readColumnNames_[i];
    const TypePtr veloxType = [&]() -> TypePtr {
      if (i < outputType_->size()) {
        VELOX_DCHECK_EQ(fieldName, outputType_->nameOf(i));
        return outputType_->childAt(i);
      }
      // Filter-only column beyond the output projection.
      const auto& dataColumns = tableHandle_->dataColumns();
      VELOX_CHECK(
          dataColumns and dataColumns->containsChild(fieldName),
          "Filter-only column missing from table schema: {}",
          fieldName);
      return dataColumns->findChild(fieldName);
    }();

    if (auto iter = split_->infoColumns.find(fieldName);
        iter != split_->infoColumns.end()) {
      injectedColumns_.push_back({i, fieldName, iter->second, veloxType});
      injectedNames.insert(fieldName);
    } else if (auto it = icebergSplit_->partitionKeys.find(fieldName);
               it != icebergSplit_->partitionKeys.end()) {
      // Partition columns: Hive migrated table. In Hive-written data
      // files, partition column values are stored in partition metadata
      // rather than in the data file itself, following Hive's
      // partitioning convention.
      injectedColumns_.push_back({i, fieldName, it->second, veloxType});
      injectedNames.insert(fieldName);
    } else if (not fileColumnNames_.contains(fieldName)) {
      // Schema evolution: Column was added after the data file was written
      // and doesn't exist in older data files.
      injectedColumns_.push_back({i, fieldName, std::nullopt, veloxType});
      injectedNames.insert(fieldName);
    }
  }

  // Remove all injected columns from readColumnNames_
  if (not injectedColumns_.empty()) {
    std::erase_if(readColumnNames_, [&injectedNames](const auto& name) {
      return injectedNames.contains(name);
    });
    // Sort injected columns by assembled-table index once here
    std::sort(
        injectedColumns_.begin(),
        injectedColumns_.end(),
        [](const auto& a, const auto& b) {
          return a.outputIndex < b.outputIndex;
        });
  }
}

std::unique_ptr<cudf::scalar> CudfIcebergSplitReader::makeInjectedScalar(
    const InjectedColumn& col) const {
  const bool readAsLocalTime =
      hiveConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());
  try {
    // DATE values arrive in two format-disjoint encodings: Iceberg-native
    // days-since-epoch integers (e.g. "20244") or Hive-migrated date strings
    // (e.g. "2025-06-05"). A bare integer is unambiguously days-since-epoch
    // (date strings contain '-' separators that fail an integer parse).
    const bool isDaysSinceEpoch = col.veloxType->isDate() and
        col.partitionValue.has_value() and
        folly::tryTo<int32_t>(col.partitionValue.value()).hasValue();
    const VectorPtr constant = velox::connector::hive::newConstantFromString(
        col.veloxType,
        col.partitionValue,
        connectorQueryCtx_->memoryPool(),
        readAsLocalTime,
        isDaysSinceEpoch);
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        cudf_velox::createCudfScalar,
        col.veloxType->kind(),
        constant,
        std::nullopt,
        stream_);
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Bad partition value. Column: {}, type: {}, value: {}, error: {}",
        col.name,
        col.veloxType->toString(),
        col.partitionValue.value_or("<null>"),
        e.what());
  }
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::buildOutputTable(
    std::unique_ptr<cudf::table>&& table,
    rmm::device_async_resource_ref mr,
    std::optional<cudf::size_type> rowCountOverride) const {
  const auto numRows = rowCountOverride.value_or(table->num_rows());
  auto columns = rowCountOverride.has_value()
      ? std::vector<std::unique_ptr<cudf::column>>{}
      : table->release();

  // Interleave file columns and injected constants at their outputIndex.
  const auto totalColumns = columns.size() + injectedColumns_.size();
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(totalColumns);

  auto injectedColIter = injectedColumns_.begin();
  auto dataColIter = columns.begin();

  for (size_t outIdx = 0; outIdx < totalColumns; ++outIdx) {
    if (injectedColIter != injectedColumns_.end() &&
        injectedColIter->outputIndex == outIdx) {
      auto scalar = makeInjectedScalar(*injectedColIter++);
      output.push_back(
          cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
    } else {
      VELOX_CHECK(
          dataColIter != columns.end(),
          "Data column index out of range during column injection");
      output.push_back(std::move(*dataColIter++));
    }
  }

  VELOX_CHECK(
      dataColIter == columns.end(),
      "Not all data columns were consumed during column injection");
  VELOX_CHECK(
      injectedColIter == injectedColumns_.end(),
      "Not all injected columns were consumed during column injection");

  return std::make_unique<cudf::table>(std::move(output));
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
