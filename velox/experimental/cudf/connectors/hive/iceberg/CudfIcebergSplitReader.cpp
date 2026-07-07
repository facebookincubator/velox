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
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/Conv.h>
#include <folly/lang/Bits.h>

#include <cstring>
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
      hiveConfig_(hiveConfig) {
  // Extract physical names of target data columns from readColumnNames
  VELOX_CHECK_GE(readColumnNames.size(), outputType->size());
  physicalNames_.assign(
      readColumnNames.begin(), readColumnNames.begin() + outputType->size());
}

void CudfIcebergSplitReader::prepareSplit(
    dwio::common::RuntimeStatistics& runtimeStats) {
  // Clear existing delete file readers and other state
  deletionVectorReader_.reset();
  equalityDeleteFileReaders_.clear();
  extraEqualityColumns_.clear();
  injectedColumns_.clear();
  fileColumnNames_.clear();
  fileColumnIndex_.clear();
  fileRowCount_.reset();
  noColumnsToRead_ = false;
  syntheticTableProduced_ = false;
  baseReadOffset_ = 0;

  // Must setup delete file readers before calling base `prepareSplit` so
  // that it can correctly determine the memory resource to construct the cuDF
  // reader. Pass the DataSource's runtimeStats so delete readers accumulate
  // stats directly into the DataSource's stats object.
  setupDeleteFileReaders(runtimeStats);

  // Subfield filters along with positional deletes are not supported as the
  // original row positions may have been altered by the filter.
  // TODO(mh): Re-enable when https://github.com/rapidsai/cudf/pull/23077 merges
  if (hasSubfieldFilter() and
      (deletionVectorReader_ or positionalDeleteFileReaders_.size())) {
    VELOX_NYI(
        "cuDF Iceberg reader does not yet support subfield filters together with positional deletes");
  }

  // Setup column projection to include any equality delete key columns that
  // are not already in the output projection. Must be called before base
  // `prepareSplit`
  setupColumnProjection();

  // Detect info and (Hive-migrated) partition columns and remove them from
  // `readColumnNames_` so the parquet reader doesn't try to read them, and
  // record them for post-read injection.
  adaptColumns();

  // Call base to setup stream, datasource, options, and the cudf reader
  CudfSplitReader::prepareSplit(runtimeStats);
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

std::optional<cudf::io::table_with_metadata>
CudfIcebergSplitReader::readNextChunk(
    rmm::device_async_resource_ref output_mr) {
  auto chunkOpt = [&]() -> std::optional<std::unique_ptr<cudf::table>> {
    // If every output column is injected, skip `readNextChunk` and
    // emit a single chunk synthesized.
    if (noColumnsToRead_) {
      // No more synthetic table chunks
      if (syntheticTableProduced_) {
        return std::nullopt;
      }
      syntheticTableProduced_ = true;
      return std::make_unique<cudf::table>(
          std::vector<std::unique_ptr<cudf::column>>{});
    }

    // Otherwise, read the next table chunk from the cuDF reader.
    auto chunkOpt =
        CudfSplitReader::readNextChunk(determineCudfMemoryResource());
    // No more table chunks
    if (not chunkOpt.has_value()) {
      return std::nullopt;
    }
    // Lazily cache column names and their positions.
    ensureFileColumnIndex(chunkOpt.value().metadata.schema_info);
    return std::move(chunkOpt.value().tbl);
  }();

  // No more table chunks
  if (not chunkOpt.has_value()) {
    return std::nullopt;
  }
  auto cudfTable = std::move(chunkOpt.value());

  // Check if all projected columns are schema-evolution (missing) columns
  const bool allColumnsMissing = cudfTable->num_columns() == 0;

  // Number of rows before any deletes from cuDF table or parquet footer if all
  // columns are missing
  const auto numRows = allColumnsMissing
      ? fileRowCount()
      : static_cast<size_t>(cudfTable->num_rows());

  // Determine if we are applying any deletes
  const auto isApplyingDeletes = numRows > 0 and
      (deletionVectorReader_ or positionalDeleteFileReaders_.size() or
       equalityDeleteFileReaders_.size());

  if (isApplyingDeletes) {
    // Equality deletes require reading their key columns, so
    // `allColumnsMissing` must be false
    VELOX_CHECK(
        not allColumnsMissing or equalityDeleteFileReaders_.empty(),
        "Equality deletes are not expected when all projected columns are missing from the file");

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
      applyDeletionVector(numRows);
    }
    // Apply positional deletes
    if (positionalDeleteFileReaders_.size()) {
      applyPositionalDeletes(numRows);
    }
    // Apply equality deletes
    if (equalityDeleteFileReaders_.size()) {
      applyEqualityDeletes(cudfTable->view());
    }

    // Only apply deletes if there are physical columns to filter
    if (not allColumnsMissing) {
      cudfTable = cudf::apply_deletion_mask(
          cudfTable->view(), deleteMaskView_, stream_, get_output_mr());
    }
  }

  // Reorder read columns, inject info and partition columns and fill
  // schema-evolution columns with typed NULLs. This must run even for 0-row
  // tables so post-delete empty chunks still have the expected number and order
  // of output columns.

  // Compute the row count override if all projected columns are missing
  auto rowCountOverride = allColumnsMissing
      ? std::optional<cudf::size_type>(
            numRows - countDeletedRows(deleteMaskView_, stream_, get_temp_mr()))
      : std::nullopt;

  // Build the output table with an optional row count override indicating if
  // all projected columns are missing
  cudfTable =
      buildOutputTable(std::move(cudfTable), output_mr, rowCountOverride);

  // Update the base read offset
  baseReadOffset_ += numRows;

  return cudf::io::table_with_metadata{std::move(cudfTable), {}};
}

void CudfIcebergSplitReader::setupDeleteFileReaders(
    dwio::common::RuntimeStatistics& runtimeStats) {
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
              runtimeStats,
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
          std::make_unique<CudfDeletionVectorReader>(deleteFile, splitOffset);
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

void CudfIcebergSplitReader::applyDeletionVector(std::size_t numRows) {
  // Apply deletion vector into the deleteMask_
  deletionVectorReader_->applyDeletes(
      deleteMaskView_, baseReadOffset_, numRows, stream_, get_temp_mr());
}

void CudfIcebergSplitReader::applyPositionalDeletes(std::size_t numRows) {
  // Apply positional deletes into the deleteMask_

  // Initialize host and device delete bitmaps
  const auto numWords = cudf::num_bitmask_words(numRows);
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
    (*iter)->readDeletePositions(baseReadOffset_, numRows, deleteBitmap_);
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

  // Apply the deletion bitmap to the row mask
  applyBitmapToMask(
      cudf::device_span<cudf::bitmask_type>(
          static_cast<cudf::bitmask_type*>(deviceBitmap_->data()), numWords),
      deleteMaskView_,
      stream_,
      get_temp_mr());
}

void CudfIcebergSplitReader::applyEqualityDeletes(cudf::table_view input) {
  // Apply equality deletes against the running deleteMask_.
  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(input, fileColumnNames_, deleteMaskView_, stream_);
  }
}

void CudfIcebergSplitReader::setupColumnProjection() {
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

void CudfIcebergSplitReader::adaptColumns() {
  // Detect info and partition columns. Split-metadata lookups use the physical
  // names as outputType may carry logical aliases
  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& fieldName = physicalNames_[i];
    const auto& veloxType = outputType_->childAt(i);

    if (auto iter = split_->infoColumns.find(fieldName);
        iter != split_->infoColumns.end()) {
      injectedColumns_.emplace(
          fieldName, InjectedColumn{iter->second, veloxType});
    } else if (auto it = icebergSplit_->partitionKeys.find(fieldName);
               it != icebergSplit_->partitionKeys.end()) {
      // Partition columns: Hive migrated table. In Hive-written data
      // files, partition column values are stored in partition metadata
      // rather than in the data file itself, following Hive's
      // partitioning convention.
      injectedColumns_.emplace(
          fieldName, InjectedColumn{it->second, veloxType});
    }
  }

  // Remove the metadata-sourced columns from readColumnNames_ so the parquet
  // reader does not attempt to read them from the file.
  if (not injectedColumns_.empty()) {
    std::erase_if(readColumnNames_, [this](const auto& name) {
      return injectedColumns_.contains(name);
    });
  }

  // If every output column is injected (info/partition), no columns need to be
  // read from the data file.
  noColumnsToRead_ = readColumnNames_.empty();
}

void CudfIcebergSplitReader::ensureFileColumnIndex(
    const std::vector<cudf::io::column_name_info>& schemaInfo) {
  // The reader returns the same columns in the same order for every chunk, so
  // build the name list and the name -> position lookup only once per split.
  if (not fileColumnNames_.empty()) {
    return;
  }
  fileColumnNames_.reserve(schemaInfo.size());
  fileColumnIndex_.reserve(schemaInfo.size());
  for (size_t i = 0; i < schemaInfo.size(); ++i) {
    fileColumnNames_.push_back(schemaInfo[i].name);
    fileColumnIndex_.emplace(schemaInfo[i].name, i);
  }
}

std::size_t CudfIcebergSplitReader::fileRowCount() {
  if (fileRowCount_.has_value()) {
    return fileRowCount_.value();
  }
  setupCudfDataSource();
  VELOX_CHECK_NOT_NULL(
      dataSource_, "CudfIcebergSplitReader failed to setup a cuDF datasource");

  // TODO(mh): Compute using selected row group when we support sub-splits
  const auto meta =
      cudf::io::read_parquet_metadata(cudf::io::source_info{dataSource_.get()});
  VELOX_CHECK_LE(
      meta.num_rows(),
      std::numeric_limits<cudf::size_type>::max(),
      "File row count exceeds max cudf::size_type value");
  fileRowCount_ = static_cast<std::size_t>(meta.num_rows());
  return fileRowCount_.value();
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::buildOutputTable(
    std::unique_ptr<cudf::table>&& table,
    rmm::device_async_resource_ref mr,
    std::optional<cudf::size_type> rowCountOverride) {
  const auto numRows = rowCountOverride.value_or(table->num_rows());
  auto columns = rowCountOverride.has_value()
      ? std::vector<std::unique_ptr<cudf::column>>{}
      : table->release();
  const bool readAsLocalTime =
      hiveConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());

  // Build a cudf scalar for an info/partition column from its optional string
  // value. A `nullopt` value yields a typed NULL partition key column.
  auto makeConstantScalar =
      [&](const std::string& name,
          const std::optional<std::string>& value,
          const TypePtr& veloxType) -> std::unique_ptr<cudf::scalar> {
    try {
      // DATE values arrive in two format-disjoint encodings: Iceberg-native
      // days-since-epoch integers (e.g. "20244") or Hive-migrated date strings
      // (e.g. "2025-06-05"). A bare integer is unambiguously days-since-epoch
      // (date strings contain '-' separators that fail an integer parse).
      const bool isDaysSinceEpoch = veloxType->isDate() and
          value.has_value() and folly::tryTo<int32_t>(value.value()).hasValue();
      const VectorPtr constant = velox::connector::hive::newConstantFromString(
          veloxType,
          value,
          connectorQueryCtx_->memoryPool(),
          readAsLocalTime,
          isDaysSinceEpoch);
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          cudf_velox::createCudfScalar, veloxType->kind(), constant);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Bad partition value '{}' for column '{}' (type {}): {}",
          value.value_or("<null>"),
          name,
          veloxType->toString(),
          e.what());
    }
  };

  // Assemble the output in output-schema order. Each output column is sourced
  // from exactly one of: an injected info/partition constant, the read data
  // table (by name), or a typed NULL constant (schema evolution). Read columns
  // that are not part of the output schema (equality-delete key columns)
  // are dropped.
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(outputType_->size());

  // Number of read columns referenced in the output table.
  size_t consumedReadColumns = 0;

  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& fieldName = physicalNames_[i];
    const auto& veloxType = outputType_->childAt(i);
    const auto cudfType = cudf_velox::veloxToCudfDataType(veloxType);

    if (auto injectedIt = injectedColumns_.find(fieldName);
        injectedIt != injectedColumns_.end()) {
      // Info column or Hive-migrated partition column: constant value.
      auto scalar =
          makeConstantScalar(fieldName, injectedIt->second.value, veloxType);
      output.push_back(
          cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
    } else if (auto fileIt = fileColumnIndex_.find(fieldName);
               fileIt != fileColumnIndex_.end()) {
      // Column read from the data file.
      output.push_back(std::move(columns[fileIt->second]));
      ++consumedReadColumns;
    } else {
      // Schema evolution: column added after this data file was written and
      // absent from the file. Emit a typed NULL column.
      auto scalar =
          cudf::make_default_constructed_scalar(cudfType, stream_, mr);
      scalar->set_valid_async(false, stream_);
      output.push_back(
          cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
    }
  }

  VELOX_CHECK_EQ(
      output.size(),
      outputType_->size(),
      "Mismatch between number of columns in the output schema and the assembled Iceberg table. Output schema: {}, Assembled columns: {}",
      outputType_->toString(),
      output.size());

  // Append remaining filter-only columns in `readColumnNames_` order.
  const std::unordered_set<std::string> equalityColumnSet(
      extraEqualityColumns_.begin(), extraEqualityColumns_.end());
  size_t filterOnlyColumns = 0;
  for (const auto& name : readColumnNames_) {
    // Drop Equality-delete key columns
    if (equalityColumnSet.contains(name)) {
      continue;
    }
    // Skip missing and already moved-out output columns leaving only
    // filter-only columns.
    auto fileIt = fileColumnIndex_.find(name);
    if (fileIt == fileColumnIndex_.end() or
        columns[fileIt->second] == nullptr) {
      continue;
    }
    output.push_back(std::move(columns[fileIt->second]));
    ++consumedReadColumns;
    ++filterOnlyColumns;
  }

  // Every read column must be accounted for: an output column, a
  // filter-only column, or intentionally dropped (equality-delete key
  // column). Any leftover means a read column was silently dropped (a bug).
  VELOX_CHECK_EQ(
      consumedReadColumns + extraEqualityColumns_.size(),
      columns.size(),
      "Read table columns were not fully accounted for during Iceberg table assembly. "
      "Read columns: {}, Consumed: {} including {} filter-only columns, Equality columns: {}",
      columns.size(),
      consumedReadColumns,
      filterOnlyColumns,
      extraEqualityColumns_.size());

  return std::make_unique<cudf::table>(std::move(output));
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
