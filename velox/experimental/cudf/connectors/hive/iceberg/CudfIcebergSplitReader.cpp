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

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

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
      hiveConfig_(hiveConfig) {}

void CudfIcebergSplitReader::prepareSplit(
    dwio::common::RuntimeStatistics& runtimeStats) {
  // Clear existing delete file readers and other state
  deletionVectorReader_.reset();
  positionalDeleteFileReaders_.clear();
  equalityDeleteFileReaders_.clear();
  extraEqualityColumns_.clear();
  injectedColumns_.clear();
  baseReadOffset_ = 0;

  // Must setup delete file readers before calling base `prepareSplit` so
  // that it can correctly determine the memory resource to construct the cuDF
  // reader. Pass the DataSource's runtimeStats so delete readers accumulate
  // stats directly into the DataSource's stats object.
  setupDeleteFileReaders(runtimeStats);

  // Setup column projection to include any equality delete key columns that
  // are not already in the output projection. Must be called before base
  // `prepareSplit`
  setupColumnProjection();

  // Schema evolution: Detect info columns, partition columns, and
  // schema-evolution missing columns. Filters them from readColumnNames_ so the
  // parquet reader doesn't try to read them, and records them for post-read
  // injection.
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

  // Number of table rows read by cuDF (before any deletes)
  const auto numRows = cudfTable->num_rows();

  // Determine if we are applying any deletes
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
      applyDeletionVector(cudfTable->view());
    }
    // Apply positional deletes
    if (positionalDeleteFileReaders_.size()) {
      applyPositionalDeletes(cudfTable->view());
    }
    // Apply equality deletes
    if (equalityDeleteFileReaders_.size()) {
      applyEqualityDeletes(cudfTable->view());
    }

    cudfTable = cudf::apply_deletion_mask(
        cudfTable->view(), deleteMaskView_, stream_, get_output_mr());
  }

  // Strip extra equality delete key columns added by
  // setupColumnProjection()
  if (not extraEqualityColumns_.empty()) {
    auto columns = cudfTable->release();
    VELOX_CHECK_GE(
        columns.size(),
        extraEqualityColumns_.size(),
        "cuDF table has fewer columns than the appended equality delete keys");
    columns.resize(columns.size() - extraEqualityColumns_.size());
    cudfTable = std::make_unique<cudf::table>(std::move(columns));
  }

  // Inject partition columns and schema-evolution NULL columns.
  // This must run even for 0-row tables so post-delete empty chunks still
  // have the expected number and order of output columns.
  if (not injectedColumns_.empty()) {
    cudfTable = injectMissingColumns(std::move(cudfTable), output_mr);
  }

  // Update the base read offset
  baseReadOffset_ += numRows;

  return cudfTable;
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

void CudfIcebergSplitReader::applyDeletionVector(cudf::table_view input) {
  // Apply deletion vector into the deleteMask_
  const auto numRows = input.num_rows();
  deletionVectorReader_->applyDeletes(
      deleteMaskView_, baseReadOffset_, numRows, stream_, get_temp_mr());
}

void CudfIcebergSplitReader::applyPositionalDeletes(cudf::table_view input) {
  // Apply positional deletes into the deleteMask_
  const auto numRows = input.num_rows();

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
  const auto numRows = input.num_rows();

  // Iteratively apply equality deletes, if any
  for (auto& reader : equalityDeleteFileReaders_) {
    reader->applyDeletes(input, readColumnNames_, deleteMaskView_, stream_);
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
  // Detect info columns and partition columns
  std::unordered_set<std::string> injectedNames;
  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& fieldName = outputType_->nameOf(i);

    if (auto iter = split_->infoColumns.find(fieldName);
        iter != split_->infoColumns.end()) {
      injectedColumns_.push_back(
          {i, fieldName, iter->second, outputType_->childAt(i)});
      injectedNames.insert(fieldName);
    } else if (auto it = icebergSplit_->partitionKeys.find(fieldName);
               it != icebergSplit_->partitionKeys.end()) {
      // Partition columns: Hive migrated table. In Hive-written data
      // files, partition column values are stored in partition metadata
      // rather than in the data file itself, following Hive's
      // partitioning convention.
      injectedColumns_.push_back(
          {i, fieldName, it->second, outputType_->childAt(i)});
      injectedNames.insert(fieldName);
    }
  }

  // Detect schema-evolution missing columns by reading the parquet
  // footer. Only needed if there are output columns not yet accounted for.
  if (injectedNames.size() < outputType_->size()) {
    setupCudfDataSource();
    VELOX_CHECK_NOT_NULL(
        dataSource_,
        "CudfIcebergSplitReader failed to setup a cuDF datasource");
    auto meta = cudf::io::read_parquet_metadata(
        cudf::io::source_info{dataSource_.get()});
    const auto& rootSchema = meta.schema().root();
    std::unordered_set<std::string> fileColumns;
    fileColumns.reserve(rootSchema.num_children());
    std::transform(
        rootSchema.children().begin(),
        rootSchema.children().end(),
        std::inserter(fileColumns, fileColumns.end()),
        [](const auto& col) { return col.name(); });

    for (size_t i = 0; i < outputType_->size(); ++i) {
      const auto& fieldName = outputType_->nameOf(i);
      if (injectedNames.contains(fieldName)) {
        continue;
      }
      if (not fileColumns.contains(fieldName)) {
        // Schema evolution: Column was added after the data file was written
        // and doesn't exist in older data files.
        injectedColumns_.push_back(
            {i, fieldName, std::nullopt, outputType_->childAt(i)});
        injectedNames.insert(fieldName);
      }
    }
  }

  // Remove all injected columns from readColumnNames_
  if (not injectedColumns_.empty()) {
    std::erase_if(readColumnNames_, [&injectedNames](const auto& name) {
      return injectedNames.contains(name);
    });
    // Sort injected columns by output index once here
    std::sort(
        injectedColumns_.begin(),
        injectedColumns_.end(),
        [](const auto& a, const auto& b) {
          return a.outputIndex < b.outputIndex;
        });
  }
}

std::unique_ptr<cudf::table> CudfIcebergSplitReader::injectMissingColumns(
    std::unique_ptr<cudf::table>&& table,
    rmm::device_async_resource_ref mr) {
  const auto numRows = table->num_rows();
  auto columns = table->release();

  // Creates a scalar from a string partition value for the given cudf type.
  auto makePartitionScalar =
      [&](const std::string& value,
          cudf::data_type cudfType,
          const InjectedColumn& col) -> std::unique_ptr<cudf::scalar> {
    if (cudfType.id() != cudf::type_id::STRING and
        cudfType.id() != cudf::type_id::INT32 and
        cudfType.id() != cudf::type_id::INT64) {
      VELOX_NYI(
          "Identity partition columns are limited to VARCHAR, INTEGER, and BIGINT only. Other types are not yet supported. Column: '{}', Type: {}",
          col.name,
          col.veloxType->toString());
    }
    try {
      switch (cudfType.id()) {
        case cudf::type_id::STRING:
          return std::make_unique<cudf::string_scalar>(
              value, true, stream_, mr);
        case cudf::type_id::INT32:
          return std::make_unique<cudf::numeric_scalar<int32_t>>(
              folly::to<int32_t>(value), true, stream_, mr);
        case cudf::type_id::INT64:
          return std::make_unique<cudf::numeric_scalar<int64_t>>(
              folly::to<int64_t>(value), true, stream_, mr);
        default:
          VELOX_UNREACHABLE();
      }
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Bad partition value '{}' for column '{}' (type {}): {}",
          value,
          col.name,
          col.veloxType->toString(),
          e.what());
    }
  };

  // Merge data columns and injected constant columns in output order.
  const auto totalColumns = columns.size() + injectedColumns_.size();
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(totalColumns);

  auto injectedColIter = injectedColumns_.begin();
  auto dataColIter = columns.begin();

  for (size_t outIdx = 0; outIdx < totalColumns; ++outIdx) {
    if (injectedColIter != injectedColumns_.end() &&
        injectedColIter->outputIndex == outIdx) {
      const auto& col = *injectedColIter;
      auto cudfType = cudf_velox::veloxToCudfDataType(col.veloxType);

      std::unique_ptr<cudf::scalar> scalar;
      if (col.partitionValue.has_value()) {
        scalar = makePartitionScalar(col.partitionValue.value(), cudfType, col);
      } else {
        scalar = cudf::make_default_constructed_scalar(cudfType, stream_, mr);
        scalar->set_valid_async(false, stream_);
      }
      output.push_back(
          cudf::make_column_from_scalar(*scalar, numRows, stream_, mr));
      injectedColIter++;
    } else {
      VELOX_CHECK(
          dataColIter != columns.end(),
          "Data column index out of range during column injection");
      output.push_back(std::move(*dataColIter));
      dataColIter++;
    }
  }

  VELOX_CHECK(
      dataColIter == columns.end(),
      "Not all data columns were consumed during column injection");

  return std::make_unique<cudf::table>(std::move(output));
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
