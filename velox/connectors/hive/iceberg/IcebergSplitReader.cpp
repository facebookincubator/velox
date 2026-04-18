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

#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"

#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {
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
    int64_t deleteFileSeqNum,
    int64_t dataSeqNum,
    bool isEqualityDelete) {
  if (deleteFileSeqNum <= 0 || dataSeqNum <= 0) {
    return false;
  }
  return isEqualityDelete ? (deleteFileSeqNum <= dataSeqNum)
                          : (deleteFileSeqNum < dataSeqNum);
}

} // namespace

IcebergSplitReader::IcebergSplitReader(
    const std::shared_ptr<const HiveIcebergSplit>& icebergSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* const fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : FileSplitReader(
          icebergSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          ioStatistics,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      icebergSplit_(icebergSplit),
      baseReadOffset_(0),
      splitOffset_(0),
      deleteBitmap_(nullptr) {}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  createReader();
  if (emptySplit_) {
    return;
  }
  auto rowType = getAdaptedRowType();

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  createRowReader(std::move(metadataFilter), std::move(rowType), std::nullopt);

  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();
  positionalDeleteFileReaders_.clear();
  deletionVectorReaders_.clear();
  equalityDeleteFileReaders_.clear();

  const auto& deleteFiles = icebergSplit_->deleteFiles;
  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kPositionalDeletes) {
      if (deleteFile.recordCount > 0) {
        if (shouldSkipBySequenceNumber(
                deleteFile.dataSequenceNumber,
                icebergSplit_->dataSequenceNumber,
                /*isEqualityDelete=*/false)) {
          continue;
        }

        // Skip the delete file if all delete positions are before this split.
        // TODO: Skip delete files where all positions are after the split, if
        // split row count becomes available.
        if (auto iter =
                deleteFile.upperBounds.find(IcebergMetadataColumn::kPosId);
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
          if (posDeleteUpperBound < splitOffset_) {
            continue;
          }
        }
        positionalDeleteFileReaders_.push_back(
            std::make_unique<PositionalDeleteFileReader>(
                deleteFile,
                fileSplit_->filePath,
                fileHandleFactory_,
                connectorQueryCtx_,
                ioExecutor_,
                fileConfig_,
                ioStatistics_,
                ioStats_,
                runtimeStats,
                splitOffset_,
                fileSplit_->connectorId));
      }
    } else if (deleteFile.content == FileContent::kEqualityDeletes) {
      if (deleteFile.recordCount > 0 && !deleteFile.equalityFieldIds.empty()) {
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
        VELOX_CHECK(
            dataColumns != nullptr,
            "Iceberg equality delete file '{}' cannot be processed because "
            "table data columns are not available in HiveTableHandle.",
            deleteFile.filePath);
        for (const auto& eqFieldId : deleteFile.equalityFieldIds) {
          // Field IDs are 1-based sequential for non-evolved schemas.
          auto colIdx = static_cast<uint32_t>(eqFieldId - 1);
          VELOX_CHECK_LT(
              colIdx,
              dataColumns->size(),
              "Equality delete field ID {} out of range. This may indicate "
              "schema evolution with non-sequential field IDs, which is "
              "not yet supported.",
              eqFieldId);
          equalityColumnNames.push_back(dataColumns->nameOf(colIdx));
          equalityColumnTypes.push_back(dataColumns->childAt(colIdx));
        }

        if (!equalityColumnNames.empty()) {
          equalityDeleteFileReaders_.push_back(
              std::make_unique<EqualityDeleteFileReader>(
                  deleteFile,
                  equalityColumnNames,
                  equalityColumnTypes,
                  fileSplit_->filePath,
                  fileHandleFactory_,
                  connectorQueryCtx_,
                  ioExecutor_,
                  fileConfig_,
                  ioStatistics_,
                  ioStats_,
                  runtimeStats,
                  fileSplit_->connectorId));
        }
      }
    } else if (deleteFile.content == FileContent::kDeletionVector) {
      if (deleteFile.recordCount > 0) {
        if (shouldSkipBySequenceNumber(
                deleteFile.dataSequenceNumber,
                icebergSplit_->dataSequenceNumber,
                /*isEqualityDelete=*/false)) {
          continue;
        }

        deletionVectorReaders_.push_back(
            std::make_unique<DeletionVectorReader>(
                deleteFile, splitOffset_, connectorQueryCtx_->memoryPool()));
      }
    } else {
      VELOX_NYI(
          "Unsupported delete file content type: {}",
          static_cast<int>(deleteFile.content));
    }
  }
}

uint64_t IcebergSplitReader::next(uint64_t size, VectorPtr& output) {
  Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  mutation.deletedRows = nullptr;

  if (deleteBitmap_) {
    std::memset(
        (void*)(deleteBitmap_->asMutable<int8_t>()), 0L, deleteBitmap_->size());
  }

  const auto actualSize = baseRowReader_->nextReadSize(size);
  baseReadOffset_ = baseRowReader_->nextRowNumber() - splitOffset_;
  if (actualSize == dwio::common::RowReader::kAtEnd) {
    return 0;
  }

  if (!positionalDeleteFileReaders_.empty() ||
      !deletionVectorReaders_.empty()) {
    auto numBytes = bits::nbytes(actualSize);
    dwio::common::ensureCapacity<int8_t>(
        deleteBitmap_, numBytes, connectorQueryCtx_->memoryPool(), false, true);

    for (auto iter = positionalDeleteFileReaders_.begin();
         iter != positionalDeleteFileReaders_.end();) {
      (*iter)->readDeletePositions(baseReadOffset_, actualSize, deleteBitmap_);

      if ((*iter)->noMoreData()) {
        iter = positionalDeleteFileReaders_.erase(iter);
      } else {
        ++iter;
      }
    }

    for (auto iter = deletionVectorReaders_.begin();
         iter != deletionVectorReaders_.end();) {
      (*iter)->readDeletePositions(baseReadOffset_, actualSize, deleteBitmap_);

      if ((*iter)->noMoreData()) {
        iter = deletionVectorReaders_.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  mutation.deletedRows = deleteBitmap_ && deleteBitmap_->size() > 0
      ? deleteBitmap_->as<uint64_t>()
      : nullptr;

  auto rowsScanned = baseRowReader_->next(actualSize, output, &mutation);

  // Apply equality deletes after reading base data. Unlike positional deletes
  // (which set bits before reading), equality deletes require the data values
  // to be available for comparison.
  if (rowsScanned > 0 && !equalityDeleteFileReaders_.empty()) {
    auto outputRowVector = std::dynamic_pointer_cast<RowVector>(output);
    VELOX_CHECK_NOT_NULL(
        outputRowVector, "Output must be a RowVector for equality deletes.");

    auto numRows = outputRowVector->size();

    // Use a separate bitmap for equality deletes to track which rows to
    // remove from the output.
    BufferPtr eqDeleteBitmap = AlignedBuffer::allocate<bool>(
        numRows, connectorQueryCtx_->memoryPool());
    std::memset(
        eqDeleteBitmap->asMutable<uint8_t>(), 0, eqDeleteBitmap->size());

    for (auto& reader : equalityDeleteFileReaders_) {
      reader->applyDeletes(outputRowVector, eqDeleteBitmap);
    }

    // Count surviving rows and compact the output if any rows were deleted.
    auto* eqBitmap = eqDeleteBitmap->as<uint8_t>();
    vector_size_t numDeleted = 0;
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (bits::isBitSet(eqBitmap, i)) {
        ++numDeleted;
      }
    }

    if (numDeleted > 0) {
      vector_size_t numSurviving = numRows - numDeleted;
      if (numSurviving == 0) {
        // All rows in this batch were deleted by equality deletes. Do not
        // return 0 here — that would be interpreted as end-of-split and
        // prematurely stop scanning remaining rows in the data file.
        // Instead, set output to an empty vector and return the original
        // scanned count so the caller continues reading.
        output = BaseVector::create(
            outputRowVector->type(), 0, connectorQueryCtx_->memoryPool());
      } else {
        // Build a list of surviving row ranges and use it to compact.
        std::vector<BaseVector::CopyRange> ranges;
        ranges.reserve(numSurviving);
        vector_size_t targetIdx = 0;
        for (vector_size_t i = 0; i < numRows; ++i) {
          if (!bits::isBitSet(eqBitmap, i)) {
            ranges.push_back({i, targetIdx++, 1});
          }
        }

        auto newOutput = BaseVector::create(
            outputRowVector->type(),
            numSurviving,
            connectorQueryCtx_->memoryPool());
        newOutput->copyRanges(outputRowVector.get(), ranges);
        newOutput->resize(numSurviving);
        output = newOutput;
        rowsScanned = numSurviving;
      }
    }

    return rowsScanned;
  }

  return rowsScanned;
}

std::vector<TypePtr> IcebergSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& tableSchema) const {
  std::vector<TypePtr> columnTypes = fileType->children();
  auto& childrenSpecs = scanSpec_->children();
  const auto& splitInfoColumns = icebergSplit_->infoColumns;
  // Iceberg table stores all column's data in data file.
  for (const auto& childSpec : childrenSpecs) {
    const std::string& fieldName = childSpec->fieldName();
    if (auto iter = splitInfoColumns.find(fieldName);
        iter != splitInfoColumns.end()) {
      auto infoColumnType = readerOutputType_->findChild(fieldName);
      auto constant = newConstantFromString(
          infoColumnType,
          iter->second,
          connectorQueryCtx_->memoryPool(),
          fileConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties()),
          false);
      childSpec->setConstantValue(constant);
    } else {
      auto fileTypeIdx = fileType->getChildIdxIfExists(fieldName);
      auto outputTypeIdx = readerOutputType_->getChildIdxIfExists(fieldName);
      if (outputTypeIdx.has_value() && fileTypeIdx.has_value()) {
        childSpec->setConstantValue(nullptr);
        auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
        columnTypes[*fileTypeIdx] = outputType;
      } else if (!fileTypeIdx.has_value()) {
        // Handle columns missing from the data file in two scenarios:
        // 1. Schema evolution: Column was added after the data file was
        // written and doesn't exist in older data files.
        // 2. Partition columns: Hive migrated table. In Hive-written data
        // files, partition column values are stored in partition metadata
        // rather than in the data file itself, following Hive's partitioning
        // convention.
        auto partitionIt = fileSplit_->partitionKeys.find(fieldName);
        if (partitionIt != fileSplit_->partitionKeys.end()) {
          setPartitionValue(childSpec.get(), fieldName, partitionIt->second);
        } else {
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  tableSchema->findChild(fieldName),
                  1,
                  connectorQueryCtx_->memoryPool()));
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

} // namespace facebook::velox::connector::hive::iceberg
