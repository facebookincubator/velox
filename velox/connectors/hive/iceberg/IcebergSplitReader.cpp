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

#include <algorithm>

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
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
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
          dataIoStats,
          metadataIoStats,
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

  configureEqualityDeleteColumns();

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
                dataIoStats_,
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

        auto [equalityColumnNames, equalityColumnTypes] =
            resolveEqualityColumns(deleteFile);

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
                  dataIoStats_,
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

void IcebergSplitReader::configureEqualityDeleteColumns() {
  // Reset partition-column tracking from any prior split before re-augmenting.
  equalityAugmentedPartitionColumns_.clear();

  std::vector<std::string> extraEqualityColumns;
  std::vector<std::string> extraNames;
  std::vector<TypePtr> extraTypes;
  const auto& deleteFiles = icebergSplit_->deleteFiles;
  const auto& splitPartitionKeys = icebergSplit_->partitionKeys;

  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content != FileContent::kEqualityDeletes ||
        deleteFile.recordCount == 0 || deleteFile.equalityFieldIds.empty()) {
      continue;
    }
    if (shouldSkipBySequenceNumber(
            deleteFile.dataSequenceNumber,
            icebergSplit_->dataSequenceNumber,
            /*isEqualityDelete=*/true)) {
      continue;
    }

    auto [equalityColumnNames, equalityColumnTypes] =
        resolveEqualityColumns(deleteFile);
    for (size_t i = 0; i < equalityColumnNames.size(); ++i) {
      const auto& name = equalityColumnNames[i];
      // Skip if this column was already added by a previous delete file.
      if (std::find(
              extraEqualityColumns.begin(), extraEqualityColumns.end(), name) !=
          extraEqualityColumns.end()) {
        continue;
      }
      auto* fieldSpec = scanSpec_->childByName(name);
      const bool alreadyInOutput =
          readerOutputType_->getChildIdxIfExists(name).has_value();
      if (fieldSpec != nullptr && fieldSpec->projectOut() && alreadyInOutput) {
        // Already projected by the user (or a previous augmentation) AND
        // present in the output type by name. The equality-delete reader can
        // probe it directly.
        continue;
      }
      // Either no spec exists, or one exists but is filter-only, or the
      // scan-spec child is projected but the column is missing from
      // 'readerOutputType_'. In all cases ensure the column ends up in
      // 'readerOutputType_' AND has a projected scan-spec child with a
      // non-conflicting channel.
      if (fieldSpec == nullptr) {
        fieldSpec = scanSpec_->getOrCreateChild(name);
      }
      fieldSpec->setProjectOut(true);
      fieldSpec->setChannel(
          static_cast<column_index_t>(
              readerOutputType_->size() + extraEqualityColumns.size()));

      // For partition columns set the partition value directly as a constant
      // on the scan-spec child. This is independent of whether the data file
      // contains the partition column physically. With the constant set
      // up-front, 'adaptColumns' does not need any special-case logic for
      // augmented partition columns and the read does not depend on the
      // writer's choice of including the partition column in the file.
      auto partitionIt = splitPartitionKeys.find(name);
      if (partitionIt != splitPartitionKeys.end()) {
        // Iceberg encodes DATE partition values as the integer number of
        // days since the Unix epoch (e.g. "19345"). The standard
        // 'setPartitionValue' helper learns this from the planner-supplied
        // ColumnHandle via 'isPartitionDateValueDaysSinceEpoch()', but no
        // ColumnHandle is available here when the partition column is not
        // in the user's projection. Derive the flag from the column type
        // instead — Iceberg always uses days-since-epoch for DATE.
        const bool isDaysSinceEpoch = equalityColumnTypes[i]->isDate();
        auto constant = newConstantFromString(
            equalityColumnTypes[i],
            partitionIt->second,
            connectorQueryCtx_->memoryPool(),
            fileConfig_->readTimestampPartitionValueAsLocalTime(
                connectorQueryCtx_->sessionProperties()),
            isDaysSinceEpoch);
        fieldSpec->setConstantValue(constant);
        // Mirror Java's PARTITION_KEY column-type marking: this column's
        // value MUST come from the partition metadata, never from the file
        // body. Track it so 'adaptColumns' Branch 1 does not later wipe the
        // constant when the file happens to also carry the column.
        equalityAugmentedPartitionColumns_.insert(name);
      }

      extraEqualityColumns.push_back(name);
      extraNames.push_back(name);
      extraTypes.push_back(equalityColumnTypes[i]);
    }
  }

  if (extraEqualityColumns.empty()) {
    return;
  }

  // Extend 'readerOutputType_' so the upstream FileDataSource allocates the
  // output RowVector wide enough for the augmented scan-spec channels. The
  // original projection columns remain at indices [0, originalSize), so
  // FileDataSource's positional projection still returns exactly the
  // user-requested columns.
  auto names = readerOutputType_->names();
  auto types = readerOutputType_->children();
  names.insert(names.end(), extraNames.begin(), extraNames.end());
  types.insert(types.end(), extraTypes.begin(), extraTypes.end());
  readerOutputType_ = ROW(std::move(names), std::move(types));
}

std::pair<std::vector<std::string>, std::vector<TypePtr>>
IcebergSplitReader::resolveEqualityColumns(
    const IcebergDeleteFile& deleteFile) const {
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
  return {std::move(equalityColumnNames), std::move(equalityColumnTypes)};
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
        if (equalityAugmentedPartitionColumns_.count(fieldName) > 0) {
          // This column was pre-installed as a partition-value constant by
          // 'configureEqualityDeleteColumns'. Mirror Java's PARTITION_KEY
          // semantics — the value comes from partition metadata, never from
          // the file body, even though the file body happens to carry the
          // column. Leave the constant in place; the column type entry for
          // the file column index stays as the file type so Velox does not
          // try to bind this output channel to a file read.
          continue;
        }
        childSpec->setConstantValue(nullptr);
        auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
        columnTypes[*fileTypeIdx] = outputType;
      } else if (!fileTypeIdx.has_value()) {
        // Handle columns missing from the data file in three scenarios:
        // 1. Schema evolution: Column was added after the data file was
        //    written and doesn't exist in older data files.
        // 2. Partition columns from a Hive-migrated table where partition
        //    column values are stored in partition metadata rather than in
        //    the data file itself.
        // 3. Equality-delete partition columns not in the user's projection:
        //    'configureEqualityDeleteColumns' has already pre-installed the
        //    partition value as a constant, so this branch leaves it alone.
        if (childSpec->isConstant()) {
          // Constant already set (case 3, or set on a previous prepareSplit
          // call for the same scanSpec). Nothing to do.
          continue;
        }
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
