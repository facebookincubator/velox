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

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {

IcebergSplitReader::IcebergSplitReader(
    const std::shared_ptr<const hive::HiveConnectorSplit>& hiveSplit,
    const HiveTableHandlePtr& hiveTableHandle,
    const HiveColumnHandleMap* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* const fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : SplitReader(
          hiveSplit,
          hiveTableHandle,
          partitionKeys,
          connectorQueryCtx,
          hiveConfig,
          readerOutputType,
          ioStatistics,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
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

  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(hiveSplit_);
  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();
  positionalDeleteFileReaders_.clear();
  deletionVectorReaders_.clear();

  const auto& deleteFiles = icebergSplit->deleteFiles;
  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kPositionalDeletes) {
      if (deleteFile.recordCount > 0) {
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
                hiveSplit_->filePath,
                fileHandleFactory_,
                connectorQueryCtx_,
                ioExecutor_,
                hiveConfig_,
                ioStatistics_,
                ioStats_,
                runtimeStats,
                splitOffset_,
                hiveSplit_->connectorId));
      }
    } else if (deleteFile.content == FileContent::kEqualityDeletes) {
      VELOX_NYI("Equality deletes are not yet supported.");
    } else if (deleteFile.content == FileContent::kDeletionVector) {
      if (deleteFile.recordCount > 0) {
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

  positionalUpdateFileReaders_.clear();
  const auto& updateFiles = icebergSplit->updateFiles;

  // Derive update column names and types from the reader output schema.
  // Following the standard Iceberg MoR pattern, update files carry all
  // output columns (full-row updates). Hoisted outside the loop since the
  // schema is identical for every update file.
  std::vector<std::string> updateColumnNames;
  std::vector<TypePtr> updateColumnTypes;
  for (auto i = 0; i < readerOutputType_->size(); ++i) {
    updateColumnNames.push_back(readerOutputType_->nameOf(i));
    updateColumnTypes.push_back(readerOutputType_->childAt(i));
  }

  for (const auto& updateFile : updateFiles) {
    VELOX_CHECK(
        updateFile.content == FileContent::kPositionalUpdates,
        "Expected positional update file but got content type: {}",
        static_cast<int>(updateFile.content));
    if (updateFile.recordCount > 0) {
      positionalUpdateFileReaders_.push_back(
          std::make_unique<PositionalUpdateFileReader>(
              updateFile,
              hiveSplit_->filePath,
              updateColumnNames,
              updateColumnTypes,
              fileHandleFactory_,
              connectorQueryCtx_,
              ioExecutor_,
              hiveConfig_,
              ioStatistics_,
              ioStats_,
              runtimeStats,
              splitOffset_,
              hiveSplit_->connectorId));
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

  if (rowsScanned > 0 && !positionalUpdateFileReaders_.empty()) {
    auto outputRowVector = std::dynamic_pointer_cast<RowVector>(output);
    VELOX_CHECK_NOT_NULL(
        outputRowVector, "Output must be a RowVector for positional updates.");

    for (auto iter = positionalUpdateFileReaders_.begin();
         iter != positionalUpdateFileReaders_.end();) {
      (*iter)->applyUpdates(
          baseReadOffset_, outputRowVector, deleteBitmap_, actualSize);

      if ((*iter)->noMoreData()) {
        iter = positionalUpdateFileReaders_.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  return rowsScanned;
}

std::vector<TypePtr> IcebergSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& tableSchema) const {
  std::vector<TypePtr> columnTypes = fileType->children();
  auto& childrenSpecs = scanSpec_->children();
  // Iceberg table stores all column's data in data file.
  for (const auto& childSpec : childrenSpecs) {
    const std::string& fieldName = childSpec->fieldName();
    if (auto iter = hiveSplit_->infoColumns.find(fieldName);
        iter != hiveSplit_->infoColumns.end()) {
      auto infoColumnType = readerOutputType_->findChild(fieldName);
      auto constant = newConstantFromString(
          infoColumnType,
          iter->second,
          connectorQueryCtx_->memoryPool(),
          hiveConfig_->readTimestampPartitionValueAsLocalTime(
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
        if (auto it = hiveSplit_->partitionKeys.find(fieldName);
            it != hiveSplit_->partitionKeys.end()) {
          setPartitionValue(childSpec.get(), fieldName, it->second);
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
