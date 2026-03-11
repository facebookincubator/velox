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
#include "velox/vector/DecodedVector.h"

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
  // Expand the file schema to include row lineage metadata columns
  // (_row_id and _last_updated_sequence_number) if they're requested
  // in the output. These are hidden metadata columns physically stored
  // in Iceberg V3 data files but not listed in the table's logical
  // schema (dataColumns). The Parquet reader needs them in the file
  // schema to read them from the file.
  auto fileSchema = baseReaderOpts_.fileSchema();
  if (fileSchema) {
    auto names = fileSchema->names();
    auto types = fileSchema->children();
    bool modified = false;
    for (const auto* colName :
         {IcebergMetadataColumn::kRowIdColumnName,
          IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName}) {
      if (readerOutputType_->containsChild(colName) &&
          !fileSchema->containsChild(colName)) {
        names.push_back(std::string(colName));
        types.push_back(BIGINT());
        modified = true;
      }
    }
    if (modified) {
      baseReaderOpts_.setFileSchema(ROW(std::move(names), std::move(types)));
    }
  }

  createReader();
  if (emptySplit_) {
    return;
  }

  // Initialize row lineage fields BEFORE getAdaptedRowType() because
  // adaptColumns() needs them to decide how to handle missing columns.
  firstRowId_ = std::nullopt;
  if (auto it = hiveSplit_->infoColumns.find("$first_row_id");
      it != hiveSplit_->infoColumns.end()) {
    auto value = folly::to<int64_t>(it->second);
    if (value >= 0) {
      firstRowId_ = value;
    }
  }

  dataSequenceNumber_ = std::nullopt;
  if (auto it = hiveSplit_->infoColumns.find("$data_sequence_number");
      it != hiveSplit_->infoColumns.end()) {
    dataSequenceNumber_ = folly::to<int64_t>(it->second);
  }

  auto rowType = getAdaptedRowType();

  // After adaptColumns(), check if row lineage columns need null replacement
  // in next(). If adaptColumns() set them as constants (column missing from
  // file), no replacement is needed. If the column is read from the file
  // (not constant), null values must be replaced per spec.
  readLastUpdatedSeqNumFromFile_ = false;
  lastUpdatedSeqNumOutputIndex_ = std::nullopt;
  if (dataSequenceNumber_.has_value()) {
    auto* seqNumSpec = scanSpec_->childByName(
        IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName);
    if (seqNumSpec && !seqNumSpec->isConstant()) {
      auto idx = readerOutputType_->getChildIdxIfExists(
          IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName);
      if (idx.has_value()) {
        readLastUpdatedSeqNumFromFile_ = true;
        lastUpdatedSeqNumOutputIndex_ = *idx;
      }
    }
  }

  computeRowId_ = false;
  rowIdOutputIndex_ = std::nullopt;
  if (firstRowId_.has_value()) {
    auto idx = readerOutputType_->getChildIdxIfExists(
        IcebergMetadataColumn::kRowIdColumnName);
    if (idx.has_value()) {
      computeRowId_ = true;
      rowIdOutputIndex_ = *idx;
    }
  }

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  createRowReader(std::move(metadataFilter), std::move(rowType), std::nullopt);

  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(hiveSplit_);
  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();
  positionalDeleteFileReaders_.clear();

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
    } else {
      VELOX_NYI();
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

  if (!positionalDeleteFileReaders_.empty()) {
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
  }

  mutation.deletedRows = deleteBitmap_ && deleteBitmap_->size() > 0
      ? deleteBitmap_->as<uint64_t>()
      : nullptr;

  auto rowsScanned = baseRowReader_->next(actualSize, output, &mutation);

  // For Iceberg V3 row lineage: replace null values in
  // _last_updated_sequence_number with the data sequence number from the
  // file's manifest entry. Per the spec, null means the value should be
  // inherited from the manifest entry's sequence number.
  if (readLastUpdatedSeqNumFromFile_ && dataSequenceNumber_.has_value() &&
      lastUpdatedSeqNumOutputIndex_.has_value() && rowsScanned > 0) {
    auto* rowOutput = output->as<RowVector>();
    if (rowOutput) {
      auto& seqNumChild = rowOutput->childAt(*lastUpdatedSeqNumOutputIndex_);
      // Load lazy vector - the Parquet reader wraps columns in LazyVector.
      seqNumChild = BaseVector::loadedVectorShared(seqNumChild);
      auto vectorSize = seqNumChild->size();

      if (seqNumChild->isConstantEncoding()) {
        if (seqNumChild->isNullAt(0)) {
          seqNumChild = std::make_shared<ConstantVector<int64_t>>(
              connectorQueryCtx_->memoryPool(),
              vectorSize,
              false,
              BIGINT(),
              static_cast<int64_t>(*dataSequenceNumber_));
        }
      } else if (seqNumChild->mayHaveNulls()) {
        // Use DecodedVector to handle any encoding (flat, dictionary, etc.).
        DecodedVector decoded(*seqNumChild);
        if (decoded.mayHaveNulls()) {
          auto pool = connectorQueryCtx_->memoryPool();
          auto newFlat = BaseVector::create<FlatVector<int64_t>>(
              BIGINT(), vectorSize, pool);
          for (vector_size_t i = 0; i < vectorSize; ++i) {
            if (decoded.isNullAt(i)) {
              newFlat->set(i, *dataSequenceNumber_);
            } else {
              newFlat->set(i, decoded.valueAt<int64_t>(i));
            }
          }
          seqNumChild = newFlat;
        }
      }
    }
  }

  // For Iceberg V3 row lineage: compute _row_id = first_row_id + _pos.
  // When the data file doesn't contain _row_id physically, compute it from
  // the manifest entry's first_row_id plus the row's position in the file.
  // When positional deletes are applied, we track actual file positions
  // using the deletion bitmap.
  if (computeRowId_ && firstRowId_.has_value() &&
      rowIdOutputIndex_.has_value() && rowsScanned > 0) {
    auto* rowOutput = output->as<RowVector>();
    if (rowOutput) {
      auto& rowIdChild = rowOutput->childAt(*rowIdOutputIndex_);
      // Load lazy vector - the Parquet reader wraps columns in LazyVector.
      rowIdChild = BaseVector::loadedVectorShared(rowIdChild);
      auto vectorSize = rowIdChild->size();

      if (rowIdChild->isConstantEncoding() && rowIdChild->isNullAt(0)) {
        // All null - compute _row_id = first_row_id + _pos for all rows.
        auto pool = connectorQueryCtx_->memoryPool();
        auto flatVec =
            BaseVector::create<FlatVector<int64_t>>(BIGINT(), vectorSize, pool);
        int64_t batchStartPos = splitOffset_ + baseReadOffset_;
        if (mutation.deletedRows == nullptr) {
          for (vector_size_t i = 0; i < vectorSize; ++i) {
            flatVec->set(
                i, static_cast<int64_t>(*firstRowId_) + batchStartPos + i);
          }
        } else {
          // With deletions: use the bitmap to find actual file positions
          // of surviving rows.
          vector_size_t outputIdx = 0;
          for (vector_size_t j = 0;
               j < static_cast<vector_size_t>(actualSize) &&
               outputIdx < vectorSize;
               ++j) {
            if (!bits::isBitSet(mutation.deletedRows, j)) {
              flatVec->set(
                  outputIdx++,
                  static_cast<int64_t>(*firstRowId_) + batchStartPos + j);
            }
          }
        }
        rowIdChild = flatVec;
      } else if (rowIdChild->mayHaveNulls()) {
        // Column is in the file but has some null values - replace nulls
        // with first_row_id + _pos. Use DecodedVector for encoding support.
        DecodedVector decoded(*rowIdChild);
        if (decoded.mayHaveNulls()) {
          auto pool = connectorQueryCtx_->memoryPool();
          auto newFlat = BaseVector::create<FlatVector<int64_t>>(
              BIGINT(), vectorSize, pool);
          int64_t batchStartPos = splitOffset_ + baseReadOffset_;
          if (mutation.deletedRows == nullptr) {
            for (vector_size_t i = 0; i < vectorSize; ++i) {
              if (decoded.isNullAt(i)) {
                newFlat->set(
                    i, static_cast<int64_t>(*firstRowId_) + batchStartPos + i);
              } else {
                newFlat->set(i, decoded.valueAt<int64_t>(i));
              }
            }
          } else {
            vector_size_t outputIdx = 0;
            for (vector_size_t j = 0;
                 j < static_cast<vector_size_t>(actualSize) &&
                 outputIdx < vectorSize;
                 ++j) {
              if (!bits::isBitSet(mutation.deletedRows, j)) {
                if (decoded.isNullAt(outputIdx)) {
                  newFlat->set(
                      outputIdx,
                      static_cast<int64_t>(*firstRowId_) + batchStartPos + j);
                } else {
                  newFlat->set(outputIdx, decoded.valueAt<int64_t>(outputIdx));
                }
                ++outputIdx;
              }
            }
          }
          rowIdChild = newFlat;
        }
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
        // 3. _last_updated_sequence_number: For Iceberg V3 row lineage, if
        // the column is not in the file, inherit the data sequence number
        // from the file's manifest entry.
        // 4. _row_id: For Iceberg V3 row lineage, if the column is not in
        // the file, set as NULL constant here. When first_row_id is
        // available, next() will replace NULL with first_row_id + _pos.
        if (auto it = hiveSplit_->partitionKeys.find(fieldName);
            it != hiveSplit_->partitionKeys.end()) {
          setPartitionValue(childSpec.get(), fieldName, it->second);
        } else if (
            fieldName ==
                IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName &&
            dataSequenceNumber_.has_value()) {
          childSpec->setConstantValue(
              std::make_shared<ConstantVector<int64_t>>(
                  connectorQueryCtx_->memoryPool(),
                  1,
                  false,
                  BIGINT(),
                  static_cast<int64_t>(*dataSequenceNumber_)));
        } else {
          // Column missing from both the file and partition keys. This
          // can be a schema evolution column (in tableSchema) or a
          // metadata column like _row_id (in readerOutputType_ but not
          // in tableSchema). Try readerOutputType_ first, then fall
          // back to tableSchema.
          auto outputIdx = readerOutputType_->getChildIdxIfExists(fieldName);
          auto colType = outputIdx.has_value()
              ? readerOutputType_->childAt(*outputIdx)
              : tableSchema->findChild(fieldName);
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  colType, 1, connectorQueryCtx_->memoryPool()));
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

} // namespace facebook::velox::connector::hive::iceberg
