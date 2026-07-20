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

#include <folly/ScopeGuard.h>
#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/vector/DecodedVector.h"

using namespace facebook::velox::dwio::common;

namespace {

// Fills null slots in a BIGINT child vector with values produced by valueAt(i),
// where i is the output row index. If the whole vector is a null constant,
// replaces it with a flat vector. If only some slots are null, copies non-null
// values and fills null slots with valueAt(i). No-op when no nulls are present.
template <typename F>
void fillNullsWithInt64(
    facebook::velox::VectorPtr& child,
    facebook::velox::memory::MemoryPool* pool,
    F valueAt) {
  using namespace facebook::velox;
  child = BaseVector::loadedVectorShared(child);
  const auto vectorSize = child->size();
  if (vectorSize == 0) {
    return;
  }
  if (child->isConstantEncoding() && child->isNullAt(0)) {
    auto flat =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), vectorSize, pool);
    for (vector_size_t i = 0; i < vectorSize; ++i) {
      flat->set(i, valueAt(i));
    }
    child = std::move(flat);
  } else if (child->mayHaveNulls()) {
    const DecodedVector decoded(*child);
    if (decoded.mayHaveNulls()) {
      auto flat =
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), vectorSize, pool);
      for (vector_size_t i = 0; i < vectorSize; ++i) {
        if (decoded.isNullAt(i)) {
          flat->set(i, valueAt(i));
        } else {
          flat->set(i, decoded.valueAt<int64_t>(i));
        }
      }
      child = std::move(flat);
    }
  }
}

} // namespace

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
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<ColumnHandleMap> columnHandles)
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
      deleteBitmap_(nullptr),
      columnHandles_(std::move(columnHandles)) {}

void IcebergSplitReader::configureBaseReaderOptions() {
  FileSplitReader::configureBaseReaderOptions();
  const auto fileFormat = fileSplit_->fileFormat;
  if (fileFormat != dwio::common::FileFormat::DWRF &&
      fileFormat != dwio::common::FileFormat::ORC) {
    // Parquet resolves field ids from physical metadata, not from the
    // attribute-based kFieldId path.
    return;
  }
  auto fieldIds = buildFieldIds();
  if (fieldIds.empty()) {
    return;
  }
  baseReaderOpts_.setColumnMappingMode(
      dwio::common::ColumnMappingMode::kFieldId);
  baseReaderOpts_.setFieldIds(std::move(fieldIds));
}

std::vector<dwio::common::ParquetFieldId> IcebergSplitReader::buildFieldIds()
    const {
  std::vector<dwio::common::ParquetFieldId> fieldIds;
  const auto& dataColumns = tableHandle_->dataColumns();
  if (dataColumns == nullptr || columnHandles_ == nullptr) {
    return fieldIds;
  }
  // Column handles are keyed by output alias; index them by the underlying
  // data-column name so we can align to dataColumns() order.
  std::unordered_map<std::string, const IcebergColumnHandle*> handleByName;
  for (const auto& [outputName, handle] : *columnHandles_) {
    if (auto* icebergHandle =
            dynamic_cast<const IcebergColumnHandle*>(handle.get())) {
      handleByName.emplace(icebergHandle->name(), icebergHandle);
    }
  }
  if (handleByName.empty()) {
    return fieldIds;
  }

  fieldIds.reserve(dataColumns->size());
  int32_t sentinelFieldId = -1;
  for (size_t i = 0; i < dataColumns->size(); ++i) {
    auto it = handleByName.find(dataColumns->nameOf(static_cast<uint32_t>(i)));
    if (it != handleByName.end()) {
      fieldIds.push_back(it->second->field());
    } else {
      fieldIds.push_back(dwio::common::ParquetFieldId{sentinelFieldId--, {}});
    }
  }
  return fieldIds;
}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  // For NIMBLE splits, switch the reader to Iceberg field-id-based column
  // resolution. The NIMBLE per-SchemaNode `attributes` slot carries
  // `iceberg.id` keys stamped on the write path; the NIMBLE reader uses
  // them to resolve projected columns under schema evolution (rename /
  // reorder / add / drop). For other formats this branch is a no-op:
  // each format-specific reader handles its own column mapping.
  if (icebergSplit_->fileFormat == dwio::common::FileFormat::NIMBLE &&
      columnHandles_ != nullptr) {
    auto fileSchema = baseReaderOpts_.fileSchema();
    if (fileSchema != nullptr && !fileSchema->names().empty()) {
      // Build the requested field-id trees, one ParquetFieldId per requested
      // file-schema column, in file-schema order. Each entry is the
      // IcebergColumnHandle's field-id tree (per-input-column).
      // Column handles are keyed by output alias; index them by physical name
      // to mirror buildFieldIds() and support renamed columns.
      std::unordered_map<std::string, const IcebergColumnHandle*> handleByName;
      for (const auto& [outputName, handle] : *columnHandles_) {
        if (auto* icebergHandle =
                dynamic_cast<const IcebergColumnHandle*>(handle.get())) {
          handleByName.emplace(icebergHandle->name(), icebergHandle);
        }
      }
      std::vector<dwio::common::ParquetFieldId> fieldIds;
      fieldIds.reserve(fileSchema->size());
      bool allResolved = true;
      for (size_t i = 0; i < fileSchema->size(); ++i) {
        const auto& name = fileSchema->nameOf(static_cast<uint32_t>(i));
        const auto it = handleByName.find(name);
        if (it == handleByName.end()) {
          allResolved = false;
          break;
        }
        const auto icebergColumn = it->second;
        if (icebergColumn == nullptr) {
          allResolved = false;
          break;
        }
        fieldIds.emplace_back(icebergColumn->field());
      }
      if (allResolved) {
        baseReaderOpts_.setColumnMappingMode(
            dwio::common::ColumnMappingMode::kFieldId);
        baseReaderOpts_.setFieldIds(std::move(fieldIds));
        // The NIMBLE reader resolves columns via the per-type "iceberg.id"
        // attribute; supply the key here so the reader stays format-agnostic.
        baseReaderOpts_.setFieldIdAttributeKey("iceberg.id");
      }
    }
  }

  // Temporarily extend the file schema with projected row-lineage columns
  // (_row_id, _last_updated_sequence_number) so the reader allocates output
  // slots for them. Scoped to createReader() so getAdaptedRowType() sees the
  // original schema.
  {
    auto originalFileSchema = baseReaderOpts_.fileSchema();
    auto restorer = folly::makeGuard([&] {
      if (originalFileSchema) {
        baseReaderOpts_.setFileSchema(originalFileSchema);
      }
    });
    if (originalFileSchema) {
      auto names = originalFileSchema->names();
      auto types = originalFileSchema->children();
      bool modified = false;
      for (const auto* colName :
           {IcebergMetadataColumn::kRowIdColumnName,
            IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName}) {
        if (readerOutputType_->containsChild(colName) &&
            !originalFileSchema->containsChild(colName)) {
          names.emplace_back(colName);
          types.emplace_back(BIGINT());
          modified = true;
        }
      }
      if (modified) {
        baseReaderOpts_.setFileSchema(ROW(std::move(names), std::move(types)));
      }
    }
    createReader(fileReadOps);
  }

  if (emptySplit_) {
    return;
  }

  configureEqualityDeleteColumns();

  firstRowId_ = std::nullopt;
  if (auto it = icebergSplit_->infoColumns.find(
          IcebergMetadataColumn::kFirstRowIdInfoColumn);
      it != icebergSplit_->infoColumns.end()) {
    try {
      firstRowId_ = folly::to<int64_t>(it->second);
    } catch (const folly::ConversionError&) {
      VELOX_FAIL(
          "Invalid $first_row_id value in split info columns: {}", it->second);
    }
    VELOX_CHECK_GE(*firstRowId_, 0, "First row ID must be non-negative");
  }

  dataSequenceNumber_ = std::nullopt;
  if (auto it = icebergSplit_->infoColumns.find(
          IcebergMetadataColumn::kDataSequenceNumberInfoColumn);
      it != icebergSplit_->infoColumns.end()) {
    try {
      dataSequenceNumber_ = folly::to<int64_t>(it->second);
    } catch (const folly::ConversionError&) {
      VELOX_FAIL(
          "Invalid $data_sequence_number value in split info columns: {}",
          it->second);
    }
    VELOX_CHECK_GE(
        *dataSequenceNumber_, 0, "Data sequence number must be non-negative");
  }

  // getAdaptedRowType() calls adaptColumns(), which may set
  // _last_updated_sequence_number to a constant. Must run after
  // dataSequenceNumber_ and firstRowId_ are initialized.
  auto rowType = getAdaptedRowType();

  lastUpdatedSeqNumOutputIndex_ = std::nullopt;
  if (dataSequenceNumber_.has_value() && firstRowId_.has_value()) {
    auto* seqNumSpec = scanSpec_->childByName(
        IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName);
    if (seqNumSpec && !seqNumSpec->isConstant()) {
      lastUpdatedSeqNumOutputIndex_ = readerOutputType_->getChildIdxIfExists(
          IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName);
    }
  }

  rowIdOutputIndex_ = std::nullopt;
  if (firstRowId_.has_value()) {
    rowIdOutputIndex_ = readerOutputType_->getChildIdxIfExists(
        IcebergMetadataColumn::kRowIdColumnName);
  }

  // Iceberg MERGE INTO row-id synthesis: detect projection of
  // $target_table_row_id and cache the constant fields (spec_id,
  // partition_data) so next() can build the composite RowVector cheaply.
  // file_path comes from the split's filePath directly; row_position is
  // computed per row in next() using either the injected row-number column
  // (filter / skip / positional-delete paths) or the contiguous formula.
  targetTableRowIdOutputIndex_ = std::nullopt;
  targetTableSpecId_ = std::nullopt;
  targetTablePartitionData_ = std::nullopt;
  if (readerOutputType_->containsChild(
          IcebergMetadataColumn::kTargetTableRowIdColumnName)) {
    targetTableRowIdOutputIndex_ = readerOutputType_->getChildIdxIfExists(
        IcebergMetadataColumn::kTargetTableRowIdColumnName);
    if (auto it = icebergSplit_->infoColumns.find(
            IcebergMetadataColumn::kSpecIdInfoColumn);
        it != icebergSplit_->infoColumns.end()) {
      try {
        targetTableSpecId_ = folly::to<int32_t>(it->second);
      } catch (const folly::ConversionError&) {
        VELOX_FAIL(
            "Invalid $spec_id value in split info columns: {}", it->second);
      }
    }
    if (auto it = icebergSplit_->infoColumns.find(
            IcebergMetadataColumn::kPartitionDataInfoColumn);
        it != icebergSplit_->infoColumns.end()) {
      targetTablePartitionData_ = it->second;
    }
  }

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  // Inject a row-number column when filters, random-skip, or positional
  // deletes make the output-to-file-position mapping non-contiguous.
  // Check split metadata rather than positionalDeleteFileReaders_ because
  // the row reader must be configured before delete files are opened. Both
  // _row_id and $target_table_row_id need accurate file-absolute positions,
  // so request injection when either is projected.
  const bool hasPositionalDeletes = std::any_of(
      icebergSplit_->deleteFiles.begin(),
      icebergSplit_->deleteFiles.end(),
      [](const IcebergDeleteFile& deleteFile) {
        return deleteFile.content == FileContent::kPositionalDeletes &&
            deleteFile.recordCount > 0;
      });
  useRowNumberColumn_ = (rowIdOutputIndex_.has_value() ||
                         targetTableRowIdOutputIndex_.has_value()) &&
      (scanSpec_->hasFilter() || baseReaderOpts_.randomSkip() != nullptr ||
       hasPositionalDeletes);
  if (useRowNumberColumn_) {
    dwio::common::RowNumberColumnInfo rowNumInfo;
    rowNumInfo.insertPosition = readerOutputType_->size();
    rowNumInfo.name = "";
    baseRowReaderOpts_.setRowNumberColumnInfo(rowNumInfo);
  } else {
    baseRowReaderOpts_.setRowNumberColumnInfo(std::nullopt);
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

        // If 'referencedDataFile' is set and does not match the split's
        // data file, log a warning. Do NOT skip the DV: silently dropping
        // a coordinator-shipped DV could mask deletes if the planner and
        // worker disagree on path normalization (trailing slash, scheme
        // prefix like s3:// vs s3a://, percent-encoding). The coordinator
        // already filters per-split, so the worker-side check is purely
        // diagnostic.
        if (!deleteFile.referencedDataFile.empty() &&
            deleteFile.referencedDataFile != fileSplit_->filePath) {
          LOG(WARNING)
              << "Iceberg DV referencedDataFile does not match split path. "
              << "Applying DV anyway. referencedDataFile='"
              << deleteFile.referencedDataFile << "' splitPath='"
              << fileSplit_->filePath << "'";
        }

        deletionVectorReaders_.push_back(
            std::make_unique<DeletionVectorReader>(
                deleteFile,
                splitOffset_,
                connectorQueryCtx_->memoryPool(),
                fileConfig_->config()));
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

  auto* pool = connectorQueryCtx_->memoryPool();

  if (rowsScanned > 0 &&
      (lastUpdatedSeqNumOutputIndex_.has_value() ||
       rowIdOutputIndex_.has_value() ||
       targetTableRowIdOutputIndex_.has_value())) {
    auto* rowOutput = output->as<RowVector>();
    VELOX_DCHECK_NOT_NULL(
        rowOutput, "Expected RowVector output from table scan");

    if (lastUpdatedSeqNumOutputIndex_.has_value()) {
      VELOX_DCHECK(dataSequenceNumber_.has_value());
      auto& seqNumChild = rowOutput->childAt(*lastUpdatedSeqNumOutputIndex_);
      const int64_t seqNum = static_cast<int64_t>(*dataSequenceNumber_);
      fillNullsWithInt64(
          seqNumChild, pool, [seqNum](vector_size_t) { return seqNum; });
    }

    // Resolve file-absolute row positions once and reuse for both
    // _row_id and $target_table_row_id. When useRowNumberColumn_ is true,
    // the reader appended a row-number column at index
    // readerOutputType_->size(). Otherwise the output is contiguous and we
    // derive positions from the split offset plus per-row index.
    const bool needRowPositions = rowIdOutputIndex_.has_value() ||
        targetTableRowIdOutputIndex_.has_value();
    std::optional<DecodedVector> decodedRowNumsHolder;
    if (needRowPositions && useRowNumberColumn_) {
      decodedRowNumsHolder.emplace(
          *rowOutput->childAt(readerOutputType_->size()));
    }
    auto rowPositionAt = [&](vector_size_t i) -> int64_t {
      if (useRowNumberColumn_) {
        return decodedRowNumsHolder->valueAt<int64_t>(i);
      }
      return static_cast<int64_t>(splitOffset_ + baseReadOffset_) +
          static_cast<int64_t>(i);
    };

    if (rowIdOutputIndex_.has_value() && firstRowId_.has_value()) {
      auto& rowIdChild = rowOutput->childAt(*rowIdOutputIndex_);
      const int64_t firstRowId = static_cast<int64_t>(*firstRowId_);
      fillNullsWithInt64(rowIdChild, pool, [&](vector_size_t i) {
        return firstRowId + rowPositionAt(i);
      });
    }

    if (targetTableRowIdOutputIndex_.has_value()) {
      // Build the 4-field composite ROW for Iceberg MERGE INTO row-id:
      //   ROW(file_path:VARCHAR, row_position:BIGINT, spec_id:INTEGER,
      //       partition_data:VARCHAR).
      // file_path and partition_data are constant per split; spec_id is
      // constant per split. row_position is per row, sourced from
      // rowPositionAt(). Use ConstantVector children for the constants so
      // we avoid copying the file path / partition data once per row.
      static const std::string emptyPartitionData;
      const auto& rowIdType =
          readerOutputType_->childAt(*targetTableRowIdOutputIndex_);
      VELOX_CHECK(
          rowIdType->isRow() && rowIdType->size() == 4,
          "$target_table_row_id must be a 4-field ROW; got {}",
          rowIdType->toString());
      const auto& rowIdRowType = rowIdType->asRow();
      const auto numRows = static_cast<vector_size_t>(rowsScanned);

      auto filePathConst = BaseVector::createConstant(
          rowIdRowType.childAt(0),
          variant(icebergSplit_->filePath),
          numRows,
          pool);

      auto rowPositionFlat = BaseVector::create<FlatVector<int64_t>>(
          rowIdRowType.childAt(1), numRows, pool);
      for (vector_size_t i = 0; i < numRows; ++i) {
        rowPositionFlat->set(i, rowPositionAt(i));
      }

      const int32_t specId = targetTableSpecId_.value_or(0);
      auto specIdConst = BaseVector::createConstant(
          rowIdRowType.childAt(2), variant(specId), numRows, pool);

      const std::string& partitionData = targetTablePartitionData_.has_value()
          ? *targetTablePartitionData_
          : emptyPartitionData;
      auto partitionDataConst = BaseVector::createConstant(
          rowIdRowType.childAt(3), variant(partitionData), numRows, pool);

      std::vector<VectorPtr> children = {
          std::move(filePathConst),
          std::move(rowPositionFlat),
          std::move(specIdConst),
          std::move(partitionDataConst)};
      rowOutput->childAt(*targetTableRowIdOutputIndex_) =
          std::make_shared<RowVector>(
              pool,
              rowIdType,
              /*nulls=*/nullptr,
              numRows,
              std::move(children));
    }

    // Strip the injected row-number column (always last, allocated when
    // useRowNumberColumn_ is true). Done once for both row-id paths.
    if (useRowNumberColumn_) {
      auto children = rowOutput->children();
      children.pop_back();
      output = std::make_shared<RowVector>(
          rowOutput->pool(),
          readerOutputType_,
          rowOutput->nulls(),
          rowOutput->size(),
          std::move(children));
    }
  }

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
    BufferPtr eqDeleteBitmap = AlignedBuffer::allocate<bool>(numRows, pool);
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
        output = BaseVector::create(outputRowVector->type(), 0, pool);
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

        auto newOutput =
            BaseVector::create(outputRowVector->type(), numSurviving, pool);
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
  const bool readTimestampAsLocalTime =
      fileConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());
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
          readTimestampAsLocalTime,
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
        if (!firstRowId_.has_value() &&
            (fieldName == IcebergMetadataColumn::kRowIdColumnName ||
             fieldName ==
                 IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName)) {
          // Null first_row_id: spec requires null for both lineage columns.
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  BIGINT(), 1, connectorQueryCtx_->memoryPool()));
          continue;
        }
        childSpec->setConstantValue(nullptr);
        auto& outputType = readerOutputType_->childAt(*outputTypeIdx);
        auto& columnType = columnTypes[*fileTypeIdx];
        if (childSpec->isFlatMapAsStruct()) {
          VELOX_CHECK(
              outputType->isRow(),
              "Unexpected output type for flat-map-as-struct column: {}.",
              outputType->toString());
          VELOX_CHECK(
              columnType->isMap(),
              "Unexpected column type for flat-map-as-struct column: {}.",
              columnType->toString());
        } else {
          columnType = outputType;
        }
      } else if (!fileTypeIdx.has_value()) {
        // Handle columns missing from the data file in several scenarios:
        // 1. Partition columns from a Hive-migrated table where partition
        //    column values are stored in partition metadata rather than in
        //    the data file itself.
        // 2. Schema evolution with default values (Iceberg V3): Column was
        //    added with an initial-default value. Use the default instead of
        //    NULL.
        // 3. Schema evolution: Column was added after the data file was
        //    written and doesn't exist in older data files.
        // 4. _last_updated_sequence_number: For Iceberg V3 row lineage, if
        //    the column is not in the file, inherit the data sequence number
        //    from the file's manifest entry.
        // 5. _row_id: For Iceberg V3 row lineage, if the column is not in
        //    the file, set as NULL constant here. When first_row_id is
        //    available, next() will replace NULL with first_row_id + file
        //    position.
        // 6. Equality-delete partition columns not in the user's projection:
        //    'configureEqualityDeleteColumns' has already pre-installed the
        //    partition value as a constant, so this branch leaves it alone.
        if (fieldName ==
            IcebergMetadataColumn::kLastUpdatedSequenceNumberColumnName) {
          // Column absent from file. Use data_sequence_number constant when
          // first_row_id is present; null otherwise (spec requirement).
          if (dataSequenceNumber_.has_value() && firstRowId_.has_value()) {
            childSpec->setConstantValue(
                std::make_shared<ConstantVector<int64_t>>(
                    connectorQueryCtx_->memoryPool(),
                    1,
                    false,
                    BIGINT(),
                    static_cast<int64_t>(*dataSequenceNumber_)));
          } else {
            childSpec->setConstantValue(
                BaseVector::createNullConstant(
                    BIGINT(), 1, connectorQueryCtx_->memoryPool()));
          }
        } else if (fieldName == IcebergMetadataColumn::kRowIdColumnName) {
          // Column absent from file. next() fills nulls with first_row_id +
          // _pos when first_row_id is present.
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  BIGINT(), 1, connectorQueryCtx_->memoryPool()));
        } else if (
            fieldName == IcebergMetadataColumn::kTargetTableRowIdColumnName) {
          // Synthesized Iceberg MERGE row-id column. Install a null
          // placeholder constant of the full ROW type so the reader
          // allocates the output slot with the right shape; next() will
          // overwrite the child with a freshly built flat RowVector whose
          // four fields (file_path, row_position, spec_id, partition_data)
          // are populated from the split's info columns and the file row
          // positions. This mirrors the Java IcebergPageSourceProvider
          // synthesis path that backs MERGE_TARGET_ROW_ID_DATA.
          auto columnType = readerOutputType_->findChild(fieldName);
          VELOX_CHECK_NOT_NULL(
              columnType,
              "$target_table_row_id type missing from readerOutputType");
          childSpec->setConstantValue(
              BaseVector::createNullConstant(
                  columnType, 1, connectorQueryCtx_->memoryPool()));
        } else if (childSpec->isConstant()) {
          // Constant already set (equality-delete partition column, or set on
          // a previous prepareSplit call for the same scanSpec). Nothing to do.
          continue;
        } else if (auto partitionIt = fileSplit_->partitionKeys.find(fieldName);
                   partitionIt != fileSplit_->partitionKeys.end()) {
          setPartitionValue(childSpec.get(), fieldName, partitionIt->second);
        } else {
          // Check if column has an initial-default value (Iceberg V3)
          bool hasDefaultValue = false;
          // The columnHandles_ map is keyed by output name (which may be an
          // alias). We need to find the column handle where the handle's name()
          // matches fieldName. fieldName is the table column name from
          // readerOutputType_.
          for (const auto& [outputName, handle] : *columnHandles_) {
            if (handle->name() == fieldName) {
              auto icebergColumnHandle =
                  std::dynamic_pointer_cast<const IcebergColumnHandle>(handle);
              if (icebergColumnHandle &&
                  icebergColumnHandle->initialDefaultValue().has_value()) {
                // Use initial-default value for schema evolution.
                auto columnType = tableSchema->findChild(fieldName);
                VELOX_CHECK_NOT_NULL(
                    columnType,
                    "Column '{}' not found in table schema",
                    fieldName);
                auto constant = newConstantFromString(
                    columnType,
                    icebergColumnHandle->initialDefaultValue().value(),
                    connectorQueryCtx_->memoryPool(),
                    readTimestampAsLocalTime,
                    false);
                childSpec->setConstantValue(constant);
                hasDefaultValue = true;
                break;
              }
            }
          }

          // Fall back to NULL if no default value
          if (!hasDefaultValue) {
            auto columnType = tableSchema->findChild(fieldName);
            VELOX_CHECK_NOT_NULL(
                columnType, "Column '{}' not found in table schema", fieldName);
            childSpec->setConstantValue(
                BaseVector::createNullConstant(
                    columnType, 1, connectorQueryCtx_->memoryPool()));
          }
        }
      }
    }
  }

  scanSpec_->resetCachedValues(false);

  return columnTypes;
}

} // namespace facebook::velox::connector::hive::iceberg
