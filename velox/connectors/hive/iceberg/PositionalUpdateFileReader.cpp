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

#include "velox/connectors/hive/iceberg/PositionalUpdateFileReader.h"

#include "velox/common/base/BitUtil.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive::iceberg {

PositionalUpdateFileReader::PositionalUpdateFileReader(
    const IcebergDeleteFile& updateFile,
    const std::string& baseFilePath,
    const std::vector<std::string>& updateColumnNames,
    const std::vector<TypePtr>& updateColumnTypes,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    uint64_t splitOffset,
    const std::string& connectorId)
    : updateFile_(updateFile),
      baseFilePath_(baseFilePath),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      hiveConfig_(hiveConfig),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      pool_(connectorQueryCtx->memoryPool()),
      filePathColumn_(IcebergMetadataColumn::icebergDeleteFilePathColumn()),
      posColumn_(IcebergMetadataColumn::icebergDeletePosColumn()),
      splitOffset_(splitOffset),
      updateColumnNames_(updateColumnNames),
      updateColumnTypes_(updateColumnTypes) {
  VELOX_CHECK(updateFile_.content == FileContent::kPositionalUpdates);
  VELOX_CHECK_GT(updateFile_.recordCount, 0);
  VELOX_CHECK(
      !updateColumnNames_.empty(),
      "Positional update file must specify at least one update column.");
  VELOX_CHECK_EQ(
      updateColumnNames_.size(),
      updateColumnTypes_.size(),
      "Update column names and types must have the same size.");

  // Build the output row type for readUpdatePositions(): (pos, update_cols...).
  // Cached to avoid reconstructing on every batch.
  {
    std::vector<std::string> outNames{posColumn_->name};
    std::vector<TypePtr> outTypes{posColumn_->type};
    outNames.insert(
        outNames.end(), updateColumnNames_.begin(), updateColumnNames_.end());
    outTypes.insert(
        outTypes.end(), updateColumnTypes_.begin(), updateColumnTypes_.end());
    outputRowType_ = ROW(std::move(outNames), std::move(outTypes));
  }

  // Build the file schema: (file_path, pos, update_col_1, update_col_2, ...).
  std::vector<std::string> columnNames;
  std::vector<TypePtr> columnTypes;
  columnNames.reserve(2 + updateColumnNames_.size());
  columnTypes.reserve(2 + updateColumnTypes_.size());
  columnNames.push_back(filePathColumn_->name);
  columnTypes.push_back(filePathColumn_->type);
  columnNames.push_back(posColumn_->name);
  columnTypes.push_back(posColumn_->type);
  for (size_t i = 0; i < updateColumnNames_.size(); ++i) {
    columnNames.push_back(updateColumnNames_[i]);
    columnTypes.push_back(updateColumnTypes_[i]);
  }
  RowTypePtr updateFileSchema =
      ROW(std::move(columnNames), std::move(columnTypes));

  // Create a ScanSpec that filters on file_path = baseFilePath_ and reads pos
  // plus the update columns.
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField(posColumn_->name, 0);
  for (size_t i = 0; i < updateColumnNames_.size(); ++i) {
    scanSpec->addField(updateColumnNames_[i], static_cast<int>(i) + 1);
  }
  auto* pathSpec = scanSpec->getOrCreateChild(filePathColumn_->name);
  pathSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>({baseFilePath_}), false));

  updateSplit_ = std::make_shared<HiveConnectorSplit>(
      connectorId,
      updateFile_.filePath,
      updateFile_.fileFormat,
      0,
      updateFile_.fileSizeInBytes);

  dwio::common::ReaderOptions updateReaderOpts(pool_);
  configureReaderOptions(
      hiveConfig_,
      connectorQueryCtx,
      updateFileSchema,
      updateSplit_,
      /*tableParameters=*/{},
      updateReaderOpts);

  const FileHandleKey fileHandleKey{
      .filename = updateFile_.filePath,
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
  auto updateFileHandleCachePtr = fileHandleFactory_->generate(fileHandleKey);
  auto updateFileInput = BufferedInputBuilder::getInstance()->create(
      *updateFileHandleCachePtr,
      updateReaderOpts,
      connectorQueryCtx,
      ioStatistics_,
      ioStats_,
      executor_);

  auto updateReader =
      dwio::common::getReaderFactory(updateReaderOpts.fileFormat())
          ->createReader(std::move(updateFileInput), updateReaderOpts);

  // Filter update columns to only those present in the file. The update file
  // may carry a subset of the requested columns (e.g., only the columns that
  // were actually modified). Columns not in the file are left unchanged.
  const auto& fileRowType = updateReader->rowType();
  {
    std::vector<std::string> filteredNames;
    std::vector<TypePtr> filteredTypes;
    for (size_t i = 0; i < updateColumnNames_.size(); ++i) {
      if (fileRowType->containsChild(updateColumnNames_[i])) {
        filteredNames.push_back(updateColumnNames_[i]);
        filteredTypes.push_back(updateColumnTypes_[i]);
      }
    }
    VELOX_CHECK(
        !filteredNames.empty(),
        "Positional update file contains none of the expected update columns. "
        "File schema: {}, file: {}",
        fileRowType->toString(),
        updateFile_.filePath);
    updateColumnNames_ = std::move(filteredNames);
    updateColumnTypes_ = std::move(filteredTypes);

    // Rebuild outputRowType_ and scanSpec after filtering.
    std::vector<std::string> outNames{posColumn_->name};
    std::vector<TypePtr> outTypes{posColumn_->type};
    outNames.insert(
        outNames.end(), updateColumnNames_.begin(), updateColumnNames_.end());
    outTypes.insert(
        outTypes.end(), updateColumnTypes_.begin(), updateColumnTypes_.end());
    outputRowType_ = ROW(std::move(outNames), std::move(outTypes));

    scanSpec = std::make_shared<common::ScanSpec>("<root>");
    scanSpec->addField(posColumn_->name, 0);
    for (size_t i = 0; i < updateColumnNames_.size(); ++i) {
      scanSpec->addField(updateColumnNames_[i], static_cast<int>(i) + 1);
    }
    pathSpec = scanSpec->getOrCreateChild(filePathColumn_->name);
    pathSpec->setFilter(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>({baseFilePath_}), false));
  }

  if (!testFilters(
          scanSpec.get(),
          updateReader.get(),
          updateSplit_->filePath,
          updateSplit_->partitionKeys,
          {},
          hiveConfig_->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx_->sessionProperties()))) {
    runtimeStats.skippedSplitBytes +=
        static_cast<int64_t>(updateSplit_->length);
    updateSplit_.reset();
    return;
  }

  dwio::common::RowReaderOptions updateRowReaderOpts;
  configureRowReaderOptions(
      {},
      scanSpec,
      nullptr,
      updateFileSchema,
      updateSplit_,
      nullptr,
      nullptr,
      nullptr,
      updateRowReaderOpts);

  updateRowReader_ = updateReader->createRowReader(updateRowReaderOpts);
}

void PositionalUpdateFileReader::readUpdatePositions(
    uint64_t baseReadOffset,
    int64_t rowNumberUpperBound) {
  if (!updateRowReader_ || !updateSplit_) {
    return;
  }

  if (!updatePositionsOutput_) {
    updatePositionsOutput_ = BaseVector::create(outputRowType_, 0, pool_);
  }

  int64_t rowNumberLowerBound =
      static_cast<int64_t>(baseReadOffset + splitOffset_);

  // Process any leftover rows from the previous batch.
  if (updatePositionsOffset_ <
      static_cast<uint64_t>(updatePositionsOutput_->size())) {
    auto rowOutput =
        std::dynamic_pointer_cast<RowVector>(updatePositionsOutput_);
    auto posVector = rowOutput->childAt(0)->as<FlatVector<int64_t>>();

    while (updatePositionsOffset_ < static_cast<uint64_t>(posVector->size())) {
      int64_t pos = posVector->valueAt(
          static_cast<vector_size_t>(updatePositionsOffset_));
      if (pos >= rowNumberUpperBound) {
        return;
      }
      if (pos >= rowNumberLowerBound) {
        pendingUpdates_[pos] = {
            rowOutput, static_cast<vector_size_t>(updatePositionsOffset_)};
      }
      ++updatePositionsOffset_;
    }
  }

  while (!readFinishedForBatch(rowNumberUpperBound)) {
    // Allocate a fresh output vector so that next() does not overwrite any
    // previously stored batch that pendingUpdates_ entries still reference.
    VectorPtr freshOutput = BaseVector::create(outputRowType_, 0, pool_);
    auto batchSize = static_cast<uint64_t>(rowNumberUpperBound);
    auto rowsScanned = updateRowReader_->next(batchSize, freshOutput);
    updatePositionsOutput_ = freshOutput;
    totalNumRowsScanned_ += rowsScanned;

    if (rowsScanned > 0) {
      auto rowOutput =
          std::dynamic_pointer_cast<RowVector>(updatePositionsOutput_);
      auto numRows = updatePositionsOutput_->size();
      if (numRows > 0) {
        updatePositionsOutput_->loadedVector();
        auto posVector = rowOutput->childAt(0)->as<FlatVector<int64_t>>();
        VELOX_CHECK_NOT_NULL(posVector);
        updatePositionsOffset_ = 0;

        for (vector_size_t i = 0; i < numRows; ++i) {
          int64_t pos = posVector->valueAt(i);
          if (pos < rowNumberLowerBound) {
            ++updatePositionsOffset_;
            continue;
          }
          if (pos >= rowNumberUpperBound) {
            updatePositionsOffset_ = i;
            return;
          }
          pendingUpdates_[pos] = {rowOutput, i};
          ++updatePositionsOffset_;
        }
      }
    } else {
      updateSplit_.reset();
      break;
    }
  }
}

void PositionalUpdateFileReader::applyUpdates(
    uint64_t baseReadOffset,
    const RowVectorPtr& output,
    const BufferPtr& deleteBitmap,
    uint64_t batchSize) {
  int64_t rowNumberUpperBound =
      static_cast<int64_t>(splitOffset_ + baseReadOffset + batchSize);
  int64_t rowNumberLowerBound =
      static_cast<int64_t>(splitOffset_ + baseReadOffset);

  readUpdatePositions(baseReadOffset, rowNumberUpperBound);

  if (pendingUpdates_.empty()) {
    return;
  }

  const auto& outputRowType = *output->rowType();

  // Build a mapping from original file position to output row index. When a
  // delete bitmap is present, deleted rows are removed from the output, so
  // file positions do not map 1:1 to output indices.
  const uint64_t* deleteBits =
      deleteBitmap ? deleteBitmap->as<uint64_t>() : nullptr;

  auto it = pendingUpdates_.lower_bound(rowNumberLowerBound);
  while (it != pendingUpdates_.end() && it->first < rowNumberUpperBound) {
    int64_t pos = it->first;
    auto& [srcBatch, srcIdx] = it->second;

    auto origIndex = static_cast<vector_size_t>(pos - rowNumberLowerBound);

    // If this position was deleted, skip the update.
    if (deleteBits && origIndex < static_cast<vector_size_t>(batchSize)) {
      if (bits::isBitSet(deleteBits, origIndex)) {
        auto nextIt = std::next(it);
        pendingUpdates_.erase(it);
        it = nextIt;
        continue;
      }
    }

    // Compute the output row index by counting non-deleted rows before this
    // position.
    vector_size_t rowIndex;
    if (deleteBits) {
      auto deletedBefore = bits::countBits(deleteBits, 0, origIndex);
      rowIndex = origIndex - static_cast<vector_size_t>(deletedBefore);
    } else {
      rowIndex = origIndex;
    }

    if (rowIndex >= output->size()) {
      ++it;
      continue;
    }

    for (size_t c = 0; c < updateColumnNames_.size(); ++c) {
      auto outputColIdx =
          outputRowType.getChildIdxIfExists(updateColumnNames_[c]);
      if (!outputColIdx.has_value()) {
        continue;
      }

      auto& outputCol = output->childAt(*outputColIdx);
      // Ensure the output column supports random writes. The base reader may
      // produce dictionary-encoded or constant vectors which do not allow copy.
      if (!outputCol->isFlatEncoding()) {
        BaseVector::flattenVector(outputCol);
      }
      // childAt(0) is pos, so update columns start at childAt(1).
      auto& srcCol = srcBatch->childAt(static_cast<column_index_t>(c + 1));
      outputCol->copy(srcCol.get(), rowIndex, srcIdx, 1);
    }

    auto nextIt = std::next(it);
    pendingUpdates_.erase(it);
    it = nextIt;
  }
}

bool PositionalUpdateFileReader::noMoreData() {
  if (!pendingUpdates_.empty()) {
    return false;
  }
  // Check for leftover positions that haven't been consumed yet.
  if (updatePositionsOutput_ &&
      updatePositionsOffset_ <
          static_cast<uint64_t>(updatePositionsOutput_->size())) {
    return false;
  }
  return totalNumRowsScanned_ >= updateFile_.recordCount;
}

bool PositionalUpdateFileReader::readFinishedForBatch(
    int64_t rowNumberUpperBound) {
  if (!updatePositionsOutput_) {
    return true;
  }

  if (totalNumRowsScanned_ >= updateFile_.recordCount) {
    return true;
  }

  auto rowOutput = std::dynamic_pointer_cast<RowVector>(updatePositionsOutput_);
  if (!rowOutput || rowOutput->size() == 0) {
    return false;
  }

  auto posVector = rowOutput->childAt(0)->as<FlatVector<int64_t>>();
  if (updatePositionsOffset_ < static_cast<uint64_t>(posVector->size()) &&
      posVector->valueAt(static_cast<vector_size_t>(updatePositionsOffset_)) >=
          rowNumberUpperBound) {
    return true;
  }

  return false;
}

} // namespace facebook::velox::connector::hive::iceberg
